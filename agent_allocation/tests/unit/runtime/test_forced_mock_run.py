"""Forced mode through the real dispatcher: ForcedClient over the fake /completions route, reservations that cover
both calls, two rate tokens per node, short THINKs retried (never terminal-complete), the ledger reconciling to
what the fake billed; plus the forced decoding hash, reservation bound, forecast, Gate 1 and the live Gate 0 wiring."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from agent_allocation.api import fake
from agent_allocation.api.forced import (
    XHIGH_INSTRUCTION, ForcedClient, ForcedSettings, forced_usage_from_raw, render_chat_prompt,
)
from agent_allocation.config import decoding_config_hash, load_config
from agent_allocation.contracts import GenerationKind, LedgerEventKind, TransportStatus
from agent_allocation.forecast import BenchmarkModel, Observation, fit_models, item_cost, observations_from_rows
from agent_allocation.gates import CallRow, evaluate_gate1
from agent_allocation.runtime.estimate import (
    FORCED_TEMPLATE_OVERHEAD_TOKENS, forced_prompt_tokens, rendered_chars, reservation_usd,
)
from agent_allocation.runtime.ledger import journal_attempts
from agent_allocation.runtime.reconcile import ledger_reconciliation
from agent_allocation.runtime.scheduler import SchedulerSettings, client_is_offline_mock
from agent_allocation.storage.raw import read_raw_response
from agent_allocation.storage.state import iter_parsed_files, read_parsed
from tests.unit.runtime.fakes import LOCK, Env, make_graph, plan_of, spec

ROOT = Path(__file__).resolve().parents[3]
FORCED_CFG = load_config(ROOT / "configs" / "forced27b.json")
FORCED_LOCK = fake.fake_forced_endpoint_lock(model=LOCK.model)     # the test specs' model, a forced endpoint


def forced_env(tmp_path, **scheduler) -> Env:
    env = Env.create(tmp_path, **scheduler)
    env.config["forced"] = deepcopy(FORCED_CFG["forced"])
    env.config["sampling"] = deepcopy(FORCED_CFG["sampling"])
    env.config["provider"]["endpoint_tag"] = FORCED_LOCK.provider_tag
    return env


def forced_client(script) -> ForcedClient:
    return ForcedClient(settings=ForcedSettings.from_config(FORCED_CFG), api_key=None,
                        transport=fake.fake_openrouter_transport(script), answer_backoff_base_seconds=0.0)


async def run_forced(env: Env, graphs, script):
    client = forced_client(script)
    assert client_is_offline_mock(client)
    dispatcher = env.dispatcher(client)
    dispatcher.lock = FORCED_LOCK
    try:
        summary = await dispatcher.run(plan_of(graphs), allow_paid=False, max_usd=Decimal("5"))
    finally:
        await client.aclose()
    return summary, dispatcher


def _graphs(n: int):
    return [make_graph(f"aime/{i}", benchmark="math500", bank=3, chain=3, team_slots=2, team_rounds=1,
                       peer_sizes=(2,), peer_rounds=1, deep=(2, 4)) for i in range(n)]


async def test_forced_mock_run_end_to_end(tmp_path):
    env = forced_env(tmp_path, concurrency_start=8, concurrency_max=16, requests_per_second_start=2000,
                     requests_per_second_max=4000, lease_stale_seconds=30)
    graphs = _graphs(3)
    short = {"n": 0}

    def hook(call):   # the first THINK of every 5th node is cut short by the "provider"
        if call.coordinates.get("phase") == "think" and str(call.coordinates.get("attempt_id", "")).endswith(":1") \
                and int(call.coordinates["node_id"][:4], 16) % 5 == 0:
            short["n"] += 1
            return fake.FakeReply(fault=fake.Fault.SHORT_THINK)
        return None

    script = fake.FakeScript(hook=hook, vary_final_by_node=True)
    summary, dispatcher = await run_forced(env, graphs, script)
    nodes = [s for g in graphs for s in g.nodes]
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == len(nodes), summary
    assert short["n"] > 0 and script.violations == []
    assert dispatcher.forced and dispatcher.http_calls_per_node == 2
    # every succeeded node realized exactly its budget, with THINK finish length
    parsed = [read_parsed(f) for f in iter_parsed_files(env.paths)]
    assert len(parsed) == len(nodes)
    budgets = {s.node_id: s.reasoning_budget for s in nodes}
    for result, candidate in parsed:
        assert result.transport_status is TransportStatus.OK
        assert result.usage.reasoning_tokens == budgets[result.node_id]
        raw = read_raw_response(env.paths, result.raw_response_path)
        assert forced_usage_from_raw(raw)["think_finish_reason"] == "length"
        assert candidate.final_answer
    # short THINKs were retried (settled at their real cost), never accepted
    attempts = journal_attempts(env.paths.ledger_journal)
    accepted = {r.attempt_id for r, _ in parsed}
    retried = [a for a in attempts if a not in accepted]
    assert len(retried) == short["n"]
    assert summary.settled_usd == script.billed_usd
    report = ledger_reconciliation(env.paths, provider={"usage_usd": script.billed_usd})
    assert report["discrepancies"] == [] and report["open_reservations_usd"] == "0"
    # the reservation covered both calls of every attempt
    for aid, info in attempts.items():
        assert info.resolution is LedgerEventKind.SETTLE and info.resolved_usd <= info.reserved_usd, aid
    # two HTTP calls per accepted node, one per retried short THINK
    assert len(script.text_completion_calls(phase="think")) == len(nodes) + short["n"]
    assert len(script.text_completion_calls(phase="answer")) == len(nodes)
    # the children saw their parents' answers (the forced ANSWER is the visible content)
    assert all("[No prior candidate" not in "".join(v) for v in env.render.seen.values())


async def test_forced_admission_takes_two_rate_tokens(tmp_path):
    env = forced_env(tmp_path, concurrency_start=8, concurrency_max=8, requests_per_second_start=4,
                     requests_per_second_max=4)
    script = fake.FakeScript()
    client = forced_client(script)
    dispatcher = env.dispatcher(client)
    dispatcher.lock = FORCED_LOCK
    taken = []
    original = dispatcher.aimd.take_token
    dispatcher.aimd.take_token = lambda: (taken.append(1), original())[1]
    graph = make_graph("aime/0", bank=2, chain=1, team_slots=0, peer_sizes=(), deep=())
    try:
        summary = await dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    finally:
        await client.aclose()
    assert summary.nodes_succeeded == 2 and len(taken) == 4


def test_forced_watchdog_is_sized_for_two_calls():
    base = load_config(ROOT / "configs" / "balanced150.json")
    chat = SchedulerSettings.from_config(base)
    forced = SchedulerSettings.from_config({**base, "forced": FORCED_CFG["forced"]})
    one_call = base["scheduler"]["connect_timeout_seconds"] + base["scheduler"]["read_timeout_seconds"]
    assert forced.watchdog_seconds >= 2 * one_call + 30 and forced.watchdog_seconds > chat.watchdog_seconds


def test_forced_decoding_hash_differs_and_chat_hash_is_unchanged():
    chat = load_config(ROOT / "configs" / "balanced150.json")
    forced = deepcopy(chat)
    forced["forced"] = deepcopy(FORCED_CFG["forced"])
    assert decoding_config_hash(forced) != decoding_config_hash(chat)
    disabled = deepcopy(forced)
    disabled["forced"]["enabled"] = False
    assert decoding_config_hash(disabled) == decoding_config_hash(chat)
    other_ban = deepcopy(forced)
    other_ban["forced"]["ban_token_ids"] = [248069]
    assert decoding_config_hash(other_ban) != decoding_config_hash(forced)
    assert decoding_config_hash(FORCED_CFG) != decoding_config_hash(chat)


def test_template_overhead_bound():
    overhead = len(render_chat_prompt([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]))
    assert FORCED_TEMPLATE_OVERHEAD_TOKENS >= -(-overhead // 2) and len(XHIGH_INSTRUCTION) < overhead


@pytest.mark.parametrize("budget,headroom", [(1024, 4096), (4096, 4096), (25600, 4096)])
@pytest.mark.parametrize("prompt_chars", [800, 6000, 30000])
async def test_forced_reservation_covers_the_realized_two_call_cost(budget, headroom, prompt_chars):
    """The fake bills at 4 chars/token (realistic for this text) and the realized THINK is the full budget; the ANSWER
    is scripted to use the whole headroom -- the worst case one attempt can bill."""
    lock = fake.fake_forced_endpoint_lock()
    messages = [{"role": "system", "content": "s" * 200}, {"role": "user", "content": "x" * prompt_chars}]
    node = replace(spec("aime/1", GenerationKind.DEEP, budget=budget, headroom=headroom), model=lock.model,
                   endpoint_lock_hash=lock.lock_hash)
    script = fake.FakeScript(hook=lambda call: fake.FakeReply(content="y" * (4 * headroom))
                             if call.coordinates.get("phase") == "answer" else None)
    client = forced_client(script)
    try:
        outcome = await client.generate(node, messages, attempt_id=f"{node.node_id}:1", lock=lock,
                                        sampling=FORCED_CFG["sampling"])
    finally:
        await client.aclose()
    assert outcome.result.usage.reasoning_tokens == budget
    reserved = reservation_usd(node, rendered_chars(messages), lock, multiplier=Decimal("1.1"), forced=True)
    assert outcome.result.usage.actual_cost_usd <= reserved
    chat_style = reservation_usd(node, rendered_chars(messages), lock, multiplier=Decimal("1.1"))
    assert reserved > chat_style                       # the ANSWER re-reads prompt + reasoning
    assert forced_prompt_tokens(rendered_chars(messages)) * 4 >= len(outcome.request_body["think"]["prompt"])


def test_forecast_forced_counts_the_reread_input():
    model = BenchmarkModel(root_prompt=500, private_prompt=900, packet_tokens=300, forced=True)
    chat = BenchmarkModel(root_prompt=500, private_prompt=900, packet_tokens=300)
    grid = load_grid()
    lock = fake.fake_forced_endpoint_lock()
    assert item_cost(model, grid, 1024, 4096, lock) > item_cost(chat, grid, 1024, 4096, lock)
    assert item_cost(chat, grid, 1024, 4096, lock, forced=True) == item_cost(model, grid, 1024, 4096, lock)
    obs = [Observation("aime", GenerationKind.ROOT, 0, 1024, 4096, 500, 1400, forced=True)]
    assert fit_models(obs)["aime"].forced


def load_grid():
    from agent_allocation.config import grids_from_config
    return grids_from_config(FORCED_CFG)["main"]


async def test_forced_rows_feed_forecast_and_gate1():
    lock = fake.fake_forced_endpoint_lock()
    rows = []
    for budget in (1024, 2048):
        node = replace(spec("aime/1", GenerationKind.ROOT, budget=budget, headroom=4096), model=lock.model,
                       endpoint_lock_hash=lock.lock_hash)
        script = fake.FakeScript()
        client = forced_client(script)
        try:
            outcome = await client.generate(node, [{"role": "user", "content": "q" * 400}],
                                            attempt_id=f"{node.node_id}:1", lock=lock,
                                            sampling=FORCED_CFG["sampling"])
        finally:
            await client.aclose()
        payload = json.loads(outcome.response_body)
        rows.append(CallRow(node, outcome.result, None, {"provider_name": lock.provider_name,
                                                         "service_tier": "standard"}, {"response_body_text":
                                                                                        json.dumps(payload)}))
    observations = observations_from_rows(rows)
    assert all(o.forced for o in observations)
    assert observations[0].prompt_tokens == forced_usage_from_raw(rows[0].raw_payload)["think_prompt_tokens"]
    locks = {lock.model: lock}
    gate = evaluate_gate1(rows, locks, key_usage_delta_usd=Decimal("0"), ledger_settled_usd=Decimal("1"))
    names = {c.name: c.passed for c in gate.criteria}
    assert "numeric_budget_applied" not in names and names["forced_exact_budget"], gate.criteria
    assert gate.passed, [(c.name, c.observed) for c in gate.criteria if not c.passed]
    short = replace(rows[0].result, usage=replace(rows[0].result.usage, reasoning_tokens=1000))
    bad = evaluate_gate1([replace(rows[0], result=short), rows[1]], locks, key_usage_delta_usd=Decimal("0"),
                         ledger_settled_usd=Decimal("1"))
    assert not {c.name: c.passed for c in bad.criteria}["forced_exact_budget"]


async def test_live_runtime_runs_gate0_before_paid_forced_dispatch(tmp_path, monkeypatch):
    from agent_allocation.live import Runtime, RunRefused
    from agent_allocation.paths import RunPaths
    cfg = deepcopy(FORCED_CFG)
    cfg["provider"]["endpoint_tag"] = "parasail/fp8"
    lock = fake.fake_forced_endpoint_lock()
    for honoured, expect_pass in ((False, False), (True, True)):
        paths = RunPaths.for_run(run_dir=tmp_path / f"run{honoured}", local_state_dir=tmp_path / f"local{honoured}")
        paths.ensure()
        script = fake.FakeScript(forced_bias_honoured=honoured)
        runtime = Runtime(cfg, paths, lock=lock, tasks={}, ceiling=Decimal("5"),
                          transport=fake.fake_openrouter_transport(script))
        assert isinstance(runtime.client, ForcedClient)
        try:
            if expect_pass:
                await runtime.run([], allow_paid=True, max_usd=Decimal("5"))
            else:
                with pytest.raises(RunRefused, match="Gate 0"):
                    await runtime.run([[make_graph("aime/0")]], allow_paid=True, max_usd=Decimal("5"))
                assert all(c.coordinates["phase"].startswith("gate0_") for c in script.text_completion_calls())
        finally:
            await runtime.aclose()
        gate = json.loads(paths.gate_result("gate0", "preflight").read_text())
        assert gate["passed"] is expect_pass
        assert (paths.reports_dir / "GATE0_FORCED_ENDPOINT.json").exists()
        charged = sum(a.resolved_usd for a in journal_attempts(paths.ledger_journal).values())
        assert charged == script.billed_usd > 0            # Gate 0's probes are charged through the ledger
