"""The deliverable mock runs: the real ``OpenRouterClient`` over agent 01's ``fake_openrouter_transport`` streams
dependent completions through the dispatcher; the ledger reconciles against what the fake billed; effective
concurrency is measured, not assumed. No network, no key. Measurements land in ``reports/mock_run_*.json``."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pytest

from agent_allocation.contracts import GenerationKind, LedgerEventKind
from agent_allocation.runtime.ledger import journal_attempts
from agent_allocation.runtime.reconcile import ledger_reconciliation
from agent_allocation.runtime.scheduler import client_is_offline_mock
from tests.unit.runtime.fakes import ABSENT, Env, make_graph, plan_of

api_fake = pytest.importorskip("agent_allocation.api.fake")
api_client = pytest.importorskip("agent_allocation.api.client")

BENCHMARKS = ("math500", "mmlu_pro", "bbh_logic4", "mbpp_plus")


def _bucket(text: str, modulus: int) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % modulus


def _graphs(items_per_benchmark: int):
    return [make_graph(f"{b}/{i}", benchmark=b, bank=6, chain=6, team_slots=3, team_rounds=2, peer_sizes=(2, 3),
                       peer_rounds=2, deep=(2,)) for i in range(items_per_benchmark) for b in BENCHMARKS]


async def _run(env: Env, graphs, hook, capsys, label: str):
    script = api_fake.FakeScript(hook=hook, vary_final_by_node=True)
    client = api_client.OpenRouterClient(api_key=None, transport=api_fake.fake_openrouter_transport(script))
    assert client_is_offline_mock(client)
    dispatcher = env.dispatcher(client)
    dispatcher.lock = api_fake.fake_endpoint_lock()
    try:
        summary = await dispatcher.run(plan_of(graphs), allow_paid=False, max_usd=Decimal("5"))
    finally:
        await client.aclose()
    report = ledger_reconciliation(env.paths, provider={"usage_usd": script.billed_usd})
    measured = {"label": label, "nodes": summary.nodes_planned, "requests": summary.requests_sent,
                "wall_seconds": round(summary.wall_seconds, 2), "configured_concurrency": summary.concurrency_start,
                "peak_in_flight": summary.peak_in_flight, "mean_in_flight": round(summary.mean_in_flight, 1),
                "mean_in_flight_active": round(summary.mean_in_flight_active, 1),
                "concurrency_final": summary.concurrency_final, "rps_final": summary.rps_final,
                "rate_limited": summary.rate_limited, "aimd_increases_decreases": [summary.extra["aimd_increases"],
                                                                                   summary.extra["aimd_decreases"]],
                "settled_usd": str(summary.settled_usd), "fake_billed_usd": str(script.billed_usd),
                "charged_at_maximum_usd": report["charged_at_maximum_usd"],
                "retry_and_failed_attempt_charges_usd": report["retry_and_failed_attempt_charges_usd"],
                "journal_batches": summary.extra["journal_batches"], "journal_max_batch": summary.extra["journal_max_batch"]}
    (env.paths.reports_dir / f"mock_run_{label}.json").write_text(json.dumps(measured, indent=2))
    with capsys.disabled():
        print(f"\nMOCK RUN {json.dumps(measured)}")
    return summary, script, report


async def test_mock_run_realizes_configured_concurrency_and_streams_dependents(tmp_path, capsys):
    env = Env.create(tmp_path, concurrency_start=48, concurrency_max=48, requests_per_second_start=4000,
                     requests_per_second_max=4000, max_calls_per_item_in_flight=8, lease_stale_seconds=30)
    graphs = _graphs(4)                                           # 16 items (one balanced wave), 416 nodes

    def hook(call):
        attempt = str(call.coordinates.get("attempt_id", ""))
        return api_fake.FakeReply(delay_seconds=0.2 + _bucket(attempt, 400) / 1000.0)   # 0.2-0.6 s latency

    summary, script, report = await _run(env, graphs, hook, capsys, "throughput")
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == summary.nodes_planned == 416
    assert script.violations == [] and report["discrepancies"] == []
    assert summary.settled_usd == script.billed_usd                 # every charge settled exactly, nothing held
    assert summary.peak_in_flight == 48
    assert summary.mean_in_flight_active >= 24                        # HTTP-only limiter keeps the pipe full
    for g in graphs:
        for s in g.nodes:
            assert ABSENT not in env.render.seen[s.node_id]            # every child saw its parents' finals
    kinds = {s.node_id: s for g in graphs for s in g.nodes}
    order = [c.coordinates["node_id"] for c in script.completion_calls()]
    first_chain_step = min(i for i, n in enumerate(order)
                           if kinds[n].kind is GenerationKind.PRIVATE and kinds[n].root_slot == 0)
    last_root = max(i for i, n in enumerate(order) if kinds[n].kind is GenerationKind.ROOT)
    assert first_chain_step < last_root                               # children stream; no global barrier


async def test_mock_run_with_faults_reconciles_every_charge(tmp_path, capsys):
    Fake, Fault = api_fake.FakeReply, api_fake.Fault
    env = Env.create(tmp_path, concurrency_start=24, concurrency_max=48, aimd_window_seconds=0.5,
                     increase_concurrency_by=8, requests_per_second_start=2000, requests_per_second_max=4000,
                     increase_rps_by=500, max_calls_per_item_in_flight=8, lease_stale_seconds=30)
    graphs = _graphs(2)                                           # 8 items, 208 nodes
    faults = {"rate_limited": 0, "read_timeout": 0, "connect_error": 0}

    def hook(call):
        attempt = str(call.coordinates.get("attempt_id", ""))
        if attempt.endswith(":1"):
            if _bucket(attempt, 60) == 0:
                faults["rate_limited"] += 1
                return Fake(fault=Fault.RATE_LIMITED, retry_after="0.2")
            if _bucket(attempt, 50) == 1:
                faults["read_timeout"] += 1
                return Fake(fault=Fault.READ_TIMEOUT)           # generated and billed; the client never sees it
            if _bucket(attempt, 45) == 2:
                faults["connect_error"] += 1
                return Fake(fault=Fault.CONNECT_ERROR)          # never reached the server
        return Fake(delay_seconds=0.02 + _bucket(attempt + "d", 100) / 1000.0)

    summary, script, report = await _run(env, graphs, hook, capsys, "faults")
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 208
    assert all(faults.values()), faults
    billed: dict[str, Decimal] = {}
    for call in script.completion_calls():
        aid = call.coordinates["attempt_id"]
        billed[aid] = billed.get(aid, Decimal("0")) + call.billed_usd
    attempts = journal_attempts(env.paths.ledger_journal)
    # attempts admitted before a 429's Retry-After pause but still waiting for the HTTP limiter never left: they are
    # released (or settled at 0 if their SENT fsync overlapped the pause) and never reach the provider
    withdrawn = {aid for aid, info in attempts.items() if info.resolution is LedgerEventKind.RELEASE_UNSENT
                 or (info.resolution is LedgerEventKind.SETTLE and info.resolved_usd == 0 and aid not in billed)}
    assert set(billed) == set(attempts) - withdrawn
    for aid, info in attempts.items():
        if aid in withdrawn:
            continue
        if info.resolution is LedgerEventKind.SETTLE:
            assert info.resolved_usd == billed[aid], aid              # settled exactly what was charged
        else:                                                         # read timeouts: never assumed free
            assert info.resolution is LedgerEventKind.CHARGE_MAX and info.resolved_usd >= billed[aid] > 0
    assert summary.settled_usd >= script.billed_usd
    assert report["discrepancies"] == [] and report["open_reservations_usd"] == "0"
    assert Decimal(report["charged_usd"]) == summary.settled_usd
    assert summary.rate_limited == faults["rate_limited"] and summary.extra["aimd_decreases"] >= 1
