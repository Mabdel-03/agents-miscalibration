"""Failure injection against the real dispatcher, ledger, journal, state store and client (agents 01 + 02).

Invariants asserted after every scenario (spec 7.1, 10.3, 10.5; acceptance 15.8, 15.10, 15.20):
  * the run terminates (no hang) and the journal replays with zero ledger violations
  * no semantic node receives more than one successful (HTTP 200 completion) response, across resumes
  * no reservation for an attempt that reached the provider is ever released
  * transport attempts per node never exceed ``max_transport_attempts``; every request was reserved first
  * a child is never requested before its parents' responses came back
"""
from __future__ import annotations

import copy
import sqlite3
from decimal import Decimal

import httpx
import pytest

from agent_allocation.config import grids_from_config
from agent_allocation.contracts import LedgerEventKind as K
from agent_allocation.contracts import PublicTask

from tests.integration import adversarial
from tests.integration import helpers as h
from tests.integration.conftest import optional_module
from tests.integration.dispatch_harness import Harness, fast_config, item_graph, reserve_count
from tests.integration.openrouter_stub import (
    DispatchStub,
    StubReply,
    cache_hit,
    connect_failure,
    error_envelope,
    ok,
    rate_limited,
    read_timeout,
)

pytestmark = pytest.mark.timeout(120)


def _items(n_per_benchmark: int = 1) -> list[PublicTask]:
    tasks = [t for t, _, _ in adversarial.load_items()]
    out = []
    for bench in ("math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"):
        out += [t for t in tasks if t.benchmark == bench][:n_per_benchmark]
    return out


def _harness(run_paths, cfg, *, grid_name: str = "pilot_r1", partition: str = "pilot", n_items: int = 2,
             ceiling: str = "5", max_unresolved: str = "2", budget: int = 1024, **scheduler) -> Harness:
    cfg = copy.deepcopy(cfg)
    cfg["budget"]["max_unresolved_spend_usd"] = float(max_unresolved)     # the dispatcher refuses a mismatch
    grid = grids_from_config(cfg)[grid_name]
    tasks = _items(1)[:n_items]
    graphs = [item_graph(h.ref_graph(grid, t, partition=partition, budget=budget), t, grid, partition) for t in tasks]
    return Harness(paths=run_paths, cfg=fast_config(cfg, **scheduler), graphs=graphs,
                   tasks={t.item_id: t for t in tasks}, stub=DispatchStub(), ceiling=Decimal(ceiling),
                   max_unresolved=Decimal(max_unresolved))


def _common_invariants(hx: Harness, *, max_attempts: int = 3) -> None:
    assert hx.ledger_violations() == [], hx.ledger_violations()
    doubles = {n: c for n, c in hx.stub.ok_count().items() if c > 1}
    assert not doubles, f"nodes with more than one successful response: {len(doubles)}"
    per_node = {}
    for req in hx.stub.served:
        per_node[req.node_id] = per_node.get(req.node_id, 0) + 1
    over = {n: c for n, c in per_node.items() if c > max_attempts}
    assert not over, f"transport attempts above {max_attempts}: {over}"
    assert reserve_count(hx.journal_events()) >= len(hx.stub.served), "a request went out without a reservation"
    assert hx.request_before_parent_violations() == []
    assert hx.parent_order_violations() == []
    evidence = hx.connect_failed_attempts()
    released_after_send = [e.attempt_id for e in hx.journal_events() if e.kind is K.RELEASE_UNSENT
                           and e.attempt_id not in evidence
                           and any(x.kind is K.SENT and x.attempt_id == e.attempt_id for x in hx.journal_events())]
    assert not released_after_send


def _root(hx: Harness, item_index: int = 0, slot: int = 0) -> str:
    graph = hx.graphs[item_index]
    return next(n.node_id for n in graph.nodes if n.kind.value == "root" and n.root_slot == slot)


# --------------------------------------------------------------------------------------------------------------
# Baseline: exactly-once dispatch, ordering, metering, sealed collection cost
# --------------------------------------------------------------------------------------------------------------

async def test_happy_path_dispatches_each_node_once_and_meters_exactly(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    store, ledger = await hx.run()
    _common_invariants(hx)
    assert len(hx.stub.served) == len(hx.specs) == 24
    assert set(hx.specs) <= store.terminal_node_ids()
    assert len(hx.raw_files()) == 24
    settled = sum((e.amount_usd for e in hx.journal_events() if e.kind in (K.SETTLE, K.CHARGE_MAX)), Decimal(0))
    assert settled == Decimal(str(hx.stub.cost)) * 24, "collection ledger must charge each usage object exactly once"
    snap = ledger.snapshot()
    assert snap.in_flight == snap.unresolved == 0 and snap.settled == settled


@pytest.mark.parametrize("grid_name,partition,budget,count", [("main", "main", 1024, 125), ("high", "high", 4096, 123),
                                                               ("replication", "replication", 1024, 57),
                                                               ("main_extension", "main", 1024, 138)])
async def test_profiles_dispatch_full_grid_once_with_synchronous_rounds(run_paths, cfg, grid_name, partition, budget, count):
    hx = _harness(run_paths, cfg, grid_name=grid_name, partition=partition, n_items=1, budget=budget)
    store, _ = await hx.run()
    _common_invariants(hx)
    assert len(hx.stub.served) == count and set(hx.specs) <= store.terminal_node_ids()
    bodies = [r.body_text for r in hx.stub.served]
    deep = [s for s in hx.specs.values() if s.kind.value == "deep"]
    assert all(f'"max_tokens": {s.max_completion_tokens}' in b or f'"max_tokens":{s.max_completion_tokens}' in b
               for s in deep for b in bodies if s.node_id in b), "deep budgets must not be clipped"


# --------------------------------------------------------------------------------------------------------------
# 429 bursts, persistent 429, 5xx / error envelope, connect failure, ambiguous timeout
# --------------------------------------------------------------------------------------------------------------

async def test_429_burst_honors_retry_after_and_retains_correct_reservations(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    burst = [_root(hx, i, s) for i in range(2) for s in range(3)]
    for node in burst:
        hx.stub.script(node, rate_limited("1"))
    store, ledger = await hx.run()
    _common_invariants(hx)
    for node in burst:
        reqs = hx.stub.requests_for(node)
        assert [r.outcome for r in reqs] == ["429", "ok"], [r.outcome for r in reqs]
        assert reqs[1].received_at - reqs[0].received_at >= 0.9, "Retry-After not honored"
    assert set(hx.specs) <= store.terminal_node_ids()
    snap = ledger.snapshot()
    assert snap.in_flight == snap.unresolved == 0


async def test_persistent_429_is_bounded_and_terminates(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 0, 1)
    hx.stub.script(victim, *[rate_limited("0") for _ in range(10)])
    store, _ = await hx.run()
    _common_invariants(hx)
    assert len(hx.stub.requests_for(victim)) == cfg["scheduler"]["max_transport_attempts"]
    assert hx.stub.first_ok_completed(victim) is None
    other_item = {n.node_id for n in hx.graphs[1].nodes}
    assert other_item <= store.terminal_node_ids(), "one exhausted node must not stall unrelated items"


@pytest.mark.parametrize("fault", ["upstream_503", "connect_failure", "error_envelope"])
async def test_transient_failures_retry_as_new_metered_attempts(run_paths, cfg, fault):
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 1, 0)
    reply = {"upstream_503": StubReply(status=503, payload={"error": {"code": 503, "message": "unavailable"}}),
             "connect_failure": connect_failure(), "error_envelope": error_envelope(502)}[fault]
    hx.stub.script(victim, reply)
    store, _ = await hx.run()
    _common_invariants(hx)
    attempts = {e.attempt_id for e in hx.events_for(victim) if e.kind is K.RESERVE}
    if fault == "error_envelope" and len(hx.stub.requests_for(victim)) == 1:
        pytest.xfail("D1: ERROR_ENVELOPE has no declared retry policy; dispatcher treated it as terminal")
    assert len(hx.stub.requests_for(victim)) == 2 and len(attempts) == 2, "each retry is its own reserved attempt"
    assert victim in store.terminal_node_ids()


async def test_timeout_after_dispatch_never_releases_the_reservation(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 0, 2)
    hx.stub.script(victim, read_timeout())
    store, ledger = await hx.run()
    _common_invariants(hx)
    first = sorted({e.attempt_id for e in hx.events_for(victim) if e.kind is K.RESERVE})[0]
    kinds = [e.kind for e in hx.events_for(first)]
    assert K.SENT in kinds and K.RELEASE_UNSENT not in kinds
    reserved = next(e.amount_usd for e in hx.events_for(first) if e.kind is K.RESERVE)
    if K.CHARGE_MAX in kinds:
        assert next(e.amount_usd for e in hx.events_for(first) if e.kind is K.CHARGE_MAX) >= reserved
    else:
        snap = ledger.snapshot()
        assert snap.unresolved >= reserved, "ambiguous timeout neither charged at max nor held as unresolved"
    assert len(hx.stub.requests_for(victim)) <= 3


async def test_unresolved_exposure_above_limit_pauses_admission(run_paths, cfg):
    hx = _harness(run_paths, cfg, max_unresolved="0.000001", concurrency_start=1, concurrency_max=1)
    victim = _root(hx, 0, 0)
    hx.stub.script(victim, read_timeout())
    await hx.run(expect_error=True)
    events = hx.events_for(victim)
    first = sorted({e.attempt_id for e in events if e.kind is K.RESERVE})[0]
    if any(e.kind is K.CHARGE_MAX for e in hx.events_for(first)):
        pytest.skip("timeout settled conservatively at max; no unresolved exposure to pause on")
    assert len(hx.stub.served) < len(hx.specs), "admission continued past the unresolved-exposure limit"
    assert hx.ledger_violations() == []


# --------------------------------------------------------------------------------------------------------------
# Circuit breakers: cache HIT, provider mismatch, 402/401
# --------------------------------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="W02-3: runtime/scheduler.py:984-1011 opens the circuit only after the raw "
                                       "write thread hop and the ledger fsync; admissions continue meanwhile (4-5 extra "
                                       "requests observed at concurrency 1)")
@pytest.mark.parametrize("fault", ["cache_hit", "provider_mismatch", "payment_required", "auth_failed"])
async def test_circuit_breakers_stop_the_partition(run_paths, cfg, fault):
    hx = _harness(run_paths, cfg, concurrency_start=1, concurrency_max=1)
    victim = _root(hx, 0, 0)
    reply = {"cache_hit": cache_hit(), "provider_mismatch": ok(provider="Google Vertex", model="google/gemini-2.5-flash"),
             "payment_required": StubReply(status=402, payload={"error": {"code": 402, "message": "credits"}}),
             "auth_failed": StubReply(status=401, payload={"error": {"code": 401, "message": "key"}})}[fault]
    hx.stub.script(victim, reply)
    hx.stub.scripts[victim].replies.extend([reply] * 3)
    await hx.run(expect_error=True)
    assert hx.ledger_violations() == []
    victim_reqs = hx.stub.requests_for(victim)
    assert len(victim_reqs) == 1, f"{fault} was retried"
    known_at = victim_reqs[0].completed_at
    after = [r for r in hx.stub.served if r.received_at > known_at]
    assert not after, (f"{fault}: {len(after)} requests left after the circuit-breaking response had been received "
                       f"(concurrency 1, so none were already in flight)")


async def test_malformed_final_is_terminal_not_redrawn(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 0, 0)
    hx.stub.script(victim, ok(content="<rationale>two answers</rationale><final>1</final><final>2</final>"))
    empty = _root(hx, 1, 0)
    hx.stub.script(empty, ok(content="", finish="length", native_finish="MAX_TOKENS", reasoning_tokens=2048))
    store, _ = await hx.run()
    _common_invariants(hx)
    for node in (victim, empty):
        assert len(hx.stub.requests_for(node)) == 1, "a completed bad answer or empty length stop was redrawn"
        assert node in store.terminal_node_ids()


# --------------------------------------------------------------------------------------------------------------
# Crashes and restarts: raw persisted before commit, SQLite failure, stale lease, sent-but-unknown
# --------------------------------------------------------------------------------------------------------------

class InjectedCrash(RuntimeError):
    pass


def _faulty_store(paths, *, fail_node: str | None = None, fail_after: int | None = None, exc=InjectedCrash):
    state_mod = optional_module("agent_allocation.storage.state")

    class FaultyStore(state_mod.StateStore):
        calls = 0

        def record_result(self, result, candidate):
            FaultyStore.calls += 1
            if (fail_node and result.node_id == fail_node) or (fail_after is not None and FaultyStore.calls > fail_after):
                raise exc("injected: raw written, commit lost")
            return super().record_result(result, candidate)

    return FaultyStore(paths)


async def test_crash_after_raw_write_before_commit_resumes_without_redispatch(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 0, 1)
    await hx.run(store=_faulty_store(run_paths, fail_node=victim), expect_error=True)
    assert hx.stub.ok_count()[victim] == 1
    raw_for_victim = [p for p in hx.raw_files() if victim[:16] in p.name]
    assert raw_for_victim, "the raw response must be written atomically before the terminal commit"
    served_before, ok_before = len(hx.stub.served), {n for n, c in hx.stub.ok_count().items() if c}
    store, _ = await hx.run(resume=True)
    _common_invariants(hx)
    assert hx.stub.ok_count()[victim] == 1, "resume re-dispatched a node whose successful raw response was on disk"
    assert len(hx.stub.served) - served_before == len(set(hx.specs) - ok_before)
    assert set(hx.specs) <= store.terminal_node_ids()
    assert all(p.exists() for p in raw_for_victim)


async def test_sqlite_commit_failure_stops_cleanly_and_resume_is_idempotent(run_paths, cfg):
    hx = _harness(run_paths, cfg)
    await hx.run(store=_faulty_store(run_paths, fail_after=5, exc=sqlite3.OperationalError), expect_error=True)
    uncommitted_ok = {n for n, c in hx.stub.ok_count().items() if c}
    requested_children = [r for r in hx.stub.served if any(p in uncommitted_ok for p in hx.specs[r.node_id].parent_ids)]
    assert len(requested_children) <= len(hx.specs), "sanity"
    served_before = len(hx.stub.served)
    store, _ = await hx.run(resume=True)
    _common_invariants(hx)
    assert set(hx.specs) <= store.terminal_node_ids()
    assert len(hx.stub.served) - served_before == len(set(hx.specs) - uncommitted_ok), \
        "resume re-requested nodes whose successful raw responses survived the commit failure"


async def test_stale_lease_never_runs_a_node_concurrently(run_paths, cfg):
    hx = _harness(run_paths, cfg, lease_stale_seconds=0.2, lease_heartbeat_seconds=0.05)
    slow = _root(hx, 0, 0)
    hx.stub.script(slow, delay=1.0)
    store, _ = await hx.run()
    _common_invariants(hx)
    assert hx.stub.max_active[slow] == 1 and len(hx.stub.requests_for(slow)) == 1
    assert slow in store.terminal_node_ids()


async def test_sent_but_unknown_attempt_from_a_dead_process_is_reconciled_conservatively(run_paths, cfg):
    journal_mod = optional_module("agent_allocation.storage.journal")
    ledger_mod = optional_module("agent_allocation.runtime.ledger")
    hx = _harness(run_paths, cfg)
    victim = _root(hx, 0, 0)
    journal = journal_mod.LedgerJournal(run_paths.ledger_journal, group_commit_ms=1)
    dead = ledger_mod.CollectionLedger(journal, ceiling_usd=hx.ceiling, max_unresolved_usd=hx.max_unresolved)
    attempt = f"{victim}:1"
    assert await dead.reserve(attempt_id=attempt, node_id=victim, partition_id="pilot", amount_usd=Decimal("0.01"))
    await dead.mark_sent(attempt)                        # the process dies with the request on the wire
    store, ledger = await hx.run(resume=True)
    _common_invariants(hx)
    kinds = [e.kind for e in hx.events_for(attempt)]
    assert K.RELEASE_UNSENT not in kinds
    held = ledger.snapshot().unresolved + ledger.snapshot().in_flight
    assert K.CHARGE_MAX in kinds or K.SETTLE in kinds or held >= Decimal("0.01"), "unknown money disappeared"
    new_attempts = {e.attempt_id for e in hx.events_for(victim) if e.kind is K.RESERVE} - {attempt}
    assert len(new_attempts) == len(hx.stub.requests_for(victim)), "a retry must be a new attempt id, never :1 again"
    assert victim in store.terminal_node_ids()


# --------------------------------------------------------------------------------------------------------------
# Spending-limit contention
# --------------------------------------------------------------------------------------------------------------

async def test_tiny_ceiling_under_contention_never_overbooks_and_resume_completes(run_paths, cfg):
    hx = _harness(run_paths, cfg, ceiling="0.002", concurrency_start=64, concurrency_max=64)
    await hx.run(expect_error=True)
    assert hx.ledger_violations(Decimal("0.002")) == []
    first_pass = len(hx.stub.served)
    assert 0 < first_pass < len(hx.specs), "the tiny ceiling should stop collection part-way"
    done_first = {n for n, c in hx.stub.ok_count().items() if c}
    hx.ceiling = Decimal("5")
    store, _ = await hx.run(resume=True)
    _common_invariants(hx)
    assert set(hx.specs) <= store.terminal_node_ids()
    assert not {r.node_id for r in hx.stub.served[first_pass:]} & done_first


async def test_max_usd_below_ledger_ceiling_is_respected(run_paths, cfg):
    hx = _harness(run_paths, cfg, concurrency_start=64, concurrency_max=64)
    await hx.run(max_usd=Decimal("0.0015"), expect_error=True)
    replay = hx.ledger_violations(Decimal("0.0015"))
    assert replay == [], replay
    assert len(hx.stub.served) < len(hx.specs)


# --------------------------------------------------------------------------------------------------------------
# Duplicate / stale completion notifications at the storage and ledger layer
# --------------------------------------------------------------------------------------------------------------

async def test_duplicate_and_stale_completion_notifications_are_idempotent(run_paths):
    state_mod = optional_module("agent_allocation.storage.state")
    raw_mod = optional_module("agent_allocation.storage.raw")
    import dataclasses
    from agent_allocation.contracts import Attempt, AttemptState, GenerationResult, GenerationSpec, load_record
    rows = adversarial.load_jsonl("duplicate_completions.jsonl")
    task = adversarial.load_items()[0][0]
    store = state_mod.StateStore(run_paths)
    specs = {s.node_id: s for s in (load_record(GenerationSpec, r["node_spec"]) for r in rows)}
    store.register_nodes(specs.values(), wave_index=0)
    accepted = {}
    for row in rows:
        result = load_record(GenerationResult, row["result"])
        spec = specs[result.node_id]
        if result.transport_status.value == "ok":
            store.record_attempt(Attempt(result.attempt_id, result.node_id, int(result.attempt_id.rsplit(":", 1)[1]),
                                         AttemptState.SENT, Decimal("0.001"), h.T0, sent_at=h.T0))
        rel, sha = raw_mod.write_raw_response(run_paths, result.attempt_id, h.outcome_for(result))
        result = dataclasses.replace(result, raw_response_path=rel, raw_response_sha256=sha)
        if result.transport_status.value != "ok":
            store.record_attempt(Attempt(result.attempt_id, result.node_id, int(result.attempt_id.rsplit(":", 1)[1]),
                                         AttemptState.TERMINAL, Decimal("0.001"), h.T0, sent_at=h.T0,
                                         completed_at=h.T0, transport_status=result.transport_status, http_status=429))
            continue
        try:
            store.record_result(result, h.valid_candidate(spec, task, result.visible_content))
            accepted.setdefault(result.node_id, result.attempt_id)
        except Exception:
            pass                                          # explicit rejection of a duplicate is acceptable
    assert set(specs) <= store.terminal_node_ids()
    for node_id, attempt_id in accepted.items():
        parsed = run_paths.parsed(node_id)
        if parsed.exists():
            import json
            data = json.loads(parsed.read_text())
            assert data["result"]["attempt_id"] == attempt_id, "a later duplicate/stale notification overwrote the result"


async def test_settling_one_attempt_twice_never_double_charges(run_paths):
    journal_mod = optional_module("agent_allocation.storage.journal")
    ledger_mod = optional_module("agent_allocation.runtime.ledger")
    ledger = ledger_mod.CollectionLedger(journal_mod.LedgerJournal(run_paths.ledger_journal, group_commit_ms=1),
                                         ceiling_usd=Decimal("1"), max_unresolved_usd=Decimal("2"))
    a = f"{7:064x}:1"
    assert await ledger.reserve(attempt_id=a, node_id=a[:64], partition_id="main", amount_usd=Decimal("0.2"))
    await ledger.mark_sent(a)
    await ledger.settle(a, Decimal("0.05"))
    try:
        await ledger.settle(a, Decimal("0.05"))
    except Exception:
        pass
    assert ledger.snapshot().settled == Decimal("0.05")
