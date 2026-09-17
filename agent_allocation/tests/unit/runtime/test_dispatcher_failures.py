"""Required failure-mode tests: crash before commit, delayed bills, stale leases, 429 bursts, slow grading,
duplicate notifications, disk failure, graceful stop, circuit breaking, unresolved exposure, exhausted nodes."""
from __future__ import annotations

import asyncio
import errno
import time
from contextlib import suppress
from dataclasses import replace
from decimal import Decimal


import agent_allocation.runtime.scheduler as scheduler_mod
from agent_allocation.contracts import (
    Attempt, AttemptState, GenerationKind, LedgerEventKind, TransportStatus,
)
from agent_allocation.runtime.control import cmd_stop
from agent_allocation.runtime.ledger import journal_attempts
from agent_allocation.runtime.reconcile import ledger_reconciliation
from agent_allocation.runtime.scheduler import Dispatcher
from agent_allocation.storage.raw import write_raw_response as real_write_raw
from tests.unit.runtime.fakes import ABSENT, Behavior, Env, FakeClient, fake_extract, make_graph, plan_of

D = Decimal
COST = D("0.0001")


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so no handler in the dispatcher catches it."""


def _kinds(env: Env) -> dict[str, LedgerEventKind | None]:
    return {aid: info.resolution for aid, info in journal_attempts(env.paths.ledger_journal).items()}


def chain_graph(item: str = "c", length: int = 2):
    return make_graph(item, bank=1, chain=length, team_slots=1, peer_sizes=(), deep=())


async def _crash_run(env: Env, client: FakeClient, plan, monkeypatch, crashed: asyncio.Event) -> None:
    async def no_final_persist(self):  # a dead process writes nothing more
        return None

    monkeypatch.setattr(Dispatcher, "_final_persist", no_final_persist)
    task = asyncio.create_task(env.dispatcher(client).run(plan, allow_paid=False, max_usd=D("5")))
    waiter = asyncio.create_task(crashed.wait())
    await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    task.cancel()
    with suppress(asyncio.CancelledError, Crash):
        await task
    waiter.cancel()
    monkeypatch.undo()


async def test_crash_after_provider_response_before_db_commit_resends_nothing(tmp_path, monkeypatch):
    env = Env.create(tmp_path)
    graph = chain_graph(length=2)
    root, child = graph.nodes
    crashed = asyncio.Event()

    def crash_after_raw(paths, attempt_id, outcome):
        out = real_write_raw(paths, attempt_id, outcome)
        if outcome.result.node_id == root.node_id:
            crashed.set()
            raise Crash()
        return out

    monkeypatch.setattr(scheduler_mod, "write_raw_response", crash_after_raw)
    first = FakeClient()
    await _crash_run(env, first, plan_of([graph]), monkeypatch, crashed)
    assert first.sent_nodes() == [root.node_id]
    assert env.paths.raw_response(f"{root.node_id}:1").exists()
    assert not env.paths.parsed(root.node_id).exists()
    assert _kinds(env) == {f"{root.node_id}:1": None}               # reserved + sent, never settled

    env.reopen(recover=True)                                         # restore -> re-ingest raw -> reconcile journal
    assert env.store.recovery_report["raw_persisted"] == 1
    assert env.store.recovery_report["journal_settled_from_raw"] == 1
    second = FakeClient()
    summary = await env.dispatcher(second).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    assert second.sent_nodes() == [child.node_id]                    # the persisted root is never re-sent
    assert env.render.seen[child.node_id] == ["42"]
    assert summary.settled_usd == 2 * COST and summary.unresolved_usd == 0
    report = ledger_reconciliation(env.paths)
    assert report["discrepancies"] == [] and report["charged_usd"] == str(2 * COST)


async def test_crash_while_request_in_flight_charges_the_full_reservation(tmp_path, monkeypatch):
    env = Env.create(tmp_path)
    graph = chain_graph(length=1)
    crashed = asyncio.Event()

    def script(spec, attempt_id, index):
        crashed.set()
        return Behavior(delay=30)

    await _crash_run(env, FakeClient(script), plan_of([graph]), monkeypatch, crashed)
    [aid] = _kinds(env)
    env.reopen(recover=True)
    assert env.store.recovery_report["journal_charged_max"] == 1
    kinds = _kinds(env)
    assert kinds[aid] is LedgerEventKind.CHARGE_MAX
    charged = journal_attempts(env.paths.ledger_journal)[aid].resolved_usd
    assert env.ledger.snapshot().settled == charged > 0             # outstanding cost did not disappear


async def test_timeout_with_later_bill_settles_from_generation_metadata(tmp_path):
    env = Env.create(tmp_path, lease_stale_seconds=3)
    graph = chain_graph(length=2)
    root, child = graph.nodes

    def script(spec, attempt_id, index):
        if attempt_id == f"{root.node_id}:1":
            return Behavior(status=TransportStatus.AMBIGUOUS_TIMEOUT, cost=None, generation_id="gen-late")
        return Behavior(delay=0.9 if spec.node_id == child.node_id else 0.0)

    client = FakeClient(script)
    client.metadata["gen-late"] = [None, {"total_cost": "0.00005"}]
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 2
    assert [a for _, a in client.calls] == [f"{root.node_id}:1", f"{root.node_id}:2", f"{child.node_id}:1"]
    attempts = journal_attempts(env.paths.ledger_journal)
    late = attempts[f"{root.node_id}:1"]
    assert late.resolution is LedgerEventKind.SETTLE and late.resolved_usd == D("0.00005")
    assert summary.unresolved_usd == 0 and summary.settled_usd == D("0.00005") + 2 * COST
    assert client.metadata_calls.count("gen-late") >= 2
    unresolved = [e for e in env.journal.replay() if e.kind is LedgerEventKind.UNRESOLVED]
    assert [e.attempt_id for e in unresolved] == [f"{root.node_id}:1"] and unresolved[0].amount_usd == 0


async def test_timeout_without_a_bill_is_charged_at_the_maximum(tmp_path):
    env = Env.create(tmp_path, lease_stale_seconds=0.3)
    graph = chain_graph(length=1)

    def script(spec, attempt_id, index):
        if attempt_id.endswith(":1"):
            return Behavior(status=TransportStatus.AMBIGUOUS_TIMEOUT, cost=None)
        return Behavior(delay=0.8)

    summary = await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    attempts = journal_attempts(env.paths.ledger_journal)
    timed_out = attempts[f"{graph.nodes[0].node_id}:1"]
    assert timed_out.resolution is LedgerEventKind.CHARGE_MAX and timed_out.resolved_usd == timed_out.reserved_usd
    assert summary.settled_usd == timed_out.reserved_usd + COST and summary.unresolved_usd == 0


async def _simulate_foreign_sent_lease(env: Env, node, *, owner: str, heartbeat_age: float) -> str:
    aid = f"{node.node_id}:1"
    assert await env.ledger.reserve(attempt_id=aid, node_id=node.node_id, partition_id="main", amount_usd=D("0.002"))
    await env.ledger.mark_sent(aid)
    env.store.register_nodes([node], wave_index=0)
    assert env.store.try_lease(node.node_id, aid, owner=owner, now=time.time() - heartbeat_age)
    env.store.owner = owner
    env.store.record_attempt(Attempt(aid, node.node_id, 1, AttemptState.SENT, D("0.002"), "t0", sent_at="t1"))
    env.store.heartbeat(owner, now=time.time() - heartbeat_age)
    return aid


async def test_stale_sent_lease_is_charged_and_retried_as_a_new_attempt(tmp_path):
    env = Env.create(tmp_path, lease_stale_seconds=5)
    graph = chain_graph(length=2)
    root, child = graph.nodes
    aid = await _simulate_foreign_sent_lease(env, root, owner="otherhost|4242|99|abc", heartbeat_age=60)
    env.reopen(recover=False)                    # same DB; new ledger replays the open reservation
    assert env.ledger.snapshot().unresolved == D("0.002")
    client = FakeClient()
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    assert [a for _, a in client.calls] == [f"{root.node_id}:2", f"{child.node_id}:1"]
    assert _kinds(env)[aid] is LedgerEventKind.CHARGE_MAX
    assert summary.settled_usd == D("0.002") + 2 * COST
    assert len([a for a in env.store.attempts_for(root.node_id)]) == 2


async def test_fresh_foreign_lease_is_never_sent_concurrently(tmp_path):
    env = Env.create(tmp_path, lease_stale_seconds=1.0, lease_heartbeat_seconds=0.1)
    graph = make_graph("f", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())
    held, free = graph.nodes
    await _simulate_foreign_sent_lease(env, held, owner="otherhost|4242|99|abc", heartbeat_age=0)
    env.reopen(recover=False)
    client = FakeClient()
    started = time.monotonic()
    sent_at: dict[str, float] = {}

    def script(spec, attempt_id, index):
        sent_at[spec.node_id] = time.monotonic() - started
        return Behavior()

    client.script = script
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    assert sent_at[free.node_id] < 0.5
    assert sent_at[held.node_id] >= 0.8                              # waited for the lease to go stale
    assert [a for n, a in client.calls if n == held.node_id] == [f"{held.node_id}:2"]


async def test_429_burst_halves_once_honors_retry_after_and_bounds_attempts(tmp_path):
    env = Env.create(tmp_path, concurrency_start=8, max_calls_per_item_in_flight=8)
    graphs = [make_graph(f"r{i}", bank=4, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(3)]
    times: list[tuple[float, str, str]] = []
    start = time.monotonic()

    def script(spec, attempt_id, index):
        times.append((time.monotonic() - start, spec.node_id, attempt_id))
        if index < 8:
            return Behavior(status=TransportStatus.RATE_LIMITED, cost=None, retry_after=0.3, delay=0.05)
        return Behavior(delay=0.01)

    client = FakeClient(script)
    summary = await env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 12
    assert summary.rate_limited == 8
    assert summary.extra["aimd_decreases"] == 1 and summary.concurrency_final == 4 and summary.rps_final == 500
    first_429_done = min(t for t, _, a in times[:8]) + 0.05
    retries = [t for t, _, a in times if a.endswith(":2")]
    assert retries and min(retries) >= first_429_done + 0.3 - 0.02   # Retry-After honoured before any retry
    assert max(int(a.rsplit(":", 1)[1]) for _, _, a in times) <= 3
    assert summary.settled_usd == 12 * COST                            # refused requests cost nothing


async def test_exhausted_node_is_terminal_without_candidate_and_children_still_run(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.01)
    env = Env.create(tmp_path)
    graph = chain_graph(length=2)
    root, child = graph.nodes

    def script(spec, attempt_id, index):
        if spec.node_id == root.node_id:
            return Behavior(status=TransportStatus.UPSTREAM_ERROR, cost=None)
        return Behavior()

    client = FakeClient(script)
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    assert [a for n, a in client.calls if n == root.node_id] == [f"{root.node_id}:{i}" for i in (1, 2, 3)]
    assert summary.nodes_failed == 1 and summary.nodes_succeeded == 1
    assert env.render.seen[child.node_id] == [ABSENT]
    row = env.store.node(root.node_id)
    assert row.state == "terminal" and row.outcome == "exhausted" and "upstream_error" in row.terminal_reason
    # restart: the exhausted node is terminal, nothing is re-sent
    env.reopen(recover=True)
    again = FakeClient()
    assert (await env.dispatcher(again).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))).requests_sent == 0


async def test_slow_grader_does_not_hold_http_capacity(tmp_path):
    env = Env.create(tmp_path, concurrency_start=16, max_calls_per_item_in_flight=8)
    graphs = [make_graph(f"s{i}", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(6)]

    def slow_extract(result, spec):
        time.sleep(0.05)                                   # CPU-bound grading/extraction in a worker thread
        return fake_extract(result, spec)

    client = FakeClient(delay=0.1)
    started = time.monotonic()
    summary = await env.dispatcher(client, extract=slow_extract).run(plan_of(graphs), allow_paid=False,
                                                                    max_usd=D("5"))
    elapsed = time.monotonic() - started
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 36
    assert summary.peak_in_flight == 16 == client.peak                 # extraction backlog never starved HTTP work
    assert elapsed < 36 * (0.1 + 0.05) / 2


async def test_duplicate_completion_notification_is_idempotent(tmp_path):
    env = Env.create(tmp_path)
    graph = chain_graph(length=2)
    root, child = graph.nodes
    dispatcher = env.dispatcher(FakeClient())
    summary = await dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    parsed_before = env.paths.parsed(root.node_id).read_bytes()
    snapshot_before = env.ledger.snapshot()
    node = dispatcher.nodes[root.node_id]
    [attempt] = env.store.attempts_for(root.node_id)
    result = env.store.result(root.node_id)
    await dispatcher._finish_ok(node, attempt, result)                # the same completion delivered again
    other = replace(result, attempt_id=f"{root.node_id}:9")
    assert env.store.commit_result(other, fake_extract(other, root)) is False
    assert dispatcher.duplicate_notifications == 1 and env.store.duplicate_results == 1
    assert dispatcher.nodes[child.node_id].waiting_parents == 0
    assert env.paths.parsed(root.node_id).read_bytes() == parsed_before
    assert env.ledger.snapshot() == snapshot_before
    await env.ledger.settle(f"{root.node_id}:1", COST)                 # duplicate settlement notification
    assert env.ledger.snapshot() == snapshot_before


async def test_disk_failure_opens_circuit_keeps_cost_and_resends_only_unpersisted(tmp_path, monkeypatch):
    env = Env.create(tmp_path, concurrency_start=4)
    graphs = [make_graph(f"d{i}", bank=3, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(2)]
    victim = graphs[0].nodes[0]

    def failing_write(paths, attempt_id, outcome):
        if outcome.result.node_id == victim.node_id:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write_raw(paths, attempt_id, outcome)

    monkeypatch.setattr(scheduler_mod, "write_raw_response", failing_write)
    client = FakeClient(lambda spec, aid, i: Behavior(delay=0.05 if spec.node_id != victim.node_id else 0.0))
    summary = await env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "circuit_open" and summary.circuit_reason.startswith("disk_failure")
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0
    assert _kinds(env)[f"{victim.node_id}:1"] is LedgerEventKind.SETTLE   # money truth kept despite the disk
    assert not env.paths.parsed(victim.node_id).exists()
    assert env.store.node(victim.node_id).state == "pending"
    assert len(client.calls) < 6                                       # admissions stopped
    for node_id in client.completed:
        if node_id != victim.node_id:
            assert env.paths.parsed(node_id).exists()                  # in-flight work drained and persisted
    monkeypatch.undo()
    env.reopen(recover=True)
    again = FakeClient()
    summary2 = await env.dispatcher(again).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert summary2.stop_reason == "completed"
    assert victim.node_id in again.sent_nodes()
    assert set(again.sent_nodes()).isdisjoint(set(client.completed) - {victim.node_id})
    assert (f"{victim.node_id}:2") in [a for _, a in again.calls]


async def test_graceful_drain_finishes_in_flight_and_sentinel_blocks_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "CONTROL_POLL_SECONDS", 0.02)
    env = Env.create(tmp_path, concurrency_start=4, max_calls_per_item_in_flight=4)
    graphs = [make_graph(f"g{i}", bank=5, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(4)]
    client = FakeClient(delay=0.3)
    run = asyncio.create_task(env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.1)
    assert cmd_stop(["--run-dir", str(env.paths.run_dir), "--drain"]) == 0
    sent_at_drain = len(client.calls)
    summary = await run
    assert summary.stop_reason == "drain"
    assert len(client.calls) <= sent_at_drain + 1 and len(client.completed) == len(client.calls)
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0
    assert all(env.paths.parsed(n).exists() for n in client.completed)
    blocked = await env.dispatcher(FakeClient()).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert blocked.stop_reason == "control_sentinel_present" and blocked.requests_sent == 0
    assert cmd_stop(["--run-dir", str(env.paths.run_dir), "--clear"]) == 0
    env.reopen(recover=True)
    resumed_client = FakeClient()
    resumed = await env.dispatcher(resumed_client).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert resumed.stop_reason == "completed"
    assert set(resumed_client.sent_nodes()).isdisjoint(client.completed)
    assert len(resumed_client.calls) + len(client.completed) == 20


async def test_stop_abandons_in_flight_but_never_releases_their_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "CONTROL_POLL_SECONDS", 0.02)
    monkeypatch.setattr(scheduler_mod, "STOP_GRACE_SECONDS", 0.1)
    env = Env.create(tmp_path, concurrency_start=3)
    graph = make_graph("slow", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(delay=20)
    run = asyncio.create_task(env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.15)
    cmd_stop(["--run-dir", str(env.paths.run_dir)])
    summary = await asyncio.wait_for(run, timeout=10)
    assert summary.stop_reason == "stop_sentinel" and len(client.calls) == 3
    kinds = list(journal_attempts(env.paths.ledger_journal).values())
    charged = [k for k in kinds if k.resolution is LedgerEventKind.CHARGE_MAX]
    released = [k for k in kinds if k.resolution is LedgerEventKind.RELEASE_UNSENT]
    # the three requests on the wire are charged in full; the prefetched attempts that were still waiting for the
    # HTTP limiter never left the process (journal SENT = committed to send) and are released, never charged
    assert len(charged) == 3 and len(released) == len(kinds) - 3
    assert {k.node_id for k in charged} == set(client.sent_nodes())
    assert summary.settled_usd == sum(k.reserved_usd for k in charged)
    assert sum(env.store.sends_by_node().values()) == 3
    assert sum(1 for n in graph.nodes if env.store.node(n.node_id).state == "pending") == 6
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0


async def test_payment_required_opens_the_circuit_and_drains(tmp_path):
    env = Env.create(tmp_path, concurrency_start=2)
    graphs = [make_graph(f"p{i}", bank=4, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(2)]

    def script(spec, attempt_id, index):
        if index == 1:
            return Behavior(status=TransportStatus.PAYMENT_REQUIRED, cost=None)
        return Behavior(delay=0.1)

    client = FakeClient(script)
    summary = await env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "circuit_open" and summary.circuit_reason.startswith("payment_required")
    assert len(client.calls) <= 3 and len(client.completed) == len(client.calls)
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0
    payment_node = client.calls[1][0]
    assert env.store.node(payment_node).state == "pending"            # not a model outcome: re-runnable later


async def test_unresolved_exposure_pauses_admissions_until_reconciled(tmp_path):
    env = Env.create(tmp_path, concurrency_start=2, lease_stale_seconds=0.6)
    env.config["budget"]["max_unresolved_spend_usd"] = 0.0008      # one ~0.00092 USD timeout exceeds it
    env.reopen(recover=False)
    graph = make_graph("u", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=())
    sends: list[tuple[float, str]] = []
    start = time.monotonic()

    def script(spec, attempt_id, index):
        sends.append((time.monotonic() - start, attempt_id))
        if index < 2:
            return Behavior(status=TransportStatus.AMBIGUOUS_TIMEOUT, cost=None)
        return Behavior(delay=0.05)

    summary = await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 6
    events = [e for _, e, _ in summary.extra["events"]]
    assert "paused_unresolved_exposure" in events
    # the first burst (2 in flight + 2 prefetched, all reserved before any timeout) may send; nothing admitted
    # after the timeouts pushed unresolved exposure over the limit is sent before the horizon charges them
    assert len(sends) == 8 and all(t >= 0.5 for t, _ in sends[4:]), sends
    charged = [k for k in journal_attempts(env.paths.ledger_journal).values()
               if k.resolution is LedgerEventKind.CHARGE_MAX]
    assert len(charged) == 2


async def test_extract_failure_persists_response_and_restart_finishes_without_resend(tmp_path):
    env = Env.create(tmp_path)
    graph = chain_graph(length=2)
    root, child = graph.nodes

    def broken_extract(result, spec):
        raise ValueError("extractor bug")

    client = FakeClient()
    summary = await env.dispatcher(client, extract=broken_extract).run(plan_of([graph]), allow_paid=False,
                                                                       max_usd=D("5"))
    assert summary.stop_reason == "circuit_open" and summary.circuit_reason.startswith("extract_failed")
    assert env.store.node(root.node_id).state == "response_persisted"
    env.reopen(recover=True)
    again = FakeClient()
    fixed = await env.dispatcher(again).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert fixed.stop_reason == "completed" and again.sent_nodes() == [child.node_id]
    assert fixed.settled_usd == 2 * COST


async def test_context_limit_makes_the_node_terminal_without_a_call_or_reservation(tmp_path):
    from agent_allocation.prompts.render import ContextLimitEvent

    env = Env.create(tmp_path)
    graph = chain_graph(length=3)
    root, middle, leaf = graph.nodes
    inner = env.render

    def render(spec, parents):
        if spec.node_id == middle.node_id:
            raise ContextLimitEvent(source_node_id=root.node_id, final_chars=9000, visible_chars=9100,
                                    envelope_chars=4000, receiver_node_id=spec.node_id)
        return inner(spec, parents)

    client = FakeClient()
    dispatcher = env.dispatcher(client)
    dispatcher.render = render
    summary = await dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_context_limited == 1
    assert client.sent_nodes() == [root.node_id, leaf.node_id]
    assert not any(a.startswith(middle.node_id) for a in _kinds(env))       # no reservation, no API call
    row = env.store.node(middle.node_id)
    assert row.state == "terminal" and row.outcome == "context_limit" and env.store.candidate(middle.node_id) is None
    assert env.render.seen[leaf.node_id] == [ABSENT]


async def test_error_envelope_is_retried_and_its_reservation_kept_until_reconciled(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.01)
    env = Env.create(tmp_path, lease_stale_seconds=0.3)
    graph = chain_graph(length=1)
    [root] = graph.nodes

    def script(spec, attempt_id, index):
        if index == 0:
            return Behavior(status=TransportStatus.ERROR_ENVELOPE, cost=None)
        return Behavior(delay=0.6)

    client = FakeClient(script)
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 1
    assert [a for _, a in client.calls] == [f"{root.node_id}:1", f"{root.node_id}:2"]
    envelope = journal_attempts(env.paths.ledger_journal)[f"{root.node_id}:1"]
    assert envelope.sent and envelope.resolution is LedgerEventKind.CHARGE_MAX
    assert env.paths.raw_response(f"{root.node_id}:1").exists()             # the error payload is preserved


async def test_connect_failure_settles_at_zero_and_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.01)
    env = Env.create(tmp_path)
    graph = chain_graph(length=1)
    [root] = graph.nodes

    def script(spec, attempt_id, index):
        return Behavior(status=TransportStatus.CONNECT_FAILED, cost=None) if index == 0 else Behavior()

    summary = await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed"
    failed = journal_attempts(env.paths.ledger_journal)[f"{root.node_id}:1"]
    assert failed.sent and failed.resolution is LedgerEventKind.SETTLE and failed.resolved_usd == 0
    from tests.integration.reference_impl import LedgerChecker
    assert LedgerChecker(D("5")).replay(list(env.journal.replay())).violations == []


async def test_urgent_sent_does_not_wait_out_the_group_commit_window(tmp_path):
    from agent_allocation.runtime.ledger import CollectionLedger
    from agent_allocation.storage.journal import LedgerJournal

    journal = LedgerJournal(tmp_path / "j.jsonl", group_commit_ms=400)
    ledger = CollectionLedger(journal, ceiling_usd=D("1"), max_unresolved_usd=D("2"))
    started = time.monotonic()
    assert await ledger.reserve(attempt_id="a:1", node_id="a", partition_id="main", amount_usd=D("0.1"))
    reserve_latency = time.monotonic() - started
    started = time.monotonic()
    await ledger.mark_sent("a:1")
    sent_latency = time.monotonic() - started
    assert reserve_latency >= 0.35 and sent_latency < 0.2
