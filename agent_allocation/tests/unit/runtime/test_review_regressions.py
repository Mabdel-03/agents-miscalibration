"""Regressions from the independent review (repro scripts: scratchpad/review/repro_*.py)."""
from __future__ import annotations

import asyncio
import shutil
from collections import Counter
from contextlib import suppress
from dataclasses import replace
from decimal import Decimal

import agent_allocation.runtime.scheduler as scheduler_mod
from agent_allocation.contracts import Attempt, AttemptState, LedgerEventKind, TransportStatus
from agent_allocation.runtime.ledger import journal_attempts
from agent_allocation.storage.raw import write_raw_response as real_write_raw
from tests.unit.runtime.fakes import Behavior, Env, FakeClient, make_graph, make_outcome, plan_of

D = Decimal


def one_root(item: str = "x"):
    return make_graph(item, bank=1, chain=1, team_slots=1, peer_sizes=(), deep=())


def terminal_events_per_node(env: Env) -> Counter:
    return Counter(info.node_id for info in journal_attempts(env.paths.ledger_journal).values()
                   if info.resolution in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX))


# (1) DOUBLE-CHARGE: a cancelled run() must never re-send a node whose OK response is already on disk ---------------

async def _cancel_after_raw_write(env: Env, graph, monkeypatch) -> None:
    written = asyncio.Event()
    loop = asyncio.get_running_loop()

    def write_then_signal(paths, attempt_id, outcome):
        out = real_write_raw(paths, attempt_id, outcome)
        loop.call_soon_threadsafe(written.set)
        return out

    monkeypatch.setattr(scheduler_mod, "write_raw_response", write_then_signal)
    run = asyncio.create_task(env.dispatcher(FakeClient()).run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await written.wait()
    run.cancel()
    with suppress(asyncio.CancelledError):
        await run
    monkeypatch.undo()


async def test_cancelled_run_keeps_a_persisted_response_and_restart_resends_nothing(tmp_path, monkeypatch):
    env = Env.create(tmp_path)
    env.group_commit_ms = 250                                   # production linger widens the window
    env.reopen(recover=False)
    graph = one_root()
    [root] = graph.nodes
    await _cancel_after_raw_write(env, graph, monkeypatch)
    assert env.paths.raw_response(f"{root.node_id}:1").exists()
    assert env.store.node(root.node_id).state == "response_persisted"          # not returned to pending
    env.reopen(recover=True)
    second = FakeClient()
    summary = await env.dispatcher(second).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and second.calls == []
    assert terminal_events_per_node(env) == Counter({root.node_id: 1}) and summary.settled_usd == D("0.0001")


async def test_recovery_reingests_ok_raw_even_when_the_attempt_row_says_terminal(tmp_path):
    env = Env.create(tmp_path)
    graph = one_root()
    [root] = graph.nodes
    aid = f"{root.node_id}:1"
    env.store.register_nodes([root], wave_index=0)
    env.store.record_attempt(Attempt(aid, root.node_id, 1, AttemptState.TERMINAL, D("0.001"), "t0", sent_at="t1",
                                     error_detail="lease_released_at_exit"))    # what the old exit path wrote
    real_write_raw(env.paths, aid, make_outcome(root, aid, [], Behavior()))
    assert env.store.node(root.node_id).state == "pending"
    env.reopen(recover=True)
    assert env.store.recovery_report["raw_persisted"] == 1
    assert env.store.node(root.node_id).state == "response_persisted"
    client = FakeClient()
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and client.calls == [] and summary.nodes_succeeded == 1


async def test_take_over_with_failing_extraction_keeps_the_response(tmp_path):
    env = Env.create(tmp_path, lease_stale_seconds=5)
    graph = one_root()
    [root] = graph.nodes
    aid = f"{root.node_id}:1"
    assert await env.ledger.reserve(attempt_id=aid, node_id=root.node_id, partition_id="main", amount_usd=D("0.001"))
    await env.ledger.mark_sent(aid)
    env.store.register_nodes([root], wave_index=0)
    host = __import__("socket").gethostname()
    assert env.store.try_lease(root.node_id, aid, owner=f"{host}|999999999|0|dead")
    env.store.record_attempt(Attempt(aid, root.node_id, 1, AttemptState.SENT, D("0.001"), "t0", sent_at="t1"))
    real_write_raw(env.paths, aid, make_outcome(root, aid, [], Behavior()))
    env.journal.release_writer()
    env.reopen(recover=False)

    def broken(result, spec):
        raise ValueError("extractor bug")

    first = FakeClient()
    s1 = await env.dispatcher(first, extract=broken).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert s1.stop_reason == "circuit_open" and first.calls == []
    assert env.store.node(root.node_id).state == "response_persisted"          # never back to pending
    second = FakeClient()
    s2 = await env.dispatcher(second).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert s2.stop_reason == "completed" and second.calls == []
    assert terminal_events_per_node(env) == Counter({root.node_id: 1})


# (2) EXTRA SEND: max_transport_attempts holds across runs ---------------------------------------------------------

async def test_attempt_limit_holds_after_a_circuit_breaking_third_send(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.01)
    env = Env.create(tmp_path)
    graph = one_root()
    [root] = graph.nodes
    first = FakeClient(lambda s, aid, i: Behavior(status=TransportStatus.UPSTREAM_ERROR if i < 2
                                                  else TransportStatus.PAYMENT_REQUIRED, cost=None))
    s1 = await env.dispatcher(first).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert s1.stop_reason == "circuit_open" and len(first.calls) == 3
    env.reopen(recover=True)
    second = FakeClient()
    s2 = await env.dispatcher(second).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert second.calls == [] and s2.nodes_failed == 1
    row = env.store.node(root.node_id)
    assert row.state == "terminal" and row.outcome == "exhausted"


async def test_attempt_limit_holds_after_restoring_an_older_backup(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.01)
    env = Env.create(tmp_path)
    graph = one_root()
    [root] = graph.nodes
    env.store.register_nodes([root], wave_index=0)
    env.store.backup()
    old_backup = tmp_path / "old_backup.sqlite"
    shutil.copy(env.paths.state_db_backup, old_backup)
    first = FakeClient(lambda s, aid, i: Behavior(status=TransportStatus.UPSTREAM_ERROR, cost=None,
                                                  generation_id="gen-x"))
    await env.dispatcher(first).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert len(first.calls) == 3
    env.store.close()
    for suffix in ("", "-wal", "-shm"):
        path = env.paths.state_db_local.with_name(env.paths.state_db_local.name + suffix)
        if path.exists():
            path.unlink()
    shutil.copy(old_backup, env.paths.state_db_backup)
    env.open(recover=True)
    second = FakeClient()
    await env.dispatcher(second).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert second.calls == [] and env.store.node(root.node_id).outcome == "exhausted"


async def test_recovery_takes_sent_from_the_journal_when_the_backup_says_leased(tmp_path):
    env = Env.create(tmp_path)
    graph = one_root()
    [root] = graph.nodes
    aid = f"{root.node_id}:1"
    env.store.register_nodes([root], wave_index=0)
    assert env.store.try_lease(root.node_id, aid, owner="otherhost|1|1|x")
    env.store.record_attempt(Attempt(aid, root.node_id, 1, AttemptState.LEASED, D("0.001"), "t0"))
    assert env.store.sends_by_node() == {}
    assert await env.ledger.reserve(attempt_id=aid, node_id=root.node_id, partition_id="main", amount_usd=D("0.001"))
    await env.ledger.mark_sent(aid)
    env.journal.release_writer()
    env.reopen(recover=True)
    assert env.store.sends_by_node() == {root.node_id: 1}


# (3) STUCK RUN: settlements below their reservations lift budget blocks --------------------------------------------

async def test_soft_ceiling_block_lifts_when_admitted_work_settles_below_its_reservation(tmp_path):
    env = Env.create(tmp_path, concurrency_start=16, max_calls_per_item_in_flight=16)
    graph = make_graph("b", bank=20, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(lambda spec, aid, i: Behavior(delay=0.01, cost=D("0.0001")))
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("0.005"))
    assert summary.stop_reason == "completed", (summary.stop_reason, summary.stop_detail)
    assert summary.requests_sent == 20 and summary.settled_usd == D("0.0020")
    assert env.ledger.peak_exposure <= D("0.005")


async def test_pending_reservations_are_not_counted_twice_during_the_journal_linger(tmp_path):
    env = Env.create(tmp_path, concurrency_start=8, max_calls_per_item_in_flight=16)
    env.group_commit_ms = 300
    env.reopen(recover=False)
    graph = make_graph("p", bank=8, chain=1, team_slots=1, peer_sizes=(), deep=())
    dispatcher = env.dispatcher(FakeClient(delay=0.01))
    run = asyncio.create_task(dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.15)                                   # reservations lingering in the group commit
    snap = env.ledger.snapshot()
    assert dispatcher.pending_reservation == 0 and snap.in_flight > 0
    summary = await run
    assert summary.stop_reason == "completed"


# (4) the Retry-After and unresolved-exposure pauses also bind attempts already waiting for the HTTP limiter ---------

async def test_retry_after_pause_holds_attempts_that_were_admitted_before_the_429(tmp_path):
    import time

    env = Env.create(tmp_path, concurrency_start=8, max_calls_per_item_in_flight=16)
    graph = make_graph("ra", bank=16, chain=1, team_slots=1, peer_sizes=(), deep=())
    start = time.monotonic()
    sends: list[float] = []

    def script(spec, attempt_id, index):
        sends.append(time.monotonic() - start)
        if index < 8:
            return Behavior(status=TransportStatus.RATE_LIMITED, cost=None, retry_after=0.6, delay=0.05)
        return Behavior(delay=0.01)

    summary = await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 16
    answered = min(sends[:8]) + 0.05
    assert all(t >= answered + 0.6 - 0.02 for t in sends[8:]), [round(t, 3) for t in sends[8:]]
    released = [a for a in journal_attempts(env.paths.ledger_journal).values()
                if a.resolution is LedgerEventKind.RELEASE_UNSENT]
    assert released and not any(a.sent for a in released)         # the prefetched attempts never left


async def test_unresolved_pause_holds_attempts_waiting_for_the_http_limiter(tmp_path):
    import time

    env = Env.create(tmp_path, concurrency_start=2, lease_stale_seconds=0.6)
    env.config["budget"]["max_unresolved_spend_usd"] = 0.0008
    env.reopen(recover=False)
    graph = make_graph("u2", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=())
    start = time.monotonic()
    sends: list[float] = []

    def script(spec, attempt_id, index):
        sends.append(time.monotonic() - start)
        if index < 2:
            return Behavior(status=TransportStatus.AMBIGUOUS_TIMEOUT, cost=None)
        return Behavior(delay=0.05)

    summary = await env.dispatcher(FakeClient(script)).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 6
    assert len(sends) == 8 and all(t >= 0.5 for t in sends[2:]), [round(t, 3) for t in sends]


# (5) cancellation leaves no running exchange and no leaked reservation ----------------------------------------------

async def test_cancelled_run_leaves_no_http_exchange_running(tmp_path):
    env = Env.create(tmp_path, concurrency_start=2)
    graph = make_graph("cancel2", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(delay=1.0)
    run = asyncio.create_task(env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.2)
    run.cancel()
    with suppress(asyncio.CancelledError):
        await run
    await asyncio.sleep(0)
    assert client.active == 0
    assert not [t for t in asyncio.all_tasks() if "generate" in repr(t.get_coro())]
    await asyncio.sleep(1.1)
    assert client.completed == []                                   # nothing finished (and billed) behind our back


async def test_cancel_inside_the_reservation_linger_releases_and_never_reuses_attempt_ids(tmp_path):
    env = Env.create(tmp_path, concurrency_start=2)
    env.group_commit_ms = 250
    env.reopen(recover=False)
    graph = make_graph("leak", bank=4, chain=1, team_slots=1, peer_sizes=(), deep=())
    dispatcher = env.dispatcher(FakeClient(delay=0.05))
    run = asyncio.create_task(dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.05)
    run.cancel()
    with suppress(asyncio.CancelledError):
        await run
    assert env.ledger.snapshot().in_flight == 0 and env.ledger.snapshot().unresolved == 0
    assert all(a.resolution is not None for a in journal_attempts(env.paths.ledger_journal).values())
    client = FakeClient()
    dispatcher.client = client
    summary = await dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 4
    attempts = journal_attempts(env.paths.ledger_journal)
    assert all(a.resolution is not None for a in attempts.values())
    assert {aid for _, aid in client.calls} <= set(attempts) and len(set(a for _, a in client.calls)) == 4


# (6) STOP written during a drain takes over ------------------------------------------------------------------------

async def test_stop_written_during_a_drain_overrides_it(tmp_path, monkeypatch):
    import time

    from agent_allocation.runtime.control import cmd_stop

    monkeypatch.setattr(scheduler_mod, "CONTROL_POLL_SECONDS", 0.02)
    monkeypatch.setattr(scheduler_mod, "STOP_GRACE_SECONDS", 0.1)
    env = Env.create(tmp_path, concurrency_start=3)
    graph = make_graph("d", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(delay=6)
    run = asyncio.create_task(env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5")))
    await asyncio.sleep(0.2)
    cmd_stop(["--run-dir", str(env.paths.run_dir), "--drain"])
    await asyncio.sleep(0.3)
    escalated = time.monotonic()
    cmd_stop(["--run-dir", str(env.paths.run_dir)])
    summary = await asyncio.wait_for(run, timeout=5)
    assert time.monotonic() - escalated < 2.0
    assert summary.stop_reason == "stop_sentinel" and summary.stop_detail == "escalated_during_drain"
    charged = [a for a in journal_attempts(env.paths.ledger_journal).values() if a.sent]
    assert len(charged) == 3 and all(a.resolution is LedgerEventKind.CHARGE_MAX for a in charged)
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0
