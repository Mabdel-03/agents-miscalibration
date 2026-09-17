"""Pilot finding (kill -9 resume): two journal writers charged the same attempts with duplicate seqs. The journal now
has one writer (in-process registry + flock) that catches up before every transition, recovery never writes it, and
the ledger is the sole owner of recovery charging."""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from agent_allocation.contracts import LedgerEventKind
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.ledger import CollectionLedger
from agent_allocation.storage.journal import JournalError, LedgerJournal, replay_events
from agent_allocation.storage.state import StateStore
from tests.integration.reference_impl import LedgerChecker
from tests.unit.runtime.fakes import Env, FakeClient, make_config, make_graph, plan_of

D = Decimal
HERE = Path(__file__).resolve().parent


def _terminal_counts(path: Path) -> Counter:
    return Counter(e.attempt_id for e in replay_events(path)
                   if e.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX, LedgerEventKind.RELEASE_UNSENT))


def _assert_journal_sound(path: Path, ceiling: Decimal = D("5")) -> list:
    events = list(replay_events(path))                       # raises on any seq regression or malformed line
    seqs = [e.seq for e in events]
    assert seqs == sorted(set(seqs)) and seqs == list(range(1, len(seqs) + 1))
    assert all(n == 1 for n in _terminal_counts(path).values())
    assert LedgerChecker(ceiling).replay(events).violations == []
    return events


async def _leave_sent_attempts(paths: RunPaths, n: int) -> list[str]:
    journal = LedgerJournal(paths.ledger_journal, group_commit_ms=0)
    ledger = CollectionLedger(journal, ceiling_usd=D("5"), max_unresolved_usd=D("2"))
    ids = []
    for i in range(n):
        aid = f"{i:064x}:1"
        assert await ledger.reserve(attempt_id=aid, node_id=aid[:64], partition_id="main", amount_usd=D("0.001155"))
        await ledger.mark_sent(aid)
        ids.append(aid)
    journal.release_writer()                                  # the process "dies" with six requests on the wire
    return ids


@pytest.mark.parametrize("order", ["ledger_then_recover", "recover_then_ledger"])
async def test_ledger_and_recover_in_one_process_charge_each_attempt_once(tmp_path, order):
    paths = RunPaths.for_run(run_dir=tmp_path / "run", local_state_dir=tmp_path / "local")
    paths.ensure()
    ids = await _leave_sent_attempts(paths, 6)
    if order == "ledger_then_recover":
        ledger = CollectionLedger(LedgerJournal(paths.ledger_journal), ceiling_usd=D("5"), max_unresolved_usd=D("2"))
        store = StateStore.recover(paths)
    else:
        store = StateStore.recover(paths)
        ledger = CollectionLedger(LedgerJournal(paths.ledger_journal), ceiling_usd=D("5"), max_unresolved_usd=D("2"))
    assert store.recovery_report["journal_open_reservations"] == 6
    assert _terminal_counts(paths.ledger_journal) == Counter()          # recovery wrote nothing
    assert ledger.snapshot().unresolved == D("0.001155") * 6
    # a second, stale ledger in the same process (the pilot's shape) must not double-charge either
    stale = CollectionLedger(LedgerJournal(paths.ledger_journal), ceiling_usd=D("5"), max_unresolved_usd=D("2"))
    assert await ledger.resolve_inherited(paths) == {"settled_from_raw": 0, "charged_max": 6, "released_unsent": 0}
    assert await stale.resolve_inherited(paths) == {"settled_from_raw": 0, "charged_max": 0,
                                                    "released_unsent": 0}                    # caught up first
    assert stale.snapshot() == ledger.snapshot() and stale.snapshot().settled == D("0.001155") * 6
    events = _assert_journal_sound(paths.ledger_journal)
    assert [e.kind for e in events].count(LedgerEventKind.CHARGE_MAX) == 6
    assert set(_terminal_counts(paths.ledger_journal)) == set(ids)
    store.close()


async def test_a_busy_writer_in_this_process_refuses_a_second_writer(tmp_path):
    path = tmp_path / "journal.jsonl"
    first = CollectionLedger(LedgerJournal(path, group_commit_ms=300), ceiling_usd=D("1"), max_unresolved_usd=D("2"))
    second = CollectionLedger(LedgerJournal(path, group_commit_ms=0), ceiling_usd=D("1"), max_unresolved_usd=D("2"))
    pending = asyncio.create_task(first.reserve(attempt_id="a:1", node_id="a", partition_id="main",
                                                amount_usd=D("0.1")))
    await asyncio.sleep(0.05)                                  # first holds the role with an append lingering
    with pytest.raises(JournalError):
        await second.reserve(attempt_id="b:1", node_id="b", partition_id="main", amount_usd=D("0.1"))
    assert await pending
    # once idle, the role moves and the second ledger catches up before deciding
    assert await second.reserve(attempt_id="b:1", node_id="b", partition_id="main", amount_usd=D("0.1"))
    # a:1 (first's, reserved only) arrived through the take-over catch-up; b:1 is second's own
    assert second.snapshot().in_flight == D("0.2") and second.inherited_attempt_ids == {"a:1"}
    with pytest.raises(Exception):
        await second.reserve(attempt_id="a:1", node_id="a", partition_id="main", amount_usd=D("0.1"))
    _assert_journal_sound(path)


def test_another_process_holding_the_writer_lock_is_refused(tmp_path):
    path = tmp_path / "journal.jsonl"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT); "
         "fcntl.flock(fd, fcntl.LOCK_EX); print('LOCKED', flush=True); time.sleep(30)", str(path) + ".lock"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        journal = LedgerJournal(path, group_commit_ms=0)
        with pytest.raises(JournalError):
            journal.ensure_writer()
    finally:
        holder.kill()
        holder.wait()
    LedgerJournal(path, group_commit_ms=0).ensure_writer()   # released with the process


async def test_kill_9_with_requests_in_flight_then_resume_charges_each_once(tmp_path):
    env_dir = tmp_path / "pilot"
    proc = subprocess.Popen([sys.executable, str(HERE / "killable_dispatcher.py"), "--tmp", str(env_dir),
                             "--inflight", "6"], stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE.parents[2]),
                                                                              str(HERE.parents[2] / "src")])})
    try:
        line = await asyncio.wait_for(asyncio.to_thread(proc.stdout.readline), timeout=60)
        assert line.strip() == "READY", line
        await asyncio.sleep(0.3)                               # the sixth SENT fsync lands
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    paths = RunPaths.for_run(run_dir=env_dir / "run", local_state_dir=env_dir / "local")
    open_before = [e.attempt_id for e in replay_events(paths.ledger_journal) if e.kind is LedgerEventKind.SENT]
    assert len(open_before) == 6 and _terminal_counts(paths.ledger_journal) == Counter()
    for order in ("ledger_then_recover", "recover_then_ledger"):
        work = tmp_path / order
        shutil.copytree(env_dir, work)
        wpaths = RunPaths.for_run(run_dir=work / "run", local_state_dir=work / "local")
        # the lead's Runtime shape: journal+ledger and StateStore.recover in one process, in either order
        if order == "ledger_then_recover":
            ledger = CollectionLedger(LedgerJournal(wpaths.ledger_journal), ceiling_usd=D("5"),
                                      max_unresolved_usd=D("2"))
            store = StateStore.recover(wpaths)
        else:
            store = StateStore.recover(wpaths)
            ledger = CollectionLedger(LedgerJournal(wpaths.ledger_journal), ceiling_usd=D("5"),
                                      max_unresolved_usd=D("2"))
        env = Env(work, wpaths, make_config(lease_stale_seconds=5), store=store, journal=ledger.journal,
                  ledger=ledger)
        client = FakeClient()
        graph = make_graph("killed", bank=10, chain=1, team_slots=1, peer_sizes=(), deep=())
        summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=D("5"))
        assert summary.stop_reason == "completed" and summary.nodes_succeeded == 10
        events = _assert_journal_sound(wpaths.ledger_journal)
        charged = [e.attempt_id for e in events if e.kind is LedgerEventKind.CHARGE_MAX]
        released = [e.attempt_id for e in events if e.kind is LedgerEventKind.RELEASE_UNSENT]
        assert sorted(charged) == sorted(open_before)             # every killed in-flight (SENT) attempt charged once
        assert len(released) == 2 and not set(released) & set(open_before)   # prefetched, never sent: released
        assert summary.unresolved_usd == 0 and summary.in_flight_usd == 0
        assert len(client.calls) == 10                           # the 6 killed attempts retried as attempt 2
        store.close()
