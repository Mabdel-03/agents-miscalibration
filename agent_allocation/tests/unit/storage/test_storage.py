"""Raw artifacts, the ledger journal, and the SQLite state store."""
from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from agent_allocation.contracts import (
    Attempt, AttemptState, GenerationKind, LedgerEvent, LedgerEventKind, TransportStatus, dump_record,
)
from agent_allocation.paths import RunPaths
from agent_allocation.storage.journal import JournalError, LedgerJournal
from agent_allocation.storage.raw import RawArtifactConflict, read_raw_response, write_raw_response
from agent_allocation.storage.state import StateStore, read_parsed
from tests.unit.runtime.fakes import Behavior, fake_extract, make_outcome, spec


@pytest.fixture()
def paths(tmp_path):
    p = RunPaths.for_run(run_dir=tmp_path / "run", local_state_dir=tmp_path / "local")
    p.ensure()
    return p


def _event(seq: int, kind=LedgerEventKind.RESERVE, amount="0.01", attempt="a:1") -> LedgerEvent:
    return LedgerEvent(seq=seq, kind=kind, amount_usd=Decimal(amount), recorded_at="t", attempt_id=attempt,
                       node_id="n", partition_id="main")


def test_raw_write_is_atomic_deterministic_and_never_overwritten(paths):
    s = spec("i1", GenerationKind.ROOT)
    attempt_id = f"{s.node_id}:1"
    outcome = make_outcome(s, attempt_id, [{"role": "user", "content": "hi"}], Behavior())
    rel, sha = write_raw_response(paths, attempt_id, outcome)
    assert rel.startswith("raw/") and rel.endswith(".json.gz")
    assert (paths.run_dir / rel) == paths.raw_response(attempt_id)
    # duplicate completion notification: same bytes, same answer
    assert write_raw_response(paths, attempt_id, outcome) == (rel, sha)
    payload = read_raw_response(paths, rel, expected_sha256=sha)
    for key in ("attempt_id", "node_id", "request_body", "response_status", "response_headers", "response_body_text"):
        assert key in payload
    assert payload["result"]["attempt_id"] == attempt_id
    different = make_outcome(s, attempt_id, [{"role": "user", "content": "hi"}], Behavior(content="<final>7</final>"))
    with pytest.raises(RawArtifactConflict):
        write_raw_response(paths, attempt_id, different)
    assert not list(paths.raw_response(attempt_id).parent.glob(".*.tmp"))
    assert json.loads(gzip.decompress(paths.raw_response(attempt_id).read_bytes()))["node_id"] == s.node_id


async def test_journal_group_commit_fsync_and_replay(paths):
    journal = LedgerJournal(paths.ledger_journal, group_commit_ms=20)
    events = [_event(i, attempt=f"a:{i}") for i in range(1, 51)]
    await asyncio.gather(*(journal.append(e) for e in events))
    assert journal.events_written == 50
    assert journal.batches_written <= 3          # one linger window covers the burst
    assert [e.seq for e in LedgerJournal(paths.ledger_journal).replay()] == list(range(1, 51))
    with pytest.raises(JournalError):
        await journal.append(_event(7))           # seq must be the next one


async def test_journal_repairs_torn_tail_but_not_a_corrupt_middle(paths):
    journal = LedgerJournal(paths.ledger_journal, group_commit_ms=0)
    await journal.append(_event(1))
    await journal.append(_event(2, attempt="a:2"))
    with open(paths.ledger_journal, "ab") as handle:
        handle.write(b'{"seq": 3, "kind": "res')
    reopened = LedgerJournal(paths.ledger_journal, group_commit_ms=0)
    assert reopened.torn_tail_bytes > 0 and reopened.last_seq == 2
    await reopened.append(_event(3, attempt="a:3"))
    assert [e.seq for e in reopened.replay()] == [1, 2, 3]
    lines = paths.ledger_journal.read_bytes().split(b"\n")
    lines[1] = b"not json"
    paths.ledger_journal.write_bytes(b"\n".join(lines))
    with pytest.raises(JournalError):
        LedgerJournal(paths.ledger_journal)


async def test_journal_failure_breaks_the_journal(paths, monkeypatch):
    journal = LedgerJournal(paths.ledger_journal, group_commit_ms=0)

    def boom(data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(journal, "_write_and_sync", boom)
    with pytest.raises(JournalError):
        await journal.append(_event(1))
    with pytest.raises(JournalError) as info:
        await journal.append(_event(2, attempt="a:2"))
    assert info.value.maybe_written is False


def test_state_store_lease_result_backup_and_recover(paths):
    store = StateStore(paths)
    root = spec("i1", GenerationKind.ROOT)
    child = spec("i1", GenerationKind.PRIVATE, update_round=1, parents=(root.node_id,))
    store.register_nodes([root, child], wave_index=0)
    store.register_nodes([root], wave_index=3)                 # idempotent
    aid = f"{root.node_id}:1"
    assert store.try_lease(root.node_id, aid, owner="me")
    assert not store.try_lease(root.node_id, f"{root.node_id}:2", owner="other")   # never two active attempts
    attempt = Attempt(aid, root.node_id, 1, AttemptState.SENT, Decimal("0.001"), "t0", sent_at="t1")
    store.record_attempt(attempt)
    outcome = make_outcome(root, aid, [], Behavior())
    rel, sha = write_raw_response(paths, aid, outcome)
    result = replace(outcome.result, raw_response_path=rel, raw_response_sha256=sha)
    candidate = fake_extract(result, root)
    store.record_persisted_response(attempt, result)
    assert store.node(root.node_id).state == "response_persisted"
    store.record_result(result, candidate)
    store.record_result(result, candidate)                     # duplicate completion notification
    assert store.duplicate_results == 1
    assert store.terminal_node_ids() == {root.node_id}
    assert store.candidate(root.node_id) == candidate
    parsed_result, parsed_candidate = read_parsed(paths.parsed(root.node_id))
    assert parsed_candidate == candidate and parsed_result.raw_response_sha256 == sha
    assert store.sends_by_node() == {root.node_id: 1}
    store.backup()
    assert paths.state_db_backup.exists()
    store.close()
    for suffix in ("", "-wal", "-shm"):
        target = paths.state_db_local.with_name(paths.state_db_local.name + suffix)
        if target.exists():
            target.unlink()
    recovered = StateStore.recover(paths)
    assert recovered.terminal_node_ids() == {root.node_id}
    assert recovered.node(child.node_id).state == "pending"
    recovered.close()


def test_recover_without_any_database_reingests_parsed_and_raw(paths):
    store = StateStore(paths)
    a, b = spec("i1", GenerationKind.ROOT, root_slot=0), spec("i1", GenerationKind.ROOT, root_slot=1)
    store.register_nodes([a, b], wave_index=0)
    for s, commit in ((a, True), (b, False)):
        aid = f"{s.node_id}:1"
        outcome = make_outcome(s, aid, [], Behavior())
        rel, sha = write_raw_response(paths, aid, outcome)
        result = replace(outcome.result, raw_response_path=rel, raw_response_sha256=sha)
        if commit:
            store.record_result(result, fake_extract(result, s))
    store.close()
    for suffix in ("", "-wal", "-shm"):
        target = paths.state_db_local.with_name(paths.state_db_local.name + suffix)
        if target.exists():
            target.unlink()
    recovered = StateStore.recover(paths)                      # no backup either
    assert recovered.recovery_report["parsed_ingested"] == 1
    assert recovered.recovery_report["raw_persisted"] == 1
    recovered.register_nodes([a, b], wave_index=0)
    assert recovered.node(a.node_id).state == "terminal"
    assert recovered.node(b.node_id).state == "response_persisted"
    [(attempt, result)] = recovered.persisted_unfinished()
    assert result.node_id == b.node_id and result.raw_response_path is not None
    assert dump_record(result)["transport_status"] == TransportStatus.OK.value
    recovered.close()
