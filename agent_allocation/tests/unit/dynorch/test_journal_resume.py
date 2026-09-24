"""Contract §5 journal + resume: call files are immutable and redacted, the run store skips terminal records,
generation-aware attempt ids never collide on resume, and ``resolve_inherited`` settles inherited SENT attempts from
the journal with ``charged_max == 0``."""
from __future__ import annotations

import gzip
import json
from decimal import Decimal

import pytest

from agent_allocation.api.fake import FakeReply, FakeScript
from agent_allocation.contracts import LedgerEventKind
from agent_allocation.dynorch.journal import (
    JOURNAL_LAYOUT, CallJournal, CallJournalConflict, RunStore, attempt_id, decode_payload, parse_attempt_id,
    redact_payload, resolve_inherited,
)
from agent_allocation.dynorch.records import RunRecord, from_json, new_run_record, to_json
from agent_allocation.runtime.ledger import CollectionLedger, attempt_number, journal_attempts
from agent_allocation.storage.journal import LedgerJournal

from .conftest import make_request, make_spec, root_messages

# assembled at import time: the doctor refuses a worktree that contains a key-shaped literal (Gate 0)
KEY = "sk-or-" + "v1-" + "0123456789abcdef" * 3


def payload(spec, call_no: int, **extra):
    base = {"layout": JOURNAL_LAYOUT, "run_key": spec.run_key, "call_no": call_no, "phases": [], "reply": None}
    base.update(extra)
    return base


# ---- immutability and redaction ------------------------------------------------------------------------------------

def test_call_journal_is_write_once_and_redacted(tmp_path):
    journal = CallJournal(tmp_path / "root")
    spec = make_spec()
    path = journal.write(spec, 1, payload(spec, 1, phases=[{"response_headers": {"Authorization": f"Bearer {KEY}",
                                                                                 "x-openrouter-cache-status": "MISS"},
                                                            "response_body_text": f"key {KEY} inside"}]))
    assert path == tmp_path / "root" / "raw" / "H1" / "4" / "1" / "aime__2025__1" / "a1" / "call_0001.json.gz"
    raw = gzip.decompress(path.read_bytes()).decode()
    assert KEY not in raw and "[REDACTED_KEY]" in raw and '"authorization":"[REDACTED]"' in raw
    first = path.read_bytes()
    journal.write(spec, 1, payload(spec, 1, phases=[{"response_headers": {"Authorization": f"Bearer {KEY}",
                                                                          "x-openrouter-cache-status": "MISS"},
                                                     "response_body_text": f"key {KEY} inside"}]))
    assert path.read_bytes() == first                       # identical bytes: accepted, unchanged
    with pytest.raises(CallJournalConflict):
        journal.write(spec, 1, payload(spec, 1, phases=[{"changed": True}]))
    assert path.read_bytes() == first
    assert journal.read(spec, 1)["phases"][0]["response_body_text"] == "key [REDACTED_KEY] inside"
    with pytest.raises(ValueError):
        journal.write(spec, 2, {"layout": "other", "run_key": spec.run_key, "call_no": 2})
    with pytest.raises(ValueError):
        journal.write(spec, 3, payload(spec, 4))


def test_redact_payload_walks_everything():
    out = redact_payload({"a": [KEY, {"cookie": "x", "nested": KEY.encode()}], "headers": {"Authorization": "Bearer abcdefghij"},
                          "n": Decimal("1.5")})
    assert out == {"a": ["[REDACTED_KEY]", {"cookie": "[REDACTED]", "nested": "[REDACTED_KEY]"}],
                   "headers": {"authorization": "[REDACTED]"}, "n": "1.5"}


def test_supersede_moves_call_files_aside(tmp_path):
    journal = CallJournal(tmp_path / "root")
    spec = make_spec()
    journal.write(spec, 1, payload(spec, 1))
    target = journal.supersede(spec)
    assert target is not None and (target / "call_0001.json.gz").exists()
    assert journal.scan(spec) == [] and len(journal.scan(spec, include_superseded=True)) == 1
    assert journal.supersede(spec) is None


# ---- run store -------------------------------------------------------------------------------------------------------

def test_run_store_round_trip_scan_and_terminal_skip(tmp_path):
    store = RunStore(tmp_path / "root")
    running = new_run_record(make_spec(item_id="aime/2025/1"), status="running", started_at="t0")
    done = new_run_record(make_spec(item_id="aime/2025/2"), status="done", started_at="t0")
    store.save(running)
    store.save(done)
    scanned = store.scan()
    assert set(scanned) == {running.spec.run_key, done.spec.run_key}
    assert scanned[done.spec.run_key] == done and from_json(RunRecord, to_json(done)) == done
    assert store.is_terminal(scanned[done.spec.run_key]) and not store.is_terminal(scanned[running.spec.run_key])
    pending = [r for r in scanned.values() if not RunStore.is_terminal(r)]
    assert [r.spec.run_key for r in pending] == [running.spec.run_key]
    assert store.load(running.spec) == running and store.load(make_spec(item_id="none/1")) is None
    # a second attempt of the same cell archives the previous record instead of overwriting it silently
    second = make_spec(item_id="aime/2025/1", attempt=2)
    assert store.archive_previous(second) == store.superseded_path(running.spec)
    assert store.load(running.spec) is None and running.spec.run_key in store.scan(include_superseded=True)
    assert store.archive_previous(second) is None


def test_records_survive_the_codec_strictly():
    record = new_run_record(make_spec(), status="planned", started_at="t0")
    data = to_json(record)
    assert data["__record__"] == "RunRecord" and data["spec"]["run_key"] == record.spec.run_key
    assert from_json(RunRecord, data) == record
    data.pop("calls")
    with pytest.raises(ValueError, match="missing"):
        from_json(RunRecord, data)
    with pytest.raises(ValueError):
        new_run_record(make_spec(), status="weird", started_at="t0")


# ---- attempt ids -----------------------------------------------------------------------------------------------------

def test_generation_aware_attempt_ids_do_not_collide_and_parse():
    first = attempt_id("rk", 0, 3, "think")
    resumed = attempt_id("rk", 1, 3, "think")
    assert first == "rk:g0:3:think:1" and resumed == "rk:g1:3:think:1" and first != resumed
    assert attempt_number(first) == 1 and attempt_number(attempt_id("rk", 1, 3, "answer", 2)) == 2
    assert parse_attempt_id(resumed) == {"run_key": "rk", "generation": 1, "call_no": 3, "phase": "think",
                                         "attempt_no": 1}
    with pytest.raises(ValueError):
        attempt_id("r:k", 0, 1, "think")
    with pytest.raises(ValueError):
        parse_attempt_id("rk:1:think")


async def test_a_resumed_generation_reserves_beside_the_old_ids(tmp_path):
    ledger = CollectionLedger(LedgerJournal(tmp_path / "j.jsonl", group_commit_ms=0), ceiling_usd=Decimal("10"),
                              max_unresolved_usd=Decimal("2"))
    assert await ledger.reserve(attempt_id=attempt_id("rk", 0, 1, "think"), node_id="rk", partition_id="main",
                                amount_usd=Decimal("0.01"))
    await ledger.mark_sent(attempt_id("rk", 0, 1, "think"))
    # same call number, next generation: a fresh id, no LedgerError
    assert await ledger.reserve(attempt_id=attempt_id("rk", 1, 1, "think"), node_id="rk", partition_id="main",
                                amount_usd=Decimal("0.01"))
    from agent_allocation.runtime.ledger import LedgerError
    with pytest.raises(LedgerError, match="already reserved"):
        await ledger.reserve(attempt_id=attempt_id("rk", 0, 1, "think"), node_id="rk", partition_id="main",
                             amount_usd=Decimal("0.01"))
    await ledger.journal.flush()
    ledger.journal.release_writer()


# ---- resolve_inherited -------------------------------------------------------------------------------------------------

async def test_resolve_inherited_settles_from_the_journal_with_zero_charge_max(tmp_path):
    """A SENT attempt whose reply was journaled (the process died between the upstream reply and the ledger
    settle) is settled at the journaled cost; a reserved-only attempt is released; nothing is charged max."""
    root = tmp_path / "root"
    store, journal = RunStore(root), CallJournal(root)
    spec = make_spec()
    store.save(new_run_record(spec, status="running", started_at="t0"))
    journal_path = root / "state" / "ledger_journal.jsonl"
    ledger = CollectionLedger(LedgerJournal(journal_path, group_commit_ms=0), ceiling_usd=Decimal("10"),
                              max_unresolved_usd=Decimal("2"))
    sent = attempt_id(spec.run_key, 0, 1, "think")
    unsent = attempt_id(spec.run_key, 0, 2, "think")
    assert await ledger.reserve(attempt_id=sent, node_id=spec.run_key, partition_id="main", amount_usd=Decimal("0.5"))
    await ledger.mark_sent(sent)
    assert await ledger.reserve(attempt_id=unsent, node_id=spec.run_key, partition_id="main",
                                amount_usd=Decimal("0.5"))
    journal.write(spec, 1, payload(spec, 1, phases=[{"phase": "think", "attempt_id": sent, "cost_usd": "0.0123",
                                                     "transport_status": "ok", "generation_id": "gen-1"}]))
    await ledger.journal.flush()
    ledger.journal.release_writer()                        # "the process died"

    resumed = CollectionLedger(LedgerJournal(journal_path, group_commit_ms=0), ceiling_usd=Decimal("10"),
                               max_unresolved_usd=Decimal("2"))
    assert set(resumed.inherited_attempt_ids) == {sent, unsent}
    assert resumed.snapshot().unresolved == Decimal("0.5") and resumed.snapshot().in_flight == Decimal("0.5")
    report = await resolve_inherited(resumed, store, journal)
    assert report == {"settled_from_journal": 1, "charged_max": 0, "released_unsent": 1, "unparsed": 0}
    snapshot = resumed.snapshot()
    assert snapshot.settled == Decimal("0.0123") and snapshot.unresolved == 0 and snapshot.in_flight == 0
    attempts = journal_attempts(resumed.journal)
    assert attempts[sent].resolution is LedgerEventKind.SETTLE and attempts[sent].resolved_usd == Decimal("0.0123")
    assert attempts[unsent].resolution is LedgerEventKind.RELEASE_UNSENT
    assert await resolve_inherited(resumed, store, journal) == {"settled_from_journal": 0, "charged_max": 0,
                                                                "released_unsent": 0, "unparsed": 0}
    await resumed.journal.flush()
    resumed.journal.release_writer()


async def test_resolve_inherited_charges_max_only_without_a_journaled_reply(tmp_path):
    root = tmp_path / "root"
    store, journal = RunStore(root), CallJournal(root)
    spec = make_spec()
    store.save(new_run_record(spec, status="running", started_at="t0"))
    journal_path = root / "state" / "ledger_journal.jsonl"
    ledger = CollectionLedger(LedgerJournal(journal_path, group_commit_ms=0), ceiling_usd=Decimal("10"),
                              max_unresolved_usd=Decimal("2"))
    lost = attempt_id(spec.run_key, 0, 1, "think")
    foreign = "not-a-dynorch-id:1"
    for aid in (lost, foreign):
        assert await ledger.reserve(attempt_id=aid, node_id=spec.run_key, partition_id="main",
                                    amount_usd=Decimal("0.25"))
        await ledger.mark_sent(aid)
    await ledger.journal.flush()
    ledger.journal.release_writer()
    resumed = CollectionLedger(LedgerJournal(journal_path, group_commit_ms=0), ceiling_usd=Decimal("10"),
                               max_unresolved_usd=Decimal("2"))
    report = await resolve_inherited(resumed, store, journal)
    assert report == {"settled_from_journal": 0, "charged_max": 2, "released_unsent": 0, "unparsed": 1}
    assert resumed.snapshot().settled == Decimal("0.5") and resumed.snapshot().unresolved == 0
    await resumed.journal.flush()
    resumed.journal.release_writer()


async def test_core_journal_files_carry_the_attempt_cost_resolve_inherited_needs(make_env):
    """End to end: the core's own journal payload is what ``resolve_inherited`` reads."""
    script = FakeScript(natural_think_tokens=50)
    script.default = FakeReply(content="<final>1</final>")
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    stored = env.journal.read(spec, 1)
    phase = stored["phases"][0]
    assert phase["attempt_id"] == attempt_id(spec.run_key, 0, 1, "think")
    assert Decimal(phase["cost_usd"]) == Decimal(reply.call.cost_usd)
    assert stored["rendered_prompt_sha256"] and stored["budget"]["granted"] == 1024
    assert stored["call"]["__record__"] == "CallRecord" and stored["harness_request"]["__record__"] == "ChatRequest"
    assert "sk-or-" not in json.dumps(stored)
