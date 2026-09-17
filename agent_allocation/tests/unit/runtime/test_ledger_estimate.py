"""Collection ledger algebra, journal replay, and the conservative reservation estimate."""
from __future__ import annotations

import asyncio
import random
from decimal import Decimal

import pytest

from agent_allocation.contracts import GenerationKind, LedgerEventKind
from agent_allocation.runtime.estimate import input_token_bound, reservation_usd
from agent_allocation.runtime.ledger import (
    CollectionLedger, LedgerError, active_ceiling_from_provider, journal_attempts,
)
from agent_allocation.storage.journal import LedgerJournal
from tests.unit.runtime.fakes import LOCK, spec

D = Decimal


def _ledger(tmp_path, ceiling="1", unresolved="2", group_commit_ms=1):
    journal = LedgerJournal(tmp_path / "journal.jsonl", group_commit_ms=group_commit_ms)
    return CollectionLedger(journal, ceiling_usd=D(ceiling), max_unresolved_usd=D(unresolved))


def test_reservation_formula_is_conservative_and_exact():
    s = spec("i", GenerationKind.ROOT, budget=1024, headroom=1024)
    amount = reservation_usd(s, 1001, LOCK, multiplier=D("1.1"))
    expected = (D(501 + 64) * D("0.0000001") + D(2048) * D("0.0000004")) * D("1.1")
    assert amount == expected
    assert input_token_bound(0) == 64 and input_token_bound(3) == 66
    with pytest.raises(ValueError):
        reservation_usd(s, 10, LOCK, multiplier=D("0.9"))
    from dataclasses import replace
    with pytest.raises(ValueError):
        reservation_usd(s, 10, replace(LOCK, output_price_per_token=""), multiplier=D("1.1"))
    pricier = replace(LOCK, internal_reasoning_price_per_token="0.000001")
    assert reservation_usd(s, 10, pricier, multiplier=D(1)) > reservation_usd(s, 10, LOCK, multiplier=D(1))


async def test_tiny_ceiling_concurrent_reservations_never_exceed_it(tmp_path):
    ledger = _ledger(tmp_path, ceiling="0.01")
    rng = random.Random(7)
    amounts = [D(rng.randint(1, 900)) / D(1_000_000) for _ in range(400)]

    async def one(i, amount):
        await asyncio.sleep(rng.random() / 1000)
        ok = await ledger.reserve(attempt_id=f"n{i}:1", node_id=f"n{i}", partition_id="main", amount_usd=amount)
        if ok and i % 3 == 0:
            await ledger.mark_sent(f"n{i}:1")
            await ledger.settle(f"n{i}:1", amount / 2)
        return ok, amount

    results = await asyncio.gather(*(one(i, a) for i, a in enumerate(amounts)))
    snap = ledger.snapshot()
    assert snap.settled + snap.in_flight + snap.unresolved <= D("0.01")
    assert ledger.peak_exposure <= D("0.01")
    accepted = [a for ok, a in results if ok]
    assert 0 < len(accepted) < len(amounts)
    replayed = CollectionLedger(LedgerJournal(tmp_path / "journal.jsonl"), ceiling_usd=D("0.01"),
                                max_unresolved_usd=D(2)).snapshot()
    assert replayed.settled == snap.settled
    assert replayed.in_flight + replayed.unresolved == snap.in_flight + snap.unresolved   # nothing disappears


async def test_transitions_and_outstanding_costs_cannot_disappear(tmp_path):
    ledger = _ledger(tmp_path, ceiling="10")
    for i in range(1, 6):
        assert await ledger.reserve(attempt_id=f"n:{i}", node_id="n", partition_id="main", amount_usd=D("0.5"))
    await ledger.mark_sent("n:1")
    await ledger.settle("n:1", D("0.1"))
    await ledger.mark_sent("n:2")
    await ledger.mark_unresolved("n:2", detail="ambiguous_timeout")
    with pytest.raises(LedgerError):
        await ledger.release_unsent("n:2")                     # an unresolved attempt is never free
    await ledger.release_unsent("n:3")                          # provably never sent
    await ledger.mark_sent("n:4")
    await ledger.charge_max("n:4")
    await ledger.settle("n:4", D("0.2"))                        # late bill after charge_max: noted, no change
    await ledger.settle("n:1", D("0.1"))                        # duplicate notification: ignored
    snap = ledger.snapshot()
    assert snap.settled == D("0.6") and snap.unresolved == D("0.5") and snap.in_flight == D("0.5")
    assert ledger.late_bills == [("n:4", D("0.2"))] and ledger.duplicate_resolutions == 2
    with pytest.raises(LedgerError):
        await ledger.reserve(attempt_id="n:1", node_id="n", partition_id="main", amount_usd=D("0.1"))
    # crash: everything open becomes unresolved exposure; nothing is lost
    restarted = _ledger(tmp_path, ceiling="10")
    after = restarted.snapshot()
    # n:2 was sent (unresolved, feeds the pause); n:5 was only reserved (provably unsent, still counted in flight)
    assert after.settled == D("0.6") and after.in_flight == D("0.5") and after.unresolved == D("0.5")
    assert restarted.inherited_attempt_ids == {"n:2", "n:5"}
    kinds = {a: j.resolution for a, j in journal_attempts(restarted.journal).items()}
    assert kinds == {"n:1": LedgerEventKind.SETTLE, "n:2": None, "n:3": LedgerEventKind.RELEASE_UNSENT,
                     "n:4": LedgerEventKind.CHARGE_MAX, "n:5": None}


async def test_sent_money_is_never_released_and_replays_as_unresolved(tmp_path):
    ledger = _ledger(tmp_path, ceiling="1")
    assert await ledger.reserve(attempt_id="s:1", node_id="s", partition_id="main", amount_usd=D("0.3"))
    await ledger.mark_sent("s:1")
    with pytest.raises(LedgerError):
        await ledger.release_unsent("s:1")                      # QA s15.08: sent money is never released
    snap = ledger.snapshot()
    assert snap.in_flight == D("0.3") and snap.settled == 0
    await ledger.mark_sent("s:1")                               # idempotent: no duplicate SENT line
    assert await ledger.reserve(attempt_id="t:1", node_id="t", partition_id="main", amount_usd=D("0.2"))
    await ledger.mark_sent("t:1")
    await ledger.mark_unresolved("t:1", detail="error_envelope")
    kinds = [e.kind for e in ledger.journal.replay()]
    assert kinds == [LedgerEventKind.RESERVE, LedgerEventKind.SENT, LedgerEventKind.RESERVE, LedgerEventKind.SENT,
                     LedgerEventKind.UNRESOLVED]
    restarted = _ledger(tmp_path, ceiling="1", unresolved="0.4")
    assert restarted.snapshot().unresolved == D("0.5") and restarted.unresolved_exceeded
    assert not await restarted.reserve(attempt_id="u:1", node_id="u", partition_id="main", amount_usd=D("0.01"))
    assert restarted.last_refusal == "unresolved_exposure"


async def test_unresolved_exposure_pauses_reservations(tmp_path):
    ledger = _ledger(tmp_path, ceiling="100", unresolved="1")
    for i in range(3):
        assert await ledger.reserve(attempt_id=f"t:{i}", node_id="t", partition_id="main", amount_usd=D("0.6"))
        await ledger.mark_sent(f"t:{i}")
    await ledger.mark_unresolved("t:0")
    assert await ledger.reserve(attempt_id="x:1", node_id="x", partition_id="main", amount_usd=D("0.1"))
    await ledger.mark_unresolved("t:1")
    assert ledger.unresolved_exceeded
    assert not await ledger.reserve(attempt_id="x:2", node_id="x", partition_id="main", amount_usd=D("0.1"))
    assert ledger.last_refusal == "unresolved_exposure"
    await ledger.settle("t:1", D("0.2"))
    assert await ledger.reserve(attempt_id="x:2", node_id="x", partition_id="main", amount_usd=D("0.1"))


async def test_active_ceiling_and_provider_conversion(tmp_path):
    ledger = _ledger(tmp_path, ceiling="135")
    ledger.set_active_ceiling(D("500"))
    assert ledger.snapshot().ceiling == D("135")               # never above the study cap
    ceiling = active_ceiling_from_provider(study_cap=D("135"), settled=D("20"), key_limit_remaining=D("30"),
                                           account_balance=None)
    assert ceiling == D("50")                                  # unrelated usage lowers the ceiling
    ledger.set_active_ceiling(ceiling)
    assert not await ledger.reserve(attempt_id="a:1", node_id="a", partition_id="main", amount_usd=D("50.01"))
