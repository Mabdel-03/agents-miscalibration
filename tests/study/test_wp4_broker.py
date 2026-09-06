"""resources/broker.py: fit/no-fit boundary, headroom release, final reserve, call caps,
16-thread no-overshoot (fixture T8), close invariants (§6.5, §10.2)."""

from __future__ import annotations

import random
import threading
import time

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.resources.broker import ROLE_SELECTOR, EpisodeLedger, LedgerError, Reservation, StopReason
from agents_scaling.study.resources.oracle import FlopOracle

ORACLE = FlopOracle.from_table("32B")
R = ORACLE.reservation(1000, 8192)


def ledger(B: int, final: int = 0, **kw) -> EpisodeLedger:
    return EpisodeLedger(B, ORACLE, final, **kw)


def test_fit_boundary_exact():
    led = ledger(3 * R)
    assert led.fits([R, R, R]) == (True, None)
    assert led.fits([R, R, R, 1]) == (False, StopReason.BUDGET)
    res = led.try_reserve([R, R, R], owner="roots")
    assert isinstance(res, Reservation) and res.total == 3 * R and led.remaining == 0
    assert led.try_reserve([1]) is None and led.last_refusal is StopReason.BUDGET
    assert led.events[-1].op == "reject" and led.events[-1].detail["reason"] == "BUDGET"
    led.debit(res, [R, R, R])
    assert led.spent == 3 * R and led.slack == 0
    led.stop(StopReason.BUDGET)
    assert led.close()["stop_reason"] == "BUDGET"


def test_debit_releases_headroom_and_refuses_over_actual():
    led = ledger(2 * R)
    res = led.try_reserve([R])
    assert led.remaining == R
    actual = ORACLE.debit(1000, 100)
    led.debit(res, [actual])
    assert led.spent == actual and led.remaining == 2 * R - actual and led.peak_reserved == R
    res2 = led.try_reserve([R])
    assert res2 is not None
    with pytest.raises(LedgerError, match="exceeds its reservation"):
        led.debit(res2, [R + 1])
    # a failed reservation stays held until settled; releasing it is allowed
    led.release(res2)
    assert led.reserved == 0 and led.spent == actual
    with pytest.raises(LedgerError, match="not held"):
        led.debit(res2, [1])


def test_zero_amount_context_failure_keeps_its_slot():
    led = ledger(10 * R, solver_call_cap=3)
    res = led.try_reserve([0, R])  # a context failure (0) and a real call
    assert led.calls_admitted() == 2
    led.debit(res, [0, ORACLE.debit(1000, 1)])
    assert led.calls_admitted() == 2 and led.call_slots_remaining() == 1
    assert led.try_reserve([R, R]) is None and led.last_refusal is StopReason.CALL_CAP


def test_final_reserve_retained_then_released():
    final = R
    led = ledger(3 * R, final, final_reserve_calls=1, solver_call_cap=3)
    assert led.remaining == 2 * R and led.call_slots_remaining() == 2
    assert led.try_reserve([R, R, R]) is None and led.last_refusal is StopReason.CALL_CAP
    assert led.try_reserve([2 * R, 1]) is None and led.last_refusal is StopReason.BUDGET
    res = led.try_reserve([R, R])
    led.debit(res, [R, R])
    assert led.try_reserve([1]) is None  # nothing but the final reserve is left
    assert led.release_final() == final
    with pytest.raises(LedgerError):
        led.release_final()
    fin = led.try_reserve([final], keep_final=False, owner="final")
    assert fin is not None
    led.debit(fin, [final])
    led.stop(StopReason.CYCLE_CAP)
    summary = led.close()
    assert summary["spent"] == 3 * R and summary["final_reserve_released"] is True and summary["calls_admitted"] == 3


def test_set_final_reserve_rules():
    led = ledger(2 * R)
    led.set_final_reserve(R, 1)
    assert led.final_reserve == R and led.remaining == R
    with pytest.raises(LedgerError, match="cannot start"):
        ledger(R).set_final_reserve(R + 1)
    with pytest.raises(LedgerError, match="cannot start"):
        EpisodeLedger(R, ORACLE, R + 1)
    res = led.try_reserve([R])
    with pytest.raises(LedgerError, match="before the first reservation"):
        led.set_final_reserve(0)
    led.release(res)


def test_selector_calls_are_separate():
    led = ledger(100 * R, selector_call_cap=2)
    a = led.try_reserve([1, 1], role=ROLE_SELECTOR)
    assert a is not None and led.calls_admitted(ROLE_SELECTOR) == 2 and led.calls_admitted() == 0
    assert led.try_reserve([1], role=ROLE_SELECTOR) is None and led.last_refusal is StopReason.CALL_CAP
    led.debit(a, [1, 1])
    with pytest.raises(ValueError):
        led.try_reserve([1], role="judge")


def test_close_invariants_and_stop_once():
    led = ledger(2 * R)
    res = led.try_reserve([R])
    with pytest.raises(LedgerError, match="never debited"):
        led.close()
    led.debit(res, [R])
    with pytest.raises(LedgerError, match="without a recorded stop reason"):
        led.close()
    led.stop("CALL_CAP", note="x")
    with pytest.raises(LedgerError, match="already recorded"):
        led.stop(StopReason.BUDGET)
    summary = led.close()
    assert summary["closed"] and summary["stop_detail"] == {"note": "x"}
    assert isinstance(summary["B_flops"], int) and summary["events"][-1]["op"] == "close"
    with pytest.raises(LedgerError, match="closed"):
        led.try_reserve([1])
    with pytest.raises(ValueError):
        led.try_reserve([])
    with pytest.raises(ValueError):
        led.try_reserve([-1])
    assert isinstance(LedgerError("x"), T.ProtocolError)


def test_no_overshoot_under_16_threads():
    """T8: 16 threads reserving and debiting random amounts never exceed B, and every
    reservation is settled."""
    B = 50 * R
    led = ledger(B, solver_call_cap=10_000)
    rng = random.Random(7)
    violations: list[str] = []
    stop_at = time.monotonic() + 1.5
    lock = threading.Lock()
    admitted = [0]

    def worker(seed: int) -> None:
        local = random.Random(seed)
        while time.monotonic() < stop_at:
            amounts = [local.randint(1, 2 * R) for _ in range(local.randint(1, 4))]
            res = led.try_reserve(amounts, owner=f"t{seed}")
            snap = led.summary()
            if snap["spent"] + snap["reserved_open"] > B:
                violations.append("allocation over B")
            if res is None:
                continue
            with lock:
                admitted[0] += 1
            time.sleep(local.random() * 0.002)
            led.debit(res, [local.randint(0, a) for a in amounts])
            snap = led.summary()
            if snap["spent"] > B:
                violations.append("committed over B")

    threads = [threading.Thread(target=worker, args=(rng.randint(0, 10**6),)) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not violations and admitted[0] > 16
    assert led.reserved == 0 and led.spent <= B
    led.stop(StopReason.BUDGET)
    summary = led.close()
    assert summary["spent"] <= B and summary["peak_allocated"] <= B
    reserve_ids = {e.detail["reservation_id"] for e in led.events if e.op == "reserve"}
    debit_ids = {e.detail["reservation_id"] for e in led.events if e.op == "debit"}
    assert reserve_ids == debit_ids  # ledger acyclic: every reservation settled exactly once
