"""Per-episode budget broker: reserve-before-launch, debit-actual, frozen stop reasons (WP4).

Spec §6.5 (allocation policy steps 1–4: reserve the known-prompt prefill plus full-cap
decode for the next indivisible call or symmetric group *and* the retained final-answer
reserve; launch only if the complete reservation fits; debit actual work and release the
headroom; stop optional work when the next legal request does not fit; overshoot is a
harness defect → suspend the cell), §6.7 (logical uncached accounting: an aliased store hit
is debited at its recorded actual cost), §10.2 ("admission is a transaction ... Reserving
two children from the same remaining balance without an atomic update is a correctness
defect"), §4.2 (64 solver-invocation cap; ≤64 selector calls), §11.4 (per-episode ledger
records).  Architecture §1.10; audit §2.0 items 9–10 (fixtures T4, T8, T10); corrections
§4 item 7 (a context failure is a used opportunity — reserved at 0 FLOPs, it still consumes
its call slot).

Everything is exact ``int`` FLOPs.  One :class:`threading.Lock` guards every mutation so
parallel members of a group can never double-spend the same balance.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents_scaling.study.types import ProtocolError

ROLE_SOLVER = "solver"
ROLE_SELECTOR = "selector"


class StopReason(str, Enum):
    """Why an episode stopped allocating (§6.5 "report ... whether each episode stopped
    because of budget, a context/call cap or ordinary completion"; audit §2.0 item 9)."""

    BUDGET = "BUDGET"
    CALL_CAP = "CALL_CAP"
    ROUND_CAP = "ROUND_CAP"
    CYCLE_CAP = "CYCLE_CAP"
    VOLUNTARY_FINISH = "VOLUNTARY_FINISH"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    COMPLETED = "COMPLETED"


class LedgerError(ProtocolError):
    """A ledger invariant was violated by the caller (harness defect → suspend)."""


def _amount(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


@dataclass
class Reservation:
    """One admitted indivisible group (a single call is a group of one)."""

    reservation_id: int
    owner: str
    role: str
    amounts: tuple[int, ...]
    created_at: float
    state: str = "held"  # held | settled
    actual: tuple[int | None, ...] | None = None

    @property
    def total(self) -> int:
        return sum(self.amounts)

    @property
    def calls(self) -> int:
        return len(self.amounts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "owner": self.owner,
            "role": self.role,
            "amounts": list(self.amounts),
            "total": self.total,
            "calls": self.calls,
            "created_at": self.created_at,
            "state": self.state,
            "actual": None if self.actual is None else list(self.actual),
        }


@dataclass(frozen=True)
class ResourceEvent:
    """Append-only ledger line (architecture §2.2 ``ledger.events[]``, §11.4)."""

    t: float
    op: str  # reserve | reject | debit | release_final | stop | close
    amount: int
    remaining: int
    reserved: int
    committed: int
    owner: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "op": self.op,
            "amount": self.amount,
            "remaining": self.remaining,
            "reserved": self.reserved,
            "committed": self.committed,
            "owner": self.owner,
            "detail": dict(self.detail),
        }


class EpisodeLedger:
    """One episode's allowance ``B`` with the retained final-answer reserve (§6.5).

    Invariants (checked on every transition, violations raise :class:`LedgerError`):

    * ``committed + reserved + final_reserve_held <= B`` after every admission;
    * ``committed <= B`` at :meth:`close` (overshoot → suspend, never trim);
    * solver/selector call slots: ``admitted + group + final_reserve_calls <= cap``;
    * an actual debit never exceeds its reservation (the reservation was computed from the
      known prompt and the full decode cap, so a larger actual means a broken oracle input).

    ``final_reserve`` is FLOPs held back for the method's legal final-answer path (CEN_FLAT's
    reserved final hub call; 0 for the deterministic VOTE selectors) and
    ``final_reserve_calls`` the solver call slots it needs.  :meth:`release_final` hands the
    reserve back so the final call can be admitted with ``keep_final=False``.
    """

    def __init__(
        self,
        B: int,
        oracle: Any,
        final_reserve: int,
        *,
        final_reserve_calls: int = 0,
        solver_call_cap: int = 64,
        selector_call_cap: int = 64,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.B = _amount(B, "B")
        self.oracle = oracle
        self.final_reserve = _amount(final_reserve, "final_reserve")
        self.final_reserve_calls = _amount(final_reserve_calls, "final_reserve_calls")
        self.solver_call_cap = _amount(solver_call_cap, "solver_call_cap")
        self.selector_call_cap = _amount(selector_call_cap, "selector_call_cap")
        if self.final_reserve > self.B:
            raise LedgerError(
                f"final reserve {self.final_reserve} exceeds B={self.B}: the budget cannot start the mandatory protocol"
            )
        if self.final_reserve_calls > self.solver_call_cap:
            raise LedgerError("final_reserve_calls exceeds the solver call cap")
        self._clock = clock
        self._lock = threading.Lock()
        self._reserved = 0
        self._committed = 0
        self._peak_reserved = 0
        self._peak_allocated = self.final_reserve
        self._final_released = False
        self._admitted = {ROLE_SOLVER: 0, ROLE_SELECTOR: 0}
        self._events: list[ResourceEvent] = []
        self._reservations: dict[int, Reservation] = {}
        self._next_id = 1
        self._stop_reason: StopReason | None = None
        self._stop_detail: dict[str, Any] = {}
        self._closed = False
        self.last_refusal: StopReason | None = None

    # ---- read-only views -------------------------------------------------------------
    @property
    def final_reserve_held(self) -> int:
        return 0 if self._final_released else self.final_reserve

    @property
    def remaining(self) -> int:
        """Headroom for new optional work: ``B - committed - reserved - final_reserve_held``."""
        with self._lock:
            return self._remaining_locked()

    def _remaining_locked(self) -> int:
        return self.B - self._committed - self._reserved - self.final_reserve_held

    @property
    def spent(self) -> int:
        with self._lock:
            return self._committed

    @property
    def committed(self) -> int:
        return self.spent

    @property
    def reserved(self) -> int:
        with self._lock:
            return self._reserved

    @property
    def peak_reserved(self) -> int:
        with self._lock:
            return self._peak_reserved

    @property
    def peak_allocated(self) -> int:
        """Maximum of ``committed + reserved + final_reserve_held`` observed."""
        with self._lock:
            return self._peak_allocated

    @property
    def slack(self) -> int:
        with self._lock:
            return self.B - self._committed

    @property
    def events(self) -> list[ResourceEvent]:
        with self._lock:
            return list(self._events)

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason

    @property
    def stop_detail(self) -> dict[str, Any]:
        return dict(self._stop_detail)

    @property
    def closed(self) -> bool:
        return self._closed

    def calls_admitted(self, role: str = ROLE_SOLVER) -> int:
        with self._lock:
            return self._admitted[self._role(role)]

    def call_slots_remaining(self, role: str = ROLE_SOLVER, *, keep_final: bool = True) -> int:
        with self._lock:
            return self._slots_locked(self._role(role), keep_final)

    def _slots_locked(self, role: str, keep_final: bool) -> int:
        cap = self.solver_call_cap if role == ROLE_SOLVER else self.selector_call_cap
        held = self.final_reserve_calls if (keep_final and role == ROLE_SOLVER and not self._final_released) else 0
        return cap - self._admitted[role] - held

    @staticmethod
    def _role(role: str) -> str:
        if role not in (ROLE_SOLVER, ROLE_SELECTOR):
            raise ValueError(f"role must be {ROLE_SOLVER!r} or {ROLE_SELECTOR!r}, got {role!r}")
        return role

    def set_final_reserve(self, amount: int, calls: int = 0) -> None:
        """Declare the method's final-answer reserve before any admission (§6.5 "every
        retained budget must still reserve a legal final-answer path").

        Policies call this first thing in ``run`` (0 for the deterministic VOTE selectors;
        CEN_FLAT's reserved final hub call otherwise).  Refused once anything was reserved,
        committed or stopped, and when the reserve alone exceeds ``B`` (a budget that cannot
        start the mandatory protocol is a configuration error, never a fallback).
        """
        amount = _amount(amount, "final_reserve")
        calls = _amount(calls, "final_reserve_calls")
        with self._lock:
            self._check_open()
            if self._events or self._reserved or self._committed or self._final_released or self._stop_reason is not None:
                raise LedgerError("set_final_reserve is only legal before the first reservation")
            if amount > self.B:
                raise LedgerError(f"final reserve {amount} exceeds B={self.B}: the budget cannot start the mandatory protocol")
            if calls > self.solver_call_cap:
                raise LedgerError("final_reserve_calls exceeds the solver call cap")
            self.final_reserve = amount
            self.final_reserve_calls = calls
            self._peak_allocated = amount

    # ---- transactions --------------------------------------------------------------
    def fits(self, group: Sequence[int], *, keep_final: bool = True, role: str = ROLE_SOLVER) -> tuple[bool, StopReason | None]:
        """Non-mutating admission query: ``(fits, refusal reason)``."""
        amounts = tuple(_amount(a, "group amount") for a in group)
        role = self._role(role)
        with self._lock:
            return self._fits_locked(amounts, keep_final, role)

    def _fits_locked(self, amounts: tuple[int, ...], keep_final: bool, role: str) -> tuple[bool, StopReason | None]:
        if not keep_final and self.final_reserve > 0 and not self._final_released:
            raise LedgerError("keep_final=False before release_final(): the final-answer reserve is not this caller's to spend")
        if len(amounts) > self._slots_locked(role, keep_final):
            return False, StopReason.CALL_CAP
        held = self.final_reserve_held if keep_final else 0
        if self._committed + self._reserved + sum(amounts) + held > self.B:
            return False, StopReason.BUDGET
        return True, None

    def try_reserve(
        self,
        group: Sequence[int],
        *,
        keep_final: bool = True,
        owner: str = "",
        role: str = ROLE_SOLVER,
    ) -> Reservation | None:
        """All-or-nothing admission of one indivisible group (§6.5 steps 1–2, §10.2).

        Returns ``None`` (and sets :attr:`last_refusal` to ``BUDGET`` or ``CALL_CAP``) when
        the group plus the retained final reserve does not fit; a refusal is recorded as a
        ``reject`` event.  Zero-FLOP entries are allowed: a context failure still consumes
        its call slot (§4.3; corrections §4 item 7).  ``keep_final=False`` is only for the
        final-answer path itself (after :meth:`release_final`).
        """
        amounts = tuple(_amount(a, "group amount") for a in group)
        if not amounts:
            raise ValueError("cannot reserve an empty group")
        role = self._role(role)
        with self._lock:
            self._check_open()
            ok, reason = self._fits_locked(amounts, keep_final, role)
            if not ok:
                self.last_refusal = reason
                self._event("reject", sum(amounts), owner, {"reason": reason.value, "calls": len(amounts), "role": role, "keep_final": keep_final})
                return None
            res = Reservation(self._next_id, owner, role, amounts, self._clock())
            self._next_id += 1
            self._reservations[res.reservation_id] = res
            self._reserved += res.total
            self._admitted[role] += len(amounts)
            self._peak_reserved = max(self._peak_reserved, self._reserved)
            self._peak_allocated = max(self._peak_allocated, self._committed + self._reserved + self.final_reserve_held)
            self.last_refusal = None
            self._event("reserve", res.total, owner, {"reservation_id": res.reservation_id, "calls": len(amounts), "amounts": list(amounts), "role": role, "keep_final": keep_final})
            return res

    def debit(self, res: Reservation, actual: Sequence[int | None]) -> int:
        """§6.5 step 3: commit the actual work of ``res`` and release the unused headroom.

        ``actual[i]`` is the realized cost of call ``i`` (``oracle.debit(prompt, completion)``,
        also for an aliased store hit — §6.7 logical accounting) or ``None`` for a call that
        was never completed (infrastructure failure, or a placeholder reservation such as
        CEN_FLAT's not-yet-planned workers) whose reservation is released uncommitted.
        Returns the committed amount.
        """
        if not isinstance(res, Reservation):
            raise TypeError("res must be a Reservation")
        actual_t = tuple(None if a is None else _amount(a, "actual") for a in actual)
        if len(actual_t) != res.calls:
            raise LedgerError(f"debit of reservation {res.reservation_id}: {len(actual_t)} actuals for {res.calls} calls")
        with self._lock:
            self._check_open()
            live = self._reservations.get(res.reservation_id)
            if live is None or live is not res or res.state != "held":
                raise LedgerError(f"reservation {res.reservation_id} is not held by this ledger")
            for i, (reserved_i, actual_i) in enumerate(zip(res.amounts, actual_t)):
                if actual_i is not None and actual_i > reserved_i:
                    raise LedgerError(
                        f"reservation {res.reservation_id} call {i}: actual {actual_i} exceeds its reservation {reserved_i}"
                        " (the reservation must be computed from the known prompt and the full decode cap)"
                    )
            committed_now = sum(a for a in actual_t if a is not None)
            released_slots = sum(1 for a in actual_t if a is None)
            self._reserved -= res.total
            self._committed += committed_now
            self._admitted[res.role] -= released_slots
            res.state = "settled"
            res.actual = actual_t
            del self._reservations[res.reservation_id]
            if self._committed > self.B:
                self._event("overshoot", committed_now, res.owner, {"reservation_id": res.reservation_id})
                raise LedgerError(f"budget overshoot: committed {self._committed} > B={self.B} (§6.5 harness defect)")
            self._event(
                "debit",
                committed_now,
                res.owner,
                {
                    "reservation_id": res.reservation_id,
                    "released": res.total - committed_now,
                    "released_calls": released_slots,
                    "actual": list(actual_t),
                },
            )
            return committed_now

    def release(self, res: Reservation) -> None:
        """Release a held reservation entirely (nothing launched): ``debit(res, [None]*calls)``."""
        self.debit(res, [None] * res.calls)

    def release_final(self) -> int:
        """Hand the final-answer reserve back (once) so the final call can be admitted with
        ``keep_final=False``.  Returns the released amount."""
        with self._lock:
            self._check_open()
            if self._final_released:
                raise LedgerError("final reserve already released")
            self._final_released = True
            self._event("release_final", self.final_reserve, "final", {"calls": self.final_reserve_calls})
            return self.final_reserve

    # ---- termination ---------------------------------------------------------------
    def stop(self, reason: StopReason | str, **detail: Any) -> None:
        """Record the frozen stop reason exactly once (§6.5 reporting)."""
        reason = StopReason(reason)
        with self._lock:
            self._check_open()
            if self._stop_reason is not None:
                raise LedgerError(f"stop reason already recorded ({self._stop_reason.value}); refusing {reason.value}")
            self._stop_reason = reason
            self._stop_detail = dict(detail)
            self._event("stop", 0, "episode", {"reason": reason.value, **detail})

    def close(self) -> dict[str, Any]:
        """Assert the §6.5 invariants and return the ledger summary for the item file.

        Raises :class:`LedgerError` (a :class:`ProtocolError`) on overshoot, on a reservation
        that was never settled, or when no stop reason was recorded — every one of those is
        a harness defect that must suspend the cell rather than produce a scored item.
        """
        with self._lock:
            self._check_open()
            if self._committed > self.B:
                raise LedgerError(f"budget overshoot at close: committed {self._committed} > B={self.B}")
            if self._reservations:
                held = sorted(self._reservations)
                raise LedgerError(f"reservations {held} were never debited or released")
            if self._stop_reason is None:
                raise LedgerError("episode closed without a recorded stop reason")
            self._closed = True
            self._event("close", 0, "episode", {"committed": self._committed, "slack": self.B - self._committed})
            return self._summary_locked()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict[str, Any]:
        return {
            "B_flops": self.B,
            "spent": self._committed,
            "reserved_open": self._reserved,
            "peak_reserved": self._peak_reserved,
            "peak_allocated": self._peak_allocated,
            "slack": self.B - self._committed,
            "final_reserve": self.final_reserve,
            "final_reserve_calls": self.final_reserve_calls,
            "final_reserve_released": self._final_released,
            "calls_admitted": self._admitted[ROLE_SOLVER],
            "selector_calls_admitted": self._admitted[ROLE_SELECTOR],
            "solver_call_cap": self.solver_call_cap,
            "selector_call_cap": self.selector_call_cap,
            "stop_reason": None if self._stop_reason is None else self._stop_reason.value,
            "stop_detail": dict(self._stop_detail),
            "closed": self._closed,
            "oracle": getattr(self.oracle, "oracle_hash", None),
            "events": [e.to_dict() for e in self._events],
        }

    # ---- internals ---------------------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise LedgerError("ledger is closed")

    def _event(self, op: str, amount: int, owner: str, detail: dict[str, Any]) -> None:
        self._events.append(
            ResourceEvent(
                t=self._clock(),
                op=op,
                amount=amount,
                remaining=self._remaining_locked(),
                reserved=self._reserved,
                committed=self._committed,
                owner=owner,
                detail=detail,
            )
        )


__all__ = [
    "EpisodeLedger",
    "LedgerError",
    "ROLE_SELECTOR",
    "ROLE_SOLVER",
    "Reservation",
    "ResourceEvent",
    "StopReason",
]
