"""The collection ledger: exact-decimal global spending protection (MASTER_SPEC §7.1).

Invariant, checked atomically on every reservation::

    settled + in_flight + unresolved + new_reservation <= active_ceiling

Buckets per attempt:

* ``in_flight``  - reserved (``RESERVE``) or sent (``SENT``) and not yet resolved;
* ``unresolved`` - the HTTP exchange ended without a confirmed cost (ambiguous timeout, error envelope, missing usage;
  journaled as ``UNRESOLVED`` with amount 0) or the attempt was inherited from a previous process; kept at full
  reservation until ``settle`` (confirmed cost, e.g. from generation metadata) or ``charge_max``;
* ``settled``    - confirmed cost (``SETTLE``) or conservative maximum (``CHARGE_MAX``).

``SENT`` is journaled (urgent fsync) immediately before the request leaves, never earlier. Only ``release_unsent``
returns a reservation without charge, and only for attempts that were reserved but never marked sent; after ``SENT``
it raises (unknown money is never released - a provably unsent CONNECT_FAILED settles at 0 instead). Every transition is journaled (``LedgerJournal.append`` returns after fsync); the in-memory state is
updated *before* the await so concurrent coroutines always see the reservation, and the constructor rebuilds the
state by replaying the journal. Attempts reserved by an earlier process and never resolved come back as
``unresolved`` (``inherited_attempt_ids``); recovery settles them from a persisted response or charges the maximum.

Reservations above ``max_unresolved_usd`` of unresolved exposure are refused (``last_refusal ==
"unresolved_exposure"``) so admissions pause until reconciliation.

``set_active_ceiling`` takes ``min(study cap, key remaining, account balance)`` expressed in *study* dollars; use
:func:`active_ceiling_from_provider` to convert provider "remaining" figures so a shared account's unrelated usage
lowers the ceiling instead of being counted as study spend. ``set_run_cap`` is the dispatcher's per-``run()`` cap
(``--max-usd``, pilot allowance); the enforced (and snapshot) ceiling is ``min(active ceiling, run cap)``, and clearing
the run cap never raises the active ceiling the lead set.

Config keys consumed: none directly (the lead passes ``budget.credit_ceiling_usd`` / the run's ``--max-usd`` as
``ceiling_usd`` and ``budget.max_unresolved_spend_usd`` as ``max_unresolved_usd``; the dispatcher verifies both).
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent_allocation.contracts import GenerationResult, LedgerEvent, LedgerEventKind, TransportStatus
from agent_allocation.storage.journal import JournalError, LedgerJournal

ZERO = Decimal("0")

_RESERVED = "reserved"
_SENT = "sent"
_UNRESOLVED = "unresolved"


class LedgerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dec(value: Any, name: str) -> Decimal:
    if isinstance(value, float):
        value = repr(value)
    try:
        out = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001
        raise LedgerError(f"{name} is not a decimal amount: {value!r}") from exc
    if not out.is_finite():
        raise LedgerError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class LedgerSnapshot:
    settled: Decimal
    in_flight: Decimal
    unresolved: Decimal
    ceiling: Decimal

    @property
    def exposure(self) -> Decimal:
        return self.settled + self.in_flight + self.unresolved

    @property
    def headroom(self) -> Decimal:
        return self.ceiling - self.exposure


@dataclass
class _Open:
    attempt_id: str
    node_id: str | None
    partition_id: str | None
    amount: Decimal
    state: str


@dataclass(frozen=True)
class OpenAttempt:
    attempt_id: str
    node_id: str | None
    partition_id: str | None
    amount_usd: Decimal
    state: str               # reserved | sent | unresolved


def active_ceiling_from_provider(*, study_cap: Decimal, settled: Decimal, key_limit_remaining: Decimal | None,
                                 account_balance: Decimal | None) -> Decimal:
    """``min(study cap, settled + key remaining, settled + balance)`` in study dollars.

    Provider "remaining" figures already exclude everything billed so far (this study's settled spend and any
    unrelated usage on the account), so ``settled + remaining`` is what this study may still reach; unrelated
    usage lowers it rather than being mistaken for study spend. ``None`` (unknown / no key limit) does not bind.
    """
    ceiling = _dec(study_cap, "study_cap")
    for remaining in (key_limit_remaining, account_balance):
        if remaining is not None:
            ceiling = min(ceiling, _dec(settled, "settled") + max(ZERO, _dec(remaining, "remaining")))
    return ceiling


class CollectionLedger:
    def __init__(self, journal: LedgerJournal, *, ceiling_usd: Decimal, max_unresolved_usd: Decimal) -> None:
        self.journal = journal
        self.study_cap = _dec(ceiling_usd, "ceiling_usd")
        if self.study_cap < ZERO:
            raise LedgerError("ceiling_usd must be >= 0")
        self.max_unresolved_usd = _dec(max_unresolved_usd, "max_unresolved_usd")
        self._ceiling = self.study_cap
        self._run_cap: Decimal | None = None
        self._lock = threading.Lock()
        self._open: dict[str, _Open] = {}
        self._resolved: dict[str, tuple[LedgerEventKind, Decimal]] = {}
        self._settled = ZERO
        self._in_flight = ZERO
        self._unresolved = ZERO
        self.last_refusal: str | None = None
        self.duplicate_resolutions = 0
        self.late_bills: list[tuple[str, Decimal]] = []
        self.overruns: list[tuple[str, Decimal, Decimal]] = []
        self.peak_exposure = ZERO
        self._replay()
        self.inherited_attempt_ids: frozenset[str] = frozenset(self._open)

    # ---- replay -------------------------------------------------------------------------------------------
    def _replay(self) -> None:
        for event in self.journal.replay():
            aid = event.attempt_id
            if event.kind is LedgerEventKind.RESERVE:
                if aid is None or aid in self._open or aid in self._resolved:
                    raise LedgerError(f"journal seq {event.seq}: duplicate or anonymous reservation {aid!r}")
                self._open[aid] = _Open(aid, event.node_id, event.partition_id, event.amount_usd, _RESERVED)
            elif event.kind is LedgerEventKind.SENT:
                if aid in self._open:
                    self._open[aid].state = _SENT
            elif event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX, LedgerEventKind.RELEASE_UNSENT):
                if aid in self._resolved:
                    continue
                opened = self._open.pop(aid, None)
                if opened is None:
                    raise LedgerError(f"journal seq {event.seq}: {event.kind.value} for unknown attempt {aid!r}")
                if event.kind is not LedgerEventKind.RELEASE_UNSENT:
                    self._settled += event.amount_usd
                self._resolved[aid] = (event.kind, event.amount_usd)
            elif event.kind is LedgerEventKind.UNRESOLVED:
                if aid in self._open:
                    self._open[aid].state = _UNRESOLVED
        # Nothing is in flight in a fresh process: every inherited open attempt (SENT or not) is unresolved exposure
        # and feeds the unresolved-exposure pause until recovery settles or charges it.
        for opened in self._open.values():
            opened.state = _UNRESOLVED
            self._unresolved += opened.amount
        self.peak_exposure = self._settled + self._unresolved

    # ---- views --------------------------------------------------------------------------------------------
    def snapshot(self) -> LedgerSnapshot:
        with self._lock:
            return LedgerSnapshot(self._settled, self._in_flight, self._unresolved, self._effective_ceiling_locked())

    def fits(self, amount_usd: Decimal, *, ceiling_usd: Decimal | None = None) -> bool:
        """Non-mutating: would ``reserve(amount)`` pass (optionally under a lower soft ceiling)?"""
        amount = _dec(amount_usd, "amount_usd")
        with self._lock:
            return self._refusal_locked(amount, ceiling_usd) is None

    def open_attempts(self) -> list[OpenAttempt]:
        with self._lock:
            return [OpenAttempt(o.attempt_id, o.node_id, o.partition_id, o.amount, o.state) for o in self._open.values()]

    def state_of(self, attempt_id: str) -> str | None:
        """``reserved|sent|unresolved`` for open attempts, the resolving event kind value, or None if unknown."""
        with self._lock:
            if attempt_id in self._open:
                return self._open[attempt_id].state
            if attempt_id in self._resolved:
                return self._resolved[attempt_id][0].value
            return None

    def reservation_of(self, attempt_id: str) -> Decimal | None:
        with self._lock:
            opened = self._open.get(attempt_id)
            return None if opened is None else opened.amount

    @property
    def unresolved_exceeded(self) -> bool:
        with self._lock:
            return self._unresolved > self.max_unresolved_usd

    # ---- ceiling ------------------------------------------------------------------------------------------
    def _effective_ceiling_locked(self) -> Decimal:
        return self._ceiling if self._run_cap is None else min(self._ceiling, self._run_cap)

    @property
    def active_ceiling(self) -> Decimal:
        """The lead's provider-derived ceiling, without the per-run cap."""
        with self._lock:
            return self._ceiling

    def set_run_cap(self, cap_usd: Decimal | None) -> None:
        cap = None if cap_usd is None else _dec(cap_usd, "cap_usd")
        if cap is not None and cap < ZERO:
            raise LedgerError("run cap must be >= 0")
        with self._lock:
            self._run_cap = cap

    def set_active_ceiling(self, ceiling_usd: Decimal) -> None:
        ceiling = _dec(ceiling_usd, "ceiling_usd")
        if ceiling < ZERO:
            raise LedgerError("active ceiling must be >= 0")
        with self._lock:
            self._ceiling = min(ceiling, self.study_cap)

    # ---- transitions --------------------------------------------------------------------------------------
    def _refusal_locked(self, amount: Decimal, ceiling_usd: Decimal | None) -> str | None:
        if self._unresolved > self.max_unresolved_usd:
            return "unresolved_exposure"
        effective = self._effective_ceiling_locked()
        ceiling = effective if ceiling_usd is None else min(effective, _dec(ceiling_usd, "ceiling_usd"))
        if self._settled + self._in_flight + self._unresolved + amount > ceiling:
            return "ceiling"
        return None

    def _event_locked(self, kind: LedgerEventKind, amount: Decimal, opened: _Open | None, *, attempt_id: str,
                      detail: str | None = None) -> LedgerEvent:
        return LedgerEvent(
            seq=self.journal.next_seq(), kind=kind, amount_usd=amount, recorded_at=utc_now(), attempt_id=attempt_id,
            node_id=None if opened is None else opened.node_id,
            partition_id=None if opened is None else opened.partition_id, detail=detail)

    async def reserve(self, *, attempt_id: str, node_id: str, partition_id: str, amount_usd: Decimal) -> bool:
        amount = _dec(amount_usd, "amount_usd")
        if amount <= ZERO:
            raise LedgerError(f"reservation for {attempt_id} must be positive, got {amount}")
        with self._lock:
            if attempt_id in self._open or attempt_id in self._resolved:
                raise LedgerError(f"attempt {attempt_id} was already reserved")
            refusal = self._refusal_locked(amount, None)
            self.last_refusal = refusal
            if refusal is not None:
                return False
            opened = _Open(attempt_id, node_id, partition_id, amount, _RESERVED)
            self._open[attempt_id] = opened
            self._in_flight += amount
            self.peak_exposure = max(self.peak_exposure, self._settled + self._in_flight + self._unresolved)
            try:
                event = LedgerEvent(seq=self.journal.next_seq(), kind=LedgerEventKind.RESERVE, amount_usd=amount,
                                    recorded_at=utc_now(), attempt_id=attempt_id, node_id=node_id,
                                    partition_id=partition_id)
                line_future = self.journal.submit(event)
            except BaseException:
                self._open.pop(attempt_id, None)
                self._in_flight -= amount
                raise
        try:
            await asyncio.shield(line_future)
        except JournalError as exc:
            with self._lock:
                if not exc.maybe_written and self._open.get(attempt_id) is opened and opened.state == _RESERVED:
                    self._open.pop(attempt_id)
                    self._in_flight -= amount
                elif self._open.get(attempt_id) is opened and opened.state == _RESERVED:
                    # The line may be on disk: never pretend the reservation did not exist.
                    opened.state = _UNRESOLVED
                    self._in_flight -= amount
                    self._unresolved += amount
            raise
        return True

    async def mark_sent(self, attempt_id: str) -> None:
        """Journal ``SENT`` (urgent fsync) right before the request leaves; from here on it is never released."""
        with self._lock:
            opened = self._open.get(attempt_id)
            if opened is None:
                raise LedgerError(f"mark_sent for unknown or resolved attempt {attempt_id}")
            if opened.state != _RESERVED:
                return
            event = self._event_locked(LedgerEventKind.SENT, opened.amount, opened, attempt_id=attempt_id)
            pending = self.journal.submit(event, urgent=True)   # raises (state unchanged) if the journal is broken
            opened.state = _SENT
        await asyncio.shield(pending)

    async def mark_unresolved(self, attempt_id: str, *, detail: str = "") -> None:
        """Sent attempt whose exchange ended without a confirmed cost: keep the full reservation as unresolved."""
        with self._lock:
            opened = self._open.get(attempt_id)
            if opened is None:
                raise LedgerError(f"mark_unresolved for unknown or resolved attempt {attempt_id}")
            if opened.state == _UNRESOLVED:
                return
            opened.state = _UNRESOLVED
            self._in_flight -= opened.amount
            self._unresolved += opened.amount
            event = self._event_locked(LedgerEventKind.UNRESOLVED, ZERO, opened, attempt_id=attempt_id,
                                       detail=detail or None)
            pending = self.journal.submit(event)
        await asyncio.shield(pending)

    def _remove_locked(self, opened: _Open) -> None:
        del self._open[opened.attempt_id]
        if opened.state == _UNRESOLVED:
            self._unresolved -= opened.amount
        else:
            self._in_flight -= opened.amount

    async def settle(self, attempt_id: str, cost_usd: Decimal) -> None:
        cost = _dec(cost_usd, "cost_usd")
        if cost < ZERO:
            raise LedgerError(f"negative cost {cost} for {attempt_id}")
        with self._lock:
            opened = self._open.get(attempt_id)
            if opened is None:
                prior = self._resolved.get(attempt_id)
                if prior is None:
                    raise LedgerError(f"settle for unknown attempt {attempt_id}")
                self.duplicate_resolutions += 1
                if prior[0] is not LedgerEventKind.CHARGE_MAX:
                    return
                # A bill that arrives after the conservative charge: note it, change nothing.
                self.late_bills.append((attempt_id, cost))
                event = LedgerEvent(seq=self.journal.next_seq(), kind=LedgerEventKind.RECONCILE, amount_usd=ZERO,
                                    recorded_at=utc_now(), attempt_id=attempt_id,
                                    detail=f"late_bill_after_charge_max:{cost}")
                pending = self.journal.submit(event)
            else:
                detail = None
                if cost > opened.amount:
                    self.overruns.append((attempt_id, opened.amount, cost))
                    detail = f"cost_exceeds_reservation:{opened.amount}"
                self._remove_locked(opened)
                self._settled += cost
                self._resolved[attempt_id] = (LedgerEventKind.SETTLE, cost)
                self.peak_exposure = max(self.peak_exposure, self._settled + self._in_flight + self._unresolved)
                event = self._event_locked(LedgerEventKind.SETTLE, cost, opened, attempt_id=attempt_id, detail=detail)
                pending = self.journal.submit(event)
        await asyncio.shield(pending)

    async def release_unsent(self, attempt_id: str) -> None:
        with self._lock:
            opened = self._open.get(attempt_id)
            if opened is None:
                if attempt_id in self._resolved:
                    self.duplicate_resolutions += 1
                    return
                raise LedgerError(f"release_unsent for unknown attempt {attempt_id}")
            if opened.state != _RESERVED:
                raise LedgerError(f"attempt {attempt_id} is {opened.state}: a reservation that may have left the "
                                  "process is never released (settle a provably unsent attempt at 0 instead)")
            self._remove_locked(opened)
            self._resolved[attempt_id] = (LedgerEventKind.RELEASE_UNSENT, opened.amount)
            event = self._event_locked(LedgerEventKind.RELEASE_UNSENT, opened.amount, opened, attempt_id=attempt_id)
            pending = self.journal.submit(event)
        await asyncio.shield(pending)

    async def charge_max(self, attempt_id: str) -> None:
        with self._lock:
            opened = self._open.get(attempt_id)
            if opened is None:
                if attempt_id in self._resolved:
                    self.duplicate_resolutions += 1
                    return
                raise LedgerError(f"charge_max for unknown attempt {attempt_id}")
            self._remove_locked(opened)
            self._settled += opened.amount
            self._resolved[attempt_id] = (LedgerEventKind.CHARGE_MAX, opened.amount)
            event = self._event_locked(LedgerEventKind.CHARGE_MAX, opened.amount, opened, attempt_id=attempt_id)
            pending = self.journal.submit(event)
        await asyncio.shield(pending)

    async def note(self, detail: str, *, attempt_id: str | None = None) -> None:
        """Provider-side reconciliation note (``RECONCILE``, amount 0, no reservation change)."""
        with self._lock:
            event = LedgerEvent(seq=self.journal.next_seq(), kind=LedgerEventKind.RECONCILE, amount_usd=ZERO,
                                recorded_at=utc_now(), attempt_id=attempt_id, detail=detail)
            pending = self.journal.submit(event)
        await asyncio.shield(pending)


# ---------------------------------------------------------------------------------------------------------------
# Shared cost rules (dispatcher, recovery, status) - one place decides what a finished exchange costs.
# ---------------------------------------------------------------------------------------------------------------

# Statuses whose response itself proves no generation was billed when it carries no usage: the request was refused
# before any model ran (rate limit, auth, payment, bad request).
NEVER_BILLED_WITHOUT_USAGE = frozenset({
    TransportStatus.RATE_LIMITED,
    TransportStatus.AUTH_FAILED,
    TransportStatus.PAYMENT_REQUIRED,
    TransportStatus.BAD_REQUEST,
})


def confirmed_cost(result: GenerationResult | None) -> Decimal | None:
    """The cost a finished exchange provably incurred, or ``None`` when it is unknown (keep it unresolved).

    ``usage.actual_cost_usd`` wins whenever present. Without it: refusals in ``NEVER_BILLED_WITHOUT_USAGE``,
    CONNECT_FAILED (provably never sent) and ``UPSTREAM_ERROR`` without a generation id cost 0; everything else
    (timeouts, 200 responses without usage, error envelopes, cache hits, provider mismatches, 5xx with a generation
    id) is unknown and stays unresolved until generation metadata or ``charge_max``.
    """
    if result is None:
        return None
    if result.usage is not None and result.usage.actual_cost_usd is not None:
        return _dec(result.usage.actual_cost_usd, "actual_cost_usd")
    status = result.transport_status
    if status in NEVER_BILLED_WITHOUT_USAGE or status is TransportStatus.CONNECT_FAILED:
        return ZERO
    if status is TransportStatus.UPSTREAM_ERROR and not result.generation_id:
        return ZERO
    return None


@dataclass(frozen=True)
class JournalAttempt:
    attempt_id: str
    node_id: str | None
    partition_id: str | None
    reserved_usd: Decimal
    sent: bool
    resolution: LedgerEventKind | None
    resolved_usd: Decimal | None


def journal_attempts(journal: "LedgerJournal | Path") -> dict[str, JournalAttempt]:
    """Every attempt the journal knows, with its first resolution (``None`` = still open)."""
    from agent_allocation.storage.journal import replay_events

    events = journal.replay() if isinstance(journal, LedgerJournal) else replay_events(journal)
    found: dict[str, dict[str, Any]] = {}
    for event in events:
        aid = event.attempt_id
        if aid is None:
            continue
        if event.kind is LedgerEventKind.RESERVE:
            found[aid] = {"attempt_id": aid, "node_id": event.node_id, "partition_id": event.partition_id,
                          "reserved_usd": event.amount_usd, "sent": False, "resolution": None, "resolved_usd": None}
        elif aid in found:
            row = found[aid]
            if event.kind is LedgerEventKind.SENT:
                row["sent"] = True
            elif event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX, LedgerEventKind.RELEASE_UNSENT) \
                    and row["resolution"] is None:
                row["resolution"] = event.kind
                row["resolved_usd"] = event.amount_usd
    return {aid: JournalAttempt(**row) for aid, row in found.items()}


def attempt_number(attempt_id: str) -> int:
    try:
        return int(attempt_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise LedgerError(f"attempt id {attempt_id!r} is not '<node_id>:<attempt_no>'") from exc
