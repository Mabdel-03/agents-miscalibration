"""The ready-node DAG dispatcher: one asyncio coordinator, one SQLite writer, one global ledger (MASTER_SPEC §10).

How a node runs (outbox ordering; every step durable before the next):

1. *ready* - every parent is terminal (with or without a candidate); the node's wave is open.
2. *admission* (coordinator, synchronous) - fair pick, render, conservative estimate, budget pre-check, rate token.
3. ``ledger.reserve`` (journal fsync) -> ``store.try_lease`` -> ``Attempt(LEASED)`` -> acquire the HTTP limiter ->
   re-check stop/drain/circuit (withdraw with ``release_unsent`` if set) -> ``ledger.mark_sent`` (urgent fsync, so
   ``SENT`` is durable before the request leaves) -> ``Attempt(SENT)`` -> ``client.generate`` -> release the limiter.
   The limiter covers the HTTP exchange plus that one urgent fsync: never rendering, reservation waits, backoff
   sleeps, extraction or database queues.
4. ``write_raw_response`` (fsync, before any commit) -> settle (confirmed cost; CONNECT_FAILED settles 0) or keep
   unresolved -> ``store.record_persisted_response`` -> ``extract`` (worker thread) -> ``store.record_result``
   (parsed/ + DB) -> children unlocked immediately.

Context limit (R03-1): when ``render`` raises ``prompts.render.ContextLimitEvent`` the node becomes terminal with
outcome ``context_limit``: no reservation, no API call, no candidate; its children render the absent marker.

Transport outcomes (``contracts``): OK -> candidate (every completion status is terminal, never redrawn);
``RETRYABLE_TRANSPORT`` -> new attempt after randomized backoff, at most ``max_transport_attempts`` attempts per node,
then the node is terminal without a candidate and its children still run (absent-response marker);
``CIRCUIT_BREAKING_TRANSPORT`` -> circuit opens: admissions stop, in-flight work drains, reason recorded; any other
status -> terminal without candidate. AMBIGUOUS_TIMEOUT and ERROR_ENVELOPE (and any exchange without a confirmed
cost) keep the full reservation as *unresolved* until generation metadata confirms the cost or the reconciliation
horizon (``lease_stale_seconds``) passes, then ``charge_max``; admissions pause while unresolved exposure exceeds
``max_unresolved_spend_usd``.

``RunSummary.stop_reason`` is one of ``STOP_REASONS`` (``completed``, ``budget_exhausted``, ``circuit_open``,
``stop_sentinel``, ``drain``, ``control_sentinel_present``, ``stalled``); ``circuit_reason`` / ``stop_detail`` carry the
specifics.

Waves: waves open in plan order, at most ``max_active_waves`` incomplete at once, and a new one opens only when the
ready work of open waves cannot fill the dispatcher. Pick order among ready nodes: older wave, then critical path
(class 0: root[0] and its private chain; class 1: roots 1-4; class 2: the rest), then the least-served
(benchmark, kind) bucket, then FIFO. At most ``max_calls_per_item_in_flight`` attempts per item are active.
Optional (replication) waves open only after every core wave is complete when ``optional_requires_core_completion``.

AIMD: start at ``concurrency_start`` in-flight / ``requests_per_second_start`` admissions per second; after each
``aimd_window_seconds`` window with no 429 in which a limit was binding, add ``increase_concurrency_by`` /
``increase_rps_by`` up to ``concurrency_max`` / ``requests_per_second_max``; on a 429 for a request sent after the
last decrease, multiply both by ``decrease_factor`` and pause admissions for ``Retry-After``.

Config keys consumed (each changes behaviour):
``scheduler``: concurrency_start, concurrency_max, requests_per_second_start, requests_per_second_max,
aimd_window_seconds, increase_concurrency_by, increase_rps_by, decrease_factor, max_transport_attempts,
connect_timeout_seconds + read_timeout_seconds (watchdog around ``client.generate``), lease_heartbeat_seconds,
lease_stale_seconds (foreign-lease takeover and the unresolved reconciliation horizon), max_active_waves,
wave_items (maximum items per wave), max_calls_per_item_in_flight, single_paid_dispatcher (refuse to start while
another live dispatcher owns the state).
``budget``: credit_ceiling_usd (``max_usd`` above it is refused), soft_admission_ceiling_usd (first attempts must fit
under it; retries may use the reserve up to the hard ceiling), pilot_allowance_usd (caps plans made only of canary/
pilot partitions), reservation_multiplier, max_unresolved_spend_usd (must equal the ledger's), and
optional_requires_core_completion.
Not consumed: scheduler.checkpoint_after_every_response (always true here), scheduler.api_sdk_automatic_retries
(client), budget.cash_target_usd, budget.contingency_usd (implied by credit - soft ceilings).
"""
from __future__ import annotations

import asyncio
import math
import os
import random
import socket
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from agent_allocation.contracts import (
    CIRCUIT_BREAKING_TRANSPORT, RETRYABLE_TRANSPORT, Attempt, AttemptState, Candidate, EndpointLock,
    GenerationKind, GenerationOutcome, GenerationResult, GenerationSpec, ItemGraph, Partition, TransportStatus,
)
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.estimate import rendered_chars, reservation_usd
from agent_allocation.runtime.ledger import CollectionLedger, LedgerError, confirmed_cost
from agent_allocation.storage.raw import read_raw_response, result_from_raw, sha256_file, write_raw_response
from agent_allocation.storage.state import OUTCOME_SUCCEEDED, StateStore

RenderFn = Callable[[GenerationSpec, Sequence[tuple[GenerationSpec, Candidate | None]]], list[dict[str, str]]]
ExtractFn = Callable[[GenerationResult, GenerationSpec], Candidate]

STATUS_INTERVAL_SECONDS = 10.0
BACKUP_INTERVAL_SECONDS = 120.0
RECONCILE_POLL_SECONDS = 30.0
CONTROL_POLL_SECONDS = 1.0
STOP_GRACE_SECONDS = 30.0
WATCHDOG_MARGIN_SECONDS = 30.0
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 60.0
MIN_RPS = 0.25
NOT_SENT = "not_sent"

STOP_COMPLETED = "completed"
STOP_BUDGET_EXHAUSTED = "budget_exhausted"
STOP_CIRCUIT_OPEN = "circuit_open"
STOP_SENTINEL = "stop_sentinel"
STOP_DRAIN = "drain"
STOP_CONTROL_SENTINEL_PRESENT = "control_sentinel_present"
STOP_STALLED = "stalled"
STOP_REASONS = frozenset({STOP_COMPLETED, STOP_BUDGET_EXHAUSTED, STOP_CIRCUIT_OPEN, STOP_SENTINEL, STOP_DRAIN,
                          STOP_CONTROL_SENTINEL_PRESENT, STOP_STALLED})

OUTCOME_CONTEXT_LIMIT = "context_limit"
OPTIONAL_PARTITIONS = frozenset({Partition.REPLICATION.value})
PILOT_PARTITIONS = frozenset({Partition.CANARY.value, Partition.PILOT.value})


class PlanError(ValueError):
    """The admission plan is malformed (missing parent, oversized wave, conflicting duplicate)."""


class DispatchRefused(RuntimeError):
    """The dispatcher refuses to start (unsafe client, budget configuration, another live dispatcher)."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class AdmissionPlan:
    waves: tuple[tuple[ItemGraph, ...], ...]


@dataclass(frozen=True)
class RunSummary:
    stop_reason: str
    nodes_planned: int
    nodes_succeeded: int
    nodes_failed: int
    nodes_pending: int
    requests_sent: int
    attempts_by_status: Mapping[str, int]
    rate_limited: int
    retries_scheduled: int
    settled_usd: Decimal
    in_flight_usd: Decimal
    unresolved_usd: Decimal
    ceiling_usd: Decimal
    soft_ceiling_usd: Decimal
    spent_this_run_usd: Decimal
    peak_in_flight: int
    mean_in_flight: float
    mean_in_flight_active: float
    concurrency_start: int
    concurrency_final: int
    concurrency_max_reached: int
    rps_final: float
    wall_seconds: float
    waves_total: int
    waves_opened: int
    waves_completed: int
    circuit_reason: str | None
    duplicate_notifications: int
    stop_detail: str | None = None
    nodes_context_limited: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------

def _need(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise DispatchRefused(f"config {where}.{key} is required")
    return block[key]


@dataclass(frozen=True)
class SchedulerSettings:
    concurrency_start: int
    concurrency_max: int
    rps_start: float
    rps_max: float
    aimd_window_seconds: float
    increase_concurrency_by: int
    increase_rps_by: float
    decrease_factor: float
    max_transport_attempts: int
    watchdog_seconds: float
    lease_heartbeat_seconds: float
    lease_stale_seconds: float
    max_active_waves: int
    wave_items: int
    max_calls_per_item_in_flight: int
    single_paid_dispatcher: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SchedulerSettings":
        s = _need(config, "scheduler", "")
        out = cls(
            concurrency_start=int(_need(s, "concurrency_start", "scheduler")),
            concurrency_max=int(_need(s, "concurrency_max", "scheduler")),
            rps_start=float(_need(s, "requests_per_second_start", "scheduler")),
            rps_max=float(_need(s, "requests_per_second_max", "scheduler")),
            aimd_window_seconds=float(_need(s, "aimd_window_seconds", "scheduler")),
            increase_concurrency_by=int(_need(s, "increase_concurrency_by", "scheduler")),
            increase_rps_by=float(_need(s, "increase_rps_by", "scheduler")),
            decrease_factor=float(_need(s, "decrease_factor", "scheduler")),
            max_transport_attempts=int(_need(s, "max_transport_attempts", "scheduler")),
            watchdog_seconds=float(_need(s, "connect_timeout_seconds", "scheduler"))
            + float(_need(s, "read_timeout_seconds", "scheduler")) + WATCHDOG_MARGIN_SECONDS,
            lease_heartbeat_seconds=float(_need(s, "lease_heartbeat_seconds", "scheduler")),
            lease_stale_seconds=float(_need(s, "lease_stale_seconds", "scheduler")),
            max_active_waves=int(_need(s, "max_active_waves", "scheduler")),
            wave_items=int(_need(s, "wave_items", "scheduler")),
            max_calls_per_item_in_flight=int(_need(s, "max_calls_per_item_in_flight", "scheduler")),
            single_paid_dispatcher=bool(_need(s, "single_paid_dispatcher", "scheduler")),
        )
        if not (1 <= out.concurrency_start <= out.concurrency_max):
            raise DispatchRefused("scheduler concurrency_start must be within [1, concurrency_max]")
        if not (0 < out.rps_start <= out.rps_max):
            raise DispatchRefused("scheduler requests_per_second_start must be within (0, requests_per_second_max]")
        if not (0 < out.decrease_factor < 1):
            raise DispatchRefused("scheduler decrease_factor must be in (0, 1)")
        if out.max_transport_attempts < 1 or out.max_active_waves < 1 or out.max_calls_per_item_in_flight < 1:
            raise DispatchRefused("scheduler attempt/wave/item limits must be >= 1")
        return out


@dataclass(frozen=True)
class BudgetSettings:
    credit_ceiling_usd: Decimal
    soft_admission_ceiling_usd: Decimal
    pilot_allowance_usd: Decimal
    reservation_multiplier: Decimal
    max_unresolved_spend_usd: Decimal
    optional_requires_core_completion: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BudgetSettings":
        b = _need(config, "budget", "")
        dec = lambda key: Decimal(str(_need(b, key, "budget")))  # noqa: E731
        out = cls(dec("credit_ceiling_usd"), dec("soft_admission_ceiling_usd"), dec("pilot_allowance_usd"),
                  dec("reservation_multiplier"), dec("max_unresolved_spend_usd"),
                  bool(_need(b, "optional_requires_core_completion", "budget")))
        if out.soft_admission_ceiling_usd > out.credit_ceiling_usd:
            raise DispatchRefused("budget soft_admission_ceiling_usd exceeds credit_ceiling_usd")
        if out.reservation_multiplier < 1:
            raise DispatchRefused("budget reservation_multiplier must be >= 1")
        return out


# ---------------------------------------------------------------------------------------------------------------
# helpers: client guard, owner identity, limiter, AIMD, meter
# ---------------------------------------------------------------------------------------------------------------

def _find_transports(obj: Any, depth: int = 0, seen: set[int] | None = None) -> list[Any]:
    import httpx

    seen = set() if seen is None else seen
    if obj is None or id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    if isinstance(obj, httpx.AsyncBaseTransport):
        found = [obj]
        # Wrapping transports (e.g. retries/logging) expose the inner one.
        for name in ("_transport", "transport", "_wrapped", "wrapped", "_pool"):
            inner = getattr(obj, name, None)
            if inner is not None and inner is not obj:
                found.extend(_find_transports(inner, depth + 1, seen))
        return found
    if isinstance(obj, (str, bytes, int, float, bool, Decimal)):
        return []
    if isinstance(obj, httpx.AsyncClient):
        found = _find_transports(getattr(obj, "_transport", None), depth + 1, seen)
        for mounted in (getattr(obj, "_mounts", None) or {}).values():
            found.extend(_find_transports(mounted, depth + 1, seen))
        return found
    found: list[Any] = []
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict):
        for value in attrs.values():
            if isinstance(value, (httpx.AsyncClient, httpx.AsyncBaseTransport)) or hasattr(value, "__dict__"):
                found.extend(_find_transports(value, depth + 1, seen))
    return found


def client_is_offline_mock(client: Any) -> bool:
    """True only if every transport reachable from the client is (or wraps nothing but) an ``httpx.MockTransport``."""
    import httpx

    transports = _find_transports(client)
    if not transports:
        return False
    wrapped = {id(getattr(t, n)) for t in transports for n in ("_transport", "transport", "_wrapped", "wrapped")
               if isinstance(getattr(t, n, None), httpx.AsyncBaseTransport)}
    leaves = [t for t in transports if id(t) not in wrapped or isinstance(t, httpx.MockTransport)]
    leaves = [t for t in leaves if not any(isinstance(getattr(t, n, None), httpx.AsyncBaseTransport)
                                           for n in ("_transport", "transport", "_wrapped", "wrapped"))]
    return bool(leaves) and all(isinstance(t, httpx.MockTransport) for t in leaves)


def _proc_start_time(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b")", 1)[1].split()
        return fields[19].decode()
    except (OSError, IndexError):
        return None


def make_owner_id() -> str:
    pid = os.getpid()
    return f"{socket.gethostname()}|{pid}|{_proc_start_time(pid)}|{uuid.uuid4().hex[:12]}"


def owner_is_dead(owner: str | None) -> bool:
    """Provably dead: recorded on this host and the pid is gone or belongs to a different process now."""
    if not owner:
        return True
    try:
        host, pid_s, start, _ = owner.split("|", 3)
        pid = int(pid_s)
    except ValueError:
        return False
    if host != socket.gethostname():
        return False
    current = _proc_start_time(pid)
    return current is None or current != start


class HttpLimiter:
    """Counting semaphore with a mutable limit; held only around ``client.generate``."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._cond = asyncio.Condition()

    def set_limit(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    async def __aenter__(self) -> "HttpLimiter":
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < self.limit)
            self.active += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()


class Aimd:
    def __init__(self, settings: SchedulerSettings, clock: Callable[[], float]) -> None:
        self.s = settings
        self.clock = clock
        self.concurrency = float(settings.concurrency_start)
        self.rps = float(settings.rps_start)
        self.window_start = clock()
        self.last_decrease_at = -math.inf
        self.bound_in_window = False
        self.pause_until = 0.0
        self.increases = 0
        self.decreases = 0
        self.max_concurrency_reached = int(self.concurrency)
        self._tokens = 1.0
        self._token_time = clock()

    @property
    def concurrency_limit(self) -> int:
        return max(1, int(self.concurrency))

    def tick(self, now: float) -> bool:
        if now - self.window_start < self.s.aimd_window_seconds:
            return False
        changed = False
        if self.bound_in_window and (self.concurrency < self.s.concurrency_max or self.rps < self.s.rps_max):
            self.concurrency = min(float(self.s.concurrency_max), self.concurrency + self.s.increase_concurrency_by)
            self.rps = min(self.s.rps_max, self.rps + self.s.increase_rps_by)
            self.increases += 1
            self.max_concurrency_reached = max(self.max_concurrency_reached, self.concurrency_limit)
            changed = True
        self.window_start = now
        self.bound_in_window = False
        return changed

    def on_rate_limited(self, now: float, sent_at: float, retry_after: float | None) -> bool:
        if retry_after is not None and retry_after > 0:
            self.pause_until = max(self.pause_until, now + retry_after)
        if sent_at < self.last_decrease_at:
            return False  # this request was admitted under the old allowance: one decrease per congestion event
        self.concurrency = max(1.0, math.floor(self.concurrency * self.s.decrease_factor))
        self.rps = max(MIN_RPS, self.rps * self.s.decrease_factor)
        self.last_decrease_at = now
        self.window_start = now
        self.bound_in_window = False
        self.decreases += 1
        self._tokens = min(self._tokens, 1.0)
        return True

    def token_available(self, now: float) -> bool:
        self._tokens = min(max(1.0, self.rps), self._tokens + (now - self._token_time) * self.rps)
        self._token_time = now
        return self._tokens >= 1.0

    def take_token(self) -> None:
        self._tokens -= 1.0

    def seconds_until_token(self, now: float) -> float:
        self.token_available(now)
        return 0.0 if self._tokens >= 1.0 else (1.0 - self._tokens) / self.rps


class InFlightMeter:
    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.start = clock()
        self.value = 0
        self.peak = 0
        self._last = self.start
        self._area = 0.0
        self.first_send: float | None = None
        self.last_done: float | None = None

    def _advance(self, now: float) -> None:
        self._area += self.value * (now - self._last)
        self._last = now

    def change(self, delta: int) -> None:
        now = self.clock()
        self._advance(now)
        self.value += delta
        self.peak = max(self.peak, self.value)
        if delta > 0 and self.first_send is None:
            self.first_send = now
        if delta < 0:
            self.last_done = now

    def means(self) -> tuple[float, float]:
        now = self.clock()
        self._advance(now)
        total = now - self.start
        overall = self._area / total if total > 0 else 0.0
        if self.first_send is None:
            return overall, 0.0
        end = self.last_done if self.last_done is not None else now
        active = end - self.first_send
        return overall, (self._area / active if active > 0 else float(self.peak))


# ---------------------------------------------------------------------------------------------------------------
# in-memory graph
# ---------------------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class _Node:
    spec: GenerationSpec
    node_id: str
    wave: int
    item_key: tuple[str, str]
    crit: int
    bucket: tuple[str, str]
    seq: int
    parents: tuple[str, ...]
    children: list["_Node"] = field(default_factory=list)
    waiting_parents: int = 0
    terminal: bool = False
    succeeded: bool = False
    attempts_used: int = 0           # highest attempt number used (attempt ids are never reused)
    sends: int = 0                   # attempts that (may have) left the process: the transport-attempt limit
    not_before: float = 0.0
    status: str = "blocked"          # blocked | ready | active | parked | done
    foreign_lease: str | None = None


@dataclass(eq=False)
class _Wave:
    index: int
    nodes: list[_Node]
    optional: bool
    opened: bool = False
    remaining: int = 0


@dataclass
class _Unresolved:
    attempt_id: str
    node_id: str
    generation_id: str | None
    deadline: float


def _is_context_limit(exc: BaseException) -> bool:
    try:
        from agent_allocation.prompts.render import ContextLimitEvent
    except ImportError:  # pragma: no cover - agent 03's module not present
        return type(exc).__name__ == "ContextLimitEvent"
    return isinstance(exc, ContextLimitEvent)


def critical_class(spec: GenerationSpec) -> int:
    if spec.kind is GenerationKind.ROOT and spec.root_slot == 0:
        return 0
    if spec.kind is GenerationKind.PRIVATE and spec.root_slot == 0 and spec.team_size == 0:
        return 0
    if spec.kind is GenerationKind.ROOT and spec.root_slot < 5:
        return 1
    return 2


# ---------------------------------------------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------------------------------------------

class Dispatcher:
    def __init__(self, *, config: Mapping[str, Any], paths: RunPaths, store: StateStore, ledger: CollectionLedger,
                 client: Any, lock: EndpointLock, render: RenderFn, extract: ExtractFn) -> None:
        self.config = config
        self.paths = paths
        self.store = store
        self.ledger = ledger
        self.client = client
        self.lock = lock
        self.render = render
        self.extract = extract
        self.settings = SchedulerSettings.from_config(config)
        self.budget = BudgetSettings.from_config(config)
        self.sampling = dict(config.get("sampling", {}))
        self.clock: Callable[[], float] = time.monotonic
        self.rng = random.Random()
        self.owner = make_owner_id()
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.aimd = Aimd(self.settings, self.clock)
        self.meter = InFlightMeter(self.clock)
        self.nodes: dict[str, _Node] = {}
        self.waves: list[_Wave] = []
        self.ready: dict[str, _Node] = {}
        self.item_active: dict[tuple[str, str], int] = defaultdict(int)
        self.bucket_served: dict[tuple[str, str], int] = defaultdict(int)
        self.unresolved: dict[str, _Unresolved] = {}
        self.tasks: set[asyncio.Task] = set()
        self.admit_blocker = "empty"
        self.pre_send = 0                          # admitted, not yet holding the HTTP limiter
        self.pending_reservation = Decimal("0")    # admitted, reservation not yet journaled
        self.requests_sent = 0
        self.rate_limited = 0
        self.retries_scheduled = 0
        self.duplicate_notifications = 0
        self.context_limited = 0
        self.attempt_statuses: dict[str, int] = defaultdict(int)
        self.circuit_reason: str | None = None
        self.stop_mode: str | None = None          # "stop" | "drain"
        self.budget_blocked_at: Decimal | None = None
        self.stop_reason: str | None = None
        self.stop_detail: str | None = None
        self.events: list[tuple[float, str, str]] = []
        self.soft_ceiling = Decimal("0")
        self.run_ceiling = Decimal("0")
        self._seq = 0

    # ---- public entry ---------------------------------------------------------------------------------------
    async def run(self, plan: AdmissionPlan, *, allow_paid: bool, max_usd: Decimal) -> RunSummary:
        self._reset_run_state()
        started = self.clock()
        if not allow_paid and not client_is_offline_mock(self.client):
            raise DispatchRefused("allow_paid=False but the client transport is not an httpx.MockTransport")
        self._configure_budget(plan, Decimal(str(max_usd)))
        settled_before = self.ledger.snapshot().settled
        self.store.owner = self.owner
        self._build(plan)
        self._loop = asyncio.get_running_loop()
        self.http = HttpLimiter(self.aimd.concurrency_limit)
        self.hard_stop: asyncio.Future = self._loop.create_future()
        self.wake = asyncio.Event()
        if self.paths.stop_sentinel.exists() or self.paths.drain_sentinel.exists():
            self._finish(STOP_CONTROL_SENTINEL_PRESENT, "STOP" if self.paths.stop_sentinel.exists() else "DRAIN")
            return self._summary(started, settled_before)
        self._claim_dispatcher()
        self.cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix="extract")
        background: list[asyncio.Task] = []
        try:
            await self._startup_reconcile()
            background = [self._loop.create_task(coro) for coro in (
                self._status_loop(), self._backup_loop(), self._heartbeat_loop(), self._reconcile_loop())]
            await self._main_loop()
            await self._drain()
        finally:
            for task in background:
                task.cancel()
            for task in background:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            self.cpu.shutdown(wait=True)
            await self._final_persist()
            if self.store.get_meta("dispatcher_owner") == self.owner:
                self.store.delete_meta("dispatcher_owner")
        return self._summary(started, settled_before)

    # ---- setup ----------------------------------------------------------------------------------------------
    def _configure_budget(self, plan: AdmissionPlan, max_usd: Decimal) -> None:
        b = self.budget
        if max_usd > b.credit_ceiling_usd:
            raise DispatchRefused(f"max_usd {max_usd} exceeds budget.credit_ceiling_usd {b.credit_ceiling_usd}")
        if self.ledger.max_unresolved_usd != b.max_unresolved_spend_usd:
            raise DispatchRefused("ledger max_unresolved_usd differs from budget.max_unresolved_spend_usd")
        if self.ledger.study_cap > b.credit_ceiling_usd:
            raise DispatchRefused("ledger ceiling exceeds budget.credit_ceiling_usd")
        ceiling = min(max_usd, b.credit_ceiling_usd)
        soft = min(b.soft_admission_ceiling_usd, ceiling)
        partitions = {g.partition_id for wave in plan.waves for g in wave}
        if partitions and partitions <= PILOT_PARTITIONS:
            ceiling = min(ceiling, b.pilot_allowance_usd)
            soft = min(soft, b.pilot_allowance_usd)
        self.ledger.set_active_ceiling(min(self.ledger.snapshot().ceiling, ceiling))
        self.run_ceiling = self.ledger.snapshot().ceiling
        self.soft_ceiling = min(soft, self.run_ceiling)

    def _build(self, plan: AdmissionPlan) -> None:
        seen_items: set[tuple[str, str]] = set()
        for index, wave_graphs in enumerate(plan.waves):
            if len(wave_graphs) > self.settings.wave_items:
                raise PlanError(f"wave {index} has {len(wave_graphs)} items; scheduler.wave_items is "
                                f"{self.settings.wave_items}")
            wave = _Wave(index, [], optional=bool(wave_graphs) and all(
                g.partition_id in OPTIONAL_PARTITIONS for g in wave_graphs))
            for graph in wave_graphs:
                key = (graph.partition_id, graph.item_id)
                if key in seen_items:
                    raise PlanError(f"item {key} appears twice in the plan")
                seen_items.add(key)
                for spec in graph.nodes:
                    node_id = spec.node_id
                    if node_id in self.nodes:
                        raise PlanError(f"node {node_id} appears in two item graphs")
                    if spec.item_id != graph.item_id or spec.partition_id != graph.partition_id:
                        raise PlanError(f"node {node_id} does not belong to item graph {key}")
                    self._seq += 1
                    node = _Node(spec, node_id, index, key, critical_class(spec), (spec.benchmark, spec.kind.value),
                                 self._seq, tuple(spec.parent_ids))
                    self.nodes[node_id] = node
                    wave.nodes.append(node)
            self.waves.append(wave)
        missing = sorted({(n.node_id, p) for n in self.nodes.values() for p in n.parents if p not in self.nodes})
        if missing:
            raise PlanError(f"{len(missing)} parent references are not in the plan, e.g. node {missing[0][0]} -> "
                            f"missing parent {missing[0][1]}")
        for wave in self.waves:
            self.store.register_nodes((n.spec for n in wave.nodes), wave_index=wave.index)
        rows = self.store.nodes(self.nodes)
        sends = self.store.sends_by_node()
        for node in self.nodes.values():
            for parent in node.parents:
                self.nodes[parent].children.append(node)
            row = rows[node.node_id]
            node.attempts_used = row.attempts_used
            node.sends = sends.get(node.node_id, 0)
            if row.state == AttemptState.TERMINAL.value:
                node.terminal = True
                node.succeeded = row.outcome == OUTCOME_SUCCEEDED
                node.status = "done"
        for node in self.nodes.values():
            node.waiting_parents = sum(1 for p in node.parents if not self.nodes[p].terminal)
        for wave in self.waves:
            wave.remaining = sum(1 for n in wave.nodes if not n.terminal)

    def _claim_dispatcher(self) -> None:
        if self.settings.single_paid_dispatcher:
            owner = self.store.get_meta("dispatcher_owner")
            beat = self.store.get_meta("dispatcher_heartbeat")
            if owner and owner != self.owner and not owner_is_dead(owner) and beat is not None \
                    and time.time() - float(beat) < self.settings.lease_stale_seconds:
                raise DispatchRefused(f"another dispatcher ({owner}) holds this run's state")
        self.store.set_meta("dispatcher_owner", self.owner)
        self.store.set_meta("dispatcher_heartbeat", repr(time.time()))

    async def _startup_reconcile(self) -> None:
        # (a) reservations an earlier process left open: settle from a persisted response or charge the maximum.
        for attempt in self.ledger.open_attempts():
            if attempt.attempt_id in self.ledger.inherited_attempt_ids:
                await self._resolve_inherited(attempt.attempt_id)
        # (b) leases held by other owners: take over only when provably dead or stale.
        for row in self.store.leased_nodes():
            node = self.nodes.get(row.node_id)
            if row.lease_owner == self.owner or node is None:
                continue
            await self._maybe_take_over(node, row.lease_owner, row.lease_attempt_id, row.heartbeat_at)
        # (c) persisted responses never committed: finish them without sending anything.
        for attempt, result in self.store.persisted_unfinished():
            node = self.nodes.get(attempt.node_id)
            if node is None or node.terminal:
                continue
            await self._finish_ok(node, attempt, result)

    async def _resolve_inherited(self, attempt_id: str) -> None:
        raw_path = self.paths.raw_response(attempt_id)
        cost = None
        if raw_path.exists():
            relative = self.paths.relative(raw_path)
            result = result_from_raw(read_raw_response(self.paths, relative), relative_path=relative,
                                     sha256=sha256_file(raw_path))
            cost = confirmed_cost(result)
        if cost is not None:
            await self.ledger.settle(attempt_id, cost)
        else:
            await self.ledger.charge_max(attempt_id)

    async def _maybe_take_over(self, node: _Node, owner: str | None, attempt_id: str | None,
                               heartbeat_at: float | None) -> bool:
        stale = heartbeat_at is None or time.time() - heartbeat_at >= self.settings.lease_stale_seconds
        if not (stale or owner_is_dead(owner)):
            node.foreign_lease = owner or "unknown"
            return False
        if attempt_id is not None:
            if self.ledger.state_of(attempt_id) in ("reserved", "sent", "unresolved"):
                await self.ledger.charge_max(attempt_id)
            attempts = {a.attempt_id: a for a in self.store.attempts_for(node.node_id)}
            prior = attempts.get(attempt_id)
            if prior is not None and prior.state is not AttemptState.TERMINAL:
                self.store.record_attempt(replace(prior, state=AttemptState.TERMINAL, completed_at=_utc(),
                                                  error_detail=f"lease_taken_over:{'stale' if stale else 'dead'}"))
            self.store.take_over_lease(node.node_id, attempt_id)
        node.foreign_lease = None
        self._event("lease_taken_over", node.node_id)
        if node.sends >= self.settings.max_transport_attempts and not node.terminal:
            self.store.mark_exhausted(node.node_id, "attempts_exhausted:lease_taken_over")
            self._on_terminal(node, succeeded=False)
        elif node.status == "ready" or node.waiting_parents == 0:
            self._make_ready(node)
        return True

    # ---- main loop ------------------------------------------------------------------------------------------
    async def _main_loop(self) -> None:
        last_control = -math.inf
        while True:
            now = self.clock()
            if now - last_control >= CONTROL_POLL_SECONDS:
                self._poll_control()
                last_control = now
            if self.aimd.tick(now):
                self.http.set_limit(self.aimd.concurrency_limit)
                self._event("aimd_increase", f"{self.aimd.concurrency_limit}/{self.aimd.rps:.2f}")
            if self.stop_mode is not None:
                self._finish(STOP_SENTINEL if self.stop_mode == "stop" else STOP_DRAIN, None)
                return
            if self.circuit_reason is not None:
                self._finish(STOP_CIRCUIT_OPEN, self.circuit_reason)
                return
            if all(w.remaining == 0 for w in self.waves):
                self._finish(STOP_COMPLETED, None)
                return
            self._maybe_open_waves()
            self._admit(now)
            idle = self.pre_send == 0 and not self._live_tasks()
            if idle:
                verdict = self._idle_verdict(now)
                if verdict is not None:
                    self._finish(*verdict)
                    return
            timeout = self._next_wakeup(now)
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    def _finish(self, reason: str, detail: str | None) -> None:
        if reason not in STOP_REASONS:
            raise AssertionError(f"undeclared stop reason {reason!r}")
        self.stop_reason = reason
        self.stop_detail = detail

    def _idle_verdict(self, now: float) -> tuple[str, str | None] | None:
        """Nothing is running. Return a stop reason if nothing can ever run again without outside help."""
        if self.aimd.pause_until > now or any(n.not_before > now for n in self.ready.values()):
            return None                                   # backoff / Retry-After: time will unblock it
        if any(n.foreign_lease for n in self.nodes.values() if not n.terminal):
            return None                                   # the heartbeat loop re-checks foreign leases
        if self.unresolved and self.ledger.unresolved_exceeded:
            return None                                   # the reconcile loop will settle or charge them
        if self.budget_blocked_at is not None and self.ready:
            if any(u.generation_id for u in self.unresolved.values()):
                return None                               # a confirmed bill below the reservation may free budget
            return STOP_BUDGET_EXHAUSTED, self.ledger.last_refusal
        if self.ready:
            return None                                   # admissible next iteration (rate token)
        unopened = [w for w in self.waves if not w.opened and w.remaining]
        if unopened:
            if self.ledger.snapshot().exposure >= self.soft_ceiling:
                return STOP_BUDGET_EXHAUSTED, "soft_admission_ceiling"
            if all(w.optional for w in unopened) and not self._core_complete():
                return STOP_STALLED, "optional_block_waits_for_incomplete_core"
            return None
        parked = sum(1 for n in self.nodes.values() if n.status == "parked" and not n.terminal)
        return STOP_STALLED, f"no_runnable_nodes:parked={parked}"

    def _next_wakeup(self, now: float) -> float:
        # Completions, settlements and unlocks set ``wake``; timers only cover what time alone unblocks, so the
        # coordinator never spins while capacity is exhausted.
        candidates = [CONTROL_POLL_SECONDS]
        if self.ready:
            if self.admit_blocker == "token":
                candidates.append(self.aimd.seconds_until_token(now))
            future = [n.not_before - now for n in self.ready.values() if n.not_before > now]
            if future:
                candidates.append(min(future))
        if self.aimd.pause_until > now:
            candidates.append(self.aimd.pause_until - now)
        return max(0.001, min(candidates))

    def _poll_control(self) -> None:
        if self.paths.stop_sentinel.exists():
            self.stop_mode = "stop"
        elif self.paths.drain_sentinel.exists() and self.stop_mode is None:
            self.stop_mode = "drain"

    def _core_complete(self) -> bool:
        return all(w.remaining == 0 for w in self.waves if not w.optional)

    def _maybe_open_waves(self) -> None:
        open_incomplete = [w for w in self.waves if w.opened and w.remaining > 0]
        if len(open_incomplete) >= self.settings.max_active_waves:
            return
        capacity = self._capacity()
        runnable = sum(1 for n in self.ready.values() if not n.foreign_lease)
        if open_incomplete and runnable >= capacity:
            return
        snap = self.ledger.snapshot()
        if snap.exposure >= self.soft_ceiling:
            return
        for wave in self.waves:
            if wave.opened:
                continue
            if wave.optional and self.budget.optional_requires_core_completion and not self._core_complete():
                return
            wave.opened = True
            self._event("wave_opened", str(wave.index))
            for node in wave.nodes:
                if not node.terminal and node.waiting_parents == 0 and node.status == "blocked":
                    self._make_ready(node)
            return

    def _capacity(self) -> int:
        limit = self.aimd.concurrency_limit
        prefetch = max(2, limit // 4)
        return limit + prefetch - self.pre_send - self.http.active

    def _make_ready(self, node: _Node) -> None:
        if node.terminal or node.status in ("active", "parked"):
            return
        if not self.waves[node.wave].opened:
            node.status = "blocked"
            return
        node.status = "ready"
        self.ready[node.node_id] = node

    def _pick(self, now: float) -> _Node | None:
        best: _Node | None = None
        best_key: tuple | None = None
        cap = self.settings.max_calls_per_item_in_flight
        for node in self.ready.values():
            if node.not_before > now or node.foreign_lease or self.item_active[node.item_key] >= cap:
                continue
            key = (node.wave, node.crit, self.bucket_served[node.bucket], node.seq)
            if best_key is None or key < best_key:
                best, best_key = node, key
        return best

    def _admit(self, now: float) -> None:
        self.admit_blocker = self._admit_inner(now)

    def _admit_inner(self, now: float) -> str:
        if self.stop_mode or self.circuit_reason:
            return "closed"
        if not self.ready:
            return "empty"
        if self.aimd.pause_until > now:
            return "retry_after"
        if self.ledger.unresolved_exceeded:
            self._event_once("paused_unresolved_exposure")
            return "unresolved_exposure"
        snap = self.ledger.snapshot()
        if self.budget_blocked_at is not None:
            if snap.exposure >= self.budget_blocked_at:
                return "budget"
            self.budget_blocked_at = None
        while self.ready:
            if self._capacity() <= 0:
                self.aimd.bound_in_window = True
                return "capacity"
            if not self.aimd.token_available(now):
                self.aimd.bound_in_window = True
                return "token"
            node = self._pick(now)
            if node is None:
                return "none_eligible"
            try:
                parents = [(self.nodes[p].spec, self.store.candidate(p) if self.nodes[p].succeeded else None)
                           for p in node.parents]
                messages = self.render(node.spec, parents)
                amount = reservation_usd(node.spec, rendered_chars(messages), self.lock,
                                         multiplier=self.budget.reservation_multiplier)
            except Exception as exc:  # noqa: BLE001 - a render/estimate defect must stop spending, loudly
                if _is_context_limit(exc):
                    # R03-1: terminal without an API call, a reservation or a candidate (the lead's render wrapper
                    # logs the event); children render the absent-response marker.
                    self.store.mark_terminal_without_candidate(node.node_id, OUTCOME_CONTEXT_LIMIT,
                                                               f"context_limit:{exc}"[:500])
                    self.context_limited += 1
                    self._event("context_limit", node.node_id)
                    self._on_terminal(node, succeeded=False)
                    continue
                self._open_circuit(f"render_failed:{type(exc).__name__}:{exc}")
                return "closed"
            first_attempt = node.sends == 0
            if not self.ledger.fits(amount + self.pending_reservation,
                                    ceiling_usd=self.soft_ceiling if first_attempt else None):
                self.budget_blocked_at = snap.exposure if snap.exposure > 0 else Decimal("0.000000001")
                self._event_once(f"budget_blocked:{self.ledger.last_refusal or 'ceiling'}")
                return "budget"
            self.aimd.take_token()
            self.bucket_served[node.bucket] += 1
            del self.ready[node.node_id]
            node.status = "active"
            node.attempts_used += 1
            self.item_active[node.item_key] += 1
            self.pre_send += 1
            self.pending_reservation += amount
            task = self._loop.create_task(self._attempt(node, node.attempts_used, messages, amount))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            snap = self.ledger.snapshot()
        return "empty"

    # ---- one attempt ----------------------------------------------------------------------------------------
    async def _attempt(self, node: _Node, attempt_no: int, messages: list[dict[str, str]], amount: Decimal) -> None:
        spec = node.spec
        attempt_id = f"{node.node_id}:{attempt_no}"
        attempt = Attempt(attempt_id, node.node_id, attempt_no, AttemptState.PENDING, amount, _utc())
        phase = {"pre_send": True}
        try:
            try:
                reserved = await self.ledger.reserve(attempt_id=attempt_id, node_id=node.node_id,
                                                     partition_id=spec.partition_id, amount_usd=amount)
            finally:
                self.pending_reservation -= amount
            if not reserved:
                node.attempts_used -= 1          # a refused reservation is not journaled: the id is still free
                snap = self.ledger.snapshot()
                self.budget_blocked_at = snap.exposure if snap.exposure > 0 else Decimal("0.000000001")
                self._requeue(node)
                return
            if not self.store.try_lease(node.node_id, attempt_id, owner=self.owner):
                await self.ledger.release_unsent(attempt_id)
                row = self.store.node(node.node_id)
                if row is not None and row.state == AttemptState.TERMINAL.value:
                    self._on_terminal(node, succeeded=row.outcome == OUTCOME_SUCCEEDED)
                    return
                node.foreign_lease = (row.lease_owner if row is not None and row.lease_owner else
                                      f"state:{None if row is None else row.state}")
                self._event("lease_refused", node.node_id)
                self._requeue(node)
                return
            attempt = replace(attempt, state=AttemptState.LEASED)
            self.store.record_attempt(attempt)
            outcome, error, sent_at, attempt = await self._exchange(node, messages, attempt, phase)
            if error == NOT_SENT:
                # Stop/drain/circuit arrived while this attempt waited for the HTTP limiter: it never left.
                await self.ledger.release_unsent(attempt_id)
                self.store.record_unsent_attempt(attempt, detail="not_sent:stopping")
                self._requeue(node, park=True)
                return
            await self._handle(node, attempt, outcome, error, sent_at)
        except Exception as exc:  # noqa: BLE001 - harness defect: stop spending, keep every reservation counted
            self._open_circuit(f"attempt_failed:{type(exc).__name__}:{exc}")
            state = self.ledger.state_of(attempt_id)
            try:
                if state == "reserved":
                    await self.ledger.release_unsent(attempt_id)
                elif state == "sent":
                    await self.ledger.mark_unresolved(attempt_id, detail="harness_exception")
                    self.unresolved.setdefault(attempt_id, _Unresolved(
                        attempt_id, node.node_id, None, self.clock() + self.settings.lease_stale_seconds))
            except Exception:  # noqa: BLE001
                pass
            node.status = "parked"
        finally:
            if phase["pre_send"]:
                self.pre_send -= 1
            self.item_active[node.item_key] -= 1
            self.wake.set()

    async def _exchange(self, node: _Node, messages: list[dict[str, str]], attempt: Attempt,
                        phase: dict[str, bool]) -> tuple[GenerationOutcome | None, str | None, float, Attempt]:
        async with self.http:
            phase["pre_send"] = False
            self.pre_send -= 1
            if self.stop_mode is not None or self.circuit_reason is not None:
                return None, NOT_SENT, self.clock(), attempt
            await self.ledger.mark_sent(attempt.attempt_id)       # urgent fsync: SENT is durable before sending
            node.sends += 1
            attempt = replace(attempt, state=AttemptState.SENT, sent_at=_utc())
            self.store.record_attempt(attempt)
            sent_at = self.clock()
            self.requests_sent += 1
            self.meter.change(+1)
            generation = self._loop.create_task(self.client.generate(
                node.spec, messages, attempt_id=attempt.attempt_id, lock=self.lock, sampling=self.sampling))
            try:
                done, _ = await asyncio.wait({generation, self.hard_stop}, timeout=self.settings.watchdog_seconds,
                                             return_when=asyncio.FIRST_COMPLETED)
                if generation in done:
                    try:
                        return generation.result(), None, sent_at, attempt
                    except Exception as exc:  # noqa: BLE001 - the client must classify; an escape is ambiguous
                        return None, f"client_exception:{type(exc).__name__}", sent_at, attempt
                generation.cancel()
                await asyncio.wait({generation}, timeout=5.0)
                return None, ("cancelled_by_stop" if self.hard_stop.done() else "watchdog_timeout"), sent_at, attempt
            finally:
                self.meter.change(-1)

    async def _handle(self, node: _Node, attempt: Attempt, outcome: GenerationOutcome | None, error: str | None,
                      sent_at: float) -> None:
        attempt_id = attempt.attempt_id
        now = self.clock()
        if outcome is None:
            self.attempt_statuses[TransportStatus.AMBIGUOUS_TIMEOUT.value] += 1
            await self.ledger.mark_unresolved(attempt_id, detail=error or "no_outcome")
            self.unresolved[attempt_id] = _Unresolved(attempt_id, node.node_id, None,
                                                      now + self.settings.lease_stale_seconds)
            self.store.record_attempt(replace(attempt, state=AttemptState.TERMINAL, completed_at=_utc(),
                                              transport_status=TransportStatus.AMBIGUOUS_TIMEOUT, error_detail=error))
            self._retry_or_exhaust(node, TransportStatus.AMBIGUOUS_TIMEOUT, None)
            return
        result = outcome.result
        if result.node_id != node.node_id or result.attempt_id != attempt_id:
            await self.ledger.mark_unresolved(attempt_id, detail="client_identity_mismatch")
            self.unresolved[attempt_id] = _Unresolved(attempt_id, node.node_id, None,
                                                      now + self.settings.lease_stale_seconds)
            self._open_circuit(f"client_identity_mismatch:{result.attempt_id}")
            node.status = "parked"
            return
        status = result.transport_status
        self.attempt_statuses[status.value] += 1
        raw_error: BaseException | None = None
        try:
            relative, digest = await asyncio.to_thread(write_raw_response, self.paths, attempt_id, outcome)
            result = replace(result, raw_response_path=relative, raw_response_sha256=digest)
        except Exception as exc:  # noqa: BLE001
            raw_error = exc
        cost = confirmed_cost(result)                   # CONNECT_FAILED without usage: provably unsent, 0
        if cost is not None:
            await self.ledger.settle(attempt_id, cost)
        else:
            await self.ledger.mark_unresolved(attempt_id, detail=status.value)
            self.unresolved[attempt_id] = _Unresolved(attempt_id, node.node_id, result.generation_id,
                                                      now + self.settings.lease_stale_seconds)
        finished = replace(attempt, completed_at=_utc(), transport_status=status, http_status=result.http_status,
                           generation_id=result.generation_id, settled_usd=cost,
                           retry_after_seconds=result.retry_after_seconds, error_detail=result.error_detail)
        if raw_error is not None:
            self.store.record_attempt(replace(finished, state=AttemptState.TERMINAL,
                                              error_detail=f"raw_write_failed:{type(raw_error).__name__}"))
            self._open_circuit(f"disk_failure:{type(raw_error).__name__}:{raw_error}")
            node.status = "parked"
            return
        if status is TransportStatus.OK:
            await self._finish_ok(node, finished, result)
            return
        self.store.record_attempt(replace(finished, state=AttemptState.TERMINAL))
        if status in CIRCUIT_BREAKING_TRANSPORT:
            self._open_circuit(f"{status.value}:{node.node_id}")
            node.status = "parked"
            return
        if status in RETRYABLE_TRANSPORT:
            if status is TransportStatus.RATE_LIMITED:
                self.rate_limited += 1
                if self.aimd.on_rate_limited(now, sent_at, result.retry_after_seconds):
                    self.http.set_limit(self.aimd.concurrency_limit)
                    self._event("aimd_decrease", f"{self.aimd.concurrency_limit}/{self.aimd.rps:.2f}")
            self._retry_or_exhaust(node, status, result.retry_after_seconds)
            return
        self.store.mark_exhausted(node.node_id, f"non_retryable:{status.value}")
        self._on_terminal(node, succeeded=False)

    async def _finish_ok(self, node: _Node, attempt: Attempt, result: GenerationResult) -> None:
        if node.terminal:
            self.duplicate_notifications += 1
            return
        self.store.record_persisted_response(attempt, result)
        try:
            candidate = await self._loop.run_in_executor(self.cpu, self.extract, result, node.spec)
        except Exception as exc:  # noqa: BLE001 - response is persisted; a fixed extractor finishes it on restart
            self._open_circuit(f"extract_failed:{type(exc).__name__}:{exc}")
            node.status = "parked"
            return
        try:
            self.store.record_result(result, candidate)
        except Exception as exc:  # noqa: BLE001
            self._open_circuit(f"commit_failed:{type(exc).__name__}:{exc}")
            node.status = "parked"
            return
        self._on_terminal(node, succeeded=True)

    def _retry_or_exhaust(self, node: _Node, status: TransportStatus, retry_after: float | None) -> None:
        if node.sends >= self.settings.max_transport_attempts:
            self.store.mark_exhausted(node.node_id, f"attempts_exhausted:{status.value}")
            self._on_terminal(node, succeeded=False)
            return
        exponent = max(0, node.sends - 1)
        delay = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2 ** exponent)) * self.rng.uniform(0.5, 1.5)
        if retry_after is not None and retry_after > 0:
            delay = max(delay, retry_after * self.rng.uniform(1.0, 1.25))
        node.not_before = self.clock() + delay
        self.retries_scheduled += 1
        self._requeue(node)

    def _requeue(self, node: _Node, *, park: bool = False) -> None:
        if node.terminal:
            return
        node.status = "blocked"
        if park or self.stop_mode is not None or self.circuit_reason is not None:
            node.status = "parked"
            return
        self._make_ready(node)

    def _on_terminal(self, node: _Node, *, succeeded: bool) -> None:
        if node.terminal:
            self.duplicate_notifications += 1
            return
        node.terminal = True
        node.succeeded = succeeded
        node.status = "done"
        self.ready.pop(node.node_id, None)
        self.waves[node.wave].remaining -= 1
        for child in node.children:
            child.waiting_parents -= 1
            if child.waiting_parents == 0 and not child.terminal and child.status == "blocked":
                self._make_ready(child)
        self.wake.set()

    def _open_circuit(self, reason: str) -> None:
        if self.circuit_reason is None:
            self.circuit_reason = reason
            self._event("circuit_open", reason)
        self.wake.set()

    def _event(self, kind: str, detail: str) -> None:
        self.events.append((time.time(), kind, detail))

    def _event_once(self, kind: str) -> None:
        if not self.events or self.events[-1][1] != kind:
            self._event(kind, "")

    # ---- drain / background -------------------------------------------------------------------------------
    async def _drain(self) -> None:
        if self.stop_mode == "stop":
            deadline = self.clock() + STOP_GRACE_SECONDS
            while self._live_tasks() and self.clock() < deadline:
                await asyncio.wait(self._live_tasks(), timeout=max(0.01, deadline - self.clock()))
            if not self.hard_stop.done():
                self.hard_stop.set_result(True)
        while self._live_tasks():
            await asyncio.wait(self._live_tasks())
        await self._reconcile_unresolved(final=True)
        await self.ledger.journal.flush()

    def _live_tasks(self) -> set[asyncio.Task]:
        return {t for t in self.tasks if not t.done()}

    async def _reconcile_unresolved(self, *, final: bool = False) -> None:
        now = self.clock()
        for item in list(self.unresolved.values()):
            if self.ledger.state_of(item.attempt_id) != "unresolved":
                self.unresolved.pop(item.attempt_id, None)
                continue
            cost: Decimal | None = None
            if item.generation_id and hasattr(self.client, "generation_metadata"):
                try:
                    meta = await self.client.generation_metadata(item.generation_id)
                except Exception:  # noqa: BLE001 - metadata lag or outage: try again later
                    meta = None
                if meta:
                    raw_cost = meta.get("total_cost", meta.get("cost"))
                    if raw_cost is not None:
                        cost = Decimal(str(raw_cost))
            if cost is not None:
                await self.ledger.settle(item.attempt_id, cost)
                self.unresolved.pop(item.attempt_id, None)
            elif final or now >= item.deadline:
                await self.ledger.charge_max(item.attempt_id)
                self.unresolved.pop(item.attempt_id, None)
        self.wake.set()

    async def _reconcile_loop(self) -> None:
        interval = min(RECONCILE_POLL_SECONDS, max(0.05, self.settings.lease_stale_seconds / 4))
        while True:
            await asyncio.sleep(interval)
            await self._reconcile_unresolved()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.lease_heartbeat_seconds)
            self.store.heartbeat(self.owner)
            self.store.set_meta("dispatcher_heartbeat", repr(time.time()))
            for row in self.store.leased_nodes():
                node = self.nodes.get(row.node_id)
                if node is not None and node.foreign_lease and row.lease_owner != self.owner:
                    await self._maybe_take_over(node, row.lease_owner, row.lease_attempt_id, row.heartbeat_at)
            self.wake.set()

    async def _backup_loop(self) -> None:
        while True:
            await asyncio.sleep(BACKUP_INTERVAL_SECONDS)
            try:
                await asyncio.to_thread(self.store.backup)
            except Exception as exc:  # noqa: BLE001
                self._open_circuit(f"backup_failed:{type(exc).__name__}:{exc}")

    async def _status_loop(self) -> None:
        while True:
            await self._write_status()
            await asyncio.sleep(STATUS_INTERVAL_SECONDS)

    async def _write_status(self) -> None:
        from agent_allocation.runtime.status import write_status_json

        payload = self.status_payload()
        try:
            await asyncio.to_thread(write_status_json, self.paths, payload)
        except Exception:  # noqa: BLE001 - status is advisory; never stop spending accounting for it
            pass

    async def _final_persist(self) -> None:
        try:
            await self.ledger.journal.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.to_thread(self.store.backup)
        except Exception as exc:  # noqa: BLE001
            self._event("backup_failed", repr(exc))
        await self._write_status()

    # ---- reporting ------------------------------------------------------------------------------------------
    def status_payload(self) -> dict[str, Any]:
        snap = self.ledger.snapshot()
        overall, active = self.meter.means()
        return {
            "written_at": _utc(),
            "owner": self.owner,
            "state": "running" if self.stop_reason is None else "finished",
            "stop_reason": self.stop_reason,
            "stop_mode": self.stop_mode,
            "spend": {"settled_usd": str(snap.settled), "in_flight_usd": str(snap.in_flight),
                      "unresolved_usd": str(snap.unresolved), "ceiling_usd": str(snap.ceiling),
                      "soft_ceiling_usd": str(self.soft_ceiling), "exposure_usd": str(snap.exposure),
                      "max_unresolved_usd": str(self.ledger.max_unresolved_usd)},
            "counts": self.store.counts(),
            "attempts_by_status": dict(self.attempt_statuses),
            "concurrency": {"in_flight_now": self.meter.value, "peak_in_flight": self.meter.peak,
                            "mean_in_flight": overall, "mean_in_flight_active": active,
                            "configured_start": self.settings.concurrency_start,
                            "configured_now": self.aimd.concurrency_limit,
                            "configured_max": self.settings.concurrency_max,
                            "rps_now": self.aimd.rps, "rps_max": self.settings.rps_max,
                            "increases": self.aimd.increases, "decreases": self.aimd.decreases},
            "requests_sent": self.requests_sent,
            "rate_limited_429": self.rate_limited,
            "retries_scheduled": self.retries_scheduled,
            "circuit": {"open": self.circuit_reason is not None, "reason": self.circuit_reason},
            "waves": {"total": len(self.waves), "opened": sum(w.opened for w in self.waves),
                      "complete": sum(1 for w in self.waves if w.remaining == 0),
                      "active": sum(1 for w in self.waves if w.opened and w.remaining > 0)},
            "ready": len(self.ready),
            "unresolved_attempts": len(self.unresolved),
            "duplicate_notifications": self.duplicate_notifications,
        }

    def _summary(self, started: float, settled_before: Decimal) -> RunSummary:
        snap = self.ledger.snapshot()
        overall, active = self.meter.means()
        succeeded = sum(1 for n in self.nodes.values() if n.terminal and n.succeeded)
        failed = sum(1 for n in self.nodes.values() if n.terminal and not n.succeeded)
        return RunSummary(
            stop_reason=self.stop_reason or STOP_STALLED,
            stop_detail=self.stop_detail,
            nodes_context_limited=self.context_limited,
            nodes_planned=len(self.nodes),
            nodes_succeeded=succeeded,
            nodes_failed=failed,
            nodes_pending=len(self.nodes) - succeeded - failed,
            requests_sent=self.requests_sent,
            attempts_by_status=dict(self.attempt_statuses),
            rate_limited=self.rate_limited,
            retries_scheduled=self.retries_scheduled,
            settled_usd=snap.settled,
            in_flight_usd=snap.in_flight,
            unresolved_usd=snap.unresolved,
            ceiling_usd=snap.ceiling,
            soft_ceiling_usd=self.soft_ceiling,
            spent_this_run_usd=snap.settled - settled_before,
            peak_in_flight=self.meter.peak,
            mean_in_flight=overall,
            mean_in_flight_active=active,
            concurrency_start=self.settings.concurrency_start,
            concurrency_final=self.aimd.concurrency_limit,
            concurrency_max_reached=self.aimd.max_concurrency_reached,
            rps_final=self.aimd.rps,
            wall_seconds=self.clock() - started,
            waves_total=len(self.waves),
            waves_opened=sum(w.opened for w in self.waves),
            waves_completed=sum(1 for w in self.waves if w.remaining == 0),
            circuit_reason=self.circuit_reason,
            duplicate_notifications=self.duplicate_notifications,
            extra={"events": list(self.events[-200:]), "aimd_increases": self.aimd.increases,
                   "aimd_decreases": self.aimd.decreases, "ledger_overruns": len(self.ledger.overruns),
                   "journal_batches": self.ledger.journal.batches_written,
                   "journal_max_batch": self.ledger.journal.max_batch},
        )
