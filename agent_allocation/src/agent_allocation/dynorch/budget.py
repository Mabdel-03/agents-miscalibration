"""WP-A, contract §5 / plan B2-B6: ``TokenBudget`` -- the per-run completion-token cap with a FIFO lock.

One ``TokenBudget`` per run. ``acquire()`` takes the run's ``asyncio.Lock`` (FIFO: waiters are served in arrival
order) and mints a ``Grant`` of ``max_tokens = cap - spent`` (E0 capped arms: ``min(cap - spent, root_ceiling)``
for root calls); the lock is held until the grant is settled or released, so exactly one upstream call per run is
in flight (frozen table "per-run serialisation") and the grants of concurrent sub-agent calls can never add up
beyond ``cap``. ``queue_wait_seconds`` is recorded on the grant.

A grant that was marked sent is never released: ``release_unsent`` raises for it, and ``settle`` charges what the
usage reports (``settle_full`` charges the whole grant for an ambiguous outcome). ``record_grace`` books the one
root-only grace (``max_grace_events_per_run = 1``, outside ``cap``); a second call raises. ``guard`` answers
``max_calls`` / ``max_prompt_tokens`` (B6). ``exhausted_for(root)`` is B4's boundary rule: ``remaining <
terminal_threshold``.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


class BudgetError(RuntimeError):
    """A programming error against the budget protocol (never a harness-visible condition)."""


@dataclass
class Grant:
    call_no: int
    max_tokens: int
    spent_before: int
    queue_wait_seconds: float
    root: bool
    ceiling_bound: bool = False              # E0: the root ceiling, not the cap, limited this grant
    sent: bool = False
    settled: bool = False
    realized_completion: int | None = None
    realized_prompt: int = 0
    _token: object = field(default_factory=object, repr=False)


class TokenBudget:
    def __init__(self, cap: int, *, terminal_threshold: int, grace: int, max_calls: int,
                 max_prompt_tokens: int) -> None:
        for name, value in (("cap", cap), ("terminal_threshold", terminal_threshold), ("grace", grace),
                            ("max_calls", max_calls), ("max_prompt_tokens", max_prompt_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BudgetError(f"{name} must be a positive integer, got {value!r}")
        if terminal_threshold >= grace:
            raise BudgetError("terminal_threshold must be below the grace")
        self.cap = cap
        self.terminal_threshold = terminal_threshold
        self.grace = grace
        self.max_calls = max_calls
        self.max_prompt_tokens = max_prompt_tokens
        self.spent = 0
        self.prompt_total = 0
        self.calls = 0                       # grants minted (upstream calls admitted), duplicates excluded
        self.grace_tokens = 0
        self.grace_kind = "none"
        self.grace_used = False
        self.peak_granted = 0
        self.max_queue_wait_seconds = 0.0
        self._lock = asyncio.Lock()
        self._active: Grant | None = None

    # ---- views --------------------------------------------------------------------------------------------------
    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.spent)

    @property
    def closed(self) -> bool:
        """After the grace the run accepts no further call (B4)."""
        return self.grace_used

    def exhausted_for(self, root: bool) -> bool:
        """B4: ``remaining < terminal_threshold`` at a call boundary (the caller picks TERMINAL vs 402)."""
        return self.remaining < self.terminal_threshold

    def guard(self) -> str | None:
        """B6: ``call_limit`` when the next call would exceed ``max_calls``, ``input_limit`` when the prompt tokens
        already served reach ``max_prompt_tokens``; ``None`` otherwise."""
        if self.calls >= self.max_calls:
            return "call_limit"
        if self.prompt_total >= self.max_prompt_tokens:
            return "input_limit"
        return None

    @property
    def overdraft_tokens(self) -> int:
        """B5: ``completion_total + grace - C`` (never below zero)."""
        return max(0, self.spent + self.grace_tokens - self.cap)

    def view(self) -> dict[str, int]:
        return {"cap": self.cap, "spent": self.spent, "remaining": self.remaining, "calls": self.calls,
                "grace_tokens": self.grace_tokens, "prompt_total": self.prompt_total}

    # ---- the lock -----------------------------------------------------------------------------------------------
    async def lock(self) -> float:
        """Take the run's FIFO lock without minting a grant (the caller decides TERMINAL / refusal / duplicate under
        it); returns the wait in seconds. Release with ``unlock``."""
        started = time.monotonic()
        await self._lock.acquire()
        waited = time.monotonic() - started
        self.max_queue_wait_seconds = max(self.max_queue_wait_seconds, waited)
        return waited

    def unlock(self) -> None:
        if self._active is not None:
            raise BudgetError("unlock while a grant is active: settle or release it first")
        if self._lock.locked():
            self._lock.release()

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    # ---- grants -------------------------------------------------------------------------------------------------
    def grant(self, *, root: bool, queue_wait_seconds: float, root_ceiling: int | None = None) -> Grant:
        """Mint the grant under the held lock: ``max_tokens = cap - spent`` (root with an E0 ceiling:
        ``min(cap - spent, ceiling)``). Raises when the lock is not held, a grant is active, or nothing remains."""
        if not self._lock.locked():
            raise BudgetError("grant without the run lock")
        if self._active is not None:
            raise BudgetError("a grant is already active: one upstream call per run")
        remaining = self.remaining
        if remaining < 1:
            raise BudgetError("grant with nothing remaining (the caller must apply B4 first)")
        max_tokens = remaining
        ceiling_bound = False
        if root and root_ceiling is not None:
            if isinstance(root_ceiling, bool) or not isinstance(root_ceiling, int) or root_ceiling < 1:
                raise BudgetError(f"root_ceiling must be a positive integer, got {root_ceiling!r}")
            if root_ceiling < remaining:
                max_tokens, ceiling_bound = root_ceiling, True
        self.calls += 1
        grant = Grant(call_no=self.calls, max_tokens=max_tokens, spent_before=self.spent,
                      queue_wait_seconds=queue_wait_seconds, root=root, ceiling_bound=ceiling_bound)
        self.peak_granted = max(self.peak_granted, max_tokens)
        self._active = grant
        return grant

    async def acquire(self, *, root: bool = True, root_ceiling: int | None = None) -> Grant:
        """FIFO lock + grant in one step (contract §5 signature; the core uses ``lock``/``grant`` separately so it
        can apply B4 under the lock before minting)."""
        waited = await self.lock()
        try:
            return self.grant(root=root, queue_wait_seconds=waited, root_ceiling=root_ceiling)
        except BaseException:
            self._lock.release()
            raise

    def mark_sent(self, grant: Grant) -> None:
        self._check_active(grant)
        grant.sent = True

    def settle(self, grant: Grant, completion_tokens: int, prompt_tokens: int) -> None:
        """Charge the reported completion tokens (inside ``cap``) and release the lock. The grant is never
        'released' after ``mark_sent``: an unknown completion count must go through ``settle_full``."""
        self._check_active(grant)
        if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens < 0:
            raise BudgetError(f"completion_tokens must be a non-negative integer, got {completion_tokens!r}")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise BudgetError(f"prompt_tokens must be a non-negative integer, got {prompt_tokens!r}")
        self.spent += completion_tokens
        self.prompt_total += prompt_tokens
        grant.realized_completion = completion_tokens
        grant.realized_prompt = prompt_tokens
        grant.settled = True
        self._active = None
        self._lock.release()

    def settle_full(self, grant: Grant, prompt_tokens: int = 0) -> None:
        """B2: an ambiguous outcome settles the whole grant (conservative)."""
        self.settle(grant, grant.max_tokens, prompt_tokens)

    def release_unsent(self, grant: Grant) -> None:
        """Give a never-sent grant back (e.g. the $ ledger refused before anything left). Raises after
        ``mark_sent``: a sent grant is never released."""
        self._check_active(grant)
        if grant.sent:
            raise BudgetError(f"grant {grant.call_no} was sent: a sent grant is never released")
        self.calls -= 1
        grant.settled = True
        self._active = None
        self._lock.release()

    def _check_active(self, grant: Grant) -> None:
        if grant.settled:
            raise BudgetError(f"grant {grant.call_no} was already settled")
        if self._active is not grant:
            raise BudgetError(f"grant {grant.call_no} is not the active grant")

    # ---- grace --------------------------------------------------------------------------------------------------
    def record_grace(self, kind: str, tokens: int) -> None:
        """B5: the single grace event of the run (``mid_call`` / ``visible_continuation`` / ``terminal``), outside
        ``cap``, at most ``grace`` tokens. A second grace raises: the rule is enforced here, not by convention."""
        if self.grace_used:
            raise BudgetError("a second grace event is not allowed (max_grace_events_per_run = 1)")
        if kind not in ("mid_call", "visible_continuation", "terminal"):
            raise BudgetError(f"unknown grace kind {kind!r}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise BudgetError(f"grace tokens must be a non-negative integer, got {tokens!r}")
        self.grace_used = True
        self.grace_kind = kind
        self.grace_tokens = tokens
