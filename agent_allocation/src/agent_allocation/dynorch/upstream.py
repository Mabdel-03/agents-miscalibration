"""WP-A, contract §5 / plan §5: ``Upstream`` -- one capped raw ``/completions`` exchange with the $ ledger around it.

``Upstream.complete(shard_id, prompt, max_tokens, attempt_id, coordinates)``:

1. body = ``api/request.build_capped_body`` (provider pinned from the shard's sealed ``EndpointLock``, sampling from
   the config, ``usage.include``, no ``logit_bias``); $ reservation = ``(input_token_bound(chars) * input price +
   max_tokens * completion price) * reservation_multiplier`` via ``CollectionLedger.reserve`` (attempt ids
   ``f"{run_key}:g{gen}:{call_no}:{phase}:{n}"``; a refusal raises ``ProxyError(ledger_refused)`` -- nothing left the
   process, so the caller releases its token grant);
2. per-shard in-flight limiter (``runtime/scheduler.HttpLimiter``, plan: Parasail 40 / Mancer 48) and rps limiter
   (8 / 12) around ``mark_sent`` -> ``ForcedClient.raw_completion`` (the coordinates travel only as the httpx
   extension so ``api/fake.py`` can key scripted replies on ``phase`` / ``call_no``);
3. classification with the existing ``api/normalize`` rules: transport-OK with usage -> ``settle(usage.cost)``;
   429 / 5xx (not 504/524) / connect failures -> settle at the confirmed 0 and retry (``max_transport_attempts``
   3, ``ForcedClient``'s backoff, ``Retry-After`` honoured); a response-cache HIT -> ``charge_max`` and retry once
   (B9); ambiguous outcomes (timeouts, gateway 504/524, error envelopes, a 200 whose usage is missing, a 5xx
   carrying a generation id) -> ``charge_max`` and ``UpstreamFailed`` (``upstream_failed``, non-retryable for the
   harness; the caller settles the full token grant); auth / payment / bad-request refusals -> settle 0 and
   ``UpstreamFailed`` (an upstream 402 is an infrastructure stop, B6, never a harness exhaustion).

Every exchange is returned as a journal ``phase`` dict (headers/text redacted through ``apikey``), attached to the
``UpstreamResult`` or the ``UpstreamFailed``.
"""
from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from typing import Any, Callable, Mapping

from agent_allocation.api.forced import (
    ANSWER_BACKOFF_BASE_SECONDS, ANSWER_BACKOFF_CAP_SECONDS, ANSWER_RETRY_AFTER_CAP_SECONDS, ForcedCall,
    ForcedClient,
)
from agent_allocation.api.normalize import split_reasoning
from agent_allocation.api.request import build_capped_body
from agent_allocation.apikey import redact, redact_headers
from agent_allocation.contracts import EndpointLock, TransportStatus
from agent_allocation.dynorch.records import ProxyError, UpstreamResult
from agent_allocation.runtime.estimate import input_token_bound
from agent_allocation.runtime.ledger import GATEWAY_TIMEOUT_MAY_BE_BILLED, CollectionLedger
from agent_allocation.runtime.scheduler import HttpLimiter

MAX_TRANSPORT_ATTEMPTS = 3
RETRY_STATUSES = frozenset({TransportStatus.RATE_LIMITED, TransportStatus.UPSTREAM_ERROR,
                            TransportStatus.CONNECT_FAILED})
REFUSAL_STATUSES = frozenset({TransportStatus.AUTH_FAILED, TransportStatus.PAYMENT_REQUIRED,
                              TransportStatus.BAD_REQUEST})


class UpstreamFailed(ProxyError):
    """``upstream_failed`` with what the ledger did (for the journal and the run record)."""

    def __init__(self, message: str, *, transport_status: TransportStatus, phases: tuple[dict[str, Any], ...],
                 cost_usd: Decimal, charged_max: bool, infrastructure_stop: bool = False) -> None:
        super().__init__("upstream_failed", message, detail=transport_status.value)
        self.transport_status = transport_status
        self.phases = phases
        self.cost_usd = cost_usd
        self.charged_max = charged_max
        self.infrastructure_stop = infrastructure_stop


class RateLimiter:
    """A per-shard requests-per-second gate (one send every ``1/rps`` seconds, no bursts)."""

    def __init__(self, rps: int, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] | None = None) -> None:
        if isinstance(rps, bool) or not isinstance(rps, int) or rps < 1:
            raise ValueError(f"rps must be a positive integer, got {rps!r}")
        self.interval = 1.0 / rps
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - now
        if delay > 0:
            await self._sleep(delay)


def sibling_attempt_id(attempt_id: str, number: int) -> str:
    """``...:<phase>:1`` -> ``...:<phase>:<number>`` (transport retries of one call keep the ledger's
    ``'<node>:<attempt_no>'`` shape, see ``runtime/ledger.attempt_number``)."""
    head, _, tail = attempt_id.rpartition(":")
    if not head or not tail.isdigit():
        raise ValueError(f"attempt id {attempt_id!r} must end in ':<attempt_no>'")
    return f"{head}:{number}"


def _body_text(call: ForcedCall) -> str:
    payload = call.body_json
    if payload is None:
        return ""
    if isinstance(payload, str):
        return redact(payload) or ""
    return redact(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) or ""


def phase_record(phase: str, attempt_id: str, body: Mapping[str, Any], call: ForcedCall, *, cost_usd: Decimal | None,
                 charged_max: bool) -> dict[str, Any]:
    """One journal ``phases[]`` entry (contract §5), redacted."""
    return {
        "phase": phase,
        "attempt_id": attempt_id,
        "request_body": {k: (redact(v) if isinstance(v, str) else v) for k, v in body.items()},
        "response_status": call.http_status,
        "response_headers": redact_headers(dict(call.headers)),
        "response_body_text": _body_text(call),
        "latency_seconds": call.latency_seconds,
        "transport_status": call.transport_status.value,
        "generation_id": call.generation_id,
        "cost_usd": None if cost_usd is None else str(cost_usd),
        "charged_max": charged_max,
        "error_detail": None if call.error_detail is None else (redact(call.error_detail) or ""),
    }


class Upstream:
    def __init__(self, client: ForcedClient, locks: Mapping[str, EndpointLock], sampling: Mapping[str, Any],
                 in_flight: Mapping[str, int], rps: Mapping[str, int], ledger: CollectionLedger, *,
                 reservation_multiplier: Decimal = Decimal("1.1"), max_transport_attempts: int = MAX_TRANSPORT_ATTEMPTS,
                 backoff_base_seconds: float = ANSWER_BACKOFF_BASE_SECONDS,
                 backoff_cap_seconds: float = ANSWER_BACKOFF_CAP_SECONDS,
                 sleep: Callable[[float], Any] | None = None) -> None:
        if not locks:
            raise ValueError("Upstream needs at least one shard lock")
        missing = [s for s in locks if s not in in_flight or s not in rps]
        if missing:
            raise ValueError(f"shards without in_flight/rps limits: {missing}")
        multiplier = Decimal(str(reservation_multiplier))
        if not multiplier.is_finite() or multiplier < 1:
            raise ValueError(f"reservation_multiplier must be >= 1, got {reservation_multiplier!r}")
        if isinstance(max_transport_attempts, bool) or not isinstance(max_transport_attempts, int) \
                or max_transport_attempts < 1:
            raise ValueError("max_transport_attempts must be a positive integer")
        self.client = client
        self.locks = dict(locks)
        self.sampling = dict(sampling)
        self.ledger = ledger
        self.multiplier = multiplier
        self.max_transport_attempts = max_transport_attempts
        self._backoff_base = max(0.0, float(backoff_base_seconds))
        self._backoff_cap = max(0.0, float(backoff_cap_seconds))
        self._sleep = sleep or asyncio.sleep
        self._limiters = {shard: HttpLimiter(int(in_flight[shard])) for shard in locks}
        self._rps = {shard: RateLimiter(int(rps[shard]), sleep=self._sleep) for shard in locks}
        self.calls_sent = 0
        self.retries = 0

    # ---- reservation ----------------------------------------------------------------------------------------
    def reservation_usd(self, prompt: str, max_tokens: int, lock: EndpointLock) -> Decimal:
        """Conservative ex-ante $ for one capped call (``runtime/estimate.input_token_bound`` for the prompt)."""
        input_price = Decimal(lock.input_price_per_token)
        completion_price = max(Decimal(lock.output_price_per_token),
                               Decimal(lock.internal_reasoning_price_per_token or "0"))
        amount = (Decimal(input_token_bound(len(prompt))) * input_price
                  + Decimal(int(max_tokens)) * completion_price) * self.multiplier
        return amount if amount > 0 else Decimal("0.000001")

    def _delay(self, number: int, retry_after: float | None) -> float:
        delay = min(self._backoff_cap, self._backoff_base * (2 ** (number - 1)))
        if retry_after is not None and retry_after > 0:
            delay = max(delay, min(retry_after, ANSWER_RETRY_AFTER_CAP_SECONDS))
        return delay

    # ---- one call --------------------------------------------------------------------------------------------
    async def complete(self, *, shard_id: str, prompt: str, max_tokens: int, attempt_id: str,
                       coordinates: Mapping[str, Any]) -> UpstreamResult:
        """§5: reserve, send (retrying transport failures), settle. Raises ``ProxyError(ledger_refused)`` before
        anything is sent, ``UpstreamFailed`` after the $ side has been settled or charged conservatively."""
        if shard_id not in self.locks:
            raise ProxyError("bad_request", f"unknown shard {shard_id!r}")
        lock = self.locks[shard_id]
        body = build_capped_body(prompt, lock, self.sampling, max_tokens=max_tokens)
        amount = self.reservation_usd(prompt, max_tokens, lock)
        node_id = str(coordinates.get("run_key") or coordinates.get("node_id") or "")
        partition = str(coordinates.get("partition") or "main")
        phase = str(coordinates.get("phase") or "think")
        if not node_id:
            raise ValueError("coordinates must carry run_key")
        phases: list[dict[str, Any]] = []
        settled_total = Decimal("0")
        charged_max_any = False
        cache_retry_used = False
        number = 1
        while True:
            aid = attempt_id if number == 1 else sibling_attempt_id(attempt_id, number)
            if not await self.ledger.reserve(attempt_id=aid, node_id=node_id, partition_id=partition,
                                             amount_usd=amount):
                raise ProxyError("ledger_refused", f"ledger refused {amount} USD ({self.ledger.last_refusal})",
                                 detail=self.ledger.last_refusal)
            async with self._limiters[shard_id]:
                await self._rps[shard_id].wait()
                await self.ledger.mark_sent(aid)
                self.calls_sent += 1
                call = await self.client.raw_completion(
                    body, lock=lock, coordinates={**dict(coordinates), "attempt_id": aid, "transport_attempt": number})
            status = call.transport_status
            usage = call.usage
            tokens_known = usage is not None and usage.completion_tokens is not None
            # ---- $ settlement (the ledger's confirmed-cost rules, conservative otherwise) ----
            if status is TransportStatus.OK and tokens_known and call.cost_usd is not None:
                await self.ledger.settle(aid, call.cost_usd)
                cost, charged = call.cost_usd, False
            elif status in RETRY_STATUSES and call.cost_usd is not None:
                await self.ledger.settle(aid, call.cost_usd)      # provably 0: refused before any model ran
                cost, charged = call.cost_usd, False
            elif status in REFUSAL_STATUSES:
                await self.ledger.settle(aid, call.cost_usd if call.cost_usd is not None else Decimal("0"))
                cost, charged = (call.cost_usd if call.cost_usd is not None else Decimal("0")), False
            elif status is TransportStatus.PROVIDER_MISMATCH and call.cost_usd is not None:
                await self.ledger.settle(aid, call.cost_usd)
                cost, charged = call.cost_usd, False
            else:
                await self.ledger.charge_max(aid)                  # unknown money is never released (B2)
                cost, charged = amount, True
            settled_total += cost
            charged_max_any = charged_max_any or charged
            phases.append(phase_record(phase, aid, body, call, cost_usd=cost, charged_max=charged))

            if status is TransportStatus.OK and tokens_known:
                reasoning, text, closed = split_reasoning(call.reasoning, call.text)
                return UpstreamResult(
                    reasoning=reasoning, text=text, closed=closed, finish_reason=call.finish_reason, usage=usage,
                    generation_id=call.generation_id, cache_status=call.cache_status,
                    latency_seconds=sum(p["latency_seconds"] for p in phases), transport_status=status,
                    http_status=call.http_status, raw=call.body_json if isinstance(call.body_json, dict) else {},
                    cost_usd=str(settled_total), attempts=tuple(p["attempt_id"] for p in phases),
                    phases=tuple(phases), request_body=dict(body))
            # ---- retry classes ----
            retryable = status in RETRY_STATUSES and call.cost_usd is not None
            if status is TransportStatus.CACHE_HIT and not cache_retry_used:
                retryable, cache_retry_used = True, True           # B9: a cache HIT is retried once
            if retryable and number < self.max_transport_attempts:
                self.retries += 1
                await self._sleep(self._delay(number, call.retry_after_seconds))
                number += 1
                continue
            detail = call.error_detail or status.value
            if status is TransportStatus.OK:
                detail = "ok_without_usage: the reply carries no completion_tokens; cannot meter"
            raise UpstreamFailed(f"upstream call failed ({status.value}): {redact(detail)}",
                                 transport_status=status, phases=tuple(phases), cost_usd=settled_total,
                                 charged_max=charged_max_any,
                                 infrastructure_stop=status in (TransportStatus.PAYMENT_REQUIRED,
                                                                TransportStatus.AUTH_FAILED))

    @staticmethod
    def is_gateway_timeout(call: ForcedCall) -> bool:
        return (call.error_detail or "").startswith(GATEWAY_TIMEOUT_MAY_BE_BILLED)
