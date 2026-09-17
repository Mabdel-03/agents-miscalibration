"""Pure OpenRouter response classification and usage normalization (no I/O, no clock except Retry-After dates).

Transport classification (``classify_http_status`` / ``classify_exception`` / ``parse_completion``):

* exceptions: connect errors / connect timeouts / pool timeouts / proxy errors (provably nothing written)
  -> CONNECT_FAILED; read/write timeouts, read/write errors, protocol errors, decoding errors and any other
  ``httpx.RequestError`` after the request may have been written -> AMBIGUOUS_TIMEOUT (money unknown).
* HTTP status: 429 -> RATE_LIMITED; 401/403 -> AUTH_FAILED; 402 -> PAYMENT_REQUIRED; 408 -> AMBIGUOUS_TIMEOUT;
  400/404/422 and every other 1xx/3xx/4xx -> BAD_REQUEST; 5xx -> UPSTREAM_ERROR; 2xx -> body inspection.
* 2xx body, in this order: ``x-openrouter-cache-status: HIT`` -> CACHE_HIT (even if the body looks fine);
  a non-null top-level ``error``, a choice-level ``error``, ``finish_reason: "error"``, or a body that is not a single
  well-formed completion (non-JSON, no/several choices, no message) -> ERROR_ENVELOPE (retryable in contract v2; an
  error body's own ``provider``/``model`` strings are only noted in ``error_detail`` -- OpenRouter's documented error
  bodies use slug forms, so they must not open the circuit); top-level ``provider`` present and
  != ``lock.provider_name`` or ``model`` present and != ``lock.model`` -> PROVIDER_MISMATCH; otherwise OK.
* 504 and 524 (gateway timeouts) are UPSTREAM_ERROR by rule but carry ``gateway_timeout_may_be_billed`` in
  ``error_detail``: the upstream may have finished and charged the generation.

Completion status (transport OK only; every value terminal): finish ``length`` or native ``MAX_TOKENS`` -> LENGTH;
finish ``content_filter`` or a blocking native reason (SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII, ...)
-> BLOCKED; empty/whitespace visible content -> EMPTY (BLOCKED when ``message.refusal`` is a non-empty string);
non-empty content with finish ``stop`` (or no finish and native ``STOP``) -> COMPLETED; anything else -> OTHER.

Usage: integers must be JSON integers >= 0, money must be a finite number >= 0 converted with ``Decimal(str(v))``;
anything absent or malformed is ``None`` (never 0). ``completion_tokens`` is kept as reported (it already includes
reasoning); ``reasoning_tokens`` is the reported subset and is never added to it.
"""
from __future__ import annotations

import copy
import email.utils
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import httpx

from agent_allocation.apikey import redact
from agent_allocation.contracts import CompletionStatus, EndpointLock, TransportStatus, Usage

MAX_ERROR_DETAIL_CHARS = 2000

BLOCKING_NATIVE_FINISH_REASONS = frozenset({
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION", "LANGUAGE", "FINISH_REASON_SAFETY", "FINISH_REASON_RECITATION",
    "FINISH_REASON_BLOCKLIST", "FINISH_REASON_PROHIBITED_CONTENT", "FINISH_REASON_SPII",
})
LENGTH_NATIVE_FINISH_REASONS = frozenset({"MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"})
STOP_NATIVE_FINISH_REASONS = frozenset({"STOP", "FINISH_REASON_STOP"})

GATEWAY_TIMEOUT_STATUSES = frozenset({504, 524})

_CONNECT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError, httpx.UnsupportedProtocol,
)


def classify_exception(exc: BaseException) -> TransportStatus:
    """CONNECT_FAILED only when httpx guarantees no request bytes left the process; everything else (including a raw
    ``ssl.SSLError``/``OSError`` that httpcore does not wrap after the handshake) is ambiguous."""
    if isinstance(exc, _CONNECT_EXCEPTIONS):
        return TransportStatus.CONNECT_FAILED
    return TransportStatus.AMBIGUOUS_TIMEOUT


def classify_http_status(status: int) -> TransportStatus | None:
    """Transport status for a non-2xx HTTP status; ``None`` means 2xx (inspect the body)."""
    if 200 <= status < 300:
        return None
    if status == 429:
        return TransportStatus.RATE_LIMITED
    if status in (401, 403):
        return TransportStatus.AUTH_FAILED
    if status == 402:
        return TransportStatus.PAYMENT_REQUIRED
    if status == 408:
        return TransportStatus.AMBIGUOUS_TIMEOUT
    if status >= 500:
        return TransportStatus.UPSTREAM_ERROR
    return TransportStatus.BAD_REQUEST


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Retry-After as seconds (delta-seconds or HTTP-date, clamped at 0); ``None`` when absent or unparseable."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = None
    if seconds is not None:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = email.utils.parsedate_to_datetime(text)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        return max(0.0, (when - reference).total_seconds())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def parse_money(value: Any) -> Decimal | None:
    """``Decimal(str(value))`` for a finite, non-negative JSON number or numeric string; otherwise ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        amount = Decimal(str(value))
    elif isinstance(value, Decimal):
        amount = value
    elif isinstance(value, str):
        try:
            amount = Decimal(value.strip())
        except InvalidOperation:
            return None
    else:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return amount


def normalize_usage(raw: Any) -> Usage | None:
    """``Usage`` from a raw ``usage`` object; ``None`` when the response carries no usage mapping at all."""
    if not isinstance(raw, Mapping):
        return None
    prompt_details = raw.get("prompt_tokens_details")
    completion_details = raw.get("completion_tokens_details")
    cost_details = raw.get("cost_details")
    prompt_details = prompt_details if isinstance(prompt_details, Mapping) else {}
    completion_details = completion_details if isinstance(completion_details, Mapping) else {}
    cost_details = cost_details if isinstance(cost_details, Mapping) else {}
    return Usage(
        prompt_tokens=_count(raw.get("prompt_tokens")),
        completion_tokens=_count(raw.get("completion_tokens")),
        reasoning_tokens=_count(completion_details.get("reasoning_tokens")),
        cached_prompt_tokens=_count(prompt_details.get("cached_tokens")),
        actual_cost_usd=parse_money(raw.get("cost")),
        raw_usage=copy.deepcopy(dict(raw)),
        upstream_inference_cost_usd=parse_money(cost_details.get("upstream_inference_cost")),
    )


def completion_status(content: str | None, finish_reason: str | None, native_finish_reason: str | None,
                      refusal: Any = None) -> CompletionStatus:
    native = (native_finish_reason or "").strip().upper()
    finish = (finish_reason or "").strip().lower()
    if finish == "length":
        return CompletionStatus.LENGTH
    if finish == "content_filter":
        return CompletionStatus.BLOCKED
    if native in BLOCKING_NATIVE_FINISH_REASONS:
        return CompletionStatus.BLOCKED
    if native in LENGTH_NATIVE_FINISH_REASONS:
        return CompletionStatus.LENGTH
    if content is None or not content.strip():
        if isinstance(refusal, str) and refusal.strip():
            return CompletionStatus.BLOCKED
        return CompletionStatus.EMPTY
    if finish == "stop" or (not finish and native in STOP_NATIVE_FINISH_REASONS):
        return CompletionStatus.COMPLETED
    return CompletionStatus.OTHER


def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # multi-part content: concatenate text parts in order
        parts = [p.get("text") for p in value if isinstance(p, Mapping) and isinstance(p.get("text"), str)]
        return "".join(parts) if parts else None
    return None


def _non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return len(value) > 0
    return False


def error_summary(payload: Any, *, prefix: str, limit: int = MAX_ERROR_DETAIL_CHARS) -> str:
    """A redacted, bounded one-line description of an error body."""
    if isinstance(payload, Mapping) and isinstance(payload.get("error"), Mapping):
        error = payload["error"]
        pieces = [prefix]
        if error.get("code") is not None:
            pieces.append(f"code={error.get('code')}")
        if error.get("message") is not None:
            pieces.append(f"message={error.get('message')}")
        metadata = error.get("metadata")
        if isinstance(metadata, Mapping):
            if metadata.get("provider_name") is not None:
                pieces.append(f"provider={metadata.get('provider_name')}")
            if metadata.get("raw") is not None:
                pieces.append(f"raw={metadata.get('raw')}")
        text = " ".join(str(p) for p in pieces)
    elif payload is None:
        text = prefix
    else:
        text = f"{prefix} body={json.dumps(payload, default=str)[:limit]}"
    return (redact(text) or "")[:limit]


@dataclass(frozen=True)
class ParsedCompletion:
    """Everything extracted from a 2xx body. ``transport_status`` is final; ``completion_status`` only for OK."""
    transport_status: TransportStatus
    completion_status: CompletionStatus | None
    generation_id: str | None
    observed_model: str | None
    observed_provider: str | None
    finish_reason: str | None
    native_finish_reason: str | None
    visible_content: str | None
    reasoning_returned: bool
    usage: Usage | None
    error_detail: str | None


def parse_completion(body: bytes, cache_status: str | None, lock: EndpointLock) -> ParsedCompletion:
    try:
        payload: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        payload = None
    parsed_ok = isinstance(payload, Mapping)
    data: Mapping[str, Any] = payload if parsed_ok else {}

    generation_id = data.get("id") if isinstance(data.get("id"), str) else None
    observed_model = data.get("model") if isinstance(data.get("model"), str) else None
    observed_provider = data.get("provider") if isinstance(data.get("provider"), str) else None
    choices = data.get("choices")
    choice: Mapping[str, Any] = {}
    message: Mapping[str, Any] = {}
    shape_problem: str | None = None
    if not parsed_ok:
        shape_problem = "malformed_body: response is not a JSON object"
    elif data.get("error") is None:
        if not isinstance(choices, list) or len(choices) != 1:
            count = len(choices) if isinstance(choices, list) else "no"
            shape_problem = f"malformed_body: expected exactly one choice, got {count}"
        elif not isinstance(choices[0], Mapping):
            shape_problem = "malformed_body: choice is not an object"
        else:
            choice = choices[0]
            if isinstance(choice.get("message"), Mapping):
                message = choice["message"]
            elif choice.get("error") is None:
                shape_problem = "malformed_body: choice has no message"
    finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
    native = choice.get("native_finish_reason")
    native_finish_reason = native if isinstance(native, str) else None
    visible_content = _text_content(message.get("content"))
    reasoning_returned = _non_empty(message.get("reasoning")) or _non_empty(message.get("reasoning_details"))
    usage = normalize_usage(data.get("usage"))

    status = TransportStatus.OK
    detail: str | None = None
    mismatches = []
    if observed_provider is not None and observed_provider != lock.provider_name:
        mismatches.append(f"provider {observed_provider!r} != locked {lock.provider_name!r}")
    if observed_model is not None and observed_model != lock.model:
        mismatches.append(f"model {observed_model!r} != locked {lock.model!r}")
    identity_note = "" if not mismatches else " [identity in error body: " + "; ".join(mismatches) + "]"
    if (cache_status or "").strip().upper() == "HIT":
        status = TransportStatus.CACHE_HIT
        detail = "x-openrouter-cache-status HIT on an independent draw"
    elif parsed_ok and data.get("error") is not None:
        status = TransportStatus.ERROR_ENVELOPE
        detail = error_summary(data, prefix="error_envelope") + identity_note
    elif parsed_ok and choice.get("error") is not None:
        status = TransportStatus.ERROR_ENVELOPE
        detail = error_summary({"error": choice.get("error")}, prefix="error_envelope choice") + identity_note
    elif shape_problem is not None:
        status = TransportStatus.ERROR_ENVELOPE
        detail = shape_problem + identity_note
    elif finish_reason is not None and finish_reason.strip().lower() == "error":
        status = TransportStatus.ERROR_ENVELOPE
        detail = "error_envelope finish_reason=error without an error object" + identity_note
    elif mismatches:
        status = TransportStatus.PROVIDER_MISMATCH
        detail = "served outside the endpoint lock: " + "; ".join(mismatches)
    return ParsedCompletion(
        transport_status=status,
        completion_status=(completion_status(visible_content, finish_reason, native_finish_reason,
                                             message.get("refusal")) if status is TransportStatus.OK else None),
        generation_id=generation_id,
        observed_model=observed_model,
        observed_provider=observed_provider,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        visible_content=visible_content,
        reasoning_returned=reasoning_returned,
        usage=usage,
        error_detail=None if detail is None else (redact(detail) or "")[:MAX_ERROR_DETAIL_CHARS],
    )
