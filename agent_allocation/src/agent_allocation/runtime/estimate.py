"""Conservative ex-ante reservation for one attempt, from the pinned endpoint prices (MASTER_SPEC §7.1).

``reservation = multiplier * (input_tokens * input_price + max_completion_tokens * completion_price)`` with

* ``input_tokens = ceil(rendered_chars / 2) + 64`` - at most one token per two characters of the rendered messages
  (real tokenizers average ~4 chars/token for this text), plus chat-template overhead;
* ``max_completion_tokens = reasoning_budget + completion_headroom`` - the request's own hard cap, so reasoning and
  visible output together can never exceed it;
* ``completion_price = max(output price, internal reasoning price)`` - reasoning tokens are billed as completion
  tokens; taking the larger price covers either billing rule.

``multiplier`` is ``budget.reservation_multiplier`` (1.1), documented as the safety factor for fees and rounding;
values below 1 are refused. Prices come from the endpoint lock (decimal strings); a missing or unparseable input or
output price is an error, never zero. Config keys consumed here: none (the dispatcher passes the multiplier).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

from agent_allocation.contracts import EndpointLock, GenerationSpec

INPUT_CHARS_PER_TOKEN_FLOOR = 2
INPUT_TOKEN_OVERHEAD = 64


class PriceError(ValueError):
    pass


def _price(value: str | None, name: str, *, required: bool) -> Decimal | None:
    if value is None or str(value).strip() == "":
        if required:
            raise PriceError(f"endpoint lock has no {name}")
        return None
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise PriceError(f"endpoint lock {name} is not a decimal: {value!r}") from exc
    if not price.is_finite() or price < 0:
        raise PriceError(f"endpoint lock {name} must be a finite non-negative decimal, got {value!r}")
    return price


def input_token_bound(rendered_chars: int) -> int:
    if rendered_chars < 0:
        raise ValueError("rendered_chars must be >= 0")
    return -(-rendered_chars // INPUT_CHARS_PER_TOKEN_FLOOR) + INPUT_TOKEN_OVERHEAD


def reservation_usd(spec: GenerationSpec, rendered_chars: int, lock: EndpointLock, *, multiplier: Decimal) -> Decimal:
    factor = Decimal(str(multiplier))
    if not factor.is_finite() or factor < 1:
        raise ValueError(f"reservation multiplier must be >= 1, got {multiplier!r}")
    input_price = _price(lock.input_price_per_token, "input_price_per_token", required=True)
    output_price = _price(lock.output_price_per_token, "output_price_per_token", required=True)
    reasoning_price = _price(lock.internal_reasoning_price_per_token, "internal_reasoning_price_per_token",
                             required=False)
    completion_price = max(output_price, reasoning_price) if reasoning_price is not None else output_price
    input_cost = Decimal(input_token_bound(rendered_chars)) * input_price
    completion_cost = Decimal(spec.max_completion_tokens) * completion_price
    return (input_cost + completion_cost) * factor


def rendered_chars(messages) -> int:
    """Characters the reservation is computed from: every message's content plus its role."""
    return sum(len(str(m.get("content", ""))) + len(str(m.get("role", ""))) for m in messages)
