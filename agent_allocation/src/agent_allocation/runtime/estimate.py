"""Conservative ex-ante reservation for one attempt, from the pinned endpoint prices (MASTER_SPEC §7.1).

``reservation = multiplier * (input_tokens * input_price + max_completion_tokens * completion_price)`` with

* ``input_tokens = ceil(rendered_chars / 2) + 64`` - at most one token per two characters of the rendered messages
  (real tokenizers average ~4 chars/token for this text), plus chat-template overhead;
* ``max_completion_tokens = reasoning_budget + completion_headroom`` - the request's own hard cap, so reasoning and
  visible output together can never exceed it;
* ``completion_price = max(output price, internal reasoning price)`` - reasoning tokens are billed as completion
  tokens; taking the larger price covers either billing rule.

Forced mode (``forced=True``: two raw ``/completions`` calls per node, THINK then ANSWER)::

    reservation = multiplier * (input_price * (2 * P + budget + FORCED_CLOSE_TOKENS)
                                + completion_price * (budget + headroom))

with ``P = input_token_bound(rendered_chars) + FORCED_TEMPLATE_OVERHEAD_TOKENS`` (the rendered chat template adds the
xhigh reasoning instruction and role tags): THINK reads P and writes exactly ``budget``; ANSWER re-reads P plus the
re-tokenized reasoning (``budget`` + 2-3 seam tokens, covered by ``FORCED_CLOSE_TOKENS``) and writes at most
``headroom``. Internally retried ANSWER failures are refusals that bill nothing (a retry with a known non-zero cost
is not retried); the reservation therefore covers one THINK and one ANSWER.

``multiplier`` is ``budget.reservation_multiplier`` (1.1), documented as the safety factor for fees and rounding;
values below 1 are refused. Prices come from the endpoint lock (decimal strings); a missing or unparseable input or
output price is an error, never zero. Config keys consumed here: none (the dispatcher passes the multiplier).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

from agent_allocation.contracts import EndpointLock, GenerationSpec

INPUT_CHARS_PER_TOKEN_FLOOR = 2
INPUT_TOKEN_OVERHEAD = 64
FORCED_TEMPLATE_OVERHEAD_TOKENS = 160   # >= ceil(len(api.forced.XHIGH_INSTRUCTION) / 2) + tags (tested)
FORCED_CLOSE_TOKENS = 16               # "\n</think>\n\n" plus the observed 2-3 token re-tokenization seam


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


def forced_prompt_tokens(rendered_chars: int) -> int:
    """Conservative token bound on a forced node's rendered THINK prompt."""
    return input_token_bound(rendered_chars) + FORCED_TEMPLATE_OVERHEAD_TOKENS


def reservation_usd(spec: GenerationSpec, rendered_chars: int, lock: EndpointLock, *, multiplier: Decimal,
                    forced: bool = False) -> Decimal:
    factor = Decimal(str(multiplier))
    if not factor.is_finite() or factor < 1:
        raise ValueError(f"reservation multiplier must be >= 1, got {multiplier!r}")
    input_price = _price(lock.input_price_per_token, "input_price_per_token", required=True)
    output_price = _price(lock.output_price_per_token, "output_price_per_token", required=True)
    reasoning_price = _price(lock.internal_reasoning_price_per_token, "internal_reasoning_price_per_token",
                             required=False)
    completion_price = max(output_price, reasoning_price) if reasoning_price is not None else output_price
    if forced:
        prompt = forced_prompt_tokens(rendered_chars)
        input_tokens = 2 * prompt + spec.reasoning_budget + FORCED_CLOSE_TOKENS
        completion_tokens = spec.reasoning_budget + spec.completion_headroom
        return (Decimal(input_tokens) * input_price + Decimal(completion_tokens) * completion_price) * factor
    input_cost = Decimal(input_token_bound(rendered_chars)) * input_price
    completion_cost = Decimal(spec.max_completion_tokens) * completion_price
    return (input_cost + completion_cost) * factor


def rendered_chars(messages) -> int:
    """Characters the reservation is computed from: every message's content plus its role."""
    return sum(len(str(m.get("content", ""))) + len(str(m.get("role", ""))) for m in messages)
