"""ARD36 decoding values: the split thinking/report budget (WP1).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §6.2
(generation contract: thinking on, temperature 0.6 / top-p 0.95 / top-k 20, the split
budget driven by vLLM's native ``thinking_token_budget``) and §6.3 (the independent
256-token report allowance).  Plan §5.2.

Every value here is derived from :mod:`agents_scaling.ard.contract`; nothing is
re-declared.  One ARD decoding is a frozen :class:`agents_scaling.study.types.Decoding`
carrying two extra integers:

``thinking_tokens``
    the vLLM ``thinking_token_budget`` ``T`` sent in ``extra_body``.  The sampler forces
    ``</think>`` (151668) the moment the generated thinking count reaches ``T``.
``final_tokens``
    the post-``</think>`` report allowance the parser reads (``contract.FINAL_TOKENS``).

Both enter ``decoding_hash`` and therefore the request identity: they change what is sent
to the engine, so two budget arms must never collide on one cached request (contract §0.3).

``max_tokens`` is ``contract.max_tokens_for(T) = 1 + T + 1 + 256``.  The two delimiters
are sampled output tokens, so they must be inside ``max_tokens`` or a budget cap would eat
into the report.

Root aliasing (spec §10 E2d, plan §5.2): the *root* draws of every arm use one frozen
decoding at ``contract.ROOT_THINKING_TOKENS``.  A high-budget arm varies only the
*revision* decoding, so its root requests alias into the existing bank instead of being
regenerated.  :func:`decoding_for_purpose` is the single place that rule lives.
"""

from __future__ import annotations

from typing import Final

from agents_scaling.ard import contract as K
from agents_scaling.study.types import Decoding

__all__ = [
    "ARD_DECODING_ROLE",
    "CONTINUATION_DECODING",
    "CONTINUATION_PREFILL_TEXT",
    "CONTINUATION_ROLE",
    "HIGH_DECODING",
    "LOW_DECODING",
    "ROOT_DECODING",
    "THINKING_BUDGETS",
    "ard_decoding",
    "budget_of",
    "continuation_decoding",
    "decoding_for",
    "decoding_for_purpose",
    "is_continuation",
    "is_root_purpose",
    "max_tokens_for",
]

#: Re-exported from the frozen contract so callers never re-derive ``1 + T + 1 + 256``.
max_tokens_for = K.max_tokens_for

#: ``RequestSpec.role`` label of an ARD decision (ledger metadata only, not identity).
ARD_DECODING_ROLE: Final[str] = "ard_decision"


def ard_decoding(thinking_tokens: int, *, final_tokens: int = K.FINAL_TOKENS) -> Decoding:
    """The ARD sampling recipe at thinking budget ``thinking_tokens`` (spec §6.2).

    General constructor: any positive budget is accepted so the registration-time
    conformance probe (plan §5.2, ``thinking_token_budget=8``) and the unit tests can
    build a small-budget decoding.  Production budgets go through :func:`decoding_for`.
    """
    budget = int(thinking_tokens)
    final = int(final_tokens)
    if budget < 1:
        raise ValueError(f"thinking budget must be >= 1, got {thinking_tokens!r}")
    if final < 1:
        raise ValueError(f"final allowance must be >= 1, got {final_tokens!r}")
    return Decoding(
        temperature=K.TEMPERATURE,
        top_p=K.TOP_P,
        top_k=K.TOP_K,
        min_p=K.MIN_P,
        presence_penalty=K.PRESENCE_PENALTY,
        repetition_penalty=K.REPETITION_PENALTY,
        max_tokens=1 + budget + 1 + final,
        enable_thinking=K.ENABLE_THINKING,
        guided_json=False,
        thinking_tokens=budget,
        final_tokens=final,
    )


#: The two frozen production arms (spec §6.2; E2d and the receiver high arm use HIGH).
LOW_DECODING: Final[Decoding] = ard_decoding(K.THINKING_TOKENS_LOW)
HIGH_DECODING: Final[Decoding] = ard_decoding(K.THINKING_TOKENS_HIGH)
#: One frozen root decoding for every arm, so roots alias instead of being regenerated.
ROOT_DECODING: Final[Decoding] = ard_decoding(K.ROOT_THINKING_TOKENS)

THINKING_BUDGETS: Final[tuple[int, ...]] = (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH)

_BY_BUDGET: Final[dict[int, Decoding]] = {
    K.THINKING_TOKENS_LOW: LOW_DECODING,
    K.THINKING_TOKENS_HIGH: HIGH_DECODING,
}


def decoding_for(thinking_tokens: int) -> Decoding:
    """The frozen decoding of one production budget arm; unknown budgets are refused."""
    budget = int(thinking_tokens)
    try:
        return _BY_BUDGET[budget]
    except KeyError as exc:
        raise ValueError(
            f"thinking budget {budget} is not a frozen ARD36 arm {THINKING_BUDGETS}"
        ) from exc


def is_root_purpose(purpose: str) -> bool:
    """Whether ``purpose`` names a stateless-bank draw (roots and FRESH draws)."""
    return str(purpose) == K.PURPOSE_ROOT


def decoding_for_purpose(purpose: str, thinking_tokens: int) -> Decoding:
    """Decoding of one decision: roots ignore the arm's budget (spec §10 E2d line 341).

    A root/FRESH draw always uses :data:`ROOT_DECODING`, so the *same* root spec has one
    request id under both the low- and the high-budget arm and no root request is ever
    issued at ``THINKING_TOKENS_HIGH``.  Every other purpose uses the arm's budget.
    """
    if is_root_purpose(purpose):
        return ROOT_DECODING
    return decoding_for(thinking_tokens)


def budget_of(decoding: Decoding) -> int:
    """The thinking budget carried by ``decoding`` (``0`` when it is not an ARD decoding)."""
    return int(getattr(decoding, "thinking_tokens", 0) or 0)


# --------------------------------------------------------------------------- phase 2


#: ``RequestSpec.role`` of a phase-2 report continuation (ledger metadata only).
CONTINUATION_ROLE: Final[str] = "ard_continuation"

#: The report opening prefilled into the phase-2 *prompt* (plan §5.2 contingency).
#:
#: A forced ``</think>`` removes the delimiter from the stream but does not make the model
#: switch into report mode: measured on the pilot roots cell, a third of the cap-hit draws
#: emit no ``<final>`` at all and simply carry on deriving.  Prefilling the opening tag is
#: what makes phase 2 work — the model resumes reasoning without it — and the leading
#: ``\n\n`` is what a Qwen3 completion writes after ``</think>`` anyway, so the prefix
#: tokenizes exactly as a natural close would have.
CONTINUATION_PREFILL_TEXT: Final[str] = "\n\n" + K.FINAL_OPEN


def continuation_decoding(*, final_tokens: int = K.FINAL_TOKENS) -> Decoding:
    """Sampling recipe of the phase-2 continuation (spec §6.3 report allowance).

    The same §6.2 sampler settings as the decision itself — a continuation is the *same*
    draw, resumed — with three differences that are all forced by the route it takes:

    ``max_tokens`` is the bare report allowance (no delimiters to pay for, no thinking
    channel); ``thinking_tokens`` is 0 because no ``thinking_token_budget`` is sent (the
    thinking channel is already closed, inside the prompt); and ``enable_thinking`` is
    false because ``/v1/completions`` renders no chat template at all — the prompt is an
    exact token prefix, so the flag is identity bookkeeping and records "no template".
    """
    final = int(final_tokens)
    if final < 1:
        raise ValueError(f"final allowance must be >= 1, got {final_tokens!r}")
    return Decoding(
        temperature=K.TEMPERATURE,
        top_p=K.TOP_P,
        top_k=K.TOP_K,
        min_p=K.MIN_P,
        presence_penalty=K.PRESENCE_PENALTY,
        repetition_penalty=K.REPETITION_PENALTY,
        max_tokens=final,
        enable_thinking=False,
        guided_json=False,
        thinking_tokens=0,
        final_tokens=final,
    )


#: The one frozen phase-2 decoding (``contract.FINAL_TOKENS`` report tokens).
CONTINUATION_DECODING: Final[Decoding] = continuation_decoding()


def is_continuation(decoding: Decoding) -> bool:
    """Whether ``decoding`` is a phase-2 report continuation rather than a decision."""
    return (
        budget_of(decoding) == 0
        and int(getattr(decoding, "final_tokens", 0) or 0) > 0
        and int(decoding.max_tokens) == int(decoding.final_tokens)
    )
