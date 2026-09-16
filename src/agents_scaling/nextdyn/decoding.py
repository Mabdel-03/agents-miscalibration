"""The one place a :class:`~agents_scaling.study.types.Decoding` is built (plan D5, D6).

ARD36 froze two decodings as module constants and every caller that needed a third budget
re-derived ``1 + T + 1 + 256`` by hand, so the ``max_tokens`` identity ended up asserted in
the client, far from where it was constructed.  Here construction and assertion are the same
function: :func:`decoding_for` asks the adapter for its decoding and then refuses any answer
whose ``max_tokens`` is not that adapter's declared output envelope --
``1 + thinking_cap + 1 + final_cap`` for a native-thinking cell (amendment A1: the two
delimiters are *sampled* tokens and sit inside ``max_tokens``, so the main contract is 3,074
and not the spec's 3,072) and ``ModelCell.total_output_tokens`` for the spec-5.6 envelope
companion.

Two facts live here and nowhere else:

``CONTINUATION_PREFILL_TEXT``
    the JSON report opening prefilled into the *prompt* of a phase-2 continuation (plan D5,
    amendment A3).  It is prompt bytes, so it enters the request identity, and its sha256 is
    hashed into ``prompts.lock.json`` by
    :func:`agents_scaling.nextdyn.locks.lock_prompts`, which reads this very attribute
    (``locks.CONTINUATION_PREFILL_ATTR``).  Runtime reads the *locked* bytes through
    :func:`prefill_text`, which fails closed when the lock has drifted from this
    declaration: the string is declared once here and verified against the lock, never
    declared twice (R1).

no guided field
    ``guided_json`` is accepted but **not enforced** on vLLM 0.21 (measured; contract §3), so
    ``study.protocol.final_phase_constrained_decoding`` is frozen false and no decoding this
    module builds ever carries a guided flag.  The JSON object comes from prompt-only
    formatting (plan D6).

Spec: 5.5 (sampling recipe and token contract), 5.6 (the total-output envelope companion),
6.1 (public output limits).  Plan: D5, D6, §1.7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study.types import Decoding

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agents_scaling.nextdyn.adapters import ModelAdapter
    from agents_scaling.nextdyn.locks import PromptsLock

# --------------------------------------------------------------------------- the frozen prefill

#: The report opening prefilled into a phase-2 continuation's prompt (plan D5, amendment A3).
#:
#: ARD36 measured why this exists: a forced ``</think>`` takes the delimiter out of the stream
#: but does not put the model into report mode, and about a third of cap-hit draws simply
#: carry on deriving.  Committed to the opening of the JSON object, the model writes the
#: object.  The leading ``\n\n`` is what a Qwen3 completion emits after ``</think>`` anyway, so
#: the prefix tokenizes exactly as a natural close would have, and D6's object opens with
#: ``reasoning``, so the prefilled opening is also the spec-6.1 field order's first key.
CONTINUATION_PREFILL_TEXT: Final[str] = '\n\n{\n  "reasoning": "'

#: ``RequestSpec.role`` labels: ledger metadata only, never part of the request identity.
DECISION_ROLE: Final[str] = "nd1_decision"
CONTINUATION_ROLE: Final[str] = "nd1_continuation"

#: Logprob width of an ordinary phase: the sampled token alone.  Only the receiver phases ask
#: for the spec-11.2 answer-logit margin, and the width is a request argument rather than a
#: ``Decoding`` field, so it travels on
#: :class:`~agents_scaling.nextdyn.inference.DecisionRequest` and enters the attempt ledger.
TOP_LOGPROBS_SAMPLED: Final[int] = 1


def prefill_text(lock: "PromptsLock | None" = None) -> str:
    """The continuation prefill bytes, read from ``prompts.lock.json`` when one is given.

    :data:`CONTINUATION_PREFILL_TEXT` is the declaration and the lock is its frozen copy
    (``locks.lock_prompts`` hashes exactly this attribute).  A caller holding the lock passes
    it and gets the *locked* bytes back after they have been checked against the declaration;
    a drifted lock is a :class:`ContractViolation`, never a silent preference for one of the
    two.  Without a lock the declaration is returned unchanged, which is what ``lock-inputs``
    itself needs before any lock exists.
    """
    if lock is None:
        return CONTINUATION_PREFILL_TEXT
    locked = lock.continuation_prefill_text
    if locked is None:
        raise ContractViolation(
            "prompts.lock.json carries no continuation prefill; run `lock-inputs` before "
            "generating (the prefill is prompt bytes and enters every phase-2 request id)")
    if locked != CONTINUATION_PREFILL_TEXT:
        raise ContractViolation(
            "prompts.lock.json's continuation prefill has drifted from "
            f"decoding.CONTINUATION_PREFILL_TEXT: locked {locked!r} != declared "
            f"{CONTINUATION_PREFILL_TEXT!r}; re-lock or restore the declaration")
    return locked


def _squeeze(text: str) -> str:
    return "".join(text.split())


def continuation_prefill_ids(tokenizer: Any, text: str | None = None) -> tuple[int, ...]:
    """Token ids of the prefilled report opening, round-trip checked.

    ``add_special_tokens=False`` because the phase-2 prefix already carries every control
    token the chat template produced.  The round-trip check is whitespace-insensitive: the
    pinned Qwen3 tokenizer reproduces the bytes exactly, a whitespace tokenizer (the fake
    server's) reproduces the *tokens* and normalises the runs between them, and what must
    never drift is the JSON opening itself.
    """
    source = CONTINUATION_PREFILL_TEXT if text is None else text
    ids = tuple(int(i) for i in tokenizer.encode(source, add_special_tokens=False))
    if not ids:
        raise ContractViolation(f"the tokenizer produced no ids for the prefill {source!r}")
    decoded = tokenizer.decode(list(ids))
    if not isinstance(decoded, str):
        raise ContractViolation("tokenizer.decode did not return text")
    if _squeeze(decoded) != _squeeze(source):
        raise ContractViolation(
            "the prefilled report opening does not round-trip through the tokenizer: "
            f"{source!r} -> {ids} -> {decoded!r}")
    return ids


# --------------------------------------------------------------------------- decodings


def build_decoding(sampling: K.Sampling, *, max_tokens: int, enable_thinking: bool,
                   thinking_tokens: int, final_tokens: int) -> Decoding:
    """One :class:`Decoding` from the study's single sampling recipe (spec 5.5).

    Every sampler field is copied out by name, so a recipe field the study declares and the
    wire cannot carry becomes a refusal instead of a silent drop: ``study.types.Decoding`` has
    no ``frequency_penalty`` slot, so a non-zero ``frequency_penalty`` could reach vLLM only
    outside the request identity, which R5 forbids.
    """
    if float(sampling.frequency_penalty) != 0.0:
        raise ContractViolation(
            f"sampling.frequency_penalty is {sampling.frequency_penalty!r}, but "
            "study.types.Decoding carries no frequency_penalty slot: the value could only "
            "reach vLLM outside the request identity, which R5 forbids.  Keep it at 0.0, or "
            "widen Decoding (a study/ change this study may not make)")
    if int(max_tokens) < 1:
        raise ContractViolation(f"max_tokens must be >= 1, got {max_tokens!r}")
    return Decoding(
        temperature=float(sampling.temperature),
        top_p=float(sampling.top_p),
        top_k=int(sampling.top_k),
        min_p=float(sampling.min_p),
        presence_penalty=float(sampling.presence_penalty),
        repetition_penalty=float(sampling.repetition_penalty),
        max_tokens=int(max_tokens),
        enable_thinking=bool(enable_thinking),
        guided_json=False,          # plan D6: not enforced on vLLM 0.21, so never sent
        thinking_tokens=int(thinking_tokens),
        final_tokens=int(final_tokens),
    )


def envelope_tokens_of(adapter: "ModelAdapter", tokens: K.TokenContract) -> int:
    """``max_tokens`` the adapter's output envelope allows for one decision.

    Native thinking pays for the two sampled delimiters and keeps an independent final
    allowance; the spec-5.6 companion has one total output cap carried on the adapter
    (``ModelCell.total_output_tokens``, bound by :func:`adapters.adapter_for`).
    """
    if adapter.name == K.ADAPTER_NATIVE_THINKING:
        return int(tokens.max_new_tokens)
    if adapter.name == K.ADAPTER_TOTAL_OUTPUT_ENVELOPE:
        cap = adapter.total_output_tokens
        if cap is None:
            raise ContractViolation(
                f"{adapter.name}: total_output_tokens is null; bind the adapter to its model "
                "cell with adapters.adapter_for(model) before building a decoding (spec 5.6)")
        return int(cap)
    raise ContractViolation(f"unknown adapter {adapter.name!r}; declared: {K.ADAPTERS}")


def decoding_for(adapter: "ModelAdapter", sampling: K.Sampling, tokens: K.TokenContract, *,
                 purpose: str) -> Decoding:
    """The decoding of one decision, with the adapter's output envelope asserted.

    The adapter decides the shape; this function refuses any answer whose ``max_tokens``
    does not match :func:`envelope_tokens_of`, so the cap can never drift away from the
    contract the arm was planned under, and refuses any answer carrying a guided field.
    """
    if not purpose or purpose not in K.PHASES:
        raise ContractViolation(f"unknown purpose {purpose!r}; the phases are {K.PHASES}")
    decoding = adapter.decoding(sampling, tokens, purpose=purpose)
    expected = envelope_tokens_of(adapter, tokens)
    if int(decoding.max_tokens) != expected:
        raise ContractViolation(
            f"{adapter.name}: max_tokens {decoding.max_tokens} != {expected} for purpose "
            f"{purpose!r} (native: 1 + {tokens.thinking_cap} + 1 + {tokens.final_cap}; "
            "envelope: ModelCell.total_output_tokens)")
    if decoding.guided_json:
        raise ContractViolation(
            "no nextdyn request carries a guided field: guided_json is accepted but NOT "
            "enforced on vLLM 0.21, so protocol.final_phase_constrained_decoding is frozen "
            "false and the JSON object comes from the prompt alone (plan D6)")
    return decoding


def decoding_for_contract(contract: K.StudyContract, adapter: "ModelAdapter",
                          tokens: K.TokenContract, *, purpose: str) -> Decoding:
    """:func:`decoding_for` under the study's one sampling recipe (``contract.sampling``).

    The recipe is a single study-wide fact (spec 5.5), so no producer chooses it per call:
    every generation-side caller comes through here and an arm differs only in ``tokens``.
    """
    return decoding_for(adapter, contract.sampling, tokens, purpose=purpose)


def continuation_decoding(tokens: K.TokenContract, sampling: K.Sampling) -> Decoding:
    """Sampling recipe of the phase-2 report continuation (plan D5, amendment A3).

    The same recipe as the decision it continues -- a continuation is the *same* draw,
    resumed -- with the three differences the route forces: ``max_tokens`` is the bare
    ``final_cap`` (no delimiters left to pay for and no thinking channel), ``thinking_tokens``
    is 0 because no ``thinking_token_budget`` is sent (the channel is already closed, inside
    the prompt), and ``enable_thinking`` is false because ``/v1/completions`` renders no chat
    template at all, so the flag is identity bookkeeping recording "no template".

    Takes the recipe explicitly (contract §2.1 append, WP1): the sampler is a config value and
    no number in this package may be a module constant, so the recipe cannot be defaulted.
    """
    return build_decoding(sampling, max_tokens=tokens.final_cap, enable_thinking=False,
                          thinking_tokens=0, final_tokens=tokens.final_cap)


def is_continuation(decoding: Decoding) -> bool:
    """Whether ``decoding`` is a phase-2 report continuation rather than a decision."""
    return (int(decoding.thinking_tokens) == 0 and int(decoding.final_tokens) > 0
            and int(decoding.max_tokens) == int(decoding.final_tokens)
            and not decoding.enable_thinking)


__all__ = [
    "CONTINUATION_PREFILL_TEXT", "CONTINUATION_ROLE", "DECISION_ROLE", "TOP_LOGPROBS_SAMPLED",
    "build_decoding", "continuation_decoding", "continuation_prefill_ids", "decoding_for",
    "decoding_for_contract", "envelope_tokens_of", "is_continuation", "prefill_text",
]
