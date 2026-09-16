"""Model adapters: everything that differs between two ways of asking for a thought.

Spec 5.6 compares two *generation contracts* on the same public template -- Qwen3 with the
native thinking channel and a ``thinking_token_budget``, and a companion served with
``enable_thinking=false`` under one total output envelope -- and the ARD36 way of expressing
that (branching on the checkpoint name inside the client) is exactly how the token accounting
drifted.  Here each contract is one object that knows three things and nothing else:

* the **decoding** it needs (``1 + T + 1 + F`` versus one total cap), built through
  :mod:`agents_scaling.nextdyn.decoding` so the envelope is asserted at construction;
* the **request kwargs** it adds to the wire -- the native cell's
  ``extra_body.thinking_token_budget``, the envelope cell's nothing (its
  ``chat_template_kwargs.enable_thinking=false`` already rides on ``Decoding``, so it is
  inside ``decoding_hash`` and therefore inside the request id, R5);
* how to **split** a committed response into the native reasoning span and the public span.

The split is the one place the two channels are separated, and it is deliberately *not* the
parser: it works on token ids and the tokenizer's own delimiter ids, decides
``boundary_injected`` from the arithmetic ``thinking_tokens == budget`` (exact on vLLM 0.21,
contract §3) and hands WP2's :func:`schema.parse_public_output` a final span it never has to
hunt for.  ``native_reasoning_text`` and the public ``reasoning`` field of the JSON object are
different things (spec 6.1) and confusing them is what this boundary prevents.

The delimiter ids come from the *tokenizer*, never from a constant: both pinned snapshots
share one tokenizer blob (``<think>``=151667, ``</think>``=151668, contract §3), and a new
model cell is meant to be one adapter plus one lock entry (plan §1.10), not a new constant.

The family-transfer cell (Llama, weights gated, amendment A4) compiles -- it has a lane, a
profile and an adapter -- and raises :class:`BlockedAdapter` the moment anything asks it for a
request, so the planner can express E3C_FAMILY as a NOT_RUN row without a special case.

Spec: 5.4, 5.6, 6.1, 17.4.  Plan: D5, D6, D11, §1.7, §1.10.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final, Mapping, Protocol, runtime_checkable

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn.errors import ContractViolation, NextdynError
from agents_scaling.study.types import Decoding

#: Text forms of the reasoning delimiters.  The *ids* are resolved per tokenizer (see
#: :func:`delimiter_ids`); these strings are vocabulary, not knobs.
THINK_OPEN: Final[str] = "<think>"
THINK_CLOSE: Final[str] = "</think>"
#: The end-of-turn control token.  It terminates the completion and is not part of the public
#: span (ARD36 counted it into the report span and every final-token count was one too many).
TURN_END: Final[str] = "<|im_end|>"


class BlockedAdapter(NextdynError):
    """A model cell that compiles for planning but may never issue a request (plan D11).

    Raised at *request* time, not at compile time: the study must be able to plan, cost and
    report E3C_FAMILY as NOT_RUN (amendment A4) without the config refusing to load.
    """


@dataclass(frozen=True)
class SplitCompletion:
    """One committed completion, separated into its two channels.

    ``delimiter_positions`` are the indices of ``<think>`` and ``</think>`` inside
    ``completion_token_ids`` (empty when the channel is absent, one entry when it opened and
    never closed), so WP2 can report ``NO_THINK_OPEN`` / ``PREMATURE_REASONING_EOS`` from the
    same object without re-scanning the ids.

    ``boundary_injected`` is the *thinking* event: the sampler forced ``</think>`` because the
    thinking count reached the budget.  ``natural_stop`` is the *whole-response* event: the
    completion ended on the model's own turn token with both delimiters well formed.  They are
    independent -- a forced boundary leaves the final allowance untouched -- and the
    continuation trigger is ``boundary_injected`` alone (plan D5).
    """

    native_ids: tuple[int, ...]
    delimiter_positions: tuple[int, ...]
    final_ids: tuple[int, ...]
    boundary_injected: bool
    natural_stop: bool
    #: Tokens of the final span beyond ``final_cap``; they are dropped from ``final_ids`` and
    #: counted here so ``final_overrun`` is a number, not just a flag.
    overrun_tokens: int = 0
    #: Tokens the completion emitted after the forced boundary and before the turn token.
    #: On a continued draw the continuation replaces them, and they are still real GPU work:
    #: the D-02 fix is that they are attributed rather than dropped (see ``telemetry``).
    discarded_span_tokens: int = 0

    @property
    def thinking_tokens(self) -> int:
        return len(self.native_ids)

    @property
    def opened(self) -> bool:
        return len(self.delimiter_positions) >= 1

    @property
    def closed(self) -> bool:
        return len(self.delimiter_positions) == 2


@runtime_checkable
class ModelAdapter(Protocol):
    """One generation contract (spec 5.6).

    ``total_output_tokens`` is contract §2.1's append (WP1): ``decoding.decoding_for`` is
    required to assert ``max_tokens == total_output_tokens`` for the envelope adapter and
    receives only the adapter, so the cap has to be on the adapter.  It is ``None`` on the
    native contract, whose caps are ``TokenContract``'s and live in one place only (R1).
    """

    name: str
    total_output_tokens: int | None

    def decoding(self, sampling: K.Sampling, tokens: K.TokenContract, *, purpose: str) -> Decoding: ...

    def request_kwargs(self, decoding: Decoding) -> dict[str, Any]: ...

    def split(self, record: Any, tokenizer: Any) -> SplitCompletion: ...

    def continuation_supported(self) -> bool: ...

    def assert_compatible(self, model: K.ModelCell, cfg_block: Mapping[str, Any] | None) -> None: ...


# --------------------------------------------------------------------------- delimiter ids


def _token_id(tokenizer: Any, token: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        value = convert(token)
    except Exception:                                   # noqa: BLE001 - a tokenizer that cannot
        return None                                     # answer is "absent", not a crash
    unknown = getattr(tokenizer, "unk_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value == unknown:
        return None
    return int(value)


def delimiter_ids(tokenizer: Any) -> tuple[int, int]:
    """``(<think>, </think>)`` ids of ``tokenizer``; a tokenizer without them is refused.

    Read from the tokenizer rather than frozen as constants: a new model cell is one adapter
    plus one lock entry (plan §1.10), and the two pinned snapshots agreeing on 151667/151668
    is a *measured fact about them* (contract §3), not a property of the study.
    """
    start, end = _token_id(tokenizer, THINK_OPEN), _token_id(tokenizer, THINK_CLOSE)
    if start is None or end is None:
        raise ContractViolation(
            f"the tokenizer does not know the reasoning delimiters {THINK_OPEN!r}/{THINK_CLOSE!r}; "
            "a native-thinking cell cannot be split without them (spec 5.6)")
    return start, end


def turn_end_id(tokenizer: Any) -> int | None:
    """``<|im_end|>`` id, or ``None`` when the tokenizer has no turn token."""
    return _token_id(tokenizer, TURN_END)


def _completion_ids(record: Any) -> tuple[int, ...]:
    return tuple(int(i) for i in (record.response.get("token_ids") or ()))


def _sent(record: Any, field: str) -> int:
    """A sampler field as it was actually sent (``RequestRecord.sampling``).

    The budget the split compares against is the budget the engine received, not one
    re-derived from a contract the caller happens to hold: a resumed cell measures exactly
    what the original run measured.
    """
    value = record.sampling.get(field)
    if value is None:
        raise ContractViolation(f"committed request {record.request_id[:12]} carries no {field!r}")
    return int(value)


def _trim_turn(ids: tuple[int, ...], tokenizer: Any) -> tuple[int, ...]:
    end = turn_end_id(tokenizer)
    if end is not None and ids and ids[-1] == end:
        return ids[:-1]
    return ids


# --------------------------------------------------------------------------- the two adapters


@dataclass(frozen=True)
class NativeThinkingAdapter:
    """Qwen3 with the native thinking channel (spec 5.5, plan D5).

    ``thinking_token_budget`` is an engine argument, so it rides on ``Decoding.thinking_tokens``
    and is therefore inside ``decoding_hash`` and inside the request id before it is ever put
    in ``extra_body`` (R5).  ``boundary_injected == (thinking_tokens == budget)`` is exact on
    vLLM 0.21: forcing the delimiter is a logits boost, so ``</think>`` is genuinely sampled
    into ``token_ids`` and nothing in the response marks the injection (contract §3).
    """

    name: str = K.ADAPTER_NATIVE_THINKING
    total_output_tokens: int | None = None

    def decoding(self, sampling: K.Sampling, tokens: K.TokenContract, *, purpose: str) -> Decoding:
        return D.build_decoding(sampling, max_tokens=tokens.max_new_tokens, enable_thinking=True,
                                thinking_tokens=tokens.thinking_cap, final_tokens=tokens.final_cap)

    def request_kwargs(self, decoding: Decoding) -> dict[str, Any]:
        budget = int(decoding.thinking_tokens)
        if budget < 1:
            raise ContractViolation(
                f"a {self.name} request must carry a thinking budget; decoding "
                f"{decoding.hash[:12]} has thinking_tokens={budget}")
        return {"extra_body": {"thinking_token_budget": budget}}

    def continuation_supported(self) -> bool:
        return True

    def assert_compatible(self, model: K.ModelCell, cfg_block: Mapping[str, Any] | None) -> None:
        if model.adapter != self.name:
            raise ContractViolation(f"model cell {model.name!r} declares adapter {model.adapter!r}, "
                                    f"not {self.name!r}")
        if model.total_output_tokens is not None:
            raise ContractViolation(
                f"model cell {model.name!r}: total_output_tokens belongs to the "
                f"{K.ADAPTER_TOTAL_OUTPUT_ENVELOPE} adapter; the native cap is "
                "TokenContract.thinking_cap and lives in one place only (R1)")
        _refuse_native_budget_override(model, cfg_block)

    def split(self, record: Any, tokenizer: Any) -> SplitCompletion:
        """Separate the native channel from the public span, on token ids.

        Everything after the closing delimiter and before the turn token is the public span.
        A span longer than the final allowance is truncated to it and the excess counted: the
        engine may over-run by a token or two under a forced boundary, and silently keeping
        the extra tokens would make ``final_tokens`` disagree with the allowance the arm was
        costed at.
        """
        start_id, end_id = delimiter_ids(tokenizer)
        budget, final_cap = _sent(record, "thinking_tokens"), _sent(record, "final_tokens")
        ids = _completion_ids(record)
        finish_reason = str(record.response.get("finish_reason"))

        if not ids or ids[0] != start_id:
            # The think-on template ends at ``assistant\n``; the model emits ``<think>`` itself.
            # Nothing here is attributable to either channel, so both spans stay empty and WP2
            # reports NO_THINK_OPEN from ``delimiter_positions``.
            return SplitCompletion(native_ids=(), delimiter_positions=(), final_ids=(),
                                   boundary_injected=False, natural_stop=False)
        try:
            end = ids.index(end_id, 1)
        except ValueError:
            return SplitCompletion(native_ids=ids[1:], delimiter_positions=(0,), final_ids=(),
                                   boundary_injected=False, natural_stop=False)

        native = ids[1:end]
        tail = _trim_turn(ids[end + 1:], tokenizer)
        kept = tail[:final_cap]
        boundary_injected = len(native) == budget
        natural_stop = finish_reason == "stop" and not boundary_injected
        return SplitCompletion(
            native_ids=native,
            delimiter_positions=(0, end),
            final_ids=kept,
            boundary_injected=boundary_injected,
            natural_stop=natural_stop,
            overrun_tokens=max(0, len(tail) - len(kept)),
            discarded_span_tokens=len(tail) if boundary_injected else 0,
        )


@dataclass(frozen=True)
class TotalOutputEnvelopeAdapter:
    """The spec-5.6 companion: one total output envelope, no thinking channel.

    ``chat_template_kwargs.enable_thinking=false`` rides on ``Decoding.enable_thinking``, so it
    is in the request identity; no ``thinking_token_budget`` field exists for this contract at
    all, and a config that declares one is the spec's error (refused in
    :meth:`assert_compatible`).  There is no forced boundary and therefore no continuation:
    the whole output is the public span.

    An instance with ``blocked`` set is a cell that compiles but may never issue a request
    (the gated Llama weights, amendment A4).
    """

    total_output_tokens: int | None = None
    blocked: bool = False
    blocked_reason: str = ""
    name: str = K.ADAPTER_TOTAL_OUTPUT_ENVELOPE

    def _guard(self) -> None:
        if self.blocked:
            raise BlockedAdapter(
                f"{self.name}: this model cell is not dispatchable ({self.blocked_reason}); "
                "it compiles so the plan can carry it, and reports NOT_RUN (amendment A4)")

    def decoding(self, sampling: K.Sampling, tokens: K.TokenContract, *, purpose: str) -> Decoding:
        self._guard()
        cap = D.envelope_tokens_of(self, tokens)
        # ``final_tokens`` is the whole envelope: under this contract every generated token is
        # public output, so the public allowance and the output cap are the same number.
        return D.build_decoding(sampling, max_tokens=cap, enable_thinking=False,
                                thinking_tokens=0, final_tokens=cap)

    def request_kwargs(self, decoding: Decoding) -> dict[str, Any]:
        self._guard()
        if int(decoding.thinking_tokens) != 0:
            raise ContractViolation(
                f"{self.name} sends no thinking_token_budget (spec 5.6), but the decoding "
                f"carries thinking_tokens={decoding.thinking_tokens}")
        return {}

    def continuation_supported(self) -> bool:
        return False

    def assert_compatible(self, model: K.ModelCell, cfg_block: Mapping[str, Any] | None) -> None:
        if model.adapter != self.name:
            raise ContractViolation(f"model cell {model.name!r} declares adapter {model.adapter!r}, "
                                    f"not {self.name!r}")
        if model.total_output_tokens is None:
            raise ContractViolation(f"model cell {model.name!r}: the {self.name} adapter needs "
                                    "total_output_tokens (spec 5.6)")
        _refuse_native_budget_override(model, cfg_block)

    def split(self, record: Any, tokenizer: Any) -> SplitCompletion:
        """The whole completion is the public span: there is no native channel to separate."""
        self._guard()
        cap = _sent(record, "final_tokens")
        ids = _trim_turn(_completion_ids(record), tokenizer)
        kept = ids[:cap]
        return SplitCompletion(
            native_ids=(), delimiter_positions=(), final_ids=kept,
            boundary_injected=False,
            natural_stop=str(record.response.get("finish_reason")) == "stop",
            overrun_tokens=max(0, len(ids) - len(kept)),
        )


def _refuse_native_budget_override(model: K.ModelCell, cfg_block: Mapping[str, Any] | None) -> None:
    """``models.<cell>.thinking_tokens`` is the spec-17.4 trap field: null is its only value.

    On the envelope adapter a value is the spec's stated error; on a native cell it would be a
    per-cell override nothing implements in v1, because the one native cap is
    ``budgets.thinking_tokens_main`` (R1).  ``config`` refuses it at compile time; the adapter
    refuses it again here so an adapter built from a hand-made block cannot slip past.
    """
    if cfg_block is None:
        return
    if cfg_block.get("thinking_tokens") is not None:
        raise ContractViolation(
            f"model cell {model.name!r} declares thinking_tokens="
            f"{cfg_block['thinking_tokens']!r}: the only admitted value is null (spec 17.4).  "
            f"On {K.ADAPTER_TOTAL_OUTPUT_ENVELOPE} a budget is the spec's error; on "
            f"{K.ADAPTER_NATIVE_THINKING} the one cap is budgets.thinking_tokens_main")


# --------------------------------------------------------------------------- the registry

#: The two generation contracts, keyed by ``contract.ADAPTER_*``.  A new model family is one
#: entry here plus a ``models.lock.json`` entry plus a serving profile (plan §1.10).
ADAPTERS: Final[Mapping[str, ModelAdapter]] = {
    K.ADAPTER_NATIVE_THINKING: NativeThinkingAdapter(),
    K.ADAPTER_TOTAL_OUTPUT_ENVELOPE: TotalOutputEnvelopeAdapter(),
}


def adapter_for(model: K.ModelCell, cfg_block: Mapping[str, Any] | None = None) -> ModelAdapter:
    """The adapter of one model cell, bound to that cell's caps and dispatchability.

    The registry entries are prototypes; binding is what turns ``ModelCell.total_output_tokens``
    into the envelope adapter's cap and an unpinned cell into a :class:`BlockedAdapter`
    (no revision means no weights: the Llama entry, amendment A4).
    """
    try:
        prototype = ADAPTERS[model.adapter]
    except KeyError as exc:
        raise ContractViolation(
            f"model cell {model.name!r} declares adapter {model.adapter!r}; "
            f"declared adapters: {sorted(ADAPTERS)}") from exc
    prototype.assert_compatible(model, cfg_block)
    if model.adapter != K.ADAPTER_TOTAL_OUTPUT_ENVELOPE:
        return prototype
    return replace(
        prototype,
        total_output_tokens=model.total_output_tokens,
        blocked=not model.pinned,
        blocked_reason=(f"model cell {model.name!r} ({model.hf_id}) has no pinned revision: its "
                        "weights are not in the offline cache"),
    )


__all__ = [
    "ADAPTERS", "BlockedAdapter", "ModelAdapter", "NativeThinkingAdapter", "SplitCompletion",
    "THINK_CLOSE", "THINK_OPEN", "TURN_END", "TotalOutputEnvelopeAdapter", "adapter_for",
    "delimiter_ids", "turn_end_id",
]
