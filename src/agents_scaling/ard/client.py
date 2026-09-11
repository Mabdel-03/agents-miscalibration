"""ARD36 inference: one logical decision == one chat request (WP1).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §6.2 (split
budget through vLLM's native ``thinking_token_budget``; ``max_tokens = 1 + T + 1 + 256``),
§6.3 (the report is read from the post-``</think>`` span only), §18 (cache identity: every
request is content-addressed and aliases across cells).  Plan §5.2 and §4.2.  Frozen
interface: ``docs/ard36/00_contract.md`` §2 (WP1).

What this module adds on top of the frozen study inference layer, which it reuses rather
than re-implements (``study/inference/client.py`` ``EndpointPool``/``VllmChatClient``: the
§4.3 prompt-token-id equality check, the §10.4 two-retry exogenous policy and endpoint
rotation; ``study/inference/store.py`` ``RequestStore``: first-writer-wins aliasing):

1. ``extra_body["thinking_token_budget"]`` plus ``logprobs``/``top_logprobs``
   (``contract.RECEIVER_TOP_LOGPROBS`` for the receiver purpose, 1 elsewhere), sent
   through :class:`ArdChatClient`.  All of it is derived from the ``RequestSpec`` — the
   budget from ``Decoding.thinking_tokens`` (hashed into the request id, so two budget
   arms can never share a cache entry) and the logprob width from the seed key's purpose.
2. A build-time context assertion (``serving.context.preflight_chat_context``) so an
   over-long prompt raises :class:`ContextTooLong` *before* an HTTP request rather than
   letting vLLM answer 400 — a non-retriable ``InfraFailure`` that would kill a cell.
3. Delimiter accounting over the returned token ids: ``<think>`` (151667) must open the
   completion; ``thinking_tokens`` are the ids strictly between it and the first
   ``</think>`` (151668); ``boundary_injected`` is ``thinking_tokens == budget`` (vLLM 0.21
   forces the boundary by a logits boost exactly when the count reaches the budget, so the
   equality is exact and no response field marks the injection); the report span is what
   follows.  A completion that never opens or never closes the thinking channel maps to
   :class:`~agents_scaling.ard.contract.ParseStatus` ``NO_THINK_OPEN`` /
   ``PREMATURE_REASONING_EOS`` instead of raising.
4. The FLOP oracle, attached exactly as ``study/runner.py:254-261`` does
   (``FlopOracle.from_checkpoint`` → ``make_cost_fn``), so ``DecisionRecord.flops`` is a
   populated dict and never the ``None`` placeholder, plus :func:`oracle_tables` /
   :func:`verify_oracle_hash` for ``FROZEN.yaml``.
5. The **two-phase report continuation** (plan §5.2 contingency), behind the same
   :func:`generate_decision` interface.  A forced ``</think>`` takes the delimiter out of
   the stream but does not make the model switch into report mode: measured on the pilot
   roots cell, 53% of 8B draws hit the thinking cap and about a third of *those* emit no
   ``<final>`` at all — they simply keep deriving — which floors valid reports at 0.82
   against the 0.95 gate and, because cap hits track item difficulty, biases every arm.
   So whenever phase 1's boundary was injected, phase 2 re-sends the exact phase-1 token
   prefix through ``</think>`` with ``<final>`` prefilled and takes the report from a
   fresh ``/v1/completions`` call — which is also the more faithful reading of §6.2's
   "independent 256-token final-output allowance" under a forced boundary.  The trigger is
   ``boundary_injected`` alone, never what phase 1's span happened to contain, so no
   correctness-adjacent judgement is taken per draw.  Draws that close their reasoning
   naturally keep the single-call path exactly as it was.

Report *parsing* is not here: this module never looks at report tags.  It hands WP2's
``ard/report.py`` the post-``</think>`` text and the token ids.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Final

from agents_scaling.ard import contract as K
from agents_scaling.ard.decoding import (
    ARD_DECODING_ROLE,
    CONTINUATION_DECODING,
    CONTINUATION_PREFILL_TEXT,
    CONTINUATION_ROLE,
    budget_of,
    decoding_for_purpose,
)
from agents_scaling.serving.context import (
    ContextCapacityError,
    ContextPreflight,
    preflight_chat_context,
)
from agents_scaling.serving.profiles import ServingProfile, get_serving_profile
from agents_scaling.study import identity
from agents_scaling.study.inference.client import (
    COMPLETED_FINISH_REASONS,
    EndpointPool,
    VllmChatClient,
)
# The completions route must validate its response exactly as the frozen chat route does
# (§4.3 token-id equality, §10.4 abort-is-exogenous), so it reuses those three helpers
# rather than growing a second, drifting copy of them.
from agents_scaling.study.inference.client import (  # noqa: PLC2701
    _AbortedCompletion,
    _extension,
    _int_list,
)
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.resources.oracle import (
    FlopOracle,
    make_cost_fn,
    oracles_from_checkpoints,
)
from agents_scaling.study.resources.profile import oracle_tables as _study_oracle_tables
from agents_scaling.study.types import (
    Checkpoint,
    ProtocolError,
    RequestRecord,
    RequestSpec,
    SeedKey,
)

__all__ = [
    "ArdChatClient",
    "ArdCompletionClient",
    "ContextTooLong",
    "ContinuationOutcome",
    "ContinuationSpec",
    "DecisionContext",
    "DecisionRecord",
    "DecisionRequest",
    "EndpointAffinity",
    "bind_context",
    "continuation_prefill_ids",
    "continuation_prefix_ids",
    "continuation_seed_key",
    "continuation_spec",
    "continue_decision",
    "cost_fn_for",
    "decision_from_record",
    "default_context",
    "generate_decision",
    "oracle_for",
    "oracle_tables",
    "preflight_continuation",
    "preflight_decision",
    "request_spec",
    "top_logprobs_for",
    "verify_oracle_hash",
]


# --------------------------------------------------------------------------- errors


class ContextTooLong(ProtocolError):
    """The rendered prompt plus ``max_tokens`` plus the reserve exceeds the profile.

    Raised at build time, before any HTTP request.  Carries the auditable
    :class:`~agents_scaling.serving.context.ContextPreflight` accounting and the
    ``CONTEXT_FAILURE`` parse status the record must be labelled with (spec §6.3).
    """

    parse_status = K.ParseStatus.CONTEXT_FAILURE

    def __init__(self, preflight: ContextPreflight, cause: ContextCapacityError | None = None) -> None:
        self.preflight = preflight
        super().__init__(
            "ARD36 prompt refused at build time (spec §6.3 context_failure): "
            f"profile={preflight.profile_name!r} prompt={preflight.prompt_tokens} "
            f"max_tokens={preflight.requested_output_tokens} reserve={preflight.reserve_tokens} "
            f"required={preflight.required_tokens} > max_model_len={preflight.served_context}"
        )
        self.__cause__ = cause


# --------------------------------------------------------------------------- frozen interface


@dataclass(frozen=True)
class DecisionRequest:
    """One logical decision (contract §2).

    ``checkpoint`` is ``"8B"`` / ``"32B"`` (``contract.CHECKPOINT_SMALL`` / ``_LARGE``);
    the :class:`DecisionContext` resolves it to a pinned
    :class:`~agents_scaling.study.types.Checkpoint`.  ``thinking_tokens`` is the *arm's*
    budget; a root purpose ignores it and uses the frozen root decoding so the bank
    aliases (see :func:`agents_scaling.ard.decoding.decoding_for_purpose`).
    """

    messages: tuple[Mapping[str, Any], ...]
    checkpoint: str
    thinking_tokens: int
    seed_key: SeedKey
    purpose: str

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a decision needs at least one chat message")
        if self.purpose != self.seed_key.purpose:
            raise ValueError(
                f"purpose {self.purpose!r} disagrees with seed_key.purpose "
                f"{self.seed_key.purpose!r}; the seed key is part of the identity"
            )


@dataclass(frozen=True)
class ContinuationOutcome:
    """What phase 2 of one decision cost, and where it ran (plan §5.2, §5.8).

    Attached to the merged :class:`DecisionRecord` so a decision that was executed as two
    physical requests can be audited as such: its own request id, the prefix it re-sent,
    whether the prefix landed back on the endpoint that had just prefilled it (``pinned``)
    or had to be paid again elsewhere (``reprefilled``), and its own FLOPs — which are
    already *inside* the merged record's summed ``flops``, and are kept here only so the
    split can be recovered.
    """

    request_id: str
    prefix_tokens: int
    prefill_token_ids: tuple[int, ...]
    generated_tokens: int
    endpoint_id: str
    #: ``None`` when nothing was issued (the continuation was already committed).
    endpoint_pinned: bool | None
    reprefilled: bool
    cached_prefill_tokens: int
    finish_reason: str
    latency_s: float
    flops: dict
    aliased: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefix_tokens": self.prefix_tokens,
            "prefill_token_ids": list(self.prefill_token_ids),
            "generated_tokens": self.generated_tokens,
            "endpoint_id": self.endpoint_id,
            "endpoint_pinned": self.endpoint_pinned,
            "reprefilled": self.reprefilled,
            "cached_prefill_tokens": self.cached_prefill_tokens,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
            "flops": dict(self.flops),
            "aliased": self.aliased,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """The measured outcome of one decision (contract §2, plan §5.8 decision grain).

    ``boundary_injected`` and ``cap_hit`` are different events.  ``boundary_injected`` is
    the *thinking* budget: the sampler forced ``</think>`` because the thinking count
    reached ``budget_tokens``, and the 256-token report allowance is untouched.
    ``cap_hit`` is the *whole-response* cap: ``finish_reason == "length"``, so generation
    stopped at ``max_tokens`` and the report span may be cut (WP2 maps that to
    ``TRUNCATED``).  A forced boundary alone leaves ``cap_hit`` false.

    Beyond the frozen contract fields this carries ``budget_tokens`` (the effective
    thinking budget, which a root does not take from the arm), ``think_status`` (the
    delimiter-level :class:`~agents_scaling.ard.contract.ParseStatus`, ``None`` when the
    thinking channel is well formed and WP2's report parser decides the status) and
    ``aliased`` (the record was already committed; no GPU work was spent).

    A *continued* decision (``boundary_injected``, so the report came from the phase-2
    continuation) is still exactly one record: the thinking side is phase 1's and the
    report side — ``final_tokens``, ``final_text``, ``answer_logprob``, ``finish_reason``,
    ``cap_hit`` — is phase 2's, because that is the report that was delivered.  ``flops``
    is the sum over both physical requests, ``physical_request_ids`` names them, and
    :attr:`continuation` holds everything else phase 2 measured.  ``aliased`` is true only
    when *neither* half was generated.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    thinking_tokens: int
    final_tokens: int
    boundary_injected: bool
    cap_hit: bool
    final_overrun: bool
    final_text: str
    answer_logprob: float | None
    prompt_tokens: int
    cached_prefill_tokens: int
    finish_reason: str
    latency_s: float
    endpoint_id: str
    flops: dict
    budget_tokens: int
    think_status: K.ParseStatus | None = None
    aliased: bool = False
    #: Top-k alternatives over the final span, populated only when the request asked
    #: for more than one (the receiver purpose).  Spec §11's answer-logit margin is
    #: uncomputable without them, and would otherwise be missing for every E3 decision.
    #: Rows are indexed by the *final span* (see :attr:`final_text`), so on a two-phase
    #: decision the harness-prefilled ``<final>`` leads with an empty row: those ids were
    #: prompt, not generation, and carry no alternatives.
    final_top_logprobs: tuple[tuple[dict[str, Any], ...], ...] | None = None
    #: Every physical request this logical decision was executed as, in order.  One id on
    #: the single-call path; phase 1 then phase 2 on a continued decision.  Cost
    #: accounting must never count one request where two were issued (plan §5.8).
    physical_request_ids: tuple[str, ...] = ()
    #: Phase-2 accounting, ``None`` on the single-call path.
    continuation: ContinuationOutcome | None = None

    def __post_init__(self) -> None:
        if not self.physical_request_ids:
            object.__setattr__(self, "physical_request_ids", (self.request_id,))

    @property
    def valid_delimiters(self) -> bool:
        """The completion opened and closed exactly one thinking channel."""
        return self.think_status is None

    @property
    def continued(self) -> bool:
        """Was the report taken from a phase-2 continuation (spec §6.2 under a forced boundary)?"""
        return self.continuation is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready row for ``decisions.parquet`` (WP4 compaction)."""
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "thinking_tokens": self.thinking_tokens,
            "final_tokens": self.final_tokens,
            "budget_tokens": self.budget_tokens,
            "boundary_injected": self.boundary_injected,
            "cap_hit": self.cap_hit,
            "final_overrun": self.final_overrun,
            "final_text": self.final_text,
            "answer_logprob": self.answer_logprob,
            "final_top_logprobs": (None if self.final_top_logprobs is None else
                                   [list(row) for row in self.final_top_logprobs]),
            "prompt_tokens": self.prompt_tokens,
            "cached_prefill_tokens": self.cached_prefill_tokens,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
            "endpoint_id": self.endpoint_id,
            "flops": dict(self.flops),
            "think_status": None if self.think_status is None else self.think_status.value,
            "aliased": self.aliased,
            "physical_request_ids": list(self.physical_request_ids),
            "continuation": None if self.continuation is None else self.continuation.to_dict(),
        }


# --------------------------------------------------------------------------- context


@dataclass(frozen=True)
class DecisionContext:
    """Everything a decision needs beyond the request itself.

    The frozen ``generate_decision(req, *, store, pool, cost_fn)`` signature carries no
    way to turn ``"8B"`` into a pinned checkpoint, a tokenizer or the study seed, so those
    live here.  Pass it explicitly, or install one process-wide with :func:`bind_context`
    (WP4's runner does this once per cell) and call the frozen signature unchanged.
    """

    checkpoints: Mapping[str, Checkpoint]
    tokenizers: Mapping[str, Any]
    study_seed_hex: str
    study_id: str = K.STUDY_ID
    run_id: str | None = None
    hf_home: str | os.PathLike | None = None

    def checkpoint(self, size: str) -> Checkpoint:
        try:
            return self.checkpoints[str(size)]
        except KeyError as exc:
            raise ProtocolError(
                f"unknown ARD36 checkpoint {size!r}; declared: {sorted(self.checkpoints)}"
            ) from exc

    def tokenizer(self, size: str) -> Any:
        try:
            return self.tokenizers[str(size)]
        except KeyError as exc:
            raise ProtocolError(
                f"no tokenizer for checkpoint {size!r}; declared: {sorted(self.tokenizers)}"
            ) from exc

    def profile(self, size: str) -> ServingProfile:
        return get_serving_profile(self.checkpoint(size).profile)


_BOUND: dict[str, DecisionContext] = {}


def bind_context(context: DecisionContext | None) -> DecisionContext | None:
    """Install (or clear with ``None``) the process-wide default context; returns the old one."""
    previous = _BOUND.get("context")
    if context is None:
        _BOUND.pop("context", None)
    else:
        _BOUND["context"] = context
    return previous


def default_context() -> DecisionContext:
    context = _BOUND.get("context")
    if context is None:
        raise ProtocolError(
            "no ARD36 DecisionContext: pass context=... to generate_decision or install "
            "one with ard.client.bind_context()"
        )
    return context


# --------------------------------------------------------------------------- request shaping


def top_logprobs_for(purpose: str) -> int:
    """Logprob width of one purpose (spec §14: the receiver needs the answer-logit margin)."""
    if str(purpose) == K.PURPOSE_RECEIVER:
        return int(K.RECEIVER_TOP_LOGPROBS)
    return 1


def request_spec(req: DecisionRequest, context: DecisionContext) -> RequestSpec:
    """The identity-bearing :class:`RequestSpec` of ``req`` (§18 cache identity)."""
    checkpoint = context.checkpoint(req.checkpoint)
    if req.seed_key.model_cell != checkpoint.model_cell:
        raise ProtocolError(
            f"seed_key.model_cell {req.seed_key.model_cell!r} is not the {req.checkpoint} "
            f"checkpoint cell {checkpoint.model_cell!r}"
        )
    decoding = decoding_for_purpose(req.purpose, req.thinking_tokens)
    return RequestSpec(
        messages=tuple(dict(m) for m in req.messages),
        decoding=decoding,
        checkpoint=checkpoint,
        seed_key=req.seed_key,
        role=ARD_DECODING_ROLE,
        study_id=context.study_id,
        study_seed_hex=context.study_seed_hex,
    )


def preflight_decision(
    spec: RequestSpec, context: DecisionContext, *, size: str
) -> ContextPreflight:
    """``len(prompt_ids) + max_tokens + 128 <= profile.max_model_len``, before the request.

    Plan §5.2: a vLLM 400 is a non-retriable ``InfraFailure`` that would kill a cell
    mid-run, so the refusal happens here, in the harness, with auditable accounting.
    """
    profile = context.profile(size)
    try:
        return preflight_chat_context(
            context.tokenizer(size),
            list(spec.messages),
            requested_output_tokens=int(spec.decoding.max_tokens),
            served_context=int(profile.max_model_len),
            profile_name=profile.name,
            enable_thinking=bool(spec.decoding.enable_thinking),
            reserve_tokens=int(K.CONTEXT_RESERVE_TOKENS),
        )
    except ContextCapacityError as exc:
        raise ContextTooLong(exc.preflight, exc) from exc


class ArdChatClient(VllmChatClient):
    """``VllmChatClient`` plus the ARD wire fields (spec §6.2, plan §5.2).

    ``request_kwargs`` and ``_parse_response`` are the only overrides; the retry policy,
    the prompt-token-id equality check and the record assembly are the frozen study ones.
    Everything added to the wire is derived from the spec, so nothing reaches vLLM that is
    not already inside the request id.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._continuation: ArdCompletionClient | None = None
        self._continuation_lock = threading.Lock()

    def continuation(self) -> "ArdCompletionClient":
        """This client's phase-2 client, built once and then reused.

        One client per continuation would mean one OpenAI SDK client — and its connection
        pool — per cap-hit draw, i.e. tens of thousands of them over a run.  The cell's
        continuations therefore share one client, whose pool is wrapped in an
        :class:`EndpointAffinity` so each continuation can still prefer the endpoint that
        served its own phase 1.
        """
        with self._continuation_lock:
            if self._continuation is None:
                self._continuation = ArdCompletionClient(
                    EndpointAffinity(self.pool),
                    self.tokenizer,
                    request_timeout_s=self.request_timeout_s,
                    max_exogenous_retries=self.max_exogenous_retries,
                    cost_fn=self.cost_fn,
                    run_id=self.run_id,
                    clock=self._clock,
                )
            return self._continuation

    @staticmethod
    def request_kwargs(
        spec: RequestSpec, guided_json_schema: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        kwargs = VllmChatClient.request_kwargs(spec, guided_json_schema)
        budget = budget_of(spec.decoding)
        if budget <= 0:
            raise ProtocolError(
                "an ARD36 request must carry a thinking budget; "
                f"decoding {spec.decoding.hash[:12]} has thinking_tokens={budget}"
            )
        expected = 1 + budget + 1 + int(spec.decoding.final_tokens)
        if int(spec.decoding.max_tokens) != expected:
            raise ProtocolError(
                f"max_tokens {spec.decoding.max_tokens} != 1 + {budget} + 1 + "
                f"{spec.decoding.final_tokens} (spec §6.2)"
            )
        # The budget is an engine argument, so it is already inside decoding_hash.
        kwargs["extra_body"]["thinking_token_budget"] = budget
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs_for(spec.seed_key.purpose)
        return kwargs

    def _parse_response(self, response: Any, local_ids: tuple[int, ...]) -> dict[str, Any]:
        parsed = super()._parse_response(response, local_ids)
        choice = response.choices[0]
        parsed["token_logprobs"] = _token_logprobs(choice)
        parsed["top_logprobs"] = _top_logprobs(choice)
        parsed["cached_prefill_tokens"] = _cached_tokens(getattr(response, "usage", None))
        return parsed

    def _build_record(self, spec: RequestSpec, parsed: Mapping[str, Any], **kwargs: Any) -> RequestRecord:
        record = super()._build_record(spec, parsed, **kwargs)
        if record.flops.get("total") is None:
            # Refuse here, before ``store.publish``: a committed placeholder record would
            # alias forever and the run would end with no FLOP number at all (contract §2).
            raise ProtocolError(
                f"request {record.request_id[:12]} would commit the FLOP placeholder; "
                "attach FlopOracle.from_checkpoint + make_cost_fn (cost_fn_for)"
            )
        return record


# --------------------------------------------------------------------------- response accounting


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _token_logprobs(choice: Any) -> list[float | None] | None:
    """Per-token logprobs of ``choices[0].logprobs.content``, or ``None`` when absent.

    Non-finite values (``-inf`` for an impossible token) become ``None``: the record is
    written through a JSON encoder with ``allow_nan=False`` and the store refuses
    non-finite constants on read-back.
    """
    block = getattr(choice, "logprobs", None)
    if block is None:
        return None
    content = getattr(block, "content", None)
    if content is None and isinstance(block, Mapping):
        content = block.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    out: list[float | None] = []
    for entry in content:
        value = entry.get("logprob") if isinstance(entry, Mapping) else getattr(entry, "logprob", None)
        out.append(_finite(value))
    return out


def _top_logprobs(choice: Any) -> list[list[dict[str, Any]]] | None:
    """Per-token top-k alternatives of ``choices[0].logprobs.content[i].top_logprobs``.

    Needed only by the receiver assay, whose answer-logit margin (spec §11) is the logprob
    difference between the two supplied candidates' first differing answer token.  Without
    the alternatives that margin is uncomputable and would have to be recorded as missing
    for every receiver decision.  Kept over the final span only (see
    :func:`decision_from_record`), so an ordinary decision carries nothing extra.
    """
    block = getattr(choice, "logprobs", None)
    if block is None:
        return None
    content = getattr(block, "content", None)
    if content is None and isinstance(block, Mapping):
        content = block.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    out: list[list[dict[str, Any]]] = []
    for entry in content:
        alts = (entry.get("top_logprobs") if isinstance(entry, Mapping)
                else getattr(entry, "top_logprobs", None)) or ()
        row: list[dict[str, Any]] = []
        for alt in alts:
            token = alt.get("token") if isinstance(alt, Mapping) else getattr(alt, "token", None)
            value = _finite(alt.get("logprob") if isinstance(alt, Mapping)
                            else getattr(alt, "logprob", None))
            if token is not None and value is not None:
                row.append({"token": str(token), "logprob": float(value)})
        out.append(row)
    return out


def _cached_tokens(usage: Any) -> int:
    """``usage.prompt_tokens_details.cached_tokens`` — vLLM's prefix-cache hit count."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    value = (
        details.get("cached_tokens")
        if isinstance(details, Mapping)
        else getattr(details, "cached_tokens", None)
    )
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _final_span_logprobs(
    token_logprobs: Sequence[float | None] | None,
    *,
    n_completion: int,
    start: int,
    stop: int,
) -> float | None:
    """Mean logprob over completion ids ``[start, stop)``.

    vLLM returns one ``logprobs.content`` entry per generated token, so the list aligns
    with ``token_ids`` by index.  A server that reports only the post-reasoning channel is
    tolerated by matching the span length; anything else yields ``None`` rather than a
    silently misaligned number.
    """
    if not token_logprobs or stop <= start:
        return None
    n = len(token_logprobs)
    if n == n_completion:
        return _mean(token_logprobs[start:stop])
    final_len = n_completion - start
    if n == final_len:
        return _mean(token_logprobs[: stop - start])
    return None


def decision_from_record(
    record: RequestRecord,
    *,
    budget_tokens: int,
    tokenizer: Any,
    aliased: bool = False,
) -> DecisionRecord:
    """Delimiter accounting over a committed record (spec §6.2 post-response rules).

    Works identically on a freshly generated record and on one read back from the store,
    so a resumed cell measures exactly what the original run measured.
    """
    response = record.response
    ids = tuple(int(i) for i in response.get("token_ids") or ())
    finish_reason = str(response.get("finish_reason"))
    flops = dict(record.flops or {})
    if flops.get("total") is None:
        raise ProtocolError(
            f"request {record.request_id[:12]} carries the FLOP placeholder; attach "
            "FlopOracle.from_checkpoint + make_cost_fn (contract §2 WP1)"
        )

    think_status: K.ParseStatus | None = None
    thinking_tokens = 0
    final_ids: tuple[int, ...] = ()
    if not ids or ids[0] != K.THINK_START_ID:
        # The think-on template ends at ``assistant\n``; the model emits ``<think>`` itself.
        think_status = K.ParseStatus.NO_THINK_OPEN
    else:
        try:
            end = ids.index(K.THINK_END_ID, 1)
        except ValueError:
            think_status = K.ParseStatus.PREMATURE_REASONING_EOS
            thinking_tokens = len(ids) - 1
        else:
            thinking_tokens = end - 1
            final_ids = ids[end + 1 :]
            if final_ids and final_ids[-1] == K.IM_END_ID:
                # The terminating turn token is not part of the report span.
                final_ids = final_ids[:-1]

    boundary_injected = bool(think_status is None and thinking_tokens == int(budget_tokens))
    kept = final_ids[: K.FINAL_TOKENS]
    final_overrun = len(final_ids) > K.FINAL_TOKENS
    final_text = tokenizer.decode(list(kept)) if kept else ""
    if not isinstance(final_text, str):
        raise ProtocolError("tokenizer.decode did not return text")

    start = len(ids) - len(final_ids) if final_ids else len(ids)
    answer_logprob = _final_span_logprobs(
        response.get("token_logprobs"),
        n_completion=len(ids),
        start=start,
        stop=start + len(kept),
    )

    top_block = response.get("top_logprobs")
    final_top = None
    if top_block and len(kept):
        width = max((len(row) for row in top_block), default=0)
        if width > 1:                     # only the receiver purpose asks for more than one
            n = len(ids)
            lo = start if len(top_block) == n else max(0, len(top_block) - (n - start))
            final_top = tuple(tuple(row) for row in top_block[lo:lo + len(kept)])

    endpoint = record.endpoint or {}
    return DecisionRecord(
        request_id=record.request_id,
        prompt_token_ids=tuple(int(i) for i in record.prompt_token_ids),
        completion_token_ids=ids,
        thinking_tokens=thinking_tokens,
        final_tokens=len(final_ids),
        boundary_injected=boundary_injected,
        cap_hit=finish_reason == "length",
        final_overrun=final_overrun,
        final_text=final_text,
        answer_logprob=answer_logprob,
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(response.get("cached_prefill_tokens") or 0),
        finish_reason=finish_reason,
        latency_s=float((record.timing or {}).get("latency_s") or 0.0),
        endpoint_id=f"{endpoint.get('host')}:{endpoint.get('port')}",
        flops=flops,
        budget_tokens=int(budget_tokens),
        think_status=think_status,
        final_top_logprobs=final_top,
        aliased=bool(aliased),
        physical_request_ids=(record.request_id,),
    )


# --------------------------------------------------------------------------- phase 2: identity


#: Domain tag folded into a phase-2 request id.  A continuation is not a decision and must
#: never alias one even if every frozen identity component happened to agree.
CONTINUATION_ID_TAG: Final[str] = "ard36/continuation/v1"

#: Suffix appended to the decision's seed-key namespace to derive the continuation's seed.
#: Phase 2 is a different request with a different prompt, so reusing the decision's engine
#: seed would be an accident waiting to be explained; deriving it keeps the pair
#: deterministic, so a resumed cell re-derives exactly the ids the original run committed.
CONTINUATION_NAMESPACE_SUFFIX: Final[str] = "|CONTINUATION"


@dataclass(frozen=True)
class ContinuationSpec(RequestSpec):
    """A phase-2 request: the prompt is an exact token prefix, not chat messages.

    ``messages`` is deliberately empty — there is no chat template to render, and a
    synthesised message list would be a fiction that the §4.3 prompt-token-id check would
    then have to work around.  The prompt's identity is carried instead by
    :attr:`prefix_hash`, a tenth component folded into :attr:`request_id` on top of the
    frozen nine of ``study/identity.request_id``.  Contract §0.3 still holds exactly:
    everything sent to the engine is inside the id — the token prefix through
    ``prefix_hash``, the sampler and the report allowance through ``decoding_hash``, the
    pins and the seed through the frozen components.
    """

    prompt_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ProtocolError("a continuation spec needs its phase-1 token prefix")
        if self.messages:
            raise ProtocolError("a continuation carries a token prefix, not chat messages")

    @property
    def prefix_hash(self) -> str:
        """SHA-256 of JCS(prompt token ids) — the prompt-bytes identity of a phase-2 request."""
        return identity.sha256_hex(identity.jcs([int(i) for i in self.prompt_ids]))

    def identity_fields(self) -> dict[str, Any]:
        return {**super().identity_fields(), "prefix_hash": self.prefix_hash}

    @property
    def request_id(self) -> str:
        """The frozen nine-component id, re-hashed with the token prefix and a domain tag."""
        fields = dict(self.identity_fields())
        prefix_hash = str(fields.pop("prefix_hash"))
        return identity.sha256_hex(
            identity.jcs([identity.request_id(**fields), CONTINUATION_ID_TAG, prefix_hash])
        )


def continuation_seed_key(seed_key: SeedKey) -> SeedKey:
    """The decision's seed key in the continuation namespace (spec §18 cache identity)."""
    return replace(seed_key, namespace=f"{seed_key.namespace}{CONTINUATION_NAMESPACE_SUFFIX}")


def continuation_prefill_ids(tokenizer: Any) -> tuple[int, ...]:
    """Token ids of the prefilled report opening (``\\n\\n<final>``; plan §5.2).

    Prefilling the opening tag into the *prompt* is the part that works: with the prefix
    ending at the forced ``</think>`` and nothing more, the model resumes its derivation
    exactly as it does in phase 1.  Committed to the opening tag, it writes the answer
    first.  ``add_special_tokens=False`` because the prefix already carries every control
    token the template produced.
    """
    ids = tuple(int(i) for i in tokenizer.encode(CONTINUATION_PREFILL_TEXT, add_special_tokens=False))
    if not ids:
        raise ProtocolError(f"tokenizer produced no ids for {CONTINUATION_PREFILL_TEXT!r}")
    decoded = tokenizer.decode(list(ids))
    if not isinstance(decoded, str) or K.FINAL_OPEN not in decoded:
        raise ProtocolError(
            f"the prefilled report opening does not round-trip through the tokenizer: "
            f"{CONTINUATION_PREFILL_TEXT!r} -> {ids} -> {decoded!r}"
        )
    return ids


def continuation_prefix_ids(record: RequestRecord, prefill_ids: Sequence[int]) -> tuple[int, ...]:
    """The exact phase-2 prompt: phase-1 prompt + completion through ``</think>`` + prefill.

    Token ids, never re-rendered text: the continuation must resume the *same* sequence the
    engine already has in its prefix cache, and a decode/re-encode round trip is not
    guaranteed to reproduce it.
    """
    ids = tuple(int(i) for i in record.response.get("token_ids") or ())
    try:
        end = ids.index(K.THINK_END_ID, 1)
    except ValueError as exc:
        raise ProtocolError(
            f"request {record.request_id[:12]} has no {K.THINK_END_ID} to continue from"
        ) from exc
    return tuple(int(i) for i in record.prompt_token_ids) + ids[: end + 1] + tuple(int(i) for i in prefill_ids)


def continuation_spec(
    spec: RequestSpec, record: RequestRecord, *, prefill_ids: Sequence[int]
) -> ContinuationSpec:
    """The identity-bearing spec of the phase-2 request of one cap-hit decision."""
    return ContinuationSpec(
        messages=(),
        decoding=CONTINUATION_DECODING,
        checkpoint=spec.checkpoint,
        seed_key=continuation_seed_key(spec.seed_key),
        role=CONTINUATION_ROLE,
        study_id=spec.study_id,
        study_seed_hex=spec.study_seed_hex,
        prompt_ids=continuation_prefix_ids(record, prefill_ids),
    )


def preflight_continuation(
    spec: ContinuationSpec, context: DecisionContext, *, size: str
) -> ContextPreflight:
    """``len(prefix) + FINAL_TOKENS + 128 <= profile.max_model_len``, before the request.

    The prefix is a decision's whole prompt *plus* its whole thinking span, so unlike the
    phase-1 preflight this one is not implied by the frozen prompt budget and is checked
    on its own.
    """
    profile = context.profile(size)
    preflight = ContextPreflight(
        profile_name=profile.name,
        prompt_tokens=len(spec.prompt_ids),
        requested_output_tokens=int(spec.decoding.max_tokens),
        reserve_tokens=int(K.CONTEXT_RESERVE_TOKENS),
        served_context=int(profile.max_model_len),
    )
    if not preflight.fits:
        raise ContextTooLong(preflight)
    return preflight


# --------------------------------------------------------------------------- phase 2: endpoints


class EndpointAffinity:
    """An :class:`EndpointPool` façade that hands out one preferred endpoint first.

    A continuation re-sends the decision's prompt *plus* its whole thinking span — about
    1.3k tokens — so it is worth landing back on the server that has just prefilled them:
    vLLM's prefix cache is on by default (block 16, contract §3) and a miss pays the
    prefill twice.  The preference is per thread, because one decision is generated by one
    worker thread, and is consumed by the first :meth:`pick`; every retry after it goes
    back to the pool, so the frozen §10.4 rotation and cooldown are untouched.
    :meth:`outcome` reports whether the pin was honoured, and whether the prefix had to be
    re-prefilled somewhere else.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._state = threading.local()

    def prefer(self, host: Any = None, port: Any = None) -> None:
        """Prefer ``host:port`` for this thread's next :meth:`pick` (``None`` clears it)."""
        state = self._state
        state.want = None if host is None or port is None else (str(host), int(port))
        state.used = False
        state.pinned = False
        state.fell_back = False

    def outcome(self) -> tuple[bool, bool]:
        """``(pinned, reprefilled)`` of the request just issued on this thread."""
        state = self._state
        return bool(getattr(state, "pinned", False)), bool(getattr(state, "fell_back", False))

    def pick(self) -> Any:
        state = self._state
        want = getattr(state, "want", None)
        if want is not None and not getattr(state, "used", False):
            state.used = True
            for entry in self._pool.endpoints():
                if (str(entry.host), int(entry.port)) == want:
                    state.pinned = True
                    return entry
            # The phase-1 endpoint is gone (preempted, or rotated out): the prefix will be
            # prefilled again wherever the pool sends us.
            state.fell_back = True
        return self._pool.pick()

    def report_success(self, entry: Any) -> Any:
        return self._pool.report_success(entry)

    def report_failure(self, entry: Any) -> Any:
        state = self._state
        if getattr(state, "want", None) == (str(entry.host), int(entry.port)):
            state.pinned, state.fell_back = False, True
        return self._pool.report_failure(entry)

    def __getattr__(self, name: str) -> Any:
        # endpoints(), wait(), profile, shard, ... — everything else is the pool's.
        return getattr(self._pool, name)


# --------------------------------------------------------------------------- phase 2: the client


def _completion_logprobs(choice: Any) -> tuple[list[float | None] | None, list[list[dict[str, Any]]] | None]:
    """``choices[0].logprobs`` of ``/v1/completions``, in the shapes the chat path produces.

    Completions report ``token_logprobs`` (one per generated token) and ``top_logprobs``
    (one ``token -> logprob`` mapping per generated token) where chat reports a ``content``
    list of per-token objects.  Normalising here means :func:`decision_from_record` and the
    merge read one shape, and the alternatives keep a defined order (most probable first,
    ties broken by token) instead of a dict's insertion order.
    """
    block = getattr(choice, "logprobs", None)
    if block is None:
        return None, None

    def field(name: str) -> Any:
        value = getattr(block, name, None)
        if value is None and isinstance(block, Mapping):
            value = block.get(name)
        return value

    def is_list(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    raw = field("token_logprobs")
    values = [_finite(v) for v in raw] if is_list(raw) else None

    raw_rows = field("top_logprobs")
    rows: list[list[dict[str, Any]]] | None = None
    if is_list(raw_rows):
        rows = []
        for entry in raw_rows:
            row: list[dict[str, Any]] = []
            if isinstance(entry, Mapping):
                for token, value in entry.items():
                    logprob = _finite(value)
                    if token is not None and logprob is not None:
                        row.append({"token": str(token), "logprob": float(logprob)})
                row.sort(key=lambda alt: (-alt["logprob"], alt["token"]))
            rows.append(row)
    return values, rows


class ArdCompletionClient(ArdChatClient):
    """Phase 2: ``/v1/completions`` on an exact token prefix (plan §5.2 contingency).

    Everything that governs *how* a request is issued is the frozen study client's: the
    §10.4 two-retry exogenous policy, the endpoint rotation and failure reporting, the
    attempt ledger, the record assembly and :class:`ArdChatClient`'s refusal to commit the
    FLOP placeholder.  Only what a raw-prefix request must change is changed — the prompt
    is the spec's token ids instead of a rendered chat template, the body is a completions
    body (``prompt`` as ``list[int]``, ``logprobs`` as a width rather than a flag), the
    route is ``/v1/completions``, and the response is the completions shape (per-choice
    ``prompt_token_ids``/``token_ids``, ``choices[0].text``, and the ``tokens`` /
    ``token_logprobs`` / ``top_logprobs`` logprob block).  vLLM 0.21 supports all of it:
    ``vllm/entrypoints/openai/completion/protocol.py``.
    """

    def render_prompt_ids(self, spec: RequestSpec) -> tuple[int, ...]:
        """The spec's token prefix — no chat template is rendered for a continuation."""
        ids = tuple(int(i) for i in getattr(spec, "prompt_ids", ()) or ())
        if not ids:
            raise ProtocolError("a continuation spec carries no token prefix")
        return ids

    def chat_template_hash(self, spec: RequestSpec) -> str:
        """No template is applied, so the prefix hash *is* the rendered-bytes hash."""
        return spec.prefix_hash

    @staticmethod
    def request_kwargs(
        spec: RequestSpec, guided_json_schema: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        decoding = spec.decoding
        if guided_json_schema is not None or bool(decoding.guided_json):
            raise ProtocolError("a report continuation is never guided (spec §6.3)")
        if int(decoding.max_tokens) != int(decoding.final_tokens) or int(decoding.final_tokens) < 1:
            raise ProtocolError(
                "a continuation asks for exactly the report allowance and nothing else: "
                f"max_tokens={decoding.max_tokens} final_tokens={decoding.final_tokens} "
                f"(spec §6.3, contract.FINAL_TOKENS={K.FINAL_TOKENS})"
            )
        if budget_of(decoding) != 0:
            raise ProtocolError(
                "a continuation sends no thinking_token_budget: the thinking channel is "
                f"already closed inside the prompt (got {budget_of(decoding)})"
            )
        return {
            "model": spec.checkpoint.served_model_name,
            "prompt": [int(i) for i in spec.prompt_ids],
            "temperature": float(decoding.temperature),
            "max_tokens": int(decoding.max_tokens),
            "seed": int(spec.engine_seed),
            # Completions take the width itself, not a flag plus a width.
            "logprobs": top_logprobs_for(spec.seed_key.purpose),
            "extra_body": {
                "top_k": int(decoding.top_k),
                "top_p": float(decoding.top_p),
                "min_p": float(decoding.min_p),
                "presence_penalty": float(decoding.presence_penalty),
                "repetition_penalty": float(decoding.repetition_penalty),
                "return_token_ids": True,
            },
        }

    def _sdk_client(self, entry: Any) -> Any:
        """The SDK client with ``/v1/completions`` under the ``chat.completions`` name.

        ``VllmChatClient.generate`` owns the retry accounting and calls exactly
        ``self._sdk_client(entry).chat.completions.create(**kwargs)``.  The route is the
        one line a continuation changes, so it is re-pointed here rather than by forking
        the frozen retry loop into a second copy that would then have to be kept in step.
        """
        completions = super()._sdk_client(entry).completions
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def _parse_response(self, response: Any, local_ids: tuple[int, ...]) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ProtocolError(
                f"server returned {0 if not choices else len(choices)} choices, expected exactly 1"
            )
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in COMPLETED_FINISH_REASONS:
            raise _AbortedCompletion(f"finish_reason={finish_reason!r}")

        # /v1/completions carries the prompt ids per choice, not at the top level.
        present, prompt_ids_raw = _extension(choice, "prompt_token_ids")
        if not present or prompt_ids_raw is None:
            present, prompt_ids_raw = _extension(response, "prompt_token_ids")
        if not present or prompt_ids_raw is None:
            raise ProtocolError(
                "completion response lacks prompt_token_ids (return_token_ids unsupported?)"
            )
        prompt_ids = _int_list(prompt_ids_raw, what="prompt_token_ids")
        if tuple(prompt_ids) != tuple(local_ids):
            raise ProtocolError(
                "server prompt_token_ids disagree with the phase-1 token prefix (§4.3): "
                f"local={len(local_ids)} tokens "
                f"sha256={identity.sha256_hex(identity.jcs(list(local_ids)))[:16]}, "
                f"server={len(prompt_ids)} tokens "
                f"sha256={identity.sha256_hex(identity.jcs(prompt_ids))[:16]}"
            )
        present, token_ids_raw = _extension(choice, "token_ids")
        if not present or token_ids_raw is None:
            raise ProtocolError("completion choice lacks token_ids (return_token_ids unsupported?)")
        token_ids = _int_list(token_ids_raw, what="token_ids")

        text = getattr(choice, "text", None)
        if text is not None and not isinstance(text, str):
            raise ProtocolError("completion choice text is neither text nor null")

        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] | None = None
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            for key, ids in (("prompt_tokens", prompt_ids), ("completion_tokens", token_ids)):
                count = usage_dict[key]
                if count is not None and int(count) != len(ids):
                    raise ProtocolError(
                        f"usage.{key}={count} but the server returned {len(ids)} token ids"
                    )

        token_logprobs, top_logprobs = _completion_logprobs(choice)
        return {
            "content": text,
            "reasoning": None,
            "reasoning_field_name": None,
            "token_ids": token_ids,
            "completion_tokens": len(token_ids),
            # The thinking channel is inside the prompt, already closed: a continuation
            # generates report tokens only.
            "reasoning_tokens": 0,
            "reasoning_tokens_source": "continuation",
            "finish_reason": finish_reason,
            "stop_reason": _extension(choice, "stop_reason")[1],
            "usage": usage_dict,
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "prompt_token_ids": prompt_ids,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "cached_prefill_tokens": _cached_tokens(usage),
            "route": "/v1/completions",
        }


# --------------------------------------------------------------------------- phase 2: the merge


def _sum_flops(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict:
    """Both physical requests' FLOPs, summed (contract §2; plan §5.8 ``cost_summary``).

    Cost accounting must not count one request where two were executed.  Both phases run
    the same checkpoint, so they must carry the same oracle hash; a disagreement means the
    two records are not two halves of one decision and the sum would be meaningless.

    The oracle is a pure function of token counts, so the re-sent prefix is counted again
    even when vLLM served it from its prefix cache: the summed number is the work the
    analytic model attributes to the decision, and ``cached_prefill_tokens`` records how
    much of it the engine actually skipped.
    """
    oracle, other = first.get("oracle"), second.get("oracle")
    if oracle != other:
        raise ProtocolError(
            f"phase-1 oracle {str(oracle)[:12]} != phase-2 oracle {str(other)[:12]}: "
            "the two requests of one decision must share the checkpoint's oracle"
        )
    summed: dict[str, Any] = {"oracle": oracle}
    for key in ("prefill", "decode", "total"):
        left, right = first.get(key), second.get(key)
        if left is None or right is None:
            raise ProtocolError(f"a physical request carries the FLOP placeholder for {key!r}")
        summed[key] = int(left) + int(right)
    return summed


def _continuation_client(
    client: Any,
    pool: EndpointPool,
    tokenizer: Any,
    *,
    cost_fn: Callable[..., Mapping[str, Any]],
    context: DecisionContext,
) -> ArdCompletionClient:
    """The phase-2 client: the phase-1 client's own, or a fresh one for an injected client."""
    maker = getattr(client, "continuation", None)
    if callable(maker):
        return maker()
    return ArdCompletionClient(
        EndpointAffinity(pool), tokenizer, cost_fn=cost_fn, run_id=context.run_id
    )


def _merged_decision(
    first: DecisionRecord,
    record: RequestRecord,
    *,
    spec: ContinuationSpec,
    prefill_ids: Sequence[int],
    tokenizer: Any,
    endpoint_pinned: bool | None,
    reprefilled: bool,
    aliased: bool,
) -> DecisionRecord:
    """One :class:`DecisionRecord` for a decision executed as two physical requests.

    The thinking side is phase 1's and the report side is phase 2's: ``thinking_tokens``,
    ``budget_tokens``, ``prompt_tokens``, ``prompt_token_ids`` and the logical
    ``request_id`` come from the decision; ``final_tokens`` (the tokens phase 2 actually
    *generated* — the prefilled opening was prompt, not generation), ``final_text``,
    ``answer_logprob``, ``finish_reason``, ``cap_hit`` and ``final_overrun`` describe the
    report that was delivered, because that is the one WP2 parses and votes on.
    ``completion_token_ids`` is the logical completion — phase 1 through the forced
    ``</think>``, then the prefilled opening, then phase 2's report — so the delimiter
    accounting re-derived from it agrees with the fields; phase 1's discarded post-boundary
    span is not in it and is preserved in the phase-1 store record.

    ``flops`` is the only field summed across the two, because cost is the one quantity
    that is genuinely additive; every other per-request measurement of phase 2 is kept in
    :class:`ContinuationOutcome` rather than blended into a field whose meaning is tied to
    phase 1's prompt.  ``latency_s`` is the exception it has to be: it is the wall time of
    the decision, which is both calls.
    """
    prefix = tuple(int(i) for i in spec.prompt_ids)
    prefill = tuple(int(i) for i in prefill_ids)
    if prefix[: len(first.prompt_token_ids)] != tuple(first.prompt_token_ids):
        raise ProtocolError("the continuation prefix does not start with the decision's prompt")
    if prefill and prefix[-len(prefill):] != prefill:
        raise ProtocolError("the continuation prefix does not end with the prefilled report opening")
    response = record.response
    ids = tuple(int(i) for i in response.get("token_ids") or ())
    # The terminating turn token is not part of the report span (as in phase 1).
    generated = ids[:-1] if ids and ids[-1] == K.IM_END_ID else ids
    # A full 256 generated tokens plus the prefilled opening is the one case where the span
    # exceeds FINAL_TOKENS; it is also exactly the case where phase 2 stopped on the cap, so
    # WP2 reads it as TRUNCATED either way (``report.span_cut``).
    kept = generated[: K.FINAL_TOKENS]
    final_ids = prefill + kept
    final_text = tokenizer.decode(list(final_ids)) if final_ids else ""
    if not isinstance(final_text, str):
        raise ProtocolError("tokenizer.decode did not return text")

    answer_logprob = _final_span_logprobs(
        response.get("token_logprobs"), n_completion=len(ids), start=0, stop=len(kept)
    )
    top_block = response.get("top_logprobs")
    final_top = None
    if top_block and kept:
        width = max((len(row) for row in top_block), default=0)
        if width > 1:                     # only the receiver purpose asks for more than one
            rows = [tuple(row) for row in top_block[: len(kept)]]
            # The prefilled ids were prompt: they have no alternatives, but the rows stay
            # indexed by the final span.
            final_top = tuple([()] * len(prefill) + rows)

    endpoint = record.endpoint or {}
    endpoint_id = f"{endpoint.get('host')}:{endpoint.get('port')}"
    latency = float((record.timing or {}).get("latency_s") or 0.0)
    finish_reason = str(response.get("finish_reason"))
    flops = _sum_flops(first.flops, dict(record.flops or {}))
    outcome = ContinuationOutcome(
        request_id=record.request_id,
        prefix_tokens=len(prefix),
        prefill_token_ids=prefill,
        generated_tokens=len(generated),
        endpoint_id=endpoint_id,
        endpoint_pinned=endpoint_pinned,
        reprefilled=bool(reprefilled),
        cached_prefill_tokens=int(response.get("cached_prefill_tokens") or 0),
        finish_reason=finish_reason,
        latency_s=latency,
        flops=dict(record.flops or {}),
        aliased=bool(aliased),
    )
    return DecisionRecord(
        request_id=first.request_id,
        prompt_token_ids=first.prompt_token_ids,
        completion_token_ids=prefix[len(first.prompt_token_ids):] + generated,
        thinking_tokens=first.thinking_tokens,
        final_tokens=len(generated),
        boundary_injected=True,
        cap_hit=finish_reason == "length",
        final_overrun=len(generated) > K.FINAL_TOKENS,
        final_text=final_text,
        answer_logprob=answer_logprob,
        prompt_tokens=first.prompt_tokens,
        # Phase 1's, not the sum: ``prompt_tokens`` is phase 1's prompt, and a consumer
        # computing "prefill actually paid" as prompt_tokens - cached_prefill_tokens must
        # not be handed a cache count from a prompt it is not counting (a phase-2 prefix is
        # usually a near-total cache hit and would drive that difference to zero).
        # Phase 2's own count is in ``continuation.cached_prefill_tokens``.
        cached_prefill_tokens=first.cached_prefill_tokens,
        finish_reason=finish_reason,
        latency_s=first.latency_s + latency,
        endpoint_id=first.endpoint_id,
        flops=flops,
        budget_tokens=first.budget_tokens,
        think_status=first.think_status,
        final_top_logprobs=final_top,
        aliased=bool(first.aliased and aliased),
        physical_request_ids=(first.request_id, record.request_id),
        continuation=outcome,
    )


def continue_decision(
    first: DecisionRecord,
    record: RequestRecord,
    *,
    spec: RequestSpec,
    size: str,
    context: DecisionContext,
    store: RequestStore,
    pool: EndpointPool,
    cost_fn: Callable[..., Mapping[str, Any]],
    client: VllmChatClient | None = None,
    cell_id: str | None = None,
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> DecisionRecord:
    """Phase 2 of one cap-hit decision: the report, continued from the exact token prefix.

    Called by :func:`generate_decision` whenever — and only whenever — phase 1's boundary
    was injected.  The trigger is deliberately blind to what phase 1's report span
    contained: a rule that continued only the draws whose report failed to parse would be
    a per-draw judgement correlated with difficulty, and the uniform rule is also the
    faithful reading of §6.2's independent report allowance, which under a forced boundary
    is exactly this continuation.

    Phase 2 commits as its own store record (the store has no transaction spanning two),
    so a cell that dies between the two phases resumes by aliasing phase 1 and issuing only
    the continuation.
    """
    if not first.boundary_injected:
        raise ProtocolError(
            f"request {first.request_id[:12]} closed its reasoning naturally; "
            "phase 2 runs only under a forced boundary (spec §6.2)"
        )
    tokenizer = context.tokenizer(size)
    prefill = continuation_prefill_ids(tokenizer)
    spec2 = continuation_spec(spec, record, prefill_ids=prefill)
    preflight_continuation(spec2, context, size=size)   # raises ContextTooLong first
    generator = _continuation_client(client, pool, tokenizer, cost_fn=cost_fn, context=context)
    affinity = generator.pool if isinstance(generator.pool, EndpointAffinity) else None
    if affinity is not None:
        endpoint = record.endpoint or {}
        affinity.prefer(endpoint.get("host"), endpoint.get("port"))
    committed, aliased = store.get_or_generate(spec2, generator, cell_id, on_event=on_event)
    pinned: bool | None = None
    reprefilled = False
    if not aliased and affinity is not None:
        pinned, reprefilled = affinity.outcome()
    if on_event is not None:
        on_event(
            "continuation",
            {
                "request_id": first.request_id,
                "continuation_request_id": committed.request_id,
                "cell_id": cell_id,
                "prefix_tokens": len(spec2.prompt_ids),
                "endpoint_pinned": pinned,
                "reprefilled": reprefilled,
                "aliased": aliased,
            },
        )
    return _merged_decision(
        first,
        committed,
        spec=spec2,
        prefill_ids=prefill,
        tokenizer=tokenizer,
        endpoint_pinned=pinned,
        reprefilled=reprefilled,
        aliased=aliased,
    )


# --------------------------------------------------------------------------- main entry


def generate_decision(
    req: DecisionRequest,
    *,
    store: RequestStore,
    pool: EndpointPool,
    cost_fn: Callable[..., Mapping[str, Any]],
    context: DecisionContext | None = None,
    client: VllmChatClient | None = None,
    cell_id: str | None = None,
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> DecisionRecord:
    """One decision → one :class:`DecisionRecord` (contract §2), in one or two requests.

    One ``/v1/chat/completions`` call (plus the frozen §10.4 exogenous retries), or none at
    all when the store already holds the request id.  When that call's ``</think>`` was
    *forced* — ``boundary_injected``, i.e. the thinking count reached the budget — the
    report is not taken from what followed the forced boundary: the model is mid-derivation
    there and a third of such draws never open ``<final>`` at all.  A second request
    (:func:`continue_decision`) continues the exact token prefix through ``</think>`` with
    the report opening prefilled, and the returned record carries phase 1's thinking side,
    phase 2's report, both ``physical_request_ids`` and the summed ``flops``.  A naturally
    closed draw is untouched: one request, one id.

    ``context`` defaults to the process-wide one installed by :func:`bind_context`, so the
    frozen four-argument signature works unchanged.
    """
    ctx = context if context is not None else default_context()
    spec = request_spec(req, ctx)
    preflight_decision(spec, ctx, size=req.checkpoint)  # raises ContextTooLong first
    tokenizer = ctx.tokenizer(req.checkpoint)
    generator = client if client is not None else ArdChatClient(
        pool, tokenizer, cost_fn=cost_fn, run_id=ctx.run_id
    )
    record, aliased = store.get_or_generate(spec, generator, cell_id, on_event=on_event)
    decision = decision_from_record(
        record,
        budget_tokens=budget_of(spec.decoding),
        tokenizer=tokenizer,
        aliased=aliased,
    )
    if not decision.boundary_injected:
        return decision
    return continue_decision(
        decision,
        record,
        spec=spec,
        size=req.checkpoint,
        context=ctx,
        store=store,
        pool=pool,
        cost_fn=cost_fn,
        client=generator,
        cell_id=cell_id,
        on_event=on_event,
    )


# --------------------------------------------------------------------------- FLOP oracle


def oracle_for(checkpoint: Checkpoint, home: str | os.PathLike | None = None) -> FlopOracle:
    """The verified analytic oracle of one checkpoint (``study/runner.py:254-261``)."""
    return FlopOracle.from_checkpoint(checkpoint, home)


def cost_fn_for(
    checkpoints: Mapping[str, Checkpoint], home: str | os.PathLike | None = None
) -> Callable[..., Mapping[str, Any]]:
    """``cost_fn(checkpoint, prompt_tokens, completion_tokens) -> flops`` for the ARD client.

    Attached exactly as ``study/runner.py:254-261`` attaches it: read the pinned
    ``config.json`` per checkpoint, then :func:`make_cost_fn`.  Without this every
    ``DecisionRecord.flops`` would be the ``None`` placeholder and the run would produce no
    FLOP number at all (contract §2).
    """
    return make_cost_fn(oracles_from_checkpoints(dict(checkpoints), home))


def oracle_tables(
    checkpoints: Mapping[str, Checkpoint], home: str | os.PathLike | None = None
) -> dict[str, Any]:
    """Per-checkpoint FLOP tables + ``oracle_hash`` for ``FROZEN.yaml`` (spec §10.2)."""
    return _study_oracle_tables(oracles_from_checkpoints(dict(checkpoints), home))


def verify_oracle_hash(frozen_tables: Mapping[str, Any], oracle: FlopOracle, size: str) -> None:
    """Refuse an oracle that differs from the frozen one (``study/runner.py:239-246``)."""
    frozen_hash = (frozen_tables.get(size) or {}).get("oracle_hash")
    if frozen_hash is not None and frozen_hash != oracle.oracle_hash:
        raise ProtocolError(
            f"oracle for {size} ({oracle.oracle_hash[:12]}) differs from the frozen one "
            f"({str(frozen_hash)[:12]})"
        )
