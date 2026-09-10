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

Report *parsing* is not here: this module never looks at report tags.  It hands WP2's
``ard/report.py`` the post-``</think>`` text and the token ids.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agents_scaling.ard import contract as K
from agents_scaling.ard.decoding import (
    ARD_DECODING_ROLE,
    budget_of,
    decoding_for_purpose,
)
from agents_scaling.serving.context import (
    ContextCapacityError,
    ContextPreflight,
    preflight_chat_context,
)
from agents_scaling.serving.profiles import ServingProfile, get_serving_profile
from agents_scaling.study.inference.client import EndpointPool, VllmChatClient
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
    "ContextTooLong",
    "DecisionContext",
    "DecisionRecord",
    "DecisionRequest",
    "bind_context",
    "cost_fn_for",
    "decision_from_record",
    "default_context",
    "generate_decision",
    "oracle_for",
    "oracle_tables",
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

    @property
    def valid_delimiters(self) -> bool:
        """The completion opened and closed exactly one thinking channel."""
        return self.think_status is None

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
            "prompt_tokens": self.prompt_tokens,
            "cached_prefill_tokens": self.cached_prefill_tokens,
            "finish_reason": self.finish_reason,
            "latency_s": self.latency_s,
            "endpoint_id": self.endpoint_id,
            "flops": dict(self.flops),
            "think_status": None if self.think_status is None else self.think_status.value,
            "aliased": self.aliased,
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
        aliased=bool(aliased),
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
    """One decision → one committed request → one :class:`DecisionRecord` (contract §2).

    Exactly one ``/v1/chat/completions`` call (plus the frozen §10.4 exogenous retries),
    or none at all when the store already holds the request id.  ``context`` defaults to
    the process-wide one installed by :func:`bind_context`, so the frozen four-argument
    signature works unchanged.
    """
    ctx = context if context is not None else default_context()
    spec = request_spec(req, ctx)
    preflight_decision(spec, ctx, size=req.checkpoint)  # raises ContextTooLong first
    tokenizer = ctx.tokenizer(req.checkpoint)
    generator = client if client is not None else ArdChatClient(
        pool, tokenizer, cost_fn=cost_fn, run_id=ctx.run_id
    )
    record, aliased = store.get_or_generate(spec, generator, cell_id, on_event=on_event)
    return decision_from_record(
        record,
        budget_tokens=budget_of(spec.decoding),
        tokenizer=tokenizer,
        aliased=aliased,
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
