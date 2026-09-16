"""Phase 2: the report, continued from the exact phase-1 token prefix (plan D5, amendment A3).

Why it exists, measured on ARD36's pilot roots cell: vLLM's forced ``</think>`` takes the
delimiter out of the stream but does not put the model into report mode.  Half the draws hit
the thinking cap, and about a third of *those* produce no report at all, because the model is
mid-derivation when the boundary lands.  Cap hits track item difficulty, so the loss is not
noise -- it biases every arm.  A prompt-wording repair moved it by less than a point: a model
mid-derivation is not attending to the instruction.

So a cap-hit decision is finished by a second physical request: the exact phase-1 token prefix
through ``</think>``, with the JSON report opening prefilled, continued through
``/v1/completions``.  The trigger is ``boundary_injected`` **alone** -- never whether phase 1's
span happened to parse.  Conditioning on the parse would be a per-draw judgement correlated
with difficulty; the uniform rule is also the faithful reading of the independent final
allowance, which under a forced boundary is exactly this continuation.

**Vendored, parametrically.**  This module is ``ard/client.py:978-1358``
(``ArdCompletionClient``, ``continue_decision``, ``_merged_decision``) with every ARD36
constant replaced by a parameter: the final cap comes from
:class:`~agents_scaling.nextdyn.contract.TokenContract`, the prefill bytes from
``prompts.lock.json`` through :func:`agents_scaling.nextdyn.decoding.prefill_text`, the turn
token from the tokenizer.  ``agents_scaling.ard.client`` is not on the ``nextdyn.ARD_IMPORTS``
allowlist, so the copy carries a parity test:
``tests/nextdyn/test_continuation.py::test_vendored_continuation_matches_ard_on_the_ard_contract``
runs the ARD original and this copy on the same fake-server phase-1 record under the ARD
constants and asserts identical merged token ids and summed FLOPs.

Phase 2 commits as its own store record (the store has no transaction spanning two), so a cell
that dies between the phases resumes by aliasing phase 1 and issuing only the continuation.
Both ids go under one ``generation_uid``: one sample, two requests.

Spec: 5.5, 6.1, 15.3, 17.2.  Plan: D5, §1.7, §1.11 (VENDORED table).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Final

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.inference import (
    COMPLETED_FINISH_REASONS,
    ContextTooLong,
    ContinuationOutcome,
    DecisionRecord,
    DecisionRequest,
    DispatchContext,
    NextdynChatClient,
    ProducerStamp,
    ReplicaStamp,
    replica_of,
    server_batch_id,
    span_logprob_mean,
    uncertain_execution_of,
)
from agents_scaling.serving.context import ContextPreflight
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.study import identity
from agents_scaling.study.inference.client import (  # noqa: PLC2701 - the shared
    _AbortedCompletion,   # response-shape helpers of the frozen study client; ard/client.py
    _extension,           # imports the same three for the same reason.
    _int_list,
)
from agents_scaling.study.types import ProtocolError, RequestRecord, RequestSpec, SeedKey

#: Domain tag folded into a phase-2 request id.  A continuation is not a decision and must
#: never alias one even if every frozen identity component happened to agree.
CONTINUATION_ID_TAG: Final[str] = "nd1/continuation/v1"

#: Suffix appended to the decision's seed-key namespace to derive the continuation's seed.
#: Phase 2 is a different request with a different prompt, so reusing the decision's engine
#: seed would be an accident waiting to be explained; deriving it keeps the pair deterministic,
#: so a resumed cell re-derives exactly the ids the original run committed.
CONTINUATION_NAMESPACE_SUFFIX: Final[str] = "|CONTINUATION"


@dataclass(frozen=True)
class ContinuationContract:
    """Everything the phase-2 path needs that ARD36 read off module constants.

    Holding them in one object is what makes the vendored copy testable against the original:
    the parity test builds this with the ARD36 values and runs the same code.
    """

    tokens: K.TokenContract
    prefill_text: str
    id_tag: str = CONTINUATION_ID_TAG
    namespace_suffix: str = CONTINUATION_NAMESPACE_SUFFIX
    role: str = D.CONTINUATION_ROLE

    def __post_init__(self) -> None:
        if not self.prefill_text:
            raise ContractViolation("the continuation prefill text is empty")

    @classmethod
    def of(cls, tokens: K.TokenContract, prefill_text: str) -> "ContinuationContract":
        return cls(tokens=tokens, prefill_text=prefill_text)

    @property
    def final_cap(self) -> int:
        """The phase-2 allowance: the decision's own final cap, never a second number (R1)."""
        return int(self.tokens.final_cap)

    @property
    def served_window(self) -> int:
        return int(self.tokens.served_window)

    @property
    def reserve(self) -> int:
        return int(self.tokens.reserve)


# --------------------------------------------------------------------------- identity


@dataclass(frozen=True)
class ContinuationSpec(RequestSpec):
    """A phase-2 request: the prompt is an exact token prefix, not chat messages.

    ``messages`` is deliberately empty -- there is no chat template to render, and a
    synthesised message list would be a fiction the prompt-token-id equality check would then
    have to work around.  The prompt's identity is carried by :attr:`prefix_hash`, a tenth
    component folded into :attr:`request_id` on top of ``study/identity.request_id``'s frozen
    nine.  R5 still holds exactly: everything sent to the engine is inside the id -- the token
    prefix through ``prefix_hash``, the sampler and the final allowance through
    ``decoding_hash``, the pins and the seed through the frozen components.
    """

    prompt_ids: tuple[int, ...] = ()
    id_tag: str = CONTINUATION_ID_TAG

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ProtocolError("a continuation spec needs its phase-1 token prefix")
        if self.messages:
            raise ProtocolError("a continuation carries a token prefix, not chat messages")

    @property
    def prefix_hash(self) -> str:
        """SHA-256 of JCS(prompt token ids) -- the prompt-bytes identity of a phase-2 request."""
        return identity.sha256_hex(identity.jcs([int(i) for i in self.prompt_ids]))

    def identity_fields(self) -> dict[str, Any]:
        return {**super().identity_fields(), "prefix_hash": self.prefix_hash}

    @property
    def request_id(self) -> str:
        """The frozen nine-component id, re-hashed with the token prefix and a domain tag."""
        fields = dict(self.identity_fields())
        prefix_hash = str(fields.pop("prefix_hash"))
        return identity.sha256_hex(
            identity.jcs([identity.request_id(**fields), self.id_tag, prefix_hash]))


def continuation_seed_key(seed_key: SeedKey, *, suffix: str = CONTINUATION_NAMESPACE_SUFFIX) -> SeedKey:
    """The decision's seed key in the continuation namespace (spec 17.2 cache identity)."""
    return replace(seed_key, namespace=f"{seed_key.namespace}{suffix}")


def continuation_prefix_ids(record: RequestRecord, prefill_ids: Sequence[int], *,
                            think_end_id: int) -> tuple[int, ...]:
    """The exact phase-2 prompt: phase-1 prompt + completion through ``</think>`` + prefill.

    Token ids, never re-rendered text: the continuation must resume the *same* sequence the
    engine already has in its prefix cache, and a decode/re-encode round trip is not
    guaranteed to reproduce it.
    """
    ids = tuple(int(i) for i in record.response.get("token_ids") or ())
    try:
        end = ids.index(int(think_end_id), 1)
    except ValueError as exc:
        raise ProtocolError(
            f"request {record.request_id[:12]} has no closing reasoning delimiter "
            f"({think_end_id}) to continue from") from exc
    return (tuple(int(i) for i in record.prompt_token_ids) + ids[: end + 1]
            + tuple(int(i) for i in prefill_ids))


def continuation_spec(spec: RequestSpec, record: RequestRecord, *, prefill_ids: Sequence[int],
                      cc: ContinuationContract, sampling: K.Sampling,
                      think_end_id: int) -> ContinuationSpec:
    """The identity-bearing spec of the phase-2 request of one cap-hit decision."""
    return ContinuationSpec(
        messages=(),
        decoding=D.continuation_decoding(cc.tokens, sampling),
        checkpoint=spec.checkpoint,
        seed_key=continuation_seed_key(spec.seed_key, suffix=cc.namespace_suffix),
        role=cc.role,
        study_id=spec.study_id,
        study_seed_hex=spec.study_seed_hex,
        prompt_ids=continuation_prefix_ids(record, prefill_ids, think_end_id=think_end_id),
        id_tag=cc.id_tag,
    )


def preflight_continuation(spec: ContinuationSpec, *, cc: ContinuationContract,
                           profile_name: str) -> ContextPreflight:
    """``len(prefix) + final_cap + reserve <= served_window``, before the request.

    The prefix is a decision's whole prompt *plus* its whole thinking span, so unlike the
    phase-1 preflight this one is not implied by the prompt budget and is checked on its own.
    """
    preflight = ContextPreflight(
        profile_name=profile_name,
        prompt_tokens=len(spec.prompt_ids),
        requested_output_tokens=int(spec.decoding.max_tokens),
        reserve_tokens=int(cc.reserve),
        served_context=int(cc.served_window),
    )
    if not preflight.fits:
        raise ContextTooLong(preflight)
    return preflight


# --------------------------------------------------------------------------- endpoints


class EndpointAffinity:
    """An endpoint-pool facade that hands out one preferred endpoint first.

    A continuation re-sends the decision's prompt *plus* its whole thinking span, so it is
    worth landing back on the server that has just prefilled them: vLLM's prefix cache is on by
    default and a miss pays the prefill twice.  The preference is per thread, because one
    decision is generated by one worker thread, and is consumed by the first :meth:`pick`;
    every retry after it goes back to the pool, so the frozen rotation and cooldown are
    untouched.  :meth:`outcome` reports whether the pin was honoured, and whether the prefix
    had to be re-prefilled somewhere else.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._state = threading.local()

    def prefer(self, host: Any = None, port: Any = None) -> None:
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
        # endpoints(), wait(), profile, shard, replica_stamp, ... -- everything else is the pool's.
        return getattr(self._pool, name)


# --------------------------------------------------------------------------- the client


def _completion_logprobs(choice: Any) -> tuple[list[float | None] | None, list[list[dict[str, Any]]] | None]:
    """``choices[0].logprobs`` of ``/v1/completions``, in the shapes the chat path produces.

    Completions report parallel arrays (``token_logprobs``, and ``top_logprobs`` as one
    ``token -> logprob`` mapping per generated token) where chat reports a ``content`` list of
    per-token objects.  Normalising here means the merge reads one shape and the alternatives
    keep a defined order (most probable first, ties broken by token).
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
    values = [_as_float(v) for v in raw] if is_list(raw) else None

    raw_rows = field("top_logprobs")
    rows: list[list[dict[str, Any]]] | None = None
    if is_list(raw_rows):
        rows = []
        for entry in raw_rows:
            row: list[dict[str, Any]] = []
            if isinstance(entry, Mapping):
                for token, value in entry.items():
                    logprob = _as_float(value)
                    if token is not None and logprob is not None:
                        row.append({"token": str(token), "logprob": float(logprob)})
                row.sort(key=lambda alt: (-alt["logprob"], alt["token"]))
            rows.append(row)
    return values, rows


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


class NextdynCompletionClient(NextdynChatClient):
    """Phase 2: ``/v1/completions`` on an exact token prefix.

    Everything that governs *how* a request is issued is the frozen study client's -- the
    exogenous-retry policy, the endpoint rotation and failure reporting, the attempt ledger,
    the record assembly and the refusal to commit a FLOP placeholder.  Only what a raw-prefix
    request must change is changed: the prompt is the spec's token ids instead of a rendered
    chat template, the body is a completions body (``prompt`` as ``list[int]``, ``logprobs`` as
    a width rather than a flag), the route is ``/v1/completions``, and the response is the
    completions shape.
    """

    def __init__(self, pool: Any, tokenizer: Any, **kwargs: Any) -> None:
        # No adapter: a continuation's wire body is a full override (the route, the prompt and
        # the logprob shape all change), so there is nothing for an adapter to contribute.
        super().__init__(pool, tokenizer, adapter=None, **kwargs)

    def render_prompt_ids(self, spec: RequestSpec) -> tuple[int, ...]:
        """The spec's token prefix -- no chat template is rendered for a continuation."""
        ids = tuple(int(i) for i in getattr(spec, "prompt_ids", ()) or ())
        if not ids:
            raise ProtocolError("a continuation spec carries no token prefix")
        return ids

    def chat_template_hash(self, spec: RequestSpec) -> str:
        """No template is applied, so the prefix hash *is* the rendered-bytes hash."""
        return spec.prefix_hash

    def request_kwargs(self, spec: RequestSpec,
                       guided_json_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
        decoding = spec.decoding
        if guided_json_schema is not None or bool(decoding.guided_json):
            raise ContractViolation("a report continuation is never guided (plan D6)")
        if int(decoding.max_tokens) != int(decoding.final_tokens) or int(decoding.final_tokens) < 1:
            raise ContractViolation(
                "a continuation asks for exactly the final allowance and nothing else: "
                f"max_tokens={decoding.max_tokens} final_tokens={decoding.final_tokens}")
        if int(decoding.thinking_tokens) != 0:
            raise ContractViolation(
                "a continuation sends no thinking_token_budget: the thinking channel is "
                f"already closed inside the prompt (got {decoding.thinking_tokens})")
        return {
            "model": spec.checkpoint.served_model_name,
            "prompt": [int(i) for i in spec.prompt_ids],
            "temperature": float(decoding.temperature),
            "max_tokens": int(decoding.max_tokens),
            "seed": int(spec.engine_seed),
            # Completions take the width itself, not a flag plus a width.
            "logprobs": self.top_logprobs,
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
        ``self._sdk_client(entry).chat.completions.create(**kwargs)``.  The route is the one
        line a continuation changes, so it is re-pointed here rather than by forking the frozen
        retry loop into a second copy that would then have to be kept in step.
        """
        completions = super()._sdk_client(entry).completions
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def _parse_response(self, response: Any, local_ids: tuple[int, ...]) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ProtocolError(
                f"server returned {0 if not choices else len(choices)} choices, expected exactly 1")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in COMPLETED_FINISH_REASONS:
            raise _AbortedCompletion(f"finish_reason={finish_reason!r}")

        # /v1/completions carries the prompt ids per choice, not at the top level.
        present, prompt_ids_raw = _extension(choice, "prompt_token_ids")
        if not present or prompt_ids_raw is None:
            present, prompt_ids_raw = _extension(response, "prompt_token_ids")
        if not present or prompt_ids_raw is None:
            raise ProtocolError("completion response lacks prompt_token_ids "
                                "(return_token_ids unsupported?)")
        prompt_ids = _int_list(prompt_ids_raw, what="prompt_token_ids")
        if tuple(prompt_ids) != tuple(local_ids):
            raise ProtocolError(
                "server prompt_token_ids disagree with the phase-1 token prefix: "
                f"local={len(local_ids)} tokens "
                f"sha256={identity.sha256_hex(identity.jcs(list(local_ids)))[:16]}, "
                f"server={len(prompt_ids)} tokens "
                f"sha256={identity.sha256_hex(identity.jcs(prompt_ids))[:16]}")
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
                        f"usage.{key}={count} but the server returned {len(ids)} token ids")

        token_logprobs, top_logprobs = _completion_logprobs(choice)
        from agents_scaling.nextdyn.inference import _cached_tokens
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


# --------------------------------------------------------------------------- the merge


def sum_flops(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict:
    """Both physical requests' FLOPs, summed (defect D-02: never count one where two ran).

    Both phases run the same checkpoint, so they must carry the same oracle hash; a
    disagreement means the two records are not two halves of one decision and the sum would be
    meaningless.  The oracle is a pure function of token counts, so the re-sent prefix is
    counted again even when vLLM served it from its prefix cache: the summed number is the work
    the analytic model attributes to the decision, and ``cached_prefill_tokens`` records how
    much of it the engine actually skipped.
    """
    oracle, other = first.get("oracle"), second.get("oracle")
    if oracle != other:
        raise ProtocolError(
            f"phase-1 oracle {str(oracle)[:12]} != phase-2 oracle {str(other)[:12]}: the two "
            "requests of one decision must share the checkpoint's oracle")
    summed: dict[str, Any] = {"oracle": oracle}
    for key in ("prefill", "decode", "total"):
        left, right = first.get(key), second.get(key)
        if left is None or right is None:
            raise ProtocolError(f"a physical request carries the FLOP placeholder for {key!r}")
        summed[key] = int(left) + int(right)
    return summed


def merged_decision(first: DecisionRecord, record: RequestRecord, *, spec: ContinuationSpec,
                    prefill_ids: Sequence[int], cc: ContinuationContract, tokenizer: Any,
                    replica: ReplicaStamp, endpoint_pinned: bool | None, reprefilled: bool,
                    aliased: bool) -> DecisionRecord:
    """One :class:`DecisionRecord` for a decision executed as two physical requests.

    The thinking side is phase 1's and the report side is phase 2's: ``thinking_tokens``,
    ``native_reasoning_text``, ``prompt_tokens`` and ``prompt_token_ids`` come from the
    decision; ``final_tokens``, ``final_text``, ``answer_logprob``, ``finish_reason``,
    ``cap_hit`` and ``final_overrun`` describe the report that was *delivered*, because that is
    the one WP2 parses and votes on.  ``completion_token_ids`` is the logical completion --
    phase 1 through the forced ``</think>``, then the prefilled opening, then phase 2's report
    -- so delimiter accounting re-derived from it agrees with the fields.

    Phase 1's discarded post-boundary span is not in it; it is preserved in the phase-1 store
    record and its *length* is carried as ``discarded_span_tokens``, so the cost ledger
    attributes the GPU seconds that produced it instead of dropping them (defect D-02).

    ``flops`` is the only field summed across the two, because cost is the one genuinely
    additive quantity; ``latency_s`` is the exception it has to be, being the wall time of the
    decision, which is both calls.
    """
    prefix = tuple(int(i) for i in spec.prompt_ids)
    prefill = tuple(int(i) for i in prefill_ids)
    if prefix[: len(first.prompt_token_ids)] != tuple(first.prompt_token_ids):
        raise ProtocolError("the continuation prefix does not start with the decision's prompt")
    if prefill and prefix[-len(prefill):] != prefill:
        raise ProtocolError("the continuation prefix does not end with the prefilled report opening")

    response = record.response
    ids = tuple(int(i) for i in response.get("token_ids") or ())
    turn_end = A.turn_end_id(tokenizer)
    generated = ids[:-1] if ids and turn_end is not None and ids[-1] == turn_end else ids
    kept = generated[: cc.final_cap]
    final_ids = prefill + kept
    final_text = tokenizer.decode(list(final_ids)) if final_ids else ""
    if not isinstance(final_text, str):
        raise ProtocolError("tokenizer.decode did not return text")

    answer_logprob = span_logprob_mean(response.get("token_logprobs"), n_completion=len(ids),
                                       start=0, stop=len(kept))
    top_block = response.get("top_logprobs")
    final_top = None
    if top_block and kept and max((len(row) for row in top_block), default=0) > 1:
        # The prefilled ids were prompt: they have no alternatives, but the rows stay indexed
        # by the final span.
        final_top = tuple([()] * len(prefill) + [tuple(row) for row in top_block[: len(kept)]])

    timing = record.timing or {}
    latency = float(timing.get("latency_s") or 0.0)
    arrival = float(timing.get("submitted_at") or 0.0)
    finish_reason = str(response.get("finish_reason"))
    outcome = ContinuationOutcome(
        request_id=record.request_id,
        prefix_tokens=len(prefix),
        prefill_token_ids=prefill,
        generated_tokens=len(generated),
        endpoint_id=replica.endpoint_id,
        replica_id=replica.replica_id,
        host=replica.host,
        gpu_type=replica.gpu_type,
        partition=replica.partition,
        endpoint_pinned=endpoint_pinned,
        reprefilled=bool(reprefilled),
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(response.get("cached_prefill_tokens") or 0),
        finish_reason=finish_reason,
        latency_s=latency,
        arrival_ts=arrival,
        flops=dict(record.flops or {}),
        aliased=bool(aliased),
        n_attempts=len(tuple(timing.get("attempts") or ())),
        first_aliased=bool(first.aliased),
    )
    physical = (*first.physical_request_ids, record.request_id)
    return DecisionRecord(
        request_ids=(*first.request_ids, spec.request_id),
        physical_request_ids=physical,
        generation_uid=I.generation_uid(physical),
        prompt_token_ids=first.prompt_token_ids,
        completion_token_ids=prefix[len(first.prompt_token_ids):] + generated,
        native_reasoning_text=first.native_reasoning_text,
        final_text=final_text,
        thinking_tokens=first.thinking_tokens,
        final_tokens=len(kept),
        boundary_injected=True,
        natural_stop=False,
        cap_hit=finish_reason == "length",
        final_overrun=len(generated) > cc.final_cap,
        answer_logprob=answer_logprob,
        final_top_logprobs=final_top,
        prompt_tokens=first.prompt_tokens,
        # Phase 1's, not the sum: a consumer computing "prefill actually paid" as
        # prompt_tokens - cached_prefill_tokens must not be handed a cache count from a prompt
        # it is not counting (a phase-2 prefix is usually a near-total cache hit and would
        # drive that difference to zero).  Phase 2's own count is in ``continuation``.
        cached_prefill_tokens=first.cached_prefill_tokens,
        finish_reason=finish_reason,
        latency_s=first.latency_s + latency,
        endpoint_id=first.endpoint_id,
        replica_id=first.replica_id,
        server_batch_id=first.server_batch_id,
        arrival_ts=first.arrival_ts,
        gpu_type=first.gpu_type,
        partition=first.partition,
        host=first.host,
        flops=sum_flops(first.flops, dict(record.flops or {})),
        continuation=outcome.to_dict(),
        discarded_span_tokens=int(first.discarded_span_tokens),
        aliased=bool(first.aliased and aliased),
        uncertain_execution=bool(first.uncertain_execution
                                 or uncertain_execution_of(tuple(timing.get("attempts") or ()))),
        attempts=first.attempts + tuple(timing.get("attempts") or ()),
    )


# --------------------------------------------------------------------------- entry points


def continue_from(first: DecisionRecord, record: RequestRecord, *, spec: RequestSpec,
                  cc: ContinuationContract, sampling: K.Sampling, tokenizer: Any, store: Any,
                  pool: Any, cost_fn: Callable[..., Mapping[str, Any]],
                  failure_policy: K.FailurePolicy, top_logprobs: int, profile_name: str,
                  cell_id: str, run_id: str, producer: ProducerStamp | None = None,
                  on_event: Callable[[str, Mapping[str, Any]], None] | None = None) -> DecisionRecord:
    """Phase 2 of one cap-hit decision, under an explicit :class:`ContinuationContract`.

    The parametric core: every ARD36 constant is an argument here, which is what lets the
    parity test run this exact code under the ARD36 values against ``ard/client.py``.
    """
    if not first.boundary_injected:
        raise ContractViolation(
            f"decision {first.physical_request_ids[0][:12]} closed its reasoning naturally; "
            "phase 2 runs only under a forced boundary (plan D5)")
    _start, think_end = A.delimiter_ids(tokenizer)
    prefill = D.continuation_prefill_ids(tokenizer, cc.prefill_text)
    spec2 = continuation_spec(spec, record, prefill_ids=prefill, cc=cc, sampling=sampling,
                              think_end_id=think_end)
    preflight_continuation(spec2, cc=cc, profile_name=profile_name)   # ContextTooLong first
    affinity = EndpointAffinity(pool)
    generator = NextdynCompletionClient(
        affinity, tokenizer, failure_policy=failure_policy, top_logprobs=int(top_logprobs),
        cost_fn=cost_fn, run_id=run_id)
    if producer is not None:
        generator.stamp(producer)
    endpoint = record.endpoint or {}
    affinity.prefer(endpoint.get("host"), endpoint.get("port"))
    committed, aliased = store.get_or_generate(spec2, generator, cell_id, on_event=on_event)
    pinned: bool | None = None
    reprefilled = False
    if not aliased:
        pinned, reprefilled = affinity.outcome()
    if on_event is not None:
        on_event("continuation", {
            "request_id": first.physical_request_ids[0],
            "continuation_request_id": committed.request_id,
            "cell_id": cell_id,
            "prefix_tokens": len(spec2.prompt_ids),
            "endpoint_pinned": pinned,
            "reprefilled": reprefilled,
            "aliased": aliased,
        })
    return merged_decision(first, committed, spec=spec2, prefill_ids=prefill, cc=cc,
                           tokenizer=tokenizer, replica=replica_of(pool, committed),
                           endpoint_pinned=pinned, reprefilled=reprefilled, aliased=aliased)


def continue_decision(first: DecisionRecord, record: RequestRecord, split: A.SplitCompletion, *,
                      spec: RequestSpec, req: DecisionRequest, ctx: DispatchContext,
                      client: NextdynChatClient | None = None,
                      on_event: Callable[[str, Mapping[str, Any]], None] | None = None) -> DecisionRecord:
    """Phase 2 of one cap-hit decision, with the contract read off the study config.

    Called by :func:`agents_scaling.nextdyn.inference.generate_decision` whenever -- and only
    whenever -- phase 1's boundary was forced.  The prefill bytes come from ``prompts.lock.json``
    through the dispatch context, so what is re-sent is the locked prompt bytes and nothing else.
    """
    if not split.boundary_injected:
        raise ContractViolation("continue_decision is called only on a forced boundary (plan D5)")
    model = ctx.contract.model(req.model.name)
    cc = ContinuationContract.of(req.tokens, D.prefill_text() if ctx.prefill_text is None
                                 else ctx.prefill_text)
    pool = client.pool if client is not None else ctx.pool(model)
    return continue_from(
        first, record, spec=spec, cc=cc, sampling=ctx.contract.sampling,
        tokenizer=ctx.tokenizer(model), store=ctx.store, pool=pool, cost_fn=ctx.cost_fn,
        failure_policy=ctx.contract.failure_policy, top_logprobs=int(req.top_logprobs),
        profile_name=get_serving_profile(model.profile).name, cell_id=ctx.cell_id,
        run_id=ctx.run_id,
        producer=client.producer if client is not None else None,
        on_event=on_event,
    )


__all__ = [
    "CONTINUATION_ID_TAG", "CONTINUATION_NAMESPACE_SUFFIX", "ContinuationContract",
    "ContinuationSpec", "EndpointAffinity", "NextdynCompletionClient", "continuation_prefix_ids",
    "continuation_seed_key", "continuation_spec", "continue_decision", "continue_from",
    "merged_decision", "preflight_continuation", "sum_flops",
]
