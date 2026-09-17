"""One logical decision -> one :class:`DecisionRecord`, in one or two physical requests.

This is the generation front door.  Everything that governs *how* a request is issued is the
frozen study client's -- the exogenous-retry policy, the prompt-token-id equality check, the
endpoint rotation and the attempt ledger (``study/inference/client.py``) -- and only what this
study adds is added: the adapter's wire fields, the ``max_tokens`` identity, the refusal to
commit a FLOP placeholder, the full producer stamp, and the partition filter that keeps a
deadline-bearing lane off preemptable endpoints.

Spec 15.3 says the operational failure policy in one line: *no extra model calls may repair
malformed answers; infrastructure retries are allowed only under the fixed retry policy*.
ARD36 expressed that as a comment; here :class:`NextdynChatClient` asserts it against the
compiled :class:`~agents_scaling.nextdyn.contract.FailurePolicy` at construction, so a config
that quietly allowed a repair call could not build a client at all.  ``COMPLETED_FINISH_REASONS``
(``stop``/``length``) are terminal regardless of what the text turned out to be: a malformed
object is an observation, never a reason to sample again (spec 4.6, 15.3).

Two-phase draws.  A forced ``</think>`` does not put the model in report mode, so a cap-hit
draw is finished by a second physical request from the exact phase-1 token prefix (plan D5,
amendment A3; :mod:`agents_scaling.nextdyn.continuation`).  The trigger is ``boundary_injected``
alone -- never whether phase 1 happened to parse, which would be a per-draw judgement
correlated with difficulty.  Both physical ids sit under one ``generation_uid``: one sample,
two requests, and cost accounting that counts both (defect D-02).

Spec: 5.5, 5.6, 15.3, 17.2 (replica/batch metadata), 18.5.  Plan: §1.7, D5, D6, D11.
"""

from __future__ import annotations

import math
import os
import socket
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn.errors import ContractViolation, NextdynError
from agents_scaling.serving.context import ContextCapacityError, ContextPreflight, preflight_chat_context
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.serving.registry import ServerEntry
from agents_scaling.study.inference.client import (
    COMPLETED_FINISH_REASONS,
    AttemptInfo,
    EndpointPool,
    VllmChatClient,
)
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn
from agents_scaling.study.types import Checkpoint, ProtocolError, RequestSpec, SeedKey

#: SDK error types that mean "the request may have completed on the server".  A retry after
#: one of these is not a second *sample* -- the same request id and the same seed are re-sent
#: (spec 15.3) -- but the replica may have done the work twice, so the decision is marked and
#: the cost ledger keeps both attempts (plan §3.7).
UNCERTAIN_ERROR_TYPES: Final[frozenset[str]] = frozenset({"APITimeoutError"})

#: One-second arrival bucket: the batch proxy of the documented attribution approximation.
#: Vendored from ``ard/resources.py:177-186`` (that module's ``server_batch_id`` is not on the
#: ``nextdyn.ARD_IMPORTS`` allowlist), with a parity test in ``tests/nextdyn/test_telemetry.py``.
ARRIVAL_BUCKET_S: Final[float] = 1.0


class ContextTooLong(NextdynError):
    """The rendered prompt plus the output envelope plus the reserve exceeds the served window.

    Typed and never retried: ``failure_policy.context_overflow`` is ``typed_failure``, and
    another endpoint would refuse identically.  WP2's episode loop turns it into a
    context-failure decision and the round continues (plan §1.7).
    """

    def __init__(self, preflight: ContextPreflight, cause: Exception | None = None) -> None:
        self.preflight = preflight
        super().__init__(
            "prompt refused before dispatch (spec 6.3 context_failure): "
            f"profile={preflight.profile_name!r} prompt={preflight.prompt_tokens} "
            f"max_tokens={preflight.requested_output_tokens} reserve={preflight.reserve_tokens} "
            f"required={preflight.required_tokens} > served_window={preflight.served_context}")
        self.__cause__ = cause


def server_batch_id(endpoint_id: str, arrival_ts: float) -> str:
    """``(endpoint_id, one-second arrival bucket)`` -- the batch proxy, as a string key.

    The OpenAI-compatible API exposes no batch id, so this stands in for one; it is the join
    key of the GPU-second attribution and is recorded per physical request (spec 17.3).
    """
    return f"{endpoint_id}@{int(math.floor(float(arrival_ts) / ARRIVAL_BUCKET_S))}"


# --------------------------------------------------------------------------- checkpoints, cost


def checkpoint_for(model: K.ModelCell) -> Checkpoint:
    """The pinned :class:`~agents_scaling.study.types.Checkpoint` of one nextdyn model cell.

    ``size`` is the cell's *lane* (``8B``/``32B``), which is the scale class the FLOP oracle
    and the arch table are keyed by; ``served_model_name`` is what actually goes on the wire,
    and the two are deliberately different strings (the envelope companion serves ``8B`` under
    its own cell, plan D11).
    """
    if not model.pinned:
        raise ContractViolation(
            f"model cell {model.name!r} has no pinned revision; run `lock-inputs` "
            "(confirmation may not use main/latest, spec 5.4)")
    return Checkpoint(
        size=model.lane,
        hf_id=model.hf_id,
        model_revision=model.revision,
        tokenizer_revision=model.tokenizer_revision,
        profile=model.profile,
        tp_size=model.tp,
        served_model_name=model.served_model_name,
    )


def flop_cost_fn(models: Mapping[str, K.ModelCell], home: str | os.PathLike | None = None):
    """``cost_fn(checkpoint, prompt_tokens, completion_tokens) -> flops`` over the pinned cells.

    The ``ard/resources.flop_cost_fn`` pattern, re-keyed to model cells: read each pinned
    ``config.json`` from the offline HF cache, verify it against the frozen arch table, and
    adapt the oracles to the study client's hook.  Without this every ``flops`` block is the
    ``None`` placeholder and the run ends with no FLOP number at all, which is why
    :meth:`NextdynChatClient._build_record` refuses to commit one.
    """
    oracles: dict[str, FlopOracle] = {}
    for model in models.values():
        if not model.pinned:
            continue
        checkpoint = checkpoint_for(model)
        if checkpoint.size not in oracles:
            oracles[checkpoint.size] = FlopOracle.from_checkpoint(checkpoint, home)
    if not oracles:
        raise ContractViolation("no pinned model cell to build a FLOP oracle from")
    return make_cost_fn(oracles)


# --------------------------------------------------------------------------- the frozen records


@dataclass(frozen=True)
class DecisionRequest:
    """One logical decision, before anything has been sent (contract §2.1).

    ``key`` is the twelve-field :class:`~agents_scaling.nextdyn.ids.CompositeKey` this decision
    *is*; ``seed_key`` is what the engine seed comes from and carries nothing about the
    condition (D4).  ``purpose`` is the phase and must agree with both, so a producer cannot
    seed one phase and record another.
    """

    messages: tuple[Mapping[str, Any], ...]
    model: K.ModelCell
    adapter: str
    tokens: K.TokenContract
    seed_key: SeedKey
    purpose: str
    top_logprobs: int
    key: I.CompositeKey

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(dict(m) for m in self.messages))
        if not self.messages:
            raise ContractViolation("a decision needs at least one chat message")
        if self.purpose != self.seed_key.purpose:
            raise ContractViolation(
                f"purpose {self.purpose!r} disagrees with seed_key.purpose "
                f"{self.seed_key.purpose!r}; the seed key is part of the identity")
        if self.purpose != self.key.phase:
            raise ContractViolation(
                f"purpose {self.purpose!r} disagrees with key.phase {self.key.phase!r}")
        if self.adapter != self.model.adapter:
            raise ContractViolation(
                f"request adapter {self.adapter!r} != model cell adapter {self.model.adapter!r}")
        if int(self.top_logprobs) < 1:
            raise ContractViolation("top_logprobs must be >= 1 (the sampled token is always scored)")


@dataclass(frozen=True)
class ReplicaStamp:
    """Where one physical request ran (spec 17.3 ``requests``; plan §3.7).

    ``gpu_type`` and ``partition`` are properties of the *Slurm job* behind the endpoint, not
    of the registry entry, so they are resolved by :class:`NextdynEndpointPool` -- the object
    that already has to know a job's partition in order to keep a deadline lane off a
    preemptable one.
    """

    endpoint_id: str
    replica_id: str
    host: str
    gpu_type: str | None
    partition: str | None

    @classmethod
    def of(cls, entry: ServerEntry, *, gpu_type: str | None = None,
           partition: str | None = None) -> "ReplicaStamp":
        replica = entry.replica_id
        return cls(
            endpoint_id=f"{entry.host}:{int(entry.port)}",
            replica_id=str(replica) if replica is not None else f"{entry.host}:{int(entry.port)}",
            host=str(entry.host),
            gpu_type=gpu_type,
            partition=partition,
        )


@dataclass(frozen=True)
class ProducerStamp:
    """Who produced one physical request (``requests.producer_json``).

    ARD36 recorded ``{cell_id, run_id, slurm_job_id, host}`` and then could not answer "which
    GPU type did this cost land on" without re-joining the registry by hand.  The endpoint and
    its hardware go in at write time instead.
    """

    cell_id: str
    run_id: str
    slurm_job_id: str | None
    host: str
    endpoint_id: str
    gpu_type: str | None
    partition: str | None

    @classmethod
    def of(cls, *, cell_id: str, run_id: str, replica: ReplicaStamp,
           slurm_job_id: str | None = None) -> "ProducerStamp":
        return cls(
            cell_id=cell_id, run_id=run_id,
            slurm_job_id=slurm_job_id if slurm_job_id is not None else os.environ.get("SLURM_JOB_ID"),
            host=replica.host, endpoint_id=replica.endpoint_id,
            gpu_type=replica.gpu_type, partition=replica.partition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "run_id": self.run_id, "slurm_job_id": self.slurm_job_id,
                "host": self.host, "endpoint_id": self.endpoint_id, "gpu_type": self.gpu_type,
                "partition": self.partition}


@dataclass(frozen=True)
class ContinuationOutcome:
    """What phase 2 of one decision cost and where it ran (plan D5, §3.7).

    Already summed into the merged record's ``flops``; kept whole so the split can be
    recovered, and so ``discarded_span_tokens`` -- phase 1's post-boundary span, which the
    continuation replaced -- is attributable rather than invisible (defect D-02).
    """

    request_id: str
    prefix_tokens: int
    prefill_token_ids: tuple[int, ...]
    generated_tokens: int
    endpoint_id: str
    replica_id: str
    host: str
    gpu_type: str | None
    partition: str | None
    endpoint_pinned: bool | None
    reprefilled: bool
    prompt_tokens: int
    cached_prefill_tokens: int
    finish_reason: str
    latency_s: float
    arrival_ts: float
    flops: dict
    aliased: bool
    #: Phase 2's own attempt count and phase 1's own alias status.  The merged record's
    #: ``attempts`` is both phases' ledgers concatenated and its ``aliased`` is their
    #: conjunction, so without these two the per-request rows could not be split back apart
    #: (``telemetry.request_rows`` writes one row per physical request).
    n_attempts: int
    first_aliased: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "prefix_tokens": self.prefix_tokens,
            "prefill_token_ids": list(self.prefill_token_ids),
            "generated_tokens": self.generated_tokens, "endpoint_id": self.endpoint_id,
            "replica_id": self.replica_id, "host": self.host, "gpu_type": self.gpu_type,
            "partition": self.partition,
            "endpoint_pinned": self.endpoint_pinned, "reprefilled": self.reprefilled,
            "prompt_tokens": self.prompt_tokens,
            "cached_prefill_tokens": self.cached_prefill_tokens,
            "finish_reason": self.finish_reason, "latency_s": self.latency_s,
            "arrival_ts": self.arrival_ts, "flops": dict(self.flops), "aliased": self.aliased,
            "n_attempts": self.n_attempts, "first_aliased": self.first_aliased,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """The measured outcome of one decision (contract §2.1).

    ``request_ids`` are the identities that were *asked for* (``RequestSpec.request_id``, one
    per phase) and ``physical_request_ids`` the ids the store actually holds; the merge
    asserts they agree element-wise, so a race that committed something else is a typed
    failure rather than a silently mislabelled sample.  ``generation_uid`` hashes
    ``physical_request_ids`` in phase order: one sample, however many requests it took.

    ``boundary_injected`` and ``cap_hit`` are different events.  The first is the *thinking*
    budget -- the sampler forced ``</think>`` -- and leaves the final allowance untouched; the
    second is ``finish_reason == "length"``.  ``natural_stop`` is the completion's own turn
    token with both delimiters well formed.

    On a continued draw the thinking side is phase 1's and the report side is phase 2's,
    because that is the report that was delivered; ``flops`` is summed over both physical
    requests and ``latency_s`` is the wall time of the decision.  ``final_tokens`` is the
    length of the *delivered* span after the ``final_cap`` truncation and ``answer_logprob`` is
    the mean over exactly those tokens, so ``sum = mean * final_tokens`` holds by construction
    (``telemetry.decision_row`` relies on it); ``final_overrun`` says the raw span was longer.
    """

    request_ids: tuple[str, ...]
    physical_request_ids: tuple[str, ...]
    generation_uid: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    native_reasoning_text: str
    final_text: str
    thinking_tokens: int
    final_tokens: int
    boundary_injected: bool
    natural_stop: bool
    cap_hit: bool
    final_overrun: bool
    answer_logprob: float | None
    final_top_logprobs: Any
    prompt_tokens: int
    cached_prefill_tokens: int
    finish_reason: str
    latency_s: float
    endpoint_id: str
    replica_id: str
    server_batch_id: str
    arrival_ts: float
    gpu_type: str | None
    partition: str | None
    host: str
    flops: dict
    continuation: dict | None
    discarded_span_tokens: int
    aliased: bool
    uncertain_execution: bool
    attempts: tuple[AttemptInfo, ...]

    @property
    def continued(self) -> bool:
        return self.continuation is not None

    @property
    def n_physical_requests(self) -> int:
        return len(self.physical_request_ids)


#: What an episode hands the loop: ``generate(req) -> DecisionRecord`` (plan §1.7).
Generate = Callable[[DecisionRequest], DecisionRecord]


# --------------------------------------------------------------------------- endpoint pool


@dataclass(frozen=True)
class JobInfo:
    """The partition and GPU type of one Slurm job behind an endpoint."""

    partition: str | None
    gpu_type: str | None


#: ``job_info(slurm_job_id) -> JobInfo``.  Injected so a test (and a dry run) can answer
#: without Slurm; ``fleet`` supplies the live one.
JobInfoFn = Callable[[str], JobInfo]


class NextdynEndpointPool(EndpointPool):
    """An :class:`EndpointPool` that only ever hands out endpoints on allowed partitions.

    The deadline-bearing lanes must not run on preemptable partitions: a preemption there is
    not a slow cell, it is a cell that dies mid-episode and leaves a half-written round.  The
    registry record names the Slurm job, not its partition, so the partition is resolved
    through ``job_info`` and cached per job id.

    Fail closed: with ``allowed_partitions`` set, an endpoint whose partition cannot be
    resolved is *excluded*.  An unknown partition might be a preemptable one, and the whole
    point of the filter is that a deadline lane never finds out the hard way.  With
    ``allowed_partitions`` ``None`` the pool is the frozen study one, unfiltered.
    """

    def __init__(self, *args: Any, allowed_partitions: Iterable[str] | None = None,
                 job_info: JobInfoFn | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allowed_partitions: frozenset[str] | None = (
            None if allowed_partitions is None else frozenset(str(p) for p in allowed_partitions))
        self._job_info = job_info
        self._job_cache: dict[str, JobInfo] = {}
        self._job_lock = threading.Lock()

    def job_info(self, slurm_job_id: str | None) -> JobInfo:
        """Partition and GPU type of ``slurm_job_id`` (cached); unknown when unresolvable."""
        if slurm_job_id is None or self._job_info is None:
            return JobInfo(partition=None, gpu_type=None)
        key = str(slurm_job_id)
        with self._job_lock:
            cached = self._job_cache.get(key)
        if cached is not None:
            return cached
        try:
            info = self._job_info(key)
        except Exception:                               # noqa: BLE001 - squeue is best effort
            info = JobInfo(partition=None, gpu_type=None)
        with self._job_lock:
            self._job_cache[key] = info
        return info

    def replica_stamp(self, entry: ServerEntry) -> ReplicaStamp:
        """Where a request on ``entry`` ran, hardware included (spec 17.3 ``requests``)."""
        info = self.job_info(entry.slurm_job_id)
        return ReplicaStamp.of(entry, gpu_type=info.gpu_type, partition=info.partition)

    def endpoints(self, *, force: bool = False) -> list[ServerEntry]:
        entries = super().endpoints(force=force)
        if self.allowed_partitions is None:
            return entries
        return [e for e in entries if self.job_info(e.slurm_job_id).partition in self.allowed_partitions]


# --------------------------------------------------------------------------- the client


class NextdynChatClient(VllmChatClient):
    """The study client plus this study's wire fields, assertions and stamp.

    ``request_kwargs`` is an *instance* method here (the study's is static) because the wire
    fields are the adapter's, and the adapter is a property of the model cell this client
    serves.  Everything it adds is already inside the request id: the thinking budget rides on
    ``Decoding.thinking_tokens`` and the logprob width is fixed per phase.

    The failure policy is asserted at construction, not documented: ``transient_retries`` is
    the retry cap, ``retry_same_request_and_seed`` must hold (a retry that re-seeded would be a
    second *sample* of the same logical decision), ``model_output_repair_calls`` must be zero
    and ``wrong_answer_retry`` false (spec 15.3, 4.6).
    """

    def __init__(self, pool: EndpointPool, tokenizer: Any, *, adapter: A.ModelAdapter | None,
                 failure_policy: K.FailurePolicy, top_logprobs: int, **kwargs: Any) -> None:
        if not failure_policy.retry_same_request_and_seed:
            raise ContractViolation(
                "failure_policy.retry_same_request_and_seed is false: an infrastructure retry "
                "that changed the request or the seed would be a second sample of the same "
                "logical decision, not a retry (spec 15.3)")
        if int(failure_policy.model_output_repair_calls) != 0:
            raise ContractViolation(
                "failure_policy.model_output_repair_calls is "
                f"{failure_policy.model_output_repair_calls}: no extra model call may repair a "
                "malformed answer in the primary arms (spec 15.3, 4.6)")
        if failure_policy.wrong_answer_retry:
            raise ContractViolation(
                "failure_policy.wrong_answer_retry is true: a wrong answer is an observation "
                "(W), never a reason to sample again (spec 4.6)")
        if failure_policy.context_overflow != "typed_failure":
            raise ContractViolation(
                f"failure_policy.context_overflow is {failure_policy.context_overflow!r}: an "
                "over-long prompt is a typed failure refused before dispatch, never a retry")
        kwargs.setdefault("max_exogenous_retries", int(failure_policy.transient_retries))
        super().__init__(pool, tokenizer, **kwargs)
        self.adapter = adapter
        self.failure_policy = failure_policy
        self.top_logprobs = int(top_logprobs)
        self._producer: ProducerStamp | None = None
        self._continuation: Any = None
        self._continuation_lock = threading.Lock()

    # ---- wire -----------------------------------------------------------------------
    def request_kwargs(self, spec: RequestSpec,
                       guided_json_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if guided_json_schema is not None or bool(spec.decoding.guided_json):
            raise ContractViolation(
                "no nextdyn request carries a guided field: guided decoding is accepted but "
                "not enforced on vLLM 0.21, so protocol.final_phase_constrained_decoding is "
                "frozen false (plan D6)")
        if self.adapter is None:
            raise ContractViolation("this client has no adapter; it cannot build a chat body")
        kwargs = VllmChatClient.request_kwargs(spec, None)
        extra = self.adapter.request_kwargs(spec.decoding).get("extra_body", {})
        kwargs["extra_body"].update(extra)
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = self.top_logprobs
        return kwargs

    def _parse_response(self, response: Any, local_ids: tuple[int, ...]) -> dict[str, Any]:
        parsed = super()._parse_response(response, local_ids)
        choice = response.choices[0]
        parsed["token_logprobs"] = _token_logprobs(choice)
        parsed["top_logprobs"] = _top_logprobs(choice)
        parsed["cached_prefill_tokens"] = _cached_tokens(getattr(response, "usage", None))
        return parsed

    def stamp(self, producer: ProducerStamp) -> None:
        """Install the producer stamp written into every record this client commits."""
        self._producer = producer

    @property
    def producer(self) -> ProducerStamp | None:
        """The stamp installed by :meth:`stamp` (the phase-2 client inherits the cell's)."""
        return self._producer

    def _build_record(self, spec: RequestSpec, parsed: Mapping[str, Any], **kwargs: Any) -> Any:
        record = super()._build_record(spec, parsed, **kwargs)
        if record.flops.get("total") is None:
            # Refuse here, before ``store.publish``: a committed placeholder would alias for
            # the rest of the run and the study would end with no FLOP number at all.
            raise ProtocolError(
                f"request {record.request_id[:12]} would commit the FLOP placeholder; attach "
                "inference.flop_cost_fn(contract.models) as cost_fn")
        entry = kwargs.get("entry")
        if self._producer is not None and entry is not None:
            producer = self._producer
            record.producer = ProducerStamp(
                cell_id=producer.cell_id, run_id=producer.run_id,
                slurm_job_id=producer.slurm_job_id, host=producer.host,
                endpoint_id=f"{entry.host}:{int(entry.port)}",
                gpu_type=producer.gpu_type, partition=producer.partition,
            ).to_dict()
            record = record.with_content_sha256()
        return record


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _token_logprobs(choice: Any) -> list[float | None] | None:
    """Per-token logprobs of ``choices[0].logprobs.content``, or ``None`` when absent.

    Non-finite values (``-inf`` for an impossible token) become ``None``: the record is
    written through a JSON encoder with ``allow_nan=False``.
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
    """``[{token, logprob}, ...]`` per generated token, most probable first."""
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
        alts = entry.get("top_logprobs") if isinstance(entry, Mapping) else getattr(entry, "top_logprobs", None)
        row: list[dict[str, Any]] = []
        if isinstance(alts, Sequence) and not isinstance(alts, (str, bytes)):
            for alt in alts:
                token = alt.get("token") if isinstance(alt, Mapping) else getattr(alt, "token", None)
                value = _finite(alt.get("logprob") if isinstance(alt, Mapping)
                                else getattr(alt, "logprob", None))
                if token is not None and value is not None:
                    row.append({"token": str(token), "logprob": float(value)})
            row.sort(key=lambda a: (-a["logprob"], a["token"]))
        out.append(row)
    return out


def _cached_tokens(usage: Any) -> int:
    """``usage.prompt_tokens_details.cached_tokens`` -- vLLM's prefix-cache hit count."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    value = (details.get("cached_tokens") if isinstance(details, Mapping)
             else getattr(details, "cached_tokens", None))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def span_logprob_mean(token_logprobs: Sequence[float | None] | None, *, n_completion: int,
                      start: int, stop: int) -> float | None:
    """Mean logprob over completion ids ``[start, stop)``.

    vLLM returns one ``logprobs.content`` entry per generated token, so the list aligns with
    ``token_ids`` by index.  A server reporting only the post-reasoning channel is tolerated by
    matching the span length; anything else yields ``None`` rather than a silently misaligned
    number.
    """
    if not token_logprobs or stop <= start:
        return None
    values = [v for v in token_logprobs[start:stop] if v is not None] \
        if len(token_logprobs) == n_completion else None
    if values is None:
        final_len = n_completion - start
        if len(token_logprobs) != final_len:
            return None
        values = [v for v in token_logprobs[: stop - start] if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def uncertain_execution_of(attempts: Sequence[Mapping[str, Any]]) -> bool:
    """Did an earlier attempt possibly complete on the server (spec 15.3)?

    A timed-out attempt followed by a successful one is the case: the request and the seed
    were identical, so the *sample* is well defined, but the replica may have generated twice
    and the GPU-second ledger must not pretend otherwise.
    """
    return any((a.get("error") or {}).get("type") in UNCERTAIN_ERROR_TYPES for a in attempts)


# --------------------------------------------------------------------------- dispatch


@dataclass
class DispatchContext:
    """Everything a decision needs beyond the request itself (contract §2.1).

    ``pool_for`` is keyed by serving profile and ``tokenizer_for`` by model-cell name, because
    those are the two things that are genuinely per-deployment and per-model; both are
    injected so a dry run and a test can drive the whole path without Slurm or a GPU.
    """

    store: RequestStore
    pool_for: Callable[[str], EndpointPool]
    cost_fn: Callable[..., Mapping[str, Any]]
    tokenizer_for: Callable[[str], Any]
    contract: K.StudyContract
    cell_id: str
    run_id: str
    #: Continuation prefill bytes; ``None`` means "the declaration in ``decoding``".  The
    #: runner passes ``decoding.prefill_text(prompts_lock)`` so the locked bytes are used.
    prefill_text: str | None = None
    #: Per-phase prefills (A9).  ``None`` means "the declarations in ``decoding``"; the runner
    #: passes the locked map so what is re-sent is locked prompt bytes and nothing else.
    phase_prefill_text: Mapping[str, str] | None = None
    _clients: dict[str, NextdynChatClient] = field(default_factory=dict)
    _clients_lock: threading.Lock = field(default_factory=threading.Lock)

    def prefill_for(self, phase: str | None) -> str:
        """The continuation prefill this phase's report opens with (A9).

        One resolver, so the two continuation call sites cannot disagree about which bytes a
        phase gets -- the same R1 reason the prefill itself is declared once.
        """
        if self.phase_prefill_text is not None and phase is not None:
            declared = dict(self.phase_prefill_text).get(str(phase))
            if declared is not None:
                return declared
        if self.prefill_text is not None and str(phase) not in D.PHASE_CONTINUATION_PREFILL_TEXT:
            return self.prefill_text
        return D.prefill_text(phase=phase)

    def tokenizer(self, model: K.ModelCell) -> Any:
        return self.tokenizer_for(model.name)

    def pool(self, model: K.ModelCell) -> EndpointPool:
        return self.pool_for(model.profile)

    def client(self, req: DecisionRequest) -> NextdynChatClient:
        """The cell's client for one model cell, built once and reused.

        One client per decision would mean one OpenAI SDK client -- and its connection pool --
        per decision, i.e. tens of thousands over a run.
        """
        key = f"{req.model.name}|{req.top_logprobs}"
        with self._clients_lock:
            client = self._clients.get(key)
            if client is None:
                pool = self.pool(req.model)
                client = NextdynChatClient(
                    pool, self.tokenizer(req.model),
                    adapter=A.adapter_for(req.model),
                    failure_policy=self.contract.failure_policy,
                    top_logprobs=int(req.top_logprobs),
                    cost_fn=self.cost_fn, run_id=self.run_id,
                )
                client.stamp(ProducerStamp(
                    cell_id=self.cell_id, run_id=self.run_id,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
                    endpoint_id="", gpu_type=None, partition=None))
                self._clients[key] = client
            return client


def request_spec(req: DecisionRequest, ctx: DispatchContext) -> RequestSpec:
    """The identity-bearing :class:`RequestSpec` of ``req`` (R5: nothing outside the id).

    ``seed_key.model_cell`` is the 16-hex model hash the composite key already carries
    (``ids.seed_key_for``), so the check is against the key rather than against a string a
    producer chose.
    """
    model = ctx.contract.model(req.model.name)
    expected = I.model_hash(model)
    if req.seed_key.model_cell != expected:
        raise ContractViolation(
            f"seed_key.model_cell {req.seed_key.model_cell!r} is not model_hash({model.name}) "
            f"= {expected!r}; the generating model is in every key (spec 17.2)")
    if req.key.receiver_model_hash != expected:
        raise ContractViolation(
            f"key.receiver_model_hash {req.key.receiver_model_hash!r} != model_hash({model.name})")
    adapter = A.adapter_for(model)
    decoding = D.decoding_for_contract(ctx.contract, adapter, req.tokens, purpose=req.purpose)
    return RequestSpec(
        messages=tuple(dict(m) for m in req.messages),
        decoding=decoding,
        checkpoint=checkpoint_for(model),
        seed_key=req.seed_key,
        role=D.DECISION_ROLE,
        study_id=ctx.contract.study_id,
        study_seed_hex=ctx.contract.study_seed_hex,
    )


def preflight(req: DecisionRequest, ctx: DispatchContext,
              spec: RequestSpec | None = None) -> ContextPreflight:
    """Refuse an over-long prompt in the harness, before it can reach an endpoint.

    A vLLM 400 is a non-retriable ``InfraFailure`` that kills a cell mid-episode; the refusal
    happens here instead, at the *served window* with the declared reserve, with auditable
    accounting attached to the exception (spec 6.3, plan D5).  ``spec`` is the caller's already
    built :class:`RequestSpec` where it has one, so a decision renders its chat template once.
    """
    spec = request_spec(req, ctx) if spec is None else spec
    profile = get_serving_profile(req.model.profile)
    window = int(req.tokens.served_window)
    if int(profile.max_model_len) < window:
        raise ContractViolation(
            f"serving profile {profile.name!r} serves {profile.max_model_len} tokens but the "
            f"token contract declares a served window of {window}: both compared models must "
            "be served at the same declared window (spec 5.5)")
    try:
        return preflight_chat_context(
            ctx.tokenizer(req.model), list(spec.messages),
            requested_output_tokens=int(spec.decoding.max_tokens),
            served_context=window,
            profile_name=profile.name,
            enable_thinking=bool(spec.decoding.enable_thinking),
            reserve_tokens=int(req.tokens.reserve),
        )
    except ContextCapacityError as exc:
        raise ContextTooLong(exc.preflight, exc) from exc


def decision_from_record(record: Any, split: A.SplitCompletion, *, spec: RequestSpec,
                         tokenizer: Any, replica: ReplicaStamp, aliased: bool) -> DecisionRecord:
    """Delimiter-aware accounting over one committed record (single-call path).

    Works identically on a freshly generated record and on one read back from the store, so a
    resumed cell measures exactly what the original run measured.
    """
    flops = dict(record.flops or {})
    if flops.get("total") is None:
        raise ProtocolError(
            f"request {record.request_id[:12]} carries the FLOP placeholder; attach "
            "inference.flop_cost_fn(contract.models) as cost_fn")
    if record.request_id != spec.request_id:
        raise ProtocolError(
            f"committed record {record.request_id[:12]} is not the spec {spec.request_id[:12]}")
    response = record.response
    ids = tuple(int(i) for i in response.get("token_ids") or ())
    finish_reason = str(response.get("finish_reason"))
    native_text = _decode(tokenizer, split.native_ids)
    final_text = _decode(tokenizer, split.final_ids)
    # The final span starts one token after the closing delimiter; with no closing
    # delimiter there is no final span at all and the window is empty.
    start = (split.delimiter_positions[1] + 1) if split.closed else len(ids) - len(split.final_ids)
    timing = record.timing or {}
    attempts = tuple(timing.get("attempts") or ())
    return DecisionRecord(
        request_ids=(spec.request_id,),
        physical_request_ids=(record.request_id,),
        generation_uid=I.generation_uid([record.request_id]),
        prompt_token_ids=tuple(int(i) for i in record.prompt_token_ids),
        completion_token_ids=ids,
        native_reasoning_text=native_text,
        final_text=final_text,
        thinking_tokens=split.thinking_tokens,
        final_tokens=len(split.final_ids),
        boundary_injected=split.boundary_injected,
        natural_stop=split.natural_stop,
        cap_hit=finish_reason == "length",
        final_overrun=split.overrun_tokens > 0,
        answer_logprob=span_logprob_mean(response.get("token_logprobs"), n_completion=len(ids),
                                         start=start, stop=start + len(split.final_ids)),
        final_top_logprobs=_final_top(response.get("top_logprobs"), n_completion=len(ids),
                                      start=start, length=len(split.final_ids)),
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(response.get("cached_prefill_tokens") or 0),
        finish_reason=finish_reason,
        latency_s=float(timing.get("latency_s") or 0.0),
        endpoint_id=replica.endpoint_id,
        replica_id=replica.replica_id,
        server_batch_id=server_batch_id(replica.endpoint_id, float(timing.get("submitted_at") or 0.0)),
        arrival_ts=float(timing.get("submitted_at") or 0.0),
        gpu_type=replica.gpu_type,
        partition=replica.partition,
        host=replica.host,
        flops=flops,
        continuation=None,
        # What a continuation would replace.  It is zero on every path that keeps the span:
        # the envelope contract never forces a boundary, and a native forced boundary is always
        # continued, so a span is only ever counted here when it was in fact discarded (D-02).
        discarded_span_tokens=split.discarded_span_tokens,
        aliased=bool(aliased),
        uncertain_execution=uncertain_execution_of(attempts),
        attempts=attempts,
    )


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    if not ids:
        return ""
    text = tokenizer.decode(list(ids))
    if not isinstance(text, str):
        raise ProtocolError("tokenizer.decode did not return text")
    return text


def _final_top(block: Any, *, n_completion: int, start: int, length: int) -> Any:
    """Top-k alternatives over the final span only, or ``None`` when only the sampled token
    was scored (every phase but the receiver's; spec 11.2)."""
    if not block or not length:
        return None
    if max((len(row) for row in block), default=0) <= 1:
        return None
    lo = start if len(block) == n_completion else max(0, len(block) - (n_completion - start))
    return tuple(tuple(row) for row in block[lo:lo + length])


def generate_decision(req: DecisionRequest, *, ctx: DispatchContext,
                      on_event: Callable[[str, Mapping[str, Any]], None] | None = None) -> DecisionRecord:
    """One decision -> one :class:`DecisionRecord`, in one or two physical requests.

    Store first (an already-committed request is returned untouched and no GPU work is spent),
    then the adapter's split, then -- if and only if the boundary was forced -- the phase-2
    continuation.  A naturally closed draw is untouched: one request, one id.
    """
    from agents_scaling.nextdyn import continuation as C   # late: continuation needs this module

    model = ctx.contract.model(req.model.name)
    adapter = A.adapter_for(model)
    spec = request_spec(req, ctx)
    preflight(req, ctx, spec)                 # raises ContextTooLong first, never a retry
    tokenizer = ctx.tokenizer(model)
    client = ctx.client(req)
    pool = client.pool
    record, aliased = ctx.store.get_or_generate(spec, client, ctx.cell_id, on_event=on_event)
    replica = replica_of(pool, record)
    split = adapter.split(record, tokenizer)
    decision = decision_from_record(record, split, spec=spec, tokenizer=tokenizer,
                                    replica=replica, aliased=aliased)
    if not (split.boundary_injected and adapter.continuation_supported()):
        return decision
    return C.continue_decision(decision, record, split, spec=spec, req=req, ctx=ctx,
                               client=client, on_event=on_event)


def request_specs_for(req: DecisionRequest, ctx: DispatchContext,
                      record: DecisionRecord) -> tuple[RequestSpec, ...]:
    """Every identity-bearing spec ``record`` was generated from, in phase order.

    ``telemetry.request_rows`` writes one row per physical request and derives the decoding
    hash, the seed and the engine digest from the spec rather than from a second reading of the
    wire.  A continued decision has two specs, and phase 2's is rebuilt here -- from the
    committed phase-1 record and the prefill ids the merge recorded -- so a caller never has to
    keep one alive across the two calls, and a resumed cell derives exactly what the original
    run wrote.
    """
    from agents_scaling.nextdyn import continuation as C

    spec = request_spec(req, ctx)
    if record.continuation is None:
        return (spec,)
    model = ctx.contract.model(req.model.name)
    tokenizer = ctx.tokenizer(model)
    committed = ctx.store.get(record.physical_request_ids[0])
    if committed is None:
        raise ProtocolError(
            f"phase-1 record {record.physical_request_ids[0][:12]} is not in the store; a "
            "continued decision's specs cannot be rebuilt without it")
    cc = C.ContinuationContract.of(req.tokens, ctx.prefill_for(req.purpose))
    _start, think_end = A.delimiter_ids(tokenizer)
    spec2 = C.continuation_spec(spec, committed,
                                prefill_ids=record.continuation["prefill_token_ids"], cc=cc,
                                sampling=ctx.contract.sampling, think_end_id=think_end)
    return (spec, spec2)


def replica_of(pool: EndpointPool, record: Any) -> ReplicaStamp:
    """Where the committed record ran, as the pool understands that endpoint.

    A record read back from the store carries the endpoint it ran on originally, which may no
    longer be live; the stamp is rebuilt from the record so a resumed cell reports the replica
    that actually did the work.
    """
    endpoint = record.endpoint or {}
    host, port = str(endpoint.get("host")), int(endpoint.get("port") or 0)
    job_id = endpoint.get("slurm_job_id")
    entry = ServerEntry(model_size="", hf_id="", host=host, port=port, slurm_job_id=job_id,
                        serving_profile=endpoint.get("serving_profile"))
    if isinstance(pool, NextdynEndpointPool):
        return pool.replica_stamp(entry)
    return ReplicaStamp.of(entry)


__all__ = [
    "ARRIVAL_BUCKET_S", "COMPLETED_FINISH_REASONS", "AttemptInfo", "ContextTooLong",
    "ContinuationOutcome", "DecisionRecord", "DecisionRequest", "DispatchContext", "Generate",
    "JobInfo", "NextdynChatClient", "NextdynEndpointPool", "ProducerStamp", "ReplicaStamp",
    "UNCERTAIN_ERROR_TYPES", "checkpoint_for", "decision_from_record", "flop_cost_fn", "generate_decision", "preflight", "replica_of", "request_spec",
    "request_specs_for", "server_batch_id",
    "span_logprob_mean", "uncertain_execution_of",
]
