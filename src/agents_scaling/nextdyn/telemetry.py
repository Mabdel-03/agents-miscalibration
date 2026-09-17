"""Rows and cost: the one site that builds a ``decisions`` row, and the compute ledger.

**R6, the single constructor.**  ARD36's defect D-02 was not a wrong number, it was a second
place that built a row carrying physical ids: two constructors drifted, and the phase-2 request
of every continued draw fell out of the cost ledger entirely -- its prefill, its decode and the
phase-1 span it replaced were never attributed to anything.  :func:`decision_row` is therefore
the **only** ``DecisionRow`` constructor in the package (AST-checked), and
:func:`request_rows` emits one ``requests`` row per *physical* request, in phase order, so a
two-request decision is two rows and a one-request decision is one.

**The D-02 fix, structurally.**  :func:`request_resources` builds the GPU-second attribution
inputs from the ``requests`` rows rather than from ``decisions``.  That is the whole fix: a
continued draw contributes both its phase-1 row (whose completion tokens include the
post-boundary span the continuation discarded -- real generated tokens, real GPU seconds) and
its phase-2 row (whose prompt is the re-sent prefix).  Nothing is dropped because nothing is
read at decision grain.

**Attribution is an approximation, and says so.**  The OpenAI-compatible API exposes no batch
id, so ``server_batch_id = (endpoint_id, one-second arrival bucket)`` and a request receives
its wall time divided by the time-weighted number of concurrently in-flight requests on that
replica (``ard.resources.attribute_gpu_seconds``, imported).  The un-attributed server-level
totals are kept verbatim beside it, and :func:`conservation_report` states the three
inequalities that make the approximation auditable: attributed <= union busy <= allocated, per
replica, with coverage recorded explicitly rather than left as a silent NULL.

**Campaign cost and per-system cost are different questions** (spec 16.1-16.2):
:func:`campaign_cost` is what the study spent, :func:`per_system_cost` is what one protocol
costs per delivered item, which is the number the efficiency frontier compares.  They are
separate functions over separate groupings so neither can be quoted as the other.

Spec: 4.5, 4.6 (parse/delivery/infrastructure are separate fields), 6.1, 16.1-16.2, 17.3, 18.5.
Plan: §1.4, §3.7, D5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from agents_scaling.ard.resources import (
    attribute_gpu_seconds,
    capture_sacct_gpu_seconds,
    parse_nvidia_smi_csv,
)
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn.errors import MaterializeError
from agents_scaling.nextdyn.inference import (
    ARRIVAL_BUCKET_S,
    DecisionRecord,
    ProducerStamp,
    server_batch_id,
    uncertain_execution_of,
)
from agents_scaling.study.types import RequestSpec

if TYPE_CHECKING:  # pragma: no cover - WP2 owns the parser; telemetry only reads its fields
    from agents_scaling.nextdyn.schema import PublicOutput

# --------------------------------------------------------------------------- vocabularies

#: ``decisions.infrastructure_status`` (spec 4.6: missingness is not semantic error).  A
#: malformed answer is *not* an infrastructure failure -- that distinction is the whole point
#: of keeping the fields apart -- so a committed record is ``ok`` unless an attempt may have
#: run twice.
INFRA_OK: Final[str] = "ok"
INFRA_UNCERTAIN: Final[str] = "uncertain_execution"
INFRA_CONTEXT_FAILURE: Final[str] = "context_failure"
INFRASTRUCTURE_STATUSES: Final[tuple[str, ...]] = (INFRA_OK, INFRA_UNCERTAIN, INFRA_CONTEXT_FAILURE)

#: ``decisions.logprob_norm_scheme``: how ``answer_logprob_mean`` was normalised.  Recorded
#: rather than assumed, because a mean over a final span and a sum over one are different
#: numbers and the calibration panel consumes whichever it is told.
LOGPROB_NORM_MEAN_FINAL_SPAN: Final[str] = "mean_over_final_span"

#: ``decisions.termination_reason``: vLLM's terminal ``finish_reason`` for the request that
#: produced the *delivered* span.  ``boundary_injected`` and ``continuation_used`` are their own
#: columns, so this one stays the engine's own word (R1).
TERMINATION_REASONS: Final[tuple[str, ...]] = ("stop", "length")

#: The attribution rule, recorded on every attributed row so a reader never has to guess which
#: approximation produced the number (spec 18; plan §3.7).
ATTRIBUTION_RULE: Final[str] = (
    "server_batch_id = (endpoint_id, floor(arrival_ts / 1 s)); a physical request receives its "
    "wall time divided by the time-weighted number of concurrently in-flight requests on that "
    "replica.  Inputs are the requests rows, so a continued decision contributes both physical "
    "requests -- the phase-2 prefill and the discarded post-boundary span included (D-02).")


# --------------------------------------------------------------------------- row context


@dataclass(frozen=True)
class CellContext:
    """What a ``decisions`` row needs that neither the record nor the composite key carries.

    ``episode_id`` is the caller's because only the caller knows which id builder applies:
    ``ids.episode_id`` for EPISODE/RECEIVER, ``ids.bank_draw_id`` for a bank kind,
    ``ids.audit_episode_id`` for SELECTION_AUDIT (R4 -- and defect D-11 was exactly a bank draw
    wearing an episode's id).  ``display_label`` is the human name of the arm and never enters
    any hash; ``bundle_export_hash`` ties an episode row to the label-free root bundle it was
    generated from.
    """

    cell_id: str
    episode_id: str
    display_label: str | None
    bundle_export_hash: str | None

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise MaterializeError("CellContext.cell_id is empty")
        if not self.episode_id:
            raise MaterializeError("CellContext.episode_id is empty")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def infrastructure_status(record: DecisionRecord) -> str:
    """Spec 4.6's ``infrastructure_status`` for one committed record.

    A possibly-completed timeout followed by a successful retry is ``uncertain_execution``: the
    request and the seed were identical, so the *sample* is well defined, but the replica may
    have generated twice and the ledger must not pretend otherwise.
    """
    return INFRA_UNCERTAIN if record.uncertain_execution else INFRA_OK


def decision_row(record: DecisionRecord, key: I.CompositeKey, out: "PublicOutput", *,
                 cell: CellContext, parents: tuple[str, ...],
                 seen_transmissions: tuple[str, ...], aliased: bool, reuse_of: str | None,
                 is_reference: bool = False,
                 native_reasoning_ref: str | None = None, plan_parsed: bool | None = None) -> T.DecisionRow:
    """THE only ``DecisionRow`` constructor (R6).

    ``canonical_class`` is ``None`` here and always: grading is writer-side and joins on
    ``generation_uid`` afterwards, so a generation-side row that already carried a correctness
    class would be a gold leak into the generation closure (R7).

    Spec 4.6 keeps five things apart and so does this row: ``parse_status`` (did the object
    parse), ``validity`` (was an answer extractable at all), ``delivery_success`` (operational
    delivery -- an answer, produced without an uncertain execution), ``infrastructure_status``
    and ``termination_reason``.  A refusal or a malformed object scores zero delivery success
    without manufacturing a semantic transition.

    ``answer_logprob_sum`` is ``mean * answer_token_count`` by construction:
    :class:`~agents_scaling.nextdyn.inference.DecisionRecord` defines ``answer_logprob`` as the
    mean over exactly the ``final_tokens`` delivered tokens, so the two agree rather than being
    measured twice over slightly different spans (R1).
    """
    if not isinstance(key, I.CompositeKey):
        raise MaterializeError("decision_row needs the twelve-field CompositeKey")
    if record.generation_uid != I.generation_uid(record.physical_request_ids):
        raise MaterializeError(
            f"record.generation_uid {record.generation_uid[:14]} is not the uid of its physical "
            f"request ids {list(record.physical_request_ids)} (one sample, one uid)")
    mean = record.answer_logprob
    count = int(record.final_tokens)
    valid = out.final_answer is not None
    status = infrastructure_status(record)
    return T.DecisionRow(
        decision_id=key.decision_id(),
        study_id=key.study_id,
        cell_kind=key.cell_kind.value,
        source_item_id=key.source_item_id,
        item_variant_id=key.item_variant_id,
        root_team_id=key.root_team_id,
        protocol_hash=key.protocol_hash,
        receiver_model_hash=key.receiver_model_hash,
        budget_hash=key.budget_hash,
        replicate_id=int(key.replicate_id),
        round=int(key.round),
        agent_slot=int(key.agent_slot),
        phase=key.phase,
        episode_id=cell.episode_id,
        cell_id=cell.cell_id,
        display_label=cell.display_label,
        generation_uid=record.generation_uid,
        content_hash=I.content_hash(record.final_text.encode("utf-8")),
        request_hashes_json=_json(list(record.physical_request_ids)),
        aliased=bool(aliased),
        reuse_of=reuse_of,
        is_reference=bool(is_reference),
        first_decision_id=None,          # materialize fills it per generation_uid
        bundle_export_hash=cell.bundle_export_hash,
        parent_decision_ids_json=_json(list(parents)),
        seen_transmission_ids_json=_json(list(seen_transmissions)),
        native_reasoning_ref=native_reasoning_ref,
        public_reasoning=out.reasoning,
        shareable_claims_json=None if out.shareable_claims is None else _json(list(out.shareable_claims)),
        final_answer_raw=out.final_answer,
        canonical_class=None,            # writer-side (R7); grades join on generation_uid
        confidence=out.confidence,
        answer_logprob_sum=None if mean is None else float(mean) * count,
        answer_logprob_mean=mean,
        answer_token_count=count,
        logprob_norm_scheme=None if mean is None else LOGPROB_NORM_MEAN_FINAL_SPAN,
        parse_status=str(getattr(out.parse_status, "value", out.parse_status)),
        escape_repaired=bool(getattr(out, "escape_repaired", False)),
        trailing_chars=int(getattr(out, "trailing_chars", 0) or 0),
        plan_parsed=plan_parsed,
        field_order_violation=bool(out.order_violation),
        validity=valid,
        delivery_success=bool(valid and status == INFRA_OK),
        infrastructure_status=status,
        termination_reason=record.finish_reason,
        boundary_injected=bool(record.boundary_injected),
        continuation_used=record.continuation is not None,
        discarded_span_tokens=int(record.discarded_span_tokens),
        thinking_tokens=int(record.thinking_tokens),
        final_tokens=count,
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(record.cached_prefill_tokens),
        token_offsets_json=_json({
            "final_answer_span": list(out.final_answer_span) if out.final_answer_span else None,
            "confidence_span": list(out.confidence_span) if out.confidence_span else None,
            "confidence_emitted_before_answer": bool(out.confidence_emitted_before_answer),
        }),
        endpoint_id=record.endpoint_id,
        latency_s=float(record.latency_s),
        flops_prefill=_int_or_none(record.flops.get("prefill")),
        flops_decode=_int_or_none(record.flops.get("decode")),
        flops_total=_int_or_none(record.flops.get("total")),
        # Null here for the same reason as ``first_decision_id``: both need facts no single
        # decision carries.  ``materialize`` fills this one through
        # :func:`linear_compute_proxy`, whose parameter count is a property of the model cell,
        # not of the row's key (which carries only the model's hash).
        linear_compute_proxy=None,
    )


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def request_rows(record: DecisionRecord, key: I.CompositeKey, *, cell_id: str,
                 request_specs: Sequence[RequestSpec],
                 producer: ProducerStamp) -> tuple[T.RequestRow, ...]:
    """One ``requests`` row per *physical* request of one decision, in phase order.

    The derived columns come from the spec, not from a second reading of the wire: the decoding
    hash, the seed, the model and tokenizer revisions and the engine digest are all properties
    of ``RequestSpec`` and therefore already inside the request id.  ``first_decision_id`` is
    ``None`` at write time; materialize fills it per ``generation_uid`` once every cell that
    aliased the sample is known.

    A continued decision is two rows.  That is what makes the D-02 fix structural: everything
    downstream that meters or attributes reads this table, so the phase-2 prefill and the
    phase-1 span it replaced cannot fall out of the ledger without a row going missing.
    """
    if len(request_specs) != len(record.physical_request_ids):
        raise MaterializeError(
            f"{len(request_specs)} request specs for {len(record.physical_request_ids)} physical "
            "requests; one row per physical request, in phase order")
    if tuple(record.request_ids) != tuple(s.request_id for s in request_specs):
        raise MaterializeError(
            "request_specs are not the specs this record was generated from "
            f"({list(record.request_ids)} != {[s.request_id for s in request_specs]})")
    continuation = record.continuation or {}
    n_phase2 = int(continuation.get("n_attempts", 0))
    attempts = tuple(record.attempts)
    phase1_attempts = attempts[: len(attempts) - n_phase2] if continuation else attempts
    phase2_attempts = attempts[len(attempts) - n_phase2:] if continuation else ()
    rows: list[T.RequestRow] = []
    for index, (spec, request_id) in enumerate(zip(request_specs, record.physical_request_ids)):
        phase_two = index == 1
        source: Mapping[str, Any] = continuation if phase_two else {}
        arrival = float(source["arrival_ts"]) if phase_two else record.arrival_ts
        latency = float(source["latency_s"]) if phase_two else record.latency_s
        endpoint_id = str(source["endpoint_id"]) if phase_two else record.endpoint_id
        own_attempts = phase2_attempts if phase_two else phase1_attempts
        rows.append(T.RequestRow(
            request_hash=request_id,
            generation_uid=record.generation_uid,
            phase_index=index,
            first_decision_id=None,
            prompt_token_hash=_ids_hash(list(spec.prompt_ids) if phase_two
                                        else list(record.prompt_token_ids)),
            completion_token_hash=_ids_hash(_phase_completion_ids(record, phase_two=phase_two)),
            decoding_hash=spec.decoding_hash,
            seed_hex=spec.semantic_seed_hex,
            model_hash=key.receiver_model_hash,
            tokenizer_hash=spec.checkpoint.tokenizer_revision,
            engine_digest=spec.engine_digest,
            adapter=_adapter_of(spec),
            endpoint_id=endpoint_id,
            replica_id=str(source["replica_id"]) if phase_two else record.replica_id,
            host=str(source["host"]) if phase_two else record.host,
            gpu_type=source["gpu_type"] if phase_two else record.gpu_type,
            partition=source["partition"] if phase_two else record.partition,
            server_batch_id=server_batch_id(endpoint_id, arrival),
            arrival_ts=arrival,
            finish_ts=arrival + latency,
            attempts_json=_json([_attempt_dict(a) for a in own_attempts]),
            uncertain_execution=uncertain_execution_of([_attempt_dict(a) for a in own_attempts]),
            aliased=bool(source["aliased"]) if phase_two
            else bool(continuation.get("first_aliased", record.aliased)),
            prompt_tokens=int(source["prompt_tokens"]) if phase_two else record.prompt_tokens,
            completion_tokens=int(source["generated_tokens"]) if phase_two
            else _phase_one_generated(record),
            cached_prefill_tokens=int(source["cached_prefill_tokens"]) if phase_two
            else record.cached_prefill_tokens,
            latency_s=latency,
            cell_id=cell_id,
            producer_json=_json(producer.to_dict()),
        ))
    return tuple(rows)


def _adapter_of(spec: RequestSpec) -> str:
    """Which generation contract issued this request, read off what was actually sent."""
    if int(spec.decoding.thinking_tokens) > 0:
        return K.ADAPTER_NATIVE_THINKING
    if spec.decoding.enable_thinking:
        return K.ADAPTER_NATIVE_THINKING
    return K.ADAPTER_TOTAL_OUTPUT_ENVELOPE


def _ids_hash(ids: Sequence[int]) -> str:
    from agents_scaling.study import identity

    return identity.sha256_hex(identity.jcs([int(i) for i in ids]))


def _continued_tail(record: DecisionRecord) -> int:
    """Tokens of the merged completion that belong to phase 2: its prefill plus what it
    generated.  Zero on the single-call path."""
    if not record.continuation:
        return 0
    return (len(record.continuation["prefill_token_ids"])
            + int(record.continuation["generated_tokens"]))


def generated_tokens(record: DecisionRecord) -> int:
    """Every token the engine decoded for one decision, both phases and the discarded span.

    Equal to the sum of the ``requests`` rows' ``completion_tokens`` by construction, so the
    per-decision meter and the per-request ledger cannot disagree (R1).  It is *larger* than
    ``thinking_tokens + final_tokens + discarded_span_tokens`` by the two reasoning delimiters,
    which are genuinely sampled tokens (amendment A1 buys ``max_tokens`` for them) and belong to
    neither span.
    """
    return _phase_one_generated(record) + int((record.continuation or {}).get("generated_tokens", 0))


def _phase_one_generated(record: DecisionRecord) -> int:
    """How many tokens phase 1 actually decoded -- the D-02 number.

    The merged completion stops at the forced ``</think>``, so counting it would price phase 1
    at the span the continuation kept and silently drop the post-boundary span the continuation
    replaced.  Those tokens were decoded on a GPU and this is the count the attribution reads,
    so the discarded span is added back.  (The turn token is outside every span by the same rule
    everywhere in this package, so it is outside this count too.)
    """
    ids = len(record.completion_token_ids) - _continued_tail(record)
    return ids + int(record.discarded_span_tokens)


def _phase_completion_ids(record: DecisionRecord, *, phase_two: bool) -> list[int]:
    """The completion ids of one physical request, out of the merged logical completion.

    The merged completion is phase 1 through the forced ``</think>``, then the prefilled
    opening, then phase 2's report.  Phase 1's own row therefore hashes everything before the
    prefill and phase 2's hashes what it generated.  Phase 1's *discarded* post-boundary span is
    in neither -- it is preserved whole in the phase-1 store record, and its length rides on
    ``decisions.discarded_span_tokens`` so the ledger still prices it (defect D-02).
    """
    ids = list(record.completion_token_ids)
    tail = _continued_tail(record)
    if not phase_two:
        return ids[: len(ids) - tail]
    return ids[len(ids) - int(record.continuation["generated_tokens"]):]


def _attempt_dict(attempt: Any) -> dict[str, Any]:
    return attempt if isinstance(attempt, Mapping) else attempt.to_dict()


# --------------------------------------------------------------------------- metering


@dataclass(frozen=True)
class Metering:
    """What one decision cost, by phase (spec 4.5 "Cost", plan §3.7).

    ``unused_thinking_tokens`` / ``unused_final_tokens`` are the allowance the draw did not
    spend: the efficiency frontier reports unused allowance beside delivered accuracy, and a
    protocol that systematically leaves budget on the table is a different finding from one
    that is cheap.
    """

    calls: int
    thinking_tokens: int
    final_tokens: int
    discarded_span_tokens: int
    generated_tokens: int
    prompt_tokens: int
    cached_prefill_tokens: int
    uncached_prompt_tokens: int
    latency_s: float
    tool_elapsed_s: float
    flops_prefill: int | None
    flops_decode: int | None
    flops_total: int | None
    unused_thinking_tokens: int
    unused_final_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def metering(record: DecisionRecord, tokens: K.TokenContract, *,
             tool_elapsed_s: float = 0.0) -> Metering:
    """The per-decision cost row (spec 4.5).

    ``generated_tokens`` counts every token the GPUs produced for this decision -- the two
    sampled reasoning delimiters and the discarded post-boundary span included -- and equals the
    sum of this decision's ``requests`` rows' ``completion_tokens``.  Pricing the discarded span
    at zero is exactly the ARD36 understatement (D-02).  ``prompt_tokens`` is phase 1's logical prompt; the phase-2
    prefix is counted as its own request in ``requests`` rather than folded in here, where it
    would corrupt "prefill actually paid" (a phase-2 prefix is a near-total cache hit).
    """
    continuation = record.continuation or {}
    return Metering(
        calls=len(record.physical_request_ids),
        thinking_tokens=int(record.thinking_tokens),
        final_tokens=int(record.final_tokens),
        discarded_span_tokens=int(record.discarded_span_tokens),
        generated_tokens=generated_tokens(record),
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(record.cached_prefill_tokens)
        + int(continuation.get("cached_prefill_tokens", 0)),
        uncached_prompt_tokens=max(0, int(record.prompt_tokens) - int(record.cached_prefill_tokens))
        + max(0, int(continuation.get("prompt_tokens", 0))
              - int(continuation.get("cached_prefill_tokens", 0))),
        latency_s=float(record.latency_s),
        tool_elapsed_s=float(tool_elapsed_s),
        flops_prefill=_int_or_none(record.flops.get("prefill")),
        flops_decode=_int_or_none(record.flops.get("decode")),
        flops_total=_int_or_none(record.flops.get("total")),
        unused_thinking_tokens=max(0, int(tokens.thinking_cap) - int(record.thinking_tokens)),
        unused_final_tokens=max(0, int(tokens.final_cap) - int(record.final_tokens)),
    )


def linear_compute_proxy(param_count: int, uncached_prompt: int, generated: int) -> int:
    """``2 * P * (uncached prompt + generated)`` -- the reported model-compute estimate.

    Deliberately linear and deliberately *not* called FLOPs: spec 18's rule is "do not claim
    exact per-request FLOPs from token count alone".  The quadratic attention term lives in the
    analytic oracle (``flops_*``); this number is the comparable, arch-only proxy that survives
    being compared across models, and it prices the tokens the engine actually had to compute
    (cached prefill excluded, discarded span included).
    """
    for name, value in (("param_count", param_count), ("uncached_prompt", uncached_prompt),
                        ("generated", generated)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaterializeError(f"linear_compute_proxy: {name} must be a non-negative int")
    return 2 * int(param_count) * (int(uncached_prompt) + int(generated))


# --------------------------------------------------------------------------- attribution


@dataclass(frozen=True)
class RequestResource:
    """One physical request, as the GPU-second attribution sees it.

    Structurally the input ``ard.resources.attribute_gpu_seconds`` expects (only
    ``attribute_gpu_seconds`` itself is on the ``ARD_IMPORTS`` allowlist, not its row type), and
    built from ``requests`` rows so both phases of a continued decision are present.
    """

    request_id: str
    endpoint_id: str
    arrival_ts: float
    latency_s: float
    prompt_tokens: int = 0
    cached_prefill_tokens: int = 0
    decode_tokens: int = 0
    retries: int = 0
    errors: int = 0
    decision_id: str | None = None

    @property
    def prefill_tokens(self) -> int:
        """Physical prefill: the logical prompt minus what the prefix cache already held."""
        return max(0, int(self.prompt_tokens) - int(self.cached_prefill_tokens))

    @property
    def end_ts(self) -> float:
        return float(self.arrival_ts) + max(0.0, float(self.latency_s))


def request_resources(rows: Iterable[T.RequestRow],
                      decision_of: Mapping[str, str] | None = None) -> list[RequestResource]:
    """Attribution inputs from the ``requests`` table -- the D-02 fix, structurally.

    Reading ``requests`` rather than ``decisions`` is the whole difference: a continued draw has
    two rows here, so its phase-2 prefill is priced and the phase-1 post-boundary span it
    replaced is inside phase 1's ``completion_tokens``.  ``decision_of`` maps a request hash to
    the decision it served, when the caller wants per-decision roll-ups.
    """
    lookup = dict(decision_of or {})
    out: list[RequestResource] = []
    for row in rows:
        attempts = json.loads(row.attempts_json) if row.attempts_json else []
        errors = sum(1 for a in attempts if isinstance(a, Mapping) and a.get("error"))
        out.append(RequestResource(
            request_id=row.request_hash,
            endpoint_id=row.endpoint_id,
            arrival_ts=float(row.arrival_ts),
            latency_s=float(row.latency_s),
            prompt_tokens=int(row.prompt_tokens),
            cached_prefill_tokens=int(row.cached_prefill_tokens),
            decode_tokens=int(row.completion_tokens),
            retries=max(0, len(attempts) - 1),
            errors=errors,
            decision_id=lookup.get(row.request_hash),
        ))
    return out


def attribute(rows: Iterable[T.RequestRow], decision_of: Mapping[str, str] | None = None
              ) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """``(per-request attributed rows, per-replica server totals)`` for ``resources.parquet``.

    The attribution itself is ``ard.resources.attribute_gpu_seconds`` (imported, not vendored);
    what this study adds is the input set.  Every row is stamped with
    :data:`ATTRIBUTION_RULE`, so a reader never has to infer which approximation produced it.
    """
    attributed, totals = attribute_gpu_seconds(request_resources(rows, decision_of))
    for row in attributed:
        row["attribution_rule"] = ATTRIBUTION_RULE
    return attributed, totals


@dataclass(frozen=True)
class ReplicaRef:
    """A replica, as the registry and Slurm each name it (the join key of the ledger)."""

    endpoint_id: str
    host: str
    slurm_job_id: str | None


@dataclass(frozen=True)
class ConservationRow:
    """One replica's three-way accounting (spec 18: retain server-level totals).

    ``coverage`` is a word, never a silent NULL: ``sampled`` (nvidia-smi rows joined),
    ``allocated_only`` (sacct but no samples), ``unsampled`` (neither).  An unsampled replica is
    a known gap in the ledger, which is a different thing from a replica that idled.
    """

    endpoint_id: str
    n_requests: float
    attributed_seconds: float
    server_busy_seconds: float
    summed_request_seconds: float
    allocated_gpu_seconds: float | None
    mean_utilization: float | None
    n_samples: int
    coverage: str
    attributed_within_busy: bool
    busy_within_allocated: bool | None


def conservation_report(totals: Mapping[str, Mapping[str, float]], *,
                        replicas: Sequence[ReplicaRef] = (),
                        gpu_samples: Sequence[Mapping[str, Any]] = (),
                        sacct_rows: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """``conservation_report.json``: attributed <= union busy <= allocated, per replica.

    The middle term is the *union* of the request intervals on a replica, never their sum: a
    batched replica serving eight streams at once is busy once, and comparing the sum against
    ``sacct``'s allocated GPU-seconds would fail an inequality that was never supposed to hold.
    A violated inequality is reported, not raised -- the report is the artifact that makes the
    approximation auditable, and a run that ends with one bad replica should still produce it.
    """
    by_job: dict[str, float] = {}
    for row in sacct_rows:
        job = str(row.get("job_id") or "")
        by_job[job] = by_job.get(job, 0.0) + float(row.get("allocated_gpu_seconds") or 0.0)
    samples_by_job: dict[str, list[float]] = {}
    for sample in gpu_samples:
        job = str(sample.get("job_id") or "")
        value = sample.get("utilization")
        if value is not None:
            samples_by_job.setdefault(job, []).append(float(value))
    refs = {r.endpoint_id: r for r in replicas}

    rows: list[ConservationRow] = []
    for endpoint, total in sorted(totals.items()):
        ref = refs.get(endpoint)
        job = str(ref.slurm_job_id) if ref is not None and ref.slurm_job_id is not None else None
        allocated = by_job.get(job) if job is not None else None
        samples = samples_by_job.get(job or "", [])
        busy = float(total.get("server_busy_seconds", 0.0))
        attributed = float(total.get("attributed_seconds", 0.0))
        coverage = "sampled" if samples else ("allocated_only" if allocated is not None else "unsampled")
        rows.append(ConservationRow(
            endpoint_id=endpoint,
            n_requests=float(total.get("n_requests", 0.0)),
            attributed_seconds=attributed,
            server_busy_seconds=busy,
            summed_request_seconds=float(total.get("summed_request_seconds", 0.0)),
            allocated_gpu_seconds=allocated,
            mean_utilization=(sum(samples) / len(samples)) if samples else None,
            n_samples=len(samples),
            coverage=coverage,
            attributed_within_busy=attributed <= busy + _TOL,
            busy_within_allocated=None if allocated is None else busy <= allocated + _TOL,
        ))
    violations = [r.endpoint_id for r in rows
                  if not r.attributed_within_busy or r.busy_within_allocated is False]
    return {
        "attribution_rule": ATTRIBUTION_RULE,
        "arrival_bucket_s": ARRIVAL_BUCKET_S,
        "n_replicas": len(rows),
        "n_unsampled_replicas": sum(1 for r in rows if r.coverage == "unsampled"),
        "attributed_seconds": math.fsum(r.attributed_seconds for r in rows),
        "server_busy_seconds": math.fsum(r.server_busy_seconds for r in rows),
        "allocated_gpu_seconds": math.fsum(r.allocated_gpu_seconds or 0.0 for r in rows),
        "conserves": not violations,
        "violations": violations,
        "replicas": [r.__dict__ for r in rows],
    }


_TOL: Final[float] = 1e-6


# --------------------------------------------------------------------------- the two cost views


def campaign_cost(meterings: Iterable[Metering], attributed: Iterable[Mapping[str, Any]]
                  ) -> dict[str, Any]:
    """What the *study* spent (``cost_campaign.parquet``; spec 16.2).

    One total over everything that ran, pilots and discarded spans included.  It answers "did
    this fit the authorized ceiling", and it is never the number a protocol comparison quotes.
    """
    meterings = list(meterings)
    return {
        "scope": "campaign",
        "n_decisions": len(meterings),
        "n_calls": sum(m.calls for m in meterings),
        "generated_tokens": sum(m.generated_tokens for m in meterings),
        "discarded_span_tokens": sum(m.discarded_span_tokens for m in meterings),
        "uncached_prompt_tokens": sum(m.uncached_prompt_tokens for m in meterings),
        "flops_total": sum((m.flops_total or 0) for m in meterings),
        "attributed_gpu_seconds": math.fsum(
            float(r.get("attributed_gpu_seconds") or 0.0) for r in attributed),
    }


def per_system_cost(meterings_by_system: Mapping[str, Sequence[Metering]],
                    delivered_by_system: Mapping[str, int]) -> list[dict[str, Any]]:
    """What one *system* costs per delivered item (``cost_per_system.parquet``; spec 16.1).

    The efficiency frontier compares protocols at equal cost, so the denominator is the number
    of delivered items of that system, not the number of calls it happened to make.  A system
    with no delivered item has a null unit cost -- never a zero, which would read as free.
    """
    out: list[dict[str, Any]] = []
    for system in sorted(meterings_by_system):
        meterings = list(meterings_by_system[system])
        delivered = int(delivered_by_system.get(system, 0))
        flops = sum((m.flops_total or 0) for m in meterings)
        generated = sum(m.generated_tokens for m in meterings)
        out.append({
            "scope": "system",
            "system": system,
            "n_decisions": len(meterings),
            "n_calls": sum(m.calls for m in meterings),
            "delivered_items": delivered,
            "generated_tokens": generated,
            "flops_total": flops,
            "flops_per_delivered_item": (flops / delivered) if delivered else None,
            "generated_tokens_per_delivered_item": (generated / delivered) if delivered else None,
            "unused_thinking_tokens": sum(m.unused_thinking_tokens for m in meterings),
            "unused_final_tokens": sum(m.unused_final_tokens for m in meterings),
        })
    return out


__all__ = [
    "ATTRIBUTION_RULE", "CellContext", "ConservationRow", "INFRASTRUCTURE_STATUSES", "INFRA_OK",
    "INFRA_CONTEXT_FAILURE", "INFRA_UNCERTAIN", "LOGPROB_NORM_MEAN_FINAL_SPAN", "Metering",
    "ProducerStamp", "ReplicaRef", "RequestResource", "TERMINATION_REASONS", "attribute",
    "attribute_gpu_seconds", "campaign_cost", "capture_sacct_gpu_seconds", "conservation_report",
    "decision_row", "generated_tokens", "infrastructure_status", "linear_compute_proxy",
    "metering",
    "parse_nvidia_smi_csv", "per_system_cost", "request_resources", "request_rows",
]
