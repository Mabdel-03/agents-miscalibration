"""The per-question result record (one JSONL row) and the per-cell meta record.

One row captures everything needed to (a) score performance, (b) compute efficiency, and
(c) compute calibration under multiple confidence definitions — both per-agent and at the
system level — without re-running anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agents_scaling.agents.message_builder import (
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_COT_CHAR_LIMIT,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
)
from agents_scaling.serving.client import (
    GENERATION_CENSOR_PROTOCOL_HASH,
    GENERATION_CENSOR_PROTOCOL_VERSION,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
)
from agents_scaling.experiment.transport_censor import (
    TRANSPORT_CENSOR_PROTOCOL_HASH,
    TRANSPORT_CENSOR_PROTOCOL_VERSION,
)


ARTIFACT_SCHEMA_VERSION = 5
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({4, ARTIFACT_SCHEMA_VERSION})

TERMINATION_COMPLETED = "completed"
TERMINATION_LENGTH_CENSORED = "length_censored"
TERMINATION_PROTOCOL_CENSORED = "protocol_censored"
TERMINATION_TRANSPORT_CENSORED = "transport_censored"
SCHEMA_4_TERMINATION_STATUSES = frozenset(
    {TERMINATION_COMPLETED, TERMINATION_LENGTH_CENSORED}
)
TERMINATION_STATUSES = frozenset(
    {
        TERMINATION_COMPLETED,
        TERMINATION_LENGTH_CENSORED,
        TERMINATION_PROTOCOL_CENSORED,
        TERMINATION_TRANSPORT_CENSORED,
    }
)

_COORDINATE_PROVENANCE_FIELDS = (
    "capacity_generation",
    "endpoint_generation",
    "fleet_contract_sha256",
    "release_fleet_contract_sha256",
    "rollout_generation",
)


def _empty_coordinate_provenance_counts() -> dict[str, dict[str, int]]:
    return {field: {} for field in _COORDINATE_PROVENANCE_FIELDS}


def termination_statuses_for_schema(schema_version: int) -> frozenset[str]:
    if schema_version == 4:
        return SCHEMA_4_TERMINATION_STATUSES
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        return TERMINATION_STATUSES
    raise ValueError(f"unsupported artifact schema {schema_version!r}")

SELF_CONSISTENCY_PROTOCOL_V1_VERSION = 1
_SELF_CONSISTENCY_PROTOCOL_SPEC = {
    "scheduled_seed": "cell.seed + 1000 + sample_index",
    "outcome_contract": "one completed AgentOutput or one full length censor per sample",
    "aggregate_censor_policy": "all confidence/entropy aggregates undefined if any censor",
    "cost_accounting": "auxiliary generation costs separate from topology efficiency",
    "replacement_sampling": "forbidden",
}
SELF_CONSISTENCY_PROTOCOL_V1_HASH = hashlib.sha256(
    json.dumps(
        _SELF_CONSISTENCY_PROTOCOL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

SELF_CONSISTENCY_PROTOCOL_VERSION = 3
_SELF_CONSISTENCY_PROTOCOL_V2_SPEC = {
    "version": SELF_CONSISTENCY_PROTOCOL_VERSION,
    "scheduled_seed": "cell.seed + 1000 + sample_index",
    "outcome_contract": (
        "one completed AgentOutput, one full length censor, one full protocol censor, "
        "or one durable transport censor per scheduled sample"
    ),
    "aggregate_censor_policy": "all confidence/entropy aggregates undefined if any censor",
    "cost_accounting": "auxiliary generation costs separate from topology efficiency",
    "replacement_sampling": "forbidden",
}
SELF_CONSISTENCY_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        _SELF_CONSISTENCY_PROTOCOL_V2_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass
class QuestionResult:
    cell_id: str
    qid: str
    benchmark: str
    model_size: str
    topology: str
    context_share_level: str
    prompt_complexity_level: int
    reasoning_level: str

    final_answer: str | None
    answer_key: str
    correct: bool

    # Immutable normalized-benchmark provenance.  Completion validation requires both
    # values for current-schema rows: the per-Question digest binds prompt/options/gold,
    # while the ordered contract digest binds the entire selected benchmark tranche.
    question_sha256: str
    benchmark_contract_sha256: str
    per_agent: list[dict[str, Any]] = field(default_factory=list)   # AgentOutput.to_dict()
    system_conf: dict[str, float] = field(default_factory=dict)     # multiple definitions
    self_consistency: dict[str, Any] = field(default_factory=dict)  # samples + agreement
    efficiency_raw: dict[str, Any] = field(default_factory=dict)    # turns/messages/tokens/wall
    timestamp: float = 0.0
    # Runtime serving provenance belongs on every newly generated question.  Historical
    # rows predate these fields; completion.py interprets an entirely absent/null trio as
    # the cell's standard (model-size-named) profile without rewriting the retained row.
    serving_profile: str | None = None
    effective_context_limit: int | None = None
    tensor_parallel_size: int | None = None
    # Homogeneous schema-5 production lineage.  These stay nullable so the same reader
    # can audit sealed schema-4 and pre-policy schema-5 artifacts.  A run-level artifact
    # policy promotes every field below to mandatory and completion.py validates it
    # against the frozen release/model/environment pins.
    release_id: str | None = None
    environment_hash: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_contract_sha256: str | None = None
    # The effective fleet hash may change only through a controlled capacity
    # generation. ``release_fleet_contract_sha256`` retains immutable release lineage.
    fleet_contract_sha256: str | None = None
    release_fleet_contract_sha256: str | None = None
    capacity_generation: int | None = None
    endpoint_generation: str | None = None
    # Exact counts over stochastic coordinates.  Scalar fleet/capacity/rollout fields
    # are populated only when these counts are homogeneous; a resumed QID may
    # truthfully span capacity, rollout, and endpoint generations.  The marginal
    # ``coordinate_provenance_counts`` remain convenient summaries, but cannot by
    # themselves prove which generation values co-occurred.  Schema-5 production
    # therefore also retains exact joint identity/count rows over the complete
    # release/fleet/capacity/rollout/endpoint tuple.
    coordinate_provenance_counts: dict[str, dict[str, int]] = field(
        default_factory=_empty_coordinate_provenance_counts
    )
    coordinate_provenance_identity_counts: list[dict[str, Any]] = field(
        default_factory=list
    )
    effective_context: int | None = None
    rollout_generation: int | None = None
    # A finite model context cannot guarantee that a sampled trajectory emits EOS.
    # ``length_censored`` is therefore a first-class observed outcome: no parseable
    # prefix is accepted as an answer and the rejected response is retained under
    # ``censored_generation``.  This avoids outcome-conditioned resampling while still
    # giving the completion contract exactly one truthful record per benchmark QID.
    termination_status: str = TERMINATION_COMPLETED
    censored_generation: dict[str, Any] | None = None
    # An ambiguous connection/timeout or a restarted pending request is a trusted,
    # terminal observation but has no model response envelope.  Keep it separate from
    # ``censored_generation`` so no token provenance is fabricated.
    transport_censor: dict[str, Any] | None = None
    # Schema 5 preserves every stochastic topology coordinate that completed before a
    # censor terminated the QID.  Completed rows use an empty list.
    observed_topology_coordinates: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellMeta:
    cell_id: str
    config: dict[str, Any]            # ExperimentCell.to_dict()
    config_hash: str
    model_hf_id: str
    served_model_name: str
    prompt_token_count: int
    prompt_quality: dict[str, Any]    # heuristic + llm_judge + features
    # Ordered normalized-Question contract for this cell's exact benchmark selection.
    # Current metadata cannot certify completion unless this is the registered digest.
    benchmark_contract_sha256: str
    # Runtime layout is recorded separately from the scientific model-size treatment.
    serving_profile: str | None = None
    effective_context_limit: int | None = None
    tensor_parallel_size: int | None = None
    # Counts make a resume across runtime layouts explicit.  ``serving_profile`` is the
    # profile name for homogeneous cells and ``"mixed"`` otherwise.  Inferred counts
    # identify retained historical rows whose per-question provenance fields predate the
    # provenance contract.
    serving_profile_counts: dict[str, int] = field(default_factory=dict)
    serving_profile_inferred_counts: dict[str, int] = field(default_factory=dict)
    release_id: str | None = None
    environment_hash: str | None = None
    serving_environment_hash: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_contract_sha256: str | None = None
    fleet_contract_sha256: str | None = None
    release_fleet_contract_sha256: str | None = None
    capacity_generation: int | None = None
    artifact_policy_sha256: str | None = None
    endpoint_generation: str | None = None
    coordinate_provenance_counts: dict[str, dict[str, int]] = field(
        default_factory=_empty_coordinate_provenance_counts
    )
    coordinate_provenance_identity_counts: list[dict[str, Any]] = field(
        default_factory=list
    )
    effective_context: int | None = None
    rollout_generation: int | None = None
    git_commit: str | None = None
    n_questions: int = 0
    completed_question_count: int = 0
    length_censored_question_count: int = 0
    protocol_censored_question_count: int = 0
    transport_censored_question_count: int = 0
    transport_affected_question_count: int = 0
    transport_censored_coordinate_count: int = 0
    artifact_schema_counts: dict[str, int] = field(default_factory=dict)
    # Exact token-ID means are never pooled with historical whitespace counts.
    # ``mean_reasoning_tokens`` is populated only when every canonical question is
    # exact; ``mean_reasoning_tokens_exact`` describes the exact subset of a mixed cell.
    mean_reasoning_tokens: float | None = None
    mean_reasoning_tokens_exact: float | None = None
    exact_reasoning_question_count: int = 0
    nonexact_reasoning_question_count: int = 0
    mean_reasoning_word_count_legacy: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    # Versioned scientific-protocol provenance.  Historical metadata has none of these
    # fields and is validated under the explicit legacy contract.
    peer_context_protocol_version: int = PEER_CONTEXT_PROTOCOL_VERSION
    peer_context_protocol_hash: str = PEER_CONTEXT_PROTOCOL_HASH
    peer_cot_char_limit: int = PEER_COT_CHAR_LIMIT
    peer_rendered_block_token_limit: int = PEER_RENDERED_BLOCK_TOKEN_LIMIT
    thinking_budget_protocol_version: int = THINKING_BUDGET_PROTOCOL_VERSION
    thinking_budget_protocol_hash: str = THINKING_BUDGET_PROTOCOL_HASH
    generation_censor_protocol_version: int = GENERATION_CENSOR_PROTOCOL_VERSION
    generation_censor_protocol_hash: str = GENERATION_CENSOR_PROTOCOL_HASH
    transport_censor_protocol_version: int = TRANSPORT_CENSOR_PROTOCOL_VERSION
    transport_censor_protocol_hash: str = TRANSPORT_CENSOR_PROTOCOL_HASH
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
