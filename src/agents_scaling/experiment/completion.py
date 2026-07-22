"""Semantic completion, cell locking, and durable failure state.

The presence of ``meta.json`` is not a completion contract.  A cell is complete only
when its metadata matches the frozen manifest cell and its canonical result records
contain exactly one valid row for every question produced by the benchmark loader.

This module intentionally contains no scheduler policy.  It exposes one shared status
model used by runners, dispatchers, monitors, and analysis so those components cannot
disagree about whether a cell is done.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agents_scaling.benchmarks.loaders import load_benchmark
from agents_scaling.benchmarks.grading import extract_answer, grade
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.benchmarks.contracts import (
    BenchmarkContractKey,
    FrozenBenchmarkContracts,
    build_question_contract,
    canonical_question_sha256,
    load_frozen_benchmark_contracts,
    load_verified_questions,
)
from agents_scaling.agents.aggregate import majority_vote, system_confidences
from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.calibration.semantic_entropy import semantic_entropy_conf
from agents_scaling.calibration.signals import majority_answer, self_consistency_conf
from agents_scaling.agents.message_builder import (
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_COT_CHAR_LIMIT,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
)
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.artifact_policy import (
    ArtifactPolicy,
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest
from agents_scaling.experiment.result_schema import (
    ARTIFACT_SCHEMA_VERSION,
    SCHEMA_4_TERMINATION_STATUSES,
    SELF_CONSISTENCY_PROTOCOL_HASH,
    SELF_CONSISTENCY_PROTOCOL_V1_HASH,
    SELF_CONSISTENCY_PROTOCOL_V1_VERSION,
    SELF_CONSISTENCY_PROTOCOL_VERSION,
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
    TERMINATION_STATUSES,
)
from agents_scaling.models import get_model
from agents_scaling.serving.context import (
    CONTEXT_RESERVE_TOKENS,
    ContextCapacityError,
    TokenizerInitializationError,
)
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    GENERATION_CENSOR_PROTOCOL_HASH,
    GENERATION_CENSOR_PROTOCOL_VERSION,
    GenerationProtocolCensorError,
    LEGACY_ANSWER_GENERATION_TOKEN_ALLOWANCE,
    QWEN_END_OF_TEXT_TOKEN_ID,
    QWEN_IM_END_TOKEN_ID,
    QWEN_THINK_END_TOKEN_ID,
    QWEN_THINK_START_TOKEN_ID,
    ServerResponseProtocolError,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
    UNLIMITED_THINKING_TOKEN_ALLOWANCE,
    generation_protocol_violation_codes,
)
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.serving.model_contracts import ModelContractError, load_model_contracts


RESULTS_FILENAME = "results.jsonl"
META_FILENAME = "meta.json"
FAILURE_FILENAME = "failure.json"
LOCK_FILENAME = ".cell.lock"
FAILURE_SCHEMA_VERSION = 2
LEGACY_FAILURE_SCHEMA_VERSIONS = frozenset({1})


class CompletionState(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    ACTIVE = "active"
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    CORRUPT = "corrupt"
    MISSING = "missing"


class FailureClass(str, Enum):
    CONNECTION = "connection"
    CONTEXT_CAPACITY = "context_capacity"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"


class FailureDisposition(str, Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"


class CellLockUnavailable(RuntimeError):
    """Raised when another worker already owns a cell's advisory lock."""


class CorruptArtifactError(ValueError):
    """Raised when a completion artifact cannot be parsed or validated."""


@dataclass(frozen=True)
class ReasoningTokenSummary:
    """Honest aggregate of exact v3 token counts and historical word counts.

    Historical rows stored whitespace-word counts under the name
    ``reasoning_tokens``.  They remain scientifically useful, but cannot be averaged
    with exact token-ID spans from the native vLLM protocol.
    """

    all_exact: bool
    exact_question_count: int
    nonexact_question_count: int
    mean_reasoning_tokens: float | None
    mean_reasoning_tokens_exact: float | None
    mean_reasoning_word_count_legacy: float


class ExperimentConfigurationError(RuntimeError):
    """A cell cannot run until its code/configuration changes."""


@dataclass(frozen=True)
class QuestionServingProvenance:
    """Resolved runtime layout for one retained question result.

    ``inferred`` is true only for historical rows that predate per-question serving
    provenance.  Such rows came from the original model-size-named serving pools, so the
    only truthful inference is the cell's standard profile -- never the profile to which
    the cell happens to route when it is repaired or resumed today.
    """

    serving_profile: str
    effective_context_limit: int
    tensor_parallel_size: int
    inferred: bool = False


@dataclass(frozen=True)
class _OutcomeView:
    """Minimal AgentOutput-compatible view for shared aggregation functions."""

    answer_choice: str | None
    option_logprobs: dict[str, float]
    verbalized_conf: float | None


@dataclass(frozen=True)
class CanonicalResults:
    """First valid result row per expected QID, plus discarded-input diagnostics."""

    records: tuple[dict[str, Any], ...]
    raw_rows: int
    malformed_lines: int = 0
    invalid_rows: int = 0
    duplicate_qids: tuple[str, ...] = ()
    unexpected_qids: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()
    out_of_order: bool = False

    @property
    def qids(self) -> tuple[str, ...]:
        return tuple(str(record["qid"]) for record in self.records)

    @property
    def has_corruption(self) -> bool:
        return bool(
            self.malformed_lines
            or self.invalid_rows
            or self.duplicate_qids
            or self.unexpected_qids
            or self.validation_errors
        )

    @property
    def needs_rewrite(self) -> bool:
        return self.has_corruption or self.out_of_order or self.raw_rows != len(self.records)


@dataclass(frozen=True)
class _ResolvedQuestionContract:
    """Scientific truth and immutable digests used to certify one cell.

    ``questions_by_qid`` and both hash fields are absent only under the private
    structural parser bypass.  They are inseparable on every scientific path.
    """

    ordered_qids: tuple[str, ...]
    questions_by_qid: Mapping[str, Question] | None
    question_sha256_by_qid: Mapping[str, str] | None
    benchmark_contract_sha256: str | None


@dataclass(frozen=True)
class FailureRecord:
    schema_version: int
    cell_id: str
    config_hash: str
    classification: str
    disposition: str
    attempts: int
    first_failed_at: float
    last_failed_at: float
    last_error: dict[str, str]
    next_eligible_at: float | None = None
    serving_profile: str | None = None
    code_version: str | None = None
    server_pool_generation: str | None = None
    dormant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureRecord":
        try:
            record = cls(
                schema_version=int(value["schema_version"]),
                cell_id=str(value["cell_id"]),
                config_hash=str(value["config_hash"]),
                classification=FailureClass(value["classification"]).value,
                disposition=FailureDisposition(value["disposition"]).value,
                attempts=int(value["attempts"]),
                first_failed_at=float(value["first_failed_at"]),
                last_failed_at=float(value["last_failed_at"]),
                last_error={
                    "type": str(value["last_error"]["type"]),
                    "message": str(value["last_error"]["message"]),
                },
                next_eligible_at=(
                    None
                    if value.get("next_eligible_at") is None
                    else float(value["next_eligible_at"])
                ),
                serving_profile=(
                    None if value.get("serving_profile") is None else str(value["serving_profile"])
                ),
                code_version=(
                    None if value.get("code_version") is None else str(value["code_version"])
                ),
                server_pool_generation=(
                    None
                    if value.get("server_pool_generation", value.get("endpoint_generation"))
                    is None
                    else str(
                        value.get("server_pool_generation", value.get("endpoint_generation"))
                    )
                ),
                dormant=bool(value.get("dormant", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptArtifactError(f"invalid {FAILURE_FILENAME}: {exc}") from exc
        if record.schema_version not in {
            FAILURE_SCHEMA_VERSION,
            *LEGACY_FAILURE_SCHEMA_VERSIONS,
        }:
            supported = sorted(
                LEGACY_FAILURE_SCHEMA_VERSIONS | {FAILURE_SCHEMA_VERSION}
            )
            raise CorruptArtifactError(
                f"unsupported failure schema {record.schema_version}; "
                f"expected one of {supported}"
            )
        if record.attempts < 1:
            raise CorruptArtifactError("failure attempts must be >= 1")
        if not _valid_timestamp(record.first_failed_at) or not _valid_timestamp(
            record.last_failed_at
        ):
            raise CorruptArtifactError("failure timestamps must be finite and positive")
        if record.last_failed_at < record.first_failed_at:
            raise CorruptArtifactError("last_failed_at precedes first_failed_at")
        if record.next_eligible_at is not None and not _valid_timestamp(record.next_eligible_at):
            raise CorruptArtifactError("next_eligible_at must be finite and positive")
        intrinsic = record.classification in {
            FailureClass.CONTEXT_CAPACITY.value,
            FailureClass.CONFIGURATION.value,
        }
        transient = record.classification in {
            FailureClass.CONNECTION.value,
            FailureClass.RUNTIME.value,
        }
        if intrinsic and record.disposition != FailureDisposition.PERMANENT.value:
            raise CorruptArtifactError(
                "context/configuration failures must have permanent disposition"
            )
        if transient and record.disposition != FailureDisposition.RETRYABLE.value:
            raise CorruptArtifactError(
                "connection/runtime failures must have retryable disposition"
            )
        if intrinsic and (record.dormant or record.next_eligible_at is not None):
            raise CorruptArtifactError(
                "permanent context/configuration failures cannot be dormant or scheduled"
            )
        if transient and record.dormant and record.next_eligible_at is not None:
            raise CorruptArtifactError(
                "dormant transient failures cannot have next_eligible_at"
            )
        if transient and not record.dormant and record.next_eligible_at is None:
            raise CorruptArtifactError(
                "active transient failures require next_eligible_at"
            )
        return record


@dataclass(frozen=True)
class CompletionStatus:
    status: CompletionState
    cell_id: str
    expected_count: int
    valid_count: int
    completed_question_count: int = 0
    length_censored_question_count: int = 0
    protocol_censored_question_count: int = 0
    missing_qids: tuple[str, ...] = ()
    duplicate_qids: tuple[str, ...] = ()
    unexpected_qids: tuple[str, ...] = ()
    malformed_lines: int = 0
    invalid_rows: int = 0
    errors: tuple[str, ...] = ()
    next_eligible_at: float | None = None
    eligible_for_retry: bool = False
    failure: FailureRecord | None = None

    @property
    def is_complete(self) -> bool:
        return self.status is CompletionState.COMPLETE

    @property
    def is_terminal(self) -> bool:
        return self.status in (CompletionState.COMPLETE, CompletionState.PERMANENT)

    @property
    def state(self) -> CompletionState:
        """Compatibility alias for callers that use ``state`` terminology."""
        return self.status


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONNumber(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _strict_json_loads(payload: str) -> Any:
    def reject_constant(token: str) -> None:
        raise _NonFiniteJSONNumber(f"non-finite JSON number {token!r}")

    return json.loads(
        payload,
        object_pairs_hook=_strict_object,
        parse_constant=reject_constant,
    )


def _valid_timestamp(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _is_integer(value: Any, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return False
    return minimum is None or float(value) >= minimum


def _schema_contract(value: Mapping[str, Any], artifact: str) -> tuple[bool, list[str]]:
    """Return ``(is_versioned, errors)`` for legacy or registered artifacts.

    Legacy artifacts are identified only by an absent schema_version.  Once a producer
    writes a version field, unknown, null, boolean, and future versions are rejected
    instead of being silently interpreted as legacy.
    """
    if "schema_version" not in value:
        return False, []
    observed = value.get("schema_version")
    if (
        not isinstance(observed, int)
        or isinstance(observed, bool)
        or observed not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
    ):
        return True, [
            f"{artifact} schema_version must be one of "
            f"{sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}"
        ]
    return True, []


_QUESTION_PROVENANCE_FIELDS = (
    "serving_profile",
    "effective_context_limit",
    "tensor_parallel_size",
)

_BENCHMARK_PROVENANCE_FIELDS = (
    "question_sha256",
    "benchmark_contract_sha256",
)

_EFFICIENCY_INTEGER_FIELDS = (
    "n_turns",
    "n_messages",
    "n_rounds",
    "n_agents",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_reasoning_tokens",
)

_AGENT_STRING_FIELDS = (
    "agent_id",
    "cot_text",
    "intermediate_results",
    "reasoning_text",
)
_CURRENT_AGENT_FIELDS = frozenset(
    ("answer" if field.name == "answer_choice" else field.name)
    for field in fields(AgentOutput)
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _requested_generation_tokens(cell: ExperimentCell) -> int:
    total = ANSWER_GENERATION_TOKEN_ALLOWANCE
    if cell.reasoning_level.enable_thinking:
        total += (
            cell.reasoning_level.thinking_budget
            if cell.reasoning_level.thinking_budget is not None
            else UNLIMITED_THINKING_TOKEN_ALLOWANCE
        )
    return total


def _expected_agent_seed(cell: ExperimentCell, agent: Mapping[str, Any]) -> int | None:
    """Mirror the topology's frozen request-seed schedule for artifact validation."""
    agent_id = agent.get("agent_id")
    round_index = agent.get("round")
    if not isinstance(agent_id, str) or not _is_integer(round_index):
        return None
    match = re.fullmatch(r"agent(\d+)", agent_id)
    if match is None:
        return None
    index = int(match.group(1))
    topology = cell.topology.value
    if topology == "single_agent":
        return cell.seed if index == 0 and round_index == 0 else None
    if topology == "independent":
        return cell.seed + index if 0 <= index < cell.n_agents and round_index == 0 else None
    if not 0 <= round_index < cell.rounds or not 0 <= index < cell.n_agents:
        return None
    if topology == "decentralized":
        return cell.seed + round_index * 100 + index
    if topology == "centralized":
        offset = 999 if index == 0 else index - 1
        return cell.seed + round_index * 100 + offset
    return None


def _token_id_sequence(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        for token_id in value
    )


def _token_id_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _censored_generation_errors(
    value: Any,
    *,
    cell: ExperimentCell,
    qid: str,
    termination_status: str,
    serving_profile: Any,
    effective_context_limit: Any,
) -> list[str]:
    """Validate one non-resampled exact length or response-protocol censor.

    A censored response is scientific data, not a completed answer.  The complete token
    sequences and request envelope are required so a row cannot use the censor label to
    conceal a smaller ad-hoc cap, a changed seed, or a parseable truncated prefix.
    """

    if not isinstance(value, dict):
        return ["censored_generation must be an object"]
    errors: list[str] = []
    base_fields = {
        "reason",
        "finish_reason",
        "sampling_attempt_count",
        "thinking_budget_protocol_version",
        "thinking_budget_protocol_hash",
        "qid",
        "agent_id",
        "round",
        "generation_role",
        "sample_index",
        "seed",
        "serving_profile",
        "effective_context_limit",
        "context_reserve_tokens",
        "prompt_tokens",
        "requested_output_tokens",
        "output_capacity_floor_tokens",
        "completion_tokens",
        "prompt_token_id_sha256",
        "completion_token_id_sha256",
        "prompt_token_ids",
        "completion_token_ids",
        "decoded_completion",
        "server_content",
        "server_reasoning",
        "prompt_think_start_positions",
        "prompt_think_end_positions",
        "completion_think_start_positions",
        "completion_think_end_positions",
        "endpoint_generation",
        "created_at",
    }
    protocol_only_fields = {
        "protocol_violation_codes",
        "actual_terminal_token_id",
        "generation_censor_protocol_version",
        "generation_censor_protocol_hash",
    }
    expected_fields = base_fields | (
        protocol_only_fields
        if termination_status == TERMINATION_PROTOCOL_CENSORED
        else set()
    )
    observed_fields = set(value)
    if observed_fields not in {
        frozenset(expected_fields),
        frozenset(expected_fields | {"topology", "benchmark"}),
    }:
        errors.append("censored_generation has the wrong fields")
    if "topology" in value or "benchmark" in value:
        if value.get("topology") != cell.topology.value:
            errors.append("censored_generation.topology does not match the manifest")
        if value.get("benchmark") != cell.benchmark:
            errors.append("censored_generation.benchmark does not match the manifest")
    expected_reason = (
        "protocol"
        if termination_status == TERMINATION_PROTOCOL_CENSORED
        else "length"
    )
    if value.get("reason") != expected_reason:
        errors.append(
            f"censored_generation.reason must be {expected_reason}"
        )
    if (
        termination_status == TERMINATION_LENGTH_CENSORED
        and value.get("finish_reason") != "length"
    ):
        errors.append("length censor finish_reason must be length")
    if value.get("sampling_attempt_count") != 1:
        errors.append("censored_generation must record exactly one sampling attempt")
    if value.get("qid") != qid:
        errors.append("censored_generation.qid does not match the result QID")

    identity = {
        "agent_id": value.get("agent_id"),
        "round": value.get("round"),
    }
    if not isinstance(identity["agent_id"], str) or not identity["agent_id"]:
        errors.append("censored_generation.agent_id must be a non-empty string")
    if not _is_integer(identity["round"]):
        errors.append("censored_generation.round must be a non-negative integer")
    role = value.get("generation_role")
    if role == "topology":
        expected_seed = _expected_agent_seed(cell, identity)
        if expected_seed is None:
            errors.append(
                "censored generation agent identity/round is invalid for topology"
            )
        elif value.get("seed") != expected_seed:
            errors.append("censored generation seed does not match topology schedule")
        if value.get("sample_index") is not None:
            errors.append("topology censor cannot have a self-consistency sample index")
    elif role == "self_consistency":
        sample_index = value.get("sample_index")
        if (
            cell.topology.value != "single_agent"
            or identity != {"agent_id": "agent0", "round": 0}
            or not _is_integer(sample_index)
            or sample_index >= cell.n_samples
        ):
            errors.append("censored self-consistency sample identity is invalid")
        elif value.get("seed") != cell.seed + 1000 + sample_index:
            errors.append("censored self-consistency seed does not match schedule")
    else:
        errors.append("censored_generation.generation_role is not registered")

    integer_fields = (
        "prompt_tokens",
        "requested_output_tokens",
        "output_capacity_floor_tokens",
        "completion_tokens",
        "context_reserve_tokens",
        "effective_context_limit",
    )
    for field in integer_fields:
        minimum = 1
        if field == "context_reserve_tokens" or (
            field == "completion_tokens"
            and termination_status == TERMINATION_PROTOCOL_CENSORED
        ):
            minimum = 0
        if not _is_integer(value.get(field), minimum=minimum):
            errors.append(f"censored_generation.{field} must be an integer >= {minimum}")

    prompt_ids = value.get("prompt_token_ids")
    completion_ids = value.get("completion_token_ids")
    prompt_ids_valid = _token_id_sequence(prompt_ids)
    completion_ids_valid = _token_id_sequence(completion_ids)
    if not prompt_ids_valid or not prompt_ids:
        errors.append("censored_generation.prompt_token_ids must be non-empty token IDs")
        prompt_ids = []
    if not completion_ids_valid or (
        termination_status == TERMINATION_LENGTH_CENSORED and not completion_ids
    ):
        errors.append(
            "censored_generation.completion_token_ids must be token IDs"
            + (" and non-empty for a length censor" if termination_status == TERMINATION_LENGTH_CENSORED else "")
        )
        completion_ids = []
    for field, token_ids in (
        ("prompt_token_id_sha256", prompt_ids),
        ("completion_token_id_sha256", completion_ids),
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"censored_generation.{field} must be a lowercase SHA-256")
        elif (
            (field.startswith("prompt") and prompt_ids_valid)
            or (field.startswith("completion") and completion_ids_valid)
        ) and digest != _token_id_sha256(token_ids):
            errors.append(f"censored_generation.{field} does not match retained token IDs")

    if prompt_ids and _is_integer(value.get("prompt_tokens")) and len(prompt_ids) != value["prompt_tokens"]:
        errors.append("censored generation prompt count does not match token IDs")
    if (
        completion_ids_valid
        and _is_integer(value.get("completion_tokens"))
        and len(completion_ids) != value["completion_tokens"]
    ):
        errors.append("censored generation completion count does not match token IDs")
    if (
        termination_status == TERMINATION_LENGTH_CENSORED
        and _is_integer(value.get("completion_tokens"), minimum=1)
        and _is_integer(value.get("requested_output_tokens"), minimum=1)
        and value["completion_tokens"] != value["requested_output_tokens"]
    ):
        errors.append("length-censored completion must exhaust its requested output envelope")
    if (
        termination_status == TERMINATION_PROTOCOL_CENSORED
        and
        _is_integer(value.get("completion_tokens"), minimum=1)
        and _is_integer(value.get("requested_output_tokens"), minimum=1)
        and value["completion_tokens"] > value["requested_output_tokens"]
    ):
        errors.append("protocol-censored completion exceeds its requested output envelope")

    expected_floor = _requested_generation_tokens(cell)
    if value.get("output_capacity_floor_tokens") != expected_floor:
        errors.append("censored generation output floor does not match manifest reasoning level")
    if (
        _is_integer(value.get("requested_output_tokens"), minimum=1)
        and value["requested_output_tokens"] < expected_floor
    ):
        errors.append(
            "censored generation requested output is below the registered output floor"
        )
    if value.get("context_reserve_tokens") != CONTEXT_RESERVE_TOKENS:
        errors.append("censored generation context reserve is not registered")
    if value.get("effective_context_limit") != effective_context_limit:
        errors.append("censored generation context limit does not match result provenance")
    if value.get("serving_profile") != serving_profile:
        errors.append("censored generation serving profile does not match result provenance")
    if (
        all(_is_integer(value.get(field)) for field in (
            "prompt_tokens",
            "requested_output_tokens",
            "context_reserve_tokens",
            "effective_context_limit",
        ))
        and value["prompt_tokens"]
        + value["requested_output_tokens"]
        + value["context_reserve_tokens"]
        != value["effective_context_limit"]
    ):
        errors.append("censored generation did not use the full registered context envelope")

    if not isinstance(value.get("decoded_completion"), str):
        errors.append("censored_generation.decoded_completion must be a string")
    for field in ("server_content", "server_reasoning"):
        if value.get(field) is not None and not isinstance(value.get(field), str):
            errors.append(
                f"censored_generation.{field} must be a string or null"
            )
    position_contracts = (
        ("prompt_think_start_positions", prompt_ids, QWEN_THINK_START_TOKEN_ID),
        ("prompt_think_end_positions", prompt_ids, QWEN_THINK_END_TOKEN_ID),
        (
            "completion_think_start_positions",
            completion_ids,
            QWEN_THINK_START_TOKEN_ID,
        ),
        (
            "completion_think_end_positions",
            completion_ids,
            QWEN_THINK_END_TOKEN_ID,
        ),
    )
    for field, token_ids, delimiter_id in position_contracts:
        positions = value.get(field)
        if not isinstance(positions, list) or not all(_is_integer(position) for position in positions):
            errors.append(f"censored_generation.{field} must contain token indexes")
        else:
            expected_positions = [
                index
                for index, token_id in enumerate(token_ids)
                if token_id == delimiter_id
            ]
            if positions != expected_positions:
                errors.append(
                    f"censored_generation.{field} does not exactly match retained "
                    "delimiter token IDs"
                )
    if value.get("thinking_budget_protocol_version") != THINKING_BUDGET_PROTOCOL_VERSION:
        errors.append("censored generation protocol version is not registered")
    if value.get("thinking_budget_protocol_hash") != THINKING_BUDGET_PROTOCOL_HASH:
        errors.append("censored generation protocol hash is not registered")
    protocol_fields = protocol_only_fields
    if termination_status == TERMINATION_PROTOCOL_CENSORED:
        codes = value.get("protocol_violation_codes")
        if (
            not isinstance(codes, list)
            or not codes
            or not all(isinstance(code, str) and code for code in codes)
            or codes != sorted(set(codes))
        ):
            errors.append(
                "protocol censor violation codes must be a non-empty sorted unique list"
            )
            codes = []
        expected_codes = generation_protocol_violation_codes(
            finish_reason=value.get("finish_reason"),
            completion_token_ids=(
                completion_ids if completion_ids_valid else []
            ),
            enable_thinking=cell.reasoning_level.enable_thinking,
            thinking_budget=cell.reasoning_level.thinking_budget,
        )
        if tuple(codes) != expected_codes:
            errors.append(
                "protocol censor violation codes do not match retained response"
            )
        terminal = completion_ids[-1] if completion_ids else None
        if value.get("actual_terminal_token_id") != terminal:
            errors.append(
                "protocol censor actual_terminal_token_id does not match completion"
            )
        if (
            value.get("generation_censor_protocol_version")
            != GENERATION_CENSOR_PROTOCOL_VERSION
        ):
            errors.append("generation censor protocol version is not registered")
        if (
            value.get("generation_censor_protocol_hash")
            != GENERATION_CENSOR_PROTOCOL_HASH
        ):
            errors.append("generation censor protocol hash is not registered")
    else:
        unexpected_protocol_fields = sorted(protocol_fields.intersection(value))
        if unexpected_protocol_fields:
            errors.append(
                "length censor contains protocol-censor-only fields: "
                + ", ".join(unexpected_protocol_fields)
            )
    if not _valid_timestamp(value.get("created_at")):
        errors.append("censored_generation.created_at must be finite and positive")
    endpoint_generation = value.get("endpoint_generation")
    if not isinstance(endpoint_generation, str) or not endpoint_generation:
        errors.append("censored_generation.endpoint_generation must be non-empty")
    return errors


def _expected_peer_block_count(
    cell: ExperimentCell, agent: Mapping[str, Any]
) -> int | None:
    """Return the peer blocks this exact topology role/round must have consumed."""

    agent_id = agent.get("agent_id")
    round_index = agent.get("round")
    if not isinstance(agent_id, str) or not _is_integer(round_index):
        return None
    match = re.fullmatch(r"agent(\d+)", agent_id)
    if match is None:
        return None
    index = int(match.group(1))
    if not 0 <= index < (1 if cell.topology.value == "single_agent" else cell.n_agents):
        return None
    topology = cell.topology.value
    if topology in {"single_agent", "independent"}:
        return 0 if round_index == 0 else None
    if not 0 <= round_index < cell.rounds:
        return None
    if topology == "decentralized":
        return 0 if round_index == 0 else max(0, cell.n_agents - 1)
    if topology == "centralized":
        if index == 0:  # orchestrator consumes every sub-agent output each round
            return max(0, cell.n_agents - 1)
        # Sub-agents are independent in round zero, then consume the prior synthesis.
        return 0 if round_index == 0 else 1
    return None


def _agent_errors(
    agent: Any,
    *,
    current_schema: bool,
    cell: ExperimentCell,
    effective_context_limit: Any,
    scheduled_seed: int | None = None,
    question: Question | None = None,
) -> list[str]:
    if not isinstance(agent, dict):
        return ["agent output must be an object"]
    errors: list[str] = []
    if current_schema and set(agent) != _CURRENT_AGENT_FIELDS:
        errors.append("current agent output has the wrong fields")
    for field in _AGENT_STRING_FIELDS:
        value = agent.get(field)
        if not isinstance(value, str) or (field == "agent_id" and not value):
            errors.append(f"agent {field} must be a string")
    if not _is_integer(agent.get("round")):
        errors.append("agent round must be a non-negative integer")
    answer = agent.get("answer")
    if answer is not None and not isinstance(answer, str):
        errors.append("agent answer must be a string or null")
    confidence = agent.get("verbalized_conf")
    if confidence is not None and (
        not _is_finite_number(confidence)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("agent verbalized_conf must be null or finite in [0, 1]")
    option_logprobs = agent.get("option_logprobs")
    if not isinstance(option_logprobs, dict):
        errors.append("agent option_logprobs must be an object")
    else:
        for option, probability in option_logprobs.items():
            if not isinstance(option, str) or not option:
                errors.append("agent option_logprobs keys must be non-empty strings")
                break
            if (
                not _is_finite_number(probability)
                or not 0.0 <= float(probability) <= 1.0
            ):
                errors.append("agent option_logprobs values must be finite in [0, 1]")
                break
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        if not _is_integer(agent.get(field)):
            errors.append(f"agent {field} must be a non-negative integer")

    finite_budget = cell.reasoning_level.thinking_budget
    if not current_schema:
        if _is_integer(agent.get("completion_tokens")):
            legacy_cap = LEGACY_ANSWER_GENERATION_TOKEN_ALLOWANCE
            if cell.reasoning_level.enable_thinking:
                legacy_cap += (
                    finite_budget
                    if finite_budget is not None
                    else UNLIMITED_THINKING_TOKEN_ALLOWANCE
                )
            if agent["completion_tokens"] >= legacy_cap:
                errors.append(
                    "legacy generation reached its old output cap and "
                    "cannot certify complete termination"
                )
        return errors

    raw_text = agent.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        errors.append("v3 agent raw_text must be a non-empty string")

    if question is not None and isinstance(option_logprobs, dict):
        expected_options = (
            set(question.option_letters)
            if question.answer_type is AnswerType.MCQ
            else set()
        )
        if set(option_logprobs) != expected_options:
            errors.append(
                "current agent option_logprobs keys do not match benchmark options"
            )
        elif expected_options and all(
            _is_finite_number(option_logprobs[letter])
            for letter in question.option_letters
        ):
            if not math.isclose(
                math.fsum(float(option_logprobs[letter]) for letter in question.option_letters),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append(
                    "current agent option_logprobs must sum to one over benchmark options"
                )

        if isinstance(raw_text, str) and raw_text:
            expected_answer = extract_answer(question, raw_text)
            if (
                expected_answer is None
                and question.answer_type is AnswerType.MCQ
                and set(option_logprobs) == expected_options
                and expected_options
            ):
                expected_answer = max(
                    question.option_letters,
                    key=lambda letter: float(option_logprobs[letter]),
                )
            if agent.get("answer") != expected_answer:
                errors.append(
                    "current agent answer does not match raw_text/option-probe fallback"
                )

    finish_reason = agent.get("finish_reason")
    if finish_reason != "stop":
        errors.append("v3 agent finish_reason must be stop")

    expected_source = (
        "vllm_native_token_ids"
        if cell.reasoning_level.enable_thinking
        else "none"
    )
    if agent.get("reasoning_token_source") != expected_source:
        errors.append(
            "v3 agent reasoning_token_source does not match the native token-ID protocol"
        )
    word_count = agent.get("reasoning_word_count_legacy")
    if not _is_integer(word_count):
        errors.append(
            "v3 agent reasoning_word_count_legacy must be a non-negative integer"
        )
    elif isinstance(agent.get("reasoning_text"), str) and word_count != len(
        agent["reasoning_text"].split()
    ):
        errors.append(
            "v3 agent reasoning_word_count_legacy does not match reasoning_text"
        )

    if agent.get("thinking_budget_protocol_version") != THINKING_BUDGET_PROTOCOL_VERSION:
        errors.append("v3 agent thinking_budget_protocol_version is not registered")
    if agent.get("thinking_budget_protocol_hash") != THINKING_BUDGET_PROTOCOL_HASH:
        errors.append("v3 agent thinking_budget_protocol_hash is not registered")
    endpoint_generation = agent.get("endpoint_generation")
    if not isinstance(endpoint_generation, str) or not endpoint_generation:
        errors.append("current agent endpoint_generation must be non-empty")
    if agent.get("thinking_budget_requested") != finite_budget:
        errors.append("v3 agent thinking_budget_requested does not match manifest")

    saturated = agent.get("thinking_budget_saturated")
    if not isinstance(saturated, bool):
        errors.append("v3 agent thinking_budget_saturated must be a boolean")
    consumed = agent.get("thinking_budget_consumed_tokens")
    if not _is_integer(consumed):
        errors.append(
            "v3 agent thinking_budget_consumed_tokens must be a non-negative integer"
        )

    phase_count = agent.get("generation_phase_count")
    if phase_count != 1:
        errors.append("v3 native generation_phase_count must equal 1")
    phase_fields = (
        "generation_phase_finish_reasons",
        "generation_phase_seeds",
        "generation_phase_prompt_tokens",
        "generation_phase_completion_tokens",
        "generation_phase_requested_max_tokens",
        "generation_phase_prompt_token_id_hashes",
        "generation_phase_completion_token_id_hashes",
    )
    phase_values: dict[str, list[Any]] = {}
    for field in phase_fields:
        value = agent.get(field)
        if not isinstance(value, list) or len(value) != 1:
            errors.append(f"v3 agent {field} must contain exactly one phase")
            phase_values[field] = []
        else:
            phase_values[field] = value

    finishes = phase_values.get("generation_phase_finish_reasons", [])
    if finishes != ["stop"]:
        errors.append("v3 native generation phase must finish stop")
    expected_seed = (
        scheduled_seed
        if scheduled_seed is not None
        else _expected_agent_seed(cell, agent)
    )
    seeds = phase_values.get("generation_phase_seeds", [])
    if expected_seed is None:
        errors.append("v3 agent identity/round is not valid for the manifest topology")
    elif seeds != [expected_seed]:
        errors.append("v3 agent generation seed does not match the topology schedule")

    phase_prompts = phase_values.get("generation_phase_prompt_tokens", [])
    phase_completions = phase_values.get("generation_phase_completion_tokens", [])
    phase_requested = phase_values.get("generation_phase_requested_max_tokens", [])
    if not all(_is_integer(value, minimum=1) for value in phase_prompts):
        errors.append("v3 generation prompt tokens must be positive integers")
    if not all(_is_integer(value, minimum=1) for value in phase_completions):
        errors.append("v3 generation completion tokens must be positive integers")
    if not all(_is_integer(value, minimum=1) for value in phase_requested):
        errors.append("current generation requested max tokens must be positive integers")
    if phase_prompts and phase_prompts[0] != agent.get("prompt_tokens"):
        errors.append("v3 phase prompt tokens do not equal prompt_tokens")
    if phase_completions and phase_completions[0] != agent.get("completion_tokens"):
        errors.append("v3 phase completion tokens do not equal completion_tokens")

    for field in (
        "generation_phase_prompt_token_id_hashes",
        "generation_phase_completion_token_id_hashes",
    ):
        values = phase_values.get(field, [])
        if values and not (
            isinstance(values[0], str) and _SHA256_RE.fullmatch(values[0])
        ):
            errors.append(f"v3 agent {field} must contain one lowercase SHA-256")

    if agent.get("injected_transition_tokens") != 0:
        errors.append("v3 native generation cannot inject transition tokens")

    peer_tokens = agent.get("peer_context_tokens")
    peer_hash = agent.get("peer_context_sha256")
    peer_block_counts = agent.get("peer_context_block_token_counts")
    peer_truncations = agent.get("peer_context_truncation_marker_count")
    expected_peer_blocks = _expected_peer_block_count(cell, agent)
    if not _is_integer(peer_tokens):
        errors.append("v3 peer_context_tokens must be a non-negative integer")
    if not isinstance(peer_hash, str) or not _SHA256_RE.fullmatch(peer_hash):
        errors.append("v3 peer_context_sha256 must be a lowercase SHA-256")
    if not isinstance(peer_block_counts, list) or not all(
        _is_integer(count, minimum=1)
        and count <= PEER_RENDERED_BLOCK_TOKEN_LIMIT
        for count in peer_block_counts
    ):
        errors.append(
            "v3 peer_context_block_token_counts must contain positive bounded integers"
        )
        peer_block_counts = []
    if not _is_integer(peer_truncations) or peer_truncations > len(peer_block_counts):
        errors.append(
            "v3 peer_context_truncation_marker_count must fit the rendered block count"
        )
    if expected_peer_blocks is None:
        errors.append("v3 peer context role/round is not valid for the manifest topology")
    elif len(peer_block_counts) != expected_peer_blocks:
        errors.append(
            "v3 peer block count does not match the manifest topology role/round"
        )
    if _is_integer(peer_tokens) and expected_peer_blocks is not None:
        if expected_peer_blocks > 0 and peer_tokens == 0:
            errors.append("v3 non-empty peer blocks require positive peer_context_tokens")
        if expected_peer_blocks == 0:
            if peer_tokens != 0:
                errors.append(
                    "v3 peer_context_tokens must be zero when no peer blocks are expected"
                )
            if peer_hash != _EMPTY_SHA256:
                errors.append(
                    "v3 empty peer context must use the registered empty SHA-256"
                )
            if peer_truncations != 0:
                errors.append("v3 empty peer context cannot record truncation markers")

    output_floor = _requested_generation_tokens(cell)
    if agent.get("output_capacity_floor_tokens") != output_floor:
        errors.append(
            "current output_capacity_floor_tokens does not match manifest reasoning level"
        )
    prompt_tokens = agent.get("prompt_tokens")
    completion_tokens = agent.get("completion_tokens")
    requested_output = phase_requested[0] if phase_requested else None
    if (
        _is_integer(completion_tokens)
        and _is_integer(requested_output, minimum=1)
        and completion_tokens > requested_output
    ):
        errors.append("current completion_tokens exceeds the requested max")
    if (
        _is_integer(requested_output, minimum=1)
        and requested_output < output_floor
    ):
        errors.append("current requested max is below the registered output floor")
    if (
        _is_integer(prompt_tokens, minimum=1)
        and _is_integer(effective_context_limit, minimum=1)
        and _is_integer(requested_output, minimum=1)
        and prompt_tokens
        + requested_output
        + CONTEXT_RESERVE_TOKENS
        != effective_context_limit
    ):
        errors.append(
            "current generation did not use the full registered context-capacity envelope"
        )

    start_index = agent.get("reasoning_start_token_index")
    end_index = agent.get("reasoning_end_token_index")
    start_count = agent.get("reasoning_start_token_count")
    end_count = agent.get("reasoning_end_token_count")
    if not _is_integer(start_count) or not _is_integer(end_count):
        errors.append("v3 reasoning delimiter counts must be non-negative integers")

    reasoning_tokens = agent.get("reasoning_tokens")
    if cell.reasoning_level.enable_thinking:
        if start_count != 1 or not _is_integer(end_count, minimum=1):
            errors.append(
                "v3 thinking generation requires one generated start and at least "
                "one generated end delimiter"
            )
        if (
            _is_integer(end_count, minimum=1)
            and _is_integer(completion_tokens, minimum=1)
            and _is_integer(reasoning_tokens)
            # The completion must contain, in order, one start token, the exact
            # reasoning span, every recorded end token (the first boundary plus any
            # literal ends in answer content), and the terminal im_end token.
            and end_count > completion_tokens - reasoning_tokens - 2
        ):
            errors.append(
                "v3 generated end-delimiter count cannot fit before terminal im_end"
            )
        if not _is_integer(start_index) or not _is_integer(end_index):
            errors.append("v3 thinking generation requires delimiter token indexes")
        elif _is_integer(prompt_tokens) and _is_integer(completion_tokens):
            combined_tokens = prompt_tokens + completion_tokens
            if not (
                start_index == prompt_tokens
                and start_index < end_index < combined_tokens - 1
            ):
                errors.append(
                    "v3 reasoning delimiter indexes violate the Qwen output contract"
                )
            elif _is_integer(reasoning_tokens) and (
                end_index - start_index - 1 != reasoning_tokens
            ):
                errors.append(
                    "v3 reasoning_tokens does not match the exact delimiter span"
                )
        if _is_integer(consumed) and consumed != reasoning_tokens:
            errors.append(
                "v3 thinking_budget_consumed_tokens does not match reasoning_tokens"
            )
        if finite_budget is not None and _is_integer(consumed):
            if consumed > finite_budget:
                errors.append("v3 native reasoning exceeds the manifest budget")
            if isinstance(saturated, bool) and saturated != (consumed == finite_budget):
                errors.append(
                    "v3 thinking_budget_saturated does not match budget consumption"
                )
        elif saturated is not False:
            errors.append("v3 unlimited reasoning cannot claim finite-budget saturation")
    else:
        if reasoning_tokens != 0 or consumed != 0:
            errors.append("v3 thinking-disabled generation must record zero reasoning")
        if saturated is not False:
            errors.append("v3 thinking-disabled generation cannot saturate a budget")
        if start_index is not None or end_index is not None:
            errors.append(
                "v3 thinking-disabled generation cannot record generated delimiter indexes"
            )
        if start_count != 0 or end_count != 0:
            errors.append(
                "v3 thinking-disabled generation cannot record generated delimiter counts"
            )
    return errors


def _efficiency_errors(
    efficiency: Any,
    agents: Sequence[Mapping[str, Any]],
    cell: ExperimentCell,
) -> list[str]:
    if not isinstance(efficiency, dict):
        return ["efficiency_raw must be an object"]
    errors: list[str] = []
    for field in _EFFICIENCY_INTEGER_FIELDS:
        minimum = 1 if field in {"n_turns", "n_rounds", "n_agents"} else 0
        if not _is_integer(efficiency.get(field), minimum=minimum):
            errors.append(f"efficiency_raw.{field} must be an integer >= {minimum}")
    if not _is_finite_number(efficiency.get("wall_ms"), minimum=0.0):
        errors.append("efficiency_raw.wall_ms must be finite and non-negative")
    if errors:
        return errors

    expected_agents = 1 if cell.topology.value == "single_agent" else cell.n_agents
    expected_rounds = (
        1
        if cell.topology.value in {"single_agent", "independent"}
        else cell.rounds
    )
    expected_turns = (
        (max(1, expected_agents - 1) + 1) * expected_rounds
        if cell.topology.value == "centralized"
        else expected_agents * expected_rounds
    )
    if efficiency["n_agents"] != expected_agents:
        errors.append("efficiency_raw.n_agents does not match manifest topology")
    if efficiency["n_rounds"] != expected_rounds:
        errors.append("efficiency_raw.n_rounds does not match manifest topology")
    if efficiency["n_turns"] != expected_turns:
        errors.append("efficiency_raw.n_turns does not match manifest topology")
    if len(agents) != efficiency["n_turns"]:
        errors.append("per_agent length does not match efficiency_raw.n_turns")

    if cell.topology.value in {"single_agent", "independent"}:
        expected_messages = 0
    elif cell.topology.value == "decentralized":
        expected_messages = expected_agents * (expected_agents - 1) * (expected_rounds - 1)
    else:  # centralized: subagents -> orchestrator each round, reverse after round zero
        subagents = max(1, expected_agents - 1)
        expected_messages = subagents * (2 * expected_rounds - 1)
    if efficiency["n_messages"] != expected_messages:
        errors.append("efficiency_raw.n_messages does not match manifest topology")

    token_fields_are_valid = all(
        _is_integer(agent.get(field))
        for agent in agents
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    )
    if token_fields_are_valid:
        sums = {
            "total_prompt_tokens": sum(int(agent.get("prompt_tokens", 0)) for agent in agents),
            "total_completion_tokens": sum(
                int(agent.get("completion_tokens", 0)) for agent in agents
            ),
            "total_reasoning_tokens": sum(
                int(agent.get("reasoning_tokens", 0)) for agent in agents
            ),
        }
        for field, expected in sums.items():
            if efficiency[field] != expected:
                errors.append(f"efficiency_raw.{field} does not match per_agent sum")
    return errors


def _outcome_view(agent: Mapping[str, Any]) -> _OutcomeView | None:
    answer = agent.get("answer")
    option_logprobs = agent.get("option_logprobs")
    verbalized = agent.get("verbalized_conf")
    if answer is not None and not isinstance(answer, str):
        return None
    if not isinstance(option_logprobs, dict) or not all(
        isinstance(key, str) and _is_finite_number(value)
        for key, value in option_logprobs.items()
    ):
        return None
    if verbalized is not None and not _is_finite_number(verbalized):
        return None
    return _OutcomeView(
        answer_choice=answer,
        option_logprobs={
            str(key): float(value) for key, value in option_logprobs.items()
        },
        verbalized_conf=None if verbalized is None else float(verbalized),
    )


def _topology_outcome_errors(
    record: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    cell: ExperimentCell,
    *,
    current_schema: bool,
) -> list[str]:
    """Recompute the published answer and calibration from retained producers."""

    indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for agent in agents:
        match = re.fullmatch(r"agent(\d+)", str(agent.get("agent_id", "")))
        round_index = agent.get("round")
        if match is None or not _is_integer(round_index):
            return []  # coordinate errors are reported by _agent_errors
        indexed[(int(match.group(1)), int(round_index))] = agent

    topology = cell.topology.value
    if topology == "single_agent":
        relevant_coordinates = [(0, 0)]
        producer_coordinates: list[tuple[int, int]] | None = None
    elif topology == "independent":
        relevant_coordinates = [(index, 0) for index in range(cell.n_agents)]
        producer_coordinates = None
    elif topology == "decentralized":
        relevant_coordinates = [
            (index, cell.rounds - 1) for index in range(cell.n_agents)
        ]
        producer_coordinates = None
    else:  # centralized: final-round subagents are pooled; agent0 is the producer
        relevant_coordinates = [
            (index, cell.rounds - 1) for index in range(1, cell.n_agents)
        ]
        producer_coordinates = [(0, cell.rounds - 1)]

    if any(coordinate not in indexed for coordinate in relevant_coordinates):
        return []
    if producer_coordinates and any(
        coordinate not in indexed for coordinate in producer_coordinates
    ):
        return []
    relevant = [indexed[coordinate] for coordinate in relevant_coordinates]
    relevant_views = [_outcome_view(agent) for agent in relevant]
    if any(view is None for view in relevant_views):
        return []
    pooled = [view for view in relevant_views if view is not None]

    if topology == "centralized":
        assert producer_coordinates is not None
        producer = indexed[producer_coordinates[0]]
        producer_view = _outcome_view(producer)
        if producer_view is None:
            return []
        expected_final = producer_view.answer_choice
        expected_conf = system_confidences(
            pooled, expected_final, producers=[producer_view]
        )
        if expected_final is not None:
            expected_conf["orchestrator_logprob"] = producer_view.option_logprobs.get(
                expected_final, 0.0
            )
            if producer_view.verbalized_conf is not None:
                expected_conf["orchestrator_verbal"] = producer_view.verbalized_conf
    else:
        expected_final = majority_vote(pooled)
        expected_conf = system_confidences(pooled, expected_final)

    errors: list[str] = []
    if record.get("final_answer") != expected_final:
        errors.append("final_answer does not match retained topology outputs")
    if not current_schema:
        return errors

    observed_conf = record.get("system_conf")
    if not isinstance(observed_conf, dict):
        return errors
    if set(observed_conf) != set(expected_conf):
        errors.append("system_conf keys do not match retained topology outputs")
        return errors
    for field, expected in expected_conf.items():
        observed = observed_conf.get(field)
        if not _is_finite_number(observed) or not math.isclose(
            float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append(
                f"system_conf.{field} does not match retained topology outputs"
            )
    return errors


def _auxiliary_censored_reasoning_tokens(
    censored: Mapping[str, Any], cell: ExperimentCell
) -> int:
    """Mirror runner accounting for the exact reasoning span in a censored draw."""

    if not cell.reasoning_level.enable_thinking:
        return 0
    completion_ids = censored.get("completion_token_ids")
    starts = censored.get("completion_think_start_positions")
    ends = censored.get("completion_think_end_positions")
    if not isinstance(completion_ids, list) or not completion_ids:
        return 0
    if not isinstance(starts, list) or len(starts) != 1:
        return 0
    start = starts[0]
    if not _is_integer(start) or start >= len(completion_ids):
        return 0
    valid_ends = sorted(
        end
        for end in (ends if isinstance(ends, list) else [])
        if _is_integer(end) and start < end <= len(completion_ids)
    )
    if valid_ends:
        boundary = valid_ends[0]
    else:
        # A stop-terminated response with an open reasoning span retains its terminal
        # EOS token in ``completion_token_ids``.  That terminal is protocol evidence,
        # not a reasoning token.  A length-censored response has no terminal EOS, so
        # every token after the start delimiter belongs to the observed span.
        boundary = (
            max(start + 1, len(completion_ids) - 1)
            if censored.get("finish_reason") == "stop"
            else len(completion_ids)
        )
    return max(0, boundary - start - 1)


def _self_consistency_errors(
    value: Any,
    *,
    cell: ExperimentCell,
    qid: str,
    primary_answer: Any,
    serving_profile: Any,
    effective_context_limit: Any,
    question: Question | None,
    schema_version: int,
) -> list[str]:
    """Validate every scheduled auxiliary draw and its censor-aware aggregates."""

    enabled = cell.topology.value == "single_agent" and cell.n_samples > 1
    if not enabled:
        return [] if value == {} else [
            "self_consistency must be empty when auxiliary sampling is not configured"
        ]
    if not isinstance(value, dict) or not value:
        return [
            "self_consistency must record every configured auxiliary sample"
        ]

    errors: list[str] = []
    expected_fields = {
        "protocol_version",
        "protocol_hash",
        "sample_count",
        "completed_sample_count",
        "length_censored_sample_count",
        "samples",
        "majority",
        "self_consistency_conf",
        "semantic_entropy_conf",
        "semantic_entropy",
        "auxiliary_efficiency_raw",
    }
    if schema_version >= 5:
        expected_fields.add("protocol_censored_sample_count")
    if set(value) != expected_fields:
        errors.append(
            "self_consistency fields do not match the registered protocol"
        )
    expected_protocol_version = (
        SELF_CONSISTENCY_PROTOCOL_V1_VERSION
        if schema_version == 4
        else SELF_CONSISTENCY_PROTOCOL_VERSION
    )
    expected_protocol_hash = (
        SELF_CONSISTENCY_PROTOCOL_V1_HASH
        if schema_version == 4
        else SELF_CONSISTENCY_PROTOCOL_HASH
    )
    if value.get("protocol_version") != expected_protocol_version:
        errors.append("self_consistency protocol version is not registered")
    if value.get("protocol_hash") != expected_protocol_hash:
        errors.append("self_consistency protocol hash is not registered")

    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != cell.n_samples:
        errors.append(
            "self_consistency.samples must contain exactly manifest n_samples outcomes"
        )
        samples = []

    completed_outputs: list[Mapping[str, Any]] = []
    length_censored_outputs: list[Mapping[str, Any]] = []
    protocol_censored_outputs: list[Mapping[str, Any]] = []
    expected_sample_fields = {
        "sample_index",
        "seed",
        "termination_status",
        "agent_output",
        "censored_generation",
    }
    for sample_index, sample in enumerate(samples):
        label = f"self_consistency.samples[{sample_index}]"
        if not isinstance(sample, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(sample) != expected_sample_fields:
            errors.append(f"{label} fields do not match the registered protocol")
        expected_seed = cell.seed + 1000 + sample_index
        if sample.get("sample_index") != sample_index:
            errors.append(f"{label}.sample_index does not match its schedule position")
        if sample.get("seed") != expected_seed:
            errors.append(f"{label}.seed does not match the frozen schedule")
        status = sample.get("termination_status")
        if status == TERMINATION_COMPLETED:
            output = sample.get("agent_output")
            if sample.get("censored_generation") is not None:
                errors.append(f"{label} completed outcome cannot contain a censor")
            if not isinstance(output, dict):
                errors.append(f"{label}.agent_output must be a complete AgentOutput")
                continue
            if output.get("agent_id") != "agent0" or output.get("round") != 0:
                errors.append(
                    f"{label}.agent_output must use single-agent coordinate agent0/round0"
                )
            errors.extend(
                f"{label}.agent_output: {error}"
                for error in _agent_errors(
                    output,
                    current_schema=True,
                    cell=cell,
                    effective_context_limit=effective_context_limit,
                    scheduled_seed=expected_seed,
                    question=question,
                )
            )
            completed_outputs.append(output)
        elif status in {
            TERMINATION_LENGTH_CENSORED,
            TERMINATION_PROTOCOL_CENSORED,
        }:
            if status == TERMINATION_PROTOCOL_CENSORED and schema_version < 5:
                errors.append(f"{label}.termination_status is not registered in schema 4")
            censor = sample.get("censored_generation")
            if sample.get("agent_output") is not None:
                errors.append(f"{label} censored outcome cannot contain AgentOutput")
            if isinstance(censor, dict):
                if censor.get("sample_index") != sample_index:
                    errors.append(
                        f"{label}.censored_generation.sample_index does not match wrapper"
                    )
                if censor.get("seed") != expected_seed:
                    errors.append(
                        f"{label}.censored_generation.seed does not match wrapper schedule"
                    )
            errors.extend(
                f"{label}: {error}"
                for error in _censored_generation_errors(
                    censor,
                    cell=cell,
                    qid=qid,
                    termination_status=status,
                    serving_profile=serving_profile,
                    effective_context_limit=effective_context_limit,
                )
            )
            if isinstance(censor, dict):
                if status == TERMINATION_LENGTH_CENSORED:
                    length_censored_outputs.append(censor)
                else:
                    protocol_censored_outputs.append(censor)
        else:
            errors.append(f"{label}.termination_status is not registered")

    completed_count = len(completed_outputs)
    length_censored_count = len(length_censored_outputs)
    protocol_censored_count = len(protocol_censored_outputs)
    censored_outputs = length_censored_outputs + protocol_censored_outputs
    censored_count = len(censored_outputs)
    if value.get("sample_count") != cell.n_samples:
        errors.append("self_consistency.sample_count does not match manifest n_samples")
    if value.get("completed_sample_count") != completed_count:
        errors.append("self_consistency.completed_sample_count is incorrect")
    if value.get("length_censored_sample_count") != length_censored_count:
        errors.append("self_consistency.length_censored_sample_count is incorrect")
    if schema_version >= 5 and (
        value.get("protocol_censored_sample_count") != protocol_censored_count
    ):
        errors.append("self_consistency.protocol_censored_sample_count is incorrect")
    if completed_count + censored_count != cell.n_samples:
        errors.append("self_consistency does not retain one valid outcome per sample")

    aggregate_fields = (
        "majority",
        "self_consistency_conf",
        "semantic_entropy_conf",
        "semantic_entropy",
    )
    if censored_count:
        for field in aggregate_fields:
            if value.get(field) is not None:
                errors.append(
                    f"self_consistency.{field} must be undefined when any sample censors"
                )
    elif completed_count == cell.n_samples:
        answers = [output.get("answer") for output in completed_outputs]
        expected_majority = majority_answer(answers)
        expected_consistency = self_consistency_conf(
            answers, primary_answer or ""
        )
        expected_semantic_conf, expected_entropy = semantic_entropy_conf(
            [answer or "" for answer in answers]
        )
        if value.get("majority") != expected_majority:
            errors.append("self_consistency.majority does not match retained samples")
        for field, expected in (
            ("self_consistency_conf", expected_consistency),
            ("semantic_entropy_conf", expected_semantic_conf),
            ("semantic_entropy", expected_entropy),
        ):
            observed = value.get(field)
            if not _is_finite_number(observed) or not math.isclose(
                float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
            ):
                errors.append(f"self_consistency.{field} does not match retained samples")

    auxiliary = value.get("auxiliary_efficiency_raw")
    expected_auxiliary_fields = {
        "n_samples",
        "completed_samples",
        "length_censored_samples",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_reasoning_tokens",
        "wall_ms",
    }
    if schema_version >= 5:
        expected_auxiliary_fields.add("protocol_censored_samples")
    if not isinstance(auxiliary, dict):
        errors.append("self_consistency.auxiliary_efficiency_raw must be an object")
    else:
        if set(auxiliary) != expected_auxiliary_fields:
            errors.append(
                "self_consistency.auxiliary_efficiency_raw fields are not registered"
            )
        def retained_count(item: Mapping[str, Any], field: str) -> int:
            observed = item.get(field)
            return int(observed) if _is_integer(observed) else 0

        expected_costs = {
            "n_samples": cell.n_samples,
            "completed_samples": completed_count,
            "length_censored_samples": length_censored_count,
            "total_prompt_tokens": sum(
                retained_count(output, "prompt_tokens") for output in completed_outputs
            )
            + sum(
                retained_count(censor, "prompt_tokens") for censor in censored_outputs
            ),
            "total_completion_tokens": sum(
                retained_count(output, "completion_tokens")
                for output in completed_outputs
            )
            + sum(
                retained_count(censor, "completion_tokens")
                for censor in censored_outputs
            ),
            "total_reasoning_tokens": sum(
                retained_count(output, "reasoning_tokens")
                for output in completed_outputs
            )
            + sum(
                _auxiliary_censored_reasoning_tokens(censor, cell)
                for censor in censored_outputs
            ),
        }
        if schema_version >= 5:
            expected_costs["protocol_censored_samples"] = protocol_censored_count
        for field, expected in expected_costs.items():
            if auxiliary.get(field) != expected:
                errors.append(
                    f"self_consistency.auxiliary_efficiency_raw.{field} is incorrect"
                )
        if not _is_finite_number(auxiliary.get("wall_ms"), minimum=0.0):
            errors.append(
                "self_consistency.auxiliary_efficiency_raw.wall_ms must be finite and non-negative"
            )
    return errors


def serving_provenance_for_record(
    cell: ExperimentCell, record: Mapping[str, Any]
) -> QuestionServingProvenance:
    """Resolve and validate the serving provenance of one question row.

    An entirely absent (or null) provenance trio denotes a historical pre-contract row.
    Those rows are inferred as the standard profile named by ``cell.model_size``.  A
    partially populated trio is rejected: silently filling it would conceal ambiguous
    runtime provenance.
    """
    values = tuple(record.get(field) for field in _QUESTION_PROVENANCE_FIELDS)
    if all(value is None for value in values):
        profile = get_serving_profile(cell.model_size)
        return QuestionServingProvenance(
            serving_profile=profile.name,
            effective_context_limit=profile.max_model_len,
            tensor_parallel_size=profile.tp_size,
            inferred=True,
        )
    if any(value is None for value in values):
        raise CorruptArtifactError(
            "question serving provenance must provide profile, context limit, and TP size together"
        )

    profile_name, context_limit, tp_size = values
    if not isinstance(profile_name, str) or not profile_name:
        raise CorruptArtifactError("question serving_profile must be a non-empty string")
    if (
        not isinstance(context_limit, int)
        or isinstance(context_limit, bool)
        or context_limit <= 0
    ):
        raise CorruptArtifactError(
            "question effective_context_limit must be a positive integer"
        )
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size <= 0:
        raise CorruptArtifactError("question tensor_parallel_size must be a positive integer")
    try:
        profile = get_serving_profile(profile_name)
    except KeyError as exc:
        raise CorruptArtifactError(f"unknown question serving_profile {profile_name!r}") from exc
    if profile.model_size != cell.model_size:
        raise CorruptArtifactError(
            "question serving_profile does not serve the manifest model size"
        )
    if context_limit != profile.max_model_len:
        raise CorruptArtifactError(
            "question effective_context_limit does not match serving_profile"
        )
    if tp_size != profile.tp_size:
        raise CorruptArtifactError(
            "question tensor_parallel_size does not match serving_profile"
        )
    return QuestionServingProvenance(
        serving_profile=profile.name,
        effective_context_limit=context_limit,
        tensor_parallel_size=tp_size,
        inferred=False,
    )


def summarize_serving_provenance(
    cell: ExperimentCell, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return deterministic cell-level provenance fields for canonical rows.

    Homogeneous cells report their concrete profile and runtime dimensions.  Cells that
    retain rows from more than one layout report ``serving_profile="mixed"`` and null
    scalar dimensions; the exact profile counts remain available for analysis.  Inferred
    counts make the legacy-row assumption auditable rather than silently laundering it as
    directly observed provenance.
    """
    counts: dict[str, int] = {}
    inferred_counts: dict[str, int] = {}
    layouts: dict[str, QuestionServingProvenance] = {}
    for record in records:
        provenance = serving_provenance_for_record(cell, record)
        name = provenance.serving_profile
        counts[name] = counts.get(name, 0) + 1
        layouts.setdefault(name, provenance)
        if provenance.inferred:
            inferred_counts[name] = inferred_counts.get(name, 0) + 1

    ordered_counts = {name: counts[name] for name in sorted(counts)}
    ordered_inferred = {
        name: inferred_counts[name] for name in sorted(inferred_counts)
    }
    if len(layouts) == 1:
        only = next(iter(layouts.values()))
        profile_name: str | None = only.serving_profile
        context_limit: int | None = only.effective_context_limit
        tp_size: int | None = only.tensor_parallel_size
    elif layouts:
        profile_name = "mixed"
        context_limit = None
        tp_size = None
    else:
        profile_name = None
        context_limit = None
        tp_size = None
    return {
        "serving_profile": profile_name,
        "effective_context_limit": context_limit,
        "tensor_parallel_size": tp_size,
        "serving_profile_counts": ordered_counts,
        "serving_profile_inferred_counts": ordered_inferred,
    }


def summarize_endpoint_generation(
    records: Sequence[Mapping[str, Any]],
) -> str | None:
    """Summarize exact nested serving-process identities across canonical rows.

    Current AgentOutput and censor payloads bind every stochastic request to an endpoint
    process.  The cell/QID summary is the concrete identity when homogeneous and the
    literal ``"mixed"`` when endpoint recovery crossed process generations.  Nested
    identities remain the authoritative detail; a pre-existing top-level summary is
    deliberately ignored so it cannot certify itself.
    """

    generations: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            endpoint = value.get("endpoint_generation")
            if isinstance(endpoint, str) and endpoint and endpoint != "mixed":
                generations.add(endpoint)
            for key, nested in value.items():
                if key != "endpoint_generation":
                    collect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                collect(nested)

    for record in records:
        # Do not inspect the result row's own summary field.
        for field, value in record.items():
            if field != "endpoint_generation":
                collect(value)
    if not generations:
        return None
    if len(generations) == 1:
        return next(iter(generations))
    return "mixed"


@lru_cache(maxsize=None)
def _expected_questions_cached(
    benchmark: str, n_questions: int | None, seed: int
) -> tuple[Question, ...]:
    questions = tuple(load_benchmark(benchmark, n=n_questions, seed=seed))
    qids = tuple(question.qid for question in questions)
    if len(qids) != len(set(qids)):
        raise ValueError(f"benchmark loader produced duplicate qids for {benchmark!r}")
    return questions


def expected_questions_for_cell(cell: ExperimentCell) -> tuple[Question, ...]:
    """Return the exact deterministic benchmark contract for ``cell``."""

    return _expected_questions_cached(cell.benchmark, cell.n_questions, cell.seed)


def expected_qids_for_cell(cell: ExperimentCell) -> tuple[str, ...]:
    """Return the exact deterministic QID contract for ``cell``."""
    return tuple(question.qid for question in expected_questions_for_cell(cell))


def clear_expected_qid_cache() -> None:
    """Clear the loader-derived QID cache (primarily useful to isolated tests)."""
    _expected_questions_cached.cache_clear()


def _normalise_question_contracts(
    cell: ExperimentCell,
    expected_qids: Iterable[str] | None,
    expected_questions: Iterable[Question] | None,
    *,
    cell_directory: str | os.PathLike | None = None,
    verified_benchmark_contracts: FrozenBenchmarkContracts | None = None,
    verified_manifest: ManifestSnapshot | None = None,
    structural_only: bool = False,
) -> _ResolvedQuestionContract:
    """Resolve ordered QIDs and, when supplied/derived, their scientific truth.

    Scientific callers always derive or supply Questions. ``structural_only`` is a
    deliberately private escape hatch for isolated parser tests and incident tooling;
    it must never be used to certify completion.

    A directory shaped as ``<run>/cells/<cell_id>`` identifies a real frozen run.  If
    Questions were not supplied by an already-verified caller, load them through that
    run's immutable benchmark sidecar rather than trusting the current loader alone.
    """

    questions: tuple[Question, ...] | None
    frozen_entry: Mapping[str, Any] | None = None
    cdir = None if cell_directory is None else Path(cell_directory)
    run_root = (
        cdir.parent.parent
        if cdir is not None and cdir.parent.name == "cells"
        else None
    )
    if not structural_only and run_root is not None:
        manifest = load_manifest(run_root) if verified_manifest is None else verified_manifest
        try:
            manifest_root_matches = manifest.path.parent.resolve() == run_root.resolve()
        except OSError:
            manifest_root_matches = False
        matching_cells = [
            manifest_cell
            for manifest_cell in manifest.cells
            if manifest_cell.cell_id == cell.cell_id
        ]
        if (
            not manifest_root_matches
            or len(matching_cells) != 1
            or matching_cells[0] != cell
        ):
            raise ValueError(
                "cell is not the exact entry in the frozen run manifest"
            )
    if expected_questions is None:
        if structural_only:
            questions = None
        else:
            if run_root is None:
                questions = expected_questions_for_cell(cell)
            else:
                if cdir.name != cell.cell_id:
                    raise ValueError(
                        "cell directory name does not match the manifest cell_id"
                    )
                questions, frozen_entry = load_verified_questions(run_root, cell)
    else:
        questions = tuple(expected_questions)
        # A caller-supplied Question sequence is useful for run-scoped caching, but it is
        # not itself proof of scientific identity.  On a real run path, always compare it
        # with the immutable sidecar.  Trusted operational callers may pass the already
        # verified run object to avoid reparsing the manifest and sidecar for every cell.
        if not structural_only and run_root is not None:
            if cdir is None or cdir.name != cell.cell_id:
                raise ValueError(
                    "cell directory name does not match the manifest cell_id"
                )
            frozen = (
                load_frozen_benchmark_contracts(run_root)
                if verified_benchmark_contracts is None
                else verified_benchmark_contracts
            )
            try:
                if frozen.path.parent.resolve() != run_root.resolve():
                    raise ValueError(
                        "verified benchmark contracts belong to a different run"
                    )
            except OSError as exc:
                raise ValueError("cannot resolve frozen benchmark contract path") from exc
            frozen_entry = frozen.verify_questions(cell, questions)
    if questions is None:
        if expected_qids is None:
            raise ValueError("structural-only validation requires explicit expected_qids")
        return _ResolvedQuestionContract(
            ordered_qids=_normalise_expected_qids(cell, expected_qids),
            questions_by_qid=None,
            question_sha256_by_qid=None,
            benchmark_contract_sha256=None,
        )

    question_qids = tuple(question.qid for question in questions)
    qids = (
        question_qids
        if expected_qids is None
        else _normalise_expected_qids(cell, expected_qids)
    )
    if qids != question_qids:
        raise ValueError("expected question order/QIDs do not match expected_qids")
    if any(question.benchmark != cell.benchmark for question in questions):
        raise ValueError("expected question benchmark does not match manifest cell")
    if len(question_qids) != len(set(question_qids)):
        raise ValueError("expected question QIDs must be unique")

    contract = (
        dict(frozen_entry)
        if frozen_entry is not None
        else build_question_contract(BenchmarkContractKey.from_cell(cell), questions)
    )
    observed_hashes = tuple(canonical_question_sha256(question) for question in questions)
    contract_qids = tuple(contract["ordered_qids"])
    contract_hashes = tuple(contract["question_sha256s"])
    if contract_qids != question_qids or contract_hashes != observed_hashes:
        # A frozen entry reaches this branch only if the contract verifier itself has
        # regressed.  An explicit Question set reaches it only if the shared builder and
        # canonical hash disagree.  Either case must fail closed.
        raise ValueError("normalized Questions do not match their benchmark contract")
    contract_sha256 = contract.get("question_contract_sha256")
    if not isinstance(contract_sha256, str) or _SHA256_RE.fullmatch(contract_sha256) is None:
        raise ValueError("benchmark contract has an invalid ordered contract SHA-256")
    return _ResolvedQuestionContract(
        ordered_qids=qids,
        questions_by_qid=dict(zip(question_qids, questions, strict=True)),
        question_sha256_by_qid=dict(
            zip(question_qids, observed_hashes, strict=True)
        ),
        benchmark_contract_sha256=contract_sha256,
    )


def _normalise_expected_qids(
    cell: ExperimentCell, expected_qids: Iterable[str] | None
) -> tuple[str, ...]:
    qids = expected_qids_for_cell(cell) if expected_qids is None else tuple(expected_qids)
    if any(not isinstance(qid, str) or not qid for qid in qids):
        raise ValueError("expected qids must be non-empty strings")
    if len(qids) != len(set(qids)):
        raise ValueError("expected qids must be unique")
    return qids


def _observed_topology_coordinate_errors(
    value: Any,
    *,
    cell: ExperimentCell,
    qid: str,
    serving_profile: Any,
    effective_context_limit: Any,
    question: Question | None,
    terminal_censor: Any,
) -> tuple[list[str], dict[str, int]]:
    """Validate schema-5 censor snapshots and derive their exact consumed costs."""

    empty_costs = {
        "n_turns": 0,
        "n_messages": 0,
        "n_rounds": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0,
    }
    if not isinstance(value, list) or not value:
        return ["censored schema-5 result requires observed topology coordinates"], empty_costs

    errors: list[str] = []
    keys: set[str] = set()
    rounds: set[int] = set()
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    message_count = 0
    observed_censors: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    observed_schedule: dict[tuple[int, int], str] = {}
    expected_entry_fields = {
        "coordinate_key",
        "request",
        "outcome",
        "observed_at",
        "producer_wall_ms",
    }
    expected_request_fields = {
        "generation_role",
        "qid",
        "agent_id",
        "round",
        "seed",
        "sample_index",
        "peer_context",
        "max_tokens",
        "elicit_cot",
    }
    for index, entry in enumerate(value):
        label = f"observed_topology_coordinates[{index}]"
        if not isinstance(entry, dict) or set(entry) != expected_entry_fields:
            errors.append(f"{label} has the wrong fields")
            continue
        key = entry.get("coordinate_key")
        request = entry.get("request")
        outcome = entry.get("outcome")
        if not isinstance(key, str) or not key or key in keys:
            errors.append(f"{label}.coordinate_key must be unique non-empty text")
        else:
            keys.add(key)
        if not _valid_timestamp(entry.get("observed_at")):
            errors.append(f"{label}.observed_at must be finite and positive")
        if not _is_finite_number(entry.get("producer_wall_ms"), minimum=0.0):
            errors.append(f"{label}.producer_wall_ms must be finite and non-negative")
        if not isinstance(request, dict):
            errors.append(f"{label}.request must be an object")
            continue
        if set(request) != expected_request_fields:
            errors.append(f"{label}.request has the wrong fields")
        agent_id = request.get("agent_id")
        round_index = request.get("round")
        expected_key = f"topology:{agent_id}:{round_index}"
        if key != expected_key:
            errors.append(f"{label}.coordinate_key does not match request")
        if request.get("generation_role") != "topology":
            errors.append(f"{label}.request generation_role must be topology")
        if request.get("qid") != qid:
            errors.append(f"{label}.request qid does not match result")
        if request.get("sample_index") is not None:
            errors.append(f"{label}.request sample_index must be null")
        if request.get("max_tokens") != ANSWER_GENERATION_TOKEN_ALLOWANCE:
            errors.append(f"{label}.request max_tokens is not registered")
        if request.get("elicit_cot") is not True:
            errors.append(f"{label}.request elicit_cot must be true")
        peer = request.get("peer_context")
        if not isinstance(peer, dict) or set(peer) != {"sha256", "utf8_bytes"}:
            errors.append(f"{label}.request peer_context identity is invalid")
        elif (
            not isinstance(peer.get("sha256"), str)
            or not _SHA256_RE.fullmatch(peer["sha256"])
            or not _is_integer(peer.get("utf8_bytes"))
        ):
            errors.append(f"{label}.request peer_context identity is invalid")
        expected_seed = _expected_agent_seed(
            cell, {"agent_id": agent_id, "round": round_index}
        )
        if expected_seed is None or request.get("seed") != expected_seed:
            errors.append(f"{label}.request coordinate/seed is not on the manifest schedule")
        if _is_integer(round_index):
            rounds.add(int(round_index))
        agent_match = (
            re.fullmatch(r"agent(\d+)", agent_id)
            if isinstance(agent_id, str)
            else None
        )
        if (
            agent_match is not None
            and _is_integer(round_index)
            and expected_seed is not None
        ):
            coordinate = (int(agent_match.group(1)), int(round_index))
            if coordinate in observed_schedule:
                errors.append(f"{label}.request duplicates a topology coordinate")
            else:
                observed_schedule[coordinate] = str(
                    outcome.get("termination_status")
                    if isinstance(outcome, dict)
                    else "invalid"
                )
        expected_peer_blocks = _expected_peer_block_count(cell, request)
        if expected_peer_blocks is not None:
            message_count += expected_peer_blocks

        if not isinstance(outcome, dict) or set(outcome) != {
            "termination_status",
            "agent_output",
            "censored_generation",
        }:
            errors.append(f"{label}.outcome has the wrong fields")
            continue
        status = outcome.get("termination_status")
        output = outcome.get("agent_output")
        censor = outcome.get("censored_generation")
        if status == TERMINATION_COMPLETED:
            if not isinstance(output, dict) or censor is not None:
                errors.append(f"{label} completed outcome is malformed")
                continue
            normalized = dict(output)
            normalized["answer"] = normalized.pop("answer_choice", None)
            if (
                output.get("agent_id") != agent_id
                or output.get("round") != round_index
            ):
                errors.append(
                    f"{label}.agent_output coordinate does not match request"
                )
            errors.extend(
                f"{label}.agent_output: {error}"
                for error in _agent_errors(
                    normalized,
                    current_schema=True,
                    cell=cell,
                    effective_context_limit=effective_context_limit,
                    scheduled_seed=(
                        int(request["seed"])
                        if _is_integer(request.get("seed"))
                        else None
                    ),
                    question=question,
                )
            )
            if (
                isinstance(peer, dict)
                and isinstance(peer.get("sha256"), str)
                and output.get("peer_context_sha256") != peer.get("sha256")
            ):
                errors.append(
                    f"{label}.agent_output peer-context hash does not match request"
                )
            prompt_tokens += (
                int(output["prompt_tokens"])
                if _is_integer(output.get("prompt_tokens"))
                else 0
            )
            completion_tokens += (
                int(output["completion_tokens"])
                if _is_integer(output.get("completion_tokens"))
                else 0
            )
            reasoning_tokens += (
                int(output["reasoning_tokens"])
                if _is_integer(output.get("reasoning_tokens"))
                else 0
            )
        elif status in {
            TERMINATION_LENGTH_CENSORED,
            TERMINATION_PROTOCOL_CENSORED,
        }:
            if output is not None or not isinstance(censor, dict):
                errors.append(f"{label} censored outcome is malformed")
                continue
            if any(
                censor.get(field) != request.get(field)
                for field in (
                    "qid",
                    "agent_id",
                    "round",
                    "seed",
                    "generation_role",
                    "sample_index",
                )
            ):
                errors.append(
                    f"{label}.censored_generation coordinate does not match request"
                )
            errors.extend(
                f"{label}: {error}"
                for error in _censored_generation_errors(
                    censor,
                    cell=cell,
                    qid=qid,
                    termination_status=status,
                    serving_profile=serving_profile,
                    effective_context_limit=effective_context_limit,
                )
            )
            observed_censors.append((request, censor))
            prompt_tokens += (
                int(censor["prompt_tokens"])
                if _is_integer(censor.get("prompt_tokens"))
                else 0
            )
            completion_tokens += (
                int(censor["completion_tokens"])
                if _is_integer(censor.get("completion_tokens"))
                else 0
            )
            reasoning_tokens += _auxiliary_censored_reasoning_tokens(censor, cell)
        else:
            errors.append(f"{label}.outcome termination_status is not registered")

    # A topology censor may terminate only after a complete concurrent wave.  Earlier
    # rounds must be complete; siblings in the terminal wave may independently finish
    # or censor because ThreadPoolExecutor waits for all submitted futures before the
    # first exception is propagated.
    if observed_schedule:
        terminal_round = max(round_index for _, round_index in observed_schedule)
        prior_censors = [
            coordinate
            for coordinate, status in observed_schedule.items()
            if coordinate[1] < terminal_round
            and status
            in {TERMINATION_LENGTH_CENSORED, TERMINATION_PROTOCOL_CENSORED}
        ]
        if prior_censors:
            errors.append("censored topology snapshot continues after a censor")

        n_agents = 1 if cell.topology.value == "single_agent" else cell.n_agents
        all_agents = set(range(n_agents))
        expected_coordinates: set[tuple[int, int]] = set()
        topology = cell.topology.value
        if topology in {"single_agent", "independent"}:
            expected_coordinates = {(index, 0) for index in all_agents}
            if terminal_round != 0:
                errors.append("censored topology snapshot has an invalid terminal round")
        elif topology == "decentralized":
            expected_coordinates = {
                (index, round_index)
                for round_index in range(terminal_round + 1)
                for index in all_agents
            }
        else:  # centralized: a sub-agent censor prevents that round's orchestrator.
            expected_coordinates = {
                (index, round_index)
                for round_index in range(terminal_round)
                for index in all_agents
            }
            terminal_subagents = {
                index
                for (index, round_index), status in observed_schedule.items()
                if round_index == terminal_round
                and index != 0
                and status
                in {TERMINATION_LENGTH_CENSORED, TERMINATION_PROTOCOL_CENSORED}
            }
            if terminal_subagents:
                expected_coordinates.update(
                    (index, terminal_round) for index in all_agents if index != 0
                )
            else:
                expected_coordinates.update(
                    (index, terminal_round) for index in all_agents
                )
                if observed_schedule.get((0, terminal_round)) not in {
                    TERMINATION_LENGTH_CENSORED,
                    TERMINATION_PROTOCOL_CENSORED,
                }:
                    errors.append(
                        "centralized terminal wave must contain an orchestrator censor"
                    )
        if set(observed_schedule) != expected_coordinates:
            errors.append(
                "censored topology coordinates do not form an exact legal terminal wave"
            )
        if terminal_round >= (
            1 if topology in {"single_agent", "independent"} else cell.rounds
        ):
            errors.append("censored topology snapshot exceeds manifest rounds")

    if not observed_censors:
        errors.append("censored topology snapshot must contain at least one censor")
    elif isinstance(terminal_censor, dict):
        comparable_terminal = {
            key: item
            for key, item in terminal_censor.items()
            if key not in {"topology", "benchmark"}
        }
        terminal_matches = [
            (request, observed)
            for request, observed in observed_censors
            if dict(observed) == comparable_terminal
        ]
        if len(terminal_matches) != 1:
            errors.append(
                "terminal censor must match exactly one observed coordinate"
            )
        else:
            terminal_request, _ = terminal_matches[0]
            terminal_round = terminal_request.get("round")
            terminal_agent = terminal_request.get("agent_id")
            terminal_agent_match = (
                re.fullmatch(r"agent(\d+)", terminal_agent)
                if isinstance(terminal_agent, str)
                else None
            )
            wave = [
                (request, observed)
                for request, observed in observed_censors
                if request.get("round") == terminal_round
                and (
                    cell.topology.value != "centralized"
                    or (
                        terminal_agent == "agent0"
                        and request.get("agent_id") == "agent0"
                    )
                    or (
                        terminal_agent != "agent0"
                        and request.get("agent_id") != "agent0"
                    )
                )
            ]
            ordered_wave = sorted(
                wave,
                key=lambda item: int(
                    re.fullmatch(r"agent(\d+)", str(item[0].get("agent_id"))).group(1)
                ),
            ) if all(
                re.fullmatch(r"agent(\d+)", str(item[0].get("agent_id")))
                for item in wave
            ) else []
            if (
                terminal_agent_match is None
                or not ordered_wave
                or ordered_wave[0][0] != terminal_request
            ):
                errors.append(
                    "terminal censor is not the deterministic map-order censor"
                )
    return errors, {
        "n_turns": len(value),
        "n_messages": message_count,
        "n_rounds": len(rounds),
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_reasoning_tokens": reasoning_tokens,
    }


def _record_errors(
    record: Any,
    cell: ExperimentCell,
    expected: set[str],
    questions_by_qid: Mapping[str, Question] | None = None,
    question_sha256_by_qid: Mapping[str, str] | None = None,
    benchmark_contract_sha256: str | None = None,
) -> tuple[list[str], str | None, bool]:
    """Return (errors, qid, unexpected_qid)."""
    if not isinstance(record, dict):
        return ["row is not a JSON object"], None, False
    qid = record.get("qid")
    if not isinstance(qid, str) or not qid:
        return ["qid is missing or not a non-empty string"], None, False
    if qid not in expected:
        return [], qid, True

    errors: list[str] = []
    current_schema, schema_errors = _schema_contract(record, "question result")
    errors.extend(schema_errors)
    record_schema = record.get("schema_version") if current_schema else None
    termination_status = record.get("termination_status", TERMINATION_COMPLETED)
    if current_schema:
        allowed_statuses = (
            SCHEMA_4_TERMINATION_STATUSES
            if record_schema == 4
            else TERMINATION_STATUSES
        )
        if termination_status not in allowed_statuses:
            errors.append(
                "question result termination_status is not registered"
            )
    elif "termination_status" in record or "censored_generation" in record:
        errors.append(
            "legacy question result cannot contain termination/censor provenance"
        )
    is_censored = current_schema and termination_status in {
        TERMINATION_LENGTH_CENSORED,
        TERMINATION_PROTOCOL_CENSORED,
    }
    exact_fields = {
        "cell_id": cell.cell_id,
        "benchmark": cell.benchmark,
        "model_size": cell.model_size,
        "topology": cell.topology.value,
        "context_share_level": cell.context_share_level.value,
        "prompt_complexity_level": cell.prompt_complexity_level,
        "reasoning_level": cell.reasoning_level.value,
    }
    for field, expected_value in exact_fields.items():
        if record.get(field) != expected_value:
            errors.append(f"{field} does not match manifest cell")
    if not _valid_timestamp(record.get("timestamp")):
        errors.append("timestamp must be finite and positive")
    if record.get("final_answer") is not None and not isinstance(record.get("final_answer"), str):
        errors.append("final_answer must be a string or null")
    if not isinstance(record.get("answer_key"), str):
        errors.append("answer_key must be a string")
    if not isinstance(record.get("correct"), bool):
        errors.append("correct must be a boolean")
    question = None if questions_by_qid is None else questions_by_qid.get(qid)
    if question is not None:
        if record.get("answer_key") != question.answer_key:
            errors.append("answer_key does not match the benchmark question")
        final_answer = record.get("final_answer")
        if final_answer is None or isinstance(final_answer, str):
            expected_correct = (
                False if final_answer is None else grade(question, final_answer)
            )
            if isinstance(record.get("correct"), bool) and record.get("correct") != expected_correct:
                errors.append("correct does not match benchmark grading of final_answer")
    if current_schema and question_sha256_by_qid is not None:
        if record.get("question_sha256") != question_sha256_by_qid[qid]:
            errors.append(
                "question_sha256 does not match the normalized benchmark question"
            )
        if record.get("benchmark_contract_sha256") != benchmark_contract_sha256:
            errors.append(
                "benchmark_contract_sha256 does not match the ordered benchmark contract"
            )
    elif not current_schema:
        present_benchmark_provenance = [
            field for field in _BENCHMARK_PROVENANCE_FIELDS if field in record
        ]
        if present_benchmark_provenance:
            errors.append(
                "legacy question result must not contain benchmark provenance without "
                "schema_version: "
                + ", ".join(present_benchmark_provenance)
            )
    system_conf = record.get("system_conf")
    if not isinstance(system_conf, dict):
        errors.append("system_conf must be an object")
    else:
        for name, confidence in system_conf.items():
            if not isinstance(name, str) or not name:
                errors.append("system_conf keys must be non-empty strings")
                break
            if (
                not _is_finite_number(confidence)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                errors.append("system_conf values must be finite in [0, 1]")
                break

    self_consistency = record.get("self_consistency")
    if current_schema and not is_censored:
        errors.extend(
            _self_consistency_errors(
                self_consistency,
                cell=cell,
                qid=qid,
                primary_answer=record.get("final_answer"),
                serving_profile=record.get("serving_profile"),
                effective_context_limit=record.get("effective_context_limit"),
                question=question,
                schema_version=(
                    int(record_schema)
                    if isinstance(record_schema, int)
                    and record_schema in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
                    else ARTIFACT_SCHEMA_VERSION
                ),
            )
        )
    elif not isinstance(self_consistency, dict):
        errors.append("self_consistency must be an object")
    elif not current_schema:
        if "samples" in self_consistency and not (
            isinstance(self_consistency["samples"], list)
            and all(
                sample is None or isinstance(sample, str)
                for sample in self_consistency["samples"]
            )
        ):
            errors.append("self_consistency.samples must be a list of strings/nulls")
        if "majority" in self_consistency and self_consistency["majority"] is not None and not isinstance(
            self_consistency["majority"], str
        ):
            errors.append("self_consistency.majority must be a string or null")
        for field in ("self_consistency_conf", "semantic_entropy_conf"):
            if field in self_consistency and (
                not _is_finite_number(self_consistency[field])
                or not 0.0 <= float(self_consistency[field]) <= 1.0
            ):
                errors.append(f"self_consistency.{field} must be finite in [0, 1]")
        if "semantic_entropy" in self_consistency and not _is_finite_number(
            self_consistency["semantic_entropy"], minimum=0.0
        ):
            errors.append("self_consistency.semantic_entropy must be finite and non-negative")

    per_agent = record.get("per_agent")
    if not isinstance(per_agent, list):
        errors.append("per_agent must be a list")
        per_agent = []
    elif not per_agent and not is_censored:
        errors.append("per_agent must contain at least one agent output")
    if is_censored:
        if per_agent:
            errors.append(
                "censored result must not publish partial agent outputs"
            )
        if record.get("final_answer") is not None:
            errors.append("censored result final_answer must be null")
        if record.get("correct") is not False:
            errors.append("censored result must count as incorrect")
        if record.get("system_conf") != {}:
            errors.append("censored result system_conf must be empty")
        if record.get("self_consistency") != {}:
            errors.append("censored result self_consistency must be empty")
        errors.extend(
            _censored_generation_errors(
                record.get("censored_generation"),
                cell=cell,
                qid=qid,
                termination_status=termination_status,
                serving_profile=record.get("serving_profile"),
                effective_context_limit=record.get("effective_context_limit"),
            )
        )
    else:
        if current_schema and record.get("censored_generation") is not None:
            errors.append("completed result cannot contain censored_generation data")
        for index, agent in enumerate(per_agent):
            errors.extend(
                f"per_agent[{index}]: {error}"
                for error in _agent_errors(
                    agent,
                    current_schema=current_schema,
                    cell=cell,
                    effective_context_limit=record.get("effective_context_limit"),
                    question=question,
                )
            )
    mapping_agents = [agent for agent in per_agent if isinstance(agent, Mapping)]
    if len(mapping_agents) == len(per_agent) and all(
        isinstance(agent.get("agent_id"), str) and _is_integer(agent.get("round"))
        for agent in mapping_agents
    ):
        pairs = [
            (agent.get("agent_id"), agent.get("round")) for agent in mapping_agents
        ]
        if len(pairs) != len(set(pairs)):
            errors.append("per_agent contains duplicate (agent_id, round) outputs")
        elif not is_censored:
            errors.extend(
                _topology_outcome_errors(
                    record,
                    mapping_agents,
                    cell,
                    current_schema=current_schema,
                )
            )
    observed_costs: dict[str, int] | None = None
    if record_schema == ARTIFACT_SCHEMA_VERSION:
        observed_coordinates = record.get("observed_topology_coordinates")
        if is_censored:
            coordinate_errors, observed_costs = _observed_topology_coordinate_errors(
                observed_coordinates,
                cell=cell,
                qid=qid,
                serving_profile=record.get("serving_profile"),
                effective_context_limit=record.get("effective_context_limit"),
                question=question,
                terminal_censor=record.get("censored_generation"),
            )
            errors.extend(coordinate_errors)
        elif observed_coordinates != []:
            errors.append(
                "completed schema-5 result observed_topology_coordinates must be empty"
            )
    elif current_schema and "observed_topology_coordinates" in record:
        errors.append(
            "schema-4 result cannot contain schema-5 topology-coordinate provenance"
        )

    if is_censored:
        efficiency = record.get("efficiency_raw")
        if not isinstance(efficiency, dict):
            errors.append("censored efficiency_raw must be an object")
        else:
            expected_agents = 1 if cell.topology.value == "single_agent" else cell.n_agents
            if record_schema == ARTIFACT_SCHEMA_VERSION and observed_costs is not None:
                for field, expected_value in observed_costs.items():
                    if efficiency.get(field) != expected_value:
                        errors.append(
                            f"censored efficiency_raw.{field} does not match "
                            "observed topology coordinates"
                        )
            else:
                for field in (
                    "n_turns",
                    "n_messages",
                    "n_rounds",
                    "total_reasoning_tokens",
                ):
                    if efficiency.get(field) != 0:
                        errors.append(
                            f"schema-4 length-censored efficiency_raw.{field} must be zero"
                        )
            if efficiency.get("n_agents") != expected_agents:
                errors.append(
                    "censored efficiency_raw.n_agents does not match topology"
                )
            censor = record.get("censored_generation")
            if isinstance(censor, dict) and record_schema == 4:
                for output_field, censor_field in (
                    ("total_prompt_tokens", "prompt_tokens"),
                    ("total_completion_tokens", "completion_tokens"),
                ):
                    if efficiency.get(output_field) != censor.get(censor_field):
                        errors.append(
                            f"schema-4 length-censored efficiency_raw.{output_field} "
                            "does not match censored generation"
                        )
            if not _is_finite_number(efficiency.get("wall_ms"), minimum=0.0):
                errors.append(
                    "censored efficiency_raw.wall_ms must be finite and non-negative"
                )
    else:
        errors.extend(
            _efficiency_errors(record.get("efficiency_raw"), mapping_agents, cell)
        )

    provenance_keys = [field for field in _QUESTION_PROVENANCE_FIELDS if field in record]
    if not current_schema and provenance_keys:
        errors.append(
            "legacy question result must not contain serving provenance without schema_version"
        )
    try:
        provenance = serving_provenance_for_record(cell, record)
        if current_schema and provenance.inferred:
            errors.append("current question result requires explicit serving provenance")
    except CorruptArtifactError as exc:
        errors.append(str(exc))
    return errors, qid, False


def read_canonical_results(
    cell: ExperimentCell,
    cell_directory: str | os.PathLike,
    *,
    expected_qids: Iterable[str] | None = None,
    expected_questions: Iterable[Question] | None = None,
    verified_benchmark_contracts: FrozenBenchmarkContracts | None = None,
    verified_manifest: ManifestSnapshot | None = None,
    _structural_only: bool = False,
    _resolved_question_contract: _ResolvedQuestionContract | None = None,
) -> CanonicalResults:
    """Parse results and retain the first valid record for every expected QID.

    Parsing never mutates the source.  Malformed lines, invalid records, unexpected QIDs,
    and later valid duplicates are reported so callers do not accidentally call a dirty
    file complete.
    """
    cdir = Path(cell_directory)
    path = cdir / RESULTS_FILENAME
    question_contract = (
        _normalise_question_contracts(
            cell,
            expected_qids,
            expected_questions,
            cell_directory=cdir,
            verified_benchmark_contracts=verified_benchmark_contracts,
            verified_manifest=verified_manifest,
            structural_only=_structural_only,
        )
        if _resolved_question_contract is None
        else _resolved_question_contract
    )
    expected_order = question_contract.ordered_qids
    expected_set = set(expected_order)
    if not path.exists():
        return CanonicalResults(records=(), raw_rows=0)

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return CanonicalResults(
            records=(), raw_rows=0, validation_errors=(f"cannot read results: {exc}",)
        )

    by_qid: dict[str, dict[str, Any]] = {}
    seen_valid_order: list[str] = []
    malformed = 0
    invalid = 0
    duplicates: list[str] = []
    unexpected: list[str] = []
    validation_errors: list[str] = []
    raw_rows = 0
    for line_number, line in enumerate(lines, start=1):
        raw_rows += 1
        if not line.strip():
            malformed += 1
            validation_errors.append(f"line {line_number}: blank JSONL row")
            continue
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, _DuplicateJSONKey, _NonFiniteJSONNumber) as exc:
            malformed += 1
            validation_errors.append(f"line {line_number}: malformed JSON ({exc})")
            continue
        errors, qid, is_unexpected = _record_errors(
            value,
            cell,
            expected_set,
            question_contract.questions_by_qid,
            question_contract.question_sha256_by_qid,
            question_contract.benchmark_contract_sha256,
        )
        if is_unexpected:
            unexpected.append(qid or "<missing>")
            continue
        if errors:
            invalid += 1
            validation_errors.extend(f"line {line_number}: {error}" for error in errors)
            continue
        assert qid is not None
        if qid in by_qid:
            duplicates.append(qid)
            continue
        by_qid[qid] = value
        seen_valid_order.append(qid)

    canonical = tuple(by_qid[qid] for qid in expected_order if qid in by_qid)
    canonical_order = [qid for qid in expected_order if qid in by_qid]
    return CanonicalResults(
        records=canonical,
        raw_rows=raw_rows,
        malformed_lines=malformed,
        invalid_rows=invalid,
        duplicate_qids=tuple(dict.fromkeys(duplicates)),
        unexpected_qids=tuple(dict.fromkeys(unexpected)),
        validation_errors=tuple(validation_errors),
        out_of_order=seen_valid_order != canonical_order,
    )


def canonicalize_results(
    cell: ExperimentCell,
    cell_directory: str | os.PathLike,
    *,
    expected_qids: Iterable[str] | None = None,
    expected_questions: Iterable[Question] | None = None,
    verified_benchmark_contracts: FrozenBenchmarkContracts | None = None,
    verified_manifest: ManifestSnapshot | None = None,
    _structural_only: bool = False,
    write: bool = True,
) -> CanonicalResults:
    """Retain first-valid-per-QID rows and atomically rewrite ``results.jsonl``.

    The caller must hold :func:`cell_lock` whenever ``write`` is true.
    """
    cdir = Path(cell_directory)
    parsed = read_canonical_results(
        cell,
        cdir,
        expected_qids=expected_qids,
        expected_questions=expected_questions,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=verified_manifest,
        _structural_only=_structural_only,
    )
    path = cdir / RESULTS_FILENAME
    if write and path.exists() and parsed.needs_rewrite:
        io.write_jsonl(path, parsed.records)
    return parsed


def reasoning_token_summary(
    records: Sequence[Mapping[str, Any]],
) -> ReasoningTokenSummary:
    """Summarize exact token spans without pooling them with legacy word counts."""

    exact_totals: list[float] = []
    legacy_word_totals: list[float] = []
    for record in records:
        agents = record.get("per_agent")
        if not isinstance(agents, list):
            agents = []
        row_exact = (
            record.get("schema_version") in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS
            and bool(agents)
            and all(
                isinstance(agent, Mapping)
                and agent.get("reasoning_token_source")
                in {"vllm_native_token_ids", "none"}
                for agent in agents
            )
        )
        if row_exact:
            exact_totals.append(
                float(record["efficiency_raw"]["total_reasoning_tokens"])
            )

        word_total = 0.0
        for agent in agents:
            if not isinstance(agent, Mapping):
                continue
            if row_exact:
                word_total += float(agent.get("reasoning_word_count_legacy", 0))
            else:
                # In schema-less rows this field was a whitespace count despite its
                # historical name.  Do not reinterpret it as a tokenizer count.
                word_total += float(agent.get("reasoning_tokens", 0))
        legacy_word_totals.append(word_total)

    exact_count = len(exact_totals)
    nonexact_count = len(records) - exact_count
    exact_mean = math.fsum(exact_totals) / exact_count if exact_count else None
    word_mean = (
        math.fsum(legacy_word_totals) / len(legacy_word_totals)
        if legacy_word_totals
        else 0.0
    )
    return ReasoningTokenSummary(
        all_exact=bool(records) and nonexact_count == 0,
        exact_question_count=exact_count,
        nonexact_question_count=nonexact_count,
        mean_reasoning_tokens=exact_mean if records and nonexact_count == 0 else None,
        mean_reasoning_tokens_exact=exact_mean,
        mean_reasoning_word_count_legacy=word_mean,
    )


def mean_reasoning_tokens(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Compatibility wrapper returning a mean only when every row is exact."""

    return reasoning_token_summary(records).mean_reasoning_tokens


def _historical_mean_reasoning_word_count(
    records: Sequence[Mapping[str, Any]],
) -> float:
    """Reproduce the schema-less metadata statistic for legacy validation only."""

    if not records:
        return 0.0
    totals = [
        float(record["efficiency_raw"]["total_reasoning_tokens"])
        for record in records
    ]
    return math.fsum(totals) / len(totals)


def _valid_profile_counts(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, dict)
        and (allow_empty or bool(value))
        and all(
        isinstance(name, str)
        and bool(name)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        for name, count in value.items()
        )
    )


def _prompt_quality_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["meta prompt_quality must be an object"]
    errors: list[str] = []
    heuristic = value.get("heuristic")
    if not _is_finite_number(heuristic) or not 0.0 <= float(heuristic) <= 100.0:
        errors.append("meta prompt_quality.heuristic must be finite in [0, 100]")
    judge = value.get("llm_judge")
    if judge is not None and (
        not _is_finite_number(judge) or not 0.0 <= float(judge) <= 100.0
    ):
        errors.append("meta prompt_quality.llm_judge must be null or finite in [0, 100]")
    features = value.get("features")
    if not isinstance(features, dict) or not features:
        errors.append("meta prompt_quality.features must be a non-empty object")
    elif not all(
        isinstance(name, str)
        and bool(name)
        and _is_finite_number(feature_value)
        for name, feature_value in features.items()
    ):
        errors.append("meta prompt_quality.features must contain finite numeric values")
    return errors


def _policy_for_cell_directory(
    cell_directory: str | os.PathLike | None,
) -> tuple[ArtifactPolicy | None, str | None]:
    """Resolve a run policy from ``<run>/cells/<cell>`` without affecting legacy runs."""

    if cell_directory is None:
        return None, None
    cdir = Path(cell_directory)
    if cdir.parent.name != "cells":
        return None, None
    try:
        return load_artifact_policy(cdir.parent.parent), None
    except ArtifactPolicyError as exc:
        return None, f"invalid schema-5 artifact policy: {exc}"


def _artifact_policy_payload_errors(
    cell: ExperimentCell,
    policy: ArtifactPolicy,
    records: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any] | None,
    model_contract_path: str | os.PathLike | None = None,
) -> list[str]:
    """Apply the homogeneous production contract without weakening legacy readers."""

    errors: list[str] = []
    for index, record in enumerate(records):
        if record.get("schema_version") != policy.required_artifact_schema_version:
            errors.append(
                f"record {index}: authoritative policy requires artifact schema "
                f"{policy.required_artifact_schema_version}"
            )
    try:
        contracts = load_model_contracts(
            model_contract_path,
            expected_sha256=policy.accepted_model_contract_sha256,
        )
        identity = contracts.for_size(cell.model_size)
    except ModelContractError as exc:
        errors.append(f"schema-5 model contract is invalid: {exc}")
        identity = None

    # Partial/active cells have no success metadata yet, but every retained row must
    # already belong to this exact production contract.  Otherwise resume would keep a
    # first-valid QID from another release and make truthful completion impossible.
    for index, record in enumerate(records):
        row_fixed = {
            "release_id": policy.release.release_id,
            "environment_hash": policy.environment.harness_sha256,
            "model_contract_sha256": policy.accepted_model_contract_sha256,
        }
        if identity is not None:
            row_fixed.update(
                model_revision=identity.model_revision,
                tokenizer_revision=identity.tokenizer_revision,
            )
        for field, expected in row_fixed.items():
            if record.get(field) != expected:
                errors.append(
                    f"record {index}: schema-5 production {field} does not match policy"
                )
        if record.get("effective_context") != record.get("effective_context_limit"):
            errors.append(
                f"record {index}: schema-5 production effective_context is inconsistent"
            )
        expected_row_endpoint = summarize_endpoint_generation([record])
        if (
            expected_row_endpoint is None
            or record.get("endpoint_generation") != expected_row_endpoint
        ):
            errors.append(
                f"record {index}: schema-5 production endpoint_generation is inconsistent"
            )
        if not _is_integer(record.get("rollout_generation"), minimum=1):
            errors.append(
                f"record {index}: schema-5 production rollout_generation must be positive"
            )
    if meta is None:
        return errors
    if meta.get("schema_version") != policy.required_artifact_schema_version:
        errors.append(
            "authoritative policy requires schema-5 completion metadata"
        )
    missing = [
        field for field in policy.required_metadata_fields if field not in meta
    ]
    if missing:
        errors.append(
            "schema-5 production meta is missing required provenance fields: "
            + ", ".join(missing)
        )

    fixed_meta = {
        "release_id": policy.release.release_id,
        "environment_hash": policy.environment.harness_sha256,
        "serving_environment_hash": policy.environment.serving_sha256,
        "model_contract_sha256": policy.accepted_model_contract_sha256,
        "artifact_policy_sha256": policy.file_sha256,
        "git_commit": policy.release.git_commit,
    }
    for field, expected in fixed_meta.items():
        if meta.get(field) != expected:
            errors.append(f"schema-5 production meta {field} does not match policy")

    if identity is not None:
        for field, expected in (
            ("model_revision", identity.model_revision),
            ("tokenizer_revision", identity.tokenizer_revision),
        ):
            if meta.get(field) != expected:
                errors.append(f"schema-5 production meta {field} does not match contract")

    expected_endpoint = summarize_endpoint_generation(records)
    if expected_endpoint is None:
        errors.append("schema-5 production results lack exact endpoint generations")
    if meta.get("endpoint_generation") != expected_endpoint:
        errors.append(
            "schema-5 production meta endpoint_generation does not match canonical results"
        )
    try:
        expected_serving = summarize_serving_provenance(cell, records)
    except CorruptArtifactError as exc:
        errors.append(f"schema-5 production serving provenance is invalid: {exc}")
    else:
        expected_context = expected_serving["effective_context_limit"]
        if meta.get("effective_context") != expected_context:
            errors.append(
                "schema-5 production meta effective_context does not match canonical results"
            )
    rollout = meta.get("rollout_generation")
    if not _is_integer(rollout, minimum=1):
        errors.append("schema-5 production meta rollout_generation must be positive")

    return errors


def _meta_errors(
    cell: ExperimentCell,
    meta: Any,
    expected_qids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    benchmark_contract_sha256: str | None = None,
    artifact_policy: ArtifactPolicy | None = None,
    model_contract_path: str | os.PathLike | None = None,
) -> list[str]:
    if not isinstance(meta, dict):
        return ["meta.json must contain one JSON object"]
    errors: list[str] = []
    current_schema, schema_errors = _schema_contract(meta, "meta")
    errors.extend(schema_errors)
    meta_schema = meta.get("schema_version") if current_schema else None
    if meta.get("cell_id") != cell.cell_id:
        errors.append("meta cell_id does not match manifest cell")
    if meta.get("config_hash") != cell.config_hash():
        errors.append("meta config_hash does not match manifest cell")
    if meta.get("config") != cell.to_dict():
        errors.append("meta config does not match manifest cell")
    if meta.get("n_questions") != len(expected_qids):
        errors.append("meta n_questions does not match loader-derived QID count")
    if current_schema:
        if (
            benchmark_contract_sha256 is not None
            and meta.get("benchmark_contract_sha256")
            != benchmark_contract_sha256
        ):
            errors.append(
                "meta benchmark_contract_sha256 does not match the ordered "
                "benchmark contract"
            )
        completed_count = sum(
            record.get("termination_status", TERMINATION_COMPLETED)
            == TERMINATION_COMPLETED
            for record in records
        )
        censored_count = sum(
            record.get("termination_status") == TERMINATION_LENGTH_CENSORED
            for record in records
        )
        protocol_censored_count = sum(
            record.get("termination_status") == TERMINATION_PROTOCOL_CENSORED
            for record in records
        )
        if meta.get("completed_question_count") != completed_count:
            errors.append(
                "meta completed_question_count does not match canonical results"
            )
        if meta.get("length_censored_question_count") != censored_count:
            errors.append(
                "meta length_censored_question_count does not match canonical results"
            )
        if meta_schema == ARTIFACT_SCHEMA_VERSION:
            if (
                meta.get("protocol_censored_question_count")
                != protocol_censored_count
            ):
                errors.append(
                    "meta protocol_censored_question_count does not match canonical results"
                )
            expected_schema_counts: dict[str, int] = {}
            for record in records:
                schema_key = str(record.get("schema_version", "legacy"))
                expected_schema_counts[schema_key] = (
                    expected_schema_counts.get(schema_key, 0) + 1
                )
            expected_schema_counts = {
                key: expected_schema_counts[key]
                for key in sorted(expected_schema_counts)
            }
            if meta.get("artifact_schema_counts") != expected_schema_counts:
                errors.append(
                    "meta artifact_schema_counts does not match canonical results"
                )
        elif protocol_censored_count:
            errors.append("schema-4 meta cannot certify protocol-censored results")
        if completed_count + censored_count + protocol_censored_count != len(records):
            errors.append("meta termination counts do not cover canonical results")
    model = get_model(cell.model_size)
    if meta.get("model_hf_id") != model.hf_id:
        errors.append("meta model_hf_id does not match manifest model")
    if meta.get("served_model_name") != cell.model_size:
        errors.append("meta served_model_name does not match manifest model")
    if not _is_integer(meta.get("prompt_token_count")):
        errors.append("meta prompt_token_count must be a non-negative integer")
    errors.extend(_prompt_quality_errors(meta.get("prompt_quality")))
    git_commit = meta.get("git_commit")
    if current_schema:
        if not isinstance(git_commit, str) or not git_commit:
            errors.append("current meta git_commit must be a non-empty code version")
    elif git_commit is not None and (not isinstance(git_commit, str) or not git_commit):
        errors.append("legacy meta git_commit must be a non-empty string or null")
    started = meta.get("started_at")
    finished = meta.get("finished_at")
    if not _valid_timestamp(started):
        errors.append("meta started_at must be finite and positive")
    if not _valid_timestamp(finished):
        errors.append("meta finished_at must be finite and positive")
    if _valid_timestamp(started) and _valid_timestamp(finished) and float(finished) < float(started):
        errors.append("meta finished_at precedes started_at")
    observed_mean = meta.get("mean_reasoning_tokens")
    if current_schema:
        reasoning_fields = (
            "mean_reasoning_tokens",
            "mean_reasoning_tokens_exact",
            "exact_reasoning_question_count",
            "nonexact_reasoning_question_count",
            "mean_reasoning_word_count_legacy",
        )
        missing_reasoning_fields = [
            field for field in reasoning_fields if field not in meta
        ]
        if missing_reasoning_fields:
            errors.append(
                "current meta is missing reasoning summary fields: "
                + ", ".join(missing_reasoning_fields)
            )
        try:
            expected_reasoning = reasoning_token_summary(records)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"cannot summarize reasoning provenance: {exc}")
        else:
            for field, expected_value in (
                ("mean_reasoning_tokens", expected_reasoning.mean_reasoning_tokens),
                (
                    "mean_reasoning_tokens_exact",
                    expected_reasoning.mean_reasoning_tokens_exact,
                ),
            ):
                observed_value = meta.get(field)
                if expected_value is None:
                    if observed_value is not None:
                        errors.append(f"meta {field} must be null for canonical results")
                elif not _is_finite_number(observed_value):
                    errors.append(f"meta {field} must be finite for canonical results")
                elif not math.isclose(
                    float(observed_value),
                    expected_value,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    errors.append(f"meta {field} does not match canonical results")
            for field, expected_value in (
                (
                    "exact_reasoning_question_count",
                    expected_reasoning.exact_question_count,
                ),
                (
                    "nonexact_reasoning_question_count",
                    expected_reasoning.nonexact_question_count,
                ),
            ):
                if meta.get(field) != expected_value:
                    errors.append(f"meta {field} does not match canonical results")
            observed_word_mean = meta.get("mean_reasoning_word_count_legacy")
            if not _is_finite_number(observed_word_mean):
                errors.append(
                    "meta mean_reasoning_word_count_legacy must be finite"
                )
            elif not math.isclose(
                float(observed_word_mean),
                expected_reasoning.mean_reasoning_word_count_legacy,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                errors.append(
                    "meta mean_reasoning_word_count_legacy does not match canonical results"
                )
    elif not _is_finite_number(observed_mean):
        errors.append("legacy meta mean_reasoning_tokens must be finite")
    elif len(records) == len(expected_qids):
        expected_mean = _historical_mean_reasoning_word_count(records)
        if not math.isclose(
            float(observed_mean), expected_mean, rel_tol=1e-12, abs_tol=1e-9
        ):
            errors.append(
                "legacy meta mean_reasoning_tokens does not match canonical results"
            )

    # Serving fields were not present in the original metadata schema.  The two schemas
    # are explicit: legacy metadata has no schema/provenance fields, while current requires a
    # complete summary derived from the canonical rows.
    serving_fields = (
        "serving_profile",
        "effective_context_limit",
        "tensor_parallel_size",
        "serving_profile_counts",
        "serving_profile_inferred_counts",
    )
    protocol_fields = (
        "peer_context_protocol_version",
        "peer_context_protocol_hash",
        "peer_cot_char_limit",
        "peer_rendered_block_token_limit",
        "thinking_budget_protocol_version",
        "thinking_budget_protocol_hash",
        "generation_censor_protocol_version",
        "generation_censor_protocol_hash",
    )
    if not current_schema:
        current_reasoning_fields = (
            "mean_reasoning_tokens_exact",
            "exact_reasoning_question_count",
            "nonexact_reasoning_question_count",
            "mean_reasoning_word_count_legacy",
            "completed_question_count",
            "length_censored_question_count",
            "protocol_censored_question_count",
            "artifact_schema_counts",
        )
        present = [
            field
            for field in serving_fields + protocol_fields + current_reasoning_fields
            if field in meta
        ]
        if "benchmark_contract_sha256" in meta:
            present.append("benchmark_contract_sha256")
        if present:
            errors.append(
                "legacy meta must not contain current provenance fields without schema_version: "
                + ", ".join(present)
            )
        if any("schema_version" in record for record in records):
            errors.append("legacy meta cannot certify versioned question results")

    try:
        expected_provenance = summarize_serving_provenance(cell, records)
    except CorruptArtifactError as exc:
        errors.append(f"cannot summarize serving provenance: {exc}")
    else:
        if current_schema:
            missing_fields = [field for field in serving_fields if field not in meta]
            if missing_fields:
                errors.append(
                    "current meta is missing serving summary fields: "
                    + ", ".join(missing_fields)
                )
            observed_profile = meta.get("serving_profile")
            if observed_profile is not None and (
                not isinstance(observed_profile, str) or not observed_profile
            ):
                errors.append("meta serving_profile must be a non-empty string or null")
            elif observed_profile != expected_provenance["serving_profile"]:
                errors.append("meta serving_profile does not match canonical results")

            for field in ("effective_context_limit", "tensor_parallel_size"):
                observed_value = meta.get(field)
                if observed_value is not None and not _is_integer(observed_value, minimum=1):
                    errors.append(f"meta {field} must be a positive integer or null")
                elif observed_value != expected_provenance[field]:
                    errors.append(f"meta {field} does not match canonical results")

            for field, allow_empty in (
                ("serving_profile_counts", False),
                ("serving_profile_inferred_counts", True),
            ):
                observed_counts = meta.get(field)
                if not _valid_profile_counts(observed_counts, allow_empty=allow_empty):
                    errors.append(f"meta {field} must map profile names to positive counts")
                elif observed_counts != expected_provenance[field]:
                    errors.append(f"meta {field} does not match canonical results")

    if current_schema:
        expected_protocol = {
            "peer_context_protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
            "peer_context_protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
            "peer_cot_char_limit": PEER_COT_CHAR_LIMIT,
            "peer_rendered_block_token_limit": PEER_RENDERED_BLOCK_TOKEN_LIMIT,
            "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
            "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
        }
        if meta_schema == ARTIFACT_SCHEMA_VERSION:
            expected_protocol.update(
                {
                    "generation_censor_protocol_version": (
                        GENERATION_CENSOR_PROTOCOL_VERSION
                    ),
                    "generation_censor_protocol_hash": (
                        GENERATION_CENSOR_PROTOCOL_HASH
                    ),
                }
            )
        else:
            unexpected = [
                field
                for field in (
                    "generation_censor_protocol_version",
                    "generation_censor_protocol_hash",
                    "protocol_censored_question_count",
                    "artifact_schema_counts",
                )
                if field in meta
            ]
            if unexpected:
                errors.append(
                    "schema-4 meta contains schema-5 fields: "
                    + ", ".join(unexpected)
                )
        for field, expected_value in expected_protocol.items():
            if meta.get(field) != expected_value:
                errors.append(f"current meta {field} does not match registered protocol")
    if artifact_policy is not None:
        errors.extend(
            _artifact_policy_payload_errors(
                cell,
                artifact_policy,
                records,
                meta,
                model_contract_path=model_contract_path,
            )
        )
    return errors


def validate_completion_payload(
    cell: ExperimentCell,
    expected_qids: Iterable[str],
    records: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    expected_questions: Iterable[Question] | None = None,
    cell_directory: str | os.PathLike | None = None,
    verified_benchmark_contracts: FrozenBenchmarkContracts | None = None,
    verified_manifest: ManifestSnapshot | None = None,
    model_contract_path: str | os.PathLike | None = None,
    _structural_only: bool = False,
) -> tuple[str, ...]:
    """Validate in-memory artifacts before publishing the completion metadata."""
    question_contract = _normalise_question_contracts(
        cell,
        expected_qids,
        expected_questions,
        cell_directory=cell_directory,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=verified_manifest,
        structural_only=_structural_only,
    )
    qids = question_contract.ordered_qids
    expected_set = set(qids)
    errors: list[str] = []
    artifact_policy, policy_read_error = _policy_for_cell_directory(cell_directory)
    if policy_read_error is not None:
        errors.append(policy_read_error)
    if artifact_policy is not None:
        if verified_manifest is None:
            errors.append(
                "schema-5 artifact policy requires a verified frozen manifest"
            )
        elif verified_manifest.sha256 != artifact_policy.accepted_manifest_sha256:
            errors.append("schema-5 artifact policy manifest pin mismatch")
        if verified_benchmark_contracts is None:
            errors.append(
                "schema-5 artifact policy requires verified benchmark contracts"
            )
        elif (
            verified_benchmark_contracts.sidecar_sha256
            != artifact_policy.accepted_benchmark_contracts_sha256
        ):
            errors.append("schema-5 artifact policy benchmark pin mismatch")
    seen: set[str] = set()
    for index, record in enumerate(records):
        row_errors, qid, unexpected = _record_errors(
            record,
            cell,
            expected_set,
            question_contract.questions_by_qid,
            question_contract.question_sha256_by_qid,
            question_contract.benchmark_contract_sha256,
        )
        if unexpected:
            errors.append(f"record {index}: unexpected qid {qid!r}")
        errors.extend(f"record {index}: {error}" for error in row_errors)
        if qid is not None and not unexpected and not row_errors:
            if qid in seen:
                errors.append(f"record {index}: duplicate qid {qid!r}")
            seen.add(qid)
    missing = [qid for qid in qids if qid not in seen]
    if missing:
        errors.append(f"missing {len(missing)} expected qids")
    if len(records) != len(qids):
        errors.append(f"record count {len(records)} != expected count {len(qids)}")
    errors.extend(
        _meta_errors(
            cell,
            meta,
            qids,
            records,
            question_contract.benchmark_contract_sha256,
            artifact_policy,
            model_contract_path,
        )
    )
    return tuple(errors)


def _read_meta(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJSONKey,
        _NonFiniteJSONNumber,
    ) as exc:
        return None, f"invalid meta.json: {exc}"
    if not isinstance(value, dict):
        return None, "meta.json must contain one JSON object"
    return value, None


def read_failure(cell_directory: str | os.PathLike) -> FailureRecord | None:
    path = Path(cell_directory) / FAILURE_FILENAME
    if not path.exists():
        return None
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJSONKey,
        _NonFiniteJSONNumber,
    ) as exc:
        raise CorruptArtifactError(f"invalid {FAILURE_FILENAME}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorruptArtifactError(f"{FAILURE_FILENAME} must contain one JSON object")
    return FailureRecord.from_dict(value)


def infer_failure_class(error: BaseException) -> FailureClass:
    if isinstance(error, ContextCapacityError) or type(error).__name__ == "ContextCapacityError":
        return FailureClass.CONTEXT_CAPACITY
    # Protocol v4 submits one draw with the entire exact remaining model context.  A
    # leaked length-censor exception therefore describes profile capacity, not a broken
    # local configuration and not a connection failure.  The runner normally converts
    # it into an explicit censored QID row; this classification is the fail-closed path.
    if type(error).__name__ == "GenerationTruncationError":
        return FailureClass.CONTEXT_CAPACITY
    # A structured protocol censor should be journaled and published, never recorded as
    # failure.  If it leaks past that boundary, fail permanently rather than resampling.
    if isinstance(error, GenerationProtocolCensorError) or type(error).__name__ == (
        "GenerationProtocolCensorError"
    ):
        return FailureClass.CONFIGURATION
    # Remaining response-protocol failures mean the exact observation envelope itself
    # could not be trusted (token IDs, usage, prompt identity, or parser agreement).
    # Retrying after a response was sampled can condition on validation success, so these
    # are blocked until code/operator repair rather than entering ordinary backoff.
    if (
        isinstance(error, ServerResponseProtocolError)
        or type(error).__name__ == "ServerResponseProtocolError"
    ):
        return FailureClass.CONFIGURATION
    if isinstance(
        error, (ExperimentConfigurationError, TokenizerInitializationError)
    ) or type(error).__name__ in {
        "ConfigurationError",
        "ExperimentConfigurationError",
        "ThinkingBudgetProtocolError",
        "TokenizerInitializationError",
        "BadRequestError",
    }:
        return FailureClass.CONFIGURATION
    if type(error).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError"}:
        return FailureClass.CONNECTION
    return FailureClass.RUNTIME


def record_failure(
    cell_directory: str | os.PathLike,
    cell: ExperimentCell,
    error: BaseException,
    *,
    classification: FailureClass | str | None = None,
    now: float | None = None,
    serving_profile: str | None = None,
    code_version: str | None = None,
    server_pool_generation: str | None = None,
    base_backoff_s: float = 60.0,
    max_backoff_s: float = 3600.0,
    max_attempts: int = 5,
) -> FailureRecord:
    """Atomically record a classified failure and compute retry eligibility.

    Context-capacity failures stay permanent until the serving profile changes;
    configuration failures stay permanent until the code version changes.  Connection
    and other runtime failures back off exponentially and become dormant after
    ``max_attempts`` rather than consuming scheduler slots forever.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_backoff_s < 0 or max_backoff_s < 0:
        raise ValueError("backoff durations must be non-negative")
    cdir = Path(cell_directory)
    cdir.mkdir(parents=True, exist_ok=True)
    failure_class = (
        infer_failure_class(error)
        if classification is None
        else FailureClass(classification)
    )
    timestamp = time.time() if now is None else float(now)
    if not _valid_timestamp(timestamp):
        raise ValueError("failure timestamp must be finite and positive")

    previous: FailureRecord | None
    try:
        previous = read_failure(cdir)
    except CorruptArtifactError:
        previous = None
    same_environment = bool(
        previous
        and (
            failure_class
            not in {FailureClass.CONNECTION, FailureClass.RUNTIME}
            or previous.server_pool_generation == server_pool_generation
        )
    )
    same_failure = bool(
        previous
        and previous.cell_id == cell.cell_id
        and previous.config_hash == cell.config_hash()
        and previous.classification == failure_class.value
        and same_environment
    )
    attempts = previous.attempts + 1 if same_failure and previous is not None else 1
    first_failed_at = (
        previous.first_failed_at if same_failure and previous is not None else timestamp
    )

    intrinsically_permanent = failure_class in {
        FailureClass.CONTEXT_CAPACITY,
        FailureClass.CONFIGURATION,
    }
    exhausted = attempts >= max_attempts
    if intrinsically_permanent:
        disposition = FailureDisposition.PERMANENT
        next_eligible_at = None
    elif exhausted:
        # A transient outage is not a scientific/configuration impossibility.  Keep it
        # retryable but dormant so it consumes no slots until an operator resets it or
        # the recorded runtime environment changes.
        disposition = FailureDisposition.RETRYABLE
        next_eligible_at = None
    else:
        disposition = FailureDisposition.RETRYABLE
        delay = min(max_backoff_s, base_backoff_s * (2 ** (attempts - 1)))
        next_eligible_at = timestamp + delay

    record = FailureRecord(
        schema_version=FAILURE_SCHEMA_VERSION,
        cell_id=cell.cell_id,
        config_hash=cell.config_hash(),
        classification=failure_class.value,
        disposition=disposition.value,
        attempts=attempts,
        first_failed_at=first_failed_at,
        last_failed_at=timestamp,
        last_error={"type": type(error).__name__, "message": str(error)[:4000]},
        next_eligible_at=next_eligible_at,
        serving_profile=serving_profile,
        code_version=code_version,
        server_pool_generation=server_pool_generation,
        dormant=exhausted and not intrinsically_permanent,
    )
    io.write_json(cdir / FAILURE_FILENAME, record.to_dict())
    return record


def clear_failure(cell_directory: str | os.PathLike) -> None:
    io.remove_file(Path(cell_directory) / FAILURE_FILENAME)


def _quarantine_artifact(
    cell_directory: str | os.PathLike, filename: str, label: str
) -> Path | None:
    cdir = Path(cell_directory)
    source = cdir / filename
    if not source.exists():
        return None
    quarantine = cdir / "quarantine"
    # time_ns + pid is collision-resistant across concurrent repair tools.  The cell lock
    # is still required by callers; the suffix is defensive and aids provenance.
    destination = quarantine / f"{label}.{time.time_ns()}.{os.getpid()}.json"
    counter = 0
    while destination.exists():
        counter += 1
        destination = (
            quarantine / f"{label}.{time.time_ns()}.{os.getpid()}.{counter}.json"
        )
    io.move_file(source, destination)
    return destination


def quarantine_metadata(cell_directory: str | os.PathLike) -> Path | None:
    """Move invalid completion metadata aside without destroying audit evidence."""
    return _quarantine_artifact(cell_directory, META_FILENAME, "meta")


def quarantine_failure(cell_directory: str | os.PathLike) -> Path | None:
    """Move an invalid failure ledger aside without destroying audit evidence."""
    return _quarantine_artifact(cell_directory, FAILURE_FILENAME, "failure")


def _failure_state(
    failure: FailureRecord,
    *,
    now: float,
    serving_profile: str | None,
    code_version: str | None,
    server_pool_generation: str | None,
) -> tuple[CompletionState, bool, float | None]:
    profile_changed = bool(
        failure.classification == FailureClass.CONTEXT_CAPACITY.value
        and failure.serving_profile is not None
        and serving_profile is not None
        and failure.serving_profile != serving_profile
    )
    code_changed = bool(
        failure.classification
        in {
            FailureClass.CONFIGURATION.value,
            FailureClass.CONTEXT_CAPACITY.value,
        }
        and failure.code_version is not None
        and code_version is not None
        and failure.code_version != code_version
    )
    pool_changed = bool(
        server_pool_generation is not None
        and failure.server_pool_generation != server_pool_generation
    )
    if profile_changed or code_changed:
        return CompletionState.RETRYABLE, True, None
    if failure.disposition == FailureDisposition.PERMANENT.value:
        return CompletionState.PERMANENT, False, None
    if failure.dormant:
        environment_changed = bool(
            pool_changed
            or (
                failure.serving_profile is not None
                and serving_profile is not None
                and failure.serving_profile != serving_profile
            )
            or (
                failure.code_version is not None
                and code_version is not None
                and failure.code_version != code_version
            )
        )
        return CompletionState.RETRYABLE, environment_changed, None
    eligible = failure.next_eligible_at is None or now >= failure.next_eligible_at
    return CompletionState.RETRYABLE, eligible, failure.next_eligible_at


@contextmanager
def cell_lock(
    cell_directory: str | os.PathLike, *, blocking: bool = False
) -> Iterator[None]:
    """Acquire a per-cell advisory lock, nonblocking by default."""
    cdir = Path(cell_directory)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(fd, operation)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise CellLockUnavailable(f"cell is active: {cdir.name}") from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def is_cell_active(cell_directory: str | os.PathLike) -> bool:
    """Return whether another process currently holds the cell lock."""
    path = Path(cell_directory) / LOCK_FILENAME
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def get_completion_status(
    cell: ExperimentCell,
    cell_directory: str | os.PathLike,
    *,
    expected_qids: Iterable[str] | None = None,
    expected_questions: Iterable[Question] | None = None,
    verified_benchmark_contracts: FrozenBenchmarkContracts | None = None,
    verified_manifest: ManifestSnapshot | None = None,
    _structural_only: bool = False,
    check_active: bool = True,
    now: float | None = None,
    serving_profile: str | None = None,
    code_version: str | None = None,
    server_pool_generation: str | None = None,
    model_contract_path: str | os.PathLike | None = None,
) -> CompletionStatus:
    """Return the semantic state of one manifest cell without mutating its artifacts."""
    cdir = Path(cell_directory)
    question_contract = _normalise_question_contracts(
        cell,
        expected_qids,
        expected_questions,
        cell_directory=cdir,
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=verified_manifest,
        structural_only=_structural_only,
    )
    qids = question_contract.ordered_qids
    active = bool(check_active and cdir.exists() and is_cell_active(cdir))

    parsed = read_canonical_results(
        cell,
        cdir,
        expected_qids=qids,
        expected_questions=(
            None
            if question_contract.questions_by_qid is None
            else tuple(question_contract.questions_by_qid[qid] for qid in qids)
        ),
        verified_benchmark_contracts=verified_benchmark_contracts,
        verified_manifest=verified_manifest,
        _structural_only=_structural_only,
        _resolved_question_contract=question_contract,
    )
    completed_question_count = sum(
        record.get("termination_status", TERMINATION_COMPLETED)
        == TERMINATION_COMPLETED
        for record in parsed.records
    )
    length_censored_question_count = sum(
        record.get("termination_status") == TERMINATION_LENGTH_CENSORED
        for record in parsed.records
    )
    protocol_censored_question_count = sum(
        record.get("termination_status") == TERMINATION_PROTOCOL_CENSORED
        for record in parsed.records
    )
    valid_qids = set(parsed.qids)
    missing = tuple(qid for qid in qids if qid not in valid_qids)
    errors = list(parsed.validation_errors)
    artifact_policy, policy_read_error = _policy_for_cell_directory(cdir)
    if policy_read_error is not None:
        errors.append(policy_read_error)
    if artifact_policy is not None:
        if verified_manifest is None:
            errors.append("schema-5 artifact policy requires a verified frozen manifest")
        elif verified_manifest.sha256 != artifact_policy.accepted_manifest_sha256:
            errors.append("schema-5 artifact policy manifest pin mismatch")
        if verified_benchmark_contracts is None:
            errors.append(
                "schema-5 artifact policy requires verified benchmark contracts"
            )
        elif (
            verified_benchmark_contracts.sidecar_sha256
            != artifact_policy.accepted_benchmark_contracts_sha256
        ):
            errors.append("schema-5 artifact policy benchmark pin mismatch")

    # Active is an admission state, not a claim that no work has been retained yet.
    # Report the last fully valid JSONL rows so monitors expose real QID progress.  Do
    # not classify transient append-tail diagnostics as corruption while the owner may
    # still be writing; the next unlocked poll performs the full contract decision.
    if active:
        return CompletionStatus(
            status=CompletionState.ACTIVE,
            cell_id=cell.cell_id,
            expected_count=len(qids),
            valid_count=len(parsed.records),
            completed_question_count=completed_question_count,
            length_censored_question_count=length_censored_question_count,
            protocol_censored_question_count=protocol_censored_question_count,
            missing_qids=missing,
        )

    meta, meta_read_error = _read_meta(cdir / META_FILENAME)
    if meta_read_error:
        errors.append(meta_read_error)
    if meta is not None:
        errors.extend(
            _meta_errors(
                cell,
                meta,
                qids,
                parsed.records,
                question_contract.benchmark_contract_sha256,
                artifact_policy,
                model_contract_path,
            )
        )
        if missing:
            errors.append("meta.json exists but expected QID coverage is incomplete")
    elif artifact_policy is not None and parsed.records:
        errors.extend(
            _artifact_policy_payload_errors(
                cell,
                artifact_policy,
                parsed.records,
                None,
                model_contract_path=model_contract_path,
            )
        )

    failure: FailureRecord | None = None
    try:
        failure = read_failure(cdir)
    except CorruptArtifactError as exc:
        errors.append(str(exc))
    if failure is not None:
        if failure.cell_id != cell.cell_id:
            errors.append("failure cell_id does not match manifest cell")
        if failure.config_hash != cell.config_hash():
            errors.append("failure config_hash does not match manifest cell")

    if parsed.has_corruption or errors:
        return CompletionStatus(
            status=CompletionState.CORRUPT,
            cell_id=cell.cell_id,
            expected_count=len(qids),
            valid_count=len(parsed.records),
            completed_question_count=completed_question_count,
            length_censored_question_count=length_censored_question_count,
            protocol_censored_question_count=protocol_censored_question_count,
            missing_qids=missing,
            duplicate_qids=parsed.duplicate_qids,
            unexpected_qids=parsed.unexpected_qids,
            malformed_lines=parsed.malformed_lines,
            invalid_rows=parsed.invalid_rows,
            errors=tuple(errors),
            failure=failure,
        )

    if not missing and meta is not None:
        return CompletionStatus(
            status=CompletionState.COMPLETE,
            cell_id=cell.cell_id,
            expected_count=len(qids),
            valid_count=len(parsed.records),
            completed_question_count=completed_question_count,
            length_censored_question_count=length_censored_question_count,
            protocol_censored_question_count=protocol_censored_question_count,
            failure=failure,
        )

    if failure is not None:
        current = time.time() if now is None else float(now)
        state, eligible, next_eligible = _failure_state(
            failure,
            now=current,
            serving_profile=serving_profile,
            code_version=code_version,
            server_pool_generation=server_pool_generation,
        )
        return CompletionStatus(
            status=state,
            cell_id=cell.cell_id,
            expected_count=len(qids),
            valid_count=len(parsed.records),
            completed_question_count=completed_question_count,
            length_censored_question_count=length_censored_question_count,
            protocol_censored_question_count=protocol_censored_question_count,
            missing_qids=missing,
            next_eligible_at=next_eligible,
            eligible_for_retry=eligible,
            failure=failure,
        )

    has_any_artifact = any(
        (cdir / name).exists() for name in (RESULTS_FILENAME, META_FILENAME, FAILURE_FILENAME)
    )
    return CompletionStatus(
        status=CompletionState.PARTIAL if has_any_artifact else CompletionState.MISSING,
        cell_id=cell.cell_id,
        expected_count=len(qids),
        valid_count=len(parsed.records),
        completed_question_count=completed_question_count,
        length_censored_question_count=length_censored_question_count,
        protocol_censored_question_count=protocol_censored_question_count,
        missing_qids=missing,
        eligible_for_retry=True,
    )


# Concise aliases for integrations that prefer inspect/validate terminology.
inspect_completion = get_completion_status
validate_completion = get_completion_status
