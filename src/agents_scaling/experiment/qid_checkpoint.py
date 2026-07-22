"""Durable, fail-closed checkpoints for stochastic work within one benchmark QID.

The cell result row is the publication boundary, but a topology can contain many
independently sampled coordinates.  A worker or endpoint can fail after some siblings
have returned.  Reissuing those siblings would turn an infrastructure retry into
replacement sampling.  This module journals every observed coordinate immediately and
atomically, then lets a restarted topology replay the exact :class:`AgentOutput` or
length-censor instead of contacting a server again.

One :class:`QIDCheckpoint` is used by all topology threads for a question.  The cell's
advisory lock serializes processes; the condition variable below serializes duplicate
coordinates within a process while allowing distinct siblings to run concurrently.
Checkpoints are deliberately fail-closed: malformed data, an integrity failure, or any
identity mismatch is never repaired or ignored because doing so could resample an
already observed scientific outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from agents_scaling.agents.base_agent import (
    Agent,
    AgentOutput,
    SelfConsistencySample,
    requested_generation_tokens,
)
from agents_scaling.agents.message_builder import (
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
    PeerContextRender,
)
from agents_scaling.agents.topologies.base import TopologyResult
from agents_scaling.benchmarks.contracts import (
    canonical_question_payload,
    canonical_question_sha256,
)
from agents_scaling.benchmarks.schema import Question
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.result_schema import (
    ARTIFACT_SCHEMA_VERSION,
    SELF_CONSISTENCY_PROTOCOL_HASH,
    SELF_CONSISTENCY_PROTOCOL_VERSION,
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    GENERATION_CENSOR_PROTOCOL_HASH,
    GENERATION_CENSOR_PROTOCOL_VERSION,
    QWEN_THINK_END_TOKEN_ID,
    QWEN_THINK_START_TOKEN_ID,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
    GenerationProtocolCensorError,
    GenerationTruncationError,
    generation_protocol_violation_codes,
)
from agents_scaling.serving.context import CONTEXT_RESERVE_TOKENS
from agents_scaling.serving.profiles import ServingProfile

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_DIRECTORY = ".qid_checkpoints"

_T = TypeVar("_T")
_AGENT_OUTPUT_FIELDS = frozenset(field.name for field in fields(AgentOutput))
_TOPOLOGY_RESULT_FIELDS = frozenset(field.name for field in fields(TopologyResult))
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "coordinates",
        "topology_terminal",
        "created_at",
        "updated_at",
        "migration_history",
        "integrity_sha256",
    }
)
_MIGRATION_HISTORY_FIELDS = frozenset(
    {
        "migration_schema_version",
        "migration_type",
        "migrated_at",
        "source_checkpoint_schema_version",
        "target_checkpoint_schema_version",
        "source_file_sha256",
        "source_integrity_sha256",
        "source_identity_sha256",
        "target_identity_sha256",
        "migrator_code_version",
    }
)
_CHECKPOINT_SCHEMA_MIGRATION_TYPE = "qid_checkpoint_schema_1_to_2"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_AGENT_ID_RE = re.compile(r"agent(\d+)")
_PEER_CONTEXT_IDENTITY_FIELDS = frozenset({"sha256", "utf8_bytes"})
_TOPOLOGY_REQUEST_FIELDS = frozenset(
    {
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
)
_SELF_CONSISTENCY_REQUEST_FIELDS = frozenset(
    {
        "generation_role",
        "qid",
        "agent_id",
        "round",
        "seed",
        "sample_index",
        "base_seed",
        "peer_context",
        "max_tokens",
        "elicit_cot",
        "extra_kwargs",
    }
)
_CENSOR_BASE_FIELDS = frozenset(
    {
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
)
_PROTOCOL_CENSOR_FIELDS = frozenset(
    {
        "protocol_violation_codes",
        "actual_terminal_token_id",
        "generation_censor_protocol_version",
        "generation_censor_protocol_hash",
    }
)
_CENSOR_ANNOTATION_FIELDS = frozenset({"topology", "benchmark"})


class CheckpointError(RuntimeError):
    """Base class for a checkpoint failure that must not lead to resampling."""


class CheckpointCorruptionError(CheckpointError):
    """The durable checkpoint is malformed or failed its integrity contract."""


class CheckpointIdentityError(CheckpointError):
    """A checkpoint belongs to a different scientific or executable identity."""


class CheckpointPersistenceError(CheckpointError):
    """An observed outcome could not be durably checkpointed."""


class CoordinateAdmissionClosed(RuntimeError):
    """A graceful worker drain forbids starting another model request.

    This is deliberately *not* a :class:`CheckpointError`: no artifact is corrupt and
    no stochastic observation has failed.  The runner catches it outside ordinary
    failure accounting, releases the cell lock, and lets a later worker resume from the
    already-fsynced coordinate journal.
    """


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONNumber(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise _NonFiniteJSONNumber(f"non-finite JSON number {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _integrity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "integrity_sha256"}


def _finite_nonnegative(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointCorruptionError(f"{field_name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise CheckpointCorruptionError(f"{field_name} must be finite and non-negative")
    return converted


def _is_integer(value: Any, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _token_id_sequence(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(_is_integer(token_id) for token_id in value)
    )


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _profile_payload(profile: ServingProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "model_size": profile.model_size,
        "hf_id": profile.hf_id,
        "tp_size": profile.tp_size,
        "max_model_len": profile.max_model_len,
        "served_model_name": profile.served_model_name,
    }


def _coordinate_key_from_request(request: Mapping[str, Any]) -> str:
    if request.get("generation_role") == "self_consistency":
        return f"self_consistency:{request.get('sample_index')}"
    return f"topology:{request.get('agent_id')}:{request.get('round')}"


def checkpoint_path(cell_directory: str | Path, qid: str) -> Path:
    """Return the traversal-safe checkpoint path for ``qid`` within a cell."""

    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return Path(cell_directory) / CHECKPOINT_DIRECTORY / f"{digest}.json"


def remove_qid_checkpoint(cell_directory: str | Path, qid: str) -> None:
    """Durably remove a checkpoint after its canonical QID row already exists."""

    io.remove_file(checkpoint_path(cell_directory, qid))


def _agent_output_payload(output: AgentOutput) -> dict[str, Any]:
    if not isinstance(output, AgentOutput):
        raise CheckpointCorruptionError("coordinate did not produce AgentOutput")
    try:
        payload = asdict(output)
        # Exercise the strict JSON contract before the outcome is admitted in memory.
        _canonical_bytes(payload)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CheckpointCorruptionError(
            f"observed AgentOutput is not finite canonical JSON: {exc}"
        ) from exc
    return payload


def _agent_output_from_payload(
    payload: Any,
    *,
    expected_agent_id: str | None = None,
    expected_round: int | None = None,
) -> AgentOutput:
    if not isinstance(payload, dict) or set(payload) != _AGENT_OUTPUT_FIELDS:
        raise CheckpointCorruptionError("checkpoint AgentOutput has the wrong fields")
    try:
        output = AgentOutput(**copy.deepcopy(payload))
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"invalid checkpoint AgentOutput: {exc}"
        ) from exc
    if asdict(output) != payload:
        raise CheckpointCorruptionError(
            "checkpoint AgentOutput failed exact round trip"
        )
    if expected_agent_id is not None and output.agent_id != expected_agent_id:
        raise CheckpointCorruptionError(
            "checkpoint AgentOutput agent_id does not match its coordinate"
        )
    if expected_round is not None and output.round != expected_round:
        raise CheckpointCorruptionError(
            "checkpoint AgentOutput round does not match its coordinate"
        )
    return output


def _censor_from_payload(
    payload: Any,
    *,
    qid: str,
    agent_id: str,
    round_idx: int,
    generation_role: str,
    sample_index: int | None,
    seed: int | None,
    enable_thinking: bool,
    thinking_budget: int | None,
    output_capacity_floor_tokens: int,
    serving_profile: str,
    effective_context_limit: int,
) -> GenerationTruncationError:
    if not isinstance(payload, dict):
        raise CheckpointCorruptionError("censored generation must be an object")
    reason = payload.get("reason")
    expected_fields = set(_CENSOR_BASE_FIELDS)
    if reason == "protocol":
        expected_fields.update(_PROTOCOL_CENSOR_FIELDS)
    elif reason != "length":
        raise CheckpointCorruptionError(
            f"unknown censored generation reason {reason!r}"
        )
    observed_fields = set(payload) - _CENSOR_ANNOTATION_FIELDS
    if observed_fields != expected_fields:
        missing = sorted(expected_fields - observed_fields)
        unexpected = sorted(observed_fields - expected_fields)
        raise CheckpointCorruptionError(
            "censored generation has the wrong fields: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if payload["sampling_attempt_count"] != 1:
        raise CheckpointCorruptionError(
            "censored generation must record exactly one sampling attempt"
        )
    expected_coordinates = {
        "qid": qid,
        "agent_id": agent_id,
        "round": round_idx,
        "generation_role": generation_role,
        "sample_index": sample_index,
        "seed": seed,
    }
    for field, expected in expected_coordinates.items():
        if payload[field] != expected:
            raise CheckpointCorruptionError(
                f"censored generation {field} does not match its coordinate"
            )
    if payload["serving_profile"] != serving_profile:
        raise CheckpointCorruptionError(
            "censored generation serving profile does not match checkpoint identity"
        )
    if payload["effective_context_limit"] != effective_context_limit:
        raise CheckpointCorruptionError(
            "censored generation context limit does not match checkpoint identity"
        )
    if payload["context_reserve_tokens"] != CONTEXT_RESERVE_TOKENS:
        raise CheckpointCorruptionError(
            "censored generation context reserve is not registered"
        )
    if payload["output_capacity_floor_tokens"] != output_capacity_floor_tokens:
        raise CheckpointCorruptionError(
            "censored generation output floor does not match request treatment"
        )

    prompt_ids = payload["prompt_token_ids"]
    completion_ids = payload["completion_token_ids"]
    if not _token_id_sequence(prompt_ids, allow_empty=False):
        raise CheckpointCorruptionError(
            "censored generation prompt_token_ids must be non-empty token IDs"
        )
    if not _token_id_sequence(
        completion_ids, allow_empty=(reason == "protocol")
    ):
        raise CheckpointCorruptionError(
            "censored generation completion_token_ids violate censor semantics"
        )
    integer_fields = {
        "prompt_tokens": 1,
        "requested_output_tokens": 1,
        "output_capacity_floor_tokens": 1,
        "completion_tokens": 0 if reason == "protocol" else 1,
        "context_reserve_tokens": 0,
        "effective_context_limit": 1,
    }
    for field, minimum in integer_fields.items():
        if not _is_integer(payload[field], minimum=minimum):
            raise CheckpointCorruptionError(
                f"censored generation {field} must be an integer >= {minimum}"
            )
    if payload["prompt_tokens"] != len(prompt_ids):
        raise CheckpointCorruptionError(
            "censored generation prompt count does not match retained token IDs"
        )
    if payload["completion_tokens"] != len(completion_ids):
        raise CheckpointCorruptionError(
            "censored generation completion count does not match retained token IDs"
        )
    if payload["requested_output_tokens"] < output_capacity_floor_tokens:
        raise CheckpointCorruptionError(
            "censored generation requested output is below its treatment floor"
        )
    if (
        payload["prompt_tokens"]
        + payload["requested_output_tokens"]
        + payload["context_reserve_tokens"]
        != payload["effective_context_limit"]
    ):
        raise CheckpointCorruptionError(
            "censored generation does not use the full context envelope"
        )
    if reason == "length":
        if payload["finish_reason"] != "length":
            raise CheckpointCorruptionError(
                "length censor finish_reason must be length"
            )
        if payload["completion_tokens"] != payload["requested_output_tokens"]:
            raise CheckpointCorruptionError(
                "length censor must exhaust its requested output envelope"
            )
    elif payload["completion_tokens"] > payload["requested_output_tokens"]:
        raise CheckpointCorruptionError(
            "protocol censor exceeds its requested output envelope"
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
    for field, token_ids, delimiter in position_contracts:
        expected_positions = [
            index for index, token_id in enumerate(token_ids) if token_id == delimiter
        ]
        if payload[field] != expected_positions:
            raise CheckpointCorruptionError(
                f"censored generation {field} does not match retained token IDs"
            )
    if not isinstance(payload["decoded_completion"], str):
        raise CheckpointCorruptionError(
            "censored generation decoded_completion must be text"
        )
    for field in ("server_content", "server_reasoning"):
        if payload[field] is not None and not isinstance(payload[field], str):
            raise CheckpointCorruptionError(
                f"censored generation {field} must be text or null"
            )
    if not isinstance(payload["finish_reason"], str):
        raise CheckpointCorruptionError(
            "censored generation finish_reason must be text"
        )
    if (
        not isinstance(payload["endpoint_generation"], str)
        or not payload["endpoint_generation"]
    ):
        raise CheckpointCorruptionError(
            "censored generation endpoint_generation must be non-empty text"
        )
    if _finite_nonnegative(payload["created_at"], "censored_generation.created_at") == 0:
        raise CheckpointCorruptionError(
            "censored generation created_at must be positive"
        )

    if reason == "protocol":
        expected_codes = generation_protocol_violation_codes(
            finish_reason=payload["finish_reason"],
            completion_token_ids=completion_ids,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        codes = payload["protocol_violation_codes"]
        if not isinstance(codes, list) or tuple(codes) != expected_codes:
            raise CheckpointCorruptionError(
                "protocol censor violation codes do not match retained response"
            )
        terminal = completion_ids[-1] if completion_ids else None
        if payload["actual_terminal_token_id"] != terminal:
            raise CheckpointCorruptionError(
                "protocol censor terminal token does not match retained response"
            )
        if (
            payload["generation_censor_protocol_version"]
            != GENERATION_CENSOR_PROTOCOL_VERSION
            or payload["generation_censor_protocol_hash"]
            != GENERATION_CENSOR_PROTOCOL_HASH
        ):
            raise CheckpointCorruptionError(
                "protocol censor identity is not registered"
            )
    try:
        kwargs = dict(
            finish_reason=payload["finish_reason"],
            requested_output_tokens=payload["requested_output_tokens"],
            completion_tokens=payload["completion_tokens"],
            output_capacity_floor_tokens=payload["output_capacity_floor_tokens"],
            prompt_tokens=payload["prompt_tokens"],
            prompt_token_ids=payload["prompt_token_ids"],
            completion_token_ids=payload["completion_token_ids"],
            decoded_completion=payload["decoded_completion"],
            server_content=payload["server_content"],
            server_reasoning=payload["server_reasoning"],
            seed=payload["seed"],
            serving_profile=payload["serving_profile"],
            effective_context_limit=payload["effective_context_limit"],
            context_reserve_tokens=payload["context_reserve_tokens"],
            prompt_think_start_positions=payload["prompt_think_start_positions"],
            prompt_think_end_positions=payload["prompt_think_end_positions"],
            completion_think_start_positions=payload[
                "completion_think_start_positions"
            ],
            completion_think_end_positions=payload["completion_think_end_positions"],
            endpoint_generation=payload["endpoint_generation"],
            created_at=payload["created_at"],
        )
        if reason == "length":
            error: GenerationTruncationError = GenerationTruncationError(**kwargs)
        elif reason == "protocol":
            error = GenerationProtocolCensorError(
                protocol_violation_codes=payload["protocol_violation_codes"],
                **kwargs,
            )
        error.enrich(
            qid=qid,
            agent_id=agent_id,
            round_idx=round_idx,
            generation_role=generation_role,
            sample_index=sample_index,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"invalid censored generation in checkpoint: {exc}"
        ) from exc
    observed = error.to_censored_generation()
    unexpected = set(payload) - set(observed) - _CENSOR_ANNOTATION_FIELDS
    if unexpected or any(payload.get(key) != value for key, value in observed.items()):
        raise CheckpointCorruptionError(
            "censored generation failed exact provenance round trip"
        )
    return error


def _topology_result_payload(result: TopologyResult) -> dict[str, Any]:
    if not isinstance(result, TopologyResult):
        raise CheckpointCorruptionError("terminal outcome is not TopologyResult")
    try:
        payload = asdict(result)
        _canonical_bytes(payload)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CheckpointCorruptionError(
            f"observed TopologyResult is not finite canonical JSON: {exc}"
        ) from exc
    return payload


def _topology_result_from_payload(payload: Any) -> TopologyResult:
    if not isinstance(payload, dict) or set(payload) != _TOPOLOGY_RESULT_FIELDS:
        raise CheckpointCorruptionError("checkpoint TopologyResult has wrong fields")
    per_agent_payload = payload.get("per_agent")
    if not isinstance(per_agent_payload, list):
        raise CheckpointCorruptionError("TopologyResult.per_agent must be a list")
    outputs = [_agent_output_from_payload(item) for item in per_agent_payload]
    try:
        result = TopologyResult(
            final_answer=payload["final_answer"],
            per_agent=outputs,
            n_turns=payload["n_turns"],
            n_messages=payload["n_messages"],
            n_rounds=payload["n_rounds"],
            n_agents=payload["n_agents"],
            system_conf=copy.deepcopy(payload["system_conf"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"invalid checkpoint TopologyResult: {exc}"
        ) from exc
    if asdict(result) != payload:
        raise CheckpointCorruptionError(
            "checkpoint TopologyResult failed exact provenance round trip"
        )
    return result


class TopologyTerminal:
    """One decoded durable topology terminal."""

    def __init__(
        self,
        *,
        termination_status: str,
        wall_ms: float,
        topology_result: TopologyResult | None = None,
        censored_error: GenerationTruncationError | None = None,
    ) -> None:
        self.termination_status = termination_status
        self.wall_ms = wall_ms
        self.topology_result = topology_result
        self.censored_error = censored_error


class QIDCheckpoint:
    """Atomic checkpoint journal for a single frozen cell/question identity."""

    def __init__(
        self,
        cell_directory: str | Path,
        cell: ExperimentCell,
        question: Question,
        *,
        code_version: str | None,
        serving_profile: ServingProfile,
        benchmark_contract_sha256: str,
    ) -> None:
        self.path = checkpoint_path(cell_directory, question.qid)
        self._cell = cell
        if (
            not isinstance(benchmark_contract_sha256, str)
            or len(benchmark_contract_sha256) != 64
            or any(character not in "0123456789abcdef" for character in benchmark_contract_sha256)
        ):
            raise CheckpointIdentityError(
                "benchmark_contract_sha256 must be a lowercase SHA-256"
            )
        question_payload = canonical_question_payload(question)
        self.identity: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "cell_config": cell.to_dict(),
            "config_hash": cell.config_hash(),
            "benchmark_contract_sha256": benchmark_contract_sha256,
            "question": question_payload,
            "question_sha256": canonical_question_sha256(question),
            "code_version": code_version,
            "serving_profile": _profile_payload(serving_profile),
            "protocols": {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
                "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
                "generation_censor_protocol_version": (
                    GENERATION_CENSOR_PROTOCOL_VERSION
                ),
                "generation_censor_protocol_hash": GENERATION_CENSOR_PROTOCOL_HASH,
                "peer_context_protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
                "peer_context_protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
                "self_consistency_protocol_version": (
                    SELF_CONSISTENCY_PROTOCOL_VERSION
                ),
                "self_consistency_protocol_hash": SELF_CONSISTENCY_PROTOCOL_HASH,
            },
        }
        _canonical_bytes(self.identity)
        self._condition = threading.Condition(threading.RLock())
        self._inflight: set[str] = set()
        self._fatal: CheckpointError | None = None
        self._legacy_migration_cutoff: float | None = None
        self._payload = self._load_or_initialize()

    def assert_question(self, question: Question) -> None:
        """Reject proxy misuse with a different question before any replay/inference."""

        if canonical_question_payload(question) != self.identity["question"]:
            raise CheckpointIdentityError(
                "checkpoint proxy question does not match its frozen QID identity"
            )

    def _new_payload(self) -> dict[str, Any]:
        now = time.time()
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": copy.deepcopy(self.identity),
            "coordinates": {},
            "topology_terminal": None,
            "created_at": now,
            "updated_at": now,
            "migration_history": [],
        }
        payload["integrity_sha256"] = _sha256(payload)
        return payload

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_payload()
        try:
            raw = self.path.read_text(encoding="utf-8")
            value = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            _DuplicateJSONKey,
            _NonFiniteJSONNumber,
        ) as exc:
            raise CheckpointCorruptionError(
                f"cannot parse durable QID checkpoint {self.path}: {exc}"
            ) from exc
        if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
            raise CheckpointCorruptionError(
                f"durable QID checkpoint {self.path} has the wrong root schema"
            )
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptionError(
                f"unsupported QID checkpoint schema {value.get('schema_version')!r}"
            )
        integrity = value.get("integrity_sha256")
        if not isinstance(integrity, str) or integrity != _sha256(
            _integrity_payload(value)
        ):
            raise CheckpointCorruptionError(
                f"durable QID checkpoint {self.path} failed integrity validation"
            )
        if value.get("identity") != self.identity:
            raise CheckpointIdentityError(
                f"durable QID checkpoint {self.path} does not match the current "
                "cell/config/question/code/protocol/serving identity"
            )
        _finite_nonnegative(value.get("created_at"), "checkpoint.created_at")
        _finite_nonnegative(value.get("updated_at"), "checkpoint.updated_at")
        self._validate_migration_history(value.get("migration_history"))
        if not isinstance(value.get("coordinates"), dict):
            raise CheckpointCorruptionError("checkpoint.coordinates must be an object")
        for key, entry in value["coordinates"].items():
            self._validate_coordinate_entry(key, entry)
        terminal = value.get("topology_terminal")
        if terminal is not None:
            self._decode_terminal(terminal, coordinates=value["coordinates"])
        return value

    def _validate_migration_history(self, history: Any) -> None:
        """Validate the one registered schema transition without permitting growth."""

        if not isinstance(history, list):
            raise CheckpointCorruptionError(
                "checkpoint.migration_history must be a list"
            )
        if not history:
            self._legacy_migration_cutoff = None
            return
        if len(history) != 1:
            raise CheckpointCorruptionError(
                "checkpoint.migration_history permits exactly one registered migration"
            )
        record = history[0]
        if not isinstance(record, dict) or set(record) != _MIGRATION_HISTORY_FIELDS:
            raise CheckpointCorruptionError(
                "checkpoint migration_history record has the wrong fields"
            )
        expected = {
            "migration_schema_version": 1,
            "migration_type": _CHECKPOINT_SCHEMA_MIGRATION_TYPE,
            "source_checkpoint_schema_version": 1,
            "target_checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "migrator_code_version": self.identity["code_version"],
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                raise CheckpointCorruptionError(
                    f"checkpoint migration_history has invalid {field}"
                )
        _finite_nonnegative(
            record.get("migrated_at"), "checkpoint.migration_history.migrated_at"
        )
        for field in (
            "source_file_sha256",
            "source_integrity_sha256",
            "source_identity_sha256",
            "target_identity_sha256",
        ):
            if not _is_lower_sha256(record.get(field)):
                raise CheckpointCorruptionError(
                    f"checkpoint migration_history has invalid {field}"
                )
        if record["target_identity_sha256"] != _sha256(self.identity):
            raise CheckpointCorruptionError(
                "checkpoint migration_history target identity hash does not match identity"
            )
        self._legacy_migration_cutoff = float(record["migrated_at"])

    def _migration_cutoff(self) -> float | None:
        return self._legacy_migration_cutoff

    def _write_candidate(self, candidate: dict[str, Any]) -> None:
        try:
            candidate["updated_at"] = time.time()
            candidate["integrity_sha256"] = _sha256(_integrity_payload(candidate))
            io.write_json(self.path, candidate)
        except Exception as exc:
            raise CheckpointPersistenceError(
                f"could not atomically persist observed QID checkpoint {self.path}: {exc}"
            ) from exc

    def _latch_fatal(self, error: CheckpointError) -> None:
        with self._condition:
            if self._fatal is None:
                self._fatal = error

    def _expected_topology_seed(self, *, agent_index: int, round_idx: int) -> int:
        topology = self._cell.topology.value
        if topology == "single_agent":
            if agent_index != 0 or round_idx != 0:
                raise CheckpointCorruptionError(
                    "topology request coordinate is invalid for single_agent"
                )
            return self._cell.seed
        if topology == "independent":
            if not 0 <= agent_index < self._cell.n_agents or round_idx != 0:
                raise CheckpointCorruptionError(
                    "topology request coordinate is invalid for independent"
                )
            return self._cell.seed + agent_index
        if (
            not 0 <= agent_index < self._cell.n_agents
            or not 0 <= round_idx < self._cell.rounds
        ):
            raise CheckpointCorruptionError(
                "topology request agent/round is outside the manifest schedule"
            )
        if topology == "decentralized":
            return self._cell.seed + round_idx * 100 + agent_index
        if topology == "centralized":
            offset = 999 if agent_index == 0 else agent_index - 1
            return self._cell.seed + round_idx * 100 + offset
        raise CheckpointCorruptionError(
            f"unsupported topology request schedule {topology!r}"
        )

    def _expected_peer_blocks(self, *, agent_index: int, round_idx: int) -> int:
        topology = self._cell.topology.value
        if topology in {"single_agent", "independent"}:
            return 0
        if topology == "decentralized":
            return 0 if round_idx == 0 else max(0, self._cell.n_agents - 1)
        if topology == "centralized":
            if agent_index == 0:
                return max(0, self._cell.n_agents - 1)
            return 0 if round_idx == 0 else 1
        raise CheckpointCorruptionError(
            f"unsupported peer-context schedule {topology!r}"
        )

    def _request_agent_index(self, request: Mapping[str, Any]) -> int:
        agent_id = request.get("agent_id")
        match = _AGENT_ID_RE.fullmatch(agent_id) if isinstance(agent_id, str) else None
        if match is None:
            raise CheckpointCorruptionError(
                "coordinate request agent_id must use the registered agentN form"
            )
        return int(match.group(1))

    def _validate_peer_context_identity(
        self,
        value: Any,
        *,
        expected_blocks: int,
    ) -> None:
        if not isinstance(value, dict) or set(value) != _PEER_CONTEXT_IDENTITY_FIELDS:
            raise CheckpointCorruptionError(
                "coordinate peer_context identity has the wrong fields"
            )
        if not _is_lower_sha256(value.get("sha256")):
            raise CheckpointCorruptionError(
                "coordinate peer_context.sha256 must be a lowercase SHA-256"
            )
        if not _is_integer(value.get("utf8_bytes")):
            raise CheckpointCorruptionError(
                "coordinate peer_context.utf8_bytes must be a non-negative integer"
            )
        is_empty = value["utf8_bytes"] == 0
        if is_empty != (value["sha256"] == _EMPTY_SHA256):
            raise CheckpointCorruptionError(
                "coordinate peer_context empty-byte/hash identity is inconsistent"
            )
        if expected_blocks == 0 and not is_empty:
            raise CheckpointCorruptionError(
                "coordinate request has peer context where the topology schedules none"
            )
        if expected_blocks > 0 and is_empty:
            raise CheckpointCorruptionError(
                "coordinate request omits scheduled peer context"
            )

    def _validate_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise CheckpointCorruptionError("coordinate request must be an object")
        try:
            # Round-trip through canonical JSON to reject non-JSON/non-finite inputs.
            decoded = json.loads(_canonical_bytes(request).decode("utf-8"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptionError(
                f"invalid coordinate request: {exc}"
            ) from exc
        if decoded != request:
            raise CheckpointCorruptionError("coordinate request is not JSON-stable")

        role = request.get("generation_role")
        expected_fields = (
            _TOPOLOGY_REQUEST_FIELDS
            if role == "topology"
            else _SELF_CONSISTENCY_REQUEST_FIELDS
            if role == "self_consistency"
            else None
        )
        if expected_fields is None:
            raise CheckpointCorruptionError(
                f"unknown coordinate generation_role {role!r}"
            )
        if set(request) != expected_fields:
            raise CheckpointCorruptionError(
                f"{role} coordinate request has the wrong fields"
            )
        if request["qid"] != self.identity["question"]["qid"]:
            raise CheckpointCorruptionError(
                "coordinate request QID does not match checkpoint identity"
            )
        agent_index = self._request_agent_index(request)
        round_idx = request["round"]
        if not _is_integer(round_idx):
            raise CheckpointCorruptionError(
                "coordinate request round must be a non-negative integer"
            )
        if request["max_tokens"] != ANSWER_GENERATION_TOKEN_ALLOWANCE:
            raise CheckpointCorruptionError(
                "coordinate request max_tokens does not match the frozen protocol"
            )
        if request["elicit_cot"] is not True:
            raise CheckpointCorruptionError(
                "coordinate request elicit_cot must be true"
            )

        if role == "topology":
            if request["sample_index"] is not None:
                raise CheckpointCorruptionError(
                    "topology coordinate sample_index must be null"
                )
            expected_seed = self._expected_topology_seed(
                agent_index=agent_index,
                round_idx=round_idx,
            )
            if request["seed"] != expected_seed:
                raise CheckpointCorruptionError(
                    "topology coordinate seed does not match the manifest schedule"
                )
            expected_blocks = self._expected_peer_blocks(
                agent_index=agent_index,
                round_idx=round_idx,
            )
        else:
            if (
                self._cell.topology.value != "single_agent"
                or agent_index != 0
                or round_idx != 0
            ):
                raise CheckpointCorruptionError(
                    "self-consistency coordinate is invalid for the manifest topology"
                )
            sample_index = request["sample_index"]
            if (
                not _is_integer(sample_index)
                or sample_index >= self._cell.n_samples
            ):
                raise CheckpointCorruptionError(
                    "self-consistency sample_index is outside the manifest schedule"
                )
            expected_base_seed = self._cell.seed + 1000
            if request["base_seed"] != expected_base_seed:
                raise CheckpointCorruptionError(
                    "self-consistency base_seed does not match the manifest schedule"
                )
            if request["seed"] != expected_base_seed + sample_index:
                raise CheckpointCorruptionError(
                    "self-consistency seed does not match its scheduled sample"
                )
            if request["extra_kwargs"] != {}:
                raise CheckpointCorruptionError(
                    "self-consistency extra_kwargs are outside the frozen protocol"
                )
            expected_blocks = 0
        self._validate_peer_context_identity(
            request["peer_context"], expected_blocks=expected_blocks
        )
        return copy.deepcopy(request)

    def _validate_coordinate_entry(self, key: Any, entry: Any) -> None:
        if not isinstance(key, str) or not key:
            raise CheckpointCorruptionError("coordinate key must be non-empty text")
        if not isinstance(entry, dict) or set(entry) != {
            "request",
            "outcome",
            "observed_at",
            "producer_wall_ms",
        }:
            raise CheckpointCorruptionError(
                f"checkpoint coordinate {key!r} has the wrong fields"
            )
        request = self._validate_request(entry["request"])
        if key != _coordinate_key_from_request(request):
            raise CheckpointCorruptionError(
                f"checkpoint coordinate {key!r} does not match its request semantics"
            )
        _finite_nonnegative(entry["observed_at"], f"coordinate {key}.observed_at")
        _finite_nonnegative(
            entry["producer_wall_ms"], f"coordinate {key}.producer_wall_ms"
        )
        migration_cutoff = self._migration_cutoff()
        allow_legacy_peer_audit = (
            migration_cutoff is not None
            and float(entry["observed_at"]) <= migration_cutoff
        )
        self._validate_outcome(
            request,
            entry["outcome"],
            allow_legacy_peer_audit=allow_legacy_peer_audit,
        )

    def _validate_agent_peer_audit(
        self,
        output: AgentOutput,
        request: Mapping[str, Any],
        *,
        allow_legacy_peer_audit: bool,
    ) -> None:
        values = {
            "token_count": output.peer_context_tokens,
            "sha256": output.peer_context_sha256,
            "block_counts": output.peer_context_block_token_counts,
            "truncations": output.peer_context_truncation_marker_count,
        }
        if not _is_integer(values["token_count"]):
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer_context_tokens is invalid"
            )
        if not _is_lower_sha256(values["sha256"]):
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer_context_sha256 is invalid"
            )
        block_counts = values["block_counts"]
        if not isinstance(block_counts, list) or not all(
            _is_integer(count, minimum=1)
            and count <= PEER_RENDERED_BLOCK_TOKEN_LIMIT
            for count in block_counts
        ):
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer-context block counts are invalid"
            )
        truncations = values["truncations"]
        if (
            not _is_integer(truncations)
            or truncations > len(block_counts)
        ):
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer-context truncation count is invalid"
            )

        agent_index = self._request_agent_index(request)
        expected_blocks = (
            0
            if request["generation_role"] == "self_consistency"
            else self._expected_peer_blocks(
                agent_index=agent_index,
                round_idx=request["round"],
            )
        )
        legacy_default = (
            values["token_count"] == 0
            and values["sha256"] == _EMPTY_SHA256
            and block_counts == []
            and truncations == 0
        )
        # Schema-1 persisted AgentOutput before the topology attached these four
        # fields.  The audited request hash/byte count still identifies the consumed
        # text, and the terminal (when present) carries the later topology audit.  Keep
        # precisely that historical all-default shape readable at/before the sealed
        # migration timestamp; every schema-2 observation is validated strictly.
        if allow_legacy_peer_audit and expected_blocks > 0 and legacy_default:
            return
        if len(block_counts) != expected_blocks:
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer block count does not match topology"
            )
        request_peer = request["peer_context"]
        if values["sha256"] != request_peer["sha256"]:
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput peer-context hash does not match request"
            )
        if expected_blocks == 0:
            if not legacy_default or request_peer["utf8_bytes"] != 0:
                raise CheckpointCorruptionError(
                    "checkpoint AgentOutput empty peer-context audit is inconsistent"
                )
        elif values["token_count"] <= 0 or request_peer["utf8_bytes"] <= 0:
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput non-empty peer-context audit is inconsistent"
            )

    def _decode_censor(
        self,
        payload: Any,
        *,
        qid: str,
        agent_id: str,
        round_idx: int,
        generation_role: str,
        sample_index: int | None,
        seed: int | None,
    ) -> GenerationTruncationError:
        profile = self.identity["serving_profile"]
        return _censor_from_payload(
            payload,
            qid=qid,
            agent_id=agent_id,
            round_idx=round_idx,
            generation_role=generation_role,
            sample_index=sample_index,
            seed=seed,
            enable_thinking=self._cell.reasoning_level.enable_thinking,
            thinking_budget=self._cell.reasoning_level.thinking_budget,
            output_capacity_floor_tokens=requested_generation_tokens(
                self._cell.reasoning_level
            ),
            serving_profile=profile["name"],
            effective_context_limit=profile["max_model_len"],
        )

    def _validate_outcome(
        self,
        request: dict[str, Any],
        outcome: Any,
        *,
        allow_legacy_peer_audit: bool = False,
    ) -> None:
        role = request.get("generation_role")
        if role == "topology":
            if not isinstance(outcome, dict) or set(outcome) != {
                "termination_status",
                "agent_output",
                "censored_generation",
            }:
                raise CheckpointCorruptionError(
                    "topology coordinate outcome has the wrong fields"
                )
            status = outcome["termination_status"]
            if status == TERMINATION_COMPLETED:
                if outcome["censored_generation"] is not None:
                    raise CheckpointCorruptionError(
                        "completed topology coordinate contains a censor"
                    )
                output = _agent_output_from_payload(
                    outcome["agent_output"],
                    expected_agent_id=request.get("agent_id"),
                    expected_round=request.get("round"),
                )
                self._validate_agent_peer_audit(
                    output,
                    request,
                    allow_legacy_peer_audit=allow_legacy_peer_audit,
                )
            elif status in {
                TERMINATION_LENGTH_CENSORED,
                TERMINATION_PROTOCOL_CENSORED,
            }:
                if outcome["agent_output"] is not None:
                    raise CheckpointCorruptionError(
                        "censored topology coordinate contains AgentOutput"
                    )
                error = self._decode_censor(
                    outcome["censored_generation"],
                    qid=request.get("qid"),
                    agent_id=request.get("agent_id"),
                    round_idx=request.get("round"),
                    generation_role="topology",
                    sample_index=None,
                    seed=request.get("seed"),
                )
                expected_status = (
                    TERMINATION_PROTOCOL_CENSORED
                    if isinstance(error, GenerationProtocolCensorError)
                    else TERMINATION_LENGTH_CENSORED
                )
                if status != expected_status:
                    raise CheckpointCorruptionError(
                        "topology coordinate status does not match censor reason"
                    )
                self._validate_censor_annotations(outcome["censored_generation"])
            else:
                raise CheckpointCorruptionError(
                    f"invalid topology coordinate status {status!r}"
                )
            return
        if role == "self_consistency":
            if not isinstance(outcome, dict) or set(outcome) != {
                "sample_index",
                "seed",
                "termination_status",
                "agent_output",
                "censored_generation",
            }:
                raise CheckpointCorruptionError(
                    "self-consistency outcome has the wrong fields"
                )
            if outcome["sample_index"] != request.get("sample_index"):
                raise CheckpointCorruptionError(
                    "self-consistency sample_index does not match request"
                )
            if outcome["seed"] != request.get("seed"):
                raise CheckpointCorruptionError(
                    "self-consistency seed does not match request"
                )
            status = outcome["termination_status"]
            if status == TERMINATION_COMPLETED:
                if outcome["censored_generation"] is not None:
                    raise CheckpointCorruptionError(
                        "completed self-consistency outcome contains a censor"
                    )
                output = _agent_output_from_payload(
                    outcome["agent_output"],
                    expected_agent_id=request.get("agent_id"),
                    expected_round=request.get("round"),
                )
                self._validate_agent_peer_audit(
                    output,
                    request,
                    allow_legacy_peer_audit=allow_legacy_peer_audit,
                )
            elif status in {
                TERMINATION_LENGTH_CENSORED,
                TERMINATION_PROTOCOL_CENSORED,
            }:
                if outcome["agent_output"] is not None:
                    raise CheckpointCorruptionError(
                        "censored self-consistency outcome contains AgentOutput"
                    )
                error = self._decode_censor(
                    outcome["censored_generation"],
                    qid=request.get("qid"),
                    agent_id=request.get("agent_id"),
                    round_idx=request.get("round"),
                    generation_role="self_consistency",
                    sample_index=request.get("sample_index"),
                    seed=request.get("seed"),
                )
                expected_status = (
                    TERMINATION_PROTOCOL_CENSORED
                    if isinstance(error, GenerationProtocolCensorError)
                    else TERMINATION_LENGTH_CENSORED
                )
                if status != expected_status:
                    raise CheckpointCorruptionError(
                        "self-consistency status does not match censor reason"
                    )
                self._validate_censor_annotations(outcome["censored_generation"])
            else:
                raise CheckpointCorruptionError(
                    f"invalid self-consistency status {status!r}"
                )
            return
        raise CheckpointCorruptionError(f"unknown coordinate generation_role {role!r}")

    def _validate_censor_annotations(self, censor: Mapping[str, Any]) -> None:
        expected = {
            "topology": self.identity["cell_config"]["topology"],
            "benchmark": self.identity["question"]["benchmark"],
        }
        for field, value in expected.items():
            if field in censor and censor[field] != value:
                raise CheckpointCorruptionError(
                    f"censored generation {field} annotation does not match identity"
                )

    def execute_coordinate(
        self,
        key: str,
        request: dict[str, Any],
        producer: Callable[[], dict[str, Any]],
        *,
        admit_producer: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Produce once or replay one exact coordinate, synchronized across siblings.

        ``admit_producer`` runs only for a genuinely missing coordinate, immediately
        before that coordinate becomes in-flight.  A graceful-drain callback can reject
        new work without obstructing exact replay.  Once admitted, the producer is
        always allowed to return and its observation is atomically persisted even if a
        drain signal arrives while the request is in flight.
        """

        # Preserve the stronger identity-drift diagnostic for a slot already observed;
        # the stored request remains the only legal identity even if the challenger is
        # also outside the manifest schedule.
        with self._condition:
            prior = self._payload["coordinates"].get(key)
            if prior is not None and prior["request"] != request:
                raise CheckpointIdentityError(
                    f"coordinate {key!r} was already observed under a different "
                    "request identity"
                )
        request = self._validate_request(request)
        if key != _coordinate_key_from_request(request):
            raise CheckpointCorruptionError(
                f"coordinate key {key!r} does not match its request semantics"
            )
        with self._condition:
            if self._fatal is not None:
                raise self._fatal
            while key in self._inflight:
                self._condition.wait()
            if self._fatal is not None:
                raise self._fatal
            existing = self._payload["coordinates"].get(key)
            if existing is not None:
                if existing["request"] != request:
                    raise CheckpointIdentityError(
                        f"coordinate {key!r} was already observed under a different "
                        "request identity"
                    )
                migration_cutoff = self._migration_cutoff()
                allow_legacy_peer_audit = (
                    migration_cutoff is not None
                    and float(existing["observed_at"]) <= migration_cutoff
                )
                try:
                    self._validate_outcome(
                        request,
                        existing["outcome"],
                        allow_legacy_peer_audit=allow_legacy_peer_audit,
                    )
                except CheckpointError as exc:
                    self._fatal = exc
                    raise
                return copy.deepcopy(existing["outcome"])
            if admit_producer is not None:
                # Keep this check inside the same condition critical section as the
                # transition to ``_inflight``.  That makes admission a single local
                # boundary: no cooperating sibling can create the coordinate between
                # the drain decision and producer ownership.
                admit_producer()
            self._inflight.add(key)

        started = time.monotonic()
        observation_returned = False
        try:
            outcome = producer()
            observation_returned = True
            self._validate_outcome(request, outcome)
            entry = {
                "request": request,
                "outcome": copy.deepcopy(outcome),
                "observed_at": time.time(),
                "producer_wall_ms": (time.monotonic() - started) * 1000.0,
            }
            with self._condition:
                candidate = copy.deepcopy(self._payload)
                existing = candidate["coordinates"].get(key)
                if existing is not None:
                    if existing["request"] != request or existing["outcome"] != outcome:
                        raise CheckpointIdentityError(
                            f"coordinate {key!r} raced with a different observation"
                        )
                else:
                    candidate["coordinates"][key] = entry
                    self._write_candidate(candidate)
                    self._payload = candidate
                return copy.deepcopy(outcome)
        except CheckpointError as exc:
            # In particular, never let a duplicate waiter reissue an observation after
            # its producer returned but the atomic checkpoint write failed.
            self._latch_fatal(exc)
            raise
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            if not observation_returned:
                raise
            wrapped = CheckpointCorruptionError(
                "observed coordinate could not satisfy the durable JSON contract: "
                f"{exc}"
            )
            self._latch_fatal(wrapped)
            raise wrapped from exc
        finally:
            with self._condition:
                self._inflight.discard(key)
                self._condition.notify_all()

    def _validate_terminal_censor_binding(
        self,
        censor: Mapping[str, Any],
        *,
        coordinates: Mapping[str, Any],
    ) -> None:
        observed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for entry in coordinates.values():
            request = entry["request"]
            outcome = entry["outcome"]
            if request.get("generation_role") != "topology":
                continue
            if outcome.get("termination_status") not in {
                TERMINATION_LENGTH_CENSORED,
                TERMINATION_PROTOCOL_CENSORED,
            }:
                continue
            observed.append((request, outcome["censored_generation"]))
        exact_matches = [
            request for request, payload in observed if payload == censor
        ]
        if len(exact_matches) != 1:
            raise CheckpointCorruptionError(
                "topology terminal censor must match exactly one observed coordinate"
            )

        terminal_request = exact_matches[0]
        round_idx = terminal_request["round"]
        agent_index = self._request_agent_index(terminal_request)
        topology = self._cell.topology.value
        if topology == "centralized" and agent_index == 0:
            wave = [
                (request, payload)
                for request, payload in observed
                if request["round"] == round_idx
                and self._request_agent_index(request) == 0
            ]
        elif topology == "centralized":
            wave = [
                (request, payload)
                for request, payload in observed
                if request["round"] == round_idx
                and self._request_agent_index(request) > 0
            ]
        else:
            wave = [
                (request, payload)
                for request, payload in observed
                if request["round"] == round_idx
            ]
        if not wave:
            raise CheckpointCorruptionError(
                "topology terminal censor has no legal execution wave"
            )
        propagated_request, propagated_payload = min(
            wave,
            key=lambda item: self._request_agent_index(item[0]),
        )
        if propagated_request != terminal_request or propagated_payload != censor:
            raise CheckpointCorruptionError(
                "topology terminal censor is not the deterministic map-order censor"
            )

    def _decode_terminal(
        self,
        terminal: Any,
        *,
        coordinates: Mapping[str, Any] | None = None,
    ) -> TopologyTerminal:
        if not isinstance(terminal, dict) or set(terminal) != {
            "termination_status",
            "topology_result",
            "censored_generation",
            "wall_ms",
            "observed_at",
        }:
            raise CheckpointCorruptionError("topology terminal has the wrong fields")
        wall_ms = _finite_nonnegative(terminal["wall_ms"], "terminal.wall_ms")
        _finite_nonnegative(terminal["observed_at"], "terminal.observed_at")
        status = terminal["termination_status"]
        question = self.identity["question"]
        if status == TERMINATION_COMPLETED:
            if terminal["censored_generation"] is not None:
                raise CheckpointCorruptionError(
                    "completed topology terminal contains a censor"
                )
            result = _topology_result_from_payload(terminal["topology_result"])
            return TopologyTerminal(
                termination_status=status,
                topology_result=result,
                wall_ms=wall_ms,
            )
        if status in {
            TERMINATION_LENGTH_CENSORED,
            TERMINATION_PROTOCOL_CENSORED,
        }:
            if terminal["topology_result"] is not None:
                raise CheckpointCorruptionError(
                    "censored topology terminal contains TopologyResult"
                )
            censor = terminal["censored_generation"]
            if not isinstance(censor, dict):
                raise CheckpointCorruptionError(
                    "censored topology terminal has no censor object"
                )
            error = self._decode_censor(
                censor,
                qid=question["qid"],
                agent_id=censor.get("agent_id"),
                round_idx=censor.get("round"),
                generation_role="topology",
                sample_index=None,
                seed=censor.get("seed"),
            )
            expected_status = (
                TERMINATION_PROTOCOL_CENSORED
                if isinstance(error, GenerationProtocolCensorError)
                else TERMINATION_LENGTH_CENSORED
            )
            if status != expected_status:
                raise CheckpointCorruptionError(
                    "topology terminal status does not match censor reason"
                )
            self._validate_censor_annotations(censor)
            self._validate_terminal_censor_binding(
                censor,
                coordinates=(
                    self._payload["coordinates"]
                    if coordinates is None
                    else coordinates
                ),
            )
            return TopologyTerminal(
                termination_status=status,
                censored_error=error,
                wall_ms=wall_ms,
            )
        raise CheckpointCorruptionError(f"invalid topology terminal status {status!r}")

    def topology_terminal(self) -> TopologyTerminal | None:
        with self._condition:
            terminal = self._payload["topology_terminal"]
            return (
                None
                if terminal is None
                else self._decode_terminal(copy.deepcopy(terminal))
            )

    def _record_terminal(self, terminal: dict[str, Any]) -> None:
        try:
            decoded = self._decode_terminal(terminal)
            del decoded  # Callers consume through topology_terminal().
        except CheckpointError as exc:
            self._latch_fatal(exc)
            raise
        with self._condition:
            if self._fatal is not None:
                raise self._fatal
            existing = self._payload["topology_terminal"]
            if existing is not None:
                if existing != terminal:
                    raise CheckpointIdentityError(
                        "topology terminal was already recorded with a different outcome"
                    )
                return
            candidate = copy.deepcopy(self._payload)
            candidate["topology_terminal"] = copy.deepcopy(terminal)
            try:
                self._write_candidate(candidate)
            except CheckpointError as exc:
                self._fatal = exc
                raise
            self._payload = candidate

    def record_topology_result(self, result: TopologyResult, *, wall_ms: float) -> None:
        try:
            terminal = {
                "termination_status": TERMINATION_COMPLETED,
                "topology_result": _topology_result_payload(result),
                "censored_generation": None,
                "wall_ms": _finite_nonnegative(wall_ms, "terminal.wall_ms"),
                "observed_at": time.time(),
            }
            self._record_terminal(terminal)
        except CheckpointError as exc:
            self._latch_fatal(exc)
            raise

    def record_topology_censor(
        self, error: GenerationTruncationError, *, wall_ms: float
    ) -> None:
        try:
            status = (
                TERMINATION_PROTOCOL_CENSORED
                if isinstance(error, GenerationProtocolCensorError)
                else TERMINATION_LENGTH_CENSORED
            )
            terminal = {
                "termination_status": status,
                "topology_result": None,
                "censored_generation": error.to_censored_generation(),
                "wall_ms": _finite_nonnegative(wall_ms, "terminal.wall_ms"),
                "observed_at": time.time(),
            }
            self._record_terminal(terminal)
        except CheckpointError as exc:
            self._latch_fatal(exc)
            raise
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            wrapped = CheckpointCorruptionError(
                f"observed topology censor is not durable canonical JSON: {exc}"
            )
            self._latch_fatal(wrapped)
            raise wrapped from exc

    def upgrade_legacy_peer_audit(
        self,
        key: str,
        request: dict[str, Any],
        rendered: PeerContextRender,
    ) -> dict[str, Any]:
        """Durably enrich one migrated schema-1 output without another model draw.

        Schema 1 journaled ``AgentOutput`` immediately and the topology attached its
        peer-context fields only afterward.  On a resumed replay the exact rendered
        text is available again and is already hash-bound to the durable request.  This
        method atomically fills only those four derived fields, preserves the original
        observation timestamp/cost/outcome, and validates the upgraded outcome under
        the strict schema-2 contract before exposing it to callers or snapshots.
        """

        request = self._validate_request(request)
        if key != _coordinate_key_from_request(request):
            raise CheckpointCorruptionError(
                f"coordinate key {key!r} does not match its request semantics"
            )
        if request["generation_role"] != "topology":
            raise CheckpointCorruptionError(
                "legacy peer-audit upgrade applies only to topology coordinates"
            )
        rendered_identity = _peer_context_identity(rendered.text)
        if (
            rendered.sha256 != rendered_identity["sha256"]
            or request["peer_context"] != rendered_identity
        ):
            raise CheckpointIdentityError(
                "legacy peer-audit render does not match the durable request identity"
            )
        with self._condition:
            if self._fatal is not None:
                raise self._fatal
            entry = self._payload["coordinates"].get(key)
            if entry is None or entry["request"] != request:
                raise CheckpointIdentityError(
                    f"coordinate {key!r} is unavailable under this request identity"
                )
            outcome = entry["outcome"]
            if outcome.get("termination_status") != TERMINATION_COMPLETED:
                return copy.deepcopy(outcome)
            try:
                # A strict native or previously upgraded observation needs no write.
                self._validate_outcome(request, outcome)
                return copy.deepcopy(outcome)
            except CheckpointCorruptionError as strict_error:
                cutoff = self._migration_cutoff()
                if cutoff is None or float(entry["observed_at"]) > cutoff:
                    self._fatal = strict_error
                    raise
                # Prove that this is precisely the registered schema-1 omission, not a
                # general repair mechanism for malformed or changed observations.
                self._validate_outcome(
                    request,
                    outcome,
                    allow_legacy_peer_audit=True,
                )

            candidate = copy.deepcopy(self._payload)
            durable_output = candidate["coordinates"][key]["outcome"][
                "agent_output"
            ]
            durable_output["peer_context_tokens"] = rendered.token_count
            durable_output["peer_context_sha256"] = rendered.sha256
            durable_output["peer_context_block_token_counts"] = list(
                rendered.block_token_counts
            )
            durable_output["peer_context_truncation_marker_count"] = (
                rendered.truncation_marker_count
            )
            upgraded = candidate["coordinates"][key]["outcome"]
            try:
                self._validate_outcome(request, upgraded)
                self._write_candidate(candidate)
            except CheckpointError as exc:
                self._fatal = exc
                raise
            self._payload = candidate
            return copy.deepcopy(upgraded)

    def observed_topology_coordinates(self) -> list[dict[str, Any]]:
        """Return strict schema-2 topology observations in stable key order.

        A migrated peer-consuming coordinate must first be replay-upgraded; this keeps
        a later censor snapshot from publishing the historical all-default audit.
        """

        with self._condition:
            observed = []
            for key in sorted(self._payload["coordinates"]):
                entry = self._payload["coordinates"][key]
                if entry["request"].get("generation_role") != "topology":
                    continue
                self._validate_outcome(entry["request"], entry["outcome"])
                observed.append({"coordinate_key": key, **copy.deepcopy(entry)})
            return observed

    def topology_producer_wall_ms(self) -> float:
        """Return total producer time across all durable topology observations.

        This is additive compute time, not elapsed wall time; unlike a caller-local
        stopwatch it survives endpoint replacement and process preemption.
        """

        with self._condition:
            return sum(
                float(entry["producer_wall_ms"])
                for entry in self._payload["coordinates"].values()
                if entry["request"].get("generation_role") == "topology"
            )

    def topology_critical_path_wall_ms(self) -> float:
        """Estimate durable elapsed inference time over observed legal phases.

        Siblings within independent/decentralized rounds and centralized sub-agent
        phases execute concurrently, so their phase cost is the maximum producer time.
        Centralized synthesis is sequential after its sub-agent phase.  A partial
        checkpoint yields a lower bound; a complete coordinate schedule yields the
        checkpoint-visible critical path (exclusive of endpoint selection/retries).
        """

        with self._condition:
            coordinates = [
                copy.deepcopy(entry)
                for entry in self._payload["coordinates"].values()
                if entry["request"].get("generation_role") == "topology"
            ]
        topology = self._cell.topology.value
        if topology == "single_agent":
            return sum(float(entry["producer_wall_ms"]) for entry in coordinates)
        if topology == "independent":
            return max(
                (float(entry["producer_wall_ms"]) for entry in coordinates),
                default=0.0,
            )
        by_round: dict[int, list[dict[str, Any]]] = {}
        for entry in coordinates:
            by_round.setdefault(entry["request"]["round"], []).append(entry)
        total = 0.0
        for round_idx in sorted(by_round):
            entries = by_round[round_idx]
            if topology == "decentralized":
                total += max(float(entry["producer_wall_ms"]) for entry in entries)
                continue
            orchestrator = [
                entry
                for entry in entries
                if entry["request"]["agent_id"] == "agent0"
            ]
            sub_agents = [
                entry
                for entry in entries
                if entry["request"]["agent_id"] != "agent0"
            ]
            total += max(
                (float(entry["producer_wall_ms"]) for entry in sub_agents),
                default=0.0,
            )
            total += sum(
                float(entry["producer_wall_ms"]) for entry in orchestrator
            )
        return total

    def self_consistency_wall_ms(self) -> float:
        """Return durable producer time for all retained auxiliary coordinates."""

        with self._condition:
            return sum(
                float(entry["producer_wall_ms"])
                for entry in self._payload["coordinates"].values()
                if entry["request"].get("generation_role") == "self_consistency"
            )

    def delete(self) -> None:
        """Delete only after the caller has fsynced the canonical QID result row."""

        with self._condition:
            if self._inflight:
                raise CheckpointPersistenceError(
                    "cannot delete a QID checkpoint while coordinates are in flight"
                )
            io.remove_file(self.path)


def _peer_context_identity(peer_context: str) -> dict[str, Any]:
    if not isinstance(peer_context, str):
        raise TypeError("peer_context must be text")
    payload = peer_context.encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "utf8_bytes": len(payload),
    }


class CheckpointingAgent:
    """Duck-typed Agent proxy that durably executes each stochastic coordinate once."""

    def __init__(
        self,
        agent: Agent,
        checkpoint: QIDCheckpoint,
        *,
        agent_id: str | None = None,
        admit_coordinate: Callable[[], None] | None = None,
    ) -> None:
        self._agent = agent
        self._checkpoint = checkpoint
        resolved_agent_id = (
            agent_id if agent_id is not None else getattr(agent, "agent_id", None)
        )
        if not isinstance(resolved_agent_id, str) or not resolved_agent_id:
            raise TypeError("checkpointed agent requires a non-empty agent_id")
        self._agent_id = resolved_agent_id
        self._admit_coordinate = admit_coordinate

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def prepare_calibration(self, question: Question) -> Any:
        if self._admit_coordinate is not None:
            self._admit_coordinate()
        return self._agent.prepare_calibration(question)

    @staticmethod
    def _coordinate_key(request: Mapping[str, Any]) -> str:
        # Seed, prompt identity, and peer-context hash are attributes of one semantic
        # slot, not alternate slots.  Request drift must fail closed instead of being
        # admitted as a second draw.
        return _coordinate_key_from_request(request)

    def _validated_peer_context_render(
        self,
        peer_context: str,
        rendered: PeerContextRender | None,
    ) -> PeerContextRender:
        if not isinstance(peer_context, str):
            raise CheckpointIdentityError("checkpoint peer_context must be text")
        if rendered is None:
            if peer_context:
                raise CheckpointIdentityError(
                    "non-empty checkpoint peer_context requires its exact render audit"
                )
            rendered = PeerContextRender(
                text="",
                token_count=0,
                sha256=_EMPTY_SHA256,
            )
        if not isinstance(rendered, PeerContextRender):
            raise CheckpointIdentityError(
                "checkpoint peer-context audit must be PeerContextRender"
            )
        expected_hash = hashlib.sha256(peer_context.encode("utf-8")).hexdigest()
        if rendered.text != peer_context or rendered.sha256 != expected_hash:
            raise CheckpointIdentityError(
                "peer-context render text/hash does not match the exact request"
            )
        if not _is_integer(rendered.token_count):
            raise CheckpointIdentityError(
                "peer-context render token_count must be a non-negative integer"
            )
        if not isinstance(rendered.block_token_counts, tuple) or not all(
            _is_integer(count, minimum=1)
            and count <= PEER_RENDERED_BLOCK_TOKEN_LIMIT
            for count in rendered.block_token_counts
        ):
            raise CheckpointIdentityError(
                "peer-context render block counts violate the protocol"
            )
        if (
            not _is_integer(rendered.truncation_marker_count)
            or rendered.truncation_marker_count > len(rendered.block_token_counts)
        ):
            raise CheckpointIdentityError(
                "peer-context render truncation count violates the protocol"
            )
        if not peer_context and (
            rendered.token_count != 0
            or rendered.block_token_counts
            or rendered.truncation_marker_count != 0
        ):
            raise CheckpointIdentityError(
                "empty peer-context render has non-empty audit values"
            )
        if peer_context and (
            rendered.token_count <= 0 or not rendered.block_token_counts
        ):
            raise CheckpointIdentityError(
                "non-empty peer-context render has an empty audit"
            )

        # The production Agent's client owns the same profile tokenizer used by the
        # topology renderer.  Re-tokenize when that concrete capability is available;
        # duck-typed test agents need only satisfy the exact render/hash contract.
        client = getattr(self._agent, "client", None)
        tokenizer_factory = getattr(client, "_protocol_tokenizer", None)
        if callable(tokenizer_factory):
            try:
                tokenizer = tokenizer_factory()
                token_ids = tokenizer.encode(
                    peer_context,
                    add_special_tokens=False,
                )
                if isinstance(token_ids, Mapping):
                    token_ids = token_ids["input_ids"]
                recomputed_count = len(token_ids)
            except (KeyError, TypeError, ValueError) as exc:
                raise CheckpointIdentityError(
                    f"could not verify peer-context token count: {exc}"
                ) from exc
            if recomputed_count != rendered.token_count:
                raise CheckpointIdentityError(
                    "peer-context render token count does not match profile tokenizer"
                )
        return rendered

    @staticmethod
    def _attach_peer_context_audit(
        output: AgentOutput,
        rendered: PeerContextRender,
    ) -> AgentOutput:
        if not isinstance(output, AgentOutput):
            raise CheckpointCorruptionError(
                "coordinate did not produce AgentOutput before peer-context audit"
            )
        output.peer_context_tokens = rendered.token_count
        output.peer_context_sha256 = rendered.sha256
        output.peer_context_block_token_counts = list(rendered.block_token_counts)
        output.peer_context_truncation_marker_count = (
            rendered.truncation_marker_count
        )
        return output

    @staticmethod
    def _assert_output_matches_peer_context_audit(
        output: AgentOutput,
        rendered: PeerContextRender,
    ) -> None:
        if (
            output.peer_context_tokens != rendered.token_count
            or output.peer_context_sha256 != rendered.sha256
            or output.peer_context_block_token_counts
            != list(rendered.block_token_counts)
            or output.peer_context_truncation_marker_count
            != rendered.truncation_marker_count
        ):
            raise CheckpointCorruptionError(
                "checkpoint AgentOutput does not match the supplied peer-context audit"
            )

    def answer_with_peer_context_audit(
        self,
        question: Question,
        *,
        round_idx: int,
        peer_context_render: PeerContextRender,
        max_tokens: int,
        seed: int | None,
        elicit_cot: bool = True,
    ) -> AgentOutput:
        """Answer while binding the topology's exact render audit before persistence."""

        return self.answer(
            question,
            round_idx=round_idx,
            peer_context=peer_context_render.text,
            max_tokens=max_tokens,
            seed=seed,
            elicit_cot=elicit_cot,
            _peer_context_render=peer_context_render,
        )

    def answer(
        self,
        question: Question,
        round_idx: int = 0,
        peer_context: str = "",
        max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
        seed: int | None = None,
        elicit_cot: bool = True,
        _peer_context_render: PeerContextRender | None = None,
    ) -> AgentOutput:
        self._checkpoint.assert_question(question)
        rendered = self._validated_peer_context_render(
            peer_context,
            _peer_context_render,
        )
        request = {
            "generation_role": "topology",
            "qid": question.qid,
            "agent_id": self.agent_id,
            "round": round_idx,
            "seed": seed,
            "sample_index": None,
            "peer_context": _peer_context_identity(peer_context),
            "max_tokens": max_tokens,
            "elicit_cot": elicit_cot,
        }
        key = self._coordinate_key(request)

        def produce() -> dict[str, Any]:
            try:
                output = self._agent.answer(
                    question,
                    round_idx=round_idx,
                    peer_context=peer_context,
                    max_tokens=max_tokens,
                    seed=seed,
                    elicit_cot=elicit_cot,
                )
            except GenerationTruncationError as exc:
                exc.enrich(
                    qid=question.qid,
                    agent_id=self.agent_id,
                    round_idx=round_idx,
                    generation_role="topology",
                    sample_index=None,
                )
                return {
                    "termination_status": (
                        TERMINATION_PROTOCOL_CENSORED
                        if isinstance(exc, GenerationProtocolCensorError)
                        else TERMINATION_LENGTH_CENSORED
                    ),
                    "agent_output": None,
                    "censored_generation": exc.to_censored_generation(),
                }
            output = self._attach_peer_context_audit(output, rendered)
            return {
                "termination_status": TERMINATION_COMPLETED,
                "agent_output": _agent_output_payload(output),
                "censored_generation": None,
            }

        outcome = self._checkpoint.execute_coordinate(
            key,
            request,
            produce,
            admit_producer=self._admit_coordinate,
        )
        if outcome["termination_status"] == TERMINATION_COMPLETED:
            outcome = self._checkpoint.upgrade_legacy_peer_audit(
                key,
                request,
                rendered,
            )
            output = _agent_output_from_payload(
                outcome["agent_output"],
                expected_agent_id=self.agent_id,
                expected_round=round_idx,
            )
            self._assert_output_matches_peer_context_audit(output, rendered)
            return output
        raise self._checkpoint._decode_censor(
            outcome["censored_generation"],
            qid=question.qid,
            agent_id=self.agent_id,
            round_idx=round_idx,
            generation_role="topology",
            sample_index=None,
            seed=seed,
        )

    def sample_one(
        self,
        question: Question,
        *,
        sample_index: int,
        base_seed: int = 0,
        **kwargs: Any,
    ) -> SelfConsistencySample:
        self._checkpoint.assert_question(question)
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        round_idx = int(kwargs.get("round_idx", 0))
        peer_context = kwargs.get("peer_context", "")
        max_tokens = kwargs.get("max_tokens", ANSWER_GENERATION_TOKEN_ALLOWANCE)
        elicit_cot = kwargs.get("elicit_cot", True)
        seed = base_seed + sample_index
        extra_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"round_idx", "peer_context", "max_tokens", "elicit_cot"}
        }
        rendered = self._validated_peer_context_render(peer_context, None)
        request = {
            "generation_role": "self_consistency",
            "qid": question.qid,
            "agent_id": self.agent_id,
            "round": round_idx,
            "seed": seed,
            "sample_index": sample_index,
            "base_seed": base_seed,
            "peer_context": _peer_context_identity(peer_context),
            "max_tokens": max_tokens,
            "elicit_cot": elicit_cot,
            "extra_kwargs": extra_kwargs,
        }
        key = self._coordinate_key(request)

        def produce() -> dict[str, Any]:
            sample = self._agent.sample_one(
                question,
                sample_index=sample_index,
                base_seed=base_seed,
                **kwargs,
            )
            if not isinstance(sample, SelfConsistencySample):
                raise CheckpointCorruptionError(
                    "self-consistency coordinate did not produce SelfConsistencySample"
                )
            output = sample.agent_output
            if output is not None:
                output = self._attach_peer_context_audit(output, rendered)
            return {
                "sample_index": sample.sample_index,
                "seed": sample.seed,
                "termination_status": sample.termination_status,
                "agent_output": (
                    _agent_output_payload(output)
                    if output is not None
                    else None
                ),
                "censored_generation": copy.deepcopy(sample.censored_generation),
            }

        outcome = self._checkpoint.execute_coordinate(
            key,
            request,
            produce,
            admit_producer=self._admit_coordinate,
        )
        if outcome["termination_status"] == TERMINATION_COMPLETED:
            output = _agent_output_from_payload(
                outcome["agent_output"],
                expected_agent_id=self.agent_id,
                expected_round=round_idx,
            )
            self._assert_output_matches_peer_context_audit(output, rendered)
            return SelfConsistencySample(
                sample_index=sample_index,
                seed=seed,
                termination_status=TERMINATION_COMPLETED,
                agent_output=output,
            )
        # Reconstruct before returning so hashes, coordinates, and every censor field
        # are validated rather than trusting a dictionary merely covered by integrity.
        error = self._checkpoint._decode_censor(
            outcome["censored_generation"],
            qid=question.qid,
            agent_id=self.agent_id,
            round_idx=round_idx,
            generation_role="self_consistency",
            sample_index=sample_index,
            seed=seed,
        )
        return SelfConsistencySample(
            sample_index=sample_index,
            seed=seed,
            termination_status=(
                TERMINATION_PROTOCOL_CENSORED
                if isinstance(error, GenerationProtocolCensorError)
                else TERMINATION_LENGTH_CENSORED
            ),
            censored_generation=copy.deepcopy(outcome["censored_generation"]),
        )
