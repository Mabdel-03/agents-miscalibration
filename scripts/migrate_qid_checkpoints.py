#!/usr/bin/env python3
"""Migrate frozen-run QID checkpoints from schema 1 to schema 2.

This is an evidence-preserving, offline migration.  It considers only cells in a
checksum-frozen manifest, verifies their frozen benchmark questions and deterministic
serving profiles, and accepts only the one schema-1 executable identity emitted by the
controlled rollout.  Before replacing any checkpoint, the exact source text and both
the old and new identities are durably recorded in a run-level incident artifact.

The default is a read-only audit.  Pass ``--apply`` explicitly after all legacy workers
have drained.  Busy cells are skipped via the runner's nonblocking advisory cell lock.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Permit the documented ``python scripts/...`` invocation from a checkout.
REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.benchmarks.contracts import (  # noqa: E402
    BenchmarkContractError,
    BenchmarkLoader,
    canonical_question_payload,
    canonical_question_sha256,
)
from agents_scaling.benchmarks.loaders import load_benchmark  # noqa: E402
from agents_scaling.benchmarks.runtime_contracts import (  # noqa: E402
    VerifiedQuestionCatalog,
)
from agents_scaling.benchmarks.schema import Question  # noqa: E402
from agents_scaling.config import DEFAULT_RESULTS_ROOT, ExperimentCell  # noqa: E402
from agents_scaling.experiment import io  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    CellLockUnavailable,
    cell_lock,
    is_cell_active,
)
from agents_scaling.experiment.manifest import (  # noqa: E402
    ManifestSnapshot,
    load_manifest,
)
from agents_scaling.experiment.qid_checkpoint import (  # noqa: E402
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_SCHEMA_VERSION as EXECUTABLE_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    QIDCheckpoint,
    checkpoint_path,
)
from agents_scaling.serving.profiles import (  # noqa: E402
    ServingProfile,
    serving_profile_for_cell,
)

SOURCE_CHECKPOINT_SCHEMA_VERSION = 1
MIGRATION_SCHEMA_VERSION = 1
INCIDENT_SCHEMA_VERSION = 1
INCIDENT_FILENAME = "qid_checkpoint_schema_1_to_2_incident_v1.json"
RUN_LOCK_FILENAME = ".qid_checkpoint_schema_1_to_2.lock"
MIGRATION_TYPE = "qid_checkpoint_schema_1_to_2"

# This is deliberately an exact allowlist, not a pattern.  A checkpoint produced by a
# different dirty source tree requires a separately reviewed migration contract.
ALLOWED_SOURCE_CODE_VERSIONS = frozenset(
    {
        "404292772cd56372cb86812475e990800cd22941+source.063bd094ecd6f1c3",
    }
)

LEGACY_PROTOCOL_IDENTITY: dict[str, Any] = {
    "artifact_schema_version": 4,
    "thinking_budget_protocol_version": 4,
    "thinking_budget_protocol_hash": (
        "fb190c66f48d056a80f327b04ae2a33e733efd9cee2cd9386e979570eb354326"
    ),
    "peer_context_protocol_version": 2,
    "peer_context_protocol_hash": (
        "16068be1c93259a1aaeca27f307ef430c167b36df71c1af2b3b0cf0d6c26292b"
    ),
    "self_consistency_protocol_version": 1,
    "self_consistency_protocol_hash": (
        "30431be07be4a56373dd82434307b3c05edd31fba0eda5a0c8cc02557e9d3b39"
    ),
}

_SOURCE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "coordinates",
        "topology_terminal",
        "created_at",
        "updated_at",
        "integrity_sha256",
    }
)
_TARGET_ROOT_FIELDS = _SOURCE_ROOT_FIELDS | {"migration_history"}
_LEGACY_COORDINATE_FIELDS = frozenset(
    {"request", "outcome", "observed_at", "producer_wall_ms"}
)
_LEGACY_TOPOLOGY_OUTCOME_FIELDS = frozenset(
    {"termination_status", "agent_output", "censored_generation"}
)
_LEGACY_SELF_CONSISTENCY_OUTCOME_FIELDS = _LEGACY_TOPOLOGY_OUTCOME_FIELDS | {
    "sample_index",
    "seed",
}
_LEGACY_TERMINAL_FIELDS = frozenset(
    {
        "termination_status",
        "topology_result",
        "censored_generation",
        "wall_ms",
        "observed_at",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "cell_id",
        "cell_config",
        "config_hash",
        "benchmark_contract_sha256",
        "question",
        "question_sha256",
        "code_version",
        "serving_profile",
        "protocols",
    }
)
_MIGRATION_RECORD_FIELDS = frozenset(
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
_INCIDENT_ROOT_FIELDS = frozenset(
    {
        "incident_schema_version",
        "incident_type",
        "run_id",
        "manifest",
        "benchmark_contracts",
        "target",
        "checkpoints",
    }
)
_INCIDENT_ENTRY_FIELDS = frozenset(
    {
        "evidence_id",
        "cell_id",
        "manifest_index",
        "manifest_config_hash",
        "qid",
        "checkpoint_relative_path",
        "source_checkpoint_text",
        "source_checkpoint_sha256",
        "source_integrity_sha256",
        "source_identity",
        "source_identity_sha256",
        "target_identity",
        "target_identity_sha256",
        "migration_history_record",
        "target_checkpoint_sha256",
        "target_integrity_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_NAME_RE = re.compile(r"[0-9a-f]{64}\.json")
_DERIVED_PEER_AUDIT_FIELDS = frozenset(
    {
        "peer_context_tokens",
        "peer_context_sha256",
        "peer_context_block_token_counts",
        "peer_context_truncation_marker_count",
    }
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class MigrationError(RuntimeError):
    """The requested migration could not be proven safe."""


class _InvalidJSON(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointPlan:
    path: Path
    relative_path: str
    cell: ExperimentCell
    manifest_index: int
    question: Question
    source_text: str
    source_sha256: str
    source_payload: dict[str, Any]
    target_payload: dict[str, Any]
    target_text: str
    target_sha256: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CellScan:
    plans: tuple[CheckpointPlan, ...]
    native_current: int
    already_migrated: int


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise _InvalidJSON(f"non-finite JSON number {value!r}")


def _strict_json_loads(text: str, *, path: Path | None = None) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, _InvalidJSON) as exc:
        location = "" if path is None else f" {path}"
        raise MigrationError(f"cannot parse strict JSON{location}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"value is not canonical finite JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"candidate is not finite JSON: {exc}") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MigrationError(f"{label} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise MigrationError(f"{label} must be a finite non-negative number")
    return converted


def _profile_payload(profile: ServingProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "model_size": profile.model_size,
        "hf_id": profile.hf_id,
        "tp_size": profile.tp_size,
        "max_model_len": profile.max_model_len,
        "served_model_name": profile.served_model_name,
    }


def _integrity_sha256(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}
    )


def _read_regular_utf8(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"checkpoint is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise MigrationError(f"cannot read checkpoint {path}: {exc}") from exc
    return text, _bytes_sha256(raw)


def _require_frozen_run(
    run_root: Path,
    *,
    benchmark_loader: BenchmarkLoader,
) -> tuple[ManifestSnapshot, VerifiedQuestionCatalog]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise MigrationError(f"run root is missing or unsafe: {run_root}")
    try:
        snapshot = load_manifest(run_root, verify_frozen=True)
        catalog = VerifiedQuestionCatalog(
            run_root,
            snapshot=snapshot,
            benchmark_loader=benchmark_loader,
        )
    except (OSError, UnicodeError, ValueError, BenchmarkContractError) as exc:
        raise MigrationError(f"cannot verify frozen run {run_root}: {exc}") from exc
    # ``load_manifest`` treats a missing checksum as an unfrozen development manifest;
    # an offline production migration must instead require both regular checksum files.
    for required in (
        run_root / "cells.json",
        run_root / "cells.sha256",
        catalog.frozen.path,
        catalog.frozen.checksum_path,
    ):
        if required.is_symlink() or not required.is_file():
            raise MigrationError(
                f"required frozen-run artifact is missing or unsafe: {required}"
            )
    return snapshot, catalog


def _verify_run_unchanged(
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    *,
    target_code_version: str,
) -> None:
    try:
        observed = load_manifest(run_root, verify_frozen=True)
        catalog.verify_unchanged()
    except (OSError, UnicodeError, ValueError, BenchmarkContractError) as exc:
        raise MigrationError(f"frozen run changed during migration: {exc}") from exc
    if observed.sha256 != snapshot.sha256 or observed.cells != snapshot.cells:
        raise MigrationError("frozen manifest changed during migration")
    if io.git_commit() != target_code_version:
        raise MigrationError("executable source identity changed during migration")


def _expected_current_identity(
    staging_root: Path,
    *,
    cell: ExperimentCell,
    question: Question,
    profile: ServingProfile,
    benchmark_contract_sha256: str,
    target_code_version: str,
) -> dict[str, Any]:
    checkpoint = QIDCheckpoint(
        staging_root / "identity" / cell.cell_id,
        cell,
        question,
        code_version=target_code_version,
        serving_profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
    )
    return copy.deepcopy(checkpoint.identity)


def _validate_source_root(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _SOURCE_ROOT_FIELDS:
        raise MigrationError(f"schema-1 checkpoint has the wrong root fields: {path}")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise MigrationError(f"checkpoint is not exact schema 1: {path}")
    identity = payload.get("identity")
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS:
        raise MigrationError(
            f"schema-1 checkpoint has the wrong identity fields: {path}"
        )
    integrity = payload.get("integrity_sha256")
    if not _is_sha256(integrity) or integrity != _integrity_sha256(payload):
        raise MigrationError(f"schema-1 checkpoint failed integrity validation: {path}")
    _finite_nonnegative(payload.get("created_at"), f"{path}.created_at")
    _finite_nonnegative(payload.get("updated_at"), f"{path}.updated_at")
    if not isinstance(payload.get("coordinates"), dict):
        raise MigrationError(f"schema-1 coordinates must be an object: {path}")
    return identity


def _question_map(
    catalog: VerifiedQuestionCatalog,
    cell: ExperimentCell,
) -> dict[str, Question]:
    questions = catalog.questions_for(cell)
    by_qid = {question.qid: question for question in questions}
    if len(by_qid) != len(questions):
        raise MigrationError(
            f"frozen question contract has duplicate QIDs: {cell.cell_id}"
        )
    return by_qid


def _validate_source_identity(
    *,
    identity: Mapping[str, Any],
    path: Path,
    cell: ExperimentCell,
    questions: Mapping[str, Question],
    profile: ServingProfile,
    benchmark_contract_sha256: str,
) -> Question:
    if identity.get("cell_id") != cell.cell_id:
        raise MigrationError(f"checkpoint cell_id does not match manifest: {path}")
    if identity.get("cell_config") != cell.to_dict():
        raise MigrationError(f"checkpoint cell_config does not match manifest: {path}")
    if identity.get("config_hash") != cell.config_hash():
        raise MigrationError(f"checkpoint config_hash does not match manifest: {path}")
    if identity.get("benchmark_contract_sha256") != benchmark_contract_sha256:
        raise MigrationError(
            f"checkpoint benchmark contract does not match frozen run: {path}"
        )
    if identity.get("serving_profile") != _profile_payload(profile):
        raise MigrationError(
            f"checkpoint serving profile does not match deterministic route: {path}"
        )
    if identity.get("protocols") != LEGACY_PROTOCOL_IDENTITY:
        raise MigrationError(f"checkpoint protocol identity is not allowlisted: {path}")
    if identity.get("code_version") not in ALLOWED_SOURCE_CODE_VERSIONS:
        raise MigrationError(
            f"checkpoint source code identity is not allowlisted: {path}"
        )

    question_payload = identity.get("question")
    if not isinstance(question_payload, dict):
        raise MigrationError(f"checkpoint question identity is malformed: {path}")
    qid = question_payload.get("qid")
    question = questions.get(qid) if isinstance(qid, str) else None
    if question is None:
        raise MigrationError(
            f"checkpoint QID is absent from frozen question contract: {path}"
        )
    if question_payload != canonical_question_payload(question):
        raise MigrationError(
            f"checkpoint question content does not match frozen contract: {path}"
        )
    if identity.get("question_sha256") != canonical_question_sha256(question):
        raise MigrationError(
            f"checkpoint question hash does not match frozen contract: {path}"
        )
    if path != checkpoint_path(path.parent.parent, question.qid):
        raise MigrationError(
            f"checkpoint filename does not match its frozen QID: {path}"
        )
    return question


def _migration_record(
    *,
    migrated_at: float,
    source_sha256: str,
    source_integrity_sha256: str,
    source_identity_sha256: str,
    target_identity_sha256: str,
    target_code_version: str,
) -> dict[str, Any]:
    return {
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_type": MIGRATION_TYPE,
        "migrated_at": migrated_at,
        "source_checkpoint_schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "target_checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_file_sha256": source_sha256,
        "source_integrity_sha256": source_integrity_sha256,
        "source_identity_sha256": source_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
        "migrator_code_version": target_code_version,
    }


def _validate_migration_record(record: Any, *, target_code_version: str) -> None:
    if not isinstance(record, dict) or set(record) != _MIGRATION_RECORD_FIELDS:
        raise MigrationError("checkpoint migration_history record has the wrong fields")
    expected_scalars = {
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_type": MIGRATION_TYPE,
        "source_checkpoint_schema_version": SOURCE_CHECKPOINT_SCHEMA_VERSION,
        "target_checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "migrator_code_version": target_code_version,
    }
    for field, expected in expected_scalars.items():
        if record.get(field) != expected:
            raise MigrationError(f"checkpoint migration_history has invalid {field}")
    _finite_nonnegative(record.get("migrated_at"), "migration_history.migrated_at")
    for field in (
        "source_file_sha256",
        "source_integrity_sha256",
        "source_identity_sha256",
        "target_identity_sha256",
    ):
        if not _is_sha256(record.get(field)):
            raise MigrationError(f"checkpoint migration_history has invalid {field}")


def _evidence_id(relative_path: str, source_sha256: str) -> str:
    return _canonical_sha256(
        {
            "migration_type": MIGRATION_TYPE,
            "checkpoint_relative_path": relative_path,
            "source_checkpoint_sha256": source_sha256,
        }
    )


def _validate_candidate(
    staging_root: Path,
    *,
    target_payload: dict[str, Any],
    cell: ExperimentCell,
    question: Question,
    profile: ServingProfile,
    benchmark_contract_sha256: str,
    target_code_version: str,
    evidence_id: str,
) -> str:
    validation_root = staging_root / "candidate" / evidence_id
    target_path = checkpoint_path(validation_root, question.qid)
    target_text = _json_text(target_payload)
    projection = _schema2_executable_validation_projection(
        target_payload,
        evidence_id=evidence_id,
    )
    io.atomic_write_text(target_path, _json_text(projection))
    reopened = QIDCheckpoint(
        validation_root,
        cell,
        question,
        code_version=target_code_version,
        serving_profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
    )
    if reopened.identity != target_payload["identity"]:
        raise MigrationError("schema-2 candidate failed exact identity validation")
    return target_text


def _schema2_executable_validation_projection(
    payload: Mapping[str, Any],
    *,
    evidence_id: str,
) -> dict[str, Any]:
    """Project sealed schema-2 evidence into a disposable schema-3 validator input.

    Schema 2 intentionally remains non-executable after the no-redraw journal upgrade.
    The legacy migrator still has to validate all preserved requests, outcomes, censors,
    terminals, and migration history.  Build a validation-only schema-3 copy with
    deterministic synthetic attempt records; never write those inferred records into
    the historical checkpoint or claim that they were observed at runtime.
    """

    if not isinstance(payload, Mapping) or set(payload) != _TARGET_ROOT_FIELDS:
        raise MigrationError("schema-2 checkpoint has the wrong root fields")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise MigrationError("checkpoint is not exact sealed schema 2")
    if (
        not _is_sha256(payload.get("integrity_sha256"))
        or payload["integrity_sha256"] != _integrity_sha256(payload)
    ):
        raise MigrationError("schema-2 checkpoint failed integrity validation")
    if not _is_sha256(evidence_id):
        raise MigrationError("schema-2 validation evidence ID is invalid")
    _finite_nonnegative(payload.get("created_at"), "schema-2 checkpoint.created_at")
    _finite_nonnegative(payload.get("updated_at"), "schema-2 checkpoint.updated_at")
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise MigrationError("schema-2 checkpoint coordinates must be an object")

    projection = copy.deepcopy(dict(payload))
    projection["schema_version"] = EXECUTABLE_CHECKPOINT_SCHEMA_VERSION
    projection["pending_attempts"] = {}
    for ordinal, (key, coordinate) in enumerate(
        sorted(projection["coordinates"].items())
    ):
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(coordinate, dict)
            or set(coordinate) != _LEGACY_COORDINATE_FIELDS
        ):
            raise MigrationError(
                f"schema-2 checkpoint coordinate {key!r} has the wrong fields"
            )
        request = coordinate.get("request")
        outcome = coordinate.get("outcome")
        role = request.get("generation_role") if isinstance(request, dict) else None
        expected_outcome_fields = (
            _LEGACY_TOPOLOGY_OUTCOME_FIELDS
            if role == "topology"
            else _LEGACY_SELF_CONSISTENCY_OUTCOME_FIELDS
            if role == "self_consistency"
            else None
        )
        if (
            expected_outcome_fields is None
            or not isinstance(outcome, dict)
            or set(outcome) != expected_outcome_fields
        ):
            raise MigrationError(
                f"schema-2 checkpoint coordinate {key!r} has an invalid outcome shape"
            )
        _finite_nonnegative(
            coordinate.get("observed_at"),
            f"schema-2 checkpoint coordinate {key!r}.observed_at",
        )
        _finite_nonnegative(
            coordinate.get("producer_wall_ms"),
            f"schema-2 checkpoint coordinate {key!r}.producer_wall_ms",
        )
        output = outcome.get("agent_output")
        if isinstance(output, dict):
            # This provenance field was added with schema 3.  A null value is used
            # only inside the disposable validator projection; the sealed schema-2
            # bytes retain the exact historical AgentOutput.
            output.setdefault("calibration_endpoint_generation", None)
        outcome["transport_censor"] = None
        coordinate["attempt"] = {
            "attempt_id": hashlib.sha256(
                f"{evidence_id}:{ordinal}:{key}".encode("utf-8")
            ).hexdigest()[:32],
            "coordinate_key": key,
            "request_sha256": _canonical_sha256(request),
            "endpoint_generation": "sealed-schema2-validation-only",
            "started_at": max(
                float(coordinate["observed_at"])
                - float(coordinate["producer_wall_ms"]) / 1000.0,
                1e-9,
            ),
        }

    terminal = projection.get("topology_terminal")
    if terminal is not None:
        if not isinstance(terminal, dict) or set(terminal) != _LEGACY_TERMINAL_FIELDS:
            raise MigrationError("schema-2 checkpoint terminal has the wrong fields")
        result = terminal.get("topology_result")
        if isinstance(result, dict):
            per_agent = result.get("per_agent")
            if isinstance(per_agent, list):
                for output in per_agent:
                    if isinstance(output, dict):
                        output.setdefault("calibration_endpoint_generation", None)
        terminal["transport_censor"] = None

    projection["integrity_sha256"] = _integrity_sha256(projection)
    return projection


def _new_incident(
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    *,
    target_code_version: str,
) -> dict[str, Any]:
    return {
        "incident_schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_type": MIGRATION_TYPE,
        "run_id": run_root.name,
        "manifest": {
            "filename": snapshot.path.name,
            "sha256": snapshot.sha256,
            "cell_count": len(snapshot.cells),
        },
        "benchmark_contracts": {
            "filename": catalog.frozen.path.name,
            "sha256": catalog.frozen.sidecar_sha256,
        },
        "target": {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "code_version": target_code_version,
        },
        "checkpoints": {},
    }


def _validate_incident_entry(
    evidence_id: str,
    entry: Any,
    *,
    snapshot: ManifestSnapshot,
    target_code_version: str,
) -> None:
    if not isinstance(entry, dict) or set(entry) != _INCIDENT_ENTRY_FIELDS:
        raise MigrationError(f"incident checkpoint {evidence_id!r} has wrong fields")
    if entry.get("evidence_id") != evidence_id or not _is_sha256(evidence_id):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} has invalid evidence id"
        )
    index = entry.get("manifest_index")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(snapshot.cells)
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} has invalid manifest index"
        )
    cell = snapshot.cells[index]
    if (
        entry.get("cell_id") != cell.cell_id
        or entry.get("manifest_config_hash") != cell.config_hash()
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} does not match manifest"
        )
    qid = entry.get("qid")
    relative = entry.get("checkpoint_relative_path")
    if not isinstance(qid, str) or not qid or not isinstance(relative, str):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} has invalid path identity"
        )
    expected_relative = (
        Path("cells")
        / cell.cell_id
        / CHECKPOINT_DIRECTORY
        / (hashlib.sha256(qid.encode("utf-8")).hexdigest() + ".json")
    )
    if relative != expected_relative.as_posix():
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} has unsafe relative path"
        )
    source_text = entry.get("source_checkpoint_text")
    source_sha256 = entry.get("source_checkpoint_sha256")
    if not isinstance(source_text, str) or not _is_sha256(source_sha256):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} has invalid source evidence"
        )
    if _bytes_sha256(source_text.encode("utf-8")) != source_sha256:
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source checksum failed"
        )
    if _evidence_id(relative, source_sha256) != evidence_id:
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} evidence id is inconsistent"
        )
    source_payload = _strict_json_loads(source_text)
    if not isinstance(source_payload, dict):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source is not an object"
        )
    _validate_source_root(source_payload, Path(relative))
    if source_payload.get("identity") != entry.get("source_identity"):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source identity drifted"
        )
    if source_payload.get("integrity_sha256") != entry.get("source_integrity_sha256"):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source integrity drifted"
        )
    for name in (
        "source_integrity_sha256",
        "source_identity_sha256",
        "target_identity_sha256",
        "target_checkpoint_sha256",
        "target_integrity_sha256",
    ):
        if not _is_sha256(entry.get(name)):
            raise MigrationError(
                f"incident checkpoint {evidence_id!r} has invalid {name}"
            )
    if _canonical_sha256(entry["source_identity"]) != entry["source_identity_sha256"]:
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} old identity checksum failed"
        )
    if (
        _canonical_sha256(entry.get("target_identity"))
        != entry["target_identity_sha256"]
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} new identity checksum failed"
        )
    record = entry.get("migration_history_record")
    _validate_migration_record(record, target_code_version=target_code_version)
    source_updated_at = source_payload.get("updated_at")
    _finite_nonnegative(source_updated_at, "source checkpoint updated_at")
    if float(record["migrated_at"]) < float(source_updated_at):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} migration predates its source"
        )
    record_links = {
        "source_file_sha256": source_sha256,
        "source_integrity_sha256": entry["source_integrity_sha256"],
        "source_identity_sha256": entry["source_identity_sha256"],
        "target_identity_sha256": entry["target_identity_sha256"],
    }
    for field, expected in record_links.items():
        if record.get(field) != expected:
            raise MigrationError(
                f"incident checkpoint {evidence_id!r} record has invalid {field}"
            )
    source_identity = entry["source_identity"]
    target_identity = entry["target_identity"]
    if (
        not isinstance(source_identity, dict)
        or set(source_identity) != _IDENTITY_FIELDS
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source identity has wrong fields"
        )
    if (
        not isinstance(target_identity, dict)
        or set(target_identity) != _IDENTITY_FIELDS
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} target identity has wrong fields"
        )
    if (
        source_identity.get("code_version") not in ALLOWED_SOURCE_CODE_VERSIONS
        or source_identity.get("protocols") != LEGACY_PROTOCOL_IDENTITY
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} source identity is not allowlisted"
        )
    if target_identity.get("code_version") != target_code_version:
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} target code identity drifted"
        )
    unchanged_identity_fields = _IDENTITY_FIELDS - {"code_version", "protocols"}
    if any(
        source_identity[field] != target_identity[field]
        for field in unchanged_identity_fields
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} changed scientific identity"
        )
    # Reconstruct the exact initial schema-2 bytes from the preserved schema-1 source.
    # This binds the incident's target checksums even after a resumed worker has
    # legitimately appended new coordinates to the live migrated checkpoint.
    target_payload = copy.deepcopy(source_payload)
    target_payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    target_payload["identity"] = copy.deepcopy(target_identity)
    target_payload["migration_history"] = [copy.deepcopy(record)]
    target_payload["integrity_sha256"] = _integrity_sha256(target_payload)
    if target_payload["integrity_sha256"] != entry["target_integrity_sha256"]:
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} target integrity checksum failed"
        )
    reconstructed_text = _json_text(target_payload)
    if (
        _bytes_sha256(reconstructed_text.encode("utf-8"))
        != entry["target_checkpoint_sha256"]
    ):
        raise MigrationError(
            f"incident checkpoint {evidence_id!r} target file checksum failed"
        )


def _load_incident(
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    *,
    target_code_version: str,
) -> dict[str, Any]:
    expected = _new_incident(
        run_root,
        snapshot,
        catalog,
        target_code_version=target_code_version,
    )
    path = run_root / INCIDENT_FILENAME
    if not path.exists():
        return expected
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"incident path is unsafe: {path}")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), path=path)
    except (OSError, UnicodeError) as exc:
        raise MigrationError(f"cannot read incident {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _INCIDENT_ROOT_FIELDS:
        raise MigrationError(f"incident has the wrong root fields: {path}")
    for field in _INCIDENT_ROOT_FIELDS - {"checkpoints"}:
        if value.get(field) != expected[field]:
            raise MigrationError(f"incident has incompatible {field}: {path}")
    entries = value.get("checkpoints")
    if not isinstance(entries, dict):
        raise MigrationError(f"incident checkpoints must be an object: {path}")
    for evidence_id, entry in entries.items():
        _validate_incident_entry(
            evidence_id,
            entry,
            snapshot=snapshot,
            target_code_version=target_code_version,
        )
    return value


def _validate_current_checkpoint(
    staging_root: Path,
    *,
    path: Path,
    payload: Any,
    source_text: str,
    cell: ExperimentCell,
    manifest_index: int,
    questions: Mapping[str, Question],
    profile: ServingProfile,
    benchmark_contract_sha256: str,
    target_code_version: str,
    incident: Mapping[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        raise MigrationError(f"current checkpoint is not an object: {path}")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise MigrationError(f"current checkpoint identity is malformed: {path}")
    question_payload = identity.get("question")
    qid = question_payload.get("qid") if isinstance(question_payload, dict) else None
    question = questions.get(qid) if isinstance(qid, str) else None
    if question is None:
        raise MigrationError(
            f"current checkpoint QID is absent from frozen contract: {path}"
        )
    if path != checkpoint_path(path.parent.parent, qid):
        raise MigrationError(f"current checkpoint filename does not match QID: {path}")
    validation_root = staging_root / "current" / cell.cell_id / path.stem
    validation_path = checkpoint_path(validation_root, qid)
    projection = _schema2_executable_validation_projection(
        payload,
        evidence_id=_canonical_sha256(
            {
                "checkpoint_path": str(path),
                "checkpoint_sha256": _bytes_sha256(source_text.encode("utf-8")),
            }
        ),
    )
    io.atomic_write_text(validation_path, _json_text(projection))
    QIDCheckpoint(
        validation_root,
        cell,
        question,
        code_version=target_code_version,
        serving_profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
    )
    history = payload.get("migration_history")
    if not isinstance(history, list):
        raise MigrationError(
            f"current checkpoint migration_history is malformed: {path}"
        )
    if not history:
        return False
    if len(history) != 1:
        raise MigrationError(
            f"current checkpoint has unsupported migration history: {path}"
        )
    record = history[0]
    _validate_migration_record(record, target_code_version=target_code_version)
    relative = path.relative_to(path.parents[3]).as_posix()
    evidence_id = _evidence_id(relative, record["source_file_sha256"])
    entry = incident.get("checkpoints", {}).get(evidence_id)
    if not isinstance(entry, dict):
        raise MigrationError(f"migrated checkpoint lacks its incident evidence: {path}")
    if (
        entry.get("migration_history_record") != record
        or entry.get("target_identity") != identity
        or entry.get("cell_id") != cell.cell_id
        or entry.get("manifest_index") != manifest_index
        or entry.get("qid") != qid
        or entry.get("checkpoint_relative_path") != relative
    ):
        raise MigrationError(
            f"migrated checkpoint does not match incident evidence: {path}"
        )
    source_payload = _strict_json_loads(entry["source_checkpoint_text"])
    pristine_target = copy.deepcopy(source_payload)
    pristine_target["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    pristine_target["identity"] = copy.deepcopy(entry["target_identity"])
    pristine_target["migration_history"] = [copy.deepcopy(record)]
    pristine_target["integrity_sha256"] = _integrity_sha256(pristine_target)
    pristine_text = _json_text(pristine_target)
    if (
        pristine_target["integrity_sha256"]
        != entry.get("target_integrity_sha256")
        or _bytes_sha256(pristine_text.encode("utf-8"))
        != entry.get("target_checkpoint_sha256")
    ):
        raise MigrationError(
            f"migrated checkpoint incident target cannot be reconstructed: {path}"
        )

    # Exact pristine bytes are the initial state, not a lifetime invariant: a resumed
    # worker legitimately appends missing coordinates/terminal state, and replay can
    # fill the four schema-1 peer-audit defaults from an exact hash-bound render.  Prove
    # that the current journal is a monotone descendant of the sealed pristine target.
    immutable_root_fields = {
        "schema_version",
        "identity",
        "created_at",
        "migration_history",
    }
    for field in immutable_root_fields:
        if payload.get(field) != pristine_target.get(field):
            raise MigrationError(
                f"migrated checkpoint changed immutable {field}: {path}"
            )
    if float(payload["updated_at"]) < float(pristine_target["updated_at"]):
        raise MigrationError(f"migrated checkpoint updated_at regressed: {path}")
    pristine_coordinates = pristine_target["coordinates"]
    current_coordinates = payload["coordinates"]
    if not set(pristine_coordinates).issubset(current_coordinates):
        raise MigrationError(
            f"migrated checkpoint removed a preserved coordinate: {path}"
        )
    migrated_at = float(record["migrated_at"])
    for key, current_coordinate in current_coordinates.items():
        pristine_coordinate = pristine_coordinates.get(key)
        if pristine_coordinate is None:
            if float(current_coordinate["observed_at"]) < migrated_at:
                raise MigrationError(
                    f"migrated checkpoint appended a predating coordinate: {path}"
                )
            continue
        if current_coordinate == pristine_coordinate:
            continue
        if not _is_derived_peer_audit_upgrade(
            pristine_coordinate,
            current_coordinate,
        ):
            raise MigrationError(
                f"migrated checkpoint changed a preserved observation: {path}"
            )
    pristine_terminal = pristine_target["topology_terminal"]
    current_terminal = payload["topology_terminal"]
    if pristine_terminal is not None and current_terminal != pristine_terminal:
        raise MigrationError(
            f"migrated checkpoint changed its preserved terminal: {path}"
        )
    if (
        pristine_terminal is None
        and current_terminal is not None
        and float(current_terminal["observed_at"]) < migrated_at
    ):
        raise MigrationError(
            f"migrated checkpoint appended a predating terminal: {path}"
        )
    return True


def _is_derived_peer_audit_upgrade(
    pristine_coordinate: Any,
    current_coordinate: Any,
) -> bool:
    """Recognize only the no-redraw schema-1 peer-audit enrichment."""

    if not isinstance(pristine_coordinate, dict) or not isinstance(
        current_coordinate, dict
    ):
        return False
    if set(pristine_coordinate) != set(current_coordinate):
        return False
    for field in ("request", "observed_at", "producer_wall_ms"):
        if pristine_coordinate.get(field) != current_coordinate.get(field):
            return False
    pristine_outcome = pristine_coordinate.get("outcome")
    current_outcome = current_coordinate.get("outcome")
    if not isinstance(pristine_outcome, dict) or not isinstance(current_outcome, dict):
        return False
    if set(pristine_outcome) != set(current_outcome):
        return False
    for field in ("termination_status", "censored_generation"):
        if pristine_outcome.get(field) != current_outcome.get(field):
            return False
    if pristine_outcome.get("termination_status") != "completed":
        return False
    pristine_output = pristine_outcome.get("agent_output")
    current_output = current_outcome.get("agent_output")
    if not isinstance(pristine_output, dict) or not isinstance(current_output, dict):
        return False
    if set(pristine_output) != set(current_output):
        return False
    unchanged = set(pristine_output) - _DERIVED_PEER_AUDIT_FIELDS
    if any(pristine_output[field] != current_output[field] for field in unchanged):
        return False
    if {
        field: pristine_output.get(field) for field in _DERIVED_PEER_AUDIT_FIELDS
    } != {
        "peer_context_tokens": 0,
        "peer_context_sha256": _EMPTY_SHA256,
        "peer_context_block_token_counts": [],
        "peer_context_truncation_marker_count": 0,
    }:
        return False
    # QIDCheckpoint has already validated the current fields strictly against its
    # request/topology before this lineage check; require an actual enrichment here.
    return any(
        pristine_output[field] != current_output[field]
        for field in _DERIVED_PEER_AUDIT_FIELDS
    )


def _plan_source_checkpoint(
    staging_root: Path,
    *,
    run_root: Path,
    path: Path,
    payload: dict[str, Any],
    source_text: str,
    source_sha256: str,
    cell: ExperimentCell,
    manifest_index: int,
    questions: Mapping[str, Question],
    profile: ServingProfile,
    benchmark_contract_sha256: str,
    target_code_version: str,
    migration_timestamp: float,
    incident: Mapping[str, Any],
) -> CheckpointPlan:
    source_identity = _validate_source_root(payload, path)
    question = _validate_source_identity(
        identity=source_identity,
        path=path,
        cell=cell,
        questions=questions,
        profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
    )
    relative = path.relative_to(run_root).as_posix()
    evidence_id = _evidence_id(relative, source_sha256)
    target_identity = _expected_current_identity(
        staging_root,
        cell=cell,
        question=question,
        profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
        target_code_version=target_code_version,
    )
    source_identity_sha256 = _canonical_sha256(source_identity)
    target_identity_sha256 = _canonical_sha256(target_identity)
    existing = incident.get("checkpoints", {}).get(evidence_id)
    if existing is not None:
        # ``_load_incident`` already validated the complete entry against the real
        # frozen manifest.  Reuse its timestamp so a crash after evidence publication
        # remains byte-idempotent.
        record = copy.deepcopy(existing["migration_history_record"])
    else:
        record = _migration_record(
            migrated_at=migration_timestamp,
            source_sha256=source_sha256,
            source_integrity_sha256=payload["integrity_sha256"],
            source_identity_sha256=source_identity_sha256,
            target_identity_sha256=target_identity_sha256,
            target_code_version=target_code_version,
        )
    _validate_migration_record(record, target_code_version=target_code_version)
    if float(record["migrated_at"]) < float(payload["updated_at"]):
        raise MigrationError(
            f"checkpoint migration timestamp predates its source: {path}"
        )
    expected_record_links = {
        "source_file_sha256": source_sha256,
        "source_integrity_sha256": payload["integrity_sha256"],
        "source_identity_sha256": source_identity_sha256,
        "target_identity_sha256": target_identity_sha256,
    }
    if any(
        record.get(field) != value for field, value in expected_record_links.items()
    ):
        raise MigrationError(
            f"incident record does not match source checkpoint: {path}"
        )

    target_payload = copy.deepcopy(payload)
    target_payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    target_payload["identity"] = target_identity
    target_payload["migration_history"] = [record]
    target_payload["integrity_sha256"] = _integrity_sha256(target_payload)
    target_text = _validate_candidate(
        staging_root,
        target_payload=target_payload,
        cell=cell,
        question=question,
        profile=profile,
        benchmark_contract_sha256=benchmark_contract_sha256,
        target_code_version=target_code_version,
        evidence_id=evidence_id,
    )
    target_sha256 = _bytes_sha256(target_text.encode("utf-8"))
    evidence = {
        "evidence_id": evidence_id,
        "cell_id": cell.cell_id,
        "manifest_index": manifest_index,
        "manifest_config_hash": cell.config_hash(),
        "qid": question.qid,
        "checkpoint_relative_path": relative,
        "source_checkpoint_text": source_text,
        "source_checkpoint_sha256": source_sha256,
        "source_integrity_sha256": payload["integrity_sha256"],
        "source_identity": copy.deepcopy(source_identity),
        "source_identity_sha256": source_identity_sha256,
        "target_identity": copy.deepcopy(target_identity),
        "target_identity_sha256": target_identity_sha256,
        "migration_history_record": record,
        "target_checkpoint_sha256": target_sha256,
        "target_integrity_sha256": target_payload["integrity_sha256"],
    }
    if existing is not None and existing != evidence:
        raise MigrationError(
            f"existing incident evidence disagrees with migration plan: {path}"
        )
    return CheckpointPlan(
        path=path,
        relative_path=relative,
        cell=cell,
        manifest_index=manifest_index,
        question=question,
        source_text=source_text,
        source_sha256=source_sha256,
        source_payload=copy.deepcopy(payload),
        target_payload=target_payload,
        target_text=target_text,
        target_sha256=target_sha256,
        evidence=evidence,
    )


def _scan_cell(
    staging_root: Path,
    *,
    run_root: Path,
    cell_directory: Path,
    cell: ExperimentCell,
    manifest_index: int,
    catalog: VerifiedQuestionCatalog,
    target_code_version: str,
    migration_timestamp: float,
    incident: Mapping[str, Any],
) -> CellScan:
    checkpoint_directory = cell_directory / CHECKPOINT_DIRECTORY
    if not checkpoint_directory.exists():
        return CellScan((), 0, 0)
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise MigrationError(f"checkpoint directory is unsafe: {checkpoint_directory}")
    paths: list[Path] = []
    for path in sorted(checkpoint_directory.iterdir()):
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or _CHECKPOINT_NAME_RE.fullmatch(path.name) is None
        ):
            raise MigrationError(f"unexpected checkpoint-directory entry: {path}")
        paths.append(path)
    if not paths:
        return CellScan((), 0, 0)

    questions = _question_map(catalog, cell)
    profile = serving_profile_for_cell(cell)
    contract = catalog.frozen.contract_for_cell(cell)
    benchmark_contract_sha256 = contract.get("question_contract_sha256")
    if not _is_sha256(benchmark_contract_sha256):
        raise MigrationError(
            f"frozen benchmark contract hash is malformed: {cell.cell_id}"
        )

    plans: list[CheckpointPlan] = []
    native_current = 0
    already_migrated = 0
    for path in paths:
        source_text, source_sha256 = _read_regular_utf8(path)
        payload = _strict_json_loads(source_text, path=path)
        if not isinstance(payload, dict):
            raise MigrationError(f"checkpoint root is not an object: {path}")
        schema = payload.get("schema_version")
        if type(schema) is not int:
            raise MigrationError(f"checkpoint has a non-integer schema marker: {path}")
        if schema == SOURCE_CHECKPOINT_SCHEMA_VERSION:
            plans.append(
                _plan_source_checkpoint(
                    staging_root,
                    run_root=run_root,
                    path=path,
                    payload=payload,
                    source_text=source_text,
                    source_sha256=source_sha256,
                    cell=cell,
                    manifest_index=manifest_index,
                    questions=questions,
                    profile=profile,
                    benchmark_contract_sha256=benchmark_contract_sha256,
                    target_code_version=target_code_version,
                    migration_timestamp=migration_timestamp,
                    incident=incident,
                )
            )
        elif schema == CHECKPOINT_SCHEMA_VERSION:
            migrated = _validate_current_checkpoint(
                staging_root,
                path=path,
                payload=payload,
                source_text=source_text,
                cell=cell,
                manifest_index=manifest_index,
                questions=questions,
                profile=profile,
                benchmark_contract_sha256=benchmark_contract_sha256,
                target_code_version=target_code_version,
                incident=incident,
            )
            already_migrated += int(migrated)
            native_current += int(not migrated)
        else:
            raise MigrationError(f"unsupported checkpoint schema {schema!r}: {path}")
    return CellScan(tuple(plans), native_current, already_migrated)


def _new_report(
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    *,
    apply: bool,
    target_code_version: str,
    migration_timestamp: float,
) -> dict[str, Any]:
    cells_root = run_root / "cells"
    present = (
        {
            path.name
            for path in cells_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        }
        if cells_root.is_dir() and not cells_root.is_symlink()
        else set()
    )
    return {
        "run_root": str(run_root),
        "manifest_sha256": snapshot.sha256,
        "manifest_cells": len(snapshot.cells),
        "benchmark_contracts_sha256": catalog.frozen.sidecar_sha256,
        "target_code_version": target_code_version,
        "planned_migration_timestamp": migration_timestamp,
        "applied": apply,
        "stale_unmanifested_dirs": len(present - set(snapshot.ids)),
        "schema1_candidates": 0,
        "schema2_native": 0,
        "schema2_already_migrated": 0,
        "checkpoints_migrated": 0,
        # Exact, content-addressed rewrite plan.  This is populated in dry-run and
        # apply modes before any checkpoint is replaced, so an operator can compare
        # the complete mutation boundary with the sealed migration incident.
        "would_change_checkpoints": [],
        "active_or_locked_cells_skipped": 0,
        "unsafe_or_invalid_cells_skipped": 0,
        "incident_sha256": None,
        "errors": [],
    }


def _record_checkpoint_plans(
    report: dict[str, Any], plans: Sequence[CheckpointPlan]
) -> None:
    """Append deterministic before/after identities for every planned rewrite."""

    rows = report["would_change_checkpoints"]
    if not isinstance(rows, list):
        raise MigrationError("checkpoint migration report plan field is invalid")
    for plan in plans:
        rows.append(
            {
                "cell_id": plan.cell.cell_id,
                "manifest_index": plan.manifest_index,
                "qid": plan.question.qid,
                "checkpoint_relative_path": plan.relative_path,
                "before_sha256": plan.source_sha256,
                "after_sha256": plan.target_sha256,
                "before_size": len(plan.source_text.encode("utf-8")),
                "after_size": len(plan.target_text.encode("utf-8")),
                "after_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "evidence_id": plan.evidence["evidence_id"],
                "complete_preimage_embedded_in_incident": True,
            }
        )


def _write_and_verify_incident(
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    incident: dict[str, Any],
    *,
    target_code_version: str,
) -> None:
    path = run_root / INCIDENT_FILENAME
    io.write_json(path, incident)
    observed = _load_incident(
        run_root,
        snapshot,
        catalog,
        target_code_version=target_code_version,
    )
    if observed != incident:
        raise MigrationError(f"incident verification failed after atomic write: {path}")


def _apply_plans(
    *,
    run_root: Path,
    snapshot: ManifestSnapshot,
    catalog: VerifiedQuestionCatalog,
    target_code_version: str,
    incident: dict[str, Any],
    plans: Sequence[CheckpointPlan],
) -> int:
    changed_incident = False
    entries = incident["checkpoints"]
    for plan in plans:
        evidence_id = plan.evidence["evidence_id"]
        existing = entries.get(evidence_id)
        if existing is None:
            entries[evidence_id] = copy.deepcopy(plan.evidence)
            changed_incident = True
        elif existing != plan.evidence:
            raise MigrationError(f"incident evidence collision: {plan.path}")
    incident_path = run_root / INCIDENT_FILENAME
    if changed_incident or not incident_path.exists():
        _write_and_verify_incident(
            run_root,
            snapshot,
            catalog,
            incident,
            target_code_version=target_code_version,
        )
    else:
        observed = _load_incident(
            run_root,
            snapshot,
            catalog,
            target_code_version=target_code_version,
        )
        if observed != incident:
            raise MigrationError("incident changed before checkpoint replacement")

    migrated = 0
    for plan in plans:
        current_text, current_sha256 = _read_regular_utf8(plan.path)
        if current_text != plan.source_text or current_sha256 != plan.source_sha256:
            raise MigrationError(
                f"checkpoint changed while cell lock was held: {plan.path}"
            )
        io.atomic_write_text(plan.path, plan.target_text)
        _, observed_sha256 = _read_regular_utf8(plan.path)
        if observed_sha256 != plan.target_sha256:
            raise MigrationError(
                f"checkpoint replacement verification failed: {plan.path}"
            )
        migrated += 1
    return migrated


class _RunLock:
    def __init__(self, run_root: Path) -> None:
        self.path = run_root / RUN_LOCK_FILENAME
        self.fd: int | None = None

    def __enter__(self) -> None:
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise MigrationError(
                    f"another checkpoint migration owns {self.path}"
                ) from exc
            raise

    def __exit__(self, *exc_info: Any) -> None:
        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


def migrate_run(
    run_root: str | Path,
    *,
    apply: bool = False,
    benchmark_loader: BenchmarkLoader = load_benchmark,
    migration_timestamp: float | None = None,
    excluded_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Audit or atomically migrate one checksum-frozen run."""

    supplied_root = Path(run_root)
    if supplied_root.is_symlink():
        raise MigrationError(f"refusing symlinked run root: {supplied_root}")
    root = supplied_root.resolve()
    snapshot, catalog = _require_frozen_run(root, benchmark_loader=benchmark_loader)
    target_code_version = io.git_commit()
    if not isinstance(target_code_version, str) or not target_code_version:
        raise MigrationError("cannot determine target executable source identity")
    planned_migration_timestamp = (
        time.time() if migration_timestamp is None else migration_timestamp
    )
    _finite_nonnegative(
        planned_migration_timestamp, "planned migration timestamp"
    )
    planned_migration_timestamp = float(planned_migration_timestamp)
    incident = _load_incident(
        root,
        snapshot,
        catalog,
        target_code_version=target_code_version,
    )
    report = _new_report(
        root,
        snapshot,
        catalog,
        apply=apply,
        target_code_version=target_code_version,
        migration_timestamp=planned_migration_timestamp,
    )
    cells_root = root / "cells"
    excluded = set(excluded_cell_ids or ())
    unknown_exclusions = excluded - set(snapshot.ids)
    if unknown_exclusions:
        raise MigrationError(
            "excluded checkpoint cells are absent from the frozen manifest: "
            f"{sorted(unknown_exclusions)!r}"
        )
    report["explicitly_excluded_cells"] = sorted(excluded)

    def process(staging_root: Path) -> None:
        for manifest_index, cell in enumerate(snapshot.cells):
            if cell.cell_id in excluded:
                continue
            cell_directory = cells_root / cell.cell_id
            if not cell_directory.exists():
                continue
            if cell_directory.is_symlink() or not cell_directory.is_dir():
                report["unsafe_or_invalid_cells_skipped"] += 1
                report["errors"].append(f"unsafe cell directory: {cell_directory}")
                continue
            try:
                if is_cell_active(cell_directory):
                    report["active_or_locked_cells_skipped"] += 1
                    continue
            except OSError as exc:
                report["unsafe_or_invalid_cells_skipped"] += 1
                report["errors"].append(
                    f"{cell.cell_id}: cannot inspect cell lock: {exc}"
                )
                continue
            if not apply:
                try:
                    scan = _scan_cell(
                        staging_root,
                        run_root=root,
                        cell_directory=cell_directory,
                        cell=cell,
                        manifest_index=manifest_index,
                        catalog=catalog,
                        target_code_version=target_code_version,
                        migration_timestamp=planned_migration_timestamp,
                        incident=incident,
                    )
                except (OSError, MigrationError, BenchmarkContractError) as exc:
                    report["unsafe_or_invalid_cells_skipped"] += 1
                    report["errors"].append(f"{cell.cell_id}: {exc}")
                    continue
                report["schema1_candidates"] += len(scan.plans)
                report["schema2_native"] += scan.native_current
                report["schema2_already_migrated"] += scan.already_migrated
                _record_checkpoint_plans(report, scan.plans)
                continue

            try:
                with cell_lock(cell_directory, blocking=False):
                    _verify_run_unchanged(
                        root,
                        snapshot,
                        catalog,
                        target_code_version=target_code_version,
                    )
                    scan = _scan_cell(
                        staging_root,
                        run_root=root,
                        cell_directory=cell_directory,
                        cell=cell,
                        manifest_index=manifest_index,
                        catalog=catalog,
                        target_code_version=target_code_version,
                        migration_timestamp=planned_migration_timestamp,
                        incident=incident,
                    )
                    report["schema1_candidates"] += len(scan.plans)
                    report["schema2_native"] += scan.native_current
                    report["schema2_already_migrated"] += scan.already_migrated
                    _record_checkpoint_plans(report, scan.plans)
                    if scan.plans:
                        report["checkpoints_migrated"] += _apply_plans(
                            run_root=root,
                            snapshot=snapshot,
                            catalog=catalog,
                            target_code_version=target_code_version,
                            incident=incident,
                            plans=scan.plans,
                        )
            except CellLockUnavailable:
                report["active_or_locked_cells_skipped"] += 1
            except (OSError, MigrationError, BenchmarkContractError) as exc:
                report["unsafe_or_invalid_cells_skipped"] += 1
                report["errors"].append(f"{cell.cell_id}: {exc}")

    with tempfile.TemporaryDirectory(prefix="qid-checkpoint-migration-") as temp:
        staging_root = Path(temp)
        if apply:
            with _RunLock(root):
                process(staging_root)
        else:
            process(staging_root)

    incident_path = root / INCIDENT_FILENAME
    if incident_path.is_file() and not incident_path.is_symlink():
        report["incident_sha256"] = _bytes_sha256(incident_path.read_bytes())
    report["would_change_cells"] = sorted(
        {
            str(row["cell_id"])
            for row in report["would_change_checkpoints"]
        }
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the evidence-backed migration (default: read-only audit)",
    )
    parser.add_argument(
        "--migration-timestamp",
        type=float,
        help=(
            "bind dry-run and apply to one exact migration timestamp; reuse the "
            "planned_migration_timestamp reported by the dry-run"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    migration_timestamp = (
        time.time()
        if args.migration_timestamp is None
        else args.migration_timestamp
    )
    reports: list[dict[str, Any]] = []
    failed = False
    for run_id in args.run_id:
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise MigrationError(f"run id must be one path component: {run_id!r}")
        try:
            report = migrate_run(
                args.results_root / run_id,
                apply=args.apply,
                migration_timestamp=migration_timestamp,
            )
        except (OSError, MigrationError, BenchmarkContractError) as exc:
            report = {
                "run_root": str((args.results_root / run_id).resolve()),
                "applied": args.apply,
                "errors": [str(exc)],
            }
        failed |= bool(report["errors"])
        reports.append(report)
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
