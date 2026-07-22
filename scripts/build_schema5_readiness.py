#!/usr/bin/env python3
"""Build artifact-backed schema-5 readiness evidence, failing closed on drift.

The schema-5 controller intentionally accepts only tiny, typed gate envelopes.  This
tool is the corresponding evidence producer.  It never accepts operator-supplied
metrics: every value is derived from a sealed snapshot, the legacy consolidation
marker, an exact context audit, or live scheduler/registry/HTTP observations.

Each successful command writes its source/wrapper artifacts first and publishes the
schema-2 outer envelope last using a same-directory fsync + ``os.replace``.  The
candidate envelope is passed through the controller's own validator before it becomes
visible at ``--output``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import (  # noqa: E402
    VerifiedQuestionCatalog,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.serving.fleet_contract import (  # noqa: E402
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.launch_server import (  # noqa: E402
    _port_for,
    render_sbatch,
)
from agents_scaling.serving.model_contracts import (  # noqa: E402
    FrozenModelContracts,
    load_model_contracts,
)
from agents_scaling.serving.registry import (  # noqa: E402
    ServerEntry,
    entry_matches_frozen_provenance,
    server_pool_id,
)
from scripts.audit_context_capacity import AuditFilters, selected_cells  # noqa: E402
from slurm import keepalive  # noqa: E402
from slurm import schema5_control as control_plane  # noqa: E402


READINESS_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 1
ACTIVE_FLEET_STATE = "RUNNING"
EXPECTED_CONTEXT_MARGIN = 1_287
EXPECTED_DENSE_CELLS = 216
EXPECTED_DENSE_REQUESTS = 43_092
EXPECTED_SEVEN_CELLS = 288
EXPECTED_SEVEN_REQUESTS = 57_456


class EvidenceError(RuntimeError):
    """A readiness fact cannot be proved from its authoritative source."""


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_json(path: Path) -> dict[str, Any]:
    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise EvidenceError(f"expected a regular evidence file: {supplied}")
    path = supplied.resolve()
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"cannot parse evidence JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence must contain one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(name: str, path: Path) -> dict[str, str]:
    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise EvidenceError(f"artifact {name!r} is missing or a symlink: {supplied}")
    path = supplied.resolve()
    return {"name": name, "path": str(path), "sha256": _sha256(path)}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one JSON artifact and durably publish its directory entry."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_outer(
    *,
    control: Mapping[str, Any],
    gate: str,
    metrics: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, str]],
    output: Path,
    now: float | None = None,
) -> dict[str, Any]:
    """Validate a temporary candidate, then make the gate envelope visible last."""

    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "gate": gate,
        "passed": True,
        "immutable_sha256": control["immutable_sha256"],
        "metrics": dict(metrics),
        "artifacts": list(artifacts),
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{output.name}.candidate.", suffix=".json", dir=output.parent
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        candidate.chmod(0o444)
        digest = _sha256(candidate)
        control_plane._validate_attestation(  # type: ignore[attr-defined]
            control,
            gate,
            candidate,
            digest,
            now=time.time() if now is None else float(now),
        )
        os.replace(candidate, output)
        _fsync_directory(output.parent)
    finally:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    return payload | {"path": str(output), "sha256": _sha256(output)}


def _wrapper(
    *,
    name: str,
    immutable_sha256: str,
    metrics: Mapping[str, Any],
    references: Sequence[Mapping[str, str]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": name,
        "passed": True,
        "immutable_sha256": immutable_sha256,
        "metrics": dict(metrics),
        "referenced_artifacts": list(references),
    }
    if extra:
        payload.update(extra)
    return payload


def _artifact_path(output: Path, name: str) -> Path:
    return output.expanduser().resolve().parent / "artifacts" / f"{name}.json"


def _require_exact_metrics(
    observed: Any, expected: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if not isinstance(observed, dict) or observed != dict(expected):
        raise EvidenceError(
            f"{context} metrics drifted: expected {dict(expected)!r}, "
            f"observed {observed!r}"
        )
    return dict(observed)


def _verify_recursive_reference(reference: Mapping[str, Any], *, context: str) -> None:
    """Use the controller's recursive checksum verifier on one source graph."""

    try:
        control_plane._validate_referenced_artifact(  # type: ignore[attr-defined]
            reference,
            context=context,
            verified=set(),
            active=set(),
        )
    except control_plane.ReadinessError as exc:
        raise EvidenceError(str(exc)) from exc


def build_snapshot_gate(
    control: Mapping[str, Any],
    *,
    pre_repair_attestation: Path,
    legacy_consolidated_attestation: Path,
    output: Path,
) -> dict[str, Any]:
    sources = (
        ("pre_repair_external_attestation", pre_repair_attestation),
        ("legacy_consolidated_external_attestation", legacy_consolidated_attestation),
    )
    artifacts: list[dict[str, str]] = []
    roots: set[str] = set()
    snapshot_ids: set[str] = set()
    for name, path in sources:
        payload = _read_json(path)
        root_value = payload.get("snapshot_root")
        if not isinstance(root_value, str) or not Path(root_value).is_absolute():
            raise EvidenceError(f"snapshot attestation root is not absolute: {path}")
        roots.add(str(Path(root_value).resolve()))
        snapshot_ids.add(str(payload.get("snapshot_id")))
        artifacts.append(_artifact(name, path))
    if len(roots) != 2 or len(snapshot_ids) != 2:
        raise EvidenceError("pre-repair and consolidated snapshots must have distinct identities")
    # Publication invokes the controller's recursive validator exactly once.  That
    # validator rehashes every inventory payload and derives ID/count/bytes from the
    # sealed controls before the candidate envelope is renamed into place.
    return _publish_outer(
        control=control,
        gate="snapshot",
        metrics={
            "snapshot_count": 2,
            "pre_repair_verified": True,
            "legacy_consolidated_verified": True,
        },
        artifacts=artifacts,
        output=output,
    )


LEGACY_ARTIFACTS = {
    "response_incident_archive_report": "consolidated_response_report",
    "checkpoint_migration_report": "consolidated_checkpoint_report",
    "permanent_ledger_archive_report": "consolidated_permanent_report",
    "legacy_semantic_audit_report": "consolidated_semantic_report",
}

MIGRATION_COMPONENT_METRICS = {
    "response_incident_archive_report": {
        "protocol_incidents_total": 22,
        "protocol_already_reset": 22,
        "sealed_incident_qids": 1_064,
    },
    "checkpoint_migration_report": {
        "migrated_checkpoints": 47,
        "remaining_schema1_checkpoints": 0,
        "coordinates_preserved": True,
    },
    "permanent_ledger_archive_report": {
        "permanent_ledgers_archived": 3,
        "unresolved_permanent_ledgers": 0,
    },
}

MIGRATION_OUTER_METRICS = {
    "protocol_incidents_total": 22,
    "protocol_already_reset": 22,
    "sealed_incident_qids": 1_064,
    "migrated_checkpoints": 47,
    "remaining_schema1_checkpoints": 0,
    "permanent_ledgers_archived": 3,
    "unresolved_permanent_ledgers": 0,
}

SEMANTIC_METRICS = {
    "complete_cells": 740,
    "active_validated_qids": 888_068,
    "corrupt_cells": 0,
    "permanent_cells": 0,
    "malformed_lines": 0,
    "duplicate_qids": 0,
    "unexpected_qids": 0,
    "repair_count": 0,
}


def _load_cleanup_sources(
    cleanup_marker: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    marker_path = cleanup_marker.expanduser().resolve()
    marker = _read_json(marker_path)
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "complete"
        or marker.get("passed") is not True
        or not isinstance(marker.get("snapshot_id"), str)
        or not marker["snapshot_id"]
    ):
        raise EvidenceError("legacy cleanup marker is not a completed schema-1 marker")
    records = marker.get("artifacts")
    if not isinstance(records, list):
        raise EvidenceError("legacy cleanup marker artifacts must be an array")
    names = [row.get("name") if isinstance(row, dict) else None for row in records]
    if len(names) != len(set(names)) or set(names) != set(LEGACY_ARTIFACTS):
        raise EvidenceError(
            f"legacy cleanup marker must bind exactly {sorted(LEGACY_ARTIFACTS)}"
        )
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in records:
        assert isinstance(row, dict)
        if set(row) != {"name", "path", "sha256"}:
            raise EvidenceError("legacy cleanup artifact reference has wrong fields")
        _verify_recursive_reference(row, context=f"legacy cleanup {row['name']}")
        supplied_path = Path(str(row["path"])).expanduser()
        loaded[str(row["name"])] = (supplied_path.resolve(), _read_json(supplied_path))
    return marker, loaded


def build_migrations_gate(
    control: Mapping[str, Any], *, cleanup_marker: Path, output: Path
) -> dict[str, Any]:
    marker, sources = _load_cleanup_sources(cleanup_marker)
    _require_exact_metrics(
        marker.get("migration_metrics"),
        MIGRATION_OUTER_METRICS,
        context="legacy cleanup marker migration",
    )
    artifacts: list[dict[str, str]] = []
    for name, expected in MIGRATION_COMPONENT_METRICS.items():
        raw_path, raw = sources[name]
        if raw.get("schema_version") != 1 or raw.get("passed") is not True:
            raise EvidenceError(f"legacy consolidation source did not pass: {raw_path}")
        observed = {key: raw.get(key) for key in expected}
        _require_exact_metrics(observed, expected, context=name)
        wrapper_path = _artifact_path(output, name)
        _write_json_atomic(
            wrapper_path,
            _wrapper(
                name=name,
                immutable_sha256=str(control["immutable_sha256"]),
                metrics=expected,
                references=[_artifact(LEGACY_ARTIFACTS[name], raw_path)],
            ),
        )
        artifacts.append(_artifact(name, wrapper_path))
    return _publish_outer(
        control=control,
        gate="migrations",
        metrics=MIGRATION_OUTER_METRICS,
        artifacts=artifacts,
        output=output,
    )


def build_semantic_gate(
    control: Mapping[str, Any], *, cleanup_marker: Path, output: Path
) -> dict[str, Any]:
    marker, sources = _load_cleanup_sources(cleanup_marker)
    _require_exact_metrics(
        marker.get("semantic_metrics"), SEMANTIC_METRICS, context="cleanup semantic"
    )
    raw_path, raw = sources["legacy_semantic_audit_report"]
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "legacy_semantic_audit"
        or raw.get("passed") is not True
        or raw.get("manifest_cells") != 22_680
        or raw.get("invalid_rows") != 0
    ):
        raise EvidenceError("legacy semantic source identity/acceptance drifted")
    _require_exact_metrics(raw.get("metrics"), SEMANTIC_METRICS, context="semantic source")
    name = "legacy_semantic_audit_report"
    wrapper_path = _artifact_path(output, name)
    _write_json_atomic(
        wrapper_path,
        _wrapper(
            name=name,
            immutable_sha256=str(control["immutable_sha256"]),
            metrics=SEMANTIC_METRICS,
            references=[_artifact(LEGACY_ARTIFACTS[name], raw_path)],
        ),
    )
    return _publish_outer(
        control=control,
        gate="semantic_audit",
        metrics=SEMANTIC_METRICS,
        artifacts=[_artifact(name, wrapper_path)],
        output=output,
    )


def _run(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=False, timeout=timeout
    )


def _fleet_scheduler_rows(
    *, runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] | None = None
) -> tuple[keepalive.FleetQueueRow, ...]:
    if runner is None:
        proc = _run(
            [
                "squeue",
                "-u",
                os.environ.get("USER", ""),
                "-h",
                "-o",
                "%i|%j|%T|%P|%N|%o|%k",
            ],
            timeout=15.0,
        )
    else:
        proc = runner(
            [
                "squeue",
                "-u",
                os.environ.get("USER", ""),
                "-h",
                "-o",
                "%i|%j|%T|%P|%N|%o|%k",
            ],
            15.0,
        )
    if proc.returncode != 0:
        raise EvidenceError(f"squeue fleet query failed: {proc.stderr.strip()[:500]}")
    rows: list[keepalive.FleetQueueRow] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 6)
        if len(fields) != 7:
            raise EvidenceError(f"malformed fleet scheduler row: {line!r}")
        row = keepalive.FleetQueueRow(*(field.strip() for field in fields))
        if not row.job_id.isdigit():
            raise EvidenceError(f"fleet scheduler job ID is not exact numeric: {row.job_id!r}")
        rows.append(row)
    return tuple(rows)


def _registry_records(pool_root: Path) -> dict[str, tuple[ServerEntry, Path]]:
    servers = pool_root / "servers"
    if not servers.is_dir() or servers.is_symlink():
        raise EvidenceError(f"canonical server registry is missing: {servers}")
    records: dict[str, tuple[ServerEntry, Path]] = {}
    for path in sorted(servers.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"unsafe server registry artifact: {path}")
        payload = _read_json(path)
        try:
            entry = ServerEntry(**payload)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"malformed registry record {path}: {exc}") from exc
        replica_id = entry.replica_id
        if not isinstance(replica_id, str) or not replica_id:
            raise EvidenceError(f"registry record lacks exact replica identity: {path}")
        if replica_id in records:
            raise EvidenceError(f"duplicate registry records for replica {replica_id}")
        records[replica_id] = (entry, path.resolve())
    return records


def _http_probe(entry: ServerEntry, expected_model: str, timeout: float) -> dict[str, Any]:
    started = time.time()

    def fetch(path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(  # noqa: S310 - frozen in-cluster endpoint
                f"http://{entry.host}:{entry.port}{path}", timeout=timeout
            ) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), b""
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return 0, b""

    health_status, _ = fetch("/health")
    models_status, body = fetch("/v1/models")
    model_ids: list[str] = []
    if models_status == 200:
        try:
            payload = json.loads(body.decode("utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                model_ids = sorted(
                    str(row["id"])
                    for row in data
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                )
        except (UnicodeError, json.JSONDecodeError):
            model_ids = []
    completed = time.time()
    return {
        "health_status": health_status,
        "models_status": models_status,
        "model_ids": model_ids,
        "expected_model": expected_model,
        "probe_started_timestamp": started,
        "probe_completed_timestamp": completed,
        "healthy": (
            health_status == 200
            and models_status == 200
            and expected_model in model_ids
        ),
    }


def _launch_options(pins: Mapping[str, Any]) -> dict[str, str]:
    return {
        "release_worktree": str(pins["release_worktree"]),
        "release_id": str(pins["release_id"]),
        "environment_hash": str(pins["serving_environment_sha256"]),
        "model_contract_path": str(pins["model_contract_path"]),
        "model_contract_sha256": str(pins["model_contract_sha256"]),
        "harness_environment_prefix": str(pins["harness_environment_prefix"]),
        "serving_environment_prefix": str(pins["serving_environment_prefix"]),
        "harness_environment_manifest_path": str(
            pins["harness_environment_manifest_path"]
        ),
        "serving_environment_manifest_path": str(
            pins["serving_environment_manifest_path"]
        ),
        "harness_environment_hash": str(pins["harness_environment_sha256"]),
        "fleet_contract_path": str(pins["fleet_contract_path"]),
        "fleet_contract_sha256": str(pins["fleet_contract_sha256"]),
        "hf_home": str(pins["hf_home"]),
    }


def _verify_scheduler_and_spool(
    *,
    pool_root: Path,
    fleet: FrozenFleetContract,
    pins: Mapping[str, Any],
    rows: Iterable[keepalive.FleetQueueRow],
) -> dict[str, tuple[keepalive.FleetQueueRow, keepalive.SpooledServingProvenance]]:
    expected_by_name = {replica.scheduler_job_name: replica for replica in fleet.replicas}
    production_rows = [row for row in rows if row.job_name.startswith("asys-s5-serve-")]
    unknown = sorted({row.job_name for row in production_rows} - set(expected_by_name))
    if unknown:
        raise EvidenceError(f"unmappable schema-5 serving jobs: {unknown}")
    grouped: dict[str, list[keepalive.FleetQueueRow]] = {}
    for row in production_rows:
        grouped.setdefault(row.job_name, []).append(row)
    verified: dict[
        str, tuple[keepalive.FleetQueueRow, keepalive.SpooledServingProvenance]
    ] = {}
    options = _launch_options(pins)
    for replica in fleet.replicas:
        matches = grouped.get(replica.scheduler_job_name, [])
        if len(matches) != 1:
            raise EvidenceError(
                f"replica {replica.replica_id} has {len(matches)} scheduler allocations"
            )
        row = matches[0]
        expected_comment = (
            f"asys-schema5-pool:{replica.pool_id};"
            f"profile={replica.serving_profile};replica={replica.replica_id}"
        )
        if (
            row.state.upper() != ACTIVE_FLEET_STATE
            or row.partition != replica.partition
            or row.node in {"", "(null)", "N/A"}
            or row.comment != expected_comment
        ):
            raise EvidenceError(
                f"scheduler allocation drift for {replica.replica_id}: {asdict(row)}"
            )
        expected_script = render_sbatch(
            replica.model_size,
            str(pool_root),
            replica.partition,
            replica.gpu_type,
            replica.time_limit,
            str(pool_root / "logs"),
            replica=replica.replica_index,
            serving_profile=replica.serving_profile,
            **options,
        )
        provenance = keepalive._spooled_job_provenance(
            row.job_id,
            replica.serving_profile,
            run_root=str(pool_root),
            expected_script=expected_script,
        )
        if (
            provenance is None
            or provenance.run_root != str(pool_root)
            or provenance.server_pool_id != replica.pool_id
            or provenance.replica_id != replica.replica_id
            or provenance.replica_index != replica.replica_index
            or provenance.release_id != pins["release_id"]
            or provenance.environment_hash != pins["serving_environment_sha256"]
            or provenance.model_contract_sha256 != pins["model_contract_sha256"]
            or provenance.fleet_contract_sha256 != pins["fleet_contract_sha256"]
        ):
            raise EvidenceError(
                f"spooled immutable provenance failed for {replica.replica_id}"
            )
        verified[replica.replica_id] = (row, provenance)
    return verified


def _verify_registry(
    *,
    pool_root: Path,
    fleet: FrozenFleetContract,
    models: FrozenModelContracts,
    pins: Mapping[str, Any],
    scheduler: Mapping[
        str, tuple[keepalive.FleetQueueRow, keepalive.SpooledServingProvenance]
    ],
) -> dict[str, tuple[ServerEntry, Path]]:
    records = _registry_records(pool_root)
    expected_ids = {replica.replica_id for replica in fleet.replicas}
    if set(records) != expected_ids:
        raise EvidenceError(
            "registry replica set drifted: "
            f"missing={sorted(expected_ids - set(records))}, "
            f"unexpected={sorted(set(records) - expected_ids)}"
        )
    for replica in fleet.replicas:
        entry, path = records[replica.replica_id]
        row, provenance = scheduler[replica.replica_id]
        identity = models.for_size(replica.model_size)
        if not entry_matches_frozen_provenance(
            entry,
            replica.serving_profile,
            release_id=str(pins["release_id"]),
            environment_hash=str(pins["serving_environment_sha256"]),
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
            model_contract_sha256=str(pins["model_contract_sha256"]),
            fleet_contract_sha256=str(pins["fleet_contract_sha256"]),
            expected_server_pool_id=fleet.fleet_id,
        ):
            raise EvidenceError(f"frozen registry provenance drift for {path}")
        expected_path = (
            pool_root
            / "servers"
            / replica.serving_profile
            / f"{entry.host}_{entry.port}.json"
        ).resolve()
        if (
            entry.replica_index != replica.replica_index
            or entry.model_size != replica.model_size
            or entry.slurm_job_id != row.job_id
            or entry.host != row.node
            or entry.port != _port_for(replica.serving_profile, replica.replica_index)
            or path != expected_path
            or entry.model_revision != provenance.model_revision
            or entry.tokenizer_id != provenance.tokenizer_id
            or entry.tokenizer_revision != provenance.tokenizer_revision
        ):
            raise EvidenceError(f"registry/scheduler/spool mismatch for {replica.replica_id}")
        if (
            not isinstance(entry.started_at, (int, float))
            or isinstance(entry.started_at, bool)
            or not math.isfinite(float(entry.started_at))
            or float(entry.started_at) <= 0
        ):
            raise EvidenceError(f"invalid registration timestamp for {replica.replica_id}")
    return records


def build_fleet_gate(
    control: Mapping[str, Any],
    *,
    output: Path,
    probe_timeout: float = 5.0,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None = None,
    probe: Callable[[ServerEntry, str, float], Mapping[str, Any]] = _http_probe,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if not 0 < probe_timeout <= 60:
        raise EvidenceError("probe timeout must be in (0, 60] seconds")
    pins = control["immutable"]
    pool_root = Path(str(pins["server_pool_root"])).expanduser().resolve()
    if server_pool_id(pool_root) != "schema5-v1":
        raise EvidenceError(f"server pool root has the wrong identity: {pool_root}")
    models = load_model_contracts(
        pins["model_contract_path"], expected_sha256=pins["model_contract_sha256"]
    )
    fleet = load_fleet_contract(
        pins["fleet_contract_path"],
        model_contracts=models,
        expected_sha256=pins["fleet_contract_sha256"],
    )
    fleet.verify_pool_root(pool_root)
    rows = _fleet_scheduler_rows(runner=scheduler_runner)
    scheduler = _verify_scheduler_and_spool(
        pool_root=pool_root, fleet=fleet, pins=pins, rows=rows
    )
    records = _verify_registry(
        pool_root=pool_root,
        fleet=fleet,
        models=models,
        pins=pins,
        scheduler=scheduler,
    )

    def probe_one(replica_id: str) -> tuple[str, Mapping[str, Any]]:
        entry, _ = records[replica_id]
        replica = next(item for item in fleet.replicas if item.replica_id == replica_id)
        profile = fleet.by_profile[replica.serving_profile]
        del profile  # profile existence was frozen by the fleet loader
        from agents_scaling.serving.profiles import get_serving_profile

        expected_model = get_serving_profile(replica.serving_profile).served_model_name
        return replica_id, probe(entry, expected_model, probe_timeout)

    with ThreadPoolExecutor(max_workers=len(fleet.replicas)) as executor:
        probes = dict(executor.map(probe_one, sorted(records)))
    captured = float(now())
    unhealthy = sorted(
        replica_id
        for replica_id, result in probes.items()
        if result.get("healthy") is not True
    )
    if unhealthy:
        raise EvidenceError(f"fleet HTTP health failed for replicas: {unhealthy}")
    completed = [result.get("probe_completed_timestamp") for result in probes.values()]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) <= captured
        for value in completed
    ):
        raise EvidenceError("fleet probes have invalid completion timestamps")
    max_probe_age = max(captured - float(value) for value in completed)
    if not 0 <= max_probe_age <= 600:
        raise EvidenceError("fleet live-probe evidence is already stale")

    counts = Counter(replica.serving_profile for replica in fleet.replicas)
    profile_counts = {name: counts[name] for name in control_plane.EXPECTED_FLEET_PROFILES}
    metrics = {
        "logical_replicas": len(fleet.replicas),
        "allocated_gpus": sum(replica.gpus_per_replica for replica in fleet.replicas),
        "healthy_replicas": len(probes),
        "unhealthy_replicas": 0,
        "profile_replicas": profile_counts,
        "revision_mismatches": 0,
        "missing_profiles": 0,
        "stale_registrations": 0,
        "max_heartbeat_age_seconds": max_probe_age,
        "captured_timestamp": captured,
        "model_contract_sha256": models.sha256,
        "fleet_contract_sha256": fleet.sha256,
    }
    detail = []
    registration_refs: list[dict[str, str]] = []
    for replica in fleet.replicas:
        row, provenance = scheduler[replica.replica_id]
        entry, registry_path = records[replica.replica_id]
        detail.append(
            {
                "replica_id": replica.replica_id,
                "serving_profile": replica.serving_profile,
                "slurm_job_id": row.job_id,
                "partition": row.partition,
                "node": row.node,
                "host": entry.host,
                "port": entry.port,
                "spooled_provenance": asdict(provenance),
                "http": dict(probes[replica.replica_id]),
                "registry_path": str(registry_path),
                "registry_sha256": _sha256(registry_path),
            }
        )
        registration_refs.append(
            _artifact(f"registry_{replica.replica_id}", registry_path)
        )
    name = "fleet_health_report"
    health_path = _artifact_path(output, name)
    raw_name = "raw_fleet_health_probe"
    raw_path = output.expanduser().resolve().parent / "artifacts" / f"{raw_name}.json"
    _write_json_atomic(
        raw_path,
        {
            "schema_version": 1,
            "kind": raw_name,
            "passed": True,
            "immutable_sha256": control["immutable_sha256"],
            "server_pool_root": str(pool_root),
            "captured_timestamp": captured,
            "replicas": detail,
            "referenced_artifacts": registration_refs,
        },
    )
    wrapper = _wrapper(
        name=name,
        immutable_sha256=str(control["immutable_sha256"]),
        metrics=metrics,
        references=[_artifact(raw_name, raw_path)],
        extra={"server_pool_root": str(pool_root)},
    )
    _write_json_atomic(health_path, wrapper)
    return _publish_outer(
        control=control,
        gate="fleet",
        metrics=metrics,
        artifacts=[
            _artifact(name, health_path),
            _artifact("fleet_contract", Path(str(pins["fleet_contract_path"]))),
        ],
        output=output,
        now=captured,
    )


CONTEXT_SPECS: dict[str, dict[str, Any]] = {
    "dense_peer_context_audit": {
        "source_name": "raw_dense_peer_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["b2048", "b8192", "unlimited"],
            "prompt_levels": [3],
            "topologies": ["decentralized"],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": EXPECTED_DENSE_CELLS,
        "audited_requests": EXPECTED_DENSE_REQUESTS,
    },
    "seven_agent_context_audit": {
        "source_name": "raw_seven_agent_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["unlimited"],
            "prompt_levels": [],
            "topologies": [],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": EXPECTED_SEVEN_CELLS,
        "audited_requests": EXPECTED_SEVEN_REQUESTS,
    },
}


def _run_pin(control: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    matches = [run for run in control["immutable"]["runs"] if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise EvidenceError(f"immutable pins do not uniquely define run {run_id}")
    return matches[0]


def _validate_context_source(
    control: Mapping[str, Any],
    *,
    name: str,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = CONTEXT_SPECS[name]
    report = _read_json(path)
    run_pin = _run_pin(control, spec["run_id"])
    if (
        report.get("schema_version") != 3
        or report.get("audit") != "all_routed_profiles_context_capacity"
        or report.get("run_id") != spec["run_id"]
        or report.get("all_routed_profiles") is not True
        or report.get("filters") != spec["filters"]
    ):
        raise EvidenceError(f"context audit identity/filters drifted: {path}")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or manifest != {
        "path": str(Path(str(run_pin["manifest_path"])).resolve()),
        "sha256": run_pin["manifest_sha256"],
        "cells": run_pin["cell_count"],
    }:
        raise EvidenceError(f"context audit manifest identity drifted: {path}")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise EvidenceError(f"context audit summary is missing: {path}")
    expected_summary = {
        "passed": True,
        "selected_cells": spec["selected_cells"],
        "audited_requests": spec["audited_requests"],
        "failed_requests": 0,
        "failed_cells": 0,
        "minimum_context_headroom_tokens": EXPECTED_CONTEXT_MARGIN,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise EvidenceError(
                f"context audit {name} {field} drifted: "
                f"expected {expected!r}, observed {summary.get(field)!r}"
            )
    if report.get("failure_groups") != [] or report.get("failure_examples") != []:
        raise EvidenceError(f"context audit {name} retains failure evidence")

    snapshot = load_manifest(Path(str(run_pin["run_root"])), verify_frozen=True)
    filters = AuditFilters(
        n_agents=frozenset(spec["filters"]["n_agents"]),
        reasoning=frozenset(spec["filters"]["reasoning"]),
        prompt_levels=frozenset(spec["filters"]["prompt_levels"]),
        topologies=frozenset(spec["filters"]["topologies"]),
        context_levels=frozenset(spec["filters"]["context_levels"]),
    )
    selected = selected_cells(snapshot, filters, all_routed_profiles=True)
    if len(selected) != spec["selected_cells"]:
        raise EvidenceError(f"manifest-derived context selection drifted for {name}")
    expected_ids = {cell.cell_id for cell in selected}
    rows = report.get("cells")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise EvidenceError(f"context audit cell records are incomplete for {name}")
    observed_ids = [row.get("cell_id") if isinstance(row, dict) else None for row in rows]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise EvidenceError(f"context audit cell identities drifted for {name}")

    catalog = VerifiedQuestionCatalog(Path(str(run_pin["run_root"])), snapshot=snapshot)
    selected_by_id = {cell.cell_id: cell for cell in selected}
    request_total = 0
    margins: list[int] = []
    request_truncations = 0
    for row in rows:
        assert isinstance(row, dict)
        cell = selected_by_id[str(row["cell_id"])]
        expected_requests = len(catalog.questions_for(cell))
        margin = row.get("minimum_context_headroom_tokens")
        if (
            row.get("config_hash") != cell.config_hash()
            or row.get("audited_requests") != expected_requests
            or row.get("failed_requests") != 0
            or row.get("fits") is not True
            or not isinstance(margin, int)
            or isinstance(margin, bool)
            or margin <= 0
        ):
            raise EvidenceError(f"invalid context cell evidence for {cell.cell_id}")
        minimum_request = row.get("minimum_headroom_request")
        if (
            not isinstance(minimum_request, dict)
            or minimum_request.get("context_preflight_fits") is not True
            or minimum_request.get("output_capacity_headroom_tokens") != margin
        ):
            raise EvidenceError(f"context request evidence drifted for {cell.cell_id}")
        if minimum_request.get("context_preflight_fits") is not True:
            request_truncations += 1
        request_total += expected_requests
        margins.append(margin)
    if (
        request_total != spec["audited_requests"]
        or min(margins) != EXPECTED_CONTEXT_MARGIN
        or request_truncations != 0
    ):
        raise EvidenceError(f"context audit derived aggregate drifted for {name}")
    wrapper_metrics = {
        "selected_cells": spec["selected_cells"],
        "audited_requests": spec["audited_requests"],
        "failed_requests": 0,
        "failed_cells": 0,
        "minimum_context_headroom_tokens": EXPECTED_CONTEXT_MARGIN,
        "truncation_incidents": 0,
    }
    return report, wrapper_metrics


def build_context_gate(
    control: Mapping[str, Any],
    *,
    dense_peer_audit: Path,
    seven_agent_audit: Path,
    output: Path,
) -> dict[str, Any]:
    inputs = {
        "dense_peer_context_audit": dense_peer_audit,
        "seven_agent_context_audit": seven_agent_audit,
    }
    artifacts: list[dict[str, str]] = []
    total_failures = total_truncations = 0
    margins: list[int] = []
    for name, path in inputs.items():
        _, metrics = _validate_context_source(control, name=name, path=path)
        spec = CONTEXT_SPECS[name]
        wrapper_path = _artifact_path(output, name)
        _write_json_atomic(
            wrapper_path,
            _wrapper(
                name=name,
                immutable_sha256=str(control["immutable_sha256"]),
                metrics=metrics,
                references=[_artifact(str(spec["source_name"]), path)],
                extra={
                    "run_id": spec["run_id"],
                    "filters": spec["filters"],
                    "all_routed_profiles": True,
                },
            ),
        )
        artifacts.append(_artifact(name, wrapper_path))
        total_failures += int(metrics["failed_requests"])
        total_truncations += int(metrics["truncation_incidents"])
        margins.append(int(metrics["minimum_context_headroom_tokens"]))
    outer_metrics = {
        "dense_peer_requests": EXPECTED_DENSE_REQUESTS,
        "seven_agent_requests": EXPECTED_SEVEN_REQUESTS,
        "failed_preflights": total_failures,
        "truncation_incidents": total_truncations,
        "minimum_context_margin_tokens": min(margins),
    }
    return _publish_outer(
        control=control,
        gate="context_audit",
        metrics=outer_metrics,
        artifacts=artifacts,
        output=output,
    )


def build_email_gate(
    control: Mapping[str, Any],
    *,
    output: Path,
    apply: bool,
    mail_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if not apply:
        raise EvidenceError("email readiness requires --apply; no synthetic receipt is valid")
    recipient = control.get("alert_email")
    if not isinstance(recipient, str) or not recipient:
        raise EvidenceError("control alert_email is missing")
    submitted = float(now())
    subject = "[agents-scaling] schema-5 readiness delivery test"
    body = (
        "This is the required live email-path readiness test for immutable schema-5 "
        f"control {control['immutable_sha256']}.\n"
    )
    try:
        proc = mail_runner(
            ["mail", "-s", subject, recipient],
            input=body,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"mail submission failed: {exc}") from exc
    receipt_metrics = {
        "recipient": recipient,
        "delivery_succeeded": proc.returncode == 0,
        "returncode": int(proc.returncode),
    }
    if proc.returncode != 0:
        raise EvidenceError(
            f"mail command rejected readiness message rc={proc.returncode}: "
            f"{proc.stderr.strip()[:500]}"
        )
    raw_receipt = output.expanduser().resolve().parent / "artifacts" / "mail_submission_receipt.json"
    _write_json_atomic(
        raw_receipt,
        {
            "schema_version": 1,
            "kind": "mail_submission_receipt",
            "passed": True,
            "immutable_sha256": control["immutable_sha256"],
            "metrics": receipt_metrics,
            "submitted_timestamp": submitted,
            "subject": subject,
            "message_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "referenced_artifacts": [],
        },
    )
    name = "email_delivery_receipt"
    wrapper_path = _artifact_path(output, name)
    _write_json_atomic(
        wrapper_path,
        _wrapper(
            name=name,
            immutable_sha256=str(control["immutable_sha256"]),
            metrics=receipt_metrics,
            references=[_artifact("mail_submission_receipt", raw_receipt)],
        ),
    )
    return _publish_outer(
        control=control,
        gate="email_test",
        metrics=receipt_metrics,
        artifacts=[_artifact(name, wrapper_path)],
        output=output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="gate", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--pre-repair-attestation", required=True, type=Path)
    snapshot.add_argument("--legacy-consolidated-attestation", required=True, type=Path)

    for name in ("migrations", "semantic-audit"):
        command = subparsers.add_parser(name)
        command.add_argument("--cleanup-marker", required=True, type=Path)

    fleet = subparsers.add_parser("fleet")
    fleet.add_argument("--probe-timeout", type=float, default=5.0)

    context = subparsers.add_parser("context-audit")
    context.add_argument("--dense-peer-audit", required=True, type=Path)
    context.add_argument("--seven-agent-audit", required=True, type=Path)

    email = subparsers.add_parser("email-test")
    email.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        current = control_plane.load_control(
            args.state_dir.expanduser().resolve(), verify_files=True
        )
        if args.gate == "snapshot":
            report = build_snapshot_gate(
                current,
                pre_repair_attestation=args.pre_repair_attestation,
                legacy_consolidated_attestation=args.legacy_consolidated_attestation,
                output=args.output,
            )
        elif args.gate == "migrations":
            report = build_migrations_gate(
                current, cleanup_marker=args.cleanup_marker, output=args.output
            )
        elif args.gate == "semantic-audit":
            report = build_semantic_gate(
                current, cleanup_marker=args.cleanup_marker, output=args.output
            )
        elif args.gate == "fleet":
            report = build_fleet_gate(
                current, output=args.output, probe_timeout=args.probe_timeout
            )
        elif args.gate == "context-audit":
            report = build_context_gate(
                current,
                dense_peer_audit=args.dense_peer_audit,
                seven_agent_audit=args.seven_agent_audit,
                output=args.output,
            )
        elif args.gate == "email-test":
            report = build_email_gate(current, output=args.output, apply=args.apply)
        else:  # pragma: no cover - argparse closes the command set
            raise AssertionError(args.gate)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "type": type(exc).__name__, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
