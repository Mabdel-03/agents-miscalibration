#!/usr/bin/env python
"""Archive and reset cells whose sampled responses were discarded by protocol v4.

``ServerResponseProtocolError`` used to escape the runner after a model response had
already been sampled.  An ordinary retry could then accept a different response and
condition the dataset on parser success.  Those cells must be rerun from an empty cell
state under artifact schema 5, but none of their old evidence may be destroyed.

This utility discovers affected cells from two independent sources:

* a manifested cell's exact ``failure.json`` exception type; and
* dispatcher traceback logs joined to immutable batch tasks through the dispatcher
  ledger and batch manifest.

The second source also recovers cells whose old failure ledger was cleared by a later
successful QID.  Every join is checked against the checksum-frozen run manifest.  The
tool fails closed on an unmappable protocol traceback, task/config drift, symlinks, or
source bytes that change between discovery and the locked transaction.

Dry-run is the default.  ``--apply`` acquires the runner's nonblocking per-cell lock,
atomically publishes a deterministic incident archive, removes only the archived active
cell artifacts, and writes a reset-complete marker before releasing the lock.  A crash
after archive publication is recoverable and idempotent.  Once the marker exists, later
schema-5 rerun artifacts are never touched by this incident.  A new failure ledger or
dispatcher traceback after that marker is reported as a separate operator incident; it
is never hidden as an idempotent old reset and never folded into the sealed archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Support direct ``python scripts/...`` use from a checkout.
REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT, ExperimentCell  # noqa: E402
from agents_scaling.experiment import io  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    FAILURE_FILENAME,
    META_FILENAME,
    RESULTS_FILENAME,
    CellLockUnavailable,
    cell_lock,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest  # noqa: E402
from agents_scaling.experiment.qid_checkpoint import CHECKPOINT_DIRECTORY  # noqa: E402
from agents_scaling.experiment.result_schema import ARTIFACT_SCHEMA_VERSION  # noqa: E402


INCIDENT_SCHEMA_VERSION = 1
RESET_MARKER_SCHEMA_VERSION = 1
DISPATCHER_LEDGER_SCHEMA_VERSION = 1
TARGET_ARTIFACT_SCHEMA_VERSION = 5
INCIDENT_KIND = "discarded_server_response_protocol_v4"
INCIDENT_DIRECTORY = Path("incidents") / INCIDENT_KIND
INCIDENT_FILENAME = "incident.json"
RESET_MARKER_FILENAME = "reset_complete.json"
PROTOCOL_ERROR_TYPE = "ServerResponseProtocolError"
TRACEBACK_MARKER = b"agents_scaling.serving.client.ServerResponseProtocolError:"
LOG_PATTERN = re.compile(r"dispatch_(?P<job>[0-9]+)_(?P<task>[0-9]+)\.out")
ACTIVE_ARTIFACT_FILENAMES = (RESULTS_FILENAME, META_FILENAME, FAILURE_FILENAME)


class IncidentError(RuntimeError):
    """The utility cannot prove that an archive/reset operation is safe."""


class NewProtocolIncidentRequired(IncidentError):
    """A post-reset response-protocol failure needs a distinct operator incident."""


class _InvalidJSON(ValueError):
    pass


@dataclass(frozen=True)
class DispatcherEvidence:
    state_dir: Path
    ledger_path: Path
    ledger_sha256: str
    job_id: str
    task_index: int
    task: dict[str, Any]
    job_mapping: dict[str, Any]
    log_path: Path
    log_sha256: str
    log_size: int
    batch_path: Path
    batch_sha256: str
    batch_size: int

    @property
    def run_id(self) -> str:
        return str(self.task["run_id"])

    @property
    def cell_id(self) -> str:
        return str(self.task["cell_id"])

    @property
    def evidence_id(self) -> str:
        return _sha256_bytes(
            "\0".join(
                (
                    str(self.state_dir),
                    self.job_id,
                    str(self.task_index),
                    self.log_sha256,
                    self.batch_sha256,
                )
            ).encode("utf-8")
        )


@dataclass(frozen=True)
class FailureEvidence:
    path: Path
    sha256: str
    size: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    cell: ExperimentCell
    manifest_index: int
    failure: FailureEvidence | None
    dispatcher: tuple[DispatcherEvidence, ...]

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id


@dataclass(frozen=True)
class ArtifactSnapshot:
    source: Path
    source_relative_path: str
    archived_relative_path: str
    payload: bytes
    sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(text: str, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise _InvalidJSON(f"non-finite JSON number {value!r}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _InvalidJSON(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeError, _InvalidJSON) as exc:
        raise IncidentError(f"cannot parse {label}: {exc}") from exc


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise IncidentError(f"refusing non-regular {label}: {path}")
    return path.read_bytes()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_bytes(path, label=label)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IncidentError(f"{label} is not UTF-8: {path}: {exc}") from exc
    value = _strict_json_loads(text, label=f"{label} {path}")
    if not isinstance(value, dict):
        raise IncidentError(f"{label} must contain one JSON object: {path}")
    return value, payload


def _require_safe_component(value: str, *, label: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise IncidentError(f"unsafe {label}: {value!r}")


def _require_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise IncidentError(f"missing or unsafe {label}: {path}")
    return path.resolve()


def _require_frozen_manifest(run_root: Path) -> ManifestSnapshot:
    checksum = run_root / "cells.sha256"
    manifest = run_root / "cells.json"
    _read_regular_bytes(checksum, label="manifest checksum")
    _read_regular_bytes(manifest, label="manifest")
    try:
        snapshot = load_manifest(run_root, verify_frozen=True)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IncidentError(f"cannot load checksum-frozen manifest {run_root}: {exc}") from exc
    for cell_id in snapshot.ids:
        _require_safe_component(cell_id, label="manifest cell id")
    return snapshot


def _normalise_run_root(results_root: Path, run_id: str) -> Path:
    _require_safe_component(run_id, label="run id")
    root = _require_directory(results_root, label="results root")
    requested = root / run_id
    resolved = _require_directory(requested, label="run root")
    if resolved.parent != root or resolved.name != run_id or requested.is_symlink():
        raise IncidentError(f"run root escapes results root or is a symlink: {requested}")
    return resolved


def _path_within(path: Path, parent: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise IncidentError(f"refusing symlink {label}: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise IncidentError(f"{label} escapes its trusted root: {path}") from exc
    return resolved


def _load_dispatcher_evidence(state_dir: Path) -> tuple[DispatcherEvidence, ...]:
    """Join every protocol traceback to one exact dispatcher batch task."""

    state = _require_directory(state_dir, label="dispatcher state directory")
    ledger_path = state / "ledger.json"
    ledger, ledger_bytes = _read_json_object(ledger_path, label="dispatcher ledger")
    if ledger.get("schema_version") != DISPATCHER_LEDGER_SCHEMA_VERSION:
        raise IncidentError(
            f"unsupported dispatcher ledger schema in {ledger_path}: "
            f"{ledger.get('schema_version')!r}"
        )
    jobs = ledger.get("jobs")
    if not isinstance(jobs, dict):
        raise IncidentError(f"dispatcher ledger jobs must be an object: {ledger_path}")
    ledger_sha256 = _sha256_bytes(ledger_bytes)
    logs_root = _require_directory(state / "logs", label="dispatcher logs directory")
    batches_root = _require_directory(
        state / "batches", label="dispatcher batches directory"
    )
    events: list[DispatcherEvidence] = []

    for log_path in sorted(logs_root.iterdir(), key=lambda item: item.name):
        if log_path.is_symlink() or not log_path.is_file():
            raise IncidentError(f"refusing unsafe entry in dispatcher logs: {log_path}")
        log_bytes = log_path.read_bytes()
        if TRACEBACK_MARKER not in log_bytes:
            continue
        match = LOG_PATTERN.fullmatch(log_path.name)
        if match is None:
            raise IncidentError(f"protocol traceback has unmappable log name: {log_path}")
        job_id = match.group("job")
        task_index = int(match.group("task"))
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            raise IncidentError(
                f"protocol traceback {log_path} has no dispatcher ledger job mapping"
            )
        tasks = job.get("tasks")
        if not isinstance(tasks, list) or not 0 <= task_index < len(tasks):
            raise IncidentError(
                f"protocol traceback {log_path} has invalid dispatcher task mapping"
            )
        task = tasks[task_index]
        if not isinstance(task, dict):
            raise IncidentError(f"dispatcher task is not an object for {log_path}")
        required_task_fields = {
            "run_id",
            "run_root",
            "cell_id",
            "config_hash",
            "manifest_sha256",
        }
        if not required_task_fields.issubset(task):
            raise IncidentError(f"dispatcher task lacks identity fields for {log_path}")
        _require_safe_component(str(task["run_id"]), label="dispatcher run id")
        _require_safe_component(str(task["cell_id"]), label="dispatcher cell id")

        raw_batch_path = job.get("batch_manifest")
        if not isinstance(raw_batch_path, str) or not raw_batch_path:
            raise IncidentError(f"dispatcher job lacks batch_manifest for {log_path}")
        batch_path = _path_within(
            Path(raw_batch_path), batches_root, label="dispatcher batch manifest"
        )
        batch, batch_bytes = _read_json_object(batch_path, label="batch manifest")
        if batch.get("schema_version") != 1:
            raise IncidentError(f"unsupported batch schema in {batch_path}")
        batch_tasks = batch.get("tasks")
        if not isinstance(batch_tasks, list) or not 0 <= task_index < len(batch_tasks):
            raise IncidentError(f"batch task mapping is invalid for {log_path}")
        if batch_tasks[task_index] != task:
            raise IncidentError(
                f"dispatcher ledger/batch task drift for {job_id}_{task_index}"
            )
        if job.get("batch_id") != batch.get("batch_id"):
            raise IncidentError(f"dispatcher ledger/batch id drift for {job_id}")

        events.append(
            DispatcherEvidence(
                state_dir=state,
                ledger_path=ledger_path,
                ledger_sha256=ledger_sha256,
                job_id=job_id,
                task_index=task_index,
                task=dict(task),
                job_mapping={
                    key: job.get(key)
                    for key in (
                        "job_id",
                        "batch_id",
                        "batch_manifest",
                        "sbatch_path",
                        "submitted_at",
                        "task_count",
                    )
                },
                log_path=log_path.resolve(),
                log_sha256=_sha256_bytes(log_bytes),
                log_size=len(log_bytes),
                batch_path=batch_path,
                batch_sha256=_sha256_bytes(batch_bytes),
                batch_size=len(batch_bytes),
            )
        )
    return tuple(events)


def _failure_evidence(cell_dir: Path, cell: ExperimentCell) -> FailureEvidence | None:
    path = cell_dir / FAILURE_FILENAME
    if not path.exists():
        return None
    value, payload = _read_json_object(path, label="failure ledger")
    last_error = value.get("last_error")
    if not isinstance(last_error, dict) or last_error.get("type") != PROTOCOL_ERROR_TYPE:
        return None
    if value.get("cell_id") != cell.cell_id:
        raise IncidentError(f"protocol failure cell_id drift in {path}")
    if value.get("config_hash") != cell.config_hash():
        raise IncidentError(f"protocol failure config_hash drift in {path}")
    return FailureEvidence(
        path=path,
        sha256=_sha256_bytes(payload),
        size=len(payload),
        payload=value,
    )


def discover_candidates(
    run_root: Path,
    snapshot: ManifestSnapshot,
    dispatcher_events: Iterable[DispatcherEvidence],
) -> tuple[tuple[Candidate, ...], int]:
    """Discover only checksum-manifested cells, retaining stale-dir audit count."""

    cells_root = run_root / "cells"
    if cells_root.exists() and (cells_root.is_symlink() or not cells_root.is_dir()):
        raise IncidentError(f"unsafe cells directory: {cells_root}")
    manifest_by_id = {cell.cell_id: (index, cell) for index, cell in enumerate(snapshot.cells)}
    event_map: dict[str, list[DispatcherEvidence]] = {}
    for event in dispatcher_events:
        if event.run_id != run_root.name:
            continue
        task_run_root = Path(str(event.task["run_root"]))
        if task_run_root.is_symlink() or task_run_root.resolve() != run_root:
            raise IncidentError(
                f"dispatcher task run_root drift for {event.job_id}_{event.task_index}"
            )
        if event.task.get("manifest_sha256") != snapshot.sha256:
            raise IncidentError(
                f"dispatcher task manifest drift for {event.job_id}_{event.task_index}"
            )
        mapped = manifest_by_id.get(event.cell_id)
        if mapped is None:
            raise IncidentError(
                f"dispatcher protocol traceback targets unmanifested cell {event.cell_id!r}"
            )
        _, cell = mapped
        if event.task.get("config_hash") != cell.config_hash():
            raise IncidentError(
                f"dispatcher task config drift for {event.job_id}_{event.task_index}"
            )
        event_map.setdefault(event.cell_id, []).append(event)

    candidates: list[Candidate] = []
    for index, cell in enumerate(snapshot.cells):
        cell_dir = cells_root / cell.cell_id
        if cell_dir.exists() and (cell_dir.is_symlink() or not cell_dir.is_dir()):
            raise IncidentError(f"unsafe manifested cell directory: {cell_dir}")
        failure = _failure_evidence(cell_dir, cell) if cell_dir.exists() else None
        events = tuple(sorted(event_map.get(cell.cell_id, ()), key=lambda item: item.evidence_id))
        # An archive without its reset marker is a recoverable transaction interrupted
        # after durable publication.  Include archives in discovery even when the
        # failure source was already removed before the interruption and no dispatcher
        # state directory is available on the recovery host.
        archive_present = _archive_directory(run_root, cell.cell_id).exists()
        if failure is not None or events or archive_present:
            candidates.append(Candidate(cell, index, failure, events))

    present = {
        path.name
        for path in cells_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    } if cells_root.exists() else set()
    stale_count = len(present - set(snapshot.ids))
    return tuple(candidates), stale_count


def _archive_directory(run_root: Path, cell_id: str) -> Path:
    _require_safe_component(cell_id, label="archive cell id")
    return run_root / INCIDENT_DIRECTORY / cell_id


def _snapshot_active_artifacts(cell_dir: Path) -> tuple[ArtifactSnapshot, ...]:
    artifacts: list[ArtifactSnapshot] = []
    for filename in ACTIVE_ARTIFACT_FILENAMES:
        path = cell_dir / filename
        if not path.exists():
            continue
        payload = _read_regular_bytes(path, label="active cell artifact")
        artifacts.append(
            ArtifactSnapshot(
                source=path,
                source_relative_path=filename,
                archived_relative_path=f"artifacts/{filename}",
                payload=payload,
                sha256=_sha256_bytes(payload),
            )
        )

    checkpoint_dir = cell_dir / CHECKPOINT_DIRECTORY
    if checkpoint_dir.exists():
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise IncidentError(f"unsafe QID checkpoint directory: {checkpoint_dir}")
        for path in sorted(checkpoint_dir.iterdir(), key=lambda item: item.name):
            _require_safe_component(path.name, label="checkpoint filename")
            payload = _read_regular_bytes(path, label="QID checkpoint")
            relative = f"{CHECKPOINT_DIRECTORY}/{path.name}"
            artifacts.append(
                ArtifactSnapshot(
                    source=path,
                    source_relative_path=relative,
                    archived_relative_path=f"artifacts/{relative}",
                    payload=payload,
                    sha256=_sha256_bytes(payload),
                )
            )
    return tuple(artifacts)


def _dispatcher_payload(event: DispatcherEvidence) -> dict[str, Any]:
    evidence_root = f"evidence/dispatcher/{event.evidence_id}"
    return {
        "evidence_id": event.evidence_id,
        "state_dir": str(event.state_dir),
        "ledger": {
            "original_path": str(event.ledger_path),
            "sha256": event.ledger_sha256,
            "job_mapping": event.job_mapping,
        },
        "job_id": event.job_id,
        "task_index": event.task_index,
        "task": event.task,
        "log": {
            "original_path": str(event.log_path),
            "archived_path": f"{evidence_root}/worker.out",
            "sha256": event.log_sha256,
            "size": event.log_size,
        },
        "batch_manifest": {
            "original_path": str(event.batch_path),
            "archived_path": f"{evidence_root}/batch.json",
            "sha256": event.batch_sha256,
            "size": event.batch_size,
        },
    }


def _incident_payload(
    run_root: Path,
    snapshot: ManifestSnapshot,
    candidate: Candidate,
    artifacts: Sequence[ArtifactSnapshot],
) -> dict[str, Any]:
    return {
        "incident_schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_type": INCIDENT_KIND,
        "source_protocol_error_type": PROTOCOL_ERROR_TYPE,
        "target_artifact_schema_version": TARGET_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_root.name,
        "manifest": {
            "path": "cells.json",
            "sha256": snapshot.sha256,
            "cell_count": len(snapshot.cells),
        },
        "cell": {
            "cell_id": candidate.cell_id,
            "manifest_index": candidate.manifest_index,
            "config_hash": candidate.cell.config_hash(),
            "config": candidate.cell.to_dict(),
        },
        "discovery": {
            "failure": None
            if candidate.failure is None
            else {
                "original_path": str(candidate.failure.path),
                "sha256": candidate.failure.sha256,
                "size": candidate.failure.size,
                "last_error": candidate.failure.payload.get("last_error"),
            },
            "dispatcher_events": [
                _dispatcher_payload(event) for event in candidate.dispatcher
            ],
        },
        "active_artifacts": [
            {
                "source_relative_path": artifact.source_relative_path,
                "archived_path": artifact.archived_relative_path,
                "sha256": artifact.sha256,
                "size": len(artifact.payload),
            }
            for artifact in artifacts
        ],
        "reset_contract": {
            "scope": "entire_active_cell_state",
            "active_files": list(ACTIVE_ARTIFACT_FILENAMES),
            "checkpoint_directory": CHECKPOINT_DIRECTORY,
            "rerun_from_empty_cell_under_schema": TARGET_ARTIFACT_SCHEMA_VERSION,
        },
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError(f"zero-byte write while archiving {path}")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _mkdir_trusted(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise IncidentError(f"unsafe archive directory: {path}")


def _verify_event_sources(event: DispatcherEvidence) -> tuple[bytes, bytes]:
    log_bytes = _read_regular_bytes(event.log_path, label="dispatcher protocol log")
    batch_bytes = _read_regular_bytes(event.batch_path, label="dispatcher batch manifest")
    if _sha256_bytes(log_bytes) != event.log_sha256:
        raise IncidentError(f"dispatcher log changed since discovery: {event.log_path}")
    if _sha256_bytes(batch_bytes) != event.batch_sha256:
        raise IncidentError(
            f"dispatcher batch manifest changed since discovery: {event.batch_path}"
        )
    return log_bytes, batch_bytes


def _publish_archive(
    archive_dir: Path,
    incident: Mapping[str, Any],
    artifacts: Sequence[ArtifactSnapshot],
    events: Sequence[DispatcherEvidence],
) -> None:
    if archive_dir.exists():
        return
    parent = archive_dir.parent
    _mkdir_trusted(parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{archive_dir.name}.", dir=parent))
    try:
        for artifact in artifacts:
            _atomic_write_bytes(staging / artifact.archived_relative_path, artifact.payload)
        for event in events:
            log_bytes, batch_bytes = _verify_event_sources(event)
            evidence_root = staging / "evidence" / "dispatcher" / event.evidence_id
            _atomic_write_bytes(evidence_root / "worker.out", log_bytes)
            _atomic_write_bytes(evidence_root / "batch.json", batch_bytes)
        _atomic_write_bytes(staging / INCIDENT_FILENAME, _canonical_json_bytes(incident))
        try:
            os.rename(staging, archive_dir)
        except FileExistsError:
            # Another tool cannot own the same cell lock, but retain a fail-safe race
            # path for filesystems with unusual advisory-lock behavior.
            if not archive_dir.is_dir() or archive_dir.is_symlink():
                raise IncidentError(f"unsafe existing incident archive: {archive_dir}")
        _fsync_directory(parent)
    finally:
        if staging.exists():
            # Staging contains only freshly-created copies.  Remove it narrowly; never
            # recurse through a symlink or an operator-supplied path.
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_symlink():
                    path.unlink()
                elif path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def _load_archive(
    archive_dir: Path,
    run_root: Path,
    snapshot: ManifestSnapshot,
    candidate: Candidate,
) -> tuple[dict[str, Any], bytes]:
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        raise IncidentError(f"unsafe incident archive: {archive_dir}")
    incident, payload = _read_json_object(
        archive_dir / INCIDENT_FILENAME, label="incident record"
    )
    if incident.get("incident_schema_version") != INCIDENT_SCHEMA_VERSION:
        raise IncidentError(f"unsupported incident schema in {archive_dir}")
    if incident.get("incident_type") != INCIDENT_KIND:
        raise IncidentError(f"incident type drift in {archive_dir}")
    if incident.get("run_id") != run_root.name:
        raise IncidentError(f"incident run drift in {archive_dir}")
    manifest = incident.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("sha256") != snapshot.sha256:
        raise IncidentError(f"incident manifest drift in {archive_dir}")
    cell = incident.get("cell")
    expected_cell = {
        "cell_id": candidate.cell_id,
        "manifest_index": candidate.manifest_index,
        "config_hash": candidate.cell.config_hash(),
        "config": candidate.cell.to_dict(),
    }
    if cell != expected_cell:
        raise IncidentError(f"incident cell identity drift in {archive_dir}")
    if incident.get("target_artifact_schema_version") != TARGET_ARTIFACT_SCHEMA_VERSION:
        raise IncidentError(f"incident target schema drift in {archive_dir}")

    artifact_rows = incident.get("active_artifacts")
    if not isinstance(artifact_rows, list):
        raise IncidentError(f"incident artifact index is invalid in {archive_dir}")
    seen_sources: set[str] = set()
    for row in artifact_rows:
        if not isinstance(row, dict):
            raise IncidentError(f"invalid incident artifact row in {archive_dir}")
        source_relative = row.get("source_relative_path")
        archived_relative = row.get("archived_path")
        if not isinstance(source_relative, str) or source_relative in seen_sources:
            raise IncidentError(f"duplicate/invalid incident artifact source in {archive_dir}")
        seen_sources.add(source_relative)
        if not isinstance(archived_relative, str):
            raise IncidentError(f"invalid archived artifact path in {archive_dir}")
        archived = _path_within(
            archive_dir / archived_relative, archive_dir, label="archived artifact"
        )
        archived_bytes = _read_regular_bytes(archived, label="archived artifact")
        if (
            _sha256_bytes(archived_bytes) != row.get("sha256")
            or len(archived_bytes) != row.get("size")
        ):
            raise IncidentError(f"archived artifact integrity failure: {archived}")

    discovery = incident.get("discovery")
    events = discovery.get("dispatcher_events") if isinstance(discovery, dict) else None
    if not isinstance(events, list):
        raise IncidentError(f"incident dispatcher evidence is invalid in {archive_dir}")
    for event in events:
        if not isinstance(event, dict):
            raise IncidentError(f"invalid dispatcher evidence in {archive_dir}")
        for field in ("log", "batch_manifest"):
            item = event.get(field)
            if not isinstance(item, dict) or not isinstance(item.get("archived_path"), str):
                raise IncidentError(f"invalid archived {field} evidence in {archive_dir}")
            archived = _path_within(
                archive_dir / item["archived_path"],
                archive_dir,
                label=f"archived {field}",
            )
            evidence_bytes = _read_regular_bytes(archived, label=f"archived {field}")
            if (
                _sha256_bytes(evidence_bytes) != item.get("sha256")
                or len(evidence_bytes) != item.get("size")
            ):
                raise IncidentError(f"archived {field} integrity failure: {archived}")
    return incident, payload


def _source_from_relative(cell_dir: Path, relative: str) -> Path:
    path = cell_dir / relative
    resolved_parent = path.parent.resolve()
    cell_resolved = cell_dir.resolve()
    try:
        resolved_parent.relative_to(cell_resolved)
    except ValueError as exc:
        raise IncidentError(f"incident source path escapes cell: {relative!r}") from exc
    return path


def _reset_archived_sources(cell_dir: Path, incident: Mapping[str, Any]) -> None:
    artifact_rows = incident["active_artifacts"]
    for row in artifact_rows:
        source = _source_from_relative(cell_dir, str(row["source_relative_path"]))
        if not source.exists():
            continue
        payload = _read_regular_bytes(source, label="active artifact pending reset")
        if _sha256_bytes(payload) != row["sha256"] or len(payload) != row["size"]:
            raise IncidentError(
                f"active artifact changed after archival; refusing reset: {source}"
            )
        io.remove_file(source)

    checkpoint_dir = cell_dir / CHECKPOINT_DIRECTORY
    if checkpoint_dir.exists():
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise IncidentError(f"unsafe checkpoint directory during reset: {checkpoint_dir}")
        remaining = list(checkpoint_dir.iterdir())
        if remaining:
            # Anything remaining was not in the archive and may be a newly-observed draw.
            raise IncidentError(
                f"unarchived checkpoint appeared during reset: {remaining[0]}"
            )
        checkpoint_dir.rmdir()
        _fsync_directory(cell_dir)

    # The snapshot enumerates all active artifacts.  An unexpected active file means a
    # worker wrote after the archive snapshot, so never widen the deletion set.
    archived_sources = {row["source_relative_path"] for row in artifact_rows}
    for filename in ACTIVE_ARTIFACT_FILENAMES:
        path = cell_dir / filename
        if path.exists() and filename not in archived_sources:
            raise IncidentError(f"unarchived active artifact appeared during reset: {path}")


def _reset_marker_payload(incident_bytes: bytes) -> dict[str, Any]:
    return {
        "reset_marker_schema_version": RESET_MARKER_SCHEMA_VERSION,
        "incident_sha256": _sha256_bytes(incident_bytes),
        "target_artifact_schema_version": TARGET_ARTIFACT_SCHEMA_VERSION,
        "active_cell_state_removed": True,
    }


def _write_reset_marker(archive_dir: Path, incident_bytes: bytes) -> None:
    marker_path = archive_dir / RESET_MARKER_FILENAME
    marker = _reset_marker_payload(incident_bytes)
    if marker_path.exists():
        existing, _ = _read_json_object(marker_path, label="reset marker")
        if existing != marker:
            raise IncidentError(f"reset marker drift in {archive_dir}")
        return
    _atomic_write_bytes(marker_path, _canonical_json_bytes(marker), mode=0o444)


def _verify_reset_marker(archive_dir: Path, incident_bytes: bytes) -> bool:
    marker_path = archive_dir / RESET_MARKER_FILENAME
    if not marker_path.exists():
        return False
    marker, _ = _read_json_object(marker_path, label="reset marker")
    expected = _reset_marker_payload(incident_bytes)
    if marker != expected:
        raise IncidentError(f"reset marker integrity failure in {archive_dir}")
    return True


def _post_reset_incident_error(
    incident: Mapping[str, Any], candidate: Candidate
) -> str | None:
    """Return a fail-closed warning for evidence newer than the sealed incident.

    Old dispatcher logs intentionally remain discoverable forever.  Compare their
    content-addressed event identities with the sealed archive so those expected logs do
    not create false alerts.  Conversely, any newly-created failure ledger or any new
    job/task/log identity is a separate sampled-response incident.  This utility cannot
    append it to an immutable archive or reset schema-5 data under the authority granted
    for the old protocol-v4 migration.
    """

    if candidate.failure is not None:
        return (
            f"post-reset {PROTOCOL_ERROR_TYPE} failure exists for {candidate.cell_id}; "
            "preserve schema-5 artifacts and open a new operator incident"
        )
    discovery = incident.get("discovery")
    archived_events = (
        discovery.get("dispatcher_events") if isinstance(discovery, dict) else None
    )
    if not isinstance(archived_events, list):
        raise IncidentError("sealed incident has invalid dispatcher evidence")
    archived_ids = {
        str(event.get("evidence_id"))
        for event in archived_events
        if isinstance(event, dict) and isinstance(event.get("evidence_id"), str)
    }
    new_events = [
        event.evidence_id
        for event in candidate.dispatcher
        if event.evidence_id not in archived_ids
    ]
    if new_events:
        return (
            f"post-reset {PROTOCOL_ERROR_TYPE} dispatcher evidence exists for "
            f"{candidate.cell_id}: {new_events!r}; preserve schema-5 artifacts and "
            "open a new operator incident"
        )
    return None


def _relative_report_path(path: Path, run_root: Path) -> str:
    """Return a stable run-relative path for a recovery mutation report."""

    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _candidate_mutation_plan(
    run_root: Path,
    snapshot: ManifestSnapshot,
    candidate: Candidate,
) -> dict[str, Any]:
    """Describe every byte-level publication/removal before a reset is attempted."""

    archive_dir = _archive_directory(run_root, candidate.cell_id)
    cell_dir = run_root / "cells" / candidate.cell_id
    archive_publications: list[dict[str, Any]] = []

    if archive_dir.exists():
        incident, incident_bytes = _load_archive(
            archive_dir, run_root, snapshot, candidate
        )
        if _verify_reset_marker(archive_dir, incident_bytes):
            return {
                "would_change": False,
                "active_artifact_changes": [],
                "archive_publications": [],
                "complete_preimages_archived": True,
            }
        artifact_rows = incident["active_artifacts"]
    else:
        artifacts = _snapshot_active_artifacts(cell_dir)
        incident = _incident_payload(run_root, snapshot, candidate, artifacts)
        incident_bytes = _canonical_json_bytes(incident)
        artifact_rows = incident["active_artifacts"]
        for artifact in artifacts:
            archive_publications.append(
                {
                    "path": _relative_report_path(
                        archive_dir / artifact.archived_relative_path, run_root
                    ),
                    "operation": "create_archived_preimage",
                    "before_sha256": None,
                    "after_sha256": artifact.sha256,
                    "after_size": len(artifact.payload),
                }
            )
        for event in candidate.dispatcher:
            evidence_root = archive_dir / "evidence" / "dispatcher" / event.evidence_id
            archive_publications.extend(
                (
                    {
                        "path": _relative_report_path(
                            evidence_root / "worker.out", run_root
                        ),
                        "operation": "create_dispatcher_evidence",
                        "before_sha256": None,
                        "after_sha256": event.log_sha256,
                        "after_size": event.log_size,
                    },
                    {
                        "path": _relative_report_path(
                            evidence_root / "batch.json", run_root
                        ),
                        "operation": "create_dispatcher_evidence",
                        "before_sha256": None,
                        "after_sha256": event.batch_sha256,
                        "after_size": event.batch_size,
                    },
                )
            )
        archive_publications.append(
            {
                "path": _relative_report_path(
                    archive_dir / INCIDENT_FILENAME, run_root
                ),
                "operation": "create_incident_index",
                "before_sha256": None,
                "after_sha256": _sha256_bytes(incident_bytes),
                "after_size": len(incident_bytes),
            }
        )

    active_changes: list[dict[str, Any]] = []
    for row in artifact_rows:
        source = _source_from_relative(cell_dir, str(row["source_relative_path"]))
        if not source.exists():
            continue
        payload = _read_regular_bytes(source, label="active artifact mutation plan")
        digest = _sha256_bytes(payload)
        if digest != row["sha256"] or len(payload) != row["size"]:
            raise IncidentError(
                f"active artifact differs from planned archive preimage: {source}"
            )
        active_changes.append(
            {
                "path": _relative_report_path(source, run_root),
                "operation": "remove_after_verified_archive",
                "before_sha256": digest,
                "after_sha256": None,
                "before_size": len(payload),
                "archived_path": _relative_report_path(
                    archive_dir / str(row["archived_path"]), run_root
                ),
            }
        )

    marker_bytes = _canonical_json_bytes(_reset_marker_payload(incident_bytes))
    archive_publications.append(
        {
            "path": _relative_report_path(
                archive_dir / RESET_MARKER_FILENAME, run_root
            ),
            "operation": "create_reset_marker",
            "before_sha256": None,
            "after_sha256": _sha256_bytes(marker_bytes),
            "after_size": len(marker_bytes),
        }
    )
    return {
        "would_change": True,
        "active_artifact_changes": sorted(
            active_changes, key=lambda row: str(row["path"])
        ),
        "archive_publications": sorted(
            archive_publications, key=lambda row: str(row["path"])
        ),
        "complete_preimages_archived": all(
            row.get("archived_path") for row in active_changes
        ),
    }


def _seal_archive(archive_dir: Path) -> None:
    for path in sorted(archive_dir.rglob("*"), reverse=True):
        if path.is_symlink():
            raise IncidentError(f"symlink appeared in incident archive: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    archive_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def _process_candidate(
    run_root: Path,
    snapshot: ManifestSnapshot,
    candidate: Candidate,
    *,
    apply: bool,
) -> str:
    archive_dir = _archive_directory(run_root, candidate.cell_id)
    if archive_dir.exists():
        incident, incident_bytes = _load_archive(
            archive_dir, run_root, snapshot, candidate
        )
        if _verify_reset_marker(archive_dir, incident_bytes):
            post_reset_error = _post_reset_incident_error(incident, candidate)
            if post_reset_error is not None:
                raise NewProtocolIncidentRequired(post_reset_error)
            return "already_reset"
        if not apply:
            return "archive_pending_reset"
    elif not apply:
        return "candidate"

    cell_dir = run_root / "cells" / candidate.cell_id
    try:
        with cell_lock(cell_dir):
            if candidate.failure is not None:
                path = cell_dir / FAILURE_FILENAME
                if not path.exists():
                    raise IncidentError(
                        f"protocol failure disappeared after discovery: {path}"
                    )
                failure_bytes = _read_regular_bytes(path, label="protocol failure")
                if _sha256_bytes(failure_bytes) != candidate.failure.sha256:
                    raise IncidentError(
                        f"protocol failure changed after discovery: {path}"
                    )

            if archive_dir.exists():
                incident, incident_bytes = _load_archive(
                    archive_dir, run_root, snapshot, candidate
                )
                if _verify_reset_marker(archive_dir, incident_bytes):
                    post_reset_error = _post_reset_incident_error(incident, candidate)
                    if post_reset_error is not None:
                        raise NewProtocolIncidentRequired(post_reset_error)
                    return "already_reset"
            else:
                artifacts = _snapshot_active_artifacts(cell_dir)
                incident = _incident_payload(
                    run_root, snapshot, candidate, artifacts
                )
                _publish_archive(
                    archive_dir, incident, artifacts, candidate.dispatcher
                )
                incident, incident_bytes = _load_archive(
                    archive_dir, run_root, snapshot, candidate
                )

            _reset_archived_sources(cell_dir, incident)
            _write_reset_marker(archive_dir, incident_bytes)
            _seal_archive(archive_dir)
            return "reset"
    except CellLockUnavailable:
        return "locked"


def archive_reset_run(
    run_root: Path,
    *,
    dispatcher_state_dirs: Sequence[Path],
    apply: bool = False,
    selected_cell_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Discover, archive, and optionally reset one checksum-frozen run."""

    requested_run_root = Path(run_root)
    if requested_run_root.is_symlink():
        raise IncidentError(f"refusing symlink run root: {requested_run_root}")
    run_root = _require_directory(requested_run_root, label="run root")
    snapshot = _require_frozen_manifest(run_root)
    if ARTIFACT_SCHEMA_VERSION != TARGET_ARTIFACT_SCHEMA_VERSION:
        raise IncidentError(
            "protocol incident reset is pinned to executable artifact schema 5; "
            f"current schema is {ARTIFACT_SCHEMA_VERSION}"
        )
    all_events: list[DispatcherEvidence] = []
    for state_dir in dispatcher_state_dirs:
        all_events.extend(_load_dispatcher_evidence(Path(state_dir)))
    candidates, stale_count = discover_candidates(run_root, snapshot, all_events)

    selection = set(selected_cell_ids or ())
    unknown_selection = selection - set(snapshot.ids)
    if unknown_selection:
        raise IncidentError(
            f"selected cell ids are not in frozen manifest: {sorted(unknown_selection)!r}"
        )
    if selection:
        candidates = tuple(candidate for candidate in candidates if candidate.cell_id in selection)
        undiscovered = selection - {candidate.cell_id for candidate in candidates}
        if undiscovered:
            raise IncidentError(
                "selected cells lack ServerResponseProtocolError evidence: "
                f"{sorted(undiscovered)!r}"
            )

    counts = {
        "candidate": 0,
        "reset": 0,
        "already_reset": 0,
        "archive_pending_reset": 0,
        "locked": 0,
        "new_incident_required": 0,
        "error": 0,
    }
    cells: list[dict[str, Any]] = []
    for candidate in candidates:
        mutation_plan: dict[str, Any] | None = None
        try:
            mutation_plan = _candidate_mutation_plan(
                run_root, snapshot, candidate
            )
            outcome = _process_candidate(
                run_root, snapshot, candidate, apply=apply
            )
            error = None
        except NewProtocolIncidentRequired as exc:
            outcome = "new_incident_required"
            error = str(exc)
        except IncidentError as exc:
            outcome = "error"
            error = str(exc)
        counts[outcome] += 1
        cells.append(
            {
                "cell_id": candidate.cell_id,
                "manifest_index": candidate.manifest_index,
                "config_hash": candidate.cell.config_hash(),
                "failure_evidence": candidate.failure is not None,
                "dispatcher_event_count": len(candidate.dispatcher),
                "outcome": outcome,
                "error": error,
                "archive": str(_archive_directory(run_root, candidate.cell_id)),
                "mutation_plan": mutation_plan,
            }
        )

    would_change_cells = sorted(
        row["cell_id"]
        for row in cells
        if isinstance(row.get("mutation_plan"), dict)
        and row["mutation_plan"].get("would_change") is True
    )
    return {
        "run_root": str(run_root),
        "run_id": run_root.name,
        "manifest_sha256": snapshot.sha256,
        "manifest_cells": len(snapshot.cells),
        "target_artifact_schema_version": TARGET_ARTIFACT_SCHEMA_VERSION,
        "dispatcher_state_dirs": [str(Path(path).resolve()) for path in dispatcher_state_dirs],
        "dispatcher_protocol_events_all_runs": len(all_events),
        "affected_manifest_cells": len(candidates),
        "stale_unmanifested_dirs": stale_count,
        "outcomes": counts,
        "cells": cells,
        "would_change_cells": would_change_cells,
        "applied": apply,
        "errors": [row["error"] for row in cells if row["error"] is not None],
    }


def _error_report(run_root: Path, error: BaseException, *, apply: bool) -> dict[str, Any]:
    return {
        "run_root": str(run_root),
        "applied": apply,
        "errors": [f"{type(error).__name__}: {error}"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--results-root",
        default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--dispatcher-state-dir",
        action="append",
        help="repeat for every dispatcher state directory; defaults to .dispatcher-v3",
    )
    parser.add_argument(
        "--cell-id",
        action="append",
        default=[],
        help="optional manifested/evidence-backed filter; never adds an unproven target",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="publish immutable archives and reset their exact active artifact snapshots",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).expanduser()
    state_dirs = (
        [Path(path).expanduser() for path in args.dispatcher_state_dir]
        if args.dispatcher_state_dir
        else [results_root / ".dispatcher-v3"]
    )
    reports: list[dict[str, Any]] = []
    for run_id in args.run_id:
        requested = results_root / run_id
        try:
            run_root = _normalise_run_root(results_root, run_id)
            reports.append(
                archive_reset_run(
                    run_root,
                    dispatcher_state_dirs=state_dirs,
                    apply=args.apply,
                    selected_cell_ids=set(args.cell_id),
                )
            )
        except (IncidentError, OSError) as exc:
            reports.append(_error_report(requested, exc, apply=args.apply))
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))
    return 1 if any(report.get("errors") for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
