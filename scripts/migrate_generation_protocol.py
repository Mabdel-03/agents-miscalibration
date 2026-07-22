#!/usr/bin/env python
"""Retire explicit pre-current generation-protocol artifacts without losing evidence.

The generation protocol is part of the scientific artifact contract.  Rows and metadata
written with an explicit, superseded artifact schema must not be silently interpreted by
the current runner, and pilot ``GenerationTruncationError`` ledgers must not permanently
poison their cells.  This utility performs the deliberately narrow migration required to
make those cells resumable:

* only cells named by a checksum-frozen ``cells.json`` are considered;
* schema-less legacy rows/metadata and current-schema artifacts are retained;
* every removed payload, including its exact UTF-8 source text and SHA-256, is first
  recorded in one atomic run-level incident file;
* mutations occur only while holding the runner's nonblocking per-cell advisory lock.

The default is a read-only dry run.  Pass ``--apply`` explicitly to write anything.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Support the documented direct ``python scripts/...`` invocation from a checkout, not
# only environments where this project has already been installed editable.
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
    expected_qids_for_cell,
    is_cell_active,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest  # noqa: E402
from agents_scaling.experiment.result_schema import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
)


INCIDENT_SCHEMA_VERSION = 1
INCIDENT_FILENAME = "generation_protocol_migration_incident_v1.json"
CURRENT_INCIDENT_FILENAME = (
    f"generation_protocol_migration_incident_schema_{ARTIFACT_SCHEMA_VERSION}_v1.json"
)
RUN_LOCK_FILENAME = ".generation_protocol_migration.lock"
TRUNCATION_ERROR_TYPE = "GenerationTruncationError"
# Both explicit schemas were emitted only by the superseded controlled-rollout pilots.
# Schema 2 briefly added serving provenance to otherwise legacy rows; schema 3 used the
# now-retired multi-phase generation protocol.  Neither is admissible as current schema
# 4 evidence, and both are preserved byte-for-byte in the incident record before removal.
TARGET_ARTIFACT_SCHEMA_VERSIONS = frozenset({2, 3})
SEALED_HISTORICAL_ARTIFACT_SCHEMA_VERSIONS = frozenset({4})


class MigrationError(RuntimeError):
    """Raised when the migration cannot prove that a requested operation is safe."""


class _InvalidJSON(ValueError):
    pass


@dataclass(frozen=True)
class ResultsScan:
    source_text: str | None
    source_sha256: str | None
    retained_text: str | None
    retained_sha256: str | None
    removed_rows: tuple[dict[str, Any], ...]
    retained_qids: frozenset[str]
    malformed_lines_retained: int = 0


@dataclass(frozen=True)
class JSONArtifactScan:
    source_text: str | None
    source_sha256: str | None
    payload: dict[str, Any] | None
    targeted: bool
    parse_error: str | None = None


@dataclass(frozen=True)
class CellScan:
    results: ResultsScan
    metadata: JSONArtifactScan
    failure: JSONArtifactScan

    @property
    def targeted(self) -> bool:
        return bool(
            self.results.removed_rows or self.metadata.targeted or self.failure.targeted
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise _InvalidJSON(f"non-finite JSON number {value!r}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise _InvalidJSON(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeError, _InvalidJSON) as exc:
        raise _InvalidJSON(str(exc)) from exc


def _is_registered_schema(value: Any) -> bool:
    # ``bool`` is an ``int`` subclass and JSON 4.0 compares equal to integer 4.  Neither
    # is the explicit integer schema marker required by the artifact contract.
    return type(value) is int and value in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS


def _is_target_schema(value: Any) -> bool:
    return type(value) is int and value in TARGET_ARTIFACT_SCHEMA_VERSIONS


def _scan_results(path: Path) -> ResultsScan:
    if not path.exists():
        return ResultsScan(None, None, None, None, (), frozenset())
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"refusing non-regular results artifact: {path}")
    source = path.read_bytes()
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MigrationError(f"results artifact is not valid UTF-8: {path}: {exc}") from exc

    removed: list[dict[str, Any]] = []
    retained_lines: list[str] = []
    retained_qids: set[str] = set()
    malformed = 0
    for line_number, raw_line in enumerate(source_text.splitlines(keepends=True), start=1):
        try:
            value = _strict_json_loads(raw_line)
        except _InvalidJSON:
            # This is not a general canonicalizer.  Unrelated malformed evidence remains
            # byte-for-byte in place and can be handled by the normal repair workflow.
            malformed += 1
            retained_lines.append(raw_line)
            continue

        if isinstance(value, dict) and "schema_version" in value:
            schema = value["schema_version"]
            if not _is_registered_schema(schema) and not _is_target_schema(schema):
                raise MigrationError(
                    f"refusing unknown explicit result schema {schema!r} "
                    f"at {path}:{line_number}"
                )
        if isinstance(value, dict) and _is_target_schema(value.get("schema_version")):
            raw_bytes = raw_line.encode("utf-8")
            removed.append(
                {
                    "line_number": line_number,
                    "raw_line": raw_line,
                    "row": value,
                    "row_sha256": _sha256(raw_bytes),
                }
            )
            continue

        retained_lines.append(raw_line)
        if isinstance(value, dict):
            qid = value.get("qid")
            if isinstance(qid, str) and qid:
                retained_qids.add(qid)

    retained_text = "".join(retained_lines)
    return ResultsScan(
        source_text=source_text,
        source_sha256=_sha256(source),
        retained_text=retained_text,
        retained_sha256=_sha256(retained_text.encode("utf-8")),
        removed_rows=tuple(removed),
        retained_qids=frozenset(retained_qids),
        malformed_lines_retained=malformed,
    )


def _scan_json_artifact(path: Path, *, predicate: Any) -> JSONArtifactScan:
    if not path.exists():
        return JSONArtifactScan(None, None, None, False)
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"refusing non-regular JSON artifact: {path}")
    source = path.read_bytes()
    digest = _sha256(source)
    try:
        text = source.decode("utf-8")
        value = _strict_json_loads(text)
    except (UnicodeDecodeError, _InvalidJSON) as exc:
        return JSONArtifactScan(None, digest, None, False, str(exc))
    if not isinstance(value, dict):
        return JSONArtifactScan(text, digest, None, False, "payload is not a JSON object")
    return JSONArtifactScan(text, digest, value, bool(predicate(value)))


def _old_explicit_metadata(value: Mapping[str, Any]) -> bool:
    if "schema_version" not in value:
        return False
    schema = value["schema_version"]
    if _is_registered_schema(schema):
        return False
    if _is_target_schema(schema):
        return True
    raise MigrationError(f"refusing unknown explicit metadata schema {schema!r}")


def _truncation_failure(value: Mapping[str, Any]) -> bool:
    # The retired pilot emitted this exact legacy failure contract.  Matching only the
    # exception class is unsafe: schema-v2/current runners can retain the same exception
    # name as a fail-closed context-capacity event, and quarantining that ledger would
    # make an already observed trajectory eligible for replacement sampling.
    last_error = value.get("last_error")
    return (
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("classification") == "configuration"
        and value.get("disposition") == "permanent"
        and isinstance(last_error, Mapping)
        and last_error.get("type") == TRUNCATION_ERROR_TYPE
    )


def _scan_cell(cell_directory: Path) -> CellScan:
    return CellScan(
        results=_scan_results(cell_directory / RESULTS_FILENAME),
        metadata=_scan_json_artifact(
            cell_directory / META_FILENAME, predicate=_old_explicit_metadata
        ),
        failure=_scan_json_artifact(
            cell_directory / FAILURE_FILENAME, predicate=_truncation_failure
        ),
    )


def _require_frozen_manifest(run_root: Path) -> ManifestSnapshot:
    checksum = run_root / "cells.sha256"
    manifest = run_root / "cells.json"
    if not checksum.is_file() or checksum.is_symlink():
        raise MigrationError(
            f"immutable manifest checksum is missing or unsafe: {checksum}"
        )
    if not manifest.is_file() or manifest.is_symlink():
        raise MigrationError(f"immutable manifest is missing or unsafe: {manifest}")
    try:
        return load_manifest(run_root, verify_frozen=True)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise MigrationError(f"cannot load frozen manifest {run_root}: {exc}") from exc


def _new_incident(run_root: Path, snapshot: ManifestSnapshot) -> dict[str, Any]:
    return {
        "incident_schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_type": "generation_protocol_artifact_migration",
        "current_artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "target_artifact_schema_versions": sorted(TARGET_ARTIFACT_SCHEMA_VERSIONS),
        "run_id": run_root.name,
        "manifest": {
            "path": "cells.json",
            "sha256": snapshot.sha256,
            "cell_count": len(snapshot.cells),
        },
        "cells": {},
    }


def _validate_incident_evidence(
    value: Mapping[str, Any], *, path: Path, snapshot: ManifestSnapshot
) -> None:
    """Validate the internal byte hashes of an immutable migration history."""

    cells_by_id = {cell.cell_id: (index, cell) for index, cell in enumerate(snapshot.cells)}
    incident_cells = value["cells"]
    for cell_id, entry in incident_cells.items():
        if not isinstance(entry, Mapping):
            raise MigrationError(f"invalid incident cell entry {cell_id!r}: {path}")
        index, cell = cells_by_id[cell_id]
        if (
            entry.get("cell_id") != cell_id
            or entry.get("manifest_index") != index
            or entry.get("manifest_config_hash") != cell.config_hash()
        ):
            raise MigrationError(f"incident cell identity mismatch for {cell_id}: {path}")
        observed_removed_ids: set[str] = set()
        for removed in entry.get("removed_result_rows", []):
            if not isinstance(removed, Mapping):
                raise MigrationError(f"invalid removed-row evidence for {cell_id}: {path}")
            raw_line = removed.get("raw_line")
            source_sha = removed.get("source_results_sha256")
            line_number = removed.get("line_number")
            if not isinstance(raw_line, str) or not isinstance(source_sha, str):
                raise MigrationError(f"incomplete removed-row evidence for {cell_id}: {path}")
            row_sha = _sha256(raw_line.encode("utf-8"))
            expected_id = _evidence_id(source_sha, str(line_number), row_sha)
            if removed.get("row_sha256") != row_sha or removed.get("evidence_id") != expected_id:
                raise MigrationError(f"removed-row checksum mismatch for {cell_id}: {path}")
            observed_removed_ids.add(expected_id)
        for label, source_key in (
            ("quarantined_metadata", "source_meta_sha256"),
            ("quarantined_failures", "source_failure_sha256"),
        ):
            for evidence in entry.get(label, []):
                if not isinstance(evidence, Mapping) or not isinstance(evidence.get("raw_json"), str):
                    raise MigrationError(f"invalid {label} evidence for {cell_id}: {path}")
                raw_json = evidence["raw_json"]
                if _sha256(raw_json.encode("utf-8")) != evidence.get(source_key):
                    raise MigrationError(f"{label} checksum mismatch for {cell_id}: {path}")
                if _strict_json_loads(raw_json) != evidence.get("payload"):
                    raise MigrationError(f"{label} payload mismatch for {cell_id}: {path}")
        for rewrite in entry.get("result_rewrites", []):
            if not isinstance(rewrite, Mapping):
                raise MigrationError(f"invalid rewrite evidence for {cell_id}: {path}")
            source_sha = rewrite.get("source_results_sha256")
            rewritten_sha = rewrite.get("rewritten_results_sha256")
            if (
                not isinstance(source_sha, str)
                or not isinstance(rewritten_sha, str)
                or rewrite.get("rewrite_id") != _evidence_id(source_sha, rewritten_sha)
            ):
                raise MigrationError(f"rewrite checksum mismatch for {cell_id}: {path}")
            referenced = rewrite.get("removed_row_evidence_ids")
            if not isinstance(referenced, list) or not set(referenced) <= observed_removed_ids:
                raise MigrationError(f"rewrite references unknown evidence for {cell_id}: {path}")


def _load_incident(
    run_root: Path,
    snapshot: ManifestSnapshot,
    *,
    path: Path | None = None,
    allowed_artifact_schemas: frozenset[int] | None = None,
) -> dict[str, Any]:
    path = run_root / INCIDENT_FILENAME if path is None else path
    if not path.exists():
        return _new_incident(run_root, snapshot)
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"refusing unsafe incident artifact: {path}")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, _InvalidJSON) as exc:
        raise MigrationError(f"cannot read existing incident file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"existing incident file is not a JSON object: {path}")
    expected = _new_incident(run_root, snapshot)
    for field in (
        "incident_schema_version",
        "incident_type",
        "target_artifact_schema_versions",
        "run_id",
    ):
        if value.get(field) != expected[field]:
            raise MigrationError(f"existing incident has incompatible {field}: {path}")
    allowed = allowed_artifact_schemas or frozenset({ARTIFACT_SCHEMA_VERSION})
    recorded_schema = value.get("current_artifact_schema_version")
    if type(recorded_schema) is not int or recorded_schema not in allowed:
        raise MigrationError(
            f"existing incident has incompatible current_artifact_schema_version "
            f"{recorded_schema!r}: {path}"
        )
    manifest = value.get("manifest")
    if manifest != expected["manifest"]:
        raise MigrationError(f"existing incident does not match frozen manifest: {path}")
    incident_cells = value.get("cells")
    if not isinstance(incident_cells, dict):
        raise MigrationError(f"existing incident cells field is invalid: {path}")
    unknown = set(incident_cells) - set(snapshot.ids)
    if unknown:
        raise MigrationError(
            f"existing incident contains unmanifested cell ids: {sorted(unknown)!r}"
        )
    _validate_incident_evidence(value, path=path, snapshot=snapshot)
    return value


def _quarantine_relative_path(label: str, source_sha256: str) -> str:
    return f"quarantine/{label}.generation-protocol-migration.{source_sha256}.json"


def _infer_next_missing_qid(
    cell: ExperimentCell, retained_qids: frozenset[str]
) -> tuple[str | None, bool | None, str | None]:
    try:
        expected = expected_qids_for_cell(cell)
    except Exception as exc:  # dataset availability is an environmental concern here
        return None, None, f"{type(exc).__name__}: {exc}"
    for qid in expected:
        if qid not in retained_qids:
            return qid, False, None
    return None, True, None


def _evidence_id(*parts: str) -> str:
    return _sha256("\0".join(parts).encode("utf-8"))


def _merge_unique(
    existing: list[dict[str, Any]], additions: Sequence[dict[str, Any]], *, key: str
) -> bool:
    changed = False
    observed = {str(item.get(key)) for item in existing if isinstance(item, dict)}
    for item in additions:
        identity = str(item[key])
        if identity in observed:
            continue
        existing.append(item)
        observed.add(identity)
        changed = True
    existing.sort(key=lambda item: str(item.get(key, "")))
    return changed


def _merge_cell_evidence(
    incident: dict[str, Any],
    *,
    cell: ExperimentCell,
    manifest_index: int,
    scan: CellScan,
) -> bool:
    cells = incident["cells"]
    entry = cells.get(cell.cell_id)
    changed = False
    if entry is None:
        entry = {
            "cell_id": cell.cell_id,
            "manifest_index": manifest_index,
            "manifest_config_hash": cell.config_hash(),
            "removed_result_rows": [],
            "result_rewrites": [],
            "quarantined_metadata": [],
            "quarantined_failures": [],
            "inferred_next_missing_qid": None,
            "all_expected_qids_present_after_migration": None,
            "qid_inference_error": None,
        }
        cells[cell.cell_id] = entry
        changed = True
    elif not isinstance(entry, dict):
        raise MigrationError(f"incident entry for {cell.cell_id} is invalid")
    if (
        entry.get("manifest_index") != manifest_index
        or entry.get("manifest_config_hash") != cell.config_hash()
    ):
        raise MigrationError(f"incident entry for {cell.cell_id} does not match manifest")

    removed: list[dict[str, Any]] = []
    removed_ids: list[str] = []
    for row in scan.results.removed_rows:
        evidence_id = _evidence_id(
            str(scan.results.source_sha256),
            str(row["line_number"]),
            str(row["row_sha256"]),
        )
        removed_ids.append(evidence_id)
        removed.append(
            {
                **row,
                "evidence_id": evidence_id,
                "source_results_sha256": scan.results.source_sha256,
            }
        )
    changed |= _merge_unique(
        entry.setdefault("removed_result_rows", []), removed, key="evidence_id"
    )
    if removed:
        rewrite_id = _evidence_id(
            str(scan.results.source_sha256), str(scan.results.retained_sha256)
        )
        rewrite = {
            "rewrite_id": rewrite_id,
            "source_results_sha256": scan.results.source_sha256,
            "rewritten_results_sha256": scan.results.retained_sha256,
            "removed_row_evidence_ids": sorted(removed_ids),
        }
        changed |= _merge_unique(
            entry.setdefault("result_rewrites", []), [rewrite], key="rewrite_id"
        )

    if scan.metadata.targeted:
        assert scan.metadata.payload is not None
        assert scan.metadata.source_text is not None
        assert scan.metadata.source_sha256 is not None
        metadata_evidence = {
            "source_meta_sha256": scan.metadata.source_sha256,
            "raw_json": scan.metadata.source_text,
            "payload": scan.metadata.payload,
            "quarantine_path": _quarantine_relative_path(
                "meta", scan.metadata.source_sha256
            ),
        }
        changed |= _merge_unique(
            entry.setdefault("quarantined_metadata", []),
            [metadata_evidence],
            key="source_meta_sha256",
        )

    if scan.failure.targeted:
        assert scan.failure.payload is not None
        assert scan.failure.source_text is not None
        assert scan.failure.source_sha256 is not None
        failure_evidence = {
            "source_failure_sha256": scan.failure.source_sha256,
            "raw_json": scan.failure.source_text,
            "payload": scan.failure.payload,
            "quarantine_path": _quarantine_relative_path(
                "failure", scan.failure.source_sha256
            ),
        }
        changed |= _merge_unique(
            entry.setdefault("quarantined_failures", []),
            [failure_evidence],
            key="source_failure_sha256",
        )

    next_qid, all_present, inference_error = _infer_next_missing_qid(
        cell, scan.results.retained_qids
    )
    inferred = {
        "inferred_next_missing_qid": next_qid,
        "all_expected_qids_present_after_migration": all_present,
        "qid_inference_error": inference_error,
    }
    for key, value in inferred.items():
        if entry.get(key) != value:
            entry[key] = value
            changed = True
    return changed


def _move_to_evidence_quarantine(
    source: Path, *, label: str, source_sha256: str
) -> Path:
    destination = source.parent / _quarantine_relative_path(label, source_sha256)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise MigrationError(f"refusing unsafe quarantine destination: {destination}")
        if _path_sha256(destination) != source_sha256:
            raise MigrationError(
                f"quarantine checksum collision for {source}: {destination}"
            )
        # A prior interrupted application already preserved these exact bytes.  Removing
        # the reappeared source is safe because both the quarantine and incident copies
        # have been verified.
        if _path_sha256(source) != source_sha256:
            raise MigrationError(f"source changed before quarantine: {source}")
        io.remove_file(source)
        return destination
    if _path_sha256(source) != source_sha256:
        raise MigrationError(f"source changed before quarantine: {source}")
    io.move_file(source, destination)
    if _path_sha256(destination) != source_sha256:
        raise MigrationError(f"quarantine verification failed: {destination}")
    return destination


@contextmanager
def _run_lock(run_root: Path) -> Iterator[None]:
    path = run_root / RUN_LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise MigrationError(f"another migration owns the run lock: {run_root}") from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _empty_report(
    run_root: Path, snapshot: ManifestSnapshot, *, apply: bool
) -> dict[str, Any]:
    cells_root = run_root / "cells"
    present = (
        {path.name for path in cells_root.iterdir() if path.is_dir()}
        if cells_root.is_dir()
        else set()
    )
    return {
        "run_root": str(run_root),
        "manifest_sha256": snapshot.sha256,
        "manifest_cells": len(snapshot.cells),
        "stale_unmanifested_dirs": len(present - set(snapshot.ids)),
        "applied": apply,
        "candidate_cells": 0,
        "old_explicit_result_rows": 0,
        "old_explicit_metadata": 0,
        "generation_truncation_failures": 0,
        "malformed_result_lines_retained": 0,
        "results_rewritten": 0,
        "metadata_quarantined": 0,
        "failures_quarantined": 0,
        "active_or_locked_cells_skipped": 0,
        "unsafe_or_unreadable_cells_skipped": 0,
        "incident_sha256": None,
        "current_incident_path": None,
        "current_incident_sha256": None,
        "sealed_historical_incident_path": None,
        "sealed_historical_incident_sha256": None,
        "sealed_historical_artifact_schema_version": None,
        "errors": [],
    }


def _record_scan_counts(report: dict[str, Any], scan: CellScan) -> None:
    report["candidate_cells"] += 1
    report["old_explicit_result_rows"] += len(scan.results.removed_rows)
    report["old_explicit_metadata"] += int(scan.metadata.targeted)
    report["generation_truncation_failures"] += int(scan.failure.targeted)
    report["malformed_result_lines_retained"] += scan.results.malformed_lines_retained


def _apply_locked_cell(
    *,
    run_root: Path,
    snapshot: ManifestSnapshot,
    incident: dict[str, Any],
    cell: ExperimentCell,
    manifest_index: int,
    cell_directory: Path,
    scan: CellScan,
    report: dict[str, Any],
    incident_path: Path,
) -> bool:
    incident_changed = _merge_cell_evidence(
        incident,
        cell=cell,
        manifest_index=manifest_index,
        scan=scan,
    )
    if incident_changed or not incident_path.exists():
        io.write_json(incident_path, incident)
    # Evidence must be durable and verifiably readable before any source artifact is
    # replaced or moved.  Keep newly merged in-memory evidence if this verification
    # fails: no source is mutated, and a later cell/restart may safely retry the same
    # idempotent incident write without copying an arbitrarily large evidence document.
    verified = _load_incident(
        run_root,
        snapshot,
        path=incident_path,
        allowed_artifact_schemas=frozenset({ARTIFACT_SCHEMA_VERSION}),
    )
    if verified != incident:
        raise MigrationError(f"incident verification failed: {incident_path}")

    changed = False
    if scan.results.removed_rows:
        results_path = cell_directory / RESULTS_FILENAME
        if _path_sha256(results_path) != scan.results.source_sha256:
            raise MigrationError(f"results changed while cell lock was held: {results_path}")
        assert scan.results.retained_text is not None
        io.atomic_write_text(results_path, scan.results.retained_text)
        if _path_sha256(results_path) != scan.results.retained_sha256:
            raise MigrationError(f"results rewrite verification failed: {results_path}")
        report["results_rewritten"] += 1
        changed = True
    if scan.metadata.targeted:
        assert scan.metadata.source_sha256 is not None
        _move_to_evidence_quarantine(
            cell_directory / META_FILENAME,
            label="meta",
            source_sha256=scan.metadata.source_sha256,
        )
        report["metadata_quarantined"] += 1
        changed = True
    if scan.failure.targeted:
        assert scan.failure.source_sha256 is not None
        _move_to_evidence_quarantine(
            cell_directory / FAILURE_FILENAME,
            label="failure",
            source_sha256=scan.failure.source_sha256,
        )
        report["failures_quarantined"] += 1
        changed = True
    return changed


def migrate_run(run_root: str | Path, *, apply: bool = False) -> dict[str, Any]:
    """Audit or migrate one checksum-frozen run and return a machine-readable report."""
    root = Path(run_root).resolve()
    snapshot = _require_frozen_manifest(root)
    report = _empty_report(root, snapshot, apply=apply)
    legacy_incident_path = root / INCIDENT_FILENAME
    historical_incident_path: Path | None = None
    if legacy_incident_path.exists():
        legacy_incident = _load_incident(
            root,
            snapshot,
            path=legacy_incident_path,
            allowed_artifact_schemas=(
                SEALED_HISTORICAL_ARTIFACT_SCHEMA_VERSIONS
                | frozenset({ARTIFACT_SCHEMA_VERSION})
            ),
        )
        if legacy_incident.get("current_artifact_schema_version") == ARTIFACT_SCHEMA_VERSION:
            incident_path = legacy_incident_path
            incident = legacy_incident
        else:
            # Schema-4 incidents are immutable scientific history.  Schema-5 may
            # validate them but never append new evidence to them; an unexpected new
            # schema-2/3 artifact is written to a generation-addressed incident.
            historical_incident_path = legacy_incident_path
            incident_path = root / CURRENT_INCIDENT_FILENAME
            incident = _load_incident(
                root,
                snapshot,
                path=incident_path,
                allowed_artifact_schemas=frozenset({ARTIFACT_SCHEMA_VERSION}),
            )
            report["sealed_historical_incident_path"] = str(historical_incident_path)
            report["sealed_historical_incident_sha256"] = _path_sha256(
                historical_incident_path
            )
            report["sealed_historical_artifact_schema_version"] = legacy_incident[
                "current_artifact_schema_version"
            ]
    else:
        incident_path = legacy_incident_path
        incident = _new_incident(root, snapshot)
    report["current_incident_path"] = str(incident_path)
    cells_root = root / "cells"

    def process_cells() -> None:
        for index, cell in enumerate(snapshot.cells):
            cdir = cells_root / cell.cell_id
            # Missing manifest cells are intentionally not created.  Symlinked cell
            # directories are never traversed because they could escape manifest scope.
            if not cdir.exists():
                continue
            if cdir.is_symlink() or not cdir.is_dir():
                report["unsafe_or_unreadable_cells_skipped"] += 1
                report["errors"].append(f"unsafe cell directory: {cdir}")
                continue
            try:
                active = is_cell_active(cdir)
            except OSError as exc:
                report["unsafe_or_unreadable_cells_skipped"] += 1
                report["errors"].append(f"{cell.cell_id}: cannot inspect cell lock: {exc}")
                continue
            if active:
                report["active_or_locked_cells_skipped"] += 1
                continue
            try:
                preliminary = _scan_cell(cdir)
            except (OSError, MigrationError) as exc:
                report["unsafe_or_unreadable_cells_skipped"] += 1
                report["errors"].append(f"{cell.cell_id}: {exc}")
                continue
            if not preliminary.targeted:
                continue
            if not apply:
                _record_scan_counts(report, preliminary)
                continue
            try:
                with cell_lock(cdir, blocking=False):
                    # The preliminary scan only avoids creating lock files for irrelevant
                    # cells.  All decisions and checksums used for mutation are recomputed
                    # after acquiring the authoritative cell lock.
                    locked_scan = _scan_cell(cdir)
                    if not locked_scan.targeted:
                        continue
                    _record_scan_counts(report, locked_scan)
                    _apply_locked_cell(
                        run_root=root,
                        snapshot=snapshot,
                        incident=incident,
                        cell=cell,
                        manifest_index=index,
                        cell_directory=cdir,
                        scan=locked_scan,
                        report=report,
                        incident_path=incident_path,
                    )
            except CellLockUnavailable:
                report["active_or_locked_cells_skipped"] += 1
            except (OSError, MigrationError) as exc:
                report["unsafe_or_unreadable_cells_skipped"] += 1
                report["errors"].append(f"{cell.cell_id}: {exc}")

    if apply:
        with _run_lock(root):
            process_cells()
    else:
        process_cells()

    if incident_path.is_file() and not incident_path.is_symlink():
        report["current_incident_sha256"] = _path_sha256(incident_path)
    report["incident_sha256"] = (
        report["current_incident_sha256"]
        or report["sealed_historical_incident_sha256"]
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--results-root",
        default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform evidence-backed, lock-protected migration (default: dry run)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    results_root = Path(args.results_root)
    reports: list[dict[str, Any]] = []
    failed = False
    for run_id in args.run_id:
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise MigrationError(f"run id must be one path component: {run_id!r}")
        try:
            report = migrate_run(results_root / run_id, apply=args.apply)
        except (OSError, MigrationError) as exc:
            report = {
                "run_root": str((results_root / run_id).resolve()),
                "applied": args.apply,
                "errors": [str(exc)],
            }
        failed |= bool(report["errors"])
        reports.append(report)
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
