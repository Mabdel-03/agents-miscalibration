#!/usr/bin/env python3
"""Idempotently consolidate legacy sweep evidence in the only safe recovery order.

Dry-run is the default.  ``--apply`` first proves maintenance, scheduler quiescence,
unheld cell locks, and a fully verified pre-repair byte snapshot.  It then performs and
revalidates, in order: the 22 response-protocol incident resets, 47 schema-1 checkpoint
migrations, three obsolete permanent-ledger archives, and the canonical repair audit.
Named checksummed reports are published for typed schema-5 readiness evidence, followed
by ``LEGACY_CLEANUP_COMPLETE.json`` only after the exact semantic baseline passes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    CompletionState,
    get_completion_status,
    is_cell_active,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from scripts.archive_checkpoint_permanent_failures import (  # noqa: E402
    PermanentFailureArchiveError,
    TARGET_CELL_CONFIG_HASHES,
    archive_run,
)
from scripts.archive_protocol_incidents import archive_reset_run  # noqa: E402
from scripts.audit_repair_run import audit_run  # noqa: E402
from scripts.create_recovery_snapshot import (  # noqa: E402
    SnapshotError,
    verify_snapshot,
)
from scripts.migrate_generation_protocol import migrate_run as validate_generation  # noqa: E402
from scripts.migrate_qid_checkpoints import migrate_run as migrate_checkpoints  # noqa: E402


RUN_IDS = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
)
EXPECTED_MANIFEST_CELLS = 22_680
EXPECTED_COMPLETE_CELLS = 740
EXPECTED_ACTIVE_QIDS = 888_068
EXPECTED_SEALED_QIDS = 1_064
EXPECTED_FROZEN_BASELINE_QIDS = 889_132
EXPECTED_INCIDENTS = 22
EXPECTED_NEWLY_SEALED_INCIDENTS = 8
EXPECTED_HISTORICAL_INCIDENTS = 14
EXPECTED_HISTORICAL_QIDS = 355
EXPECTED_MIGRATED_CHECKPOINTS = 47
EXPECTED_PERMANENT_ARCHIVES = 3
COMPLETE_FILENAME = "LEGACY_CLEANUP_COMPLETE.json"
CONSOLIDATION_LOCK_FILENAME = ".legacy_consolidation.lock"

_COMPLETE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "passed",
        "snapshot_id",
        "precheck",
        "mutation_precheck",
        "migration_metrics",
        "evidence_accounting",
        "semantic_metrics",
        "artifacts",
    }
)

_EXPECTED_REPORT_FILENAMES = {
    "response_incident_archive_report": "response_incident_archive_report.json",
    "checkpoint_migration_report": "checkpoint_migration_report.json",
    "permanent_ledger_archive_report": "permanent_ledger_archive_report.json",
    "legacy_semantic_audit_report": "legacy_semantic_audit_report.json",
}


class ConsolidationError(RuntimeError):
    """Legacy consolidation cannot prove an exact safe transition."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read one regular JSON object and reject aliases or duplicate keys."""

    if path.is_symlink() or not path.is_file():
        raise ConsolidationError(f"{label} is missing or unsafe: {path}")

    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ConsolidationError(
                    f"{label} has duplicate key {key!r}: {path}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ConsolidationError(f"{label} contains non-finite {token!r}")
            ),
        )
    except ConsolidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsolidationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsolidationError(f"{label} must be one JSON object: {path}")
    return value


@contextmanager
def _consolidation_lock(recovery_root: Path):
    """Serialize the complete legacy mutation transaction across nodes."""

    recovery_root.mkdir(parents=True, exist_ok=True)
    path = recovery_root / CONSOLIDATION_LOCK_FILENAME
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ConsolidationError(
                    f"another legacy consolidation owns {path}"
                ) from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _artifact(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path.resolve()), "sha256": _sha256(path)}


def _publish_report(
    operations_root: Path,
    filename: str,
    payload: Mapping[str, Any],
    *,
    referenced_paths: Sequence[tuple[str, Path]] = (),
) -> Path:
    report = dict(payload)
    report["referenced_artifacts"] = [
        _artifact(name, path) for name, path in referenced_paths
    ]
    path = operations_root / filename
    _atomic_json(path, report)
    return path


def _verify_external_snapshot_attestation(
    *, results_root: Path, recovery_root: Path, attestation_path: Path
) -> dict[str, Any]:
    snapshot_root = (recovery_root / "pre_repair").resolve()
    verified = verify_snapshot(snapshot_root)
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsolidationError(f"cannot load pre-repair attestation: {exc}") from exc
    if (
        not isinstance(attestation, dict)
        or attestation.get("kind") != "recovery_snapshot_external_attestation"
        or attestation.get("passed") is not True
        or Path(str(attestation.get("snapshot_root", ""))).resolve() != snapshot_root
        or attestation.get("snapshot_id") != verified["snapshot_id"]
    ):
        raise ConsolidationError("pre-repair snapshot attestation has the wrong identity")
    controls = attestation.get("control_artifacts")
    if not isinstance(controls, dict) or len(controls) != 5:
        raise ConsolidationError("pre-repair snapshot control envelope is incomplete")
    for filename, record in controls.items():
        path = snapshot_root / filename
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.stat().st_size != record.get("size")
            or _sha256(path) != record.get("sha256")
        ):
            raise ConsolidationError(f"pre-repair snapshot control drift: {filename}")
    catalog_path = snapshot_root / "SNAPSHOT_CATALOG.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsolidationError(f"cannot load pre-repair snapshot catalog: {exc}") from exc
    expected_sources = [
        {"name": run_id, "path": str((results_root / run_id).resolve())}
        for run_id in RUN_IDS
    ] + [
        {
            "name": "dispatcher_v3",
            "path": str((results_root / ".dispatcher-v3").resolve()),
        },
        {
            "name": "recovery_evidence",
            "path": str((recovery_root / "pre_repair_inventory").resolve()),
        },
    ]
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or catalog.get("copy_contract")
        != "independent_regular_files_no_hardlinks_no_symlinks"
        or catalog.get("sources") != expected_sources
        or catalog.get("snapshot_id") != verified["snapshot_id"]
        or catalog.get("source_inventory_sha256")
        != verified["snapshot_inventory_sha256"]
        or catalog.get("snapshot_inventory_sha256")
        != verified["snapshot_inventory_sha256"]
    ):
        raise ConsolidationError(
            "pre-repair snapshot does not cover the exact five recovery sources"
        )
    try:
        marker = json.loads(
            (snapshot_root / "SNAPSHOT_COMPLETE.json").read_text(encoding="utf-8")
        )
        completed_at = marker["completed_at"]
        completed_timestamp = datetime.fromisoformat(completed_at).timestamp()
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ConsolidationError(
            "pre-repair snapshot has no stable completion timestamp"
        ) from exc
    if completed_timestamp < 0:
        raise ConsolidationError("pre-repair snapshot completion timestamp is invalid")
    return dict(attestation) | {
        "snapshot_completed_at": completed_at,
        "snapshot_completed_timestamp": completed_timestamp,
    }


def _maintenance_precheck(results_root: Path, recovery_root: Path) -> dict[str, Any]:
    interlock_path = recovery_root / "MAINTENANCE_INTERLOCK.json"
    try:
        interlock = json.loads(interlock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsolidationError(f"maintenance interlock is unavailable: {exc}") from exc
    if (
        interlock.get("desired_state") != "maintenance"
        or interlock.get("admission_enabled") is not False
        or tuple(interlock.get("retired_run_ids", ())) != RUN_IDS
    ):
        raise ConsolidationError("maintenance interlock does not freeze the exact legacy runs")
    scheduler_user = os.environ.get("USER")
    if not scheduler_user:
        raise ConsolidationError("USER is unset; scheduler quiescence cannot be scoped")
    proc = subprocess.run(
        ["squeue", "-u", scheduler_user, "-h", "-r", "-o", "%i|%j|%T|%o"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ConsolidationError(f"squeue failed during maintenance check: {proc.stderr[:500]}")
    unsafe_rows = [
        line
        for line in proc.stdout.splitlines()
        if any(run_id in line for run_id in RUN_IDS)
        or any(prefix in line for prefix in ("asys-cells", "asys-dispatch", "asys-driver", "asys-serve"))
    ]
    if unsafe_rows:
        raise ConsolidationError(
            "legacy jobs remain live during cleanup: " + "; ".join(unsafe_rows[:10])
        )
    active_locks: list[str] = []
    inspected_cells = 0
    for run_id in RUN_IDS:
        cells_root = results_root / run_id / "cells"
        if not cells_root.is_dir():
            continue
        for cell_dir in cells_root.iterdir():
            if not cell_dir.is_dir() or cell_dir.is_symlink():
                continue
            inspected_cells += 1
            if is_cell_active(cell_dir):
                active_locks.append(f"{run_id}/{cell_dir.name}")
    if active_locks:
        raise ConsolidationError(f"held legacy cell locks remain: {active_locks[:10]}")
    return {
        "maintenance_interlock": _artifact("maintenance_interlock", interlock_path),
        "scheduler_rows": len(proc.stdout.splitlines()),
        "legacy_jobs": 0,
        "inspected_cell_directories": inspected_cells,
        "held_cell_locks": 0,
    }


def _sealed_incident_summary(
    results_root: Path, *, pre_repair_snapshot_root: Path
) -> dict[str, int]:
    incidents = newly_sealed_incidents = historical_incidents = 0
    newly_sealed_rows = historical_rows = 0
    for run_id in RUN_IDS:
        root = (
            results_root
            / run_id
            / "incidents"
            / "discarded_server_response_protocol_v4"
        )
        if not root.exists():
            continue
        for archive in sorted(path for path in root.iterdir() if path.is_dir()):
            incident_path = archive / "incident.json"
            marker_path = archive / "reset_complete.json"
            if not incident_path.is_file() or not marker_path.is_file():
                raise ConsolidationError(f"unsealed protocol incident: {archive}")
            incident_bytes = incident_path.read_bytes()
            incident = json.loads(incident_bytes)
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("incident_sha256") != hashlib.sha256(incident_bytes).hexdigest():
                raise ConsolidationError(f"protocol incident marker drifted: {archive}")
            result_records = [
                record
                for record in incident.get("active_artifacts", [])
                if record.get("source_relative_path") == "results.jsonl"
            ]
            if len(result_records) > 1:
                raise ConsolidationError(f"protocol incident has duplicate results preimages: {archive}")
            archive_rows = 0
            if result_records:
                record = result_records[0]
                path = archive / str(record["archived_path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != record.get("size")
                    or _sha256(path) != record.get("sha256")
                ):
                    raise ConsolidationError(f"protocol results preimage drifted: {archive}")
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not value.get("qid"):
                        raise ConsolidationError(
                            f"invalid sealed result {path}:{line_number}"
                        )
                    archive_rows += 1
            snapshot_marker = (
                pre_repair_snapshot_root
                / run_id
                / "incidents"
                / "discarded_server_response_protocol_v4"
                / archive.name
                / "reset_complete.json"
            )
            if snapshot_marker.is_file():
                historical_incidents += 1
                historical_rows += archive_rows
            else:
                newly_sealed_incidents += 1
                newly_sealed_rows += archive_rows
            incidents += 1
    return {
        "protocol_incidents_total": incidents,
        "newly_sealed_incidents": newly_sealed_incidents,
        "sealed_incident_qids": newly_sealed_rows,
        "preexisting_historical_incidents": historical_incidents,
        "preexisting_historical_qids": historical_rows,
        "all_sealed_incident_qids": newly_sealed_rows + historical_rows,
    }


def semantic_audit(
    results_root: Path, *, repair_count: int, check_active: bool = True
) -> dict[str, Any]:
    states: Counter[str] = Counter()
    total_valid = malformed = duplicates = unexpected = invalid = 0
    run_reports: dict[str, Any] = {}
    manifested = 0
    for run_id in RUN_IDS:
        run_root = results_root / run_id
        snapshot = load_manifest(run_root, verify_frozen=True)
        catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
        run_states: Counter[str] = Counter()
        run_valid = 0
        for cell in snapshot.cells:
            questions = catalog.questions_for(cell)
            status = get_completion_status(
                cell,
                run_root / "cells" / cell.cell_id,
                expected_qids=tuple(question.qid for question in questions),
                expected_questions=questions,
                verified_benchmark_contracts=catalog.frozen,
                verified_manifest=catalog.snapshot,
                check_active=check_active,
            )
            states[status.status.value] += 1
            run_states[status.status.value] += 1
            total_valid += status.valid_count
            run_valid += status.valid_count
            malformed += status.malformed_lines
            duplicates += len(status.duplicate_qids)
            unexpected += len(status.unexpected_qids)
            invalid += status.invalid_rows
        manifested += len(snapshot.cells)
        run_reports[run_id] = {
            "manifest_sha256": snapshot.sha256,
            "benchmark_contracts_sha256": catalog.sidecar_sha256,
            "manifest_cells": len(snapshot.cells),
            "active_validated_qids": run_valid,
            "states": dict(sorted(run_states.items())),
        }
    metrics = {
        "complete_cells": states[CompletionState.COMPLETE.value],
        "active_validated_qids": total_valid,
        "corrupt_cells": states[CompletionState.CORRUPT.value],
        "permanent_cells": states[CompletionState.PERMANENT.value],
        "malformed_lines": malformed,
        "duplicate_qids": duplicates,
        "unexpected_qids": unexpected,
        "repair_count": repair_count,
    }
    return {
        "schema_version": 1,
        "kind": "legacy_semantic_audit",
        "passed": (
            manifested == EXPECTED_MANIFEST_CELLS
            and metrics
            == {
                "complete_cells": EXPECTED_COMPLETE_CELLS,
                "active_validated_qids": EXPECTED_ACTIVE_QIDS,
                "corrupt_cells": 0,
                "permanent_cells": 0,
                "malformed_lines": 0,
                "duplicate_qids": 0,
                "unexpected_qids": 0,
                "repair_count": 0,
            }
            and invalid == 0
        ),
        "manifest_cells": manifested,
        "metrics": metrics,
        "invalid_rows": invalid,
        "states": dict(sorted(states.items())),
        "runs": run_reports,
    }


def _validate_recorded_precheck(
    value: object, *, recovery_root: Path, label: str
) -> None:
    expected_fields = {
        "maintenance_interlock",
        "scheduler_rows",
        "legacy_jobs",
        "inspected_cell_directories",
        "held_cell_locks",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ConsolidationError(f"{label} has the wrong fields")
    interlock = recovery_root / "MAINTENANCE_INTERLOCK.json"
    expected_interlock = _artifact("maintenance_interlock", interlock)
    integer_fields = (
        "scheduler_rows",
        "legacy_jobs",
        "inspected_cell_directories",
        "held_cell_locks",
    )
    if value.get("maintenance_interlock") != expected_interlock or any(
        isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
        for field in integer_fields
    ):
        raise ConsolidationError(f"{label} is invalid")
    if (
        value["scheduler_rows"] < 0
        or value["legacy_jobs"] != 0
        or value["inspected_cell_directories"] < 0
        or value["held_cell_locks"] != 0
    ):
        raise ConsolidationError(f"{label} did not prove quiescence")


def _verify_report_graph(
    path: Path, *, operations_root: Path, seen: set[Path]
) -> dict[str, Any]:
    """Verify one immutable consolidation report and its complete local graph."""

    supplied = path
    if supplied.is_symlink() or not supplied.is_file():
        raise ConsolidationError(f"consolidation report is missing or unsafe: {path}")
    resolved = supplied.resolve()
    try:
        resolved.relative_to(operations_root)
    except ValueError as exc:
        raise ConsolidationError(
            f"consolidation report escapes its operation root: {resolved}"
        ) from exc
    info = supplied.stat(follow_symlinks=False)
    if info.st_nlink != 1 or info.st_mode & 0o222:
        raise ConsolidationError(f"consolidation report is not sealed: {path}")
    if resolved in seen:
        raise ConsolidationError(f"duplicate/cyclic consolidation report: {resolved}")
    seen.add(resolved)
    payload = _read_strict_json(resolved, label="consolidation report")
    references = payload.get("referenced_artifacts")
    if references is None:
        return payload
    if not isinstance(references, list):
        raise ConsolidationError(
            f"consolidation report references are not an array: {resolved}"
        )
    names: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "name",
            "path",
            "sha256",
        }:
            raise ConsolidationError(
                f"consolidation report has an invalid reference: {resolved}"
            )
        name = reference.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ConsolidationError(
                f"consolidation report reference names are invalid: {resolved}"
            )
        names.add(name)
        nested = Path(str(reference.get("path", "")))
        if not nested.is_absolute() or _sha256(nested) != reference.get("sha256"):
            raise ConsolidationError(
                f"consolidation report reference drifted: {nested}"
            )
        _verify_report_graph(nested, operations_root=operations_root, seen=seen)
    return payload


def _verify_live_migration_and_permanent_evidence(
    *,
    results_root: Path,
    operations_root: Path,
    checkpoint_report: Mapping[str, Any],
) -> None:
    """Rebind sealed cleanup reports to the live migrated checkpoint/archive bytes.

    The completed-consolidation fast path must remain read-only, but validating only
    the sealed reports would miss later drift in the live checkpoint or permanent
    incident trees.  The apply reports already record every target checkpoint hash;
    compare those hashes to every live schema-2 checkpoint carrying migration history,
    reject any residual schema-1 checkpoint, and invoke the permanent archive's
    idempotent read-only validator.
    """

    references = checkpoint_report.get("referenced_artifacts")
    if not isinstance(references, list) or len(references) != len(RUN_IDS):
        raise ConsolidationError(
            "checkpoint migration report does not bind every legacy run"
        )
    expected: dict[tuple[str, str], str] = {}
    referenced_runs: set[str] = set()
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "name",
            "path",
            "sha256",
        }:
            raise ConsolidationError("checkpoint migration reference is invalid")
        run_id = reference.get("name")
        if run_id not in RUN_IDS or run_id in referenced_runs:
            raise ConsolidationError("checkpoint migration run references drifted")
        referenced_runs.add(str(run_id))
        detail_path = Path(str(reference.get("path", "")))
        try:
            detail_path.resolve().relative_to(operations_root)
        except (OSError, ValueError) as exc:
            raise ConsolidationError(
                "checkpoint migration detail escapes the operation root"
            ) from exc
        detail = _read_strict_json(
            detail_path, label=f"checkpoint migration detail for {run_id}"
        )
        if detail.get("schema_version") != 1 or detail.get("run_id") != run_id:
            raise ConsolidationError("checkpoint migration detail identity drifted")
        applied = detail.get("apply")
        if not isinstance(applied, dict):
            raise ConsolidationError("checkpoint migration apply evidence is invalid")
        rows = applied.get("would_change_checkpoints")
        if not isinstance(rows, list):
            raise ConsolidationError("checkpoint migration hash plan is invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ConsolidationError("checkpoint migration hash row is invalid")
            relative = row.get("checkpoint_relative_path")
            checksum = row.get("after_sha256")
            parts = Path(relative).parts if isinstance(relative, str) else ()
            if (
                not isinstance(relative, str)
                or len(parts) != 4
                or parts[0] != "cells"
                or parts[2] != ".qid_checkpoints"
                or not parts[1]
                or not parts[3].endswith(".json")
                or not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ConsolidationError("checkpoint migration target identity is invalid")
            key = (str(run_id), relative)
            if key in expected:
                raise ConsolidationError("duplicate checkpoint migration target")
            expected[key] = checksum
    if referenced_runs != set(RUN_IDS) or len(expected) != EXPECTED_MIGRATED_CHECKPOINTS:
        raise ConsolidationError(
            "checkpoint migration report does not bind the exact 47 live targets"
        )

    observed: dict[tuple[str, str], str] = {}
    for run_id in RUN_IDS:
        run_root = (results_root / run_id).resolve()
        cells_root = run_root / "cells"
        if cells_root.is_symlink() or not cells_root.is_dir():
            raise ConsolidationError(f"legacy cells root is unsafe: {cells_root}")
        for cell_directory in sorted(cells_root.iterdir(), key=lambda path: path.name):
            if cell_directory.is_symlink() or not cell_directory.is_dir():
                raise ConsolidationError(
                    f"legacy cell directory is unsafe: {cell_directory}"
                )
            checkpoint_root = cell_directory / ".qid_checkpoints"
            if not checkpoint_root.exists():
                continue
            if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                raise ConsolidationError(
                    f"checkpoint directory is unsafe: {checkpoint_root}"
                )
            for path in sorted(checkpoint_root.iterdir(), key=lambda item: item.name):
                if path.is_symlink() or not path.is_file():
                    raise ConsolidationError(f"checkpoint artifact is unsafe: {path}")
                info = path.stat(follow_symlinks=False)
                if info.st_nlink != 1:
                    raise ConsolidationError(f"checkpoint has a hardlink alias: {path}")
                payload = _read_strict_json(path, label="live checkpoint")
                if payload.get("schema_version") != 2:
                    raise ConsolidationError(
                        f"legacy checkpoint migration is incomplete: {path}"
                    )
                history = payload.get("migration_history")
                if not isinstance(history, list):
                    raise ConsolidationError(
                        f"legacy checkpoint migration history is invalid: {path}"
                    )
                if not history:
                    continue
                relative = path.relative_to(run_root).as_posix()
                key = (run_id, relative)
                checksum = _sha256(path)
                if expected.get(key) != checksum:
                    raise ConsolidationError(
                        f"live migrated checkpoint differs from sealed evidence: {path}"
                    )
                observed[key] = checksum
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ConsolidationError(
            "live migrated checkpoint inventory drifted: "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )

    try:
        permanent = archive_run(results_root / RUN_IDS[0], apply=False)
    except (OSError, PermanentFailureArchiveError, ValueError) as exc:
        raise ConsolidationError(
            f"live permanent incident archive no longer validates: {exc}"
        ) from exc
    if (
        permanent.get("target_count") != EXPECTED_PERMANENT_ARCHIVES
        or permanent.get("counts") != {
            "already_reset": EXPECTED_PERMANENT_ARCHIVES
        }
        or permanent.get("would_change_cells") != []
    ):
        raise ConsolidationError("live permanent incident archive inventory drifted")


def _verify_completed_consolidation(
    *, results_root: Path, recovery_root: Path, snapshot_id: str
) -> dict[str, Any]:
    """Revalidate a marker-last cleanup without invoking any mutation primitive."""

    marker_path = recovery_root / COMPLETE_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ConsolidationError(
            f"legacy cleanup completion marker is missing or unsafe: {marker_path}"
        )
    marker_info = marker_path.stat(follow_symlinks=False)
    if marker_info.st_nlink != 1 or marker_info.st_mode & 0o222:
        raise ConsolidationError("legacy cleanup completion marker is not sealed")
    marker = _read_strict_json(marker_path, label="legacy cleanup completion marker")
    if set(marker) != _COMPLETE_FIELDS:
        raise ConsolidationError("legacy cleanup completion marker has wrong fields")
    if (
        type(marker.get("schema_version")) is not int
        or marker.get("schema_version") != 1
        or marker.get("status") != "complete"
        or marker.get("passed") is not True
        or marker.get("snapshot_id") != snapshot_id
    ):
        raise ConsolidationError("legacy cleanup completion marker has wrong identity")
    _validate_recorded_precheck(
        marker.get("precheck"), recovery_root=recovery_root, label="initial precheck"
    )
    _validate_recorded_precheck(
        marker.get("mutation_precheck"),
        recovery_root=recovery_root,
        label="mutation precheck",
    )
    expected_migrations = {
        "protocol_incidents_total": EXPECTED_INCIDENTS,
        "protocol_already_reset": EXPECTED_INCIDENTS,
        "sealed_incident_qids": EXPECTED_SEALED_QIDS,
        "migrated_checkpoints": EXPECTED_MIGRATED_CHECKPOINTS,
        "remaining_schema1_checkpoints": 0,
        "permanent_ledgers_archived": EXPECTED_PERMANENT_ARCHIVES,
        "unresolved_permanent_ledgers": 0,
    }
    expected_semantic = {
        "complete_cells": EXPECTED_COMPLETE_CELLS,
        "active_validated_qids": EXPECTED_ACTIVE_QIDS,
        "corrupt_cells": 0,
        "permanent_cells": 0,
        "malformed_lines": 0,
        "duplicate_qids": 0,
        "unexpected_qids": 0,
        "repair_count": 0,
    }
    if marker.get("migration_metrics") != expected_migrations:
        raise ConsolidationError("legacy cleanup migration metrics drifted")
    if marker.get("semantic_metrics") != expected_semantic:
        raise ConsolidationError("legacy cleanup semantic metrics drifted")
    evidence = marker.get("evidence_accounting")
    if not isinstance(evidence, dict) or evidence != {
        "newly_sealed_incidents": EXPECTED_NEWLY_SEALED_INCIDENTS,
        "newly_sealed_qids": EXPECTED_SEALED_QIDS,
        "preexisting_historical_incidents": EXPECTED_HISTORICAL_INCIDENTS,
        "preexisting_historical_qids": EXPECTED_HISTORICAL_QIDS,
        "all_sealed_incident_qids": EXPECTED_SEALED_QIDS
        + EXPECTED_HISTORICAL_QIDS,
        "frozen_baseline_validated_qids": EXPECTED_FROZEN_BASELINE_QIDS,
    }:
        raise ConsolidationError("legacy cleanup evidence accounting drifted")

    operations = (recovery_root / "operations" / "legacy_consolidation").resolve()
    records = marker.get("artifacts")
    if not isinstance(records, list) or len(records) != len(_EXPECTED_REPORT_FILENAMES):
        raise ConsolidationError("legacy cleanup artifact inventory is incomplete")
    by_name = {
        row.get("name"): row for row in records if isinstance(row, dict)
    }
    if len(by_name) != len(records) or set(by_name) != set(_EXPECTED_REPORT_FILENAMES):
        raise ConsolidationError("legacy cleanup artifact names drifted")
    reports: dict[str, dict[str, Any]] = {}
    seen: set[Path] = set()
    for name, filename in _EXPECTED_REPORT_FILENAMES.items():
        row = by_name[name]
        if set(row) != {"name", "path", "sha256"}:
            raise ConsolidationError(f"legacy cleanup artifact {name} has wrong fields")
        path = operations / filename
        if row.get("path") != str(path) or row.get("sha256") != _sha256(path):
            raise ConsolidationError(f"legacy cleanup artifact drifted: {name}")
        reports[name] = _verify_report_graph(
            path, operations_root=operations, seen=seen
        )
    response = reports["response_incident_archive_report"]
    checkpoint = reports["checkpoint_migration_report"]
    permanent = reports["permanent_ledger_archive_report"]
    semantic_report = reports["legacy_semantic_audit_report"]
    if any(
        report.get("schema_version") != 1 or report.get("passed") is not True
        for report in (response, checkpoint, permanent, semantic_report)
    ):
        raise ConsolidationError("legacy cleanup report identity drifted")
    for key in (
        "protocol_incidents_total",
        "protocol_already_reset",
        "sealed_incident_qids",
    ):
        if response.get(key) != expected_migrations[key]:
            raise ConsolidationError("response cleanup report metrics drifted")
    for key in (
        "migrated_checkpoints",
        "remaining_schema1_checkpoints",
    ):
        if checkpoint.get(key) != expected_migrations[key]:
            raise ConsolidationError("checkpoint cleanup report metrics drifted")
    if checkpoint.get("coordinates_preserved") is not True:
        raise ConsolidationError("checkpoint coordinate preservation drifted")
    for key in (
        "permanent_ledgers_archived",
        "unresolved_permanent_ledgers",
    ):
        if permanent.get(key) != expected_migrations[key]:
            raise ConsolidationError("permanent cleanup report metrics drifted")
    if (
        semantic_report.get("kind") != "legacy_semantic_audit"
        or semantic_report.get("manifest_cells") != EXPECTED_MANIFEST_CELLS
        or semantic_report.get("invalid_rows") != 0
        or semantic_report.get("metrics") != expected_semantic
    ):
        raise ConsolidationError("legacy semantic report drifted")
    _verify_live_migration_and_permanent_evidence(
        results_root=results_root,
        operations_root=operations,
        checkpoint_report=checkpoint,
    )
    # Completed roots may already have been retired read-only.  Activity is proven by
    # the recorded maintenance boundaries; reopening 0444 lock files O_RDWR would be
    # both unnecessary and impossible after retirement.
    live_semantic = semantic_audit(
        results_root, repair_count=0, check_active=False
    )
    if live_semantic.get("passed") is not True or live_semantic.get(
        "metrics"
    ) != expected_semantic:
        raise ConsolidationError("completed legacy cleanup no longer validates")
    return marker | {
        "status": "already_complete",
        "completion_marker": str(marker_path),
    }


def _pending_response_resets(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Return only cells whose sealed response transaction still changes live state."""

    pending: dict[str, set[str]] = {run_id: set() for run_id in RUN_IDS}
    for report in reports:
        run_id = report.get("run_id")
        if run_id not in pending:
            raise ConsolidationError(f"response preflight returned unknown run: {run_id!r}")
        rows = report.get("cells")
        if not isinstance(rows, list):
            raise ConsolidationError(f"response preflight has invalid cells: {run_id}")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("cell_id"), str):
                raise ConsolidationError(f"response preflight has invalid cell row: {run_id}")
            plan = row.get("mutation_plan")
            if isinstance(plan, Mapping) and plan.get("would_change") is True:
                pending[run_id].add(str(row["cell_id"]))
    return pending


def _checkpoint_projection_for_permanent_ledgers(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Select exact target checkpoint hashes needed by the permanent-ledger preflight."""

    projection: dict[str, list[Mapping[str, Any]]] = {}
    targets = set(TARGET_CELL_CONFIG_HASHES)
    for report in reports:
        rows = report.get("would_change_checkpoints")
        if not isinstance(rows, list):
            raise ConsolidationError("checkpoint preflight has an invalid mutation plan")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("cell_id"), str):
                raise ConsolidationError("checkpoint preflight has an invalid mutation row")
            cell_id = str(row["cell_id"])
            if cell_id in targets:
                projection.setdefault(cell_id, []).append(row)
    return projection


def _permanent_plan_identity(report: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    rows = report.get("cells")
    if not isinstance(rows, list):
        raise ConsolidationError("permanent-ledger report has invalid cells")
    return {
        str(row["cell_id"]): (
            row.get("before_sha256"),
            row.get("after_sha256"),
            row.get("archive"),
            row.get("checkpoint_count"),
            row.get("checkpoint_evidence"),
            row.get("incident_sha256"),
        )
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }


def _repair_mutation_count(reports: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(report["results_would_rewrite"])
        + int(report["metadata_would_quarantine"])
        + int(report["failure_would_quarantine"])
        for report in reports
    )


def _consolidate_unlocked(
    *,
    results_root: Path,
    recovery_root: Path,
    pre_repair_attestation: Path,
    apply: bool,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    recovery_root = recovery_root.expanduser().resolve()
    operations = recovery_root / "operations" / "legacy_consolidation"
    snapshot_attestation = _verify_external_snapshot_attestation(
        results_root=results_root,
        recovery_root=recovery_root,
        attestation_path=pre_repair_attestation.expanduser().resolve(),
    )
    complete_path = recovery_root / COMPLETE_FILENAME
    if complete_path.exists() or complete_path.is_symlink():
        return _verify_completed_consolidation(
            results_root=results_root,
            recovery_root=recovery_root,
            snapshot_id=str(snapshot_attestation["snapshot_id"]),
        )
    precheck = _maintenance_precheck(results_root, recovery_root)
    dispatcher = results_root / ".dispatcher-v3"
    run_roots = [results_root / run_id for run_id in RUN_IDS]

    # Validate every run and every dispatcher join before the first live artifact is
    # touched.  This keeps a late unmappable traceback or manifest drift from turning
    # an otherwise safe, resumable transaction into an avoidable partial rollout.
    generation_reports = [validate_generation(root, apply=False) for root in run_roots]
    if any(report["errors"] for report in generation_reports):
        raise ConsolidationError("sealed generation-protocol history did not validate")
    response_preflight = [
        archive_reset_run(root, dispatcher_state_dirs=[dispatcher], apply=False)
        for root in run_roots
    ]
    if any(report["errors"] for report in response_preflight):
        raise ConsolidationError("response incident recovery preflight reported errors")
    preflight_total = sum(
        report["affected_manifest_cells"] for report in response_preflight
    )
    if preflight_total != EXPECTED_INCIDENTS:
        raise ConsolidationError(
            f"response incident preflight expected {EXPECTED_INCIDENTS} cells, "
            f"found {preflight_total}"
        )

    # Complete the entire read-only mutation plan before the first response reset.
    # Pending response incidents project their cells as absent, which is the exact
    # post-reset state in which checkpoint migration and canonical repair will run.
    pending_response = _pending_response_resets(response_preflight)
    checkpoint_migration_timestamp = float(
        snapshot_attestation["snapshot_completed_timestamp"]
    )
    checkpoint_preflight = [
        migrate_checkpoints(
            root,
            apply=False,
            migration_timestamp=checkpoint_migration_timestamp,
            excluded_cell_ids=pending_response[root.name],
        )
        for root in run_roots
    ]
    if any(report["errors"] for report in checkpoint_preflight):
        raise ConsolidationError("checkpoint migration preflight reported errors")
    checkpoint_accounted = sum(
        report["schema1_candidates"] + report["schema2_already_migrated"]
        for report in checkpoint_preflight
    )
    if checkpoint_accounted != EXPECTED_MIGRATED_CHECKPOINTS:
        raise ConsolidationError(
            "checkpoint migration preflight did not account for the exact legacy set: "
            f"{checkpoint_accounted}"
        )

    checkpoint_projection = _checkpoint_projection_for_permanent_ledgers(
        checkpoint_preflight
    )
    permanent_preflight = archive_run(
        run_roots[0],
        apply=False,
        projected_checkpoint_plans=checkpoint_projection,
    )
    permanent_accounted = sum(
        permanent_preflight["counts"].get(status, 0)
        for status in ("would_reset", "already_reset")
    )
    if permanent_accounted != EXPECTED_PERMANENT_ARCHIVES:
        raise ConsolidationError(
            "permanent-ledger preflight did not account for the exact legacy set: "
            f"{permanent_accounted}"
        )

    repair_preflight = [
        audit_run(
            root,
            apply=False,
            excluded_cell_ids=pending_response[root.name],
        )
        for root in run_roots
    ]
    would_repair = _repair_mutation_count(repair_preflight)
    if would_repair != 0:
        raise ConsolidationError(
            f"canonical repair preflight unexpectedly found {would_repair} mutations"
        )

    if not apply:
        return {
            "status": "dry_run",
            "precheck": precheck,
            "snapshot_id": snapshot_attestation["snapshot_id"],
            "snapshot_completed_at": snapshot_attestation["snapshot_completed_at"],
            "generation_protocol": generation_reports,
            "response_incidents": response_preflight,
            "pending_response_reset_cells": {
                run_id: sorted(cell_ids)
                for run_id, cell_ids in pending_response.items()
            },
            "checkpoint_migration_timestamp": checkpoint_migration_timestamp,
            "checkpoint_migrations": checkpoint_preflight,
            "permanent_ledgers": permanent_preflight,
            "canonical_repair": repair_preflight,
            "would_repair_artifacts": would_repair,
        }

    # The complete dry-run above can be expensive.  Re-query both scheduler truth and
    # every advisory lock at the final boundary before the first response artifact is
    # changed, while the process-wide transaction lock is still held.
    mutation_precheck = _maintenance_precheck(results_root, recovery_root)

    response_apply: list[dict[str, Any]] = []
    if apply:
        response_apply = [
            archive_reset_run(root, dispatcher_state_dirs=[dispatcher], apply=True)
            for root in run_roots
        ]
        if any(report["errors"] for report in response_apply):
            raise ConsolidationError("response incident recovery reported errors")
        for before, after in zip(response_preflight, response_apply, strict=True):
            before_plans = {
                row["cell_id"]: row.get("mutation_plan")
                for row in before["cells"]
            }
            after_plans = {
                row["cell_id"]: row.get("mutation_plan")
                for row in after["cells"]
            }
            if before_plans != after_plans:
                raise ConsolidationError(
                    f"response mutation plan drifted for {before['run_root']}"
                )
    response_reports = [
        archive_reset_run(root, dispatcher_state_dirs=[dispatcher], apply=False)
        for root in run_roots
    ] if apply else response_preflight
    if any(report["errors"] for report in response_reports):
        raise ConsolidationError("response incident recovery verification reported errors")
    response_total = sum(report["affected_manifest_cells"] for report in response_reports)
    response_already = sum(
        report["outcomes"]["already_reset"] for report in response_reports
    )
    sealed = _sealed_incident_summary(
        results_root, pre_repair_snapshot_root=recovery_root / "pre_repair"
    )
    if apply and (
        response_total != EXPECTED_INCIDENTS
        or response_already != EXPECTED_INCIDENTS
        or sealed["protocol_incidents_total"] != EXPECTED_INCIDENTS
        or sealed["newly_sealed_incidents"] != EXPECTED_NEWLY_SEALED_INCIDENTS
        or sealed["sealed_incident_qids"] != EXPECTED_SEALED_QIDS
        or sealed["preexisting_historical_incidents"] != EXPECTED_HISTORICAL_INCIDENTS
        or sealed["preexisting_historical_qids"] != EXPECTED_HISTORICAL_QIDS
        or sealed["all_sealed_incident_qids"]
        != EXPECTED_SEALED_QIDS + EXPECTED_HISTORICAL_QIDS
        or EXPECTED_ACTIVE_QIDS + sealed["sealed_incident_qids"]
        != EXPECTED_FROZEN_BASELINE_QIDS
    ):
        raise ConsolidationError(
            f"response incident acceptance failed: total={response_total}, "
            f"already={response_already}, sealed={sealed}"
        )

    operations.mkdir(parents=True, exist_ok=True)
    response_run_paths: list[tuple[str, Path]] = []
    for preflight, applied_report, report in zip(
        response_preflight, response_apply, response_reports, strict=True
    ):
        path = operations / f"response.{report['run_id']}.json"
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "run_id": report["run_id"],
                "preflight": preflight,
                "apply": applied_report,
                "verify": report,
            },
        )
        response_run_paths.append((report["run_id"], path))
    response_path = _publish_report(
        operations,
        "response_incident_archive_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "protocol_incidents_total": response_total,
            "protocol_already_reset": response_already,
            "newly_sealed_incidents": sealed["newly_sealed_incidents"],
            "sealed_incident_qids": sealed["sealed_incident_qids"],
            "preexisting_historical_incidents": sealed[
                "preexisting_historical_incidents"
            ],
            "preexisting_historical_qids": sealed["preexisting_historical_qids"],
            "all_sealed_incident_qids": sealed["all_sealed_incident_qids"],
            "frozen_baseline_validated_qids": EXPECTED_FROZEN_BASELINE_QIDS,
        },
        referenced_paths=response_run_paths,
    )

    # Re-read the now-live post-response state and require it to equal the complete
    # projection made before mutation.  The sealed snapshot completion timestamp makes
    # a standalone dry-run and a later apply produce byte-identical target checkpoints.
    checkpoint_live_preflight = [
        migrate_checkpoints(
            root,
            apply=False,
            migration_timestamp=checkpoint_migration_timestamp,
        )
        for root in run_roots
    ]
    if any(report["errors"] for report in checkpoint_live_preflight):
        raise ConsolidationError("post-response checkpoint preflight reported errors")
    live_checkpoint_accounted = sum(
        report["schema1_candidates"] + report["schema2_already_migrated"]
        for report in checkpoint_live_preflight
    )
    if live_checkpoint_accounted != EXPECTED_MIGRATED_CHECKPOINTS:
        raise ConsolidationError(
            "post-response checkpoint preflight did not account for the exact legacy set: "
            f"{live_checkpoint_accounted}"
        )
    for projected, live in zip(
        checkpoint_preflight, checkpoint_live_preflight, strict=True
    ):
        if projected["would_change_checkpoints"] != live["would_change_checkpoints"]:
            raise ConsolidationError(
                f"checkpoint projection drifted after response reset for {live['run_root']}"
            )
    checkpoint_apply = [
        migrate_checkpoints(
            root,
            apply=True,
            migration_timestamp=checkpoint_migration_timestamp,
        )
        for root in run_roots
    ]
    if any(report["errors"] for report in checkpoint_apply):
        raise ConsolidationError("checkpoint migration reported errors")
    for before, after in zip(checkpoint_live_preflight, checkpoint_apply, strict=True):
        if before["would_change_checkpoints"] != after["would_change_checkpoints"]:
            raise ConsolidationError(
                f"checkpoint mutation plan drifted for {before['run_root']}"
            )
    checkpoint_verify = [
        migrate_checkpoints(
            root,
            apply=False,
            migration_timestamp=checkpoint_migration_timestamp,
        )
        for root in run_roots
    ]
    if any(report["errors"] for report in checkpoint_verify):
        raise ConsolidationError("checkpoint migration verification reported errors")
    migrated = sum(report["schema2_already_migrated"] for report in checkpoint_verify)
    remaining_schema1 = sum(report["schema1_candidates"] for report in checkpoint_verify)
    if migrated != EXPECTED_MIGRATED_CHECKPOINTS or remaining_schema1 != 0:
        raise ConsolidationError(
            f"checkpoint acceptance failed: migrated={migrated}, schema1={remaining_schema1}"
        )
    checkpoint_run_paths: list[tuple[str, Path]] = []
    for preflight, applied_report, report in zip(
        checkpoint_live_preflight, checkpoint_apply, checkpoint_verify, strict=True
    ):
        path = operations / f"checkpoint.{Path(report['run_root']).name}.json"
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "run_id": Path(report["run_root"]).name,
                "planned_migration_timestamp": checkpoint_migration_timestamp,
                "preflight": preflight,
                "apply": applied_report,
                "verify": report,
            },
        )
        checkpoint_run_paths.append((Path(report["run_root"]).name, path))
    checkpoint_path = _publish_report(
        operations,
        "checkpoint_migration_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "migrated_checkpoints": migrated,
            "remaining_schema1_checkpoints": remaining_schema1,
            "coordinates_preserved": True,
        },
        referenced_paths=checkpoint_run_paths,
    )

    permanent_live_preflight = archive_run(run_roots[0], apply=False)
    live_permanent_accounted = sum(
        permanent_live_preflight["counts"].get(status, 0)
        for status in ("would_reset", "already_reset")
    )
    if live_permanent_accounted != EXPECTED_PERMANENT_ARCHIVES:
        raise ConsolidationError(
            "post-checkpoint permanent-ledger preflight did not account for the exact "
            f"legacy set: {live_permanent_accounted}"
        )
    if _permanent_plan_identity(permanent_preflight) != _permanent_plan_identity(
        permanent_live_preflight
    ):
        raise ConsolidationError(
            "permanent-ledger projection drifted after checkpoint migration"
        )
    permanent_apply = archive_run(run_roots[0], apply=True)
    planned_permanent = _permanent_plan_identity(permanent_live_preflight)
    applied_permanent = _permanent_plan_identity(permanent_apply)
    if planned_permanent != applied_permanent:
        raise ConsolidationError("permanent-ledger mutation plan drifted during apply")
    permanent_verify = archive_run(run_roots[0], apply=False)
    permanent_archived = permanent_verify["counts"].get("already_reset", 0)
    if permanent_archived != EXPECTED_PERMANENT_ARCHIVES:
        raise ConsolidationError(
            f"permanent-ledger acceptance failed: archived={permanent_archived}"
        )
    permanent_detail = operations / "permanent.full_sweep_v1.json"
    _atomic_json(
        permanent_detail,
        {
            "schema_version": 1,
            "run_id": run_roots[0].name,
            "projected_preflight": permanent_preflight,
            "preflight": permanent_live_preflight,
            "apply": permanent_apply,
            "verify": permanent_verify,
        },
    )
    permanent_path = _publish_report(
        operations,
        "permanent_ledger_archive_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "permanent_ledgers_archived": permanent_archived,
            "unresolved_permanent_ledgers": 0,
        },
        referenced_paths=(("full_sweep_v1", permanent_detail),),
    )

    repair_dry = [audit_run(root, apply=False) for root in run_roots]
    live_would_repair = _repair_mutation_count(repair_dry)
    if live_would_repair != 0:
        raise ConsolidationError(
            "post-migration canonical repair unexpectedly found "
            f"{live_would_repair} mutations"
        )
    if any(
        projected.get("would_change_cells") != live.get("would_change_cells")
        for projected, live in zip(repair_preflight, repair_dry, strict=True)
    ):
        raise ConsolidationError("canonical repair projection drifted after migration")
    repair_apply = [audit_run(root, apply=True) for root in run_roots]
    repair_count = sum(
        report["results_rewritten"]
        + report["meta_quarantined"]
        + report["failure_quarantined"]
        for report in repair_apply
    )
    if repair_count != 0:
        raise ConsolidationError(f"canonical repair unexpectedly changed {repair_count} artifacts")

    semantic = semantic_audit(results_root, repair_count=repair_count)
    if semantic["passed"] is not True:
        raise ConsolidationError(f"legacy semantic acceptance failed: {semantic['metrics']}")
    semantic_path = _publish_report(
        operations,
        "legacy_semantic_audit_report.json",
        semantic,
    )
    migration_metrics = {
        "protocol_incidents_total": response_total,
        "protocol_already_reset": response_already,
        "sealed_incident_qids": sealed["sealed_incident_qids"],
        "migrated_checkpoints": migrated,
        "remaining_schema1_checkpoints": remaining_schema1,
        "permanent_ledgers_archived": permanent_archived,
        "unresolved_permanent_ledgers": 0,
    }
    complete = {
        "schema_version": 1,
        "status": "complete",
        "passed": True,
        "snapshot_id": snapshot_attestation["snapshot_id"],
        "precheck": precheck,
        "mutation_precheck": mutation_precheck,
        "migration_metrics": migration_metrics,
        "evidence_accounting": {
            "newly_sealed_incidents": sealed["newly_sealed_incidents"],
            "newly_sealed_qids": sealed["sealed_incident_qids"],
            "preexisting_historical_incidents": sealed[
                "preexisting_historical_incidents"
            ],
            "preexisting_historical_qids": sealed[
                "preexisting_historical_qids"
            ],
            "all_sealed_incident_qids": sealed["all_sealed_incident_qids"],
            "frozen_baseline_validated_qids": EXPECTED_FROZEN_BASELINE_QIDS,
        },
        "semantic_metrics": semantic["metrics"],
        "artifacts": [
            _artifact("response_incident_archive_report", response_path),
            _artifact("checkpoint_migration_report", checkpoint_path),
            _artifact("permanent_ledger_archive_report", permanent_path),
            _artifact("legacy_semantic_audit_report", semantic_path),
        ],
    }
    _atomic_json(complete_path, complete)
    return complete | {"completion_marker": str(complete_path)}


def consolidate(
    *,
    results_root: Path,
    recovery_root: Path,
    pre_repair_attestation: Path,
    apply: bool,
) -> dict[str, Any]:
    """Plan or execute one globally serialized legacy cleanup transaction."""

    resolved_recovery = recovery_root.expanduser().resolve()
    if not apply:
        return _consolidate_unlocked(
            results_root=results_root,
            recovery_root=resolved_recovery,
            pre_repair_attestation=pre_repair_attestation,
            apply=False,
        )
    with _consolidation_lock(resolved_recovery):
        return _consolidate_unlocked(
            results_root=results_root,
            recovery_root=resolved_recovery,
            pre_repair_attestation=pre_repair_attestation,
            apply=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--pre-repair-attestation", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = consolidate(
            results_root=args.results_root,
            recovery_root=args.recovery_root,
            pre_repair_attestation=args.pre_repair_attestation,
            apply=args.apply,
        )
    except (OSError, ValueError, SnapshotError, ConsolidationError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
