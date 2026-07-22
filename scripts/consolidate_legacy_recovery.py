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
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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
from scripts.archive_checkpoint_permanent_failures import archive_run  # noqa: E402
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
    return attestation


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


def semantic_audit(results_root: Path, *, repair_count: int) -> dict[str, Any]:
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
                check_active=True,
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


def consolidate(
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

    if not apply:
        return {
            "status": "dry_run",
            "precheck": precheck,
            "snapshot_id": snapshot_attestation["snapshot_id"],
            "response_incidents": response_reports,
            "sealed_incidents": sealed,
        }

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

    # One timestamp binds the dry-run target bytes, the archived preimages, and the
    # actual replacements.  Without it, ``migrated_at`` would make after-hashes vary
    # between preflight and apply even though the scientific source was unchanged.
    checkpoint_migration_timestamp = time.time()
    checkpoint_preflight = [
        migrate_checkpoints(
            root,
            apply=False,
            migration_timestamp=checkpoint_migration_timestamp,
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
    for before, after in zip(checkpoint_preflight, checkpoint_apply, strict=True):
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
        checkpoint_preflight, checkpoint_apply, checkpoint_verify, strict=True
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

    permanent_preflight = archive_run(run_roots[0], apply=False)
    permanent_accounted = sum(
        permanent_preflight["counts"].get(status, 0)
        for status in ("would_reset", "already_reset")
    )
    if permanent_accounted != EXPECTED_PERMANENT_ARCHIVES:
        raise ConsolidationError(
            "permanent-ledger preflight did not account for the exact legacy set: "
            f"{permanent_accounted}"
        )
    permanent_apply = archive_run(run_roots[0], apply=True)
    planned_permanent = {
        row["cell_id"]: (
            row["before_sha256"],
            row["after_sha256"],
            row["archive"],
        )
        for row in permanent_preflight["cells"]
    }
    applied_permanent = {
        row["cell_id"]: (
            row["before_sha256"],
            row["after_sha256"],
            row["archive"],
        )
        for row in permanent_apply["cells"]
    }
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
            "preflight": permanent_preflight,
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
    would_repair = sum(
        report["results_would_rewrite"]
        + report["metadata_would_quarantine"]
        + report["failure_would_quarantine"]
        for report in repair_dry
    )
    if would_repair != 0:
        raise ConsolidationError(
            f"canonical repair dry-run unexpectedly found {would_repair} mutations"
        )
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
    complete_path = recovery_root / COMPLETE_FILENAME
    _atomic_json(complete_path, complete)
    return complete | {"completion_marker": str(complete_path)}


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
