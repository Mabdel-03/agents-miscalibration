"""Ordered legacy consolidation keeps the frozen-baseline incident boundary exact."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path

import pytest

from scripts import consolidate_legacy_recovery as recovery
from scripts import create_recovery_snapshot as snapshot


def test_apply_refuses_concurrent_global_consolidation_lock(tmp_path: Path) -> None:
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    lock_path = recovery_root / recovery.CONSOLIDATION_LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(recovery.ConsolidationError, match="another legacy"):
            recovery.consolidate(
                results_root=tmp_path / "results",
                recovery_root=recovery_root,
                pre_repair_attestation=tmp_path / "attestation.json",
                apply=True,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_incident(
    results_root: Path,
    *,
    run_id: str,
    cell_id: str,
    qids: tuple[str, ...],
    historical_snapshot: Path | None,
) -> None:
    archive = (
        results_root
        / run_id
        / "incidents"
        / "discarded_server_response_protocol_v4"
        / cell_id
    )
    archive.mkdir(parents=True)
    rows = "".join(json.dumps({"qid": qid}) + "\n" for qid in qids).encode()
    preimage = archive / "results.jsonl.preimage"
    preimage.write_bytes(rows)
    incident = {
        "active_artifacts": [
            {
                "source_relative_path": "results.jsonl",
                "archived_path": preimage.name,
                "size": len(rows),
                "sha256": hashlib.sha256(rows).hexdigest(),
            }
        ]
    }
    incident_bytes = (json.dumps(incident, sort_keys=True) + "\n").encode()
    (archive / "incident.json").write_bytes(incident_bytes)
    marker = {
        "incident_sha256": hashlib.sha256(incident_bytes).hexdigest(),
    }
    (archive / "reset_complete.json").write_text(json.dumps(marker) + "\n")
    if historical_snapshot is not None:
        marker_path = (
            historical_snapshot
            / run_id
            / "incidents"
            / "discarded_server_response_protocol_v4"
            / cell_id
            / "reset_complete.json"
        )
        marker_path.parent.mkdir(parents=True)
        marker_path.write_text("historical membership\n")


def test_sealed_incidents_distinguish_preexisting_history_from_new_baseline(tmp_path):
    results = tmp_path / "results"
    snapshot = tmp_path / "pre_repair"
    _write_incident(
        results,
        run_id=recovery.RUN_IDS[0],
        cell_id="historical-cell",
        qids=("old-1", "old-2"),
        historical_snapshot=snapshot,
    )
    _write_incident(
        results,
        run_id=recovery.RUN_IDS[1],
        cell_id="newly-sealed-cell",
        qids=("new-1", "new-2", "new-3"),
        historical_snapshot=None,
    )

    observed = recovery._sealed_incident_summary(
        results, pre_repair_snapshot_root=snapshot
    )

    assert observed == {
        "protocol_incidents_total": 2,
        "newly_sealed_incidents": 1,
        "sealed_incident_qids": 3,
        "preexisting_historical_incidents": 1,
        "preexisting_historical_qids": 2,
        "all_sealed_incident_qids": 5,
    }


def test_sealed_incident_without_marker_fails_closed(tmp_path):
    archive = (
        tmp_path
        / recovery.RUN_IDS[0]
        / "incidents"
        / "discarded_server_response_protocol_v4"
        / "unsealed"
    )
    archive.mkdir(parents=True)
    (archive / "incident.json").write_text("{}\n")

    with pytest.raises(recovery.ConsolidationError, match="unsealed"):
        recovery._sealed_incident_summary(
            tmp_path, pre_repair_snapshot_root=tmp_path / "snapshot"
        )


def _exact_snapshot_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    results = tmp_path / "results"
    recovery_root = results / "recovery" / "schema5-v1"
    inventory = recovery_root / "pre_repair_inventory"
    inventory.mkdir(parents=True)
    (inventory / "scheduler.txt").write_text("no legacy jobs\n")
    sources = []
    for run_id in recovery.RUN_IDS:
        run = results / run_id
        run.mkdir(parents=True)
        (run / "cells.sha256").write_text(f"{run_id}\n")
        sources.append(snapshot.Source(run_id, run.resolve()))
    dispatcher = results / ".dispatcher-v3"
    dispatcher.mkdir()
    (dispatcher / "ledger.json").write_text("{}\n")
    sources.extend(
        (
            snapshot.Source("dispatcher_v3", dispatcher.resolve()),
            snapshot.Source("recovery_evidence", inventory.resolve()),
        )
    )
    root = recovery_root / "pre_repair"
    snapshot.create_snapshot(root, sources, apply=True)
    attestation = recovery_root / "pre_repair.attestation.json"
    snapshot.write_snapshot_attestation(root, attestation)
    return results, recovery_root, attestation


def test_pre_repair_attestation_must_cover_exact_five_sources(tmp_path):
    results, recovery_root, attestation = _exact_snapshot_fixture(tmp_path)

    observed = recovery._verify_external_snapshot_attestation(
        results_root=results,
        recovery_root=recovery_root,
        attestation_path=attestation,
    )

    assert observed["passed"] is True
    assert observed["snapshot_completed_at"]
    assert observed["snapshot_completed_timestamp"] > 0


def test_pre_repair_attestation_rejects_wrong_source_namespace(tmp_path):
    results, recovery_root, _attestation = _exact_snapshot_fixture(tmp_path)
    wrong_root = results / "recovery-wrong" / "schema5-v1"
    wrong_inventory = wrong_root / "pre_repair_inventory"
    wrong_inventory.mkdir(parents=True)
    (wrong_inventory / "scheduler.txt").write_text("wrong evidence\n")
    sources = [
        snapshot.Source(run_id, (results / run_id).resolve())
        for run_id in recovery.RUN_IDS
    ] + [
        snapshot.Source("dispatcher_v3", (results / ".dispatcher-v3").resolve()),
        snapshot.Source("wrong_evidence", wrong_inventory.resolve()),
    ]
    wrong_snapshot = wrong_root / "pre_repair"
    snapshot.create_snapshot(wrong_snapshot, sources, apply=True)
    wrong_attestation = wrong_root / "pre_repair.attestation.json"
    snapshot.write_snapshot_attestation(wrong_snapshot, wrong_attestation)

    with pytest.raises(recovery.ConsolidationError, match="exact five"):
        recovery._verify_external_snapshot_attestation(
            results_root=results,
            recovery_root=wrong_root,
            attestation_path=wrong_attestation,
        )


def test_apply_preflights_and_reuses_exact_mutation_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    recovery_root = tmp_path / "recovery"
    events: list[str] = []
    response_applied = False
    checkpoint_applied = False
    permanent_applied = False
    checkpoint_timestamps: list[float] = []
    recovery_root.mkdir()
    interlock = recovery_root / "MAINTENANCE_INTERLOCK.json"
    interlock.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        recovery,
        "_verify_external_snapshot_attestation",
        lambda **_kwargs: {
            "snapshot_id": "snapshot-id",
            "passed": True,
            "snapshot_completed_at": "2026-07-22T00:00:00+00:00",
            "snapshot_completed_timestamp": 1_774_396_800.0,
        },
    )
    def maintenance(*_args):
        events.append("maintenance")
        return {
            "maintenance_interlock": recovery._artifact(
                "maintenance_interlock", interlock
            ),
            "scheduler_rows": 0,
            "legacy_jobs": 0,
            "inspected_cell_directories": 0,
            "held_cell_locks": 0,
        }

    monkeypatch.setattr(recovery, "_maintenance_precheck", maintenance)

    def generation(run_root: Path, *, apply: bool):
        assert apply is False
        events.append(f"generation:{run_root.name}")
        return {"errors": []}

    monkeypatch.setattr(recovery, "validate_generation", generation)

    def response(run_root: Path, *, dispatcher_state_dirs, apply: bool):
        nonlocal response_applied
        del dispatcher_state_dirs
        events.append(f"response:{'apply' if apply else 'read'}:{run_root.name}")
        count = recovery.EXPECTED_INCIDENTS if run_root.name == recovery.RUN_IDS[0] else 0
        plans = [
            {
                "cell_id": f"cell-{index}",
                "mutation_plan": {
                    "would_change": not response_applied,
                    "active_artifact_changes": [
                        {
                            "before_sha256": f"{index:064x}",
                            "after_sha256": None,
                        }
                    ],
                },
            }
            for index in range(count)
        ]
        if apply:
            response_applied = True
        already = count if response_applied else recovery.EXPECTED_HISTORICAL_INCIDENTS
        return {
            "run_root": str(run_root),
            "run_id": run_root.name,
            "affected_manifest_cells": count,
            "outcomes": {"already_reset": already},
            "cells": plans,
            "errors": [],
        }

    monkeypatch.setattr(recovery, "archive_reset_run", response)
    monkeypatch.setattr(
        recovery,
        "_sealed_incident_summary",
        lambda *_args, **_kwargs: {
            "protocol_incidents_total": recovery.EXPECTED_INCIDENTS,
            "newly_sealed_incidents": recovery.EXPECTED_NEWLY_SEALED_INCIDENTS,
            "sealed_incident_qids": recovery.EXPECTED_SEALED_QIDS,
            "preexisting_historical_incidents": recovery.EXPECTED_HISTORICAL_INCIDENTS,
            "preexisting_historical_qids": recovery.EXPECTED_HISTORICAL_QIDS,
            "all_sealed_incident_qids": (
                recovery.EXPECTED_SEALED_QIDS + recovery.EXPECTED_HISTORICAL_QIDS
            ),
        },
    )

    checkpoint_plan = [
        {
            "cell_id": f"checkpoint-cell-{index}",
            "before_sha256": f"{index:064x}",
            "after_sha256": f"{index + 100:064x}",
        }
        for index in range(recovery.EXPECTED_MIGRATED_CHECKPOINTS)
    ]

    def checkpoints(
        run_root: Path,
        *,
        apply: bool,
        migration_timestamp: float,
        excluded_cell_ids=None,
    ):
        nonlocal checkpoint_applied
        del excluded_cell_ids
        checkpoint_timestamps.append(migration_timestamp)
        is_primary = run_root.name == recovery.RUN_IDS[0]
        plans = checkpoint_plan if is_primary and not checkpoint_applied else []
        candidates = len(plans)
        already = (
            recovery.EXPECTED_MIGRATED_CHECKPOINTS
            if is_primary and checkpoint_applied
            else 0
        )
        report = {
            "run_root": str(run_root),
            "errors": [],
            "schema1_candidates": candidates,
            "schema2_already_migrated": already,
            "would_change_checkpoints": plans,
        }
        if apply and is_primary:
            checkpoint_applied = True
        return report

    monkeypatch.setattr(recovery, "migrate_checkpoints", checkpoints)

    permanent_cells = [
        {
            "cell_id": f"permanent-{index}",
            "before_sha256": f"{index + 200:064x}",
            "after_sha256": None,
            "archive": f"archive-{index}",
        }
        for index in range(recovery.EXPECTED_PERMANENT_ARCHIVES)
    ]

    def permanent(_run_root: Path, *, apply: bool, projected_checkpoint_plans=None):
        nonlocal permanent_applied
        del projected_checkpoint_plans
        if apply:
            permanent_applied = True
            counts = {"reset": recovery.EXPECTED_PERMANENT_ARCHIVES}
        elif permanent_applied:
            counts = {"already_reset": recovery.EXPECTED_PERMANENT_ARCHIVES}
        else:
            counts = {"would_reset": recovery.EXPECTED_PERMANENT_ARCHIVES}
        return {"counts": counts, "cells": permanent_cells}

    monkeypatch.setattr(recovery, "archive_run", permanent)
    monkeypatch.setattr(
        recovery,
        "audit_run",
        lambda *_args, apply, excluded_cell_ids=None: {
            "results_would_rewrite": 0,
            "metadata_would_quarantine": 0,
            "failure_would_quarantine": 0,
            "results_rewritten": 0,
            "meta_quarantined": 0,
            "failure_quarantined": 0,
            "would_change_cells": [],
        },
    )
    semantic = {
        "schema_version": 1,
        "kind": "legacy_semantic_audit",
        "passed": True,
        "manifest_cells": recovery.EXPECTED_MANIFEST_CELLS,
        "metrics": {
            "complete_cells": recovery.EXPECTED_COMPLETE_CELLS,
            "active_validated_qids": recovery.EXPECTED_ACTIVE_QIDS,
            "corrupt_cells": 0,
            "permanent_cells": 0,
            "malformed_lines": 0,
            "duplicate_qids": 0,
            "unexpected_qids": 0,
            "repair_count": 0,
        },
        "invalid_rows": 0,
        "states": {},
        "runs": {},
    }
    monkeypatch.setattr(recovery, "semantic_audit", lambda *_args, **_kwargs: semantic)

    completed = recovery.consolidate(
        results_root=results,
        recovery_root=recovery_root,
        pre_repair_attestation=tmp_path / "attestation.json",
        apply=True,
    )

    first_apply = next(index for index, event in enumerate(events) if ":apply:" in event)
    assert events[0] == "maintenance"
    assert all(event.startswith("generation:") for event in events[1:4])
    assert sum(event.startswith("response:read:") for event in events[:first_apply]) == 3
    assert events[first_apply - 1] == "maintenance"
    assert len(set(checkpoint_timestamps)) == 1
    assert completed["status"] == "complete"
    assert completed["migration_metrics"] == {
        "protocol_incidents_total": 22,
        "protocol_already_reset": 22,
        "sealed_incident_qids": 1_064,
        "migrated_checkpoints": 47,
        "remaining_schema1_checkpoints": 0,
        "permanent_ledgers_archived": 3,
        "unresolved_permanent_ledgers": 0,
    }
    assert completed["evidence_accounting"]["newly_sealed_incidents"] == 8
    assert completed["evidence_accounting"]["preexisting_historical_incidents"] == 14
    assert completed["evidence_accounting"]["frozen_baseline_validated_qids"] == 889_132

    marker_hash = recovery._sha256(recovery_root / recovery.COMPLETE_FILENAME)
    operation_hashes = {
        path: recovery._sha256(path)
        for path in (recovery_root / "operations" / "legacy_consolidation").iterdir()
        if path.is_file()
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed consolidation invoked a mutation primitive")

    monkeypatch.setattr(
        recovery,
        "_verify_live_migration_and_permanent_evidence",
        lambda **_kwargs: None,
    )
    for name in (
        "validate_generation",
        "archive_reset_run",
        "migrate_checkpoints",
        "archive_run",
        "audit_run",
    ):
        monkeypatch.setattr(recovery, name, forbidden)
    again = recovery.consolidate(
        results_root=results,
        recovery_root=recovery_root,
        pre_repair_attestation=tmp_path / "attestation.json",
        apply=True,
    )
    assert again["status"] == "already_complete"
    assert recovery._sha256(recovery_root / recovery.COMPLETE_FILENAME) == marker_hash
    assert {
        path: recovery._sha256(path) for path in operation_hashes
    } == operation_hashes

    response_summary = (
        recovery_root
        / "operations"
        / "legacy_consolidation"
        / "response_incident_archive_report.json"
    )
    response_summary.chmod(0o644)
    response_summary.write_text('{"passed":false}\n', encoding="utf-8")
    with pytest.raises(recovery.ConsolidationError, match="artifact drifted"):
        recovery.consolidate(
            results_root=results,
            recovery_root=recovery_root,
            pre_repair_attestation=tmp_path / "attestation.json",
            apply=True,
        )


def test_dry_run_plans_every_stage_without_any_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    recovery_root = tmp_path / "recovery"
    calls: list[tuple[str, bool]] = []
    timestamp = 1_774_396_800.0
    monkeypatch.setattr(
        recovery,
        "_verify_external_snapshot_attestation",
        lambda **_kwargs: {
            "snapshot_id": "snapshot-id",
            "snapshot_completed_at": "2026-07-22T00:00:00+00:00",
            "snapshot_completed_timestamp": timestamp,
        },
    )
    monkeypatch.setattr(
        recovery,
        "_maintenance_precheck",
        lambda *_args: {"legacy_jobs": 0, "held_cell_locks": 0},
    )

    def generation(root: Path, *, apply: bool):
        calls.append((f"generation:{root.name}", apply))
        return {"errors": [], "sealed_history": True}

    monkeypatch.setattr(recovery, "validate_generation", generation)

    def response(root: Path, *, dispatcher_state_dirs, apply: bool):
        del dispatcher_state_dirs
        calls.append((f"response:{root.name}", apply))
        count = recovery.EXPECTED_INCIDENTS if root.name == recovery.RUN_IDS[0] else 0
        return {
            "run_root": str(root),
            "run_id": root.name,
            "affected_manifest_cells": count,
            "errors": [],
            "cells": [
                {
                    "cell_id": f"incident-{index}",
                    "mutation_plan": {
                        "would_change": True,
                        "active_artifact_changes": [
                            {
                                "before_sha256": f"{index:064x}",
                                "after_sha256": None,
                            }
                        ],
                    },
                }
                for index in range(count)
            ],
        }

    monkeypatch.setattr(recovery, "archive_reset_run", response)

    checkpoint_rows = [
        {
            "cell_id": f"checkpoint-{index}",
            "before_sha256": f"{index + 100:064x}",
            "after_sha256": f"{index + 200:064x}",
            "checkpoint_relative_path": f"cells/checkpoint-{index}/.qid_checkpoints/{index:064x}.json",
            "before_size": 10,
            "after_size": 20,
            "after_schema_version": 2,
        }
        for index in range(recovery.EXPECTED_MIGRATED_CHECKPOINTS)
    ]

    def checkpoints(
        root: Path,
        *,
        apply: bool,
        migration_timestamp: float,
        excluded_cell_ids=None,
    ):
        calls.append((f"checkpoint:{root.name}", apply))
        assert migration_timestamp == timestamp
        assert isinstance(excluded_cell_ids, set)
        primary = root.name == recovery.RUN_IDS[0]
        return {
            "run_root": str(root),
            "errors": [],
            "schema1_candidates": len(checkpoint_rows) if primary else 0,
            "schema2_already_migrated": 0,
            "would_change_checkpoints": checkpoint_rows if primary else [],
        }

    monkeypatch.setattr(recovery, "migrate_checkpoints", checkpoints)

    permanent_rows = [
        {
            "cell_id": cell_id,
            "before_sha256": f"{index + 500:064x}",
            "after_sha256": None,
            "archive": f"archive-{index}",
        }
        for index, cell_id in enumerate(recovery.TARGET_CELL_CONFIG_HASHES)
    ]

    def permanent(_root: Path, *, apply: bool, projected_checkpoint_plans=None):
        calls.append(("permanent", apply))
        assert isinstance(projected_checkpoint_plans, dict)
        return {
            "counts": {"would_reset": recovery.EXPECTED_PERMANENT_ARCHIVES},
            "cells": permanent_rows,
        }

    monkeypatch.setattr(recovery, "archive_run", permanent)

    def repair(root: Path, *, apply: bool, excluded_cell_ids=None):
        calls.append((f"repair:{root.name}", apply))
        assert isinstance(excluded_cell_ids, set)
        return {
            "results_would_rewrite": 0,
            "metadata_would_quarantine": 0,
            "failure_would_quarantine": 0,
            "would_change_cells": [],
        }

    monkeypatch.setattr(recovery, "audit_run", repair)

    report = recovery.consolidate(
        results_root=results,
        recovery_root=recovery_root,
        pre_repair_attestation=tmp_path / "attestation.json",
        apply=False,
    )

    assert report["status"] == "dry_run"
    assert report["checkpoint_migration_timestamp"] == timestamp
    assert report["checkpoint_migrations"][0]["would_change_checkpoints"][0][
        "before_sha256"
    ] == checkpoint_rows[0]["before_sha256"]
    assert report["permanent_ledgers"]["cells"][0]["after_sha256"] is None
    assert report["would_repair_artifacts"] == 0
    assert calls
    assert all(applied is False for _stage, applied in calls)
    assert not (recovery_root / "operations").exists()


def _live_migration_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], list[Path]]:
    results = tmp_path / "results"
    operations = tmp_path / "operations"
    operations.mkdir()
    rows_by_run: dict[str, list[dict[str, object]]] = {
        run_id: [] for run_id in recovery.RUN_IDS
    }
    checkpoints: list[Path] = []
    for run_id in recovery.RUN_IDS:
        (results / run_id / "cells").mkdir(parents=True)
    for index in range(recovery.EXPECTED_MIGRATED_CHECKPOINTS):
        run_id = recovery.RUN_IDS[index % len(recovery.RUN_IDS)]
        relative = (
            f"cells/cell-{index:03d}/.qid_checkpoints/{index:064x}.json"
        )
        path = results / run_id / relative
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "migration_history": [
                        {"migration_schema_version": 1, "ordinal": index}
                    ],
                    "payload": index,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoints.append(path)
        rows_by_run[run_id].append(
            {
                "checkpoint_relative_path": relative,
                "after_sha256": recovery._sha256(path),
            }
        )
    references: list[dict[str, object]] = []
    for run_id in recovery.RUN_IDS:
        detail = operations / f"checkpoint.{run_id}.json"
        detail.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "apply": {
                        "would_change_checkpoints": rows_by_run[run_id]
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        references.append(recovery._artifact(run_id, detail.resolve()))
    return results, operations, {"referenced_artifacts": references}, checkpoints


def _completed_permanent_archive_report(*_args, **_kwargs) -> dict[str, object]:
    return {
        "target_count": recovery.EXPECTED_PERMANENT_ARCHIVES,
        "counts": {"already_reset": recovery.EXPECTED_PERMANENT_ARCHIVES},
        "would_change_cells": [],
    }


def test_completed_fast_path_rebinds_live_checkpoint_and_permanent_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, operations, report, _paths = _live_migration_fixture(tmp_path)
    monkeypatch.setattr(recovery, "archive_run", _completed_permanent_archive_report)
    recovery._verify_live_migration_and_permanent_evidence(
        results_root=results,
        operations_root=operations,
        checkpoint_report=report,
    )


def test_completed_fast_path_rejects_migrated_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, operations, report, paths = _live_migration_fixture(tmp_path)
    monkeypatch.setattr(recovery, "archive_run", _completed_permanent_archive_report)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["payload"] = "drifted"
    paths[0].write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(recovery.ConsolidationError, match="differs from sealed"):
        recovery._verify_live_migration_and_permanent_evidence(
            results_root=results,
            operations_root=operations,
            checkpoint_report=report,
        )


def test_completed_fast_path_rejects_checkpoint_hardlink_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, operations, report, paths = _live_migration_fixture(tmp_path)
    monkeypatch.setattr(recovery, "archive_run", _completed_permanent_archive_report)
    os.link(paths[0], paths[0].with_name("f" * 64 + ".json"))
    with pytest.raises(recovery.ConsolidationError, match="hardlink alias"):
        recovery._verify_live_migration_and_permanent_evidence(
            results_root=results,
            operations_root=operations,
            checkpoint_report=report,
        )
