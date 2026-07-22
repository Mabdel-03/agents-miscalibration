"""Ordered legacy consolidation keeps the frozen-baseline incident boundary exact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import consolidate_legacy_recovery as recovery
from scripts import create_recovery_snapshot as snapshot


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

    monkeypatch.setattr(
        recovery,
        "_verify_external_snapshot_attestation",
        lambda **_kwargs: {"snapshot_id": "snapshot-id", "passed": True},
    )
    monkeypatch.setattr(
        recovery,
        "_maintenance_precheck",
        lambda *_args: {"legacy_jobs": 0, "held_cell_locks": 0},
    )

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

    def checkpoints(run_root: Path, *, apply: bool, migration_timestamp: float):
        nonlocal checkpoint_applied
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

    def permanent(_run_root: Path, *, apply: bool):
        nonlocal permanent_applied
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
        lambda *_args, apply: {
            "results_would_rewrite": 0,
            "metadata_would_quarantine": 0,
            "failure_would_quarantine": 0,
            "results_rewritten": 0,
            "meta_quarantined": 0,
            "failure_quarantined": 0,
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
    assert all(event.startswith("generation:") for event in events[:3])
    assert sum(event.startswith("response:read:") for event in events[:first_apply]) == 3
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
