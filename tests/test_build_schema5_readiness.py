"""Focused tests for artifact-derived schema-5 readiness publication."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from scripts import build_schema5_readiness as readiness
from scripts import create_recovery_snapshot as recovery_snapshot
from scripts import schema5_email_ack
from slurm import schema5_control as control
from slurm import keepalive
from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving import fleet_transactions as fleet_tx
from agents_scaling.serving.model_contracts import load_model_contracts
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.serving.registry import ServerEntry


IMMUTABLE_SHA = "a" * 64


def _wave_fence_fixture(
    tmp_path: Path,
    *,
    generation: int,
    wave_passed: bool,
) -> tuple[dict, dict, SimpleNamespace]:
    certificate = (
        tmp_path
        / "capacity-generations"
        / f"c{generation:06d}"
        / "PREFLIGHT_CAPACITY_CERTIFICATE.json"
    ).resolve()
    selected = 384 if wave_passed else (278 if generation == 1 else 284)
    shortfall = 384 - selected
    fleet_sha256 = "b" * 64
    contract = SimpleNamespace(
        capacity_generation=generation,
        static_feasibility_certificate_path=certificate,
        static_feasibility_certificate_sha256="c" * 64,
        static_feasibility_certificate_id="d" * 64,
        effective_fleet_contract_sha256=fleet_sha256,
        static_feasibility_wave_passed=wave_passed,
        static_feasibility_selected_cell_count=selected,
        static_feasibility_target_cell_count=384,
        static_feasibility_shortfall_cells=shortfall,
        static_feasibility_configured_client_ceiling=384,
        static_feasibility_certified_saturation_target=selected,
    )
    authority = {
        "static_feasibility_certificate_path": str(certificate),
        "static_feasibility_certificate_sha256": "c" * 64,
        "static_feasibility_certificate_id": "d" * 64,
        "effective_fleet_contract_sha256": fleet_sha256,
    }
    binding = {
        "capacity_generation": generation,
        "sha256": fleet_sha256,
        "protected_capacity": authority,
    }
    control_value: dict = {
        "desired_state": "running",
        "drain_requested": False,
        "capacity": {
            "current_generation": generation,
            "active_transition": None,
        },
        "admission_safety_hold": {
            "active": False,
            "reasons": [],
        },
    }
    return control_value, binding, contract


def test_capacity_wave_fence_accepts_generation_addressed_intermediate_tp2(
    tmp_path: Path,
) -> None:
    control_value, binding, contract = _wave_fence_fixture(
        tmp_path,
        generation=2,
        wave_passed=False,
    )
    summary = readiness._validate_capacity_wave_admission_fence(
        control_value,
        fleet_binding=binding,
        contract=contract,
    )

    assert summary == {
        "capacity_generation": 2,
        "path": str(
            (
                tmp_path
                / "capacity-generations"
                / "c000002"
                / "PREFLIGHT_CAPACITY_CERTIFICATE.json"
            ).resolve()
        ),
        "sha256": "c" * 64,
        "certificate_id": "d" * 64,
        "effective_fleet_contract_sha256": "b" * 64,
        "wave_passed": False,
        "selected_cell_count": 284,
        "target_cell_count": 384,
        "shortfall_cells": 100,
        "configured_client_ceiling": 384,
        "certified_saturation_target": 284,
    }


@pytest.mark.parametrize(
    "drift",
    (
        "certificate-path",
        "fleet-hash",
        "configured-ceiling",
        "saturation-target",
    ),
)
def test_capacity_wave_fence_rejects_unbound_intermediate(
    tmp_path: Path,
    drift: str,
) -> None:
    control_value, binding, contract = _wave_fence_fixture(
        tmp_path,
        generation=2,
        wave_passed=False,
    )
    if drift == "certificate-path":
        binding["protected_capacity"][
            "static_feasibility_certificate_path"
        ] = str((tmp_path / "wrong.json").resolve())
    elif drift == "fleet-hash":
        binding["sha256"] = "e" * 64
    elif drift == "configured-ceiling":
        contract.static_feasibility_configured_client_ceiling = 383
    else:
        contract.static_feasibility_certified_saturation_target = 283

    with pytest.raises(readiness.EvidenceError):
        readiness._validate_capacity_wave_admission_fence(
            control_value,
            fleet_binding=binding,
            contract=contract,
        )


def test_capacity_wave_fence_preserves_exact_generation_one_shortfall(
    tmp_path: Path,
) -> None:
    control_value, binding, contract = _wave_fence_fixture(
        tmp_path,
        generation=1,
        wave_passed=False,
    )
    summary = readiness._validate_capacity_wave_admission_fence(
        control_value,
        fleet_binding=binding,
        contract=contract,
    )
    assert summary["selected_cell_count"] == 278
    assert summary["shortfall_cells"] == 106
    assert summary["wave_passed"] is False
    assert summary["configured_client_ceiling"] == 384
    assert summary["certified_saturation_target"] == 278


def _capacity_readiness_control(tmp_path: Path, *, rollout_generation: int = 3):
    contract = {
        "capacity_generation": 2,
        "path": str((tmp_path / "overlay.json").resolve()),
        "sha256": "b" * 64,
        "marker_path": str((tmp_path / "overlay.complete.json").resolve()),
        "marker_sha256": "c" * 64,
        "protected_capacity_marker_path": str(
            (tmp_path / "PROTECTED_CAPACITY_COMPLETE.json").resolve()
        ),
        "protected_capacity_marker_sha256": "d" * 64,
        "protected_capacity_marker_id": "e" * 64,
        "static_feasibility_certificate_path": str(
            (tmp_path / "STATIC_FEASIBILITY_COMPLETE.json").resolve()
        ),
        "static_feasibility_certificate_sha256": "f" * 64,
        "static_feasibility_certificate_id": "1" * 64,
        "base_fleet_contract_sha256": "2" * 64,
        "additive_overlay_contract_path": str(
            (tmp_path / "overlay.json").resolve()
        ),
        "additive_overlay_contract_sha256": "b" * 64,
        "fleet_id": "schema5-v1",
        "logical_replicas": 23,
        "allocated_gpus": 25,
        "profile_replicas": {
            **control.EXPECTED_FLEET_PROFILES,
            "0.6B": control.EXPECTED_FLEET_PROFILES["0.6B"] + 1,
        },
        "activated_at": "1970-01-01T00:00:01Z",
        "activated_timestamp": 1.0,
    }
    transition_id = "capacity-g000001-to-g000002-fixture"
    return {
        "immutable_sha256": IMMUTABLE_SHA,
        "desired_state": "paused",
        "drain_requested": True,
        "rollout_generation": rollout_generation,
        "immutable": {},
        "capacity": {
            "current_generation": 2,
            "current_contract": contract,
            "active_transition": {
                "transition_id": transition_id,
                "phase": "readiness_pending",
                "to_generation": 2,
                "old_fleet_job_ids": ["900"],
                "last_retirement_error": None,
                "published_timestamp": 2.0,
                "last_fleet_launch_error": None,
                "fleet_launch_completed_timestamp": 3.0,
                "new_contract": contract,
            },
        },
        "admission_safety_hold": {
            "active": True,
            "mode": "operator",
            "reasons": [f"capacity-transition:{transition_id}"],
        },
        "readiness": {"fleet": {"passed": False}},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ref(name: str, path: Path) -> dict[str, str]:
    return {"name": name, "path": str(path.resolve()), "sha256": _sha(path)}


def _snapshot_attestation(tmp_path: Path, name: str) -> Path:
    root = tmp_path / f"{name}.sealed"
    root.mkdir()
    payload_path = root / "payload.txt"
    payload_path.write_text(f"{name}:payload\n", encoding="utf-8")
    inventory = f"{_sha(payload_path)}  payload.txt\n"
    inventory_sha = hashlib.sha256(inventory.encode("utf-8")).hexdigest()
    (root / "SOURCE_INVENTORY.sha256").write_text(inventory, encoding="utf-8")
    (root / "SNAPSHOT_INVENTORY.sha256").write_text(inventory, encoding="utf-8")
    (root / "DIRECTORY_INVENTORY.txt").write_text("", encoding="utf-8")
    snapshot_id = f"snapshot-{name}"
    _write(
        root / "SNAPSHOT_CATALOG.json",
        {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "file_count": 1,
            "directory_count": 0,
            "total_bytes": payload_path.stat().st_size,
            "source_inventory_sha256": inventory_sha,
            "snapshot_inventory_sha256": inventory_sha,
        },
    )
    _write(
        root / "SNAPSHOT_COMPLETE.json",
        {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "completed_at": "2026-07-22T00:00:00Z",
            "file_count": 1,
            "total_bytes": payload_path.stat().st_size,
            "snapshot_inventory_sha256": inventory_sha,
            "verified": True,
            "read_only": True,
        },
    )
    records = {}
    for filename in sorted(control.SNAPSHOT_CONTROL_FILENAMES):
        path = root / filename
        path.chmod(0o444)
        records[filename] = {"sha256": _sha(path), "size": path.stat().st_size}
    payload_path.chmod(0o444)
    root.chmod(0o555)
    return _write(
        tmp_path / f"{name}.attestation.json",
        {
            "schema_version": 1,
            "kind": "recovery_snapshot_external_attestation",
            "passed": True,
            "snapshot_root": str(root.resolve()),
            "snapshot_id": snapshot_id,
            "file_count": 1,
            "total_bytes": payload_path.stat().st_size,
            "control_artifacts": records,
            "attested_at": "2026-07-22T00:00:00Z",
        },
    )


def _legacy_cleanup(
    tmp_path: Path, *, evidence_accounting: dict[str, int] | None = None
) -> Path:
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    operations = recovery / "operations" / "legacy_consolidation"
    response = _write(
        operations / "response_incident_archive_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "protocol_incidents_total": 22,
            "protocol_already_reset": 22,
            "sealed_incident_qids": 1_064,
            "referenced_artifacts": [],
        },
    )
    checkpoint = _write(
        operations / "checkpoint_migration_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "migrated_checkpoints": 47,
            "remaining_schema1_checkpoints": 0,
            "coordinates_preserved": True,
            "referenced_artifacts": [],
        },
    )
    permanent = _write(
        operations / "permanent_ledger_archive_report.json",
        {
            "schema_version": 1,
            "passed": True,
            "permanent_ledgers_archived": 3,
            "unresolved_permanent_ledgers": 0,
            "referenced_artifacts": [],
        },
    )
    semantic = _write(
        operations / "legacy_semantic_audit_report.json",
        {
            "schema_version": 1,
            "kind": "legacy_semantic_audit",
            "passed": True,
            "manifest_cells": 22_680,
            "invalid_rows": 0,
            "metrics": readiness.SEMANTIC_METRICS,
            "referenced_artifacts": [],
        },
    )
    marker = _write(
        recovery / "LEGACY_CLEANUP_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "passed": True,
            "snapshot_id": "pre",
            "migration_metrics": readiness.MIGRATION_OUTER_METRICS,
            "evidence_accounting": (
                readiness.EVIDENCE_ACCOUNTING
                if evidence_accounting is None
                else evidence_accounting
            ),
            "semantic_metrics": readiness.SEMANTIC_METRICS,
            "artifacts": [
                _ref("response_incident_archive_report", response),
                _ref("checkpoint_migration_report", checkpoint),
                _ref("permanent_ledger_archive_report", permanent),
                _ref("legacy_semantic_audit_report", semantic),
            ],
        },
    )
    sources = []
    for run_id in (
        "full_sweep_v1",
        "full_sweep_agent_counts_v1",
        "full_sweep_agent_count_7_v1",
    ):
        root = results / run_id
        root.mkdir(parents=True)
        (root / "evidence.txt").write_text(f"{run_id}\n", encoding="utf-8")
        sources.append(recovery_snapshot.Source(run_id, root.resolve()))
    dispatcher = results / ".dispatcher-v3"
    dispatcher.mkdir()
    (dispatcher / "state.txt").write_text("retired\n", encoding="utf-8")
    sources.extend(
        (
            recovery_snapshot.Source("dispatcher_v3", dispatcher.resolve()),
            recovery_snapshot.Source(
                "legacy_cleanup_evidence", operations.resolve()
            ),
            recovery_snapshot.Source("legacy_cleanup_complete", marker.resolve()),
        )
    )
    sealed = recovery / "legacy_consolidated"
    recovery_snapshot.create_snapshot(sealed, sources, apply=True)
    recovery_snapshot.write_snapshot_attestation(
        sealed, recovery / "legacy_consolidated.attestation.json"
    )
    return marker


def test_snapshot_gate_verifies_both_distinct_sealed_snapshots(tmp_path: Path):
    current = {"immutable_sha256": IMMUTABLE_SHA}
    output = tmp_path / "readiness" / "snapshot.json"
    report = readiness.build_snapshot_gate(
        current,
        pre_repair_attestation=_snapshot_attestation(tmp_path, "pre"),
        legacy_consolidated_attestation=_snapshot_attestation(tmp_path, "legacy"),
        output=output,
    )
    assert report["metrics"]["snapshot_count"] == 2
    assert output.is_file()
    assert [row["name"] for row in report["artifacts"]] == [
        "pre_repair_external_attestation",
        "legacy_consolidated_external_attestation",
    ]


def test_migration_and_semantic_gates_wrap_checksums_not_assertions(tmp_path: Path):
    current = {"immutable_sha256": IMMUTABLE_SHA}
    marker = _legacy_cleanup(tmp_path)
    migrations = readiness.build_migrations_gate(
        current, cleanup_marker=marker, output=tmp_path / "migrations.json"
    )
    semantic = readiness.build_semantic_gate(
        current, cleanup_marker=marker, output=tmp_path / "semantic-gate.json"
    )
    assert migrations["metrics"] == readiness.MIGRATION_OUTER_METRICS
    assert semantic["metrics"] == readiness.SEMANTIC_METRICS
    wrapper_path = Path(migrations["artifacts"][0]["path"])
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    assert set(wrapper) == {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "metrics",
        "referenced_artifacts",
    }
    [source] = wrapper["referenced_artifacts"]
    assert source["name"] == "consolidated_response_report"
    assert source["sha256"] == _sha(Path(source["path"]))
    proxy = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    assert proxy["sealed_snapshot_member"]["logical_path"].startswith(
        "legacy_cleanup_evidence/"
    )
    [snapshot_reference] = proxy["referenced_artifacts"]
    assert snapshot_reference["name"] == "legacy_consolidated_external_attestation"


@pytest.mark.parametrize(
    ("forgery", "error"),
    (
        ("logical_path", "absent from snapshot inventory|topology differs"),
        ("sha256", "payload hash drifted"),
        ("snapshot_id", "does not address exactly one"),
        ("snapshot_root", "does not address exactly one"),
    ),
)
def test_sealed_snapshot_member_is_independently_bound_to_inventory(
    tmp_path: Path, forgery: str, error: str
) -> None:
    marker = _legacy_cleanup(tmp_path)
    migrations = readiness.build_migrations_gate(
        {"immutable_sha256": IMMUTABLE_SHA},
        cleanup_marker=marker,
        output=tmp_path / "migrations.json",
    )
    wrapper = json.loads(
        Path(migrations["artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    source_reference = wrapper["referenced_artifacts"][0]
    proxy_path = Path(source_reference["path"])
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    member = proxy["sealed_snapshot_member"]
    true_root = Path(member["snapshot_root"])
    true_candidate = true_root / member["logical_path"]

    if forgery == "logical_path":
        true_root.chmod(0o755)
        unlisted = true_root / "unlisted-member.json"
        unlisted.write_bytes(true_candidate.read_bytes())
        unlisted.chmod(0o444)
        true_root.chmod(0o555)
        member["logical_path"] = "unlisted-member.json"
    elif forgery == "sha256":
        member["sha256"] = "0" * 64
    elif forgery == "snapshot_id":
        member["snapshot_id"] = "forged-snapshot-id"
    else:
        other_root = tmp_path / "other-sealed-root"
        other_candidate = other_root / member["logical_path"]
        other_candidate.parent.mkdir(parents=True)
        other_candidate.write_bytes(true_candidate.read_bytes())
        other_candidate.chmod(0o444)
        for directory in sorted(
            [other_candidate.parent, *other_candidate.parents],
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if directory == tmp_path.parent:
                break
            if directory == tmp_path:
                continue
            if directory == other_root or other_root in directory.parents:
                directory.chmod(0o555)
        member["snapshot_root"] = str(other_root.resolve())

    proxy_path.chmod(0o644)
    proxy_path.write_text(json.dumps(proxy, sort_keys=True) + "\n", encoding="utf-8")
    proxy_path.chmod(0o444)
    forged_reference = dict(source_reference)
    forged_reference["sha256"] = _sha(proxy_path)
    with pytest.raises(control.ReadinessError, match=error):
        control._validate_referenced_artifact(
            forged_reference,
            context="forged sealed member",
            verified=set(),
            active=set(),
            snapshot_context=control._SnapshotValidationContext(full=True),
        )


def test_migration_gate_rejects_marker_metric_forgery(tmp_path: Path):
    marker = _legacy_cleanup(tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["migration_metrics"]["sealed_incident_qids"] = 1_063
    _write(marker, payload)
    with pytest.raises(readiness.EvidenceError, match="drifted"):
        readiness.build_migrations_gate(
            {"immutable_sha256": IMMUTABLE_SHA},
            cleanup_marker=marker,
            output=tmp_path / "gate.json",
        )


def test_migration_gate_rejects_cleanup_evidence_accounting_forgery(
    tmp_path: Path,
) -> None:
    forged = dict(readiness.EVIDENCE_ACCOUNTING)
    forged["frozen_baseline_validated_qids"] = 889_131
    marker = _legacy_cleanup(tmp_path, evidence_accounting=forged)
    with pytest.raises(readiness.EvidenceError, match="evidence accounting drifted"):
        readiness.build_migrations_gate(
            {"immutable_sha256": IMMUTABLE_SHA},
            cleanup_marker=marker,
            output=tmp_path / "gate.json",
        )


def test_migration_gate_rejects_live_report_drift_from_sealed_snapshot(
    tmp_path: Path,
) -> None:
    marker = _legacy_cleanup(tmp_path)
    response = (
        marker.parent
        / "operations"
        / "legacy_consolidation"
        / "response_incident_archive_report.json"
    )
    response.write_text('{"passed":false}\n', encoding="utf-8")
    with pytest.raises(readiness.EvidenceError, match="live/sealed"):
        readiness.build_migrations_gate(
            {"immutable_sha256": IMMUTABLE_SHA},
            cleanup_marker=marker,
            output=tmp_path / "gate.json",
        )


def test_published_migration_gate_depends_only_on_sealed_snapshot_graph(
    tmp_path: Path,
) -> None:
    current = {"immutable_sha256": IMMUTABLE_SHA}
    marker = _legacy_cleanup(tmp_path)
    output = tmp_path / "readiness" / "migrations.json"
    readiness.build_migrations_gate(
        current, cleanup_marker=marker, output=output
    )

    # The builder requires live and sealed evidence to agree.  Once published, the
    # controller's recursive graph terminates at the sealed snapshot attestation, not
    # at these mutable convenience copies.
    live_response = (
        marker.parent
        / "operations"
        / "legacy_consolidation"
        / "response_incident_archive_report.json"
    )
    live_response.write_text('{"passed":false}\n', encoding="utf-8")
    control._validate_attestation(
        current,
        "migrations",
        output,
        _sha(output),
    )


def test_outer_marker_is_not_published_until_controller_validation_passes(
    tmp_path: Path,
):
    output = tmp_path / "invalid.json"
    with pytest.raises(control.ReadinessError):
        readiness._publish_outer(
            control={"immutable_sha256": IMMUTABLE_SHA},
            gate="migrations",
            metrics=readiness.MIGRATION_OUTER_METRICS | {"sealed_incident_qids": 0},
            artifacts=[],
            output=output,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".invalid.json.candidate.*")) == []


def test_context_source_rejects_claimed_count_before_expensive_manifest_scan(
    tmp_path: Path,
):
    report = _write(
        tmp_path / "dense.json",
        {
            "schema_version": 3,
            "audit": "all_routed_profiles_context_capacity",
            "run_id": "full_sweep_agent_count_7_schema5_v1",
            "all_routed_profiles": True,
            "filters": readiness.CONTEXT_SPECS["dense_peer_context_audit"]["filters"],
            "manifest": {
                "path": str((tmp_path / "cells.json").resolve()),
                "sha256": "b" * 64,
                "cells": 3_600,
            },
            "benchmark_contracts": {
                "path": str((tmp_path / "benchmark_contracts.v1.json").resolve()),
                "sha256": "c" * 64,
                "verified": True,
            },
            "summary": {
                "passed": True,
                "selected_cells": 216,
                "audited_requests": 43_091,
                "failed_requests": 0,
                "failed_cells": 0,
                "minimum_context_headroom_tokens": 1_287,
            },
            "assumptions": {},
            "groups": [],
            "failure_groups": [],
            "failure_examples": [],
            "cells": [],
        },
    )
    fake_control = {
        "immutable": {
            "runs": [
                {
                    "run_id": "full_sweep_agent_count_7_schema5_v1",
                    "manifest_path": str(tmp_path / "cells.json"),
                    "manifest_sha256": "b" * 64,
                    "cell_count": 3_600,
                    "run_root": str(tmp_path / "run"),
                    "benchmark_contract_path": str(
                        tmp_path / "benchmark_contracts.v1.json"
                    ),
                    "benchmark_contract_sha256": "c" * 64,
                }
            ]
        }
    }
    with pytest.raises(readiness.EvidenceError, match="audited_requests drifted"):
        readiness._validate_context_source(
            fake_control, name="dense_peer_context_audit", path=report
        )


def test_context_source_accepts_margin_above_established_floor_before_manifest_scan(
    tmp_path: Path, monkeypatch
):
    manifest_path = tmp_path / "cells.json"
    report = _write(
        tmp_path / "dense.json",
        {
            "schema_version": 3,
            "audit": "all_routed_profiles_context_capacity",
            "run_id": "full_sweep_agent_count_7_schema5_v1",
            "all_routed_profiles": True,
            "filters": readiness.CONTEXT_SPECS["dense_peer_context_audit"][
                "filters"
            ],
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": "b" * 64,
                "cells": 3_600,
            },
            "benchmark_contracts": {
                "path": str((tmp_path / "benchmark_contracts.v1.json").resolve()),
                "sha256": "c" * 64,
                "verified": True,
            },
            "summary": {
                "passed": True,
                "selected_cells": 216,
                "audited_requests": 43_092,
                "failed_requests": 0,
                "failed_cells": 0,
                "minimum_context_headroom_tokens": 1_288,
            },
            "assumptions": {},
            "groups": [],
            "failure_groups": [],
            "failure_examples": [],
            "cells": [],
        },
    )
    fake_control = {
        "immutable": {
            "runs": [
                {
                    "run_id": "full_sweep_agent_count_7_schema5_v1",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": "b" * 64,
                    "cell_count": 3_600,
                    "run_root": str(tmp_path / "run"),
                    "benchmark_contract_path": str(
                        tmp_path / "benchmark_contracts.v1.json"
                    ),
                    "benchmark_contract_sha256": "c" * 64,
                }
            ]
        }
    }

    def reached_manifest_scan(*_args, **_kwargs):
        raise RuntimeError("reached manifest scan")

    monkeypatch.setattr(readiness, "load_manifest", reached_manifest_scan)
    with pytest.raises(RuntimeError, match="reached manifest scan"):
        readiness._validate_context_source(
            fake_control, name="dense_peer_context_audit", path=report
        )


def test_context_source_rejects_older_report_without_frozen_question_binding(
    tmp_path: Path,
):
    manifest_path = tmp_path / "cells.json"
    report = _write(
        tmp_path / "old-context-report.json",
        {
            "schema_version": 3,
            "audit": "all_routed_profiles_context_capacity",
            "run_id": "full_sweep_agent_count_7_schema5_v1",
            "all_routed_profiles": True,
            "filters": readiness.CONTEXT_SPECS[
                "dense_peer_context_audit"
            ]["filters"],
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": "b" * 64,
                "cells": 3_600,
            },
        },
    )
    fake_control = {
        "immutable": {
            "runs": [
                {
                    "run_id": "full_sweep_agent_count_7_schema5_v1",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": "b" * 64,
                    "cell_count": 3_600,
                    "run_root": str(tmp_path / "run"),
                    "benchmark_contract_path": str(
                        tmp_path / "benchmark_contracts.v1.json"
                    ),
                    "benchmark_contract_sha256": "c" * 64,
                }
            ]
        }
    }

    with pytest.raises(
        readiness.EvidenceError,
        match="frozen benchmark-contract identity drifted",
    ):
        readiness._validate_context_source(
            fake_control,
            name="dense_peer_context_audit",
            path=report,
        )


def test_context_source_rejects_self_consistent_summary_not_exact_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "full_sweep_agent_count_7_schema5_v1"
    manifest_path = tmp_path / "run" / "cells.json"
    benchmark_path = tmp_path / "run" / "benchmark_contracts.v1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    benchmark_path.write_text("{}\n", encoding="utf-8")
    cell = SimpleNamespace(
        cell_id="cell-1",
        config_hash=lambda: "config-hash",
    )
    snapshot = SimpleNamespace(cells=(cell,))
    catalog = SimpleNamespace(
        questions_for=lambda _cell: (object(),),
        frozen=SimpleNamespace(
            contract_for_cell=lambda _cell: {
                "question_contract_sha256": "q" * 64
            }
        ),
    )
    spec = {
        "source_name": "raw_dense_peer_context_audit",
        "run_id": run_id,
        "filters": {
            "n_agents": [7],
            "reasoning": ["unlimited"],
            "prompt_levels": [3],
            "topologies": ["decentralized"],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": 1,
        "audited_requests": 1,
    }
    monkeypatch.setitem(readiness.CONTEXT_SPECS, "fixture_context", spec)
    monkeypatch.setattr(readiness, "load_manifest", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        readiness, "selected_cells", lambda *_a, **_k: [cell]
    )
    monkeypatch.setattr(
        readiness, "VerifiedQuestionCatalog", lambda *_a, **_k: catalog
    )
    source = {
        "schema_version": 3,
        "audit": "all_routed_profiles_context_capacity",
        "run_id": run_id,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": "b" * 64,
            "cells": 1,
        },
        "benchmark_contracts": {
            "path": str(benchmark_path.resolve()),
            "sha256": "c" * 64,
            "verified": True,
        },
        "filters": spec["filters"],
        "all_routed_profiles": True,
        "assumptions": {"offline": False, "fabricated": True},
        "summary": {
            "passed": True,
            "selected_cells": 1,
            "audited_requests": 1,
            "failed_requests": 0,
            "failed_cells": 0,
            "minimum_context_headroom_tokens": 1_287,
        },
        "groups": [],
        "failure_groups": [],
        "failure_examples": [],
        "cells": [
            {
                "cell_id": "cell-1",
                "config_hash": "config-hash",
                "question_contract_sha256": "q" * 64,
                "audited_requests": 1,
                "failed_requests": 0,
                "fits": True,
                "minimum_context_headroom_tokens": 1_287,
                "minimum_headroom_request": {
                    "context_preflight_fits": True,
                    "output_capacity_headroom_tokens": 1_287,
                },
            }
        ],
    }
    authoritative = json.loads(json.dumps(source))
    authoritative["assumptions"] = {"offline": True, "fabricated": False}
    monkeypatch.setattr(
        readiness,
        "_recompute_context_source",
        lambda **_kwargs: authoritative,
    )
    report = _write(tmp_path / "fabricated-context.json", source)
    current = {
        "immutable": {
            "runs": [
                {
                    "run_id": run_id,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": "b" * 64,
                    "cell_count": 1,
                    "run_root": str(manifest_path.parent),
                    "benchmark_contract_path": str(benchmark_path),
                    "benchmark_contract_sha256": "c" * 64,
                }
            ]
        }
    }
    with pytest.raises(
        readiness.EvidenceError,
        match="independently rerendered",
    ):
        readiness._validate_context_source(
            current, name="fixture_context", path=report
        )


def test_double_http_probe_requires_both_rounds_to_be_healthy():
    successful = {
        "health_status": 200,
        "models_status": 200,
        "model_ids": ["4B"],
        "expected_model": "4B",
        "probe_started_timestamp": 1.0,
        "probe_completed_timestamp": 2.0,
        "healthy": True,
    }
    failed = dict(successful, health_status=0, healthy=False)
    combined = readiness._combine_http_probe_rounds(
        [successful, failed], expected_model="4B"
    )
    assert combined["healthy"] is False
    assert combined["health_status"] == 0


def test_registry_reader_rejects_duplicate_replica_identity(tmp_path: Path):
    base = {
        "model_size": "4B",
        "hf_id": "Qwen/Qwen3-4B",
        "host": "node",
        "port": 8000,
        "replica_id": "schema5-v1--4b--standard--r00",
    }
    _write(tmp_path / "servers" / "4B" / "one.json", base)
    _write(tmp_path / "servers" / "4B" / "two.json", base | {"port": 8001})
    with pytest.raises(readiness.EvidenceError, match="duplicate registry"):
        readiness._registry_records(tmp_path)


def test_fleet_runtime_proof_projects_exact_paused_next_generation(
    tmp_path: Path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": 3,
        "immutable": {},
    }
    attestation = {
        "schema_version": 1,
        "generation": 4,
        "path": str(state / "runtime.g000004.json"),
        "sha256": "1" * 64,
        "attestation_id": "2" * 64,
        "lease_path": str(state / "lease.g000004.json"),
    }
    calls = []
    monkeypatch.setattr(control, "control_lock", lambda _state: nullcontext())
    monkeypatch.setattr(control, "load_control", lambda *_args, **_kwargs: current)
    def ensure(_state, observed, *, generation, force_full):
        calls.append((observed, generation, force_full))
        return attestation

    monkeypatch.setattr(control, "ensure_runtime_integrity_attestation", ensure)

    def validate(projected, *, verify_metadata):
        assert projected["rollout_generation"] == 4
        assert projected[control.RUNTIME_ATTESTATION_STATE_KEY] == attestation
        assert verify_metadata is True

    monkeypatch.setattr(control, "validate_runtime_integrity_attestation", validate)
    monkeypatch.setattr(
        control,
        "production_environment",
        lambda projected: {
            "ASYS_RUNTIME_ATTESTATION": projected["runtime_integrity"]["path"],
            "ASYS_RUNTIME_ATTESTATION_SHA256": projected["runtime_integrity"][
                "sha256"
            ],
            "ASYS_RUNTIME_INTEGRITY_LEASE": projected["runtime_integrity"][
                "lease_path"
            ],
            "ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_SHA,
            "ASYS_ROLLOUT_GENERATION": str(projected["rollout_generation"]),
        },
    )

    observed = readiness._paused_next_generation_runtime_environment(
        current, state_dir=state
    )
    assert observed["ASYS_ROLLOUT_GENERATION"] == "4"
    assert calls == [(current, 4, False)]


def test_fleet_runtime_proof_rejects_running_control(tmp_path: Path, monkeypatch):
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "desired_state": "running",
        "drain_requested": False,
        "rollout_generation": 1,
    }
    monkeypatch.setattr(control, "control_lock", lambda _state: nullcontext())
    monkeypatch.setattr(control, "load_control", lambda *_args, **_kwargs: current)
    with pytest.raises(readiness.EvidenceError, match="paused, non-draining"):
        readiness._paused_next_generation_runtime_environment(
            current, state_dir=tmp_path
        )


def test_fleet_runtime_proof_allows_exact_capacity_readiness_drain(
    tmp_path: Path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    current = _capacity_readiness_control(tmp_path)
    attestation = {
        "path": str(state / "runtime.g000004.json"),
        "sha256": "1" * 64,
        "lease_path": str(state / "lease.g000004.json"),
    }
    monkeypatch.setattr(control, "control_lock", lambda _state: nullcontext())
    monkeypatch.setattr(control, "load_control", lambda *_args, **_kwargs: current)
    monkeypatch.setattr(
        control,
        "effective_fleet_contract_binding",
        lambda *_args, **_kwargs: {
            "capacity_generation": 2,
            "path": current["capacity"]["current_contract"]["path"],
            "sha256": current["capacity"]["current_contract"]["sha256"],
        },
    )
    monkeypatch.setattr(
        control,
        "ensure_runtime_integrity_attestation",
        lambda *_args, **_kwargs: attestation,
    )
    monkeypatch.setattr(
        control,
        "validate_runtime_integrity_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        control,
        "production_environment",
        lambda projected: {
            "ASYS_RUNTIME_ATTESTATION": attestation["path"],
            "ASYS_RUNTIME_ATTESTATION_SHA256": attestation["sha256"],
            "ASYS_RUNTIME_INTEGRITY_LEASE": attestation["lease_path"],
            "ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_SHA,
            "ASYS_ROLLOUT_GENERATION": str(projected["rollout_generation"]),
        },
    )
    observed = readiness._paused_next_generation_runtime_environment(
        current, state_dir=state
    )
    assert observed["ASYS_ROLLOUT_GENERATION"] == "4"


def test_fleet_gate_publishes_22_replica_scheduler_registry_http_graph(
    tmp_path: Path, monkeypatch
):
    model_path = Path("configs/model_contracts.v1.json").resolve()
    model_hash = _sha(model_path)
    fleet_path = Path("configs/schema5_fleet.v1.json").resolve()
    fleet_hash = _sha(fleet_path)
    models = load_model_contracts(model_path, expected_sha256=model_hash)
    fleet = load_fleet_contract(
        fleet_path, model_contracts=models, expected_sha256=fleet_hash
    )
    # A capacity generation owns a sealed operational path distinct from the release
    # path even when this focused fixture reuses the same replica bytes.
    overlay_fleet_path = tmp_path / "capacity-g000002" / "fleet.json"
    overlay_fleet_path.parent.mkdir(parents=True)
    overlay_fleet_path.write_bytes(fleet_path.read_bytes())
    overlay_fleet_path.with_suffix(".sha256").write_text(
        f"{fleet_hash}  {overlay_fleet_path.name}\n", encoding="utf-8"
    )
    overlay_fleet_path.chmod(0o444)
    profile_counts = {
        name: len(replicas) for name, replicas in fleet.by_profile.items()
    }
    certificate_path = (
        tmp_path
        / "readiness"
        / "capacity-generations"
        / "c000002"
        / "PREFLIGHT_CAPACITY_CERTIFICATE.json"
    ).resolve()
    protected_authority = {
        "capacity_generation": 2,
        "path": str(
            (tmp_path / "PROTECTED_CAPACITY_COMPLETE.json").resolve()
        ),
        "sha256": "3" * 64,
        "marker_id": "4" * 64,
        "static_feasibility_certificate_path": str(certificate_path),
        "static_feasibility_certificate_sha256": "5" * 64,
        "static_feasibility_certificate_id": "6" * 64,
        "effective_fleet_contract_sha256": fleet_hash,
    }
    protected_contract = SimpleNamespace(
        marker_id="1" * 64,
        capacity_generation=2,
        static_feasibility_certificate_path=certificate_path,
        static_feasibility_certificate_sha256="5" * 64,
        static_feasibility_certificate_id="6" * 64,
        effective_fleet_contract_sha256=fleet_hash,
        static_feasibility_wave_passed=False,
        static_feasibility_selected_cell_count=284,
        static_feasibility_target_cell_count=384,
        static_feasibility_shortfall_cells=100,
        static_feasibility_configured_client_ceiling=384,
        static_feasibility_certified_saturation_target=284,
    )
    monkeypatch.setattr(
        readiness.control_plane,
        "effective_fleet_contract_binding",
        lambda *_args, **_kwargs: {
            "capacity_generation": 2,
            "path": str(overlay_fleet_path.resolve()),
            "sha256": fleet_hash,
            "release_sha256": fleet_hash,
            "logical_replicas": len(fleet.replicas),
            "allocated_gpus": sum(
                replica.gpus_per_replica for replica in fleet.replicas
            ),
            "profile_replicas": profile_counts,
            "protected_capacity": dict(protected_authority),
        },
    )
    client_contract = {
        "partition": "ou_bcs_normal",
        "qos": "normal",
        "authorized_cell_ceiling": 384,
        "reserve_jobs": 64,
    }
    monkeypatch.setattr(
        readiness.control_plane,
        "client_capacity_contract_from_state",
        lambda *_args, **_kwargs: dict(client_contract),
    )
    monkeypatch.setattr(
        readiness.control_plane,
        "load_effective_protected_capacity_contract",
        lambda *_args, **_kwargs: protected_contract,
    )
    monkeypatch.setattr(
        readiness.control_plane,
        "effective_protected_capacity_binding",
        lambda *_args, **_kwargs: dict(protected_authority),
    )
    monkeypatch.setattr(
        readiness.control_plane,
        "_load_protected_capacity_contract",
        lambda *_args, **_kwargs: protected_contract,
    )
    monkeypatch.setattr(
        readiness.protected_capacity,
        "authorize_client",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        readiness.protected_capacity,
        "capture_live_client_capacity",
        lambda *_args, **_kwargs: {"evidence_id": "2" * 64},
    )
    monkeypatch.setattr(
        readiness.protected_capacity,
        "validate_live_client_capacity_evidence",
        lambda *_args, **_kwargs: {
            "evidence_id": "2" * 64,
            "partition": "ou_bcs_normal",
            "qos": "normal",
            "cpu_limit": 384,
            "memory_limit_mib": 384 * 4096,
            "max_submit_jobs": 448,
        },
    )
    pool = tmp_path / "results" / "server_pools" / "schema5-v1"
    transaction_directory = fleet_tx.state_directory(pool)
    ledger_path = fleet_tx.ledger_path(transaction_directory, 1)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("{}\n", encoding="utf-8")
    _write(
        transaction_directory / fleet_tx.CURRENT_FILENAME,
        {
            "schema_version": fleet_tx.STATE_SCHEMA_VERSION,
            "pool_root": str(pool.resolve()),
            "pool_id": "schema5-v1",
            "fleet_sha256": fleet_hash,
            "current_generation": 1,
            "ledger_path": str(ledger_path.resolve()),
            "ledger_sha256": _sha(ledger_path),
            "updated_at": 90.0,
        },
    )
    records = {}
    scheduler = {}
    scripts = {}
    intent_tokens = {}
    environment_hash = "c" * 64
    for replica in fleet.replicas:
        profile = get_serving_profile(replica.serving_profile)
        identity = models.for_size(replica.model_size)
        job_id = str(10_000 + len(records))
        host = f"node{len(records):02d}"
        entry = ServerEntry(
            model_size=replica.model_size,
            hf_id=profile.hf_id,
            host=host,
            port=8_000 + len(records),
            slurm_job_id=job_id,
            started_at=50.0,
            serving_profile=replica.serving_profile,
            served_model_name=profile.served_model_name,
            max_model_len=profile.max_model_len,
            tp_size=profile.tp_size,
            release_id="sweep-recovery-schema5-v1.2",
            environment_hash=environment_hash,
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
                model_contract_sha256=model_hash,
                fleet_contract_sha256=fleet_hash,
                server_pool_id="schema5-v1",
                replica_id=replica.replica_id,
                replica_index=replica.replica_index,
                release_fleet_contract_sha256=fleet_hash,
                capacity_generation=2,
                rollout_generation=1,
        )
        registry_path = (
            pool
            / "servers"
            / replica.serving_profile
            / f"{host}_{entry.port}.json"
        )
        _write(registry_path, asdict(entry))
        records[replica.replica_id] = (entry, registry_path.resolve())
        intent_token = f"{len(records):032x}"
        script_path = (
            transaction_directory
            / "sbatch"
            / "g000001"
            / f"{replica.replica_id}.{intent_token}.sbatch"
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
        script_path.chmod(0o444)
        intent_tokens[replica.replica_id] = intent_token
        comment = fleet_tx.intent_comment(
            pool_id="schema5-v1",
            profile=replica.serving_profile,
            replica_id=replica.replica_id,
            rollout_generation=1,
            intent_token=intent_token,
            fleet_sha256=fleet_hash,
        )
        row = keepalive.FleetQueueRow(
            job_id,
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            host,
            str(script_path.resolve()),
            comment,
        )
        provenance = keepalive.SpooledServingProvenance(
            run_root=str(pool.resolve()),
            server_pool_id="schema5-v1",
            replica_id=replica.replica_id,
            replica_index=replica.replica_index,
            release_id="sweep-recovery-schema5-v1.2",
            environment_hash=environment_hash,
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
                model_contract_sha256=model_hash,
                fleet_contract_sha256=fleet_hash,
                release_fleet_contract_sha256=fleet_hash,
                capacity_generation=2,
                rollout_generation=1,
                spooled_script_sha256=_sha(script_path),
            )
        scheduler[replica.replica_id] = (row, provenance)
        scripts[job_id] = script_path

    snapshot = keepalive.ReadOnlyFleetSnapshot(
        current_generation=1,
        captured_at=90.0,
        allocations=tuple(
            keepalive.ReadOnlyFleetAllocation(
                replica_id=replica_id,
                ledger_generation=1,
                intent_token=intent_tokens[replica_id],
                attempt_state="committed",
                row=row,
                spooled_provenance=provenance,
                sbatch_path=str(scripts[row.job_id].resolve()),
                sbatch_sha256=_sha(scripts[row.job_id]),
            )
            for index, (replica_id, (row, provenance)) in enumerate(
                sorted(scheduler.items()), start=1
            )
        ),
        ignored_terminal_job_ids=(),
    )
    monkeypatch.setattr(
        keepalive,
        "reconcile_fleet_read_only",
        lambda *_args, **_kwargs: snapshot,
    )
    provenance_calls = []
    monkeypatch.setattr(
        readiness.control_plane,
        "reconcile_trusted_scientific_job_provenance",
        lambda *_args, **kwargs: (
            provenance_calls.append(dict(kwargs))
            or SimpleNamespace(
                payload={
                    "trusted_live_job_ids": sorted(scripts),
                    "provenance_id": "a" * 64,
                }
            )
        ),
    )
    monkeypatch.setattr(readiness, "_verify_registry", lambda **_kwargs: records)
    monkeypatch.setattr(
        readiness,
        "_paused_next_generation_runtime_environment",
        lambda *_args, **_kwargs: {
            "ASYS_RUNTIME_ATTESTATION": str(tmp_path / "runtime.g000001.json"),
            "ASYS_RUNTIME_ATTESTATION_SHA256": "1" * 64,
            "ASYS_RUNTIME_INTEGRITY_LEASE": str(tmp_path / "lease.g000001.json"),
            "ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_SHA,
            "ASYS_ROLLOUT_GENERATION": "1",
        },
    )

    def scheduler_runner(argv, _timeout):
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                args,
                0,
                (
                    "KillWait                = 30 sec\n"
                    "PreemptMode             = REQUEUE\n"
                    "PreemptType             = preempt/partition_prio\n"
                ),
                "",
            )
        if args == [
            "scontrol",
            "show",
            "partition",
            "mit_normal",
            "-o",
        ]:
            return subprocess.CompletedProcess(
                args,
                0,
                (
                    "PartitionName=mit_normal GraceTime=0 MaxTime=12:00:00 "
                    "PreemptMode=OFF State=UP TotalNodes=50\n"
                ),
                "",
            )
        if (
            len(args) == 5
            and args[:3] == ["scontrol", "show", "partition"]
            and args[3] in {"ou_bcs_low", "ou_bcs_normal"}
            and args[4] == "-o"
        ):
            mode = "REQUEUE" if args[3] == "ou_bcs_low" else "OFF"
            return subprocess.CompletedProcess(
                args,
                0,
                (
                    f"PartitionName={args[3]} GraceTime=0 "
                    f"MaxTime=1-00:00:00 PreemptMode={mode} "
                    "State=UP TotalNodes=50\n"
                ),
                "",
            )
        if args == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "mit_normal",
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ]:
            return subprocess.CompletedProcess(
                args, 0, "mit_normal|cpu=96,mem=386G|448\n", ""
            )
        job_id = args[-1] if args[:4] == ["scontrol", "show", "job", "-o"] else args[-2]
        script_path = scripts[job_id]
        if args[:4] == ["scontrol", "show", "job", "-o"]:
            replica_id = next(
                replica_id
                for replica_id, (row, _provenance) in scheduler.items()
                if row.job_id == job_id
            )
            row, _provenance = scheduler[replica_id]
            output = (
                f"JobId={job_id} JobName={row.job_name} "
                f"JobState=RUNNING Reason=None Requeue=0 "
                f"Comment={row.comment} NodeList={row.node} "
                f"Command={script_path.resolve()}\n"
            )
        elif args[:3] == ["scontrol", "write", "batch_script"]:
            output = script_path.read_text(encoding="utf-8")
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, output, "")

    probe_calls = []

    def probe(entry, expected_model, _timeout):
        probe_calls.append(entry.replica_id)
        return {
            "health_status": 200,
            "models_status": 200,
            "model_ids": [expected_model],
            "expected_model": expected_model,
            "probe_started_timestamp": 90.0,
            "probe_completed_timestamp": 91.0,
            "healthy": True,
        }

    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "desired_state": "paused",
        "drain_requested": True,
        "rollout_generation": 0,
        "capacity": {
            "current_generation": 2,
            "active_transition": {
                "transition_id": "capacity-g000001-to-g000002-fleet-gate",
                "phase": "readiness_pending",
                "to_generation": 2,
            },
        },
        "admission_safety_hold": {
            "active": True,
            "reasons": [
                "capacity-transition:"
                "capacity-g000001-to-g000002-fleet-gate"
            ],
        },
        "immutable": {
            "release_worktree": str(Path.cwd().resolve()),
            "release_id": "sweep-recovery-schema5-v1.2",
            "git_commit": "a" * 40,
            "source_tree_sha256": "e" * 64,
            "protected_capacity_marker_path": str(
                tmp_path / "PROTECTED_CAPACITY_COMPLETE.json"
            ),
            "protected_capacity_marker_sha256": "3" * 64,
            "protected_capacity_marker_id": "4" * 64,
            "protected_client_partition": "ou_bcs_normal",
            "protected_client_qos": "normal",
            "protected_client_cpus": 384,
            "protected_client_memory_mib": 384 * 4096,
            "protected_client_submit_headroom": 448,
            "transport_uncertainty_binding": (
                readiness.scheduler_safety.expected_transport_uncertainty_binding()
            ),
            "transport_uncertainty_binding_sha256": (
                readiness.scheduler_safety.transport_uncertainty_binding_sha256()
            ),
            "model_contract_path": str(model_path),
            "model_contract_sha256": model_hash,
            "fleet_contract_path": str(fleet_path),
            "fleet_contract_sha256": fleet_hash,
            "server_pool_root": str(pool.resolve()),
            "serving_environment_sha256": environment_hash,
            "harness_environment_prefix": str(tmp_path / "harness"),
            "serving_environment_prefix": str(tmp_path / "serving"),
            "harness_environment_manifest_path": str(tmp_path / "harness.json"),
            "serving_environment_manifest_path": str(tmp_path / "serving.json"),
            "harness_environment_sha256": "d" * 64,
            "hf_home": str(tmp_path / "hf"),
        },
    }
    report = readiness.build_fleet_gate(
        current,
        output=tmp_path / "fleet.json",
        state_dir=tmp_path / "state",
        scheduler_runner=scheduler_runner,
        probe=probe,
        now=lambda: 100.0,
    )
    assert report["metrics"]["logical_replicas"] == 22
    assert report["metrics"]["fleet_contract_sha256"] == fleet_hash
    assert report["metrics"]["transport_censor_protocol_hash"] == (
        "e1a46d605eed904e902f9339be714c1019aa361a11da2cdab98b8edcf0ec3104"
    )
    assert report["metrics"]["scheduler_preemptible_partitions"] == []
    assert len(report["metrics"]["scheduler_safety_policy_contract_id"]) == 64
    fleet_artifact = next(
        artifact
        for artifact in report["artifacts"]
        if artifact["name"] == "fleet_contract"
    )
    assert fleet_artifact == {
        "name": "fleet_contract",
        "path": str(overlay_fleet_path.resolve()),
        "sha256": fleet_hash,
    }
    assert len(probe_calls) == 44
    assert all(probe_calls.count(replica.replica_id) == 2 for replica in fleet.replicas)
    wrapper = json.loads(
        Path(report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    assert set(wrapper) == {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "server_pool_root",
        "metrics",
        "referenced_artifacts",
    }
    [raw_ref] = wrapper["referenced_artifacts"]
    raw_path = Path(raw_ref["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert len(raw["replicas"]) == len(raw["referenced_artifacts"]) == 22

    # A fully reserialized raw artifact cannot substitute spooled provenance: the
    # validator re-joins immutable fleet/ledger/scheduler/script/registry truth.
    raw["replicas"][0]["spooled_provenance"]["model_revision"] = "forged"
    raw_path.chmod(0o644)
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_path.chmod(0o444)
    with pytest.raises(
        readiness.EvidenceError,
        match="raw fleet readiness proof drifted",
    ):
        readiness._validate_raw_fleet_readiness(
            raw_path,
            control=current,
            state_dir=tmp_path / "state",
            pool_root=pool.resolve(),
            expected_replica_ids={
                replica.replica_id for replica in fleet.replicas
            },
            scheduler_runner=scheduler_runner,
        )


def _bind_test_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    interpreter = tmp_path / "immutable-test-python"
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o555)
    monkeypatch.setattr(readiness.sys, "executable", str(interpreter))
    return interpreter


def test_email_gate_requires_delivery_and_one_time_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    active = tmp_path / "email-challenges" / "CURRENT.json"
    output = tmp_path / "email.json"
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    entropy = iter(bytes([index]) * 32 for index in range(1, 9))

    def random_bytes(size: int) -> bytes:
        assert size == 32
        return next(entropy)

    with pytest.raises(readiness.EvidenceError, match="requires --apply"):
        readiness.build_email_request(
            current,
            output=active,
            chain_id="c" * 64,
            challenge_generation=0,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=ack_script,
            apply=False,
        )

    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "queued\n", "")

    requested = readiness.build_email_request(
        current,
        output=active,
        chain_id="c" * 64,
        challenge_generation=0,
        release_tag="sweep-recovery-schema5-v1.2-r9",
        release_git_commit="b" * 40,
        acknowledgement_script=ack_script,
        apply=True,
        mail_runner=runner,
        now=lambda: 123.0,
        random_bytes=random_bytes,
        retry_delays=(0,),
    )
    assert requested["kind"] == "schema5_email_active_challenge"
    body = calls[0][1]["input"]
    token = body.split("One-time token:\n", 1)[1].split("\n", 1)[0]
    request = Path(requested["request"])
    acknowledgement = Path(requested["acknowledgement"])
    raw_request = json.loads(request.read_text(encoding="utf-8"))
    assert token not in request.read_text(encoding="utf-8")
    assert "--token" not in request.read_text(encoding="utf-8")
    assert "ack_command" not in raw_request
    assert raw_request["challenge_verifier"] == (
        schema5_email_ack.challenge_verifier_for(token, raw_request)
    )
    schema5_email_ack.acknowledge(
        request=request,
        output=acknowledgement,
        token=token,
        operator="tester",
        apply=True,
        now=124.0,
        executing_tool=ack_script,
    )
    report = readiness.build_email_gate(
        current,
        output=output,
        active_challenge=active,
    )
    assert report["metrics"] == {
        "recipient": "mabdel03@mit.edu",
        "delivery_succeeded": True,
        "returncode": 0,
        "acknowledged": True,
        "chain_id": "c" * 64,
        "request_id": raw_request["request_id"],
        "release_tag": "sweep-recovery-schema5-v1.2-r9",
        "release_git_commit": "b" * 40,
        "challenge_generation": 0,
        "challenge_id": raw_request["challenge_id"],
        "challenge_verifier": raw_request["challenge_verifier"],
        "acknowledged_at": json.loads(
            acknowledgement.read_text(encoding="utf-8")
        )["acknowledged_at"],
    }
    assert calls[0][0] == [
        "mail",
        "-s",
        "[agents-scaling] schema-5 readiness acknowledgement required",
        "mabdel03@mit.edu",
    ]
    wrapper = json.loads(
        Path(report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    active_ref, receipt, ack = wrapper["referenced_artifacts"]
    raw = json.loads(Path(receipt["path"]).read_text(encoding="utf-8"))
    assert raw["submitted_timestamp"] == 123.0
    assert raw["delivery_attempts"] == [
        {"attempt": 1, "returncode": 0, "timed_out": False}
    ]
    assert active_ref["path"] == str(active.resolve())
    assert ack["path"] == str(acknowledgement.resolve())

    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="already consumed",
    ):
        schema5_email_ack.acknowledge(
            request=request,
            output=acknowledgement,
            token=token,
            operator="tester",
            apply=True,
            now=125.0,
            executing_tool=ack_script,
        )


def test_email_delivery_retries_reuse_token_only_in_live_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    active = tmp_path / "email-challenges" / "CURRENT.json"
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    bodies: list[str] = []
    delays: list[float] = []

    def runner(argv, **kwargs):
        bodies.append(kwargs["input"])
        return subprocess.CompletedProcess(
            argv, 0 if len(bodies) == 3 else 75, "", "rejected"
        )

    entropy = iter(bytes([index]) * 32 for index in range(11, 15))
    report = readiness.build_email_request(
        current,
        output=active,
        chain_id="c" * 64,
        challenge_generation=3,
        release_tag="sweep-recovery-schema5-v1.2-r9",
        release_git_commit="b" * 40,
        acknowledgement_script=ack_script,
        apply=True,
        mail_runner=runner,
        now=lambda: 100.0,
        random_bytes=lambda size: next(entropy),
        sleeper=delays.append,
        retry_delays=(0, 1, 2),
    )
    assert len(set(bodies)) == 1
    assert delays == [1.0, 2.0]
    token = bodies[0].split("One-time token:\n", 1)[1].split("\n", 1)[0]
    request = Path(report["request"])
    assert token not in request.read_text(encoding="utf-8")
    assert token not in active.read_text(encoding="utf-8")
    assert json.loads(request.read_text(encoding="utf-8"))[
        "delivery_attempts"
    ] == [
        {"attempt": 1, "returncode": 75, "timed_out": False},
        {"attempt": 2, "returncode": 75, "timed_out": False},
        {"attempt": 3, "returncode": 0, "timed_out": False},
    ]
    ack_script.chmod(0o644)
    ack_script.write_text("# tampered fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="differs from the sealed request",
    ):
        schema5_email_ack.acknowledge(
            request=request,
            output=Path(report["acknowledgement"]),
            token=token,
            operator="tester",
            apply=True,
            now=101.0,
            executing_tool=ack_script,
        )


def test_email_failed_delivery_publishes_no_challenge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    active = tmp_path / "email-challenges" / "CURRENT.json"
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    entropy = iter(bytes([index]) * 32 for index in range(21, 25))

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 75, "", "rejected")

    with pytest.raises(readiness.EvidenceError, match="bounded retry"):
        readiness.build_email_request(
            current,
            output=active,
            chain_id="c" * 64,
            challenge_generation=0,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=ack_script,
            apply=True,
            mail_runner=runner,
            random_bytes=lambda size: next(entropy),
            sleeper=lambda _: None,
            retry_delays=(0, 0),
        )
    assert not active.exists()
    assert not list(active.parent.glob("g*/requests/*.json"))


def test_email_challenge_rejects_generation_and_tool_ancestor_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    root = tmp_path / "email-challenges"
    root.mkdir()
    active = root / "CURRENT.json"
    outside = tmp_path / "outside-generation"
    outside.mkdir()
    (root / "g0000").symlink_to(outside, target_is_directory=True)
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    calls = 0

    def runner(argv, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, "", "")

    entropy = iter(bytes([index]) * 32 for index in range(25, 29))
    with pytest.raises(readiness.EvidenceError, match="ancestry is unsafe"):
        readiness.build_email_request(
            current,
            output=active,
            chain_id="c" * 64,
            challenge_generation=0,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=ack_script,
            apply=True,
            mail_runner=runner,
            random_bytes=lambda size: next(entropy),
            retry_delays=(0,),
        )
    assert calls == 0
    assert not list(outside.iterdir())

    (root / "g0000").unlink()
    real_tools = tmp_path / "real-tools"
    real_tools.mkdir()
    real_tool = real_tools / "schema5_email_ack.py"
    real_tool.write_text("# fixture\n", encoding="utf-8")
    real_tool.chmod(0o444)
    linked_tools = tmp_path / "linked-tools"
    linked_tools.symlink_to(real_tools, target_is_directory=True)
    with pytest.raises(readiness.EvidenceError, match="traverses a symlink"):
        readiness.build_email_request(
            current,
            output=active,
            chain_id="c" * 64,
            challenge_generation=0,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=linked_tools / real_tool.name,
            apply=True,
            mail_runner=runner,
            random_bytes=lambda size: b"x" * size,
            retry_delays=(0,),
        )
    assert calls == 0


def test_email_repair_supersedes_pending_token_and_blocks_cross_chain_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    active = tmp_path / "email-challenges" / "CURRENT.json"
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    bodies: list[str] = []

    def runner(argv, **kwargs):
        bodies.append(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    entropy = iter(bytes([index]) * 32 for index in range(31, 39))
    first = readiness.build_email_request(
        current,
        output=active,
        chain_id="c" * 64,
        challenge_generation=0,
        release_tag="sweep-recovery-schema5-v1.2-r9",
        release_git_commit="b" * 40,
        acknowledgement_script=ack_script,
        apply=True,
        mail_runner=runner,
        now=lambda: 100.0,
        random_bytes=lambda size: next(entropy),
        retry_delays=(0,),
    )
    second = readiness.build_email_request(
        current,
        output=active,
        chain_id="c" * 64,
        challenge_generation=1,
        release_tag="sweep-recovery-schema5-v1.2-r9",
        release_git_commit="b" * 40,
        acknowledgement_script=ack_script,
        apply=True,
        mail_runner=runner,
        now=lambda: 200.0,
        random_bytes=lambda size: next(entropy),
        retry_delays=(0,),
    )
    first_token = bodies[0].split("One-time token:\n", 1)[1].split("\n", 1)[0]
    second_token = bodies[1].split("One-time token:\n", 1)[1].split("\n", 1)[0]
    assert first_token != second_token
    assert second["supersedes"]["request_id"] == first["request_id"]
    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="currently active",
    ):
        schema5_email_ack.acknowledge(
            request=Path(first["request"]),
            output=Path(first["acknowledgement"]),
            token=first_token,
            operator="tester",
            apply=True,
            now=201.0,
            executing_tool=ack_script,
        )
    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="token or email-request identity",
    ):
        schema5_email_ack.acknowledge(
            request=Path(second["request"]),
            output=Path(second["acknowledgement"]),
            token=first_token,
            operator="tester",
            apply=True,
            now=201.0,
            executing_tool=ack_script,
        )
    schema5_email_ack.acknowledge(
        request=Path(second["request"]),
        output=Path(second["acknowledgement"]),
        token=second_token,
        operator="tester",
        apply=True,
        now=201.0,
        executing_tool=ack_script,
    )
    with pytest.raises(readiness.EvidenceError, match="another chain"):
        readiness.build_email_request(
            current,
            output=active,
            chain_id="d" * 64,
            challenge_generation=1,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=ack_script,
            apply=True,
            mail_runner=runner,
            random_bytes=lambda size: b"x" * size,
            retry_delays=(0,),
        )


def test_email_activation_crash_leaves_old_pointer_and_next_repair_supersedes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _bind_test_interpreter(tmp_path, monkeypatch)
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
        "immutable": {"git_commit": "b" * 40},
    }
    active = tmp_path / "email-challenges" / "CURRENT.json"
    ack_script = tmp_path / "schema5_email_ack.py"
    ack_script.write_text("# fixture\n", encoding="utf-8")
    ack_script.chmod(0o444)
    bodies: list[str] = []

    def runner(argv, **kwargs):
        bodies.append(kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    entropy = iter(bytes([index]) * 32 for index in range(41, 53))

    def launch(generation: int, timestamp: float):
        return readiness.build_email_request(
            current,
            output=active,
            chain_id="c" * 64,
            challenge_generation=generation,
            release_tag="sweep-recovery-schema5-v1.2-r9",
            release_git_commit="b" * 40,
            acknowledgement_script=ack_script,
            apply=True,
            mail_runner=runner,
            now=lambda: timestamp,
            random_bytes=lambda size: next(entropy),
            retry_delays=(0,),
        )

    first = launch(0, 100.0)
    first_pointer = active.read_bytes()
    original_write = readiness._write_json_atomic
    failed = False

    def crash_before_activation(path: Path, payload: dict):
        nonlocal failed
        if path.expanduser().absolute() == active.absolute() and not failed:
            failed = True
            raise RuntimeError("simulated activation crash")
        return original_write(path, payload)

    monkeypatch.setattr(readiness, "_write_json_atomic", crash_before_activation)
    with pytest.raises(RuntimeError, match="simulated activation crash"):
        launch(1, 200.0)
    assert active.read_bytes() == first_pointer

    crashed_command = bodies[1].split("Run exactly once:\n", 1)[1].split(
        "\n", 1
    )[0]
    crashed_argv = shlex.split(crashed_command)
    crashed_request = Path(crashed_argv[crashed_argv.index("--request") + 1])
    crashed_output = Path(crashed_argv[crashed_argv.index("--output") + 1])
    crashed_token = crashed_argv[crashed_argv.index("--token") + 1]
    assert crashed_request.is_file()
    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="currently active",
    ):
        schema5_email_ack.acknowledge(
            request=crashed_request,
            output=crashed_output,
            token=crashed_token,
            operator="tester",
            apply=True,
            now=201.0,
            executing_tool=ack_script,
        )

    monkeypatch.setattr(readiness, "_write_json_atomic", original_write)
    repaired = launch(2, 300.0)
    assert repaired["request_id"] != first["request_id"]
    first_token = bodies[0].split("One-time token:\n", 1)[1].split("\n", 1)[0]
    with pytest.raises(
        schema5_email_ack.AcknowledgementError,
        match="currently active",
    ):
        schema5_email_ack.acknowledge(
            request=Path(first["request"]),
            output=Path(first["acknowledgement"]),
            token=first_token,
            operator="tester",
            apply=True,
            now=301.0,
            executing_tool=ack_script,
        )


def _capacity_chain_binding(tmp_path: Path) -> dict:
    manifest_path = _write(tmp_path / "chain.json", {"chain": "fixture"})
    receipt_path = _write(tmp_path / "receipt.json", {"receipt": "fixture"})
    manifest_path.chmod(0o444)
    receipt_path.chmod(0o444)
    manifest = {
        "chain_id": "e" * 64,
        "jobs": [
            {
                "name": "fleet_readiness",
                "job_name": "asys-s5-r9-fleet-ready",
            }
        ],
    }
    receipt = {
        "receipt_id": "f" * 64,
        "jobs": [
            {
                "name": "fleet_readiness",
                "job_id": "4242",
                "comment": (
                    "asys:s5-recovery-v1.2-r9:"
                    + "e" * 64
                    + ":g0000:fleet_readiness"
                ),
            }
        ],
    }
    return {
        "verified": {
                "chain_protocol": readiness.R9_PROTOCOL,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha(manifest_path),
            "manifest": manifest,
            "submission_receipt_path": str(receipt_path.resolve()),
            "submission_receipt_sha256": _sha(receipt_path),
            "submission_receipt": receipt,
        },
        "manifest_row": manifest["jobs"][0],
        "receipt_row": receipt["jobs"][0],
        "chain_generation": 0,
    }


def _capacity_fleet_payload(*, pending_reason: str = "Resources") -> dict:
    rows = []
    for index in range(22):
        common = {
            "replica_id": f"replica-{index:02d}",
            "serving_profile": f"profile-{index:02d}",
            "allocated_gpus": 2 if index < 2 else 1,
            "job_id": str(50_000 + index),
            "comment": f"comment-{index:02d}",
            "intent_token": f"{index + 1:032x}",
            "ledger_generation": 1,
            "local_script_path": f"/immutable/{index:02d}.sbatch",
            "local_script_sha256": f"{index + 1:064x}",
            "spooled_script_sha256": f"{index + 1:064x}",
            "spooled_script_proof": {
                "argv": [
                    "scontrol",
                    "write",
                    "batch_script",
                    str(50_000 + index),
                    "-",
                ],
                "local_path": f"/immutable/{index:02d}.sbatch",
                "local_sha256": f"{index + 1:064x}",
                "observed_sha256": f"{index + 1:064x}",
                "observed_bytes": 100,
                "exact_match": True,
            },
            "spooled_provenance": {
                "run_root": "/results/server_pools/schema5-v1",
                "server_pool_id": "schema5-v1",
                "replica_id": f"replica-{index:02d}",
                "replica_index": index,
                "release_id": "sweep-recovery-schema5-v1.2",
                "environment_hash": "6" * 64,
                "model_revision": "model-revision",
                "tokenizer_id": "tokenizer",
                "tokenizer_revision": "tokenizer-revision",
                "model_contract_sha256": "4" * 64,
                "fleet_contract_sha256": "3" * 64,
            },
        }
        if index == 0:
            rows.append(
                common
                | {
                    "state": "PENDING",
                    "reason": pending_reason,
                    "partition": "ou_bcs",
                    "scontrol": {
                        "argv": [
                            "scontrol",
                            "show",
                            "job",
                            "-o",
                            str(50_000 + index),
                        ],
                        "job_id": str(50_000 + index),
                        "job_state": "PENDING",
                        "reason": pending_reason,
                        "comment": f"comment-{index:02d}",
                        "job_name": f"name-{index:02d}",
                        "node": None,
                        "command": f"/immutable/{index:02d}.sbatch",
                        "raw_output_sha256": "1" * 64,
                        "effective_requeue": 0,
                    },
                }
            )
        else:
            rows.append(
                common
                | {
                    "state": "RUNNING",
                    "partition": "ou_bcs",
                    "node": f"node-{index:02d}",
                    "registry_path": f"/registry/{index:02d}.json",
                    "registry_sha256": "2" * 64,
                    "scontrol": {
                        "argv": [
                            "scontrol",
                            "show",
                            "job",
                            "-o",
                            str(50_000 + index),
                        ],
                        "job_id": str(50_000 + index),
                        "job_state": "RUNNING",
                        "reason": None,
                        "comment": f"comment-{index:02d}",
                        "job_name": f"name-{index:02d}",
                        "node": f"node-{index:02d}",
                        "command": f"/immutable/{index:02d}.sbatch",
                        "raw_output_sha256": "1" * 64,
                        "effective_requeue": 0,
                    },
                    "http": {
                        "health_status": 200,
                        "models_status": 200,
                        "model_ids": ["served-model"],
                        "expected_model": "served-model",
                        "probe_started_timestamp": 99.0,
                        "healthy": True,
                        "probe_completed_timestamp": 100.0,
                    },
                }
            )
    return {
        "control_immutable_sha256": IMMUTABLE_SHA,
        "rollout_generation": 1,
        "server_pool_root": "/results/server_pools/schema5-v1",
        "fleet_id": "schema5-v1",
        "fleet_contract_path": "/release/fleet.json",
        "fleet_contract_sha256": "3" * 64,
        "model_contract_sha256": "4" * 64,
        "current_pointer_path": "/fleet-state/CURRENT.json",
        "current_pointer_sha256": "6" * 64,
        "generation_ledger_path": "/fleet-state/ledgers/g000001.json",
        "generation_ledger_sha256": "7" * 64,
        "scheduler_captured_timestamp": 99.0,
        "scheduler_age_seconds": 1.0,
        "scheduler_sources": {"squeue": True, "sacct": True},
        "isolated_foreign_scheduler_job_ids": [],
        "logical_replicas": 22,
        "allocated_gpus": 24,
        "running_replicas": 21,
        "pending_replicas": 1,
        "ignored_current_terminal_job_ids": [],
        "sealed_successful_handoff_terminal_job_ids": [],
        "overlap_replicas": [],
        "running": rows[1:],
        "pending": rows[:1],
        "captured_timestamp": 100.0,
    }


def _materialize_capacity_archive_sources(
    tmp_path: Path, payload: dict
) -> tuple[dict, dict[str, Path]]:
    payload = json.loads(json.dumps(payload))
    pool_root = tmp_path / "server_pools" / "schema5-v1"
    payload["server_pool_root"] = str(pool_root.resolve())
    transaction_state = pool_root / fleet_tx.STATE_DIRECTORY
    rows = [*payload["pending"], *payload["running"]]
    ledger = {
        "schema_version": fleet_tx.STATE_SCHEMA_VERSION,
        "pool_root": str(pool_root.resolve()),
        "pool_id": "schema5-v1",
        "fleet_sha256": payload["fleet_contract_sha256"],
        "rollout_generation": payload["rollout_generation"],
        "visibility_grace_seconds": fleet_tx.DEFAULT_VISIBILITY_GRACE_SECONDS,
        "created_at": 1.0,
        "updated_at": 2.0,
        "replicas": {},
    }
    scripts: dict[str, Path] = {}
    for index, row in enumerate(rows):
        replica_id = row["replica_id"]
        job_id = row["job_id"]
        token = row["intent_token"]
        script_path = (
            transaction_state
            / "sbatch"
            / "g000001"
            / f"{replica_id}.{token}.sbatch"
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
        script_path.chmod(0o444)
        script_sha256 = _sha(script_path)
        scripts[job_id] = script_path
        comment = fleet_tx.intent_comment(
            pool_id="schema5-v1",
            profile=row["serving_profile"],
            replica_id=replica_id,
            rollout_generation=payload["rollout_generation"],
            intent_token=token,
            fleet_sha256=payload["fleet_contract_sha256"],
        )
        row["comment"] = comment
        row["local_script_path"] = str(script_path.resolve())
        row["local_script_sha256"] = script_sha256
        row["spooled_script_sha256"] = script_sha256
        row["spooled_script_proof"].update(
            {
                "local_path": str(script_path.resolve()),
                "local_sha256": script_sha256,
                "observed_sha256": script_sha256,
                "observed_bytes": script_path.stat().st_size,
            }
        )
        row["scontrol"]["comment"] = comment
        row["scontrol"]["command"] = str(script_path.resolve())
        row["spooled_provenance"]["run_root"] = str(pool_root.resolve())
        attempt = {
            "intent_token": token,
            "rollout_generation": payload["rollout_generation"],
            "state": "committed",
            "created_at": 1.0,
            "submit_started_at": 1.1,
            "submission_attempts": 1,
            "sbatch_path": str(script_path.resolve()),
            "sbatch_sha256": script_sha256,
            "submission_transport": (
                fleet_tx.STDIN_EXACT_SUBMISSION_TRANSPORT
            ),
            "submission_argv_sha256": (
                fleet_tx.submission_argv_sha256(comment)
            ),
            "scheduler_comment": comment,
            "job_id": job_id,
            "submitted_at": 1.2,
            "committed_at": 1.3,
            "terminal_at": None,
            "last_seen_at": 2.0,
            "missing_since": None,
            "last_error": None,
            "launch_kind": "primary",
            "lifecycle": "primary",
            "predecessor_job_id": None,
            "predecessor_end_at": None,
            "allocated_gpus": row["allocated_gpus"],
            "scheduler_start_at": 1.4 if row["state"] == "RUNNING" else None,
            "scheduler_end_at": None,
            "scheduler_time_limit_seconds": 86_400,
            "ready_probe_count": 0,
            "last_ready_probe_at": None,
            "promoted_at": None,
            "retire_requested_at": None,
            "last_retire_attempt_at": None,
            "retire_attempts": 0,
            "retire_error": None,
        }
        health = None
        if row["state"] == "RUNNING":
            host = f"node-{index:02d}"
            port = 20_000 + index
            registry_path = (
                tmp_path / "registries" / f"{row['replica_id']}.json"
            )
            _write(
                registry_path,
                {
                    "replica_id": row["replica_id"],
                    "host": host,
                    "port": port,
                },
            )
            row["registry_path"] = str(registry_path.resolve())
            row["registry_sha256"] = _sha(registry_path)
            health = {
                "job_id": job_id,
                "endpoint": f"{host}:{port}",
                "observer_generation": payload["rollout_generation"],
                "first_failure_at": None,
                "last_failure_at": None,
                "last_probe_at": None,
                "consecutive_failures": 0,
                "health_failures": 0,
                "models_failures": 0,
                "cancel_state": None,
                "cancel_requested_at": None,
                "cancel_completed_at": None,
                "cancel_error": None,
                "cancel_attempts": 0,
                "last_cancel_attempt_at": None,
                "next_cancel_eligible_at": None,
                "alert_id": None,
            }
        ledger["replicas"][replica_id] = {
            "attempts": [attempt],
            "health": health,
        }
    ledger_path = (
        transaction_state
        / fleet_tx.LEDGERS_DIRECTORY
        / "g000001.json"
    )
    _write(ledger_path, ledger)
    current = {
        "schema_version": fleet_tx.STATE_SCHEMA_VERSION,
        "pool_root": str(pool_root.resolve()),
        "pool_id": "schema5-v1",
        "fleet_sha256": payload["fleet_contract_sha256"],
        "current_generation": payload["rollout_generation"],
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": _sha(ledger_path),
        "updated_at": 2.0,
    }
    current_path = _write(
        pool_root / fleet_tx.STATE_DIRECTORY / fleet_tx.CURRENT_FILENAME,
        current,
    )
    payload["current_pointer_path"] = str(current_path.resolve())
    payload["current_pointer_sha256"] = _sha(current_path)
    payload["generation_ledger_path"] = str(ledger_path.resolve())
    payload["generation_ledger_sha256"] = _sha(ledger_path)
    return payload, scripts


def _build_capacity_preimage_fixture(
    tmp_path: Path,
) -> tuple[dict, dict, dict[str, Path]]:
    fleet, scripts = _materialize_capacity_archive_sources(
        tmp_path / "live", _capacity_fleet_payload()
    )
    binding = _archive_capacity_preimage(tmp_path, fleet, scripts)
    return binding, fleet, scripts


def _archive_capacity_preimage(
    tmp_path: Path, fleet: dict, scripts: dict[str, Path]
) -> dict:

    def scheduler_runner(argv, _timeout):
        script = scripts[str(argv[-2])]
        return subprocess.CompletedProcess(
            argv, 0, script.read_text(encoding="utf-8"), ""
        )

    archive_parent = tmp_path / "receipt"
    archive_parent.mkdir(parents=True)
    binding = readiness._build_capacity_preimage_archive(
        parent=archive_parent,
        fleet=fleet,
        chain_id="schema5-v1.2-r9",
        chain_generation=1,
        readiness_job_id="4242",
        scheduler_runner=scheduler_runner,
    )
    return binding


def _rewrite_capacity_ledger(fleet: dict, mutate) -> None:
    ledger_path = Path(fleet["generation_ledger_path"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    mutate(ledger)
    _write(ledger_path, ledger)
    fleet["generation_ledger_sha256"] = _sha(ledger_path)
    current_path = Path(fleet["current_pointer_path"])
    current = json.loads(current_path.read_text(encoding="utf-8"))
    current["ledger_sha256"] = fleet["generation_ledger_sha256"]
    _write(current_path, current)
    fleet["current_pointer_sha256"] = _sha(current_path)


def _validate_capacity_preimage_fixture(
    binding: dict, fleet: dict
) -> dict:
    return readiness._validate_capacity_preimage_archive(
        binding,
        fleet=fleet,
        chain_id="schema5-v1.2-r9",
        chain_generation=1,
        readiness_job_id="4242",
    )


def test_capacity_preimage_remains_verifiable_after_live_sources_rotate(
    tmp_path: Path,
) -> None:
    binding, fleet, scripts = _build_capacity_preimage_fixture(tmp_path)
    live_paths = [
        Path(fleet["current_pointer_path"]),
        Path(fleet["generation_ledger_path"]),
        *(Path(row["registry_path"]) for row in fleet["running"]),
        *scripts.values(),
    ]
    for index, source in enumerate(live_paths):
        source.chmod(0o644)
        source.write_bytes(f"rotated-live-source-{index}\n".encode("utf-8"))
    assert _validate_capacity_preimage_fixture(binding, fleet) == binding


def test_capacity_preimage_rejects_schema_only_empty_generation_ledger(
    tmp_path: Path,
) -> None:
    fleet, scripts = _materialize_capacity_archive_sources(
        tmp_path / "live", _capacity_fleet_payload()
    )
    _rewrite_capacity_ledger(
        fleet, lambda ledger: ledger.__setitem__("replicas", {})
    )
    with pytest.raises(readiness.EvidenceError, match="ledger"):
        _archive_capacity_preimage(tmp_path, fleet, scripts)


def test_capacity_preimage_accepts_recovered_cumulative_probe_failures(
    tmp_path: Path,
) -> None:
    fleet, scripts = _materialize_capacity_archive_sources(
        tmp_path / "live", _capacity_fleet_payload()
    )

    def recovered(ledger):
        health = ledger["replicas"]["replica-01"]["health"]
        health["first_failure_at"] = 1.5
        health["last_failure_at"] = 1.6
        health["consecutive_failures"] = 0
        health["health_failures"] = 2
        health["models_failures"] = 1

    _rewrite_capacity_ledger(fleet, recovered)
    binding = _archive_capacity_preimage(tmp_path, fleet, scripts)
    assert _validate_capacity_preimage_fixture(binding, fleet) == binding


def test_capacity_preimage_rejects_health_registry_endpoint_drift(
    tmp_path: Path,
) -> None:
    fleet, scripts = _materialize_capacity_archive_sources(
        tmp_path / "live", _capacity_fleet_payload()
    )
    _rewrite_capacity_ledger(
        fleet,
        lambda ledger: ledger["replicas"]["replica-01"]["health"].__setitem__(
            "endpoint", "other-node:9999"
        ),
    )
    with pytest.raises(readiness.EvidenceError, match="endpoint drifted"):
        _archive_capacity_preimage(tmp_path, fleet, scripts)


@pytest.mark.parametrize(
    "mutation",
    ("bytes", "symlink", "shared_inode", "unexpected_file"),
)
def test_capacity_preimage_rejects_archive_mutation(
    tmp_path: Path, mutation: str
) -> None:
    binding, fleet, _scripts = _build_capacity_preimage_fixture(tmp_path)
    root = Path(binding["root"])
    manifest = json.loads(Path(binding["manifest"]).read_text(encoding="utf-8"))
    record = manifest["files"][0]
    archived = Path(record["archive_path"])
    parent = archived.parent
    if mutation == "bytes":
        archived.chmod(0o644)
        archived.write_bytes(b"tampered archive bytes\n")
        archived.chmod(0o444)
    elif mutation == "symlink":
        parent.chmod(0o755)
        archived.unlink()
        archived.symlink_to(Path(record["source_path"]))
        parent.chmod(0o555)
    elif mutation == "shared_inode":
        source = Path(record["source_path"])
        source.chmod(0o444)
        parent.chmod(0o755)
        archived.unlink()
        os.link(source, archived)
        parent.chmod(0o555)
    else:
        root.chmod(0o755)
        unexpected = root / "unexpected"
        unexpected.write_bytes(b"not in the inventory\n")
        unexpected.chmod(0o444)
        root.chmod(0o555)
    with pytest.raises(readiness.EvidenceError):
        _validate_capacity_preimage_fixture(binding, fleet)


def test_capacity_transient_receipt_is_marker_last_read_only_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _capacity_chain_binding(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        readiness,
        "_capacity_boundary_proof",
        lambda **_kwargs: {
            "argv": ["scontrol", "show", "job", "-o", "4242"],
            "job_id": "4242",
            "comment": binding["receipt_row"]["comment"],
            "job_name": binding["manifest_row"]["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": 36_000,
            "raw_output_sha256": "5" * 64,
        },
    )
    calls = []
    fleet_payload, scripts = _materialize_capacity_archive_sources(
        tmp_path, _capacity_fleet_payload()
    )
    monkeypatch.setattr(
        readiness,
        "_collect_fleet_capacity_transient",
        lambda *_args, **_kwargs: (
            calls.append("collect") or fleet_payload
        ),
    )
    scheduler_runner = lambda argv, _timeout: subprocess.CompletedProcess(
        argv,
        0,
        scripts[str(argv[-2])].read_text(encoding="utf-8"),
        "",
    )
    output = tmp_path / "capacity" / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    report = readiness.build_fleet_capacity_transient_receipt(
        {"immutable_sha256": IMMUTABLE_SHA},
        output=output,
        state_dir=tmp_path / "state",
        chain_manifest=Path(binding["verified"]["manifest_path"]),
        submission_receipt=Path(binding["verified"]["submission_receipt_path"]),
        readiness_job_id="4242",
        scheduler_runner=scheduler_runner,
        now=lambda: 100.0,
    )
    assert report["status"] == "complete"
    evidence = output.parent / readiness.CAPACITY_TRANSIENT_EVIDENCE_NAME
    assert output.is_file() and evidence.is_file()
    assert output.stat().st_mode & 0o222 == 0
    assert evidence.stat().st_mode & 0o222 == 0
    assert calls == ["collect"]
    second = readiness.build_fleet_capacity_transient_receipt(
        {"immutable_sha256": IMMUTABLE_SHA},
        output=output,
        state_dir=tmp_path / "state",
        chain_manifest=Path(binding["verified"]["manifest_path"]),
        submission_receipt=Path(binding["verified"]["submission_receipt_path"]),
        readiness_job_id="4242",
        scheduler_runner=scheduler_runner,
        now=lambda: pytest.fail("idempotent validation consulted the clock"),
    )
    assert second["status"] == "already_complete"
    assert calls == ["collect"]


def test_capacity_transient_crash_before_marker_archives_stale_proof_and_reprobes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _capacity_chain_binding(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: binding,
    )
    clock = {"now": 100.0}
    boundary_calls: list[float] = []
    active_scripts: dict[str, Path] = {}

    def boundary(**_kwargs):
        boundary_calls.append(clock["now"])
        return {
            "argv": ["scontrol", "show", "job", "-o", "4242"],
            "job_id": "4242",
            "comment": binding["receipt_row"]["comment"],
            "job_name": binding["manifest_row"]["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": 36_000,
            "raw_output_sha256": "5" * 64,
        }

    def fleet_payload():
        payload = _capacity_fleet_payload()
        payload["scheduler_captured_timestamp"] = clock["now"] - 1
        payload["captured_timestamp"] = clock["now"]
        for row in payload["running"]:
            row["http"]["probe_started_timestamp"] = clock["now"] - 1
            row["http"]["probe_completed_timestamp"] = clock["now"]
        materialized, scripts = _materialize_capacity_archive_sources(
            tmp_path / f"capture-{int(clock['now'])}", payload
        )
        active_scripts.clear()
        active_scripts.update(scripts)
        return materialized

    def scheduler_runner(argv, _timeout):
        script_path = active_scripts[str(argv[-2])]
        return subprocess.CompletedProcess(
            argv, 0, script_path.read_text(encoding="utf-8"), ""
        )

    monkeypatch.setattr(readiness, "_capacity_boundary_proof", boundary)
    monkeypatch.setattr(
        readiness, "_collect_fleet_capacity_transient", lambda *_a, **_k: fleet_payload()
    )
    output = tmp_path / "capacity" / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    evidence_path = output.parent / readiness.CAPACITY_TRANSIENT_EVIDENCE_NAME
    original_write = readiness._write_json_atomic

    def crash_before_marker(path, payload):
        if Path(path) == output:
            raise RuntimeError("simulated crash after evidence publication")
        return original_write(path, payload)

    monkeypatch.setattr(readiness, "_write_json_atomic", crash_before_marker)
    with pytest.raises(RuntimeError, match="simulated crash"):
        readiness.build_fleet_capacity_transient_receipt(
            {"immutable_sha256": IMMUTABLE_SHA},
            output=output,
            state_dir=tmp_path / "state",
            chain_manifest=Path(binding["verified"]["manifest_path"]),
            submission_receipt=Path(
                binding["verified"]["submission_receipt_path"]
            ),
            readiness_job_id="4242",
            scheduler_runner=scheduler_runner,
            now=lambda: clock["now"],
        )
    stale_sha256 = _sha(evidence_path)
    assert evidence_path.is_file() and not output.exists()

    clock["now"] = 1_000.0
    monkeypatch.setattr(readiness, "_write_json_atomic", original_write)
    result = readiness.build_fleet_capacity_transient_receipt(
        {"immutable_sha256": IMMUTABLE_SHA},
        output=output,
        state_dir=tmp_path / "state",
        chain_manifest=Path(binding["verified"]["manifest_path"]),
        submission_receipt=Path(binding["verified"]["submission_receipt_path"]),
        readiness_job_id="4242",
        scheduler_runner=scheduler_runner,
        now=lambda: clock["now"],
    )
    incidents = list(
        (output.parent / readiness.CAPACITY_TRANSIENT_INCIDENT_DIRECTORY).glob(
            "*.json"
        )
    )
    assert result["status"] == "complete"
    assert boundary_calls == [100.0, 1_000.0]
    assert len(incidents) == 1 and _sha(incidents[0]) == stale_sha256
    assert _sha(evidence_path) != stale_sha256


def test_capacity_transient_invalid_reason_never_publishes_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _capacity_chain_binding(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        readiness,
        "_capacity_boundary_proof",
        lambda **_kwargs: {
            "argv": ["scontrol", "show", "job", "-o", "4242"],
            "job_id": "4242",
            "comment": binding["receipt_row"]["comment"],
            "job_name": binding["manifest_row"]["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": 36_000,
            "raw_output_sha256": "5" * 64,
        },
    )
    invalid_payload, scripts = _materialize_capacity_archive_sources(
        tmp_path,
        _capacity_fleet_payload(pending_reason="QOSMaxGRESPerUser"),
    )
    monkeypatch.setattr(
        readiness,
        "_collect_fleet_capacity_transient",
        lambda *_args, **_kwargs: invalid_payload,
    )
    output = tmp_path / "capacity" / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    with pytest.raises(readiness.EvidenceError, match="narrowly valid"):
        readiness.build_fleet_capacity_transient_receipt(
            {"immutable_sha256": IMMUTABLE_SHA},
            output=output,
            state_dir=tmp_path / "state",
            chain_manifest=Path(binding["verified"]["manifest_path"]),
            submission_receipt=Path(
                binding["verified"]["submission_receipt_path"]
            ),
            readiness_job_id="4242",
            scheduler_runner=lambda argv, _timeout: subprocess.CompletedProcess(
                argv,
                0,
                scripts[str(argv[-2])].read_text(encoding="utf-8"),
                "",
            ),
            now=lambda: 100.0,
        )
    assert not output.exists()


def test_capacity_transient_recollects_after_live_preimage_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _capacity_chain_binding(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        readiness,
        "_capacity_boundary_proof",
        lambda **_kwargs: {
            "argv": ["scontrol", "show", "job", "-o", "4242"],
            "job_id": "4242",
            "comment": binding["receipt_row"]["comment"],
            "job_name": binding["manifest_row"]["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": 36_000,
            "raw_output_sha256": "5" * 64,
        },
    )
    collections = 0
    active_scripts: dict[str, Path] = {}

    def collect(*_args, **_kwargs):
        nonlocal collections
        collections += 1
        fleet, scripts = _materialize_capacity_archive_sources(
            tmp_path / f"capture-{collections}", _capacity_fleet_payload()
        )
        active_scripts.clear()
        active_scripts.update(scripts)
        if collections == 1:
            Path(fleet["current_pointer_path"]).write_bytes(
                b"concurrent fleet supervisor rotation\n"
            )
        return fleet

    monkeypatch.setattr(
        readiness, "_collect_fleet_capacity_transient", collect
    )
    output = tmp_path / "capacity" / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    report = readiness.build_fleet_capacity_transient_receipt(
        {"immutable_sha256": IMMUTABLE_SHA},
        output=output,
        state_dir=tmp_path / "state",
        chain_manifest=Path(binding["verified"]["manifest_path"]),
        submission_receipt=Path(binding["verified"]["submission_receipt_path"]),
        readiness_job_id="4242",
        scheduler_runner=lambda argv, _timeout: subprocess.CompletedProcess(
            argv,
            0,
            active_scripts[str(argv[-2])].read_text(encoding="utf-8"),
            "",
        ),
        now=lambda: 100.0,
    )
    assert report["status"] == "complete"
    assert collections == 2
    assert output.is_file()
    assert len(
        list(
            (
                output.parent
                / readiness.CAPACITY_TRANSIENT_INCIDENT_DIRECTORY
            ).glob(f"{readiness.CAPACITY_PREIMAGE_ROOT_NAME}.partial-*")
        )
    ) == 1


def test_capacity_transient_three_preimage_drifts_publish_no_evidence_or_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _capacity_chain_binding(tmp_path)
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: binding,
    )
    monkeypatch.setattr(
        readiness,
        "_capacity_boundary_proof",
        lambda **_kwargs: {
            "argv": ["scontrol", "show", "job", "-o", "4242"],
            "job_id": "4242",
            "comment": binding["receipt_row"]["comment"],
            "job_name": binding["manifest_row"]["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": 36_000,
            "raw_output_sha256": "5" * 64,
        },
    )
    collections = 0

    def collect(*_args, **_kwargs):
        nonlocal collections
        collections += 1
        fleet, _scripts = _materialize_capacity_archive_sources(
            tmp_path / f"capture-{collections}", _capacity_fleet_payload()
        )
        Path(fleet["current_pointer_path"]).write_bytes(
            f"concurrent drift {collections}\n".encode("utf-8")
        )
        return fleet

    monkeypatch.setattr(
        readiness, "_collect_fleet_capacity_transient", collect
    )
    output = tmp_path / "capacity" / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    with pytest.raises(
        readiness.EvidenceError, match="did not stabilize after three"
    ):
        readiness.build_fleet_capacity_transient_receipt(
            {"immutable_sha256": IMMUTABLE_SHA},
            output=output,
            state_dir=tmp_path / "state",
            chain_manifest=Path(binding["verified"]["manifest_path"]),
            submission_receipt=Path(
                binding["verified"]["submission_receipt_path"]
            ),
            readiness_job_id="4242",
            scheduler_runner=lambda *_args, **_kwargs: pytest.fail(
                "CURRENT drift must fail before a scheduler spool copy"
            ),
            now=lambda: 100.0,
        )
    assert collections == 3
    assert not output.exists()
    assert not (
        output.parent / readiness.CAPACITY_TRANSIENT_EVIDENCE_NAME
    ).exists()
    assert len(
        list(
            (
                output.parent
                / readiness.CAPACITY_TRANSIENT_INCIDENT_DIRECTORY
            ).glob(f"{readiness.CAPACITY_PREIMAGE_ROOT_NAME}.partial-*")
        )
    ) == 3


def test_capacity_exact_scontrol_parser_rejects_ambiguous_identity() -> None:
    with pytest.raises(readiness.EvidenceError, match="exactly one"):
        readiness._parse_scontrol_record(
            (
                "JobId=4242 JobState=PENDING Reason=Resources Requeue=0\n"
                "JobId=4242 JobState=PENDING Reason=Resources Requeue=0\n"
            ),
            expected_job_id="4242",
            description="fixture",
        )
    with pytest.raises(readiness.EvidenceError, match="exact JobId"):
        readiness._parse_scontrol_record(
            "JobId=9999 JobState=PENDING Reason=Resources Requeue=0\n",
            expected_job_id="4242",
            description="fixture",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("JobState", "RUNNING"),
        ("Reason", "QOSMaxGRESPerUser"),
        ("Requeue", "1"),
        ("Comment", "foreign-comment"),
        ("JobName", "foreign-name"),
        ("Command", "/immutable/other.sbatch"),
        ("JobId", "9999"),
    ],
)
def test_pending_capacity_scontrol_contract_rejects_each_identity_drift(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    script = tmp_path / "replica.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    fields = {
        "JobId": "4242",
        "JobState": "PENDING",
        "Reason": "Priority",
        "Requeue": "0",
        "Comment": "exact-comment",
        "JobName": "exact-name",
        "Command": str(script.resolve()),
    }
    fields[field] = value
    with pytest.raises(readiness.EvidenceError, match="solely blocked"):
        readiness._validate_pending_capacity_scontrol(
            fields,
            job_id="4242",
            expected_comment="exact-comment",
            expected_job_name="exact-name",
            script_path=script,
        )


def test_pending_capacity_scontrol_contract_accepts_only_priority_or_resources(
    tmp_path: Path,
) -> None:
    script = tmp_path / "replica.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    base = {
        "JobId": "4242",
        "JobState": "PENDING",
        "Requeue": "0",
        "Comment": "exact-comment",
        "JobName": "exact-name",
        "Command": str(script.resolve()),
    }
    for reason in ("Priority", "Resources"):
        assert readiness._validate_pending_capacity_scontrol(
            base | {"Reason": reason},
            job_id="4242",
            expected_comment="exact-comment",
            expected_job_name="exact-name",
            script_path=script,
        ) == reason


def test_capacity_handoff_history_accepts_multi_hop_sealed_retirements() -> None:
    def attempt(
        token: str,
        job_id: str,
        *,
        state: str,
        lifecycle: str,
        launch_kind: str,
        predecessor: str | None,
    ) -> dict:
        historical = state == "terminal"
        return {
            "intent_token": token,
            "job_id": job_id,
            "state": state,
            "lifecycle": lifecycle,
            "launch_kind": launch_kind,
            "predecessor_job_id": predecessor,
            "committed_at": 1.0,
            "last_error": None,
            "retire_error": None,
            "terminal_at": 3.0 if historical else None,
            "retire_requested_at": 2.0 if historical else None,
            "retire_attempts": 1 if historical else 0,
            "last_retire_attempt_at": 2.0 if historical else None,
            "promoted_at": 1.5 if launch_kind == "handoff" else None,
            "ready_probe_count": 2 if launch_kind == "handoff" else 0,
        }

    first = attempt(
        "1" * 32,
        "101",
        state="terminal",
        lifecycle="retiring",
        launch_kind="primary",
        predecessor=None,
    )
    second = attempt(
        "2" * 32,
        "102",
        state="terminal",
        lifecycle="retiring",
        launch_kind="handoff",
        predecessor="101",
    )
    active = attempt(
        "3" * 32,
        "103",
        state="committed",
        lifecycle="promoted",
        launch_kind="handoff",
        predecessor="102",
    )
    ledger = {
        "replicas": {
            "replica": {
                "attempts": [first, second, active],
                "health": {},
            }
        }
    }
    allocation = SimpleNamespace(
        replica_id="replica",
        attempt={"intent_token": active["intent_token"]},
        row=SimpleNamespace(job_id="103"),
    )
    assert readiness._successful_capacity_handoff_history(
        current_ledger=ledger,
        logical_allocations=[allocation],
    ) == {"101", "102"}
    second["retire_error"] = "ambiguous cancellation"
    with pytest.raises(readiness.EvidenceError, match="failed/orphaned"):
        readiness._successful_capacity_handoff_history(
            current_ledger=ledger,
            logical_allocations=[allocation],
        )


def test_capacity_scheduler_scope_ignores_isolated_canary_but_rejects_collision() -> None:
    fleet_sha = "a" * 64
    replica = SimpleNamespace(
        replica_id="schema5-v1--0.6B--r00",
        serving_profile="0.6B",
        scheduler_job_name="asys-s5-serve-0.6B-r00",
    )
    fleet = SimpleNamespace(
        fleet_id="schema5-v1",
        sha256=fleet_sha,
        replicas=[replica],
    )
    production_comment = readiness.fleet_tx.intent_comment(
        pool_id="schema5-v1",
        profile="0.6B",
        replica_id=replica.replica_id,
        rollout_generation=1,
        intent_token="1" * 32,
        fleet_sha256=fleet_sha,
    )
    canary_comment = readiness.fleet_tx.intent_comment(
        pool_id="schema5-canary",
        profile="canary",
        replica_id="canary-r00",
        rollout_generation=1,
        intent_token="2" * 32,
        fleet_sha256="b" * 64,
    )
    production = keepalive.FleetQueueRow(
        "101",
        replica.scheduler_job_name,
        "RUNNING",
        "partition",
        "node",
        "/jobs/production.sbatch",
        production_comment,
    )
    canary = keepalive.FleetQueueRow(
        "102",
        "asys-s5-serve-canary-r00",
        "COMPLETED",
        "partition",
        "node",
        "/jobs/canary.sbatch",
        canary_comment,
    )
    scoped, isolated = readiness._scope_capacity_scheduler_rows(
        [canary, production], fleet=fleet
    )
    assert scoped == (production,)
    assert isolated == ("102",)
    collision = keepalive.FleetQueueRow(
        "103",
        replica.scheduler_job_name,
        "COMPLETED",
        "partition",
        "node",
        "/jobs/collision.sbatch",
        canary_comment,
    )
    with pytest.raises(readiness.EvidenceError, match="partially collides"):
        readiness._scope_capacity_scheduler_rows(
            [production, collision], fleet=fleet
        )


def test_pending_replica_must_have_no_promoted_registry_pointer() -> None:
    record = (SimpleNamespace(), Path("/registry/pending.json"))
    with pytest.raises(readiness.EvidenceError, match="pending_pointers"):
        readiness._validate_capacity_registry_membership(
            {"pending": record, "running": record},
            running_ids={"running"},
            pending_ids={"pending"},
        )
    readiness._validate_capacity_registry_membership(
        {"running": record},
        running_ids={"running"},
        pending_ids={"pending"},
    )


def test_exact_spooled_script_proof_rejects_different_slurm_bytes(
    tmp_path: Path,
) -> None:
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")

    def runner(_argv, _timeout):
        return subprocess.CompletedProcess([], 0, "#!/bin/bash\nfalse\n", "")

    with pytest.raises(readiness.EvidenceError, match="differs"):
        readiness._exact_spooled_script_proof(
            job_id="4242",
            script_path=script,
            scheduler_runner=runner,
        )
