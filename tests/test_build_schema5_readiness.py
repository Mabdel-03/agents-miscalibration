"""Focused tests for artifact-derived schema-5 readiness publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from scripts import build_schema5_readiness as readiness
from slurm import schema5_control as control
from slurm import keepalive
from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving.model_contracts import load_model_contracts
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.serving.registry import ServerEntry


IMMUTABLE_SHA = "a" * 64


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


def _legacy_cleanup(tmp_path: Path) -> Path:
    response = _write(
        tmp_path / "response.json",
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
        tmp_path / "checkpoint.json",
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
        tmp_path / "permanent.json",
        {
            "schema_version": 1,
            "passed": True,
            "permanent_ledgers_archived": 3,
            "unresolved_permanent_ledgers": 0,
            "referenced_artifacts": [],
        },
    )
    semantic = _write(
        tmp_path / "semantic.json",
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
    return _write(
        tmp_path / "LEGACY_CLEANUP_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "passed": True,
            "snapshot_id": "pre",
            "migration_metrics": readiness.MIGRATION_OUTER_METRICS,
            "semantic_metrics": readiness.SEMANTIC_METRICS,
            "artifacts": [
                _ref("response_incident_archive_report", response),
                _ref("checkpoint_migration_report", checkpoint),
                _ref("permanent_ledger_archive_report", permanent),
                _ref("legacy_semantic_audit_report", semantic),
            ],
        },
    )


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


def test_migration_gate_rejects_marker_metric_forgery(tmp_path: Path):
    marker = _legacy_cleanup(tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["migration_metrics"]["sealed_incident_qids"] = 1_063
    _write(marker, payload)
    with pytest.raises(readiness.EvidenceError, match="metrics drifted"):
        readiness.build_migrations_gate(
            {"immutable_sha256": IMMUTABLE_SHA},
            cleanup_marker=marker,
            output=tmp_path / "gate.json",
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
            "summary": {
                "passed": True,
                "selected_cells": 216,
                "audited_requests": 43_091,
                "failed_requests": 0,
                "failed_cells": 0,
                "minimum_context_headroom_tokens": 1_287,
            },
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
                }
            ]
        }
    }
    with pytest.raises(readiness.EvidenceError, match="audited_requests drifted"):
        readiness._validate_context_source(
            fake_control, name="dense_peer_context_audit", path=report
        )


def test_scheduler_parser_rejects_non_numeric_allocation_id():
    def runner(_argv, _timeout):
        return subprocess.CompletedProcess(
            [], 0, "not-a-job|asys-s5-serve-4b-s-r00|RUNNING|p|n|cmd|comment\n", ""
        )

    with pytest.raises(readiness.EvidenceError, match="not exact numeric"):
        readiness._fleet_scheduler_rows(runner=runner)


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
    pool = tmp_path / "results" / "server_pools" / "schema5-v1"
    records = {}
    scheduler = {}
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
            release_id="sweep-recovery-schema5-v1",
            environment_hash=environment_hash,
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
            model_contract_sha256=model_hash,
            fleet_contract_sha256=fleet_hash,
            server_pool_id="schema5-v1",
            replica_id=replica.replica_id,
            replica_index=replica.replica_index,
        )
        registry_path = (
            pool
            / "servers"
            / replica.serving_profile
            / f"{host}_{entry.port}.json"
        )
        _write(registry_path, asdict(entry))
        records[replica.replica_id] = (entry, registry_path.resolve())
        row = keepalive.FleetQueueRow(
            job_id,
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            host,
            "spooled",
            (
                f"asys-schema5-pool:schema5-v1;profile={replica.serving_profile};"
                f"replica={replica.replica_id}"
            ),
        )
        provenance = keepalive.SpooledServingProvenance(
            run_root=str(pool.resolve()),
            server_pool_id="schema5-v1",
            replica_id=replica.replica_id,
            replica_index=replica.replica_index,
            release_id="sweep-recovery-schema5-v1",
            environment_hash=environment_hash,
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
            model_contract_sha256=model_hash,
            fleet_contract_sha256=fleet_hash,
        )
        scheduler[replica.replica_id] = (row, provenance)

    monkeypatch.setattr(
        readiness,
        "_verify_scheduler_and_spool",
        lambda **_kwargs: scheduler,
    )
    monkeypatch.setattr(readiness, "_verify_registry", lambda **_kwargs: records)

    def scheduler_runner(_argv, _timeout):
        return subprocess.CompletedProcess([], 0, "", "")

    def probe(_entry, expected_model, _timeout):
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
        "immutable": {
            "release_id": "sweep-recovery-schema5-v1",
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
        scheduler_runner=scheduler_runner,
        probe=probe,
        now=lambda: 100.0,
    )
    assert report["metrics"]["logical_replicas"] == 22
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
    raw = json.loads(Path(raw_ref["path"]).read_text(encoding="utf-8"))
    assert len(raw["replicas"]) == len(raw["referenced_artifacts"]) == 22


def test_email_gate_requires_apply_and_preserves_submission_receipt(tmp_path: Path):
    current = {
        "immutable_sha256": IMMUTABLE_SHA,
        "alert_email": "mabdel03@mit.edu",
    }
    output = tmp_path / "email.json"
    with pytest.raises(readiness.EvidenceError, match="requires --apply"):
        readiness.build_email_gate(current, output=output, apply=False)

    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "queued\n", "")

    report = readiness.build_email_gate(
        current,
        output=output,
        apply=True,
        mail_runner=runner,
        now=lambda: 123.0,
    )
    assert report["metrics"] == {
        "recipient": "mabdel03@mit.edu",
        "delivery_succeeded": True,
        "returncode": 0,
    }
    assert calls[0][0] == [
        "mail",
        "-s",
        "[agents-scaling] schema-5 readiness delivery test",
        "mabdel03@mit.edu",
    ]
    wrapper = json.loads(
        Path(report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    [receipt] = wrapper["referenced_artifacts"]
    raw = json.loads(Path(receipt["path"]).read_text(encoding="utf-8"))
    assert raw["submitted_timestamp"] == 123.0
    assert raw["stdout"] == "queued\n"
