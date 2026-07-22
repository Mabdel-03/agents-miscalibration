from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

from slurm import schema5_control as control
from agents_scaling import runtime_integrity, snapshot_integrity


CLUSTER_FIXTURES = Path(__file__).parent / "fixtures" / "slurm_schema5_cluster"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, payload: str) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return str(path.resolve()), _sha(path)


def _checksummed(path: Path, payload: bytes) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return str(path.resolve()), digest


def _synthetic_scientific_contract(
    *, run_id: str, cell_count: int, expected_qids: int
) -> tuple[bytes, bytes]:
    low = expected_qids // cell_count
    high = low + 1
    high_cells = expected_qids - low * cell_count
    rows = [
        {"benchmark": f"synthetic-{high}", "n_questions": high, "seed": 0}
        for _ in range(high_cells)
    ] + [
        {"benchmark": f"synthetic-{low}", "n_questions": low, "seed": 0}
        for _ in range(cell_count - high_cells)
    ]
    manifest = (json.dumps(rows, separators=(",", ":")) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    contracts = []
    for count in sorted({low, high}):
        contracts.append(
            {
                "key": {
                    "benchmark": f"synthetic-{count}",
                    "n_questions": count,
                    "seed": 0,
                },
                "question_count": count,
                "ordered_qids": [f"{run_id}-{count}-{index}" for index in range(count)],
            }
        )
    benchmark = (
        json.dumps(
            {
                "schema_version": 1,
                "manifest_filename": "cells.json",
                "manifest_sha256": manifest_sha,
                "manifest_cell_count": cell_count,
                "contracts": contracts,
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return manifest, benchmark


def make_pins(tmp_path: Path) -> dict:
    release = tmp_path / "release"
    (release / "slurm").mkdir(parents=True)
    source_slurm = Path(control.__file__).resolve().parent
    for name in (
        "schema5_control.py",
        "schema5_dispatcher.sbatch.tmpl",
        "schema5_fleet_supervisor.sbatch.tmpl",
        "dispatch_sweeps.py",
        "run_dispatch_batch.sbatch.tmpl",
        "keepalive.py",
        "common.sh",
    ):
        shutil.copy2(source_slurm / name, release / "slurm" / name)

    harness = tmp_path / "envs" / "harness"
    serving = tmp_path / "envs" / "serving"
    (harness / "bin").mkdir(parents=True)
    (serving / "bin").mkdir(parents=True)
    (harness / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    (release / "scripts").mkdir()
    (release / "configs").mkdir()
    (release / "scripts" / "schema5_monitor.py").write_text(
        "#!/usr/bin/env python3\n", encoding="utf-8"
    )
    (release / "configs" / "schema5_monitoring.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    model_path, model_hash = _artifact(tmp_path / "model_contracts.v1.json", "{}\n")
    fleet_profiles = []
    for profile_name, replica_count in control.EXPECTED_FLEET_PROFILES.items():
        model_size = profile_name.removesuffix("-long")
        fleet_profiles.append(
            {
                "serving_profile": profile_name,
                "model_size": model_size,
                "model_revision": f"revision-{model_size}",
                "tokenizer_id": f"tokenizer-{model_size}",
                "tokenizer_revision": f"tokenizer-revision-{model_size}",
                "replicas": [
                    {
                        "replica_id": f"schema5-v1--{profile_name}--r{index:02d}",
                        "replica_index": index,
                        "partition": "test-partition",
                    }
                    for index in range(replica_count)
                ],
            }
        )
    fleet_payload = {
        "schema_version": 1,
        "release_id": "sweep-recovery-schema5-v1.1",
        "fleet_id": "schema5-v1",
        "logical_replica_count": 22,
        "allocated_gpu_count": 24,
        "model_contract_sha256": model_hash,
        "server_pool": {
            "root_suffix": "server_pools/schema5-v1",
            "no_requeue": True,
        },
        "profiles": fleet_profiles,
    }
    fleet_path, fleet_hash = _artifact(
        tmp_path / "fleet_contract.v1.json", json.dumps(fleet_payload) + "\n"
    )
    results_root = tmp_path / "results"
    server_pool = results_root / "server_pools" / "schema5-v1"
    server_pool.mkdir(parents=True)

    runs = []
    for run_id, cell_count in control.REQUIRED_RUNS.items():
        run_root = results_root / run_id
        run_root.mkdir()
        manifest, benchmark = _synthetic_scientific_contract(
            run_id=run_id,
            cell_count=cell_count,
            expected_qids=control.REQUIRED_RUN_QIDS[run_id],
        )
        manifest_path, manifest_hash = _checksummed(run_root / "cells.json", manifest)
        benchmark_path, benchmark_hash = _checksummed(
            run_root / "benchmark_contracts.v1.json", benchmark
        )
        lineage_path, lineage_hash = _checksummed(
            run_root / "lineage.schema5-v1.json", f"{run_id}:lineage\n".encode()
        )
        policy_path, policy_hash = _checksummed(
            run_root / "artifact_policy.schema5-v1.json", f"{run_id}:policy\n".encode()
        )
        runs.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "cell_count": cell_count,
                "expected_qids": control.REQUIRED_RUN_QIDS[run_id],
                "weight": 1.0,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
                "lineage_path": lineage_path,
                "lineage_sha256": lineage_hash,
                "policy_path": policy_path,
                "policy_sha256": policy_hash,
                "benchmark_contract_path": benchmark_path,
                "benchmark_contract_sha256": benchmark_hash,
            }
        )
    release_bundle = tmp_path / "release-bundle"
    release_bundle.mkdir()
    harness_manifest_path = release_bundle / control.HARNESS_ENVIRONMENT_FILENAME
    serving_manifest_path = release_bundle / control.SERVING_ENVIRONMENT_FILENAME
    for prefix in (harness, serving):
        for path in sorted(prefix.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        prefix.chmod(0o555)
    (harness / "bin" / "python").chmod(0o555)
    environment_inventories = {}
    for role, prefix, manifest_path in (
        ("harness", harness, harness_manifest_path),
        ("serving", serving, serving_manifest_path),
    ):
        inventory = runtime_integrity.directory_inventory(prefix)
        environment_inventories[role] = inventory
        runtime = {"fixture": True}
        locks = {"conda_explicit": [], "pip_freeze_all": []}
        payload = {
            "schema_version": 1,
            "release_id": "sweep-recovery-schema5-v1.1",
            "role": role,
            "prefix": str(prefix),
            "sealed_read_only": True,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            "runtime": runtime,
            "locks": locks,
            "release_package": None,
            "directory_inventory": inventory,
        }
        payload["environment_content_sha256"] = hashlib.sha256(
            runtime_integrity.canonical_bytes(
                {
                    "runtime": runtime,
                    "locks": locks,
                    "release_package": None,
                    "inventory_sha256": inventory["inventory_sha256"],
                }
            )
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    harness_hash = _sha(harness_manifest_path)
    serving_hash = _sha(serving_manifest_path)
    source_tree_sha256 = control.sha256_tree(release)
    pins = {
        "release_id": "sweep-recovery-schema5-v1.1",
        "release_bundle_root": str(release_bundle),
        "release_bundle_id": "pending",
        "release_worktree": str(release),
        "git_commit": "a" * 40,
        "source_tree_sha256": source_tree_sha256,
        "model_contract_path": model_path,
        "model_contract_sha256": model_hash,
        "fleet_contract_path": fleet_path,
        "fleet_contract_sha256": fleet_hash,
        "harness_environment_prefix": str(harness),
        "harness_environment_manifest_path": str(harness_manifest_path),
        "harness_environment_sha256": harness_hash,
        "serving_environment_prefix": str(serving),
        "serving_environment_manifest_path": str(serving_manifest_path),
        "serving_environment_sha256": serving_hash,
        "hf_home": str(hf_home),
        "results_root": str(results_root),
        "server_pool_root": str(server_pool),
        "dispatcher_command": [],
        "fleet_supervisor_command": [],
        "runs": runs,
    }
    pins["dispatcher_command"] = control.expected_dispatcher_command(pins)
    pins["fleet_supervisor_command"] = control.expected_fleet_supervisor_command(pins)
    materialization_paths = {
        "source_repository": str(release),
        "release_worktree": str(release),
        "source_harness_prefix": str(harness) + ".source",
        "source_serving_prefix": str(serving) + ".source",
        "harness_prefix": str(harness),
        "serving_prefix": str(serving),
    }
    materialization_stage_records = {}
    for role, filename in control.MATERIALIZATION_STAGE_FILENAMES.items():
        stage_payload = {
            "schema_version": 2,
            "release_id": pins["release_id"],
            "stage": role,
        }
        stage_payload["record_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    stage_payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        stage_path = release_bundle.parent / filename
        stage_path.write_text(
            json.dumps(
                stage_payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        stage_path.chmod(0o444)
        materialization_stage_records[role] = {
            "filename": filename,
            "sha256": _sha(stage_path),
            "record_sha256": stage_payload["record_sha256"],
        }
    materialization_marker = {
        "schema_version": 2,
        "release_id": pins["release_id"],
        "git_tag": pins["release_id"],
        "tag_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
        "paths": materialization_paths,
        "complete": True,
        "stage_records": materialization_stage_records,
    }
    materialization_marker["materialization_id"] = hashlib.sha256(
        (
            json.dumps(
                materialization_marker,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    materialization_marker_path = (
        release_bundle.parent / control.MATERIALIZATION_COMPLETE_FILENAME
    )
    materialization_marker_path.write_text(
        json.dumps(
            materialization_marker,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    materialization_marker_path.chmod(0o444)
    materialization_binding = {
        "schema_version": 2,
        "release_id": pins["release_id"],
        "root": str(release_bundle.parent),
        "marker_path": str(materialization_marker_path),
        "marker_sha256": _sha(materialization_marker_path),
        "materialization_id": materialization_marker["materialization_id"],
        "tag_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
        "paths": materialization_paths,
        "stage_records": materialization_stage_records,
    }
    fragment_fields = {
        "release_id",
        "release_worktree",
        "git_commit",
        "source_tree_sha256",
        "model_contract_path",
        "model_contract_sha256",
        "fleet_contract_path",
        "fleet_contract_sha256",
        "harness_environment_prefix",
        "harness_environment_manifest_path",
        "harness_environment_sha256",
        "serving_environment_prefix",
        "serving_environment_manifest_path",
        "serving_environment_sha256",
    }
    identity_path = release_bundle / control.RELEASE_IDENTITY_FILENAME
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": pins["release_id"],
                "git": {
                    "git_commit": pins["git_commit"],
                    "git_tag": pins["release_id"],
                    "source_tree_sha256": pins["source_tree_sha256"],
                },
                "release_worktree": pins["release_worktree"],
                "worktree_sealed_read_only": True,
                "materialization": materialization_binding,
                "offline_environment": {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
                "model_contract": {"path": model_path, "sha256": model_hash},
                "fleet_contract": {
                    "path": fleet_path,
                    "sha256": fleet_hash,
                    "fleet_id": "schema5-v1",
                    "logical_replica_count": 22,
                    "allocated_gpu_count": 24,
                },
                "environments": {
                    "harness": {
                        "prefix": str(harness),
                        "manifest_path": str(harness_manifest_path),
                        "manifest_sha256": harness_hash,
                        "directory_inventory_sha256": environment_inventories[
                            "harness"
                        ]["inventory_sha256"],
                    },
                    "serving": {
                        "prefix": str(serving),
                        "manifest_path": str(serving_manifest_path),
                        "manifest_sha256": serving_hash,
                        "directory_inventory_sha256": environment_inventories[
                            "serving"
                        ]["inventory_sha256"],
                    },
                },
                "publication": {
                    "protocol": "fsync_verify_marker_last",
                    "complete_marker": control.RELEASE_COMPLETE_FILENAME,
                },
                "control_pin_fragment": {
                    field: pins[field] for field in fragment_fields
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {}
    for path in (harness_manifest_path, serving_manifest_path, identity_path):
        checksum_path = release_bundle / (path.name + ".sha256")
        checksum_path.write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")
    for path in sorted(release_bundle.iterdir()):
        artifacts[path.name] = {"sha256": _sha(path), "size": path.stat().st_size}
        path.chmod(0o444)
    marker = {
        "schema_version": 1,
        "release_id": pins["release_id"],
        "complete": True,
        "publication_protocol": "fsync_verify_marker_last",
        "artifacts": artifacts,
        "git_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
    }
    marker["release_bundle_id"] = control.sha256_value(marker)
    marker_path = release_bundle / control.RELEASE_COMPLETE_FILENAME
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    marker_path.chmod(0o444)
    pins["release_bundle_id"] = marker["release_bundle_id"]
    for root in (release, harness, serving, release_bundle):
        root.chmod(0o555)
    return pins


def make_preparable_pins(tmp_path: Path) -> dict:
    """Upgrade the compact base fixture to the real clone lineage/policy schema."""

    pins = make_pins(tmp_path)
    for run in pins["runs"]:
        run_id = run["run_id"]
        run_root = Path(run["run_root"])
        target_artifacts = {
            "cells.json": {
                "sha256": run["manifest_sha256"],
                "size": Path(run["manifest_path"]).stat().st_size,
            },
            "benchmark_contracts.v1.json": {
                "sha256": run["benchmark_contract_sha256"],
                "size": Path(run["benchmark_contract_path"]).stat().st_size,
            },
        }
        lineage = {
            "schema_version": 1,
            "clone_mode": "byte_for_byte_scientific_contract",
            "source_run_id": run_id.replace("_schema5_v1", "_v1"),
            "target_run_id": run_id,
            "source_artifacts": copy.deepcopy(target_artifacts),
            "target_artifacts": target_artifacts,
            "manifest_cell_count": run["cell_count"],
            "imported_result_rows": 0,
            "transformations": [],
            "lineage_id": "1" * 64,
        }
        policy = {
            "schema_version": 1,
            "run_id": run_id,
            "authoritative": True,
            "required_artifact_schema_version": 5,
            "accepted_manifest_sha256": run["manifest_sha256"],
            "accepted_benchmark_contracts_sha256": run["benchmark_contract_sha256"],
            "accepted_model_contract_sha256": pins["model_contract_sha256"],
            "release": {
                "release_id": pins["release_id"],
                "git_commit": pins["git_commit"],
                "source_tree_sha256": pins["source_tree_sha256"],
            },
            "environment": {
                "harness_sha256": pins["harness_environment_sha256"],
                "serving_sha256": pins["serving_environment_sha256"],
            },
            "required_metadata_fields": [],
            "legacy_result_import_allowed": False,
            "policy_id": "2" * 64,
        }
        for name, payload in (("lineage", lineage), ("policy", policy)):
            path = run_root / (
                "lineage.schema5-v1.json"
                if name == "lineage"
                else "artifact_policy.schema5-v1.json"
            )
            path_value, digest = _checksummed(
                path,
                (json.dumps(payload, sort_keys=True) + "\n").encode(),
            )
            run[f"{name}_path"] = path_value
            run[f"{name}_sha256"] = digest
    return pins


def initialize(tmp_path: Path) -> tuple[Path, dict]:
    state_dir = tmp_path / "results" / ".dispatcher-schema5-v1"
    pins = make_pins(tmp_path)
    return state_dir, control.initialize_control(state_dir, pins=pins, now=10.0)


def _write_json_artifact(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {"name": path.stem, "path": str(path.resolve()), "sha256": _sha(path)}


def _snapshot_envelope(state_dir: Path, name: str) -> Path:
    root = state_dir / "readiness" / f"{name}.sealed"
    root.mkdir(parents=True)
    payload_path = root / "payload.bin"
    payload_path.write_bytes(f"{name}:payload\n".encode())
    payload_path.chmod(0o444)
    inventory = f"{_sha(payload_path)}  payload.bin\n"
    inventory_sha = hashlib.sha256(inventory.encode()).hexdigest()
    snapshot_id = f"schema5-v1-{name}-{inventory_sha[:16]}"
    completed_at = "2026-01-01T00:00:00Z"
    (root / "SOURCE_INVENTORY.sha256").write_text(inventory, encoding="utf-8")
    (root / "SNAPSHOT_INVENTORY.sha256").write_text(inventory, encoding="utf-8")
    (root / "DIRECTORY_INVENTORY.txt").write_text("", encoding="utf-8")
    (root / "SNAPSHOT_CATALOG.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "file_count": 1,
                "total_bytes": payload_path.stat().st_size,
                "source_inventory_sha256": inventory_sha,
                "snapshot_inventory_sha256": inventory_sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "SNAPSHOT_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "completed_at": completed_at,
                "file_count": 1,
                "total_bytes": payload_path.stat().st_size,
                "snapshot_inventory_sha256": inventory_sha,
                "verified": True,
                "read_only": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controls = {}
    for filename in sorted(control.SNAPSHOT_CONTROL_FILENAMES):
        path = root / filename
        path.chmod(0o444)
        controls[filename] = {"sha256": _sha(path), "size": path.stat().st_size}
    root.chmod(0o555)
    envelope = state_dir / "readiness" / f"{name}.json"
    envelope.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "recovery_snapshot_external_attestation",
                "passed": True,
                "snapshot_root": str(root.resolve()),
                "snapshot_id": snapshot_id,
                "file_count": 1,
                "total_bytes": payload_path.stat().st_size,
                "control_artifacts": controls,
                "attested_at": "2026-01-01T00:00:01Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return envelope


def _gate_metrics(control_state: dict, gate: str) -> dict:
    if gate == "snapshot":
        return {
            "snapshot_count": 2,
            "pre_repair_verified": True,
            "legacy_consolidated_verified": True,
        }
    if gate == "migrations":
        return {
            "protocol_incidents_total": 22,
            "protocol_already_reset": 22,
            "sealed_incident_qids": 1_064,
            "migrated_checkpoints": 47,
            "remaining_schema1_checkpoints": 0,
            "permanent_ledgers_archived": 3,
            "unresolved_permanent_ledgers": 0,
        }
    if gate == "semantic_audit":
        return {
            "complete_cells": 740,
            "active_validated_qids": 888_068,
            "corrupt_cells": 0,
            "permanent_cells": 0,
            "malformed_lines": 0,
            "duplicate_qids": 0,
            "unexpected_qids": 0,
            "repair_count": 0,
        }
    if gate == "fleet":
        return {
            "logical_replicas": 22,
            "allocated_gpus": 24,
            "healthy_replicas": 22,
            "unhealthy_replicas": 0,
            "profile_replicas": control.EXPECTED_FLEET_PROFILES,
            "revision_mismatches": 0,
            "missing_profiles": 0,
            "stale_registrations": 0,
            "max_heartbeat_age_seconds": 0,
            "captured_timestamp": 23.0,
            "model_contract_sha256": control_state["immutable"][
                "model_contract_sha256"
            ],
            "fleet_contract_sha256": control_state["immutable"][
                "fleet_contract_sha256"
            ],
        }
    if gate == "context_audit":
        return {
            "dense_peer_requests": 43_092,
            "seven_agent_requests": 57_456,
            "failed_preflights": 0,
            "truncation_incidents": 0,
            "minimum_context_margin_tokens": 1_287,
        }
    if gate == "smoke_runs":
        return {
            "long_32b_cells": 15,
            "selective_long_cells": 20,
            "standard_canary_cells": 6,
            "schema5_complete_cells": 41,
            "context_incidents": 0,
            "protocol_incidents": 0,
            "truncation_incidents": 0,
            "provenance_failures": 0,
        }
    if gate == "email_test":
        return {
            "recipient": control_state["alert_email"],
            "delivery_succeeded": True,
            "returncode": 0,
        }
    raise AssertionError(gate)


def attest_all_non_scheduler(state_dir: Path) -> None:
    current = control.load_control(state_dir)
    immutable_hash = current["immutable_sha256"]
    seven_run = next(
        run
        for run in current["immutable"]["runs"]
        if run["run_id"] == "full_sweep_agent_count_7_schema5_v1"
    )

    def source_reference(name: str, payload: dict) -> dict:
        source = state_dir / "readiness" / "sources" / f"{name}.json"
        _write_json_artifact(source, payload)
        return {"name": name, "path": str(source.resolve()), "sha256": _sha(source)}

    for index, gate in enumerate(control.REQUIRED_GATES[:-1]):
        evidence = state_dir / "readiness" / f"{gate}.json"
        evidence.parent.mkdir(exist_ok=True)
        artifacts = []
        for name in control.READINESS_ARTIFACT_NAMES[gate]:
            if gate == "snapshot":
                path = _snapshot_envelope(state_dir, name)
            elif gate == "fleet" and name == "fleet_contract":
                path = Path(current["immutable"]["fleet_contract_path"])
            else:
                path = state_dir / "readiness" / "artifacts" / f"{name}.json"
                identity = {
                    "schema_version": 1,
                    "kind": name,
                    "passed": True,
                    "immutable_sha256": immutable_hash,
                }
                if gate == "migrations":
                    if name == "response_incident_archive_report":
                        artifact_metrics = {
                            "protocol_incidents_total": 22,
                            "protocol_already_reset": 22,
                            "sealed_incident_qids": 1_064,
                        }
                        source_name = "consolidated_response_report"
                    elif name == "checkpoint_migration_report":
                        artifact_metrics = {
                            "migrated_checkpoints": 47,
                            "remaining_schema1_checkpoints": 0,
                            "coordinates_preserved": True,
                        }
                        source_name = "consolidated_checkpoint_report"
                    else:
                        artifact_metrics = {
                            "permanent_ledgers_archived": 3,
                            "unresolved_permanent_ledgers": 0,
                        }
                        source_name = "consolidated_permanent_report"
                    raw = {"schema_version": 1, "passed": True, **artifact_metrics}
                    payload = {
                        **identity,
                        "metrics": artifact_metrics,
                        "referenced_artifacts": [source_reference(source_name, raw)],
                    }
                elif gate == "semantic_audit":
                    artifact_metrics = _gate_metrics(current, gate)
                    raw = {
                        "schema_version": 1,
                        "kind": "legacy_semantic_audit",
                        "passed": True,
                        "manifest_cells": 22_680,
                        "invalid_rows": 0,
                        "metrics": artifact_metrics,
                    }
                    payload = {
                        **identity,
                        "metrics": artifact_metrics,
                        "referenced_artifacts": [
                            source_reference("consolidated_semantic_report", raw)
                        ],
                    }
                elif gate == "fleet":
                    fleet_contract = json.loads(
                        Path(current["immutable"]["fleet_contract_path"]).read_text()
                    )
                    replica_rows = []
                    registry_references = []
                    replica_number = 0
                    for profile in fleet_contract["profiles"]:
                        for replica in profile["replicas"]:
                            replica_id = replica["replica_id"]
                            host = f"node-{replica_number:02d}"
                            port = 20_000 + replica_number
                            job_id = str(9_000 + replica_number)
                            registry_path = (
                                state_dir
                                / "readiness"
                                / "registries"
                                / f"{replica_id}.json"
                            )
                            registry = {
                                "replica_id": replica_id,
                                "replica_index": replica["replica_index"],
                                "serving_profile": profile["serving_profile"],
                                "slurm_job_id": job_id,
                                "host": host,
                                "port": port,
                                "release_id": current["immutable"]["release_id"],
                                "environment_hash": current["immutable"][
                                    "serving_environment_sha256"
                                ],
                                "model_contract_sha256": current["immutable"][
                                    "model_contract_sha256"
                                ],
                                "fleet_contract_sha256": current["immutable"][
                                    "fleet_contract_sha256"
                                ],
                            }
                            _write_json_artifact(registry_path, registry)
                            registry_reference = {
                                "name": f"registry_{replica_id}",
                                "path": str(registry_path.resolve()),
                                "sha256": _sha(registry_path),
                            }
                            registry_references.append(registry_reference)
                            replica_rows.append(
                                {
                                    "replica_id": replica_id,
                                    "serving_profile": profile["serving_profile"],
                                    "slurm_job_id": job_id,
                                    "partition": replica["partition"],
                                    "node": host,
                                    "host": host,
                                    "port": port,
                                    "spooled_provenance": {
                                        "run_root": current["immutable"][
                                            "server_pool_root"
                                        ],
                                        "server_pool_id": "schema5-v1",
                                        "replica_id": replica_id,
                                        "replica_index": replica["replica_index"],
                                        "release_id": current["immutable"][
                                            "release_id"
                                        ],
                                        "environment_hash": current["immutable"][
                                            "serving_environment_sha256"
                                        ],
                                        "model_revision": profile["model_revision"],
                                        "tokenizer_id": profile["tokenizer_id"],
                                        "tokenizer_revision": profile[
                                            "tokenizer_revision"
                                        ],
                                        "model_contract_sha256": current["immutable"][
                                            "model_contract_sha256"
                                        ],
                                        "fleet_contract_sha256": current["immutable"][
                                            "fleet_contract_sha256"
                                        ],
                                    },
                                    "http": {
                                        "health_status": 200,
                                        "models_status": 200,
                                        "model_ids": [profile["serving_profile"]],
                                        "expected_model": profile["serving_profile"],
                                        "probe_started_timestamp": 22.0,
                                        "probe_completed_timestamp": 23.0,
                                        "healthy": True,
                                    },
                                    "registry_path": str(registry_path.resolve()),
                                    "registry_sha256": _sha(registry_path),
                                }
                            )
                            replica_number += 1
                    raw_fleet = {
                        "schema_version": 1,
                        "kind": "raw_fleet_health_probe",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "server_pool_root": current["immutable"]["server_pool_root"],
                        "captured_timestamp": 23.0,
                        "replicas": replica_rows,
                        "referenced_artifacts": registry_references,
                    }
                    payload = {
                        **identity,
                        "server_pool_root": current["immutable"]["server_pool_root"],
                        "metrics": _gate_metrics(current, gate),
                        "referenced_artifacts": [
                            source_reference("raw_fleet_health_probe", raw_fleet)
                        ],
                    }
                elif gate == "context_audit":
                    dense = name == "dense_peer_context_audit"
                    filters = (
                        {
                            "n_agents": [7],
                            "reasoning": ["b2048", "b8192", "unlimited"],
                            "prompt_levels": [3],
                            "topologies": ["decentralized"],
                            "context_levels": ["plus_cot"],
                        }
                        if dense
                        else {
                            "n_agents": [7],
                            "reasoning": ["unlimited"],
                            "prompt_levels": [],
                            "topologies": [],
                            "context_levels": ["plus_cot"],
                        }
                    )
                    selected_cells = 216 if dense else 288
                    audited_requests = 43_092 if dense else 57_456
                    artifact_metrics = {
                        "selected_cells": selected_cells,
                        "audited_requests": audited_requests,
                        "failed_requests": 0,
                        "failed_cells": 0,
                        "minimum_context_headroom_tokens": 1_287,
                        "truncation_incidents": 0,
                    }
                    raw = {
                        "schema_version": 3,
                        "audit": "all_routed_profiles_context_capacity",
                        "run_id": seven_run["run_id"],
                        "manifest": {
                            "path": seven_run["manifest_path"],
                            "sha256": seven_run["manifest_sha256"],
                            "cells": seven_run["cell_count"],
                        },
                        "filters": filters,
                        "all_routed_profiles": True,
                        "summary": {
                            "passed": True,
                            **{
                                key: artifact_metrics[key]
                                for key in (
                                    "selected_cells",
                                    "audited_requests",
                                    "failed_requests",
                                    "failed_cells",
                                    "minimum_context_headroom_tokens",
                                )
                            },
                        },
                        "failure_groups": [],
                        "failure_examples": [],
                    }
                    source_name = (
                        "raw_dense_peer_context_audit"
                        if dense
                        else "raw_seven_agent_context_audit"
                    )
                    payload = {
                        **identity,
                        "run_id": seven_run["run_id"],
                        "filters": filters,
                        "all_routed_profiles": True,
                        "metrics": artifact_metrics,
                        "referenced_artifacts": [source_reference(source_name, raw)],
                    }
                elif gate == "smoke_runs":
                    smoke_contract = {
                        "long_32b_smoke": ("schema5_smoke_32b_long_v1", 15),
                        "selective_long_smoke": (
                            "schema5_smoke_selective_long_v1",
                            20,
                        ),
                        "standard_canary_smoke": (
                            "schema5_smoke_standard_canaries_v1",
                            6,
                        ),
                    }
                    smoke_profiles = {
                        "long_32b_smoke": {"32B-long": 15},
                        "selective_long_smoke": {
                            "0.6B": 1,
                            "0.6B-long": 3,
                            "1.7B": 1,
                            "1.7B-long": 3,
                            "4B": 1,
                            "4B-long": 3,
                            "8B": 1,
                            "8B-long": 3,
                            "14B": 1,
                            "14B-long": 3,
                        },
                        "standard_canary_smoke": {
                            "0.6B": 1,
                            "1.7B": 1,
                            "4B": 1,
                            "8B": 1,
                            "14B": 1,
                            "32B": 1,
                        },
                    }
                    run_id, expected_cells = smoke_contract[name]
                    zero_fields = {
                        "context_incidents": 0,
                        "protocol_incidents": 0,
                        "truncation_incidents": 0,
                        "provenance_failures": 0,
                        "semantic_validation_failures": 0,
                        "top_level_length_censored_qids": 0,
                        "top_level_protocol_censored_qids": 0,
                        "auxiliary_length_censored_draws": 0,
                        "auxiliary_protocol_censored_draws": 0,
                    }
                    payload = {
                        **identity,
                        "run_id": run_id,
                        "expected_cells": expected_cells,
                        "schema5_complete_cells": expected_cells,
                        **zero_fields,
                        "concurrency": 1,
                        "resumable": True,
                        "execution_halted": False,
                        "execution_halt_reason": None,
                        "provenance": {
                            "release_id": current["immutable"]["release_id"],
                            "git_commit": current["immutable"]["git_commit"],
                            "source_tree_sha256": current["immutable"][
                                "source_tree_sha256"
                            ],
                            "harness_environment_sha256": current["immutable"][
                                "harness_environment_sha256"
                            ],
                            "serving_environment_sha256": current["immutable"][
                                "serving_environment_sha256"
                            ],
                            "model_contract_sha256": current["immutable"][
                                "model_contract_sha256"
                            ],
                            "fleet_contract_sha256": current["immutable"][
                                "fleet_contract_sha256"
                            ],
                            "server_pool_root": current["immutable"][
                                "server_pool_root"
                            ],
                            "rollout_generation": 1,
                        },
                        "suite_identity": {
                            "run_id": run_id,
                            "cell_count": expected_cells,
                            "manifest_sha256": "a" * 64,
                            "benchmark_contracts_sha256": "b" * 64,
                            "artifact_policy_sha256": "c" * 64,
                            "lineage_sha256": "d" * 64,
                            "serving_profile_counts": smoke_profiles[name],
                            "estimand_excluded": True,
                        },
                        "cells": [
                            {
                                "cell_id": f"{run_id}-{cell_index}",
                                "status": "complete",
                                "valid_qids": 1,
                                "expected_qids": 1,
                                "context_incidents": 0,
                                "protocol_incidents": 0,
                                "truncation_incidents": 0,
                                "provenance_failures": 0,
                                "top_level_length_censored_qids": 0,
                                "top_level_protocol_censored_qids": 0,
                                "auxiliary_length_censored_draws": 0,
                                "auxiliary_protocol_censored_draws": 0,
                                "provenance_errors": [],
                                "semantic_errors": [],
                                "failure": None,
                            }
                            for cell_index in range(expected_cells)
                        ],
                    }
                elif gate == "email_test":
                    receipt_metrics = _gate_metrics(current, gate)
                    raw_receipt = {
                        "schema_version": 1,
                        "kind": "mail_submission_receipt",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "metrics": receipt_metrics,
                        "submitted_timestamp": 23.0,
                        "subject": "[agents-scaling] schema-5 readiness delivery test",
                        "message_sha256": "e" * 64,
                        "stdout": "",
                        "stderr": "",
                        "referenced_artifacts": [],
                    }
                    payload = {
                        **identity,
                        "metrics": receipt_metrics,
                        "referenced_artifacts": [
                            source_reference("mail_submission_receipt", raw_receipt)
                        ],
                    }
                else:
                    raise AssertionError((gate, name))
                _write_json_artifact(path, payload)
            artifacts.append(
                {"name": name, "path": str(path.resolve()), "sha256": _sha(path)}
            )
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": control.READINESS_EVIDENCE_SCHEMA_VERSION,
                    "gate": gate,
                    "passed": True,
                    "immutable_sha256": immutable_hash,
                    "metrics": _gate_metrics(current, gate),
                    "artifacts": artifacts,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        control.attest_gate(
            state_dir, gate=gate, evidence_path=evidence, now=20 + index
        )


def reconcile_clean(state_dir: Path, *, now: float = 40.0) -> dict:
    return control.reconcile_control(
        state_dir,
        snapshot=control.SchedulerSnapshot((), now),
        all_jobs=True,
        no_admit=True,
        now=now,
    )


def make_ready(state_dir: Path) -> None:
    attest_all_non_scheduler(state_dir)
    assert reconcile_clean(state_dir)["passed"] is True


def complete_drill_marker(state_dir: Path, *, now: float = 45.0) -> dict:
    """Install a fully bound synthetic drill proof for tests unrelated to live drilling."""

    current = control.load_control(state_dir)
    reconciliation_path, reconciliation_sha = control._publish_drill_reconciliation(
        state_dir, "test-drill"
    )
    state = {
        "schema_version": 1,
        "protocol": "schema5-controller-drill-v1",
        "drill_id": "test-drill",
        "shared_fencing_primitive": control.SHARED_CONTROLLER_FENCING_PRIMITIVE,
        "shared_controller_primitives": control.shared_controller_primitive_contract(),
        "phase": "running",
        "immutable_sha256": current["immutable_sha256"],
        "created_at": control.utc_timestamp(now - 2),
        "created_timestamp": now - 2,
        "updated_at": control.utc_timestamp(now),
        "updated_timestamp": now,
        "baseline": control.controller_drill_baseline(state_dir, current),
        "roles": {},
        "transition_history": [],
        "final_reconciliation_path": None,
        "final_reconciliation_sha256": None,
        "completed_at": control.utc_timestamp(now),
        "completed_timestamp": now,
    }
    for index, role in enumerate(control.ROLE_NAMES):
        successor_generation = 2
        successor_intent = f"synthetic-successor-{role}"
        successor_token = control.drill_job_token(
            state["drill_id"], role, successor_generation, successor_intent
        )
        successor_sbatch = f"/synthetic/{role}-successor.sbatch"
        recovery = {
            "passed": True,
            "killed_job_id": str(600 + 2 * index),
            "successor_job_id": str(601 + 2 * index),
            "successor_job_token": successor_token,
            "successor_sbatch_path": successor_sbatch,
            "successor_dependency_job_id": str(600 + 2 * index),
            "successor_generation": successor_generation,
            "successor_intent_token": successor_intent,
            "recovered_at": control.utc_timestamp(now - 1),
            "recovered_timestamp": now - 1,
            "recovery_seconds": 1.0,
            "maximum_seconds": 900.0,
            "fencing_primitive": control.SHARED_CONTROLLER_FENCING_PRIMITIVE,
            "controller_primitives": control.shared_controller_primitive_contract(),
        }
        state["roles"][role] = {
            "next_generation": 3,
            "active": None,
            "successor": None,
            "submission_intent": None,
            "heartbeat": None,
            "last_exit": None,
            "kill": {
                "state": "cancelled",
                "job_id": recovery["killed_job_id"],
                "job_token": f"synthetic-kill-{role}",
                "sbatch_path": f"/synthetic/{role}.sbatch",
                "fencing_primitive": control.SHARED_CONTROLLER_FENCING_PRIMITIVE,
                "controller_primitives": control.shared_controller_primitive_contract(),
                "successor_job_id": recovery["successor_job_id"],
                "successor_job_token": successor_token,
                "successor_sbatch_path": successor_sbatch,
                "successor_dependency_job_id": recovery[
                    "successor_dependency_job_id"
                ],
                "started_at": control.utc_timestamp(now - 1.5),
                "started_timestamp": now - 1.5,
                "returncode": 0,
            },
            "recovery": recovery,
        }
    control._append_drill_event(
        state_dir,
        state,
        event="started_paused",
        details={"desired_state": "paused"},
        now=now - 3,
    )
    for index, role in enumerate(control.ROLE_NAMES):
        recovery = state["roles"][role]["recovery"]
        killed_job_id = recovery["killed_job_id"]
        control._append_drill_event(
            state_dir,
            state,
            event="exact_kill_intent",
            details={"role": role, "job_id": killed_job_id},
            now=now - 2.5 + index * 0.2,
        )
        control._append_drill_event(
            state_dir,
            state,
            event="exact_kill_result",
            details={"role": role, "job_id": killed_job_id, "returncode": 0},
            now=now - 2.4 + index * 0.2,
        )
        control._append_drill_event(
            state_dir,
            state,
            event="recovery_verified",
            details={"role": role, **recovery},
            now=now - 2.3 + index * 0.2,
        )
    state["phase"] = "stopping"
    control._append_drill_event(
        state_dir,
        state,
        event="stopping",
        details={"reason": "both_recoveries_passed"},
        now=now - 1,
    )
    state["phase"] = "completed"
    state["final_reconciliation_path"] = str(reconciliation_path)
    state["final_reconciliation_sha256"] = reconciliation_sha
    control._append_drill_event(
        state_dir,
        state,
        event="completed",
        details={
            "final_reconciliation_path": str(reconciliation_path),
            "final_reconciliation_sha256": reconciliation_sha,
        },
        now=now,
    )
    control._save_drill_state(state_dir, state, now=now)
    marker = {
        "schema_version": 1,
        "protocol": "schema5-controller-kill-drill-v1",
        "passed": True,
        "drill_id": state["drill_id"],
        "shared_fencing_primitive": state["shared_fencing_primitive"],
        "shared_controller_primitives": state["shared_controller_primitives"],
        "immutable_sha256": state["immutable_sha256"],
        "completed_at": state["completed_at"],
        "completed_timestamp": state["completed_timestamp"],
        "baseline": state["baseline"],
        "roles": {
            role: copy.deepcopy(state["roles"][role]["recovery"])
            for role in control.ROLE_NAMES
        },
        "final_reconciliation_path": str(reconciliation_path),
        "final_reconciliation_sha256": reconciliation_sha,
        "drill_state_path": str((state_dir / control.DRILL_STATE_FILENAME).resolve()),
        "drill_state_sha256": _sha(state_dir / control.DRILL_STATE_FILENAME),
    }
    control._atomic_write_json(state_dir / control.DRILL_COMPLETE_FILENAME, marker)
    return marker


def resume_ready(state_dir: Path, *, now: float = 50.0) -> tuple[dict, tuple]:
    if not (state_dir / control.DRILL_COMPLETE_FILENAME).exists():
        complete_drill_marker(state_dir, now=now - 1)
    jobs = []
    identifiers = iter(("700", "701", "702", "703", "704", "705"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(jobs), now)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    resumed = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=submit,
        now=now,
    )
    return resumed, tuple(jobs)


def start_live_drill(state_dir: Path, *, now: float = 50.0):
    jobs = []
    identifiers = iter(str(value) for value in range(800, 900))

    def snapshot(at: float = now):
        return control.SchedulerSnapshot(tuple(jobs), at)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        sbatch = argv[-1]
        dependency = next(
            (
                arg.split("=", 1)[1]
                for arg in argv
                if arg.startswith("--dependency=")
            ),
            "",
        )
        jobs.append(
            control.SchedulerJob(
                job_id,
                "drill",
                "PENDING",
                token,
                f"sbatch {sbatch}",
                dependency=dependency,
            )
        )
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    state = control.start_controller_drill(
        state_dir,
        scheduler=snapshot(),
        submit_runner=submit,
        now=now,
    )
    for index, role in enumerate(control.ROLE_NAMES):
        active = state["roles"][role]["active"]
        control.claim_drill_controller(
            state_dir,
            drill_id=state["drill_id"],
            role=role,
            generation=active["generation"],
            intent_token=active["intent_token"],
            job_id=active["job_id"],
            active_scheduler_job_ids=snapshot().active_job_ids,
            now=now + 1 + index,
        )
        control.submit_drill_intent(
            state_dir,
            role=role,
            target="successor",
            dependency_job_id=active["job_id"],
            scheduler=snapshot(),
            submit_runner=submit,
            now=now + 1 + index,
        )
        control.heartbeat_drill_controller(
            state_dir,
            drill_id=state["drill_id"],
            role=role,
            generation=active["generation"],
            intent_token=active["intent_token"],
            job_id=active["job_id"],
            now=now + 2 + index,
        )
        state = control.load_drill_state(state_dir)
    return jobs, snapshot, submit


def recover_drill_role(
    state_dir: Path,
    jobs: list,
    snapshot,
    submit,
    *,
    role: str,
    now: float,
):
    cancelled = []
    killed = control.kill_drill_controller(
        state_dir,
        role=role,
        snapshot=snapshot(now),
        cancel_runner=lambda argv: (
            cancelled.append(list(argv)) or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=now,
    )
    old = control.load_drill_state(state_dir)["roles"][role]["active"]
    jobs[:] = [job for job in jobs if job.job_id != killed["job_id"]]
    control._record_drill_exit(
        state_dir,
        drill_id=control.load_drill_state(state_dir)["drill_id"],
        role=role,
        generation=old["generation"],
        intent_token=old["intent_token"],
        job_id=old["job_id"],
        reason="signal",
        now=now + 0.5,
    )
    successor = control.load_drill_state(state_dir)["roles"][role]["successor"]
    drill_id = control.load_drill_state(state_dir)["drill_id"]
    control.claim_drill_controller(
        state_dir,
        drill_id=drill_id,
        role=role,
        generation=successor["generation"],
        intent_token=successor["intent_token"],
        job_id=successor["job_id"],
        active_scheduler_job_ids=snapshot(now + 1).active_job_ids,
        now=now + 1,
    )
    control.submit_drill_intent(
        state_dir,
        role=role,
        target="successor",
        dependency_job_id=successor["job_id"],
        scheduler=snapshot(now + 1),
        submit_runner=submit,
        now=now + 1,
    )
    control.heartbeat_drill_controller(
        state_dir,
        drill_id=drill_id,
        role=role,
        generation=successor["generation"],
        intent_token=successor["intent_token"],
        job_id=successor["job_id"],
        now=now + 2,
    )
    recovery = control.record_drill_recovery(
        state_dir, role=role, snapshot=snapshot(now + 2), now=now + 2
    )
    assert cancelled == [["scancel", killed["job_id"]]]
    return recovery


def test_init_is_paused_idempotent_and_freezes_pins(tmp_path):
    state_dir, first = initialize(tmp_path)
    second = control.initialize_control(state_dir, pins=first["immutable"], now=999.0)
    assert second["created_timestamp"] == 10.0
    assert second["desired_state"] == "paused"
    assert second["rollout_generation"] == 0
    assert (state_dir / control.IMMUTABLE_PINS_FILENAME).stat().st_mode & 0o222 == 0
    assert first["admission"] == control.DEFAULT_ADMISSION

    changed = copy.deepcopy(first["immutable"])
    changed["git_commit"] = "c" * 40
    with pytest.raises(control.ImmutablePinError, match="idempotent|completion marker"):
        control.initialize_control(state_dir, pins=changed)


def test_load_rejects_control_pin_or_history_tampering(tmp_path):
    state_dir, _ = initialize(tmp_path)
    path = state_dir / control.CONTROL_FILENAME
    payload = json.loads(path.read_text())
    payload["immutable"]["git_commit"] = "d" * 40
    payload["immutable_sha256"] = control.sha256_value(payload["immutable"])
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(control.ImmutablePinError, match="frozen pin copy"):
        control.load_control(state_dir)

    # Restore the original and independently prove the history hash-chain check.
    payload["immutable"] = json.loads(
        (state_dir / control.IMMUTABLE_PINS_FILENAME).read_text()
    )
    payload["immutable_sha256"] = control.sha256_value(payload["immutable"])
    payload["transition_history"][0]["event"] = "forged"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(control.ControlError, match="record hash"):
        control.load_control(state_dir)


def test_resume_fails_closed_then_increments_rollout_once(tmp_path):
    state_dir, _ = initialize(tmp_path)
    with pytest.raises(control.ReadinessError, match="snapshot"):
        control.resume_control(state_dir)
    make_ready(state_dir)
    resumed, _ = resume_ready(state_dir, now=50.0)
    assert resumed["desired_state"] == "running"
    assert resumed["rollout_generation"] == 1
    again = control.resume_control(
        state_dir,
        scheduler_reader=lambda: control.SchedulerSnapshot(
            tuple(
                control.SchedulerJob(
                    record["job_id"], role, "RUNNING", record["job_token"]
                )
                for role in control.ROLE_NAMES
                for record in (resumed["controllers"][role]["active"],)
            ),
            60.0,
        ),
        now=60.0,
    )
    assert again["rollout_generation"] == 1


def test_resume_waits_for_both_scheduler_visible_jobs_without_duplicate_submission(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    visible_jobs = []
    submitted = []
    identifiers = iter(("810", "811"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(visible_jobs), 50.0)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        submitted.append((job_id, token, list(argv)))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    with pytest.raises(control.SchedulerVisibilityPending):
        control.resume_control(
            state_dir,
            scheduler_reader=scheduler,
            submit_runner=submit,
            now=50.0,
        )
    resuming = control.load_control(state_dir)
    assert resuming["desired_state"] == "resuming"
    assert resuming["resume_intent"]["state"] == "submitting_controllers"
    assert resuming["resume_intent"]["controller_job_ids"] == {
        "dispatcher": "810",
        "fleet_supervisor": "811",
    }
    assert len(submitted) == 2
    with pytest.raises(control.ControlError, match="disabled by desired state"):
        control.production_environment_from_state(state_dir)

    # Visibility can lag beyond both seven-minute metadata leases.  Expiry is not
    # corruption: preserve the authenticated generation bindings but make both leases
    # historically expired, then require the retry to rescan and advance each sequence.
    lease_sequences = {}
    for name, module in (
        (control.RUNTIME_ATTESTATION_STATE_KEY, runtime_integrity),
        (control.SNAPSHOT_ATTESTATION_STATE_KEY, snapshot_integrity),
    ):
        lease_path = Path(resuming[name]["lease_path"])
        lease = json.loads(lease_path.read_text(encoding="utf-8"))
        lease_sequences[name] = int(lease["sequence"])
        lease["verified_timestamp"] = 1.0
        lease["expires_timestamp"] = 421.0
        candidate = dict(lease)
        candidate.pop("lease_id")
        lease["lease_id"] = module.sha256_bytes(module.canonical_bytes(candidate))
        lease_path.chmod(0o644)
        lease_path.write_text(json.dumps(lease) + "\n", encoding="utf-8")
        lease_path.chmod(0o444)

    with pytest.raises(control.SchedulerVisibilityPending):
        control.resume_control(
            state_dir,
            scheduler_reader=scheduler,
            submit_runner=lambda _argv: pytest.fail(
                "visibility-grace retry duplicated a controller"
            ),
            now=51.0,
        )
    assert len(submitted) == 2

    visible_jobs.extend(
        control.SchedulerJob(job_id, "controller", "PENDING", token)
        for job_id, token, _ in submitted
    )
    # A durable resume transaction may outlive the fleet evidence's 10-minute launch
    # freshness window while Slurm visibility converges.  The paused->resuming write
    # already proved freshness, so this restart validates the sealed historical gate.
    running = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=lambda _argv: pytest.fail("visible jobs must be adopted"),
        now=1_000.0,
    )
    assert running["desired_state"] == "running"
    assert running["resume_intent"]["state"] == "complete"
    assert running["resume_intent"]["scheduler_visible_job_ids"] == {
        "dispatcher": "810",
        "fleet_supervisor": "811",
    }
    refreshed = control.load_control(state_dir)
    for name in (
        control.RUNTIME_ATTESTATION_STATE_KEY,
        control.SNAPSHOT_ATTESTATION_STATE_KEY,
    ):
        lease = json.loads(
            Path(refreshed[name]["lease_path"]).read_text(encoding="utf-8")
        )
        assert lease["sequence"] == lease_sequences[name] + 1
        assert lease["expires_timestamp"] > time.time()
    assert [transition["event"] for transition in running["transition_history"]].count(
        "resumed"
    ) == 1


def test_resume_rejects_incomplete_scheduler_truth_before_any_submission(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    calls = []
    incomplete = control.SchedulerSnapshot(
        (),
        50.0,
        squeue_ok=False,
        sacct_ok=True,
        errors=("squeue unavailable",),
    )
    with pytest.raises(control.SchedulerAmbiguity, match=r"complete squeue\+sacct"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: incomplete,
            submit_runner=lambda argv: calls.append(list(argv)),
            now=50.0,
        )
    assert calls == []
    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["rollout_generation"] == 0
    assert current["resume_intent"] is None
    assert all(
        current["controllers"][role]["active"] is None for role in control.ROLE_NAMES
    )


def test_attestation_drift_blocks_resume(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    snapshot_evidence = Path(
        control.load_control(state_dir)["readiness"]["snapshot"]["evidence"]
    )
    snapshot_evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(control.ReadinessError, match="drifted"):
        control.resume_control(state_dir)


def test_typed_artifact_metrics_cannot_disagree_with_outer_envelope(tmp_path):
    state_dir, _ = initialize(tmp_path)
    attest_all_non_scheduler(state_dir)
    evidence_path = state_dir / "readiness" / "migrations.json"
    evidence = json.loads(evidence_path.read_text())
    artifact_row = next(
        row
        for row in evidence["artifacts"]
        if row["name"] == "response_incident_archive_report"
    )
    wrapper_path = Path(artifact_row["path"])
    wrapper = json.loads(wrapper_path.read_text())
    source_row = wrapper["referenced_artifacts"][0]
    source_path = Path(source_row["path"])
    source = json.loads(source_path.read_text())
    source["protocol_already_reset"] = 21
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    source_row["sha256"] = _sha(source_path)
    wrapper_path.write_text(json.dumps(wrapper) + "\n", encoding="utf-8")
    artifact_row["sha256"] = _sha(wrapper_path)
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    with pytest.raises(
        control.ReadinessError, match="does not match its typed wrapper"
    ):
        control.attest_gate(
            state_dir,
            gate="migrations",
            evidence_path=evidence_path,
            expected_sha256=_sha(evidence_path),
            now=30.0,
        )


def test_recursive_artifact_hash_drift_invalidates_attested_gate(tmp_path):
    state_dir, _ = initialize(tmp_path)
    attest_all_non_scheduler(state_dir)
    evidence_path = state_dir / "readiness" / "semantic_audit.json"
    evidence = json.loads(evidence_path.read_text())
    wrapper_row = evidence["artifacts"][0]
    wrapper = json.loads(Path(wrapper_row["path"]).read_text())
    source_path = Path(wrapper["referenced_artifacts"][0]["path"])
    source = json.loads(source_path.read_text())
    source["invalid_rows"] = 1
    source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    current = control.load_control(state_dir)
    with pytest.raises(control.ReadinessError, match="artifact drifted"):
        control.validate_readiness(
            current, state_dir=state_dir, verify_files=True, now=30.0
        )


def test_recursive_artifact_rejects_symlink_before_path_resolution(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "artifact.json"
    link.symlink_to(target)
    with pytest.raises(control.ReadinessError, match="symlink"):
        control._validate_referenced_artifact(
            {"name": "artifact", "path": str(link), "sha256": _sha(target)},
            context="test artifact",
            verified=set(),
            active=set(),
        )


@pytest.mark.parametrize("target_kind", ("payload", "control"))
def test_controller_snapshot_validation_rejects_hardlinked_regular_files(
    tmp_path, target_kind
):
    envelope = _snapshot_envelope(tmp_path, f"hardlink-{target_kind}")
    payload = json.loads(envelope.read_text())
    root = Path(payload["snapshot_root"])
    target = (
        root / "payload.bin"
        if target_kind == "payload"
        else root / "SNAPSHOT_INVENTORY.sha256"
    )
    os.link(target, tmp_path / f"outside-{target_kind}")

    with pytest.raises(control.ReadinessError, match="hardlinked"):
        control._validate_snapshot_external_attestation(
            envelope,
            payload,
            snapshot_context=control._SnapshotValidationContext(full=True),
        )


def test_snapshot_lease_path_is_bound_to_exact_control_state_generation(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    current = control.load_control(state_dir)
    drifted = copy.deepcopy(current)
    drifted[control.SNAPSHOT_ATTESTATION_STATE_KEY]["lease_path"] = str(
        (state_dir / "other" / "lease.g000001.json").resolve()
    )
    with pytest.raises(control.ReadinessError, match="escapes its state generation"):
        control.validate_snapshot_integrity_attestation(
            drifted, state_dir=state_dir, verify_lease=False
        )


def test_full_snapshot_validation_uses_fd_bound_hash_metadata_for_files(
    tmp_path, monkeypatch
):
    envelope = _snapshot_envelope(tmp_path, "fd-bound")
    payload = json.loads(envelope.read_text(encoding="utf-8"))
    root = Path(payload["snapshot_root"])
    calls = []
    original_hash = snapshot_integrity.sha256_file_with_metadata
    original_metadata = snapshot_integrity.metadata_entry

    def counted(path, *, description="sealed snapshot file"):
        calls.append(Path(path))
        return original_hash(path, description=description)

    def directories_only(path):
        candidate = Path(path)
        if candidate.is_file():
            pytest.fail(
                f"full validation recaptured file metadata after hashing: {candidate}"
            )
        return original_metadata(candidate)

    monkeypatch.setattr(snapshot_integrity, "sha256_file_with_metadata", counted)
    monkeypatch.setattr(snapshot_integrity, "metadata_entry", directories_only)
    control._validate_snapshot_external_attestation(
        envelope,
        payload,
        snapshot_context=control._SnapshotValidationContext(full=True),
    )

    assert root / "payload.bin" in calls
    assert {
        root / filename for filename in control.SNAPSHOT_CONTROL_FILENAMES
    }.issubset(set(calls))


def test_snapshot_payload_hashing_is_absent_from_every_running_hot_path(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    _running, jobs = resume_ready(state_dir, now=50.0)

    current = control.load_control(state_dir)
    seal = control.validate_snapshot_integrity_attestation(
        current, state_dir=state_dir, verify_lease=True
    )
    roots = [Path(row["snapshot_root"]) for row in seal["snapshots"]]
    internal_reads: list[Path] = []

    def record_if_internal(path) -> None:
        candidate = Path(path).absolute()
        if any(candidate == root or root in candidate.parents for root in roots):
            internal_reads.append(candidate)

    original_hash = control._snapshot_regular_sha256
    original_read_regular = snapshot_integrity.read_regular_bytes
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def counted(path, *, context):
        record_if_internal(path)
        return original_hash(path, context=context)

    def counted_regular(path, *, description):
        record_if_internal(path)
        return original_read_regular(path, description=description)

    def counted_bytes(path):
        record_if_internal(path)
        return original_read_bytes(path)

    def counted_text(path, *args, **kwargs):
        record_if_internal(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(control, "_snapshot_regular_sha256", counted)
    monkeypatch.setattr(snapshot_integrity, "read_regular_bytes", counted_regular)
    monkeypatch.setattr(Path, "read_bytes", counted_bytes)
    monkeypatch.setattr(Path, "read_text", counted_text)
    control.validate_readiness(current, state_dir=state_dir, verify_files=True)
    control.production_environment_from_state(state_dir)
    control.production_cell_execution_from_state(state_dir)
    contract = control.admission_contract_from_state(state_dir)
    assert contract["rollout_generation"] == 1
    control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=control.SchedulerSnapshot(jobs, time.time()),
        submit_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "799\n", ""),
    )

    assert internal_reads == []


def test_snapshot_metadata_drift_prevents_lease_renewal_and_expiry_fails_closed(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    current = control.load_control(state_dir)
    record = current[control.SNAPSHOT_ATTESTATION_STATE_KEY]
    lease = json.loads(Path(record["lease_path"]).read_text())
    seal = control.validate_snapshot_integrity_attestation(
        current, state_dir=state_dir, verify_lease=True
    )

    payload = Path(seal["snapshots"][0]["snapshot_root"]) / "payload.bin"
    payload.chmod(0o644)
    payload.write_bytes(payload.read_bytes() + b"drift")
    payload.chmod(0o444)
    with pytest.raises(
        snapshot_integrity.SnapshotIntegrityError, match="metadata drifted"
    ):
        snapshot_integrity.refresh_generation_lease(
            state_dir=state_dir,
            seal_path=Path(record["path"]),
            seal_sha256=record["sha256"],
            generation=1,
            immutable_pins_sha256=current["immutable_sha256"],
            now=float(lease["verified_timestamp"]) + 301.0,
            force=True,
        )
    with pytest.raises(
        snapshot_integrity.SnapshotIntegrityError, match="lease expired"
    ):
        snapshot_integrity.verify_generation_lease(
            lease_path=Path(record["lease_path"]),
            seal_path=Path(record["path"]),
            seal_sha256=record["sha256"],
            generation=1,
            immutable_pins_sha256=current["immutable_sha256"],
            now=float(lease["expires_timestamp"]),
        )


def test_immutable_topology_is_closed_and_commands_have_no_override_surface(tmp_path):
    pins = make_pins(tmp_path)
    control.validate_immutable_pins(pins, verify_files=True)
    assert set(control.REQUIRED_RUNS) == {
        "full_sweep_schema5_v1",
        "full_sweep_agent_counts_schema5_v1",
        "full_sweep_agent_count_7_schema5_v1",
    }
    assert sum(control.REQUIRED_RUNS.values()) == 22_680
    assert sum(control.REQUIRED_RUN_QIDS.values()) == 4_524_660
    assert pins["dispatcher_command"] == control.expected_dispatcher_command(pins)
    assert pins[
        "fleet_supervisor_command"
    ] == control.expected_fleet_supervisor_command(pins)
    fleet_command = pins["fleet_supervisor_command"]
    release_argument = fleet_command.index("--release-worktree")
    assert fleet_command[release_argument + 1] == pins["release_worktree"]

    extra = copy.deepcopy(pins)
    extra["operator_override"] = True
    with pytest.raises(control.ImmutablePinError, match="closed schema"):
        control.validate_immutable_pins(extra, verify_files=False)

    wrong_weight = copy.deepcopy(pins)
    wrong_weight["runs"][0]["weight"] = 2.0
    with pytest.raises(control.ImmutablePinError, match="weight"):
        control.validate_immutable_pins(wrong_weight, verify_files=False)

    wrong_command = copy.deepcopy(pins)
    wrong_command["dispatcher_command"].extend(["--operator-ceiling", "400"])
    with pytest.raises(control.ImmutablePinError, match="exact schema-5 topology"):
        control.validate_immutable_pins(wrong_command, verify_files=False)

    wrong_qids = copy.deepcopy(pins)
    wrong_qids["runs"][0]["expected_qids"] -= 1
    with pytest.raises(control.ImmutablePinError, match="benchmark QIDs"):
        control.validate_immutable_pins(wrong_qids, verify_files=False)


def test_prepare_pins_derives_exact_authority_and_initializes_idempotently(
    tmp_path, capsys
):
    expected = make_preparable_pins(tmp_path)
    state_dir = Path(expected["results_root"]) / control.CONTROL_STATE_DIRNAME
    output = Path(expected["results_root"]) / "schema5-v1.immutable-pins.json"
    canonical_pool = Path(expected["server_pool_root"])
    canonical_pool.rmdir()

    assert (
        control.main(
            [
                "--state-dir",
                str(state_dir),
                "prepare-pins",
                "--release-bundle-root",
                expected["release_bundle_root"],
                "--hf-home",
                expected["hf_home"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert canonical_pool.is_dir()
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared == expected
    assert output.stat().st_mode & 0o222 == 0
    checksum = output.with_name(output.name + ".sha256")
    assert checksum.read_text(encoding="utf-8") == f"{_sha(output)}  {output.name}\n"
    assert checksum.stat().st_mode & 0o222 == 0

    # Re-running the producer proves the exact same bytes instead of replacing authority.
    assert (
        control.main(
            [
                "--state-dir",
                str(state_dir),
                "prepare-pins",
                "--release-bundle-root",
                expected["release_bundle_root"],
                "--hf-home",
                expected["hf_home"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        control.main(
            [
                "--state-dir",
                str(state_dir),
                "init",
                "--pins-json",
                str(output),
            ]
        )
        == 0
    )
    initialized = control.load_control(state_dir, verify_files=True)
    assert initialized["immutable"] == prepared
    assert initialized["immutable_sha256"] == control.sha256_value(prepared)
    capsys.readouterr()


def test_prepare_pins_rejects_semantically_drifted_run_authority(tmp_path):
    expected = make_preparable_pins(tmp_path)
    run = expected["runs"][0]
    policy_path = Path(run["policy_path"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["release"]["git_commit"] = "f" * 40
    _checksummed(policy_path, (json.dumps(policy, sort_keys=True) + "\n").encode())

    with pytest.raises(control.ImmutablePinError, match="policy.*frozen release"):
        control.build_immutable_pins(
            state_dir=Path(expected["results_root"]) / control.CONTROL_STATE_DIRNAME,
            release_bundle_root=Path(expected["release_bundle_root"]),
            hf_home=Path(expected["hf_home"]),
        )


def test_publish_pins_refuses_to_replace_different_authority(tmp_path):
    expected = make_preparable_pins(tmp_path)
    pins = control.build_immutable_pins(
        state_dir=Path(expected["results_root"]) / control.CONTROL_STATE_DIRNAME,
        release_bundle_root=Path(expected["release_bundle_root"]),
        hf_home=Path(expected["hf_home"]),
    )
    output = tmp_path / "pins.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(control.ImmutablePinError, match="non-identical"):
        control.publish_immutable_pins(output, pins)


def test_release_bundle_requires_live_read_only_environment_seals(tmp_path):
    pins = make_pins(tmp_path)
    harness = Path(pins["harness_environment_prefix"])
    harness.chmod(0o755)
    with pytest.raises(control.ImmutablePinError, match="not sealed read-only"):
        control.validate_immutable_pins(pins, verify_files=True)


def test_runtime_release_validation_is_git_admin_independent_but_byte_exact(tmp_path):
    pins = make_pins(tmp_path)
    release = Path(pins["release_worktree"])
    release.chmod(0o755)
    (release / ".git").write_text(
        "gitdir: /source/checkout/that/was/intentionally/removed/.git/worktrees/release\n",
        encoding="utf-8",
    )
    (release / ".git").chmod(0o444)
    release.chmod(0o555)

    # Publication-time exact-tag validation is already sealed into the release bundle;
    # production does not dereference the disposable Git administrative pointer.
    control.validate_immutable_pins(pins, verify_files=True)

    nested = release / "slurm" / "common.sh"
    nested.chmod(0o644)
    nested.write_text("# drift after source checkout removal\n", encoding="utf-8")
    nested.chmod(0o444)
    with pytest.raises(control.ImmutablePinError, match="source tree drifted"):
        control.validate_immutable_pins(pins, verify_files=True)


def test_release_bundle_rejects_bound_materialization_stage_drift(tmp_path):
    pins = make_pins(tmp_path)
    stage_path = (
        Path(pins["release_bundle_root"]).parent
        / control.MATERIALIZATION_STAGE_FILENAMES["serving_clone"]
    )
    stage_path.chmod(0o644)
    stage_path.write_text("{}\n", encoding="utf-8")
    stage_path.chmod(0o444)

    with pytest.raises(control.ImmutablePinError, match="stage artifact drifted"):
        control.validate_immutable_pins(pins, verify_files=True)


def test_scheduler_reconciliation_rejects_duplicate_or_unmappable_jobs(tmp_path):
    state_dir, initialized = initialize(tmp_path)
    token = control.job_token("dispatcher", 1, "intent1")
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob("101", "x", "RUNNING", token),
            control.SchedulerJob("102", "y", "PENDING", token),
        ),
        30.0,
    )
    report = control.build_reconciliation_report(
        initialized, snapshot, all_jobs=True, no_admit=True
    )
    assert report["passed"] is False
    assert report["ambiguous_tokens"][token] == ["101", "102"]
    assert report["unmappable_job_ids"] == ["101", "102"]


def test_scheduler_parser_and_join_prefers_live_squeue():
    submit = (
        "sbatch --comment=comment --dependency=afterany:99 /state/controller.sbatch"
    )
    outputs = {
        "squeue": subprocess.CompletedProcess(
            [], 0, "123|job|RUNNING|comment|cmd|(null)\n", ""
        ),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"123|job|FAILED|comment|{submit}\n"
            "123.batch|batch|COMPLETED||\n",
            "",
        ),
    }

    def runner(argv):
        return outputs[argv[0]]

    snapshot = control.query_scheduler(runner=runner, user="u", now=100.0)
    assert len(snapshot.jobs) == 1
    assert snapshot.jobs[0].state == "RUNNING"
    assert snapshot.jobs[0].dependency == "afterany:99"
    assert snapshot.active_job_ids == {"123"}


def test_scheduler_join_recovers_cluster_blank_sacct_comment_from_submit_line():
    token = control.job_token("dispatcher", 1, "intent1")
    submit = f"sbatch --parsable --comment={token} /state/dispatch.sbatch"
    outputs = {
        "squeue": subprocess.CompletedProcess(
            [], 0, f"123|controller|RUNNING|{token}|{submit}|(null)\n", ""
        ),
        # The production cluster has AccountingStoreFlags=(null), so Comment is blank
        # even while the scheduler-owned SubmitLine retains the exact CLI token.
        "sacct": subprocess.CompletedProcess(
            [], 0, f"123|controller|RUNNING||{submit}\n", ""
        ),
    }

    snapshot = control.query_scheduler(
        runner=lambda argv: outputs[argv[0]], user="u", now=100.0
    )
    assert snapshot.jobs[0].comment == token
    assert snapshot.jobs[0].source == "squeue"


def test_recorded_controller_scheduler_contract_is_supported():
    outputs = {
        "squeue": subprocess.CompletedProcess(
            [],
            0,
            (CLUSTER_FIXTURES / "controller_squeue_comment.txt").read_text(
                encoding="utf-8"
            ),
            "",
        ),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            (CLUSTER_FIXTURES / "controller_sacct_blank_comment.txt").read_text(
                encoding="utf-8"
            ),
            "",
        ),
    }
    snapshot = control.query_scheduler(
        runner=lambda argv: outputs[argv[0]], user="u", now=100.0
    )
    assert len(snapshot.jobs) == 1
    assert control.parse_job_token(snapshot.jobs[0].comment) == {
        "role": "dispatcher",
        "generation": "1",
        "intent": "intent1",
    }


def test_cluster_sacct_contract_derives_separate_dependency_from_submitline():
    token = control.job_token("dispatcher", 1, "intent1")
    dependency = "afterany:99"
    submit = (
        f"sbatch --parsable --comment {token} --dependency {dependency} "
        "/state/dispatch.sbatch"
    )
    commands = {}

    def runner(argv):
        commands[argv[0]] = list(argv)
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(
                argv, 0, f"123|controller|PENDING||{submit}\n", ""
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            (
                f"123|controller|PENDING|{token}|{submit}|"
                f"{dependency}(unfulfilled)\n"
            ),
            "",
        )

    snapshot = control.query_scheduler(runner=runner, user="u", now=100.0)
    assert commands["sacct"][-1] == (
        "--format=JobIDRaw,JobName,State,Comment,SubmitLine"
    )
    assert snapshot.jobs[0].comment == token
    assert snapshot.jobs[0].dependency == dependency + "(unfulfilled)"
    assert control._dependency_binds_exact_afterany(
        snapshot.jobs[0].dependency, "99"
    )


def test_sacct_reassembles_multiline_submitline_before_next_record():
    token = control.job_token("dispatcher", 1, "intent1")
    rows = control.parse_scheduler_rows(
        "123.0|bash|COMPLETED||srun bash -lc set -euo pipefail\n"
        "source /state/common.sh\n"
        "for index in 1 2; do echo $index; done\n"
        f"124|controller|PENDING||sbatch --comment={token} "
        "--dependency=afterany:99 /state/dispatch.sbatch\n",
        source="sacct",
    )
    assert len(rows) == 2
    assert rows[0].job_id == "123.0"
    assert "\nsource /state/common.sh\n" in rows[0].command
    assert rows[1].job_id == "124"
    assert rows[1].comment == token
    assert rows[1].dependency == "afterany:99"


@pytest.mark.parametrize(
    "submitline",
    [
        "sbatch --dependency",
        "sbatch --dependency --parsable /state/dispatch.sbatch",
        (
            "sbatch --dependency=afterany:1 --dependency afterany:2 "
            "/state/dispatch.sbatch"
        ),
    ],
)
def test_sacct_submitline_rejects_ambiguous_dependency(submitline):
    with pytest.raises(
        control.SchedulerAmbiguity, match="(valueless|duplicate) --dependency"
    ):
        control.parse_scheduler_rows(
            f"123|controller|PENDING||{submitline}\n", source="sacct"
        )


def test_scheduler_rejects_squeue_sacct_dependency_disagreement():
    token = control.job_token("dispatcher", 1, "intent1")
    submit = (
        f"sbatch --comment={token} --dependency=afterany:99 /state/dispatch.sbatch"
    )
    outputs = {
        "sacct": subprocess.CompletedProcess(
            [], 0, f"123|controller|PENDING||{submit}\n", ""
        ),
        "squeue": subprocess.CompletedProcess(
            [], 0, f"123|controller|PENDING|{token}|{submit}|afterany:100\n", ""
        ),
    }
    with pytest.raises(control.SchedulerAmbiguity, match="dependency conflict"):
        control.query_scheduler(
            runner=lambda argv: outputs[argv[0]], user="u", now=100.0
        )


def test_scheduler_rejects_sacct_comment_submit_line_disagreement():
    token = control.job_token("dispatcher", 1, "intent1")
    wrong = control.job_token("dispatcher", 1, "other")
    outputs = {
        "squeue": subprocess.CompletedProcess([], 0, "", ""),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|FAILED|{wrong}|sbatch --comment={token} /x.sbatch\n",
            "",
        ),
    }
    with pytest.raises(control.SchedulerAmbiguity, match="Comment/SubmitLine conflict"):
        control.query_scheduler(
            runner=lambda argv: outputs[argv[0]], user="u", now=100.0
        )


def test_scheduler_query_timeout_fails_closed_and_can_be_reported():
    def runner(argv):
        if argv[0] == "squeue":
            raise subprocess.TimeoutExpired(argv, timeout=15.0)
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(control.ControlError, match="squeue failed"):
        control.query_scheduler(runner=runner, user="u", now=100.0)

    snapshot = control.query_scheduler(
        runner=runner, user="u", now=100.0, tolerate_errors=True
    )
    assert snapshot.squeue_ok is False
    assert snapshot.sacct_ok is True
    assert snapshot.errors and "squeue failed" in snapshot.errors[0]


def test_rendered_generation_files_are_immutable_and_disable_requeue(tmp_path):
    state_dir, initialized = initialize(tmp_path)
    path = control.render_generation_sbatch(
        state_dir,
        initialized,
        role="dispatcher",
        generation=7,
        intent_token="abc123",
    )
    text = path.read_text()
    assert path.name == "dispatch.g000007.abc123.sbatch"
    assert "#SBATCH --no-requeue" in text
    assert "generation=7;intent=abc123" in text
    assert '--generation 7 --intent-token "abc123"' in text
    assert 'export ASYS_RELEASE_ID="sweep-recovery-schema5-v1.1"' in text
    assert (
        'export ASYS_RELEASE_WORKTREE="'
        + initialized["immutable"]["release_worktree"]
        + '"'
        in text
    )
    assert "export ASYS_MODEL_CONTRACT_SHA256=" in text
    assert "export ASYS_FLEET_CONTRACT_SHA256=" in text
    assert "export ASYS_HARNESS_ENVIRONMENT_SHA256=" in text
    assert "export ASYS_SERVING_ENVIRONMENT_SHA256=" in text
    assert 'export ASYS_ROLLOUT_GENERATION="0"' in text
    assert "source " not in text
    assert "mamba activate" not in text
    assert (
        "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV" in text
    )
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "export PYTHONSAFEPATH=1" in text
    assert 'export HF_HOME="' + initialized["immutable"]["hf_home"] + '"' in text
    assert (
        'export ASYS_RESULTS_ROOT="' + initialized["immutable"]["results_root"] + '"'
        in text
    )
    assert "export HF_HUB_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert "export HF_DATASETS_OFFLINE=1" in text
    with pytest.raises(control.ControlError, match="ambient environment activation"):
        control.validate_controller_generation_sbatch(
            initialized,
            payload=text + '\nsource "/tmp/common.sh"\nmamba activate legacy\n',
            state_dir=state_dir,
            role="dispatcher",
            generation=7,
            intent_token="abc123",
        )
    assert path.stat().st_mode & 0o222 == 0
    assert (
        control.render_generation_sbatch(
            state_dir,
            initialized,
            role="dispatcher",
            generation=7,
            intent_token="abc123",
        )
        == path
    )

    fleet_path = control.render_generation_sbatch(
        state_dir,
        initialized,
        role="fleet_supervisor",
        generation=8,
        intent_token="def456",
    )
    fleet_text = fleet_path.read_text()
    assert "source " not in fleet_text
    assert "mamba activate" not in fleet_text
    assert "export HF_DATASETS_OFFLINE=1" in fleet_text
    assert (
        'export ASYS_RELEASE_WORKTREE="'
        + initialized["immutable"]["release_worktree"]
        + '"'
        in fleet_text
    )
    assert (
        'export ASYS_RESULTS_ROOT="' + initialized["immutable"]["results_root"] + '"'
        in fleet_text
    )
    with pytest.raises(control.ControlError, match="ambient environment activation"):
        control.validate_controller_generation_sbatch(
            initialized,
            payload=fleet_text + "\nconda activate mutable\n",
            state_dir=state_dir,
            role="fleet_supervisor",
            generation=8,
            intent_token="def456",
        )


def test_batch_provenance_environment_is_run_scoped_and_fail_closed(tmp_path):
    state_dir, initialized = initialize(tmp_path)
    run_id = "full_sweep_agent_count_7_schema5_v1"
    expected_policy = next(
        run["policy_sha256"]
        for run in initialized["immutable"]["runs"]
        if run["run_id"] == run_id
    )
    with pytest.raises(control.ControlError, match="disabled by desired state"):
        control.production_environment_from_state(state_dir, run_id=run_id)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    environment = control.production_environment_from_state(state_dir, run_id=run_id)
    assert environment == {
        "ASYS_RELEASE_ID": "sweep-recovery-schema5-v1.1",
        "ASYS_MODEL_CONTRACT_SHA256": initialized["immutable"]["model_contract_sha256"],
        "ASYS_FLEET_CONTRACT_SHA256": initialized["immutable"]["fleet_contract_sha256"],
        "ASYS_HARNESS_ENVIRONMENT_SHA256": initialized["immutable"][
            "harness_environment_sha256"
        ],
        "ASYS_SERVING_ENVIRONMENT_SHA256": initialized["immutable"][
            "serving_environment_sha256"
        ],
        "ASYS_ROLLOUT_GENERATION": "1",
        "ASYS_IMMUTABLE_PINS_SHA256": initialized["immutable_sha256"],
        "ASYS_RUNTIME_ATTESTATION": control.load_control(state_dir)[
            control.RUNTIME_ATTESTATION_STATE_KEY
        ]["path"],
        "ASYS_RUNTIME_ATTESTATION_SHA256": control.load_control(state_dir)[
            control.RUNTIME_ATTESTATION_STATE_KEY
        ]["sha256"],
        "ASYS_RUNTIME_INTEGRITY_LEASE": control.load_control(state_dir)[
            control.RUNTIME_ATTESTATION_STATE_KEY
        ]["lease_path"],
        "ASYS_ARTIFACT_POLICY_SHA256": expected_policy,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_production_batch_validator_hard_enforces_slurm_contract(tmp_path):
    _, initialized = initialize(tmp_path)
    execution = control.production_cell_execution(initialized)
    manifest_digest = "a" * 64
    payload = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --no-requeue
#SBATCH --array=0-23
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export HF_HOME="{execution["hf_home"]}"
export ASYS_RELEASE_WORKTREE="{execution["release_worktree"]}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec {shlex.quote(execution["python"])} -u {shlex.quote(execution["dispatcher_script"])} run-task \\
  --batch-manifest-sha256 "{manifest_digest}" \\
  --expected-release-root {shlex.quote(execution["release_worktree"])} \\
  --expected-harness-prefix {shlex.quote(execution["harness_prefix"])}
"""
    assert f'--batch-manifest-sha256 "{manifest_digest}" \\' in payload
    control.validate_production_batch_sbatch(
        initialized, payload=payload, task_count=24
    )
    with pytest.raises(control.ControlError, match="manifest digest"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace(
                f'  --batch-manifest-sha256 "{manifest_digest}" \\\n', ""
            ),
            task_count=24,
        )
    with pytest.raises(control.ControlError, match="missing.*no-requeue"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace("#SBATCH --no-requeue\n", ""),
            task_count=24,
        )
    with pytest.raises(control.ControlError, match="exceeds"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace("0-23", "0-24"),
            task_count=25,
        )
    with pytest.raises(control.ControlError, match="ambient environment"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload + '\nsource "/tmp/common.sh"\nmamba activate legacy\n',
            task_count=24,
        )
    with pytest.raises(control.ControlError, match="immutable runtime"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace(execution["python"], "/usr/bin/python"),
            task_count=24,
        )
    with pytest.raises(control.ControlError, match="exact HF_HOME assignment"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload + '\nexport HF_HOME="/tmp/mutable-cache"\n',
            task_count=24,
        )


def test_submission_persists_intent_and_commits_exact_job_id_once(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    calls = []

    def submit(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "321;cluster\n", "")

    empty = control.SchedulerSnapshot((), 51.0)
    record = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=empty,
        submit_runner=submit,
        now=51.0,
    )
    assert record["state"] == "submitted"
    assert record["job_id"] == "321"
    assert calls[0][0:2] == ["sbatch", "--parsable"]
    assert any(arg.startswith("--comment=asys-schema5-v1;") for arg in calls[0])
    persisted = control.load_control(state_dir)["controllers"]["dispatcher"]
    assert persisted["submission_intent"]["job_id"] == "321"
    assert persisted["successor"]["job_id"] == "321"

    live = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "321", "asys-s5-dispatch", "RUNNING", record["job_token"]
            ),
        ),
        52.0,
    )
    same = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=live,
        submit_runner=lambda _argv: pytest.fail("duplicate submission"),
        now=52.0,
    )
    assert same["job_id"] == "321"


def test_accepted_controller_id_absence_never_triggers_duplicate_submission(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    submitted = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=control.SchedulerSnapshot((), 51.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "321\n", ""),
        now=51.0,
    )

    with pytest.raises(control.SchedulerAmbiguity, match="duplicate replacement"):
        control.submit_controller_intent(
            state_dir,
            role="dispatcher",
            target="successor",
            dependency_job_id="700",
            scheduler=control.SchedulerSnapshot((), 400.0),
            submit_runner=lambda _argv: pytest.fail("must not resubmit accepted job"),
            now=400.0,
        )
    persisted = control.load_control(state_dir)["controllers"]["dispatcher"]
    assert persisted["successor"]["job_id"] == submitted["job_id"] == "321"


def test_running_successor_does_not_expire_historical_fleet_launch_gate(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)

    # Fleet readiness is intentionally recent at resume, but controller generations
    # are 12 hours long.  A successor submission must keep verifying the sealed gate
    # bytes without treating their captured timestamp as a renewable controller lease.
    much_later = 50.0 + 2 * 86_400.0
    record = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=control.SchedulerSnapshot((), much_later),
        submit_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "321\n", ""),
        now=much_later,
    )

    assert record["job_id"] == "321"
    assert record["dependency_job_id"] == "700"


def test_initial_resume_still_requires_recent_fleet_launch_gate(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir, now=999.0)

    with pytest.raises(control.ReadinessError, match="fleet readiness evidence is not recent"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 1_000.0),
            now=1_000.0,
        )

    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["rollout_generation"] == 0


def test_crash_window_intent_is_adopted_from_scheduler_without_sbatch(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)

    class Crash(BaseException):
        pass

    with pytest.raises(Crash):
        control.submit_controller_intent(
            state_dir,
            role="fleet_supervisor",
            target="successor",
            dependency_job_id="701",
            scheduler=control.SchedulerSnapshot((), 51.0),
            submit_runner=lambda _argv: (_ for _ in ()).throw(Crash()),
            now=51.0,
        )
    intent = control.load_control(state_dir)["controllers"]["fleet_supervisor"][
        "submission_intent"
    ]
    assert intent["state"] == "submitting"
    accepted = control.SchedulerSnapshot(
        (control.SchedulerJob("444", "asys-s5-fleet", "PENDING", intent["job_token"]),),
        52.0,
    )
    adopted = control.submit_controller_intent(
        state_dir,
        role="fleet_supervisor",
        target="successor",
        dependency_job_id="701",
        scheduler=accepted,
        submit_runner=lambda _argv: pytest.fail("adoption must not resubmit"),
        now=52.0,
    )
    assert adopted["job_id"] == "444"
    assert adopted["adopted_from_scheduler"] is True


def test_exact_id_claim_fences_duplicate_controller(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    submitted = control.load_control(state_dir)["controllers"]["dispatcher"]["active"]
    claimed = control.claim_controller(
        state_dir,
        role="dispatcher",
        generation=submitted["generation"],
        intent_token=submitted["intent_token"],
        job_id="700",
        active_scheduler_job_ids={"700"},
        now=52.0,
    )
    assert claimed["state"] == "running"
    assert claimed["fencing_primitive"] == control.SHARED_CONTROLLER_FENCING_PRIMITIVE
    assert (
        claimed["controller_primitives"]
        == control.shared_controller_primitive_contract()
    )
    with pytest.raises(control.ControllerFenced, match="exact IDs"):
        control.claim_controller(
            state_dir,
            role="dispatcher",
            generation=submitted["generation"],
            intent_token=submitted["intent_token"],
            job_id="701",
            active_scheduler_job_ids={"700", "701"},
            now=53.0,
        )


def test_role_singleton_lock_is_cross_process_visible(tmp_path):
    state_dir, _ = initialize(tmp_path)
    with control.role_singleton_lock(state_dir, "dispatcher"):
        with pytest.raises(control.ControlError, match="already held"):
            with control.role_singleton_lock(state_dir, "dispatcher"):
                pytest.fail("duplicate role lock entered")


def test_repair_chain_is_idempotent_for_two_roles(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    _, initial_jobs = resume_ready(state_dir, now=50.0)
    terminal_jobs = tuple(
        control.SchedulerJob(
            job.job_id,
            job.job_name,
            "COMPLETED",
            job.comment,
            job.command,
            source="sacct",
        )
        for job in initial_jobs
    )
    ids = iter(("800", "801"))
    first_calls = []

    def submit(argv):
        first_calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, next(ids) + "\n", "")

    result = control.repair_chains(
        state_dir,
        snapshot=control.SchedulerSnapshot(terminal_jobs, 240.0),
        submit_runner=submit,
        now=240.0,
    )
    assert {item["role"] for item in result["submitted"]} == set(control.ROLE_NAMES)
    persisted = control.load_control(state_dir)
    live_jobs = tuple(
        control.SchedulerJob(record["job_id"], role, "PENDING", record["job_token"])
        for role in control.ROLE_NAMES
        for record in (persisted["controllers"][role]["active"],)
    )
    second = control.repair_chains(
        state_dir,
        snapshot=control.SchedulerSnapshot(live_jobs, 52.0),
        submit_runner=lambda _argv: pytest.fail("idempotent repair resubmitted"),
        now=52.0,
    )
    assert set(second["existing"]) == set(control.ROLE_NAMES)
    assert second["submitted"] == []


def test_live_status_never_promotes_cached_jobs_to_live(tmp_path):
    state_dir, _ = initialize(tmp_path)
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "poll_number": 9,
                "updated_at": 1.0,
                "jobs": {"stale": {"state": "active"}},
                "cells": {"stale": {"state": "active"}},
            }
        ),
        encoding="utf-8",
    )
    report = control.live_status(
        state_dir, snapshot=control.SchedulerSnapshot((), 100.0)
    )
    assert report["scheduler"]["active_cell_tasks"] == 0
    assert report["dispatcher_cache"]["cached_job_records"] == 1
    assert report["dispatcher_cache"]["cached_state_only"] is True


def test_monitor_cadences_and_frozen_command_are_pinned(tmp_path):
    state_dir, initialized = initialize(tmp_path)
    rows = initialized["monitoring"]["cadences"]
    assert {
        cadence: row["interval_seconds"] for cadence, row in rows.items()
    } == control.MONITOR_CADENCE_SECONDS
    assert control.monitor_due_cadences(initialized, now=10.0) == ()

    command = control.schema5_monitor_command(state_dir, initialized, cadence="health")
    immutable = initialized["immutable"]
    assert command[:3] == [
        str(Path(immutable["harness_environment_prefix"]) / "bin" / "python"),
        "-u",
        str(Path(immutable["release_worktree"]) / "scripts" / "schema5_monitor.py"),
    ]
    assert command[command.index("--results-root") + 1] == immutable["results_root"]
    assert command[command.index("--state-dir") + 1] == str(state_dir.resolve())
    assert command[command.index("--config") + 1] == str(
        Path(immutable["release_worktree"]) / "configs" / "schema5_monitoring.v1.json"
    )
    assert "--persist" in command
    assert "--send-email" in command
    assert "--probe-endpoints" in command
    semantic = control.schema5_monitor_command(
        state_dir, initialized, cadence="semantic"
    )
    assert "--probe-endpoints" not in semantic

    elsewhere = tmp_path / "other-state"
    with pytest.raises(control.ImmutablePinError, match="pinned results-root"):
        control.schema5_monitor_command(elsewhere, initialized, cadence="health")


def test_monitor_timestamps_survive_successor_and_exit_two_is_success(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    current = control.load_control(state_dir)
    assert set(control.monitor_due_cadences(current, now=50.0)) == {
        "health",
        "semantic",
        "daily",
    }

    attempt = control.begin_monitor_attempt(
        state_dir,
        cadence="health",
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=50.0,
    )
    started = control.load_control(state_dir)["monitoring"]["cadences"]["health"]
    assert started["next_due_timestamp"] == 310.0
    completed = control.complete_monitor_attempt(
        state_dir,
        cadence="health",
        attempt_id=attempt["attempt_id"],
        returncode=2,
        now=55.0,
    )
    assert completed["last_success_timestamp"] == 55.0
    assert completed["consecutive_failures"] == 0
    assert completed["active_attempt"] is None

    successor_attempt = control.begin_monitor_attempt(
        state_dir,
        cadence="health",
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=310.0,
    )
    recovered = control.recover_abandoned_monitor_attempts(
        state_dir,
        controller_generation=2,
        controller_intent_token="successor",
        controller_job_id="701",
        now=320.0,
    )
    assert [row["attempt_id"] for row in recovered] == [successor_attempt["attempt_id"]]
    restarted = control.load_control(state_dir)["monitoring"]["cadences"]["health"]
    assert restarted["active_attempt"] is None
    assert restarted["next_due_timestamp"] == 320.0
    assert restarted["consecutive_failures"] == 1


def test_monitor_execution_failure_retries_in_sixty_seconds(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    attempt = control.begin_monitor_attempt(
        state_dir,
        cadence="semantic",
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=50.0,
    )
    failed = control.complete_monitor_attempt(
        state_dir,
        cadence="semantic",
        attempt_id=attempt["attempt_id"],
        returncode=1,
        now=51.0,
    )
    assert failed["last_failure_timestamp"] == 51.0
    assert failed["next_due_timestamp"] == 111.0
    assert failed["consecutive_failures"] == 1
    state = control.load_control(state_dir)
    assert "semantic" not in control.monitor_due_cadences(state, now=110.0)
    assert "semantic" in control.monitor_due_cadences(state, now=111.0)


def test_monitor_launch_failures_alert_without_escaping_supervision(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    original_record_alert = control.record_alert

    def no_mail(*args, **kwargs):
        kwargs["send_email"] = False
        return original_record_alert(*args, **kwargs)

    monkeypatch.setattr(control, "record_alert", no_mail)

    def cannot_launch(*_args, **_kwargs):
        raise OSError("simulated exec failure")

    processes = {}
    # The fail-safe service contract is that one (or all) cadence failures are
    # persisted and returned from, never raised into the dispatcher supervisor.
    control._service_schema5_monitors(
        state_dir,
        processes=processes,
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=50.0,
        popen_factory=cannot_launch,
    )
    assert processes == {}
    persisted = control.load_control(state_dir)
    assert persisted["desired_state"] == "running"
    assert all(
        row["active_attempt"] is None
        and row["consecutive_failures"] == 1
        and row["next_due_timestamp"] == 110.0
        for row in persisted["monitoring"]["cadences"].values()
    )
    assert {
        alert["dedupe_key"]
        for alert in persisted["alerts"]
        if alert["resolved_at"] is None
    } == {
        "monitor-supervisor:health",
        "monitor-supervisor:semantic",
        "monitor-supervisor:daily",
    }


def test_throughput_epochs_survive_successors_and_close_on_pause(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    first = control.record_successful_poll(
        state_dir, validated_qids=100, fleet_generation="fleet-a", now=60.0
    )
    second = control.record_successful_poll(
        state_dir, validated_qids=150, fleet_generation="fleet-a", now=70.0
    )
    assert len(first["throughput_epochs"]) == 1
    assert len(second["throughput_epochs"]) == 1
    assert len(second["throughput_epochs"][0]["samples"]) == 2
    changed = control.record_successful_poll(
        state_dir, validated_qids=151, fleet_generation="fleet-b", now=80.0
    )
    assert len(changed["throughput_epochs"]) == 2
    assert changed["throughput_epochs"][0]["close_reason"] == "material_fleet_change"
    paused = control.pause_control(state_dir, drain=True, now=90.0)
    assert paused["throughput_epochs"][-1]["close_reason"] == "pause"
    with pytest.raises(control.ControlError, match="while paused"):
        control.record_successful_poll(
            state_dir, validated_qids=200, fleet_generation="fleet-b", now=100.0
        )


def _ramp_evidence(
    state_dir: Path,
    *,
    committed_at: float,
    cadence: str,
    fleet_generation: str,
    qids: dict[str, int] | None = None,
    health_clean: bool = True,
    semantic_clean: bool = True,
    critical_keys: tuple[str, ...] = (),
    blocking_keys: tuple[str, ...] = (),
) -> tuple[Path, str]:
    state = control.load_control(state_dir)
    captured_at = committed_at - 1.0
    observation = {
        "schema_version": 1,
        "protocol": control.ADMISSION_RAMP_EVIDENCE_PROTOCOL,
        "captured_timestamp": captured_at,
        "committed_timestamp": committed_at,
        "cadence": cadence,
        "control_immutable_sha256": state["immutable_sha256"],
        "rollout_generation": state["rollout_generation"],
        "admission_ceiling": state["admission"]["current_ceiling"],
        "fleet_generation": fleet_generation,
        "production_health_clean": health_clean,
        "semantic_integrity_clean": semantic_clean if cadence != "health" else None,
        "critical_finding_keys": sorted(critical_keys),
        "promotion_blocking_finding_keys": sorted(
            set(critical_keys) | set(blocking_keys)
        ),
        "run_validated_qids": copy.deepcopy(qids) if cadence != "health" else None,
    }
    report = {
        "schema_version": 1,
        "cadence": cadence,
        "captured_timestamp": captured_at,
        "health": {
            "fleet_generation": fleet_generation,
            "control": {
                "rollout_generation": state["rollout_generation"],
                "admission": {
                    "current_ceiling": state["admission"]["current_ceiling"]
                },
            },
        },
        "ramp_observation": observation,
    }
    if cadence != "health":
        assert qids is not None
        report["semantic"] = {
            "outcomes": {"validated_qids": sum(qids.values())},
            "runs": {
                run_id: {"outcomes": {"validated_qids": qids[run_id]}}
                for run_id in control.REQUIRED_RUNS
            },
        }
    path = (
        state_dir
        / "monitoring"
        / cadence
        / f"ramp-{int(committed_at * 1_000_000):020d}.json"
    )
    control._atomic_write_json(path, report)
    path.chmod(0o444)
    return path, _sha(path)


def _commit_ramp_evidence(
    state_dir: Path, *, committed_at: float, cadence: str, **kwargs
) -> dict:
    path, digest = _ramp_evidence(
        state_dir, committed_at=committed_at, cadence=cadence, **kwargs
    )
    return control.record_admission_ramp_observation(
        state_dir,
        evidence_path=path,
        evidence_sha256=digest,
        now=committed_at,
    )


def test_admission_ramp_automatically_promotes_only_from_complete_clean_windows(
    tmp_path,
):
    assert control.ADMISSION_RAMP_REQUIREMENTS == {
        24: {"next_ceiling": 96, "clean_seconds": 3_600.0},
        96: {"next_ceiling": 192, "clean_seconds": 21_600.0},
        192: {"next_ceiling": 384, "clean_seconds": 43_200.0},
    }
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    finished_run = "full_sweep_schema5_v1"
    qids = {
        run_id: (
            control.REQUIRED_RUN_QIDS[run_id] - 1
            if run_id == finished_run
            else 10
        )
        for run_id in control.REQUIRED_RUNS
    }
    control.record_successful_poll(
        state_dir,
        validated_qids=sum(qids.values()),
        fleet_generation="fleet-a",
        now=90.0,
    )
    started = _commit_ramp_evidence(
        state_dir,
        committed_at=100.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=qids,
    )
    assert started["admission_ramp"]["last_action"]["action"] == "window_started"

    # Every stage is continuously covered by observations no more than 10 minutes
    # apart; promotion itself is always a semantic observation.  All-run progress is
    # required only for the first stage, so the run completed there can remain fixed.
    stage_boundaries = (
        (100.0, 3_600.0, 96),
        (3_700.0, 21_600.0, 192),
        (25_300.0, 43_200.0, 384),
    )
    for start, duration, target in stage_boundaries:
        final = start + duration
        timestamp = start + 600.0
        while timestamp < final:
            _commit_ramp_evidence(
                state_dir,
                committed_at=timestamp,
                cadence="health",
                fleet_generation="fleet-a",
            )
            timestamp += 600.0
        qids = {
            run_id: (
                value
                if value == control.REQUIRED_RUN_QIDS[run_id]
                else value + 1
            )
            for run_id, value in qids.items()
        }
        control.record_successful_poll(
            state_dir,
            validated_qids=sum(qids.values()),
            fleet_generation="fleet-a",
            now=final - 0.5,
        )
        promoted = _commit_ramp_evidence(
            state_dir,
            committed_at=final,
            cadence="semantic",
            fleet_generation="fleet-a",
            qids=qids,
        )
        assert promoted["admission"]["current_ceiling"] == target
        assert promoted["admission_ramp"]["last_action"]["action"] == "promoted"

    final = control.load_control(state_dir)
    assert [row["to_ceiling"] for row in final["admission_ramp"]["promotions"]] == [
        96,
        192,
        384,
    ]
    assert all(
        set(row["final_run_validated_qids"]) == set(control.REQUIRED_RUNS)
        and row["observations"]
        for row in final["admission_ramp"]["promotions"]
    )
    assert final["admission_ramp"]["promotions"][0]["all_runs_progressed"] is True
    assert all(
        row["all_runs_progressed"] is False
        and row["all_run_progress_required"] is False
        for row in final["admission_ramp"]["promotions"][1:]
    )

    # A new material fleet identity closes the throughput epoch and immediately
    # returns admission to the initial safety stage.
    changed = control.record_successful_poll(
        state_dir,
        validated_qids=sum(qids.values()),
        fleet_generation="fleet-b",
        now=68_600.0,
    )
    assert changed["admission"]["current_ceiling"] == 24
    assert changed["admission_ramp"]["window"] is None
    assert changed["throughput_epochs"][-2]["close_reason"] == "material_fleet_change"


def test_admission_ramp_resets_on_blocking_alert_gap_pause_and_rejects_manual_raise(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    qids = {run_id: 10 for run_id in control.REQUIRED_RUNS}
    control.record_successful_poll(
        state_dir,
        validated_qids=sum(qids.values()),
        fleet_generation="fleet-a",
        now=90.0,
    )
    _commit_ramp_evidence(
        state_dir,
        committed_at=100.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=qids,
    )
    # Exercise rollback from an already elevated stage without spending another full
    # staged timeline in this independent failure-path test.
    elevated = control.load_control(state_dir)
    elevated["admission"]["current_ceiling"] = 192
    elevated["admission_ramp"]["current_ceiling"] = 192
    elevated["admission_ramp"]["window"] = None
    control._save_control(state_dir, elevated, now=399.0)
    qos_blocked = _commit_ramp_evidence(
        state_dir,
        committed_at=400.0,
        cadence="health",
        fleet_generation="fleet-a",
        blocking_keys=("monitor:qos-memory",),
    )
    assert qos_blocked["admission_ramp"]["window"] is None
    assert qos_blocked["admission"]["current_ceiling"] == 24
    assert (
        qos_blocked["admission_ramp"]["last_action"]["reason"]
        == "promotion_blocking_alert"
    )

    _commit_ramp_evidence(
        state_dir,
        committed_at=500.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=qids,
    )
    starved = _commit_ramp_evidence(
        state_dir,
        committed_at=800.0,
        cadence="health",
        fleet_generation="fleet-a",
        blocking_keys=("monitor:starvation",),
    )
    assert starved["admission_ramp"]["window"] is None
    _commit_ramp_evidence(
        state_dir,
        committed_at=900.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=qids,
    )
    gap = _commit_ramp_evidence(
        state_dir,
        committed_at=1_600.0,
        cadence="health",
        fleet_generation="fleet-a",
    )
    assert gap["admission_ramp"]["window"] is None
    assert gap["admission_ramp"]["resets"][-1]["reason"] == "health_observation_gap"

    with pytest.raises(control.ControlError, match="monitor-controlled"):
        control.set_admission_ceiling(state_dir, ceiling=96, now=1_700.0)
    paused = control.pause_control(state_dir, drain=True, now=1_800.0)
    assert paused["admission"]["current_ceiling"] == 24
    assert paused["admission_ramp"]["window"] is None
    assert paused["admission_ramp"]["last_action"]["reason"] == "pause"

    evidence, digest = _ramp_evidence(
        state_dir,
        committed_at=1_900.0,
        cadence="health",
        fleet_generation="fleet-a",
    )
    alias = evidence.with_name("hardlink-alias.json")
    os.link(evidence, alias)
    with pytest.raises(control.ControlError, match="hardlink aliases"):
        control.record_admission_ramp_observation(
            state_dir,
            evidence_path=evidence,
            evidence_sha256=digest,
            now=1_900.0,
        )


def test_pause_cancels_only_token_verified_recorded_successor(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    submitted = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=control.SchedulerSnapshot((), 51.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "701\n", ""),
        now=51.0,
    )
    calls = []

    def cancel(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob("701", "successor", "PENDING", submitted["job_token"]),
            control.SchedulerJob("999", "unrelated", "PENDING", ""),
        ),
        52.0,
    )
    control.pause_control(
        state_dir, drain=True, scheduler=snapshot, cancel_runner=cancel, now=52.0
    )
    assert calls == [["scancel", "701"]]


def test_pause_takes_scheduler_snapshot_only_after_admission_boundary(tmp_path):
    state_dir, _ = initialize(tmp_path)
    holder = control.admission_boundary_lock(state_dir)
    holder.__enter__()
    started = threading.Event()
    snapshot_taken = threading.Event()
    failures = []

    def pause():
        started.set()
        try:
            control.pause_control(
                state_dir,
                drain=True,
                scheduler_reader=lambda: (
                    snapshot_taken.set()
                    or control.SchedulerSnapshot((), 52.0)
                ),
                now=52.0,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    worker = threading.Thread(target=pause)
    worker.start()
    assert started.wait(timeout=2.0)
    assert not snapshot_taken.wait(timeout=0.1)
    holder.__exit__(None, None, None)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert not failures
    assert snapshot_taken.is_set()


def test_pause_fails_closed_while_accepted_cell_is_inside_visibility_grace(tmp_path):
    state_dir, _ = initialize(tmp_path)
    sbatch_path = (state_dir / "batches" / "batch-batch1.sbatch").resolve()
    sbatch_path.parent.mkdir()
    sbatch_path.write_text("#!/bin/bash\n", encoding="utf-8")
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "jobs": {
                    "900": {
                        "job_id": "900",
                        "batch_id": "batch1",
                        "sbatch_path": str(sbatch_path),
                        "state": "submitted",
                    }
                },
                "intents": {
                    "batch1": {
                        "state": "submitted",
                        # A long validation/planning poll may predate the actual
                        # scheduler boundary by far more than the visibility grace.
                        "created_at": 1.0,
                        "submit_started_at": 55.0,
                        "job_id": "900",
                        "sbatch_path": str(sbatch_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(control.SchedulerVisibilityPending, match="not yet visible"):
        control.pause_control(
            state_dir,
            drain=True,
            scheduler=control.SchedulerSnapshot((), 60.0),
            now=60.0,
        )
    persisted = control.load_control(state_dir)
    assert persisted["desired_state"] == "paused"
    assert persisted["drain_requested"] is True


def test_pause_can_cancel_visible_idless_ambiguous_submit_by_exact_intent(tmp_path):
    state_dir, _ = initialize(tmp_path)
    sbatch_path = (state_dir / "batches" / "batch-batch1.sbatch").resolve()
    sbatch_path.parent.mkdir()
    sbatch_path.write_text("#!/bin/bash\n", encoding="utf-8")
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "jobs": {},
                "intents": {
                    "batch1": {
                        "state": "submitting",
                        "created_at": 1.0,
                        "submit_started_at": 55.0,
                        "job_id": None,
                        "sbatch_path": str(sbatch_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_4",
                "asys-dispatch-batch1",
                "PENDING",
                "asys-schema5-intent:batch1",
                f"sbatch {sbatch_path}",
                "squeue",
            ),
        ),
        60.0,
    )
    calls = []
    paused = control.pause_control(
        state_dir,
        drain=True,
        scheduler=snapshot,
        cancel_runner=lambda argv: (
            calls.append(list(argv)) or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=60.0,
    )
    assert calls == [["scancel", "900_4"]]
    assert paused["drain_intent"]["state"] == "complete"


def test_pause_usr1_signals_only_exact_ledger_bound_running_cell_tasks(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    sbatch_path = state_dir / "batches" / "batch-batch1.sbatch"
    sbatch_path.parent.mkdir()
    sbatch_path.write_text("#!/bin/bash\n", encoding="utf-8")
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "jobs": {
                    "900": {
                        "job_id": "900",
                        "batch_id": "batch1",
                        "sbatch_path": str(sbatch_path),
                    }
                },
                "intents": {
                    "batch1": {
                        "job_id": "900",
                        "state": "submitted",
                        "sbatch_path": str(sbatch_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-batch1",
                "RUNNING",
                "asys-schema5-intent:batch1",
                f"sbatch {sbatch_path}",
                "squeue",
            ),
            control.SchedulerJob(
                "900_4",
                "asys-dispatch-batch1",
                "PENDING",
                "asys-schema5-intent:batch1",
                f"sbatch {sbatch_path}",
                "squeue",
            ),
            control.SchedulerJob(
                "999",
                "unrelated",
                "RUNNING",
                "",
                "/tmp/unrelated.sbatch",
                "squeue",
            ),
        ),
        60.0,
    )
    calls = []
    paused = control.pause_control(
        state_dir,
        drain=True,
        scheduler=snapshot,
        cancel_runner=lambda argv: (
            calls.append(list(argv)) or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=60.0,
    )
    assert calls == [
        ["scancel", "--batch", "--signal=USR1", "900_3"],
        ["scancel", "900_4"],
    ]
    assert paused["drain_intent"]["state"] == "complete"
    assert paused["drain_intent"]["cell_task_ids"] == ["900_3"]
    assert paused["drain_intent"]["pending_cell_task_ids"] == ["900_4"]
    assert paused["drain_intent"]["results"]["cell_usr1:900_3"]["returncode"] == 0
    assert (
        paused["drain_intent"]["results"]["pending_cell_cancel:900_4"]["returncode"]
        == 0
    )


def test_pause_refuses_all_signals_when_any_cell_mapping_is_ambiguous(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    sbatch_path = state_dir / "batches" / "batch-batch1.sbatch"
    sbatch_path.parent.mkdir()
    sbatch_path.write_text("#!/bin/bash\n", encoding="utf-8")
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "jobs": {
                    "900": {
                        "job_id": "900",
                        "batch_id": "different-batch",
                        "sbatch_path": str(sbatch_path),
                    }
                },
                    "intents": {
                        "batch1": {
                            "state": "submitted",
                            "job_id": "900",
                            "sbatch_path": str(sbatch_path),
                        }
                    },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-batch1",
                "RUNNING",
                "asys-schema5-intent:batch1",
                f"sbatch {sbatch_path}",
                "squeue",
            ),
        ),
        60.0,
    )
    calls = []
    with pytest.raises(control.SchedulerAmbiguity, match="exact cell drain mapping"):
        control.pause_control(
            state_dir,
            drain=True,
            scheduler=snapshot,
            cancel_runner=lambda argv: calls.append(list(argv)),
            now=60.0,
        )
    assert calls == []
    assert control.load_control(state_dir)["desired_state"] == "paused"


def test_stale_heartbeat_fences_exact_id_and_releases_recorded_successor(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    active = control.load_control(state_dir)["controllers"]["dispatcher"]["active"]
    control.claim_controller(
        state_dir,
        role="dispatcher",
        generation=active["generation"],
        intent_token=active["intent_token"],
        job_id="700",
        active_scheduler_job_ids={"700"},
        now=52.0,
    )
    successor = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id="700",
        scheduler=control.SchedulerSnapshot((), 53.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "701\n", ""),
        now=53.0,
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob("700", "active", "RUNNING", active["job_token"]),
            control.SchedulerJob("701", "successor", "PENDING", successor["job_token"]),
        ),
        700.0,
    )
    calls = []
    outcome = control.trigger_stale_takeover(
        state_dir,
        role="dispatcher",
        snapshot=snapshot,
        cancel_runner=lambda argv: (
            calls.append(list(argv)) or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        submit_runner=lambda _argv: pytest.fail("recorded successor must be reused"),
        now=700.0,
    )
    assert calls == [["scancel", "700"]]
    assert outcome["action"] == "released_recorded_successor"
    assert outcome["successor_job_id"] == "701"


def test_alerts_are_persistent_and_deduplicated(tmp_path):
    state_dir, _ = initialize(tmp_path)
    first = control.record_alert(
        state_dir,
        kind="stale_heartbeat",
        severity="critical",
        message="missing",
        dedupe_key="controller:dispatcher",
        now=20.0,
    )
    second = control.record_alert(
        state_dir,
        kind="stale_heartbeat",
        severity="critical",
        message="still missing",
        dedupe_key="controller:dispatcher",
        now=30.0,
    )
    assert second["alert_id"] == first["alert_id"]
    assert second["occurrences"] == 2
    resolved = control.resolve_alert(
        state_dir, dedupe_key="controller:dispatcher", now=40.0
    )
    assert resolved["alerts"][0]["resolved_at"] is not None
    assert len((state_dir / control.ALERT_JOURNAL).read_text().splitlines()) == 3


def test_drill_tokens_are_distinct_and_render_no_production_managed_plane(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    state = control.load_drill_state(state_dir)
    assert control.load_control(state_dir)["desired_state"] == "paused"
    assert (
        control.build_reconciliation_report(
            control.load_control(state_dir), snapshot(), all_jobs=True, no_admit=True
        )["scheduler"]["schema5_job_count"]
        == 0
    )
    for role in control.ROLE_NAMES:
        active = state["roles"][role]["active"]
        assert (
            active["fencing_primitive"] == control.SHARED_CONTROLLER_FENCING_PRIMITIVE
        )
        assert (
            active["controller_primitives"]
            == control.shared_controller_primitive_contract()
        )
        assert active["job_token"].startswith(control.DRILL_TOKEN_PREFIX + ";")
        assert control.parse_job_token(active["job_token"]) is None
        text = Path(active["sbatch_path"]).read_text()
        assert "#SBATCH --no-requeue" in text
        assert "#SBATCH --time=01:00:00" in text
        assert "supervise-drill" in text
        assert " dispatch --results-root " not in text
        assert "keepalive.py" not in text
        assert Path(active["sbatch_path"]).stat().st_mode & 0o222 == 0
    assert len(jobs) == 4


def test_drill_start_requires_readiness_and_paused_empty_production_state(tmp_path):
    state_dir, _ = initialize(tmp_path)
    with pytest.raises(control.ReadinessError, match="snapshot"):
        control.start_controller_drill(
            state_dir, scheduler=control.SchedulerSnapshot((), 20.0), now=20.0
        )
    make_ready(state_dir)
    current = control.load_control(state_dir)
    current["desired_state"] = "resuming"
    current["resume_intent"] = {
        "state": "submitting_controllers",
        "rollout_generation": 1,
    }
    # This synthetic state tests the drill's desired-state fence, not the separate
    # generation-zero preseal invariant.
    current[control.SNAPSHOT_ATTESTATION_STATE_KEY] = None
    control._save_control(state_dir, current, now=30.0)
    with pytest.raises(control.ControlError, match="paused"):
        control.start_controller_drill(
            state_dir, scheduler=control.SchedulerSnapshot((), 31.0), now=31.0
        )


def test_exact_drill_kill_rejects_token_or_command_mismatch_without_scancel(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    state = control.load_drill_state(state_dir)
    active = state["roles"]["dispatcher"]["active"]
    jobs[:] = [
        (
            control.SchedulerJob(
                job.job_id,
                job.job_name,
                job.state,
                job.comment,
                "/wrong/spooled-command",
                dependency=job.dependency,
            )
            if job.job_id == active["job_id"]
            else job
        )
        for job in jobs
    ]
    calls = []
    with pytest.raises(control.SchedulerAmbiguity, match="recorded sbatch"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(60.0),
            cancel_runner=lambda argv: calls.append(list(argv)),
            now=60.0,
        )
    assert calls == []
    assert control.load_drill_state(state_dir)["roles"]["dispatcher"]["kill"] is None


def test_exact_drill_kill_rejects_duplicate_token_before_scancel(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    active = control.load_drill_state(state_dir)["roles"]["dispatcher"]["active"]
    jobs.append(
        control.SchedulerJob(
            "999",
            "duplicate",
            "PENDING",
            active["job_token"],
            f"sbatch {active['sbatch_path']}",
        )
    )
    calls = []

    with pytest.raises(control.SchedulerAmbiguity, match="duplicate drill token"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(60.0),
            cancel_runner=lambda argv: calls.append(list(argv)),
            now=60.0,
        )

    assert calls == []


def test_exact_drill_kill_rejects_successor_with_wrong_scheduler_dependency(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    successor = control.load_drill_state(state_dir)["roles"]["dispatcher"][
        "successor"
    ]
    jobs[:] = [
        (
            control.SchedulerJob(
                job.job_id,
                job.job_name,
                job.state,
                job.comment,
                job.command,
                source=job.source,
                dependency="afterany:999999",
            )
            if job.job_id == successor["job_id"]
            else job
        )
        for job in jobs
    ]
    calls = []

    with pytest.raises(control.SchedulerAmbiguity, match="dependency"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(60.0),
            cancel_runner=lambda argv: calls.append(list(argv)),
            now=60.0,
        )

    assert calls == []
    assert control.load_drill_state(state_dir)["roles"]["dispatcher"]["kill"] is None


def test_exact_drill_kill_recovers_crash_after_scancel_from_sacct(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    old = control.load_drill_state(state_dir)["roles"]["dispatcher"]["active"]

    def crash_after_cancel(_argv):
        jobs[:] = [
            control.SchedulerJob(
                job.job_id,
                job.job_name,
                "CANCELLED" if job.job_id == old["job_id"] else job.state,
                job.comment,
                job.command,
                source="sacct" if job.job_id == old["job_id"] else job.source,
                dependency=job.dependency,
            )
            for job in jobs
        ]
        raise RuntimeError("caller died after scancel acceptance")

    with pytest.raises(RuntimeError, match="after scancel acceptance"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(60.0),
            cancel_runner=crash_after_cancel,
            now=60.0,
        )
    assert (
        control.load_drill_state(state_dir)["roles"]["dispatcher"]["kill"]["state"]
        == "cancelling"
    )

    reconciled = control.kill_drill_controller(
        state_dir,
        role="dispatcher",
        snapshot=snapshot(61.0),
        cancel_runner=lambda _argv: pytest.fail("terminal exact kill must be adopted"),
        now=61.0,
    )
    assert reconciled["returncode"] == 0
    assert reconciled["state"] == "cancelled_reconciled_terminal"
    assert reconciled["scheduler_terminal_state"] == "CANCELLED"

    control._record_drill_exit(
        state_dir,
        drill_id=control.load_drill_state(state_dir)["drill_id"],
        role="dispatcher",
        generation=old["generation"],
        intent_token=old["intent_token"],
        job_id=old["job_id"],
        reason="scheduler_terminal",
        now=61.5,
    )
    successor = control.load_drill_state(state_dir)["roles"]["dispatcher"]["successor"]
    control.claim_drill_controller(
        state_dir,
        drill_id=control.load_drill_state(state_dir)["drill_id"],
        role="dispatcher",
        generation=successor["generation"],
        intent_token=successor["intent_token"],
        job_id=successor["job_id"],
        active_scheduler_job_ids=snapshot(62.0).active_job_ids,
        now=62.0,
    )
    control.submit_drill_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id=successor["job_id"],
        scheduler=snapshot(62.0),
        submit_runner=submit,
        now=62.0,
    )
    control.heartbeat_drill_controller(
        state_dir,
        drill_id=control.load_drill_state(state_dir)["drill_id"],
        role="dispatcher",
        generation=successor["generation"],
        intent_token=successor["intent_token"],
        job_id=successor["job_id"],
        now=63.0,
    )
    recovery = control.record_drill_recovery(
        state_dir,
        role="dispatcher",
        snapshot=snapshot(63.0),
        now=63.0,
    )
    assert recovery["killed_job_id"] == old["job_id"]


def test_exact_drill_kill_does_not_adopt_unrelated_terminal_state(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    old = control.load_drill_state(state_dir)["roles"]["dispatcher"]["active"]

    def crash_then_fail(_argv):
        jobs[:] = [
            control.SchedulerJob(
                job.job_id,
                job.job_name,
                "FAILED" if job.job_id == old["job_id"] else job.state,
                job.comment,
                job.command,
                source="sacct" if job.job_id == old["job_id"] else job.source,
                dependency=job.dependency,
            )
            for job in jobs
        ]
        raise RuntimeError("caller died")

    with pytest.raises(RuntimeError, match="caller died"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(60.0),
            cancel_runner=crash_then_fail,
            now=60.0,
        )

    with pytest.raises(control.SchedulerAmbiguity, match="does not prove scancel"):
        control.kill_drill_controller(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(61.0),
            cancel_runner=lambda _argv: pytest.fail("FAILED must not be adopted"),
            now=61.0,
        )


def test_both_exact_kills_recover_by_unique_successors_within_fifteen_minutes(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    dispatcher = recover_drill_role(
        state_dir,
        jobs,
        snapshot,
        submit,
        role="dispatcher",
        now=60.0,
    )
    fleet = recover_drill_role(
        state_dir,
        jobs,
        snapshot,
        submit,
        role="fleet_supervisor",
        now=70.0,
    )
    assert dispatcher["successor_job_id"] != dispatcher["killed_job_id"]
    assert fleet["successor_job_id"] != fleet["killed_job_id"]
    assert dispatcher["recovery_seconds"] == 2.0
    assert fleet["recovery_seconds"] == 2.0
    assert control.load_control(state_dir)["desired_state"] == "paused"
    assert all(
        control.load_control(state_dir)["controllers"][role]["active"] is None
        for role in control.ROLE_NAMES
    )


def test_drill_recovery_rejects_a_later_replacement_of_the_frozen_successor(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0
    )
    state = control.load_drill_state(state_dir)
    drill_id = state["drill_id"]
    first_successor = state["roles"]["dispatcher"]["active"]
    later_successor = state["roles"]["dispatcher"]["successor"]
    jobs[:] = [job for job in jobs if job.job_id != first_successor["job_id"]]
    control._record_drill_exit(
        state_dir,
        drill_id=drill_id,
        role="dispatcher",
        generation=first_successor["generation"],
        intent_token=first_successor["intent_token"],
        job_id=first_successor["job_id"],
        reason="second_failure",
        now=63.0,
    )
    control.claim_drill_controller(
        state_dir,
        drill_id=drill_id,
        role="dispatcher",
        generation=later_successor["generation"],
        intent_token=later_successor["intent_token"],
        job_id=later_successor["job_id"],
        active_scheduler_job_ids=snapshot(64.0).active_job_ids,
        now=64.0,
    )
    control.submit_drill_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id=later_successor["job_id"],
        scheduler=snapshot(64.0),
        submit_runner=submit,
        now=64.0,
    )
    control.heartbeat_drill_controller(
        state_dir,
        drill_id=drill_id,
        role="dispatcher",
        generation=later_successor["generation"],
        intent_token=later_successor["intent_token"],
        job_id=later_successor["job_id"],
        now=65.0,
    )
    with control.drill_lock(state_dir):
        state = control.load_drill_state(state_dir)
        state["roles"]["dispatcher"]["recovery"] = None
        control._save_drill_state(state_dir, state, now=65.0)

    with pytest.raises(control.ControllerFenced, match="exact successor frozen"):
        control.record_drill_recovery(
            state_dir,
            role="dispatcher",
            snapshot=snapshot(65.0),
            now=65.0,
        )


def test_drill_finish_is_marker_last_and_completed_live_status_succeeds(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    with pytest.raises(control.ReadinessError, match="drill marker is missing"):
        control.resume_control(state_dir, now=50.0)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )
    cancelled = []

    def cancel(argv):
        cancelled.append(list(argv))
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    marker = control.finish_controller_drill(
        state_dir,
        scheduler_reader=lambda: snapshot(80.0),
        cancel_runner=cancel,
        timeout_seconds=1.0,
        poll_seconds=0.0,
        now=80.0,
    )
    assert marker["passed"] is True
    recovered_state = control.load_drill_state(state_dir)
    completed_events = [
        event
        for event in recovered_state["transition_history"]
        if event["event"] == "completed"
    ]
    assert len(completed_events) == 1
    assert completed_events[0]["timestamp"] == recovered_state["completed_timestamp"]
    assert marker["roles"].keys() == set(control.ROLE_NAMES)
    assert (state_dir / control.DRILL_COMPLETE_FILENAME).is_file()
    assert control.load_drill_state(state_dir)["phase"] == "completed"
    assert cancelled
    validated = control.validate_controller_drill_marker(
        state_dir, control.load_control(state_dir)
    )
    assert validated == marker
    final_snapshot = snapshot(81.0)
    status = control.controller_drill_status(
        state_dir, snapshot=final_snapshot, now=81.0
    )
    assert status["healthy"] is True
    assert status["completed"] is True
    assert status["ready_for_kill"] is False
    monkeypatch.setattr(control, "_scheduler_for_cli", lambda **_kwargs: final_snapshot)
    assert (
        control.main(["--state-dir", str(state_dir), "drill", "status", "--live"]) == 0
    )


def test_resume_and_completed_start_reject_live_drill_namespace_before_mutation(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    marker = complete_drill_marker(state_dir)
    token = control.drill_job_token(marker["drill_id"], "dispatcher", 9, "orphan")
    orphan = control.SchedulerJob(
        "998", "orphan", "RUNNING", token, "sbatch /orphan.sbatch"
    )
    snapshot = control.SchedulerSnapshot((orphan,), 50.0)

    with pytest.raises(control.SchedulerAmbiguity, match="still has live jobs"):
        control.start_controller_drill(state_dir, scheduler=snapshot, now=50.0)
    with pytest.raises(control.SchedulerAmbiguity, match="live controller-drill jobs"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: snapshot,
            now=50.0,
        )

    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["rollout_generation"] == 0


@pytest.mark.parametrize(
    "comment",
    (
        control.job_token("dispatcher", 9, "orphan"),
        control.TOKEN_PREFIX + ";malformed",
    ),
)
def test_resume_rejects_live_production_namespace_before_state_change(
    tmp_path, comment
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    orphan = control.SchedulerJob(
        "997", "orphan-production", "RUNNING", comment, "sbatch /orphan.sbatch"
    )
    submissions = []

    with pytest.raises(control.SchedulerAmbiguity, match="unclean live production join"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((orphan,), 50.0),
            submit_runner=lambda argv: submissions.append(list(argv)),
            now=50.0,
        )

    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["rollout_generation"] == 0
    assert submissions == []


def test_malformed_live_drill_namespace_fences_start_resume_and_status(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    malformed = control.SchedulerJob(
        "996",
        "malformed-drill",
        "RUNNING",
        control.DRILL_TOKEN_PREFIX + ";broken",
        "sbatch /malformed.sbatch",
    )
    snapshot = control.SchedulerSnapshot((malformed,), 50.0)

    with pytest.raises(control.SchedulerAmbiguity, match="malformed live"):
        control.start_controller_drill(state_dir, scheduler=snapshot, now=50.0)
    with pytest.raises(control.SchedulerAmbiguity, match="live controller-drill jobs"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: snapshot,
            now=50.0,
        )
    status = control.controller_drill_status(state_dir, snapshot=snapshot, now=50.0)
    assert status["healthy"] is False
    assert any("malformed live" in error for error in status["errors"])


def test_final_resume_commit_rejoins_and_rejects_late_malformed_drill_job(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    jobs = []
    reads = 0
    identifiers = iter(("920", "921"))

    def scheduler():
        nonlocal reads
        reads += 1
        visible = list(jobs)
        if reads >= 4:
            visible.append(
                control.SchedulerJob(
                    "999",
                    "late-malformed-drill",
                    "RUNNING",
                    control.DRILL_TOKEN_PREFIX + ";broken",
                    "sbatch /late.sbatch",
                )
            )
        return control.SchedulerSnapshot(tuple(visible), 50.0)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    with pytest.raises(control.SchedulerAmbiguity, match="live controller-drill jobs"):
        control.resume_control(
            state_dir,
            scheduler_reader=scheduler,
            submit_runner=submit,
            now=50.0,
        )

    current = control.load_control(state_dir)
    assert reads >= 4
    assert current["desired_state"] == "resuming"
    assert current["resume_intent"]["state"] == "submitting_controllers"


def test_resuming_retry_rejoins_namespace_before_another_submission(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)

    with pytest.raises(control.ControlError, match="sbatch rejected"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 50.0),
            submit_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, "", "temporary rejection"
            ),
            now=50.0,
        )
    assert control.load_control(state_dir)["desired_state"] == "resuming"

    malformed = control.SchedulerJob(
        "998",
        "retry-malformed-drill",
        "RUNNING",
        control.DRILL_TOKEN_PREFIX + ";broken",
        "sbatch /retry.sbatch",
    )
    submissions = []
    with pytest.raises(control.SchedulerAmbiguity, match="live controller-drill jobs"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((malformed,), 51.0),
            submit_runner=lambda argv: submissions.append(list(argv)),
            now=51.0,
        )
    assert submissions == []


def test_resume_rejects_live_cell_namespace_until_worker_exit(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    live_cell = control.SchedulerJob(
        "930_7",
        "asys-dispatch-schema5-batch",
        "COMPLETING",
        control.CELL_INTENT_PREFIX + "batch-seven",
        "sbatch /cells.sbatch",
    )
    submissions = []

    with pytest.raises(control.SchedulerAmbiguity, match="cell jobs have not drained"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((live_cell,), 50.0),
            submit_runner=lambda argv: submissions.append(list(argv)),
            now=50.0,
        )

    assert submissions == []
    assert control.load_control(state_dir)["desired_state"] == "paused"


def test_later_pause_resume_revalidates_static_drill_proof_without_live_baseline(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, _jobs = resume_ready(state_dir, now=50.0)
    first_proof = running["resume_intent"]["controller_drill_marker_sha256"]
    paused = control.pause_control(
        state_dir,
        drain=True,
        scheduler=control.SchedulerSnapshot((), 60.0),
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        now=60.0,
    )
    assert paused["drain_requested"] is True
    assert paused["drain_intent"]["state"] == "complete"
    current = control.load_control(state_dir)
    for role_state in current["controllers"].values():
        role_state["active"] = None
        role_state["successor"] = None
        role_state["submission_intent"] = None
        role_state["heartbeat"] = None
    control._save_control(state_dir, current, now=60.0)
    jobs = []
    identifiers = iter(("910", "911"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(jobs), 61.0)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    resumed = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=submit,
        now=61.0,
    )

    assert resumed["desired_state"] == "running"
    assert resumed["rollout_generation"] == 2
    assert resumed["resume_intent"]["controller_drill_marker_sha256"] == first_proof


def test_later_pause_resume_rejects_deleted_historical_drill_marker(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, _jobs = resume_ready(state_dir, now=50.0)
    current = control.load_control(state_dir)
    current["desired_state"] = "paused"
    current["drain_requested"] = False
    current["drain_intent"] = None
    for role_state in current["controllers"].values():
        role_state["active"] = None
        role_state["successor"] = None
        role_state["submission_intent"] = None
        role_state["heartbeat"] = None
    control._save_control(state_dir, current, now=60.0)
    (state_dir / control.DRILL_COMPLETE_FILENAME).unlink()

    with pytest.raises(control.ReadinessError, match="marker is missing"):
        control.resume_control(state_dir, now=61.0)

    assert running["resume_intent"]["controller_drill_marker_sha256"]
    assert control.load_control(state_dir)["desired_state"] == "paused"


def test_later_pause_resume_rejects_incomplete_drain(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    current = control.load_control(state_dir)
    current["desired_state"] = "paused"
    current["drain_requested"] = True
    current["drain_intent"] = {"state": "signaling"}
    for role_state in current["controllers"].values():
        role_state["active"] = None
        role_state["successor"] = None
        role_state["submission_intent"] = None
        role_state["heartbeat"] = None
    control._save_control(state_dir, current, now=60.0)

    with pytest.raises(control.ReadinessError, match="incomplete pause drain"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 61.0),
            now=61.0,
        )

    assert control.load_control(state_dir)["desired_state"] == "paused"


def test_crashed_later_resume_rechecks_historical_marker_hash(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, _jobs = resume_ready(state_dir, now=50.0)
    marker_sha = running["resume_intent"]["controller_drill_marker_sha256"]
    current = control.load_control(state_dir)
    generation_two_attestation = control.ensure_runtime_integrity_attestation(
        state_dir, current, generation=2, force_full=True
    )
    generation_two_snapshot = control.ensure_snapshot_integrity_attestation(
        state_dir,
        current,
        generation=2,
        validation_context=control.validate_readiness(
            current,
            state_dir=state_dir,
            verify_files=True,
            snapshot_full=True,
        ),
    )
    current["desired_state"] = "resuming"
    current["rollout_generation"] = 2
    current[control.RUNTIME_ATTESTATION_STATE_KEY] = generation_two_attestation
    current[control.SNAPSHOT_ATTESTATION_STATE_KEY] = generation_two_snapshot
    current["drain_requested"] = False
    current["drain_intent"] = None
    control._reset_admission_ramp(
        state_dir,
        current,
        reason="synthetic_crashed_resume",
        now=60.0,
        ceiling=24,
        rollout_generation=2,
        clear_last_observation=True,
    )
    current["resume_intent"] = {
        "resume_id": "crashed-second-resume",
        "state": "submitting_controllers",
        "rollout_generation": 2,
        "created_at": control.utc_timestamp(60.0),
        "created_timestamp": 60.0,
        "controller_job_ids": {},
        "controller_drill_marker_sha256": marker_sha,
    }
    for role_state in current["controllers"].values():
        role_state["active"] = None
        role_state["successor"] = None
        role_state["submission_intent"] = None
        role_state["heartbeat"] = None
    control._save_control(state_dir, current, now=60.0)
    (state_dir / control.DRILL_COMPLETE_FILENAME).unlink()

    with pytest.raises(control.ReadinessError, match="marker is missing"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 61.0),
            now=61.0,
        )

    assert control.load_control(state_dir)["desired_state"] == "resuming"


def test_final_resume_commit_rechecks_marker_hash_under_lock(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    jobs = []
    identifiers = iter(("940", "941"))
    submissions = 0

    def scheduler():
        return control.SchedulerSnapshot(tuple(jobs), 50.0)

    def submit(argv):
        nonlocal submissions
        submissions += 1
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        if submissions == 2:
            marker_path = state_dir / control.DRILL_COMPLETE_FILENAME
            marker_path.write_text(
                marker_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    with pytest.raises(control.ReadinessError, match="marker drifted"):
        control.resume_control(
            state_dir,
            scheduler_reader=scheduler,
            submit_runner=submit,
            now=50.0,
        )

    assert submissions == 2
    assert control.load_control(state_dir)["desired_state"] == "resuming"


def test_resuming_retry_revalidates_marker_bound_drill_state(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    with pytest.raises(control.ControlError, match="sbatch rejected"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 50.0),
            submit_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, "", "temporary rejection"
            ),
            now=50.0,
        )

    drill_path = state_dir / control.DRILL_STATE_FILENAME
    drill = json.loads(drill_path.read_text(encoding="utf-8"))
    drill["roles"]["dispatcher"]["recovery"]["recovery_seconds"] += 0.25
    control._atomic_write_json(drill_path, drill)
    submissions = []

    with pytest.raises(control.ReadinessError, match="state drifted"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 51.0),
            submit_runner=lambda argv: submissions.append(list(argv)),
            now=51.0,
        )
    assert submissions == []


def test_drill_finish_reconciles_and_cancels_late_afterany_successor(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )
    state = control.load_drill_state(state_dir)
    late_id = state["roles"]["fleet_supervisor"]["successor"]["job_id"]
    reads = 0
    cancelled = []

    def scheduler():
        nonlocal reads
        reads += 1
        visible = [job for job in jobs if not (reads == 1 and job.job_id == late_id)]
        return control.SchedulerSnapshot(tuple(visible), 80.0)

    def cancel(argv):
        cancelled.append(list(argv))
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    marker = control.finish_controller_drill(
        state_dir,
        scheduler_reader=scheduler,
        cancel_runner=cancel,
        timeout_seconds=1.0,
        poll_seconds=0.0,
        now=80.0,
    )

    assert marker["passed"] is True
    assert ["scancel", late_id] in cancelled
    assert late_id in control.load_drill_state(state_dir)["cleanup_job_ids"]


def test_drill_finish_recovers_sealed_reconciliation_before_completed_state(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )

    def cancel(argv):
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    real_save = control._save_drill_state
    crashed = False

    def crash_before_completed_state(path, state, *, now):
        nonlocal crashed
        if state.get("phase") == "completed" and not crashed:
            crashed = True
            raise RuntimeError("simulated death after reconciliation seal")
        return real_save(path, state, now=now)

    monkeypatch.setattr(control, "_save_drill_state", crash_before_completed_state)
    with pytest.raises(RuntimeError, match="after reconciliation seal"):
        control.finish_controller_drill(
            state_dir,
            scheduler_reader=lambda: snapshot(80.0),
            cancel_runner=cancel,
            timeout_seconds=1.0,
            poll_seconds=0.0,
            now=80.0,
        )
    sealed = control._drill_reconciliation_path(
        state_dir, control.load_drill_state(state_dir)["drill_id"]
    )
    assert sealed.is_file()
    assert control.load_drill_state(state_dir)["phase"] == "stopping"
    monkeypatch.setattr(control, "_save_drill_state", real_save)

    marker = control.finish_controller_drill(
        state_dir,
        scheduler_reader=lambda: snapshot(81.0),
        cancel_runner=cancel,
        timeout_seconds=1.0,
        poll_seconds=0.0,
        now=81.0,
    )
    assert marker["passed"] is True
    recovered = control.load_drill_state(state_dir)
    completed = [
        event for event in recovered["transition_history"] if event["event"] == "completed"
    ]
    assert len(completed) == 1
    assert completed[0]["timestamp"] == recovered["completed_timestamp"]


def test_drill_finish_recovers_completed_state_before_marker(tmp_path, monkeypatch):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )

    def cancel(argv):
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    real_write = control._atomic_write_json
    crashed = False

    def crash_before_marker(path, payload):
        nonlocal crashed
        if Path(path).name == control.DRILL_COMPLETE_FILENAME and not crashed:
            crashed = True
            raise RuntimeError("simulated death before marker")
        return real_write(path, payload)

    monkeypatch.setattr(control, "_atomic_write_json", crash_before_marker)
    with pytest.raises(RuntimeError, match="before marker"):
        control.finish_controller_drill(
            state_dir,
            scheduler_reader=lambda: snapshot(80.0),
            cancel_runner=cancel,
            timeout_seconds=1.0,
            poll_seconds=0.0,
            now=80.0,
        )
    assert control.load_drill_state(state_dir)["phase"] == "completed"
    assert not (state_dir / control.DRILL_COMPLETE_FILENAME).exists()
    monkeypatch.setattr(control, "_atomic_write_json", real_write)

    marker = control.finish_controller_drill(
        state_dir,
        scheduler_reader=lambda: snapshot(81.0),
        cancel_runner=cancel,
        timeout_seconds=1.0,
        poll_seconds=0.0,
        now=81.0,
    )
    assert marker["passed"] is True


def test_drill_finish_waits_for_inflight_sbatch_before_stopping_and_sealing(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )
    with control.drill_lock(state_dir):
        state = control.load_drill_state(state_dir)
        superseded = state["roles"]["dispatcher"]["successor"]
        jobs[:] = [job for job in jobs if job.job_id != superseded["job_id"]]
        state["roles"]["dispatcher"]["successor"] = None
        control._save_drill_state(state_dir, state, now=75.0)

    sbatch_entered = threading.Event()
    release_sbatch = threading.Event()
    submit_errors = []
    finish_errors = []
    finish_result = []

    def blocked_submit(argv):
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        sbatch_entered.set()
        assert release_sbatch.wait(timeout=5.0)
        jobs.append(
            control.SchedulerJob(
                "990",
                "drill",
                "PENDING",
                token,
                f"sbatch {argv[-1]}",
                dependency=next(
                    arg.split("=", 1)[1]
                    for arg in argv
                    if arg.startswith("--dependency=")
                ),
            )
        )
        return subprocess.CompletedProcess(argv, 0, "990\n", "")

    def run_submit():
        try:
            active = control.load_drill_state(state_dir)["roles"]["dispatcher"][
                "active"
            ]
            control.submit_drill_intent(
                state_dir,
                role="dispatcher",
                target="successor",
                dependency_job_id=active["job_id"],
                scheduler=snapshot(76.0),
                submit_runner=blocked_submit,
                now=76.0,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            submit_errors.append(exc)

    def cancel(argv):
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    def run_finish():
        try:
            finish_result.append(
                control.finish_controller_drill(
                    state_dir,
                    scheduler_reader=lambda: snapshot(80.0),
                    cancel_runner=cancel,
                    timeout_seconds=2.0,
                    poll_seconds=0.0,
                    now=80.0,
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            finish_errors.append(exc)

    submitting = threading.Thread(target=run_submit)
    finishing = threading.Thread(target=run_finish)
    submitting.start()
    assert sbatch_entered.wait(timeout=5.0)
    finishing.start()
    time.sleep(0.05)
    assert control.load_drill_state(state_dir)["phase"] == "running"
    assert not (state_dir / control.DRILL_COMPLETE_FILENAME).exists()
    release_sbatch.set()
    submitting.join(timeout=5.0)
    finishing.join(timeout=5.0)

    assert not submitting.is_alive()
    assert not finishing.is_alive()
    assert submit_errors == []
    assert finish_errors == []
    assert finish_result[0]["passed"] is True
    assert "990" in control.load_drill_state(state_dir)["cleanup_job_ids"]


def test_drill_finish_rejects_foreign_live_drill_namespace(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )
    foreign_token = control.drill_job_token(
        "foreign-drill", "dispatcher", 1, "foreign-intent"
    )
    jobs.append(
        control.SchedulerJob(
            "991", "foreign", "PENDING", foreign_token, "sbatch /foreign.sbatch"
        )
    )

    with pytest.raises(control.SchedulerAmbiguity, match="foreign controller-drill"):
        control.finish_controller_drill(
            state_dir,
            scheduler_reader=lambda: snapshot(80.0),
            cancel_runner=lambda _argv: subprocess.CompletedProcess([], 0, "", ""),
            timeout_seconds=1.0,
            poll_seconds=0.0,
            now=80.0,
        )
    assert not (state_dir / control.DRILL_COMPLETE_FILENAME).exists()


def test_completed_drill_marker_rejects_journal_suffix_not_in_state(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    with (state_dir / control.DRILL_JOURNAL_FILENAME).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps({"unexpected": "suffix"}) + "\n")

    with pytest.raises(control.ReadinessError, match="exactly match its journal"):
        control.validate_controller_drill_marker(
            state_dir,
            control.load_control(state_dir),
        )


def test_completed_drill_marker_exactly_binds_state_recoveries(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    marker_path = state_dir / control.DRILL_COMPLETE_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["roles"]["dispatcher"]["successor_job_id"] = "999"
    control._atomic_write_json(marker_path, marker)

    with pytest.raises(control.ReadinessError, match="does not exactly match state"):
        control.validate_controller_drill_marker(
            state_dir,
            control.load_control(state_dir),
        )


def test_drill_marker_keeps_sealed_reconciliation_across_new_clean_reconcile(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    marker = complete_drill_marker(state_dir)
    sealed_path = Path(marker["final_reconciliation_path"])
    sealed_bytes = sealed_path.read_bytes()

    assert reconcile_clean(state_dir, now=55.0)["passed"] is True
    assert (
        _sha(state_dir / control.RECONCILIATION_FILENAME)
        != marker["final_reconciliation_sha256"]
    )
    assert sealed_path.read_bytes() == sealed_bytes
    assert sealed_path.stat().st_mode & 0o222 == 0
    assert (
        control.validate_controller_drill_marker(
            state_dir,
            control.load_control(state_dir),
        )
        == marker
    )


def test_async_snapshot_refresh_heartbeats_while_cross_node_scan_blocks(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    refresh_calls = []
    heartbeats = []

    def slow_refresh(state_dir, current, *, now=None):
        refresh_calls.append((state_dir, current, now))
        started.set()
        assert release.wait(timeout=2.0)
        return {"cached": False, "record": {"sequence": 2}}

    monkeypatch.setattr(control, "refresh_snapshot_integrity_lease", slow_refresh)
    monkeypatch.setattr(
        control,
        "heartbeat_controller",
        lambda state_dir, **kwargs: heartbeats.append((state_dir, kwargs)) or kwargs,
    )
    task = control._start_snapshot_lease_refresh(
        tmp_path, {"rollout_generation": 7}
    )
    assert started.wait(timeout=1.0)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        result = control._await_snapshot_lease_refresh_with_heartbeats(
            task,
            state_dir=tmp_path,
            role="dispatcher",
            generation=7,
            intent_token="intent",
            job_id="700",
            heartbeat_interval=0.01,
        )
    finally:
        release.set()
        timer.cancel()
        timer.join(timeout=1.0)
        task.thread.join(timeout=1.0)

    assert result == {"cached": False, "record": {"sequence": 2}}
    assert len(refresh_calls) == 1
    assert refresh_calls[0][2] is None
    assert len(heartbeats) >= 3
    assert all(item[1]["job_id"] == "700" for item in heartbeats)


def test_async_snapshot_refresh_surfaces_worker_failure(tmp_path, monkeypatch):
    calls = []

    def failed_refresh(*_args, **_kwargs):
        calls.append("refresh")
        raise control.ReadinessError("metadata scan failed")

    monkeypatch.setattr(control, "refresh_snapshot_integrity_lease", failed_refresh)
    monkeypatch.setattr(
        control,
        "heartbeat_controller",
        lambda *_args, **_kwargs: calls.append("heartbeat"),
    )
    task = control._start_snapshot_lease_refresh(
        tmp_path, {"rollout_generation": 8}
    )
    with pytest.raises(control.ReadinessError, match="metadata scan failed"):
        control._await_snapshot_lease_refresh_with_heartbeats(
            task,
            state_dir=tmp_path,
            role="fleet_supervisor",
            generation=8,
            intent_token="intent",
            job_id="701",
            heartbeat_interval=0.01,
        )
    task.thread.join(timeout=1.0)
    assert calls.count("refresh") == 1
    assert calls.count("heartbeat") >= 1


def test_async_runtime_refresh_heartbeats_while_metadata_scan_blocks(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    refresh_calls = []
    heartbeats = []

    def slow_refresh(state_dir, current, *, now=None):
        refresh_calls.append((state_dir, current, now))
        started.set()
        assert release.wait(timeout=2.0)
        return {"cached": False, "record": {"sequence": 3}}

    monkeypatch.setattr(control, "refresh_runtime_integrity_lease", slow_refresh)
    monkeypatch.setattr(
        control,
        "heartbeat_controller",
        lambda state_dir, **kwargs: heartbeats.append((state_dir, kwargs)) or kwargs,
    )
    task = control._start_runtime_lease_refresh(
        tmp_path, {"rollout_generation": 9}
    )
    assert started.wait(timeout=1.0)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        result = control._await_runtime_lease_refresh_with_heartbeats(
            task,
            state_dir=tmp_path,
            role="dispatcher",
            generation=9,
            intent_token="intent",
            job_id="702",
            heartbeat_interval=0.01,
        )
    finally:
        release.set()
        timer.cancel()
        timer.join(timeout=1.0)
        task.thread.join(timeout=1.0)

    assert result == {"cached": False, "record": {"sequence": 3}}
    # Many heartbeat timeouts must never create a second local renewal.
    assert len(refresh_calls) == 1
    assert refresh_calls[0][2] is None
    assert len(heartbeats) >= 3
    assert all(item[1]["job_id"] == "702" for item in heartbeats)


def test_async_runtime_refresh_surfaces_worker_failure(tmp_path, monkeypatch):
    calls = []

    def failed_refresh(*_args, **_kwargs):
        calls.append("refresh")
        raise control.ImmutablePinError("runtime metadata scan failed")

    monkeypatch.setattr(control, "refresh_runtime_integrity_lease", failed_refresh)
    monkeypatch.setattr(
        control,
        "heartbeat_controller",
        lambda *_args, **_kwargs: calls.append("heartbeat"),
    )
    task = control._start_runtime_lease_refresh(
        tmp_path, {"rollout_generation": 10}
    )
    with pytest.raises(control.ImmutablePinError, match="runtime metadata scan failed"):
        control._await_runtime_lease_refresh_with_heartbeats(
            task,
            state_dir=tmp_path,
            role="fleet_supervisor",
            generation=10,
            intent_token="intent",
            job_id="703",
            heartbeat_interval=0.01,
        )
    task.thread.join(timeout=1.0)
    assert calls.count("refresh") == 1
    assert calls.count("heartbeat") >= 1


def test_drill_finish_rejects_any_production_run_mutation_before_marker(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir)
    recover_drill_role(state_dir, jobs, snapshot, submit, role="dispatcher", now=60.0)
    recover_drill_role(
        state_dir, jobs, snapshot, submit, role="fleet_supervisor", now=70.0
    )
    run_root = Path(control.load_control(state_dir)["immutable"]["runs"][0]["run_root"])
    (run_root / "unexpected-result.jsonl").write_text("{}\n", encoding="utf-8")

    def cancel(argv):
        jobs[:] = [job for job in jobs if job.job_id != argv[-1]]
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(control.ControlError, match="changed production"):
        control.finish_controller_drill(
            state_dir,
            scheduler_reader=lambda: snapshot(80.0),
            cancel_runner=cancel,
            timeout_seconds=1.0,
            poll_seconds=0.0,
            now=80.0,
        )
    assert not (state_dir / control.DRILL_COMPLETE_FILENAME).exists()


def test_drill_submission_crash_adopts_unique_scheduler_token(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs = []

    def crash_after_accept(argv):
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(
            control.SchedulerJob("880", "drill", "PENDING", token, f"sbatch {argv[-1]}")
        )
        raise RuntimeError("simulated caller death after scheduler acceptance")

    with pytest.raises(RuntimeError, match="caller death"):
        control.start_controller_drill(
            state_dir,
            scheduler=control.SchedulerSnapshot((), 50.0),
            submit_runner=crash_after_accept,
            now=50.0,
        )
    adopted = control.submit_drill_intent(
        state_dir,
        role="dispatcher",
        target="active",
        dependency_job_id=None,
        scheduler=control.SchedulerSnapshot(tuple(jobs), 51.0),
        submit_runner=lambda _argv: pytest.fail(
            "unique accepted intent must be adopted"
        ),
        now=51.0,
    )
    assert adopted["job_id"] == "880"
    assert adopted["adopted_from_scheduler"] is True


def test_resume_rejects_drill_marker_or_state_drift(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    marker_path = state_dir / control.DRILL_COMPLETE_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["roles"]["dispatcher"]["recovery_seconds"] = 901.0
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(control.ReadinessError, match="recovery is invalid"):
        control.resume_control(state_dir, now=50.0)
