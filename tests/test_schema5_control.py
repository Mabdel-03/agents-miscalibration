from __future__ import annotations

import copy
import hashlib
import json
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from slurm import schema5_control as control


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
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
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
        "release_id": "sweep-recovery-schema5-v1",
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
    harness_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "sweep-recovery-schema5-v1",
                "role": "harness",
                "prefix": str(harness),
                "sealed_read_only": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    serving_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "sweep-recovery-schema5-v1",
                "role": "serving",
                "prefix": str(serving),
                "sealed_read_only": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    harness_hash = _sha(harness_manifest_path)
    serving_hash = _sha(serving_manifest_path)
    source_tree_sha256 = control.sha256_tree(release)
    pins = {
        "release_id": "sweep-recovery-schema5-v1",
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
                        "directory_inventory_sha256": "b" * 64,
                    },
                    "serving": {
                        "prefix": str(serving),
                        "manifest_path": str(serving_manifest_path),
                        "manifest_sha256": serving_hash,
                        "directory_inventory_sha256": "c" * 64,
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
            "accepted_benchmark_contracts_sha256": run[
                "benchmark_contract_sha256"
            ],
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
            "model_contract_sha256": control_state["immutable"]["model_contract_sha256"],
            "fleet_contract_sha256": control_state["immutable"]["fleet_contract_sha256"],
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
                                        "release_id": current["immutable"]["release_id"],
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
                            "server_pool_root": current["immutable"]["server_pool_root"],
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
        control.attest_gate(state_dir, gate=gate, evidence_path=evidence, now=20 + index)


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


def resume_ready(state_dir: Path, *, now: float = 50.0) -> tuple[dict, tuple]:
    jobs = []
    identifiers = iter(("700", "701", "702", "703", "704", "705"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(jobs), now)

    def submit(argv):
        job_id = next(identifiers)
        token = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment="))
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    resumed = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=submit,
        now=now,
    )
    return resumed, tuple(jobs)


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
    visible_jobs = []
    submitted = []
    identifiers = iter(("810", "811"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(visible_jobs), 50.0)

    def submit(argv):
        job_id = next(identifiers)
        token = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment="))
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
    running = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=lambda _argv: pytest.fail("visible jobs must be adopted"),
        now=52.0,
    )
    assert running["desired_state"] == "running"
    assert running["resume_intent"]["state"] == "complete"
    assert running["resume_intent"]["scheduler_visible_job_ids"] == {
        "dispatcher": "810",
        "fleet_supervisor": "811",
    }
    assert [
        transition["event"] for transition in running["transition_history"]
    ].count("resumed") == 1


def test_resume_rejects_incomplete_scheduler_truth_before_any_submission(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
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
    assert current["desired_state"] == "resuming"
    assert all(
        current["controllers"][role]["active"] is None
        for role in control.ROLE_NAMES
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
    with pytest.raises(control.ReadinessError, match="does not match its typed wrapper"):
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
        control.validate_readiness(current, verify_files=True, now=30.0)


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
    assert pins["fleet_supervisor_command"] == control.expected_fleet_supervisor_command(
        pins
    )
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

    assert control.main(
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
    ) == 0
    assert canonical_pool.is_dir()
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared == expected
    assert output.stat().st_mode & 0o222 == 0
    checksum = output.with_name(output.name + ".sha256")
    assert checksum.read_text(encoding="utf-8") == f"{_sha(output)}  {output.name}\n"
    assert checksum.stat().st_mode & 0o222 == 0

    # Re-running the producer proves the exact same bytes instead of replacing authority.
    assert control.main(
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
    ) == 0
    assert control.main(
        [
            "--state-dir",
            str(state_dir),
            "init",
            "--pins-json",
            str(output),
        ]
    ) == 0
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
    outputs = {
        "squeue": subprocess.CompletedProcess(
            [], 0, "123|job|RUNNING|comment|cmd\n", ""
        ),
        "sacct": subprocess.CompletedProcess(
            [], 0, "123|job|FAILED|comment|submit\n123.batch|batch|COMPLETED||\n", ""
        ),
    }

    def runner(argv):
        return outputs[argv[0]]

    snapshot = control.query_scheduler(runner=runner, user="u", now=100.0)
    assert len(snapshot.jobs) == 1
    assert snapshot.jobs[0].state == "RUNNING"
    assert snapshot.active_job_ids == {"123"}


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
    assert "--generation 7 --intent-token \"abc123\"" in text
    assert 'export ASYS_RELEASE_ID="sweep-recovery-schema5-v1"' in text
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
        "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV"
        in text
    )
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "export PYTHONSAFEPATH=1" in text
    assert 'export HF_HOME="' + initialized["immutable"]["hf_home"] + '"' in text
    assert (
        'export ASYS_RESULTS_ROOT="'
        + initialized["immutable"]["results_root"]
        + '"'
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
    assert control.render_generation_sbatch(
        state_dir,
        initialized,
        role="dispatcher",
        generation=7,
        intent_token="abc123",
    ) == path

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
        'export ASYS_RESULTS_ROOT="'
        + initialized["immutable"]["results_root"]
        + '"'
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
        "ASYS_RELEASE_ID": "sweep-recovery-schema5-v1",
        "ASYS_MODEL_CONTRACT_SHA256": initialized["immutable"][
            "model_contract_sha256"
        ],
        "ASYS_FLEET_CONTRACT_SHA256": initialized["immutable"][
            "fleet_contract_sha256"
        ],
        "ASYS_HARNESS_ENVIRONMENT_SHA256": initialized["immutable"][
            "harness_environment_sha256"
        ],
        "ASYS_SERVING_ENVIRONMENT_SHA256": initialized["immutable"][
            "serving_environment_sha256"
        ],
        "ASYS_ROLLOUT_GENERATION": "1",
        "ASYS_IMMUTABLE_PINS_SHA256": initialized["immutable_sha256"],
        "ASYS_ARTIFACT_POLICY_SHA256": expected_policy,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_production_batch_validator_hard_enforces_slurm_contract(tmp_path):
    _, initialized = initialize(tmp_path)
    execution = control.production_cell_execution(initialized)
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
exec {shlex.quote(execution["python"])} -u {shlex.quote(execution["dispatcher_script"])} run-task \
  --expected-release-root {shlex.quote(execution["release_worktree"])} \
  --expected-harness-prefix {shlex.quote(execution["harness_prefix"])}
"""
    control.validate_production_batch_sbatch(
        initialized, payload=payload, task_count=24
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
        (
            control.SchedulerJob(
                "444", "asys-s5-fleet", "PENDING", intent["job_token"]
            ),
        ),
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
    resume_ready(state_dir, now=50.0)
    ids = iter(("800", "801"))
    first_calls = []

    def submit(argv):
        first_calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, next(ids) + "\n", "")

    result = control.repair_chains(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 240.0),
        submit_runner=submit,
        now=240.0,
    )
    assert {item["role"] for item in result["submitted"]} == set(control.ROLE_NAMES)
    persisted = control.load_control(state_dir)
    live_jobs = tuple(
        control.SchedulerJob(
            record["job_id"], role, "PENDING", record["job_token"]
        )
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

    command = control.schema5_monitor_command(
        state_dir, initialized, cadence="health"
    )
    immutable = initialized["immutable"]
    assert command[:3] == [
        str(Path(immutable["harness_environment_prefix"]) / "bin" / "python"),
        "-u",
        str(Path(immutable["release_worktree"]) / "scripts" / "schema5_monitor.py"),
    ]
    assert command[command.index("--results-root") + 1] == immutable["results_root"]
    assert command[command.index("--state-dir") + 1] == str(state_dir.resolve())
    assert command[command.index("--config") + 1] == str(
        Path(immutable["release_worktree"])
        / "configs"
        / "schema5_monitoring.v1.json"
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
    assert [row["attempt_id"] for row in recovered] == [
        successor_attempt["attempt_id"]
    ]
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
                "intents": {"batch1": {"job_id": "900"}},
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
            calls.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=60.0,
    )
    assert calls == [["scancel", "--signal=USR1", "900_3"]]
    assert paused["drain_intent"]["state"] == "complete"
    assert paused["drain_intent"]["cell_task_ids"] == ["900_3"]
    assert paused["drain_intent"]["results"]["cell_usr1:900_3"]["returncode"] == 0


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
                "intents": {"batch1": {"job_id": "900"}},
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
            calls.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "", "")
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
