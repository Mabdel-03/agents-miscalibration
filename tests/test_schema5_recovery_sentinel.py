"""Marker-last, classification, and delivery tests for the r2 failure sentinel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Sequence

import pytest

from scripts import build_schema5_readiness as readiness
from scripts import schema5_recovery_sentinel as sentinel


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert 0 <= seconds <= 60
        self.value += seconds


def _verified(tmp_path: Path) -> dict[str, Any]:
    manifest_path = tmp_path / "chain.json"
    receipt_path = tmp_path / "receipt.json"
    jobs = [
        {
            "name": "first",
            "dependencies": [],
            "dependency_type": "afterok",
            "job_name": "asys-first",
        },
        {
            "name": "second",
            "dependencies": ["first"],
            "dependency_type": "afterok",
            "job_name": "asys-second",
        },
        {
            "name": "failure_sentinel",
            "dependencies": ["first", "second"],
            "dependency_type": "afterany",
            "job_name": "asys-sentinel",
        },
    ]
    receipt_jobs = []
    for index, row in enumerate(jobs, start=101):
        receipt_jobs.append(
            {
                "name": row["name"],
                "job_id": str(index),
                "comment": f"comment-{row['name']}",
            }
        )
    manifest = {
        "protocol": sentinel.R11_PROTOCOL,
        "chain_id": "a" * 64,
        "slurm_user": "tester",
        "jobs": jobs,
    }
    receipt = {"receipt_id": "b" * 64, "jobs": receipt_jobs}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    receipt_path.chmod(0o444)
    return {
        "schema_version": 1,
        "chain_protocol": sentinel.R11_PROTOCOL,
        "renderer": "render_schema5_recovery_chain_v12",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sentinel._sha256(manifest_path),
        "manifest": manifest,
        "chain_report": {"passed": True},
        "submission_receipt_path": str(receipt_path),
        "submission_receipt_sha256": sentinel._sha256(receipt_path),
        "submission_receipt": receipt,
    }


def _failfast_verified(tmp_path: Path) -> dict[str, Any]:
    manifest_path = tmp_path / "failfast-chain.json"
    receipt_path = tmp_path / "failfast-receipt.json"
    production: list[dict[str, Any]] = []
    for index, name in enumerate(sentinel.PRODUCTION_STAGE_NAMES):
        production.append(
            {
                "name": name,
                "dependencies": (
                    [] if index == 0 else [sentinel.PRODUCTION_STAGE_NAMES[index - 1]]
                ),
                "dependency_type": "afterok",
                "job_name": f"asys-{name}",
            }
        )
    observers = [
        {
            "name": f"{sentinel.STAGE_SENTINEL_PREFIX}{stage}",
            "dependencies": [stage],
            "dependency_type": "afterany",
            "job_name": f"asys-alert-{index:02d}",
        }
        for index, stage in enumerate(sentinel.PRODUCTION_STAGE_NAMES)
    ]
    jobs = [
        *production,
        *observers,
        {
            "name": "failure_sentinel",
            "dependencies": [
                *sentinel.PRODUCTION_STAGE_NAMES,
                *sentinel.STAGE_SENTINEL_NAMES,
            ],
            "dependency_type": "afterany",
            "job_name": "asys-sentinel",
        },
    ]
    receipt_jobs = [
        {
            "name": row["name"],
            "job_id": str(10_000 + index),
            "comment": f"comment-{row['name']}",
        }
        for index, row in enumerate(jobs)
    ]
    manifest = {
        "protocol": sentinel.R11_PROTOCOL,
        "chain_id": "c" * 64,
        "slurm_user": "tester",
        "jobs": jobs,
    }
    receipt = {"receipt_id": "d" * 64, "jobs": receipt_jobs}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    receipt_path.chmod(0o444)
    return {
        "schema_version": 1,
        "chain_protocol": sentinel.R11_PROTOCOL,
        "renderer": "render_schema5_recovery_chain_v12",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sentinel._sha256(manifest_path),
        "manifest": manifest,
        "chain_report": {"passed": True},
        "submission_receipt_path": str(receipt_path),
        "submission_receipt_sha256": sentinel._sha256(receipt_path),
        "submission_receipt": receipt,
    }


class FailFastScheduler:
    def __init__(
        self,
        verified: dict[str, Any],
        *,
        target_stage: str,
        target_state: str,
        target_exit_code: str,
        target_reason: str = "NonZeroExitCode",
    ) -> None:
        self.verified = verified
        self.target_stage = target_stage
        self.target_state = target_state
        self.target_exit_code = target_exit_code
        self.target_reason = target_reason
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        observer_name = (
            f"{sentinel.STAGE_SENTINEL_PREFIX}{self.target_stage}"
        )
        if args[0] == "squeue":
            output = ""
            for manifest_row, receipt_row in zip(
                self.verified["manifest"]["jobs"],
                self.verified["submission_receipt"]["jobs"],
                strict=True,
            ):
                if manifest_row["name"] == self.target_stage:
                    continue
                state = (
                    "RUNNING"
                    if manifest_row["name"] == observer_name
                    else "PENDING"
                )
                reason = "None" if state == "RUNNING" else "Dependency"
                output += (
                    f"{receipt_row['job_id']}|{state}|{reason}|"
                    f"{receipt_row['comment']}|{manifest_row['job_name']}\n"
                )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] == "sacct":
            output = _sacct_row(
                self.verified,
                self.target_stage,
                self.target_state,
                exit_code=self.target_exit_code,
                reason=self.target_reason,
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        raise AssertionError(args)


def _capacity_verified(tmp_path: Path) -> dict[str, Any]:
    verified = _verified(tmp_path)
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    manifest["jobs"][0].update(
        {"name": "fleet_readiness", "job_name": "asys-fleet-readiness"}
    )
    manifest["jobs"][1]["dependencies"] = ["fleet_readiness"]
    manifest["jobs"][2]["dependencies"] = ["fleet_readiness", "second"]
    receipt["jobs"][0].update(
        {
            "name": "fleet_readiness",
            "comment": "comment-fleet-readiness",
        }
    )
    authority_root = tmp_path / "capacity-authority"
    authority_root.mkdir()
    pool_root = tmp_path / "results" / "server_pools" / "schema5-v1"
    pool_root.mkdir(parents=True)
    model_contract_path = authority_root / "model_contracts.v1.json"
    fleet_contract_path = authority_root / "schema5_fleet.v1.json"
    replica_specs: list[dict[str, Any]] = []
    models: dict[str, dict[str, str]] = {}
    profiles: list[dict[str, Any]] = []
    for index in range(22):
        model_size = f"model-size-{index:02d}"
        replica_id = f"replica-{index:02d}"
        serving_profile = f"profile-{index:02d}"
        served_model = f"model-{index:02d}"
        model_revision = f"model-revision-{index:02d}"
        tokenizer_id = f"tokenizer-{index:02d}"
        tokenizer_revision = f"tokenizer-revision-{index:02d}"
        allocated_gpus = 2 if index < 2 else 1
        scheduler_job_name = f"asys-fleet-{replica_id}"
        models[model_size] = {
            "hf_id": f"hf/{model_size}",
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        }
        profiles.append(
            {
                "model_size": model_size,
                "hf_id": f"hf/{model_size}",
                "model_revision": model_revision,
                "tokenizer_id": tokenizer_id,
                "tokenizer_revision": tokenizer_revision,
                "serving_profile": serving_profile,
                "served_model_name": served_model,
                "effective_context_limit": 32_768,
                "tensor_parallel_size": allocated_gpus,
                "gpus_per_replica": allocated_gpus,
                "replicas": [
                    {
                        "pool_id": "schema5-v1",
                        "replica_id": replica_id,
                        "replica_index": index,
                        "scheduler_job_name": scheduler_job_name,
                        "partition": "gpu",
                    }
                ],
            }
        )
        replica_specs.append(
            {
                "replica_id": replica_id,
                "serving_profile": serving_profile,
                "served_model": served_model,
                "model_size": model_size,
                "hf_id": f"hf/{model_size}",
                "model_revision": model_revision,
                "tokenizer_id": tokenizer_id,
                "tokenizer_revision": tokenizer_revision,
                "allocated_gpus": allocated_gpus,
                "replica_index": index,
                "scheduler_job_name": scheduler_job_name,
                "partition": "gpu",
                "effective_context_limit": 32_768,
                "tensor_parallel_size": allocated_gpus,
            }
        )
    model_contract = {
        "schema_version": 1,
        "family": "test",
        "offline_required": True,
        "models": models,
    }
    model_contract_path.write_bytes(sentinel._canonical_json(model_contract))
    model_contract_path.chmod(0o444)
    model_hash = sentinel._sha256(model_contract_path)
    fleet_contract = {
        "fleet_id": "schema5-v1",
        "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
        "logical_replica_count": 22,
        "allocated_gpu_count": 24,
        "model_contract_sha256": model_hash,
        "profiles": profiles,
    }
    fleet_contract_path.write_bytes(sentinel._canonical_json(fleet_contract))
    fleet_contract_path.chmod(0o444)
    fleet_hash = sentinel._sha256(fleet_contract_path)

    state_root = tmp_path / "schema5-state"
    state_root.mkdir()
    immutable_pins_path = tmp_path / "immutable_pins.schema5-v1.json"
    release_root = tmp_path / "release"
    release_root.mkdir()
    git_commit = "1" * 40
    source_tree_sha256 = "2" * 64
    release_fragment = {
        "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
        "release_worktree": str(release_root / "worktree"),
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256,
        "model_contract_path": str(model_contract_path.resolve()),
        "model_contract_sha256": model_hash,
        "fleet_contract_path": str(fleet_contract_path.resolve()),
        "fleet_contract_sha256": fleet_hash,
        "harness_environment_prefix": str(release_root / "harness"),
        "harness_environment_manifest_path": str(
            release_root / "harness.json"
        ),
        "harness_environment_sha256": "3" * 64,
        "serving_environment_prefix": str(release_root / "serving"),
        "serving_environment_manifest_path": str(
            release_root / "serving.json"
        ),
        "serving_environment_sha256": "e" * 64,
    }
    identity_path = release_root / "release_identity.schema5-v1.json"
    release_identity = {
        "schema_version": 2,
        "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
        "release_worktree": release_fragment["release_worktree"],
        "worktree_sealed_read_only": True,
        "control_pin_fragment": release_fragment,
    }
    identity_path.write_bytes(sentinel._canonical_json(release_identity))
    identity_path.chmod(0o444)
    marker = {
        "schema_version": 2,
        "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
        "complete": True,
        "publication_protocol": "fsync_verify_marker_last",
        "artifacts": {
            identity_path.name: {
                "sha256": sentinel._sha256(identity_path),
                "size": identity_path.stat().st_size,
            }
        },
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256,
    }
    marker["release_bundle_id"] = sentinel._capacity_value_sha256(marker)
    release_marker_path = release_root / "RELEASE_COMPLETE.json"
    release_marker_path.write_bytes(sentinel._canonical_json(marker))
    release_marker_path.chmod(0o444)
    pins = release_fragment | {
        "release_bundle_root": str(release_root),
        "release_bundle_id": marker["release_bundle_id"],
        "server_pool_root": str(pool_root),
    }
    immutable_pins_path.write_bytes(sentinel._canonical_json(pins))
    immutable_pins_path.chmod(0o444)
    frozen_pins_path = state_root / "immutable_pins.json"
    frozen_pins_path.write_bytes(sentinel._canonical_json(pins))
    frozen_pins_path.chmod(0o444)
    control = {
        "immutable": pins,
        "immutable_sha256": sentinel._capacity_value_sha256(pins),
    }
    (state_root / "control.json").write_bytes(sentinel._canonical_json(control))
    manifest.update(
        {
            "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
            "release_root": str(release_root),
            "release_git_commit": git_commit,
            "state_root": str(state_root),
            "server_pool_root": str(pool_root),
            "immutable_pins": str(immutable_pins_path),
        }
    )
    verified["capacity_authority"] = {
        "replicas": replica_specs,
        "model_contract_sha256": model_hash,
        "fleet_contract_path": str(fleet_contract_path.resolve()),
        "fleet_contract_sha256": fleet_hash,
        "immutable_sha256": sentinel._capacity_value_sha256(pins),
        "server_pool_root": str(pool_root),
    }
    manifest_path = Path(verified["manifest_path"])
    receipt_path = Path(verified["submission_receipt_path"])
    manifest_path.chmod(0o644)
    receipt_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    receipt_path.chmod(0o444)
    verified["manifest_sha256"] = sentinel._sha256(manifest_path)
    verified["submission_receipt_sha256"] = sentinel._sha256(receipt_path)
    return verified


def _capacity_receipt(
    tmp_path: Path,
    verified: dict[str, Any],
    *,
    pending_reason: str = "Priority",
    recovered_failure_counts: tuple[int, int] = (0, 0),
    root: Path | None = None,
) -> Path:
    root = root or (tmp_path / "capacity")
    root.mkdir(parents=True)
    fleet_row = next(
        row
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == "fleet_readiness"
    )
    manifest_row = next(
        row
        for row in verified["manifest"]["jobs"]
        if row["name"] == "fleet_readiness"
    )
    authority = verified["capacity_authority"]
    script_root = (
        Path(authority["server_pool_root"])
        / ".fleet-transactions-v1"
        / "sbatch"
        / "g000001"
    )
    script_root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(authority["replicas"]):
        allocated_gpus = expected["allocated_gpus"]
        replica_id = expected["replica_id"]
        serving_profile = expected["serving_profile"]
        job_id = str(10_000 + index)
        intent_token = f"{index + 1:032x}"
        script_file = script_root / f"{replica_id}.{intent_token}.sbatch"
        script_file.write_text(
            f"#!/bin/bash\n# immutable {replica_id}\n", encoding="utf-8"
        )
        script_file.chmod(0o444)
        script_path = str(script_file.resolve())
        script_sha256 = sentinel._sha256(script_file)
        comment = (
            "asys-s5-fleet:pool=schema5-v1;"
            f"profile={serving_profile};replica={replica_id};generation=1;"
            f"intent={intent_token};fleet={authority['fleet_contract_sha256']}"
        )
        provenance = {
            "run_root": authority["server_pool_root"],
            "server_pool_id": "schema5-v1",
            "replica_id": replica_id,
            "replica_index": expected["replica_index"],
            "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
            "environment_hash": "e" * 64,
            "model_revision": expected["model_revision"],
            "tokenizer_id": expected["tokenizer_id"],
            "tokenizer_revision": expected["tokenizer_revision"],
            "model_contract_sha256": authority["model_contract_sha256"],
            "fleet_contract_sha256": authority["fleet_contract_sha256"],
        }
        spool_proof = {
            "argv": [
                "scontrol",
                "write",
                "batch_script",
                job_id,
                "-",
            ],
            "local_path": script_path,
            "local_sha256": script_sha256,
            "observed_sha256": script_sha256,
            "observed_bytes": script_file.stat().st_size,
            "exact_match": True,
        }
        common = {
            "replica_id": replica_id,
            "serving_profile": serving_profile,
            "allocated_gpus": allocated_gpus,
            "job_id": job_id,
            "partition": "gpu",
            "comment": comment,
            "intent_token": intent_token,
            "ledger_generation": 1,
            "local_script_path": script_path,
            "local_script_sha256": script_sha256,
            "spooled_script_sha256": script_sha256,
            "spooled_script_proof": spool_proof,
            "spooled_provenance": provenance,
        }
        if index == 0:
            rows.append(
                common
                | {
                    "state": "PENDING",
                    "reason": pending_reason,
                    "scontrol": {
                        "argv": [
                            "scontrol",
                            "show",
                            "job",
                            "-o",
                            job_id,
                        ],
                        "job_id": job_id,
                        "job_state": "PENDING",
                        "reason": pending_reason,
                        "comment": comment,
                        "job_name": expected["scheduler_job_name"],
                        "node": None,
                        "command": script_path,
                        "effective_requeue": 0,
                        "raw_output_sha256": f"{index + 100:064x}",
                    },
                }
            )
        else:
            registry_path = (
                Path(authority["server_pool_root"])
                / "servers"
                / serving_profile
                / f"{replica_id}.json"
            )
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry = {
                "model_size": expected["model_size"],
                "hf_id": expected["hf_id"],
                "host": f"node-{index:02d}",
                "port": 20_000 + index,
                "slurm_job_id": job_id,
                "started_at": 35_000.0,
                "serving_profile": serving_profile,
                "served_model_name": expected["served_model"],
                "max_model_len": expected["effective_context_limit"],
                "tp_size": expected["tensor_parallel_size"],
                "release_id": sentinel.CAPACITY_TRANSIENT_RELEASE_ID,
                "environment_hash": "e" * 64,
                "model_revision": expected["model_revision"],
                "tokenizer_id": expected["tokenizer_id"],
                "tokenizer_revision": expected["tokenizer_revision"],
                "model_contract_sha256": authority["model_contract_sha256"],
                "fleet_contract_sha256": authority["fleet_contract_sha256"],
                "server_pool_id": "schema5-v1",
                "replica_id": replica_id,
                "replica_index": expected["replica_index"],
            }
            registry_path.write_bytes(sentinel._canonical_json(registry))
            rows.append(
                common
                | {
                    "state": "RUNNING",
                    "node": f"node-{index:02d}",
                    "scontrol": {
                        "argv": [
                            "scontrol",
                            "show",
                            "job",
                            "-o",
                            job_id,
                        ],
                        "job_id": job_id,
                        "job_state": "RUNNING",
                        "reason": None,
                        "comment": comment,
                        "job_name": expected["scheduler_job_name"],
                        "node": f"node-{index:02d}",
                        "command": script_path,
                        "effective_requeue": 0,
                        "raw_output_sha256": f"{index + 100:064x}",
                    },
                    "registry_path": str(registry_path.resolve()),
                    "registry_sha256": sentinel._sha256(registry_path),
                    "http": {
                        "health_status": 200,
                        "models_status": 200,
                        "model_ids": [expected["served_model"]],
                        "expected_model": expected["served_model"],
                        "probe_started_timestamp": 35_998.0,
                        "probe_completed_timestamp": 35_999.0,
                        "healthy": True,
                    },
                }
            )
    ledger_replicas: dict[str, dict[str, Any]] = {}
    for row in rows:
        attempt = {
            "intent_token": row["intent_token"],
            "rollout_generation": 1,
            "state": "committed",
            "created_at": 34_000.0,
            "submit_started_at": 34_001.0,
            "submission_attempts": 1,
            "sbatch_path": row["local_script_path"],
            "sbatch_sha256": row["local_script_sha256"],
            "submission_transport": (
                readiness.fleet_tx.STDIN_EXACT_SUBMISSION_TRANSPORT
            ),
            "submission_argv_sha256": readiness.fleet_tx.submission_argv_sha256(
                row["comment"]
            ),
            "scheduler_comment": row["comment"],
            "job_id": row["job_id"],
            "submitted_at": 34_002.0,
            "committed_at": 34_003.0,
            "terminal_at": None,
            "last_seen_at": 35_999.0,
            "missing_since": None,
            "last_error": None,
            "launch_kind": "primary",
            "lifecycle": "primary",
            "predecessor_job_id": None,
            "predecessor_end_at": None,
            "allocated_gpus": row["allocated_gpus"],
            "scheduler_start_at": (
                None if row["state"] == "PENDING" else 34_010.0
            ),
            "scheduler_end_at": None,
            "scheduler_time_limit_seconds": 43_200,
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
            registry = json.loads(
                Path(row["registry_path"]).read_text(encoding="utf-8")
            )
            health = {
                "job_id": row["job_id"],
                "endpoint": f"{registry['host']}:{registry['port']}",
                "observer_generation": 1,
                "first_failure_at": None,
                "last_failure_at": None,
                "last_probe_at": 35_999.0,
                "consecutive_failures": 0,
                "health_failures": recovered_failure_counts[0],
                "models_failures": recovered_failure_counts[1],
                "cancel_state": None,
                "cancel_requested_at": None,
                "cancel_completed_at": None,
                "cancel_error": None,
                "cancel_attempts": 0,
                "last_cancel_attempt_at": None,
                "next_cancel_eligible_at": None,
                "alert_id": None,
            }
        ledger_replicas[row["replica_id"]] = {
            "attempts": [attempt],
            "health": health,
        }
    ledger = {
        "schema_version": 1,
        "pool_root": authority["server_pool_root"],
        "pool_id": "schema5-v1",
        "fleet_sha256": authority["fleet_contract_sha256"],
        "rollout_generation": 1,
        "visibility_grace_seconds": 300.0,
        "created_at": 34_000.0,
        "updated_at": 35_999.0,
        "replicas": ledger_replicas,
    }
    ledger_path = (
        Path(authority["server_pool_root"])
        / ".fleet-transactions-v1"
        / "ledgers"
        / "g000001.json"
    )
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(sentinel._canonical_json(ledger))
    ledger_path.chmod(0o444)
    current_path = ledger_path.parents[1] / "CURRENT.json"
    current = {
        "schema_version": 1,
        "pool_root": authority["server_pool_root"],
        "pool_id": "schema5-v1",
        "fleet_sha256": authority["fleet_contract_sha256"],
        "current_generation": 1,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": sentinel._sha256(ledger_path),
        "updated_at": 35_999.0,
    }
    current_path.write_bytes(sentinel._canonical_json(current))
    current_path.chmod(0o444)
    fleet = {
        "control_immutable_sha256": authority["immutable_sha256"],
        "rollout_generation": 1,
        "server_pool_root": authority["server_pool_root"],
        "fleet_id": "schema5-v1",
        "fleet_contract_path": authority["fleet_contract_path"],
        "fleet_contract_sha256": authority["fleet_contract_sha256"],
        "model_contract_sha256": authority["model_contract_sha256"],
        "current_pointer_path": str(current_path.resolve()),
        "current_pointer_sha256": sentinel._sha256(current_path),
        "generation_ledger_path": str(ledger_path.resolve()),
        "generation_ledger_sha256": sentinel._sha256(ledger_path),
        "scheduler_captured_timestamp": 35_999.0,
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
        "captured_timestamp": 36_000.0,
    }
    archive_key = sentinel._sha256_bytes(
        sentinel._canonical_json(
            {
                "chain_id": verified["manifest"]["chain_id"],
                "chain_generation": 0,
                "fleet_readiness_job_id": fleet_row["job_id"],
                "rollout_generation": fleet["rollout_generation"],
                "current_pointer_sha256": fleet["current_pointer_sha256"],
                "generation_ledger_sha256": fleet[
                    "generation_ledger_sha256"
                ],
                "scheduler_captured_timestamp": fleet[
                    "scheduler_captured_timestamp"
                ],
                "captured_timestamp": fleet["captured_timestamp"],
            }
        )
    )
    archive_root = (
        root
        / f"{sentinel.CAPACITY_PREIMAGE_ROOT_NAME}-{archive_key[:24]}"
    )
    archive_root.mkdir()
    archive_records: list[dict[str, Any]] = []

    def archive_file(
        *,
        kind: str,
        logical_path: str,
        payload: bytes,
        source_path: str | None,
        source_argv: list[str] | None,
        replica_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        archived = archive_root / logical_path
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(payload)
        archived.chmod(0o444)
        archive_records.append(
            {
                "kind": kind,
                "replica_id": replica_id,
                "job_id": job_id,
                "logical_path": logical_path,
                "archive_path": str(archived.resolve()),
                "source_path": source_path,
                "source_argv": source_argv,
                "sha256": sentinel._sha256(archived),
                "bytes": archived.stat().st_size,
            }
        )

    archive_file(
        kind="current_pointer",
        logical_path="fleet-state/CURRENT.json",
        payload=current_path.read_bytes(),
        source_path=str(current_path.resolve()),
        source_argv=None,
    )
    archive_file(
        kind="generation_ledger",
        logical_path="fleet-state/generation-ledger.json",
        payload=ledger_path.read_bytes(),
        source_path=str(ledger_path.resolve()),
        source_argv=None,
    )
    for row in sorted(rows, key=lambda item: item["replica_id"]):
        replica_id = row["replica_id"]
        job_id = row["job_id"]
        local = Path(row["local_script_path"])
        archive_file(
            kind="local_script",
            logical_path=f"jobs/{replica_id}/local.sbatch",
            payload=local.read_bytes(),
            source_path=str(local.resolve()),
            source_argv=None,
            replica_id=replica_id,
            job_id=job_id,
        )
        archive_file(
            kind="spooled_script",
            logical_path=f"jobs/{replica_id}/spooled.sbatch",
            payload=local.read_bytes(),
            source_path=None,
            source_argv=[
                "scontrol",
                "write",
                "batch_script",
                job_id,
                "-",
            ],
            replica_id=replica_id,
            job_id=job_id,
        )
        if row["state"] == "RUNNING":
            registry = Path(row["registry_path"])
            archive_file(
                kind="registry",
                logical_path=f"jobs/{replica_id}/registry.json",
                payload=registry.read_bytes(),
                source_path=str(registry.resolve()),
                source_argv=None,
                replica_id=replica_id,
                job_id=job_id,
            )
    archive_records.sort(key=lambda record: record["logical_path"])
    inventory_payload = "".join(
        f"{record['sha256']}  {record['logical_path']}\n"
        for record in archive_records
    ).encode("utf-8")
    inventory_path = archive_root / sentinel.CAPACITY_PREIMAGE_INVENTORY_NAME
    inventory_path.write_bytes(inventory_payload)
    inventory_path.chmod(0o444)
    total_bytes = sum(record["bytes"] for record in archive_records)
    archive_manifest_path = (
        archive_root / sentinel.CAPACITY_PREIMAGE_MANIFEST_NAME
    )
    archive_manifest = {
        "schema_version": 1,
        "protocol": sentinel.CAPACITY_PREIMAGE_PROTOCOL,
        "chain_id": verified["manifest"]["chain_id"],
        "chain_generation": 0,
        "fleet_readiness_job_id": fleet_row["job_id"],
        "rollout_generation": 1,
        "fleet_contract_sha256": authority["fleet_contract_sha256"],
        "files": archive_records,
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sentinel._sha256(inventory_path),
        "file_count": len(archive_records),
        "total_bytes": total_bytes,
    }
    archive_manifest["manifest_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(archive_manifest)
    )
    archive_manifest_path.write_bytes(
        sentinel._canonical_json(archive_manifest)
    )
    archive_manifest_path.chmod(0o444)
    completion_path = archive_root / sentinel.CAPACITY_PREIMAGE_COMPLETE_NAME
    completion = {
        "schema_version": 1,
        "protocol": sentinel.CAPACITY_PREIMAGE_PROTOCOL,
        "passed": True,
        "root": str(archive_root.resolve()),
        "manifest": str(archive_manifest_path.resolve()),
        "manifest_sha256": sentinel._sha256(archive_manifest_path),
        "manifest_id": archive_manifest["manifest_id"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sentinel._sha256(inventory_path),
        "file_count": len(archive_records),
        "total_bytes": total_bytes,
    }
    completion["archive_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(completion)
    )
    completion_path.write_bytes(sentinel._canonical_json(completion))
    completion_path.chmod(0o444)
    for directory in sorted(
        (path for path in archive_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    archive_root.chmod(0o555)
    preimage_archive = {
        "schema_version": 1,
        "protocol": sentinel.CAPACITY_PREIMAGE_PROTOCOL,
        "root": str(archive_root.resolve()),
        "completion": str(completion_path.resolve()),
        "completion_sha256": sentinel._sha256(completion_path),
        "archive_id": completion["archive_id"],
        "manifest": str(archive_manifest_path.resolve()),
        "manifest_sha256": sentinel._sha256(archive_manifest_path),
        "manifest_id": archive_manifest["manifest_id"],
        "inventory": str(inventory_path.resolve()),
        "inventory_sha256": sentinel._sha256(inventory_path),
        "file_count": len(archive_records),
        "total_bytes": total_bytes,
    }
    evidence_path = root / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
    evidence = {
        "schema_version": 1,
        "protocol": sentinel.CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL,
        "passed": True,
        "chain_protocol": sentinel.R11_PROTOCOL,
        "chain_id": verified["manifest"]["chain_id"],
        "chain_generation": 0,
        "manifest": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "submission_receipt": verified["submission_receipt_path"],
        "submission_receipt_sha256": verified["submission_receipt_sha256"],
        "submission_receipt_id": verified["submission_receipt"]["receipt_id"],
        "fleet_readiness_job_id": fleet_row["job_id"],
        "fleet_readiness_comment": fleet_row["comment"],
        "boundary_seconds": sentinel.CAPACITY_TRANSIENT_BOUNDARY_SECONDS,
        "boundary_proof": {
            "argv": [
                "scontrol",
                "show",
                "job",
                "-o",
                fleet_row["job_id"],
            ],
            "job_id": fleet_row["job_id"],
            "comment": fleet_row["comment"],
            "job_name": manifest_row["job_name"],
            "job_state": "RUNNING",
            "effective_requeue": 0,
            "runtime_seconds": sentinel.CAPACITY_TRANSIENT_BOUNDARY_SECONDS,
            "raw_output_sha256": "f" * 64,
        },
        "fleet": fleet,
        "preimage_archive": preimage_archive,
        "observed_at": "1970-01-01T10:00:00+00:00",
        "observed_timestamp": 36_000.0,
    }
    evidence["evidence_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(evidence)
    )
    evidence_path.write_bytes(sentinel._canonical_json(evidence))
    evidence_path.chmod(0o444)
    marker_path = root / sentinel.CAPACITY_TRANSIENT_MARKER_NAME
    marker = {
        "schema_version": 1,
        "protocol": sentinel.CAPACITY_TRANSIENT_PROTOCOL,
        "passed": True,
        "capacity_transient_root": True,
        "chain_id": verified["manifest"]["chain_id"],
        "chain_generation": 0,
        "fleet_readiness_job_id": fleet_row["job_id"],
        "fleet_readiness_comment": fleet_row["comment"],
        "evidence": str(evidence_path.resolve()),
        "evidence_sha256": sentinel._sha256(evidence_path),
        "evidence_id": evidence["evidence_id"],
        "preimage_archive": preimage_archive,
        "published_at": "1970-01-01T10:00:00+00:00",
        "published_timestamp": 36_000.0,
    }
    marker["receipt_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(marker)
    )
    marker_path.write_bytes(sentinel._canonical_json(marker))
    marker_path.chmod(0o444)
    return marker_path


def _rehash_capacity_receipt(
    marker_path: Path,
    mutate: Any,
) -> None:
    evidence_path = marker_path.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    mutate(evidence)
    evidence.pop("evidence_id", None)
    evidence["evidence_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(evidence)
    )
    evidence_path.chmod(0o644)
    evidence_path.write_bytes(sentinel._canonical_json(evidence))
    evidence_path.chmod(0o444)

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.update(
        {
            "evidence_sha256": sentinel._sha256(evidence_path),
            "evidence_id": evidence["evidence_id"],
            "preimage_archive": evidence["preimage_archive"],
        }
    )
    marker.pop("receipt_id", None)
    marker["receipt_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(marker)
    )
    marker_path.chmod(0o644)
    marker_path.write_bytes(sentinel._canonical_json(marker))
    marker_path.chmod(0o444)


def _assert_capacity_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: dict[str, Any],
    capacity: Path,
    *,
    expected_error: str,
) -> dict[str, Any]:
    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state="FAILED",
        first_exit="75:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=tmp_path / "corrupt-output",
        capacity_transient_receipt=capacity,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
    )
    assert report["classification"] == "requires_superseding_release"
    assert report["same_generation_repair_allowed"] is False
    evidence = json.loads(
        (
            tmp_path
            / "corrupt-output"
            / sentinel.SCHEDULER_EVIDENCE_NAME
        ).read_text(encoding="utf-8")
    )
    corruption = evidence["outcome"]["capacity_transient_corruption"]
    assert expected_error in corruption["error"]
    assert evidence["capacity_transient_receipt"] is None
    return evidence


def _sacct_row(
    verified: dict[str, Any],
    name: str,
    state: str,
    *,
    exit_code: str = "0:0",
    reason: str = "None",
    start: str = "2026-07-23T00:00:00",
    elapsed: str = "00:00:01",
) -> str:
    manifest_row = next(
        row for row in verified["manifest"]["jobs"] if row["name"] == name
    )
    receipt_row = next(
        row
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == name
    )
    submit_line = (
        f"sbatch --comment={receipt_row['comment']} /jobs/{name}.sbatch"
    )
    return (
        f"{receipt_row['job_id']}|{state}|{exit_code}|{reason}|{start}|{elapsed}|"
        f"(null)|{manifest_row['job_name']}|{submit_line}\n"
    )


class FakeScheduler:
    def __init__(
        self,
        verified: dict[str, Any],
        *,
        first_state: str = "COMPLETED",
        first_exit: str = "0:0",
        first_reason: str = "None",
        first_start: str = "2026-07-23T00:00:00",
        first_elapsed: str = "00:00:01",
        second_state: str = "COMPLETED",
        second_exit: str = "0:0",
        second_reason: str = "None",
        second_start: str = "2026-07-23T00:00:01",
        second_elapsed: str = "00:00:01",
        duplicate_sacct: bool = False,
        fail_sacct: bool = False,
        first_name: str = "first",
    ) -> None:
        self.verified = verified
        self.first = (
            first_state,
            first_exit,
            first_reason,
            first_start,
            first_elapsed,
        )
        self.second = (
            second_state,
            second_exit,
            second_reason,
            second_start,
            second_elapsed,
        )
        self.duplicate_sacct = duplicate_sacct
        self.fail_sacct = fail_sacct
        self.first_name = first_name
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[0] == "squeue":
            row = next(
                row
                for row in self.verified["submission_receipt"]["jobs"]
                if row["name"] == "failure_sentinel"
            )
            manifest = next(
                row
                for row in self.verified["manifest"]["jobs"]
                if row["name"] == "failure_sentinel"
            )
            output = (
                f"{row['job_id']}|RUNNING|None|{row['comment']}|"
                f"{manifest['job_name']}\n"
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] != "sacct":
            raise AssertionError(args)
        if self.fail_sacct:
            return subprocess.CompletedProcess(args, 1, "", "accounting unavailable")
        output = _sacct_row(
            self.verified,
            self.first_name,
            self.first[0],
            exit_code=self.first[1],
            reason=self.first[2],
            start=self.first[3],
            elapsed=self.first[4],
        )
        output += _sacct_row(
            self.verified,
            "second",
            self.second[0],
            exit_code=self.second[1],
            reason=self.second[2],
            start=self.second[3],
            elapsed=self.second[4],
        )
        output += _sacct_row(
            self.verified,
            "failure_sentinel",
            "RUNNING",
            start="2026-07-23T00:00:02",
            elapsed="00:00:01",
        )
        if self.duplicate_sacct:
            output += _sacct_row(
                self.verified,
                self.first_name,
                self.first[0],
                exit_code=self.first[1],
                reason=self.first[2],
                start=self.first[3],
                elapsed=self.first[4],
            )
        return subprocess.CompletedProcess(args, 0, output, "")


def _patch_verified(
    monkeypatch: pytest.MonkeyPatch, verified: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        sentinel,
        "_verified_inputs",
        lambda _manifest, _receipt: verified,
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: dict[str, Any],
    scheduler: FakeScheduler,
    *,
    apply: bool,
    output: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _patch_verified(monkeypatch, verified)
    return sentinel.run_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output or tmp_path / "out",
        recipient="mabdel03@mit.edu",
        apply=apply,
        runner=scheduler,
        **kwargs,
    )


def test_failfast_stage_failure_is_recorded_and_alerted_without_repair_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _failfast_verified(tmp_path)
    target = "context_readiness"
    observer_name = f"{sentinel.STAGE_SENTINEL_PREFIX}{target}"
    observer_id = next(
        row["job_id"]
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == observer_name
    )
    scheduler = FailFastScheduler(
        verified,
        target_stage=target,
        target_state="FAILED",
        target_exit_code="2:0",
    )
    _patch_verified(monkeypatch, verified)
    delivered: list[tuple[list[str], str]] = []
    output = tmp_path / "stage-output"
    report = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output,
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=scheduler,
        mail_runner=lambda argv, body: (
            delivered.append((list(argv), body))
            or subprocess.CompletedProcess(list(argv), 0, "", "")
        ),
        clock=FakeClock(),
    )
    assert report["classification"] == "stage_failure_observed"
    assert report["authoritative_repair_classification"] is False
    assert report["alert_requested"] is True
    assert report["alert_delivered"] is True
    assert len(delivered) == 1
    assert target in delivered[0][1]
    assert "Repair authority: none" in delivered[0][1]
    evidence_path = output / sentinel.STAGE_SCHEDULER_EVIDENCE_NAME
    marker_path = output / sentinel.STAGE_COMPLETE_MARKER_NAME
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["outcome"]["target_stage"] == target
    assert evidence["outcome"]["target_state"] == "FAILED"
    assert evidence["outcome"]["same_generation_repair_allowed"] is False
    assert stat.S_IMODE(evidence_path.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(marker_path.stat().st_mode) & 0o222 == 0

    repeated = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output,
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=lambda _argv: pytest.fail(
            "idempotent stage observer requeried scheduler"
        ),
        mail_runner=lambda _argv, _body: pytest.fail(
            "confirmed fail-fast alert was duplicated"
        ),
        clock=FakeClock(2_000.0),
    )
    assert repeated["status"] == "already_complete"
    assert repeated["mail_attempt_count"] == 1


def test_failfast_stage_success_records_without_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _failfast_verified(tmp_path)
    target = "source_checkout"
    observer_name = f"{sentinel.STAGE_SENTINEL_PREFIX}{target}"
    observer_id = next(
        row["job_id"]
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == observer_name
    )
    scheduler = FailFastScheduler(
        verified,
        target_stage=target,
        target_state="COMPLETED",
        target_exit_code="0:0",
    )
    _patch_verified(monkeypatch, verified)
    report = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=tmp_path / "stage-success",
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=scheduler,
        mail_runner=lambda _argv, _body: pytest.fail(
            "successful stage observer sent mail"
        ),
        clock=FakeClock(),
    )
    assert report["classification"] == "complete"
    assert report["alert_requested"] is False
    assert report["mail_attempt_count"] == 0


def test_failfast_stage_marker_never_waits_for_email_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _failfast_verified(tmp_path)
    target = "release_freeze"
    observer_name = f"{sentinel.STAGE_SENTINEL_PREFIX}{target}"
    observer_id = next(
        row["job_id"]
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == observer_name
    )
    scheduler = FailFastScheduler(
        verified,
        target_stage=target,
        target_state="FAILED",
        target_exit_code="9:0",
    )
    _patch_verified(monkeypatch, verified)
    clock = FakeClock()
    calls: list[str] = []
    output = tmp_path / "stage-mail-outage"

    report = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output,
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=scheduler,
        mail_runner=lambda argv, _body: (
            calls.append("failed")
            or subprocess.CompletedProcess(
                list(argv), 1, "", "mail transport unavailable"
            )
        ),
        clock=clock,
        sleeper=lambda _seconds: pytest.fail(
            "stage sentinel slept for email retry"
        ),
        # An expansive caller request must still be capped to one synchronous
        # attempt by the sentinel publication contract.
        maximum_mail_attempts=5,
    )
    assert clock.value == 1_000.0
    assert calls == ["failed"]
    assert report["classification"] == "stage_failure_observed"
    assert report["mail_attempt_count"] == 1
    assert report["alert_delivered"] is False
    assert report["alert_retry_pending"] is True
    assert report["mail_next_retry_timestamp"] == pytest.approx(1_005.0)
    assert (output / sentinel.STAGE_COMPLETE_MARKER_NAME).is_file()
    evidence = json.loads(
        (output / sentinel.STAGE_SCHEDULER_EVIDENCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["outcome"]["classification"] == "stage_failure_observed"
    assert evidence["outcome"]["authoritative_repair_classification"] is False
    mail = json.loads(
        (output / sentinel.MAIL_STATE_NAME).read_text(encoding="utf-8")
    )
    assert mail["next_retry_timestamp"] == pytest.approx(1_005.0)

    # Before next eligibility, an idempotent successor returns immediately without
    # another send or scheduler query and without mutating scientific classification.
    repeated = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output,
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=lambda _argv: pytest.fail("sealed stage evidence was requeried"),
        mail_runner=lambda _argv, _body: pytest.fail(
            "ineligible stage mail retry was attempted"
        ),
        clock=clock,
        sleeper=lambda _seconds: pytest.fail(
            "stage successor slept for email retry"
        ),
        maximum_mail_attempts=5,
    )
    assert repeated["status"] == "already_complete"
    assert repeated["classification"] == report["classification"]
    assert repeated["mail_attempt_count"] == 1
    assert clock.value == 1_000.0

    clock.value = 1_005.0
    retried = sentinel.run_stage_sentinel(
        chain_manifest=verified["manifest_path"],
        submission_receipt=verified["submission_receipt_path"],
        output_root=output,
        recipient="mabdel03@mit.edu",
        stage_name=target,
        stage_sentinel_job_id=observer_id,
        apply=True,
        runner=lambda _argv: pytest.fail("sealed stage evidence was requeried"),
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
        clock=clock,
        sleeper=lambda _seconds: pytest.fail(
            "eligible stage successor slept for email retry"
        ),
        maximum_mail_attempts=5,
    )
    assert retried["classification"] == report["classification"]
    assert retried["alert_delivered"] is True
    assert retried["mail_attempt_count"] == 2


def test_failfast_stage_rejects_wrong_running_observer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _failfast_verified(tmp_path)
    target = "release_materialize"
    scheduler = FailFastScheduler(
        verified,
        target_stage=target,
        target_state="FAILED",
        target_exit_code="1:0",
    )
    _patch_verified(monkeypatch, verified)
    with pytest.raises(
        sentinel.SentinelError,
        match="exact fail-fast stage observer",
    ):
        sentinel.run_stage_sentinel(
            chain_manifest=verified["manifest_path"],
            submission_receipt=verified["submission_receipt_path"],
            output_root=tmp_path / "wrong-observer",
            recipient="mabdel03@mit.edu",
            stage_name=target,
            stage_sentinel_job_id="999999",
            apply=False,
            runner=scheduler,
        )


def test_aggregate_classifier_requires_all_stage_observers_terminal() -> None:
    rows: list[dict[str, Any]] = []
    for stage in sentinel.PRODUCTION_STAGE_NAMES:
        rows.append(
            {
                "name": stage,
                "job_id": str(len(rows) + 1),
                "dependencies": [],
                "dependency_type": "afterok",
                "state": "COMPLETED",
                "active": False,
                "exit_code": "0:0",
                "reason": "None",
                "start": "2026-07-23T00:00:00",
                "elapsed": "00:00:01",
            }
        )
    for stage, observer_name in zip(
        sentinel.PRODUCTION_STAGE_NAMES,
        sentinel.STAGE_SENTINEL_NAMES,
        strict=True,
    ):
        rows.append(
            {
                "name": observer_name,
                "job_id": str(len(rows) + 1),
                "dependencies": [stage],
                "dependency_type": "afterany",
                "state": "COMPLETED",
                "active": False,
                "exit_code": "0:0",
                "reason": "None",
                "start": "2026-07-23T00:00:01",
                "elapsed": "00:00:01",
            }
        )
    rows.append(
        {
            "name": "failure_sentinel",
            "job_id": str(len(rows) + 1),
            "dependencies": [
                *sentinel.PRODUCTION_STAGE_NAMES,
                *sentinel.STAGE_SENTINEL_NAMES,
            ],
            "dependency_type": "afterany",
            "state": "RUNNING",
            "active": True,
            "exit_code": "0:0",
            "reason": "None",
            "start": "2026-07-23T00:00:02",
            "elapsed": "00:00:01",
        }
    )
    outcome = sentinel.classify_recovery_jobs(rows)
    assert outcome["classification"] == "complete"
    assert outcome["completed_stage_observer_count"] == 21

    observer = next(
        row
        for row in rows
        if row["name"]
        == f"{sentinel.STAGE_SENTINEL_PREFIX}release_materialize"
    )
    observer.update(
        state="NODE_FAIL",
        active=False,
        exit_code="0:0",
        reason="NodeFailure",
    )
    outcome = sentinel.classify_recovery_jobs(rows)
    assert outcome["classification"] == "transient_repairable"
    assert observer["name"] in outcome["transient_roots"]


def test_dry_run_is_fully_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(verified)
    output = tmp_path / "absent"
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=False,
        output=output,
        mail_runner=lambda _argv, _body: pytest.fail("dry-run sent mail"),
    )
    assert report["classification"] == "complete"
    assert report["writes_performed"] is False
    assert not output.exists()
    assert [call[0] for call in scheduler.calls] == ["squeue", "sacct"]


def test_apply_publishes_marker_last_read_only_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(verified)
    output = tmp_path / "out"
    writes: list[str] = []
    original = sentinel._atomic_json

    def recording_write(path: Path, payload: object, *, mode: int) -> None:
        writes.append(path.name)
        original(path, payload, mode=mode)

    monkeypatch.setattr(sentinel, "_atomic_json", recording_write)
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=output,
    )
    assert report["classification"] == "complete"
    assert report["alert_requested"] is False
    assert writes[-1] == sentinel.COMPLETE_MARKER_NAME
    for name in (sentinel.SCHEDULER_EVIDENCE_NAME, sentinel.COMPLETE_MARKER_NAME):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o444

    calls = len(scheduler.calls)
    second = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("idempotent run queried Slurm"),  # type: ignore[arg-type]
        apply=True,
        output=output,
    )
    assert second["status"] == "already_complete"
    assert len(scheduler.calls) == calls


def test_self_rehashed_scheduler_outcome_must_be_derived_from_archived_raw_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    output = tmp_path / "out"
    _run(
        tmp_path,
        monkeypatch,
        verified,
        FakeScheduler(verified),
        apply=True,
        output=output,
    )
    evidence_path = output / sentinel.SCHEDULER_EVIDENCE_NAME
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert "stdout" in evidence["scheduler_queries"]["squeue"]
    assert "stdout" in evidence["scheduler_queries"]["sacct"]
    evidence["outcome"]["classification"] = "requires_superseding_release"
    evidence["outcome"]["run_succeeded"] = False
    evidence["outcome"]["requires_superseding_release"] = True
    identity = dict(evidence)
    identity.pop("evidence_id")
    evidence["evidence_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(identity)
    )
    evidence_path.chmod(0o644)
    evidence_path.write_bytes(sentinel._canonical_json(evidence))
    evidence_path.chmod(0o444)

    with pytest.raises(sentinel.SentinelError, match="not derived"):
        _run(
            tmp_path,
            monkeypatch,
            verified,
            lambda _argv: pytest.fail("sealed evidence re-queried Slurm"),  # type: ignore[arg-type]
            apply=True,
            output=output,
        )


def test_self_rehashed_scheduler_query_hash_must_match_archived_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    output = tmp_path / "out"
    _run(
        tmp_path,
        monkeypatch,
        verified,
        FakeScheduler(verified),
        apply=True,
        output=output,
    )
    evidence_path = output / sentinel.SCHEDULER_EVIDENCE_NAME
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["scheduler_queries"]["sacct"]["stdout_sha256"] = "f" * 64
    identity = dict(evidence)
    identity.pop("evidence_id")
    evidence["evidence_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(identity)
    )
    evidence_path.chmod(0o644)
    evidence_path.write_bytes(sentinel._canonical_json(evidence))
    evidence_path.chmod(0o444)

    with pytest.raises(sentinel.SentinelError, match="query evidence drifted"):
        _run(
            tmp_path,
            monkeypatch,
            verified,
            lambda _argv: pytest.fail("sealed evidence re-queried Slurm"),  # type: ignore[arg-type]
            apply=True,
            output=output,
        )


def test_transient_root_and_true_dependency_cancelled_suffix_are_repairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_state="NODE_FAIL",
        first_exit="1:0",
        first_reason="NodeDown",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    sent: list[list[str]] = []
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        mail_runner=lambda argv, _body: (
            sent.append(list(argv))
            or subprocess.CompletedProcess(list(argv), 0, "", "")
        ),
    )
    assert report["classification"] == "transient_repairable"
    assert report["same_generation_repair_allowed"] is True
    assert report["requires_superseding_release"] is False
    assert len(sent) == 1
    evidence = json.loads(
        (tmp_path / "out" / sentinel.SCHEDULER_EVIDENCE_NAME).read_text()
    )
    assert evidence["outcome"]["transient_roots"] == ["first"]
    assert evidence["outcome"]["dependency_cancelled_suffix"] == ["second"]


def test_exact_failed_75_capacity_receipt_is_repairable_and_bound_in_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)
    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state="FAILED",
        first_exit="75:0",
        first_reason="NonZeroExitCode",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        capacity_transient_receipt=capacity,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
    )
    assert report["classification"] == "transient_repairable"
    evidence = json.loads(
        (tmp_path / "out" / sentinel.SCHEDULER_EVIDENCE_NAME).read_text()
    )
    assert evidence["outcome"]["capacity_transient_roots"] == [
        "fleet_readiness"
    ]
    assert evidence["outcome"]["dependency_cancelled_suffix"] == ["second"]
    assert evidence["outcome"]["dispositions"]["fleet_readiness"] == (
        "capacity_transient_root"
    )
    assert evidence["capacity_transient_receipt"] == {
        "path": str(capacity.resolve()),
        "sha256": sentinel._sha256(capacity),
        "receipt_id": json.loads(capacity.read_text())["receipt_id"],
        "evidence": str(
            (capacity.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME).resolve()
        ),
        "evidence_sha256": sentinel._sha256(
            capacity.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
        ),
        "evidence_id": json.loads(
            (
                capacity.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
            ).read_text()
        )["evidence_id"],
        "chain_generation": 0,
        "fleet_readiness_job_id": "101",
        "preimage_archive": json.loads(capacity.read_text())[
            "preimage_archive"
        ],
    }


def test_real_producer_receipt_with_full_recovered_ledger_is_repairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    seed = _capacity_receipt(
        tmp_path,
        verified,
        recovered_failure_counts=(7, 11),
    )
    seed_evidence = json.loads(
        (
            seed.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
        ).read_text(encoding="utf-8")
    )
    fleet = seed_evidence["fleet"]
    manifest_row = next(
        row
        for row in verified["manifest"]["jobs"]
        if row["name"] == "fleet_readiness"
    )
    receipt_row = next(
        row
        for row in verified["submission_receipt"]["jobs"]
        if row["name"] == "fleet_readiness"
    )
    producer_binding = {
        "verified": verified,
        "manifest_row": manifest_row,
        "receipt_row": receipt_row,
        "chain_generation": 0,
    }
    monkeypatch.setattr(
        readiness,
        "_verified_capacity_chain_binding",
        lambda **_kwargs: producer_binding,
    )
    monkeypatch.setattr(
        readiness,
        "_capacity_boundary_proof",
        lambda **_kwargs: seed_evidence["boundary_proof"],
    )
    monkeypatch.setattr(
        readiness,
        "_collect_fleet_capacity_transient",
        lambda *_args, **_kwargs: fleet,
    )
    scripts = {
        row["job_id"]: Path(row["local_script_path"])
        for row in [*fleet["pending"], *fleet["running"]]
    }

    def scheduler_runner(
        argv: Sequence[str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        job_id = str(argv[-2])
        return subprocess.CompletedProcess(
            list(argv),
            0,
            scripts[job_id].read_text(encoding="utf-8"),
            "",
        )

    producer_marker = (
        tmp_path
        / "producer-capacity"
        / readiness.CAPACITY_TRANSIENT_MARKER_NAME
    )
    produced = readiness.build_fleet_capacity_transient_receipt(
        {"immutable_sha256": fleet["control_immutable_sha256"]},
        output=producer_marker,
        state_dir=tmp_path / "producer-state",
        chain_manifest=Path(verified["manifest_path"]),
        submission_receipt=Path(verified["submission_receipt_path"]),
        readiness_job_id=receipt_row["job_id"],
        scheduler_runner=scheduler_runner,
        now=lambda: 36_000.0,
    )
    assert produced["status"] == "complete"
    producer_evidence = json.loads(
        (
            producer_marker.parent
            / readiness.CAPACITY_TRANSIENT_EVIDENCE_NAME
        ).read_text(encoding="utf-8")
    )
    archive_manifest = json.loads(
        Path(
            producer_evidence["preimage_archive"]["manifest"]
        ).read_text(encoding="utf-8")
    )
    sealed_ledger_record = next(
        row
        for row in archive_manifest["files"]
        if row["kind"] == "generation_ledger"
    )
    sealed_ledger = json.loads(
        Path(sealed_ledger_record["archive_path"]).read_text(encoding="utf-8")
    )
    assert len(sealed_ledger["replicas"]) == 22
    assert any(
        replica["health"]["health_failures"] == 7
        and replica["health"]["models_failures"] == 11
        and replica["health"]["consecutive_failures"] == 0
        for replica in sealed_ledger["replicas"].values()
        if replica["health"] is not None
    )

    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state="FAILED",
        first_exit="75:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=tmp_path / "producer-sentinel-output",
        capacity_transient_receipt=producer_marker,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
    )
    assert report["classification"] == "transient_repairable"


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [
        ("FAILED", "1:0"),
        ("FAILED", "75:1"),
        ("TIMEOUT", "0:0"),
    ],
)
def test_arbitrary_fleet_failure_without_capacity_receipt_is_superseding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    exit_code: str,
) -> None:
    verified = _capacity_verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state=state,
        first_exit=exit_code,
        first_reason="TimeLimit" if state == "TIMEOUT" else "NonZeroExitCode",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=False,
    )
    assert report["classification"] == "requires_superseding_release"
    assert report["outcome"]["capacity_transient_roots"] == []


def test_invalid_capacity_reason_publishes_data_trust_failure_and_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(
        tmp_path, verified, pending_reason="QOSMaxGRESPerUser"
    )
    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state="FAILED",
        first_exit="75:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    output = tmp_path / "invalid-capacity"
    sent: list[tuple[list[str], str]] = []
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=output,
        capacity_transient_receipt=capacity,
        mail_runner=lambda argv, body: (
            sent.append((list(argv), body))
            or subprocess.CompletedProcess(list(argv), 0, "", "")
        ),
    )
    assert report["classification"] == "requires_superseding_release"
    assert report["same_generation_repair_allowed"] is False
    assert report["requires_superseding_release"] is True
    assert len(sent) == 1
    evidence_path = output / sentinel.SCHEDULER_EVIDENCE_NAME
    marker_path = output / sentinel.COMPLETE_MARKER_NAME
    assert evidence_path.is_file()
    assert marker_path.is_file()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    corruption = evidence["outcome"]["capacity_transient_corruption"]
    assert evidence["capacity_transient_receipt"] is None
    assert evidence["outcome"]["capacity_transient_roots"] == []
    assert corruption["kind"] == "capacity_transient_data_trust_failure"
    assert corruption["path"] == str(capacity.resolve())
    assert corruption["sha256"] == sentinel._sha256(capacity)
    assert corruption["evidence_sha256"] == sentinel._sha256(
        capacity.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
    )
    assert "pending reason is not admissible" in corruption["error"]
    assert "Requires superseding release: True" in sent[0][1]
    assert "Capacity receipt data-trust failure" in sent[0][1]
    capacity.parent.rename(tmp_path / "preserved-corrupt-capacity")
    second = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("sealed corruption rerun queried Slurm"),  # type: ignore[arg-type]
        apply=True,
        output=output,
        capacity_transient_receipt=capacity,
        mail_runner=lambda _argv, _body: pytest.fail(
            "confirmed corruption alert was redelivered"
        ),
    )
    assert second["status"] == "already_complete"
    assert second["classification"] == "requires_superseding_release"


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("boundary", "receipt identity is invalid"),
        ("pending_spool", "spooled-script proof is invalid"),
        ("running_requeue", "exact scontrol proof is invalid"),
        ("provenance", "spooled provenance is invalid"),
        ("http", "dual HTTP proof is invalid"),
        ("topology_hash", "fleet identity/trust proof is invalid"),
        ("invented_replica", "unknown replica"),
        ("ledger_token", "committed ledger attempt"),
        ("registry", "source binding drifted"),
    ],
)
def test_rehashed_nested_capacity_tampering_is_nonrepairable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_error: str,
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)

    def mutate(evidence: dict[str, Any]) -> None:
        fleet = evidence["fleet"]
        pending = fleet["pending"][0]
        running = fleet["running"][0]
        if case == "boundary":
            evidence["boundary_proof"]["argv"][-1] = "999999"
        elif case == "pending_spool":
            pending["spooled_script_proof"]["exact_match"] = False
        elif case == "running_requeue":
            running["scontrol"]["effective_requeue"] = 1
        elif case == "provenance":
            running["spooled_provenance"]["environment_hash"] = "9" * 64
        elif case == "http":
            running["http"]["expected_model"] = "invented-model"
            running["http"]["model_ids"] = ["invented-model"]
        elif case == "topology_hash":
            fleet["model_contract_sha256"] = "9" * 64
        elif case == "invented_replica":
            running["replica_id"] = "invented-replica"
        elif case == "ledger_token":
            old_token = pending["intent_token"]
            new_token = "f" * 32
            pending["intent_token"] = new_token
            pending["comment"] = pending["comment"].replace(
                f"intent={old_token}", f"intent={new_token}"
            )
            pending["scontrol"]["comment"] = pending["comment"]
        elif case == "registry":
            registry_path = Path(running["registry_path"])
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["model_revision"] = "invented-revision"
            registry_path.write_bytes(sentinel._canonical_json(registry))
            running["registry_sha256"] = sentinel._sha256(registry_path)
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(case)

    _rehash_capacity_receipt(capacity, mutate)
    _assert_capacity_corruption(
        tmp_path,
        monkeypatch,
        verified,
        capacity,
        expected_error=expected_error,
    )


def test_capacity_receipt_survives_rotation_of_all_live_fleet_preimages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)
    evidence = json.loads(
        (
            capacity.parent / sentinel.CAPACITY_TRANSIENT_EVIDENCE_NAME
        ).read_text(encoding="utf-8")
    )
    fleet = evidence["fleet"]
    script_root = Path(fleet["running"][0]["local_script_path"]).parent
    servers_root = Path(fleet["server_pool_root"]) / "servers"
    transaction_root = Path(fleet["current_pointer_path"]).parents[1]
    script_root.rename(script_root.with_name(f"{script_root.name}.rotated"))
    servers_root.rename(servers_root.with_name(f"{servers_root.name}.rotated"))
    transaction_root.rename(
        transaction_root.with_name(f"{transaction_root.name}.rotated")
    )

    scheduler = FakeScheduler(
        verified,
        first_name="fleet_readiness",
        first_state="FAILED",
        first_exit="75:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=tmp_path / "rotated-output",
        capacity_transient_receipt=capacity,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
    )
    assert report["classification"] == "transient_repairable"
    scheduler_evidence = json.loads(
        (
            tmp_path
            / "rotated-output"
            / sentinel.SCHEDULER_EVIDENCE_NAME
        ).read_text(encoding="utf-8")
    )
    assert scheduler_evidence["capacity_transient_receipt"][
        "preimage_archive"
    ] == evidence["preimage_archive"]


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("bytes", "bytes/inode drifted"),
        ("symlink", "record identity is invalid"),
        ("shared_inode", "bytes/inode drifted"),
        ("writable_directory", "directory is writable"),
        ("unexpected_member", "unlisted/missing members"),
    ],
)
def test_capacity_preimage_archive_corruption_is_nonrepairable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_error: str,
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)
    marker = json.loads(capacity.read_text(encoding="utf-8"))
    archive_binding = marker["preimage_archive"]
    archive_root = Path(archive_binding["root"])
    manifest = json.loads(
        Path(archive_binding["manifest"]).read_text(encoding="utf-8")
    )
    records = manifest["files"]
    if mode == "bytes":
        target = Path(records[0]["archive_path"])
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"tamper")
        target.chmod(0o444)
    elif mode == "symlink":
        record = next(row for row in records if row["kind"] == "local_script")
        target = Path(record["archive_path"])
        parent = target.parent
        backup = target.with_suffix(".preimage")
        parent.chmod(0o755)
        target.rename(backup)
        target.symlink_to(backup.name)
        parent.chmod(0o555)
    elif mode == "shared_inode":
        replica_id = next(
            row["replica_id"]
            for row in records
            if row["kind"] == "local_script"
        )
        local = Path(
            next(
                row["archive_path"]
                for row in records
                if row["kind"] == "local_script"
                and row["replica_id"] == replica_id
            )
        )
        spooled = Path(
            next(
                row["archive_path"]
                for row in records
                if row["kind"] == "spooled_script"
                and row["replica_id"] == replica_id
            )
        )
        parent = local.parent
        parent.chmod(0o755)
        spooled.unlink()
        os.link(local, spooled)
        parent.chmod(0o555)
    elif mode == "writable_directory":
        (archive_root / "jobs").chmod(0o755)
    elif mode == "unexpected_member":
        archive_root.chmod(0o755)
        unexpected = archive_root / "unexpected"
        unexpected.write_bytes(b"not listed in the sealed inventory\n")
        unexpected.chmod(0o444)
        archive_root.chmod(0o555)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mode)
    _assert_capacity_corruption(
        tmp_path,
        monkeypatch,
        verified,
        capacity,
        expected_error=expected_error,
    )


def test_rehashed_capacity_archive_root_substitution_is_nonrepairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)

    def mutate(evidence: dict[str, Any]) -> None:
        archive = evidence["preimage_archive"]
        original = Path(archive["root"])
        substitute = original.with_name(f"{original.name}.substitute")
        substitute.mkdir()
        substitute.chmod(0o555)
        archive["root"] = str(substitute.resolve())

    _rehash_capacity_receipt(capacity, mutate)
    _assert_capacity_corruption(
        tmp_path,
        monkeypatch,
        verified,
        capacity,
        expected_error="archive root is invalid",
    )


def test_rehashed_stale_capacity_marker_is_nonrepairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)
    marker = json.loads(capacity.read_text(encoding="utf-8"))
    marker["published_timestamp"] = 36_601.0
    marker["published_at"] = sentinel._utc(36_601.0)
    marker.pop("receipt_id")
    marker["receipt_id"] = sentinel._sha256_bytes(
        sentinel._canonical_json(marker)
    )
    capacity.chmod(0o644)
    capacity.write_bytes(sentinel._canonical_json(marker))
    capacity.chmod(0o444)
    _assert_capacity_corruption(
        tmp_path,
        monkeypatch,
        verified,
        capacity,
        expected_error="receipt identity is invalid",
    )


def test_rehashed_control_and_fleet_substitution_cannot_bypass_sealed_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _capacity_verified(tmp_path)
    capacity = _capacity_receipt(tmp_path, verified)
    manifest = verified["manifest"]
    pins_path = Path(manifest["immutable_pins"])
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    fleet_path = Path(pins["fleet_contract_path"])
    fleet_contract = json.loads(fleet_path.read_text(encoding="utf-8"))
    fleet_contract["profiles"][0]["served_model_name"] = "invented-model"
    fleet_path.chmod(0o644)
    fleet_path.write_bytes(sentinel._canonical_json(fleet_contract))
    fleet_path.chmod(0o444)
    invented_fleet_hash = sentinel._sha256(fleet_path)
    pins["fleet_contract_sha256"] = invented_fleet_hash
    for path in (
        pins_path,
        Path(manifest["state_root"]) / "immutable_pins.json",
    ):
        path.chmod(0o644)
        path.write_bytes(sentinel._canonical_json(pins))
        path.chmod(0o444)
    control_path = Path(manifest["state_root"]) / "control.json"
    control = {
        "immutable": pins,
        "immutable_sha256": sentinel._capacity_value_sha256(pins),
    }
    control_path.write_bytes(sentinel._canonical_json(control))

    def mutate(evidence: dict[str, Any]) -> None:
        evidence["fleet"]["control_immutable_sha256"] = (
            sentinel._capacity_value_sha256(pins)
        )
        evidence["fleet"]["fleet_contract_sha256"] = invented_fleet_hash

    _rehash_capacity_receipt(capacity, mutate)
    _assert_capacity_corruption(
        tmp_path,
        monkeypatch,
        verified,
        capacity,
        expected_error="sealed release authority drifted",
    )


def test_classifier_requires_exact_sentinel_to_be_running() -> None:
    jobs = [
        {
            "name": "stage",
            "dependencies": [],
            "dependency_type": "afterok",
            "state": "COMPLETED",
            "active": False,
            "exit_code": "0:0",
        },
        {
            "name": "failure_sentinel",
            "dependencies": ["stage"],
            "dependency_type": "afterany",
            "state": "COMPLETED",
            "active": False,
            "exit_code": "0:0",
        },
    ]
    with pytest.raises(sentinel.SentinelError, match="must be RUNNING"):
        sentinel.classify_recovery_jobs(jobs)


@pytest.mark.parametrize(
    ("state", "reason", "start", "elapsed"),
    [
        ("FAILED", "NonZeroExitCode", "2026-07-23T00:00:00", "00:00:01"),
        ("OUT_OF_MEMORY", "OutOfMemory", "2026-07-23T00:00:00", "00:00:01"),
        ("TIMEOUT", "TimeLimit", "2026-07-23T00:00:00", "01:00:00"),
        ("CANCELLED", "CancelledByUser", "Unknown", "00:00:00"),
        ("CANCELLED", "DependencyNeverSatisfied", "Unknown", "00:00:00"),
    ],
)
def test_deterministic_or_direct_failure_requires_superseding_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    reason: str,
    start: str,
    elapsed: str,
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_state=state,
        first_exit="1:0",
        first_reason=reason,
        first_start=start,
        first_elapsed=elapsed,
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=False,
    )
    assert report["classification"] == "requires_superseding_release"
    assert report["outcome"]["same_generation_repair_allowed"] is False


def test_exact_qualification_capacity_exit_has_dedicated_transition_disposition() -> None:
    failed = {
        "name": "throughput_qualification",
        "job_id": "18",
        "dependencies": [],
        "dependency_type": "afterok",
        "comment": "qualification-comment",
        "state": "FAILED",
        "active": False,
        "exit_code": "76:0",
        "reason": "NonZeroExitCode",
        "start": "2026-07-23T00:00:00",
        "elapsed": "00:10:00",
    }
    cancelled = {
        "name": "controller_drill",
        "job_id": "19",
        "dependencies": ["throughput_qualification"],
        "dependency_type": "afterok",
        "comment": "controller-comment",
        "state": "CANCELLED",
        "active": False,
        "exit_code": "0:0",
        "reason": "DependencyNeverSatisfied",
        "start": "Unknown",
        "elapsed": "00:00:00",
    }
    aggregate = {
        "name": "failure_sentinel",
        "job_id": "42",
        "dependencies": [
            "throughput_qualification",
            "controller_drill",
        ],
        "dependency_type": "afterany",
        "comment": "sentinel-comment",
        "state": "RUNNING",
        "active": True,
        "exit_code": None,
        "reason": "",
        "start": "2026-07-23T00:11:00",
        "elapsed": "00:00:01",
    }
    binding = {
        "failed_job_id": "18",
        "failed_comment": "qualification-comment",
    }

    outcome = sentinel.classify_recovery_jobs(
        [failed, cancelled, aggregate],
        qualification_capacity_failure=binding,
    )
    assert (
        outcome["classification"]
        == "qualification_capacity_transition_required"
    )
    assert outcome["qualification_capacity_transition_roots"] == [
        "throughput_qualification"
    ]
    assert outcome["same_generation_repair_allowed"] is False
    assert outcome["requires_superseding_release"] is False

    for exit_code, supplied in (("2:0", binding), ("76:0", None)):
        failed["exit_code"] = exit_code
        rejected = sentinel.classify_recovery_jobs(
            [failed, cancelled, aggregate],
            qualification_capacity_failure=supplied,
        )
        assert rejected["classification"] == "requires_superseding_release"
        assert rejected["same_generation_repair_allowed"] is False


def test_exit76_requires_exact_sealed_current_failure_preimage(
    tmp_path: Path,
) -> None:
    results_root = (tmp_path / "results").resolve()
    readiness_root = (
        results_root / "recovery" / "schema5-v1" / "readiness"
    )
    server_pool_root = results_root / "server_pools" / "schema5-v1"
    base = readiness_root / sentinel.QUALIFICATION_ROOT_NAME
    pointer_root = base / sentinel.QUALIFICATION_POINTER_ROOT_NAME
    server_pool_root.mkdir(parents=True)
    pointer_root.mkdir(parents=True)
    readiness_generation = {
        "catalog_id": "1" * 64,
        "marker_path": str(
            (readiness_root / "TRUSTED_GENERATION.json").resolve()
        ),
        "marker_sha256": "2" * 64,
        "inventory_sha256": "3" * 64,
        "catalog_payload_sha256": "4" * 64,
        "allowed_generation_tuple_count": 24,
        "release_fleet_contract_sha256": "5" * 64,
        "fleet_contract_sha256": "6" * 64,
        "capacity_generation": 1,
        "rollout_generation": 1,
    }
    attempt_id = (
        f"g000001-c000001-{readiness_generation['catalog_id']}"
    )
    attempt_root = (
        base / sentinel.QUALIFICATION_ATTEMPT_ROOT_NAME / attempt_id
    )
    run_root = (
        results_root
        / sentinel.QUALIFICATION_RUN_ROOT_NAME
        / attempt_id
        / sentinel.QUALIFICATION_ROOT_NAME
    )
    attempt_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    pointer_path = pointer_root / f"000001-{attempt_id}.json"

    def identified(
        value: dict[str, Any], identity_field: str
    ) -> dict[str, Any]:
        result = dict(value)
        result[identity_field] = sentinel._sha256_bytes(
            sentinel._canonical_json(result)
        )
        return result

    pointer = identified(
        {
            "schema_version": 1,
            "protocol": sentinel.QUALIFICATION_POINTER_PROTOCOL,
            "chain_id": "a" * 64,
            "attempt_ordinal": 1,
            "attempt_id": attempt_id,
            "attempt_root": str(attempt_root),
            "run_root": str(run_root),
            "dispatcher_state": str(attempt_root / "dispatcher"),
            "readiness_generation": readiness_generation,
            "predecessor": None,
            "additive_retry": None,
            "created_at": sentinel._utc(1_000.0),
            "created_timestamp": 1_000.0,
        },
        "pointer_id",
    )
    pointer_path.write_bytes(sentinel._canonical_json(pointer))
    pointer_path.chmod(0o444)
    current = identified(
        {
            "schema_version": 1,
            "protocol": sentinel.QUALIFICATION_CURRENT_PROTOCOL,
            "attempt_id": attempt_id,
            "pointer": str(pointer_path),
            "pointer_sha256": sentinel._sha256(pointer_path),
            "pointer_id": pointer["pointer_id"],
        },
        "current_id",
    )
    current_path = base / sentinel.QUALIFICATION_CURRENT_NAME
    current_path.write_bytes(sentinel._canonical_json(current))
    current_path.chmod(0o444)
    attempt_binding = {
        "path": str(pointer_path),
        "sha256": sentinel._sha256(pointer_path),
        "pointer_id": pointer["pointer_id"],
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "run_root": str(run_root),
        "rollout_generation": 1,
        "capacity_generation": 1,
        "trusted_generation_catalog_id": readiness_generation["catalog_id"],
    }
    failure_reason = "qualification throughput is below threshold"
    observation_path = (
        attempt_root / "observations" / "OBSERVATION_000000.json"
    )
    observation_path.parent.mkdir()
    observation = identified(
        {
            "schema_version": 3,
            "protocol": "test-qualification-observation-v3",
        },
        "observation_id",
    )
    observation_path.write_bytes(sentinel._canonical_json(observation))
    observation_path.chmod(0o444)
    drain_intent = identified(
        {
            "schema_version": (
                sentinel.QUALIFICATION_FAILURE_SCHEMA_VERSION
            ),
            "protocol": sentinel.QUALIFICATION_FAILURE_DRAIN_PROTOCOL,
            "qualification_intent_id": "7" * 64,
            "reason": failure_reason,
            "admission_closed": True,
            "state": "draining",
            "requested_at": sentinel._utc(2_000.0),
            "requested_timestamp": 2_000.0,
        },
        "failure_drain_intent_id",
    )
    drain_intent_path = (
        attempt_root / sentinel.QUALIFICATION_FAILURE_DRAIN_NAME
    )
    drain_intent_path.write_bytes(
        sentinel._canonical_json(drain_intent)
    )
    drain_intent_path.chmod(0o444)
    run_member = run_root / "fixture.jsonl"
    run_member.write_text('{"qid":"fixture"}\n', encoding="utf-8")
    run_member.chmod(0o444)
    run_root.chmod(0o555)
    cycle_inventory = sentinel._qualification_tree_inventory(run_root)
    admission_certificate = identified(
        {
            "schema_version": (
                sentinel.QUALIFICATION_FAILURE_SCHEMA_VERSION
            ),
            "protocol": (
                sentinel.QUALIFICATION_PREFLIGHT_CERTIFICATE_PROTOCOL
            ),
            "passed": True,
            "capacity_generation": 1,
            "proposed_effective_fleet_contract_sha256": (
                readiness_generation["fleet_contract_sha256"]
            ),
            "effective_logical_replicas": 22,
            "effective_active_gpus": 24,
            "selected_cell_count": 384,
            "wave": {
                "passed": True,
                "selected_cell_count": 384,
                "target_active_cells": 384,
                "shortfall_cells": 0,
            },
        },
        "certificate_id",
    )
    admission_certificate_path = (
        readiness_root / "PREFLIGHT_CAPACITY_CERTIFICATE.json"
    )
    admission_certificate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    admission_certificate_path.write_bytes(
        sentinel._canonical_json(admission_certificate)
    )
    admission_certificate_path.chmod(0o444)
    failure = identified(
        {
            "schema_version": (
                sentinel.QUALIFICATION_FAILURE_SCHEMA_VERSION
            ),
            "protocol": sentinel.QUALIFICATION_FAILURE_PROTOCOL,
            "passed": False,
            "intent_id": "7" * 64,
            "attempt": attempt_binding,
            "readiness_generation": readiness_generation,
            "reason": failure_reason,
            "admission_capacity_certificate": {
                "path": str(admission_certificate_path),
                "sha256": sentinel._sha256(
                    admission_certificate_path
                ),
                "certificate_id": admission_certificate[
                    "certificate_id"
                ],
                "capacity_generation": 1,
                "effective_fleet_contract_sha256": (
                    readiness_generation["fleet_contract_sha256"]
                ),
                "effective_logical_replicas": 22,
                "effective_active_gpus": 24,
                "wave_passed": True,
                "selected_cell_count": 384,
                "target_cell_count": 384,
                "shortfall_cells": 0,
                "theoretical_packing_upper_bound": 384,
            },
            "additive_scaling_requirement": {
                "serving_profile": "8B",
                "server_pool_root": str(server_pool_root),
                "backlog_fanout_work": 100,
                "live_replicas": 1,
                "backlog_work_per_replica": 100.0,
                "additional_replicas": 1,
                "tensor_parallel_size": 1,
                "additional_gpus": 1,
                "requirement": "add one replica (1 GPU)",
                "capacity_mutated": False,
            },
            "scheduler_capacity_mutated": False,
            "failure_drain_intent": {
                "path": str(drain_intent_path),
                "sha256": sentinel._sha256(drain_intent_path),
                "failure_drain_intent_id": drain_intent[
                    "failure_drain_intent_id"
                ],
                "drain_observation": {
                    "path": str(observation_path),
                    "sha256": sentinel._sha256(observation_path),
                    "observation_id": observation["observation_id"],
                },
            },
            "cycle_run_roots": [
                {
                    "cycle_index": 0,
                    "cycle_id": "8" * 64,
                    "run_id": sentinel.QUALIFICATION_ROOT_NAME,
                    "semantic_reference": True,
                    "estimand_excluded": True,
                    "primary_analysis_eligible": False,
                    **cycle_inventory,
                }
            ],
            "refill_reconciliations": [],
            "rerun_requirement": "fresh exact additive generation required",
        },
        "failure_id",
    )
    failure_path = attempt_root / sentinel.QUALIFICATION_FAILURE_NAME
    failure_path.write_bytes(sentinel._canonical_json(failure))
    failure_path.chmod(0o444)
    observation_path.parent.chmod(0o555)
    attempt_root.chmod(0o555)
    run_root.chmod(0o555)
    manifest = {
        "chain_id": "a" * 64,
        "readiness_root": str(readiness_root),
        "results_root": str(results_root),
        "server_pool_root": str(server_pool_root),
    }
    receipt = {"receipt_id": "b" * 64}
    verified = {"manifest": manifest, "submission_receipt": receipt}
    failed = {
        "name": "throughput_qualification",
        "job_id": "18",
        "dependencies": [],
        "dependency_type": "afterok",
        "comment": "qualification-comment",
        "state": "FAILED",
        "active": False,
        "exit_code": "76:0",
        "reason": "NonZeroExitCode",
        "start": "2026-07-23T00:00:00",
        "elapsed": "00:10:00",
    }
    cancelled = {
        "name": "controller_drill",
        "job_id": "19",
        "dependencies": ["throughput_qualification"],
        "dependency_type": "afterok",
        "comment": "controller-comment",
        "state": "CANCELLED",
        "active": False,
        "exit_code": "0:0",
        "reason": "DependencyNeverSatisfied",
        "start": "Unknown",
        "elapsed": "00:00:00",
    }
    aggregate = {
        "name": "failure_sentinel",
        "job_id": "42",
        "dependencies": [
            "throughput_qualification",
            "controller_drill",
        ],
        "dependency_type": "afterany",
        "comment": "sentinel-comment",
        "state": "RUNNING",
        "active": True,
        "exit_code": None,
        "reason": "",
        "start": "2026-07-23T00:11:00",
        "elapsed": "00:00:01",
    }
    jobs = [failed, cancelled, aggregate]
    binding = sentinel._qualification_capacity_failure_binding(
        verified=verified,
        jobs=jobs,
    )
    assert binding is not None
    assert binding["failure_id"] == failure["failure_id"]
    assert binding["attempt"] == attempt_binding
    assert (
        sentinel.classify_recovery_jobs(
            jobs,
            qualification_capacity_failure=binding,
        )["classification"]
        == "qualification_capacity_transition_required"
    )

    # Even a fully rehashed scientific/protocol drift never inherits exit-76
    # repair authority; it deterministically falls back to superseding release.
    attempt_root.chmod(0o755)
    failure_path.chmod(0o644)
    invalid = dict(failure)
    invalid.pop("failure_id")
    invalid["reason"] = "semantic integrity incident"
    invalid = identified(invalid, "failure_id")
    failure_path.write_bytes(sentinel._canonical_json(invalid))
    failure_path.chmod(0o444)
    attempt_root.chmod(0o555)
    observed = sentinel._observe_qualification_capacity_failure(
        verified=verified,
        jobs=jobs,
    )
    assert observed is None
    rejected = sentinel.classify_recovery_jobs(
        jobs,
        qualification_capacity_failure=observed,
    )
    assert rejected["classification"] == "requires_superseding_release"

    # A valid, re-sealed failure marker cannot authorize repair if one of
    # its supposedly immutable scientific cycle roots changed afterward.
    attempt_root.chmod(0o755)
    failure_path.chmod(0o644)
    failure_path.write_bytes(sentinel._canonical_json(failure))
    failure_path.chmod(0o444)
    run_root.chmod(0o755)
    run_member.chmod(0o644)
    run_member.write_text('{"qid":"drifted"}\n', encoding="utf-8")
    run_member.chmod(0o444)
    run_root.chmod(0o555)
    attempt_root.chmod(0o555)
    observed = sentinel._observe_qualification_capacity_failure(
        verified=verified,
        jobs=jobs,
    )
    assert observed is None
    rejected = sentinel.classify_recovery_jobs(
        jobs,
        qualification_capacity_failure=observed,
    )
    assert rejected["classification"] == "requires_superseding_release"


def test_scheduler_query_failure_or_duplicate_prevents_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    for index, scheduler in enumerate(
        (
            FakeScheduler(verified, fail_sacct=True),
            FakeScheduler(verified, duplicate_sacct=True),
        )
    ):
        output = tmp_path / f"out-{index}"
        with pytest.raises(sentinel.SentinelError):
            _run(
                tmp_path,
                monkeypatch,
                verified,
                scheduler,
                apply=True,
                output=output,
            )
        assert not (output / sentinel.COMPLETE_MARKER_NAME).exists()
        assert not (output / sentinel.SCHEDULER_EVIDENCE_NAME).exists()


def test_failed_mail_is_persisted_retried_and_only_confirmed_send_dedupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_state="NODE_FAIL",
        first_exit="1:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    output = tmp_path / "out"
    clock = FakeClock()
    calls: list[str] = []

    def failed_mail(argv: Sequence[str], _body: str) -> subprocess.CompletedProcess[str]:
        state = json.loads((output / sentinel.MAIL_STATE_NAME).read_text())
        assert state["in_flight_token"]
        calls.append("failed")
        return subprocess.CompletedProcess(list(argv), 1, "", "temporary failure")

    first = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=output,
        mail_runner=failed_mail,
        clock=clock,
        sleeper=clock.sleep,
        maximum_mail_attempts=1,
    )
    assert first["alert_delivered"] is False
    state = json.loads((output / sentinel.MAIL_STATE_NAME).read_text())
    assert state["attempt_count"] == 1
    assert state["next_retry_timestamp"] == pytest.approx(clock.value + 5)

    clock.value += 5

    def delivered_mail(
        argv: Sequence[str], _body: str
    ) -> subprocess.CompletedProcess[str]:
        calls.append("delivered")
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    second = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("sealed evidence was not reused"),  # type: ignore[arg-type]
        apply=True,
        output=output,
        mail_runner=delivered_mail,
        clock=clock,
        sleeper=clock.sleep,
        maximum_mail_attempts=1,
    )
    assert second["alert_delivered"] is True
    assert second["mail_attempt_count"] == 2
    delivered_state = json.loads(
        (output / sentinel.MAIL_STATE_NAME).read_text()
    )
    assert delivered_state["delivery_receipt"] == {
        "protocol": "mail-command-returncode-v1",
        "attempt_count": 2,
        "returncode": 0,
        "completed_timestamp": delivered_state["delivered_timestamp"],
        "in_flight_token_sha256": delivered_state["delivery_receipt"][
            "in_flight_token_sha256"
        ],
    }
    assert len(
        delivered_state["delivery_receipt"]["in_flight_token_sha256"]
    ) == 64

    third = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("sealed evidence was not reused"),  # type: ignore[arg-type]
        apply=True,
        output=output,
        mail_runner=lambda _argv, _body: pytest.fail(
            "confirmed delivery was not deduplicated"
        ),
        clock=clock,
        sleeper=clock.sleep,
        maximum_mail_attempts=1,
    )
    assert third["alert_delivered"] is True
    assert calls == ["failed", "delivered"]


def test_aggregate_marker_never_waits_for_email_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_state="NODE_FAIL",
        first_exit="1:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    output = tmp_path / "aggregate-mail-outage"
    clock = FakeClock()
    calls: list[str] = []
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=output,
        mail_runner=lambda argv, _body: (
            calls.append("failed")
            or subprocess.CompletedProcess(
                list(argv), 1, "", "mail transport unavailable"
            )
        ),
        clock=clock,
        sleeper=lambda _seconds: pytest.fail(
            "aggregate sentinel slept for email retry"
        ),
        maximum_mail_attempts=5,
    )
    assert clock.value == 1_000.0
    assert calls == ["failed"]
    assert report["classification"] == "transient_repairable"
    assert report["same_generation_repair_allowed"] is True
    assert report["mail_attempt_count"] == 1
    assert report["alert_delivered"] is False
    assert report["alert_retry_pending"] is True
    assert report["mail_next_retry_timestamp"] == pytest.approx(1_005.0)
    assert (output / sentinel.COMPLETE_MARKER_NAME).is_file()
    evidence = json.loads(
        (output / sentinel.SCHEDULER_EVIDENCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["outcome"]["classification"] == "transient_repairable"
    assert evidence["outcome"]["same_generation_repair_allowed"] is True
    mail = json.loads(
        (output / sentinel.MAIL_STATE_NAME).read_text(encoding="utf-8")
    )
    assert mail["next_retry_timestamp"] == pytest.approx(1_005.0)
    assert sentinel.SYNCHRONOUS_MAIL_ATTEMPT_LIMIT == 1
    assert sentinel.MAIL_COMMAND_TIMEOUT_SECONDS <= 30.0

    repeated = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("sealed aggregate evidence was requeried"),  # type: ignore[arg-type]
        apply=True,
        output=output,
        mail_runner=lambda _argv, _body: pytest.fail(
            "ineligible aggregate mail retry was attempted"
        ),
        clock=clock,
        sleeper=lambda _seconds: pytest.fail(
            "aggregate successor slept for email retry"
        ),
        maximum_mail_attempts=5,
    )
    assert repeated["status"] == "already_complete"
    assert repeated["classification"] == report["classification"]
    assert repeated["mail_attempt_count"] == 1
    assert clock.value == 1_000.0


def test_forged_delivered_mail_boolean_without_confirmed_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(
        verified,
        first_state="NODE_FAIL",
        first_exit="1:0",
        second_state="CANCELLED",
        second_reason="DependencyNeverSatisfied",
        second_start="Unknown",
        second_elapsed="00:00:00",
    )
    output = tmp_path / "out"
    clock = FakeClock()
    _run(
        tmp_path,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=output,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 1, "", "temporary failure"
        ),
        clock=clock,
        sleeper=clock.sleep,
        maximum_mail_attempts=1,
    )
    state_path = output / sentinel.MAIL_STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["delivered"] = True
    state["delivered_at"] = state["attempted_at"]
    state["delivered_timestamp"] = state["attempted_timestamp"]
    state["next_retry_at"] = None
    state["next_retry_timestamp"] = None
    state["error"] = None
    state_path.write_bytes(sentinel._canonical_json(state))

    with pytest.raises(sentinel.SentinelError, match="confirmed-send receipt"):
        _run(
            tmp_path,
            monkeypatch,
            verified,
            lambda _argv: pytest.fail("sealed evidence re-queried Slurm"),  # type: ignore[arg-type]
            apply=True,
            output=output,
            mail_runner=lambda _argv, _body: pytest.fail(
                "forged delivery was deduplicated"
            ),
            clock=clock,
            sleeper=clock.sleep,
            maximum_mail_attempts=1,
        )


def test_crash_after_scheduler_evidence_reuses_exact_preimage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified(tmp_path)
    scheduler = FakeScheduler(verified)
    output = tmp_path / "out"
    original = sentinel._atomic_json
    crashed = False

    def crash_before_marker(path: Path, payload: object, *, mode: int) -> None:
        nonlocal crashed
        if path.name == sentinel.COMPLETE_MARKER_NAME and not crashed:
            crashed = True
            raise RuntimeError("simulated marker boundary crash")
        original(path, payload, mode=mode)

    monkeypatch.setattr(sentinel, "_atomic_json", crash_before_marker)
    with pytest.raises(RuntimeError, match="simulated"):
        _run(
            tmp_path,
            monkeypatch,
            verified,
            scheduler,
            apply=True,
            output=output,
        )
    evidence_bytes = (output / sentinel.SCHEDULER_EVIDENCE_NAME).read_bytes()
    assert not (output / sentinel.COMPLETE_MARKER_NAME).exists()

    monkeypatch.setattr(sentinel, "_atomic_json", original)
    report = _run(
        tmp_path,
        monkeypatch,
        verified,
        lambda _argv: pytest.fail("crash recovery requeried Slurm"),  # type: ignore[arg-type]
        apply=True,
        output=output,
    )
    assert report["status"] == "complete"
    assert (output / sentinel.SCHEDULER_EVIDENCE_NAME).read_bytes() == evidence_bytes
