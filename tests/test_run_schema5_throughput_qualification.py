"""Focused tests for the isolated schema-5 throughput qualification producer."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace

import pytest

from agents_scaling.serving import fleet_contract as fleet_contract_runtime
from scripts import render_schema5_recovery_chain_v12 as renderer
from scripts import run_schema5_throughput_qualification as qualification


COMMIT = "1" * 40
TAG_OBJECT = "2" * 40
MANIFEST_SHA256 = "a" * 64


def _sealed(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(qualification._canonical_bytes(value))
    path.chmod(0o444)
    return path


def _prerequisite(
    recovery_root: Path,
    name: str,
    *,
    protocol: str,
    identity_field: str,
    identity: str,
) -> dict[str, object]:
    return {
        "marker": str((recovery_root / name).resolve()),
        "marker_sha256": "b" * 64,
        "marker_size": 123,
        "protocol": protocol,
        identity_field: identity,
    }


def _protected_capacity_prerequisite(
    recovery_root: Path,
    *,
    source_tree_sha256: str = "5" * 64,
    dispatcher_source_sha256: str = "4" * 64,
    qualification_runner_source_sha256: str = "6" * 64,
) -> dict[str, object]:
    fleet_fixture = {"kind": "zero-delta-fleet-test-fixture"}
    base_fleet_path = _sealed(
        recovery_root / "contracts" / "base-fleet.json",
        fleet_fixture,
    )
    effective_fleet_path = _sealed(
        recovery_root / "contracts" / "effective-fleet.json",
        fleet_fixture,
    )
    base_fleet_sha256 = qualification._sha256_file(base_fleet_path)
    effective_fleet_sha256 = qualification._sha256_file(
        effective_fleet_path
    )
    fixture = _protected_capacity_fixture_payloads(
        base_fleet_sha256,
        effective_fleet_sha256,
        source_tree_sha256,
        dispatcher_source_sha256,
        qualification_runner_source_sha256,
    )
    certificate_path = _sealed(
        recovery_root
        / "readiness"
        / qualification.PREFLIGHT_CAPACITY_CERTIFICATE_NAME,
        fixture["certificate"],
    )
    certificate_sha256 = qualification._sha256_file(certificate_path)
    marker_identity = {
        "schema_version": qualification.protected_capacity.SCHEMA_VERSION,
        "protocol": qualification.protected_capacity.PROTOCOL,
        "passed": True,
        "release_id": qualification.protected_capacity.RELEASE_ID,
        "release_tag": qualification.protected_capacity.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": qualification.protected_capacity.CHAIN_NAMESPACE,
        "source_tree_sha256": source_tree_sha256,
        "dispatcher_source_sha256": dispatcher_source_sha256,
        "qualification_runner_source_sha256": (
            qualification_runner_source_sha256
        ),
        "capacity_generation": 1,
        "base_fleet_contract_path": str(base_fleet_path.resolve()),
        "base_fleet_contract_sha256": base_fleet_sha256,
        "effective_fleet_contract_path": str(
            effective_fleet_path.resolve()
        ),
        "effective_fleet_contract_sha256": effective_fleet_sha256,
        "additive_overlay_contract_path": str(
            effective_fleet_path.resolve()
        ),
        "additive_overlay_contract_sha256": effective_fleet_sha256,
        "static_feasibility_certificate": {
            "path": str(certificate_path.resolve()),
            "sha256": certificate_sha256,
            "certificate_id": fixture["certificate"]["certificate_id"],
        },
        "static_feasibility_wave_passed": False,
        "static_feasibility_configured_client_ceiling": 384,
        "static_feasibility_certified_saturation_target": 278,
        "static_feasibility_selected_cell_count": 278,
        "static_feasibility_target_cell_count": 384,
        "static_feasibility_shortfall_cells": 106,
        "base_active_logical_replicas": 22,
        "base_active_gpus": 24,
        "base_active_topology": fixture["base_topology"],
        "base_active_topology_sha256": fixture[
            "base_topology_sha256"
        ],
        "additive_reserved_logical_replicas": 0,
        "additive_reserved_gpus": 0,
        "additive_reserved_tp1_replicas": 0,
        "additive_reserved_tp2_replicas": 0,
        "additive_reserved_topology": fixture["additive_topology"],
        "additive_reserved_topology_sha256": fixture[
            "additive_topology_sha256"
        ],
        "effective_active_logical_replicas": 22,
        "effective_active_gpus": 24,
        "effective_active_topology": fixture["effective_topology"],
        "effective_active_topology_sha256": fixture[
            "effective_topology_sha256"
        ],
        "retained_warm_turnover_job_elements": 3,
        "retained_warm_turnover_gpus": 4,
        "retained_warm_turnover_tp1_allocations": 2,
        "retained_warm_turnover_tp2_allocations": 1,
        "retained_warm_turnover_topology": fixture["warm_topology"],
        "retained_warm_turnover_topology_sha256": fixture[
            "warm_topology_sha256"
        ],
        "attested_total_gpus": 28,
        "job_element_accounting": {
            "cell_job_elements": 384,
            "active_server_job_elements": 22,
            "warm_turnover_job_elements": 3,
            "controller_monitor_other_held_job_elements": 39,
            "total_non_cell_reserve_job_elements": 64,
            "total_canary_job_elements": 448,
        },
        "active_gpus": 24,
        "warm_headroom_gpus": 4,
        "cell_ceiling": 384,
        "reserve_jobs": 64,
        "submit_headroom": 448,
        "cpu": 384,
        "memory_mib": 384 * 4_096,
        "preempt_type": "preempt/partition_prio",
        "capacity_source": qualification.protected_capacity.CAPACITY_SOURCE,
        "scheduler_cluster": "test_cluster",
        "scheduler_account": "test_account",
        "scheduler_user": "test_user",
        "scheduler_max_jobs": 409,
        "scheduler_max_submit_jobs": 500,
        "running_scientific_jobs": 409,
        "minimum_scientific_wall_seconds": 86_400,
        "scientific_qos_contracts": [
            {
                "qos": "gpu_qos",
                "max_wall_seconds": 86_400,
                "max_jobs_per_user": 25,
                "max_submit_jobs_per_user": 500,
                "required_wall_seconds": 86_400,
                "required_running_jobs": 25,
                "required_submit_jobs": 25,
            },
            {
                "qos": "normal",
                "max_wall_seconds": 86_400,
                "max_jobs_per_user": 384,
                "max_submit_jobs_per_user": 500,
                "required_wall_seconds": 43_200,
                "required_running_jobs": 384,
                "required_submit_jobs": 423,
            },
        ],
        "partition_cpus": 384,
        "partition_memory_mib": 384 * 4_096,
        "partition_gpus": 28,
        "fleet_contract_sha256": effective_fleet_sha256,
        "active_fleet_topology_sha256": fixture[
            "effective_topology_sha256"
        ],
        "scientific_server_preempt_mode": "OFF",
        "scientific_client_preempt_mode": "OFF",
        "squeue_complete": True,
        "sacct_complete": True,
        "scheduler_evidence_id": "3" * 64,
        "scheduler_evidence_sha256": "4" * 64,
        "canary_id": "5" * 64,
        "canary_evidence_sha256": "6" * 64,
        "scientific_server_placements": [
            {
                "partition": "gpu_protected",
                "qos": "gpu_qos",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "cluster",
                "base_active_gpus": 24,
                "reserved_additive_gpus": 0,
                "effective_active_gpus": 24,
                "retained_warm_turnover_gpus": 4,
                "attested_total_gpus": 28,
                "partition_cpus": 4096,
                "partition_memory_mib": 33_554_432,
                "partition_gpus": 64,
                "partition_nodes": 8,
            }
        ],
        "scientific_client_placements": [
            {
                "partition": "ou_bcs_normal",
                "qos": "normal",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "cluster",
                "slots": 384,
                "cpus": 384,
                "memory_mib": 384 * 4_096,
                "reserve_jobs": 64,
                "submit_headroom": 448,
            }
        ],
    }
    marker = qualification._with_identity(marker_identity, "marker_id")
    path = _sealed(
        recovery_root / renderer.PROTECTED_CAPACITY_MARKER_NAME,
        marker,
    )
    raw = path.read_bytes()
    return {
        "marker": str(path.resolve()),
        "marker_sha256": qualification._sha256_bytes(raw),
        "marker_size": len(raw),
        "protocol": renderer.PROTECTED_CAPACITY_PROTOCOL,
        "marker_id": marker["marker_id"],
    }


def _chain_manifest(tmp_path: Path) -> Path:
    results_root = (tmp_path / "results").resolve()
    recovery_root = results_root / "recovery" / "schema5-v1"
    readiness_root = recovery_root / "readiness"
    release_root = recovery_root / "release"
    release_worktree = release_root / "worktree"
    dispatcher_source = _sealed(
        release_worktree / "slurm" / "dispatch_sweeps.py",
        {"fixture": "exact dispatcher source"},
    )
    qualification_runner_source = _sealed(
        release_worktree
        / "scripts"
        / "run_schema5_throughput_qualification.py",
        {"fixture": "exact qualification runner source"},
    )
    source_tree_sha256 = qualification.control.sha256_tree(
        release_worktree
    )
    control_fragment = {
        "release_id": renderer.RELEASE_ID,
        "release_worktree": str(release_worktree),
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "source_tree_sha256": source_tree_sha256,
    }
    release_identity = {
        "schema_version": 3,
        "release_id": renderer.RELEASE_ID,
        "git": {
            "git_commit": COMMIT,
            "git_tag": renderer.RELEASE_TAG,
            "git_tag_object": TAG_OBJECT,
            "source_tree_sha256": source_tree_sha256,
        },
        "release_worktree": str(release_worktree),
        "worktree_sealed_read_only": True,
        "control_pin_fragment": control_fragment,
    }
    release_identity_path = _sealed(
        release_root
        / "identity"
        / "release_identity.schema5-v1.json",
        release_identity,
    )
    release_identity_checksum = Path(
        str(release_identity_path) + ".sha256"
    )
    release_identity_checksum.write_text(
        f"{qualification._sha256_file(release_identity_path)}  "
        f"{release_identity_path.name}\n",
        encoding="ascii",
    )
    release_identity_checksum.chmod(0o444)
    identity = {
        "schema_version": renderer.CHAIN_SCHEMA_VERSION,
        "protocol": "schema5-v1.2-r11-recovery-chain",
        "namespace": renderer.CHAIN_NAMESPACE,
        "release_id": renderer.RELEASE_ID,
        "release_tag": renderer.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "results_root": str(results_root),
        "recovery_root": str(recovery_root),
        "readiness_root": str(readiness_root),
        "state_root": str(
            results_root / qualification.control.CONTROL_STATE_DIRNAME
        ),
        "server_pool_root": str(
            results_root / "server_pools" / "schema5-v1"
        ),
        "release_root": str(release_root),
        "hf_home": str((tmp_path / "hf").resolve()),
        "prerequisite_evidence": {
            "protected_capacity": _protected_capacity_prerequisite(
                recovery_root,
                source_tree_sha256=source_tree_sha256,
                dispatcher_source_sha256=(
                    qualification._sha256_file(dispatcher_source)
                ),
                qualification_runner_source_sha256=(
                    qualification._sha256_file(
                        qualification_runner_source
                    )
                ),
            ),
        },
    }
    manifest = qualification._with_identity(identity, "chain_id")
    return _sealed(tmp_path / "CHAIN.json", manifest)


def _guard(*, rollout_generation: int = 0) -> dict[str, object]:
    value: dict[str, object] = {
        "immutable_sha256": "f" * 64,
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": rollout_generation,
        "production_run_ids": list(qualification.control.REQUIRED_RUNS),
        "admission": {"current_ceiling": 24},
        "admission_ramp": {"current_ceiling": 24},
        "admission_safety_hold": {"active": False},
    }
    value["guard_sha256"] = qualification._sha256_bytes(
        qualification._canonical_bytes(value)
    )
    return value


def _readiness_generation(tmp_path: Path) -> dict[str, object]:
    return {
        "catalog_id": "7" * 64,
        "marker_path": str((tmp_path / "TRUSTED_GENERATION.json").resolve()),
        "marker_sha256": "8" * 64,
        "inventory_sha256": "9" * 64,
        "catalog_payload_sha256": "a" * 64,
        "allowed_generation_tuple_count": 24,
        "release_fleet_contract_sha256": "b" * 64,
        "fleet_contract_sha256": "c" * 64,
        "capacity_generation": 1,
        "rollout_generation": 1,
    }


def _context_and_intent(
    tmp_path: Path,
    *,
    write_execution_authority: bool = True,
) -> tuple[qualification.QualificationContext, dict[str, object]]:
    base = qualification.load_qualification_context(
        _chain_manifest(tmp_path), verify_chain=False
    )
    readiness = _readiness_generation(tmp_path)
    readiness["release_fleet_contract_sha256"] = (
        base.protected_capacity_contract.base_fleet_contract_sha256
    )
    readiness["fleet_contract_sha256"] = (
        base.protected_capacity_contract.effective_fleet_contract_sha256
    )
    context = qualification.create_or_load_attempt_context(
        base,
        control_value={},
        readiness_generation=readiness,
        now=850.0,
    )
    intent = qualification._with_identity(
        qualification._intent_identity(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            readiness_generation=readiness,
            control_guard=_guard(),
            admission_capacity_certificate=(
                qualification
                ._validated_admission_capacity_certificate_binding(
                    context
                )
            ),
            created_timestamp=900.0,
        ),
        "intent_id",
    )
    context.qualification_root.mkdir(parents=True, exist_ok=True)
    context.run_root.mkdir(parents=True, exist_ok=True)
    qualification._write_once(
        context.qualification_root / qualification.INTENT_NAME,
        intent,
        description="test intent",
    )
    qualification._write_once(
        context.qualification_root / qualification.PLAN_NAME,
        qualification.build_load_plan(),
        description="test plan",
    )
    if write_execution_authority:
        authority = qualification._with_identity(
            {
                "schema_version": (
                    qualification.EXECUTION_AUTHORITY_SCHEMA_VERSION
                ),
                "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
                "intent_id": intent["intent_id"],
                "chain_id": context.chain_id,
                "run_id": qualification.QUALIFICATION_RUN_ID,
                "run_root": str(context.run_root),
                "readiness_generation": dict(readiness),
            },
            "authority_id",
        )
        qualification._write_once(
            context.qualification_root
            / qualification.EXECUTION_AUTHORITY_NAME,
            authority,
            description="test execution authority",
        )
    return context, intent


def _progress(total: int) -> dict[str, int]:
    labels = qualification.expected_stratum_labels()
    quotient, remainder = divmod(total, len(labels))
    return {
        label: quotient + int(index < remainder)
        for index, label in enumerate(labels)
    }


def _job(sequence: int) -> dict[str, str]:
    return {
        "job_id": str(10_000 + sequence),
        "job_name": f"asys-dispatch-{sequence:010d}",
        "state": "RUNNING",
        "comment": f"asys-schema5-intent:test-{sequence}",
        "command": f"/sealed/batch-{sequence}.sbatch",
        "source": "squeue",
        "dependency": "",
    }


def _synthetic_accepted_transaction(
    cycle: qualification.LoadCycleContext,
    *,
    job_id: str,
    batch_id: str,
    tasks: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    batches = cycle.dispatcher_state / "batches"
    sbatch_path = (batches / f"{batch_id}.sbatch").resolve()
    manifest_path = sbatch_path.with_suffix(".json")
    spool_path = sbatch_path.with_suffix(".spooled.json")
    _sealed(manifest_path, {"batch_id": batch_id, "tasks": tasks})
    sbatch_raw = b"#!/bin/bash\n# synthetic accepted batch\n"
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_bytes(sbatch_raw)
    sbatch_path.chmod(0o444)
    sbatch_sha256 = qualification._sha256_bytes(sbatch_raw)
    spool = {
        "schema_version": 1,
        "kind": "schema5_dispatch_spooled_script_receipt",
        "batch_id": batch_id,
        "job_id": job_id,
        "job_name": f"asys-dispatch-{batch_id[-10:]}",
        "scheduler_comment": f"asys-schema5-intent:{batch_id}",
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "spooled_sbatch_sha256": sbatch_sha256,
        "verified_at": 1_000.0,
    }
    _sealed(spool_path, spool)
    artifacts = {
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": qualification._sha256_file(
            manifest_path
        ),
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "spooled_sbatch_sha256": sbatch_sha256,
        "spooled_receipt_path": str(spool_path),
        "spooled_receipt_sha256": qualification._sha256_file(spool_path),
        "submission_transport": (
            qualification.dispatch_sweeps.STDIN_EXACT_SUBMISSION_TRANSPORT
        ),
        "submission_argv_sha256": (
            qualification.dispatch_sweeps._stdin_submission_argv_sha256(
                batch_id
            )
        ),
    }
    intent_record = {
        "state": "submitted",
        "job_id": job_id,
        "tasks": tasks,
        "fairness_committed": True,
        **artifacts,
    }
    job_record = {
        "job_id": job_id,
        "batch_id": batch_id,
        "task_count": len(tasks),
        "tasks": tasks,
        **artifacts,
    }
    return job_record, intent_record


def _canonical_cycle_task(
    context: qualification.QualificationContext,
    cycle: qualification.LoadCycleContext,
) -> dict[str, object]:
    cell = qualification.generate_qualification_cells()[0]
    profile = qualification.serving_profile_for_cell(cell).name
    return {
        "run_id": cycle.run_id,
        "run_root": str(cycle.run_root),
        "source_index": 0,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "manifest_sha256": "a" * 64,
        "benchmark_contracts_sha256": "b" * 64,
        "model_size": cell.model_size,
        "serving_profile": profile,
        "fanout_cost": qualification.dispatch_sweeps.fanout_cost(cell),
        "server_pool_id": str(context.server_pool_root),
        "server_run_id": str(context.server_pool_root),
        "server_pool_root": str(context.server_pool_root),
    }


def _write_synthetic_cycle_initialization(
    cycle: qualification.LoadCycleContext,
) -> None:
    cycle.run_root.mkdir(parents=True, exist_ok=True)
    marker = qualification._with_identity(
        {
            "schema_version": qualification.LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": qualification.CYCLE_INITIALIZED_PROTOCOL,
            "cycle_id": cycle.cycle_id,
            "run_id": cycle.run_id,
            "run_root": str(cycle.run_root),
            "manifest_sha256": "a" * 64,
            "benchmark_contracts_sha256": "b" * 64,
            "artifact_policy_sha256": "c" * 64,
            "lineage_id": "d" * 64,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
        },
        "initialization_id",
    )
    qualification._write_once(
        cycle.evidence_root / qualification.CYCLE_INITIALIZED_NAME,
        marker,
        description=f"synthetic cycle {cycle.cycle_index} initialization",
    )


def _write_synthetic_cycle_authority(
    context: qualification.QualificationContext,
    intent: dict[str, object],
    cycle: qualification.LoadCycleContext,
) -> dict[str, object]:
    path = cycle.execution_authority_path
    if path.exists():
        return qualification._read_json(
            path,
            description=f"synthetic cycle {cycle.cycle_index} authority",
            sealed=True,
        )
    authority = qualification._with_identity(
        {
            "schema_version": (
                qualification.EXECUTION_AUTHORITY_SCHEMA_VERSION
            ),
            "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": intent["intent_id"],
            "chain_id": context.chain_id,
            "run_id": cycle.run_id,
            "run_root": str(cycle.run_root),
            "readiness_generation": dict(intent["readiness_generation"]),
        },
        "authority_id",
    )
    qualification._write_once(
        path,
        authority,
        description=f"synthetic cycle {cycle.cycle_index} authority",
    )
    return authority


def _ensure_passing_cycle_roots(
    context: qualification.QualificationContext,
    intent: dict[str, object],
) -> tuple[qualification.LoadCycleContext, qualification.LoadCycleContext]:
    cycle0 = qualification.create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=0,
        now=950.0,
    )
    _write_synthetic_cycle_initialization(cycle0)
    _write_synthetic_cycle_authority(context, intent, cycle0)
    cycle1 = qualification.create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=1,
        now=960.0,
    )
    (cycle1.run_root / "cells").mkdir(parents=True, exist_ok=True)
    (cycle1.run_root / "cells" / "synthetic-row.jsonl").write_text(
        '{"synthetic":true}\n',
        encoding="utf-8",
    )
    _write_synthetic_cycle_initialization(cycle1)
    _write_synthetic_cycle_authority(context, intent, cycle1)
    return cycle0, cycle1


def _record(
    context: qualification.QualificationContext,
    intent: dict[str, object],
    *,
    sequence: int,
    timestamp: float,
    ceiling: int,
    active: int,
    useful: int,
    complete: bool = False,
    execution_events: int | None = None,
    replay_events: int = 0,
    unfinished: int | None = None,
    replay_status: str = "active",
) -> None:
    events = useful if execution_events is None else execution_events
    reference_events = events - replay_events
    if unfinished is None:
        unfinished = 0 if complete else qualification.CELL_COUNT
    cycle_directory = (
        context.qualification_root / qualification.LOAD_CYCLE_DIRECTORY
    )
    cycle0_intent = cycle_directory / "cycle-000000" / (
        qualification.CYCLE_INTENT_NAME
    )
    cycle1_intent = cycle_directory / "cycle-000001" / (
        qualification.CYCLE_INTENT_NAME
    )
    reference_cycle_id = (
        qualification._read_json(
            cycle0_intent,
            description="test reference cycle intent",
            sealed=True,
        )["cycle_id"]
        if cycle0_intent.exists()
        else "0" * 64
    )
    replay_cycle_id = (
        qualification._read_json(
            cycle1_intent,
            description="test replay cycle intent",
            sealed=True,
        )["cycle_id"]
        if cycle1_intent.exists()
        else "1" * 64
    )
    replay_run_id = (
        qualification._read_json(
            cycle1_intent,
            description="test replay cycle intent",
            sealed=True,
        )["run_id"]
        if cycle1_intent.exists()
        else (
            f"{qualification.QUALIFICATION_RUN_ID}__test"
            "__cycle_000001"
        )
    )
    cycle_inventory = [
        {
            "cycle_index": 0,
            "cycle_id": reference_cycle_id,
            "run_id": qualification.QUALIFICATION_RUN_ID,
            "semantic_reference": True,
            "status": "complete" if complete else "active",
            "validated_execution_events": reference_events,
            "unfinished_assignments": 0 if complete else unfinished,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
        }
    ]
    if replay_events:
        cycle_inventory[0]["unfinished_assignments"] = 0
        cycle_inventory.append(
            {
                "cycle_index": 1,
                "cycle_id": replay_cycle_id,
                "run_id": replay_run_id,
                "semantic_reference": False,
                "status": replay_status,
                "validated_execution_events": replay_events,
                "unfinished_assignments": unfinished,
                "estimand_excluded": True,
                "primary_analysis_eligible": False,
            }
        )
    active_cycle_ids = (
        []
        if active == 0
        else [
            record["cycle_id"]
            for record in cycle_inventory
            if record["status"] == "active"
        ]
    )
    execution_authority = (
        qualification._execution_authority_evidence_binding(intent)
    )
    jobs = [] if active == 0 else [_job(sequence)]
    scheduler = qualification.make_scheduler_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=timestamp,
        ceiling=ceiling,
        jobs=jobs,
        qualification_job_ids=(
            [] if active == 0 else [jobs[0]["job_id"]]
        ),
        active_qualification_cells=active,
        unfinished_load_assignments=unfinished,
        active_cycle_ids=active_cycle_ids,
        dispatcher_ledger_sha256=(
            None if active == 0 else "9" * 64
        ),
        production_control_guard_sha256=str(
            intent["control_guard"]["guard_sha256"]  # type: ignore[index]
        ),
        client_partition=str(intent["client_partition"]),
        client_qos=str(intent["client_qos"]),
        protected_capacity_marker_id=str(
            intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
        ),
        protected_capacity_marker_sha256=str(
            intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
        ),
        readiness_rollout_generation=int(
            intent["readiness_generation"]["rollout_generation"]  # type: ignore[index]
        ),
        trusted_generation_catalog_id=str(
            intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
        ),
        qualification_execution_authority_id=(
            execution_authority["authority_id"]
        ),
        qualification_execution_authority_sha256=(
            execution_authority["sha256"]
        ),
        cycle_execution_authorities=[
            {
                "cycle_id": cycle_id,
                "run_id": next(
                    record["run_id"]
                    for record in cycle_inventory
                    if record["cycle_id"] == cycle_id
                ),
                "authority_id": execution_authority["authority_id"],
                "authority_sha256": execution_authority["sha256"],
            }
            for cycle_id in active_cycle_ids
        ]
        or [
            {
                "cycle_id": reference_cycle_id,
                "run_id": qualification.QUALIFICATION_RUN_ID,
                "authority_id": execution_authority["authority_id"],
                "authority_sha256": execution_authority["sha256"],
            }
        ],
    )
    semantic = qualification.make_semantic_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=timestamp,
        manifest_sha256=MANIFEST_SHA256,
        states=(
            {"complete": qualification.CELL_COUNT}
            if complete
            else {
                "active": active,
                "missing": qualification.CELL_COUNT - active,
            }
        ),
        validated_qids=useful,
        useful_qids=useful,
        strata_progress=_progress(useful),
        artifact_schema_counts=({} if useful == 0 else {"5": useful}),
        semantic_reference_cycle=reference_cycle_id,
        trusted_qid_execution_events=events,
        replay_qid_execution_events=replay_events,
        load_strata_progress=_progress(events),
        load_cycle_inventory=cycle_inventory,
        unfinished_load_assignments=unfinished,
    )
    qualification.record_observation(
        context.qualification_root,
        intent=intent,
        scheduler=scheduler,
        semantic=semantic,
    )


def _passing_observations(
    context: qualification.QualificationContext,
    intent: dict[str, object],
) -> list[dict[str, object]]:
    _ensure_passing_cycle_roots(context, intent)
    saturation_target = int(intent["certified_saturation_target"])
    rows = [
        (1_000.0, 24, 0, 0, 0, False, 768, "active"),
        (1_010.0, 24, 24, 24, 24, False, 768, "active"),
        (1_020.0, 96, 96, 120, 120, False, 768, "active"),
        (1_030.0, 192, 192, 312, 312, False, 768, "active"),
        (
            1_040.0,
            384,
            saturation_target,
            696,
            696,
            False,
            768,
            "active",
        ),
    ]
    for index in range(1, 13):
        timestamp = 1_040.0 + 600.0 * index
        events = 696 + (
            qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS * index // 12
        )
        useful = min(qualification.TOTAL_QIDS, events)
        completed = useful == qualification.TOTAL_QIDS
        replay_events = max(0, events - qualification.TOTAL_QIDS)
        rows.append(
            (
                timestamp,
                384,
                saturation_target,
                useful,
                events,
                completed,
                600 if replay_events else 768,
                "active",
            )
        )
    # End-before-drain is followed by a quiescent observation.  The unfinished
    # replay remains preserved and explicitly excluded as load_window_drained.
    rows.append(
        (
            8_241.0,
            384,
            0,
            qualification.TOTAL_QIDS,
            696 + qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS,
            True,
            600,
            "load_window_drained",
        )
    )
    for sequence, (
        timestamp,
        ceiling,
        active,
        useful,
        events,
        complete,
        unfinished,
        replay_status,
    ) in enumerate(rows):
        _record(
            context,
            intent,
            sequence=sequence,
            timestamp=timestamp,
            ceiling=ceiling,
            active=active,
            useful=useful,
            complete=complete,
            execution_events=events,
            replay_events=max(0, events - qualification.TOTAL_QIDS),
            unfinished=unfinished,
            replay_status=replay_status,
        )
    return qualification.load_observations(
        context.qualification_root, intent=intent
    )


def _prepare_terminal_failure_drain(
    context: qualification.QualificationContext,
    intent: dict[str, object],
    *,
    reason: str,
) -> None:
    _ensure_passing_cycle_roots(context, intent)
    qualification.request_failure_drain(
        context,
        intent=intent,
        reason=reason,
        now=990.0,
    )
    _record(
        context,
        intent,
        sequence=0,
        timestamp=1_000.0,
        ceiling=24,
        active=0,
        useful=qualification.TOTAL_QIDS,
        complete=True,
        execution_events=qualification.TOTAL_QIDS + 1,
        replay_events=1,
        unfinished=1,
        replay_status="active",
    )
    observations = qualification.load_observations(
        context.qualification_root, intent=intent
    )
    qualification.mark_load_cycles_drained(
        context,
        intent=intent,
        cycle_inventory=observations[-1]["semantic"][
            "load_cycle_inventory"
        ],
        now=1_001.0,
    )
    _record(
        context,
        intent,
        sequence=1,
        timestamp=1_002.0,
        ceiling=24,
        active=0,
        useful=qualification.TOTAL_QIDS,
        complete=True,
        execution_events=qualification.TOTAL_QIDS + 1,
        replay_events=1,
        unfinished=1,
        replay_status="load_window_drained",
    )


def test_load_design_is_exact_deterministic_and_balanced() -> None:
    first = qualification.generate_qualification_cells()
    second = qualification.generate_qualification_cells()

    assert first == second
    assert len(first) == 768
    assert len({cell.cell_id for cell in first}) == 768
    assert {cell.n_questions for cell in first} == {20}
    assert len({qualification._stratum_tuple(cell) for cell in first}) == 570
    assert all(
        cell.cell_id not in qualification.PRODUCTION_RUN_IDS for cell in first
    )

    plan = qualification.build_load_plan()
    assert plan["cell_count"] == 768
    assert plan["qids"] == 15_360
    assert plan["ceilings"] == [24, 96, 192, 384]
    assert plan["health_soak_384_seconds"] == 7_200
    assert plan["minimum_loaded_384_observations"] == 2
    assert plan["balance"]["benchmark"] == {
        "gpqa": 192,
        "math": 192,
        "mmlu_pro": 192,
        "truthfulqa": 192,
    }
    assert plan["balance"]["seed"] == {"0": 256, "1": 256, "2": 256}
    assert plan["balance"]["prompt_complexity_level"] == {
        "0": 270,
        "1": 228,
        "3": 270,
    }
    assert plan["balance"]["sharing_topology_context"] == {
        "artifact_only": 180,
        "plus_cot": 180,
    }
    qualification.validate_load_plan(plan)


def test_load_plan_strict_schema_and_self_hash_reject_drift() -> None:
    plan = qualification.build_load_plan()
    identity = dict(plan)
    observed = identity.pop("plan_id")
    assert observed == qualification._sha256_bytes(
        qualification._canonical_bytes(identity)
    )

    drifted = dict(plan)
    drifted["qids"] -= 1
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="invalid plan_id",
    ):
        qualification.validate_load_plan(drifted)

    extra = dict(plan)
    extra["operator_override"] = True
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="fields drifted",
    ):
        qualification.validate_load_plan(extra)


def _base_profile_replicas() -> dict[str, int]:
    return {
        "0.6B": 2,
        "0.6B-long": 1,
        "1.7B": 2,
        "1.7B-long": 1,
        "4B": 2,
        "4B-long": 1,
        "8B": 3,
        "8B-long": 1,
        "14B": 2,
        "14B-long": 1,
        "32B": 4,
        "32B-long": 2,
    }


def _expanded_profiles(
    additions: dict[str, int],
) -> dict[str, int]:
    result = _base_profile_replicas()
    for profile, count in additions.items():
        result[profile] += count
    return result


def _feasible_eighteen_gpu_overlay() -> dict[str, int]:
    return _expanded_profiles(
        {
            "0.6B": 3,
            "1.7B": 3,
            "4B": 3,
            "8B": 2,
            "14B": 3,
            "32B": 4,
        }
    )


def _feasible_nineteen_gpu_overlay() -> dict[str, int]:
    return _expanded_profiles(
        {
            "0.6B": 3,
            "1.7B": 3,
            "4B": 2,
            "8B": 2,
            "14B": 4,
            "1.7B-long": 1,
            "4B-long": 1,
            "14B-long": 1,
            "32B-long": 1,
        }
    )


@lru_cache(maxsize=None)
def _protected_capacity_fixture_payloads(
    base_fleet_sha256: str,
    effective_fleet_sha256: str,
    source_tree_sha256: str = "5" * 64,
    dispatcher_source_sha256: str = "4" * 64,
    qualification_runner_source_sha256: str = "6" * 64,
) -> dict[str, object]:
    """Build one semantically real v4 capacity authority for local tests."""

    base_profiles = _base_profile_replicas()
    effective_profiles = dict(base_profiles)
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=1,
        release_git_commit=COMMIT,
        source_tree_sha256=source_tree_sha256,
        release_fleet_contract_sha256=base_fleet_sha256,
        base_fleet_contract_sha256=base_fleet_sha256,
        proposed_effective_fleet_contract_sha256=effective_fleet_sha256,
        additive_overlay_contract_sha256=effective_fleet_sha256,
        base_profile_replicas=base_profiles,
        effective_profile_replicas=effective_profiles,
        dispatcher_source_sha256=dispatcher_source_sha256,
        qualification_runner_source_sha256=(
            qualification_runner_source_sha256
        ),
    )

    def rows(
        prefix: str,
        counts: dict[str, int],
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for profile in sorted(counts):
            tp_size = int(
                qualification.SERVING_PROFILE_REGISTRY[profile].tp_size
            )
            for index in range(counts[profile]):
                result.append(
                    {
                        "shape_id": (
                            f"{prefix}-{profile.replace('.', '_')}-{index:02d}"
                        ),
                        "serving_profile": profile,
                        "tasks": 1,
                        "cpus": 1,
                        "memory_mib": 1,
                        "gpus": tp_size,
                        "time_limit_seconds": 86_400,
                    }
                )
        return result

    delta = {
        profile: effective_profiles[profile] - base_profiles[profile]
        for profile in base_profiles
    }
    base_topology = rows("base", base_profiles)
    additive_topology = rows("additive", delta)
    effective_topology = [*base_topology, *additive_topology]
    warm_topology = [
        {
            "shape_id": f"warm-{index}",
            "serving_profile": profile,
            "tasks": 1,
            "cpus": 1,
            "memory_mib": 1,
            "gpus": gpus,
            "time_limit_seconds": 86_400,
        }
        for index, (profile, gpus) in enumerate(
            (("0.6B", 1), ("1.7B", 1), ("32B-long", 2))
        )
    ]
    return {
        "certificate": certificate,
        "base_topology": base_topology,
        "base_topology_sha256": qualification._sha256_bytes(
            qualification._canonical_bytes(base_topology)
        ),
        "additive_topology": additive_topology,
        "additive_topology_sha256": qualification._sha256_bytes(
            qualification._canonical_bytes(additive_topology)
        ),
        "effective_topology": effective_topology,
        "effective_topology_sha256": qualification._sha256_bytes(
            qualification._canonical_bytes(effective_topology)
        ),
        "warm_topology": warm_topology,
        "warm_topology_sha256": qualification._sha256_bytes(
            qualification._canonical_bytes(warm_topology)
        ),
    }


def test_capacity_bounds_distinguish_theory_from_executable_policy() -> None:
    base = _base_profile_replicas()
    plus_four = _expanded_profiles(
        {"0.6B": 1, "1.7B": 1, "4B": 1, "14B": 1}
    )
    plus_seven = _expanded_profiles(
        {"0.6B": 1, "1.7B": 1, "4B": 2, "8B": 2, "14B": 1}
    )

    assert qualification.theoretical_profile_packing_upper_bound(
        base
    )["total_fit"] == 315
    assert qualification.theoretical_profile_packing_upper_bound(
        plus_four
    )["total_fit"] == 361
    assert qualification.theoretical_profile_packing_upper_bound(
        plus_seven
    )["total_fit"] == 385

    base_wave = qualification.simulate_preflight_capacity_wave(base)
    plus_four_wave = qualification.simulate_preflight_capacity_wave(
        plus_four
    )
    plus_seven_wave = qualification.simulate_preflight_capacity_wave(
        plus_seven
    )
    assert base_wave["plan_fanout_total"] == 2_658
    assert base_wave["selected_cell_count"] == 278
    assert plus_four_wave["selected_cell_count"] == 306
    assert plus_seven_wave["selected_cell_count"] == 324
    assert all(
        wave["passed"] is False
        for wave in (base_wave, plus_four_wave, plus_seven_wave)
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="cannot admit the exact 384-cell",
    ):
        qualification.validate_preflight_capacity_wave(plus_seven_wave)


def test_capacity_certificate_recomputes_exact_sixteen_batch_wave() -> None:
    effective = _feasible_eighteen_gpu_overlay()
    wave = qualification.simulate_preflight_capacity_wave(effective)
    assert wave["passed"] is True
    assert wave["selected_cell_count"] == 384
    assert wave["microbatch_count"] == 16
    assert {batch["selected_count"] for batch in wave["microbatches"]} == {
        24
    }
    assert len(
        {record["cell_id"] for record in wave["selected_wave"]}
    ) == 384
    assert (
        wave["selected_cell_ids_sha256"]
        == "e73981e73556725f2b054734bc8afef6787acdf000813cf2d22e023bcf6bf30f"
    )
    assert all(
        row["selected_fanout"] <= row["fanout_capacity"]
        for row in wave["profile_summary"].values()
    )
    qualification.validate_preflight_capacity_wave(wave)

    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=2,
        release_git_commit=COMMIT,
        source_tree_sha256="5" * 64,
        release_fleet_contract_sha256="1" * 64,
        base_fleet_contract_sha256="1" * 64,
        proposed_effective_fleet_contract_sha256="2" * 64,
        additive_overlay_contract_sha256="2" * 64,
        base_profile_replicas=_base_profile_replicas(),
        effective_profile_replicas=effective,
        dispatcher_source_sha256="4" * 64,
        qualification_runner_source_sha256="6" * 64,
    )
    assert certificate["base_logical_replicas"] == 22
    assert certificate["base_allocated_gpus"] == 24
    assert certificate["effective_logical_replicas"] == 40
    assert certificate["effective_active_gpus"] == 42
    assert certificate["additive_tp1_logical_replicas"] == 18
    assert certificate["additive_tp2_logical_replicas"] == 0
    assert certificate["additive_allocated_gpus"] == 18
    assert certificate["selected_cell_count"] == 384
    qualification.validate_preflight_capacity_certificate(certificate)

    tampered = json.loads(json.dumps(certificate))
    tampered["wave"]["microbatches"][0]["selected"].reverse()
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="invalid certificate_id",
    ):
        qualification.validate_preflight_capacity_certificate(tampered)
    lying_counts = json.loads(json.dumps(certificate))
    lying_counts["effective_profile_replicas"]["0.6B"] += 1
    lying_counts.pop("certificate_id")
    lying_counts = qualification._with_identity(
        lying_counts, "certificate_id"
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="384-cell qualification wave|source-bound sequential-policy",
    ):
        qualification.validate_preflight_capacity_certificate(
            lying_counts
        )


def test_generation_one_certificate_is_exact_zero_delta_baseline() -> None:
    base = _base_profile_replicas()
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=1,
        release_git_commit=COMMIT,
        source_tree_sha256="5" * 64,
        release_fleet_contract_sha256="1" * 64,
        base_fleet_contract_sha256="1" * 64,
        proposed_effective_fleet_contract_sha256="1" * 64,
        additive_overlay_contract_sha256="1" * 64,
        base_profile_replicas=base,
        effective_profile_replicas=base,
        dispatcher_source_sha256="4" * 64,
        qualification_runner_source_sha256="6" * 64,
    )

    assert certificate["base_logical_replicas"] == 22
    assert certificate["effective_logical_replicas"] == 22
    assert certificate["base_allocated_gpus"] == 24
    assert certificate["effective_active_gpus"] == 24
    assert certificate["additive_tp1_logical_replicas"] == 0
    assert certificate["additive_tp2_logical_replicas"] == 0
    assert certificate["additive_allocated_gpus"] == 0
    assert certificate["selected_cell_count"] == 278
    assert certificate["wave"]["passed"] is False
    assert certificate["wave"]["shortfall_cells"] == 106
    assert [
        batch["selected_count"]
        for batch in certificate["wave"]["microbatches"]
    ] == [24] * 11 + [14]
    qualification.validate_preflight_capacity_certificate(certificate)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="generation one must bind the exact zero-delta",
    ):
        qualification.build_preflight_capacity_certificate(
            capacity_generation=1,
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            release_fleet_contract_sha256="1" * 64,
            base_fleet_contract_sha256="1" * 64,
            proposed_effective_fleet_contract_sha256="2" * 64,
            additive_overlay_contract_sha256="2" * 64,
            base_profile_replicas=base,
            effective_profile_replicas=_feasible_eighteen_gpu_overlay(),
            dispatcher_source_sha256="4" * 64,
            qualification_runner_source_sha256="6" * 64,
        )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="post-baseline capacity generations require a positive",
    ):
        qualification.build_preflight_capacity_certificate(
            capacity_generation=2,
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            release_fleet_contract_sha256="1" * 64,
            base_fleet_contract_sha256="1" * 64,
            proposed_effective_fleet_contract_sha256="1" * 64,
            additive_overlay_contract_sha256="1" * 64,
            base_profile_replicas=base,
            effective_profile_replicas=base,
            dispatcher_source_sha256="4" * 64,
            qualification_runner_source_sha256="6" * 64,
        )


def test_alternate_nineteen_gpu_overlay_is_exactly_replayable() -> None:
    wave = qualification.simulate_preflight_capacity_wave(
        _feasible_nineteen_gpu_overlay()
    )
    assert wave["passed"] is True
    assert wave["selected_cell_count"] == 384
    assert wave["microbatch_count"] == 16
    assert (
        wave["selected_cell_ids_sha256"]
        == "1f5ddee1f44262cdcd73752d586f3fb2f824cafd348d992c5f2e78bf38e02438"
    )


def test_preflight_capacity_cli_path_publishes_certificate_or_shortfall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_profile_replicas()
    effective = _feasible_eighteen_gpu_overlay()

    def fleet(path: Path, replicas: dict[str, int]) -> Path:
        return _sealed(
            path,
            {
                "profiles": [
                    {
                        "serving_profile": profile,
                        "replicas": [
                            {"replica_index": index}
                            for index in range(count)
                        ],
                    }
                    for profile, count in sorted(replicas.items())
                ]
            },
        )

    base_path = fleet(tmp_path / "base.json", base)
    effective_path = fleet(tmp_path / "effective.json", effective)
    dispatcher = tmp_path / "dispatch_sweeps.py"
    dispatcher.write_text("# frozen dispatcher fixture\n", encoding="utf-8")
    dispatcher.chmod(0o444)
    runner_source = tmp_path / "run_schema5_throughput_qualification.py"
    runner_source.write_text(
        "# frozen qualification fixture\n", encoding="utf-8"
    )
    runner_source.chmod(0o444)
    monkeypatch.setattr(
        qualification.dispatch_sweeps, "__file__", str(dispatcher)
    )
    monkeypatch.setattr(qualification, "__file__", str(runner_source))
    monkeypatch.setattr(
        qualification.control,
        "_assert_additive_capacity_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        qualification,
        "_verify_preflight_release_source_binding",
        lambda **_kwargs: {
            "release_git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "source_tree_sha256": "5" * 64,
            "dispatcher_source_sha256": qualification._sha256_file(
                dispatcher
            ),
            "qualification_runner_source_sha256": (
                qualification._sha256_file(runner_source)
            ),
        },
    )

    def parsed_contract(path: Path) -> SimpleNamespace:
        replicas = (
            effective
            if Path(path).resolve() == effective_path.resolve()
            else base
        )
        by_profile = {
            profile: tuple(
                SimpleNamespace(
                    gpus_per_replica=int(
                        qualification.SERVING_PROFILE_REGISTRY[
                            profile
                        ].tp_size
                    )
                )
                for _ in range(count)
            )
            for profile, count in replicas.items()
        }
        return SimpleNamespace(
            by_profile=by_profile,
            replicas=tuple(
                replica
                for profile in sorted(by_profile)
                for replica in by_profile[profile]
            ),
        )

    monkeypatch.setattr(
        qualification,
        "_load_preflight_fleet_contracts",
        lambda *, base_path, effective_path: (
            parsed_contract(base_path),
            parsed_contract(effective_path),
        ),
    )
    output = (
        tmp_path
        / "readiness"
        / "capacity-generations"
        / "c000002"
        / qualification.PREFLIGHT_CAPACITY_CERTIFICATE_NAME
    )
    report = qualification.preflight_capacity_report(
        base_fleet_contract=base_path,
        effective_fleet_contract=effective_path,
        additive_overlay_contract=effective_path,
        capacity_generation=2,
        release_git_commit=COMMIT,
        source_tree_sha256="5" * 64,
        dispatcher_source=dispatcher,
        qualification_runner_source=runner_source,
        output=output,
        apply=True,
    )
    assert report["status"] == "complete"
    assert output.is_file()
    assert stat.S_IMODE(output.stat().st_mode) & 0o222 == 0
    qualification.validate_preflight_capacity_certificate(
        qualification._read_json(
            output,
            description="test preflight capacity certificate",
            sealed=True,
        )
    )
    unrelated_overlay = _sealed(
        tmp_path / "unrelated-overlay.json", {"additive": True}
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="exact proposed effective fleet",
    ):
        qualification.preflight_capacity_report(
            base_fleet_contract=base_path,
            effective_fleet_contract=effective_path,
            additive_overlay_contract=unrelated_overlay,
            capacity_generation=2,
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            dispatcher_source=dispatcher,
            qualification_runner_source=runner_source,
            output=(
                tmp_path
                / "bad-overlay"
                / "capacity-generations"
                / "c000002"
                / qualification.PREFLIGHT_CAPACITY_CERTIFICATE_NAME
            ),
            apply=False,
        )
    wrong_dispatcher = tmp_path / "wrong-dispatch_sweeps.py"
    wrong_dispatcher.write_text("# wrong source\n", encoding="utf-8")
    wrong_dispatcher.chmod(0o444)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="exact immutable imported source",
    ):
        qualification.preflight_capacity_report(
            base_fleet_contract=base_path,
            effective_fleet_contract=effective_path,
            additive_overlay_contract=effective_path,
            capacity_generation=2,
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            dispatcher_source=wrong_dispatcher,
            qualification_runner_source=runner_source,
            output=(
                tmp_path
                / "bad-source"
                / "capacity-generations"
                / "c000002"
                / qualification.PREFLIGHT_CAPACITY_CERTIFICATE_NAME
            ),
            apply=False,
        )

    baseline_output = (
        tmp_path
        / "baseline"
        / qualification.PREFLIGHT_CAPACITY_CERTIFICATE_NAME
    )
    baseline = qualification.preflight_capacity_report(
        base_fleet_contract=base_path,
        effective_fleet_contract=base_path,
        additive_overlay_contract=base_path,
        capacity_generation=1,
        release_git_commit=COMMIT,
        source_tree_sha256="5" * 64,
        dispatcher_source=dispatcher,
        qualification_runner_source=runner_source,
        output=baseline_output,
        apply=True,
    )
    assert baseline["status"] == "complete"
    assert baseline["passed"] is True
    assert baseline_output.is_file()
    certificate = qualification.validate_preflight_capacity_certificate(
        qualification._read_json(
            baseline_output,
            description="test baseline preflight capacity certificate",
            sealed=True,
        )
    )
    assert certificate["base_logical_replicas"] == 22
    assert certificate["effective_logical_replicas"] == 22
    assert certificate["base_allocated_gpus"] == 24
    assert certificate["effective_active_gpus"] == 24
    assert certificate["additive_tp1_logical_replicas"] == 0
    assert certificate["additive_tp2_logical_replicas"] == 0
    assert certificate["additive_allocated_gpus"] == 0
    assert certificate["selected_cell_count"] == 278
    assert certificate["wave"]["passed"] is False
    assert certificate["wave"]["shortfall_cells"] == 106


def test_preflight_fleet_loader_parses_complete_additive_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = (tmp_path / "release-worktree").resolve()
    configs = worktree / "configs"
    configs.mkdir(parents=True)

    def frozen_copy(source: Path, target: Path) -> Path:
        shutil.copyfile(source, target)
        target.chmod(0o444)
        return target

    source_configs = Path(qualification.__file__).resolve().parents[1] / "configs"
    model_path = frozen_copy(
        source_configs / "model_contracts.v1.json",
        configs / "model_contracts.v1.json",
    )
    frozen_copy(
        source_configs / "model_contracts.v1.sha256",
        configs / "model_contracts.v1.sha256",
    )
    base_path = frozen_copy(
        source_configs / "schema5_fleet.v1.json",
        configs / "schema5_fleet.v1.json",
    )
    frozen_copy(
        source_configs / "schema5_fleet.v1.sha256",
        configs / "schema5_fleet.v1.sha256",
    )
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    additions = {
        "0.6B": 3,
        "1.7B": 3,
        "4B": 3,
        "8B": 2,
        "14B": 3,
        "32B": 4,
    }
    for profile in payload["profiles"]:
        name = profile["serving_profile"]
        for _ in range(additions.get(name, 0)):
            index = len(profile["replicas"])
            replica = dict(profile["replicas"][-1])
            replica.update(
                {
                    "replica_index": index,
                    "replica_id": (
                        fleet_contract_runtime.expected_replica_id(
                            name, index
                        )
                    ),
                    "scheduler_job_name": (
                        fleet_contract_runtime.expected_scheduler_job_name(
                            name, index
                        )
                    ),
                }
            )
            profile["replicas"].append(replica)
    payload["logical_replica_count"] = 40
    payload["allocated_gpu_count"] = 42
    effective_path = configs / "schema5_fleet.capacity.v1.json"
    effective_path.write_bytes(qualification._canonical_bytes(payload))
    effective_sha256 = qualification._sha256_file(effective_path)
    effective_checksum = effective_path.with_suffix(".sha256")
    effective_checksum.write_text(
        f"{effective_sha256}  {effective_path.name}\n",
        encoding="ascii",
    )
    effective_path.chmod(0o444)
    effective_checksum.chmod(0o444)
    monkeypatch.setattr(qualification, "REPO", worktree)

    base, effective = qualification._load_preflight_fleet_contracts(
        base_path=base_path,
        effective_path=effective_path,
    )
    assert len(base.replicas) == 22
    assert len(effective.replicas) == 40
    assert sum(row.gpus_per_replica for row in effective.replicas) == 42
    assert model_path.is_file()

    effective_path.chmod(0o644)
    effective_checksum.chmod(0o644)
    drifted = json.loads(effective_path.read_text(encoding="utf-8"))
    expanded = next(
        row
        for row in drifted["profiles"]
        if row["serving_profile"] == "0.6B"
    )
    expanded["replicas"][-1]["memory"] = "121G"
    effective_path.write_bytes(qualification._canonical_bytes(drifted))
    effective_checksum.write_text(
        f"{qualification._sha256_file(effective_path)}  "
        f"{effective_path.name}\n",
        encoding="ascii",
    )
    effective_path.chmod(0o444)
    effective_checksum.chmod(0o444)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="topology failed closed",
    ):
        qualification._load_preflight_fleet_contracts(
            base_path=base_path,
            effective_path=effective_path,
        )


def test_preflight_source_binding_recomputes_exact_tag_and_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = tmp_path / "release-worktree"
    dispatcher = worktree / "slurm" / "dispatch_sweeps.py"
    runner_source = (
        worktree / "scripts" / "run_schema5_throughput_qualification.py"
    )
    dispatcher.parent.mkdir(parents=True)
    runner_source.parent.mkdir(parents=True)
    dispatcher.write_text("# exact dispatcher\n", encoding="utf-8")
    runner_source.write_text("# exact qualification\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "REPO", worktree)
    monkeypatch.setattr(qualification, "__file__", str(runner_source))
    dirty = False
    dispatcher_blob = "3" * 40
    runner_blob = "4" * 40
    tagged_dispatcher = dispatcher.read_bytes()
    tagged_runner = runner_source.read_bytes()

    def git_runner(argv, **kwargs):
        assert kwargs["check"] is False
        nonlocal dirty
        arguments = argv[3:]
        if arguments == ["rev-parse", "HEAD"]:
            output = COMMIT
        elif arguments == ["rev-parse", "--show-toplevel"]:
            output = str(worktree)
        elif arguments == [
            "rev-parse",
            f"refs/tags/{qualification.renderer.RELEASE_TAG}",
        ]:
            output = TAG_OBJECT
        elif arguments == ["cat-file", "-t", TAG_OBJECT]:
            output = "tag"
        elif arguments == [
            "rev-parse",
            f"{TAG_OBJECT}^{{commit}}",
        ]:
            output = COMMIT
        elif arguments == [
            "rev-parse",
            f"{COMMIT}:slurm/dispatch_sweeps.py",
        ]:
            output = dispatcher_blob
        elif arguments == [
            "rev-parse",
            (
                f"{COMMIT}:scripts/"
                "run_schema5_throughput_qualification.py"
            ),
        ]:
            output = runner_blob
        elif arguments in (
            ["cat-file", "-t", dispatcher_blob],
            ["cat-file", "-t", runner_blob],
        ):
            output = "blob"
        elif arguments == ["cat-file", "blob", dispatcher_blob]:
            output = tagged_dispatcher
        elif arguments == ["cat-file", "blob", runner_blob]:
            output = tagged_runner
        elif arguments == [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]:
            output = " M slurm/dispatch_sweeps.py" if dirty else ""
        else:
            raise AssertionError(arguments)
        if kwargs["text"] is False:
            assert isinstance(output, bytes)
            return subprocess.CompletedProcess(argv, 0, output, b"")
        assert isinstance(output, str)
        return subprocess.CompletedProcess(argv, 0, output + "\n", "")

    monkeypatch.setattr(qualification.subprocess, "run", git_runner)
    monkeypatch.setattr(
        qualification.control,
        "sha256_tree",
        lambda _root: "5" * 64,
    )
    report = qualification._verify_preflight_release_source_binding(
        release_git_commit=COMMIT,
        source_tree_sha256="5" * 64,
        dispatcher_source=dispatcher.resolve(),
        qualification_runner_source=runner_source.resolve(),
    )
    assert report["release_tag_object"] == TAG_OBJECT
    assert report["source_tree_sha256"] == "5" * 64

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="clean exact annotated release",
    ):
        qualification._verify_preflight_release_source_binding(
            release_git_commit=COMMIT,
            source_tree_sha256="6" * 64,
            dispatcher_source=dispatcher.resolve(),
            qualification_runner_source=runner_source.resolve(),
        )
    dirty = True
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="clean exact annotated release",
    ):
        qualification._verify_preflight_release_source_binding(
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            dispatcher_source=dispatcher.resolve(),
            qualification_runner_source=runner_source.resolve(),
        )
    dirty = False
    dispatcher.write_text("# untagged dispatcher\n", encoding="utf-8")
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="clean exact annotated release",
    ):
        # Git still returns the original blob while the worktree source has
        # drifted and all caller-supplied identities remain plausible.
        qualification._verify_preflight_release_source_binding(
            release_git_commit=COMMIT,
            source_tree_sha256="5" * 64,
            dispatcher_source=dispatcher.resolve(),
            qualification_runner_source=runner_source.resolve(),
        )


def test_sealed_json_rejects_noncanonical_and_hardlinked_files(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"value": 1}\n', encoding="utf-8")
    noncanonical.chmod(0o444)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="not canonically encoded",
    ):
        qualification._read_json(
            noncanonical,
            description="noncanonical fixture",
            sealed=True,
        )

    source = _sealed(tmp_path / "shared.json", {"value": 1})
    alias = tmp_path / "shared-alias.json"
    alias.hardlink_to(source)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unique read-only regular file",
    ):
        qualification._read_json(
            source,
            description="hardlinked fixture",
            sealed=True,
        )


def test_sealed_json_rejects_mutation_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _sealed(tmp_path / "mutable.json", {"value": 1})
    original_read = qualification.os.read
    mutated = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            path.chmod(0o644)
            path.write_bytes(
                qualification._canonical_bytes({"value": 2})
            )
            path.chmod(0o444)
        return chunk

    monkeypatch.setattr(qualification.os, "read", mutate_after_read)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="changed during its sealed read",
    ):
        qualification._read_json(
            path,
            description="concurrently changed fixture",
            sealed=True,
        )


def test_dry_run_and_dispatch_commands_never_name_production_runs(
    tmp_path: Path,
) -> None:
    manifest = _chain_manifest(tmp_path)
    context = qualification.load_qualification_context(
        manifest, verify_chain=False
    )
    report = qualification.dry_run_report(
        manifest,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        verify_chain=False,
    )
    derived = qualification.dry_run_report(
        manifest,
        verify_chain=False,
    )

    assert report["status"] == "dry_run"
    assert report["submitted"] is False
    assert report["cells"] == 768
    assert report["qids"] == 15_360
    assert derived["client_placement"] == report["client_placement"]
    assert not context.qualification_root.exists()
    for stage in report["stages"]:
        argv = stage["dispatcher_argv"]
        joined = " ".join(argv)
        assert qualification.QUALIFICATION_RUN_ID in joined
        assert "--control-state-dir" not in argv
        assert argv[argv.index("--cell-partition") + 1] == "ou_bcs_normal"
        assert argv[argv.index("--cell-qos") + 1] == "normal"
        assert (
            argv[argv.index("--protected-capacity-marker-id") + 1]
            == context.protected_capacity_contract.marker_id
        )
        assert "--protected-capacity-release-git-commit" in argv
        assert argv[
            argv.index("--qualification-execution-authority") + 1
        ] == str(
            context.qualification_base
            / qualification.ATTEMPT_DIRECTORY
            / "VERIFIED_GENERATION_AT_EXECUTE"
            / qualification.EXECUTION_AUTHORITY_NAME
        )
        assert str(context.state_root) not in argv
        assert not any(
            run_id in joined for run_id in qualification.PRODUCTION_RUN_IDS
        )

    argv = qualification.dispatcher_command(
        context,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        max_batch=7,
    )
    assert argv[argv.index("--max-batch") + 1] == "7"
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="within 0..24",
    ):
        qualification.dispatcher_command(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            max_batch=25,
        )
    reconcile_argv = qualification.dispatcher_command(
        context,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        max_batch=0,
    )
    assert reconcile_argv[
        reconcile_argv.index("--max-batch") + 1
    ] == "0"
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="must be supplied together",
    ):
        qualification.dry_run_report(
            manifest,
            client_partition="ou_bcs_normal",
            verify_chain=False,
        )


def test_context_rejects_self_consistent_wrong_release_tag_object(
    tmp_path: Path,
) -> None:
    manifest = _chain_manifest(tmp_path)
    release_root = (
        tmp_path
        / "results"
        / "recovery"
        / "schema5-v1"
        / "release"
    )
    identity_path = (
        release_root
        / "identity"
        / "release_identity.schema5-v1.json"
    )
    checksum_path = Path(str(identity_path) + ".sha256")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    wrong_tag_object = "9" * 40
    identity["git"]["git_tag_object"] = wrong_tag_object
    identity["control_pin_fragment"][
        "release_tag_object"
    ] = wrong_tag_object
    identity_path.chmod(0o644)
    checksum_path.chmod(0o644)
    identity_path.write_bytes(qualification._canonical_bytes(identity))
    checksum_path.write_text(
        f"{qualification._sha256_file(identity_path)}  "
        f"{identity_path.name}\n",
        encoding="ascii",
    )
    identity_path.chmod(0o444)
    checksum_path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="exact annotated tag object",
    ):
        qualification.load_qualification_context(
            manifest, verify_chain=False
        )


def test_marker_first_attempt_pointer_recovers_missing_current_cache(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    base = qualification.load_qualification_context(
        context.chain_manifest, verify_chain=False
    )
    pointer_path = context.attempt_pointer_path
    assert pointer_path is not None
    pointer_bytes = pointer_path.read_bytes()
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    current_path.unlink()

    recovered = qualification.create_or_load_attempt_context(
        base,
        control_value={},
        readiness_generation=intent["readiness_generation"],
        now=9_999.0,
    )

    assert recovered.qualification_root == context.qualification_root
    assert recovered.run_root == context.run_root
    assert recovered.attempt_pointer == context.attempt_pointer
    assert pointer_path.read_bytes() == pointer_bytes
    assert stat.S_IMODE(current_path.stat().st_mode) == 0o444
    assert len(qualification._load_attempt_pointers(base)) == 1


def test_current_attempt_cache_rejects_stale_publication_orphan(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    base = qualification.load_qualification_context(
        context.chain_manifest, verify_chain=False
    )
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    orphan = current_path.with_name(
        f".{current_path.name}.crashed.publishing"
    )
    orphan.write_bytes(b"incomplete")
    orphan.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unreconciled stale publication",
    ):
        qualification.create_or_load_attempt_context(
            base,
            control_value={},
            readiness_generation=intent["readiness_generation"],
            now=9_999.0,
        )


@pytest.mark.parametrize(
    "target_name",
    [qualification.EVIDENCE_NAME, qualification.MARKER_NAME],
)
def test_completion_rejects_stale_attempt_publication_orphan(
    tmp_path: Path,
    target_name: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    if target_name == qualification.MARKER_NAME:
        qualification.publish_load_window_end_intent(
            context.qualification_root,
            intent=intent,
            observations=observations,
        )
        evaluation = qualification.evaluate_observations(observations)
        qualification.publish_load_window_drain(
            context.qualification_root,
            intent=intent,
            observations=observations,
            evaluation=evaluation,
        )
        for cycle in qualification.load_cycle_inventory(
            context, intent=intent
        ):
            qualification._seal_tree_read_only(
                cycle.run_root,
                description=f"test load cycle {cycle.cycle_index}",
            )
        evidence = qualification._evidence_summary(
            intent=intent,
            observations=observations,
            evaluation=evaluation,
        )
        qualification._write_once(
            context.qualification_root / qualification.EVIDENCE_NAME,
            evidence,
            description="test aggregate evidence",
        )
    target = context.qualification_root / target_name
    orphan = target.with_name(f".{target.name}.crashed.publishing")
    orphan.write_bytes(b"incomplete")
    orphan.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unreconciled stale publication",
    ):
        qualification.publish_completion(context, intent=intent)
    assert not (
        context.qualification_base / qualification.MARKER_NAME
    ).exists()


def test_terminal_tree_seal_rejects_external_hardlink_alias(
    tmp_path: Path,
) -> None:
    context, _intent = _context_and_intent(tmp_path)
    evidence = context.run_root / "sealed-evidence.json"
    evidence.write_bytes(qualification._canonical_bytes({"sealed": True}))
    evidence.chmod(0o444)
    alias = tmp_path / "external-hardlink-alias.json"
    alias.hardlink_to(evidence)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="non-unique or unsafe member",
    ):
        qualification._seal_tree_read_only(
            context.run_root,
            description="hardlinked terminal run",
        )


def test_sealed_observations_prove_all_gates_and_publish_renderer_marker(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    evaluation = qualification.evaluate_observations(observations)
    assert evaluation["passed"] is True
    assert evaluation["health_soak_384_seconds"] == 7_200
    assert evaluation["loaded_384_seconds"] == 7_200
    assert (
        evaluation["loaded_384_useful_qids"]
        == qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS
        == 16_833
    )
    assert evaluation["loaded_384_observation_count"] == 13
    assert evaluation["configured_client_ceiling"] == 384
    assert evaluation["certified_saturation_target"] == 278
    assert evaluation["certified_saturation_target_cuts"] is True
    assert evaluation["load_execution"]["configured_client_ceiling"] == 384
    assert evaluation["load_execution"]["certified_saturation_target"] == 278
    assert (
        evaluation["load_execution"]["work_conserving_refill"]
        is True
    )
    assert (
        evaluation["load_execution"][
            "rate_denominator_includes_refill_wall_time"
        ]
        is True
    )
    assert evaluation["throughput_qids_per_day"] >= 201_994
    assert (
        evaluation["throughput_unit"]
        == "trusted_qid_execution_events"
    )
    assert evaluation["unique_design"]["cells"] == 768
    assert evaluation["unique_design"]["qids"] == 15_360
    assert len(
        evaluation["unique_design"]["semantic_reference_cycle"]
    ) == 64
    assert evaluation["load_execution"]["repeated_coordinates"] is True
    assert evaluation["peak_active_cells"] == {
        "24": 24,
        "96": 96,
        "192": 192,
        "384": 278,
    }

    marker = qualification.publish_completion(context, intent=intent)
    marker_path = context.qualification_root / qualification.MARKER_NAME
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o444
    assert set(marker) == {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "chain_id",
        "manifest",
        "manifest_sha256",
            "protected_capacity",
            "attempt",
            "evidence",
            "cells",
        "qids",
        "unique_design",
        "load_execution",
        "ceilings",
        "health_soak_384_seconds",
        "loaded_384_seconds",
        "loaded_384_useful_qids",
        "loaded_384_observation_count",
        "configured_client_ceiling",
        "certified_saturation_target",
        "certified_saturation_target_cuts",
        "throughput_qids_per_day",
        "throughput_unit",
        "every_stratum_progress",
        "integrity_incidents",
        "transport_censor_incidents",
            "qualification_id",
        }
    assert marker["evidence"]["path"] == str(
        (context.qualification_root / qualification.EVIDENCE_NAME).resolve()
    )
    assert len(marker["evidence"]["evidence_id"]) == 64
    identity = dict(marker)
    qualification_id = identity.pop("qualification_id")
    assert qualification_id == renderer._sha256_bytes(
        renderer._canonical_json(identity)
    )

    verified = qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert verified["qualification_id"] == qualification_id
    assert verified["cells"] == 768
    assert verified["qids"] == 15_360


def test_fixed_completion_marker_is_committed_after_attempt_trees_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _passing_observations(context, intent)
    root_marker = context.qualification_base / qualification.MARKER_NAME
    write_once = qualification._write_once

    def interrupt_fixed_commit(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        if path == root_marker:
            raise qualification.ThroughputQualificationError(
                "injected fixed-marker interruption"
            )
        write_once(
            path,
            payload,
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(
        qualification,
        "_write_once",
        interrupt_fixed_commit,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="injected fixed-marker interruption",
    ):
        qualification.publish_completion(context, intent=intent)
    assert not root_marker.exists()
    qualification._assert_tree_read_only(
        context.qualification_root,
        description="interrupted successful attempt",
    )
    qualification._assert_tree_read_only(
        context.run_root,
        description="interrupted successful run",
    )

    monkeypatch.setattr(qualification, "_write_once", write_once)
    marker = qualification.publish_completion(context, intent=intent)
    verified = qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert root_marker.read_bytes() == qualification._canonical_bytes(marker)
    assert verified["attempt_id"] == context.attempt_pointer["attempt_id"]


def test_verify_only_rejects_tampered_sealed_semantic_evidence(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _passing_observations(context, intent)
    qualification.publish_completion(context, intent=intent)

    semantic_path = (
        context.qualification_root
        / qualification.SEMANTIC_DIRECTORY
        / qualification._evidence_filename("SEMANTIC", 1)
    )
    payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    payload["useful_qids"] += 1
    semantic_path.chmod(0o644)
    semantic_path.write_bytes(qualification._canonical_bytes(payload))
    semantic_path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="invalid semantic_id",
    ):
        qualification.verify_completed_qualification(
            context.chain_manifest,
            verify_chain=False,
            verify_renderer=False,
        )


def test_partial_orphan_observation_fails_closed(tmp_path: Path) -> None:
    context, intent = _context_and_intent(tmp_path)
    orphan = (
        context.qualification_root
        / qualification.SCHEDULER_DIRECTORY
        / qualification._evidence_filename("SCHEDULER", 0)
    )
    _sealed(orphan, {"partial": True})

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="orphaned partial",
    ):
        qualification.load_observations(
            context.qualification_root, intent=intent
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("censor", "integrity, censor, or namespace incident"),
        ("stratum", "semantic or execution-event progress regressed"),
        ("gap", "evidence gap"),
        ("throughput", "below 201,994"),
    ],
)
def test_acceptance_fails_closed_on_scientific_or_timing_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    if mutation == "censor":
        observations[5]["semantic"]["transport_censor_incidents"] = 1
    elif mutation == "stratum":
        observations[6]["semantic"]["useful_qids"] = 1
    elif mutation == "gap":
        # Preserve increasing time while introducing a >660-second 384 gap.
        observations[6]["receipt"]["captured_timestamp"] = (
            observations[5]["receipt"]["captured_timestamp"] + 661
        )
    else:
        # One event below the exact 7,200-second threshold cannot qualify.
        events = (
            696 + qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS - 1
        )
        for observation in observations[-2:]:
            semantic = observation["semantic"]
            semantic["trusted_qid_execution_events"] = events
            semantic["replay_qid_execution_events"] = (
                events - qualification.TOTAL_QIDS
            )
            semantic["load_strata_progress"] = _progress(events)
            semantic["load_cycle_inventory"][1][
                "validated_execution_events"
            ] = events - qualification.TOTAL_QIDS

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match=message,
    ):
        qualification.evaluate_observations(observations)


def test_peak_384_once_then_idle_cannot_substitute_for_loaded_interval(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    # Preserve the exact peak and the full two-hour clean ceiling-384 soak, but
    # make every later pre-completion scheduler observation idle even though all
    # 768 cells remain unfinished.  The old peak-plus-soak contract accepted this.
    for observation in observations[5:]:
        semantic = observation["semantic"]
        if semantic["states"] == {"complete": qualification.CELL_COUNT}:
            continue
        observation["scheduler"]["active_qualification_cells"] = 0
        semantic["states"] = {"missing": qualification.CELL_COUNT}

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="non-resettable load window lost its certified capacity",
    ):
        qualification.evaluate_observations(observations)


def test_exact_384_occupancy_without_trusted_progress_is_not_loaded(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    # Retain repeated exact occupancy while backlog permits, but move all useful
    # progress to the terminal observation.  Idle scientific work cannot be
    # relabeled as a loaded interval merely because clients occupied slots.
    for observation in observations:
        if (
            observation["scheduler"]["ceiling"] == 384
            and observation["scheduler"]["active_qualification_cells"]
            == intent["certified_saturation_target"]
        ):
            semantic = observation["semantic"]
            semantic["trusted_qid_execution_events"] = 696
            semantic["replay_qid_execution_events"] = 0
            semantic["load_strata_progress"] = _progress(696)
            semantic["load_cycle_inventory"] = [
                {
                    **semantic["load_cycle_inventory"][0],
                    "validated_execution_events": 696,
                    "unfinished_assignments": semantic[
                        "unfinished_load_assignments"
                    ],
                }
            ]

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="trusted execution progress|load window",
    ):
        qualification.evaluate_observations(observations)


def test_rotation_counts_are_optimal_under_structural_context_constraints() -> None:
    cells = qualification.generate_qualification_cells()
    by_topology = Counter(cell.topology.value for cell in cells)
    assert by_topology == {
        "single_agent": 228,
        "independent": 180,
        "decentralized": 180,
        "centralized": 180,
    }
    # Context is balanced exactly where it is a real treatment.  Single-agent and
    # independent cells are correctly pinned to the canonical artifact-only value.
    sharing = [
        cell
        for cell in cells
        if cell.topology
        in {qualification.Topology.DECENTRALIZED, qualification.Topology.CENTRALIZED}
    ]
    assert Counter(cell.context_share_level.value for cell in sharing) == {
        "artifact_only": 180,
        "plus_cot": 180,
    }
    assert all(
        cell.context_share_level.value == "artifact_only"
        for cell in cells
        if cell.topology
        in {qualification.Topology.SINGLE_AGENT, qualification.Topology.INDEPENDENT}
    )


def test_paused_but_draining_control_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = qualification.load_qualification_context(
        _chain_manifest(tmp_path), verify_chain=False
    )
    control_value = {
        "desired_state": "paused",
        "drain_requested": True,
        "rollout_generation": 0,
        "immutable_sha256": "f" * 64,
        "immutable": {
            "runs": [
                {"run_id": run_id}
                for run_id in sorted(qualification.PRODUCTION_RUN_IDS)
            ]
        },
        "readiness": {"smoke_runs": {"passed": True}},
        "admission": {},
        "admission_ramp": {},
        "admission_safety_hold": {},
    }
    monkeypatch.setattr(
        qualification.control,
        "load_control",
        lambda *_args, **_kwargs: control_value,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="requires paused",
    ):
        qualification.load_paused_control(context)


def test_paused_control_requires_current_capacity_authority_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = qualification.load_qualification_context(
        _chain_manifest(tmp_path), verify_chain=False
    )
    control_value = {
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": 0,
        "immutable_sha256": "f" * 64,
        "immutable": {
            "runs": [
                {"run_id": run_id}
                for run_id in sorted(qualification.PRODUCTION_RUN_IDS)
            ]
        },
        "readiness": {"smoke_runs": {"passed": True}},
        "admission": {},
        "admission_ramp": {},
        "admission_safety_hold": {},
    }
    monkeypatch.setattr(
        qualification.control,
        "load_control",
        lambda *_args, **_kwargs: control_value,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="static-capacity, protected-capacity",
    ):
        qualification.load_paused_control(context)


def test_runtime_generation_is_catalog_bound_not_blindly_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(tmp_path)
    generation = dict(intent["readiness_generation"])
    generation.update(
        {
            "rollout_generation": 7,
            "capacity_generation": 3,
            "fleet_contract_sha256": "d" * 64,
            "release_fleet_contract_sha256": "e" * 64,
        }
    )
    runtime_intent = {**intent, "readiness_generation": generation}
    control_value = {
        "rollout_generation": 6,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    attestation = {
                "schema_version": (
                    qualification.EXECUTION_AUTHORITY_SCHEMA_VERSION
                ),
        "generation": 7,
        "path": str((tmp_path / "attestation.json").resolve()),
        "sha256": "2" * 64,
        "attestation_id": "3" * 64,
        "lease_path": str((tmp_path / "lease.json").resolve()),
    }
    inherited = {
        key: "x"
        for key in qualification.dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS
    }
    inherited.update(
        {
            "ASYS_RELEASE_GIT_COMMIT": COMMIT,
            "ASYS_PROTECTED_CAPACITY_MARKER": str(
                context.protected_capacity_contract.path
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
                context.protected_capacity_contract.sha256
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": (
                context.protected_capacity_contract.marker_id
            ),
            "ASYS_MODEL_CONTRACT_SHA256": "4" * 64,
            "ASYS_FLEET_CONTRACT_SHA256": "d" * 64,
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "fleet.json").resolve()
            ),
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "e" * 64,
            "ASYS_CAPACITY_GENERATION": "3",
            "ASYS_HARNESS_ENVIRONMENT_SHA256": "5" * 64,
            "ASYS_SERVING_ENVIRONMENT_SHA256": "6" * 64,
            "ASYS_ROLLOUT_GENERATION": "7",
            "ASYS_IMMUTABLE_PINS_SHA256": "7" * 64,
            "ASYS_RUNTIME_ATTESTATION": attestation["path"],
            "ASYS_RUNTIME_ATTESTATION_SHA256": attestation["sha256"],
            "ASYS_RUNTIME_INTEGRITY_LEASE": attestation["lease_path"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    projected: dict[str, object] = {}
    monkeypatch.setattr(
        qualification.control,
        "ensure_runtime_integrity_attestation",
        lambda state, value, *, generation, force_full: (
            projected.update(
                state=state,
                source=value,
                generation=generation,
                force_full=force_full,
            )
            or attestation
        ),
    )
    monkeypatch.setattr(
        qualification.control,
        "validate_runtime_integrity_attestation",
        lambda value, *, verify_metadata: (
            projected.update(
                projected_control=value,
                verify_metadata=verify_metadata,
            )
            or attestation
        ),
    )
    monkeypatch.setattr(
        qualification.control,
        "production_environment",
        lambda value: (
            projected.update(production_control=value) or inherited
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_artifact_policy",
        lambda *_args, **_kwargs: SimpleNamespace(file_sha256="1" * 64),
    )
    environment = qualification._execution_environment(
        context,
        control_value=control_value,
        intent=runtime_intent,
    )
    assert environment["ASYS_ROLLOUT_GENERATION"] == "7"
    assert projected["generation"] == 7
    assert projected["state"] == context.qualification_root
    assert projected["force_full"] is False
    projected_control = projected["projected_control"]
    assert isinstance(projected_control, dict)
    assert projected_control["rollout_generation"] == 7
    assert (
        projected_control[qualification.control.RUNTIME_ATTESTATION_STATE_KEY]
        == attestation
    )
    assert projected["production_control"] is projected_control
    assert projected["verify_metadata"] is True
    result_row = {
        field: generation[field]
        for field in (
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
        )
    }
    assert qualification._row_matches_readiness_generation(
        result_row, generation
    )
    assert not qualification._row_matches_readiness_generation(
        {**result_row, "rollout_generation": 8},
        generation,
    )

    drifted = {
        **runtime_intent,
        "readiness_generation": {
            **generation,
            "rollout_generation": 8,
        },
    }
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="exact next rollout",
    ):
        qualification._execution_environment(
            context,
            control_value=control_value,
            intent=drifted,
        )


def test_execution_authority_is_sealed_and_dispatcher_consumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(
        tmp_path, write_execution_authority=False
    )
    python = context.harness_prefix / "bin" / "python"
    dispatcher = (
        context.release_worktree / "slurm" / "dispatch_sweeps.py"
    )
    template = (
        context.release_worktree
        / "slurm"
        / "run_dispatch_batch.sbatch.tmpl"
    )
    python.parent.mkdir(parents=True)
    dispatcher.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    template.write_bytes(
        qualification.dispatch_sweeps.ARRAY_TEMPLATE.read_bytes()
    )
    python.chmod(0o555)
    dispatcher.chmod(0o444)
    template.chmod(0o444)
    environment = {
        key: "x"
        for key in qualification.dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS
    }
    readiness = intent["readiness_generation"]
    assert isinstance(readiness, dict)
    environment.update(
        {
            "ASYS_RELEASE_GIT_COMMIT": COMMIT,
            "ASYS_PROTECTED_CAPACITY_MARKER": str(
                context.protected_capacity_contract.path
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
                context.protected_capacity_contract.sha256
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": (
                context.protected_capacity_contract.marker_id
            ),
            "ASYS_MODEL_CONTRACT_SHA256": "3" * 64,
            "ASYS_FLEET_CONTRACT_SHA256": readiness[
                "fleet_contract_sha256"
            ],
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "fleet.json").resolve()
            ),
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": readiness[
                "release_fleet_contract_sha256"
            ],
            "ASYS_CAPACITY_GENERATION": str(
                readiness["capacity_generation"]
            ),
            "ASYS_HARNESS_ENVIRONMENT_SHA256": "4" * 64,
            "ASYS_SERVING_ENVIRONMENT_SHA256": "5" * 64,
            "ASYS_ROLLOUT_GENERATION": str(
                readiness["rollout_generation"]
            ),
            "ASYS_IMMUTABLE_PINS_SHA256": "6" * 64,
            "ASYS_RUNTIME_ATTESTATION": str(
                (tmp_path / "attestation.json").resolve()
            ),
            "ASYS_RUNTIME_ATTESTATION_SHA256": "7" * 64,
            "ASYS_RUNTIME_INTEGRITY_LEASE": str(
                (tmp_path / "lease.json").resolve()
            ),
            "ASYS_ARTIFACT_POLICY_SHA256": "8" * 64,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    verified: list[dict[str, object]] = []
    monkeypatch.setattr(
        qualification.dispatch_sweeps.runtime_integrity,
        "verify_generation_lease",
        lambda **kwargs: verified.append(kwargs) or {},
    )

    authority = qualification.create_or_load_execution_authority(
        context,
        intent=intent,
        control_value={},
        environment=environment,
    )
    path = context.qualification_root / qualification.EXECUTION_AUTHORITY_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert authority.runtime_environment == environment
    assert authority.payload["intent_id"] == intent["intent_id"]
    assert authority.execution["python"] == str(python.resolve())
    assert verified[-1]["generation"] == readiness["rollout_generation"]
    command = qualification.dispatcher_command(
        context,
        client_partition="ou_bcs_normal",
        client_qos="normal",
    )
    assert command[
        command.index("--qualification-execution-authority") + 1
    ] == str(path)


def test_terminal_failure_names_additive_bottleneck_and_requires_fresh_intent(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycles = _ensure_passing_cycle_roots(context, intent)
    ledger = qualification.dispatch_sweeps._empty_ledger()
    ledger["qualification_profile_pressure"] = {
        "eight": {
            "server_pool_root": str(context.server_pool_root),
            "serving_profile": "8B",
            "eligible_cells": 10,
            "backlog_fanout_work": 100,
            "live_replicas": 2,
            "backlog_work_per_replica": 50.0,
            "observed_poll": 2,
        },
        "long": {
            "server_pool_root": str(context.server_pool_root),
            "serving_profile": "32B-long",
            "eligible_cells": 8,
            "backlog_fanout_work": 120,
            "live_replicas": 2,
            "backlog_work_per_replica": 60.0,
            "observed_poll": 3,
        },
    }
    _bind_test_pressure_ledger(context, intent, cycles[0], ledger)
    context.dispatcher_state.mkdir(parents=True)
    qualification.dispatch_sweeps._atomic_write_json(
        context.dispatcher_state / "ledger.json", ledger
    )

    scaling = qualification._scaling_requirement(context)
    assert scaling == {
        "serving_profile": "32B-long",
        "server_pool_root": str(context.server_pool_root),
        "backlog_fanout_work": 120,
        "live_replicas": 2,
        "backlog_work_per_replica": 60.0,
        "additional_replicas": 1,
        "tensor_parallel_size": 2,
        "additional_gpus": 2,
        "requirement": "add one TP=2 replica pair (2 GPUs)",
        "capacity_mutated": False,
    }
    failure_reason = "qualification throughput is below threshold"
    _prepare_terminal_failure_drain(
        context, intent, reason=failure_reason
    )
    failure = qualification._publish_terminal_failure(
        context,
        intent=intent,
        reason=failure_reason,
    )
    assert failure["scheduler_capacity_mutated"] is False
    message = qualification._terminal_failure_message(context)
    assert message is not None
    assert "32B-long" in message
    assert "add one TP=2 replica pair (2 GPUs)" in message
    assert "fresh qualification namespace and intent" in message

    intent_path = context.qualification_root / qualification.INTENT_NAME
    original = intent_path.read_bytes()
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="generation changed after qualification intent",
    ):
        qualification.create_or_load_intent(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            readiness_generation={
                **intent["readiness_generation"],  # type: ignore[arg-type]
                "rollout_generation": 2,
            },
            control_guard=intent["control_guard"],  # type: ignore[arg-type]
            admission_capacity_certificate=intent[
                "admission_capacity_certificate"
            ],  # type: ignore[arg-type]
            now=1_000.0,
        )
    assert intent_path.read_bytes() == original


@pytest.mark.parametrize(
    ("reason", "expected_type"),
    [
        (
            qualification._capacity_shortfall_reason(
                qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS - 1,
                qualification.HEALTH_SOAK_384_SECONDS,
            ),
            qualification.QualificationCapacityTransitionRequired,
        ),
        (
            "qualification contains an integrity, censor, or namespace incident",
            qualification.ThroughputQualificationError,
        ),
    ],
)
def test_terminal_failure_first_run_and_restart_preserve_exact_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    expected_type: type[qualification.ThroughputQualificationError],
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _ensure_passing_cycle_roots(context, intent)
    _write_test_pressure_ledger(context, intent)
    _prepare_terminal_failure_drain(context, intent, reason=reason)

    with pytest.raises(expected_type) as first_run:
        qualification._publish_and_raise_terminal_failure(
            context,
            intent=intent,
            reason=reason,
        )
    assert type(first_run.value) is expected_type

    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_qualification_run",
        lambda *_args, **_kwargs: pytest.fail(
            "sealed failure replay must remain read-only"
        ),
    )

    with pytest.raises(expected_type) as replay:
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            capacity_certificate_loader=lambda *_args, **_kwargs: intent[
                "admission_capacity_certificate"
            ],
        )
    assert type(replay.value) is expected_type


def test_main_maps_only_capacity_shortfall_to_exit_76(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "execute",
        "--chain-manifest",
        str(tmp_path / "chain.json"),
    ]
    capacity_reason = qualification._capacity_shortfall_reason(
        qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS - 1,
        qualification.HEALTH_SOAK_384_SECONDS,
    )

    def raise_capacity(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise qualification.QualificationCapacityTransitionRequired(
            capacity_reason
        )

    monkeypatch.setattr(
        qualification,
        "execute_qualification",
        raise_capacity,
    )
    assert qualification.main(argv) == 76
    capacity_output = capsys.readouterr()
    assert capacity_output.out == ""
    assert "exact additive capacity transition" in capacity_output.err

    def raise_integrity(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise qualification.ThroughputQualificationError(
            "qualification contains an integrity incident"
        )

    monkeypatch.setattr(
        qualification,
        "execute_qualification",
        raise_integrity,
    )
    assert qualification.main(argv) == 2
    integrity_output = capsys.readouterr()
    assert integrity_output.out == ""
    assert "failed closed" in integrity_output.err


def test_scaling_requirement_includes_replay_only_bottleneck(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycle0, cycle1 = _ensure_passing_cycle_roots(context, intent)
    for cycle, profile, work, replicas in (
        (cycle0, "8B", 100, 2),
        (cycle1, "32B-long", 900, 3),
    ):
        ledger = qualification.dispatch_sweeps._empty_ledger()
        ledger["qualification_profile_pressure"] = {
            profile: {
                "server_pool_root": str(context.server_pool_root),
                "serving_profile": profile,
                "eligible_cells": 10,
                "backlog_fanout_work": work,
                "live_replicas": replicas,
                "backlog_work_per_replica": work / replicas,
                "observed_poll": cycle.cycle_index + 1,
            }
        }
        _bind_test_pressure_ledger(
            context, intent, cycle, ledger
        )
        cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
        qualification.dispatch_sweeps._atomic_write_json(
            cycle.dispatcher_state / "ledger.json", ledger
        )

    scaling = qualification._scaling_requirement(context)
    assert scaling["serving_profile"] == "32B-long"
    assert scaling["backlog_fanout_work"] == 900
    assert scaling["backlog_work_per_replica"] == 300.0
    assert scaling["additional_gpus"] == 2


def test_signed_sub384_saturation_target_is_healthy_and_bounds_admission(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    certificate = intent["admission_capacity_certificate"]
    assert certificate["wave_passed"] is False
    assert intent["configured_client_ceiling"] == 384
    assert intent["certified_saturation_target"] == 278
    assert qualification._stage_dispatch_batch(
        ceiling=384,
        active=254,
        useful_qids=0,
        saturation_target=278,
    ) == 24
    assert qualification._stage_dispatch_batch(
        ceiling=384,
        active=278,
        useful_qids=0,
        saturation_target=278,
    ) == 0
    assert not (
        context.qualification_root / qualification.FAILURE_DRAIN_INTENT_NAME
    ).exists()


def _legacy_failed_attempt_is_preserved_and_exact_additive_generation_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context1, intent1 = _context_and_intent(
        tmp_path, write_execution_authority=False
    )
    base = qualification.load_qualification_context(
        context1.chain_manifest, verify_chain=False
    )
    immutable_release = {
        "git_commit": COMMIT,
        "release_tag_object": base.release_tag_object,
        "source_tree_sha256": base.source_tree_sha256,
    }
    old_fleet = (tmp_path / "old-fleet.json").resolve()
    new_fleet = (tmp_path / "new-fleet.json").resolve()
    replica0 = {
        "replica_index": 0,
        "replica_id": "replica-0",
        "scheduler_job_name": "server-0",
    }
    replica1 = {
        "replica_index": 1,
        "replica_id": "replica-1",
        "scheduler_job_name": "server-1",
    }
    common = {
        "schema_version": 1,
        "fleet_id": "schema5-v1",
        "release_id": renderer.RELEASE_ID,
        "model_contract_sha256": "3" * 64,
        "offline_environment": {"HF_HUB_OFFLINE": "1"},
        "server_pool": {"pool_id": "schema5-v1"},
    }
    profile = {
        "serving_profile": "8B",
        "model_size": "8B",
    }
    old_payload = {
        **common,
        "logical_replica_count": 1,
        "allocated_gpu_count": 1,
        "profiles": [{**profile, "replicas": [replica0]}],
    }
    new_payload = {
        **common,
        "logical_replica_count": 2,
        "allocated_gpu_count": 2,
        "profiles": [{**profile, "replicas": [replica0, replica1]}],
    }
    old_fleet.write_bytes(qualification._canonical_bytes(old_payload))
    new_fleet.write_bytes(qualification._canonical_bytes(new_payload))
    old_fleet.chmod(0o444)
    new_fleet.chmod(0o444)
    old_sha = qualification._sha256_file(old_fleet)
    new_sha = qualification._sha256_file(new_fleet)
    readiness1 = dict(intent1["readiness_generation"])
    readiness1["fleet_contract_sha256"] = old_sha
    # The attempt pointer is the generation authority, so its fleet digest must
    # already name the old contract used by the failed task authority.
    pointer1_path = context1.attempt_pointer_path
    assert pointer1_path is not None
    pointer1 = dict(context1.attempt_pointer)
    pointer1_identity = dict(pointer1)
    pointer1_identity.pop("pointer_id")
    pointer1_identity["readiness_generation"] = readiness1
    pointer1 = qualification._with_identity(
        pointer1_identity, "pointer_id"
    )
    pointer1_path.chmod(0o644)
    pointer1_path.write_bytes(qualification._canonical_bytes(pointer1))
    pointer1_path.chmod(0o444)
    context1 = qualification._attempt_context_from_pointer(
        base, path=pointer1_path, pointer=pointer1
    )
    # Repair the current cache to the test's exact old-fleet pointer bytes.
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    current_path.unlink()
    qualification._replace_current_attempt(
        base,
        path=pointer1_path,
        pointer=pointer1,
        known=[(pointer1_path, pointer1)],
    )
    intent1_identity = qualification._intent_identity(
        context1,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        readiness_generation=readiness1,
        control_guard=intent1["control_guard"],
        created_timestamp=float(intent1["created_timestamp"]),
    )
    intent1 = qualification._with_identity(intent1_identity, "intent_id")
    intent1_path = context1.qualification_root / qualification.INTENT_NAME
    intent1_path.chmod(0o644)
    intent1_path.write_bytes(qualification._canonical_bytes(intent1))
    intent1_path.chmod(0o444)
    authority1 = qualification._with_identity(
        {
            "schema_version": (
                qualification.EXECUTION_AUTHORITY_SCHEMA_VERSION
            ),
            "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": intent1["intent_id"],
            "chain_id": context1.chain_id,
            "run_id": qualification.QUALIFICATION_RUN_ID,
            "run_root": str(context1.run_root),
            "readiness_generation": dict(
                intent1["readiness_generation"]
                ),
                "release_git_commit": COMMIT,
                "release_tag_object": base.release_tag_object,
                "source_tree_sha256": base.source_tree_sha256,
                "qualification_runner_source_sha256": (
                    base.qualification_runner_source_sha256
                ),
                "runtime_environment": {
                "ASYS_FLEET_CONTRACT_PATH": str(old_fleet),
                "ASYS_FLEET_CONTRACT_SHA256": old_sha,
            },
        },
        "authority_id",
    )
    qualification._write_once(
        context1.qualification_root
        / qualification.EXECUTION_AUTHORITY_NAME,
        authority1,
        description="old attempt authority",
    )
    cycles1 = _ensure_passing_cycle_roots(context1, intent1)
    ledger = qualification.dispatch_sweeps._empty_ledger()
    ledger["qualification_profile_pressure"] = {
        "eight": {
            "server_pool_root": str(context1.server_pool_root),
            "serving_profile": "8B",
            "eligible_cells": 10,
            "backlog_fanout_work": 100,
            "live_replicas": 1,
            "backlog_work_per_replica": 100.0,
            "observed_poll": 1,
        }
    }
    _bind_test_pressure_ledger(
        context1, intent1, cycles1[0], ledger
    )
    context1.dispatcher_state.mkdir(parents=True)
    qualification.dispatch_sweeps._atomic_write_json(
        context1.dispatcher_state / "ledger.json", ledger
    )
    failure_reason = "qualification throughput is below threshold"
    _prepare_terminal_failure_drain(
        context1, intent1, reason=failure_reason
    )
    failure = qualification._publish_terminal_failure(
        context1,
        intent=intent1,
        reason=failure_reason,
    )
    old_pointer_bytes = pointer1_path.read_bytes()
    old_failure_bytes = (
        context1.qualification_root / qualification.FAILURE_NAME
    ).read_bytes()
    qualification._assert_tree_read_only(
        context1.qualification_root,
        description="failed attempt",
    )
    qualification._assert_tree_read_only(
        context1.run_root,
        description="failed run",
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="fresh additive fleet/readiness generation",
    ):
        qualification.create_or_load_attempt_context(
            base,
            control_value={"immutable": immutable_release},
            readiness_generation=readiness1,
            now=1_900.0,
        )
    assert len(
        qualification._load_attempt_pointers(base)
    ) == 1

    readiness2 = {
        **readiness1,
        "catalog_id": "d" * 64,
        "marker_path": str((tmp_path / "TRUSTED_GENERATION_2.json").resolve()),
        "marker_sha256": "e" * 64,
        "inventory_sha256": "f" * 64,
        "catalog_payload_sha256": "0" * 64,
        "fleet_contract_sha256": new_sha,
        "capacity_generation": 2,
        "rollout_generation": 2,
    }
    new_replica = SimpleNamespace(replica_index=1, gpus_per_replica=1)
    monkeypatch.setattr(
        qualification.control,
        "effective_fleet_contract_binding",
        lambda *_args, **_kwargs: {
            "capacity_generation": 2,
            "path": str(new_fleet),
            "sha256": new_sha,
            "fleet_id": "schema5-v1",
            "logical_replicas": 2,
            "allocated_gpus": 2,
            "profile_replicas": {"8B": 2},
            "is_capacity_overlay": True,
        },
    )
    monkeypatch.setattr(
        qualification.control,
        "load_effective_fleet_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            by_profile={
                "8B": (
                    SimpleNamespace(
                        replica_index=0, gpus_per_replica=1
                    ),
                    new_replica,
                )
            }
        ),
    )
    failed_comment = (
        f"asys:s5-recovery-v1.2-r11:{base.chain_id}:"
        "g0000:throughput_qualification"
    )
    receipt = qualification._with_identity(
        {
            "schema_version": 1,
            "protocol": "schema5-v1.2-r11-recovery-chain-submission",
            "passed": True,
            "chain_id": base.chain_id,
            "manifest": str(base.chain_manifest),
            "manifest_sha256": base.chain_manifest_sha256,
            "jobs": [
                {
                    "name": "throughput_qualification",
                    "job_id": "12345",
                    "comment": failed_comment,
                }
            ],
        },
        "receipt_id",
    )
    receipt_path = _sealed(tmp_path / "SUBMISSION.json", receipt)
    transition_control = {"immutable": immutable_release}
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (transition_control, _guard(rollout_generation=1)),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda _context, _control: readiness2,
    )
    transition_report = (
        qualification.publish_capacity_transition_authority(
            context1.chain_manifest,
            submission_receipt=receipt_path,
            failed_job_id="12345",
            failed_comment=failed_comment,
            apply=True,
            verify_chain=False,
            now=1_950.0,
        )
    )
    transition = transition_report["transition"]
    assert transition_report["status"] == "published"
    assert transition["failure"]["failure_id"] == failure["failure_id"]
    assert transition["additive_transition"][
        "to_fleet_contract_sha256"
    ] == new_sha
    assert transition["failed_stage"] == {
        "name": "throughput_qualification",
        "job_id": "12345",
        "comment": failed_comment,
    }
    context2 = qualification.create_or_load_attempt_context(
        base,
        control_value={"immutable": immutable_release},
        readiness_generation=readiness2,
        now=2_000.0,
    )
    assert context2.qualification_root != context1.qualification_root
    assert context2.run_root != context1.run_root
    assert context2.attempt_pointer["attempt_ordinal"] == 2
    assert context2.attempt_pointer["additive_retry"] == {
        "previous_failure_id": failure["failure_id"],
        "serving_profile": "8B",
        "additional_replicas": 1,
        "tensor_parallel_size": 1,
        "from_capacity_generation": 1,
        "to_capacity_generation": 2,
        "from_rollout_generation": 1,
        "to_rollout_generation": 2,
        "from_fleet_contract_sha256": old_sha,
        "to_fleet_contract_sha256": new_sha,
        "validation": (
            "schema5_control._assert_additive_capacity_contract+"
            "exact-required-profile-delta"
        ),
    }
    assert pointer1_path.read_bytes() == old_pointer_bytes
    assert (
        context1.qualification_root / qualification.FAILURE_NAME
    ).read_bytes() == old_failure_bytes

    guard2 = _guard(rollout_generation=1)
    intent2 = qualification.create_or_load_intent(
        context2,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        readiness_generation=readiness2,
        control_guard=guard2,
        now=2_100.0,
    )
    qualification._write_once(
        context2.qualification_root / qualification.PLAN_NAME,
        qualification.build_load_plan(),
        description="second attempt plan",
    )
    authority2 = qualification._with_identity(
        {
            "schema_version": (
                qualification.EXECUTION_AUTHORITY_SCHEMA_VERSION
            ),
            "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": intent2["intent_id"],
        },
        "authority_id",
    )
    qualification._write_once(
        context2.qualification_root
        / qualification.EXECUTION_AUTHORITY_NAME,
        authority2,
        description="second attempt authority",
    )
    context2.run_root.mkdir(parents=True)
    _passing_observations(context2, intent2)
    marker = qualification.publish_completion(context2, intent=intent2)
    verified = qualification.verify_completed_qualification(
        context2.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert verified["attempt_id"] == context2.attempt_pointer["attempt_id"]
    assert marker["attempt"]["rollout_generation"] == 2
    assert marker["attempt"]["capacity_generation"] == 2
    assert (
        qualification._read_json(
            base.qualification_base / qualification.MARKER_NAME,
            description="root success marker",
            sealed=True,
        )
        == marker
    )


@pytest.mark.parametrize(
    "failure_description",
    ["semantic observation 0", "qualification observation 0"],
)
def test_observation_transaction_resumes_exact_crash_preimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_description: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    original = qualification._write_once
    crashed = False

    def interrupted_write(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal crashed
        if description == failure_description and not crashed:
            crashed = True
            raise RuntimeError("injected crash")
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(qualification, "_write_once", interrupted_write)
    with pytest.raises(RuntimeError, match="injected crash"):
        _record(
            context,
            intent,
            sequence=0,
            timestamp=1_000.0,
            ceiling=24,
            active=0,
            useful=0,
        )
    monkeypatch.setattr(qualification, "_write_once", original)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="transaction is incomplete",
    ):
        qualification.load_observations(
            context.qualification_root, intent=intent
        )
    recovered = qualification.load_observations(
        context.qualification_root,
        intent=intent,
        recover_transactions=True,
    )
    assert len(recovered) == 1
    assert recovered[0]["scheduler"]["active_qualification_cells"] == 0
    assert recovered[0]["semantic"]["useful_qids"] == 0


@pytest.mark.parametrize(
    ("ceiling", "active", "useful", "expected"),
    [
        (24, 0, 0, 24),
        (24, 18, 100, 6),
        (96, 71, 1_000, 24),
        (96, 90, 1_100, 6),
        (384, 383, 15_000, 1),
        # Semantic-reference completion does not stop estimand-excluded load
        # replay; only the durable window-end admission fence does.
        (384, 0, qualification.TOTAL_QIDS, 24),
    ],
)
def test_stage_fill_is_exact_when_tasks_finish_between_polls(
    ceiling: int,
    active: int,
    useful: int,
    expected: int,
) -> None:
    assert (
        qualification._stage_dispatch_batch(
            ceiling=ceiling,
            active=active,
            useful_qids=useful,
        )
        == expected
    )


def test_stage_fill_stops_only_after_window_end_admission_fence() -> None:
    assert (
        qualification._stage_dispatch_batch(
            ceiling=384,
            active=0,
            useful_qids=qualification.TOTAL_QIDS,
            admission_closed=True,
        )
        == 0
    )


@pytest.mark.parametrize(
    "fence_name",
    [
        qualification.LOAD_WINDOW_END_INTENT_NAME,
        qualification.FAILURE_DRAIN_INTENT_NAME,
    ],
)
def test_load_window_refill_is_disabled_by_either_durable_fence(
    tmp_path: Path,
    fence_name: str,
) -> None:
    root = tmp_path / "qualification"
    _sealed(root / qualification.LOAD_WINDOW_INTENT_NAME, {"start": 1})
    assert qualification._load_window_refill_permitted(root) is True
    _sealed(root / fence_name, {"admission_closed": True})
    assert qualification._load_window_refill_permitted(root) is False


def test_accepted_fast_array_is_counted_through_scheduler_visibility_grace() -> None:
    ledger = {
        "jobs": {
            "12345": {
                "state": "submitted",
                "submitted_at": 1_000.0,
                "task_count": 24,
                "tasks": [
                    {"run_id": qualification.QUALIFICATION_RUN_ID}
                    for _ in range(24)
                ],
            }
        }
    }
    active, job_ids, foreign = qualification._active_task_count(
        scheduler_jobs=[],
        ledger=ledger,
        captured_timestamp=1_001.0,
    )
    assert active == 24
    assert job_ids == ["12345"]
    assert foreign == []

    expired, _, _ = qualification._active_task_count(
        scheduler_jobs=[],
        ledger=ledger,
        captured_timestamp=1_300.0,
    )
    assert expired == 0

    terminal = SimpleNamespace(
        active=False,
        job_id="12345_0",
        job_name="asys-dispatch-terminal",
        comment="",
        command="/sealed/batch.sbatch",
    )
    terminal_visible, _, _ = qualification._active_task_count(
        scheduler_jobs=[terminal],
        ledger=ledger,
        captured_timestamp=1_001.0,
    )
    assert terminal_visible == 0

    active_row = SimpleNamespace(
        active=True,
        job_id="12345_1",
        job_name="asys-dispatch-visible",
        comment="",
        command="/sealed/batch.sbatch",
    )
    mixed_visible, _, mixed_foreign = (
        qualification._active_task_count(
            scheduler_jobs=[terminal, active_row],
            ledger=ledger,
            captured_timestamp=1_001.0,
        )
    )
    assert mixed_visible == 1
    assert mixed_foreign == []

    stray = SimpleNamespace(
        active=True,
        job_id="99999_0",
        job_name="asys-dispatch-foreign",
        comment="",
        command="/tmp/foreign.sbatch",
    )
    _, _, foreign = qualification._active_task_count(
        scheduler_jobs=[stray],
        ledger={"jobs": {}},
        captured_timestamp=1_001.0,
    )
    assert foreign == ["unmapped-active-cell-job:99999_0"]


def test_no_admit_reconciliation_adopts_before_foreign_job_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycle0, _ = _ensure_passing_cycle_roots(context, intent)
    adopted_ledger = qualification.dispatch_sweeps._empty_ledger()
    adopted_ledger["jobs"] = {
        "77777": {
            "state": "active",
            "submitted_at": 1_000.0,
            "task_count": 1,
            "tasks": [
                {"run_id": qualification.QUALIFICATION_RUN_ID}
            ],
        }
    }
    calls: list[int] = []

    def reconcile_once(
        _context: qualification.QualificationContext,
        *,
        max_batch: int,
        cycle: qualification.LoadCycleContext,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append(max_batch)
        assert cycle.cycle_index in {0, 1}
        cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
        (cycle.dispatcher_state / "ledger.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "poll_number": 2,
            "selected": [],
            "submission": None,
            "submission_error": None,
            "unmappable_cell_jobs": [],
            "validation_errors": [],
        }

    monkeypatch.setattr(
        qualification, "_run_dispatcher_once", reconcile_once
    )
    monkeypatch.setattr(
        qualification.dispatch_sweeps,
        "_load_ledger",
        lambda _path: adopted_ledger,
    )
    report = qualification._reconcile_dispatchers_no_admit(
        context,
        intent=intent,
        control_value={},
        runner=lambda *_args, **_kwargs: pytest.fail(
            "no direct sbatch is permitted during reconciliation"
        ),
    )
    assert calls == [0, 0]
    assert len(report["aggregate_ledger_sha256"]) == 64

    row = SimpleNamespace(
        active=True,
        job_id="77777_0",
        job_name="asys-dispatch-adopted",
        comment="",
        command="/sealed/adopted.sbatch",
    )
    active, job_ids, foreign = qualification._active_task_count(
        scheduler_jobs=[row],
        ledger=adopted_ledger,
        captured_timestamp=1_001.0,
        allowed_run_ids=[qualification.QUALIFICATION_RUN_ID],
    )
    assert active == 1
    assert job_ids == ["77777"]
    assert foreign == []


def test_execute_reconciles_restart_before_next_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _record(
        context,
        intent,
        sequence=0,
        timestamp=1_000.0,
        ceiling=24,
        active=0,
        useful=0,
    )
    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    events: list[str] = []

    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_qualification_run",
        lambda *_args, **_kwargs: {},
    )
    fake_cycle = SimpleNamespace(
        cycle_id="0" * 64,
        run_id=qualification.QUALIFICATION_RUN_ID,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_load_cycle",
        lambda *_args, **_kwargs: fake_cycle,
    )
    monkeypatch.setattr(
        qualification,
        "_ensure_load_backlog",
        lambda *_args, **_kwargs: [fake_cycle],
    )
    monkeypatch.setattr(
        qualification,
        "_select_dispatch_cycle",
        lambda *_args, **_kwargs: fake_cycle,
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_execution_authority",
        lambda *_args, **_kwargs: {},
    )
    def reconcile_before_scan(
        *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        events.append("reconcile:no-admit")
        return {
            "cycles": [],
            "aggregate_ledger_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        qualification,
        "_reconcile_dispatchers_no_admit",
        reconcile_before_scan,
    )

    def scheduler_scan(
        _context: qualification.QualificationContext,
        *,
        intent: dict[str, object],
        sequence: int,
        captured_timestamp: float,
        ceiling: int,
        unfinished_load_assignments: int,
        scheduler_reader: object,
    ) -> dict[str, object]:
        del _context, scheduler_reader
        events.append(f"capture:{ceiling}")
        execution_authority = (
            qualification._execution_authority_evidence_binding(intent)
        )
        return qualification.make_scheduler_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=sequence,
            captured_timestamp=captured_timestamp,
            ceiling=ceiling,
            jobs=[_job(sequence)],
            qualification_job_ids=[_job(sequence)["job_id"]],
            active_qualification_cells=24,
            unfinished_load_assignments=unfinished_load_assignments,
            active_cycle_ids=["0" * 64],
            dispatcher_ledger_sha256="9" * 64,
            production_control_guard_sha256=str(
                intent["control_guard"]["guard_sha256"]  # type: ignore[index]
            ),
            client_partition=str(intent["client_partition"]),
            client_qos=str(intent["client_qos"]),
            protected_capacity_marker_id=str(
                intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
            ),
            protected_capacity_marker_sha256=str(
                intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
            ),
            readiness_rollout_generation=1,
            trusted_generation_catalog_id=str(
                intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
            ),
            qualification_execution_authority_id=(
                execution_authority["authority_id"]
            ),
            qualification_execution_authority_sha256=(
                execution_authority["sha256"]
            ),
        )

    monkeypatch.setattr(qualification, "_scheduler_scan", scheduler_scan)

    def dispatch_once(
        _context: qualification.QualificationContext,
        *,
        intent: dict[str, object],
        control_value: dict[str, object],
        max_batch: int,
        runner: object,
        cycle: object,
    ) -> dict[str, object]:
        del _context, intent, control_value, runner, cycle
        events.append(f"dispatch:{max_batch}")
        if events.count(f"dispatch:{max_batch}") == 2:
            raise RuntimeError("second immediate dispatch")
        return {"selected": [{"run_id": qualification.QUALIFICATION_RUN_ID}]}

    monkeypatch.setattr(
        qualification, "_run_dispatcher_once", dispatch_once
    )

    def semantic_reader(**kwargs: object) -> dict[str, object]:
        return qualification.make_semantic_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=int(kwargs["sequence"]),
            captured_timestamp=float(kwargs["captured_timestamp"]),
            manifest_sha256=MANIFEST_SHA256,
            states={"active": 24, "missing": qualification.CELL_COUNT - 24},
            validated_qids=0,
            useful_qids=0,
            strata_progress=_progress(0),
            artifact_schema_counts={},
        )

    def unexpected_sleep(_seconds: float) -> None:
        raise AssertionError("ramp slept before exact stage fill")

    clock_values = iter(1_000.0 + 0.25 * index for index in range(100))
    with pytest.raises(RuntimeError, match="second immediate dispatch"):
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            semantic_reader=semantic_reader,
            clock=lambda: next(clock_values),
            sleeper=unexpected_sleep,
            capacity_certificate_loader=lambda *_args, **_kwargs: intent[
                "admission_capacity_certificate"
            ],
        )

    # The stale baseline is never used for admission: restart first sees the 24 live
    # tasks, closes ceiling 24, then advances under the ceiling-96 stage.
    assert events == [
        "reconcile:no-admit",
        "capture:24",
        "dispatch:24",
        "reconcile:no-admit",
        "capture:96",
        "dispatch:24",
    ]


def test_execute_fails_static_capacity_before_attempt_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: pytest.fail(
            "attempt namespace must not exist before capacity certification"
        ),
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="static capacity is not certified",
    ):
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            capacity_certificate_loader=(
                lambda *_args, **_kwargs: (
                    _ for _ in ()
                ).throw(
                    qualification.ThroughputQualificationError(
                        "static capacity is not certified"
                    )
                )
            ),
        )


def _start_exact_384_window(
    context: qualification.QualificationContext,
    intent: dict[str, object],
) -> qualification.LoadCycleContext:
    cycle0, _ = _ensure_passing_cycle_roots(context, intent)
    rows = (
        (1_000.0, 24, 0, 0),
        (1_010.0, 24, 24, 24),
        (1_020.0, 96, 96, 120),
        (1_030.0, 192, 192, 312),
        (
            1_040.0,
            384,
            int(intent["certified_saturation_target"]),
            696,
        ),
    )
    for sequence, (timestamp, ceiling, active, events) in enumerate(rows):
        _record(
            context,
            intent,
            sequence=sequence,
            timestamp=timestamp,
            ceiling=ceiling,
            active=active,
            useful=events,
            execution_events=events,
            unfinished=768,
        )
    assert (
        context.qualification_root
        / qualification.LOAD_WINDOW_INTENT_NAME
    ).is_file()
    return cycle0


def test_live_window_refills_completions_before_committing_exact_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycle = _start_exact_384_window(context, intent)
    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    active_values = iter((248, 248, 272, 276, 278))
    dispatch_sizes: list[int] = []
    accepted_job_ids: list[str] = []
    fake_ledger = qualification.dispatch_sweeps._empty_ledger()
    cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
    qualification.dispatch_sweeps._atomic_write_json(
        cycle.dispatcher_state / "ledger.json",
        fake_ledger,
    )
    monkeypatch.setattr(
        qualification.dispatch_sweeps,
        "_load_ledger",
        lambda _path: fake_ledger,
    )

    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_qualification_run",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        qualification,
        "initialize_load_cycle",
        lambda *_args, **_kwargs: cycle,
    )
    monkeypatch.setattr(
        qualification,
        "_ensure_load_backlog",
        lambda *_args, **_kwargs: [cycle],
    )
    monkeypatch.setattr(
        qualification,
        "_select_dispatch_cycle",
        lambda *_args, **_kwargs: cycle,
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_execution_authority",
        lambda *_args, **_kwargs: {},
    )
    reconciliation_calls = 0

    def reconcile(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        nonlocal reconciliation_calls
        reconciliation_calls += 1
        unresolved = (
            [
                {
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "ledger_path": str(
                        (
                            cycle.dispatcher_state / "ledger.json"
                        ).resolve()
                    ),
                    "batch_id": "f" * 64,
                    "intent_state": "prepared",
                    "task_count": 1,
                    "tasks_sha256": "e" * 64,
                }
            ]
            if reconciliation_calls == 1
            else []
        )
        return {
            "cycles": [
                {
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "ledger_path": str(
                        (
                            cycle.dispatcher_state / "ledger.json"
                        ).resolve()
                    ),
                }
            ],
            "unresolved_intent_reservations": unresolved,
            "aggregate_ledger_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        qualification,
        "_reconcile_dispatchers_no_admit",
        reconcile,
    )

    def scheduler_scan(
        _context: qualification.QualificationContext,
        *,
        intent: dict[str, object],
        sequence: int,
        captured_timestamp: float,
        ceiling: int,
        unfinished_load_assignments: int,
        scheduler_reader: object,
    ) -> dict[str, object]:
        del _context, scheduler_reader
        active = next(active_values)
        authority = qualification._execution_authority_evidence_binding(
            intent
        )
        return qualification.make_scheduler_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=sequence,
            captured_timestamp=captured_timestamp,
            ceiling=ceiling,
            jobs=[_job(sequence)] if active else [],
            qualification_job_ids=list(accepted_job_ids) or ["19000"],
            active_qualification_cells=active,
            unfinished_load_assignments=unfinished_load_assignments,
            active_cycle_ids=[cycle.cycle_id] if active else [],
            dispatcher_ledger_sha256="9" * 64,
            production_control_guard_sha256=str(
                intent["control_guard"]["guard_sha256"]  # type: ignore[index]
            ),
            client_partition=str(intent["client_partition"]),
            client_qos=str(intent["client_qos"]),
            protected_capacity_marker_id=str(
                intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
            ),
            protected_capacity_marker_sha256=str(
                intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
            ),
            readiness_rollout_generation=1,
            trusted_generation_catalog_id=str(
                intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
            ),
            qualification_execution_authority_id=authority["authority_id"],
            qualification_execution_authority_sha256=authority["sha256"],
            cycle_execution_authorities=[
                {
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "authority_id": authority["authority_id"],
                    "authority_sha256": authority["sha256"],
                }
            ],
        )

    monkeypatch.setattr(qualification, "_scheduler_scan", scheduler_scan)

    def semantic_reader(**kwargs: object) -> dict[str, object]:
        return qualification.make_semantic_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=int(kwargs["sequence"]),
            captured_timestamp=float(kwargs["captured_timestamp"]),
            manifest_sha256=MANIFEST_SHA256,
            states={"active": 384, "missing": 384},
            validated_qids=696,
            useful_qids=696,
            strata_progress=_progress(696),
            artifact_schema_counts={"5": 696},
            semantic_reference_cycle=cycle.cycle_id,
            trusted_qid_execution_events=700,
            load_strata_progress=_progress(700),
            load_cycle_inventory=[
                {
                    "cycle_index": 0,
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "semantic_reference": True,
                    "status": "active",
                    "validated_execution_events": 700,
                    "unfinished_assignments": 768,
                    "estimand_excluded": True,
                    "primary_analysis_eligible": False,
                }
            ],
            unfinished_load_assignments=768,
        )

    def dispatch_once(
        _context: qualification.QualificationContext,
        *,
        max_batch: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        dispatch_sizes.append(max_batch)
        job_id = str(20_000 + len(dispatch_sizes))
        accepted_job_ids.append(job_id)
        batch_id = f"{len(dispatch_sizes):064x}"
        tasks = [
            {"run_id": cycle.run_id}
            for _ in range(max_batch)
        ]
        job_record, intent_record = _synthetic_accepted_transaction(
            cycle,
            job_id=job_id,
            batch_id=batch_id,
            tasks=tasks,
        )
        fake_ledger["jobs"][job_id] = job_record
        fake_ledger["intents"][batch_id] = intent_record
        cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
        qualification.dispatch_sweeps._atomic_write_json(
            cycle.dispatcher_state / "ledger.json",
            fake_ledger,
        )
        return {
            "selected": [
                {"run_id": cycle.run_id}
                for _ in range(max_batch)
            ],
            "submission": {
                "job_id": job_id,
                "batch_id": batch_id,
                "tasks": max_batch,
            },
        }

    monkeypatch.setattr(
        qualification, "_run_dispatcher_once", dispatch_once
    )

    class TestClock:
        value = 1_340.0

        def __call__(self) -> float:
            self.value += 0.25
            return self.value

    sleep_calls: list[float] = []

    def sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) > 1:
            raise RuntimeError("stop after exact refill")

    with pytest.raises(RuntimeError, match="stop after exact refill"):
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            semantic_reader=semantic_reader,
            clock=TestClock(),
            sleeper=sleep,
            poll_seconds=17.0,
            capacity_certificate_loader=lambda *_args, **_kwargs: intent[
                "admission_capacity_certificate"
            ],
        )

    assert sleep_calls == [17.0, 17.0]
    assert dispatch_sizes == [24, 6, 2]
    observations = qualification.load_observations(
        context.qualification_root, intent=intent
    )
    assert len(observations) == 6
    assert observations[-1]["scheduler"][
        "active_qualification_cells"
    ] == 278
    refill = qualification.load_refill_reconciliations(
        context.qualification_root, intent=intent
    )
    assert [row["active_deficit"] for row in refill] == [
        30,
        30,
        6,
        2,
        0,
    ]
    assert [row["measurement_eligible"] for row in refill] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_refill_journal_adopts_crash_after_deficit_scan_and_binds_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _start_exact_384_window(context, intent)
    loaded = qualification.load_observations(
        context.qualification_root, intent=intent
    )[-1]
    scheduler_identity = dict(loaded["scheduler"])
    scheduler_identity.pop("scheduler_id")
    scheduler_identity["active_qualification_cells"] = 254
    deficit_scheduler = qualification._with_identity(
        scheduler_identity, "scheduler_id"
    )
    original = qualification._write_once
    crashed = False

    def crash_after_scan(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal crashed
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )
        if description == "refill reconciliation 0" and not crashed:
            crashed = True
            raise RuntimeError("crash after durable deficit scan")

    monkeypatch.setattr(qualification, "_write_once", crash_after_scan)
    with pytest.raises(
        RuntimeError, match="crash after durable deficit scan"
    ):
        qualification.record_refill_reconciliation(
            context.qualification_root,
            intent=intent,
            scheduler=deficit_scheduler,
            semantic=loaded["semantic"],
            preceding_dispatch=None,
        )
    monkeypatch.setattr(qualification, "_write_once", original)
    adopted = qualification.load_refill_reconciliations(
        context.qualification_root, intent=intent
    )
    assert len(adopted) == 1
    assert adopted[0]["active_deficit"] == 24

    exact_scheduler_identity = dict(loaded["scheduler"])
    exact_scheduler_identity.pop("scheduler_id")
    exact_scheduler_identity["captured_timestamp"] = 1_041.0
    exact_scheduler = qualification._with_identity(
        exact_scheduler_identity, "scheduler_id"
    )
    semantic_identity = dict(loaded["semantic"])
    semantic_identity.pop("semantic_id")
    semantic_identity["captured_timestamp"] = 1_041.0
    exact_semantic = qualification._with_identity(
        semantic_identity, "semantic_id"
    )
    job_id = str(exact_scheduler["qualification_job_ids"][0])
    cycle = qualification.load_cycle_context(
        context, intent=intent, cycle_index=0
    )
    tasks = [{"run_id": cycle.run_id} for _ in range(24)]
    batch_id = "f" * 64
    job_record, intent_record = _synthetic_accepted_transaction(
        cycle,
        job_id=job_id,
        batch_id=batch_id,
        tasks=tasks,
    )
    ledger = qualification.dispatch_sweeps._empty_ledger(1_041.0)
    ledger["jobs"][job_id] = job_record
    ledger["intents"][batch_id] = intent_record
    cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
    qualification.dispatch_sweeps._atomic_write_json(
        cycle.dispatcher_state / "ledger.json", ledger
    )
    binding = qualification._refill_dispatch_from_ledger(
        cycle=cycle,
        ledger=ledger,
        job_id=job_id,
        requested_tasks=24,
    )
    postfill = qualification.record_refill_reconciliation(
        context.qualification_root,
        intent=intent,
        scheduler=exact_scheduler,
        semantic=exact_semantic,
        preceding_dispatch=binding,
    )
    assert postfill["measurement_eligible"] is True
    assert postfill["preceding_dispatch"]["job_id"] == job_id


def test_real_dispatch_reconciliation_adopts_lost_reply_once_into_refill(
    tmp_path: Path,
) -> None:
    """Exercise the real submitting-intent/spool/fairness crash boundary."""

    context, intent = _context_and_intent(tmp_path)
    cycle = _start_exact_384_window(context, intent)
    loaded = qualification.load_observations(
        context.qualification_root, intent=intent
    )[-1]
    scheduler_identity = dict(loaded["scheduler"])
    scheduler_identity.pop("scheduler_id")
    scheduler_identity["active_qualification_cells"] = 277
    deficit_scheduler = qualification._with_identity(
        scheduler_identity, "scheduler_id"
    )
    qualification.record_refill_reconciliation(
        context.qualification_root,
        intent=intent,
        scheduler=deficit_scheduler,
        semantic=loaded["semantic"],
        preceding_dispatch=None,
    )

    batch_id = "a" * 64
    job_id = "321"
    task = _canonical_cycle_task(context, cycle)
    sbatch_path = (
        cycle.dispatcher_state / "batches" / f"{batch_id}.sbatch"
    ).resolve()
    manifest_path = sbatch_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        qualification._canonical_bytes(
            {"batch_id": batch_id, "tasks": [task]}
        )
    )
    sbatch_path.write_bytes(b"#!/bin/bash\n# exact crash-boundary batch\n")
    manifest_sha256 = (
        qualification.dispatch_sweeps._seal_dispatch_artifact(
            manifest_path
        )
    )
    sbatch_sha256 = (
        qualification.dispatch_sweeps._seal_dispatch_artifact(
            sbatch_path
        )
    )
    ledger = qualification.dispatch_sweeps._empty_ledger(1_040.0)
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 1_040.0,
        "submit_started_at": 1_040.5,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "submission_transport": (
            qualification.dispatch_sweeps.STDIN_EXACT_SUBMISSION_TRANSPORT
        ),
        "submission_argv_sha256": (
            qualification.dispatch_sweeps._stdin_submission_argv_sha256(
                batch_id
            )
        ),
        "tasks": [task],
        "fairness_after": {
            "cursor": 1,
            "deficits": {cycle.run_id: 0.5},
        },
        "fairness_committed": False,
    }
    ledger_path = cycle.dispatcher_state / "ledger.json"
    qualification.dispatch_sweeps._atomic_write_json(
        ledger_path, ledger
    )
    command = " ".join(
        qualification.dispatch_sweeps._stdin_submission_argv(batch_id)
    )
    scheduler_job = qualification.control.SchedulerJob(
        f"{job_id}_0",
        f"asys-dispatch-{batch_id[-10:]}",
        "RUNNING",
        f"asys-schema5-intent:{batch_id}",
        command,
        "squeue",
        "",
        "ou_bcs_normal",
        "normal",
    )
    snapshot = qualification.control.SchedulerSnapshot(
        (scheduler_job,),
        1_041.0,
        squeue_ok=True,
        sacct_ok=True,
        accounting_start_timestamp=1_000.0,
    )
    spool_reads: list[str] = []
    warnings, errors = (
        qualification.dispatch_sweeps._reconcile_schema5_intents(
            ledger,
            scheduler_snapshot=snapshot,
            now=1_041.0,
            spooled_script_reader=lambda observed_job: (
                spool_reads.append(observed_job)
                or sbatch_path.read_bytes()
            ),
        )
    )
    assert warnings and errors == []
    assert spool_reads == [job_id]
    assert ledger["intents"][batch_id]["fairness_committed"] is True
    fairness = json.loads(json.dumps(ledger["fairness"]))
    qualification.dispatch_sweeps._atomic_write_json(
        ledger_path, ledger
    )
    reloaded = qualification.dispatch_sweeps._load_ledger(ledger_path)
    assert reloaded["fairness"] == fairness
    assert Path(
        reloaded["intents"][batch_id]["spooled_receipt_path"]
    ).is_file()

    exact_scheduler_identity = dict(loaded["scheduler"])
    exact_scheduler_identity.pop("scheduler_id")
    exact_scheduler_identity["captured_timestamp"] = 1_041.0
    exact_scheduler_identity["qualification_job_ids"] = [
        *loaded["scheduler"]["qualification_job_ids"],
        job_id,
    ]
    exact_scheduler = qualification._with_identity(
        exact_scheduler_identity, "scheduler_id"
    )
    semantic_identity = dict(loaded["semantic"])
    semantic_identity.pop("semantic_id")
    semantic_identity["captured_timestamp"] = 1_041.0
    exact_semantic = qualification._with_identity(
        semantic_identity, "semantic_id"
    )
    reconciliation = {
        "cycles": [
            {
                "cycle_id": cycle.cycle_id,
                "run_id": cycle.run_id,
                "ledger_path": str(ledger_path.resolve()),
            }
        ],
        "unresolved_intent_reservations": [],
    }
    recovered, unresolved = (
        qualification._recover_unconsumed_refill_dispatch(
            context,
            intent=intent,
            reconciliation=reconciliation,
            scheduler=exact_scheduler,
            proposed=None,
        )
    )
    assert unresolved == []
    assert recovered is not None
    assert recovered["batch_id"] == batch_id
    assert recovered["job_id"] == job_id
    assert recovered["tasks_sha256"] == qualification._sha256_bytes(
        qualification._canonical_bytes([task])
    )
    assert recovered["sbatch_sha256"] == sbatch_sha256
    refill = qualification.record_refill_reconciliation(
        context.qualification_root,
        intent=intent,
        scheduler=exact_scheduler,
        semantic=exact_semantic,
        preceding_dispatch=recovered,
    )
    assert refill["preceding_dispatch"] == recovered

    # A successor replays the real reconciliation boundary.  It reuses the sealed
    # spool receipt, commits no fairness twice, and cannot consume the admission
    # into a second refill record.
    reloaded = qualification.dispatch_sweeps._load_ledger(ledger_path)
    replay_warnings, replay_errors = (
        qualification.dispatch_sweeps._reconcile_schema5_intents(
            reloaded,
            scheduler_snapshot=snapshot,
            now=1_042.0,
            spooled_script_reader=lambda _job_id: pytest.fail(
                "sealed spool receipt must make adoption replay read-only"
            ),
        )
    )
    assert replay_errors == []
    assert replay_warnings
    assert reloaded["fairness"] == fairness
    qualification.dispatch_sweeps._atomic_write_json(
        ledger_path, reloaded
    )
    recovered_again, _ = (
        qualification._recover_unconsumed_refill_dispatch(
            context,
            intent=intent,
            reconciliation=reconciliation,
            scheduler=exact_scheduler,
            proposed=None,
        )
    )
    assert recovered_again is None
    assert len(
        [
            record
            for record in qualification.load_refill_reconciliations(
                context.qualification_root, intent=intent
            )
            if record["preceding_dispatch"] is not None
        ]
    ) == 1


def test_real_dispatch_reconciliation_reserves_grace_and_fails_ambiguity(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycle = _start_exact_384_window(context, intent)
    task = _canonical_cycle_task(context, cycle)
    batch_id = "b" * 64
    sbatch_path = (
        cycle.dispatcher_state / "batches" / f"{batch_id}.sbatch"
    ).resolve()
    manifest_path = sbatch_path.with_suffix(".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        qualification._canonical_bytes(
            {"batch_id": batch_id, "tasks": [task]}
        )
    )
    sbatch_path.write_bytes(b"#!/bin/bash\n# ambiguous boundary\n")
    manifest_sha256 = (
        qualification.dispatch_sweeps._seal_dispatch_artifact(
            manifest_path
        )
    )
    sbatch_sha256 = (
        qualification.dispatch_sweeps._seal_dispatch_artifact(
            sbatch_path
        )
    )
    ledger = qualification.dispatch_sweeps._empty_ledger(1_040.0)
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 1_040.0,
        "submit_started_at": 1_040.5,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "submission_transport": (
            qualification.dispatch_sweeps.STDIN_EXACT_SUBMISSION_TRANSPORT
        ),
        "submission_argv_sha256": (
            qualification.dispatch_sweeps._stdin_submission_argv_sha256(
                batch_id
            )
        ),
        "tasks": [task],
        "fairness_after": {
            "cursor": 1,
            "deficits": {cycle.run_id: 0.5},
        },
        "fairness_committed": False,
    }
    absent = qualification.control.SchedulerSnapshot(
        (),
        1_041.0,
        squeue_ok=True,
        sacct_ok=True,
        accounting_start_timestamp=1_000.0,
    )
    warnings, errors = (
        qualification.dispatch_sweeps._reconcile_schema5_intents(
            ledger,
            scheduler_snapshot=absent,
            now=1_041.0,
            spooled_script_reader=lambda _job_id: pytest.fail(
                "an invisible intent has no spool proof yet"
            ),
        )
    )
    assert warnings == [] and errors == []
    assert ledger["intents"][batch_id]["state"] == "submitting"
    assert ledger["intents"][batch_id]["fairness_committed"] is False
    active, job_ids, foreign = qualification._active_task_count(
        scheduler_jobs=(),
        ledger=ledger,
        captured_timestamp=1_041.0,
        allowed_run_ids=[cycle.run_id],
    )
    assert (active, job_ids, foreign) == (1, [], [])

    command = " ".join(
        qualification.dispatch_sweeps._stdin_submission_argv(batch_id)
    )
    ambiguous_jobs = tuple(
        qualification.control.SchedulerJob(
            f"{job_id}_0",
            f"asys-dispatch-{batch_id[-10:]}",
            "RUNNING",
            f"asys-schema5-intent:{batch_id}",
            command,
            "squeue",
        )
        for job_id in ("321", "322")
    )
    _, ambiguity = qualification.dispatch_sweeps._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=qualification.control.SchedulerSnapshot(
            ambiguous_jobs,
            1_042.0,
            squeue_ok=True,
            sacct_ok=True,
            accounting_start_timestamp=1_000.0,
        ),
        now=1_042.0,
        spooled_script_reader=lambda _job_id: sbatch_path.read_bytes(),
    )
    assert ambiguity and "ambiguously maps" in ambiguity[0]
    assert ledger["jobs"] == {}
    assert ledger["intents"][batch_id]["fairness_committed"] is False


def test_live_window_fences_when_refill_exceeds_real_time_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycle = _start_exact_384_window(context, intent)
    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_qualification_run",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        qualification,
        "initialize_load_cycle",
        lambda *_args, **_kwargs: cycle,
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_execution_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        qualification,
        "_reconcile_dispatchers_no_admit",
        lambda *_args, **_kwargs: {
            "cycles": [],
            "unresolved_intent_reservations": [],
            "aggregate_ledger_sha256": "9" * 64,
        },
    )

    def semantic_reader(**kwargs: object) -> dict[str, object]:
        return qualification.make_semantic_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=int(kwargs["sequence"]),
            captured_timestamp=float(kwargs["captured_timestamp"]),
            manifest_sha256=MANIFEST_SHA256,
            states={"active": 354, "missing": 414},
            validated_qids=696,
            useful_qids=696,
            strata_progress=_progress(696),
            artifact_schema_counts={"5": 696},
            semantic_reference_cycle=cycle.cycle_id,
            trusted_qid_execution_events=700,
            load_strata_progress=_progress(700),
            load_cycle_inventory=[
                {
                    "cycle_index": 0,
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "semantic_reference": True,
                    "status": "active",
                    "validated_execution_events": 700,
                    "unfinished_assignments": 768,
                    "estimand_excluded": True,
                    "primary_analysis_eligible": False,
                }
            ],
            unfinished_load_assignments=768,
        )

    authority = qualification._execution_authority_evidence_binding(
        intent
    )
    monkeypatch.setattr(
        qualification,
        "_scheduler_scan",
        lambda _context, **kwargs: qualification.make_scheduler_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=int(kwargs["sequence"]),
            captured_timestamp=float(kwargs["captured_timestamp"]),
            ceiling=int(kwargs["ceiling"]),
            jobs=[_job(int(kwargs["sequence"]))],
            qualification_job_ids=["19000"],
            active_qualification_cells=354,
            unfinished_load_assignments=int(
                kwargs["unfinished_load_assignments"]
            ),
            active_cycle_ids=[cycle.cycle_id],
            dispatcher_ledger_sha256="9" * 64,
            production_control_guard_sha256=str(
                intent["control_guard"]["guard_sha256"]  # type: ignore[index]
            ),
            client_partition=str(intent["client_partition"]),
            client_qos=str(intent["client_qos"]),
            protected_capacity_marker_id=str(
                intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
            ),
            protected_capacity_marker_sha256=str(
                intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
            ),
            readiness_rollout_generation=1,
            trusted_generation_catalog_id=str(
                intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
            ),
            qualification_execution_authority_id=authority["authority_id"],
            qualification_execution_authority_sha256=authority["sha256"],
            cycle_execution_authorities=[
                {
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "authority_id": authority["authority_id"],
                    "authority_sha256": authority["sha256"],
                }
            ],
        ),
    )
    times = iter((1_050.0, 1_050.0, 1_701.0, 1_702.0, 1_703.0))

    with pytest.raises(RuntimeError, match="stop after fence"):
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            semantic_reader=semantic_reader,
            clock=lambda: next(times),
            sleeper=lambda _seconds: (_ for _ in ()).throw(
                RuntimeError("stop after fence")
            ),
            capacity_certificate_loader=lambda *_args, **_kwargs: intent[
                "admission_capacity_certificate"
            ],
        )
    fence = qualification._read_json(
        context.qualification_root
        / qualification.FAILURE_DRAIN_INTENT_NAME,
        description="test cadence failure fence",
        sealed=True,
    )
    assert "660-second evidence cadence" in fence["reason"]
    observations = qualification.load_observations(
        context.qualification_root, intent=intent
    )
    assert len(observations) == 5


def _synthetic_cycle_run_evidence() -> dict[str, object]:
    return {
        "run_id": qualification.QUALIFICATION_RUN_ID,
        "run_root": "/synthetic/reference",
        "manifest_sha256": "a" * 64,
        "benchmark_contracts_sha256": "b" * 64,
        "artifact_policy_sha256": "c" * 64,
        "lineage_id": "d" * 64,
        "cell_count": qualification.CELL_COUNT,
        "qids": qualification.TOTAL_QIDS,
        "estimand_excluded": True,
    }


def test_cycle_intent_and_initialization_resume_exact_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    monkeypatch.setattr(
        qualification,
        "verify_qualification_run",
        lambda *_args, **_kwargs: _synthetic_cycle_run_evidence(),
    )
    original = qualification._write_once
    interrupted = False

    def crash_before_initialization_receipt(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal interrupted
        if (
            description == "load cycle 0 initialization receipt"
            and not interrupted
        ):
            interrupted = True
            raise RuntimeError("cycle initialization crash")
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(
        qualification, "_write_once", crash_before_initialization_receipt
    )
    with pytest.raises(RuntimeError, match="cycle initialization crash"):
        qualification.initialize_load_cycle(
            context,
            intent=intent,
            control_value={},
            cycle_index=0,
            now=1_100.0,
        )
    cycle_intent = (
        context.qualification_root
        / qualification.LOAD_CYCLE_DIRECTORY
        / "cycle-000000"
        / qualification.CYCLE_INTENT_NAME
    )
    assert cycle_intent.is_file()
    assert not (
        cycle_intent.parent / qualification.CYCLE_INITIALIZED_NAME
    ).exists()

    monkeypatch.setattr(qualification, "_write_once", original)
    cycle = qualification.initialize_load_cycle(
        context,
        intent=intent,
        control_value={},
        cycle_index=0,
        now=9_999.0,
    )
    assert cycle.cycle_index == 0
    assert (
        cycle.evidence_root / qualification.CYCLE_INITIALIZED_NAME
    ).is_file()
    assert cycle.intent["created_timestamp"] == 1_100.0


def test_window_intent_recovers_crash_after_observation_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    for sequence, (timestamp, ceiling, active, useful) in enumerate(
        (
            (1_000.0, 24, 0, 0),
            (1_010.0, 24, 24, 24),
            (1_020.0, 96, 96, 120),
            (1_030.0, 192, 192, 312),
        )
    ):
        _record(
            context,
            intent,
            sequence=sequence,
            timestamp=timestamp,
            ceiling=ceiling,
            active=active,
            useful=useful,
        )
    original = qualification._write_once

    def crash_on_window_intent(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        if description == "non-resettable load-window intent":
            raise RuntimeError("window intent crash")
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(qualification, "_write_once", crash_on_window_intent)
    with pytest.raises(RuntimeError, match="window intent crash"):
        _record(
            context,
            intent,
            sequence=4,
            timestamp=1_040.0,
            ceiling=384,
            active=int(intent["certified_saturation_target"]),
            useful=696,
            unfinished=768,
        )
    assert (
        context.qualification_root
        / qualification.OBSERVATION_DIRECTORY
        / qualification._evidence_filename("OBSERVATION", 4)
    ).is_file()
    assert not (
        context.qualification_root / qualification.LOAD_WINDOW_INTENT_NAME
    ).exists()

    monkeypatch.setattr(qualification, "_write_once", original)
    recovered = qualification.load_observations(
        context.qualification_root,
        intent=intent,
        recover_transactions=True,
    )
    window = qualification._read_json(
        context.qualification_root / qualification.LOAD_WINDOW_INTENT_NAME,
        description="recovered load-window intent",
        sealed=True,
    )
    assert len(recovered) == 5
    assert window["start_sequence"] == 4
    assert window["start_trusted_qid_execution_events"] == 696


def test_window_intent_cannot_reset_or_cherry_pick_later_observation(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    path = context.qualification_root / qualification.LOAD_WINDOW_INTENT_NAME
    original = qualification._read_json(
        path, description="original load-window intent", sealed=True
    )
    replacement = qualification._load_window_intent_payload(
        context.qualification_root,
        intent=intent,
        observation=observations[5],
    )
    assert replacement["start_sequence"] > original["start_sequence"]
    path.chmod(0o644)
    path.write_bytes(qualification._canonical_bytes(replacement))
    path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="reset or cherry-pick is forbidden",
    ):
        qualification.load_observations(
            context.qualification_root, intent=intent
        )


def test_exact_7200_second_event_threshold_rejects_16832_accepts_16833(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    accepted = qualification.evaluate_observations(observations)
    assert accepted["load_execution"]["trusted_execution_events"] == 16_833
    assert accepted["load_execution"]["throughput_events_per_day"] == 201_996
    assert 16_832 * 86_400 // 7_200 == 201_984

    rejected = [dict(observation) for observation in observations]
    for index in (-2, -1):
        rejected[index] = dict(rejected[index])
        semantic = dict(rejected[index]["semantic"])
        semantic["trusted_qid_execution_events"] -= 1
        semantic["replay_qid_execution_events"] -= 1
        semantic["load_strata_progress"] = _progress(
            semantic["trusted_qid_execution_events"]
        )
        inventory = [
            dict(record) for record in semantic["load_cycle_inventory"]
        ]
        inventory[1]["validated_execution_events"] -= 1
        semantic["load_cycle_inventory"] = inventory
        rejected[index]["semantic"] = semantic
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match=r"16,832.*201,984/day.*below 201,994",
    ):
        qualification.evaluate_observations(rejected)


def test_mature_low_rate_followed_by_censor_is_generic_scientific_failure(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    low_events = (
        696 + qualification.MIN_LOAD_WINDOW_EXECUTION_EVENTS - 1
    )
    for observation in observations[-2:]:
        semantic = observation["semantic"]
        semantic["trusted_qid_execution_events"] = low_events
        semantic["replay_qid_execution_events"] = (
            low_events - qualification.TOTAL_QIDS
        )
        semantic["load_strata_progress"] = _progress(low_events)
        semantic["load_cycle_inventory"][1][
            "validated_execution_events"
        ] = low_events - qualification.TOTAL_QIDS
    observations[-1]["semantic"]["load_censor_incidents"] = 1

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="integrity, censor, or namespace incident",
    ) as caught:
        qualification.evaluate_observations(observations)
    assert type(caught.value) is qualification.ThroughputQualificationError


def test_end_intent_is_durable_before_partial_cycle_drain(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    end = qualification.publish_load_window_end_intent(
        context.qualification_root,
        intent=intent,
        observations=observations,
    )
    assert end["admission_closed"] is True
    assert end["state"] == "draining"
    assert not (
        context.qualification_root / qualification.LOAD_WINDOW_DRAIN_NAME
    ).exists()

    evaluation = qualification.evaluate_observations(observations)
    drain = qualification.publish_load_window_drain(
        context.qualification_root,
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )
    assert drain["active_assignments"] == 0
    assert drain["analysis_ingestion_allowed"] is False
    assert drain["cycle_inventory"][1]["status"] == "load_window_drained"
    assert drain["cycle_inventory"][1]["unfinished_assignments"] == 600


def test_coordinate_redraw_conflicts_within_cycle_but_replay_cycle_is_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    monkeypatch.setattr(
        qualification,
        "verify_qualification_run",
        lambda *_args, **_kwargs: _synthetic_cycle_run_evidence(),
    )
    cycle0 = qualification.initialize_load_cycle(
        context,
        intent=intent,
        control_value={},
        cycle_index=0,
        now=1_100.0,
    )
    cycle1 = qualification.create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=1,
        now=1_200.0,
    )
    cell = qualification.generate_qualification_cells()[0]
    first = qualification.record_load_execution_event(
        context,
        intent=intent,
        cycle=cycle0,
        cell=cell,
        qid="q-test",
        result_record_sha256="a" * 64,
    )
    assert (
        qualification.record_load_execution_event(
            context,
            intent=intent,
            cycle=cycle0,
            cell=cell,
            qid="q-test",
            result_record_sha256="a" * 64,
        )
        == first
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="conflicts with the immutable transaction",
    ):
        qualification.record_load_execution_event(
            context,
            intent=intent,
            cycle=cycle0,
            cell=cell,
            qid="q-test",
            result_record_sha256="b" * 64,
        )
    second = qualification.record_load_execution_event(
        context,
        intent=intent,
        cycle=cycle1,
        cell=cell,
        qid="q-test",
        result_record_sha256="b" * 64,
    )
    assert first["event_id"] != second["event_id"]
    assert len(
        qualification.load_cycle_execution_events(
            context, intent=intent, cycle=cycle0
        )
    ) == 1
    assert len(
        qualification.load_cycle_execution_events(
            context, intent=intent, cycle=cycle1
        )
    ) == 1


def test_cycle_lineage_and_intents_are_never_primary_analysis_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    monkeypatch.setattr(
        qualification,
        "verify_qualification_run",
        lambda *_args, **_kwargs: _synthetic_cycle_run_evidence(),
    )
    qualification.initialize_load_cycle(
        context,
        intent=intent,
        control_value={},
        cycle_index=0,
        now=1_100.0,
    )
    cycle1 = qualification.create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=1,
        now=1_200.0,
    )
    lineage = qualification._qualification_lineage(
        plan=qualification.build_load_plan(),
        manifest_sha256="a" * 64,
        benchmark_sha256="b" * 64,
        run_id=cycle1.run_id,
        cycle_index=cycle1.cycle_index,
        cycle_id=cycle1.cycle_id,
        semantic_reference=False,
    )
    assert cycle1.run_id not in qualification.PRODUCTION_RUN_IDS
    assert cycle1.intent["estimand_excluded"] is True
    assert cycle1.intent["primary_analysis_eligible"] is False
    assert lineage["estimand_excluded"] is True
    assert lineage["primary_analysis_eligible"] is False
    assert lineage["semantic_reference"] is False


def _bind_test_pressure_ledger(
    context: qualification.QualificationContext,
    intent: dict[str, object],
    cycle: qualification.LoadCycleContext,
    ledger: dict[str, object],
) -> None:
    authority = _write_synthetic_cycle_authority(
        context, intent, cycle
    )
    ledger["qualification_execution_authority"] = {
        "path": str(cycle.execution_authority_path),
        "sha256": qualification._sha256_file(
            cycle.execution_authority_path
        ),
        "authority_id": authority["authority_id"],
        "intent_id": intent["intent_id"],
        "chain_id": context.chain_id,
        "run_id": cycle.run_id,
        "run_root": str(cycle.run_root),
    }
    ledger["protected_capacity_authority"] = {
        "path": str(context.protected_capacity_contract.path),
        "sha256": context.protected_capacity_contract.sha256,
        "marker_id": context.protected_capacity_contract.marker_id,
        "release_git_commit": context.release_git_commit,
        "partition": intent["client_partition"],
        "qos": intent["client_qos"],
        "authorized_cell_slots": qualification.CEILINGS[-1],
        "reserve_jobs": qualification.QOS_RESERVE,
    }


def _write_test_pressure_ledger(
    context: qualification.QualificationContext,
    intent: dict[str, object],
) -> None:
    cycle = qualification.load_cycle_context(
        context,
        intent=intent,
        cycle_index=0,
    )
    ledger = qualification.dispatch_sweeps._empty_ledger()
    ledger["qualification_profile_pressure"] = {
        "test": {
            "server_pool_root": str(context.server_pool_root),
            "serving_profile": "8B",
            "eligible_cells": 10,
            "backlog_fanout_work": 100,
            "live_replicas": 2,
            "backlog_work_per_replica": 50.0,
            "observed_poll": 1,
        }
    }
    _bind_test_pressure_ledger(context, intent, cycle, ledger)
    context.dispatcher_state.mkdir(parents=True, exist_ok=True)
    qualification.dispatch_sweeps._atomic_write_json(
        context.dispatcher_state / "ledger.json", ledger
    )


def test_failure_fence_crash_is_adopted_without_readmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    original = qualification._write_once
    crashed = False

    def crash_after_fence(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal crashed
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )
        if (
            description == "qualification failure-drain admission fence"
            and not crashed
        ):
            crashed = True
            raise RuntimeError("crash after failure fence")

    monkeypatch.setattr(qualification, "_write_once", crash_after_fence)
    with pytest.raises(RuntimeError, match="crash after failure fence"):
        qualification.request_failure_drain(
            context,
            intent=intent,
            reason="timeout",
            now=1_000.0,
        )
    fence_path = (
        context.qualification_root
        / qualification.FAILURE_DRAIN_INTENT_NAME
    )
    assert fence_path.is_file()

    monkeypatch.setattr(qualification, "_write_once", original)
    adopted = qualification.request_failure_drain(
        context,
        intent=intent,
        reason="timeout",
        now=9_999.0,
    )
    assert adopted["requested_timestamp"] == 1_000.0
    assert adopted["admission_closed"] is True


@pytest.mark.parametrize("now", [0.0, float("inf"), float("nan")])
def test_failure_fence_rejects_nonfinite_or_nonpositive_timestamp(
    tmp_path: Path,
    now: float,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="timestamp is invalid",
    ):
        qualification.request_failure_drain(
            context,
            intent=intent,
            reason="timeout",
            now=now,
        )


def test_failure_fence_replay_rejects_timestamp_rendering_drift(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    fence = qualification.request_failure_drain(
        context,
        intent=intent,
        reason="timeout",
        now=1_000.0,
    )
    path = (
        context.qualification_root
        / qualification.FAILURE_DRAIN_INTENT_NAME
    )
    identity = dict(fence)
    identity.pop("failure_drain_intent_id")
    identity["requested_at"] = "1970-01-01T00:00:01Z"
    drifted = qualification._with_identity(
        identity, "failure_drain_intent_id"
    )
    path.chmod(0o644)
    path.write_bytes(qualification._canonical_bytes(drifted))
    path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="replay conflicts",
    ):
        qualification.request_failure_drain(
            context,
            intent=intent,
            reason="timeout",
            now=2_000.0,
        )


def test_terminal_failure_refuses_to_seal_under_active_writer(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    cycles = _ensure_passing_cycle_roots(context, intent)
    _write_test_pressure_ledger(context, intent)
    qualification.request_failure_drain(
        context,
        intent=intent,
        reason="timeout",
        now=990.0,
    )
    _record(
        context,
        intent,
        sequence=0,
        timestamp=1_000.0,
        ceiling=24,
        active=24,
        useful=0,
        unfinished=1_536,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="scientific writers are active",
    ):
        qualification._publish_terminal_failure(
            context,
            intent=intent,
            reason="timeout",
        )
    assert not (
        context.qualification_root / qualification.FAILURE_NAME
    ).exists()
    assert any(
        stat.S_IMODE(cycle.run_root.stat().st_mode) & 0o200
        for cycle in cycles
    )


def test_terminal_failure_refuses_unmapped_active_namespace_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _ensure_passing_cycle_roots(context, intent)
    _write_test_pressure_ledger(context, intent)
    _prepare_terminal_failure_drain(context, intent, reason="timeout")
    observations = qualification.load_observations(
        context.qualification_root, intent=intent
    )
    final = {
        **observations[-1],
        "scheduler": {
            **observations[-1]["scheduler"],
            "qualification_tasks_only": False,
            "production_run_ids": [
                "unmapped-active-cell-job:18889999_0"
            ],
        },
    }
    monkeypatch.setattr(
        qualification,
        "load_observations",
        lambda *_args, **_kwargs: [
            *observations[:-1],
            final,
        ],
    )

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="scheduler namespace truth is incomplete",
    ):
        qualification._publish_terminal_failure(
            context, intent=intent, reason="timeout"
        )
    assert not (
        context.qualification_root / qualification.FAILURE_NAME
    ).exists()


def test_failure_publish_resumes_post_drain_preseal_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _ensure_passing_cycle_roots(context, intent)
    _write_test_pressure_ledger(context, intent)
    _prepare_terminal_failure_drain(context, intent, reason="timeout")
    original = qualification._seal_tree_read_only
    crashed = False

    def crash_before_first_seal(
        root: Path,
        *,
        description: str,
    ) -> None:
        nonlocal crashed
        if description.startswith("failed load cycle") and not crashed:
            crashed = True
            raise RuntimeError("crash before cycle sealing")
        original(root, description=description)

    monkeypatch.setattr(
        qualification, "_seal_tree_read_only", crash_before_first_seal
    )
    with pytest.raises(RuntimeError, match="crash before cycle sealing"):
        qualification._publish_terminal_failure(
            context, intent=intent, reason="timeout"
        )
    assert not (
        context.qualification_root / qualification.FAILURE_NAME
    ).exists()

    monkeypatch.setattr(
        qualification, "_seal_tree_read_only", original
    )
    failure = qualification._publish_terminal_failure(
        context, intent=intent, reason="timeout"
    )
    assert failure["passed"] is False


def test_failure_publish_resumes_sealed_roots_before_marker_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _ensure_passing_cycle_roots(context, intent)
    _write_test_pressure_ledger(context, intent)
    _prepare_terminal_failure_drain(context, intent, reason="timeout")
    cycles = qualification.load_cycle_inventory(context, intent=intent)
    original = qualification._write_once
    crashed = False

    def crash_before_failure_marker(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal crashed
        if description == "terminal qualification failure" and not crashed:
            crashed = True
            raise RuntimeError("crash before failure marker")
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(
        qualification, "_write_once", crash_before_failure_marker
    )
    with pytest.raises(RuntimeError, match="crash before failure marker"):
        qualification._publish_terminal_failure(
            context, intent=intent, reason="timeout"
        )
    assert not (
        context.qualification_root / qualification.FAILURE_NAME
    ).exists()
    for cycle in cycles:
        qualification._assert_tree_read_only(
            cycle.run_root,
            description="cycle sealed before failure marker",
        )

    monkeypatch.setattr(qualification, "_write_once", original)
    failure = qualification._publish_terminal_failure(
        context, intent=intent, reason="timeout"
    )
    assert failure["failure_drain_intent"][
        "failure_drain_intent_id"
    ]


@pytest.mark.parametrize("mutation", ["writable", "tamper", "remove"])
def test_verify_rejects_post_completion_replay_root_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _passing_observations(context, intent)
    replay = qualification.load_cycle_inventory(
        context, intent=intent
    )[1]
    row = replay.run_root / "cells" / "synthetic-row.jsonl"
    qualification.publish_completion(context, intent=intent)
    qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )

    if mutation == "writable":
        replay.run_root.chmod(0o755)
        expected = "not recursively read-only"
    elif mutation == "tamper":
        row.chmod(0o644)
        row.write_text('{"synthetic":"tampered"}\n', encoding="utf-8")
        row.chmod(0o444)
        expected = "aggregate evidence inventory drifted"
    else:
        cells = row.parent
        cells.chmod(0o755)
        row.unlink()
        cells.chmod(0o555)
        expected = "aggregate evidence inventory drifted"

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match=expected,
    ):
        qualification.verify_completed_qualification(
            context.chain_manifest,
            verify_chain=False,
            verify_renderer=False,
        )


def test_verify_rejects_post_completion_refill_journal_drift(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    loaded_cut = next(
        observation
        for observation in observations
        if observation["scheduler"]["active_qualification_cells"]
        == intent["certified_saturation_target"]
    )
    refill = qualification.record_refill_reconciliation(
        context.qualification_root,
        intent=intent,
        scheduler=loaded_cut["scheduler"],
        semantic=loaded_cut["semantic"],
        preceding_dispatch=None,
    )
    for cycle in qualification.load_cycle_inventory(
        context, intent=intent
    ):
        cycle.dispatcher_state.mkdir(parents=True, exist_ok=True)
        qualification.dispatch_sweeps._atomic_write_json(
            cycle.dispatcher_state / "ledger.json",
            qualification.dispatch_sweeps._empty_ledger(),
        )
    qualification.publish_completion(context, intent=intent)
    qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    path = (
        context.qualification_root
        / qualification.REFILL_RECONCILIATION_DIRECTORY
        / qualification._evidence_filename(
            "REFILL", refill["refill_index"]
        )
    )
    path.chmod(0o644)
    value = qualification._read_json(
        path,
        description="test refill record",
    )
    identity = dict(value)
    identity.pop("refill_id")
    identity["active_deficit"] = 1
    path.write_bytes(
        qualification._canonical_bytes(
            qualification._with_identity(identity, "refill_id")
        )
    )
    path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="accounting drifted",
    ):
        qualification.verify_completed_qualification(
            context.chain_manifest,
            verify_chain=False,
            verify_renderer=False,
        )
