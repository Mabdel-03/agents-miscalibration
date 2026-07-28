from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import schema5_control as control
from slurm import dispatch_sweeps as dispatcher
from agents_scaling import runtime_integrity, snapshot_integrity
from agents_scaling.serving import (
    external_watchdog,
    protected_capacity,
    registry,
    scheduler_safety,
)
from agents_scaling.serving.fleet_contract import (
    expected_replica_id,
    expected_scheduler_job_name,
)
from scripts import run_schema5_throughput_qualification as qualification


CLUSTER_FIXTURES = Path(__file__).parent / "fixtures" / "slurm_schema5_cluster"


def _toolchain_binding(root: str | Path) -> dict:
    toolchain = Path(root).resolve()
    binding = {
        "schema_version": control.conda_toolchain.SCHEMA_VERSION,
        "protocol": control.conda_toolchain.PROTOCOL,
        "release_tag": control.PRODUCTION_OPERATIONAL_TAG,
        "chain_namespace": "schema5-v1.2-r11",
        "toolchain_root": str(toolchain),
        "base_prefix": str(toolchain / "base"),
        "completion_marker": {
            "path": str(
                toolchain / control.conda_toolchain.MARKER_NAME
            ),
            "sha256": "1" * 64,
            "size": 1,
        },
        "marker_id": "2" * 64,
        "installer_contract": (
            control.conda_toolchain.PINNED_INSTALLER_CONTRACT.as_dict()
        ),
        "intent_id": "3" * 64,
        "conda_executable": {
            "path": str(toolchain / "base/bin/conda"),
            "sha256": "4" * 64,
            "size": 1,
            "mode": 0o555,
            "link_count": 1,
        },
        "runtime_identity_sha256": "5" * 64,
        "complete_prefix_inventory_sha256": "6" * 64,
        "read_only_probes": {"probe_count": 2},
    }
    binding["binding_id"] = hashlib.sha256(
        (
            json.dumps(
                binding,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return binding


def _package_cache_seed_input(path: str | Path) -> dict:
    binding = {
        "source_package_cache": str(Path(path).resolve()),
        "inventory_sha256": "7" * 64,
        "inventory_entry_count": 1,
        "inventory_file_count": 1,
        "inventory_total_file_bytes": 1,
        "requirements_sha256": "8" * 64,
        "required_package_count": 1,
        "archive_count": 0,
        "selected_top_level_entries": ["cache", "urls", "urls.txt"],
    }
    binding["input_id"] = hashlib.sha256(
        (
            json.dumps(
                binding,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return binding


@pytest.fixture(autouse=True)
def _stub_conda_toolchain_verification(monkeypatch):
    monkeypatch.setattr(
        control.conda_toolchain,
        "verified_conda_toolchain_binding",
        lambda root, exercise=True: _toolchain_binding(root),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_control_subprocess_environment_rejects_ambient_command_authority(
    monkeypatch,
):
    hostile = {
        "PATH": "/tmp/hostile-bin:/usr/bin",
        "ASYS_RELEASE_ID": "ambient-release",
        "BASH_ENV": "/tmp/hostile-bash-env",
        "BASH_FUNC_squeue%%": "() { echo forged; }",
        "CONDA_PREFIX": "/tmp/hostile-conda",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": "/tmp/hostile-hook",
        "GIT_DIR": "/tmp/hostile-git",
        "GIT_REPLACE_REF_BASE": "refs/hostile/",
        "HF_HOME": "/tmp/hostile-hf",
        "LD_AUDIT": "/tmp/hostile-audit.so",
        "LD_PRELOAD": "/tmp/hostile.so",
        "PYTHONPATH": "/tmp/hostile-python",
        "SACCT_FORMAT": "forged",
        "SBATCH_PARTITION": "forged",
        "SCONTROL_ALL": "forged",
        "SLURM_CONF": "/tmp/hostile-slurm.conf",
        "SQUEUE_FORMAT": "forged",
        "TRANSFORMERS_CACHE": "/tmp/hostile-transformers",
        "VLLM_CONFIG_ROOT": "/tmp/hostile-vllm",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)

    environment = control._sanitized_subprocess_environment()

    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["LANG"] == environment["LC_ALL"] == "C"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not (set(hostile) - {"PATH"}).intersection(environment)


def test_default_control_scheduler_runner_uses_absolute_tool_and_clean_env(
    monkeypatch,
):
    observed: dict[str, object] = {}
    monkeypatch.setenv("PATH", "/tmp/hostile-bin")
    monkeypatch.setenv("SBATCH_PARTITION", "hostile")
    monkeypatch.setenv("GIT_DIR", "/tmp/hostile-git")

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(control.subprocess, "run", fake_run)
    completed = control._run_subprocess(["squeue", "--version"])

    assert completed.returncode == 0
    assert observed["argv"] == ["/usr/bin/squeue", "--version"]
    environment = observed["kwargs"]["env"]
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert "SBATCH_PARTITION" not in environment
    assert "GIT_DIR" not in environment


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
    (release / "scripts" / "render_schema5_recovery_chain_v12.py").write_text(
        "#!/usr/bin/env python3\n", encoding="utf-8"
    )
    (release / "scripts" / "render_schema5_recovery_chain_v12.py").chmod(0o444)
    shutil.copy2(
        Path(qualification.__file__).resolve(),
        release / "scripts" / "run_schema5_throughput_qualification.py",
    )
    (release / "configs" / "schema5_monitoring.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    config_root = Path(control.__file__).resolve().parent.parent / "configs"
    model_source = config_root / "model_contracts.v1.json"
    model_checksum_source = config_root / "model_contracts.v1.sha256"
    model_target = tmp_path / "model_contracts.v1.json"
    shutil.copy2(model_source, model_target)
    shutil.copy2(
        model_checksum_source, tmp_path / "model_contracts.v1.sha256"
    )
    model_path = str(model_target.resolve())
    model_hash = _sha(model_target)
    fleet_source = config_root / "schema5_fleet.v1.json"
    fleet_target = tmp_path / "fleet_contract.v1.json"
    shutil.copy2(fleet_source, fleet_target)
    fleet_path = str(fleet_target.resolve())
    fleet_hash = _sha(fleet_target)
    (tmp_path / "fleet_contract.v1.sha256").write_text(
        f"{fleet_hash}  fleet_contract.v1.json\n", encoding="utf-8"
    )
    fleet_payload = json.loads(fleet_target.read_text(encoding="utf-8"))
    effective_payload = copy.deepcopy(fleet_payload)
    effective_target = tmp_path / "fleet_contract.capacity-v1.json"
    effective_target.write_bytes(fleet_target.read_bytes())
    effective_hash = _sha(effective_target)
    effective_target.with_suffix(".sha256").write_text(
        f"{effective_hash}  {effective_target.name}\n",
        encoding="utf-8",
    )
    fleet_target.chmod(0o444)
    effective_target.chmod(0o444)
    release_source_tree_sha256 = control.sha256_tree(release)
    dispatcher_source_sha256 = _sha(
        release / "slurm" / "dispatch_sweeps.py"
    )
    qualification_runner_source_sha256 = _sha(
        release / "scripts" / "run_schema5_throughput_qualification.py"
    )
    results_root = tmp_path / "results"
    server_pool = results_root / "server_pools" / "schema5-v1"
    server_pool.mkdir(parents=True)
    protected_marker_path = protected_capacity.canonical_marker_path(
        results_root
    )
    protected_marker_path.parent.mkdir(parents=True)
    base_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in fleet_payload["profiles"]
    }
    effective_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in effective_payload["profiles"]
    }
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=1,
        release_git_commit="a" * 40,
        source_tree_sha256=release_source_tree_sha256,
        release_fleet_contract_sha256=fleet_hash,
        base_fleet_contract_sha256=fleet_hash,
        proposed_effective_fleet_contract_sha256=effective_hash,
        additive_overlay_contract_sha256=effective_hash,
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256=dispatcher_source_sha256,
        qualification_runner_source_sha256=(
            qualification_runner_source_sha256
        ),
    )
    certificate_path = (
        protected_marker_path.parent
        / protected_capacity.STATIC_FEASIBILITY_FILENAME
    )
    certificate_path.write_bytes(
        protected_capacity.canonical_bytes(certificate)
    )
    certificate_path.chmod(0o444)
    certificate_sha256 = _sha(certificate_path)

    def fleet_topology(payload: dict) -> list[dict]:
        rows = []
        for profile in payload["profiles"]:
            for replica in profile["replicas"]:
                memory = str(replica["memory"])
                assert memory.endswith("G")
                rows.append(
                    {
                        "shape_id": replica["replica_id"],
                        "serving_profile": profile["serving_profile"],
                        "tasks": 1,
                        "cpus": replica["cpus_per_task"],
                        "memory_mib": int(memory[:-1]) * 1024,
                        "gpus": profile["tensor_parallel_size"],
                        "time_limit_seconds": 86_400,
                    }
                )
        return rows

    base_topology = fleet_topology(fleet_payload)
    effective_topology = fleet_topology(effective_payload)
    base_ids = {row["shape_id"] for row in base_topology}
    additive_topology = [
        row for row in effective_topology if row["shape_id"] not in base_ids
    ]
    warm_topology = [
        {
            "shape_id": f"warm-tp1-{index:02d}",
            "serving_profile": "warm-tp1",
            "tasks": 1,
            "cpus": 8,
            "memory_mib": 120 * 1024,
            "gpus": 1,
            "time_limit_seconds": 86_400,
        }
        for index in range(2)
    ] + [
        {
            "shape_id": "warm-tp2-00",
            "serving_profile": "warm-tp2",
            "tasks": 1,
            "cpus": 16,
            "memory_mib": 240 * 1024,
            "gpus": 2,
            "time_limit_seconds": 86_400,
        }
    ]
    protected_marker = {
        "schema_version": protected_capacity.SCHEMA_VERSION,
        "protocol": protected_capacity.PROTOCOL,
        "passed": True,
        "release_id": protected_capacity.RELEASE_ID,
        "release_tag": protected_capacity.RELEASE_TAG,
        "release_git_commit": "a" * 40,
        "release_tag_object": "b" * 40,
        "source_tree_sha256": release_source_tree_sha256,
        "dispatcher_source_sha256": dispatcher_source_sha256,
        "qualification_runner_source_sha256": (
            qualification_runner_source_sha256
        ),
        "chain_namespace": protected_capacity.CHAIN_NAMESPACE,
        "capacity_generation": 1,
        "base_fleet_contract_path": str(fleet_target.resolve()),
        "base_fleet_contract_sha256": fleet_hash,
        "effective_fleet_contract_path": str(effective_target.resolve()),
        "effective_fleet_contract_sha256": effective_hash,
        "additive_overlay_contract_path": str(effective_target.resolve()),
        "additive_overlay_contract_sha256": effective_hash,
        "static_feasibility_certificate": {
            "path": str(certificate_path.resolve()),
            "sha256": certificate_sha256,
            "certificate_id": certificate["certificate_id"],
        },
        "static_feasibility_wave_passed": certificate["wave"]["passed"],
        "static_feasibility_selected_cell_count": certificate[
            "selected_cell_count"
        ],
        "static_feasibility_target_cell_count": certificate["wave"][
            "target_active_cells"
        ],
        "static_feasibility_shortfall_cells": certificate["wave"][
            "shortfall_cells"
        ],
        "static_feasibility_configured_client_ceiling": certificate["wave"][
            "target_active_cells"
        ],
        "static_feasibility_certified_saturation_target": certificate[
            "selected_cell_count"
        ],
        "base_active_logical_replicas": 22,
        "base_active_gpus": 24,
        "base_active_topology": base_topology,
        "base_active_topology_sha256": (
            protected_capacity._sha256_value(base_topology)
        ),
        "additive_reserved_logical_replicas": 0,
        "additive_reserved_gpus": 0,
        "additive_reserved_tp1_replicas": 0,
        "additive_reserved_tp2_replicas": 0,
        "additive_reserved_topology": additive_topology,
        "additive_reserved_topology_sha256": (
            protected_capacity._sha256_value(additive_topology)
        ),
        "effective_active_logical_replicas": 22,
        "effective_active_gpus": 24,
        "effective_active_topology": effective_topology,
        "effective_active_topology_sha256": (
            protected_capacity._sha256_value(effective_topology)
        ),
        "retained_warm_turnover_job_elements": 3,
        "retained_warm_turnover_gpus": 4,
        "retained_warm_turnover_tp1_allocations": 2,
        "retained_warm_turnover_tp2_allocations": 1,
        "retained_warm_turnover_topology": warm_topology,
        "retained_warm_turnover_topology_sha256": (
            protected_capacity._sha256_value(warm_topology)
        ),
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
        "memory_mib": 384 * 4096,
        "preempt_type": "preempt/qos",
        "capacity_source": protected_capacity.CAPACITY_SOURCE,
        "scheduler_cluster": "test_cluster",
        "scheduler_account": "test_account",
        "scheduler_user": "test_user",
        "scheduler_max_jobs": None,
        "scheduler_max_submit_jobs": 500,
        "running_scientific_jobs": 409,
        "minimum_scientific_wall_seconds": 86_400,
        "scientific_qos_contracts": [
            {
                "qos": "normal",
                "max_wall_seconds": 86_400,
                "max_jobs_per_user": None,
                "max_submit_jobs_per_user": 500,
                "required_wall_seconds": 86_400,
                "required_running_jobs": 409,
                "required_submit_jobs": 448,
            }
        ],
        "partition_cpus": 384,
        "partition_memory_mib": 384 * 4096,
        "partition_gpus": 28,
        "fleet_contract_sha256": effective_hash,
        "active_fleet_topology_sha256": (
            protected_capacity._sha256_value(effective_topology)
        ),
        "scientific_server_preempt_mode": "OFF",
        "scientific_client_preempt_mode": "OFF",
        "squeue_complete": True,
        "sacct_complete": True,
        "scheduler_evidence_id": "c" * 64,
        "scheduler_evidence_sha256": "d" * 64,
        "canary_id": "e" * 64,
        "canary_evidence_sha256": "f" * 64,
        "scientific_server_placements": [
            {
                "partition": "ou_bcs_normal",
                "qos": "normal",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "OFF",
                "base_active_gpus": 24,
                "reserved_additive_gpus": 0,
                "effective_active_gpus": 24,
                "retained_warm_turnover_gpus": 4,
                "attested_total_gpus": 28,
                "partition_cpus": 4096,
                "partition_memory_mib": 33_554_432,
                "partition_gpus": 64,
                "partition_nodes": 8,
            },
        ],
        "scientific_client_placements": [
            {
                "partition": "ou_bcs_normal",
                "qos": "normal",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "OFF",
                "slots": 384,
                "cpus": 384,
                "memory_mib": 384 * 4096,
                "reserve_jobs": 64,
                "submit_headroom": 448,
            }
        ],
    }
    protected_marker["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(protected_marker)
    ).hexdigest()
    protected_marker_path.write_bytes(
        protected_capacity.canonical_bytes(protected_marker)
    )
    protected_marker_path.chmod(0o444)
    protected_marker_sha256 = _sha(protected_marker_path)

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
    conda_creation_tool = {
        "path": str((tmp_path / "conda").resolve()),
        "sha256": "1" * 64,
    }
    conda_toolchain = _toolchain_binding(
        tmp_path / control.conda_toolchain.TOOLCHAIN_DIRECTORY_NAME
    )
    package_cache_seed_input = _package_cache_seed_input(
        tmp_path / "source-package-cache"
    )
    ownership_policy = {
        "path": str((tmp_path / "environment_ownership_policy.v1.json").resolve()),
        "sha256": "2" * 64,
    }
    integrity_normalization_policy = {
        "path": str(
            (
                tmp_path
                / "environment_integrity_normalization_policy.v1.json"
            ).resolve()
        ),
        "sha256": "b" * 64,
    }
    conda_package_cache_sha256 = "3" * 64
    conda_package_cache_seed_sha256 = "c" * 64
    capture_id = "4" * 64
    capture_marker_sha256 = "5" * 64
    for role, prefix, manifest_path in (
        ("harness", harness, harness_manifest_path),
        ("serving", serving, serving_manifest_path),
    ):
        inventory = runtime_integrity.directory_inventory(prefix)
        environment_inventories[role] = inventory
        runtime = {"fixture": True}
        locks = {
            "conda_explicit": [
                "@EXPLICIT",
                "https://repo.example.invalid/fixture.conda#" + "6" * 64,
            ],
            "pip_freeze_all": [],
        }
        environment_seed = {
            "capture_id": capture_id,
            "capture_marker_sha256": capture_marker_sha256,
            "prefix": str((tmp_path / "captured-seeds" / role).resolve()),
            "normalized_content_inventory_sha256": (
                "7" * 64 if role == "harness" else "8" * 64
            ),
        }
        normalization_receipt = {
            "id": "9" * 64 if role == "harness" else "a" * 64
        }
        installed_files = {
            "inventory_sha256": inventory["inventory_sha256"],
            "entry_count": inventory["entry_count"],
            "file_count": inventory["file_count"],
            "total_file_bytes": inventory["total_file_bytes"],
        }
        payload = {
            "schema_version": control.ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
            "release_id": control.PRODUCTION_RELEASE_ID,
            "role": role,
            "prefix": str(prefix),
            "sealed_read_only": True,
            "offline_environment": control.REQUIRED_OFFLINE_ENVIRONMENT,
            "conda_toolchain": conda_toolchain,
            "conda_creation_tool": conda_creation_tool,
            "environment_seed": environment_seed,
            "ownership_policy": ownership_policy,
            "integrity_normalization_policy": integrity_normalization_policy,
            "normalization_receipt": normalization_receipt,
            "conda_package_cache_sha256": conda_package_cache_sha256,
            "conda_package_cache_seed_sha256": (
                conda_package_cache_seed_sha256
            ),
            "runtime": runtime,
            "locks": locks,
            "release_package": None,
            "installed_files": installed_files,
            "directory_inventory": inventory,
        }
        payload["environment_content_sha256"] = hashlib.sha256(
            runtime_integrity.canonical_bytes(
                {
                    "runtime": runtime,
                    "locks": locks,
                    "release_package": None,
                    "environment_seed": environment_seed,
                    "ownership_policy": ownership_policy,
                    "integrity_normalization_policy": (
                        integrity_normalization_policy
                    ),
                    "normalization_receipt": normalization_receipt,
                    "conda_toolchain": conda_toolchain,
                    "conda_creation_tool": conda_creation_tool,
                    "conda_package_cache_sha256": conda_package_cache_sha256,
                    "conda_package_cache_seed_sha256": (
                        conda_package_cache_seed_sha256
                    ),
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
    assert source_tree_sha256 == release_source_tree_sha256
    transport_binding = (
        control.scheduler_safety.expected_transport_uncertainty_binding()
    )
    transport_binding_sha256 = (
        control.scheduler_safety.transport_uncertainty_binding_sha256(
            transport_binding
        )
    )
    pins = {
        "release_id": control.PRODUCTION_RELEASE_ID,
        "release_bundle_root": str(release_bundle),
        "release_bundle_id": "pending",
        "release_worktree": str(release),
        "git_commit": "a" * 40,
        "release_tag_object": "b" * 40,
        "source_tree_sha256": source_tree_sha256,
        "transport_uncertainty_binding": transport_binding,
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
        "model_contract_path": model_path,
        "model_contract_sha256": model_hash,
        "fleet_contract_path": fleet_path,
        "fleet_contract_sha256": fleet_hash,
        "protected_capacity_marker_path": str(protected_marker_path),
        "protected_capacity_marker_sha256": protected_marker_sha256,
        "protected_capacity_marker_id": protected_marker["marker_id"],
        "protected_client_partition": "ou_bcs_normal",
        "protected_client_qos": "normal",
        "protected_client_slots": 384,
        "protected_client_cpus": 384,
        "protected_client_memory_mib": 384 * 4096,
        "protected_client_reserve_jobs": 64,
        "protected_client_submit_headroom": 448,
        "harness_environment_prefix": str(harness),
        "harness_environment_manifest_path": str(harness_manifest_path),
        "harness_environment_sha256": harness_hash,
        "serving_environment_prefix": str(serving),
        "serving_environment_manifest_path": str(serving_manifest_path),
        "serving_environment_sha256": serving_hash,
        "conda_toolchain": conda_toolchain,
        "package_cache_seed_input": package_cache_seed_input,
        "conda_package_cache_seed_sha256": (
            conda_package_cache_seed_sha256
        ),
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
        "environment_capture_root": str((tmp_path / "captured-seeds").resolve()),
        "release_worktree": str(release),
        "source_harness_prefix": str(harness) + ".source",
        "source_serving_prefix": str(serving) + ".source",
        "source_package_cache": str(
            (tmp_path / "source-package-cache").resolve()
        ),
        "harness_prefix": str(harness),
        "serving_prefix": str(serving),
    }
    environment_capture = {
        "capture_id": capture_id,
        "capture_marker_sha256": capture_marker_sha256,
        "ownership_policy_path": ownership_policy["path"],
        "ownership_policy_sha256": ownership_policy["sha256"],
        "integrity_normalization_policy_path": (
            integrity_normalization_policy["path"]
        ),
        "integrity_normalization_policy_sha256": (
            integrity_normalization_policy["sha256"]
        ),
        "seed_prefixes": {
            "harness": str((tmp_path / "captured-seeds" / "harness").resolve()),
            "serving": str((tmp_path / "captured-seeds" / "serving").resolve()),
        },
        "stage_records": {
            "harness": {"normalization_receipt_id": "9" * 64},
            "serving": {"normalization_receipt_id": "a" * 64},
        },
    }
    materialization_stage_records = {}
    for role, filename in control.MATERIALIZATION_STAGE_FILENAMES.items():
        stage_payload = {
            "schema_version": control.MATERIALIZATION_SCHEMA_VERSION,
            "release_id": pins["release_id"],
            "stage": role,
        }
        if role == "package_cache":
            stage_payload["content_inventory"] = {
                "content_inventory_sha256": conda_package_cache_sha256
            }
            stage_payload["package_cache_seed"] = {
                "seed_content_inventory_sha256": (
                    conda_package_cache_seed_sha256
                )
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
        "schema_version": control.MATERIALIZATION_SCHEMA_VERSION,
        "release_id": pins["release_id"],
        "git_tag": control.PRODUCTION_OPERATIONAL_TAG,
        "tag_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
        "paths": materialization_paths,
        "complete": True,
        "publication_protocol": "stage_records_fsync_marker_last",
        "stage_records": materialization_stage_records,
        "environment_capture": environment_capture,
        "conda_toolchain": conda_toolchain,
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": package_cache_seed_input,
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
        "schema_version": control.MATERIALIZATION_SCHEMA_VERSION,
        "release_id": pins["release_id"],
        "root": str(release_bundle.parent),
        "marker_path": str(materialization_marker_path),
        "marker_sha256": _sha(materialization_marker_path),
        "materialization_id": materialization_marker["materialization_id"],
        "tag_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
        "paths": materialization_paths,
        "stage_records": materialization_stage_records,
        "environment_capture": environment_capture,
        "conda_toolchain": conda_toolchain,
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": package_cache_seed_input,
        "conda_package_cache_sha256": conda_package_cache_sha256,
        "conda_package_cache_seed_sha256": (
            conda_package_cache_seed_sha256
        ),
    }
    fragment_fields = {
        "release_id",
        "release_worktree",
        "git_commit",
        "release_tag_object",
        "source_tree_sha256",
        "transport_uncertainty_binding",
        "transport_uncertainty_binding_sha256",
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
        "conda_toolchain",
        "package_cache_seed_input",
        "conda_package_cache_seed_sha256",
    }
    identity_path = release_bundle / control.RELEASE_IDENTITY_FILENAME
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": control.RELEASE_BUNDLE_SCHEMA_VERSION,
                "release_id": pins["release_id"],
                "git": {
                    "git_commit": pins["git_commit"],
                    "git_tag": control.PRODUCTION_OPERATIONAL_TAG,
                    "git_tag_object": pins["release_tag_object"],
                    "source_tree_sha256": pins["source_tree_sha256"],
                },
                "release_worktree": pins["release_worktree"],
                "worktree_sealed_read_only": True,
                "materialization": materialization_binding,
                "offline_environment": control.REQUIRED_OFFLINE_ENVIRONMENT,
                "transport_uncertainty": {
                    "binding": transport_binding,
                    "binding_sha256": transport_binding_sha256,
                    "source_tree_sha256": pins["source_tree_sha256"],
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
        "schema_version": control.RELEASE_BUNDLE_SCHEMA_VERSION,
        "release_id": pins["release_id"],
        "complete": True,
        "publication_protocol": "fsync_verify_marker_last",
        "artifacts": artifacts,
        "git_commit": pins["git_commit"],
        "source_tree_sha256": pins["source_tree_sha256"],
        "transport_uncertainty_binding": transport_binding,
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
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


def _write_fleet_current_index(
    control_state: dict, *, rollout_generation: int
) -> None:
    pool_root = Path(control_state["immutable"]["server_pool_root"])
    directory = control.fleet_transactions.state_directory(pool_root)
    ledger_path = control.fleet_transactions.ledger_path(
        directory, rollout_generation
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({"fixture_generation": rollout_generation}) + "\n",
        encoding="utf-8",
    )
    fleet_binding = control.effective_fleet_contract_binding(
        control_state, verify_files=True
    )
    current = {
        "schema_version": control.fleet_transactions.STATE_SCHEMA_VERSION,
        "pool_root": str(pool_root.resolve()),
        "pool_id": "schema5-v1",
        "fleet_sha256": fleet_binding["sha256"],
        "current_generation": rollout_generation,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": _sha(ledger_path),
        "updated_at": 20.0,
    }
    _write_json_artifact(
        directory / control.fleet_transactions.CURRENT_FILENAME, current
    )


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


def _client_capacity_evidence(control_state: dict) -> dict:
    immutable = control_state["immutable"]
    contract = protected_capacity.load_contract(
        immutable["protected_capacity_marker_path"],
        expected_release_git_commit=immutable["git_commit"],
        expected_marker_id=immutable["protected_capacity_marker_id"],
        expected_sha256=immutable["protected_capacity_marker_sha256"],
    )

    def runner(argv, *, timeout):
        assert timeout == 30.0
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            stdout = (
                "KillWait                = 30 sec\n"
                "PreemptMode             = REQUEUE\n"
                "PreemptType             = preempt/qos\n"
            )
        elif args == [
            "scontrol",
            "show",
            "partition",
            "ou_bcs_normal",
            "-o",
        ]:
            stdout = (
                "PartitionName=ou_bcs_normal GraceTime=0 MaxTime=1-00:00:00 "
                "PreemptMode=OFF State=UP TotalNodes=50 AllowQos=normal "
                "TRES=cpu=384,mem=1572864M,node=50,billing=384,"
                "gres/gpu=28\n"
            )
        elif args == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "normal",
            (
                "format=Name,PreemptMode,MaxJobsPerUser,"
                "MaxSubmitJobsPerUser,MaxTRESPerUser,MaxWall"
            ),
        ]:
            stdout = "normal|OFF||500||1-00:00:00\n"
        elif args == [
            "sacctmgr",
            "-nP",
            "show",
            "assoc",
            "user=test_user",
            "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
        ]:
            stdout = "test_cluster|test_account|test_user|normal||500\n"
        elif args == [
            "squeue",
            "-h",
            "-r",
            "-u",
            "test_user",
            (
                "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,"
                "RESIZING,SUSPENDED"
            ),
            "-o",
            "%i|%T|%a|%q",
        ]:
            stdout = ""
        elif (
            args[:6]
            == ["sacct", "-nP", "-X", "--array", "-u", "test_user"]
            and args[-2:] == ["-o", "JobID,State,Account,QOS"]
        ):
            stdout = ""
        else:  # pragma: no cover - defensive fixture boundary
            raise AssertionError(f"unexpected scheduler-safety command: {args!r}")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    return protected_capacity.capture_live_client_capacity(
        contract,
        partition="ou_bcs_normal",
        qos="normal",
        required_time_limit_seconds=43_200,
        trusted_scientific_job_provenance=(
            protected_capacity.build_trusted_scientific_job_provenance(
                scheduler_job_states={},
                scheduler_captured_timestamp=23.0,
                trusted_cell_job_ids=(),
                trusted_fleet_job_ids=(),
                dispatcher_ledger_updated_timestamp=None,
                exact_cell_quiescence=True,
                dispatcher_provenance_id="1" * 64,
                fleet_provenance_id="2" * 64,
                fleet_contract_sha256=contract.fleet_contract_sha256,
                fleet_generation=1,
                scheduler_truth_id="3" * 64,
                now=23.0,
            )
        ),
        runner=runner,
        captured_timestamp=23.0,
    )


def _fleet_scheduler_safety(
    control_state: dict,
) -> tuple[dict, dict, dict[str, int]]:
    requirements = {"ou_bcs_normal": 86_400}

    def runner(argv, *, timeout):
        del timeout
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            stdout = (
                "KillWait                = 30 sec\n"
                "PreemptMode             = REQUEUE\n"
                "PreemptType             = preempt/partition_prio\n"
            )
        elif (
            len(args) == 5
            and args[:3] == ["scontrol", "show", "partition"]
            and args[3] in requirements
            and args[4] == "-o"
        ):
            stdout = (
                f"PartitionName={args[3]} GraceTime=0 "
                "MaxTime=1-00:00:00 PreemptMode=OFF "
                "State=UP TotalNodes=50\n"
            )
        else:  # pragma: no cover - defensive fixture boundary
            raise AssertionError(f"unexpected fleet scheduler command: {args!r}")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    evidence = scheduler_safety.capture_scheduler_safety_evidence(
        list(requirements),
        runner=runner,
        captured_timestamp=23.0,
    )
    policy = scheduler_safety.validate_scheduler_safety_evidence(
        evidence,
        expected_partitions=list(requirements),
        required_time_limits_seconds=requirements,
    )
    return evidence, policy, requirements


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
    if gate == "static_feasibility_certificate":
        authority = control.effective_protected_capacity_binding(
            control_state, verify_files=True
        )
        certificate = protected_capacity.load_static_feasibility_certificate(
            authority["static_feasibility_certificate_path"],
            expected_sha256=authority[
                "static_feasibility_certificate_sha256"
            ],
            expected_certificate_id=authority[
                "static_feasibility_certificate_id"
            ],
            expected_capacity_generation=authority["capacity_generation"],
            expected_base_fleet_contract_sha256=authority[
                "base_fleet_contract_sha256"
            ],
            expected_effective_fleet_contract_sha256=authority[
                "effective_fleet_contract_sha256"
            ],
            expected_additive_overlay_contract_sha256=authority[
                "additive_overlay_contract_sha256"
            ],
            expected_release_git_commit=control_state["immutable"][
                "git_commit"
            ],
        )
        return {
            "capacity_generation": certificate.capacity_generation,
            "certificate_id": certificate.certificate_id,
            "certificate_sha256": certificate.sha256,
            "base_fleet_contract_sha256": (
                certificate.base_fleet_contract_sha256
            ),
            "effective_fleet_contract_sha256": (
                certificate.effective_fleet_contract_sha256
            ),
            "additive_overlay_contract_sha256": (
                certificate.additive_overlay_contract_sha256
            ),
            "effective_logical_replicas": (
                certificate.effective_logical_replicas
            ),
            "effective_active_gpus": certificate.effective_active_gpus,
            "selected_cell_count": int(
                certificate.payload["selected_cell_count"]
            ),
        }
    if gate == "protected_capacity":
        authority = control.effective_protected_capacity_binding(
            control_state, verify_files=True
        )
        contract = protected_capacity.load_contract(
            authority["path"],
            expected_release_git_commit=control_state["immutable"][
                "git_commit"
            ],
            expected_marker_id=authority["marker_id"],
            expected_sha256=authority["sha256"],
        )
        return {
            "capacity_generation": contract.capacity_generation,
            "marker_id": contract.marker_id,
            "marker_sha256": contract.sha256,
            "certificate_id": contract.static_feasibility_certificate_id,
            "effective_fleet_contract_sha256": (
                contract.effective_fleet_contract_sha256
            ),
            "effective_logical_replicas": (
                contract.effective_active_logical_replicas
            ),
            "effective_active_gpus": contract.effective_active_gpus,
            "retained_warm_turnover_gpus": (
                contract.retained_warm_turnover_gpus
            ),
            "attested_total_gpus": contract.attested_total_gpus,
        }
    if gate == "fleet":
        fleet_binding = control.effective_fleet_contract_binding(
            control_state, verify_files=True
        )
        effective_fleet = control.load_effective_fleet_contract(
            control_state
        )
        client_capacity = _client_capacity_evidence(control_state)
        contract = protected_capacity.load_contract(
            control_state["immutable"]["protected_capacity_marker_path"],
            expected_release_git_commit=control_state["immutable"]["git_commit"],
            expected_marker_id=control_state["immutable"][
                "protected_capacity_marker_id"
            ],
            expected_sha256=control_state["immutable"][
                "protected_capacity_marker_sha256"
            ],
        )
        client_summary = (
            protected_capacity.validate_live_client_capacity_evidence(
                contract,
                client_capacity,
                partition="ou_bcs_normal",
                qos="normal",
                required_time_limit_seconds=43_200,
            )
        )
        _scheduler_evidence, scheduler_policy, requirements = (
            _fleet_scheduler_safety(control_state)
        )
        transport = control_state["immutable"][
            "transport_uncertainty_binding"
        ]
        return {
            "logical_replicas": len(effective_fleet.replicas),
            "allocated_gpus": sum(
                replica.gpus_per_replica
                for replica in effective_fleet.replicas
            ),
            "healthy_replicas": len(effective_fleet.replicas),
            "unhealthy_replicas": 0,
            "profile_replicas": control._capacity_profile_counts(
                effective_fleet
            ),
            "revision_mismatches": 0,
            "missing_profiles": 0,
            "stale_registrations": 0,
            "max_heartbeat_age_seconds": 0,
            "captured_timestamp": 23.0,
            "model_contract_sha256": control_state["immutable"][
                "model_contract_sha256"
            ],
            "fleet_contract_sha256": fleet_binding["sha256"],
            "client_capacity_evidence_id": client_summary["evidence_id"],
            "client_partition": client_summary["partition"],
            "client_qos": client_summary["qos"],
            "client_cpu_limit": client_summary["cpu_limit"],
            "client_memory_limit_mib": client_summary["memory_limit_mib"],
            "client_max_submit_jobs": client_summary["max_submit_jobs"],
            "transport_censor_protocol_version": transport[
                "transport_censor_protocol_version"
            ],
            "transport_censor_protocol_hash": transport[
                "transport_censor_protocol_hash"
            ],
            "transport_uncertainty_binding_sha256": control_state[
                "immutable"
            ]["transport_uncertainty_binding_sha256"],
            "scheduler_safety_evidence_id": scheduler_policy[
                "scheduler_evidence_id"
            ],
            "scheduler_safety_policy_id": scheduler_policy["policy_id"],
            "scheduler_safety_policy_contract_id": scheduler_policy[
                "policy_contract_id"
            ],
            "scheduler_preemptible_partitions": scheduler_policy[
                "preemptible_partitions"
            ],
            "scheduler_partition_time_requirements_seconds": requirements,
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
            "transport_incidents": 0,
            "truncation_incidents": 0,
            "provenance_failures": 0,
            "trusted_generation_failures": 0,
        }
    if gate == "email_test":
        request_identity = {
            "immutable_sha256": control_state["immutable_sha256"],
            "chain_id": "a" * 64,
            "request_id": "b" * 64,
            "release_tag": control.PRODUCTION_OPERATIONAL_TAG,
            "release_git_commit": control_state["immutable"]["git_commit"],
            "recipient": control_state["alert_email"],
            "challenge_generation": 1,
            "challenge_nonce": "c" * 64,
            "challenge_salt": "d" * 64,
        }
        return {
            "recipient": control_state["alert_email"],
            "delivery_succeeded": True,
            "returncode": 0,
            "acknowledged": True,
            "chain_id": request_identity["chain_id"],
            "request_id": request_identity["request_id"],
            "release_tag": request_identity["release_tag"],
            "release_git_commit": request_identity[
                "release_git_commit"
            ],
            "challenge_generation": request_identity[
                "challenge_generation"
            ],
            "challenge_id": control.schema5_email_ack.challenge_id_for(
                request_identity
            ),
            "challenge_verifier": "e" * 64,
            "acknowledged_at": "1970-01-01T00:00:23Z",
        }
    raise AssertionError(gate)


def attest_all_non_scheduler(state_dir: Path) -> None:
    current = control.load_control(state_dir)
    _write_fleet_current_index(current, rollout_generation=1)
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

    smoke_catalog = None
    smoke_attempt_binding = None
    for index, gate in enumerate(control.REQUIRED_GATES[:-1]):
        evidence = state_dir / "readiness" / f"{gate}.json"
        evidence.parent.mkdir(exist_ok=True)
        artifacts = []
        for name in control.READINESS_ARTIFACT_NAMES[gate]:
            if gate == "snapshot":
                path = _snapshot_envelope(state_dir, name)
            elif gate == "static_feasibility_certificate":
                authority = control.effective_protected_capacity_binding(
                    current, verify_files=True
                )
                path = Path(
                    authority["static_feasibility_certificate_path"]
                )
            elif gate == "protected_capacity":
                authority = control.effective_protected_capacity_binding(
                    current, verify_files=True
                )
                path = Path(authority["path"])
            elif gate == "fleet" and name == "fleet_contract":
                path = Path(
                    control.effective_fleet_contract_binding(
                        current, verify_files=True
                    )["path"]
                )
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
                    fleet_binding = control.effective_fleet_contract_binding(
                        current, verify_files=True
                    )
                    fleet_contract = json.loads(
                        Path(fleet_binding["path"]).read_text()
                    )
                    (
                        fleet_scheduler_evidence,
                        fleet_scheduler_policy,
                        _fleet_requirements,
                    ) = _fleet_scheduler_safety(current)
                    probe_transport_provenance = {
                        "transport_censor_protocol_version": current[
                            "immutable"
                        ]["transport_uncertainty_binding"][
                            "transport_censor_protocol_version"
                        ],
                        "transport_censor_protocol_hash": current[
                            "immutable"
                        ]["transport_uncertainty_binding"][
                            "transport_censor_protocol_hash"
                        ],
                        "transport_uncertainty_binding_sha256": current[
                            "immutable"
                        ]["transport_uncertainty_binding_sha256"],
                        "source_tree_sha256": current["immutable"][
                            "source_tree_sha256"
                        ],
                    }
                    replica_rows = []
                    registry_references = []
                    replica_number = 0
                    for profile in fleet_contract["profiles"]:
                        for replica in profile["replicas"]:
                            replica_id = replica["replica_id"]
                            runtime_profile = control.get_serving_profile(
                                profile["serving_profile"]
                            )
                            host = f"node-{replica_number:02d}"
                            port = 20_000 + replica_number
                            job_id = str(9_000 + replica_number)
                            registry_path = (
                                Path(current["immutable"]["server_pool_root"])
                                / "servers"
                                / profile["serving_profile"]
                                / f"{host}_{port}.json"
                            )
                            registry = {
                                "replica_id": replica_id,
                                "replica_index": replica["replica_index"],
                                "server_pool_id": "schema5-v1",
                                "serving_profile": profile["serving_profile"],
                                "model_size": profile["model_size"],
                                "hf_id": runtime_profile.hf_id,
                                "served_model_name": (
                                    runtime_profile.served_model_name
                                ),
                                "max_model_len": runtime_profile.max_model_len,
                                "tp_size": runtime_profile.tp_size,
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
                                "fleet_contract_sha256": fleet_binding[
                                    "sha256"
                                ],
                                "release_fleet_contract_sha256": current["immutable"][
                                    "fleet_contract_sha256"
                                ],
                                "capacity_generation": 1,
                                "rollout_generation": 1,
                                "model_revision": profile["model_revision"],
                                "tokenizer_id": profile["tokenizer_id"],
                                "tokenizer_revision": profile[
                                    "tokenizer_revision"
                                ],
                            }
                            _write_json_artifact(registry_path, registry)
                            registry_reference = {
                                "name": f"registry_{replica_id}",
                                "path": str(registry_path.resolve()),
                                "sha256": _sha(registry_path),
                            }
                            registry_references.append(registry_reference)
                            intent_token = f"{replica_number + 1:032x}"
                            script_path = (
                                control.fleet_transactions.state_directory(
                                    current["immutable"]["server_pool_root"]
                                )
                                / "sbatch"
                                / "g000001"
                                / f"{replica_id}.{intent_token}.sbatch"
                            )
                            script_path.parent.mkdir(parents=True, exist_ok=True)
                            script_path.write_text(
                                "#!/bin/bash\ntrue\n", encoding="utf-8"
                            )
                            script_path.chmod(0o444)
                            script_sha256 = _sha(script_path)
                            comment = control.fleet_transactions.intent_comment(
                                pool_id="schema5-v1",
                                profile=profile["serving_profile"],
                                replica_id=replica_id,
                                rollout_generation=1,
                                intent_token=intent_token,
                                fleet_sha256=fleet_binding["sha256"],
                            )
                            raw_scontrol = (
                                f"JobId={job_id} "
                                f"JobName={replica['scheduler_job_name']} "
                                "JobState=RUNNING Reason=None Requeue=0 "
                                f"Comment={shlex.quote(comment)} "
                                f"NodeList={host} "
                                f"Command={shlex.quote(str(script_path.resolve()))}\n"
                            )
                            replica_rows.append(
                                {
                                    "replica_id": replica_id,
                                    "serving_profile": profile["serving_profile"],
                                    "ledger_generation": 1,
                                    "intent_token": intent_token,
                                    "attempt_state": "committed",
                                    "slurm_job_id": job_id,
                                    "comment": comment,
                                    "job_name": replica["scheduler_job_name"],
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
                                        "fleet_contract_sha256": fleet_binding[
                                            "sha256"
                                        ],
                                        "release_fleet_contract_sha256": current[
                                            "immutable"
                                        ]["fleet_contract_sha256"],
                                        "capacity_generation": 1,
                                        "rollout_generation": 1,
                                        "spooled_script_sha256": script_sha256,
                                    },
                                    "local_script_path": str(
                                        script_path.resolve()
                                    ),
                                    "local_script_sha256": script_sha256,
                                    "spooled_script_sha256": script_sha256,
                                    "spooled_script_proof": {
                                        "argv": [
                                            "scontrol",
                                            "write",
                                            "batch_script",
                                            job_id,
                                            "-",
                                        ],
                                        "local_path": str(
                                            script_path.resolve()
                                        ),
                                        "local_sha256": script_sha256,
                                        "observed_sha256": script_sha256,
                                        "observed_bytes": script_path.stat().st_size,
                                        "exact_match": True,
                                    },
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
                                        "job_name": replica[
                                            "scheduler_job_name"
                                        ],
                                        "node": host,
                                        "command": str(script_path.resolve()),
                                        "effective_requeue": 0,
                                        "raw_output": raw_scontrol,
                                        "raw_output_sha256": hashlib.sha256(
                                            raw_scontrol.encode("utf-8")
                                        ).hexdigest(),
                                    },
                                    "http": {
                                        "health_status": 200,
                                        "models_status": 200,
                                        "model_ids": [
                                            runtime_profile.served_model_name
                                        ],
                                        "expected_model": (
                                            runtime_profile.served_model_name
                                        ),
                                        "probe_started_timestamp": 22.0,
                                        "probe_completed_timestamp": 23.0,
                                        "healthy": True,
                                    },
                                    "probe_transport_provenance": (
                                        probe_transport_provenance
                                    ),
                                    "registry_path": str(registry_path.resolve()),
                                    "registry_sha256": _sha(registry_path),
                                }
                            )
                            replica_number += 1
                    raw_fleet = {
                        "schema_version": 3,
                        "kind": "raw_fleet_health_probe",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "server_pool_root": current["immutable"]["server_pool_root"],
                        "rollout_generation": 1,
                        "isolated_foreign_scheduler_job_ids": [],
                        "sealed_successful_handoff_terminal_job_ids": [],
                        "ignored_terminal_job_ids": [],
                        "client_capacity_contract": _client_capacity_evidence(
                            current
                        ),
                        "fleet_scheduler_safety_evidence": (
                            fleet_scheduler_evidence
                        ),
                        "fleet_scheduler_safety_policy": (
                            fleet_scheduler_policy
                        ),
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
                    fleet_binding = control.effective_fleet_contract_binding(
                        current, verify_files=True
                    )
                    if smoke_catalog is None:
                        fleet_contract = json.loads(
                            Path(fleet_binding["path"]).read_text(
                                encoding="utf-8"
                            )
                        )
                        endpoint_job = 98_000
                        for profile_index, profile in enumerate(
                            fleet_contract["profiles"]
                        ):
                            for replica_index, _replica in enumerate(
                                profile["replicas"]
                            ):
                                _seal_control_catalog_endpoint(
                                    state_dir,
                                    profile_index=profile_index,
                                    replica_index=replica_index,
                                    rollout_generation=1,
                                    job_id=str(endpoint_job),
                                )
                                endpoint_job += 1
                        smoke_catalog = (
                            control.refresh_trusted_generation_catalog(
                                state_dir,
                                target_rollout_generation=1,
                                now=19.0,
                            )
                        )
                        smoke_attempt_binding = {
                            "protocol": (
                                "schema5-v1.2-r11-smoke-attempt-binding-v1"
                            ),
                            "attempt_id": (
                                "a000001-g000001-c000001-"
                                f"{smoke_catalog.catalog_id[:16]}"
                            ),
                            "attempt_ordinal": 1,
                            "immutable_sha256": immutable_hash,
                            "capacity_generation": 1,
                            "rollout_generation": 1,
                            "fleet_contract_sha256": fleet_binding["sha256"],
                            "release_fleet_contract_sha256": current[
                                "immutable"
                            ]["fleet_contract_sha256"],
                            "trusted_catalog_id": smoke_catalog.catalog_id,
                        }
                    assert smoke_catalog is not None
                    assert smoke_attempt_binding is not None
                    catalog_binding = {
                        "catalog_id": smoke_catalog.catalog_id,
                        "marker_path": str(smoke_catalog.marker_path),
                        "marker_sha256": smoke_catalog.marker_sha256,
                        "inventory_sha256": smoke_catalog.inventory_sha256,
                        "catalog_payload_sha256": smoke_catalog.catalog_sha256,
                        "allowed_generation_tuple_count": len(
                            smoke_catalog.allowed_generation_tuples
                        ),
                    }
                    allowed_tuple = next(
                        iter(smoke_catalog.allowed_generation_tuples)
                    )
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
                    path = (
                        state_dir
                        / "readiness"
                        / control.SMOKE_ATTEMPT_BASE_NAME
                        / "attempts"
                        / smoke_attempt_binding["attempt_id"]
                        / "artifacts"
                        / f"{name}.json"
                    )
                    zero_fields = {
                        "context_incidents": 0,
                        "protocol_incidents": 0,
                        "transport_incidents": 0,
                        "truncation_incidents": 0,
                        "provenance_failures": 0,
                        "trusted_generation_failures": 0,
                        "semantic_validation_failures": 0,
                        "top_level_length_censored_qids": 0,
                        "top_level_protocol_censored_qids": 0,
                        "top_level_transport_censored_qids": 0,
                        "transport_affected_qids": 0,
                        "transport_censored_coordinates": 0,
                        "auxiliary_length_censored_draws": 0,
                        "auxiliary_protocol_censored_draws": 0,
                        "auxiliary_transport_censored_draws": 0,
                    }
                    payload = {
                        **identity,
                        "run_id": run_id,
                        "expected_cells": expected_cells,
                        "schema5_complete_cells": expected_cells,
                        **zero_fields,
                        "transport_censor_protocol_version": (
                            control.TRANSPORT_CENSOR_PROTOCOL_VERSION
                        ),
                        "transport_censor_protocol_hash": (
                            control.TRANSPORT_CENSOR_PROTOCOL_HASH
                        ),
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
                            "fleet_contract_sha256": fleet_binding["sha256"],
                            "release_fleet_contract_sha256": current["immutable"][
                                "fleet_contract_sha256"
                            ],
                            "capacity_generation": fleet_binding[
                                "capacity_generation"
                            ],
                            "server_pool_root": current["immutable"][
                                "server_pool_root"
                            ],
                            "rollout_generation": (
                                int(current["rollout_generation"]) + 1
                                if isinstance(
                                    current.get("capacity", {}).get(
                                        "active_transition"
                                    ),
                                    dict,
                                )
                                else 1
                            ),
                            "trusted_generation_catalog": catalog_binding,
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
                            "smoke_attempt": smoke_attempt_binding,
                        },
                        "cells": [],
                    }
                    for cell_index in range(expected_cells):
                        cell_id = f"{run_id}-{cell_index}"
                        result_path = (
                            Path(current["immutable"]["results_root"])
                            / control.SMOKE_ATTEMPT_RUNS_NAME
                            / smoke_attempt_binding["attempt_id"]
                            / run_id
                            / "cells"
                            / cell_id
                            / "results.jsonl"
                        )
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        result_path.write_text(
                            json.dumps(
                                {
                                    "coordinate_provenance_identity_counts": [
                                        {
                                            "release_fleet_contract_sha256": (
                                                allowed_tuple[0]
                                            ),
                                            "fleet_contract_sha256": (
                                                allowed_tuple[1]
                                            ),
                                            "capacity_generation": (
                                                allowed_tuple[2]
                                            ),
                                            "rollout_generation": (
                                                allowed_tuple[3]
                                            ),
                                            "endpoint_generation": (
                                                allowed_tuple[4]
                                            ),
                                            "count": 1,
                                        }
                                    ]
                                },
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        result_path.chmod(0o444)
                        payload["cells"].append(
                            {
                                "cell_id": cell_id,
                                "status": "complete",
                                "valid_qids": 1,
                                "expected_qids": 1,
                                "context_incidents": 0,
                                "protocol_incidents": 0,
                                "transport_incidents": 0,
                                "truncation_incidents": 0,
                                "provenance_failures": 0,
                                "top_level_length_censored_qids": 0,
                                "top_level_protocol_censored_qids": 0,
                                "top_level_transport_censored_qids": 0,
                                "transport_affected_qids": 0,
                                "transport_censored_coordinates": 0,
                                "auxiliary_length_censored_draws": 0,
                                "auxiliary_protocol_censored_draws": 0,
                                "auxiliary_transport_censored_draws": 0,
                                "provenance_errors": [],
                                "trusted_generation_errors": [],
                                "results_artifact": {
                                    "path": str(result_path.resolve()),
                                    "sha256": _sha(result_path),
                                    "row_count": 1,
                                },
                                "semantic_errors": [],
                                "failure": None,
                            }
                        )
                elif gate == "email_test":
                    receipt_metrics = _gate_metrics(current, gate)
                    source_root = state_dir / "readiness" / "sources"
                    active_path = (
                        source_root / "active_email_challenge.json"
                    ).resolve()
                    request_path = (
                        source_root / "email_ack_request.json"
                    ).resolve()
                    acknowledgement_path = (
                        source_root / "email_acknowledgement.json"
                    ).resolve()
                    request_payload = {
                        "schema_version": 2,
                        "kind": "schema5_email_ack_request",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "chain_id": receipt_metrics["chain_id"],
                        "request_id": receipt_metrics["request_id"],
                        "release_tag": receipt_metrics["release_tag"],
                        "release_git_commit": receipt_metrics[
                            "release_git_commit"
                        ],
                        "recipient": current["alert_email"],
                        "challenge_generation": receipt_metrics[
                            "challenge_generation"
                        ],
                        "challenge_nonce": "c" * 64,
                        "challenge_salt": "d" * 64,
                        "challenge_verifier_algorithm": (
                            control.schema5_email_ack
                            .EMAIL_CHALLENGE_VERIFIER_ALGORITHM
                        ),
                        "challenge_verifier": receipt_metrics[
                            "challenge_verifier"
                        ],
                        "challenge_id": receipt_metrics["challenge_id"],
                        "active_challenge": str(active_path),
                        "acknowledgement": str(
                            acknowledgement_path
                        ),
                        "delivery_succeeded": True,
                        "returncode": 0,
                        "delivery_attempts": [
                            {
                                "attempt": 1,
                                "returncode": 0,
                                "timed_out": False,
                            }
                        ],
                        "submitted_timestamp": 22.0,
                        "expires_timestamp": 3_622.0,
                        "subject": (
                            control.schema5_email_ack.EMAIL_CHALLENGE_SUBJECT
                        ),
                        "body_template": (
                            control.schema5_email_ack.EMAIL_CHALLENGE_BODY_TEMPLATE
                        ),
                        "acknowledgement_tool": {
                            "path": str(Path(control.__file__).resolve()),
                            "sha256": _sha(Path(control.__file__).resolve()),
                        },
                    }
                    request_ref = source_reference(
                        "email_ack_request", request_payload
                    )
                    active_stable = {
                        "schema_version": 2,
                        "kind": "schema5_email_active_challenge",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "chain_id": receipt_metrics["chain_id"],
                        "request_id": receipt_metrics["request_id"],
                        "release_tag": receipt_metrics["release_tag"],
                        "release_git_commit": receipt_metrics[
                            "release_git_commit"
                        ],
                        "recipient": current["alert_email"],
                        "challenge_generation": receipt_metrics[
                            "challenge_generation"
                        ],
                        "challenge_id": receipt_metrics["challenge_id"],
                        "request": request_ref["path"],
                        "request_sha256": request_ref["sha256"],
                        "acknowledgement": str(
                            acknowledgement_path
                        ),
                        "supersedes": None,
                        "activated_timestamp": 22.0,
                    }
                    active_payload = {
                        **active_stable,
                        "active_challenge_id": (
                            control.schema5_email_ack.identity_sha256(
                                active_stable
                            )
                        ),
                    }
                    active_ref = source_reference(
                        "active_email_challenge", active_payload
                    )
                    acknowledgement_stable = {
                        "schema_version": 2,
                        "kind": "schema5_email_acknowledgement",
                        "passed": True,
                        "immutable_sha256": immutable_hash,
                        "chain_id": receipt_metrics["chain_id"],
                        "request_id": receipt_metrics["request_id"],
                        "release_tag": receipt_metrics["release_tag"],
                        "release_git_commit": receipt_metrics[
                            "release_git_commit"
                        ],
                        "recipient": current["alert_email"],
                        "challenge_generation": receipt_metrics[
                            "challenge_generation"
                        ],
                        "challenge_id": receipt_metrics["challenge_id"],
                        "request": request_ref["path"],
                        "request_sha256": request_ref["sha256"],
                        "operator": "test-operator",
                        "acknowledged_at": receipt_metrics["acknowledged_at"],
                        "acknowledged_timestamp": 23.0,
                    }
                    acknowledgement_payload = {
                        **acknowledgement_stable,
                        "acknowledgement_id": (
                            control.schema5_email_ack.identity_sha256(
                                acknowledgement_stable
                            )
                        ),
                    }
                    payload = {
                        **identity,
                        "metrics": receipt_metrics,
                        "referenced_artifacts": [
                            active_ref,
                            request_ref,
                            source_reference(
                                "email_acknowledgement",
                                acknowledgement_payload,
                            ),
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


def _seal_authorization_fixture(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path.resolve()


def refresh_test_watchdog_mirror(
    state_dir: Path, *, now: float
) -> dict:
    """Publish one exact synthetic five-minute watchdog cycle."""

    for captured in (now - 60.0, now):
        snapshot = control.SchedulerSnapshot((), captured)
        report = control.live_status(
            state_dir, snapshot=snapshot, now=captured
        )
        control.record_external_watchdog_status_observation(
            state_dir, report=report, now=captured
        )
    return control.publish_external_watchdog_cycle(state_dir, now=now)


def attest_test_production_authorizations(
    state_dir: Path, *, now: float = 46.0
) -> None:
    """Publish sealed synthetic verifier outputs for control-plane unit tests."""

    current = control.load_control(state_dir)
    if all(
        current["readiness"][gate].get("passed") is True
        for gate in control.PRODUCTION_AUTHORIZATION_GATES
    ):
        return
    root = state_dir / "production-authorization-fixture"
    manifest_path = root / "RECOVERY_CHAIN_SCHEMA5_V1_2_R11.json"
    chain_id = "1" * 64
    manifest = {
        "schema_version": 1,
        "release_id": control.PRODUCTION_RELEASE_ID,
        "release_tag": control.PRODUCTION_OPERATIONAL_TAG,
        "release_git_commit": current["immutable"]["git_commit"],
        "release_tag_object": "b" * 40,
        "chain_id": chain_id,
    }
    if not manifest_path.exists():
        _seal_authorization_fixture(manifest_path, manifest)
    manifest_sha = _sha(manifest_path)

    qualification_root = root / "schema5_throughput_qualification_v1"
    fixed_marker = qualification_root / "THROUGHPUT_QUALIFICATION_COMPLETE.json"
    attempt_root = qualification_root / "attempts" / "attempt-1"
    attempt_marker = attempt_root / fixed_marker.name
    pointer_path = qualification_root / "attempt-pointers" / "000001-attempt-1.json"
    catalog_id = control.load_trusted_generation_catalog(
        state_dir
    ).catalog_id
    fleet_binding = control.effective_fleet_contract_binding(
        current, verify_files=True
    )
    attempt_id = f"g000001-c000001-{catalog_id}"
    qualification_id = "3" * 64
    qualification_marker = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r11-throughput-qualification-v1",
        "passed": True,
        "release_id": control.PRODUCTION_RELEASE_ID,
        "release_tag": control.PRODUCTION_OPERATIONAL_TAG,
        "release_git_commit": current["immutable"]["git_commit"],
        "release_tag_object": "b" * 40,
        "chain_id": chain_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "qualification_id": qualification_id,
    }
    if not fixed_marker.exists():
        _seal_authorization_fixture(fixed_marker, qualification_marker)
        _seal_authorization_fixture(attempt_marker, qualification_marker)
    readiness_generation = {
        "rollout_generation": 1,
        "capacity_generation": 1,
        "catalog_id": catalog_id,
        "fleet_contract_sha256": fleet_binding["sha256"],
        "release_fleet_contract_sha256": current["immutable"][
            "fleet_contract_sha256"
        ],
    }
    pointer = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r11-throughput-qualification-attempt-pointer-v1"
        ),
        "chain_id": chain_id,
        "attempt_id": attempt_id,
        "readiness_generation": readiness_generation,
        "pointer_id": "4" * 64,
    }
    if not pointer_path.exists():
        _seal_authorization_fixture(pointer_path, pointer)
    current_attempt = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r11-throughput-qualification-current-attempt-v1"
        ),
        "attempt_id": attempt_id,
        "pointer": str(pointer_path.resolve()),
        "pointer_sha256": _sha(pointer_path),
        "pointer_id": pointer["pointer_id"],
        "current_id": "5" * 64,
    }
    current_attempt_path = qualification_root / "CURRENT_ATTEMPT.json"
    if not current_attempt_path.exists():
        _seal_authorization_fixture(current_attempt_path, current_attempt)

    watchdog_drill = root / "EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json"
    watchdog_ready = root / "WATCHDOG_READY.json"
    drill_id = "6" * 64
    watchdog_id = "7" * 64
    drill_payload = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r11-external-watchdog-drill-v1",
        "drill_id": drill_id,
        "control_sha256": current["immutable_sha256"],
    }
    watchdog_payload = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r11-external-watchdog-v1",
        "marker_id": watchdog_id,
        "control_sha256": current["immutable_sha256"],
    }
    if not watchdog_drill.exists():
        _seal_authorization_fixture(watchdog_drill, drill_payload)
        _seal_authorization_fixture(watchdog_ready, watchdog_payload)

    reports = {
        "verify-throughput-qualification": {
            "passed": True,
            "chain_id": chain_id,
            "manifest": str(manifest_path.resolve()),
            "qualification_marker": str(fixed_marker.resolve()),
            "attempt_marker": str(attempt_marker.resolve()),
            "attempt_id": attempt_id,
            "attempt_pointer": str(pointer_path.resolve()),
            "qualification_id": qualification_id,
            "cells": 768,
            "qids": 15_360,
            "health_soak_384_seconds": 7_200.0,
            "loaded_384_seconds": 3_600.0,
            "loaded_384_useful_qids": 15_360,
            "loaded_384_observation_count": 2,
            "loaded_384_exact_saturation": True,
            "throughput_qids_per_day": 201_994.0,
            "chain_verification": {"passed": True, "chain_id": chain_id},
        },
        "verify-watchdog-readiness": {
            "passed": True,
            "chain_id": chain_id,
            "manifest": str(manifest_path.resolve()),
            "control_sha256": current["immutable_sha256"],
            "external_watchdog_drill": {
                "marker": str(watchdog_drill.resolve()),
                "marker_sha256": _sha(watchdog_drill),
                "marker_size": watchdog_drill.stat().st_size,
                "protocol": drill_payload["protocol"],
                "drill_id": drill_id,
            },
            "watchdog_ready": {
                "marker": str(watchdog_ready.resolve()),
                "marker_sha256": _sha(watchdog_ready),
                "marker_size": watchdog_ready.stat().st_size,
                "protocol": watchdog_payload["protocol"],
                "marker_id": watchdog_id,
            },
        },
    }

    def verifier_runner(argv, environment, timeout):
        assert argv[0] == str(
            Path(current["immutable"]["harness_environment_prefix"])
            / "bin"
            / "python"
        )
        assert argv[1] == "-I"
        assert argv[2] == str(
            Path(current["immutable"]["release_worktree"])
            / control.PRODUCTION_AUTHORIZATION_VERIFIER
        )
        assert argv[4:] == ["--chain-manifest", str(manifest_path.resolve())]
        assert timeout == control.PRODUCTION_AUTHORIZATION_VERIFIER_TIMEOUT_SECONDS
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONPATH" not in environment
        return subprocess.CompletedProcess(
            argv, 0, json.dumps(reports[argv[3]]), ""
        )

    refreshed = control.load_control(state_dir)
    if (
        refreshed["readiness"]["throughput_qualification"].get("passed")
        is not True
    ):
        control.attest_gate(
            state_dir,
            gate="throughput_qualification",
            evidence_path=fixed_marker,
            chain_manifest=manifest_path,
            now=now,
            verifier_runner=verifier_runner,
        )
    refreshed = control.load_control(state_dir)
    if refreshed["readiness"]["external_watchdog"].get("passed") is not True:
        control.attest_gate(
            state_dir,
            gate="external_watchdog",
            evidence_path=watchdog_ready,
            chain_manifest=manifest_path,
            now=now + 0.1,
            verifier_runner=verifier_runner,
        )
    # Production authorization is deliberately insufficient without a current
    # cluster-side mirror of the VM's two independent status cuts.
    refresh_test_watchdog_mirror(state_dir, now=now)


def _record_test_watchdog_status_cuts(
    state_dir: Path, *, first: float, second: float
) -> None:
    for captured in (first, second):
        report = control.live_status(
            state_dir,
            snapshot=control.SchedulerSnapshot((), captured),
            now=captured,
        )
        control.record_external_watchdog_status_observation(
            state_dir, report=report, now=captured
        )


def test_external_watchdog_mirror_freshness_future_nan_and_five_minute_refresh(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)

    fresh = control.external_watchdog_mirror_status(
        state_dir, now=100.0
    )
    assert fresh["required"] is True
    assert fresh["healthy"] is True
    assert fresh["status"] == "fresh"
    assert fresh["sequence"] == 1
    assert control.external_watchdog_mirror_status(
        state_dir, now=700.0
    )["healthy"] is True
    assert control.external_watchdog_mirror_status(
        state_dir, now=700.001
    )["status"] == "stale"
    assert control.external_watchdog_mirror_status(
        state_dir, now=99.0
    )["status"] == "invalid"
    for invalid_now in (float("nan"), float("inf"), float("-inf")):
        invalid = control.external_watchdog_mirror_status(
            state_dir, now=invalid_now
        )
        assert invalid["healthy"] is False
        assert invalid["status"] == "invalid"

    refreshed = refresh_test_watchdog_mirror(state_dir, now=400.0)
    assert refreshed["receipt"]["sequence"] == 2
    current = control.external_watchdog_mirror_status(
        state_dir, now=400.0
    )
    assert current["healthy"] is True
    assert current["sequence"] == 2
    mirror_root = (
        state_dir / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
    )
    before_status = {
        path.relative_to(state_dir): path.read_bytes()
        for path in state_dir.rglob("*")
        if path.is_file()
        and (
            mirror_root in path.parents
            or path.name == "external-watchdog-mirror.lock"
        )
    }
    live = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 400.0),
        now=400.0,
    )
    assert live["external_watchdog_mirror"]["receipt_id"] == current[
        "receipt_id"
    ]
    after_status = {
        path.relative_to(state_dir): path.read_bytes()
        for path in state_dir.rglob("*")
        if path.is_file()
        and (
            mirror_root in path.parents
            or path.name == "external-watchdog-mirror.lock"
        )
    }
    assert after_status == before_status


def test_ordinary_live_status_does_not_write_watchdog_observations(tmp_path):
    state_dir, _ = initialize(tmp_path)
    report = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 10.0),
        now=10.0,
    )
    assert report["external_watchdog_mirror"]["required"] is False
    assert not (
        state_dir / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
    ).exists()


def test_resume_fails_when_watchdog_ready_but_cluster_mirror_is_missing(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir, now=50.0)
    latest = (
        state_dir
        / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
        / "LATEST.json"
    )
    latest.unlink()
    with pytest.raises(control.ReadinessError, match="cluster-mirrored"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 50.0),
            now=50.0,
        )


def test_watchdog_mirror_rejects_pointer_replay_and_current_identity_drift(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    pointer_path = (
        state_dir
        / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
        / "LATEST.json"
    )
    old_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    refresh_test_watchdog_mirror(state_dir, now=400.0)
    control._atomic_write_json(pointer_path, old_pointer)
    replayed = control.external_watchdog_mirror_status(
        state_dir, now=400.0
    )
    assert replayed["healthy"] is False
    assert replayed["status"] == "invalid"
    assert "pointer conflicts" in replayed["error"]

    # The fixed observe selector may repair a mutable pointer only from the sealed
    # immutable journal tail.
    repaired = control.publish_external_watchdog_cycle(
        state_dir, now=401.0
    )
    assert repaired["recovered"] is True
    assert control.external_watchdog_mirror_status(
        state_dir, now=401.0
    )["healthy"] is True

    current = control.load_control(state_dir)
    wrong_rollout = copy.deepcopy(current)
    wrong_rollout["rollout_generation"] += 1
    assert control.external_watchdog_mirror_status(
        state_dir, control=wrong_rollout, now=401.0
    )["status"] == "invalid"
    wrong_control = copy.deepcopy(current)
    wrong_control["immutable_sha256"] = "f" * 64
    assert control.external_watchdog_mirror_status(
        state_dir, control=wrong_control, now=401.0
    )["status"] == "invalid"


def test_watchdog_cycle_crash_replays_intent_and_pointer_exactly_once(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )
    original = control._atomic_publish_readonly_json
    crashed = False

    def crash_before_receipt(path, value):
        nonlocal crashed
        if path.parent.name == "cycle_receipts" and not crashed:
            crashed = True
            raise OSError("synthetic receipt publication crash")
        return original(path, value)

    monkeypatch.setattr(
        control, "_atomic_publish_readonly_json", crash_before_receipt
    )
    with pytest.raises(OSError, match="synthetic"):
        control.publish_external_watchdog_cycle(state_dir, now=221.0)
    monkeypatch.setattr(
        control, "_atomic_publish_readonly_json", original
    )
    recovered = control.publish_external_watchdog_cycle(
        state_dir, now=222.0
    )
    assert recovered["recovered"] is True
    assert recovered["receipt"]["sequence"] == 2
    assert len(control._load_watchdog_cycle_receipts(state_dir)) == 2


def test_watchdog_pointer_crash_adopts_sealed_receipt_without_duplicate(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )
    original = control._atomic_write_json
    crashed = False

    def crash_before_pointer(path, value):
        nonlocal crashed
        if path.name == "LATEST.json" and not crashed:
            crashed = True
            raise OSError("synthetic pointer publication crash")
        return original(path, value)

    monkeypatch.setattr(control, "_atomic_write_json", crash_before_pointer)
    with pytest.raises(OSError, match="synthetic"):
        control.publish_external_watchdog_cycle(state_dir, now=221.0)
    monkeypatch.setattr(control, "_atomic_write_json", original)
    recovered = control.publish_external_watchdog_cycle(
        state_dir, now=222.0
    )
    assert recovered["recovered"] is True
    assert recovered["receipt"]["sequence"] == 2
    assert len(control._load_watchdog_cycle_receipts(state_dir)) == 2
    assert control.external_watchdog_mirror_status(
        state_dir, now=222.0
    )["healthy"] is True


def test_watchdog_mutator_cleans_writable_publication_and_pointer_temps(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    mirror_root = (
        state_dir / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
    )
    status_root = mirror_root / "status_receipts"
    status_root.mkdir(parents=True)
    destination = f"{1:012d}-{'a' * 64}.json"
    partial = b'{"incomplete":true'
    publication_temp = status_root / (
        f".{destination}.publish."
        f"{hashlib.sha256(partial).hexdigest()}.{'b' * 32}.tmp"
    )
    publication_temp.write_bytes(partial)
    pointer_temp = mirror_root / ".LATEST.json.interrupted.tmp"
    pointer_temp.write_text("incomplete", encoding="utf-8")

    report = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 10.0),
        now=10.0,
    )
    receipt = control.record_external_watchdog_status_observation(
        state_dir, report=report, now=10.0
    )

    assert receipt["sequence"] == 1
    assert not publication_temp.exists()
    assert not pointer_temp.exists()
    assert not [
        path
        for path in status_root.iterdir()
        if path.name.startswith(".")
    ]


def test_watchdog_mutator_adopts_sealed_publication_prelink(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    original, original_path = control._load_watchdog_status_observations(
        state_dir
    )[0]
    raw = original_path.read_bytes()
    publication_temp = original_path.parent / (
        f".{original_path.name}.publish."
        f"{hashlib.sha256(raw).hexdigest()}.{'c' * 32}.tmp"
    )
    original_path.rename(publication_temp)
    assert not original_path.exists()
    assert publication_temp.exists()
    assert publication_temp.stat().st_mode & 0o222 == 0

    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )

    assert original_path.read_bytes() == raw
    assert not publication_temp.exists()
    observations = control._load_watchdog_status_observations(state_dir)
    assert observations[0][0] == original
    assert len(observations) == 4


def test_watchdog_foreign_hidden_preimage_is_rejected_not_deleted(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    status_root = (
        state_dir
        / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
        / "status_receipts"
    )
    foreign = status_root / ".foreign.tmp"
    foreign.write_text("do not delete", encoding="utf-8")

    status = control.external_watchdog_mirror_status(
        state_dir, now=100.0
    )
    assert status["healthy"] is False
    assert status["status"] == "invalid"
    assert "hidden entry" in status["error"]
    with pytest.raises(control.ControlError, match="foreign publication"):
        _record_test_watchdog_status_cuts(
            state_dir, first=160.0, second=220.0
        )
    assert foreign.read_text(encoding="utf-8") == "do not delete"


def test_concurrent_watchdog_observe_publishes_one_cycle(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )
    successes = []
    failures = []

    def publish(timestamp):
        try:
            successes.append(
                control.publish_external_watchdog_cycle(
                    state_dir, now=timestamp
                )
            )
        except control.ControlError as exc:
            failures.append(str(exc))

    threads = [
        threading.Thread(target=publish, args=(221.0,)),
        threading.Thread(target=publish, args=(222.0,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1
    assert len(failures) == 1
    assert "two unconsumed" in failures[0]
    assert len(control._load_watchdog_cycle_receipts(state_dir)) == 2


def test_watchdog_action_receipt_is_adopted_and_consumed_once(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )
    intent = control.begin_external_watchdog_action(
        state_dir, action="repair-chain", now=221.0
    )
    receipt = control.complete_external_watchdog_action(
        state_dir,
        intent=intent,
        result={"desired_state": "paused", "submitted": []},
        now=222.0,
    )
    cycle = control.publish_external_watchdog_cycle(
        state_dir, now=223.0
    )
    assert cycle["receipt"]["action_receipt"]["action_receipt_id"] == receipt[
        "action_receipt_id"
    ]

    _record_test_watchdog_status_cuts(
        state_dir, first=280.0, second=340.0
    )
    second_intent = control.begin_external_watchdog_action(
        state_dir, action="repair-chain", now=341.0
    )
    second_receipt = control.complete_external_watchdog_action(
        state_dir,
        intent=second_intent,
        result={"desired_state": "paused", "submitted": []},
        now=342.0,
    )
    adopted = control.adopt_pending_external_watchdog_action(
        state_dir, now=342.5
    )
    assert adopted["action_receipt_id"] == second_receipt[
        "action_receipt_id"
    ]
    with pytest.raises(control.ControlError, match="must be mirrored"):
        control.begin_external_watchdog_action(
            state_dir, action="finalizer-reconcile", now=342.6
        )
    consumed = control.publish_external_watchdog_cycle(
        state_dir, now=343.0
    )
    assert consumed["receipt"]["consumed_action_sequence"] == 2
    assert control.adopt_pending_external_watchdog_action(
        state_dir, now=343.5
    ) is None


def test_interrupted_watchdog_action_is_recovered_after_current_cuts(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    intent = control.begin_external_watchdog_action(
        state_dir, action="repair-chain", now=101.0
    )
    assert intent["sequence"] == 1
    _record_test_watchdog_status_cuts(
        state_dir, first=160.0, second=220.0
    )
    recovered = control.adopt_pending_external_watchdog_action(
        state_dir, now=221.0
    )
    assert recovered["outcome"] == "recovered_interrupted"
    assert recovered["recovered_now"] is True
    cycle = control.publish_external_watchdog_cycle(
        state_dir, now=222.0
    )
    assert cycle["receipt"]["action_receipt"]["outcome"] == (
        "recovered_interrupted"
    )
    assert cycle["receipt"]["status_observations"][-1][
        "report_captured_timestamp"
    ] == 220.0
    assert control.adopt_pending_external_watchdog_action(
        state_dir, now=223.0
    ) is None


def test_watchdog_cycle_semantics_reject_gap_replay_skip_and_action_timing(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir, now=100.0)
    statuses_list = control._load_watchdog_status_observations(state_dir)
    statuses = {
        row["sequence"]: (copy.deepcopy(row), path)
        for row, path in statuses_list
    }
    cycles = control._load_watchdog_cycle_receipts(state_dir)
    valid = copy.deepcopy(cycles[-1][0])

    gap_59 = copy.deepcopy(statuses)
    gap_59[2][0]["report_captured_timestamp"] = (
        gap_59[1][0]["report_captured_timestamp"] + 59.0
    )
    with pytest.raises(control.ControlError, match="gap"):
        control._validate_watchdog_cycle_semantics(
            valid,
            statuses=gap_59,
            actions={},
            previous=None,
            description="gap-59",
        )

    repeated = copy.deepcopy(valid)
    repeated["status_observations"][1]["sequence"] = repeated[
        "status_observations"
    ][0]["sequence"]
    with pytest.raises(control.ControlError, match="repeated"):
        control._validate_watchdog_cycle_semantics(
            repeated,
            statuses=statuses,
            actions={},
            previous=None,
            description="repeated-cut",
        )

    wrong_consumption = copy.deepcopy(valid)
    wrong_consumption["consumed_status_sequence"] = 1
    with pytest.raises(control.ControlError, match="consumption"):
        control._validate_watchdog_cycle_semantics(
            wrong_consumption,
            statuses=statuses,
            actions={},
            previous=None,
            description="replayed-consumption",
        )

    # Action receipts are contiguous: unlike deliberately skippable status noise,
    # sequence three cannot be consumed while sequence two is still pending.
    action_template = {
        "sequence": 3,
        "identity_before": valid["identity"],
        "identity_after": valid["identity"],
        "intent": {"started_timestamp": 101.0},
        "cluster_server_timestamp": 102.0,
    }
    skipped_action = copy.deepcopy(valid)
    skipped_action["action_receipt"] = {"sequence": 3}
    skipped_action["consumed_action_sequence"] = 3
    skipped_action["cluster_server_timestamp"] = 103.0
    with pytest.raises(control.ControlError, match="replayed or inconsistent"):
        control._validate_watchdog_cycle_semantics(
            skipped_action,
            statuses=statuses,
            actions={3: (action_template, Path("/unused"))},
            previous=None,
            description="skipped-action",
        )

    action_template["sequence"] = 1
    action_template["intent"]["started_timestamp"] = 99.0
    early_action = copy.deepcopy(valid)
    early_action["action_receipt"] = {"sequence": 1}
    early_action["consumed_action_sequence"] = 1
    early_action["cluster_server_timestamp"] = 103.0
    with pytest.raises(control.ControlError, match="timing"):
        control._validate_watchdog_cycle_semantics(
            early_action,
            statuses=statuses,
            actions={1: (action_template, Path("/unused"))},
            previous=None,
            description="early-action",
        )


def _seal_control_catalog_endpoint(
    state_dir: Path,
    *,
    profile_index: int = 0,
    replica_index: int = 0,
    rollout_generation: int = 1,
    job_id: str = "98765",
) -> registry.EndpointHistoryRecord:
    current = control.load_control(state_dir)
    immutable = current["immutable"]
    binding = control.effective_fleet_contract_binding(
        current, verify_files=True
    )
    contract = json.loads(
        Path(binding["path"]).read_text(encoding="utf-8")
    )
    profile = contract["profiles"][profile_index]
    replica = profile["replicas"][replica_index]
    runtime = control.get_serving_profile(profile["serving_profile"])
    pool = Path(immutable["server_pool_root"]).resolve()
    entry = registry.ServerEntry(
        model_size=profile["model_size"],
        hf_id=runtime.hf_id,
        host="catalog-node-001",
        port=19_876,
        slurm_job_id=job_id,
        started_at=41.0,
        serving_profile=profile["serving_profile"],
        served_model_name=runtime.served_model_name,
        max_model_len=runtime.max_model_len,
        tp_size=runtime.tp_size,
        release_id=immutable["release_id"],
        environment_hash=immutable["serving_environment_sha256"],
        model_revision=profile["model_revision"],
        tokenizer_id=profile["tokenizer_id"],
        tokenizer_revision=profile["tokenizer_revision"],
        model_contract_sha256=immutable["model_contract_sha256"],
        fleet_contract_sha256=binding["sha256"],
        server_pool_id="schema5-v1",
        replica_id=replica["replica_id"],
        replica_index=replica["replica_index"],
        release_fleet_contract_sha256=immutable[
            "fleet_contract_sha256"
        ],
        capacity_generation=binding["capacity_generation"],
        rollout_generation=rollout_generation,
    )
    registry.write_standby_entry(pool, entry)
    intent_token = f"{int(job_id):032x}"[-32:]
    script = (
        pool
        / ".fleet-transactions-v1"
        / "sbatch"
        / f"g{rollout_generation:06d}"
        / f"{entry.replica_id}.{intent_token}.sbatch"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script_payload = "#!/bin/bash\ntrue\n"
    if script.exists():
        assert (
            not script.is_symlink()
            and script.read_text(encoding="utf-8") == script_payload
            and script.stat().st_mode & 0o222 == 0
        )
    else:
        script.write_text(script_payload, encoding="utf-8")
        script.chmod(0o444)
    script_sha256 = _sha(script)
    spooled_provenance = {
        "run_root": str(pool),
        "server_pool_id": entry.server_pool_id,
        "replica_id": entry.replica_id,
        "replica_index": entry.replica_index,
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": (
            entry.release_fleet_contract_sha256
        ),
        "fleet_contract_sha256": entry.fleet_contract_sha256,
        "capacity_generation": entry.capacity_generation,
        "rollout_generation": entry.rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": script_sha256,
    }
    comment = control.fleet_transactions.intent_comment(
        pool_id="schema5-v1",
        profile=str(entry.serving_profile),
        replica_id=str(entry.replica_id),
        rollout_generation=rollout_generation,
        intent_token=intent_token,
        fleet_sha256=str(entry.fleet_contract_sha256),
    )
    return registry.seal_endpoint_history(
        pool,
        entry,
        release_fleet_contract_sha256=str(
            entry.release_fleet_contract_sha256
        ),
        capacity_generation=int(entry.capacity_generation),
        rollout_generation=rollout_generation,
        ledger_generation=rollout_generation,
        intent_token=intent_token,
        intent_state="committed",
        committed_at=40.0,
        local_script_path=script,
        local_script_sha256=script_sha256,
        spooled_script=script.read_bytes(),
        spooled_provenance=spooled_provenance,
        scheduler_job_name=replica["scheduler_job_name"],
        scheduler_comment=comment,
        sealed_at=42.0,
    )


def _ensure_resume_generation_catalog(
    state_dir: Path, *, now: float
) -> control.TrustedGenerationCatalog:
    current = control.load_control(state_dir)
    rollout_generation = int(current["rollout_generation"]) + 1
    try:
        existing = control.load_trusted_generation_catalog(state_dir)
        control._validate_resume_catalog_authority(
            state_dir,
            current,
            target_rollout_generation=rollout_generation,
            catalog=existing,
        )
        return existing
    except (control.ControlError, control.GenerationCatalogError):
        pass
    fleet = json.loads(
        Path(
            control.effective_fleet_contract_binding(
                current, verify_files=True
            )["path"]
        ).read_text(encoding="utf-8")
    )
    job_number = 98_000 + rollout_generation * 1_000
    for profile_index, profile in enumerate(fleet["profiles"]):
        for replica_index, _replica in enumerate(profile["replicas"]):
            _seal_control_catalog_endpoint(
                state_dir,
                profile_index=profile_index,
                replica_index=replica_index,
                rollout_generation=rollout_generation,
                job_id=str(job_number),
            )
            job_number += 1
    return control.refresh_trusted_generation_catalog(
        state_dir,
        target_rollout_generation=rollout_generation,
        now=now,
    )


def test_control_publishes_and_reloads_exact_trusted_generation_catalog(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    published = _ensure_resume_generation_catalog(state_dir, now=50.0)
    loaded = control.load_trusted_generation_catalog(state_dir)
    assert loaded.catalog_id == published.catalog_id
    assert loaded.marker_sha256 == published.marker_sha256
    endpoint = next(
        identity
        for identity in loaded.allowed_generation_tuples
        if identity[3] == 1
    )
    release, fleet, capacity, rollout, endpoint_generation = (
        endpoint
    )
    assert (
        release,
        fleet,
        capacity + 1,
        rollout,
        endpoint_generation,
    ) not in loaded.allowed_generation_tuples


def test_resume_fails_closed_when_trusted_generation_pointer_is_missing(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    current_pointer = (
        control.trusted_generation_catalog_root(state_dir) / "CURRENT.json"
    )
    current_pointer.unlink()

    with pytest.raises(control.ReadinessError, match="CURRENT.json is missing"):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 50.0),
            submit_runner=lambda _argv: pytest.fail(
                "resume must not submit without generation authority"
            ),
            now=50.0,
        )

    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["rollout_generation"] == 0
    assert current["resume_intent"] is None


@pytest.mark.parametrize("finalization_state", ["complete", "blocked"])
def test_resume_requires_finalization_state_exactly_idle_before_submission(
    tmp_path, monkeypatch, finalization_state
):
    state_dir, _ = initialize(tmp_path)
    current = control.load_control(state_dir)
    current["finalization"]["state"] = finalization_state
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: copy.deepcopy(current),
    )

    with pytest.raises(
        control.ReadinessError,
        match="finalization state exactly idle",
    ):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: pytest.fail(
                "scheduler queried through non-idle finalization"
            ),
            submit_runner=lambda _argv: pytest.fail(
                "controller submitted through non-idle finalization"
            ),
            now=50.0,
        )


def test_resume_authority_rejects_catalog_missing_one_fleet_endpoint(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    trusted = _ensure_resume_generation_catalog(state_dir, now=50.0)
    current = control.load_control(state_dir)
    target = sorted(
        identity
        for identity in trusted.allowed_generation_tuples
        if identity[3] == 1
    )
    assert len(target) > 1
    incomplete = control.TrustedGenerationCatalog(
        marker_path=trusted.marker_path,
        marker_sha256=trusted.marker_sha256,
        inventory_sha256=trusted.inventory_sha256,
        catalog_sha256=trusted.catalog_sha256,
        catalog_id=trusted.catalog_id,
        entries=trusted.entries,
        generations=trusted.generations,
        allowed_generation_tuples=frozenset(target[:-1]),
    )

    with pytest.raises(control.ReadinessError, match="every exact fleet endpoint"):
        control._validate_resume_catalog_authority(
            state_dir,
            current,
            target_rollout_generation=1,
            catalog=incomplete,
        )


def _rewrite_rehashed_fleet_evidence(state_dir: Path, mutate) -> None:
    """Rewrite every reference hash to model a self-consistent evidence forgery."""

    current = control.load_control(state_dir)
    outer_path = Path(current["readiness"]["fleet"]["evidence"])
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    wrapper_ref = next(
        item
        for item in outer["artifacts"]
        if item["name"] == "fleet_health_report"
    )
    wrapper_path = Path(wrapper_ref["path"])
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    raw_ref = wrapper["referenced_artifacts"][0]
    raw_path = Path(raw_ref["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    mutate(raw)
    raw_path.write_text(
        json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_ref["sha256"] = _sha(raw_path)
    wrapper_path.write_text(
        json.dumps(wrapper, sort_keys=True) + "\n", encoding="utf-8"
    )
    wrapper_ref["sha256"] = _sha(wrapper_path)
    outer_path.write_text(
        json.dumps(outer, sort_keys=True) + "\n", encoding="utf-8"
    )
    current["readiness"]["fleet"]["sha256"] = _sha(outer_path)
    control._save_control(state_dir, current, now=49.0)


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
    _ensure_resume_generation_catalog(state_dir, now=now + 0.01)
    attest_test_production_authorizations(state_dir, now=now + 0.1)
    return marker


def resume_ready(
    state_dir: Path,
    *,
    now: float = 50.0,
    accounting_jobs: tuple[control.SchedulerJob, ...] = (),
) -> tuple[dict, tuple]:
    if not (state_dir / control.DRILL_COMPLETE_FILENAME).exists():
        complete_drill_marker(state_dir, now=now - 1)
    _ensure_resume_generation_catalog(state_dir, now=now - 0.75)
    attest_test_production_authorizations(state_dir, now=now - 0.5)
    jobs = list(accounting_jobs)
    identifiers = iter(
        ("1700", "1701", "1702", "1703", "1704", "1705")
        if accounting_jobs
        else ("700", "701", "702", "703", "704", "705")
    )

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
        assert isinstance(argv, control._ExactSbatchInvocation)
        assert argv.stdin_bytes.startswith(b"#!/bin/bash\n")
        assert str(state_dir).encode() in argv.stdin_bytes
        assert all(not item.endswith(".sbatch") for item in argv)
        assert "--hold" in argv
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
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
                "sbatch --stdin",
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


def _watchdog_deployment_evidence(state_dir: Path, path: Path) -> Path:
    current = control.load_control(state_dir)
    value = {
        "schema_version": 1,
        "protocol": "schema5-external-watchdog-deployment-evidence-v1",
        "passed": True,
        "release_id": control.PRODUCTION_RELEASE_ID,
        "release_tag": control.PRODUCTION_OPERATIONAL_TAG,
        "release_git_commit": current["immutable"]["git_commit"],
        "release_tag_object": "b" * 40,
        "chain_namespace": "schema5-v1.2-r11",
        "deployment_id": "1" * 64,
        "watchdog_code_sha256": "2" * 64,
        "immutable_release_sha256": "3" * 64,
        "control_sha256": current["immutable_sha256"],
        "liveness_email": current["alert_email"],
        "forced_command_only": True,
        "timer_seconds": 300,
        "systemd_service_loaded": True,
        "systemd_timer_active": True,
        "systemd_timer_enabled": True,
    }
    value["evidence_id"] = control._watchdog_self_hash(
        value, "evidence_id"
    )
    path.write_bytes(control._watchdog_canonical_json(value))
    path.chmod(0o444)
    return path


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


def test_static_release_input_bindings_reject_self_consistent_substitution(
    tmp_path,
):
    toolchain = _toolchain_binding(
        tmp_path / control.conda_toolchain.TOOLCHAIN_DIRECTORY_NAME
    )
    assert control._validate_conda_toolchain_binding_static(toolchain) == toolchain
    substituted_toolchain = copy.deepcopy(toolchain)
    substituted_toolchain["conda_executable"]["path"] = str(
        tmp_path / "foreign-conda"
    )
    substituted_toolchain["binding_id"] = hashlib.sha256(
        (
            json.dumps(
                {
                    key: value
                    for key, value in substituted_toolchain.items()
                    if key != "binding_id"
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(control.ImmutablePinError, match="toolchain binding"):
        control._validate_conda_toolchain_binding_static(
            substituted_toolchain
        )

    cache = _package_cache_seed_input(tmp_path / "source-package-cache")
    assert control._validate_package_cache_seed_input_binding(cache) == cache
    substituted_cache = copy.deepcopy(cache)
    substituted_cache["selected_top_level_entries"] = ["../escape"]
    substituted_cache["input_id"] = hashlib.sha256(
        (
            json.dumps(
                {
                    key: value
                    for key, value in substituted_cache.items()
                    if key != "input_id"
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(control.ImmutablePinError, match="package-cache"):
        control._validate_package_cache_seed_input_binding(substituted_cache)


@pytest.mark.parametrize(
    ("temporary_payload", "sealed", "already_linked"),
    (
        (b"", False, False),
        (b"partial", False, False),
        (b"immutable-payload", True, False),
        (b"immutable-payload", True, True),
    ),
)
def test_readonly_publisher_recovers_every_owned_temp_boundary(
    tmp_path, temporary_payload, sealed, already_linked
):
    target = tmp_path / "MARKER.json"
    payload = b"immutable-payload"
    digest = hashlib.sha256(payload).hexdigest()
    temporary = tmp_path / (
        f".{target.name}.publish.{digest}." + "1" * 32 + ".tmp"
    )
    temporary.write_bytes(temporary_payload)
    if sealed:
        temporary.chmod(0o444)
    if already_linked:
        os.link(temporary, target)
        assert target.stat().st_nlink == 2

    control._atomic_publish_readonly_bytes(target, payload)

    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o222 == 0
    assert target.stat().st_nlink == 1
    assert not temporary.exists()


def test_readonly_publisher_rejects_foreign_namespace_and_never_clobbers(
    tmp_path,
):
    target = tmp_path / "MARKER.json"
    foreign = tmp_path / f".{target.name}.publish.foreign.tmp"
    foreign.write_bytes(b"unowned")
    with pytest.raises(control.ImmutablePinError, match="foreign"):
        control._atomic_publish_readonly_bytes(target, b"expected")
    assert not target.exists()
    assert foreign.read_bytes() == b"unowned"

    foreign.unlink()
    control._atomic_publish_readonly_bytes(target, b"winner")
    with pytest.raises(control.ImmutablePinError, match="different bytes"):
        control._atomic_publish_readonly_bytes(target, b"loser")
    assert target.read_bytes() == b"winner"


def test_readonly_publisher_serializes_concurrent_same_and_different_payloads(
    tmp_path,
):
    same_target = tmp_path / "same.json"
    same_errors: list[BaseException] = []

    def publish_same():
        try:
            control._atomic_publish_readonly_bytes(same_target, b"same")
        except BaseException as exc:  # pragma: no cover - asserted below.
            same_errors.append(exc)

    threads = [threading.Thread(target=publish_same) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert same_errors == []
    assert same_target.read_bytes() == b"same"
    assert same_target.stat().st_nlink == 1

    different_target = tmp_path / "different.json"
    outcomes: list[str] = []

    def publish_different(payload: bytes):
        try:
            control._atomic_publish_readonly_bytes(different_target, payload)
            outcomes.append("published")
        except control.ImmutablePinError:
            outcomes.append("rejected")

    threads = [
        threading.Thread(target=publish_different, args=(payload,))
        for payload in (b"first", b"second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["published", "rejected"]
    assert different_target.read_bytes() in {b"first", b"second"}
    assert different_target.stat().st_nlink == 1


def test_init_recovers_readonly_pins_after_link_crash(tmp_path, monkeypatch):
    state_dir = tmp_path / "results" / ".dispatcher-schema5-v1"
    pins = make_pins(tmp_path)
    pins_path = state_dir / control.IMMUTABLE_PINS_FILENAME
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == pins_path:
            crashed["value"] = True
            raise RuntimeError("death after immutable pins link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="immutable pins link"):
            control.initialize_control(state_dir, pins=pins, now=10.0)

    assert pins_path.is_file()
    assert pins_path.stat().st_nlink == 1
    assert pins_path.stat().st_mode & 0o222 == 0
    before = pins_path.read_bytes()
    with pytest.raises(control.ControlError, match="conflicts"):
        control.initialize_control(
            state_dir,
            pins=pins,
            alert_email="different@example.invalid",
            now=999.0,
        )
    recovered = control.initialize_control(state_dir, pins=pins, now=999.0)
    assert recovered["created_timestamp"] == 10.0
    assert recovered["transition_history"][0]["timestamp"] == 10.0
    assert pins_path.read_bytes() == before
    assert control.load_control(state_dir, verify_files=True) == recovered


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
    refresh_test_watchdog_mirror(state_dir, now=120.0)
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
            120.0,
        ),
        now=120.0,
    )
    assert again["rollout_generation"] == 1


def test_direct_resume_rejects_missing_production_authorizations_before_scheduler(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    current = control.load_control(state_dir)
    for gate in control.PRODUCTION_AUTHORIZATION_GATES:
        current["readiness"][gate] = {
            "passed": False,
            "evidence": None,
            "attested_at": None,
        }
    control._save_control(state_dir, current, now=49.0)
    with pytest.raises(
        control.ReadinessError, match="production authorization"
    ):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: pytest.fail(
                "scheduler queried without production authorization"
            ),
            submit_runner=lambda _argv: pytest.fail(
                "controller submitted without production authorization"
            ),
            now=50.0,
        )


def test_production_authorization_chain_tamper_blocks_resume(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    record = control.load_control(state_dir)["readiness"][
        "throughput_qualification"
    ]
    manifest = Path(record["chain_manifest"]["path"])
    manifest.chmod(0o644)
    manifest.write_bytes(manifest.read_bytes() + b" ")
    manifest.chmod(0o444)
    with pytest.raises(
        control.ReadinessError, match="chain manifest.*drifted"
    ):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: pytest.fail(
                "scheduler queried after chain tamper"
            ),
            now=50.0,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("rollout_generation", 2),
        ("capacity_generation", 2),
        ("fleet_contract_sha256", "d" * 64),
        ("release_fleet_contract_sha256", "e" * 64),
        ("catalog_id", "f" * 64),
    ),
)
def test_initial_launch_rejects_mismatched_qualification_generation(
    tmp_path,
    field,
    replacement,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir, now=49.0)
    current = control.load_control(state_dir)
    pin = control._load_resume_catalog_pin(
        state_dir, current, target_rollout_generation=1
    )
    control._validate_initial_qualification_generation(
        current,
        resume_catalog_pin=pin,
        target_rollout_generation=1,
    )
    current["readiness"]["throughput_qualification"][
        "qualification_generation"
    ][field] = replacement
    with pytest.raises(
        control.ReadinessError,
        match="initial production launch throughput qualification",
    ):
        control._validate_initial_qualification_generation(
            current,
            resume_catalog_pin=pin,
            target_rollout_generation=1,
        )


def test_initial_resume_rejects_catalog_published_after_qualification(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    complete_drill_marker(state_dir, now=49.0)
    qualified = control.load_control(state_dir)["readiness"][
        "throughput_qualification"
    ]["qualification_generation"]["catalog_id"]
    _seal_control_catalog_endpoint(
        state_dir,
        profile_index=0,
        replica_index=0,
        rollout_generation=1,
        job_id="199999",
    )
    replacement = control.refresh_trusted_generation_catalog(
        state_dir, target_rollout_generation=1, now=49.5
    )
    assert replacement.catalog_id != qualified
    submitted = []
    with pytest.raises(
        control.ReadinessError,
        match="initial production launch throughput qualification",
    ):
        control.resume_control(
            state_dir,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 50.0),
            submit_runner=lambda argv: submitted.append(argv)
            or subprocess.CompletedProcess(argv, 0, "700\n", ""),
            now=50.0,
        )
    assert submitted == []
    assert control.load_control(state_dir)["desired_state"] == "paused"


def test_running_admission_rehashes_production_authorization_artifacts(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    record = control.load_control(state_dir)["readiness"][
        "external_watchdog"
    ]
    marker = Path(record["artifacts"]["watchdog_ready"]["path"])
    marker.chmod(0o644)
    value = json.loads(marker.read_text(encoding="utf-8"))
    value["marker_id"] = "8" * 64
    marker.write_text(json.dumps(value) + "\n", encoding="utf-8")
    marker.chmod(0o444)
    with pytest.raises(
        control.ReadinessError, match="watchdog_ready.*drifted"
    ):
        control.admission_contract_from_state(state_dir)


def test_production_authorization_record_replay_from_other_control_is_rejected(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path / "first")
    make_ready(state_dir)
    complete_drill_marker(state_dir)
    other_state, _ = initialize(tmp_path / "second")
    make_ready(other_state)
    complete_drill_marker(other_state)
    current = control.load_control(state_dir)
    foreign = control.load_control(other_state)["readiness"][
        "throughput_qualification"
    ]
    current["readiness"]["throughput_qualification"] = copy.deepcopy(foreign)
    with pytest.raises(
        control.ReadinessError, match="authorization.*invalid|journal-bound"
    ):
        control._save_control(state_dir, current, now=49.0)


def test_production_authorization_rejects_wrong_chain_verifier_output(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir)
    current = control.load_control(state_dir)
    record = copy.deepcopy(
        current["readiness"]["throughput_qualification"]
    )
    current["readiness"]["throughput_qualification"] = {
        "passed": False,
        "evidence": None,
        "attested_at": None,
    }
    control._save_control(state_dir, current, now=47.0)

    def wrong_chain(argv, _environment, _timeout):
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "passed": True,
                    "chain_id": "8" * 64,
                    "manifest": record["chain_manifest"]["path"],
                }
            ),
            "",
        )

    with pytest.raises(control.ReadinessError, match="exact chain manifest"):
        control.attest_gate(
            state_dir,
            gate="throughput_qualification",
            evidence_path=Path(
                record["artifacts"]["qualification_marker"]["path"]
            ),
            chain_manifest=Path(record["chain_manifest"]["path"]),
            verifier_runner=wrong_chain,
            now=48.0,
        )


def test_production_authorization_two_phase_commit_discards_stale_verification(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    attest_test_production_authorizations(state_dir)
    current = control.load_control(state_dir)
    valid_record = copy.deepcopy(
        current["readiness"]["throughput_qualification"]
    )
    current["readiness"]["throughput_qualification"] = {
        "passed": False,
        "evidence": None,
        "attested_at": None,
    }
    control._save_control(state_dir, current, now=47.0)

    def mutate_during_verification(*_args, **_kwargs):
        changed = control.load_control(state_dir)
        changed["alert_email"] = "changed-during-verification@example.invalid"
        control._save_control(state_dir, changed, now=47.5)
        return copy.deepcopy(valid_record)

    monkeypatch.setattr(
        control,
        "_prepare_production_authorization_record",
        mutate_during_verification,
    )
    with pytest.raises(control.ReadinessError, match="control changed"):
        control.attest_production_authorization(
            state_dir,
            gate="throughput_qualification",
            evidence_path=Path(
                valid_record["artifacts"]["qualification_marker"]["path"]
            ),
            chain_manifest=Path(valid_record["chain_manifest"]["path"]),
            now=48.0,
        )
    assert (
        control.load_control(state_dir)["readiness"][
            "throughput_qualification"
        ]["passed"]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "running_requeue",
        "scheduler_identity",
        "spooled_bytes",
        "spooled_environment",
        "http_model",
        "registry_context",
        "unmapped_terminal",
        "rollout_generation",
    ),
)
def test_fleet_gate_rejects_self_consistently_rehashed_provenance(
    tmp_path, mutation
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)

    def mutate(raw):
        row = raw["replicas"][0]
        if mutation == "running_requeue":
            row["scontrol"]["effective_requeue"] = 1
        elif mutation == "scheduler_identity":
            row["job_name"] = "forged-fleet-name"
            row["scontrol"]["job_name"] = "forged-fleet-name"
        elif mutation == "spooled_bytes":
            forged = "0" * 64
            row["local_script_sha256"] = forged
            row["spooled_script_sha256"] = forged
            row["spooled_script_proof"]["local_sha256"] = forged
            row["spooled_script_proof"]["observed_sha256"] = forged
        elif mutation == "spooled_environment":
            row["spooled_provenance"]["environment_hash"] = "0" * 64
        elif mutation == "http_model":
            row["http"]["expected_model"] = "invented-model"
            row["http"]["model_ids"] = ["invented-model"]
        elif mutation == "registry_context":
            registry_path = Path(row["registry_path"])
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["max_model_len"] += 1
            registry_path.write_text(
                json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8"
            )
            row["registry_sha256"] = _sha(registry_path)
            reference = next(
                item
                for item in raw["referenced_artifacts"]
                if item["name"] == f"registry_{row['replica_id']}"
            )
            reference["sha256"] = row["registry_sha256"]
        elif mutation == "unmapped_terminal":
            raw["ignored_terminal_job_ids"] = ["999999"]
        elif mutation == "rollout_generation":
            raw["rollout_generation"] += 1
            for row in raw["replicas"]:
                row["ledger_generation"] = raw["rollout_generation"]
                parsed = control.fleet_transactions.parse_intent_comment(
                    row["comment"]
                )
                row["comment"] = control.fleet_transactions.intent_comment(
                    pool_id=parsed["pool"],
                    profile=parsed["profile"],
                    replica_id=parsed["replica"],
                    rollout_generation=raw["rollout_generation"],
                    intent_token=parsed["intent"],
                    fleet_sha256=parsed["fleet"],
                )
                row["scontrol"]["comment"] = row["comment"]
        else:  # pragma: no cover - exhaustive parameter contract.
            raise AssertionError(mutation)

    _rewrite_rehashed_fleet_evidence(state_dir, mutate)
    with pytest.raises(control.ReadinessError, match="fleet|resume gates"):
        control.validate_readiness(
            control.load_control(state_dir),
            state_dir=state_dir,
            now=50.0,
        )


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

    refresh_test_watchdog_mirror(state_dir, now=110.0)
    with pytest.raises(control.SchedulerVisibilityPending):
        control.resume_control(
            state_dir,
            scheduler_reader=scheduler,
            submit_runner=lambda _argv: pytest.fail(
                "visibility-grace retry duplicated a controller"
            ),
            now=110.0,
        )
    assert len(submitted) == 2

    visible_jobs.extend(
        control.SchedulerJob(job_id, "controller", "PENDING", token)
        for job_id, token, _ in submitted
    )
    # A durable resume transaction may outlive the fleet evidence's 10-minute launch
    # freshness window while Slurm visibility converges.  The paused->resuming write
    # already proved freshness, so this restart validates the sealed historical gate.
    refresh_test_watchdog_mirror(state_dir, now=1_000.0)
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
    partition_argument = pins["dispatcher_command"].index("--cell-partition")
    assert (
        pins["dispatcher_command"][partition_argument + 1]
        == control.PRODUCTION_CELL_PARTITION
        == "ou_bcs_normal"
    )
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


def test_prepare_pins_rejects_stale_canonical_server_pool(tmp_path):
    expected = make_preparable_pins(tmp_path)
    state_dir = Path(expected["results_root"]) / control.CONTROL_STATE_DIRNAME
    canonical_pool = Path(expected["server_pool_root"])
    stale_registration = canonical_pool / "registrations" / "stale.json"
    stale_registration.parent.mkdir()
    stale_registration.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        control.ImmutablePinError,
        match="server pool must be empty before v1.2 pin publication",
    ):
        control.build_immutable_pins(
            state_dir=state_dir,
            release_bundle_root=Path(expected["release_bundle_root"]),
            hf_home=Path(expected["hf_home"]),
        )


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


def test_v12_release_and_environment_schema_downgrades_are_rejected(tmp_path):
    pins = make_pins(tmp_path)
    assert pins["release_id"] == "sweep-recovery-schema5-v1.2"
    assert control.PRODUCTION_OPERATIONAL_TAG == "sweep-recovery-schema5-v1.2-r11"

    downgraded_pins = copy.deepcopy(pins)
    downgraded_pins["release_id"] = "sweep-recovery-schema5-v1.1"
    with pytest.raises(control.ImmutablePinError, match="downgrade"):
        control.validate_immutable_pins(downgraded_pins, verify_files=False)

    harness_manifest = json.loads(
        Path(pins["harness_environment_manifest_path"]).read_text(encoding="utf-8")
    )
    downgraded_environment = copy.deepcopy(harness_manifest)
    downgraded_environment["schema_version"] = 1
    with pytest.raises(control.ImmutablePinError, match="sealed schema 4"):
        control._validate_schema4_environment_manifest_static(
            downgraded_environment, role="harness"
        )

    provenance_drift = copy.deepcopy(harness_manifest)
    provenance_drift["ownership_policy"]["sha256"] = "b" * 64
    with pytest.raises(control.ImmutablePinError, match="content identity"):
        control._validate_schema4_environment_manifest_static(
            provenance_drift, role="harness"
        )

    integrity_policy_drift = copy.deepcopy(harness_manifest)
    integrity_policy_drift["integrity_normalization_policy"]["sha256"] = "c" * 64
    with pytest.raises(control.ImmutablePinError, match="content identity"):
        control._validate_schema4_environment_manifest_static(
            integrity_policy_drift, role="harness"
        )


def test_release_bundle_rejects_materialization_schema_downgrade(tmp_path):
    pins = make_pins(tmp_path)
    bundle = Path(pins["release_bundle_root"])
    identity_path = bundle / control.RELEASE_IDENTITY_FILENAME
    identity_checksum = bundle / (control.RELEASE_IDENTITY_FILENAME + ".sha256")
    marker_path = bundle / control.RELEASE_COMPLETE_FILENAME

    bundle.chmod(0o755)
    identity_path.chmod(0o644)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["materialization"]["schema_version"] = 2
    identity_path.write_text(
        json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8"
    )
    identity_path.chmod(0o444)
    identity_checksum.chmod(0o644)
    identity_checksum.write_text(
        f"{_sha(identity_path)}  {identity_path.name}\n", encoding="utf-8"
    )
    identity_checksum.chmod(0o444)

    marker_path.chmod(0o644)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for path in (identity_path, identity_checksum):
        marker["artifacts"][path.name] = {
            "sha256": _sha(path),
            "size": path.stat().st_size,
        }
    marker.pop("release_bundle_id")
    marker["release_bundle_id"] = control.sha256_value(marker)
    marker_path.write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path.chmod(0o444)
    bundle.chmod(0o555)
    pins["release_bundle_id"] = marker["release_bundle_id"]

    with pytest.raises(control.ImmutablePinError, match="materialization binding"):
        control.validate_immutable_pins(pins, verify_files=True)


def test_context_readiness_accepts_margin_above_floor_and_rejects_below(
    tmp_path, monkeypatch
):
    _, initialized = initialize(tmp_path)
    monkeypatch.setattr(control, "_validate_context_artifacts", lambda *args, **kw: None)
    metrics = _gate_metrics(initialized, "context_audit")
    metrics["minimum_context_margin_tokens"] = 1_288
    control._validate_gate_metrics(
        initialized,
        "context_audit",
        metrics,
        {},
        now=23.0,
        snapshot_context=control._SnapshotValidationContext(full=True),
    )

    metrics["minimum_context_margin_tokens"] = 1_286
    with pytest.raises(control.ReadinessError, match="at least.*1,287"):
        control._validate_gate_metrics(
            initialized,
            "context_audit",
            metrics,
            {},
            now=23.0,
            snapshot_context=control._SnapshotValidationContext(full=True),
        )


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
            [],
            0,
            "123|job|RUNNING|controller_cpu|normal|comment|cmd|(null)\n",
            "",
        ),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"123|job|FAILED|controller_cpu|normal|comment|{submit}\n"
            "123.batch|batch|COMPLETED|controller_cpu|normal||\n",
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


def test_scheduler_query_preserves_logical_array_ids_with_parent_summary():
    token = "asys-schema5-intent:20260721T000000-array"
    submit = f"sbatch --comment={token} /state/batch-array.sbatch"
    commands = {}

    def runner(argv):
        commands[argv[0]] = list(argv)
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(
                argv,
                0,
                f"321_0|array|RUNNING|protected_client|client_qos|{token}|"
                f"{submit}|(null)\n"
                f"321_1|array|PENDING|protected_client|client_qos|{token}|"
                f"{submit}|(null)\n",
                "",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            f"321|array|PENDING|protected_client|client_qos||{submit}\n"
            f"321_0|array|RUNNING|protected_client|client_qos||{submit}\n"
            f"321_1|array|PENDING|protected_client|client_qos||{submit}\n",
            "",
        )

    snapshot = control.query_scheduler(runner=runner, user="u", now=100.0)

    assert commands["squeue"][-1] == (
        "%i|%j|%T|%P|%q|%k|%o|%E"
    )
    assert commands["sacct"][-1] == (
        "--format=JobID,JobName,State,Partition,QOS,Comment,SubmitLine"
    )
    assert "JobIDRaw" not in commands["sacct"][-1]
    assert [job.job_id for job in snapshot.jobs] == [
        "321",
        "321_0",
        "321_1",
    ]
    assert {
        job.job_id: job.source for job in snapshot.jobs
    } == {
        "321": "sacct",
        "321_0": "squeue",
        "321_1": "squeue",
    }
    assert {
        (job.partition, job.qos) for job in snapshot.jobs
    } == {("protected_client", "client_qos")}
    assert {
        job.job_id.split("_", 1)[0] for job in snapshot.jobs
    } == {"321"}


def test_scheduler_query_rejects_active_sacct_only_job():
    token = "asys-schema5-intent:20260721T000000-crash"
    submit = f"sbatch --comment={token} /state/batch-crash.sbatch"
    outputs = {
        "squeue": subprocess.CompletedProcess([], 0, "", ""),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"654|array|PENDING|protected_client|client_qos||{submit}\n",
            "",
        ),
    }

    with pytest.raises(
        control.SchedulerAmbiguity, match="active sacct-only.*absent"
    ):
        control.query_scheduler(
            runner=lambda argv: outputs[argv[0]], user="u", now=100.0
        )


def test_scheduler_join_recovers_cluster_blank_sacct_comment_from_submit_line():
    token = control.job_token("dispatcher", 1, "intent1")
    submit = f"sbatch --parsable --comment={token} /state/dispatch.sbatch"
    outputs = {
        "squeue": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|RUNNING|controller_cpu|normal|{token}|"
            f"{submit}|(null)\n",
            "",
        ),
        # The production cluster has AccountingStoreFlags=(null), so Comment is blank
        # even while the scheduler-owned SubmitLine retains the exact CLI token.
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|RUNNING|controller_cpu|normal||{submit}\n",
            "",
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
                argv,
                0,
                f"123|controller|PENDING|controller_cpu|normal||{submit}\n",
                "",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            (
                f"123|controller|PENDING|controller_cpu|normal|{token}|{submit}|"
                f"{dependency}(unfulfilled)\n"
            ),
            "",
        )

    snapshot = control.query_scheduler(runner=runner, user="u", now=100.0)
    assert commands["sacct"][-1] == (
        "--format=JobID,JobName,State,Partition,QOS,Comment,SubmitLine"
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
            [],
            0,
            f"123|controller|PENDING|controller_cpu|normal||{submit}\n",
            "",
        ),
        "squeue": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|PENDING|controller_cpu|normal|{token}|"
            f"{submit}|afterany:100\n",
            "",
        ),
    }
    with pytest.raises(control.SchedulerAmbiguity, match="dependency conflict"):
        control.query_scheduler(
            runner=lambda argv: outputs[argv[0]], user="u", now=100.0
        )


@pytest.mark.parametrize(
    ("live_partition", "live_qos"),
    (
        ("other_partition", "client_qos"),
        ("protected_client", "other_qos"),
    ),
)
def test_scheduler_rejects_cross_source_placement_disagreement(
    live_partition, live_qos
):
    token = control.job_token("dispatcher", 1, "intent1")
    submit = f"sbatch --comment={token} /state/dispatch.sbatch"
    outputs = {
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            "123|controller|RUNNING|protected_client|client_qos|"
            f"{token}|{submit}\n",
            "",
        ),
        "squeue": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|RUNNING|{live_partition}|{live_qos}|"
            f"{token}|{submit}|(null)\n",
            "",
        ),
    }
    with pytest.raises(control.SchedulerAmbiguity, match="identity conflict"):
        control.query_scheduler(
            runner=lambda argv: outputs[argv[0]], user="u", now=100.0
        )


def test_scheduler_allows_blank_terminal_accounting_placement_only():
    outputs = {
        "squeue": subprocess.CompletedProcess([], 0, "", ""),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            "123|historical|COMPLETED||||/state/historical.sbatch\n",
            "",
        ),
    }
    snapshot = control.query_scheduler(
        runner=lambda argv: outputs[argv[0]], user="u", now=100.0
    )
    assert [(job.partition, job.qos) for job in snapshot.jobs] == [("", "")]


def test_scheduler_rejects_sacct_comment_submit_line_disagreement():
    token = control.job_token("dispatcher", 1, "intent1")
    wrong = control.job_token("dispatcher", 1, "other")
    outputs = {
        "squeue": subprocess.CompletedProcess([], 0, "", ""),
        "sacct": subprocess.CompletedProcess(
            [],
            0,
            f"123|controller|FAILED|controller_cpu|normal|{wrong}|"
            f"sbatch --comment={token} /x.sbatch\n",
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
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH --export=ALL" not in text
    assert "export PATH=/usr/bin:/bin" in text
    assert "readonly PATH" in text
    assert "export GIT_NO_REPLACE_OBJECTS=1" in text
    assert text.index("#SBATCH --export=NONE") < text.index("set -euo pipefail")
    assert text.index("set -euo pipefail") < text.index("export PATH=/usr/bin:/bin")
    assert text.count("#SBATCH --partition=mit_preemptable\n") == 1
    assert "#SBATCH --partition=mit_normal" not in text
    assert "generation=7;intent=abc123" in text
    assert '--generation 7 --intent-token "abc123"' in text
    assert f'export ASYS_RELEASE_ID="{control.PRODUCTION_RELEASE_ID}"' in text
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
    with pytest.raises(control.ControlError, match="partition|runtime contract"):
        control.validate_controller_generation_sbatch(
            initialized,
            payload=text.replace(
                "#SBATCH --partition=mit_preemptable",
                "#SBATCH --partition=mit_normal",
            ),
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
    assert fleet_text.count("#SBATCH --partition=mit_preemptable\n") == 1
    assert "#SBATCH --partition=mit_normal" not in fleet_text
    assert "#SBATCH --export=NONE" in fleet_text
    assert "export PATH=/usr/bin:/bin" in fleet_text
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


def test_controller_generation_sbatch_recovers_readonly_after_link_crash(
    tmp_path, monkeypatch
):
    state_dir, initialized = initialize(tmp_path)
    target = state_dir / "sbatch" / "dispatch.g000007.abc123.sbatch"
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == target:
            crashed["value"] = True
            raise RuntimeError("death after controller sbatch link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="controller sbatch link"):
            control.render_generation_sbatch(
                state_dir,
                initialized,
                role="dispatcher",
                generation=7,
                intent_token="abc123",
            )

    assert target.is_file()
    assert target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o222 == 0
    before = target.read_bytes()
    replayed = control.render_generation_sbatch(
        state_dir,
        initialized,
        role="dispatcher",
        generation=7,
        intent_token="abc123",
    )
    assert replayed == target
    assert target.read_bytes() == before


def test_controller_live_placement_is_exact_and_non_cell_partition():
    token = control.job_token("dispatcher", 7, "abc123")

    def runner(argv):
        assert argv == ["scontrol", "show", "job", "-o", "123"]
        return subprocess.CompletedProcess(
            argv,
            0,
            (
                "JobId=123 JobName=controller JobState=RUNNING "
                "Partition=mit_preemptable Requeue=0 "
                f"Comment={token}\n"
            ),
            "",
        )

    observed = control.validate_live_controller_placement(
        job_id="123",
        role="dispatcher",
        generation=7,
        intent_token="abc123",
        runner=runner,
    )
    assert observed["Partition"] == "mit_preemptable"

    def wrong_partition(argv):
        process = runner(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            process.stdout.replace(
                "Partition=mit_preemptable", "Partition=mit_normal"
            ),
            "",
        )

    with pytest.raises(control.ControllerFenced, match="placement/provenance"):
        control.validate_live_controller_placement(
            job_id="123",
            role="dispatcher",
            generation=7,
            intent_token="abc123",
            runner=wrong_partition,
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
    transport = initialized["immutable"]["transport_uncertainty_binding"]
    scheduler_binding = control.fleet_scheduler_policy_binding(
        control.load_control(state_dir), require_attested=True
    )
    effective_fleet = control.effective_fleet_contract_binding(
        control.load_control(state_dir), verify_files=True
    )
    assert scheduler_binding is not None
    assert environment == {
            "ASYS_RELEASE_ID": control.PRODUCTION_RELEASE_ID,
            "ASYS_RELEASE_GIT_COMMIT": initialized["immutable"][
                "git_commit"
            ],
            "ASYS_PROTECTED_CAPACITY_MARKER": initialized["immutable"][
                "protected_capacity_marker_path"
            ],
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": initialized[
                "immutable"
            ]["protected_capacity_marker_sha256"],
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": initialized["immutable"][
                "protected_capacity_marker_id"
            ],
        "ASYS_TRANSPORT_CENSOR_PROTOCOL_VERSION": str(
            transport["transport_censor_protocol_version"]
        ),
        "ASYS_TRANSPORT_CENSOR_PROTOCOL_HASH": transport[
            "transport_censor_protocol_hash"
        ],
        "ASYS_TRANSPORT_UNCERTAINTY_BINDING_SHA256": initialized[
            "immutable"
        ]["transport_uncertainty_binding_sha256"],
        "ASYS_CHECKPOINT_SCHEMA_VERSION": str(
            transport["checkpoint_schema_version"]
        ),
        "ASYS_ARTIFACT_SCHEMA_VERSION": str(
            transport["artifact_schema_version"]
        ),
        "ASYS_SELF_CONSISTENCY_PROTOCOL_VERSION": str(
            transport["self_consistency_protocol_version"]
        ),
        "ASYS_SELF_CONSISTENCY_PROTOCOL_HASH": transport[
            "self_consistency_protocol_hash"
        ],
        "ASYS_SCHEDULER_POLICY_CONTRACT_ID": scheduler_binding[
            "scheduler_safety_policy_contract_id"
        ],
        "ASYS_SCHEDULER_SAFETY_EVIDENCE_ID": scheduler_binding[
            "scheduler_safety_evidence_id"
        ],
        "ASYS_SCHEDULER_SAFETY_POLICY_ID": scheduler_binding[
            "scheduler_safety_policy_id"
        ],
        "ASYS_MODEL_CONTRACT_SHA256": initialized["immutable"]["model_contract_sha256"],
            "ASYS_FLEET_CONTRACT_SHA256": effective_fleet["sha256"],
            "ASYS_FLEET_CONTRACT_PATH": effective_fleet["path"],
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": initialized["immutable"][
                "fleet_contract_sha256"
            ],
            "ASYS_CAPACITY_GENERATION": "1",
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
#SBATCH --partition=ou_bcs_normal
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --array=0-23
set -euo pipefail
umask 027
{control._trusted_shell_prelude()}\
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
    with pytest.raises(control.ControlError, match="missing.*export=NONE"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace("#SBATCH --export=NONE\n", ""),
            task_count=24,
        )
    with pytest.raises(control.ControlError, match="non-preemptible"):
        control.validate_production_batch_sbatch(
            initialized,
            payload=payload.replace(
                "#SBATCH --partition=ou_bcs_normal",
                "#SBATCH --partition=mit_preemptable",
            ),
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
    _install_exact_cell_transaction(
        state_dir,
        batch_id="stale-cache",
        base_job_id="900",
        task_count=1,
        intent_state="submitted",
        record_state="active",
        captured_at=1.0,
    )
    report = control.live_status(
        state_dir, snapshot=control.SchedulerSnapshot((), 100.0)
    )
    assert report["scheduler"]["active_cell_tasks"] == 0
    assert report["dispatcher_cache"]["cached_job_records"] == 1
    assert report["dispatcher_cache"]["cached_state_only"] is True


def test_controller_health_and_chain_repair_reject_same_id_wrong_intent_token(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, jobs = resume_ready(state_dir, now=50.0)
    dispatcher = running["controllers"]["dispatcher"]["active"]
    fleet = running["controllers"]["fleet_supervisor"]["active"]
    wrong_token = control.job_token(
        "dispatcher",
        int(dispatcher["generation"]) + 100,
        "wrong-intent",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                str(dispatcher["job_id"]),
                "dispatcher",
                "RUNNING",
                wrong_token,
            ),
            control.SchedulerJob(
                str(fleet["job_id"]),
                "fleet",
                "RUNNING",
                str(fleet["job_token"]),
            ),
        ),
        60.0,
    )
    report = control.live_status(state_dir, snapshot=snapshot, now=60.0)
    dispatcher_status = report["controllers"]["dispatcher"]
    assert dispatcher_status["live_active"] is False
    assert dispatcher_status["active_identity_mismatch"] is True
    assert report["healthy"] is False
    reconciliation = control.build_reconciliation_report(
        running, snapshot, all_jobs=True, no_admit=True
    )
    assert reconciliation["passed"] is False
    assert any(
        "token mismatches" in error for error in reconciliation["errors"]
    )
    assert (
        control._role_has_live_chain(
            running, "dispatcher", snapshot
        )
        is False
    )
    with pytest.raises(control.SchedulerAmbiguity, match="token mismatch"):
        control.submit_controller_intent(
            state_dir,
            role="dispatcher",
            target="active",
            dependency_job_id=None,
            scheduler=snapshot,
            submit_runner=lambda _argv: pytest.fail(
                "wrong-token allocation must not suppress or cross sbatch"
            ),
            now=60.0,
        )


def test_wrong_token_same_successor_id_is_not_considered_live(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, _jobs = resume_ready(state_dir, now=50.0)
    active = running["controllers"]["dispatcher"]["active"]
    successor = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id=str(active["job_id"]),
        scheduler=control.SchedulerSnapshot((), 51.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "801\n", ""
        ),
        now=51.0,
    )
    wrong = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                str(successor["job_id"]),
                "successor",
                "PENDING",
                control.job_token(
                    "dispatcher",
                    int(successor["generation"]) + 1,
                    "wrong-successor-intent",
                ),
            ),
        ),
        52.0,
    )
    report = control.live_status(state_dir, snapshot=wrong, now=52.0)
    row = report["controllers"]["dispatcher"]
    assert row["successor_live"] is False
    assert row["successor_identity_mismatch"] is True
    with pytest.raises(control.SchedulerAmbiguity, match="token mismatch"):
        control._ensure_own_successor(
            state_dir,
            role="dispatcher",
            job_id=str(active["job_id"]),
            snapshot=wrong,
        )


def test_own_successor_requires_exact_spool_receipt_and_afterany_parent_before_work(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    running, _jobs = resume_ready(state_dir, now=50.0)
    active = running["controllers"]["dispatcher"]["active"]
    successor = control.submit_controller_intent(
        state_dir,
        role="dispatcher",
        target="successor",
        dependency_job_id=str(active["job_id"]),
        scheduler=control.SchedulerSnapshot((), 51.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "801\n", ""
        ),
        now=51.0,
    )
    exact = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                str(successor["job_id"]),
                "asys-s5-dispatch",
                "PENDING",
                str(successor["job_token"]),
                f"sbatch {successor['sbatch_path']}",
                "squeue",
                f"afterany:{active['job_id']}(unfulfilled)",
            ),
        ),
        52.0,
    )
    assert (
        control._ensure_own_successor(
            state_dir,
            role="dispatcher",
            job_id=str(active["job_id"]),
            snapshot=exact,
        )["job_id"]
        == "801"
    )

    wrong_parent = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                str(successor["job_id"]),
                "asys-s5-dispatch",
                "PENDING",
                str(successor["job_token"]),
                f"sbatch {successor['sbatch_path']}",
                "squeue",
                "afterany:999(unfulfilled)",
            ),
        ),
        53.0,
    )
    with pytest.raises(control.SchedulerAmbiguity, match="afterany parent"):
        control._ensure_own_successor(
            state_dir,
            role="dispatcher",
            job_id=str(active["job_id"]),
            snapshot=wrong_parent,
        )

    misleading_display_command = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                str(successor["job_id"]),
                "asys-s5-dispatch",
                "PENDING",
                str(successor["job_token"]),
                "sbatch /tmp/foreign.sbatch",
                "squeue",
                f"afterany:{active['job_id']}(unfulfilled)",
            ),
        ),
        54.0,
    )
    # A stdin-submitted script is authenticated by independently captured Slurm
    # spool bytes. Slurm's displayed Command/SubmitLine is not a byte authority.
    assert (
        control._ensure_own_successor(
            state_dir,
            role="dispatcher",
            job_id=str(active["job_id"]),
            snapshot=misleading_display_command,
        )["job_id"]
        == "801"
    )
    receipt_path = Path(successor["spooled_receipt_path"])
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["spooled_sbatch_sha256"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path.chmod(0o444)
    with pytest.raises(control.SchedulerAmbiguity, match="spool"):
        control._ensure_own_successor(
            state_dir,
            role="dispatcher",
            job_id=str(active["job_id"]),
            snapshot=misleading_display_command,
        )


def test_supervisor_source_never_launches_managed_plane_before_verified_successor():
    source = Path(control.__file__).read_text(encoding="utf-8")
    successor_gate = source.index(
        "if verified_successor is not None:",
        source.index("def supervise("),
    )
    launch = source.index(
        "child = subprocess.Popen(",
        source.index("def supervise("),
    )
    assert successor_gate < launch
    between = source[successor_gate:launch]
    assert "validate_live_controller_placement(" in source[
        source.index("def supervise("):successor_gate
    ]
    assert "break" in between


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


def test_monitor_owner_retries_email_when_monitor_cannot_start(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)

    def fail_mail(argv, _body):
        return subprocess.CompletedProcess(argv, 1, "", "mail unavailable")

    pending = control.record_alert(
        state_dir,
        kind="monitor-liveness-test",
        severity="warning",
        message="retry independently of monitor execution",
        dedupe_key="test:monitor-independent-email",
        send_email=True,
        mail_runner=fail_mail,
        now=100.0,
    )
    assert pending["email"]["next_retry_timestamp"] == 160.0

    original_record_alert = control.record_alert

    def no_new_mail(*args, **kwargs):
        kwargs["send_email"] = False
        return original_record_alert(*args, **kwargs)

    monkeypatch.setattr(control, "record_alert", no_new_mail)
    calls: list[str] = []

    def succeed_mail(argv, _body):
        calls.append(str(argv[-1]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def cannot_launch(*_args, **_kwargs):
        raise OSError("simulated monitor exec failure")

    control._service_schema5_monitors(
        state_dir,
        processes={},
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=160.0,
        popen_factory=cannot_launch,
        alert_mail_runner=succeed_mail,
    )

    persisted = next(
        alert
        for alert in control.load_control(state_dir)["alerts"]
        if alert["alert_id"] == pending["alert_id"]
    )
    assert calls == ["mabdel03@mit.edu"]
    assert persisted["email"]["attempt_count"] == 2
    assert persisted["email"]["delivered"] is True
    assert persisted["email"]["next_retry_timestamp"] is None
    assert any(
        alert["dedupe_key"].startswith("monitor-supervisor:")
        for alert in control.load_control(state_dir)["alerts"]
    )


def test_throughput_epochs_survive_successors_and_close_on_pause(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    first = control.record_successful_poll(
        state_dir, useful_qids=100, fleet_generation="fleet-a", now=60.0
    )
    second = control.record_successful_poll(
        state_dir, useful_qids=150, fleet_generation="fleet-a", now=70.0
    )
    assert len(first["throughput_epochs"]) == 1
    assert len(second["throughput_epochs"]) == 1
    assert len(second["throughput_epochs"][0]["samples"]) == 2
    changed = control.record_successful_poll(
        state_dir, useful_qids=151, fleet_generation="fleet-b", now=80.0
    )
    assert len(changed["throughput_epochs"]) == 2
    assert changed["throughput_epochs"][0]["close_reason"] == "material_fleet_change"
    paused = control.pause_control(state_dir, drain=True, now=90.0)
    assert paused["throughput_epochs"][-1]["close_reason"] == "pause"
    with pytest.raises(control.ControlError, match="while paused"):
        control.record_successful_poll(
            state_dir, useful_qids=200, fleet_generation="fleet-b", now=100.0
        )


def _ramp_evidence(
    state_dir: Path,
    *,
    committed_at: float,
    cadence: str,
    fleet_generation: str,
    qids: dict[str, int] | None = None,
    useful_qids: dict[str, int] | None = None,
    health_clean: bool = True,
    semantic_clean: bool = True,
    critical_keys: tuple[str, ...] = (),
    blocking_keys: tuple[str, ...] = (),
) -> tuple[Path, str]:
    state = control.load_control(state_dir)
    captured_at = committed_at - 1.0
    effective_useful_qids = qids if useful_qids is None else useful_qids
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
        "run_useful_qids": (
            copy.deepcopy(effective_useful_qids)
            if cadence != "health"
            else None
        ),
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
        assert effective_useful_qids is not None
        report["semantic"] = {
            "outcomes": {
                "validated_qids": sum(qids.values()),
                "useful_qids": sum(effective_useful_qids.values()),
            },
            "runs": {
                run_id: {
                    "outcomes": {
                        "validated_qids": qids[run_id],
                        "useful_qids": effective_useful_qids[run_id],
                    }
                }
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
    assert control.ADMISSION_RAMP_STALL_SECONDS == {
        24: 28_800.0,
        96: 46_800.0,
        192: 68_400.0,
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
        useful_qids=sum(qids.values()),
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

    # The verified non-preemptible capacity stage is continuously covered by
    # observations no more than 10 minutes apart; promotion itself is a semantic
    # observation and requires every production run to progress.
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
            useful_qids=sum(qids.values()),
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
    assert [
        row["to_ceiling"] for row in final["admission_ramp"]["promotions"]
    ] == [96, 192, 384]
    assert all(
        set(row["final_run_validated_qids"]) == set(control.REQUIRED_RUNS)
        and set(row["final_run_useful_qids"]) == set(control.REQUIRED_RUNS)
        and row["observations"]
        for row in final["admission_ramp"]["promotions"]
    )
    assert final["admission_ramp"]["promotions"][0]["all_runs_progressed"] is True
    assert {
        row["client_capacity_authorization_sha256"]
        for row in final["admission_ramp"]["promotions"]
    } == {
        final["immutable"]["protected_capacity_marker_sha256"]
    }
    assert final["admission"]["maximum_cell_tasks"] == 384

    # A new material fleet identity closes the throughput epoch and immediately
    # returns admission to the initial safety stage.
    changed = control.record_successful_poll(
        state_dir,
        useful_qids=sum(qids.values()),
        fleet_generation="fleet-b",
        now=68_600.0,
    )
    assert changed["admission"]["current_ceiling"] == 24
    assert changed["admission_ramp"]["window"] is None
    assert changed["throughput_epochs"][-2]["close_reason"] == "material_fleet_change"


def test_admission_ramp_does_not_promote_on_transport_only_cardinality_progress(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    validated = {run_id: 10 for run_id in control.REQUIRED_RUNS}
    useful = dict(validated)
    control.record_successful_poll(
        state_dir,
        useful_qids=sum(useful.values()),
        fleet_generation="fleet-a",
        now=90.0,
    )
    _commit_ramp_evidence(
        state_dir,
        committed_at=100.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=validated,
        useful_qids=useful,
    )
    for timestamp in range(700, 3_700, 600):
        _commit_ramp_evidence(
            state_dir,
            committed_at=float(timestamp),
            cadence="health",
            fleet_generation="fleet-a",
        )
    transport_only = {
        run_id: value + 1 for run_id, value in validated.items()
    }
    control.record_successful_poll(
        state_dir,
        useful_qids=sum(useful.values()),
        fleet_generation="fleet-a",
        now=3_699.5,
    )
    observed = _commit_ramp_evidence(
        state_dir,
        committed_at=3_700.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=transport_only,
        useful_qids=useful,
    )
    assert observed["admission"]["current_ceiling"] == 24
    assert observed["admission_ramp"]["promotions"] == []
    assert observed["admission_ramp"]["last_action"]["action"] == (
        "window_advanced"
    )
    assert observed["admission_ramp"]["last_action"][
        "all_runs_progressed"
    ] is False
    assert observed["admission_ramp"]["window"][
        "latest_run_validated_qids"
    ] == transport_only
    assert observed["admission_ramp"]["window"][
        "latest_run_useful_qids"
    ] == useful


def test_admission_ramp_uses_prelaunch_protected_capacity_through_384(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    staged = control.load_control(state_dir)
    staged["admission"]["current_ceiling"] = 96
    staged["admission_ramp"]["current_ceiling"] = 96
    staged["admission_ramp"]["window"] = None
    control._save_control(state_dir, staged, now=80.0)

    qids = {run_id: 10 for run_id in control.REQUIRED_RUNS}
    control.record_successful_poll(
        state_dir,
        useful_qids=sum(qids.values()),
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
    for timestamp in range(700, 21_700, 600):
        _commit_ramp_evidence(
            state_dir,
            committed_at=float(timestamp),
            cadence="health",
            fleet_generation="fleet-a",
        )
    promoted = _commit_ramp_evidence(
        state_dir,
        committed_at=21_700.0,
        cadence="semantic",
        fleet_generation="fleet-a",
        qids=qids,
    )

    assert promoted["admission"]["current_ceiling"] == 192
    assert [
        row["to_ceiling"]
        for row in promoted["admission_ramp"]["promotions"]
    ] == [192]
    assert promoted["admission_ramp"]["capacity_gate"] is None
    assert promoted["admission_safety_hold"]["active"] is False
    assert promoted["admission_ramp"]["promotions"][0][
        "client_capacity_authorization_sha256"
    ] == promoted["immutable"]["protected_capacity_marker_sha256"]
    assert not any(
        row["event"] == "client_capacity_transition_required"
        for row in promoted["transition_history"]
    )


def test_client_capacity_authorization_rehashes_scheduler_canary_and_readiness(
    tmp_path, monkeypatch
):
    production_state_dir, initialized = initialize(tmp_path / "production")
    state_dir = tmp_path / "state"
    evidence_root = (
        state_dir
        / "capacity-transitions"
        / "capacity-g000001-to-g000002"
        / "client-capacity"
    )
    evidence_root.mkdir(parents=True)
    generation = 2
    target = 192
    rollout_generation = 7
    transition_id = "capacity-g000001-to-g000002"
    (
        effective_fleet_path,
        contract_sha,
        capacity_authority,
    ) = _capacity_contract_fixture(tmp_path, initialized, monkeypatch)
    effective_fleet_payload = json.loads(
        effective_fleet_path.read_text(encoding="utf-8")
    )
    effective_fleet_marker = effective_fleet_path.with_suffix(".sha256")
    profile_replicas = {
        row["serving_profile"]: len(row["replicas"])
        for row in effective_fleet_payload["profiles"]
    }
    immutable_sha = initialized["immutable_sha256"]
    placement = {
        "capacity_generation": generation,
        "partition": "new_nonpreemptible_clients",
        "preempt_mode": "OFF",
        "nonpreemptible_client_slots": target,
        "cpu_limit": target,
        "memory_limit_mib": target * 4096,
        "max_submit_jobs": target + 64,
        "qos_limit": 448,
        "reserve_jobs": 64,
    }
    scheduler_path = evidence_root / "scheduler.json"
    canary_path = evidence_root / "canary.json"
    readiness_path = evidence_root / "readiness.json"
    submission_root = (
        evidence_root / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
    )
    submission_root.mkdir()
    canary_job_id = "765432"
    intent_token = "c" * 32
    scheduler_user = "test-user"
    sbatch_path = evidence_root / control.CLIENT_CAPACITY_CANARY_SBATCH
    sbatch_payload = control._client_capacity_canary_sbatch(
        partition=placement["partition"],
        target_ceiling=target,
        capacity_generation=generation,
        intent_token=intent_token,
        output_path=evidence_root / "client_capacity_canary_%j.out",
    )
    sbatch_path.write_text(sbatch_payload, encoding="utf-8")
    sbatch_path.chmod(0o444)
    sbatch_sha256 = _sha(sbatch_path)
    job_name = control._client_capacity_canary_job_name(
        capacity_generation=generation,
        target_ceiling=target,
    )
    job_comment = control._client_capacity_canary_comment(
        capacity_generation=generation,
        target_ceiling=target,
        intent_token=intent_token,
    )
    submit_argv = control._client_capacity_submission_argv(
        sbatch_path=sbatch_path,
        job_comment=job_comment,
    )
    submit_line = shlex.join(submit_argv)
    submission_intent_path = (
        submission_root / control.CLIENT_CAPACITY_SUBMISSION_INTENT
    )
    submission_intent = {
        "schema_version": 1,
        "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        "scheduler_user": scheduler_user,
        "control_immutable_sha256": immutable_sha,
        "capacity_generation": generation,
        "rollout_generation": rollout_generation,
        "transition_id": transition_id,
        "partition": placement["partition"],
        "target_ceiling": target,
        "intent_token": intent_token,
        "job_name": job_name,
        "job_comment": job_comment,
        "sbatch_path": str(sbatch_path.resolve()),
        "sbatch_sha256": sbatch_sha256,
        "submit_argv": submit_argv,
        "created_at": control.utc_timestamp(100.0),
        "created_timestamp": 100.0,
    }
    control._write_sealed_builder_json(
        submission_intent_path,
        submission_intent,
        description="client-capacity submission intent fixture",
    )
    attempt_root = (
        submission_root
        / control.CLIENT_CAPACITY_SUBMISSION_ATTEMPTS_DIRECTORY
        / "000001"
    )
    attempt_root.mkdir(parents=True)
    attempt_intent_path = attempt_root / "ATTEMPT_INTENT.json"
    attempt_intent = {
        "schema_version": 1,
        "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        "attempt": 1,
        "submission_intent_sha256": _sha(submission_intent_path),
        "control_immutable_sha256": immutable_sha,
        "capacity_generation": generation,
        "rollout_generation": rollout_generation,
        "transition_id": transition_id,
        "partition": placement["partition"],
        "target_ceiling": target,
        "intent_token": intent_token,
        "job_name": job_name,
        "job_comment": job_comment,
        "sbatch_path": str(sbatch_path.resolve()),
        "sbatch_sha256": sbatch_sha256,
        "submit_argv": submit_argv,
        "retry_authorization": None,
        "created_at": control.utc_timestamp(100.0),
        "created_timestamp": 100.0,
    }
    control._write_sealed_builder_json(
        attempt_intent_path,
        attempt_intent,
        description="client-capacity attempt intent fixture",
    )
    attempt_result_path = attempt_root / "SBATCH_RESULT.json"
    attempt_result = {
        "schema_version": 1,
        "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        "attempt": 1,
        "attempt_intent_sha256": _sha(attempt_intent_path),
        "returncode": 0,
        "stdout": f"{canary_job_id}\n",
        "stderr": "",
        "stdout_sha256": hashlib.sha256(
            f"{canary_job_id}\n".encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "job_id": canary_job_id,
        "outcome": "accepted_stdout",
        "completed_at": control.utc_timestamp(100.0),
        "completed_timestamp": 100.0,
    }
    control._write_sealed_builder_json(
        attempt_result_path,
        attempt_result,
        description="client-capacity sbatch result fixture",
    )
    observation_root = attempt_root / "observations" / "000001"
    observation_root.mkdir(parents=True)
    observation_references = {}
    for source, filename in (
        ("squeue", "SQUEUE_OBSERVATION.json"),
        ("sacct", "SACCT_OBSERVATION.json"),
    ):
        source_path = observation_root / filename
        source_state = "RUNNING" if source == "squeue" else "COMPLETED"
        source_stdout = (
            f"{canary_job_id}|{job_name}|{source_state}|"
            f"{placement['partition']}|normal|{job_comment}|"
            f"{submit_line}"
            f"{'|' if source == 'squeue' else ''}\n"
        )
        source_argv = (
            [
                "squeue",
                "-u",
                scheduler_user,
                "-h",
                "-r",
                "-o",
                "%i|%j|%T|%P|%q|%k|%o|%E",
                "--name",
                job_name,
            ]
            if source == "squeue"
            else [
                "sacct",
                "-u",
                scheduler_user,
                "-n",
                "-P",
                "-S",
                time.strftime(
                    "%Y-%m-%d",
                    time.localtime(101.0 - 7 * 86_400),
                ),
                (
                    "--format=JobID,JobName,State,Partition,QOS,"
                    "Comment,SubmitLine"
                ),
                "--name",
                job_name,
            ]
        )
        source_payload = {
            "schema_version": 1,
            "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
            "attempt": 1,
            "observation": 1,
            "source": source,
            "complete": True,
            "captured_at": control.utc_timestamp(101.0),
            "captured_timestamp": 101.0,
            "argv": source_argv,
            "returncode": 0,
            "stdout": source_stdout,
            "stderr": "",
            "stdout_sha256": hashlib.sha256(
                source_stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "candidate_job_ids": [canary_job_id],
        }
        control._write_sealed_builder_json(
            source_path,
            source_payload,
            description=f"client-capacity {source} observation fixture",
        )
        observation_references[source] = {
            "path": str(source_path.resolve()),
            "sha256": _sha(source_path),
        }
    observation_complete_path = (
        observation_root / "OBSERVATION_COMPLETE.json"
    )
    observation_complete = {
        "schema_version": 1,
        "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        "attempt": 1,
        "observation": 1,
        "complete": True,
        "captured_at": control.utc_timestamp(101.0),
        "captured_timestamp": 101.0,
        "candidate_job_ids": [canary_job_id],
        "candidate_sources": ["sacct", "squeue"],
        "squeue": observation_references["squeue"],
        "sacct": observation_references["sacct"],
    }
    control._write_sealed_builder_json(
        observation_complete_path,
        observation_complete,
        description="client-capacity observation completion fixture",
    )
    accepted_submission_path = (
        submission_root / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    )
    accepted_submission = {
        "schema_version": 1,
        "protocol": control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        "passed": True,
        "job_id": canary_job_id,
        "job_name": job_name,
        "job_comment": job_comment,
        "scheduler_user": scheduler_user,
        "control_immutable_sha256": immutable_sha,
        "capacity_generation": generation,
        "rollout_generation": rollout_generation,
        "transition_id": transition_id,
        "partition": placement["partition"],
        "target_ceiling": target,
        "intent_token": intent_token,
        "sbatch_path": str(sbatch_path.resolve()),
        "sbatch_sha256": sbatch_sha256,
        "submit_argv": submit_argv,
        "submission_intent": {
            "path": str(submission_intent_path.resolve()),
            "sha256": _sha(submission_intent_path),
        },
        "attempt": {
            "ordinal": 1,
            "intent_path": str(attempt_intent_path.resolve()),
            "intent_sha256": _sha(attempt_intent_path),
            "result_path": str(attempt_result_path.resolve()),
            "result_sha256": _sha(attempt_result_path),
        },
        "scheduler_observation": {
            "path": str(observation_complete_path.resolve()),
            "sha256": _sha(observation_complete_path),
        },
        "scheduler_sources": ["sacct", "squeue"],
        "accepted_at": control.utc_timestamp(101.0),
        "accepted_timestamp": 101.0,
    }
    control._write_sealed_builder_json(
        accepted_submission_path,
        accepted_submission,
        description="client-capacity accepted submission fixture",
    )

    def scheduler_runner(argv, *, timeout):
        del timeout
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            stdout = (
                "KillWait                = 30 sec\n"
                "PreemptMode             = REQUEUE\n"
                "PreemptType             = preempt/partition_prio\n"
            )
        elif args == [
            "scontrol",
            "show",
            "partition",
            placement["partition"],
            "-o",
        ]:
            stdout = (
                f"PartitionName={placement['partition']} GraceTime=0 "
                "MaxTime=12:00:00 PreemptMode=OFF State=UP TotalNodes=50\n"
            )
        elif args == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            placement["partition"],
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ]:
            stdout = (
                f"{placement['partition']}|cpu={target},"
                f"mem={target * 4}G|{target + 64}\n"
            )
        else:  # pragma: no cover - fixture boundary
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    raw_scheduler_contract = (
        scheduler_safety.capture_observed_client_capacity_contract(
            partition=placement["partition"],
            runner=scheduler_runner,
            captured_timestamp=101.0,
        )
    )
    control._atomic_write_json(
        scheduler_path,
        {
            "passed": True,
            "client_placement": placement,
            "raw_client_capacity_contract": raw_scheduler_contract,
        },
    )
    control._atomic_write_json(
        canary_path,
        {
            "passed": True,
            "capacity_generation": generation,
            "authorized_cell_ceiling": target,
            "partition": placement["partition"],
            "preempt_mode": "OFF",
            "no_requeue": True,
            "cell_cpus": 1,
            "cell_memory_mib": 4096,
            "job_id": canary_job_id,
            "sbatch_path": str(sbatch_path.resolve()),
            "sbatch_sha256": sbatch_sha256,
            "sacct_argv": [
                "sacct",
                "-X",
                "-nP",
                "-j",
                canary_job_id,
                "--format=JobIDRaw,State,Partition,ReqCPUS,ReqMem,ExitCode,SubmitLine",
            ],
            "sacct_raw_output": (
                f"{canary_job_id}|COMPLETED|{placement['partition']}|"
                f"1|4G|0:0|{submit_line}\n"
            ),
            "sacct_raw_output_sha256": hashlib.sha256(
                (
                    f"{canary_job_id}|COMPLETED|{placement['partition']}|"
                    f"1|4G|0:0|{submit_line}\n"
                ).encode("utf-8")
            ).hexdigest(),
            "submit_line": submit_line,
        },
    )
    gate_paths = {}
    for gate in ("fleet", "smoke_runs", "scheduler_reconciliation"):
        gate_path = evidence_root / f"{gate}_gate.json"
        control._atomic_write_json(
            gate_path, {"gate": gate, "passed": True}
        )
        gate_path.chmod(0o444)
        gate_paths[gate] = gate_path
    readiness_payload = {
        "passed": True,
        "capacity_generation": generation,
        "scheduler_reconciliation_passed": True,
        "fleet_readiness_passed": True,
        "client_canary_passed": True,
        "control_immutable_sha256": immutable_sha,
        "rollout_generation": rollout_generation,
        "transition_id": transition_id,
        "fleet_evidence": str(gate_paths["fleet"].resolve()),
        "fleet_evidence_sha256": _sha(gate_paths["fleet"]),
        "smoke_evidence": str(gate_paths["smoke_runs"].resolve()),
        "smoke_evidence_sha256": _sha(gate_paths["smoke_runs"]),
        "scheduler_reconciliation_evidence": str(
            gate_paths["scheduler_reconciliation"].resolve()
        ),
        "scheduler_reconciliation_evidence_sha256": _sha(
            gate_paths["scheduler_reconciliation"]
        ),
    }
    control._atomic_write_json(readiness_path, readiness_payload)
    for path in (scheduler_path, canary_path, readiness_path):
        path.chmod(0o444)
    authorization_path = evidence_root / "CLIENT_CAPACITY_COMPLETE.json"
    authorization = {
        "schema_version": control.CLIENT_CAPACITY_AUTHORIZATION_SCHEMA_VERSION,
        "protocol": control.CLIENT_CAPACITY_AUTHORIZATION_PROTOCOL,
        "passed": True,
        "control_immutable_sha256": immutable_sha,
        "capacity_generation": generation,
        "capacity_contract_sha256": contract_sha,
        "authorized_cell_ceiling": target,
        "partition": placement["partition"],
        "preempt_mode": "OFF",
        "nonpreemptible_client_slots": target,
        "cpu_limit": target,
        "memory_limit_mib": target * 4096,
        "max_submit_jobs": target + 64,
        "qos_limit": 448,
        "reserve_jobs": 64,
        "cell_cpus": 1,
        "cell_memory_mib": 4096,
        "scheduler_evidence": {
            "path": str(scheduler_path.resolve()),
            "sha256": _sha(scheduler_path),
        },
        "canary_evidence": {
            "path": str(canary_path.resolve()),
            "sha256": _sha(canary_path),
        },
        "accepted_submission_evidence": {
            "path": str(accepted_submission_path.resolve()),
            "sha256": _sha(accepted_submission_path),
        },
        "readiness_evidence": {
            "path": str(readiness_path.resolve()),
            "sha256": _sha(readiness_path),
        },
        "captured_timestamp": 101.0,
    }
    control._atomic_write_json(authorization_path, authorization)
    authorization_path.chmod(0o444)
    state = {
        "immutable_sha256": immutable_sha,
        "rollout_generation": rollout_generation,
        "capacity": {
            "current_generation": generation,
            "current_contract": {"sha256": contract_sha},
            "active_transition": None,
            "history": [
                {
                    "transition_id": transition_id,
                    "phase": "completed",
                    "from_generation": 1,
                    "to_generation": generation,
                    # The immutable authorization was captured after the fleet and
                    # readiness boundary but before transition completion.  It must
                    # remain valid after the later completion timestamp is appended.
                    "fleet_launch_completed_timestamp": 100.0,
                    "completed_timestamp": 200.0,
                }
            ],
        },
        "readiness": {
            gate: {
                "passed": True,
                "capacity_generation": generation,
                "evidence": str(path.resolve()),
                "sha256": _sha(path),
            }
            for gate, path in gate_paths.items()
        },
    }
    record = control._read_client_capacity_authorization(
        state_dir,
        state,
        authorization_path,
        _sha(authorization_path),
        attested_timestamp=102.0,
    )
    assert record["authorized_cell_ceiling"] == 192
    assert record["capacity_generation"] == 2
    assert record["preempt_mode"] == "OFF"
    assert record["scheduler_evidence_sha256"] == _sha(scheduler_path)

    def reseal(path: Path, payload: dict) -> None:
        control._atomic_write_json(path, payload)
        path.chmod(0o444)

    # A self-consistent but stale rollout assertion cannot authorize the current
    # control generation.
    stale_readiness = {**readiness_payload, "rollout_generation": 6}
    reseal(readiness_path, stale_readiness)
    authorization["readiness_evidence"]["sha256"] = _sha(readiness_path)
    reseal(authorization_path, authorization)
    with pytest.raises(control.ControlError, match="current control, rollout"):
        control._read_client_capacity_authorization(
            state_dir,
            state,
            authorization_path,
            _sha(authorization_path),
            attested_timestamp=102.0,
        )

    # Nor can a forged, correctly hashed artifact replace the exact gate currently
    # recorded by the control plane.
    forged_fleet = evidence_root / "forged_fleet_gate.json"
    reseal(forged_fleet, {"gate": "fleet", "passed": True})
    forged_readiness = {
        **readiness_payload,
        "fleet_evidence": str(forged_fleet.resolve()),
        "fleet_evidence_sha256": _sha(forged_fleet),
    }
    reseal(readiness_path, forged_readiness)
    authorization["readiness_evidence"]["sha256"] = _sha(readiness_path)
    reseal(authorization_path, authorization)
    with pytest.raises(control.ControlError, match="exact current fleet gate"):
        control._read_client_capacity_authorization(
            state_dir,
            state,
            authorization_path,
            _sha(authorization_path),
            attested_timestamp=102.0,
        )

    reseal(readiness_path, readiness_payload)
    authorization["readiness_evidence"]["sha256"] = _sha(readiness_path)
    reseal(authorization_path, authorization)

    # Capacity-transition evidence remains independently verifiable, but the
    # prelaunch protected-capacity marker is the one client-placement authority
    # for every registered 24->96->192->384 stage.
    execution = control.production_cell_execution(initialized)
    higher = copy.deepcopy(initialized)
    higher["admission"]["current_ceiling"] = 192
    higher["capacity"]["current_generation"] = 2
    higher["capacity"]["current_contract"] = {
        "capacity_generation": 2,
        "path": str(effective_fleet_path.resolve()),
        "sha256": contract_sha,
        "marker_path": str(effective_fleet_marker.resolve()),
        "marker_sha256": _sha(effective_fleet_marker),
        "protected_capacity_marker_path": str(
            capacity_authority["protected_capacity_marker_path"].resolve()
        ),
        "protected_capacity_marker_sha256": capacity_authority[
            "protected_capacity_marker_sha256"
        ],
        "protected_capacity_marker_id": capacity_authority[
            "protected_capacity_marker_id"
        ],
        "static_feasibility_certificate_path": str(
            capacity_authority[
                "static_feasibility_certificate_path"
            ].resolve()
        ),
        "static_feasibility_certificate_sha256": capacity_authority[
            "static_feasibility_certificate_sha256"
        ],
        "static_feasibility_certificate_id": capacity_authority[
            "static_feasibility_certificate_id"
        ],
        "base_fleet_contract_sha256": initialized["immutable"][
            "fleet_contract_sha256"
        ],
        "additive_overlay_contract_path": str(
            effective_fleet_path.resolve()
        ),
        "additive_overlay_contract_sha256": contract_sha,
        "fleet_id": effective_fleet_payload["fleet_id"],
        "logical_replicas": sum(profile_replicas.values()),
        "allocated_gpus": sum(
            int(row["tensor_parallel_size"]) * len(row["replicas"])
            for row in effective_fleet_payload["profiles"]
        ),
        "profile_replicas": profile_replicas,
        "activated_at": control.utc_timestamp(100.0),
        "activated_timestamp": 100.0,
    }
    higher["admission_ramp"]["current_ceiling"] = 192
    higher["admission_ramp"]["client_capacity_authorizations"] = [record]
    monkeypatch.setattr(
        control,
        "production_cell_execution_from_state",
        lambda _state_dir: execution,
    )
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: higher,
    )
    summary = control._client_capacity_contract_summary(
        higher,
        target_ceiling=192,
        verify_files=False,
    )
    batch_manifest = tmp_path / "batch.json"
    batch_manifest.write_text("{}\n", encoding="utf-8")
    rendered = dispatcher._render_batch_sbatch(
        batch_manifest,
        n_tasks=2,
        partition=str(summary["partition"]),
        qos=str(summary["qos"]),
        time_limit="12:00:00",
        memory="4G",
        log_dir=tmp_path / "logs",
        batch_tag="capacity-g2",
        batch_manifest_sha256="c" * 64,
        control_state_dir=production_state_dir,
    )
    assert "#SBATCH --partition=ou_bcs_normal" in rendered
    assert "#SBATCH --qos=normal" in rendered
    assert "#SBATCH --partition=mit_normal" not in rendered
    with pytest.raises(
        control.ControlError, match="ou_bcs_normal"
    ):
        dispatcher._render_batch_sbatch(
            batch_manifest,
            n_tasks=2,
            partition="mit_normal",
            qos="mit_normal",
            time_limit="12:00:00",
            memory="4G",
            log_dir=tmp_path / "logs",
            batch_tag="capacity-g2-unsafe",
            batch_manifest_sha256="d" * 64,
            control_state_dir=production_state_dir,
        )

    scheduler_path.chmod(0o644)
    with pytest.raises(
        control.ControlError, match="sealed, single-link regular file"
    ):
        control._read_client_capacity_authorization(
            state_dir,
            state,
            authorization_path,
            _sha(authorization_path),
            attested_timestamp=102.0,
        )


def test_client_capacity_builder_is_marker_last_idempotent_and_tamper_closed(
    tmp_path, monkeypatch
):
    state_dir, initialized = initialize(tmp_path)
    partition = "schema5_clients_g2"
    target = 192
    transition_id = "capacity-g000001-to-g000002-test"
    synthetic = copy.deepcopy(initialized)
    synthetic.update(
        {
            "desired_state": "paused",
            "drain_requested": True,
            "rollout_generation": 1,
        }
    )
    synthetic["capacity"] = {
        "schema_version": control.CAPACITY_STATE_SCHEMA_VERSION,
        "protocol": control.CAPACITY_STATE_PROTOCOL,
        "current_generation": 2,
        "current_contract": {"sha256": "b" * 64},
        "active_transition": {
            "transition_id": transition_id,
            "phase": "readiness_pending",
            "to_generation": 2,
            "fleet_launch_completed_timestamp": 50.0,
        },
        "history": [],
    }
    gate_paths = {}
    for gate_name in ("fleet", "smoke_runs", "scheduler_reconciliation"):
        gate_path = tmp_path / "builder-gates" / f"{gate_name}.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        control._atomic_write_json(
            gate_path, {"gate": gate_name, "passed": True}
        )
        gate_path.chmod(0o444)
        gate_paths[gate_name] = gate_path
    for gate_name, gate in synthetic["readiness"].items():
        gate_path = gate_paths.get(gate_name, gate_paths["fleet"])
        gate.update(
            {
                "passed": True,
                "evidence": str(gate_path.resolve()),
                "sha256": _sha(gate_path),
                "capacity_generation": 2,
            }
        )
    monkeypatch.setattr(
        control, "load_control", lambda *_args, **_kwargs: synthetic
    )
    monkeypatch.setattr(
        control,
        "validate_capacity_readiness_pending_state",
        lambda *_args, **_kwargs: synthetic,
    )
    monkeypatch.setattr(
        control, "_capacity_gates_are_fresh", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        control, "validate_readiness", lambda *_args, **_kwargs: None
    )

    def scheduler_runner(argv, *, timeout):
        del timeout
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            stdout = (
                "KillWait                = 30 sec\n"
                "PreemptMode             = REQUEUE\n"
                "PreemptType             = preempt/partition_prio\n"
            )
        elif args == [
            "scontrol",
            "show",
            "partition",
            partition,
            "-o",
        ]:
            stdout = (
                f"PartitionName={partition} GraceTime=0 MaxTime=12:00:00 "
                "PreemptMode=OFF State=UP TotalNodes=50\n"
            )
        elif args == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            partition,
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ]:
            stdout = f"{partition}|cpu=384,mem=1536G|448\n"
        else:  # pragma: no cover - fixture boundary
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    root = (
        state_dir
        / "capacity-transitions"
        / transition_id
        / f"client-capacity-{target}"
    )
    dry = control.build_client_capacity_generation(
        state_dir,
        action="dry-run",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        now=100.0,
    )
    assert dry["would_change"] is True
    assert dry["nonpreemptible_client_slots"] == 384
    assert dry.get("submit_command") != [
        "sbatch",
        str(
            (
                state_dir
                / "capacity-transitions"
                / transition_id
                / f"client-capacity-{target}"
                / control.CLIENT_CAPACITY_CANARY_SBATCH
            ).resolve()
        ),
    ]
    assert not root.exists()

    sbatch_target = root / control.CLIENT_CAPACITY_CANARY_SBATCH
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == sbatch_target:
            crashed["value"] = True
            raise RuntimeError("death after client-capacity sbatch link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="client-capacity sbatch link"):
            control.build_client_capacity_generation(
                state_dir,
                action="prepare",
                partition=partition,
                target_ceiling=target,
                scheduler_runner=scheduler_runner,
                now=100.0,
            )
    assert sbatch_target.is_file()
    assert sbatch_target.stat().st_nlink == 1
    assert sbatch_target.stat().st_mode & 0o222 == 0
    sbatch_preimage = sbatch_target.read_bytes()

    prepared = control.build_client_capacity_generation(
        state_dir,
        action="prepare",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        now=100.0,
    )
    assert sbatch_target.read_bytes() == sbatch_preimage
    assert prepared["state"] == "canary_prepared"
    assert prepared.get("submit_command") != [
        "sbatch",
        prepared["sbatch"],
    ]
    assert not (root / control.CLIENT_CAPACITY_COMPLETE).exists()
    assert not (
        root / control.CLIENT_CAPACITY_SCHEDULER_EVIDENCE
    ).exists()
    preflight = json.loads(
        (
            root / control.CLIENT_CAPACITY_SCHEDULER_PREFLIGHT_EVIDENCE
        ).read_text(encoding="utf-8")
    )
    assert (
        preflight["raw_client_capacity_contract"]["captured_timestamp"]
        == 100.0
    )
    assert (
        root / control.CLIENT_CAPACITY_BUILD_INTENT
    ).stat().st_mode & 0o222 == 0

    sbatch_path = Path(prepared["sbatch"])
    sbatch_path.chmod(0o644)
    with pytest.raises(control.ControlError, match="sbatch drifted"):
        control.build_client_capacity_generation(
            state_dir,
            action="submit",
            partition=partition,
            target_ceiling=target,
            scheduler_runner=scheduler_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(
                    _client_capacity_submission_snapshot(
                        SimpleNamespace(), complete=True
                    )
                )
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a drifted sbatch must fail before submission"
            ),
            scheduler_user="test-user",
            now=101.0,
        )
    sbatch_path.chmod(0o444)

    sbatch_lines = sbatch_path.read_text(encoding="utf-8").splitlines()
    submit_fixture = SimpleNamespace(
        sbatch_path=sbatch_path,
        partition=partition,
        comment=next(
            line.removeprefix("#SBATCH --comment=")
            for line in sbatch_lines
            if line.startswith("#SBATCH --comment=")
        ),
        job_name=next(
            line.removeprefix("#SBATCH --job-name=")
            for line in sbatch_lines
            if line.startswith("#SBATCH --job-name=")
        ),
    )
    empty = _client_capacity_submission_snapshot(submit_fixture)
    completed_scheduler = _client_capacity_submission_snapshot(
        submit_fixture,
        _exact_client_capacity_job(
            submit_fixture,
            "123",
            state="COMPLETED",
            source="sacct",
        ),
    )
    control.build_client_capacity_generation(
        state_dir,
        action="submit",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(
                empty, completed_scheduler
            )
        ),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "123\n", ""
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    control.build_client_capacity_generation(
        state_dir,
        action="submit",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(completed_scheduler)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "the exact visible canary must be adopted"
        ),
        scheduler_user="test-user",
        now=101.5,
    )

    def command_runner(argv):
        args = list(argv)
        submit_line = shlex.join(
            (
                "sbatch",
                "--parsable",
                f"--comment={submit_fixture.comment}",
                str(sbatch_path.resolve()),
            )
        )
        stdout = f"123|COMPLETED|{partition}|1|4G|0:0|{submit_line}\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    completed = control.build_client_capacity_generation(
        state_dir,
        action="apply",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        command_runner=command_runner,
        now=101.0,
    )
    assert completed["state"] == "complete"
    marker = Path(completed["authorization"])
    assert marker.name == control.CLIENT_CAPACITY_COMPLETE
    assert marker.stat().st_mode & 0o222 == 0
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    canonical_scheduler = json.loads(
        (
            root / control.CLIENT_CAPACITY_SCHEDULER_EVIDENCE
        ).read_text(encoding="utf-8")
    )
    assert (
        canonical_scheduler["raw_client_capacity_contract"][
            "captured_timestamp"
        ]
        == 101.0
    )
    assert marker_payload["captured_timestamp"] == 101.0
    assert (
        marker_payload["captured_timestamp"]
        == canonical_scheduler["raw_client_capacity_contract"][
            "captured_timestamp"
        ]
    )
    repeated = control.build_client_capacity_generation(
        state_dir,
        action="apply",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=scheduler_runner,
        command_runner=command_runner,
        now=102.0,
    )
    assert repeated["state"] == "already_complete"
    assert repeated["sha256"] == completed["sha256"]

    scheduler_path = root / control.CLIENT_CAPACITY_SCHEDULER_EVIDENCE
    scheduler_path.chmod(0o644)
    with pytest.raises(
        control.ControlError, match="sealed, single-link regular file"
    ):
        control.build_client_capacity_generation(
            state_dir,
            action="apply",
            partition=partition,
            target_ceiling=target,
            scheduler_runner=scheduler_runner,
            command_runner=command_runner,
            now=103.0,
        )


def _prepared_client_capacity_submission_fixture(tmp_path, monkeypatch):
    """Prepare one generation-2 canary for transactional submission tests."""

    state_dir, initialized = initialize(tmp_path)
    partition = "schema5_clients_g2"
    target = 192
    transition_id = "capacity-g000001-to-g000002-submit-test"
    synthetic = copy.deepcopy(initialized)
    synthetic.update(
        {
            "desired_state": "paused",
            "drain_requested": True,
            "rollout_generation": 1,
        }
    )
    synthetic["capacity"] = {
        "schema_version": control.CAPACITY_STATE_SCHEMA_VERSION,
        "protocol": control.CAPACITY_STATE_PROTOCOL,
        "current_generation": 2,
        "current_contract": {"sha256": "b" * 64},
        "active_transition": {
            "transition_id": transition_id,
            "phase": "readiness_pending",
            "to_generation": 2,
            "fleet_launch_completed_timestamp": 50.0,
        },
        "history": [],
    }
    gate_paths = {}
    for gate_name in ("fleet", "smoke_runs", "scheduler_reconciliation"):
        gate_path = tmp_path / "submit-builder-gates" / f"{gate_name}.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        control._atomic_write_json(
            gate_path, {"gate": gate_name, "passed": True}
        )
        gate_path.chmod(0o444)
        gate_paths[gate_name] = gate_path
    for gate_name, gate in synthetic["readiness"].items():
        gate_path = gate_paths.get(gate_name, gate_paths["fleet"])
        gate.update(
            {
                "passed": True,
                "evidence": str(gate_path.resolve()),
                "sha256": _sha(gate_path),
                "capacity_generation": 2,
            }
        )
    monkeypatch.setattr(
        control, "load_control", lambda *_args, **_kwargs: synthetic
    )
    monkeypatch.setattr(
        control,
        "validate_capacity_readiness_pending_state",
        lambda *_args, **_kwargs: synthetic,
    )
    monkeypatch.setattr(
        control, "_capacity_gates_are_fresh", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        control, "validate_readiness", lambda *_args, **_kwargs: None
    )

    def capacity_runner(argv, *, timeout):
        del timeout
        args = list(argv)
        if args == ["scontrol", "show", "config"]:
            stdout = (
                "KillWait                = 30 sec\n"
                "PreemptMode             = REQUEUE\n"
                "PreemptType             = preempt/partition_prio\n"
            )
        elif args == [
            "scontrol",
            "show",
            "partition",
            partition,
            "-o",
        ]:
            stdout = (
                f"PartitionName={partition} GraceTime=0 MaxTime=12:00:00 "
                "PreemptMode=OFF State=UP TotalNodes=50\n"
            )
        elif args == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            partition,
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ]:
            stdout = f"{partition}|cpu=384,mem=1536G|448\n"
        else:  # pragma: no cover - fixture boundary
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    prepared = control.build_client_capacity_generation(
        state_dir,
        action="prepare",
        partition=partition,
        target_ceiling=target,
        scheduler_runner=capacity_runner,
        now=100.0,
    )
    root = (
        state_dir
        / "capacity-transitions"
        / transition_id
        / f"client-capacity-{target}"
    )
    intent = json.loads(
        (root / control.CLIENT_CAPACITY_BUILD_INTENT).read_text(
            encoding="utf-8"
        )
    )
    sbatch_path = Path(prepared["sbatch"])
    sbatch_lines = sbatch_path.read_text(encoding="utf-8").splitlines()
    comment = next(
        line.removeprefix("#SBATCH --comment=")
        for line in sbatch_lines
        if line.startswith("#SBATCH --comment=")
    )
    job_name = next(
        line.removeprefix("#SBATCH --job-name=")
        for line in sbatch_lines
        if line.startswith("#SBATCH --job-name=")
    )
    return SimpleNamespace(
        state_dir=state_dir,
        synthetic=synthetic,
        partition=partition,
        target=target,
        transition_id=transition_id,
        root=root,
        prepared=prepared,
        intent=intent,
        sbatch_path=sbatch_path,
        sbatch_sha256=_sha(sbatch_path),
        comment=comment,
        job_name=job_name,
        capacity_runner=capacity_runner,
    )


def _client_capacity_submission_snapshot(
    fixture,
    *jobs: tuple[str, str, str, str, str, str],
    complete: bool = True,
    captured_at: float = 101.0,
) -> control.SchedulerSnapshot:
    return control.SchedulerSnapshot(
        jobs=tuple(
            control.SchedulerJob(
                job_id=job_id,
                job_name=job_name,
                state=state,
                comment=comment,
                command=command,
                source=source,
                partition=fixture.partition,
                qos=fixture.partition,
            )
            for job_id, job_name, state, comment, command, source in jobs
        ),
        captured_at=captured_at,
        squeue_ok=complete,
        sacct_ok=complete,
        errors=() if complete else ("sacct unavailable",),
    )


def _client_capacity_submission_scheduler_runner(*snapshots):
    """Render complete query_scheduler command output from fixture snapshots."""

    assert snapshots
    query_index = 0

    def runner(argv):
        nonlocal query_index
        args = list(argv)
        snapshot = snapshots[min(query_index, len(snapshots) - 1)]
        if args[0] == "squeue":
            if not snapshot.squeue_ok:
                return subprocess.CompletedProcess(
                    args, 1, "", "squeue fixture failure"
                )
            rows = []
            for job in snapshot.jobs:
                if job.source == "sacct":
                    continue
                rows.append(
                    "|".join(
                        (
                            job.job_id,
                            job.job_name,
                            job.state,
                            job.partition,
                            job.qos,
                            job.comment,
                            job.command,
                            "",
                        )
                    )
                )
            return subprocess.CompletedProcess(
                args, 0, "".join(f"{row}\n" for row in rows), ""
            )
        if args[0] == "sacct":
            query_index += 1
            if not snapshot.sacct_ok:
                return subprocess.CompletedProcess(
                    args, 1, "", "sacct fixture failure"
                )
            rows = []
            for job in snapshot.jobs:
                if job.source == "squeue":
                    continue
                rows.append(
                    "|".join(
                        (
                            job.job_id,
                            job.job_name,
                            job.state,
                            job.partition,
                            job.qos,
                            job.comment,
                            job.command,
                        )
                    )
                )
            return subprocess.CompletedProcess(
                args, 0, "".join(f"{row}\n" for row in rows), ""
            )
        raise AssertionError(args)  # pragma: no cover - fixture boundary

    return runner


def _exact_client_capacity_job(
    fixture,
    job_id: str,
    *,
    state: str = "RUNNING",
    source: str = "squeue+sacct",
) -> tuple[str, str, str, str, str, str]:
    return (
        job_id,
        fixture.job_name,
        state,
        fixture.comment,
        shlex.join(
            (
                "sbatch",
                "--parsable",
                f"--comment={fixture.comment}",
                str(fixture.sbatch_path),
            )
        ),
        source,
    )


def test_client_capacity_submit_is_marker_first_marker_last_and_replay_safe(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "701")
    )
    snapshots = iter((empty, visible))
    submit_calls = []

    def submit_runner(argv):
        submit_calls.append(list(argv))
        submission_root = (
            fixture.root / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        )
        intent_path = (
            submission_root / control.CLIENT_CAPACITY_SUBMISSION_INTENT
        )
        attempts = (
            submission_root
            / control.CLIENT_CAPACITY_SUBMISSION_ATTEMPTS_DIRECTORY
        )
        assert intent_path.is_file()
        assert intent_path.stat().st_mode & 0o222 == 0
        attempt_intents = list(attempts.rglob("ATTEMPT_INTENT.json"))
        assert len(attempt_intents) == 1
        assert attempt_intents[0].stat().st_mode & 0o222 == 0
        assert not list(attempts.rglob("SBATCH_RESULT.json"))
        assert not (
            fixture.root
            / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
            / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
        ).exists()
        return subprocess.CompletedProcess(list(argv), 0, "701\n", "")

    submitted = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(*snapshots)
        ),
        submit_runner=submit_runner,
        scheduler_user="test-user",
        now=101.0,
    )
    assert submit_calls == [
        [
            "sbatch",
            "--parsable",
            f"--comment={fixture.comment}",
            str(fixture.sbatch_path),
        ]
    ]
    accepted_path = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    )
    if submitted["state"] == "submission_pending":
        submitted = control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a visible submitted canary must be adopted"
            ),
            scheduler_user="test-user",
            now=101.5,
        )
    assert submitted["job_id"] == "701"
    assert accepted_path.is_file()
    assert accepted_path.stat().st_mode & 0o222 == 0
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    assert accepted["job_id"] == "701"
    assert accepted["job_comment"] == fixture.comment
    assert accepted["sbatch_path"] == str(fixture.sbatch_path)
    assert accepted["sbatch_sha256"] == fixture.sbatch_sha256
    accepted_sha = _sha(accepted_path)

    replayed = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(visible)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "an accepted client canary must never be resubmitted"
        ),
        scheduler_user="test-user",
        now=102.0,
    )
    assert replayed["job_id"] == "701"
    assert replayed["state"] == "already_submitted"
    assert _sha(accepted_path) == accepted_sha


def test_client_capacity_submit_adopts_post_sbatch_crash_without_duplicate(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    with pytest.raises(RuntimeError, match="simulated post-sbatch crash"):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(empty)
            ),
            submit_runner=lambda _argv: (_ for _ in ()).throw(
                RuntimeError("simulated post-sbatch crash")
            ),
            scheduler_user="test-user",
            now=101.0,
        )
    assert not (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    ).exists()

    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "702")
    )
    adopted = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(visible)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "a visible post-sbatch crash-window job must be adopted"
        ),
        scheduler_user="test-user",
        now=102.0,
    )
    assert adopted["job_id"] == "702"
    assert adopted["state"] == "adopted"


def test_client_capacity_candidate_observation_survives_accept_marker_crash(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "709")
    )
    first = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "709\n", ""
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    assert first["state"] == "submission_pending"
    real_write = control._write_sealed_builder_json

    def crash_before_accept(path, payload, *, description):
        if description == "client-capacity accepted submission":
            raise RuntimeError("simulated accepted-marker crash")
        return real_write(path, payload, description=description)

    monkeypatch.setattr(
        control, "_write_sealed_builder_json", crash_before_accept
    )
    with pytest.raises(RuntimeError, match="accepted-marker crash"):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "visible canary must be adopted without another sbatch"
            ),
            scheduler_user="test-user",
            now=102.0,
        )
    accepted_path = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    )
    assert not accepted_path.exists()
    monkeypatch.setattr(
        control, "_write_sealed_builder_json", real_write
    )

    # Even if both live and accounting views temporarily omit the short canary,
    # the sealed candidate-bearing observation is permanent acceptance evidence.
    adopted = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "historical candidate observation must prevent a second sbatch"
        ),
        scheduler_user="test-user",
        now=163.0,
    )
    assert adopted["state"] == "adopted"
    assert adopted["job_id"] == "709"
    replayed = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "sealed acceptance must remain replay-safe"
        ),
        scheduler_user="test-user",
        now=224.0,
    )
    assert replayed["state"] == "already_submitted"
    assert replayed["job_id"] == "709"


@pytest.mark.parametrize(
    ("crash_description", "recoverable"),
    [
        ("client-capacity sacct submission observation", False),
        ("client-capacity submission observation completion", True),
    ],
)
def test_client_capacity_partial_candidate_observation_never_allows_retry(
    tmp_path, monkeypatch, crash_description, recoverable
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "710")
    )
    control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "710\n", ""
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    real_write = control._write_sealed_builder_json

    def crash_partial(path, payload, *, description):
        if description == crash_description:
            raise RuntimeError("simulated partial-observation crash")
        return real_write(path, payload, description=description)

    monkeypatch.setattr(control, "_write_sealed_builder_json", crash_partial)
    with pytest.raises(RuntimeError, match="partial-observation crash"):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "visible candidate must not cross sbatch again"
            ),
            scheduler_user="test-user",
            now=102.0,
        )
    monkeypatch.setattr(control, "_write_sealed_builder_json", real_write)
    if recoverable:
        recovered = control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(empty)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "two sealed source observations must recover without sbatch"
            ),
            scheduler_user="test-user",
            now=163.0,
        )
        assert recovered["state"] == "adopted"
        assert recovered["job_id"] == "710"
    else:
        with pytest.raises(
            control.SchedulerVisibilityPending,
            match="partial durable.*exact candidate",
        ):
            control.build_client_capacity_generation(
                fixture.state_dir,
                action="submit",
                partition=fixture.partition,
                target_ceiling=fixture.target,
                scheduler_runner=fixture.capacity_runner,
                submission_scheduler_runner=(
                    _client_capacity_submission_scheduler_runner(empty)
                ),
                submit_runner=lambda _argv: pytest.fail(
                    "one partial candidate source must fail closed"
                ),
                scheduler_user="test-user",
                now=163.0,
            )


def test_client_capacity_ambiguous_submit_retries_only_after_two_absences(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    ambiguous = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: (_ for _ in ()).throw(
            OSError("ambiguous sbatch transport")
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    assert ambiguous["state"] == "submission_pending"
    assert ambiguous["job_id"] is None

    first_absence = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "one absence observation cannot authorize a retry"
        ),
        scheduler_user="test-user",
        now=120.0,
    )
    assert first_absence["state"] == "submission_pending"
    assert "two complete absence observations" in first_absence["reason"]

    retry_calls = []

    def retry_submit(argv):
        retry_calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "707\n", "")

    retried = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=retry_submit,
        scheduler_user="test-user",
        now=181.0,
    )
    assert retried["state"] == "submission_pending"
    assert retried["job_id"] == "707"
    assert retry_calls == [
        [
            "sbatch",
            "--parsable",
            f"--comment={fixture.comment}",
            str(fixture.sbatch_path),
        ]
    ]
    attempts_root = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ATTEMPTS_DIRECTORY
    )
    retry_authorization_path = (
        attempts_root / "000001" / "RETRY_AUTHORIZATION.json"
    )
    assert retry_authorization_path.is_file()
    assert retry_authorization_path.stat().st_mode & 0o222 == 0
    retry_authorization = json.loads(
        retry_authorization_path.read_text(encoding="utf-8")
    )
    assert retry_authorization["from_attempt"] == 1
    assert retry_authorization["to_attempt"] == 2
    assert len(retry_authorization["absence_observations"]) == 2
    assert retry_authorization["absence_interval_seconds"] == 61.0
    second_attempt = json.loads(
        (attempts_root / "000002" / "ATTEMPT_INTENT.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_attempt["retry_authorization"] == {
        "path": str(retry_authorization_path.resolve()),
        "sha256": _sha(retry_authorization_path),
    }
    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "707")
    )
    adopted = control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(visible)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "the one retried canary must be adopted without another sbatch"
        ),
        scheduler_user="test-user",
        now=182.0,
    )
    assert adopted["state"] == "adopted"
    assert adopted["job_id"] == "707"

    retry_authorization["absence_interval_seconds"] = 60.0
    retry_authorization_path.chmod(0o644)
    retry_authorization_path.write_text(
        json.dumps(retry_authorization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    retry_authorization_path.chmod(0o444)
    with pytest.raises(
        control.ControlError,
        match="retry authorization|retry authorization reference",
    ):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "tampered retry lineage must fail before sbatch"
            ),
            scheduler_user="test-user",
            now=183.0,
        )


def test_client_capacity_retry_rejects_tampered_absence_source(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: (_ for _ in ()).throw(
            OSError("ambiguous sbatch transport")
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(empty)
        ),
        submit_runner=lambda _argv: pytest.fail(
            "one absence cannot authorize retry"
        ),
        scheduler_user="test-user",
        now=120.0,
    )
    source = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ATTEMPTS_DIRECTORY
        / "000001"
        / "observations"
        / "000001"
        / "SQUEUE_OBSERVATION.json"
    )
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b" ")
    source.chmod(0o444)
    with pytest.raises(
        control.ControlError,
        match="observation reference drifted|observation drifted",
    ):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(empty)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "tampered absence evidence cannot authorize retry"
            ),
            scheduler_user="test-user",
            now=181.0,
        )


def test_client_capacity_submit_rejects_exact_comment_on_foreign_sbatch(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    foreign = _client_capacity_submission_snapshot(
        fixture,
        (
            "708",
            fixture.job_name,
            "RUNNING",
            fixture.comment,
            "sbatch --parsable "
            f"--comment={shlex.quote(fixture.comment)} "
            "/tmp/foreign-client-capacity.sbatch",
            "squeue+sacct",
        ),
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="foreign or drifted"
    ):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(foreign)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a foreign exact-comment candidate must prevent sbatch"
            ),
            scheduler_user="test-user",
            now=101.0,
        )


@pytest.mark.parametrize("failure", ("incomplete", "duplicate"))
def test_client_capacity_submit_fails_closed_on_incomplete_or_duplicate_truth(
    tmp_path, monkeypatch, failure
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    if failure == "incomplete":
        snapshot = _client_capacity_submission_snapshot(
            fixture, complete=False
        )
        match = "complete.*squeue.*sacct"
    else:
        snapshot = _client_capacity_submission_snapshot(
            fixture,
            _exact_client_capacity_job(fixture, "703"),
            _exact_client_capacity_job(
                fixture, "704", state="PENDING", source="sacct"
            ),
        )
        match = "multiple|exactly one|duplicate|ambiguous|absent.*squeue"
    with pytest.raises(
        (control.ControlError, control.SchedulerAmbiguity), match=match
    ):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(snapshot)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "unsafe scheduler truth must prevent sbatch"
            ),
            scheduler_user="test-user",
            now=101.0,
        )
    assert not (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    ).exists()


def test_client_capacity_submit_rejects_misbound_accepted_marker(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    visible = _client_capacity_submission_snapshot(
        fixture, _exact_client_capacity_job(fixture, "705")
    )
    snapshots = iter((empty, visible))
    control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(*snapshots)
        ),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "705\n", ""
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    marker = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    )
    if not marker.exists():
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a visible submitted canary must be adopted"
            ),
            scheduler_user="test-user",
            now=101.5,
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["sbatch_path"] = "/tmp/foreign-client-capacity.sbatch"
    marker.chmod(0o644)
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o444)

    with pytest.raises(control.ControlError, match="accepted.*identity|drift"):
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(visible)
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a misbound accepted marker must fail before sbatch"
            ),
            scheduler_user="test-user",
            now=102.0,
        )


def test_client_capacity_apply_derives_job_id_from_sealed_acceptance(
    tmp_path, monkeypatch
):
    fixture = _prepared_client_capacity_submission_fixture(
        tmp_path, monkeypatch
    )
    empty = _client_capacity_submission_snapshot(fixture)
    completed_snapshot = _client_capacity_submission_snapshot(
        fixture,
        _exact_client_capacity_job(
            fixture, "706", state="COMPLETED", source="sacct"
        ),
    )
    snapshots = iter((empty, completed_snapshot))
    control.build_client_capacity_generation(
        fixture.state_dir,
        action="submit",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        submission_scheduler_runner=(
            _client_capacity_submission_scheduler_runner(*snapshots)
        ),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            list(argv), 0, "706\n", ""
        ),
        scheduler_user="test-user",
        now=101.0,
    )
    accepted_path = (
        fixture.root
        / control.CLIENT_CAPACITY_SUBMISSION_DIRECTORY
        / control.CLIENT_CAPACITY_SUBMISSION_ACCEPTED
    )
    if not accepted_path.exists():
        control.build_client_capacity_generation(
            fixture.state_dir,
            action="submit",
            partition=fixture.partition,
            target_ceiling=fixture.target,
            scheduler_runner=fixture.capacity_runner,
            submission_scheduler_runner=(
                _client_capacity_submission_scheduler_runner(
                    completed_snapshot
                )
            ),
            submit_runner=lambda _argv: pytest.fail(
                "a completed submitted canary must be adopted"
            ),
            scheduler_user="test-user",
            now=101.5,
        )

    def canary_runner(argv):
        args = list(argv)
        assert args[0:5] == ["sacct", "-X", "-nP", "-j", "706"]
        submit_line = shlex.join(
            (
                "sbatch",
                "--parsable",
                f"--comment={fixture.comment}",
                str(fixture.sbatch_path),
            )
        )
        stdout = (
            f"706|COMPLETED|{fixture.partition}|1|4G|0:0|"
            f"{submit_line}\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout, "")

    completed = control.build_client_capacity_generation(
        fixture.state_dir,
        action="apply",
        partition=fixture.partition,
        target_ceiling=fixture.target,
        scheduler_runner=fixture.capacity_runner,
        command_runner=canary_runner,
        now=102.0,
    )
    assert completed["state"] == "complete"
    evidence = json.loads(
        (
            fixture.root / control.CLIENT_CAPACITY_CANARY_EVIDENCE
        ).read_text(encoding="utf-8")
    )
    assert evidence["job_id"] == "706"


def test_admission_ramp_resets_on_blocking_alert_gap_pause_and_rejects_manual_raise(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    qids = {run_id: 10 for run_id in control.REQUIRED_RUNS}
    control.record_successful_poll(
        state_dir,
        useful_qids=sum(qids.values()),
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
    elevated["admission"]["current_ceiling"] = 96
    elevated["admission_ramp"]["current_ceiling"] = 96
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
    _install_exact_cell_transaction(
        state_dir,
        batch_id="batch1",
        base_job_id="900",
        captured_at=60.0,
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


def test_pause_blocks_reconciled_job_still_inside_visibility_reservation(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    _install_exact_cell_transaction(
        state_dir,
        batch_id="reconciled1",
        base_job_id="900",
        intent_state="reconciled",
        record_state="visibility_grace",
        captured_at=60.0,
    )
    with pytest.raises(
        control.SchedulerVisibilityPending,
        match="not yet visible|visibility",
    ):
        control.pause_control(
            state_dir,
            drain=True,
            scheduler=control.SchedulerSnapshot((), 60.0),
            now=60.0,
        )
    persisted = json.loads(
        (state_dir / "ledger.json").read_text(encoding="utf-8")
    )
    assert persisted["intents"]["reconciled1"]["state"] == "reconciled"
    assert persisted["jobs"]["900"]["state"] == "visibility_grace"


def _install_exact_cell_transaction(
    state_dir: Path,
    *,
    batch_id: str,
    base_job_id: str,
    task_count: int = 5,
    intent_state: str = "submitted",
    record_state: str = "active",
    include_record: bool = True,
    captured_at: float = 60.0,
) -> tuple[Path, Path, list[dict[str, int]]]:
    """Install the same sealed batch transaction trusted in production."""

    batch_dir = state_dir / "batches"
    log_dir = state_dir / "logs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_root = (state_dir.parent / "full_sweep_schema5_v1").resolve()
    server_pool_root = (state_dir.parent / "server_pools" / "schema5-v1").resolve()
    tasks = [
        {
            "run_id": "full_sweep_schema5_v1",
            "run_root": str(run_root),
            "source_index": index,
            "cell_id": f"cell-{index:04d}",
            "config_hash": f"{index + 1:012x}",
            "manifest_sha256": "b" * 64,
            "benchmark_contracts_sha256": "c" * 64,
            "model_size": "8B",
            "serving_profile": "8B",
            "fanout_cost": 1,
            "server_pool_id": None,
            "server_run_id": None,
            "server_pool_root": str(server_pool_root),
        }
        for index in range(task_count)
    ]
    manifest_path = batch_dir / f"batch-{batch_id}.json"
    dispatcher._atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": captured_at - 10.0,
            "tasks": tasks,
        },
    )
    manifest_sha256 = dispatcher._seal_dispatch_artifact(manifest_path)
    sbatch_path = batch_dir / f"batch-{batch_id}.sbatch"
    sbatch_path.write_text(
        dispatcher._render_batch_sbatch(
            manifest_path,
            n_tasks=task_count,
            partition="ou_bcs_normal",
            qos="normal",
            time_limit="12:00:00",
            memory="4G",
            log_dir=log_dir,
            batch_tag=batch_id[-10:],
            batch_id=batch_id,
            batch_manifest_sha256=manifest_sha256,
        ),
        encoding="utf-8",
    )
    sbatch_sha256 = dispatcher._seal_dispatch_artifact(sbatch_path)
    ledger = dispatcher._empty_ledger(now=captured_at - 10.0)
    intent = {
        "state": intent_state,
        "created_at": captured_at - 11.0,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "task_count": task_count,
        "tasks": copy.deepcopy(tasks),
        "fairness_after": {"cursor": 0, "deficits": {}},
        "fairness_committed": intent_state in {"submitted", "reconciled"},
        "submission_transport": dispatcher.STDIN_EXACT_SUBMISSION_TRANSPORT,
        "submission_argv_sha256": dispatcher._stdin_submission_argv_sha256(
            batch_id
        ),
    }
    if intent_state in {"submitting", "submitted", "reconciled"}:
        intent["submit_started_at"] = captured_at - 5.0
    if intent_state in {"submitted", "reconciled"}:
        intent["job_id"] = base_job_id
    if intent_state == "submitted":
        intent["submitted_at"] = captured_at - 4.0
    elif intent_state == "reconciled":
        intent["reconciled_at"] = captured_at - 4.0
    elif intent_state == "integrity_blocked":
        intent.update(
            {
                "error": "sealed sbatch hash drifted before external invocation",
                "integrity_blocked_at": captured_at - 4.0,
                "integrity_alert_key": (
                    dispatcher.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
                ),
            }
        )
    ledger["intents"][batch_id] = intent
    if include_record:
        if intent_state not in {"submitted", "reconciled"}:
            raise AssertionError(
                "only accepted intents may install a durable job record"
            )
        receipt_path, receipt_sha256 = dispatcher._spooled_script_receipt(
            batch_id=batch_id,
            job_id=base_job_id,
            expected_name=f"asys-dispatch-{batch_id[-10:]}",
            expected_comment=f"asys-schema5-intent:{batch_id}",
            sbatch_path=sbatch_path,
            sbatch_sha256=sbatch_sha256,
            spooled_script_reader=lambda _job_id: sbatch_path.read_bytes(),
            now=captured_at - 4.0,
        )
        ledger["intents"][batch_id].update(
            {
                "spooled_sbatch_sha256": sbatch_sha256,
                "spooled_receipt_path": receipt_path,
                "spooled_receipt_sha256": receipt_sha256,
            }
        )
        ledger["jobs"][base_job_id] = {
            "job_id": base_job_id,
            "batch_id": batch_id,
            "batch_manifest": str(manifest_path),
            "batch_manifest_sha256": manifest_sha256,
            "sbatch_path": str(sbatch_path),
            "sbatch_sha256": sbatch_sha256,
            "state": record_state,
            "task_count": task_count,
            "tasks": copy.deepcopy(tasks),
            "submitted_at": captured_at - 4.0,
            "last_seen_at": captured_at - 1.0,
            "spooled_sbatch_sha256": sbatch_sha256,
            "spooled_receipt_path": receipt_path,
            "spooled_receipt_sha256": receipt_sha256,
            "submission_transport": dispatcher.STDIN_EXACT_SUBMISSION_TRANSPORT,
            "submission_argv_sha256": (
                dispatcher._stdin_submission_argv_sha256(batch_id)
            ),
        }
        if record_state == "inactive":
            ledger["jobs"][base_job_id]["inactive_since_at"] = (
                captured_at - 1.0
            )
    (state_dir / "ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sbatch_path, manifest_path, tasks


def _exact_cell_submission_command(batch_id: str) -> str:
    return " ".join(dispatcher._stdin_submission_argv(batch_id))


def test_pause_can_cancel_visible_idless_ambiguous_submit_by_exact_intent(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    sbatch_path, _manifest_path, _tasks = _install_exact_cell_transaction(
        state_dir,
        batch_id="batch1",
        base_job_id="900",
        intent_state="submitting",
        include_record=False,
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_4",
                "asys-dispatch-batch1",
                "PENDING",
                "asys-schema5-intent:batch1",
                _exact_cell_submission_command("batch1"),
                "squeue",
            ),
        ),
        60.0,
    )
    monkeypatch.setattr(
        dispatcher,
        "_read_spooled_batch_script",
        lambda job_id: (
            sbatch_path.read_bytes()
            if job_id == "900"
            else pytest.fail(f"unexpected spooled job {job_id}")
        ),
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
    sbatch_path, _manifest_path, _tasks = _install_exact_cell_transaction(
        state_dir,
        batch_id="batch1",
        base_job_id="900",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-batch1",
                "RUNNING",
                "asys-schema5-intent:batch1",
                _exact_cell_submission_command("batch1"),
                "squeue",
            ),
            control.SchedulerJob(
                "900_4",
                "asys-dispatch-batch1",
                "PENDING",
                "asys-schema5-intent:batch1",
                _exact_cell_submission_command("batch1"),
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
    _sbatch_path, _manifest_path, _tasks = _install_exact_cell_transaction(
        state_dir,
        batch_id="batch1",
        base_job_id="900",
    )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-batch1",
                "RUNNING",
                "asys-schema5-intent:batch1",
                _exact_cell_submission_command("different-batch"),
                "squeue",
            ),
        ),
        60.0,
    )
    calls = []
    with pytest.raises(
        control.SchedulerAmbiguity,
        match="scheduler provenance drift|exact cell drain mapping",
    ):
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


def test_critical_alert_and_hold_share_one_admission_boundary(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    alert_journal_written = threading.Event()
    contender_started = threading.Event()
    contender_acquired = threading.Event()
    original_append = control._append_jsonl

    def observe_alert_journal(path, payload):
        original_append(path, payload)
        if (
            path == state_dir / control.ALERT_JOURNAL
            and payload.get("action") == "raised"
        ):
            alert_journal_written.set()
            assert contender_started.wait(timeout=2.0)
            # The alert is already durable, but the admission boundary must remain
            # unavailable until the same transaction has persisted ceiling zero.
            assert not contender_acquired.wait(timeout=0.1)

    monkeypatch.setattr(control, "_append_jsonl", observe_alert_journal)
    observed_ceiling: list[int] = []

    def contender() -> None:
        assert alert_journal_written.wait(timeout=2.0)
        contender_started.set()
        with control.admission_boundary_lock(state_dir):
            contender_acquired.set()
            observed_ceiling.append(
                control.admission_contract_from_state(state_dir)["current_ceiling"]
            )

    thread = threading.Thread(target=contender)
    thread.start()
    control.record_alert(
        state_dir,
        kind="scheduler-query",
        severity="critical",
        message="scheduler unavailable",
        dedupe_key="monitor:scheduler",
        scheduler=control.SchedulerSnapshot((), 60.0),
        now=60.0,
    )
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert contender_acquired.is_set()
    assert observed_ceiling == [0]


def _hold_drain_scheduler_fixture(state_dir, *, captured_at=60.0):
    sbatch_path, _manifest_path, _tasks = _install_exact_cell_transaction(
        state_dir,
        batch_id="hold1",
        base_job_id="900",
        captured_at=captured_at,
    )
    return sbatch_path, control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-hold1",
                "RUNNING",
                "asys-schema5-intent:hold1",
                _exact_cell_submission_command("hold1"),
                "squeue",
            ),
            control.SchedulerJob(
                "900_4",
                "asys-dispatch-hold1",
                "PENDING",
                "asys-schema5-intent:hold1",
                _exact_cell_submission_command("hold1"),
                "squeue",
            ),
        ),
        captured_at,
    )


def test_critical_hold_commits_exact_targets_then_drains_running_and_pending_cells(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    _sbatch_path, snapshot = _hold_drain_scheduler_fixture(state_dir)
    calls: list[list[str]] = []

    control.record_alert(
        state_dir,
        kind="untrusted-response",
        severity="critical",
        message="response provenance mismatch",
        dedupe_key="monitor:untrusted",
        scheduler=snapshot,
        signal_runner=lambda argv: (
            calls.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=60.0,
    )

    assert calls == [
        ["scancel", "--batch", "--signal=USR1", "900_3"],
        ["scancel", "900_4"],
    ]
    persisted = control.load_control(state_dir)
    assert persisted["admission_safety_hold"]["active"] is True
    assert control.admission_contract_from_state(state_dir)["current_ceiling"] == 0
    intent = persisted["safety_hold_drain_intent"]
    assert intent["state"] == "complete"
    assert set(intent["targets"]) == {
        "cell_usr1:900_3",
        "pending_cell_cancel:900_4",
    }
    assert all(row["accepted"] for row in intent["results"].values())
    events = [
        json.loads(line)
        for line in (state_dir / control.SAFETY_HOLD_DRAIN_JOURNAL)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["event"] == "intent_created"
    assert events[1]["event"] == "targets_committed"
    assert events[-1]["event"] == "complete"


def test_critical_hold_refuses_all_signals_if_one_live_cell_is_unmappable(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    sbatch_path, snapshot = _hold_drain_scheduler_fixture(state_dir)
    ambiguous = control.SchedulerSnapshot(
        (
            *snapshot.jobs,
            control.SchedulerJob(
                "901_0",
                "asys-dispatch-foreign",
                "RUNNING",
                "asys-schema5-intent:foreign",
                _exact_cell_submission_command("foreign"),
                "squeue",
            ),
        ),
        snapshot.captured_at,
    )
    calls: list[list[str]] = []

    with pytest.raises(
        control.SchedulerAmbiguity,
        match="provenance drift|mapping failed",
    ):
        control.record_alert(
            state_dir,
            kind="scheduler-ambiguity",
            severity="critical",
            message="unmappable cell",
            dedupe_key="monitor:scheduler",
            scheduler=ambiguous,
            signal_runner=lambda argv: (
                calls.append(list(argv))
                or subprocess.CompletedProcess(argv, 0, "", "")
            ),
            now=60.0,
        )

    assert calls == []
    persisted = control.load_control(state_dir)
    assert persisted["admission_safety_hold"]["active"] is True
    assert persisted["safety_hold_drain_intent"]["state"] == "failed"
    with pytest.raises(
        control.ControlError, match="complete exact.*drain"
    ):
        control.admission_contract_from_state(state_dir)


def test_failed_hold_signal_retries_without_redrawing_or_reissuing_absent_target(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    sbatch_path, snapshot = _hold_drain_scheduler_fixture(state_dir)
    running_only = control.SchedulerSnapshot((snapshot.jobs[0],), 60.0)
    calls: list[list[str]] = []

    with pytest.raises(control.ControlError, match="failed safety-hold drain action"):
        control.record_alert(
            state_dir,
            kind="scheduler-ambiguity",
            severity="critical",
            message="signal transport failed",
            dedupe_key="monitor:scheduler",
            scheduler=running_only,
            signal_runner=lambda argv: (
                calls.append(list(argv))
                or subprocess.CompletedProcess(argv, 1, "", "temporary failure")
            ),
            now=60.0,
        )
    assert calls == [["scancel", "--batch", "--signal=USR1", "900_3"]]
    failed = control.load_control(state_dir)["safety_hold_drain_intent"]
    assert failed["state"] == "failed"
    assert failed["results"]["cell_usr1:900_3"]["accepted"] is False

    terminal = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-hold1",
                "CANCELLED",
                "asys-schema5-intent:hold1",
                _exact_cell_submission_command("hold1"),
                "sacct",
            ),
        ),
        61.0,
    )
    control.record_alert(
        state_dir,
        kind="scheduler-ambiguity",
        severity="critical",
        message="retry after scheduler reconciliation",
        dedupe_key="monitor:scheduler",
        scheduler=terminal,
        signal_runner=lambda argv: pytest.fail(
            f"terminal target must not be signaled again: {argv}"
        ),
        now=61.0,
    )
    recovered = control.load_control(state_dir)["safety_hold_drain_intent"]
    assert recovered["state"] == "complete"
    assert recovered["attempt_count"] == 2
    assert (
        recovered["results"]["cell_usr1:900_3"]["evidence"]
        == "scheduler_target_no_longer_live"
    )


def test_clean_polls_cannot_clear_transient_hold_before_exact_drain_recovers(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    sbatch_path, snapshot = _hold_drain_scheduler_fixture(state_dir)
    running = control.SchedulerSnapshot((snapshot.jobs[0],), 60.0)

    with pytest.raises(control.ControlError, match="failed safety-hold drain action"):
        control.record_alert(
            state_dir,
            kind="scheduler-query",
            severity="critical",
            message="initial signal failed",
            dedupe_key="monitor:scheduler",
            scheduler=running,
            signal_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, "", "temporary failure"
            ),
            now=60.0,
        )
    control.resolve_alert(
        state_dir, dedupe_key="monitor:scheduler", now=61.0
    )
    for timestamp in (62.0, 63.0):
        with pytest.raises(
            control.ControlError, match="failed safety-hold drain action"
        ):
            control.update_admission_safety_hold(
                state_dir,
                clean_poll=True,
                scheduler=running,
                signal_runner=lambda argv: subprocess.CompletedProcess(
                    argv, 1, "", "still unavailable"
                ),
                now=timestamp,
            )
        persisted = control.load_control(state_dir)
        assert persisted["admission_safety_hold"]["active"] is True
        assert (
            persisted["admission_safety_hold"]["consecutive_clean_polls"]
            == 0
        )
        assert persisted["safety_hold_drain_intent"]["state"] == "failed"
        with pytest.raises(
            control.ControlError, match="complete exact.*drain"
        ):
            control.admission_contract_from_state(state_dir)

    terminal = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "900_3",
                "asys-dispatch-hold1",
                "CANCELLED",
                "asys-schema5-intent:hold1",
                _exact_cell_submission_command("hold1"),
                "sacct",
            ),
        ),
        64.0,
    )
    once = control.update_admission_safety_hold(
        state_dir,
        clean_poll=True,
        scheduler=terminal,
        signal_runner=lambda argv: pytest.fail(
            f"terminal allocation must not be signaled: {argv}"
        ),
        now=64.0,
    )
    assert once["safety_hold_drain_intent"]["state"] == "complete"
    assert once["admission_safety_hold"]["consecutive_clean_polls"] == 1
    cleared = control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=65.0
    )
    assert cleared["admission_safety_hold"]["active"] is False


def test_same_timestamp_hold_reactivation_gets_unique_drain_identity(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    empty = control.SchedulerSnapshot((), 60.0)
    control.record_alert(
        state_dir,
        kind="scheduler-query",
        severity="critical",
        message="first incident",
        dedupe_key="monitor:scheduler",
        scheduler=empty,
        now=60.0,
    )
    first = control.load_control(state_dir)
    first_activation_id = first["admission_safety_hold"]["activation_id"]
    first_drain_id = first["safety_hold_drain_intent"]["drain_id"]
    control.resolve_alert(
        state_dir, dedupe_key="monitor:scheduler", now=60.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=60.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=60.0
    )
    assert control.load_control(state_dir)["admission_safety_hold"]["active"] is False

    _sbatch_path, live = _hold_drain_scheduler_fixture(
        state_dir, captured_at=60.0
    )
    calls: list[list[str]] = []
    control.record_alert(
        state_dir,
        kind="scheduler-query",
        severity="critical",
        message="second incident at the same timestamp",
        dedupe_key="monitor:scheduler",
        scheduler=live,
        signal_runner=lambda argv: (
            calls.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=60.0,
    )
    second = control.load_control(state_dir)
    assert second["admission_safety_hold"]["activation_id"] != first_activation_id
    assert second["safety_hold_drain_intent"]["drain_id"] != first_drain_id
    assert (
        second["safety_hold_drain_intent"]["hold_activation_id"]
        == second["admission_safety_hold"]["activation_id"]
    )
    assert calls == [
        ["scancel", "--batch", "--signal=USR1", "900_3"],
        ["scancel", "900_4"],
    ]


def test_admission_safety_hold_fences_and_clears_by_incident_class(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)

    control.record_alert(
        state_dir,
        kind="scheduler-query",
        severity="critical",
        message="scheduler unavailable",
        dedupe_key="monitor:scheduler",
        scheduler=control.SchedulerSnapshot((), 60.0),
        now=60.0,
    )
    contract = control.admission_contract_from_state(state_dir)
    assert contract["configured_ceiling"] == 24
    assert contract["current_ceiling"] == 0
    assert contract["safety_hold"]["mode"] == "transient"

    control.resolve_alert(state_dir, dedupe_key="monitor:scheduler", now=70.0)
    once = control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=80.0
    )
    assert once["admission_safety_hold"]["active"] is True
    cleared = control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=90.0
    )
    assert cleared["admission_safety_hold"]["active"] is False
    assert control.admission_contract_from_state(state_dir)["current_ceiling"] == 24

    control.record_alert(
        state_dir,
        kind="untrusted-response",
        severity="critical",
        message="provenance mismatch",
        dedupe_key="monitor:untrusted",
        scheduler=control.SchedulerSnapshot((), 100.0),
        now=100.0,
    )
    control.resolve_alert(state_dir, dedupe_key="monitor:untrusted", now=110.0)
    held = control.update_admission_safety_hold(
        state_dir,
        clean_poll=True,
        semantic_scan_clean=True,
        now=120.0,
    )
    assert held["admission_safety_hold"]["mode"] == "integrity"
    assert held["admission_safety_hold"]["semantic_clean_after_activation"] is True
    acknowledged = control.acknowledge_admission_hold(
        state_dir, note="semantic audit clean; incident reviewed", now=130.0
    )
    assert acknowledged["admission_safety_hold"]["active"] is False


def test_dispatcher_restart_recovers_blocked_intent_as_global_integrity_hold(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    batch_id = "20260726T120000-integrity"
    _install_exact_cell_transaction(
        state_dir,
        batch_id=batch_id,
        base_job_id="900",
        task_count=1,
        intent_state="integrity_blocked",
        include_record=False,
        captured_at=55.0,
    )
    ledger = json.loads(
        (state_dir / "ledger.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        control,
        "query_scheduler",
        lambda **_kwargs: control.SchedulerSnapshot((), 60.0),
    )
    monkeypatch.setattr(
        control,
        "_attempt_alert_email",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        dispatcher,
        "_submit_sbatch",
        lambda *_args, **_kwargs: pytest.fail(
            "restart recovery must run before any scientific sbatch"
        ),
    )

    recovered = dispatcher._recover_dispatcher_submission_integrity_hold(
        state_dir,
        ledger=ledger,
        now=60.0,
    )
    assert recovered == (batch_id,)
    persisted = control.load_control(state_dir)
    hold = persisted["admission_safety_hold"]
    assert hold["active"] is True
    assert hold["mode"] == "integrity"
    assert hold["reasons"] == [
        dispatcher.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
    ]
    assert control.admission_contract_from_state(state_dir)[
        "current_ceiling"
    ] == 0

    # Neither clean health polls nor controller restart can auto-clear a blocked
    # scientific-integrity intent.  Every restart reconstructs the same alert before
    # reading a nonzero admission contract.
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=70.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=80.0
    )
    dispatcher._recover_dispatcher_submission_integrity_hold(
        state_dir,
        ledger=ledger,
        now=90.0,
    )
    assert control.admission_contract_from_state(state_dir)[
        "current_ceiling"
    ] == 0


def test_no_progress_hold_closes_epoch_and_requires_progress_clean_scan_and_ack(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    running = control.record_successful_poll(
        state_dir,
        useful_qids=100,
        fleet_generation="fleet-a",
        now=55.0,
    )
    assert running["throughput_epochs"][-1]["closed_at"] is None

    control.record_alert(
        state_dir,
        kind="trusted-qid-progress-stall",
        severity="critical",
        message="two complete semantic scans made no trusted progress",
        dedupe_key="monitor:no-progress",
        scheduler=control.SchedulerSnapshot((), 60.0),
        now=60.0,
    )
    held = control.load_control(state_dir)
    assert held["admission_safety_hold"]["mode"] == "integrity"
    assert held["throughput_epochs"][-1]["close_reason"] == (
        "admission_safety_hold"
    )
    assert held["throughput_epochs"][-1]["closed_timestamp"] == 60.0

    # Clean health polls cannot erase a semantic-progress incident.
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=70.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=80.0
    )
    assert control.load_control(state_dir)["admission_safety_hold"]["active"] is True

    # Actual trusted-QID progress causes the semantic monitor to resolve the alert;
    # its same complete scan supplies the clean evidence, then an explicit review is
    # still required.
    control.resolve_alert(
        state_dir, dedupe_key="monitor:no-progress", now=90.0
    )
    clean = control.update_admission_safety_hold(
        state_dir,
        semantic_scan_clean=True,
        now=90.0,
    )
    assert clean["admission_safety_hold"]["active"] is True
    assert clean["admission_safety_hold"]["semantic_clean_after_activation"] is True
    released = control.acknowledge_admission_hold(
        state_dir,
        note="trusted QIDs advanced and the complete semantic scan is clean",
        now=100.0,
    )
    assert released["admission_safety_hold"]["active"] is False


def test_alert_email_delivery_retries_from_durable_state(tmp_path):
    state_dir, _ = initialize(tmp_path)
    calls = []

    def fail(argv, body):
        calls.append((list(argv), body))
        return subprocess.CompletedProcess(argv, 1, "", "mail unavailable")

    alert = control.record_alert(
        state_dir,
        kind="test-warning",
        severity="warning",
        message="delivery retry",
        dedupe_key="test:email-retry",
        send_email=True,
        mail_runner=fail,
        now=100.0,
    )
    assert alert["email"]["attempt_count"] == 1
    assert alert["email"]["delivered"] is False
    assert alert["email"]["next_retry_timestamp"] == 160.0
    assert control.retry_pending_alert_emails(
        state_dir, mail_runner=fail, now=159.0
    )["attempted"] == []

    def succeed(argv, body):
        calls.append((list(argv), body))
        return subprocess.CompletedProcess(argv, 0, "", "")

    retried = control.retry_pending_alert_emails(
        state_dir, mail_runner=succeed, now=160.0
    )
    assert retried["delivered"] == [alert["alert_id"]]
    persisted = control.load_control(state_dir)["alerts"][0]["email"]
    assert persisted["attempt_count"] == 2
    assert persisted["delivered"] is True
    assert persisted["next_retry_timestamp"] is None


@pytest.mark.parametrize(
    "crash_boundary",
    ("after_alert_journal", "after_transition_journal", "before_control_replace"),
)
def test_critical_alert_crash_boundaries_keep_admission_at_zero(
    tmp_path, monkeypatch, crash_boundary
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    original_append = control._append_jsonl
    original_save = control._save_control

    def crash_after_append(path, record):
        original_append(path, record)
        if (
            crash_boundary == "after_alert_journal"
            and Path(path).name == control.ALERT_JOURNAL
            and record.get("action") == "raised"
        ):
            raise RuntimeError("crash after alert journal")
        if (
            crash_boundary == "after_transition_journal"
            and Path(path).name == control.TRANSITION_JOURNAL
            and record.get("event") == "alert_raised"
        ):
            raise RuntimeError("crash after transition journal")

    def crash_before_control_replace(*args, **kwargs):
        if crash_boundary == "before_control_replace":
            raise RuntimeError("crash before control replace")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(control, "_append_jsonl", crash_after_append)
    monkeypatch.setattr(control, "_save_control", crash_before_control_replace)
    with pytest.raises(RuntimeError, match="crash"):
        control.record_alert(
            state_dir,
            kind="scheduler-ambiguity",
            severity="critical",
            message="ambiguous scheduler truth",
            dedupe_key="test:crash-critical",
            scheduler=control.SchedulerSnapshot((), 60.0),
            now=60.0,
        )

    # The dispatcher acquires the same admission boundary after the failed writer.
    # Journal-ahead critical truth must therefore fence the very next poll.
    with control.admission_boundary_lock(state_dir):
        contract = control.admission_contract_from_state(state_dir)
    assert contract["configured_ceiling"] == 24
    assert contract["current_ceiling"] == 0
    assert contract["active_critical_alerts"] == ["test:crash-critical"]


def test_journal_ahead_alert_retry_resolve_and_clean_does_not_orphan_hold(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    original_save = control._save_control
    crashed = {"value": False}

    def crash_once(*args, **kwargs):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("death before control replace")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(control, "_save_control", crash_once)
    with pytest.raises(RuntimeError, match="death"):
        control.record_alert(
            state_dir,
            kind="scheduler-ambiguity",
            severity="critical",
            message="first journal-ahead raise",
            dedupe_key="test:recovered-critical",
            scheduler=control.SchedulerSnapshot((), 60.0),
            now=60.0,
        )
    monkeypatch.setattr(control, "_save_control", original_save)
    recovered = control.record_alert(
        state_dir,
        kind="scheduler-ambiguity",
        severity="critical",
        message="idempotent retry under a new control alert ID",
        dedupe_key="test:recovered-critical",
        scheduler=control.SchedulerSnapshot((), 61.0),
        now=61.0,
    )
    control.resolve_alert(
        state_dir, dedupe_key="test:recovered-critical", now=62.0
    )
    assert control._active_critical_alerts_from_journal(state_dir) == []
    assert recovered["resolved_at"] is None

    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=63.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, now=64.0
    )
    contract = control.admission_contract_from_state(state_dir)
    assert contract["active_critical_alerts"] == []
    assert contract["safety_hold"]["active"] is False
    assert contract["current_ceiling"] == 24


@pytest.mark.parametrize(
    "corruption",
    ("unknown_action", "resolution_mismatch", "identity_drift", "unseen_resolution"),
)
def test_alert_journal_corruption_fails_admission_closed(
    tmp_path, corruption
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    raised = control.record_alert(
        state_dir,
        kind="fixture-warning",
        severity="warning",
        message="journal parser fixture",
        dedupe_key="test:journal-parser",
        now=60.0,
    )
    common = {
        "at": control.utc_timestamp(61.0),
        "timestamp": 61.0,
        "alert_id": raised["alert_id"],
    }
    if corruption == "unknown_action":
        row = {**common, "action": "unknown"}
    elif corruption == "resolution_mismatch":
        row = {
            **common,
            "action": "resolved",
            "dedupe_key": "test:different",
        }
    elif corruption == "unseen_resolution":
        row = {
            **common,
            "alert_id": "unseen-alert-id",
            "action": "resolved",
            "dedupe_key": "test:journal-parser",
        }
    else:
        row = {
            **common,
            "action": "raised",
            "kind": "changed-kind",
            "severity": "warning",
            "message": "identity drift",
            "dedupe_key": "test:journal-parser",
            "occurrences": 2,
        }
    control._append_jsonl(state_dir / control.ALERT_JOURNAL, row)
    with pytest.raises(control.ControlError, match="alert journal"):
        control.admission_contract_from_state(state_dir)


def test_alert_journal_allows_idempotent_duplicate_resolution(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    raised = control.record_alert(
        state_dir,
        kind="fixture-warning",
        severity="warning",
        message="duplicate resolution fixture",
        dedupe_key="test:duplicate-resolution",
        now=60.0,
    )
    control.resolve_alert(
        state_dir, dedupe_key="test:duplicate-resolution", now=61.0
    )
    control._append_jsonl(
        state_dir / control.ALERT_JOURNAL,
        {
            "at": control.utc_timestamp(62.0),
            "timestamp": 62.0,
            "action": "resolved",
            "alert_id": raised["alert_id"],
            "dedupe_key": "test:duplicate-resolution",
        },
    )
    assert control._active_critical_alerts_from_journal(state_dir) == []
    assert control.admission_contract_from_state(state_dir)["current_ceiling"] == 24


def test_monitor_deadline_sends_term_then_kill_and_persists_failure(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    resume_ready(state_dir, now=50.0)
    state = control.load_control(state_dir)
    for cadence in ("semantic", "daily"):
        state["monitoring"]["cadences"][cadence]["next_due_timestamp"] = 10_000.0
        state["monitoring"]["cadences"][cadence]["next_due_at"] = control.utc_timestamp(
            10_000.0
        )
    control._save_control(state_dir, state, now=50.0)
    attempt = control.begin_monitor_attempt(
        state_dir,
        cadence="health",
        controller_generation=1,
        controller_intent_token="first",
        controller_job_id="700",
        now=50.0,
    )

    class Hung:
        pid = 123

        def __init__(self):
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

    child = Hung()
    signals = []

    def signal_child(process, sig):
        signals.append(sig)
        if sig == control.signal.SIGKILL:
            process.killed = True

    monkeypatch.setattr(control, "_signal_process_group", signal_child)
    monkeypatch.setattr(control, "_monitor_failure_alert", lambda *_a, **_k: None)
    output_path = tmp_path / "health.log"
    output = output_path.open("w", encoding="utf-8")
    processes = {
        "health": control._ManagedMonitorProcess(
            cadence="health",
            attempt_id=attempt["attempt_id"],
            process=child,
            output=output,
            started_timestamp=50.0,
            timeout_seconds=240.0,
        )
    }
    kwargs = {
        "state_dir": state_dir,
        "processes": processes,
        "controller_generation": 1,
        "controller_intent_token": "first",
        "controller_job_id": "700",
    }
    control._service_schema5_monitors(**kwargs, now=290.0)
    control._service_schema5_monitors(**kwargs, now=321.0)
    control._service_schema5_monitors(**kwargs, now=322.0)
    assert signals == [control.signal.SIGTERM, control.signal.SIGKILL]
    assert processes == {}
    row = control.load_control(state_dir)["monitoring"]["cadences"]["health"]
    assert row["consecutive_failures"] == 1
    assert "execution deadline" in row["last_error"]


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
    attest_test_production_authorizations(state_dir, now=29.0)
    current = control.load_control(state_dir)
    current["desired_state"] = "resuming"
    current["resume_intent"] = {
        "state": "submitting_controllers",
        "rollout_generation": 1,
        "production_authorizations": (
            control._production_authorization_resume_binding(current)
        ),
    }
    # This synthetic state tests the drill's desired-state fence, not the separate
    # generation-zero preseal invariant.
    current[control.SNAPSHOT_ATTESTATION_STATE_KEY] = None
    control._save_control(state_dir, current, now=30.0)
    with pytest.raises(control.ControlError, match="paused"):
        control.start_controller_drill(
            state_dir, scheduler=control.SchedulerSnapshot((), 31.0), now=31.0
        )


def test_exact_drill_kill_rejects_spool_proof_mismatch_without_scancel(tmp_path):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, _ = start_live_drill(state_dir)
    state = control.load_drill_state(state_dir)
    active = state["roles"]["dispatcher"]["active"]
    active["spooled_receipt_sha256"] = "0" * 64
    control._save_drill_state(state_dir, state, now=59.0)
    calls = []
    with pytest.raises(control.SchedulerAmbiguity, match="spool/release proof"):
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


def test_external_watchdog_drill_cancels_only_sealed_namespace_and_recovers(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    jobs, snapshot, submit = start_live_drill(state_dir, now=50.0)
    deployment = _watchdog_deployment_evidence(
        state_dir, tmp_path / "DEPLOYMENT_EVIDENCE.json"
    )

    armed = control.arm_external_watchdog_drill(
        state_dir,
        snapshot=snapshot(55.0),
        deployment_evidence_path=deployment,
        now=55.0,
    )
    original_ids = {
        record["job_id"] for record in armed["intent"]["namespace"]
    }
    frozen = control.load_drill_state(state_dir)
    with pytest.raises(control.ControllerFenced, match="froze"):
        control.submit_drill_intent(
            state_dir,
            role="dispatcher",
            target="successor",
            dependency_job_id=frozen["roles"]["dispatcher"]["active"]["job_id"],
            scheduler=snapshot(55.5),
            submit_runner=submit,
            now=55.5,
        )
    cancelled = []

    def cancel(argv):
        assert argv[0] == "scancel"
        assert argv[1] in original_ids
        cancelled.append(argv[1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    first = control.cancel_external_watchdog_drill_namespace(
        state_dir,
        snapshot=snapshot(56.0),
        cancel_runner=cancel,
        now=56.0,
    )
    assert first["state"]["phase"] == "cancelling"
    assert set(cancelled) == original_ids
    assert len(cancelled) == 4

    jobs[:] = [
        control.SchedulerJob(
            job.job_id,
            job.job_name,
            "CANCELLED" if job.job_id in original_ids else job.state,
            job.comment,
            job.command,
            job.source,
            job.dependency,
        )
        for job in jobs
    ]
    reconciled = control.cancel_external_watchdog_drill_namespace(
        state_dir,
        snapshot=snapshot(60.0),
        cancel_runner=lambda argv: pytest.fail(f"duplicate cancel: {argv}"),
        now=60.0,
    )
    assert reconciled["state"]["phase"] == "cancelled"

    first_status = control.external_watchdog_drill_status(
        state_dir,
        snapshot=snapshot(61.0),
        record_observation=True,
        now=61.0,
    )
    second_status = control.external_watchdog_drill_status(
        state_dir,
        snapshot=snapshot(121.0),
        record_observation=True,
        now=121.0,
    )
    assert first_status["desired_state"] == "paused"
    assert first_status["effective_admission_ceiling"] == 0
    assert first_status["watchdog_drill"] == second_status["watchdog_drill"]
    assert all(
        not row["live_active"] and not row["successor_live"]
        for row in second_status["controllers"].values()
    )

    repair = control.repair_external_watchdog_drill(
        state_dir,
        snapshot=snapshot(122.0),
        submit_runner=submit,
        now=122.0,
    )
    assert repair["desired_state"] == "paused"
    assert {row["role"] for row in repair["submitted"]} == set(
        control.ROLE_NAMES
    )
    assert not original_ids & {
        row["job_id"] for row in repair["submitted"]
    }

    for index, role in enumerate(control.ROLE_NAMES):
        drill = control.load_drill_state(state_dir)
        active = drill["roles"][role]["active"]
        control.claim_drill_controller(
            state_dir,
            drill_id=drill["drill_id"],
            role=role,
            generation=active["generation"],
            intent_token=active["intent_token"],
            job_id=active["job_id"],
            active_scheduler_job_ids=snapshot(123.0).active_job_ids,
            now=123.0 + index,
        )
        control.submit_drill_intent(
            state_dir,
            role=role,
            target="successor",
            dependency_job_id=active["job_id"],
            scheduler=snapshot(124.0 + index),
            submit_runner=submit,
            now=124.0 + index,
        )
        control.heartbeat_drill_controller(
            state_dir,
            drill_id=drill["drill_id"],
            role=role,
            generation=active["generation"],
            intent_token=active["intent_token"],
            job_id=active["job_id"],
            now=126.0 + index,
        )

    evidence_path = tmp_path / "DRILL_EVIDENCE.json"
    evidence = control.complete_external_watchdog_drill(
        state_dir,
        snapshot=snapshot(130.0),
        output_path=evidence_path,
        now=130.0,
    )
    assert evidence["passed"] is True
    assert evidence["duplicate_jobs"] == 0
    assert evidence["duplicate_admission_intents"] == 0
    assert evidence["fairness_mutations"] == 0
    assert evidence["namespace_cancellation_recovery_seconds"] == 74.0
    assert evidence_path.stat().st_mode & 0o222 == 0
    loaded = control._load_external_watchdog_drill(state_dir)
    assert loaded is not None and loaded[1]["phase"] == "complete"
    assert (
        control.controller_drill_baseline(
            state_dir, control.load_control(state_dir)
        )
        == loaded[0]["baseline"]
    )


def test_external_watchdog_arm_recovers_marker_first_crash_and_rejects_symlink(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    _jobs, snapshot, _submit = start_live_drill(state_dir, now=50.0)
    deployment = _watchdog_deployment_evidence(
        state_dir, tmp_path / "deployment.json"
    )
    control.arm_external_watchdog_drill(
        state_dir,
        snapshot=snapshot(55.0),
        deployment_evidence_path=deployment,
        now=55.0,
    )
    control._external_watchdog_drill_state_path(state_dir).unlink()
    recovered = control.arm_external_watchdog_drill(
        state_dir,
        snapshot=snapshot(55.0),
        deployment_evidence_path=deployment,
        now=55.0,
    )
    assert recovered["state"]["phase"] == "armed"
    assert control._external_watchdog_drill_state_path(state_dir).is_file()

    other_state, _ = initialize(tmp_path / "other")
    make_ready(other_state)
    _jobs, other_snapshot, _submit = start_live_drill(other_state, now=50.0)
    other_deployment = _watchdog_deployment_evidence(
        other_state, tmp_path / "other-deployment.json"
    )
    alias = tmp_path / "deployment-alias.json"
    alias.symlink_to(other_deployment)
    with pytest.raises(control.ControlError, match="symlink"):
        control.arm_external_watchdog_drill(
            other_state,
            snapshot=other_snapshot(55.0),
            deployment_evidence_path=alias,
            now=55.0,
        )


def test_external_watchdog_deployment_binds_protected_capacity_tag_object(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    make_ready(state_dir)
    _jobs, snapshot, _submit = start_live_drill(state_dir, now=50.0)
    deployment = _watchdog_deployment_evidence(
        state_dir, tmp_path / "deployment-wrong-tag.json"
    )
    value = json.loads(deployment.read_text(encoding="utf-8"))
    value["release_tag_object"] = "c" * 40
    value["evidence_id"] = control._watchdog_self_hash(
        value, "evidence_id"
    )
    deployment.chmod(0o644)
    deployment.write_bytes(control._watchdog_canonical_json(value))
    deployment.chmod(0o444)
    with pytest.raises(control.ControlError, match="paused control"):
        control.arm_external_watchdog_drill(
            state_dir,
            snapshot=snapshot(55.0),
            deployment_evidence_path=deployment,
            now=55.0,
        )


def test_external_watchdog_control_marker_recovers_linked_temp_boundary(
    tmp_path,
):
    target = tmp_path / "INTENT.json"
    value = {"schema_version": 1, "protocol": "fixture"}
    payload = control._watchdog_canonical_json(value)
    interrupted = tmp_path / ".INTENT.json.crashed.publishing"
    interrupted.write_bytes(payload)
    interrupted.chmod(0o444)
    os.link(interrupted, target)
    assert target.stat().st_nlink == 2
    path, digest = control._publish_sealed_json_once(
        target, value, description="test intent"
    )
    assert path == target
    assert digest == hashlib.sha256(payload).hexdigest()
    assert target.stat().st_nlink == 1
    assert not interrupted.exists()


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
    _ensure_resume_generation_catalog(state_dir, now=60.5)
    jobs = []
    identifiers = iter(("910", "911"))

    def scheduler():
        return control.SchedulerSnapshot(tuple(jobs), 120.0)

    def submit(argv):
        job_id = next(identifiers)
        token = next(
            arg.split("=", 1)[1] for arg in argv if arg.startswith("--comment=")
        )
        jobs.append(control.SchedulerJob(job_id, "controller", "PENDING", token))
        return subprocess.CompletedProcess(argv, 0, job_id + "\n", "")

    refresh_test_watchdog_mirror(state_dir, now=120.0)
    resumed = control.resume_control(
        state_dir,
        scheduler_reader=scheduler,
        submit_runner=submit,
        now=120.0,
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
        "production_authorizations": (
            control._production_authorization_resume_binding(current)
        ),
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


def _prepare_capacity_transition_fixture(
    tmp_path: Path,
) -> tuple[Path, dict, control.SchedulerSnapshot]:
    state_dir, _ = initialize(tmp_path)
    # Capacity completion resumes through the normal fail-closed controller
    # path.  Preserve a genuine snapshot preseal so this fixture exercises that
    # path instead of relying on a readiness mock to synthesize snapshot
    # validation state.
    make_ready(state_dir)
    _running, controller_jobs = resume_ready(state_dir, now=28.0)
    live_controllers = control.SchedulerSnapshot(controller_jobs, 30.0)
    control.pause_control(
        state_dir,
        drain=True,
        scheduler=live_controllers,
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        now=30.0,
    )
    current = control.load_control(state_dir)
    transaction_state = (
        Path(current["immutable"]["server_pool_root"])
        / control.fleet_transactions.STATE_DIRECTORY
    )
    transaction_state.mkdir(parents=True, exist_ok=True)
    (transaction_state / control.fleet_transactions.CURRENT_FILENAME).write_text(
        '{"fixture":true}\n', encoding="utf-8"
    )
    old_fleet = control.SchedulerSnapshot(
        (
            control.SchedulerJob("900", "serve-a", "RUNNING"),
            control.SchedulerJob("901", "serve-b", "PENDING"),
        ),
        31.0,
    )
    return state_dir, current, old_fleet


def _capacity_contract_fixture(
    tmp_path: Path, current: dict, monkeypatch
) -> tuple[Path, str, dict[str, object]]:
    source = Path(
        control.effective_fleet_contract_binding(
            current, verify_files=True
        )["path"]
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    profile = payload["profiles"][0]
    prior = profile["replicas"][-1]
    index = len(profile["replicas"])
    added = dict(prior)
    added.update(
        {
            "replica_index": index,
            "replica_id": expected_replica_id(
                profile["serving_profile"], index
            ),
            "scheduler_job_name": expected_scheduler_job_name(
                profile["serving_profile"], index
            ),
        }
    )
    profile["replicas"].append(added)
    payload["logical_replica_count"] = sum(
        len(row["replicas"]) for row in payload["profiles"]
    )
    payload["allocated_gpu_count"] = sum(
        int(row["tensor_parallel_size"]) * len(row["replicas"])
        for row in payload["profiles"]
    )
    path = tmp_path / "capacity_fleet.v2.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha(path)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    path.chmod(0o444)
    marker_path = tmp_path / "PROTECTED_CAPACITY_COMPLETE.g2.json"
    marker_path.write_text('{"fixture":"generation-2"}\n', encoding="utf-8")
    marker_path.chmod(0o444)
    marker_sha256 = _sha(marker_path)
    marker_id = "8" * 64
    certificate_path = tmp_path / "PREFLIGHT_CAPACITY_CERTIFICATE.g2.json"
    certificate_path.write_text('{"fixture":"generation-2"}\n', encoding="utf-8")
    certificate_path.chmod(0o444)
    certificate_sha256 = _sha(certificate_path)
    certificate_id = "9" * 64
    original_load_contract = control.protected_capacity.load_contract
    original_authorize_fleet = control.protected_capacity.authorize_fleet
    proposed_contract = SimpleNamespace(
        path=marker_path.resolve(),
        sha256=marker_sha256,
        marker_id=marker_id,
        release_tag_object=current["immutable"]["release_tag_object"],
        capacity_generation=2,
        effective_fleet_contract_path=path.resolve(),
        effective_fleet_contract_sha256=digest,
        base_fleet_contract_sha256=current["immutable"][
            "fleet_contract_sha256"
        ],
        additive_overlay_contract_path=path.resolve(),
        additive_overlay_contract_sha256=digest,
        static_feasibility_certificate_path=certificate_path.resolve(),
        static_feasibility_certificate_sha256=certificate_sha256,
        static_feasibility_certificate_id=certificate_id,
        static_feasibility_wave_passed=False,
        static_feasibility_selected_cell_count=284,
        static_feasibility_target_cell_count=384,
        static_feasibility_shortfall_cells=100,
        static_feasibility_configured_client_ceiling=384,
        static_feasibility_certified_saturation_target=284,
    )

    def load_capacity_contract(observed_path, **kwargs):
        if Path(observed_path).resolve() == marker_path.resolve():
            assert kwargs["expected_marker_id"] == marker_id
            assert kwargs["expected_sha256"] == marker_sha256
            return proposed_contract
        return original_load_contract(observed_path, **kwargs)

    def authorize_capacity_fleet(fleet, contract):
        if contract is proposed_contract:
            return None
        return original_authorize_fleet(fleet, contract)

    monkeypatch.setattr(
        control.protected_capacity,
        "load_contract",
        load_capacity_contract,
    )
    monkeypatch.setattr(
        control.protected_capacity,
        "authorize_fleet",
        authorize_capacity_fleet,
    )
    return path, digest, {
        "protected_capacity_marker_path": marker_path,
        "protected_capacity_marker_sha256": marker_sha256,
        "protected_capacity_marker_id": marker_id,
        "static_feasibility_certificate_path": certificate_path,
        "static_feasibility_certificate_sha256": certificate_sha256,
        "static_feasibility_certificate_id": certificate_id,
    }


def _full_snapshot_context_from_current_seal(
    state_dir: Path,
) -> control._SnapshotValidationContext:
    """Rehydrate a test full-scan result from the fixture's sealed proof."""

    current = control.load_control(state_dir)
    seal_path = Path(current[control.SNAPSHOT_ATTESTATION_STATE_KEY]["path"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    baseline = json.loads(
        Path(seal["metadata_baseline_path"]).read_text(encoding="utf-8")
    )
    context = control._SnapshotValidationContext(full=True)
    context.proofs = {
        str(row["snapshot_root"]): dict(row) for row in seal["snapshots"]
    }
    context.members = {
        (str(row["snapshot_root"]), str(row["logical_path"])): dict(row)
        for row in seal["sealed_members"]
    }
    context.metadata = {
        str(row["path"]): dict(row) for row in baseline["entries"]
    }
    return context


def test_capacity_transition_is_exact_archived_and_restartable(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(tmp_path)
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    cancelled: list[str] = []

    def cancel(argv):
        assert argv == ["scancel", argv[-1]]
        cancelled.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
        **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {
            "old_fleet_job_ids": ["900", "901"],
            "fleet_generation": 7,
        },
        cancel_runner=cancel,
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "reconciled\n", ""
        ),
        now=31.0,
    )
    assert result["replacement_launch_complete"] is True
    assert result["capacity_generation"] == 2
    assert cancelled == ["900", "901"]
    current = control.load_control(state_dir)
    transition = current["capacity"]["active_transition"]
    assert transition["phase"] == "readiness_pending"
    assert current["capacity"]["current_generation"] == 2
    assert current["admission"]["current_ceiling"] == 24
    assert current["admission_safety_hold"]["active"] is True
    assert current["admission_safety_hold"]["mode"] == "operator"
    assert all(
        current["readiness"][gate]["passed"] is False
        for gate in (
            "static_feasibility_certificate",
            "protected_capacity",
            "fleet",
            "smoke_runs",
            "scheduler_reconciliation",
        )
    )
    archive = Path(transition["archive_path"])
    assert control._verify_capacity_archive(archive)["complete"] is True
    assert not any(path.stat().st_mode & 0o222 for path in archive.rglob("*"))

    # Re-running apply after the generation publication is a read-only status result.
    repeated = control.capacity_transition(
        state_dir,
        action="apply",
        scheduler_reader=lambda: control.SchedulerSnapshot((), 33.0),
        fleet_evidence_reader=lambda _state: pytest.fail(
            "published transition must reuse its durable fleet evidence"
        ),
        cancel_runner=lambda _argv: pytest.fail(
            "published transition must not cancel twice"
        ),
        now=33.0,
    )
    assert repeated["replacement_launch_complete"] is True

    monkeypatch.setattr(
        control,
        "validate_readiness",
        lambda current, **kwargs: current,
    )
    with control.control_lock(state_dir):
        ready = control.load_control(state_dir)
        for gate in (
            "static_feasibility_certificate",
            "protected_capacity",
            "fleet",
            "smoke_runs",
        ):
            ready["readiness"][gate] = {
                "passed": True,
                "evidence": f"/fresh/{gate}",
                "sha256": "f" * 64,
                "attested_at": control.utc_timestamp(34.0),
                "attested_timestamp": 34.0,
                "capacity_generation": 2,
            }
        control._save_control(state_dir, ready, now=34.0)
    assert reconcile_clean(state_dir, now=34.5)["passed"] is True
    completed = control.capacity_transition(
        state_dir,
        action="complete",
        scheduler_reader=lambda: control.SchedulerSnapshot((), 35.0),
        now=35.0,
    )
    assert completed == {
        "action": "complete",
        "completed": True,
        "capacity_generation": 2,
        "resume_ceiling": 24,
        "production_resumed": False,
    }
    final = control.load_control(state_dir)
    assert final["capacity"]["active_transition"] is None
    assert final["capacity"]["history"][-1]["phase"] == "completed"
    assert final["admission_safety_hold"]["active"] is False


def test_effective_capacity_loader_rejects_control_authority_drift(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(
        tmp_path
    )
    contract_path, contract_sha, capacity_authority = (
        _capacity_contract_fixture(tmp_path, initialized, monkeypatch)
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
        **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {
            "old_fleet_job_ids": ["900", "901"]
        },
        cancel_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=31.0,
    )
    current = control.load_control(state_dir)
    loaded = control.load_effective_protected_capacity_contract(current)
    assert loaded.capacity_generation == 2
    authority = control.effective_protected_capacity_binding(current)
    assert authority["static_feasibility_configured_client_ceiling"] == 384
    assert authority["static_feasibility_certified_saturation_target"] == 284
    assert (
        loaded.release_tag_object
        == current["immutable"]["release_tag_object"]
    )

    with control.control_lock(state_dir):
        drifted = control.load_control(state_dir)
        drifted["capacity"]["current_contract"][
            "static_feasibility_certificate_id"
        ] = "7" * 64
        drifted["capacity"]["active_transition"]["new_contract"][
            "static_feasibility_certificate_id"
        ] = "7" * 64
        control._save_control(state_dir, drifted, now=33.0)
    with pytest.raises(
        control.ImmutablePinError,
        match="control record differs from its protected",
    ):
        control.effective_fleet_contract_binding(
            control.load_control(state_dir), verify_files=True
        )


def test_capacity_transition_dry_run_is_non_mutating_and_requires_drain(tmp_path):
    state_dir, initialized = initialize(tmp_path)
    before = (state_dir / control.CONTROL_FILENAME).read_bytes()
    with pytest.raises(control.ControlError, match="pause --drain"):
        control.capacity_transition(
            state_dir,
            action="dry-run",
            scheduler_reader=lambda: control.SchedulerSnapshot((), 20.0),
            fleet_evidence_reader=lambda _state: {"old_fleet_job_ids": []},
            now=20.0,
        )
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() == before
    assert initialized["capacity"]["current_generation"] == 1


def test_capacity_transition_launch_crash_is_fenced_and_restartable(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(tmp_path)
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    cancelled: list[str] = []

    def crash_after_transactional_launch(_argv, _environment):
        raise RuntimeError("injected launcher crash")

    with pytest.raises(RuntimeError, match="injected launcher crash"):
        control.capacity_transition(
            state_dir,
            action="apply",
            fleet_contract_path=contract_path,
            fleet_contract_sha256=contract_sha,
            **capacity_authority,
            scheduler_reader=lambda: next(snapshots),
            fleet_evidence_reader=lambda _state: {
                "old_fleet_job_ids": ["900", "901"]
            },
            cancel_runner=lambda argv: (
                cancelled.append(argv[-1])
                or subprocess.CompletedProcess(argv, 0, "", "")
            ),
            fleet_launch_runner=crash_after_transactional_launch,
            now=31.0,
        )
    crashed = control.load_control(state_dir)
    assert crashed["capacity"]["active_transition"]["phase"] == "fleet_launch_pending"
    assert crashed["admission_safety_hold"]["active"] is True
    assert cancelled == ["900", "901"]

    recovered = control.capacity_transition(
        state_dir,
        action="apply",
        scheduler_reader=lambda: control.SchedulerSnapshot((), 33.0),
        fleet_evidence_reader=lambda _state: pytest.fail(
            "retry must adopt the durable retirement/contract intent"
        ),
        cancel_runner=lambda _argv: pytest.fail("old IDs must not be cancelled twice"),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "adopted durable intents\n", ""
        ),
        now=33.0,
    )
    assert recovered["replacement_launch_complete"] is True
    transition = control.load_control(state_dir)["capacity"]["active_transition"]
    assert transition["phase"] == "readiness_pending"
    assert transition["fleet_launch_attempts"] == 2


@pytest.mark.parametrize(
    "marker_name",
    (
        control.CAPACITY_ARCHIVE_COMPLETE,
        control.CAPACITY_CONTRACT_COMPLETE,
        "FLEET_TRANSACTION_STATE_RETIRED.json",
    ),
)
def test_capacity_transition_recovers_readonly_markers_after_link_crash(
    tmp_path, monkeypatch, marker_name
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(
        tmp_path
    )
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    cancelled = {"value": False}

    def scheduler():
        return (
            control.SchedulerSnapshot((), 32.0)
            if cancelled["value"]
            else old_fleet
        )

    def cancel(argv):
        cancelled["value"] = True
        return subprocess.CompletedProcess(argv, 0, "", "")

    original_link = control.os.link
    injected = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if Path(destination).name == marker_name and not injected["value"]:
            injected["value"] = True
            raise RuntimeError(f"death after readonly link of {marker_name}")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="death after readonly link"):
            control.capacity_transition(
                state_dir,
                action="apply",
                fleet_contract_path=contract_path,
                fleet_contract_sha256=contract_sha,
            **capacity_authority,
                scheduler_reader=scheduler,
                fleet_evidence_reader=lambda _state: {
                    "old_fleet_job_ids": ["900", "901"]
                },
                cancel_runner=cancel,
                fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
                    argv, 0, "", ""
                ),
                now=31.0,
            )
    transition_roots = list((state_dir / "capacity-transitions").iterdir())
    assert len(transition_roots) == 1
    pre_replay_matches = [
        path
        for path in transition_roots[0].rglob("*")
        if path.name == marker_name
    ]
    assert len(pre_replay_matches) == 1
    assert pre_replay_matches[0].stat().st_nlink == 1
    assert pre_replay_matches[0].stat().st_mode & 0o222 == 0
    marker_preimage = pre_replay_matches[0].read_bytes()

    recovered = control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
        **capacity_authority,
        scheduler_reader=scheduler,
        fleet_evidence_reader=lambda _state: {
            "old_fleet_job_ids": ["900", "901"]
        },
        cancel_runner=cancel,
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=32.0,
    )
    assert recovered["replacement_launch_complete"] is True
    marker_matches = [
        path
        for path in transition_roots[0].rglob("*")
        if path.name == marker_name
    ]
    assert len(marker_matches) == 1
    assert marker_matches[0].stat().st_mode & 0o222 == 0
    assert marker_matches[0].read_bytes() == marker_preimage


def test_capacity_transition_requires_fresh_generation_gates_before_complete(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(tmp_path)
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
            **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {"old_fleet_job_ids": ["900", "901"]},
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=31.0,
    )
    snapshot_context = _full_snapshot_context_from_current_seal(state_dir)
    monkeypatch.setattr(
        control, "validate_readiness", lambda current, **kwargs: snapshot_context
    )
    with pytest.raises(control.ReadinessError, match="generation-bound gates"):
        control.capacity_transition(
            state_dir,
            action="complete",
            scheduler_reader=lambda: control.SchedulerSnapshot((), 34.0),
            now=34.0,
        )
    current = control.load_control(state_dir)
    assert current["desired_state"] == "paused"
    assert current["admission_safety_hold"]["active"] is True
    assert current["throughput_epochs"] == []


def test_capacity_transition_does_not_clear_mid_transition_integrity_incident(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(tmp_path)
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
            **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {"old_fleet_job_ids": ["900", "901"]},
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=31.0,
    )
    control.update_admission_safety_hold(
        state_dir,
        active_critical_keys=["monitor:corrupt"],
        now=32.0,
    )
    with control.control_lock(state_dir):
        ready = control.load_control(state_dir)
        for gate in (
            "static_feasibility_certificate",
            "protected_capacity",
            "fleet",
            "smoke_runs",
        ):
            ready["readiness"][gate] = {
                "passed": True,
                "evidence": f"/fresh/{gate}",
                "sha256": "e" * 64,
                "attested_at": control.utc_timestamp(33.0),
                "attested_timestamp": 33.0,
                "capacity_generation": 2,
            }
        control._save_control(state_dir, ready, now=33.0)
    assert reconcile_clean(state_dir, now=33.5)["passed"] is True
    monkeypatch.setattr(control, "validate_readiness", lambda current, **kwargs: current)
    with pytest.raises(control.ControlError, match="non-capacity reasons"):
        control.capacity_transition(
            state_dir,
            action="complete",
            scheduler_reader=lambda: control.SchedulerSnapshot((), 34.0),
            now=34.0,
        )
    held = control.load_control(state_dir)["admission_safety_hold"]
    assert "monitor:corrupt" in held["reasons"]

    control.update_admission_safety_hold(
        state_dir, clean_poll=True, semantic_scan_clean=True, now=35.0
    )
    control.acknowledge_admission_hold(
        state_dir, note="clean post-incident semantic scan", now=36.0
    )
    layered = control.load_control(state_dir)["admission_safety_hold"]
    assert layered["active"] is True
    assert layered["mode"] == "operator"
    assert layered["reasons"][0].startswith("capacity-transition:")


def test_capacity_transition_layers_preexisting_throughput_hold(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(
        tmp_path
    )
    control.record_alert(
        state_dir,
        kind="capacity-gate-failure",
        severity="critical",
        message="48-hour throughput target missed",
        dedupe_key="monitor:throughput",
        now=30.5,
    )
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    result = control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
        **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {
            "old_fleet_job_ids": ["900", "901"]
        },
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=31.0,
    )
    assert result["replacement_launch_complete"] is True
    held = control.load_control(state_dir)["admission_safety_hold"]
    assert held["active"] is True
    assert held["mode"] == "operator"
    assert held["reasons"] == [
        "capacity-transition:"
        + control.load_control(state_dir)["capacity"]["active_transition"][
            "transition_id"
        ]
    ]
    current = control.load_control(state_dir)
    transition = current["capacity"]["active_transition"]
    assert transition["consumed_capacity_incidents"] == [
        "monitor:throughput"
    ]
    retirement_intent = json.loads(
        Path(transition["retirement_intent_path"]).read_text(encoding="utf-8")
    )
    assert retirement_intent["schema_version"] == 2
    assert retirement_intent["consumed_capacity_incidents"] == [
        "monitor:throughput"
    ]
    throughput_alerts = [
        alert
        for alert in current["alerts"]
        if alert["dedupe_key"] == "monitor:throughput"
    ]
    assert len(throughput_alerts) == 1
    assert throughput_alerts[0]["resolved_at"] == control.utc_timestamp(31.0)
    consumption = [
        row
        for row in current["transition_history"]
        if row["event"] == "capacity_remediation_incidents_consumed"
    ]
    assert consumption[-1]["details"]["dedupe_keys"] == [
        "monitor:throughput"
    ]


def test_capacity_transition_replays_incident_consumption_exactly_once(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(
        tmp_path
    )
    for dedupe_key, kind in (
        ("monitor:throughput", "capacity-gate-failure"),
        ("monitor:corrupt", "semantic-corruption"),
    ):
        control.record_alert(
            state_dir,
            kind=kind,
            severity="critical",
            message=f"fixture incident {dedupe_key}",
            dedupe_key=dedupe_key,
            now=30.5,
        )
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )

    original_save = control._save_control
    injected = {"value": False}

    def crash_before_control_replace(path, payload, *, now):
        active = payload["capacity"].get("active_transition")
        if (
            not injected["value"]
            and isinstance(active, dict)
            and active.get("phase") == "retirement_requested"
        ):
            injected["value"] = True
            raise RuntimeError("crash before capacity control replacement")
        return original_save(path, payload, now=now)

    monkeypatch.setattr(control, "_save_control", crash_before_control_replace)
    with pytest.raises(RuntimeError, match="capacity control replacement"):
        control.capacity_transition(
            state_dir,
            action="apply",
            fleet_contract_path=contract_path,
            fleet_contract_sha256=contract_sha,
            **capacity_authority,
            scheduler_reader=lambda: old_fleet,
            fleet_evidence_reader=lambda _state: {
                "old_fleet_job_ids": ["900", "901"]
            },
            cancel_runner=lambda argv: subprocess.CompletedProcess(
                argv, 0, "", ""
            ),
            fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
                argv, 0, "", ""
            ),
            now=31.0,
        )

    # The marker-first intent survives while the atomic control replacement does not.
    crashed = control.load_control(state_dir)
    assert crashed["capacity"]["active_transition"] is None
    assert set(crashed["admission_safety_hold"]["reasons"]) == {
        "monitor:throughput",
        "monitor:corrupt",
    }
    transition_roots = list((state_dir / "capacity-transitions").iterdir())
    assert len(transition_roots) == 1
    intent = json.loads(
        (
            transition_roots[0] / control.CAPACITY_RETIREMENT_INTENT
        ).read_text(encoding="utf-8")
    )
    assert intent["consumed_capacity_incidents"] == ["monitor:throughput"]

    monkeypatch.setattr(control, "_save_control", original_save)
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    recovered = control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
            **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {
            "old_fleet_job_ids": ["900", "901"]
        },
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=32.0,
    )
    assert recovered["replacement_launch_complete"] is True

    current = control.load_control(state_dir)
    active = current["capacity"]["active_transition"]
    assert active["consumed_capacity_incidents"] == ["monitor:throughput"]
    assert set(current["admission_safety_hold"]["reasons"]) == {
        f"capacity-transition:{active['transition_id']}",
        "monitor:corrupt",
    }
    by_key = {alert["dedupe_key"]: alert for alert in current["alerts"]}
    assert by_key["monitor:throughput"]["resolved_at"] == control.utc_timestamp(
        31.0
    )
    assert by_key["monitor:corrupt"]["resolved_at"] is None
    consumed_events = [
        row
        for row in current["transition_history"]
        if row["event"] == "capacity_remediation_incidents_consumed"
    ]
    committed_events = [
        row
        for row in current["transition_history"]
        if row["event"] == "capacity_transition_retirement_committed"
    ]
    assert len(consumed_events) == 1
    assert len(committed_events) == 1
    alert_resolutions = [
        row
        for row in control._read_jsonl_locked(state_dir / control.ALERT_JOURNAL)
        if row.get("action") == "resolved"
        and row.get("capacity_transition_id") == active["transition_id"]
    ]
    assert [
        (row["dedupe_key"], row["alert_id"]) for row in alert_resolutions
    ] == [
        (
            "monitor:throughput",
            by_key["monitor:throughput"]["alert_id"],
        )
    ]


@pytest.mark.parametrize(
    "dedupe_key", ["monitor:ramp-stall", "monitor:throughput"]
)
def test_capacity_gate_holds_never_auto_clear_or_accept_ack(
    tmp_path, dedupe_key
):
    state_dir, _ = initialize(tmp_path)
    control.record_alert(
        state_dir,
        kind="capacity-gate-failure",
        severity="critical",
        message="capacity evidence gate failed",
        dedupe_key=dedupe_key,
        now=10.0,
    )
    control.resolve_alert(state_dir, dedupe_key=dedupe_key, now=11.0)
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, semantic_scan_clean=True, now=12.0
    )
    control.update_admission_safety_hold(
        state_dir, clean_poll=True, semantic_scan_clean=True, now=13.0
    )
    held = control.load_control(state_dir)["admission_safety_hold"]
    assert held["active"] is True
    assert held["mode"] == "integrity"
    assert held["reasons"] == [dedupe_key]
    with pytest.raises(
        control.ControlError, match="controlled capacity-transition"
    ):
        control.acknowledge_admission_hold(
            state_dir, note="must not bypass capacity remediation", now=14.0
        )


def test_capacity_transition_explicit_completion_resumes_at_24_and_opens_epoch(
    tmp_path, monkeypatch
):
    state_dir, initialized, old_fleet = _prepare_capacity_transition_fixture(tmp_path)
    contract_path, contract_sha, capacity_authority = _capacity_contract_fixture(
        tmp_path, initialized, monkeypatch
    )
    snapshots = iter((old_fleet, control.SchedulerSnapshot((), 32.0)))
    control.capacity_transition(
        state_dir,
        action="apply",
        fleet_contract_path=contract_path,
        fleet_contract_sha256=contract_sha,
            **capacity_authority,
        scheduler_reader=lambda: next(snapshots),
        fleet_evidence_reader=lambda _state: {"old_fleet_job_ids": ["900", "901"]},
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_launch_runner=lambda argv, _env: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        now=31.0,
    )
    with control.control_lock(state_dir):
        ready = control.load_control(state_dir)
        for gate in (
            "static_feasibility_certificate",
            "protected_capacity",
            "fleet",
            "smoke_runs",
            "scheduler_reconciliation",
        ):
            evidence_path = tmp_path / f"fresh-{gate}.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "gate": gate,
                        "capacity_generation": 2,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            evidence_path.chmod(0o444)
            gate_record = {
                "passed": True,
                "evidence": str(evidence_path.resolve()),
                "sha256": _sha(evidence_path),
                "attested_at": control.utc_timestamp(33.0),
                "attested_timestamp": 33.0,
                "capacity_generation": 2,
            }
            if gate == "fleet":
                (
                    scheduler_evidence,
                    scheduler_policy,
                    _requirements,
                ) = _fleet_scheduler_safety(ready)
                transport = ready["immutable"][
                    "transport_uncertainty_binding"
                ]
                gate_record.update(
                    {
                        "scheduler_safety_evidence_id": (
                            scheduler_evidence["evidence_id"]
                        ),
                        "scheduler_safety_policy_id": scheduler_policy[
                            "policy_id"
                        ],
                        "scheduler_safety_policy_contract_id": (
                            scheduler_policy["policy_contract_id"]
                        ),
                        "transport_uncertainty_binding_sha256": ready[
                            "immutable"
                        ][
                            "transport_uncertainty_binding_sha256"
                        ],
                        "transport_censor_protocol_version": transport[
                            "transport_censor_protocol_version"
                        ],
                        "transport_censor_protocol_hash": transport[
                            "transport_censor_protocol_hash"
                        ],
                    }
                )
            ready["readiness"][gate] = gate_record
        control._save_control(state_dir, ready, now=33.0)
    assert reconcile_clean(state_dir, now=33.5)["passed"] is True
    snapshot_context = _full_snapshot_context_from_current_seal(state_dir)
    monkeypatch.setattr(
        control, "validate_readiness", lambda current, **kwargs: snapshot_context
    )
    first = control.capacity_transition(
        state_dir,
        action="complete",
        scheduler_reader=lambda: control.SchedulerSnapshot((), 34.0),
        now=34.0,
    )
    assert first["production_resumed"] is False
    pending = control.load_control(state_dir)
    assert pending["desired_state"] == "paused"
    assert pending["capacity"]["history"][-1]["production_resume_state"] == "pending"
    assert pending["throughput_epochs"] == []

    # This transition fixture deliberately replaces the expensive semantic fleet
    # attestation validator above.  Preserve that boundary while still supplying
    # immutable bytes for the generation catalog's marker-last evidence copy.
    control_evidence_path = tmp_path / "capacity-control-evidence.json"
    control_evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "schema5_test_capacity_control_evidence",
                "capacity_generation": 2,
                "rollout_generation": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    control_evidence_path.chmod(0o444)
    monkeypatch.setattr(
        control,
        "_trusted_generation_control_evidence",
        lambda *_args, **_kwargs: control.GenerationEvidence(
            kind="control_generation",
            path=control_evidence_path.resolve(),
            sha256=_sha(control_evidence_path),
        ),
    )

    with pytest.raises(control.ControlError, match="pending production resume"):
        control.capacity_transition(
            state_dir,
            action="apply",
            fleet_contract_path=contract_path,
            fleet_contract_sha256=contract_sha,
            **capacity_authority,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 35.0),
            fleet_evidence_reader=lambda _state: pytest.fail(
                "a second transition must be fenced before fleet inspection"
            ),
            now=35.0,
        )
    still_pending = control.load_control(state_dir)
    assert still_pending["capacity"]["current_generation"] == 2
    assert len(still_pending["capacity"]["history"]) == 1

    # The fixture does not run controller processes that would publish their exits.
    # Represent the prior generation through complete accounting truth and advance
    # beyond submission visibility grace so it is deterministically superseded.
    resume_at = 36.0 + control.SUBMISSION_VISIBILITY_GRACE_SECONDS
    prior_accounting = tuple(
        control.SchedulerJob(
            str(record["job_id"]),
            f"old-{role}",
            "COMPLETED",
            str(record["job_token"]),
        )
        for role in control.ROLE_NAMES
        if isinstance(
            (record := pending["controllers"][role].get("active")), dict
        )
    )
    refresh_test_watchdog_mirror(state_dir, now=resume_at)
    running, _jobs = resume_ready(
        state_dir, now=resume_at, accounting_jobs=prior_accounting
    )
    assert running["desired_state"] == "running"
    finished = control.capacity_transition(
        state_dir,
        action="complete",
        scheduler_reader=lambda: control.SchedulerSnapshot((), resume_at + 1.0),
        resume_callback=lambda path: control.load_control(path),
        now=resume_at + 1.0,
    )
    assert finished["production_resumed"] is True
    final = control.load_control(state_dir)
    assert final["admission"]["current_ceiling"] == 24
    assert final["capacity"]["history"][-1]["production_resume_state"] == "complete"
    assert len(final["throughput_epochs"]) == 1
    assert final["throughput_epochs"][0]["samples"] == []
    assert final["throughput_epochs"][0]["fleet_generation"].startswith(
        "capacity-g000002-"
    )


def _finalizer_test_catalog(current: dict) -> control.TrustedGenerationCatalog:
    state_dir = (
        Path(current["immutable"]["results_root"]).resolve()
        / control.CONTROL_STATE_DIRNAME
    )
    catalog_id = "c" * 64
    marker_path = (
        state_dir
        / "_finalizer-test-trusted-generation"
        / catalog_id
        / control.CATALOG_MARKER
    )
    if not marker_path.is_file():
        raise AssertionError("finalizer trusted-generation fixture is not installed")
    identity = (
        current["immutable"]["fleet_contract_sha256"],
        current["immutable"]["fleet_contract_sha256"],
        1,
        1,
        "fixture-endpoint-generation",
    )
    return control.TrustedGenerationCatalog(
        marker_path=marker_path.resolve(),
        marker_sha256=_sha(marker_path),
        inventory_sha256="d" * 64,
        catalog_sha256="e" * 64,
        catalog_id=catalog_id,
        entries=(),
        generations=(),
        allowed_generation_tuples=frozenset({identity}),
    )


def _trusted_catalog_binding(
    catalog: control.TrustedGenerationCatalog,
) -> dict:
    return {
        "catalog_id": catalog.catalog_id,
        "marker_path": str(catalog.marker_path),
        "marker_sha256": catalog.marker_sha256,
        "inventory_sha256": catalog.inventory_sha256,
        "catalog_payload_sha256": catalog.catalog_sha256,
        "allowed_generation_tuple_count": len(
            catalog.allowed_generation_tuples
        ),
    }


def _install_finalizer_test_catalog(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> control.TrustedGenerationCatalog:
    current = control.load_control(state_dir)
    marker_path = (
        state_dir
        / "_finalizer-test-trusted-generation"
        / ("c" * 64)
        / control.CATALOG_MARKER
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "kind": "schema5_finalizer_test_catalog",
                "catalog_id": "c" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_path.chmod(0o444)
    marker_path.parent.chmod(0o555)
    trusted = _finalizer_test_catalog(current)

    def validate(marker, *, server_pool_root):
        del server_pool_root
        observed = Path(marker).resolve()
        if (
            not observed.is_file()
            or observed.read_bytes() != trusted.marker_path.read_bytes()
        ):
            raise control.GenerationCatalogError(
                "finalizer fixture catalog marker drifted"
            )
        return control.TrustedGenerationCatalog(
            marker_path=observed,
            marker_sha256=_sha(observed),
            inventory_sha256=trusted.inventory_sha256,
            catalog_sha256=trusted.catalog_sha256,
            catalog_id=trusted.catalog_id,
            entries=trusted.entries,
            generations=trusted.generations,
            allowed_generation_tuples=trusted.allowed_generation_tuples,
        )

    monkeypatch.setattr(
        control,
        "refresh_trusted_generation_catalog",
        lambda *_args, **_kwargs: trusted,
    )
    monkeypatch.setattr(
        control, "validate_trusted_generation_catalog", validate
    )
    # Synthetic manifest rows intentionally omit production ExperimentCell fields.
    # Finalizer-worker tests exercise the lock barrier through explicit focused
    # cases below; the common fixture therefore supplies the quiescent baseline.
    monkeypatch.setattr(
        control, "_active_manifest_cell_locks", lambda _control: []
    )
    return trusted


def _complete_semantic_report(current: dict) -> dict:
    pins = {
        str(row["run_id"]): row for row in current["immutable"]["runs"]
    }
    runs = {}
    for run_id, cells in control.REQUIRED_RUNS.items():
        qids = control.REQUIRED_RUN_QIDS[run_id]
        pin = pins[run_id]
        runs[run_id] = {
            "run_root": str(Path(pin["run_root"]).resolve()),
            "manifest_sha256": pin["manifest_sha256"],
            "benchmark_contract_sha256": pin["benchmark_contract_sha256"],
            "artifact_policy_sha256": pin["policy_sha256"],
            "manifest_cells": cells,
            "expected_qids": qids,
            "states": {"complete": cells},
            "outcomes": {
                "validated_qids": qids,
                "useful_qids": qids,
                "completed_qids": qids,
                "length_censored_qids": 0,
                "protocol_censored_qids": 0,
                "transport_censored_qids": 0,
                "transport_affected_qids": 0,
                "topology_transport_censored_coordinates": 0,
                "transport_censored_coordinates": 0,
                "auxiliary_outcomes": 0,
                "auxiliary_completed": 0,
                "auxiliary_length_censored": 0,
                "auxiliary_protocol_censored": 0,
                "auxiliary_transport_censored": 0,
                "malformed_lines": 0,
                "duplicate_qids": 0,
                "unexpected_qids": 0,
                "invalid_rows": 0,
                "untrusted_valid_rows": 0,
            },
            "artifact_schema_counts": {"5": qids},
            "contract_errors": [],
            "stale_unmanifested_dirs": 0,
        }
    return {
        "semantic": {
            "scan_successful": True,
            "scan_errors": [],
            "trusted_generation_catalog": _trusted_catalog_binding(
                _finalizer_test_catalog(current)
            ),
            "transport_censor_protocol": {
                "version": control.TRANSPORT_CENSOR_PROTOCOL_VERSION,
                "hash": control.TRANSPORT_CENSOR_PROTOCOL_HASH,
            },
            "runs": runs,
            "states": {"complete": control.EXPECTED_TOTAL_CELLS},
            "outcomes": {
                "validated_qids": control.EXPECTED_TOTAL_QIDS,
                "useful_qids": control.EXPECTED_TOTAL_QIDS,
                "completed_qids": control.EXPECTED_TOTAL_QIDS,
                "length_censored_qids": 0,
                "protocol_censored_qids": 0,
                "transport_censored_qids": 0,
                "transport_affected_qids": 0,
                "topology_transport_censored_coordinates": 0,
                "transport_censored_coordinates": 0,
                "auxiliary_outcomes": 0,
                "auxiliary_completed": 0,
                "auxiliary_length_censored": 0,
                "auxiliary_protocol_censored": 0,
                "auxiliary_transport_censored": 0,
                "malformed_lines": 0,
                "duplicate_qids": 0,
                "unexpected_qids": 0,
                "invalid_rows": 0,
                "untrusted_valid_rows": 0,
                "stale_unmanifested_dirs": 0,
                "contract_errors": 0,
            },
            "artifact_schema_counts": {"5": control.EXPECTED_TOTAL_QIDS},
        },
        "final_acceptance": {
            "passed": True,
            "checks": {
                "exact_complete_cells": True,
                "no_noncomplete_cells": True,
                "exact_validated_qids": True,
                "top_level_partition_exact": True,
                "auxiliary_partition_exact": True,
                "useful_transport_partition_exact": True,
                "transport_coordinate_partition_exact": True,
                "topology_transport_coordinate_qid_bound": True,
                "transport_affected_qid_bounds_exact": True,
                "transport_protocol_exact": True,
                "zero_integrity_errors": True,
                "schema5_only": True,
                "throughput_projection_acceptable": True,
            },
        },
    }


def test_final_semantic_accepts_exact_transport_cardinality_and_useful_partition(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    run_id = "full_sweep_schema5_v1"
    run = report["semantic"]["runs"][run_id]["outcomes"]
    aggregate = report["semantic"]["outcomes"]

    # One completed top-level QID depends on one ambiguous auxiliary draw.  It remains
    # trusted cardinality, but is removed from useful throughput exactly once.
    for outcomes in (run, aggregate):
        outcomes["useful_qids"] -= 1
        outcomes["transport_affected_qids"] = 1
        outcomes["transport_censored_coordinates"] = 1
        outcomes["auxiliary_outcomes"] = 1
        outcomes["auxiliary_transport_censored"] = 1
    control._validate_final_semantic_report(report)


def test_final_semantic_accepts_one_exact_topology_transport_censor(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    run_id = "full_sweep_schema5_v1"
    run = report["semantic"]["runs"][run_id]["outcomes"]
    aggregate = report["semantic"]["outcomes"]

    for outcomes in (run, aggregate):
        outcomes["useful_qids"] -= 1
        outcomes["completed_qids"] -= 1
        outcomes["transport_censored_qids"] = 1
        outcomes["transport_affected_qids"] = 1
        outcomes["topology_transport_censored_coordinates"] = 1
        outcomes["transport_censored_coordinates"] = 1
    control._validate_final_semantic_report(report)


def test_final_semantic_accepts_concurrent_topology_transport_censors(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    run_id = "full_sweep_schema5_v1"
    run = report["semantic"]["runs"][run_id]["outcomes"]
    aggregate = report["semantic"]["outcomes"]

    # A concurrent terminal wave can durably retain two independently ambiguous
    # sibling attempts while map-order propagation selects one QID-level terminal.
    for outcomes in (run, aggregate):
        outcomes["useful_qids"] -= 1
        outcomes["completed_qids"] -= 1
        outcomes["transport_censored_qids"] = 1
        outcomes["transport_affected_qids"] = 1
        outcomes["topology_transport_censored_coordinates"] = 2
        outcomes["transport_censored_coordinates"] = 2
    control._validate_final_semantic_report(report)


def test_final_semantic_rejects_transport_partition_or_protocol_drift(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    report["semantic"]["outcomes"]["transport_affected_qids"] = 1
    with pytest.raises(control.ControlError, match="useful/transport"):
        control._validate_final_semantic_report(report)

    report = _complete_semantic_report(current)
    report["semantic"]["transport_censor_protocol"]["hash"] = "f" * 64
    with pytest.raises(control.ControlError, match="transport-censor protocol"):
        control._validate_final_semantic_report(report)


def test_final_semantic_rejects_duplicate_transport_coordinate_accounting(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    run_id = "full_sweep_schema5_v1"
    run = report["semantic"]["runs"][run_id]["outcomes"]
    aggregate = report["semantic"]["outcomes"]
    for outcomes in (run, aggregate):
        outcomes["useful_qids"] -= 1
        outcomes["transport_affected_qids"] = 1
        outcomes["transport_censored_coordinates"] = 2
        outcomes["auxiliary_outcomes"] = 1
        outcomes["auxiliary_transport_censored"] = 1
    with pytest.raises(control.ControlError, match="coordinate partition"):
        control._validate_final_semantic_report(report)


def test_final_semantic_rejects_aggregate_transport_drift_from_run_sums(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    aggregate = report["semantic"]["outcomes"]
    aggregate["useful_qids"] -= 1
    aggregate["transport_affected_qids"] = 1
    aggregate["transport_censored_coordinates"] = 1
    aggregate["auxiliary_outcomes"] = 1
    aggregate["auxiliary_transport_censored"] = 1
    with pytest.raises(control.ControlError, match="exact run sum"):
        control._validate_final_semantic_report(report)


@pytest.mark.parametrize(
    ("scope", "mutation", "match"),
    [
        ("aggregate", "omit", "aggregate outcome fields are not exact"),
        ("aggregate", "extra", "aggregate outcome fields are not exact"),
        ("aggregate", "bool", "is not a non-negative integer"),
        (
            "run",
            "omit",
            "full_sweep_schema5_v1 outcome fields are not exact",
        ),
        (
            "run",
            "extra",
            "full_sweep_schema5_v1 outcome fields are not exact",
        ),
        ("run", "bool", "is not a non-negative integer"),
    ],
)
def test_final_semantic_outcome_schemas_are_closed_and_typed(
    tmp_path, monkeypatch, scope, mutation, match
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    outcomes = report["semantic"]["outcomes"]
    if scope == "run":
        outcomes = report["semantic"]["runs"][
            "full_sweep_schema5_v1"
        ]["outcomes"]
    if mutation == "omit":
        outcomes.pop("malformed_lines")
    elif mutation == "extra":
        outcomes["unregistered_outcome"] = 0
    else:
        outcomes["malformed_lines"] = False

    with pytest.raises(control.ControlError, match=match):
        control._validate_final_semantic_report(report)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("omit-check", "check names or types are not exact"),
        ("extra-check", "check names or types are not exact"),
        ("wrong-check-type", "check names or types are not exact"),
        ("forged-check", "independently recomputed terminal acceptance"),
        ("forged-passed", "did not pass every acceptance check"),
    ],
)
def test_final_semantic_recomputes_exact_registered_acceptance_checks(
    tmp_path, monkeypatch, mutation, match
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    report = _complete_semantic_report(current)
    acceptance = report["final_acceptance"]
    checks = acceptance["checks"]
    if mutation == "omit-check":
        checks.pop("exact_validated_qids")
    elif mutation == "extra-check":
        checks["unregistered_check"] = True
    elif mutation == "wrong-check-type":
        checks["exact_validated_qids"] = 1
    elif mutation == "forged-check":
        checks["throughput_projection_acceptable"] = False
    else:
        acceptance["passed"] = False

    with pytest.raises(control.ControlError, match=match):
        control._validate_final_semantic_report(report)


def _write_complete_primary_cache(cache_root: Path, current: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "items_v1.parquet",
        "agents_v1.parquet",
        "cells_dedup_v1.parquet",
        "incident_items_scientifically_excluded_v1.parquet",
        "incident_agents_scientifically_excluded_v1.parquet",
    ):
        (cache_root / filename).write_bytes(b"fixture\n")
    immutable = current["immutable"]
    marker = {
        "analysis_mode": "primary-schema5",
        "mixed_protocol": False,
        "exact_token_policy": "schema5-required",
        "run_ids": list(control.REQUIRED_RUNS),
        "manifest_scoped": True,
        "manifest_cells_by_run": control.REQUIRED_RUNS,
        "n_cell_dirs": control.EXPECTED_TOTAL_CELLS,
        "n_cells_complete": control.EXPECTED_TOTAL_CELLS,
        "completion_states": {"complete": control.EXPECTED_TOTAL_CELLS},
        "validated_qids_by_completion_state": {
            "complete": control.EXPECTED_TOTAL_QIDS
        },
        "artifact_schema_counts": {"5": control.EXPECTED_TOTAL_QIDS},
        "n_bad_lines": 0,
        "n_bad_cells": 0,
        "n_dupes_dropped": 0,
        "n_cells_with_dupes": 0,
        "n_unexpected_n": 0,
        "unexpected_n_cells": [],
        "qid_mismatch_cells": [],
        "run_contracts": {run_id: {} for run_id in control.REQUIRED_RUNS},
        "stale_dirs_by_run": {run_id: 0 for run_id in control.REQUIRED_RUNS},
        "missing_dirs_by_run": {run_id: 0 for run_id in control.REQUIRED_RUNS},
        "supplementary_preservation": {
            "active_artifact_validated_qids": control.EXPECTED_TOTAL_QIDS,
            "sealed_scientifically_excluded_qids": 0,
            "unmanifested_opt_in_qids_not_in_acceptance": 0,
        },
        "analysis_runtime": {
            "release_worktree": str(Path(immutable["release_worktree"]).resolve()),
            "harness_prefix": str(
                Path(immutable["harness_environment_prefix"]).resolve()
            ),
            "immutable_pins_sha256": current["immutable_sha256"],
            "model_contract_sha256": immutable["model_contract_sha256"],
            "release_id": immutable["release_id"],
            "git_commit": immutable["git_commit"],
            "source_tree_sha256": immutable["source_tree_sha256"],
        },
        "trusted_generation_catalog": {
            **_trusted_catalog_binding(_finalizer_test_catalog(current)),
            "server_pool_root": str(
                Path(immutable["server_pool_root"]).resolve()
            ),
            "validated_qids": control.EXPECTED_TOTAL_QIDS,
            "observed_coordinate_generation_tuple_count": 1,
            "observed_calibration_endpoint_count": 1,
            "passed": True,
        },
    }
    marker["cache_contract_sha256"] = "9" * 64
    marker["cache_artifacts"] = {
        filename: {
            "sha256": _sha(cache_root / filename),
            "size": (cache_root / filename).stat().st_size,
        }
        for filename in (
            "items_v1.parquet",
            "agents_v1.parquet",
            "cells_dedup_v1.parquet",
            "incident_items_scientifically_excluded_v1.parquet",
            "incident_agents_scientifically_excluded_v1.parquet",
        )
    }
    marker["cache_generation_id"] = control.sha256_value(
        {
            "analysis_mode": marker["analysis_mode"],
            "run_ids": marker["run_ids"],
            "cache_contract_sha256": marker["cache_contract_sha256"],
            "cache_artifacts": marker["cache_artifacts"],
        }
    )
    (cache_root / "ingest_manifest_v1.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_primary_cache_build_marker(cache_root: Path) -> Path:
    path = cache_root / ".cache_generation_in_progress.json"
    path.write_text(
        json.dumps(
            {
                "cache_build_schema_version": 1,
                "status": "in_progress",
                "analysis_mode": "primary-schema5",
                "run_ids": list(control.REQUIRED_RUNS),
                "pid": 12345,
                "started_at": 39.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_primary_cache_adopts_manifest_published_before_transaction_cleanup(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    trusted = _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    final_root = tmp_path / "final"
    cache_root = final_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    _write_complete_primary_cache(cache_root, current)
    in_progress = _write_primary_cache_build_marker(cache_root)
    previous = cache_root / "ingest_manifest_v1.previous.json"
    previous.write_text('{"generation":"superseded"}\n', encoding="utf-8")

    adopted = control._resume_primary_analysis_cache_publication(
        cache_root,
        final_root=final_root,
        control=current,
        trusted_catalog=trusted,
        now=40.0,
    )
    assert adopted is not None
    assert not in_progress.exists()
    assert not previous.exists()
    marker, marker_sha256 = control._validate_primary_analysis_cache(
        cache_root,
        current,
        trusted_catalog=trusted,
    )
    assert adopted == (marker, marker_sha256)
    publication_id = control.sha256_value(
        {
            "cache_root": str(cache_root.resolve()),
            "cache_generation_id": marker["cache_generation_id"],
        }
    )
    recovery_root = (
        final_root
        / control.FINAL_ANALYSIS_CACHE_PUBLICATION_RECOVERY_DIRNAME
        / publication_id
    )
    assert (
        recovery_root / "RECOVERY_COMPLETE.json"
    ).stat().st_mode & 0o222 == 0
    assert {
        path.name for path in (recovery_root / "preimage").iterdir()
    } == {
        ".cache_generation_in_progress.json",
        "ingest_manifest_v1.previous.json",
    }
    assert recovery_root.stat().st_mode & 0o222 == 0


def test_primary_cache_publication_replays_crash_between_metadata_removals(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    trusted = _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    final_root = tmp_path / "final"
    cache_root = final_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    _write_complete_primary_cache(cache_root, current)
    in_progress = _write_primary_cache_build_marker(cache_root)
    previous = cache_root / "ingest_manifest_v1.previous.json"
    previous.write_text('{"generation":"superseded"}\n', encoding="utf-8")
    real_remove = control.io.remove_file
    crashed = {"value": False}

    def remove_then_crash(path):
        real_remove(path)
        if Path(path).name == previous.name and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("death after previous-manifest removal")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.io, "remove_file", remove_then_crash)
        with pytest.raises(RuntimeError, match="previous-manifest"):
            control._resume_primary_analysis_cache_publication(
                cache_root,
                final_root=final_root,
                control=current,
                trusted_catalog=trusted,
                now=40.0,
            )
    assert not previous.exists()
    assert in_progress.exists()

    recovered = control._resume_primary_analysis_cache_publication(
        cache_root,
        final_root=final_root,
        control=current,
        trusted_catalog=trusted,
        now=999.0,
    )
    assert recovered is not None
    assert not in_progress.exists()
    control._validate_primary_analysis_cache(
        cache_root,
        current,
        trusted_catalog=trusted,
    )


def test_partial_primary_cache_is_archived_and_reopened_idempotently(tmp_path):
    final_root = tmp_path / "final"
    cache_root = final_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    cache_root.mkdir(parents=True)
    (cache_root / "items_v1.parquet").write_bytes(b"partial\n")
    _write_primary_cache_build_marker(cache_root)

    recovered = control._recover_interrupted_primary_analysis_cache(
        cache_root,
        final_root=final_root,
        now=40.0,
    )
    assert recovered is not None
    assert list(cache_root.iterdir()) == []
    replay = control._recover_interrupted_primary_analysis_cache(
        cache_root,
        final_root=final_root,
        now=999.0,
    )
    assert replay is None
    recovery_root = (
        final_root / control.FINAL_ANALYSIS_CACHE_RECOVERY_DIRNAME
    )
    marker_paths = list(recovery_root.glob("*/RECOVERY_COMPLETE.json"))
    assert len(marker_paths) == 1
    assert marker_paths[0].stat().st_mode & 0o222 == 0


def test_partial_primary_cache_replays_marker_first_crash_before_rename(
    tmp_path, monkeypatch
):
    final_root = tmp_path / "final"
    cache_root = final_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    cache_root.mkdir(parents=True)
    (cache_root / "items_v1.parquet").write_bytes(b"partial\n")
    _write_primary_cache_build_marker(cache_root)
    original_replace = control.os.replace
    crashed = {"value": False}

    def crash_before_rename(source, destination):
        if Path(source) == cache_root and not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("death before primary-cache rename")
        return original_replace(source, destination)

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "replace", crash_before_rename)
        with pytest.raises(RuntimeError, match="before primary-cache rename"):
            control._recover_interrupted_primary_analysis_cache(
                cache_root,
                final_root=final_root,
                now=40.0,
            )

    recovery_roots = list(
        (final_root / control.FINAL_ANALYSIS_CACHE_RECOVERY_DIRNAME).iterdir()
    )
    assert len(recovery_roots) == 1
    assert (recovery_roots[0] / "RECOVERY_INTENT.json").is_file()
    assert not (recovery_roots[0] / "preimage").exists()
    assert (cache_root / "items_v1.parquet").is_file()

    recovered = control._recover_interrupted_primary_analysis_cache(
        cache_root,
        final_root=final_root,
        now=999.0,
    )
    assert recovered is None
    assert (recovery_roots[0] / "RECOVERY_COMPLETE.json").is_file()
    assert list(cache_root.iterdir()) == []


def test_partial_primary_cache_marker_first_replay_rejects_source_drift(
    tmp_path, monkeypatch
):
    final_root = tmp_path / "final"
    cache_root = final_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    cache_root.mkdir(parents=True)
    item = cache_root / "items_v1.parquet"
    item.write_bytes(b"partial\n")
    _write_primary_cache_build_marker(cache_root)
    original_replace = control.os.replace

    with monkeypatch.context() as boundary:
        boundary.setattr(
            control.os,
            "replace",
            lambda source, destination: (
                (_ for _ in ()).throw(RuntimeError("death before rename"))
                if Path(source) == cache_root
                else original_replace(source, destination)
            ),
        )
        with pytest.raises(RuntimeError, match="before rename"):
            control._recover_interrupted_primary_analysis_cache(
                cache_root,
                final_root=final_root,
                now=40.0,
            )
    item.write_bytes(b"tampered\n")
    with pytest.raises(control.ControlError, match="source changed"):
        control._recover_interrupted_primary_analysis_cache(
            cache_root,
            final_root=final_root,
            now=41.0,
        )


def test_independent_archive_copy_recovers_partial_owned_temp(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.json"
    destination = tmp_path / "archive" / "source.json"
    source.write_bytes(b"x" * 4096)
    original_write = control.os.write
    injected = {"value": False}

    def partial_write_then_die(descriptor, payload):
        if not injected["value"] and payload:
            injected["value"] = True
            original_write(descriptor, payload[:37])
            raise RuntimeError("death during archive copy")
        return original_write(descriptor, payload)

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "write", partial_write_then_die)
        with pytest.raises(RuntimeError, match="during archive copy"):
            control._write_independent_archive_copy(source, destination)
    leftovers = list(destination.parent.glob(f".{destination.name}.publish.*.tmp"))
    assert len(leftovers) == 1
    assert leftovers[0].stat().st_mode & 0o222

    control._write_independent_archive_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o222 == 0
    assert destination.stat().st_nlink == 1
    assert list(destination.parent.glob(f".{destination.name}.publish.*.tmp")) == []


def _final_fleet_evidence(
    current: dict, allocations: list[dict] | None = None, *, captured_at: float = 40.0
) -> dict:
    fleet_binding = control.effective_fleet_contract_binding(
        current,
        verify_files=True,
    )
    pool_id, _, _, _ = control._fleet_contract_reconciliation_identity(
        current
    )
    rows = [] if allocations is None else allocations
    return {
        "pool_root": str(
            Path(current["immutable"]["server_pool_root"]).resolve()
        ),
        "pool_id": pool_id,
        "fleet_sha256": fleet_binding["sha256"],
        "ledger_generation": 1,
        "scheduler_captured_timestamp": captured_at,
        "active_allocations": rows,
        "old_fleet_job_ids": sorted(
            [str(row["job_id"]) for row in rows], key=int
        ),
    }


def _single_finalizer_fleet_fixture(current: dict):
    contract = json.loads(
        Path(current["immutable"]["fleet_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    profile, replica = next(
        (profile, replica)
        for profile in contract["profiles"]
        for replica in profile["replicas"]
    )
    intent_token = "1" * 32
    job_id = "900"
    job = control.SchedulerJob(
        job_id,
        replica["scheduler_job_name"],
        "RUNNING",
        (
            "asys-s5-fleet:"
            f"pool={contract['fleet_id']};"
            f"replica={replica['replica_id']};"
            f"generation=1;intent={intent_token};"
            f"fleet={current['immutable']['fleet_contract_sha256']}"
        ),
    )
    sbatch_path = (
        Path(current["immutable"]["release_worktree"]) / "slurm" / "common.sh"
    ).resolve()
    allocation = {
        "job_id": job_id,
        "replica_id": replica["replica_id"],
        "profile": profile["serving_profile"],
        "ledger_generation": 1,
        "scheduler_state": "RUNNING",
        "intent_token": intent_token,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": _sha(sbatch_path),
    }
    active_ids = {job_id}
    captured_at = [40.0]

    def scheduler_reader():
        captured_at[0] += 1.0
        return control.SchedulerSnapshot(
            (job,) if job_id in active_ids else (),
            captured_at[0],
        )

    def evidence_reader(state):
        return _final_fleet_evidence(
            state,
            [allocation] if job_id in active_ids else [],
            captured_at=captured_at[0],
        )

    return job_id, active_ids, scheduler_reader, evidence_reader


def test_final_control_writer_exclusion_blocks_auxiliary_writers_and_unwinds(
    tmp_path,
):
    state_dir = tmp_path / "state"
    monitor_lock = state_dir / "monitoring" / ".persist.lock"
    watchdog_lock = (
        state_dir / "locks" / "external-watchdog-mirror.lock"
    )
    started = [threading.Event(), threading.Event()]
    acquired = [threading.Event(), threading.Event()]

    def contender(index: int, path: Path) -> None:
        started[index].set()
        with control._file_lock(path):
            acquired[index].set()

    threads = [
        threading.Thread(
            target=contender,
            args=(0, monitor_lock),
        ),
        threading.Thread(
            target=contender,
            args=(1, watchdog_lock),
        ),
    ]
    with control._final_control_writer_exclusion(state_dir):
        for thread in threads:
            thread.start()
        assert all(event.wait(timeout=2.0) for event in started)
        assert not any(event.wait(timeout=0.1) for event in acquired)
        # The direct cut builder nests this guard under the final writer guard.
        with control._final_control_writer_exclusion(state_dir):
            assert not any(event.is_set() for event in acquired)
    for thread in threads:
        thread.join(timeout=2.0)
    assert all(not thread.is_alive() for thread in threads)
    assert all(event.is_set() for event in acquired)

    # Finalization owns admission before this reverse-order try-lock.  If the
    # watchdog writer wins first, finalization must fail immediately and release
    # the monitor lock it acquired earlier in the ExitStack.
    with control._file_lock(watchdog_lock):
        started_at = time.monotonic()
        with pytest.raises(
            control.ControlError,
            match="active monitor/watchdog writer",
        ):
            with control._final_control_writer_exclusion(state_dir):
                pytest.fail("contended final cut must not enter")
        assert time.monotonic() - started_at < 1.0
        with control._file_lock(monitor_lock, nonblocking=True):
            pass


def test_final_watchdog_validation_rejects_receipt_without_intent(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    mirror = state_dir / control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME
    mirror.mkdir(parents=True)
    receipt_path = mirror / "cycle_receipts" / (
        f"{1:012d}-{'1' * 64}.json"
    )
    receipt_path.parent.mkdir()
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt_path.chmod(0o444)
    monkeypatch.setattr(
        control, "_load_watchdog_status_observations", lambda *_a: []
    )
    monkeypatch.setattr(
        control, "_load_watchdog_action_intents", lambda *_a: []
    )
    monkeypatch.setattr(
        control, "_load_watchdog_action_receipts", lambda *_a: []
    )
    monkeypatch.setattr(
        control, "_load_watchdog_cycle_intents", lambda *_a: []
    )
    monkeypatch.setattr(
        control,
        "_load_watchdog_cycle_receipts",
        lambda *_a: [({"sequence": 1}, receipt_path)],
    )

    with pytest.raises(
        control.ControlError,
        match="watchdog cycle journal is incomplete",
    ):
        control._final_watchdog_journal_validation(
            state_dir,
            control={"readiness": {}},
        )


def _write_final_snapshot_fixture(
    snapshot_root: Path,
    sources,
    *,
    completed_at: str = "1970-01-01T00:00:40Z",
    control_bindings: dict | None = None,
) -> dict:
    expected_control_bindings = (
        {"fixture": True}
        if control_bindings is None
        else dict(control_bindings)
    )

    def synthetic_control_cut(directory: Path) -> None:
        watchdog_journal = {
            "schema_version": 1,
            "kind": "schema5_final_watchdog_journal_validation",
            "required": False,
            "present": False,
            "status_receipt_count": 0,
            "action_intent_count": 0,
            "action_receipt_count": 0,
            "cycle_intent_count": 0,
            "cycle_receipt_count": 0,
            "latest_sequence": None,
            "latest_receipt_id": None,
            "latest_pointer_id": None,
            "files": [],
        }
        watchdog_journal["validation_id"] = control.sha256_value(
            watchdog_journal
        )
        projection = {
            "schema_version": 1,
            "kind": "schema5_final_control_plane_projection",
            "immutable_sha256": "a" * 64,
            "rollout_generation": 1,
            "external_watchdog_journal": watchdog_journal,
            "finalization": {"intent_id": None},
        }
        intent = {
            "schema_version": 1,
            "kind": "schema5_final_control_plane_cut_intent",
            "source_root": str(directory.resolve()),
            "immutable_sha256": "a" * 64,
            "capacity_generation": 1,
            "rollout_generation": 1,
            "finalization_intent_id": None,
            "bindings": expected_control_bindings,
            "control_projection": projection,
            "included_files": [],
            "mutable_preimages": [],
            "excluded_files": [],
            "created_at": control.utc_timestamp(40.0),
            "created_timestamp": 40.0,
        }
        intent["intent_id"] = control.sha256_value(intent)
        (directory / control.FINAL_CONTROL_CUT_INTENT_FILENAME).write_text(
            json.dumps(intent, sort_keys=True) + "\n", encoding="utf-8"
        )
        (
            directory / control.FINAL_CONTROL_CUT_PROJECTION_FILENAME
        ).write_text(
            json.dumps(projection, sort_keys=True) + "\n", encoding="utf-8"
        )
        (
            directory / control.FINAL_CONTROL_CUT_EXCLUSIONS_FILENAME
        ).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "schema5_final_control_plane_exclusions",
                    "files": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rows, inventory_sha256 = control._directory_inventory(directory)
        marker = {
            "schema_version": 1,
            "kind": "schema5_final_control_plane_cut",
            "complete": True,
            "intent_id": intent["intent_id"],
            "intent_sha256": _sha(
                directory / control.FINAL_CONTROL_CUT_INTENT_FILENAME
            ),
            "bindings": intent["bindings"],
            "inventory_sha256": inventory_sha256,
            "file_count": len(rows),
            "completed_at": control.utc_timestamp(40.0),
            "completed_timestamp": 40.0,
        }
        marker["completion_id"] = control.sha256_value(marker)
        (
            directory / control.FINAL_CONTROL_CUT_COMPLETE_FILENAME
        ).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in directory.iterdir():
            path.chmod(0o444)

    snapshot_root.mkdir(parents=True)
    source_rows = []
    inventory_rows = []
    directories = []
    total_bytes = 0
    for name, source in sorted(sources):
        source_rows.append(
            {"name": str(name), "path": str(Path(source).resolve())}
        )
        directory = snapshot_root / str(name)
        directory.mkdir()
        directories.append(str(name))
        if name in {"trusted_generation_catalog", "control"}:
            source_root = Path(source).resolve()
            if (
                name == "control"
                and (
                    not source_root.is_dir()
                    or not (
                        source_root
                        / control.FINAL_CONTROL_CUT_COMPLETE_FILENAME
                    ).is_file()
                )
            ):
                synthetic_control_cut(directory)
            else:
                for source_path in sorted(source_root.rglob("*")):
                    relative = source_path.relative_to(source_root)
                    target = directory / relative
                    if source_path.is_dir():
                        target.mkdir()
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_path, target)
                    target.chmod(0o444)
            for target in sorted(directory.rglob("*")):
                if target.is_dir():
                    target.chmod(0o555)
                    directories.append(
                        target.relative_to(snapshot_root).as_posix()
                    )
                    continue
                total_bytes += target.stat().st_size
                inventory_rows.append(
                    f"{_sha(target)}  "
                    f"{target.relative_to(snapshot_root).as_posix()}\n"
                )
        else:
            payload = f"{name}\n".encode()
            target = directory / "payload.bin"
            target.write_bytes(payload)
            target.chmod(0o444)
            total_bytes += len(payload)
            inventory_rows.append(
                f"{_sha(target)}  {name}/payload.bin\n"
            )
        directory.chmod(0o555)
    directories.sort()
    inventory_rows.sort(key=lambda row: row.partition("  ")[2])
    inventory = "".join(inventory_rows).encode()
    inventory_sha256 = hashlib.sha256(inventory).hexdigest()
    snapshot_id = (
        f"{control.FINAL_SNAPSHOT_ID_PREFIX}-{inventory_sha256[:16]}"
    )
    (snapshot_root / "SOURCE_INVENTORY.sha256").write_bytes(inventory)
    (snapshot_root / "SNAPSHOT_INVENTORY.sha256").write_bytes(inventory)
    (snapshot_root / "DIRECTORY_INVENTORY.txt").write_text(
        "".join(f"{name}\n" for name in directories),
        encoding="utf-8",
    )
    catalog = {
        "schema_version": control.FINAL_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "snapshot_kind": control.FINAL_SNAPSHOT_KIND,
        "created_at": completed_at,
        "copy_contract": "independent_regular_files_no_hardlinks_no_symlinks",
        "sources": source_rows,
        "file_count": len(inventory_rows),
        "directory_count": len(directories),
        "total_bytes": total_bytes,
        "source_inventory_sha256": inventory_sha256,
        "snapshot_inventory_sha256": inventory_sha256,
        "elapsed_seconds": 0.01,
    }
    (snapshot_root / "SNAPSHOT_CATALOG.json").write_text(
        json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "schema_version": control.FINAL_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "snapshot_kind": control.FINAL_SNAPSHOT_KIND,
        "completed_at": completed_at,
        "file_count": len(inventory_rows),
        "total_bytes": total_bytes,
        "snapshot_inventory_sha256": inventory_sha256,
        "verified": True,
        "read_only": True,
    }
    for name in (
        "SOURCE_INVENTORY.sha256",
        "SNAPSHOT_INVENTORY.sha256",
        "DIRECTORY_INVENTORY.txt",
        "SNAPSHOT_CATALOG.json",
    ):
        (snapshot_root / name).chmod(0o444)
    (snapshot_root / "SNAPSHOT_COMPLETE.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    (snapshot_root / "SNAPSHOT_COMPLETE.json").chmod(0o444)
    snapshot_root.chmod(0o555)
    return marker


@pytest.mark.parametrize("mutation", ("unlisted", "catalog_schema", "symlink"))
def test_final_snapshot_recursive_verifier_rejects_unlisted_and_unsafe_members(
    tmp_path, mutation
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    sources = []
    for name in sorted(control.FINAL_SNAPSHOT_REQUIRED_SOURCES):
        path = source_root / name
        path.write_bytes(b"source\n")
        sources.append((name, path))
    snapshot_root = tmp_path / f"snapshot-{mutation}"
    _write_final_snapshot_fixture(
        snapshot_root, sources
    )
    control._validate_final_snapshot_marker(snapshot_root)

    snapshot_root.chmod(0o755)
    if mutation == "unlisted":
        extra = snapshot_root / "unlisted.bin"
        extra.write_bytes(b"not catalogued\n")
        extra.chmod(0o444)
    elif mutation == "catalog_schema":
        catalog_path = snapshot_root / "SNAPSHOT_CATALOG.json"
        catalog_path.chmod(0o644)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog["unlisted_field"] = True
        catalog_path.write_text(
            json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8"
        )
        catalog_path.chmod(0o444)
    else:
        (snapshot_root / "unsafe-link").symlink_to(
            snapshot_root / "SNAPSHOT_COMPLETE.json"
        )
    snapshot_root.chmod(0o555)
    with pytest.raises(control.ControlError, match="snapshot"):
        control._validate_final_snapshot_marker(snapshot_root)


def test_final_snapshot_verifier_rejects_self_consistent_pre_repair_snapshot(
    tmp_path,
):
    from scripts import create_recovery_snapshot as recovery_snapshot

    source_root = tmp_path / "pre-repair-sources"
    source_root.mkdir()
    sources = []
    for name in sorted(control.FINAL_SNAPSHOT_REQUIRED_SOURCES):
        path = source_root / name
        path.write_bytes(f"{name}\n".encode("utf-8"))
        sources.append(recovery_snapshot.Source(name, path.resolve()))
    snapshot_root = tmp_path / "self-consistent-pre-repair"
    recovery_snapshot.create_snapshot(snapshot_root, sources)
    recovery_snapshot.verify_snapshot(snapshot_root)

    with pytest.raises(
        control.ControlError,
        match="final snapshot completion marker is invalid",
    ):
        control._validate_final_snapshot_marker(snapshot_root)


def test_final_snapshot_bound_replay_rejects_foreign_origins_and_control_cut(
    tmp_path,
):
    source_root = tmp_path / "canonical-sources"
    source_root.mkdir()
    sources = []
    for name in sorted(control.FINAL_SNAPSHOT_REQUIRED_SOURCES):
        path = source_root / name
        path.write_bytes(f"{name}\n".encode("utf-8"))
        sources.append((name, path.resolve()))
    bindings = {
        "immutable_sha256": "a" * 64,
        "capacity_generation": 1,
    }
    snapshot_root = tmp_path / "bound-final-snapshot"
    _write_final_snapshot_fixture(
        snapshot_root,
        sources,
        control_bindings=bindings,
    )

    control._validate_final_snapshot_marker(
        snapshot_root,
        expected_sources=sources,
        expected_control_cut_bindings=bindings,
    )

    foreign_root = tmp_path / "foreign-sources"
    foreign_root.mkdir()
    foreign_sources = []
    for name, source in sources:
        foreign = foreign_root / name
        foreign.write_bytes(source.read_bytes())
        foreign_sources.append((name, foreign.resolve()))
    with pytest.raises(
        control.FinalizerProvenanceError,
        match="source origins",
    ):
        control._validate_final_snapshot_marker(
            snapshot_root,
            expected_sources=foreign_sources,
            expected_control_cut_bindings=bindings,
        )
    with pytest.raises(control.ControlError, match="control-plane cut"):
        control._validate_final_snapshot_marker(
            snapshot_root,
            expected_sources=sources,
            expected_control_cut_bindings={
                **bindings,
                "capacity_generation": 2,
            },
        )


def _mutate_live_control_as_successor(state_dir: Path, *, now: float) -> None:
    with control.control_lock(state_dir):
        current = control.load_control(state_dir)
        control.append_transition(
            state_dir,
            current,
            event="test_successor_promoted_after_snapshot_cut",
            details={"attempt": int(now)},
            now=now,
        )
        control._save_control(state_dir, current, now=now)


def test_real_snapshot_resume_uses_immutable_control_cut_across_successor_mutation(
    tmp_path, monkeypatch
):
    from scripts import create_recovery_snapshot as recovery_snapshot

    state_dir, current = initialize(tmp_path)
    final_root = tmp_path / "final-control-cut"
    bindings = {
        "immutable_sha256": current["immutable_sha256"],
        "fixture_generation": 1,
    }
    cut_source, cut_marker, _ = control._ensure_final_control_plane_cut(
        state_dir,
        final_root=final_root,
        bindings=bindings,
        now=40.0,
    )
    assert cut_marker["bindings"] == bindings
    cut_intent = json.loads(
        (
            cut_source / control.FINAL_CONTROL_CUT_INTENT_FILENAME
        ).read_text(encoding="utf-8")
    )
    mutable_rows = {
        row["path"]: row for row in cut_intent["mutable_preimages"]
    }
    watchdog_proof = cut_intent["control_projection"][
        "external_watchdog_journal"
    ]
    assert watchdog_proof["validation_id"] == control.sha256_value(
        {
            key: value
            for key, value in watchdog_proof.items()
            if key != "validation_id"
        }
    )
    assert watchdog_proof["files"] == sorted(
        [
            {
                "path": row["path"],
                "size": row["size"],
                "sha256": row["sha256"],
            }
            for row in [
                *cut_intent["included_files"],
                *cut_intent["mutable_preimages"],
            ]
            if row["path"].startswith(
                control.EXTERNAL_WATCHDOG_MIRROR_DIRNAME + "/"
            )
        ],
        key=lambda row: row["path"],
    )
    assert control.CONTROL_FILENAME in mutable_rows
    sealed_control_preimage = (
        cut_source
        / "mutable_preimages"
        / control.CONTROL_FILENAME
    )
    assert sealed_control_preimage.is_file()
    live_before = (state_dir / control.CONTROL_FILENAME).read_bytes()
    assert sealed_control_preimage.read_bytes() == live_before
    snapshot_root = tmp_path / "real-resumable-snapshot"
    sources = [recovery_snapshot.Source("control", cut_source)]
    original_copy = recovery_snapshot._copy_one
    interrupted = {"value": False}

    def copy_projection_then_die(source, destination, logical):
        result = original_copy(source, destination, logical)
        if (
            logical
            == f"control/{control.FINAL_CONTROL_CUT_PROJECTION_FILENAME}"
            and not interrupted["value"]
        ):
            interrupted["value"] = True
            raise RuntimeError("finalizer allocation expired after control cut copy")
        return result

    with monkeypatch.context() as boundary:
        boundary.setattr(
            recovery_snapshot, "_copy_one", copy_projection_then_die
        )
        with pytest.raises(RuntimeError, match="allocation expired"):
            recovery_snapshot.create_snapshot(snapshot_root, sources)
    copied_projection = (
        snapshot_root
        / "control"
        / control.FINAL_CONTROL_CUT_PROJECTION_FILENAME
    )
    assert copied_projection.is_file()
    copied_before = copied_projection.read_bytes()
    sealed_control_before = sealed_control_preimage.read_bytes()

    _mutate_live_control_as_successor(state_dir, now=41.0)
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() != live_before
    completed = recovery_snapshot.create_snapshot(snapshot_root, sources)
    assert completed["status"] == "created"
    recovery_snapshot.verify_snapshot(snapshot_root)
    assert copied_projection.read_bytes() == copied_before
    snapshotted_control_preimage = (
        snapshot_root
        / "control"
        / "mutable_preimages"
        / control.CONTROL_FILENAME
    )
    assert snapshotted_control_preimage.read_bytes() == sealed_control_before
    assert snapshotted_control_preimage.read_bytes() == live_before
    control._validate_final_control_plane_cut(snapshot_root / "control")


def test_control_cut_replays_orphaned_intent_and_rejects_payload_tamper(
    tmp_path, monkeypatch
):
    state_dir, current = initialize(tmp_path)
    final_root = tmp_path / "final-control-cut"
    bindings = {
        "immutable_sha256": current["immutable_sha256"],
        "fixture_generation": 2,
    }
    original_copy = control._write_independent_archive_copy
    crashed = {"value": False}

    def die_before_first_copy(source, destination):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("death after control-cut intent")
        return original_copy(source, destination)

    with monkeypatch.context() as boundary:
        boundary.setattr(
            control, "_write_independent_archive_copy", die_before_first_copy
        )
        with pytest.raises(RuntimeError, match="control-cut intent"):
            control._ensure_final_control_plane_cut(
                state_dir,
                final_root=final_root,
                bindings=bindings,
                now=40.0,
            )
    intent_path = (
        final_root
        / control.FINAL_CONTROL_CUT_DIRNAME
        / control.FINAL_CONTROL_CUT_INTENT_FILENAME
    )
    assert intent_path.is_file()
    assert intent_path.stat().st_mode & 0o222 == 0

    _mutate_live_control_as_successor(state_dir, now=41.0)
    cut_source, _marker, _marker_sha256 = (
        control._ensure_final_control_plane_cut(
            state_dir,
            final_root=final_root,
            bindings=bindings,
            now=999.0,
        )
    )
    cut_root = final_root / control.FINAL_CONTROL_CUT_DIRNAME
    marker_path = (
        cut_source / control.FINAL_CONTROL_CUT_COMPLETE_FILENAME
    )
    cut_root.chmod(0o755)
    cut_source.chmod(0o755)
    stale_links = []
    for target in (intent_path, marker_path):
        payload_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        stale = target.parent / (
            f".{target.name}.publish.{payload_sha256}."
            f"{'1' * 32}.tmp"
        )
        os.link(target, stale)
        stale_links.append(stale)
        assert target.stat().st_nlink == 2
    cut_source, _marker, _marker_sha256 = (
        control._ensure_final_control_plane_cut(
            state_dir,
            final_root=final_root,
            bindings=bindings,
            now=999.5,
        )
    )
    assert all(not stale.exists() for stale in stale_links)
    assert intent_path.stat().st_nlink == 1
    assert marker_path.stat().st_nlink == 1

    projection = cut_source / control.FINAL_CONTROL_CUT_PROJECTION_FILENAME
    cut_source.chmod(0o755)
    projection.chmod(0o644)
    projection.write_bytes(projection.read_bytes() + b" ")
    with pytest.raises(control.ControlError, match="control-plane cut"):
        control._ensure_final_control_plane_cut(
            state_dir,
            final_root=final_root,
            bindings=bindings,
            now=1000.0,
        )


def test_final_fleet_retirement_accepts_active_allocations_from_sealed_generations(
    tmp_path,
):
    state_dir, _ = initialize(tmp_path)
    current = control.load_control(state_dir)
    contract = json.loads(
        Path(current["immutable"]["fleet_contract_path"]).read_text(encoding="utf-8")
    )
    replicas = [
        (profile, replica)
        for profile in contract["profiles"]
        for replica in profile["replicas"]
    ][:2]
    sbatch_path = (
        Path(current["immutable"]["release_worktree"]) / "slurm" / "common.sh"
    ).resolve()
    allocations = [
        {
            "job_id": str(900 + index),
            "replica_id": replica["replica_id"],
            "profile": profile["serving_profile"],
            "ledger_generation": generation,
            "scheduler_state": "RUNNING",
            "intent_token": f"{index + 1:032x}",
            "sbatch_path": str(sbatch_path),
            "sbatch_sha256": _sha(sbatch_path),
        }
        for index, (generation, (profile, replica)) in enumerate(
            zip((1, 2), replicas, strict=True)
        )
    ]
    evidence = _final_fleet_evidence(current, allocations)
    evidence["ledger_generation"] = 2

    assert control._validate_final_fleet_retirement_evidence(
        current, evidence
    ) == ["900", "901"]

    future = json.loads(json.dumps(evidence))
    future["active_allocations"][0]["ledger_generation"] = 3
    with pytest.raises(
        control.SchedulerAmbiguity,
        match="final fleet retirement allocation is untrusted",
    ):
        control._validate_final_fleet_retirement_evidence(current, future)


def test_finalizer_publishes_marker_last_and_is_idempotent(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    calls: list[str] = []
    snapshot_source_specs: list[list[list[str]]] = []

    def command_runner(argv, environment, timeout):
        assert environment["ASYS_IMMUTABLE_PINS_SHA256"] == current[
            "immutable_sha256"
        ]
        if any(Path(str(item)).name == "create_recovery_snapshot.py" for item in argv):
            assert timeout == control.FINALIZER_SNAPSHOT_DEADLINE_SECONDS
            assert argv[argv.index("--snapshot-kind") + 1] == "final"
            snapshot_root = Path(argv[argv.index("--snapshot-root") + 1])
            source_specs = [
                str(argv[index + 1]).split("=", 1)
                for index, item in enumerate(argv)
                if item == "--source"
            ]
            snapshot_source_specs.append(source_specs)
            if "--verify-only" in argv:
                calls.append("snapshot-verify")
                assert source_specs
                control._validate_final_snapshot_marker(snapshot_root)
            else:
                calls.append("snapshot-create")
                _write_final_snapshot_fixture(
                    snapshot_root,
                    [(name, Path(path)) for name, path in source_specs],
                )
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert timeout == control.FINALIZER_COMMAND_DEADLINE_SECONDS
        if argv[argv.index("--cadence") + 1] == "daily" if "--cadence" in argv else False:
            calls.append("semantic")
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        calls.append("cache")
        cache_root = Path(argv[argv.index("--out-dir") + 1])
        _write_complete_primary_cache(cache_root, current)
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=lambda: control.SchedulerSnapshot((), 40.0),
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_evidence_reader=lambda state: _final_fleet_evidence(state),
        command_runner=command_runner,
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=40.0,
    )
    assert result["complete"] is True
    assert result["complete_cells"] == control.EXPECTED_TOTAL_CELLS
    assert result["trusted_qids"] == control.EXPECTED_TOTAL_QIDS
    assert result["autonomous_finalizer"] is None
    assert result["zero_writer_proof"]["publisher_finalizer"] is None
    assert result["zero_writer_proof"]["active_finalizer_job_ids"] == []
    assert result["zero_writer_proof"]["successor_retirement"] is None
    assert calls == [
        "semantic",
        "cache",
        "snapshot-create",
        "snapshot-verify",
    ]
    assert snapshot_source_specs[0] == snapshot_source_specs[1]
    assert result["schema_version"] == 4
    assert result["snapshot_source_map"] == [
        {"name": name, "path": path}
        for name, path in snapshot_source_specs[0]
    ]
    assert result["control_cut_bindings"]
    assert (output_root / control.FINAL_COMPLETE_FILENAME).stat().st_mode & 0o222 == 0

    repeated = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        command_runner=lambda *_args: pytest.fail("final stages must not rerun"),
        snapshot_builder=lambda *_args: pytest.fail("snapshot must not rerun"),
        lock_scanner=lambda _state: pytest.fail("locks must not be rescanned"),
        now=99.0,
    )
    assert repeated == result

    # FINAL_COMPLETE is not a shortcut around recursive snapshot verification.
    snapshot_root = output_root / control.FINAL_SNAPSHOT_DIRNAME
    payload = next(snapshot_root.glob("*/payload.bin"))
    original = payload.read_bytes()
    snapshot_root.chmod(0o755)
    payload.parent.chmod(0o755)
    payload.chmod(0o644)
    payload.write_bytes(b"tampered snapshot payload\n")
    payload.chmod(0o444)
    payload.parent.chmod(0o555)
    snapshot_root.chmod(0o555)
    with pytest.raises(control.ControlError, match="checksum drifted"):
        control.finalize_sweep(state_dir, output_root=output_root)

    snapshot_root.chmod(0o755)
    payload.parent.chmod(0o755)
    payload.chmod(0o644)
    payload.write_bytes(original)
    payload.chmod(0o444)
    payload.parent.chmod(0o555)
    snapshot_root.chmod(0o555)
    assert control.finalize_sweep(
        state_dir, output_root=output_root
    ) == result

    # Catalog bytes are externally bound by FINAL_COMPLETE even when a forged
    # catalog remains internally well-formed.
    catalog_path = snapshot_root / "SNAPSHOT_CATALOG.json"
    catalog_original = catalog_path.read_bytes()
    snapshot_root.chmod(0o755)
    catalog_path.chmod(0o644)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["sources"][0]["path"] = "/forged/source/path"
    catalog_path.write_text(
        json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalog_path.chmod(0o444)
    snapshot_root.chmod(0o555)
    with pytest.raises(
        control.ControlError,
        match="source origins|snapshot controls drifted",
    ):
        control.finalize_sweep(state_dir, output_root=output_root)
    snapshot_root.chmod(0o755)
    catalog_path.chmod(0o644)
    catalog_path.write_bytes(catalog_original)
    catalog_path.chmod(0o444)
    snapshot_root.chmod(0o555)
    assert control.finalize_sweep(
        state_dir, output_root=output_root
    ) == result

    # Recomputing the outer ID cannot bless a foreign source origin or a stale
    # control-cut binding: both are independently derived from current final
    # provenance and recursively checked inside the snapshot.
    marker_path = output_root / control.FINAL_COMPLETE_FILENAME
    marker_original = marker_path.read_bytes()
    for field, mutation, error in (
        (
            "snapshot_source_map",
            lambda payload: payload["snapshot_source_map"][0].update(
                {"path": "/foreign/final/source"}
            ),
            "snapshot source origins",
        ),
        (
            "control_cut_bindings",
            lambda payload: payload["control_cut_bindings"].update(
                {"capacity_generation": 999}
            ),
            "control-cut bindings",
        ),
    ):
        marker_path.chmod(0o644)
        forged = json.loads(marker_original.decode("utf-8"))
        mutation(forged)
        without_id = dict(forged)
        without_id.pop("final_id")
        forged["final_id"] = control.sha256_value(without_id)
        marker_path.write_text(
            json.dumps(forged, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        marker_path.chmod(0o444)
        with pytest.raises(control.ControlError, match=error):
            control.finalize_sweep(state_dir, output_root=output_root)
        marker_path.chmod(0o644)
        marker_path.write_bytes(marker_original)
        marker_path.chmod(0o444)
        assert field in result
    assert control.finalize_sweep(
        state_dir, output_root=output_root
    ) == result

    # Even a recomputed outer final_id cannot bless a malformed zero-writer proof.
    marker_path.chmod(0o644)
    forged = json.loads(marker_path.read_text(encoding="utf-8"))
    forged["zero_writer_proof"]["active_cell_job_ids"] = ["999"]
    without_id = dict(forged)
    without_id.pop("final_id")
    forged["final_id"] = control.sha256_value(without_id)
    marker_path.write_text(
        json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path.chmod(0o444)
    with pytest.raises(control.ControlError, match="zero-writer proof"):
        control.finalize_sweep(state_dir, output_root=output_root)


def test_finalizer_exactly_retires_active_fleet_and_recovers_after_cancel_crash(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    contract = json.loads(
        Path(current["immutable"]["fleet_contract_path"]).read_text(encoding="utf-8")
    )
    replicas = [
        (profile, replica)
        for profile in contract["profiles"]
        for replica in profile["replicas"]
    ][:2]
    sbatch_path = (
        Path(current["immutable"]["release_worktree"]) / "slurm" / "common.sh"
    ).resolve()
    pinned_jobs: dict[str, control.SchedulerJob] = {}
    allocations: dict[str, dict] = {}
    for index, (profile, replica) in enumerate(replicas):
        job_id = str(900 + index)
        intent_token = f"{index + 1:032x}"
        comment = (
            "asys-s5-fleet:"
            f"pool={contract['fleet_id']};"
            f"replica={replica['replica_id']};"
            f"generation=1;intent={intent_token};"
            f"fleet={current['immutable']['fleet_contract_sha256']}"
        )
        pinned_jobs[job_id] = control.SchedulerJob(
            job_id,
            replica["scheduler_job_name"],
            "RUNNING",
            comment,
        )
        allocations[job_id] = {
            "job_id": job_id,
            "replica_id": replica["replica_id"],
            "profile": profile["serving_profile"],
            "ledger_generation": 1,
            "scheduler_state": "RUNNING",
            "intent_token": intent_token,
            "sbatch_path": str(sbatch_path),
            "sbatch_sha256": _sha(sbatch_path),
        }
    unrelated = control.SchedulerJob(
        "999", "other-research-job", "RUNNING", "unrelated"
    )
    active_ids = {*pinned_jobs, unrelated.job_id}
    captured_at = [40.0]

    def scheduler_reader():
        captured_at[0] += 1.0
        jobs = [
            *[
                job
                for job_id, job in pinned_jobs.items()
                if job_id in active_ids
            ],
            *([unrelated] if unrelated.job_id in active_ids else []),
        ]
        return control.SchedulerSnapshot(tuple(jobs), captured_at[0])

    def evidence_reader(state):
        live = [
            allocations[job_id]
            for job_id in sorted(pinned_jobs, key=int)
            if job_id in active_ids
        ]
        return _final_fleet_evidence(
            state, live, captured_at=captured_at[0]
        )

    cancelled: list[str] = []
    crash_once = [True]

    def cancel_runner(argv):
        assert len(argv) == 2 and argv[0] == "scancel"
        job_id = argv[1]
        intent_path = (
            output_root
            / control.FINAL_FLEET_RETIREMENT_DIRNAME
            / control.FINAL_FLEET_RETIREMENT_INTENT_FILENAME
        )
        # The exact immutable target set must exist before the first external change.
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        assert intent["exact_job_ids"] == ["900", "901"]
        assert job_id in intent["exact_job_ids"]
        cancelled.append(job_id)
        active_ids.discard(job_id)
        if job_id == "901" and crash_once[0]:
            crash_once[0] = False
            raise RuntimeError("simulated death after scheduler accepted scancel")
        return subprocess.CompletedProcess(argv, 0, "", "")

    calls: list[str] = []

    def command_runner(argv, _environment, _timeout):
        if "--cadence" in argv:
            calls.append("semantic")
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        calls.append("cache")
        _write_complete_primary_cache(
            Path(argv[argv.index("--out-dir") + 1]), current
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    def snapshot_builder(snapshot_root, sources, _environment):
        calls.append("snapshot")
        assert "fleet_retirement" in {name for name, _ in sources}
        return _write_final_snapshot_fixture(
            snapshot_root,
            sources,
        )

    with pytest.raises(RuntimeError, match="simulated death"):
        control.finalize_sweep(
            state_dir,
            output_root=output_root,
            scheduler_reader=scheduler_reader,
            cancel_runner=cancel_runner,
            fleet_evidence_reader=evidence_reader,
            command_runner=command_runner,
            snapshot_builder=snapshot_builder,
            lock_scanner=lambda _state: [],
            writer_guard=lambda _state, _path: control.nullcontext(),
            now=40.0,
        )
    assert cancelled == ["900", "901"]
    assert active_ids == {"999"}
    # Exact semantic acceptance is marker-last sealed before the first scancel and is
    # reused after the injected retirement crash.
    assert calls == ["semantic"]
    retirement_intent = json.loads(
        (
            output_root
            / control.FINAL_FLEET_RETIREMENT_DIRNAME
            / control.FINAL_FLEET_RETIREMENT_INTENT_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert retirement_intent["semantic_preflight_sha256"] == _sha(
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    )
    assert retirement_intent["semantic_report_sha256"] == _sha(
        output_root / control.FINAL_SEMANTIC_FILENAME
    )
    assert not (output_root / control.FINAL_COMPLETE_FILENAME).exists()

    completed = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=scheduler_reader,
        cancel_runner=lambda argv: pytest.fail(
            f"already-retired fleet job was cancelled again: {argv}"
        ),
        fleet_evidence_reader=evidence_reader,
        command_runner=command_runner,
        snapshot_builder=snapshot_builder,
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=41.0,
    )
    assert completed["complete"] is True
    assert completed["zero_writer_proof"]["active_fleet_job_ids"] == []
    assert cancelled == ["900", "901"]
    assert "999" in active_ids
    retirement_root = output_root / control.FINAL_FLEET_RETIREMENT_DIRNAME
    retirement = json.loads(
        (
            retirement_root
            / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert retirement["retired_job_ids"] == ["900", "901"]
    # A process death before the attempt journal append leaves no invented attempt;
    # complete scheduler truth on replay adopts the already-terminal allocations.
    assert retirement["attempt_count"] == 1
    assert retirement_root.stat().st_mode & 0o222 == 0
    assert calls == ["semantic", "cache", "snapshot"]


def test_final_fleet_retirement_accepts_nonzero_scancel_after_terminal_truth(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    job_id, active_ids, scheduler_reader, evidence_reader = (
        _single_finalizer_fleet_fixture(current)
    )

    def command_runner(argv, _environment, _timeout):
        if "--cadence" in argv:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        _write_complete_primary_cache(
            Path(argv[argv.index("--out-dir") + 1]), current
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    def cancel_runner(argv):
        assert argv == ["scancel", job_id]
        active_ids.remove(job_id)
        return subprocess.CompletedProcess(
            argv, 1, "", "job naturally completed before scancel"
        )

    completed = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=scheduler_reader,
        cancel_runner=cancel_runner,
        fleet_evidence_reader=evidence_reader,
        command_runner=command_runner,
        snapshot_builder=lambda root, sources, _environment: (
            _write_final_snapshot_fixture(
                root,
                sources,
            )
        ),
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=40.0,
    )
    assert completed["complete"] is True
    attempts = [
        json.loads(line)
        for line in (
            output_root
            / control.FINAL_FLEET_RETIREMENT_DIRNAME
            / control.FINAL_FLEET_RETIREMENT_ATTEMPTS_FILENAME
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert attempts[-1]["results"][job_id]["returncode"] == 1
    assert attempts[-1]["remaining_job_ids"] == []


def test_final_fleet_retirement_nonzero_with_active_job_is_retryable(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    job_id, active_ids, scheduler_reader, evidence_reader = (
        _single_finalizer_fleet_fixture(current)
    )

    def command_runner(argv, _environment, _timeout):
        if "--cadence" in argv:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        _write_complete_primary_cache(
            Path(argv[argv.index("--out-dir") + 1]), current
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(control.FinalizerNodeError, match="remains active"):
        control.finalize_sweep(
            state_dir,
            output_root=output_root,
            scheduler_reader=scheduler_reader,
            cancel_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, "", "temporary scheduler refusal"
            ),
            fleet_evidence_reader=evidence_reader,
            command_runner=command_runner,
            snapshot_builder=lambda *_args: pytest.fail(
                "retryable fleet retirement must not snapshot"
            ),
            lock_scanner=lambda _state: [],
            writer_guard=lambda _state, _path: control.nullcontext(),
            now=40.0,
        )
    assert active_ids == {job_id}
    assert not (
        output_root
        / control.FINAL_FLEET_RETIREMENT_DIRNAME
        / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
    ).exists()

    def succeed(argv):
        assert argv == ["scancel", job_id]
        active_ids.remove(job_id)
        return subprocess.CompletedProcess(argv, 0, "", "")

    completed = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=scheduler_reader,
        cancel_runner=succeed,
        fleet_evidence_reader=evidence_reader,
        command_runner=command_runner,
        snapshot_builder=lambda root, sources, _environment: (
            _write_final_snapshot_fixture(
                root,
                sources,
            )
        ),
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=41.0,
    )
    assert completed["complete"] is True
    retirement = json.loads(
        (
            output_root
            / control.FINAL_FLEET_RETIREMENT_DIRNAME
            / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert retirement["attempt_count"] == 2


def test_finalizer_refuses_partial_semantic_data_without_marker(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    cancelled: list[list[str]] = []
    partial = _complete_semantic_report(current)
    partial["semantic"]["states"] = {"complete": control.EXPECTED_TOTAL_CELLS - 1}
    partial["final_acceptance"]["passed"] = False
    with pytest.raises(control.ControlError, match="acceptance"):
        control.finalize_sweep(
            state_dir,
            output_root=output_root,
            scheduler_reader=lambda: control.SchedulerSnapshot((), 40.0),
            cancel_runner=lambda argv: (
                cancelled.append(list(argv))
                or subprocess.CompletedProcess(argv, 0, "", "")
            ),
            fleet_evidence_reader=lambda state: _final_fleet_evidence(state),
            command_runner=lambda argv, _environment, _timeout: (
                subprocess.CompletedProcess(argv, 0, json.dumps(partial), "")
            ),
            lock_scanner=lambda _state: [],
            writer_guard=lambda _state, _path: control.nullcontext(),
            now=40.0,
        )
    assert cancelled == []
    assert not (
        output_root / control.FINAL_FLEET_RETIREMENT_DIRNAME
    ).exists()
    assert not (output_root / control.FINAL_SEMANTIC_FILENAME).exists()
    assert not (
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    ).exists()
    assert not (output_root / control.FINAL_COMPLETE_FILENAME).exists()


def test_premature_finalizer_keeps_live_fleet_and_later_retry_completes(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    contract = json.loads(
        Path(current["immutable"]["fleet_contract_path"]).read_text(encoding="utf-8")
    )
    profile, replica = next(
        (profile, replica)
        for profile in contract["profiles"]
        for replica in profile["replicas"]
    )
    job = control.SchedulerJob(
        "900",
        replica["scheduler_job_name"],
        "RUNNING",
        "schema5-finalizer-premature-fixture",
    )
    active_ids = {"900"}
    captured_at = [40.0]
    sbatch_path = (
        Path(current["immutable"]["release_worktree"]) / "slurm" / "common.sh"
    ).resolve()
    allocation = {
        "job_id": "900",
        "replica_id": replica["replica_id"],
        "profile": profile["serving_profile"],
        "ledger_generation": 1,
        "scheduler_state": "RUNNING",
        "intent_token": "1" * 32,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": _sha(sbatch_path),
    }

    def scheduler_reader():
        captured_at[0] += 1.0
        jobs = (job,) if "900" in active_ids else ()
        return control.SchedulerSnapshot(jobs, captured_at[0])

    def evidence_reader(state):
        return _final_fleet_evidence(
            state,
            [allocation] if "900" in active_ids else [],
            captured_at=captured_at[0],
        )

    partial = _complete_semantic_report(current)
    partial["semantic"]["states"] = {
        "complete": control.EXPECTED_TOTAL_CELLS - 1
    }
    partial["final_acceptance"]["passed"] = False
    semantic_reports = [partial, _complete_semantic_report(current)]
    semantic_calls = [0]

    def command_runner(argv, _environment, _timeout):
        if "--cadence" in argv:
            report = semantic_reports[semantic_calls[0]]
            semantic_calls[0] += 1
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(report), ""
            )
        _write_complete_primary_cache(
            Path(argv[argv.index("--out-dir") + 1]),
            current,
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    cancelled: list[str] = []

    def cancel_runner(argv):
        assert argv == ["scancel", "900"]
        cancelled.append("900")
        active_ids.remove("900")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def snapshot_builder(snapshot_root, sources, _environment):
        return _write_final_snapshot_fixture(
            snapshot_root,
            sources,
            completed_at="1970-01-01T00:00:41Z",
        )

    common = {
        "output_root": output_root,
        "scheduler_reader": scheduler_reader,
        "cancel_runner": cancel_runner,
        "fleet_evidence_reader": evidence_reader,
        "command_runner": command_runner,
        "snapshot_builder": snapshot_builder,
        "lock_scanner": lambda _state: [],
        "writer_guard": lambda _state, _path: control.nullcontext(),
    }
    with pytest.raises(control.ControlError, match="acceptance"):
        control.finalize_sweep(state_dir, now=40.0, **common)

    assert semantic_calls == [1]
    assert cancelled == []
    assert active_ids == {"900"}
    assert not (
        output_root / control.FINAL_FLEET_RETIREMENT_DIRNAME
    ).exists()
    assert not (output_root / control.FINAL_SEMANTIC_FILENAME).exists()
    assert not (
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    ).exists()

    completed = control.finalize_sweep(state_dir, now=41.0, **common)
    assert completed["complete"] is True
    assert semantic_calls == [2]
    assert cancelled == ["900"]
    assert active_ids == set()
    assert (
        output_root / control.FINAL_SEMANTIC_FILENAME
    ).stat().st_mode & 0o222 == 0
    assert (
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    ).stat().st_mode & 0o222 == 0


def test_finalizer_adopts_valid_semantic_report_sealed_before_marker_crash(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    control.pause_control(
        state_dir,
        drain=True,
        scheduler=control.SchedulerSnapshot((), 39.0),
        now=39.0,
    )
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    output_root.mkdir()
    source_contract, source_contract_sha256 = (
        control._final_semantic_source_contract(current)
    )
    control._create_final_semantic_intent(
        output_root,
        control=current,
        source_contract=source_contract,
        source_contract_sha256=source_contract_sha256,
        now=39.0,
    )
    semantic_path = output_root / control.FINAL_SEMANTIC_FILENAME
    semantic_path.write_text(
        json.dumps(_complete_semantic_report(current), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    semantic_path.chmod(0o444)
    calls: list[str] = []

    def command_runner(argv, _environment, _timeout):
        assert "--cadence" not in argv, "sealed semantic report must be adopted"
        calls.append("cache")
        _write_complete_primary_cache(
            Path(argv[argv.index("--out-dir") + 1]),
            current,
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    def snapshot_builder(snapshot_root, sources, _environment):
        calls.append("snapshot")
        return _write_final_snapshot_fixture(
            snapshot_root,
            sources,
        )

    completed = control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=lambda: control.SchedulerSnapshot((), 40.0),
        cancel_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "", ""
        ),
        fleet_evidence_reader=lambda state: _final_fleet_evidence(state),
        command_runner=command_runner,
        snapshot_builder=snapshot_builder,
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=40.0,
    )
    assert completed["complete"] is True
    assert calls == ["cache", "snapshot"]
    preflight_path = (
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["semantic_report_sha256"] == _sha(semantic_path)
    assert preflight_path.stat().st_mode & 0o222 == 0


def test_semantic_report_without_transaction_intent_is_archived_and_recomputed(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    control.pause_control(
        state_dir,
        drain=True,
        scheduler=control.SchedulerSnapshot((), 39.0),
        now=39.0,
    )
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    output_root.mkdir()
    semantic_path = output_root / control.FINAL_SEMANTIC_FILENAME
    semantic_path.write_text(
        json.dumps(_complete_semantic_report(current), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    semantic_path.chmod(0o444)
    stale_sha256 = _sha(semantic_path)
    # Simulate death after the independent preimage copy but before the archive
    # marker.  Retry must verify/reuse the exact copy and finish marker-last.
    identity = {
        "name": control.FINAL_SEMANTIC_FILENAME,
        "size": semantic_path.stat().st_size,
        "sha256": stale_sha256,
    }
    archive_id = control.sha256_value(
        {
            "reason": "semantic_report_without_transaction_intent",
            "preimages": [identity],
        }
    )
    partial_archive = (
        output_root
        / control.FINAL_SEMANTIC_ARCHIVE_DIRNAME
        / archive_id
    )
    partial_archive.mkdir(parents=True)
    (partial_archive / control.FINAL_SEMANTIC_FILENAME).write_bytes(
        semantic_path.read_bytes()
    )
    calls = []

    def command_runner(argv, _environment, _timeout):
        calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(_complete_semantic_report(current)),
            "",
        )

    control._ensure_final_semantic_preflight(
        output_root,
        control=current,
        state_dir=state_dir,
        results_root=Path(current["immutable"]["results_root"]),
        environment={},
        harness_python=Path(
            current["immutable"]["harness_environment_prefix"]
        )
        / "bin"
        / "python",
        release_root=Path(current["immutable"]["release_worktree"]),
        run_command=command_runner,
        now=40.0,
    )
    assert len(calls) == 1
    archives = list(
        (output_root / control.FINAL_SEMANTIC_ARCHIVE_DIRNAME).iterdir()
    )
    assert len(archives) == 1
    archived_marker = json.loads(
        (archives[0] / "ARCHIVE_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert archived_marker["reason"] == "semantic_report_without_transaction_intent"
    assert _sha(archives[0] / control.FINAL_SEMANTIC_FILENAME) == stale_sha256
    assert (
        output_root / control.FINAL_SEMANTIC_INTENT_FILENAME
    ).stat().st_mode & 0o222 == 0
    assert (
        output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
    ).is_file()


def test_semantic_crash_evidence_is_rejected_after_result_source_mutation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    control.pause_control(
        state_dir,
        drain=True,
        scheduler=control.SchedulerSnapshot((), 39.0),
        now=39.0,
    )
    current = control.load_control(state_dir)
    output_root = tmp_path / "final"
    output_root.mkdir()
    source_contract, source_contract_sha256 = (
        control._final_semantic_source_contract(current)
    )
    _intent, old_intent_sha256 = control._create_final_semantic_intent(
        output_root,
        control=current,
        source_contract=source_contract,
        source_contract_sha256=source_contract_sha256,
        now=39.0,
    )
    semantic_path = output_root / control.FINAL_SEMANTIC_FILENAME
    semantic_path.write_text(
        json.dumps(_complete_semantic_report(current), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    semantic_path.chmod(0o444)

    first_run_root = Path(current["immutable"]["runs"][0]["run_root"])
    (first_run_root / "post_intent_mutation.bin").write_bytes(b"changed\n")
    semantic_calls = []

    def command_runner(argv, _environment, _timeout):
        semantic_calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(_complete_semantic_report(current)),
            "",
        )

    control._ensure_final_semantic_preflight(
        output_root,
        control=current,
        state_dir=state_dir,
        results_root=Path(current["immutable"]["results_root"]),
        environment={},
        harness_python=Path(
            current["immutable"]["harness_environment_prefix"]
        )
        / "bin"
        / "python",
        release_root=Path(current["immutable"]["release_worktree"]),
        run_command=command_runner,
        now=40.0,
    )
    assert len(semantic_calls) == 1
    archives = list(
        (output_root / control.FINAL_SEMANTIC_ARCHIVE_DIRNAME).iterdir()
    )
    assert len(archives) == 1
    assert _sha(
        archives[0] / control.FINAL_SEMANTIC_INTENT_FILENAME
    ) == old_intent_sha256
    archive_marker = json.loads(
        (archives[0] / "ARCHIVE_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert archive_marker["reason"].startswith("intent_rejected:")
    preflight = json.loads(
        (
            output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert preflight["source_contract_sha256"] != source_contract_sha256


def _write_autonomous_finalization_evidence(
    state_dir: Path,
    current: dict,
    *,
    captured_timestamp: float = 40.0,
) -> tuple[Path, dict]:
    report = _complete_semantic_report(current)
    report["captured_timestamp"] = captured_timestamp
    path = (
        state_dir
        / "monitoring"
        / "semantic"
        / f"{int(captured_timestamp * 1_000_000):020d}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o444)
    return path, report


def _replace_fixture_harness_python(
    current: dict,
    *,
    target: Path,
    target_mode: int,
    refresh_inventory: bool,
) -> tuple[dict, Path]:
    updated = copy.deepcopy(current)
    immutable = updated["immutable"]
    harness = Path(immutable["harness_environment_prefix"])
    lexical = harness / "bin" / "python"
    for directory in (harness, harness / "bin"):
        directory.chmod(0o755)
    lexical.chmod(0o644)
    lexical.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(target_mode)
    lexical.symlink_to(os.path.relpath(target, lexical.parent))
    (harness / "bin").chmod(0o555)
    harness.chmod(0o555)
    if refresh_inventory:
        manifest_path = Path(
            immutable["harness_environment_manifest_path"]
        )
        manifest_path.chmod(0o644)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["directory_inventory"] = (
            runtime_integrity.directory_inventory(harness)
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_path.chmod(0o444)
        immutable["harness_environment_sha256"] = _sha(manifest_path)
    return updated, lexical


def test_autonomous_finalizer_accepts_inventory_pinned_internal_python_symlink(
    tmp_path,
):
    state_dir, current = initialize(tmp_path)
    harness = Path(current["immutable"]["harness_environment_prefix"])
    updated, _lexical = _replace_fixture_harness_python(
        current,
        target=harness / "bin" / "python3.12",
        target_mode=0o555,
        refresh_inventory=True,
    )
    rendered = control._render_finalizer_sbatch(
        state_dir,
        updated,
        intent_id="1" * 64,
        attempt=1,
        intent_token="2" * 32,
    )
    payload = rendered.read_text(encoding="utf-8")
    assert f"exec {harness / 'bin' / 'python3.12'} -I -u" in payload
    assert "#SBATCH --export=NONE" in payload
    assert "export PATH=/usr/bin:/bin" in payload
    assert "export GIT_NO_REPLACE_OBJECTS=1" in payload
    log_root = state_dir / control.FINALIZER_STATE_DIRNAME / "logs"
    assert log_root.is_dir()
    assert not log_root.is_symlink()
    assert f"#SBATCH --output={log_root}/finalizer.a000001.%j.out" in payload


def test_finalizer_sbatch_is_readonly_at_link_crash_and_replays(
    tmp_path, monkeypatch
):
    state_dir, current = initialize(tmp_path)
    target = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / "sbatch"
        / ("finalizer.a000001." + "2" * 32 + ".sbatch")
    )
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == target:
            crashed["value"] = True
            raise RuntimeError("death after finalizer sbatch link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="sbatch link"):
            control._render_finalizer_sbatch(
                state_dir,
                current,
                intent_id="1" * 64,
                attempt=1,
                intent_token="2" * 32,
            )
    assert target.is_file()
    assert target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o222 == 0
    before = target.read_bytes()
    replay = control._render_finalizer_sbatch(
        state_dir,
        current,
        intent_id="1" * 64,
        attempt=1,
        intent_token="2" * 32,
    )
    assert replay == target
    assert target.read_bytes() == before


def _held_exact_submission_fixture(tmp_path: Path, *, job_id: str = "901"):
    state_dir = tmp_path / f"state-{job_id}"
    state_dir.mkdir(parents=True)
    sbatch_path = tmp_path / f"job-{job_id}.sbatch"
    control._atomic_publish_readonly_text(
        sbatch_path, "#!/bin/bash\n#SBATCH --no-requeue\ntrue\n"
    )
    token = (
        "asys-schema5-v1;role=dispatcher;generation=1;"
        f"intent={job_id.zfill(32)}"
    )
    submission_argv = control._exact_sbatch_submission_argv(
        token=token, dependency_job_id=None
    )
    record = {
        "job_token": token,
        "sbatch_path": str(sbatch_path.resolve()),
        "sbatch_sha256": _sha(sbatch_path),
        "submission_transport": control.EXACT_SBATCH_SUBMISSION_TRANSPORT,
        "submission_argv": submission_argv,
        "submission_argv_sha256": control._submission_argv_sha256(
            submission_argv
        ),
        "persistent_hold": True,
        "spooled_receipt_path": None,
        "spooled_receipt_sha256": None,
        "released_at": None,
        "released_timestamp": None,
        "dependency_job_id": None,
    }
    proof = control._prove_exact_sbatch_and_release(
        state_dir,
        record=record,
        job_id=job_id,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, job_id + "\n", ""
        ),
        release_after_proof=False,
        now=40.0,
    )
    record.update(proof)
    return state_dir, record


def test_exact_release_lost_reply_replays_from_running_scheduler_truth(tmp_path):
    state_dir, record = _held_exact_submission_fixture(tmp_path)
    release_calls = []

    def release_then_die(argv):
        release_calls.append(list(argv))
        raise RuntimeError("lost release reply")

    with pytest.raises(RuntimeError, match="lost release reply"):
        control._ensure_exact_sbatch_released(
            state_dir,
            record=record,
            job_id="901",
            receipt_path=Path(record["spooled_receipt_path"]),
            receipt_sha256=record["spooled_receipt_sha256"],
            now=41.0,
            release_runner=release_then_die,
            scheduler_job=control.SchedulerJob(
                "901", "controller", "PENDING", record["job_token"]
            ),
            observation_runner=None,
        )
    intent_path, complete_path = control._exact_sbatch_release_paths(
        Path(record["spooled_receipt_path"])
    )
    assert intent_path.is_file()
    assert intent_path.stat().st_mode & 0o222 == 0
    assert not complete_path.exists()

    replayed = control._ensure_exact_sbatch_released(
        state_dir,
        record=record,
        job_id="901",
        receipt_path=Path(record["spooled_receipt_path"]),
        receipt_sha256=record["spooled_receipt_sha256"],
        now=42.0,
        release_runner=lambda argv: pytest.fail(
            f"already-running replay must not release again: {argv}"
        ),
        scheduler_job=control.SchedulerJob(
            "901", "controller", "RUNNING", record["job_token"]
        ),
        observation_runner=None,
    )
    record.update(replayed)
    assert release_calls == [["scontrol", "release", "901"]]
    assert complete_path.is_file()
    control._validate_exact_sbatch_record_proof(
        record, job_id="901", context="lost-reply replay"
    )


def test_exact_release_nonzero_adopts_pending_unheld_and_rejects_mismatch(
    tmp_path,
):
    state_dir, record = _held_exact_submission_fixture(tmp_path, job_id="902")

    def nonzero(argv):
        return subprocess.CompletedProcess(argv, 1, "", "already released")

    def unheld(argv):
        assert argv == ["scontrol", "show", "job", "-o", "902"]
        return subprocess.CompletedProcess(
            argv,
            0,
            (
                "JobId=902 JobName=controller JobState=PENDING "
                f"Reason=Priority Comment={record['job_token']}\n"
            ),
            "",
        )

    proof = control._ensure_exact_sbatch_released(
        state_dir,
        record=record,
        job_id="902",
        receipt_path=Path(record["spooled_receipt_path"]),
        receipt_sha256=record["spooled_receipt_sha256"],
        now=41.0,
        release_runner=nonzero,
        scheduler_job=control.SchedulerJob(
            "902", "controller", "PENDING", record["job_token"]
        ),
        observation_runner=unheld,
    )
    record.update(proof)
    control._validate_exact_sbatch_record_proof(
        record, job_id="902", context="nonzero already released"
    )

    other_state, other = _held_exact_submission_fixture(tmp_path, job_id="903")

    def mismatched(argv):
        return subprocess.CompletedProcess(
            argv,
            0,
            "JobId=903 JobName=controller JobState=RUNNING "
            "Reason=None Comment=foreign\n",
            "",
        )

    with pytest.raises(control.SchedulerAmbiguity, match="identity mismatched"):
        control._ensure_exact_sbatch_released(
            other_state,
            record=other,
            job_id="903",
            receipt_path=Path(other["spooled_receipt_path"]),
            receipt_sha256=other["spooled_receipt_sha256"],
            now=41.0,
            release_runner=nonzero,
            scheduler_job=control.SchedulerJob(
                "903", "controller", "PENDING", other["job_token"]
            ),
            observation_runner=mismatched,
        )


def test_exact_release_nonzero_held_remains_retryable(tmp_path):
    state_dir, record = _held_exact_submission_fixture(tmp_path, job_id="904")

    def nonzero(argv):
        return subprocess.CompletedProcess(argv, 1, "", "still held")

    def held(argv):
        return subprocess.CompletedProcess(
            argv,
            0,
            (
                "JobId=904 JobName=controller JobState=PENDING "
                f"Reason=JobHeldUser Comment={record['job_token']}\n"
            ),
            "",
        )

    with pytest.raises(control.SchedulerVisibilityPending, match="remains held"):
        control._ensure_exact_sbatch_released(
            state_dir,
            record=record,
            job_id="904",
            receipt_path=Path(record["spooled_receipt_path"]),
            receipt_sha256=record["spooled_receipt_sha256"],
            now=41.0,
            release_runner=nonzero,
            scheduler_job=control.SchedulerJob(
                "904", "controller", "PENDING", record["job_token"]
            ),
            observation_runner=held,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda argv: [*argv, "/mutable/job.sbatch"],
        lambda argv: [item for item in argv if item != "--hold"],
        lambda argv: [*argv, "--dependency=afterany:999"],
        lambda argv: [*argv, "--partition=unattested"],
    ),
)
def test_exact_submission_validator_derives_the_only_allowed_held_argv(
    tmp_path, mutate
):
    _state_dir, record = _held_exact_submission_fixture(tmp_path, job_id="905")
    changed = copy.deepcopy(record)
    changed["submission_argv"] = mutate(list(changed["submission_argv"]))
    changed["submission_argv_sha256"] = control._submission_argv_sha256(
        changed["submission_argv"]
    )
    with pytest.raises(control.SchedulerAmbiguity, match="exact immutable stdin"):
        control._validate_exact_sbatch_record_proof(
            changed, job_id="905", context="mutated exact argv"
        )


def test_autonomous_finalizer_rejects_symlinked_log_directory(tmp_path):
    state_dir, current = initialize(tmp_path)
    finalizer_root = state_dir / control.FINALIZER_STATE_DIRNAME
    finalizer_root.mkdir()
    outside = tmp_path / "outside-finalizer-logs"
    outside.mkdir()
    (finalizer_root / "logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        control.ImmutablePinError,
        match="finalizer log directory is unsafe",
    ):
        control._render_finalizer_sbatch(
            state_dir,
            current,
            intent_id="1" * 64,
            attempt=1,
            intent_token="2" * 32,
        )


def test_autonomous_finalizer_rejects_escaping_python_symlink(tmp_path):
    state_dir, current = initialize(tmp_path)
    updated, _lexical = _replace_fixture_harness_python(
        current,
        target=tmp_path / "outside-python",
        target_mode=0o555,
        refresh_inventory=True,
    )
    with pytest.raises(control.ImmutablePinError, match="escapes its harness"):
        control._render_finalizer_sbatch(
            state_dir,
            updated,
            intent_id="1" * 64,
            attempt=1,
            intent_token="2" * 32,
        )


def test_autonomous_finalizer_rejects_unpinned_or_mutable_python_target(
    tmp_path,
):
    state_dir, current = initialize(tmp_path)
    harness = Path(current["immutable"]["harness_environment_prefix"])
    unpinned, _lexical = _replace_fixture_harness_python(
        current,
        target=harness / "bin" / "python3.12",
        target_mode=0o555,
        refresh_inventory=False,
    )
    with pytest.raises(control.ImmutablePinError, match="inventory-pinned"):
        control._render_finalizer_sbatch(
            state_dir,
            unpinned,
            intent_id="1" * 64,
            attempt=1,
            intent_token="2" * 32,
        )

    other_root = tmp_path / "mutable"
    other_state, other_current = initialize(other_root)
    other_harness = Path(
        other_current["immutable"]["harness_environment_prefix"]
    )
    mutable, _lexical = _replace_fixture_harness_python(
        other_current,
        target=other_harness / "bin" / "python3.12",
        target_mode=0o755,
        refresh_inventory=True,
    )
    with pytest.raises(control.ImmutablePinError, match="sealed executable"):
        control._render_finalizer_sbatch(
            other_state,
            mutable,
            intent_id="1" * 64,
            attempt=1,
            intent_token="2" * 32,
        )


def test_final_semantic_intent_is_readonly_at_link_crash_and_validates(
    tmp_path,
    monkeypatch,
):
    state_dir, current = initialize(tmp_path)
    final_root = Path(current["finalization"]["output_root"])
    final_root.mkdir(parents=True)
    source_contract, source_contract_sha256 = (
        control._final_semantic_source_contract(current)
    )
    intent_path = final_root / control.FINAL_SEMANTIC_INTENT_FILENAME
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == intent_path:
            crashed["value"] = True
            raise RuntimeError("death after semantic intent link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="semantic intent link"):
            control._create_final_semantic_intent(
                final_root,
                control=current,
                source_contract=source_contract,
                source_contract_sha256=source_contract_sha256,
                now=40.0,
            )
    assert intent_path.is_file()
    assert intent_path.stat().st_nlink == 1
    assert intent_path.stat().st_mode & 0o222 == 0
    intent, intent_sha256 = control._validate_final_semantic_intent(
        intent_path,
        final_root=final_root,
        control=current,
        source_contract=source_contract,
        source_contract_sha256=source_contract_sha256,
    )
    assert intent["created_timestamp"] == 40.0
    assert intent_sha256 == _sha(intent_path)


def _publish_manual_final_complete_fixture(
    state_dir: Path, current: dict
) -> dict:
    output_root = Path(current["finalization"]["output_root"])

    def command_runner(argv, _environment, _timeout):
        if "--cadence" in argv:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        cache_root = Path(argv[argv.index("--out-dir") + 1])
        _write_complete_primary_cache(cache_root, current)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return control.finalize_sweep(
        state_dir,
        output_root=output_root,
        scheduler_reader=lambda: control.SchedulerSnapshot((), 40.0),
        cancel_runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        fleet_evidence_reader=lambda state: _final_fleet_evidence(state),
        command_runner=command_runner,
        snapshot_builder=lambda snapshot_root, sources, _environment: (
            _write_final_snapshot_fixture(
                snapshot_root,
                sources,
            )
        ),
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        now=40.0,
    )


def test_final_complete_is_readonly_at_link_crash_and_replays(
    tmp_path, monkeypatch
):
    state_dir, current = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    marker_path = (
        Path(current["finalization"]["output_root"])
        / control.FINAL_COMPLETE_FILENAME
    )
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if not crashed["value"] and Path(destination) == marker_path:
            crashed["value"] = True
            raise RuntimeError("death after FINAL_COMPLETE link")

    with monkeypatch.context() as boundary:
        boundary.setattr(control.os, "link", link_then_crash)
        with pytest.raises(RuntimeError, match="FINAL_COMPLETE link"):
            _publish_manual_final_complete_fixture(state_dir, current)
    assert marker_path.is_file()
    assert marker_path.stat().st_nlink == 1
    assert marker_path.stat().st_mode & 0o222 == 0
    before = marker_path.read_bytes()
    replayed = _publish_manual_final_complete_fixture(state_dir, current)
    assert replayed["final_id"] == json.loads(before)["final_id"]
    assert marker_path.read_bytes() == before


def _autonomous_scheduler(
    finalization: dict,
    *,
    captured_at: float = 41.0,
    active_state: str = "RUNNING",
    successor_state: str = "PENDING",
    extra_jobs: tuple[control.SchedulerJob, ...] = (),
) -> control.SchedulerSnapshot:
    jobs = []
    for field in ("active_job", "successor_job"):
        row = finalization.get(field)
        if not isinstance(row, dict):
            continue
        dependency = (
            ""
            if row["dependency_job_id"] is None
            else f"afterany:{row['dependency_job_id']}"
        )
        jobs.append(
            control.SchedulerJob(
                str(row["job_id"]),
                f"asys-s5-final-a{row['attempt']:06d}",
                active_state if field == "active_job" else successor_state,
                str(row["job_token"]),
                str(row["sbatch_path"]),
                dependency=dependency,
            )
        )
    return control.SchedulerSnapshot(tuple([*jobs, *extra_jobs]), captured_at)


def _seed_complete_finalizer_phase_evidence(
    state_dir: Path,
    *,
    publisher: dict,
    output_root: Path,
    now: float,
) -> None:
    """Bind synthetic, structurally valid phase evidence for marker-only fixtures."""

    output_root = output_root.resolve()
    semantic_report_sha256 = "1" * 64
    semantic_intent_sha256 = "2" * 64
    semantic_intent_id = "3" * 64
    semantic_preflight_sha256 = "4" * 64
    semantic_preflight_id = "5" * 64
    retirement_intent_sha256 = "6" * 64
    retirement_id = "7" * 64
    retirement_marker_sha256 = "8" * 64
    retirement_completion_id = "9" * 64
    cache_marker_sha256 = "a" * 64
    cache_generation_id = "b" * 64
    cache_tree_sha256 = "c" * 64
    snapshot_marker_sha256 = "d" * 64
    retirement_root = (
        output_root / control.FINAL_FLEET_RETIREMENT_DIRNAME
    ).resolve()
    cache_root = (
        output_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    ).resolve()
    snapshot_root = (output_root / control.FINAL_SNAPSHOT_DIRNAME).resolve()
    control._record_autonomous_finalizer_phase(
        state_dir,
        publisher_finalizer=publisher,
        phase="validating",
        bindings={
            "semantic_report_path": str(
                (output_root / control.FINAL_SEMANTIC_FILENAME).resolve()
            ),
            "semantic_intent_path": str(
                (
                    output_root / control.FINAL_SEMANTIC_INTENT_FILENAME
                ).resolve()
            ),
            "semantic_preflight_path": str(
                (
                    output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
                ).resolve()
            ),
            "semantic_report_sha256": semantic_report_sha256,
            "semantic_intent_sha256": semantic_intent_sha256,
            "semantic_intent_id": semantic_intent_id,
            "semantic_preflight_sha256": semantic_preflight_sha256,
            "semantic_preflight_id": semantic_preflight_id,
        },
        now=now,
    )
    control._record_autonomous_finalizer_phase(
        state_dir,
        publisher_finalizer=publisher,
        phase="retiring_fleet",
        bindings={
            "semantic_report_sha256": semantic_report_sha256,
            "semantic_preflight_sha256": semantic_preflight_sha256,
            "semantic_preflight_id": semantic_preflight_id,
            "retirement_root": str(retirement_root),
            "retirement_intent_path": str(
                (
                    retirement_root
                    / control.FINAL_FLEET_RETIREMENT_INTENT_FILENAME
                ).resolve()
            ),
            "retirement_intent_sha256": retirement_intent_sha256,
            "retirement_id": retirement_id,
            "retirement_marker_path": str(
                (
                    retirement_root
                    / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
                ).resolve()
            ),
            "retirement_marker_sha256": retirement_marker_sha256,
            "retirement_completion_id": retirement_completion_id,
        },
        now=now + 1.0,
    )
    control._record_autonomous_finalizer_phase(
        state_dir,
        publisher_finalizer=publisher,
        phase="snapshotting",
        bindings={
            "retirement_root": str(retirement_root),
            "retirement_marker_path": str(
                (
                    retirement_root
                    / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
                ).resolve()
            ),
            "retirement_marker_sha256": retirement_marker_sha256,
            "retirement_completion_id": retirement_completion_id,
            "cache_root": str(cache_root),
            "cache_marker_path": str(
                (cache_root / "ingest_manifest_v1.json").resolve()
            ),
            "cache_marker_sha256": cache_marker_sha256,
            "cache_generation_id": cache_generation_id,
            "cache_tree_sha256": cache_tree_sha256,
            "snapshot_root": str(snapshot_root),
            "snapshot_marker_path": str(
                (snapshot_root / "SNAPSHOT_COMPLETE.json").resolve()
            ),
            "snapshot_marker_sha256": snapshot_marker_sha256,
            "snapshot_id": "synthetic-final-snapshot",
        },
        now=now + 2.0,
    )


def _record_finalization_critical_alert(
    state_dir: Path,
    *,
    dedupe_key: str,
    now: float,
) -> None:
    control.record_alert(
        state_dir,
        kind="finalization-capacity-fixture",
        severity="critical",
        message=f"fixture alert {dedupe_key}",
        dedupe_key=dedupe_key,
        scheduler=control.SchedulerSnapshot((), now),
        now=now,
    )


def test_exact_completion_consumes_only_capacity_holds_into_finalization_intent(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    capacity_keys = sorted(control.CAPACITY_REMEDIATION_ALERT_KEYS)
    for offset, dedupe_key in enumerate(capacity_keys):
        _record_finalization_critical_alert(
            state_dir,
            dedupe_key=dedupe_key,
            now=20.0 + offset,
        )
    # This is the monitor's real ordering: the exact semantic scan resolves the
    # alerts, while their integrity-mode hold deliberately remains latched.
    for offset, dedupe_key in enumerate(capacity_keys):
        control.resolve_alert(
            state_dir, dedupe_key=dedupe_key, now=25.0 + offset
        )
    held = control.load_control(state_dir)["admission_safety_hold"]
    assert held["active"] is True
    assert held["reasons"] == sorted(
        control.CAPACITY_REMEDIATION_ALERT_KEYS
    )

    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    result = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )

    assert result["schema_version"] == control.FINALIZATION_SCHEMA_VERSION
    assert result["consumed_capacity_incidents"] == sorted(
        control.CAPACITY_REMEDIATION_ALERT_KEYS
    )
    request_path = Path(result["request_intent"]["path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["intent_id"] == result["intent_id"]
    assert request["consumed_capacity_incidents"] == sorted(
        control.CAPACITY_REMEDIATION_ALERT_KEYS
    )
    assert [
        row["dedupe_key"]
        for row in request["capacity_alert_resolution_plan"]
    ] == sorted(control.CAPACITY_REMEDIATION_ALERT_KEYS)
    assert request_path.stat().st_mode & 0o222 == 0
    persisted = control.load_control(state_dir, verify_files=True)
    assert persisted["admission_safety_hold"]["active"] is False
    assert persisted["admission_safety_hold"]["reasons"] == []
    assert [
        row["event"]
        for row in persisted["transition_history"]
        if row["event"]
        in {
            "finalization_capacity_incidents_consumed",
            "autonomous_finalization_requested",
        }
    ] == [
        "finalization_capacity_incidents_consumed",
        "autonomous_finalization_requested",
    ]
    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir,
        persisted,
        captured_timestamp=41.0,
    )
    retriggered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=_autonomous_scheduler(result, captured_at=41.0),
        submit_runner=lambda argv: pytest.fail(
            f"active finalization retrigger must not submit: {argv}"
        ),
        now=41.0,
    )
    assert retriggered["intent_id"] == result["intent_id"]
    assert retriggered["semantic_evidence"] == result["semantic_evidence"]


@pytest.mark.parametrize(
    "crash_boundary",
    (
        "after_alert_resolution",
        "after_consumption_transition",
        "before_control_replace",
    ),
)
def test_finalization_capacity_consumption_replays_exactly_once_after_crash(
    tmp_path, monkeypatch, crash_boundary
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_save = control._save_control
    original_append = control._append_jsonl
    injected = {"value": False}

    def crash_after_journal_append(path, record):
        original_append(path, record)
        if injected["value"]:
            return
        if (
            crash_boundary == "after_alert_resolution"
            and Path(path).name == control.ALERT_JOURNAL
            and record.get("action") == "resolved"
            and record.get("finalization_intent_id")
        ):
            injected["value"] = True
            raise RuntimeError("crash after finalization alert resolution")
        if (
            crash_boundary == "after_consumption_transition"
            and Path(path).name == control.TRANSITION_JOURNAL
            and record.get("event")
            == "finalization_capacity_incidents_consumed"
        ):
            injected["value"] = True
            raise RuntimeError("crash after finalization transition")

    def crash_before_request_replace(path, payload, *, now):
        if (
            not injected["value"]
            and crash_boundary == "before_control_replace"
            and payload["finalization"]["state"] == "requested"
        ):
            injected["value"] = True
            raise RuntimeError("crash before finalization control replacement")
        return original_save(path, payload, now=now)

    monkeypatch.setattr(control, "_append_jsonl", crash_after_journal_append)
    monkeypatch.setattr(
        control, "_save_control", crash_before_request_replace
    )
    with pytest.raises(RuntimeError, match="crash"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: pytest.fail(
                f"pre-commit finalizer must not submit: {argv}"
            ),
            now=40.0,
        )

    crashed = control.load_control(state_dir)
    assert crashed["finalization"]["state"] == "idle"
    assert crashed["admission_safety_hold"]["reasons"] == [
        "monitor:throughput"
    ]
    request_path = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    )
    assert request_path.is_file()
    request_before = request_path.read_bytes()

    monkeypatch.setattr(control, "_append_jsonl", original_append)
    monkeypatch.setattr(control, "_save_control", original_save)
    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir,
        crashed,
        captured_timestamp=41.0,
    )
    submitted = iter(("901", "902"))
    recovered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=control.SchedulerSnapshot((), 41.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=41.0,
    )

    assert request_path.read_bytes() == request_before
    assert recovered["requested_timestamp"] == 40.0
    assert recovered["semantic_evidence"]["path"] == str(
        evidence_path.resolve()
    )
    assert recovered["consumed_capacity_incidents"] == [
        "monitor:throughput"
    ]
    persisted = control.load_control(state_dir, verify_files=True)
    assert persisted["admission_safety_hold"]["active"] is False
    throughput_alert = next(
        alert
        for alert in persisted["alerts"]
        if alert["dedupe_key"] == "monitor:throughput"
    )
    assert throughput_alert["resolved_timestamp"] == 40.0
    journal = control._read_jsonl_locked(
        state_dir / control.ALERT_JOURNAL
    )
    resolutions = [
        row
        for row in journal
        if row.get("action") == "resolved"
        and row.get("finalization_intent_id") == recovered["intent_id"]
    ]
    assert [
        (row["dedupe_key"], row["alert_id"]) for row in resolutions
    ] == [("monitor:throughput", throughput_alert["alert_id"])]
    for event in (
        "finalization_capacity_incidents_consumed",
        "autonomous_finalization_requested",
    ):
        assert (
            sum(
                row["event"] == event
                and row["details"].get("intent_id")
                == recovered["intent_id"]
                for row in persisted["transition_history"]
            )
            == 1
        )
    assert control._active_critical_alerts_from_journal(state_dir) == []


def test_finalization_capacity_replay_keeps_frozen_alert_ids_after_control_save(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_append = control._append_jsonl
    crashed = {"value": False}

    def crash_after_consumption_transition(path, record):
        original_append(path, record)
        if (
            not crashed["value"]
            and Path(path).name == control.TRANSITION_JOURNAL
            and record.get("event")
            == "finalization_capacity_incidents_consumed"
        ):
            crashed["value"] = True
            raise RuntimeError("crash after consumption transition")

    monkeypatch.setattr(
        control, "_append_jsonl", crash_after_consumption_transition
    )
    with pytest.raises(RuntimeError, match="crash"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: pytest.fail(
                f"pre-commit finalizer must not submit: {argv}"
            ),
            now=40.0,
        )

    request_path = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    )
    frozen_request = json.loads(request_path.read_text(encoding="utf-8"))
    frozen_alert_ids = frozen_request["capacity_alert_resolution_plan"][0][
        "alert_ids"
    ]
    assert len(frozen_alert_ids) == 1

    # A monitor retry can reconcile the journal-ahead transition and persist its own
    # alert resolution before this transaction resumes.
    monkeypatch.setattr(control, "_append_jsonl", original_append)
    control.resolve_alert(
        state_dir, dedupe_key="monitor:throughput", now=40.5
    )
    assert next(
        alert
        for alert in control.load_control(state_dir)["alerts"]
        if alert["dedupe_key"] == "monitor:throughput"
    )["resolved_timestamp"] == 40.5

    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir,
        control.load_control(state_dir),
        captured_timestamp=41.0,
    )
    submitted = iter(("901", "902"))
    recovered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=control.SchedulerSnapshot((), 41.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=41.0,
    )
    assert recovered["semantic_evidence"]["path"] == str(
        evidence_path.resolve()
    )
    persisted = control.load_control(state_dir, verify_files=True)
    consumption = next(
        row
        for row in persisted["transition_history"]
        if row["event"] == "finalization_capacity_incidents_consumed"
    )
    assert consumption["details"]["resolved_alert_ids"] == frozen_alert_ids
    assert (
        sum(
            row["event"] == "finalization_capacity_incidents_consumed"
            for row in persisted["transition_history"]
        )
        == 1
    )


def test_finalization_rejects_journal_ahead_alert_even_when_hold_names_it(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    original_save = control._save_control
    crashed = {"value": False}

    def crash_before_alert_control_save(*args, **kwargs):
        if not crashed["value"]:
            crashed["value"] = True
            raise RuntimeError("crash before alert control save")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(control, "_save_control", crash_before_alert_control_save)
    with pytest.raises(RuntimeError, match="crash"):
        _record_finalization_critical_alert(
            state_dir, dedupe_key="monitor:throughput", now=20.0
        )
    monkeypatch.setattr(control, "_save_control", original_save)

    # Persist the same fail-closed hold reason without inventing the missing alert
    # identity.  The journal/control difference must remain blocking.
    with control.control_lock(state_dir):
        current = control.load_control(state_dir)
        control._update_admission_safety_hold_locked(
            state_dir,
            current,
            active_critical_keys=("monitor:throughput",),
            clean_poll=False,
            semantic_scan_clean=None,
            timestamp=21.0,
        )
        control._save_control(state_dir, current, now=21.0)
    persisted = control.load_control(state_dir)
    assert persisted["admission_safety_hold"]["reasons"] == [
        "monitor:throughput"
    ]
    assert control._active_critical_alerts(persisted) == []
    assert control._active_critical_alerts_from_journal(state_dir) == [
        "monitor:throughput"
    ]

    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, persisted
    )
    with pytest.raises(control.ControlError, match="journal-ahead"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: pytest.fail(
                f"journal-ahead finalization must not submit: {argv}"
            ),
            now=40.0,
        )
    assert not (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    ).exists()


@pytest.mark.parametrize(
    "blocking_key", ["monitor:corrupt", "monitor:capacity-gate"]
)
def test_finalization_preserves_mixed_noncapacity_integrity_holds(
    tmp_path, monkeypatch, blocking_key
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    _record_finalization_critical_alert(
        state_dir, dedupe_key=blocking_key, now=21.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )

    with pytest.raises(control.ControlError, match="non-capacity"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: pytest.fail(
                f"blocked finalization must not submit: {argv}"
            ),
            now=40.0,
        )

    persisted = control.load_control(state_dir)
    assert persisted["finalization"]["state"] == "idle"
    assert persisted["finalization"]["consumed_capacity_incidents"] == []
    assert set(persisted["admission_safety_hold"]["reasons"]) == {
        "monitor:throughput",
        blocking_key,
    }
    assert all(
        alert["resolved_at"] is None
        for alert in persisted["alerts"]
        if alert["dedupe_key"] in {"monitor:throughput", blocking_key}
    )
    assert not (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    ).exists()


def test_finalization_request_intent_is_readonly_at_link_crash(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    request_path = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    )
    original_link = control.os.link
    crashed = {"value": False}

    def link_then_crash(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if (
            not crashed["value"]
            and Path(destination) == request_path
        ):
            crashed["value"] = True
            raise RuntimeError("death immediately after intent link")

    monkeypatch.setattr(control.os, "link", link_then_crash)
    with pytest.raises(RuntimeError, match="death immediately"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: pytest.fail(
                f"uncommitted finalization must not submit: {argv}"
            ),
            now=40.0,
        )

    assert request_path.is_file()
    assert request_path.stat().st_nlink == 1
    assert request_path.stat().st_mode & 0o222 == 0
    request_before = request_path.read_bytes()
    assert json.loads(request_before)["requested_timestamp"] == 40.0
    assert control.load_control(state_dir)["finalization"]["state"] == "idle"

    monkeypatch.setattr(control.os, "link", original_link)
    submitted = iter(("901", "902"))
    recovered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 41.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=41.0,
    )
    assert recovered["requested_timestamp"] == 40.0
    assert request_path.read_bytes() == request_before
    assert request_path.stat().st_mode & 0o222 == 0


def test_orphaned_finalization_intent_replays_after_alert_and_hold_clearance(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    with monkeypatch.context() as boundary:
        boundary.setattr(
            control,
            "_consume_finalization_capacity_incidents_locked",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("death after readonly request intent")
            ),
        )
        with pytest.raises(RuntimeError, match="readonly request intent"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                submit_runner=lambda argv: pytest.fail(
                    f"uncommitted finalization must not submit: {argv}"
                ),
                now=40.0,
            )

    request_path = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    )
    request_before = request_path.read_bytes()
    frozen = json.loads(request_before)
    assert frozen["consumed_capacity_incidents"] == [
        "monitor:throughput"
    ]
    assert len(
        frozen["capacity_alert_resolution_plan"][0]["alert_ids"]
    ) == 1
    assert control.load_control(state_dir)["finalization"]["state"] == "idle"

    # Reproduce reconciliation after the trigger process dies: the alert is resolved,
    # two clean monitor polls become durable, and the capacity-remediation workflow
    # independently clears the latched hold before this request is replayed.
    control.resolve_alert(
        state_dir, dedupe_key="monitor:throughput", now=41.0
    )
    control.update_admission_safety_hold(
        state_dir,
        clean_poll=True,
        semantic_scan_clean=True,
        now=42.0,
    )
    control.update_admission_safety_hold(
        state_dir,
        clean_poll=True,
        semantic_scan_clean=True,
        now=43.0,
    )
    transition_id = "orphan-intent-clearance-fixture"
    with control.admission_boundary_lock(state_dir):
        with control.control_lock(state_dir):
            clearance = control.load_control(state_dir)
            control._consume_capacity_remediation_incidents_locked(
                state_dir,
                clearance,
                transition_id=transition_id,
                incident_keys=("monitor:throughput",),
                now=44.0,
            )
            control._activate_operator_hold(
                clearance["admission_safety_hold"],
                reason=f"capacity-transition:{transition_id}",
                now=44.0,
            )
            control._clear_capacity_operator_hold(
                clearance["admission_safety_hold"],
                transition_id=transition_id,
                now=44.0,
            )
            control._save_control(state_dir, clearance, now=44.0)
    cleared = control.load_control(state_dir)
    assert cleared["admission_safety_hold"]["active"] is False
    assert control._active_critical_alerts(cleared) == []
    assert control._active_critical_alerts_from_journal(state_dir) == []

    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir, cleared, captured_timestamp=45.0
    )
    submitted = iter(("901", "902"))
    recovered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=control.SchedulerSnapshot((), 45.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=45.0,
    )
    assert request_path.read_bytes() == request_before
    assert recovered["requested_timestamp"] == 40.0
    assert recovered["semantic_evidence"]["path"] == str(
        evidence_path.resolve()
    )
    assert recovered["consumed_capacity_incidents"] == [
        "monitor:throughput"
    ]
    assert control.load_control(
        state_dir, verify_files=True
    )["admission_safety_hold"]["active"] is False


def test_orphaned_finalization_intent_preserves_new_noncapacity_conflict(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    with monkeypatch.context() as boundary:
        boundary.setattr(
            control,
            "_consume_finalization_capacity_incidents_locked",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("death after readonly request intent")
            ),
        )
        with pytest.raises(RuntimeError, match="readonly request intent"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                now=40.0,
            )
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:corrupt", now=41.0
    )
    protected_paths = (
        state_dir / control.CONTROL_FILENAME,
        state_dir / control.TRANSITION_JOURNAL,
        state_dir / control.ALERT_JOURNAL,
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME,
    )
    before = {path: path.read_bytes() for path in protected_paths}

    with pytest.raises(control.ControlError, match="non-capacity"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 42.0),
            submit_runner=lambda argv: pytest.fail(
                f"conflicted finalization must not submit: {argv}"
            ),
            now=42.0,
        )
    assert {path: path.read_bytes() for path in protected_paths} == before
    assert control.load_control(state_dir)["finalization"]["state"] == "idle"


def test_orphaned_finalization_intent_rejects_new_capacity_alert_identity(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=20.0
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    with monkeypatch.context() as boundary:
        boundary.setattr(
            control,
            "_consume_finalization_capacity_incidents_locked",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("death after readonly request intent")
            ),
        )
        with pytest.raises(RuntimeError, match="readonly request intent"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                now=40.0,
            )
    request_path = (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    )
    frozen = json.loads(request_path.read_text(encoding="utf-8"))
    frozen_alert_id = frozen["capacity_alert_resolution_plan"][0][
        "alert_ids"
    ][0]
    control.resolve_alert(
        state_dir, dedupe_key="monitor:throughput", now=41.0
    )
    _record_finalization_critical_alert(
        state_dir, dedupe_key="monitor:throughput", now=42.0
    )
    active_alert_id = next(
        alert["alert_id"]
        for alert in reversed(control.load_control(state_dir)["alerts"])
        if alert["dedupe_key"] == "monitor:throughput"
        and alert["resolved_at"] is None
    )
    assert active_alert_id != frozen_alert_id
    protected_paths = (
        state_dir / control.CONTROL_FILENAME,
        state_dir / control.TRANSITION_JOURNAL,
        state_dir / control.ALERT_JOURNAL,
        request_path,
    )
    before = {path: path.read_bytes() for path in protected_paths}

    with pytest.raises(
        control.ControlError, match="newly active capacity alert"
    ):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 43.0),
            submit_runner=lambda argv: pytest.fail(
                f"unbound capacity alert must not submit: {argv}"
            ),
            now=43.0,
        )
    assert {path: path.read_bytes() for path in protected_paths} == before
    assert control.load_control(state_dir)["finalization"]["state"] == "idle"


def test_finalization_semantic_override_mismatch_has_zero_mutation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, report = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    override = copy.deepcopy(report)
    override["captured_timestamp"] = 41.0
    protected_paths = (
        state_dir / control.CONTROL_FILENAME,
        state_dir / control.TRANSITION_JOURNAL,
        state_dir / control.ALERT_JOURNAL,
    )
    before = {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    }

    with pytest.raises(
        control.ControlError, match="override differs from sealed evidence"
    ):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            semantic_report=override,
            scheduler_reader=lambda: pytest.fail(
                "mismatched evidence must not query scheduler truth"
            ),
            submit_runner=lambda argv: pytest.fail(
                f"mismatched evidence must not submit: {argv}"
            ),
            now=40.0,
        )

    assert {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    } == before
    assert not (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    ).exists()


def test_finalization_semantic_evidence_rejects_hardlink_with_zero_mutation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    hardlink = evidence_path.with_name("hardlinked-semantic.json")
    os.link(evidence_path, hardlink)
    assert evidence_path.stat().st_nlink == 2
    protected_paths = (
        state_dir / control.CONTROL_FILENAME,
        state_dir / control.TRANSITION_JOURNAL,
        state_dir / control.ALERT_JOURNAL,
    )
    before = {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    }

    with pytest.raises(
        control.ControlError, match="sealed monitor semantic evidence"
    ):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=hardlink,
            scheduler_reader=lambda: pytest.fail(
                "hardlinked evidence must not query scheduler"
            ),
            now=40.0,
        )
    assert {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    } == before
    assert not (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    ).exists()


def test_finalization_semantic_evidence_rejects_ancestor_symlink_before_read(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    outside = tmp_path / "outside-semantic"
    outside.mkdir()
    report = _complete_semantic_report(current)
    report["captured_timestamp"] = 40.0
    outside_report = outside / "semantic.json"
    outside_report.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )
    outside_report.chmod(0o444)
    linked_parent = state_dir / "monitoring" / "linked-semantic"
    linked_parent.parent.mkdir(parents=True, exist_ok=True)
    linked_parent.symlink_to(outside, target_is_directory=True)
    lexical_report = linked_parent / outside_report.name
    protected_paths = (
        state_dir / control.CONTROL_FILENAME,
        state_dir / control.TRANSITION_JOURNAL,
        state_dir / control.ALERT_JOURNAL,
    )
    before = {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    }

    with pytest.raises(control.ControlError, match="traverses a symlink"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=lexical_report,
            scheduler_reader=lambda: pytest.fail(
                "symlinked evidence must not query scheduler"
            ),
            now=40.0,
        )
    assert {
        path: path.read_bytes() if path.exists() else None
        for path in protected_paths
    } == before
    assert not (
        state_dir
        / control.FINALIZER_STATE_DIRNAME
        / control.FINALIZER_REQUEST_INTENT_FILENAME
    ).exists()


def test_finalization_request_cleans_only_exact_owned_temporary(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    finalizer_root = state_dir / control.FINALIZER_STATE_DIRNAME
    finalizer_root.mkdir()
    owned = (
        finalizer_root
        / f".{control.FINALIZER_REQUEST_INTENT_FILENAME}.deadbeef.tmp"
    )
    owned.write_bytes(b'{"partial":')
    owned.chmod(0o600)
    unknown = (
        finalizer_root
        / f".{control.FINALIZER_REQUEST_INTENT_FILENAME}.not-ours.tmp"
    )
    unknown.write_bytes(b"unknown\n")
    unknown.chmod(0o600)

    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    assert requested["state"] == "draining"
    assert not owned.exists()
    assert unknown.read_bytes() == b"unknown\n"
    owned_pattern = re.compile(
        rf"^\.{re.escape(control.FINALIZER_REQUEST_INTENT_FILENAME)}\."
        r"[a-z0-9_]{8}\.tmp$"
    )
    assert not any(
        owned_pattern.fullmatch(path.name)
        for path in finalizer_root.iterdir()
    )


def test_autonomous_request_preserves_and_verifies_existing_manual_completion(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    marker = _publish_manual_final_complete_fixture(state_dir, current)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )

    before = (state_dir / control.CONTROL_FILENAME).read_bytes()
    observed = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler_reader=lambda: pytest.fail(
            "verified manual completion must not query scheduler truth"
        ),
        submit_runner=lambda argv: pytest.fail(
            f"verified manual completion must not submit a finalizer: {argv}"
        ),
        now=41.0,
    )
    assert observed["state"] == "idle"
    assert observed["active_job"] is None
    assert observed["successor_job"] is None
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() == before

    marker_path = (
        Path(current["finalization"]["output_root"])
        / control.FINAL_COMPLETE_FILENAME
    )
    marker_path.chmod(0o644)
    forged = json.loads(marker_path.read_text(encoding="utf-8"))
    forged["trusted_qids"] -= 1
    marker_path.write_text(
        json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path.chmod(0o444)
    with pytest.raises(control.ControlError, match="malformed"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler_reader=lambda: pytest.fail(
                "invalid manual completion must fail before scheduler access"
            ),
            submit_runner=lambda argv: pytest.fail(
                f"invalid manual completion must not submit a finalizer: {argv}"
            ),
            now=42.0,
        )
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() == before
    assert marker["autonomous_finalizer"] is None


def test_finalizer_active_submission_adoption_rejects_unexpected_dependency(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    with pytest.raises(control.ControlError, match="sbatch was rejected"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, "", "synthetic rejection"
            ),
            now=40.0,
        )
    record = control.load_control(state_dir)["finalization"]["active_job"]
    assert record["state"] == "submitting"
    assert record["dependency_job_id"] is None
    ambiguous = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "PENDING",
                record["job_token"],
                record["sbatch_path"],
                dependency="afterany:777",
            ),
        ),
        41.0,
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="scheduler identity is ambiguous"
    ):
        control._submit_finalizer_job(
            state_dir,
            field="active_job",
            attempt=1,
            dependency_job_id=None,
            snapshot=ambiguous,
            submit_runner=lambda argv: pytest.fail(
                f"ambiguous active finalizer must not be resubmitted: {argv}"
            ),
            now=41.0,
        )
    persisted = control.load_control(state_dir)["finalization"]["active_job"]
    assert persisted["job_id"] is None
    assert persisted["state"] == "submitting"


def test_finalizer_reconcile_waits_for_unaccepted_submission_then_rebuilds(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )

    def die_before_sbatch(_argv):
        raise RuntimeError("death before sbatch")

    with pytest.raises(RuntimeError, match="before sbatch"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=die_before_sbatch,
            now=40.0,
        )
    stranded = control.load_control(state_dir)["finalization"]
    assert stranded["active_job"]["job_id"] is None
    assert stranded["active_job"]["state"] == "submitting"
    assert stranded["next_attempt"] == 2

    with pytest.raises(
        control.SchedulerVisibilityPending, match="visibility grace"
    ):
        control.reconcile_autonomous_finalization(
            state_dir,
            snapshot=control.SchedulerSnapshot((), 41.0),
            submit_runner=lambda argv: pytest.fail(
                f"visibility grace must prevent resubmission: {argv}"
            ),
            now=41.0,
        )

    fresh_ids = iter(("902", "903"))
    rebuilt = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 221.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(fresh_ids) + "\n", ""
        ),
        now=221.0,
    )
    assert rebuilt["active_job"]["job_id"] == "902"
    assert rebuilt["active_job"]["attempt"] == 2
    assert rebuilt["successor_job"]["job_id"] == "903"
    assert rebuilt["successor_job"]["attempt"] == 3
    assert rebuilt["next_attempt"] == 4


def test_finalizer_reconcile_adopts_sbatch_accepted_before_job_id_commit(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_parse = control._parse_sbatch_job_id

    def die_after_accept(proc):
        assert original_parse(proc) == "901"
        raise RuntimeError("death after sbatch acceptance")

    with monkeypatch.context() as boundary:
        boundary.setattr(control, "_parse_sbatch_job_id", die_after_accept)
        with pytest.raises(RuntimeError, match="after sbatch acceptance"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                submit_runner=lambda argv: subprocess.CompletedProcess(
                    argv, 0, "901\n", ""
                ),
                now=40.0,
            )
    stranded = control.load_control(state_dir)["finalization"]
    active = stranded["active_job"]
    assert active["job_id"] is None
    accepted = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "RUNNING",
                active["job_token"],
                active["sbatch_path"],
            ),
        ),
        41.0,
    )
    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=accepted,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "902\n", ""
        ),
        now=41.0,
    )
    assert recovered["active_job"]["job_id"] == "901"
    assert recovered["active_job"]["state"] == "submitted"
    assert recovered["successor_job"]["job_id"] == "902"
    assert recovered["successor_job"]["dependency_job_id"] == "901"
    assert (
        sum(
            row["event"] == "finalizer_submitted"
            and row["details"] == {
                "field": "active_job",
                "attempt": 1,
                "job_id": "901",
            }
            for row in control.load_control(state_dir)["transition_history"]
        )
        == 1
    )


def test_finalizer_worker_self_adopts_sbatch_accepted_before_job_id_commit(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_parse = control._parse_sbatch_job_id

    def die_after_accept(proc):
        assert original_parse(proc) == "901"
        raise RuntimeError("death after sbatch acceptance")

    with monkeypatch.context() as boundary:
        boundary.setattr(control, "_parse_sbatch_job_id", die_after_accept)
        with pytest.raises(RuntimeError, match="after sbatch acceptance"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                submit_runner=lambda argv: subprocess.CompletedProcess(
                    argv, 0, "901\n", ""
                ),
                now=40.0,
            )
    stranded = control.load_control(state_dir)["finalization"]
    active = stranded["active_job"]
    assert active["job_id"] is None
    accepted = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "RUNNING",
                active["job_token"],
                active["sbatch_path"],
            ),
        ),
        41.0,
    )
    scheduler_calls = {"count": 0}

    def scheduler_reader():
        scheduler_calls["count"] += 1
        if scheduler_calls["count"] == 1:
            return accepted
        return _autonomous_scheduler(
            control.load_control(state_dir)["finalization"],
            captured_at=41.0 + scheduler_calls["count"],
        )

    submissions = []

    def submit(argv):
        submissions.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "902\n", "")

    returncode = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=stranded["intent_id"],
        attempt=1,
        job_id="901",
        scheduler_reader=scheduler_reader,
        submit_runner=submit,
        finalize_runner=lambda *_args, **_kwargs: {"complete": False},
        now=41.0,
    )
    assert returncode == 75
    recovered = control.load_control(state_dir)["finalization"]
    assert recovered["active_job"]["job_id"] == "901"
    assert recovered["successor_job"]["job_id"] == "902"
    assert recovered["successor_job"]["dependency_job_id"] == "901"
    assert recovered["worker_attempts"][0]["status"] == "pending"
    assert len(submissions) == 1
    assert (
        sum(
            row["event"] == "finalizer_submitted"
            and row["details"]
            == {"field": "active_job", "attempt": 1, "job_id": "901"}
            for row in control.load_control(state_dir)["transition_history"]
        )
        == 1
    )


def test_foreign_finalizer_worker_is_fenced_without_control_mutation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    before = (state_dir / control.CONTROL_FILENAME).read_bytes()
    with pytest.raises(control.ControllerFenced, match="job ID differs"):
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="999",
            scheduler_reader=lambda: pytest.fail(
                "foreign bound worker must be fenced before scheduler access"
            ),
            now=41.0,
        )
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() == before


@pytest.mark.parametrize(
    "crash_position",
    ("after_submission_journal", "after_control_replace"),
)
def test_finalizer_reconcile_replays_job_id_commit_crash_exactly_once(
    tmp_path, monkeypatch, crash_position
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_save = control._save_control
    crashed = {"value": False}

    def crash_at_commit(path, payload, *, now):
        active = payload["finalization"].get("active_job")
        should_crash = (
            not crashed["value"]
            and isinstance(active, dict)
            and active.get("job_id") == "901"
            and active.get("state") == "submitted"
        )
        if should_crash and crash_position == "after_submission_journal":
            crashed["value"] = True
            raise RuntimeError("death after submission journal")
        result = original_save(path, payload, now=now)
        if should_crash and crash_position == "after_control_replace":
            crashed["value"] = True
            raise RuntimeError("death after control replace")
        return result

    with monkeypatch.context() as boundary:
        boundary.setattr(control, "_save_control", crash_at_commit)
        with pytest.raises(RuntimeError, match="death after"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                submit_runner=lambda argv: subprocess.CompletedProcess(
                    argv, 0, "901\n", ""
                ),
                now=40.0,
            )
    stranded = control.load_control(state_dir)["finalization"]
    active = stranded["active_job"]
    accepted = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "RUNNING",
                active["job_token"],
                active["sbatch_path"],
            ),
        ),
        41.0,
    )
    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=accepted,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "902\n", ""
        ),
        now=41.0,
    )
    assert recovered["active_job"]["job_id"] == "901"
    assert recovered["successor_job"]["job_id"] == "902"
    persisted = control.load_control(state_dir)
    assert (
        sum(
            row["event"] == "finalizer_submitted"
            and row["details"] == {
                "field": "active_job",
                "attempt": 1,
                "job_id": "901",
            }
            for row in persisted["transition_history"]
        )
        == 1
    )
    assert (
        sum(
            row["event"] == "job_submitted"
            and row["details"] == {
                "field": "active_job",
                "attempt": 1,
                "job_id": "901",
            }
            for row in persisted["finalization"]["history"]
        )
        == 1
    )


def test_live_publisher_replaces_exact_cancelled_successor_once(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    cancelled = _autonomous_scheduler(
        requested, captured_at=41.0, successor_state="CANCELLED"
    )
    calls = []

    def replace(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "903\n", "")

    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=cancelled,
        submit_runner=replace,
        now=41.0,
    )
    assert len(calls) == 1
    assert recovered["active_job"]["job_id"] == "901"
    assert recovered["successor_job"]["job_id"] == "903"
    assert recovered["successor_job"]["attempt"] == 3
    assert recovered["successor_job"]["dependency_job_id"] == "901"
    retired = [
        row
        for row in recovered["history"]
        if row["event"] == "successor_retired_for_replacement"
    ]
    assert len(retired) == 1
    assert retired[0]["details"]["scheduler_state"] == "CANCELLED"
    assert retired[0]["details"]["successor_job_id"] == "902"


def test_live_publisher_rejects_successor_that_completed_execution(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    completed = _autonomous_scheduler(
        requested, captured_at=41.0, successor_state="COMPLETED"
    )
    control_path = state_dir / control.CONTROL_FILENAME
    before = control_path.read_bytes()
    with pytest.raises(
        control.SchedulerAmbiguity, match="execution terminal state"
    ):
        control.reconcile_autonomous_finalization(
            state_dir,
            snapshot=completed,
            submit_runner=lambda argv: pytest.fail(
                f"executed successor must not be replaced: {argv}"
            ),
            now=41.0,
        )
    assert control_path.read_bytes() == before


def test_live_publisher_does_not_replace_successor_during_retirement_transaction(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    binding = control._ensure_successor_retirement_intent(
        state_dir,
        publisher_record=requested["active_job"],
        now=41.0,
    )
    cancelled = _autonomous_scheduler(
        requested, captured_at=42.0, successor_state="CANCELLED"
    )
    reconciled = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=cancelled,
        submit_runner=lambda argv: pytest.fail(
            f"retirement transaction must fence replacement: {argv}"
        ),
        now=42.0,
    )
    assert reconciled["successor_retirement"] == binding
    assert reconciled["successor_job"]["job_id"] == "902"
    assert reconciled["next_attempt"] == 3


def test_live_publisher_replaces_missing_successor_after_visibility_grace(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    active = requested["active_job"]
    missing = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "RUNNING",
                active["job_token"],
                active["sbatch_path"],
            ),
        ),
        221.0,
    )
    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=missing,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        now=221.0,
    )
    assert recovered["successor_job"]["job_id"] == "903"
    assert recovered["successor_job"]["attempt"] == 3
    assert any(
        row["event"] == "successor_retired_for_replacement"
        and row["details"]["reason"] == "missing_after_visibility_grace"
        for row in recovered["history"]
    )


def test_live_publisher_replaces_rejected_submitting_successor(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    calls = {"count": 0}

    def reject_successor(argv):
        calls["count"] += 1
        if calls["count"] == 1:
            return subprocess.CompletedProcess(argv, 0, "901\n", "")
        return subprocess.CompletedProcess(
            argv, 1, "", "synthetic successor rejection"
        )

    with pytest.raises(control.ControlError, match="sbatch was rejected"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((), 40.0),
            submit_runner=reject_successor,
            now=40.0,
        )
    stranded = control.load_control(state_dir)["finalization"]
    assert stranded["active_job"]["job_id"] == "901"
    assert stranded["successor_job"]["job_id"] is None
    active = stranded["active_job"]
    running = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "RUNNING",
                active["job_token"],
                active["sbatch_path"],
            ),
        ),
        41.0,
    )
    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=running,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        now=41.0,
    )
    assert recovered["successor_job"]["job_id"] == "903"
    assert recovered["successor_job"]["attempt"] == 3
    assert any(
        row["event"] == "successor_retired_for_replacement"
        and row["details"]["reason"] == "submission_rejected"
        for row in recovered["history"]
    )


def test_autonomous_finalization_requests_chain_before_drain_and_is_live(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))

    def submit(argv):
        assert "--parsable" in argv
        return subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        )

    result = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=submit,
        now=40.0,
    )
    assert result["state"] == "draining"
    assert result["active_job"]["job_id"] == "901"
    assert result["successor_job"]["job_id"] == "902"
    assert result["successor_job"]["dependency_job_id"] == "901"
    assert result["next_attempt"] == 3
    assert control.load_control(state_dir)["desired_state"] == "paused"
    for field in ("active_job", "successor_job"):
        payload = Path(result[field]["sbatch_path"]).read_text(
            encoding="utf-8"
        )
        assert "#SBATCH --no-requeue" in payload
        assert "#SBATCH --signal=B:USR1@1200" in payload
    status = control.live_status(
        state_dir,
        snapshot=_autonomous_scheduler(result),
        now=41.0,
    )
    assert status["finalization"]["state"] == "draining"
    assert status["effective_admission_ceiling"] == 0
    assert status["immutable_sha256"] == current["immutable_sha256"]
    assert status["captured_timestamp"] == 41.0


def test_finalizer_child_deadlines_leave_bounded_signal_cleanup_margin():
    assert control.FINALIZER_COMMAND_DEADLINE_SECONDS <= 39_600
    assert control.FINALIZER_SNAPSHOT_DEADLINE_SECONDS <= 39_600
    assert control.FINALIZER_CHILD_TERMINATE_GRACE_SECONDS < 1_200


def test_finalizer_live_cell_is_durable_drain_pending_before_validation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    monkeypatch.setattr(
        control,
        "_exact_live_cell_task_actions",
        lambda _state, snapshot, **_kwargs: (
            (
                ["777_0"]
                if any(job.job_id == "777_0" and job.active for job in snapshot.jobs)
                else []
            ),
            [],
        ),
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    cell = control.SchedulerJob(
        "777_0",
        "asys-dispatch-fixture",
        "RUNNING",
        control.CELL_INTENT_PREFIX + "fixture",
    )
    submitted = iter(("901", "902"))
    signals = []
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((cell,), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        cancel_runner=lambda argv: (
            signals.append(list(argv))
            or subprocess.CompletedProcess(argv, 0, "", "")
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(requested, extra_jobs=(cell,))
    delays = []
    returncode = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=requested["intent_id"],
        attempt=1,
        job_id="901",
        scheduler_reader=lambda: scheduler,
        finalize_runner=lambda *_args, **_kwargs: pytest.fail(
            "semantic finalization must wait for the live cell"
        ),
        sleep_runner=delays.append,
        now=41.0,
    )
    assert returncode == 75
    finalization = control.load_control(state_dir)["finalization"]
    assert finalization["state"] == "draining"
    assert finalization["phase_evidence"]["validating"] is None
    assert finalization["worker_attempts"][0]["status"] == "pending"
    pending = next(
        row
        for row in reversed(finalization["history"])
        if row["event"] == "drain_pending"
    )
    assert pending["details"]["active_cell_task_ids"] == ["777_0"]
    assert pending["details"]["retry_not_before_timestamp"] == 101.0
    assert (
        control._finalizer_drain_backoff_seconds(finalization, now=42.0)
        == 59.0
    )
    assert delays == [control.FINALIZER_DRAIN_RETRY_SECONDS]
    assert signals == [
        ["scancel", "--batch", "--signal=USR1", "777_0"]
    ]


def test_finalizer_live_exact_controller_is_drain_pending(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    controller_sbatch = (tmp_path / "dispatcher-controller.sbatch").resolve()
    controller_sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    controller_sbatch.chmod(0o444)
    controller_intent = "a" * 32
    controller_token = control.job_token(
        "dispatcher", 1, controller_intent
    )
    with control.control_lock(state_dir):
        current = control.load_control(state_dir)
        current["controllers"]["dispatcher"]["active"] = {
            "generation": 1,
            "intent_token": controller_intent,
            "job_token": controller_token,
            "sbatch_path": str(controller_sbatch),
            "sbatch_sha256": _sha(controller_sbatch),
            "dependency_job_id": None,
            "job_id": "700",
            "state": "running",
        }
        control._save_control(state_dir, current, now=39.0)
    controller_job = control.SchedulerJob(
        "700",
        "asys-s5-dispatch-g000001-aaaaaaaa",
        "RUNNING",
        controller_token,
        f"sbatch {controller_sbatch}",
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((controller_job,), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(
        requested, extra_jobs=(controller_job,)
    )
    delays = []
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            finalize_runner=lambda *_args, **_kwargs: pytest.fail(
                "semantic finalization must wait for the live controller"
            ),
            sleep_runner=delays.append,
            now=41.0,
        )
        == 75
    )
    finalization = control.load_control(state_dir)["finalization"]
    assert finalization["state"] == "draining"
    assert finalization["phase_evidence"]["validating"] is None
    pending = next(
        row
        for row in reversed(finalization["history"])
        if row["event"] == "drain_pending"
    )
    assert pending["details"]["active_controller_job_ids"] == ["700"]
    assert delays == [control.FINALIZER_DRAIN_RETRY_SECONDS]


def test_finalizer_drain_rejects_noncanonical_name_for_durable_controller(
    tmp_path
):
    state_dir, _ = initialize(tmp_path)
    controller_sbatch = (tmp_path / "dispatcher-controller.sbatch").resolve()
    controller_sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    controller_sbatch.chmod(0o444)
    controller_intent = "a" * 32
    controller_token = control.job_token(
        "dispatcher", 1, controller_intent
    )
    with control.control_lock(state_dir):
        current = control.load_control(state_dir)
        current["controllers"]["dispatcher"]["active"] = {
            "generation": 1,
            "intent_token": controller_intent,
            "job_token": controller_token,
            "sbatch_path": str(controller_sbatch),
            "sbatch_sha256": _sha(controller_sbatch),
            "dependency_job_id": None,
            "job_id": "700",
            "state": "running",
        }
        control._save_control(state_dir, current, now=39.0)
    legacy_named = control.SchedulerJob(
        "700",
        "asys-s5-dispatcher",
        "RUNNING",
        controller_token,
        f"sbatch {controller_sbatch}",
    )
    with pytest.raises(
        control.SchedulerAmbiguity,
        match="provenance is invalid",
    ):
        control._exact_live_finalizer_draining_controllers(
            control.load_control(state_dir),
            control.SchedulerSnapshot((legacy_named,), 40.0),
        )


@pytest.mark.parametrize(
    ("job_name", "comment"),
    (
        ("asys-s5-dispatch-g000009-foreign", ""),
        (
            "asys-s5-fleet-g000009-foreign",
            control.TOKEN_PREFIX + ";broken",
        ),
        ("asys-s5-dispatcher", ""),
    ),
)
def test_finalizer_unmapped_controller_blocks_before_semantic_validation(
    tmp_path, monkeypatch, job_name, comment
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    foreign = control.SchedulerJob(
        "799",
        job_name,
        "RUNNING",
        comment,
        "/foreign/controller.sbatch",
    )
    scheduler = _autonomous_scheduler(requested, extra_jobs=(foreign,))
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            finalize_runner=lambda *_args, **_kwargs: pytest.fail(
                "unmapped controller must block before semantic validation"
            ),
            now=41.0,
        )
        == 2
    )
    finalization = control.load_control(state_dir)["finalization"]
    assert finalization["state"] == "blocked"
    assert finalization["last_error"]["kind"] == "FinalizerInvariantError"
    assert (
        "missing or malformed provenance"
        in finalization["last_error"]["message"]
    )
    assert finalization["phase_evidence"]["validating"] is None


@pytest.mark.parametrize(
    ("job_name", "comment"),
    (
        ("asys-s5-dispatch-g000009-foreign", ""),
        (
            "asys-s5-fleet-g000009-foreign",
            control.TOKEN_PREFIX + ";broken",
        ),
        ("asys-s5-fleet", ""),
    ),
)
def test_finalizer_quiescence_rejects_controller_name_without_valid_comment(
    job_name, comment
):
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "799",
                job_name,
                "RUNNING",
                comment,
                "/foreign/controller.sbatch",
            ),
        ),
        41.0,
    )
    with pytest.raises(
        control.SchedulerAmbiguity,
        match="controller-named jobs with missing or malformed provenance",
    ):
        control._require_complete_quiescent_scheduler(
            snapshot, operation="finalization"
        )


def test_finalizer_incomplete_post_pause_scheduler_cut_retries_while_draining(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    complete = _autonomous_scheduler(requested, captured_at=41.0)
    incomplete = control.SchedulerSnapshot(
        complete.jobs,
        42.0,
        squeue_ok=True,
        sacct_ok=False,
        errors=("sacct unavailable",),
    )
    snapshots = iter((complete, incomplete))
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: next(snapshots),
            finalize_runner=lambda *_args, **_kwargs: pytest.fail(
                "incomplete drain truth must retry before validation"
            ),
            now=41.0,
        )
        == 75
    )
    finalization = control.load_control(state_dir)["finalization"]
    assert finalization["state"] == "draining"
    assert finalization["last_error"]["kind"] == "FinalizerSchedulerReadError"
    assert finalization["last_error"]["transient"] is True
    assert finalization["worker_attempts"][0]["status"] == "retryable_failed"
    assert finalization["phase_evidence"]["validating"] is None


def test_finalizer_replays_crashed_pause_signal_transaction_before_validation(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    monkeypatch.setattr(
        control,
        "_exact_live_cell_task_actions",
        lambda _state, snapshot, **_kwargs: (
            (
                ["777_0"]
                if any(job.job_id == "777_0" and job.active for job in snapshot.jobs)
                else []
            ),
            [],
        ),
    )
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    cell = control.SchedulerJob(
        "777_0",
        "asys-dispatch-fixture",
        "RUNNING",
        control.CELL_INTENT_PREFIX + "fixture",
    )
    submitted = iter(("901", "902"))
    signal_calls = []
    crash_once = {"value": True}

    def signal(argv):
        signal_calls.append(list(argv))
        if crash_once["value"]:
            crash_once["value"] = False
            raise RuntimeError("death during pause signal")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="pause signal"):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=evidence_path,
            scheduler=control.SchedulerSnapshot((cell,), 40.0),
            submit_runner=lambda argv: subprocess.CompletedProcess(
                argv, 0, next(submitted) + "\n", ""
            ),
            cancel_runner=signal,
            now=40.0,
        )
    stranded = control.load_control(state_dir)
    assert stranded["finalization"]["state"] == "requested"
    assert stranded["desired_state"] == "paused"
    assert stranded["drain_requested"] is True
    assert stranded["drain_intent"]["state"] == "signaling"

    scheduler = _autonomous_scheduler(
        stranded["finalization"], extra_jobs=(cell,)
    )
    delays = []
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=stranded["finalization"]["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            cancel_runner=signal,
            finalize_runner=lambda *_args, **_kwargs: pytest.fail(
                "replayed pause still has a live cell and must remain draining"
            ),
            sleep_runner=delays.append,
            now=41.0,
        )
        == 75
    )
    replayed = control.load_control(state_dir)
    assert replayed["drain_intent"]["state"] == "complete"
    assert replayed["finalization"]["state"] == "draining"
    assert replayed["finalization"]["phase_evidence"]["validating"] is None
    assert len(signal_calls) == 2
    assert delays == [control.FINALIZER_DRAIN_RETRY_SECONDS]


def test_finalizer_quiescent_drain_enters_validation_once(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(requested)
    observed_states = []

    def finalizer(*_args, **_kwargs):
        observed_states.append(
            control.load_control(state_dir)["finalization"]["state"]
        )
        return {"complete": False}

    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            finalize_runner=finalizer,
            now=41.0,
        )
        == 75
    )
    finalization = control.load_control(state_dir)["finalization"]
    assert observed_states == ["validating"]
    assert finalization["state"] == "validating"
    assert finalization["phase_evidence"]["validating"] is not None
    assert (
        sum(
            row["event"] == "validating_entered"
            and row["details"]["phase"] == "validating"
            for row in finalization["history"]
        )
        == 1
    )


def test_active_finalization_retrigger_uses_next_attempt_after_promotion(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    promoted = control._promote_finalizer_successor(
        state_dir, job_id="902", attempt=2, now=41.0
    )
    assert promoted["attempt"] == 2
    before_retrigger = control.load_control(state_dir)["finalization"]
    assert before_retrigger["active_job"]["attempt"] == 2
    assert before_retrigger["successor_job"] is None
    assert before_retrigger["next_attempt"] == 3

    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir,
        control.load_control(state_dir),
        captured_timestamp=42.0,
    )
    calls = []

    def submit(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "903\n", "")

    retriggered = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=_autonomous_scheduler(
            before_retrigger, captured_at=42.0
        ),
        submit_runner=submit,
        now=42.0,
    )
    assert len(calls) == 1
    assert retriggered["active_job"]["job_id"] == "902"
    assert retriggered["active_job"]["attempt"] == 2
    assert retriggered["successor_job"]["job_id"] == "903"
    assert retriggered["successor_job"]["attempt"] == 3
    assert retriggered["successor_job"]["dependency_job_id"] == "902"
    assert retriggered["next_attempt"] == 4


def test_active_finalization_retrigger_rebuilds_both_none_from_next_attempt(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    assert requested["next_attempt"] == 3
    control._reset_finalizer_chain_for_retry(
        state_dir, reason="synthetic terminal pair", now=41.0
    )
    reset = control.load_control(state_dir)["finalization"]
    assert reset["active_job"] is None
    assert reset["successor_job"] is None
    assert reset["next_attempt"] == 3

    later_evidence, _ = _write_autonomous_finalization_evidence(
        state_dir,
        control.load_control(state_dir),
        captured_timestamp=42.0,
    )
    fresh_ids = iter(("903", "904"))
    rebuilt = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=later_evidence,
        scheduler=control.SchedulerSnapshot((), 42.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(fresh_ids) + "\n", ""
        ),
        now=42.0,
    )
    assert rebuilt["active_job"]["job_id"] == "903"
    assert rebuilt["active_job"]["attempt"] == 3
    assert rebuilt["successor_job"]["job_id"] == "904"
    assert rebuilt["successor_job"]["attempt"] == 4
    assert rebuilt["successor_job"]["dependency_job_id"] == "903"
    assert rebuilt["next_attempt"] == 5


def test_autonomous_finalize_publishes_only_after_bound_successor_retirement(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    initial = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, initial
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    current = control.load_control(state_dir)
    cancelled = False
    scheduler_timestamp = 41.0

    def scheduler_reader():
        nonlocal scheduler_timestamp
        scheduler_timestamp += 1.0
        finalization = control.load_control(state_dir)["finalization"]
        return _autonomous_scheduler(
            finalization,
            captured_at=scheduler_timestamp,
            successor_state="CANCELLED" if cancelled else "PENDING",
        )

    def cancel_runner(argv):
        nonlocal cancelled
        assert argv == ["scancel", "902"]
        binding = control.load_control(state_dir)["finalization"][
            "successor_retirement"
        ]
        assert binding["receipt_path"] is None
        assert Path(binding["intent_path"]).stat().st_mode & 0o222 == 0
        marker_path = (
            Path(requested["output_root"]) / control.FINAL_COMPLETE_FILENAME
        )
        assert not marker_path.exists()
        assert (Path(requested["output_root"]) / "snapshot").is_dir()
        cancelled = True
        return subprocess.CompletedProcess(argv, 0, "", "")

    def command_runner(argv, environment, timeout):
        assert environment["ASYS_IMMUTABLE_PINS_SHA256"] == current[
            "immutable_sha256"
        ]
        if "--cadence" in argv:
            assert timeout == control.FINALIZER_COMMAND_DEADLINE_SECONDS
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(_complete_semantic_report(current)), ""
            )
        cache_root = Path(argv[argv.index("--out-dir") + 1])
        _write_complete_primary_cache(cache_root, current)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def snapshot_builder(snapshot_root, sources, environment):
        del environment
        return _write_final_snapshot_fixture(
            snapshot_root,
            sources,
            completed_at="1970-01-01T00:00:41Z",
        )

    result = control.finalize_sweep(
        state_dir,
        output_root=Path(requested["output_root"]),
        scheduler_reader=scheduler_reader,
        cancel_runner=cancel_runner,
        fleet_evidence_reader=lambda state: _final_fleet_evidence(state),
        command_runner=command_runner,
        snapshot_builder=snapshot_builder,
        lock_scanner=lambda _state: [],
        writer_guard=lambda _state, _path: control.nullcontext(),
        publisher_finalizer=publisher,
        now=41.0,
    )
    assert cancelled is True
    assert result["schema_version"] == 4
    assert result["autonomous_finalizer"]["publisher"]["job_id"] == "901"
    assert result["autonomous_finalizer"]["successor"]["job_id"] == "902"
    retirement = result["autonomous_finalizer"]["successor_retirement"]
    assert retirement["receipt_id"]
    assert result["zero_writer_proof"]["publisher_finalizer"]["job_id"] == "901"
    assert result["zero_writer_proof"]["active_finalizer_job_ids"] == ["901"]
    assert result["zero_writer_proof"]["other_active_finalizer_job_ids"] == []
    assert result["zero_writer_proof"]["successor_retirement"] == retirement
    finalization = control.load_control(state_dir)["finalization"]
    assert finalization["state"] != "complete"
    assert finalization["successor_job"]["state"] == "cancelled"

    completed = control._commit_autonomous_finalization_complete(
        state_dir, now=50.0
    )
    assert completed["state"] == "complete"
    assert completed["completion_marker"]["final_id"] == result["final_id"]
    completed_control = control.load_control(state_dir)

    snapshot_root = (
        Path(requested["output_root"]) / control.FINAL_SNAPSHOT_DIRNAME
    )
    payload = next(snapshot_root.glob("*/payload.bin"))
    original_payload = payload.read_bytes()
    snapshot_root.chmod(0o755)
    payload.parent.chmod(0o755)
    payload.chmod(0o644)
    payload.write_bytes(b"post-completion snapshot corruption\n")
    payload.chmod(0o444)
    payload.parent.chmod(0o555)
    snapshot_root.chmod(0o555)
    with pytest.raises(control.ControlError, match="payload checksum drifted"):
        control.load_control(state_dir)
    snapshot_root.chmod(0o755)
    payload.parent.chmod(0o755)
    payload.chmod(0o644)
    payload.write_bytes(original_payload)
    payload.chmod(0o444)
    payload.parent.chmod(0o555)
    snapshot_root.chmod(0o555)
    assert control.load_control(state_dir)["finalization"]["state"] == "complete"

    marker_path = (
        Path(requested["output_root"]) / control.FINAL_COMPLETE_FILENAME
    )
    marker_path.chmod(0o644)
    forged = json.loads(marker_path.read_text(encoding="utf-8"))
    forged["autonomous_finalizer"]["successor_retirement"][
        "receipt_sha256"
    ] = "0" * 64
    without_id = dict(forged)
    without_id.pop("final_id")
    forged["final_id"] = control.sha256_value(without_id)
    marker_path.write_text(
        json.dumps(forged, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path.chmod(0o444)
    with pytest.raises(
        control.ControlError, match="marker checksum drifted"
    ):
        control.load_control(state_dir)
    with pytest.raises(
        control.ControlError, match="marker checksum drifted"
    ):
        control.live_status(
            state_dir,
            snapshot=control.SchedulerSnapshot((), 51.0),
            now=51.0,
        )
    with pytest.raises(
        control.ControlError, match="retirement binding drifted"
    ):
        control._verify_final_complete(
            Path(requested["output_root"]),
            control=completed_control,
        )
    marker_path.unlink()
    with pytest.raises(
        control.ControlError, match="marker is missing"
    ):
        control.load_control(state_dir)


def test_autonomous_finalizer_retries_transient_and_blocks_semantic_failure(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    result = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(result)
    monkeypatch.setattr(
        control,
        "record_alert",
        lambda *_args, **_kwargs: {},
    )
    transient = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=result["intent_id"],
        attempt=1,
        job_id="901",
        scheduler_reader=lambda: scheduler,
        finalize_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            control.FinalizerSchedulerReadError(
                "sacct temporarily unavailable"
            )
        ),
        now=41.0,
    )
    assert transient == 75
    retry_state = control.load_control(state_dir)["finalization"]
    assert retry_state["state"] == "validating"
    assert retry_state["last_error"]["transient"] is True

    generation_two = _autonomous_scheduler(
        retry_state,
        captured_at=42.0,
        active_state="COMPLETED",
        successor_state="RUNNING",
    )
    blocked = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=result["intent_id"],
        attempt=2,
        job_id="902",
        scheduler_reader=lambda: generation_two,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        finalize_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            control.ControlError(
                "semantic acceptance contains duplicate QIDs"
            )
        ),
        now=42.0,
    )
    assert blocked == 2
    blocked_state = control.load_control(state_dir)["finalization"]
    assert blocked_state["state"] == "blocked"
    assert blocked_state["last_error"]["transient"] is False


@pytest.mark.parametrize(
    "drain_signal",
    (control.signal.SIGUSR1, control.signal.SIGTERM),
    ids=("usr1", "term"),
)
def test_finalizer_signal_during_snapshot_reaps_before_lock_release_and_retries(
    tmp_path,
    monkeypatch,
    drain_signal,
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(requested)
    monkeypatch.setattr(
        control, "FINALIZER_CHILD_POLL_SECONDS", 0.02
    )
    monkeypatch.setattr(
        control, "FINALIZER_CHILD_TERMINATE_GRACE_SECONDS", 0.1
    )
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    controller = control._FinalizerInterruptionController()
    child_pid_path = tmp_path / "snapshot-child.pid"
    signal_sent = threading.Event()
    contender_done = threading.Event()
    contender_observation: dict[str, object] = {}

    def pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def send_drain_signal() -> None:
        deadline = time.monotonic() + 5.0
        while not child_pid_path.exists():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)
        os.kill(os.getpid(), drain_signal)
        signal_sent.set()

    def contend_for_finalizer_lock() -> None:
        if not signal_sent.wait(timeout=5.0):
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with control.finalizer_lock(state_dir):
                    pid = int(child_pid_path.read_text(encoding="utf-8"))
                    contender_observation["child_alive"] = pid_is_alive(pid)
                    contender_observation["acquired"] = True
                    contender_done.set()
                    return
            except control.ControlError as exc:
                assert "singleton lock is already held" in str(exc)
                time.sleep(0.01)

    signal_thread = threading.Thread(target=send_drain_signal)
    contender_thread = threading.Thread(target=contend_for_finalizer_lock)
    signal_thread.start()
    contender_thread.start()

    child_program = "\n".join(
        (
            "import os, signal, sys, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "with open(sys.argv[1], 'w', encoding='utf-8') as handle:",
            "    handle.write(str(os.getpid()))",
            "    handle.flush()",
            "    os.fsync(handle.fileno())",
            "while True:",
            "    time.sleep(1)",
        )
    )

    def interrupted_snapshot(*_args, **_kwargs):
        with control.finalizer_lock(state_dir):
            control._run_finalizer_process(
                [
                    str(control.sys.executable),
                    "-c",
                    child_program,
                    str(child_pid_path),
                ],
                environment=os.environ,
                timeout_seconds=60.0,
                interruption=controller,
            )
        pytest.fail("interrupted snapshot child unexpectedly completed")

    returncode = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=requested["intent_id"],
        attempt=1,
        job_id="901",
        scheduler_reader=lambda: scheduler,
        finalize_runner=interrupted_snapshot,
        interruption_controller=controller,
        now=41.0,
    )
    signal_thread.join(timeout=5.0)
    contender_thread.join(timeout=5.0)
    assert returncode == 75
    assert signal_sent.is_set()
    assert contender_done.is_set()
    assert contender_observation == {
        "child_alive": False,
        "acquired": True,
    }
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert pid_is_alive(child_pid) is False
    assert controller.active_child is None
    interrupted = control.load_control(state_dir)["finalization"]
    assert interrupted["state"] == "validating"
    assert interrupted["worker_attempts"][0]["status"] == "retryable_failed"
    assert interrupted["last_error"]["kind"] == "FinalizerInterruptedError"
    assert interrupted["last_error"]["transient"] is True

    generation_two = _autonomous_scheduler(
        interrupted,
        captured_at=42.0,
        active_state="COMPLETED",
        successor_state="RUNNING",
    )
    resumed = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=requested["intent_id"],
        attempt=2,
        job_id="902",
        scheduler_reader=lambda: generation_two,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        finalize_runner=lambda *_args, **_kwargs: {"complete": False},
        now=42.0,
    )
    assert resumed == 75
    recovered = control.load_control(state_dir)["finalization"]
    assert recovered["active_job"]["job_id"] == "902"
    assert recovered["successor_job"]["job_id"] == "903"
    assert [row["status"] for row in recovered["worker_attempts"]] == [
        "retryable_failed",
        "pending",
    ]


def test_finalizer_scheduler_timeout_is_counted_once_before_promotion(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    scheduler_calls = {"count": 0}

    def timeout_reader():
        scheduler_calls["count"] += 1
        raise TimeoutError("sacct read deadline")

    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=timeout_reader,
            now=41.0,
        )
        == 75
    )
    failed = control.load_control(state_dir)["finalization"]
    assert failed["state"] == requested["state"]
    assert failed["active_job"] == requested["active_job"]
    assert failed["successor_job"] == requested["successor_job"]
    assert failed["attempts"] == 1
    assert failed["worker_attempts"][0]["status"] == "retryable_failed"
    assert failed["last_error"]["kind"] == "FinalizerSchedulerReadError"
    assert failed["last_error"]["transient"] is True

    # Re-entering the same Slurm worker identity replays the durable result.  It
    # neither redraws the attempt nor performs another scheduler observation.
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=timeout_reader,
            now=42.0,
        )
        == 75
    )
    replayed = control.load_control(state_dir)["finalization"]
    assert replayed["attempts"] == 1
    assert replayed["worker_attempts"] == failed["worker_attempts"]
    assert scheduler_calls["count"] == 1


def test_finalizer_scheduler_failure_at_retry_cap_blocks_exactly_once(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    with control.control_lock(state_dir):
        current = control.load_control(state_dir)
        finalization = current["finalization"]
        rows = []
        for index in range(
            1, control.FINALIZER_MAX_TRANSIENT_ATTEMPTS
        ):
            timestamp = float(index)
            rows.append(
                {
                    "attempt": 1000 + index,
                    "job_id": str(10_000 + index),
                    "status": "retryable_failed",
                    "started_at": control.utc_timestamp(timestamp),
                    "started_timestamp": timestamp,
                    "finished_at": control.utc_timestamp(timestamp),
                    "finished_timestamp": timestamp,
                }
            )
        finalization["worker_attempts"] = rows
        finalization["attempts"] = len(rows)
        control._save_control(state_dir, current, now=40.5)
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )

    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: (_ for _ in ()).throw(
                TimeoutError("final scheduler read deadline")
            ),
            now=41.0,
        )
        == 2
    )
    blocked = control.load_control(state_dir)["finalization"]
    assert blocked["state"] == "blocked"
    assert blocked["attempts"] == control.FINALIZER_MAX_TRANSIENT_ATTEMPTS
    assert blocked["worker_attempts"][-1]["status"] == "blocked"
    assert blocked["last_error"]["kind"] == "FinalizerRetryLimitError"
    assert blocked["last_error"]["transient"] is False

    before = copy.deepcopy(blocked)
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: pytest.fail(
                "blocked replay must not read scheduler truth"
            ),
            now=42.0,
        )
        == 2
    )
    assert control.load_control(state_dir)["finalization"] == before


@pytest.mark.parametrize(
    ("state", "cache_marker_sha256", "expected_kind"),
    (
        ("validating", None, control.FinalizerSemanticError),
        ("retiring_fleet", None, control.FinalizerProvenanceError),
        ("snapshotting", None, control.FinalizerCacheError),
        ("snapshotting", "a" * 64, control.FinalizerSnapshotError),
        ("requested", None, control.FinalizerInvariantError),
    ),
)
def test_finalizer_control_failures_are_typed_by_durable_phase(
    state, cache_marker_sha256, expected_kind
):
    typed, retryable = control._typed_finalizer_failure(
        control.ControlError("sealed artifact drift"),
        finalization={
            "state": state,
            "phase_evidence": {
                "snapshotting": {
                    "cache_marker_sha256": cache_marker_sha256
                }
            },
        },
    )
    assert isinstance(typed, expected_kind)
    assert retryable is False


def test_finalizer_unknown_exception_blocks_instead_of_retrying(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(requested)
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            finalize_runner=lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(
                    RuntimeError("unclassified finalizer failure")
                )
            ),
            now=41.0,
        )
        == 2
    )
    blocked = control.load_control(state_dir)["finalization"]
    assert blocked["state"] == "blocked"
    assert blocked["last_error"]["kind"] == "RuntimeError"
    assert blocked["last_error"]["transient"] is False
    assert blocked["worker_attempts"][0]["status"] == "blocked"


def test_finalizer_phase_entries_and_bindings_resume_monotonically(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    output_root = Path(requested["output_root"]).resolve()
    semantic_paths = {
        "semantic_report_path": str(
            (output_root / control.FINAL_SEMANTIC_FILENAME).resolve()
        ),
        "semantic_intent_path": str(
            (
                output_root / control.FINAL_SEMANTIC_INTENT_FILENAME
            ).resolve()
        ),
        "semantic_preflight_path": str(
            (
                output_root / control.FINAL_SEMANTIC_PREFLIGHT_FILENAME
            ).resolve()
        ),
        "semantic_report_sha256": None,
        "semantic_intent_sha256": None,
        "semantic_intent_id": None,
        "semantic_preflight_sha256": None,
        "semantic_preflight_id": None,
    }
    semantic_complete = {
        "semantic_report_sha256": "1" * 64,
        "semantic_intent_sha256": "2" * 64,
        "semantic_intent_id": "3" * 64,
        "semantic_preflight_sha256": "4" * 64,
        "semantic_preflight_id": "5" * 64,
    }
    retirement_root = (
        output_root / control.FINAL_FLEET_RETIREMENT_DIRNAME
    ).resolve()
    retirement_paths = {
        "semantic_report_sha256": "1" * 64,
        "semantic_preflight_sha256": "4" * 64,
        "semantic_preflight_id": "5" * 64,
        "retirement_root": str(retirement_root),
        "retirement_intent_path": str(
            (
                retirement_root
                / control.FINAL_FLEET_RETIREMENT_INTENT_FILENAME
            ).resolve()
        ),
        "retirement_intent_sha256": None,
        "retirement_id": None,
        "retirement_marker_path": str(
            (
                retirement_root
                / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
            ).resolve()
        ),
        "retirement_marker_sha256": None,
        "retirement_completion_id": None,
    }
    retirement_complete = {
        "retirement_intent_sha256": "6" * 64,
        "retirement_id": "7" * 64,
        "retirement_marker_sha256": "8" * 64,
        "retirement_completion_id": "9" * 64,
    }
    cache_root = (
        output_root / control.FINAL_ANALYSIS_CACHE_DIRNAME
    ).resolve()
    snapshot_root = (output_root / control.FINAL_SNAPSHOT_DIRNAME).resolve()
    snapshot_paths = {
        "retirement_root": str(retirement_root),
        "retirement_marker_path": str(
            (
                retirement_root
                / control.FINAL_FLEET_RETIREMENT_COMPLETE_FILENAME
            ).resolve()
        ),
        "retirement_marker_sha256": "8" * 64,
        "retirement_completion_id": "9" * 64,
        "cache_root": str(cache_root),
        "cache_marker_path": str(
            (cache_root / "ingest_manifest_v1.json").resolve()
        ),
        "cache_marker_sha256": None,
        "cache_generation_id": None,
        "cache_tree_sha256": None,
        "snapshot_root": str(snapshot_root),
        "snapshot_marker_path": str(
            (snapshot_root / "SNAPSHOT_COMPLETE.json").resolve()
        ),
        "snapshot_marker_sha256": None,
        "snapshot_id": None,
    }
    snapshot_complete = {
        "cache_marker_sha256": "a" * 64,
        "cache_generation_id": "b" * 64,
        "cache_tree_sha256": "c" * 64,
        "snapshot_marker_sha256": "d" * 64,
        "snapshot_id": "resumable-snapshot",
    }
    original_save = control._save_control

    def crash_after_durable_save(*args, **kwargs):
        original_save(*args, **kwargs)
        raise RuntimeError("simulated death after durable phase save")

    phases = (
        ("validating", semantic_paths, semantic_complete),
        ("retiring_fleet", retirement_paths, retirement_complete),
        ("snapshotting", snapshot_paths, snapshot_complete),
    )
    event_time = 42.0
    for phase, entry, completion in phases:
        for bindings in (entry, completion):
            monkeypatch.setattr(
                control, "_save_control", crash_after_durable_save
            )
            with pytest.raises(
                RuntimeError, match="death after durable phase save"
            ):
                control._record_autonomous_finalizer_phase(
                    state_dir,
                    publisher_finalizer=publisher,
                    phase=phase,
                    bindings=bindings,
                    now=event_time,
                )
            monkeypatch.setattr(control, "_save_control", original_save)
            replayed = control._record_autonomous_finalizer_phase(
                state_dir,
                publisher_finalizer=publisher,
                phase=phase,
                bindings=bindings,
                now=event_time + 0.5,
            )
            assert all(
                replayed[field] == value
                for field, value in bindings.items()
            )
            event_time += 1.0

    completed_phases = control.load_control(state_dir)["finalization"]
    assert completed_phases["state"] == "snapshotting"
    assert all(
        completed_phases["phase_evidence"][phase] is not None
        for phase in ("validating", "retiring_fleet", "snapshotting")
    )
    before_replay = (
        state_dir / control.CONTROL_FILENAME
    ).read_bytes()
    control._record_autonomous_finalizer_phase(
        state_dir,
        publisher_finalizer=publisher,
        phase="validating",
        bindings=semantic_complete,
        now=60.0,
    )
    assert (
        control.load_control(state_dir)["finalization"]["state"]
        == "snapshotting"
    )
    assert (state_dir / control.CONTROL_FILENAME).read_bytes() == before_replay


def test_autonomous_finalizer_publishes_complete_state_and_cancels_successor(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    result = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    running = _autonomous_scheduler(result)
    terminal = _autonomous_scheduler(
        result, captured_at=42.0, successor_state="CANCELLED"
    )
    # Worker authentication/reconciliation and the scheduler-confirmed drain
    # barrier each consume a complete cut before fleet retirement begins.
    snapshots = iter((running, running, running, terminal))
    output_root = Path(result["output_root"])
    output_root.mkdir(parents=True)
    final_id = "f" * 64
    marker = output_root / control.FINAL_COMPLETE_FILENAME
    cancelled = []

    def cancel(argv):
        binding = control.load_control(state_dir)["finalization"][
            "successor_retirement"
        ]
        assert binding["receipt_path"] is None
        intent_path = Path(binding["intent_path"])
        assert intent_path.stat().st_mode & 0o222 == 0
        cancelled.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def finalize_runner(*_args, **kwargs):
        control._retire_autonomous_finalizer_successor(
            state_dir,
            publisher_record=kwargs["publisher_finalizer"],
            scheduler_reader=kwargs["scheduler_reader"],
            cancel_runner=kwargs["cancel_runner"],
            now=41.0,
        )
        _seed_complete_finalizer_phase_evidence(
            state_dir,
            publisher=kwargs["publisher_finalizer"],
            output_root=output_root,
            now=41.0,
        )
        marker.write_text(
            json.dumps({"final_id": final_id}) + "\n", encoding="utf-8"
        )
        marker.chmod(0o444)
        return {"complete": True, "final_id": final_id}

    monkeypatch.setattr(
        control,
        "_verify_final_complete",
        lambda *_args, **_kwargs: {
            "complete": True,
            "final_id": final_id,
        },
    )
    returncode = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=result["intent_id"],
        attempt=1,
        job_id="901",
        scheduler_reader=lambda: next(snapshots),
        finalize_runner=finalize_runner,
        cancel_runner=cancel,
        now=41.0,
    )
    assert returncode == 0
    completed = control.load_control(state_dir)["finalization"]
    assert completed["state"] == "complete"
    assert completed["completion_marker"]["final_id"] == final_id
    assert completed["successor_retirement"]["receipt_id"]
    assert completed["successor_job"]["state"] == "cancelled"
    assert cancelled == [["scancel", "902"]]
    terminal_status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 50.0),
        now=50.0,
    )
    assert terminal_status["finalization"]["state"] == "complete"
    assert terminal_status["finalization"]["chain_healthy"] is True
    assert terminal_status["finalization"]["namespace_active_job_ids"] == []
    assert terminal_status["healthy"] is True
    later_terminal_status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 110.0),
        now=110.0,
    )
    watchdog_decision = external_watchdog.decide(
        (terminal_status, later_terminal_status),
        expected_release_id=terminal_status["release_id"],
        expected_git_commit=terminal_status["git_commit"],
        expected_control_sha256=terminal_status["immutable_sha256"],
    )
    assert watchdog_decision.action is None
    assert "already complete" in watchdog_decision.reason


def test_autonomous_finalizer_reconcile_promotes_afterany_successor_once(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    result = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    successor = result["successor_job"]
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "901",
                "asys-s5-final-a000001",
                "COMPLETED",
                result["active_job"]["job_token"],
                result["active_job"]["sbatch_path"],
            ),
            control.SchedulerJob(
                "902",
                "asys-s5-final-a000002",
                "RUNNING",
                successor["job_token"],
                successor["sbatch_path"],
                dependency="afterany:901",
            ),
        ),
        41.0,
    )
    reconciled = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=snapshot,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        now=41.0,
    )
    assert reconciled["active_job"]["job_id"] == "902"
    assert reconciled["active_job"]["attempt"] == 2
    assert reconciled["successor_job"]["job_id"] == "903"
    assert reconciled["successor_job"]["attempt"] == 3
    assert reconciled["successor_job"]["dependency_job_id"] == "902"


def test_successor_retirement_persists_intent_before_cancel_and_adopts_terminal_replay(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    running = _autonomous_scheduler(requested, captured_at=41.0)
    observed_intent = {}

    def failed_cancel(argv):
        binding = control.load_control(state_dir)["finalization"][
            "successor_retirement"
        ]
        observed_intent.update(binding)
        assert binding["receipt_path"] is None
        assert Path(binding["intent_path"]).stat().st_mode & 0o222 == 0
        return subprocess.CompletedProcess(argv, 1, "", "temporary scancel failure")

    snapshots = iter((running, running))
    with pytest.raises(
        control.ControlError, match="scheduler rejected exact"
    ):
        control._retire_autonomous_finalizer_successor(
            state_dir,
            publisher_record=publisher,
            scheduler_reader=lambda: next(snapshots),
            cancel_runner=failed_cancel,
            now=41.0,
        )
    after_failure = control.load_control(
        state_dir, verify_files=True
    )["finalization"]
    assert after_failure["state"] != "complete"
    assert after_failure["successor_retirement"] == observed_intent
    assert after_failure["successor_retirement"]["receipt_path"] is None
    assert not (
        Path(after_failure["output_root"]) / control.FINAL_COMPLETE_FILENAME
    ).exists()

    still_running = iter((running, running))
    with pytest.raises(
        control.SchedulerVisibilityPending, match="not terminal"
    ):
        control._retire_autonomous_finalizer_successor(
            state_dir,
            publisher_record=publisher,
            scheduler_reader=lambda: next(still_running),
            cancel_runner=lambda argv: subprocess.CompletedProcess(
                argv, 0, "", ""
            ),
            now=41.5,
        )
    assert (
        control.load_control(state_dir)["finalization"][
            "successor_retirement"
        ]["receipt_path"]
        is None
    )

    terminal = _autonomous_scheduler(
        requested, captured_at=42.0, successor_state="CANCELLED"
    )
    completed = control._retire_autonomous_finalizer_successor(
        state_dir,
        publisher_record=publisher,
        scheduler_reader=lambda: terminal,
        cancel_runner=lambda argv: pytest.fail(
            f"terminal successor must be adopted without cancellation: {argv}"
        ),
        now=42.0,
    )
    receipt_path = Path(completed["receipt_path"])
    assert receipt_path.stat().st_mode & 0o222 == 0
    assert completed["receipt_id"]
    replay = control._retire_autonomous_finalizer_successor(
        state_dir,
        publisher_record=publisher,
        scheduler_reader=lambda: pytest.fail(
            "sealed terminal retirement must replay without scheduler access"
        ),
        cancel_runner=lambda argv: pytest.fail(
            f"sealed terminal retirement must not recancel: {argv}"
        ),
        now=43.0,
    )
    assert replay == completed


def test_successor_retirement_rejects_foreign_finalizer_and_receipt_tamper(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    foreign = control.SchedulerJob(
        "999",
        "asys-s5-final-a999999",
        "RUNNING",
        control.FINALIZER_JOB_TOKEN_PREFIX + ";malformed=foreign",
        "/foreign/finalizer.sbatch",
    )
    ambiguous = _autonomous_scheduler(
        requested, captured_at=41.0, extra_jobs=(foreign,)
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="unmapped autonomous finalizer"
    ):
        control._retire_autonomous_finalizer_successor(
            state_dir,
            publisher_record=publisher,
            scheduler_reader=lambda: ambiguous,
            cancel_runner=lambda argv: pytest.fail(
                f"ambiguity must fence cancellation: {argv}"
            ),
            now=41.0,
        )
    binding = control.load_control(state_dir)["finalization"][
        "successor_retirement"
    ]
    assert binding["receipt_path"] is None

    terminal = _autonomous_scheduler(
        requested, captured_at=42.0, successor_state="COMPLETED"
    )
    completed = control._retire_autonomous_finalizer_successor(
        state_dir,
        publisher_record=publisher,
        scheduler_reader=lambda: terminal,
        cancel_runner=lambda argv: pytest.fail(
            f"terminal successor must not be cancelled: {argv}"
        ),
        now=42.0,
    )
    receipt_path = Path(completed["receipt_path"])
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    receipt_path.chmod(0o444)
    with pytest.raises(
        control.ControlError, match="receipt checksum drifted"
    ):
        control.load_control(state_dir, verify_files=True)


def test_reconcile_adopts_marker_crash_without_replacing_bound_finalizer_chain(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    terminal = _autonomous_scheduler(
        requested,
        captured_at=42.0,
        active_state="COMPLETED",
        successor_state="CANCELLED",
    )
    retirement_snapshot = _autonomous_scheduler(
        requested, captured_at=41.0, successor_state="CANCELLED"
    )
    control._retire_autonomous_finalizer_successor(
        state_dir,
        publisher_record=publisher,
        scheduler_reader=lambda: retirement_snapshot,
        cancel_runner=lambda argv: pytest.fail(
            f"terminal successor must not be cancelled: {argv}"
        ),
        now=41.0,
    )
    _seed_complete_finalizer_phase_evidence(
        state_dir,
        publisher=publisher,
        output_root=Path(requested["output_root"]),
        now=41.0,
    )
    final_id = "f" * 64
    marker_path = (
        Path(requested["output_root"]) / control.FINAL_COMPLETE_FILENAME
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"final_id": final_id}) + "\n", encoding="utf-8"
    )
    marker_path.chmod(0o444)
    monkeypatch.setattr(
        control,
        "_verify_final_complete",
        lambda *_args, **_kwargs: {
            "complete": True,
            "final_id": final_id,
        },
    )
    live_retired_successor = _autonomous_scheduler(
        requested,
        captured_at=41.5,
        active_state="COMPLETED",
        successor_state="RUNNING",
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="retired successor"
    ):
        control.reconcile_autonomous_finalization(
            state_dir,
            snapshot=live_retired_successor,
            submit_runner=lambda argv: pytest.fail(
                f"ambiguous marker adoption must not submit: {argv}"
            ),
            now=41.5,
        )
    assert control.load_control(state_dir)["finalization"]["state"] != "complete"
    reconciled = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=terminal,
        submit_runner=lambda argv: pytest.fail(
            f"marker adoption must not replace its bound chain: {argv}"
        ),
        now=42.0,
    )
    assert reconciled["state"] == "complete"
    assert reconciled["active_job"]["job_id"] == "901"
    assert reconciled["successor_job"]["job_id"] == "902"
    assert reconciled["successor_retirement"]["receipt_id"]
    replay = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=terminal,
        submit_runner=lambda argv: pytest.fail(
            f"completed finalization must remain fenced: {argv}"
        ),
        now=43.0,
    )
    assert replay == reconciled
    control_path = state_dir / control.CONTROL_FILENAME
    before_replay = control_path.read_bytes()
    assert (
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: terminal,
            submit_runner=lambda argv: pytest.fail(
                f"completed worker replay must not submit: {argv}"
            ),
            now=43.5,
        )
        == 0
    )
    assert control_path.read_bytes() == before_replay
    live_retired_successor = _autonomous_scheduler(
        reconciled,
        captured_at=43.6,
        active_state="COMPLETED",
        successor_state="RUNNING",
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="active retired"
    ):
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: live_retired_successor,
            now=43.6,
        )
    assert control_path.read_bytes() == before_replay
    with pytest.raises(control.ControllerFenced, match="fenced"):
        control._promote_finalizer_successor(
            state_dir, job_id="901", attempt=1, now=44.0
        )


def test_reconcile_receipt_crash_before_marker_starts_fresh_bound_pair(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    publisher = control._promote_finalizer_successor(
        state_dir, job_id="901", attempt=1, now=41.0
    )
    terminal = _autonomous_scheduler(
        requested,
        captured_at=42.0,
        active_state="COMPLETED",
        successor_state="CANCELLED",
    )
    control._retire_autonomous_finalizer_successor(
        state_dir,
        publisher_record=publisher,
        scheduler_reader=lambda: _autonomous_scheduler(
            requested, captured_at=41.0, successor_state="CANCELLED"
        ),
        cancel_runner=lambda argv: pytest.fail(
            f"terminal successor must not be cancelled: {argv}"
        ),
        now=41.0,
    )
    old_binding = control.load_control(state_dir)["finalization"][
        "successor_retirement"
    ]
    fresh_ids = iter(("903", "904"))
    reconciled = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=terminal,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(fresh_ids) + "\n", ""
        ),
        now=42.0,
    )
    assert reconciled["active_job"]["job_id"] == "903"
    assert reconciled["successor_job"]["job_id"] == "904"
    assert reconciled["successor_retirement"] is None
    assert Path(old_binding["intent_path"]).is_file()
    assert Path(old_binding["receipt_path"]).is_file()
    assert any(
        row["event"] == "chain_reset_for_retry"
        and row["details"]["successor_retirement_transaction_id"]
        == old_binding["transaction_id"]
        for row in reconciled["history"]
    )


def test_trusted_json_preimages_reject_symlink_and_replacement_races(
    tmp_path, monkeypatch
):
    root = tmp_path / "state"
    root.mkdir()
    target = root / "ledger.json"
    target.write_text('{"schema_version":1}\n', encoding="utf-8")
    alias = root / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(control.SchedulerAmbiguity, match="symlink"):
        control._stable_mutable_json_preimage(
            alias,
            required_parent=root,
            description="fixture ledger",
        )

    replacement = root / "replacement.json"
    replacement.write_text('{"schema_version":2}\n', encoding="utf-8")
    original_read = control.os.read
    replaced = {"value": False}

    def read_then_replace(descriptor, size):
        block = original_read(descriptor, size)
        if block and not replaced["value"]:
            replaced["value"] = True
            control.os.replace(replacement, target)
        return block

    monkeypatch.setattr(control.os, "read", read_then_replace)
    with pytest.raises(control.SchedulerAmbiguity, match="changed"):
        control._stable_mutable_json_preimage(
            target,
            required_parent=root,
            description="fixture ledger",
        )


def test_trusted_readonly_preimage_rejects_writable_or_symlink_files(
    tmp_path,
):
    target = tmp_path / "immutable.sbatch"
    target.write_text("#!/bin/bash\n", encoding="utf-8")
    with pytest.raises(control.SchedulerAmbiguity, match="read-only"):
        control._stable_readonly_preimage(
            target, description="fixture sbatch"
        )
    target.chmod(0o444)
    alias = tmp_path / "alias.sbatch"
    alias.symlink_to(target)
    with pytest.raises(control.SchedulerAmbiguity, match="symlink"):
        control._stable_readonly_preimage(
            alias, description="fixture sbatch"
        )


def _write_dispatcher_provenance_ledger(
    state_dir: Path, *, updated_at: float
) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = dispatcher._empty_ledger(now=updated_at)
    ledger_path = state_dir / "ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger_path


def test_trusted_dispatcher_rejects_forged_comment_namespace_row(tmp_path):
    state_dir = tmp_path / "dispatcher"
    _write_dispatcher_provenance_ledger(state_dir, updated_at=100.0)
    forged = control.SchedulerJob(
        "777",
        "unrelated-job",
        "PENDING",
        control.CELL_INTENT_PREFIX + "forged",
        "/usr/bin/sbatch /tmp/unrelated.sbatch",
        partition="ou_bcs_normal",
        qos="normal",
    )
    snapshot = control.SchedulerSnapshot((forged,), 100.0)

    with pytest.raises(
        control.SchedulerAmbiguity, match="mapping differs"
    ):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=100.0,
        )


@pytest.mark.parametrize("ledger_state", ("missing", "stale", "symlink"))
def test_dispatcher_capacity_reconciliation_rejects_unsafe_ledger_preimage(
    tmp_path, ledger_state
):
    state_dir = tmp_path / "dispatcher"
    if ledger_state == "stale":
        _write_dispatcher_provenance_ledger(
            state_dir, updated_at=100.0
        )
    elif ledger_state == "symlink":
        state_dir.mkdir(parents=True)
        target = tmp_path / "outside-ledger.json"
        target.write_text(
            json.dumps(dispatcher._empty_ledger(now=1_000.0)) + "\n",
            encoding="utf-8",
        )
        (state_dir / "ledger.json").symlink_to(target)

    with pytest.raises(control.SchedulerAmbiguity):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=control.SchedulerSnapshot((), 1_000.0),
            now=1_000.0,
        )


def test_dispatcher_capacity_reconciliation_rejects_ledger_replacement(
    tmp_path, monkeypatch
):
    state_dir = tmp_path / "dispatcher"
    ledger_path = _write_dispatcher_provenance_ledger(
        state_dir, updated_at=1_000.0
    )
    replacement = state_dir / "replacement.json"
    replacement.write_text(
        json.dumps(dispatcher._empty_ledger(now=1_001.0)) + "\n",
        encoding="utf-8",
    )
    original_read = control.os.read
    replaced = {"value": False}

    def read_then_replace(descriptor, size):
        block = original_read(descriptor, size)
        if block and not replaced["value"]:
            replaced["value"] = True
            control.os.replace(replacement, ledger_path)
        return block

    monkeypatch.setattr(control.os, "read", read_then_replace)
    with pytest.raises(control.SchedulerAmbiguity, match="changed"):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=control.SchedulerSnapshot((), 1_000.0),
            now=1_000.0,
        )


def _trusted_fleet_binding_fixture(
    tmp_path: Path, *, symlink_ledgers: bool = False
) -> tuple[Path, dict[str, dict[str, object]], control.SchedulerSnapshot]:
    state_dir = tmp_path / "dispatcher"
    _write_dispatcher_provenance_ledger(state_dir, updated_at=1_000.0)
    transaction_dir = tmp_path / "fleet-transactions"
    script_path = (
        transaction_dir / "sbatch" / "g000001" / "8B-r00.sbatch"
    )
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "#!/bin/bash\n"
        "#SBATCH --partition=server_partition\n"
        "#SBATCH --qos=server_qos\n"
        "#SBATCH --gres=gpu:a100:1\n",
        encoding="utf-8",
    )
    script_path.chmod(0o444)
    script_sha256 = _sha(script_path)
    intent_token = "a" * 32
    comment = (
        "asys-s5-fleet:pool=schema5-v1;profile=8B;"
        "replica=8B-r00;generation=1;"
        f"intent={intent_token};fleet={'f' * 64}"
    )
    ledger_parent = transaction_dir / "ledgers"
    if symlink_ledgers:
        real_parent = tmp_path / "real-fleet-ledgers"
        real_parent.mkdir()
        ledger_parent.parent.mkdir(parents=True, exist_ok=True)
        ledger_parent.symlink_to(real_parent, target_is_directory=True)
        ledger_path = ledger_parent / "g000001.json"
    else:
        ledger_parent.mkdir(parents=True)
        ledger_path = ledger_parent / "g000001.json"
    fleet_ledger = {
        "rollout_generation": 1,
        "replicas": {
            "8B-r00": {
                "attempts": [
                    {
                        "intent_token": intent_token,
                        "job_id": "123",
                        "sbatch_path": str(script_path),
                        "sbatch_sha256": script_sha256,
                        "allocated_gpus": 1,
                        "submission_transport": (
                            control.fleet_transactions.STDIN_EXACT_SUBMISSION_TRANSPORT
                        ),
                        "submission_argv_sha256": (
                            control.fleet_transactions.submission_argv_sha256(
                                comment
                            )
                        ),
                        "state": "committed",
                    }
                ]
            }
        },
    }
    ledger_path.write_text(
        json.dumps(fleet_ledger, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = {
        "123": {
            "job_name": "asys-s5-serve-8B-r00",
            "comment": comment,
            "sbatch_path": str(script_path),
            "sbatch_sha256": script_sha256,
            "intent_token": intent_token,
            "replica_id": "8B-r00",
            "serving_profile": "8B",
            "ledger_generation": 1,
            "ledger_path": str(ledger_path),
            "ledger_sha256": _sha(ledger_path),
            "partition": "server_partition",
            "qos": "server_qos",
            "allocated_gpus": 1,
            "gpu_type": "a100",
        }
    }
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                "123",
                "asys-s5-serve-8B-r00",
                "RUNNING",
                comment,
                shlex.join(
                    control.fleet_transactions.submission_argv(comment)
                ),
                partition="server_partition",
                qos="server_qos",
            ),
        ),
        1_000.0,
    )
    return state_dir, binding, snapshot


def _trusted_cell_binding_fixture(
    tmp_path: Path,
    *,
    task_job_id: str = "123_0",
    intent_state: str = "reconciled",
    record_state: str = "active",
) -> tuple[Path, dict, control.SchedulerSnapshot]:
    state_dir = tmp_path / "dispatcher"
    state_dir.mkdir(parents=True)
    batch_id = "20260726T120000-abc123def0"
    sbatch_path, manifest_path, tasks = _install_exact_cell_transaction(
        state_dir,
        batch_id=batch_id,
        base_job_id="123",
        task_count=2,
        intent_state=intent_state,
        record_state=record_state,
        include_record=intent_state in {"submitted", "reconciled"},
        captured_at=1_000.0,
    )
    expected_name = f"asys-dispatch-{batch_id[-10:]}"
    comment = control.CELL_INTENT_PREFIX + batch_id
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                task_job_id,
                expected_name,
                "RUNNING",
                comment,
                _exact_cell_submission_command(batch_id),
                partition="ou_bcs_normal",
                qos="normal",
            ),
        ),
        1_000.0,
    )
    return (
        state_dir,
        {
            "batch_id": batch_id,
            "tasks": tasks,
            "manifest_path": manifest_path,
            "sbatch_path": sbatch_path,
        },
        snapshot,
    )


def test_trusted_cell_accepts_exact_sealed_artifact_join(tmp_path):
    state_dir, fixture, snapshot = _trusted_cell_binding_fixture(tmp_path)
    provenance = control.reconcile_trusted_scientific_job_provenance(
        state_dir,
        fleet_bindings={},
        fleet_contract_sha256="f" * 64,
        fleet_generation=1,
        scheduler_snapshot=snapshot,
        now=1_000.0,
    )
    assert provenance.payload["trusted_cell_job_ids"] == ["123_0"]
    assert provenance.payload["trusted_fleet_job_ids"] == []
    assert provenance.payload["dispatcher_provenance_id"]
    assert fixture["sbatch_path"].stat().st_mode & 0o222 == 0


def test_trusted_cell_rejects_cli_resource_override(tmp_path):
    state_dir, fixture, snapshot = _trusted_cell_binding_fixture(tmp_path)
    job = snapshot.jobs[0]
    overridden = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                job_id=job.job_id,
                job_name=job.job_name,
                state=job.state,
                comment=job.comment,
                command=(
                    f"/usr/bin/sbatch --mem=8G {fixture['sbatch_path']}"
                ),
                source=job.source,
                dependency=job.dependency,
                partition=job.partition,
                qos=job.qos,
            ),
        ),
        snapshot.captured_at,
    )
    with pytest.raises(
        control.SchedulerAmbiguity,
        match="exact stdin submission transport|resource overrides",
    ):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=overridden,
            now=1_000.0,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("writable-sbatch", "read-only"),
        ("tampered-sbatch", "hash drifted"),
        ("wrong-sbatch-hash", "hash drifted"),
        ("manifest-task-drift", "manifest differs from intent"),
        ("parent-symlink", "symlink"),
        ("out-of-range", "index exceeds"),
        ("bare-parent", "unexpanded array parent"),
        (
            "prepared-intent",
            "no accepted or ambiguous durable job record|"
            "incompatible intent state",
        ),
        ("inactive-record", "incompatible state"),
    ),
)
def test_trusted_cell_artifact_join_fails_closed(
    tmp_path, mutation, match
):
    task_job_id = (
        "123_999"
        if mutation == "out-of-range"
        else "123"
        if mutation == "bare-parent"
        else "123_0"
    )
    state_dir, fixture, snapshot = _trusted_cell_binding_fixture(
        tmp_path,
        task_job_id=task_job_id,
        intent_state=(
            "prepared" if mutation == "prepared-intent" else "reconciled"
        ),
        record_state=(
            "inactive" if mutation == "inactive-record" else "active"
        ),
    )
    ledger_path = state_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    intent = ledger["intents"][fixture["batch_id"]]
    record = ledger["jobs"].get("123")
    if mutation == "writable-sbatch":
        fixture["sbatch_path"].chmod(0o644)
    elif mutation == "tampered-sbatch":
        fixture["sbatch_path"].chmod(0o644)
        fixture["sbatch_path"].write_text(
            "#!/bin/bash\nexit 99\n", encoding="utf-8"
        )
        fixture["sbatch_path"].chmod(0o444)
    elif mutation == "wrong-sbatch-hash":
        assert isinstance(record, dict)
        intent["sbatch_sha256"] = "0" * 64
        intent["spooled_sbatch_sha256"] = "0" * 64
        record["sbatch_sha256"] = "0" * 64
        record["spooled_sbatch_sha256"] = "0" * 64
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif mutation == "manifest-task-drift":
        assert isinstance(record, dict)
        manifest = json.loads(
            fixture["manifest_path"].read_text(encoding="utf-8")
        )
        manifest["tasks"] = [{"coordinate": 99}]
        fixture["manifest_path"].chmod(0o644)
        dispatcher._atomic_write_json(fixture["manifest_path"], manifest)
        manifest_sha256 = dispatcher._seal_dispatch_artifact(
            fixture["manifest_path"]
        )
        sbatch_text = dispatcher._render_batch_sbatch(
            fixture["manifest_path"],
            n_tasks=len(fixture["tasks"]),
            partition="ou_bcs_normal",
            qos="normal",
            time_limit="12:00:00",
            memory="4G",
            log_dir=state_dir / "logs",
            batch_tag=fixture["batch_id"][-10:],
            batch_id=fixture["batch_id"],
            batch_manifest_sha256=manifest_sha256,
        )
        fixture["sbatch_path"].chmod(0o644)
        fixture["sbatch_path"].write_text(
            sbatch_text, encoding="utf-8"
        )
        sbatch_sha256 = dispatcher._seal_dispatch_artifact(
            fixture["sbatch_path"]
        )
        intent["batch_manifest_sha256"] = manifest_sha256
        record["batch_manifest_sha256"] = manifest_sha256
        intent["sbatch_sha256"] = sbatch_sha256
        record["sbatch_sha256"] = sbatch_sha256
        intent["spooled_sbatch_sha256"] = sbatch_sha256
        record["spooled_sbatch_sha256"] = sbatch_sha256
        receipt_path = Path(str(intent["spooled_receipt_path"]))
        receipt_path.unlink()
        (
            regenerated_receipt_path,
            regenerated_receipt_sha256,
        ) = dispatcher._spooled_script_receipt(
            batch_id=fixture["batch_id"],
            job_id="123",
            expected_name=f"asys-dispatch-{fixture['batch_id'][-10:]}",
            expected_comment=(
                control.CELL_INTENT_PREFIX + fixture["batch_id"]
            ),
            sbatch_path=fixture["sbatch_path"],
            sbatch_sha256=sbatch_sha256,
            spooled_script_reader=lambda _job_id: (
                fixture["sbatch_path"].read_bytes()
            ),
            now=1_000.0,
        )
        intent["spooled_receipt_path"] = regenerated_receipt_path
        intent["spooled_receipt_sha256"] = regenerated_receipt_sha256
        record["spooled_receipt_path"] = regenerated_receipt_path
        record["spooled_receipt_sha256"] = regenerated_receipt_sha256
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif mutation == "parent-symlink":
        real_batches = state_dir / "real-batches"
        fixture["sbatch_path"].parent.rename(real_batches)
        (state_dir / "batches").symlink_to(
            real_batches, target_is_directory=True
        )

    with pytest.raises(control.SchedulerAmbiguity, match=match):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=1_000.0,
        )


def test_trusted_cell_rejects_sbatch_replacement_during_stable_read(
    tmp_path, monkeypatch
):
    state_dir, fixture, snapshot = _trusted_cell_binding_fixture(tmp_path)
    replacement = fixture["sbatch_path"].with_name("replacement.sbatch")
    replacement.write_bytes(fixture["sbatch_path"].read_bytes())
    replacement.chmod(0o444)
    original_read = control.os.read
    replaced = {"value": False}

    def read_then_replace(descriptor, size):
        block = original_read(descriptor, size)
        if (
            block.startswith(b"#!/bin/bash")
            and not replaced["value"]
        ):
            replaced["value"] = True
            control.os.replace(replacement, fixture["sbatch_path"])
        return block

    monkeypatch.setattr(control.os, "read", read_then_replace)
    with pytest.raises(control.SchedulerAmbiguity, match="changed"):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=1_000.0,
        )


@pytest.mark.parametrize(
    "payload",
    (
        (
            '{"schema_version":1,"created_at":1,"updated_at":1000,'
            '"poll_number":0,"fairness":{},"validation_fairness":{},'
            '"runs":{},"jobs":{},"jobs":{},"intents":{},"cells":{}}\n'
        ),
        (
            '{"schema_version":1,"created_at":1,"updated_at":1e9999,'
            '"poll_number":0,"fairness":{},"validation_fairness":{},'
            '"runs":{},"jobs":{},"intents":{},"cells":{}}\n'
        ),
    ),
)
def test_shared_dispatcher_ledger_rejects_duplicate_or_nonfinite_json(
    tmp_path, payload
):
    state_dir = tmp_path / "dispatcher"
    state_dir.mkdir()
    (state_dir / "ledger.json").write_text(payload, encoding="utf-8")
    with pytest.raises(control.SchedulerAmbiguity):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=control.SchedulerSnapshot((), 1_000.0),
            now=1_000.0,
        )


def test_stable_readonly_preimage_rejects_post_fstat_chmod(
    tmp_path, monkeypatch
):
    target = tmp_path / "immutable.sbatch"
    target.write_text("#!/bin/bash\n", encoding="utf-8")
    target.chmod(0o444)
    original_fstat = control.os.fstat
    calls = {"value": 0}

    def chmod_after_second_fstat(descriptor):
        result = original_fstat(descriptor)
        calls["value"] += 1
        if calls["value"] == 2:
            target.chmod(0o644)
        return result

    monkeypatch.setattr(control.os, "fstat", chmod_after_second_fstat)
    with pytest.raises(control.SchedulerAmbiguity, match="changed|read-only"):
        control._stable_readonly_preimage(
            target, description="post-fstat fixture"
        )


def test_trusted_fleet_rejects_old_generation_binding(tmp_path):
    state_dir, binding, snapshot = _trusted_fleet_binding_fixture(tmp_path)
    with pytest.raises(
        control.SchedulerAmbiguity, match="non-current ledger generation"
    ):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings=binding,
            fleet_contract_sha256="f" * 64,
            fleet_generation=2,
            scheduler_snapshot=snapshot,
            now=1_000.0,
        )


def test_trusted_fleet_accepts_exact_current_generation_binding(tmp_path):
    state_dir, binding, snapshot = _trusted_fleet_binding_fixture(tmp_path)
    provenance = control.reconcile_trusted_scientific_job_provenance(
        state_dir,
        fleet_bindings=binding,
        fleet_contract_sha256="f" * 64,
        fleet_generation=1,
        scheduler_snapshot=snapshot,
        now=1_000.0,
    )
    assert provenance.payload["trusted_cell_job_ids"] == []
    assert provenance.payload["trusted_fleet_job_ids"] == ["123"]
    assert provenance.payload["fleet_generation"] == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "live-partition",
        "live-qos",
        "allocated-gpus",
        "gpu-type",
        "cli-gres-override",
    ),
)
def test_trusted_fleet_rejects_placement_or_gpu_drift(tmp_path, mutation):
    state_dir, binding, snapshot = _trusted_fleet_binding_fixture(tmp_path)
    binding = copy.deepcopy(binding)
    job = snapshot.jobs[0]
    partition = job.partition
    qos = job.qos
    command = job.command
    if mutation == "live-partition":
        partition = "other_partition"
    elif mutation == "live-qos":
        qos = "other_qos"
    elif mutation == "allocated-gpus":
        binding["123"]["allocated_gpus"] = 2
    elif mutation == "gpu-type":
        binding["123"]["gpu_type"] = "h100"
    elif mutation == "cli-gres-override":
        command = (
            f"/usr/bin/sbatch --gres=gpu:h100:1 "
            f"{binding['123']['sbatch_path']}"
        )
    snapshot = control.SchedulerSnapshot(
        (
            control.SchedulerJob(
                job_id=job.job_id,
                job_name=job.job_name,
                state=job.state,
                comment=job.comment,
                command=command,
                source=job.source,
                dependency=job.dependency,
                partition=partition,
                qos=qos,
            ),
        ),
        snapshot.captured_at,
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="exact ledger/scheduler/script"
    ):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings=binding,
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=1_000.0,
        )


def test_scheduler_truth_id_binds_unrelated_live_placement(tmp_path):
    state_dir = tmp_path / "dispatcher"
    _write_dispatcher_provenance_ledger(
        state_dir, updated_at=1_000.0
    )
    snapshots = [
        control.SchedulerSnapshot(
            (
                control.SchedulerJob(
                    "999",
                    "unrelated",
                    "RUNNING",
                    "",
                    "/sealed/unrelated.sbatch",
                    partition=partition,
                    qos="normal",
                ),
            ),
            1_000.0,
        )
        for partition in ("partition_a", "partition_b")
    ]
    values = [
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings={},
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=1_000.0,
            allow_exact_cell_quiescence=True,
        )
        for snapshot in snapshots
    ]
    assert (
        values[0].payload["scheduler_truth_id"]
        != values[1].payload["scheduler_truth_id"]
    )
    assert values[0].provenance_id != values[1].provenance_id


def test_trusted_fleet_rejects_symlinked_ledger_parent(tmp_path):
    state_dir, binding, snapshot = _trusted_fleet_binding_fixture(
        tmp_path, symlink_ledgers=True
    )
    with pytest.raises(
        control.SchedulerAmbiguity, match="outside|symlink"
    ):
        control.reconcile_trusted_scientific_job_provenance(
            state_dir,
            fleet_bindings=binding,
            fleet_contract_sha256="f" * 64,
            fleet_generation=1,
            scheduler_snapshot=snapshot,
            now=1_000.0,
        )


def test_finalization_prefence_crash_is_fenced_and_reconciler_resumes(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    original_save = control._save_control
    crashed = {"value": False}

    def save_fence_then_crash(target_state_dir, value, **kwargs):
        original_save(target_state_dir, value, **kwargs)
        if (
            not crashed["value"]
            and value["finalization"]["state"] == "idle"
            and value["transition_history"]
            and value["transition_history"][-1]["event"]
            == "autonomous_finalization_admission_fenced"
        ):
            crashed["value"] = True
            raise RuntimeError("death after finalization admission fence")

    with monkeypatch.context() as boundary:
        boundary.setattr(control, "_save_control", save_fence_then_crash)
        with pytest.raises(RuntimeError, match="admission fence"):
            control.request_autonomous_finalization(
                state_dir,
                semantic_report_path=evidence_path,
                scheduler=control.SchedulerSnapshot((), 40.0),
                submit_runner=lambda argv: pytest.fail(
                    f"sbatch crossed pre-intent crash: {argv}"
                ),
                now=40.0,
            )

    fenced = control.load_control(state_dir)
    assert fenced["desired_state"] == "paused"
    assert fenced["drain_requested"] is True
    assert fenced["finalization"]["state"] == "idle"
    assert not control._finalization_request_intent_path(fenced).exists()
    status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 40.5),
        now=40.5,
    )
    assert status["effective_admission_ceiling"] == 0
    assert status["healthy"] is False
    assert status["finalization"]["chain_healthy"] is False
    pending = status["finalization"]["pending_request"]
    assert pending["validated"] is True
    assert pending["intent_id"]

    submitted = iter(("901", "902"))
    resumed = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 4_000.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=4_000.0,
    )
    assert resumed["state"] == "draining"
    assert resumed["active_job"]["job_id"] == "901"
    assert resumed["active_job"]["created_timestamp"] == 4_000.0
    assert resumed["successor_job"]["job_id"] == "902"
    assert resumed["successor_job"]["created_timestamp"] == 4_000.0
    assert control._finalization_request_intent_path(
        control.load_control(state_dir)
    ).is_file()


def test_nonidle_finalization_fences_admission_without_alert_or_pause(
    tmp_path, monkeypatch
):
    state_dir, current = initialize(tmp_path)
    current["desired_state"] = "running"
    current["drain_requested"] = False
    current["finalization"]["state"] = "blocked"
    current["alerts"] = []
    current["admission_safety_hold"]["active"] = False
    current["admission_safety_hold"]["reasons"] = []
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: copy.deepcopy(current),
    )
    with pytest.raises(
        control.ControlError,
        match="disabled by autonomous finalization",
    ):
        control.admission_contract_from_state(state_dir)
    status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 50.0),
        now=50.0,
    )
    assert status["effective_admission_ceiling"] == 0
    assert status["healthy"] is False


def test_blocked_state_save_fences_before_failure_alert(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    scheduler = _autonomous_scheduler(requested, captured_at=41.0)
    monkeypatch.setattr(
        control,
        "record_alert",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("death before finalization alert")
        ),
    )
    with pytest.raises(RuntimeError, match="before finalization alert"):
        control.run_autonomous_finalizer_worker(
            state_dir,
            intent_id=requested["intent_id"],
            attempt=1,
            job_id="901",
            scheduler_reader=lambda: scheduler,
            finalize_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                control.ControlError(
                    "semantic acceptance contains duplicate QIDs"
                )
            ),
            now=41.0,
        )
    blocked = control.load_control(state_dir)
    assert blocked["finalization"]["state"] == "blocked"
    assert not [
        alert
        for alert in blocked["alerts"]
        if alert.get("dedupe_key") == "finalization:blocked"
    ]
    status = control.live_status(
        state_dir, snapshot=scheduler, now=41.0
    )
    assert status["effective_admission_ceiling"] == 0
    assert status["healthy"] is False


def test_generation_two_scheduler_read_failure_preserves_recoverable_pair(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    monkeypatch.setattr(
        control, "record_alert", lambda *_args, **_kwargs: {}
    )
    returncode = control.run_autonomous_finalizer_worker(
        state_dir,
        intent_id=requested["intent_id"],
        attempt=2,
        job_id="902",
        scheduler_reader=lambda: (_ for _ in ()).throw(
            control.SchedulerAmbiguity("sacct temporarily unavailable")
        ),
        now=41.0,
    )
    assert returncode == 75
    preserved = control.load_control(state_dir)["finalization"]
    assert preserved["active_job"]["job_id"] == "901"
    assert preserved["successor_job"]["job_id"] == "902"
    assert preserved["last_error"]["transient"] is True

    generation_two = _autonomous_scheduler(
        preserved,
        captured_at=42.0,
        active_state="COMPLETED",
        successor_state="RUNNING",
    )
    recovered = control.reconcile_autonomous_finalization(
        state_dir,
        snapshot=generation_two,
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "903\n", ""
        ),
        now=42.0,
    )
    assert recovered["active_job"]["job_id"] == "902"
    assert recovered["successor_job"]["job_id"] == "903"
    assert recovered["successor_job"]["dependency_job_id"] == "902"


def test_live_status_reconciles_finalizer_identity_mismatch_and_absence(
    tmp_path, monkeypatch
):
    state_dir, _ = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    current = control.load_control(state_dir)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    exact = _autonomous_scheduler(requested, captured_at=41.0)
    exact_status = control.live_status(
        state_dir, snapshot=exact, now=41.0
    )["finalization"]
    assert exact_status["chain_healthy"] is True
    assert exact_status["active"]["live"] is True
    assert exact_status["successor"]["live"] is True

    exact_jobs = list(exact.jobs)
    stale_active = exact_jobs[0]
    exact_jobs[0] = control.SchedulerJob(
        "999",
        stale_active.job_name,
        stale_active.state,
        stale_active.comment,
        stale_active.command,
        stale_active.source,
        stale_active.dependency,
    )
    stale_status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot(tuple(exact_jobs), 42.0),
        now=42.0,
    )["finalization"]
    assert stale_status["active"]["identity_mismatch"] is True
    assert stale_status["unexpected_active_job_ids"] == ["999"]
    assert stale_status["chain_healthy"] is False

    token_jobs = list(exact.jobs)
    token_active = token_jobs[0]
    token_jobs[0] = control.SchedulerJob(
        token_active.job_id,
        token_active.job_name,
        token_active.state,
        f"{control.FINALIZER_JOB_TOKEN_PREFIX};wrong=identity",
        token_active.command,
        token_active.source,
        token_active.dependency,
    )
    token_status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot(tuple(token_jobs), 43.0),
        now=43.0,
    )["finalization"]
    assert token_status["active"]["identity_mismatch"] is True
    assert token_status["chain_healthy"] is False

    empty_status = control.live_status(
        state_dir,
        snapshot=control.SchedulerSnapshot((), 1_000.0),
        now=1_000.0,
    )["finalization"]
    assert empty_status["active"]["missing_after_grace"] is True
    assert empty_status["successor"]["missing_after_grace"] is True
    assert empty_status["chain_healthy"] is False


def test_live_status_marks_unique_pre_receipt_finalizer_adoptable(
    tmp_path, monkeypatch
):
    state_dir, current = initialize(tmp_path)
    _install_finalizer_test_catalog(state_dir, monkeypatch)
    evidence_path, _ = _write_autonomous_finalization_evidence(
        state_dir, current
    )
    submitted = iter(("901", "902"))
    requested = control.request_autonomous_finalization(
        state_dir,
        semantic_report_path=evidence_path,
        scheduler=control.SchedulerSnapshot((), 40.0),
        submit_runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, next(submitted) + "\n", ""
        ),
        now=40.0,
    )
    in_memory = control.load_control(state_dir)
    record = in_memory["finalization"]["active_job"]
    record["job_id"] = None
    record["state"] = "submitting"
    record["submitted_at"] = None
    record["submitted_timestamp"] = None
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: copy.deepcopy(in_memory),
    )
    scheduler = _autonomous_scheduler(requested, captured_at=41.0)
    row = control.live_status(
        state_dir, snapshot=scheduler, now=41.0
    )["finalization"]["active"]
    assert row["recorded_job_id"] is None
    assert row["scheduler_job_id"] == "901"
    assert row["adoptable"] is True
    assert row["identity_mismatch"] is False
    assert row["missing"] is False
    assert row["provenance"]["job_id"] == "901"
