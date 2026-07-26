"""Focused contracts for the superseding schema-5 v1.2-r2 recovery chain."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import sys

import pytest

from scripts import render_schema5_recovery_chain_v12 as chain
from scripts import run_schema5_smokes as smoke_runner
from scripts import schema5_recovery_sentinel as recovery_sentinel
from scripts import schema5_conda_runtime_identity
from scripts import seal_recovery_evidence
from scripts import verify_schema5_recovery_evidence


COMMIT = "2" * 40
TAG_OBJECT = "3" * 40


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _identified_json(
    path: Path, payload: dict[str, object], *, identity_field: str
) -> dict[str, object]:
    value = dict(payload)
    value[identity_field] = chain._sha256_bytes(chain._canonical_json(value))
    _json(path, value)
    return value


def test_conda_runtime_identity_rejects_links_outside_the_base_prefix(
    tmp_path: Path,
) -> None:
    base = tmp_path / "conda-base"
    (base / "bin").mkdir(parents=True)
    interpreter = base / "bin" / "python"
    interpreter.write_bytes(b"fixture interpreter\n")
    interpreter.chmod(0o755)
    executable = base / "bin" / "conda"
    executable.write_text(f"#!{interpreter}\n", encoding="utf-8")
    executable.chmod(0o755)
    outside = tmp_path / "outside-runtime"
    outside.write_bytes(b"untrusted\n")
    (base / "escaping-link").symlink_to(outside)

    with pytest.raises(
        schema5_conda_runtime_identity.CondaRuntimeIdentityError,
        match="escapes base prefix",
    ):
        schema5_conda_runtime_identity.conda_runtime_identity(executable)


def _paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> chain.RecoveryPaths:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    release_binding = {
        "schema_version": 1,
        "passed": True,
        "release_id": chain.RELEASE_ID,
        "release_tag": chain.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": chain.CHAIN_NAMESPACE,
    }
    durable_bundle_root = (
        recovery / chain.durable_git.BUNDLE_DIRECTORY
    )
    durable_bundle_root.mkdir()
    durable_bundle = durable_bundle_root / chain.durable_git.BUNDLE_NAME
    durable_bundle.write_bytes(b"fixture durable Git bundle\n")
    durable_bundle.chmod(0o444)
    durable_bundle_sha256 = hashlib.sha256(
        durable_bundle.read_bytes()
    ).hexdigest()
    durable_checksum = (
        durable_bundle_root / chain.durable_git.CHECKSUM_NAME
    )
    durable_checksum.write_text(
        f"{durable_bundle_sha256}  {chain.durable_git.BUNDLE_NAME}\n",
        encoding="ascii",
    )
    durable_checksum.chmod(0o444)
    durable_marker_path = (
        recovery / chain.DURABLE_GIT_RELEASE_MARKER_NAME
    )
    durable_marker = {
        **release_binding,
        "protocol": chain.DURABLE_GIT_RELEASE_PROTOCOL,
        "clean_checkout": True,
        "annotated_tag": True,
        "remote_query_read_only": True,
        "remote": "durable",
        "remote_commit_ref": "refs/heads/release",
        "remote_commit": COMMIT,
        "remote_tag_object": TAG_OBJECT,
        "remote_peeled_commit": COMMIT,
        "remote_url_sha256": "0" * 64,
        "bundle_path": str(durable_bundle),
        "bundle_sha256": durable_bundle_sha256,
        "bundle_size": durable_bundle.stat().st_size,
        "checksum_path": str(durable_checksum),
        "checksum_sha256": hashlib.sha256(
            durable_checksum.read_bytes()
        ).hexdigest(),
        "published_at": "2026-07-25T00:00:00+00:00",
    }
    durable_marker["marker_id"] = chain.durable_git._self_hash(
        durable_marker, "marker_id"
    )
    durable_marker_path.write_bytes(
        chain.durable_git._canonical(durable_marker)
    )
    durable_marker_path.chmod(0o444)
    durable_binding = chain.durable_git.marker_binding(
        durable_marker_path
    )
    _identified_json(
        recovery / chain.PROTECTED_CAPACITY_MARKER_NAME,
        {
            **release_binding,
            "schema_version": 2,
            "protocol": chain.PROTECTED_CAPACITY_PROTOCOL,
            "active_gpus": 24,
            "warm_headroom_gpus": 4,
            "cell_ceiling": 384,
            "reserve_jobs": 64,
            "submit_headroom": 448,
            "cpu": 384,
            "memory_mib": 1_572_864,
            "preempt_type": "preempt/qos",
            "capacity_source": chain.PROTECTED_CAPACITY_SOURCE,
            "scheduler_cluster": "test_cluster",
            "scheduler_account": "test_account",
            "scheduler_user": "test_user",
            "scheduler_max_submit_jobs": 500,
            "partition_cpus": 384,
            "partition_memory_mib": 1_572_864,
            "partition_gpus": 0,
            "fleet_contract_sha256": "7" * 64,
            "active_fleet_topology_sha256": "8" * 64,
            "scientific_server_preempt_mode": "OFF",
            "scientific_client_preempt_mode": "OFF",
            "scientific_server_placements": [
                {
                    "partition": "gpu_protected",
                    "qos": "gpu_science",
                    "partition_preempt_mode": "OFF",
                    "qos_preempt_mode": "OFF",
                    "active_serving_gpus": 24,
                    "warm_headroom_gpus": 4,
                }
            ],
            "scientific_client_placements": [
                {
                    "partition": "cpu_protected",
                    "qos": "client_science",
                    "partition_preempt_mode": "OFF",
                    "qos_preempt_mode": "OFF",
                    "slots": 384,
                    "cpus": 384,
                    "memory_mib": 1_572_864,
                    "reserve_jobs": 64,
                    "submit_headroom": 448,
                }
            ],
            "scheduler_evidence_id": "3" * 64,
            "scheduler_evidence_sha256": "4" * 64,
            "canary_id": "5" * 64,
            "canary_evidence_sha256": "6" * 64,
            "squeue_complete": True,
            "sacct_complete": True,
        },
        identity_field="marker_id",
    )
    drill = _identified_json(
        recovery / chain.EXTERNAL_WATCHDOG_DRILL_MARKER_NAME,
        {
            **release_binding,
            "protocol": chain.EXTERNAL_WATCHDOG_DRILL_PROTOCOL,
            "deployment_id": "9" * 64,
            "watchdog_code_sha256": "a" * 64,
            "immutable_release_sha256": "b" * 64,
            "control_sha256": "c" * 64,
            "namespace_cancellation_recovery_seconds": 600,
            "duplicate_jobs": 0,
            "duplicate_admission_intents": 0,
            "fairness_mutations": 0,
        },
        identity_field="drill_id",
    )
    _identified_json(
        recovery / chain.WATCHDOG_READY_MARKER_NAME,
        {
            **release_binding,
            "protocol": chain.WATCHDOG_READY_PROTOCOL,
            "deployment_id": drill["deployment_id"],
            "watchdog_code_sha256": drill["watchdog_code_sha256"],
            "immutable_release_sha256": drill[
                "immutable_release_sha256"
            ],
            "control_sha256": drill["control_sha256"],
            "forced_command_only": True,
            "timer_seconds": 300,
            "scheduler_observations": [1_000.0, 1_060.0],
            "liveness_email_ack": True,
            "external_watchdog_drill": {
                "marker": str(
                    recovery
                    / chain.EXTERNAL_WATCHDOG_DRILL_MARKER_NAME
                ),
                "marker_sha256": chain._sha256(
                    recovery
                    / chain.EXTERNAL_WATCHDOG_DRILL_MARKER_NAME
                ),
                "drill_id": drill["drill_id"],
            },
            "namespace_cancellation_recovery_seconds": drill[
                "namespace_cancellation_recovery_seconds"
            ],
            "duplicate_jobs": 0,
            "duplicate_admission_intents": 0,
            "fairness_mutations": 0,
        },
        identity_field="marker_id",
    )
    for run_id in chain.LEGACY_RUN_IDS:
        (results / run_id).mkdir()
    hf_home = tmp_path / "hf"
    source_harness = tmp_path / "source-harness"
    source_serving = tmp_path / "source-serving"
    for prefix in (source_harness, source_serving):
        (prefix / "conda-meta").mkdir(parents=True)
    hf_home.mkdir()
    (source_harness / "bin").mkdir()
    stdlib = source_harness / "lib" / "python3.11"
    (stdlib / "lib-dynload").mkdir(parents=True)
    (stdlib / "os.py").write_text("# bootstrap stdlib fixture\n", encoding="utf-8")
    dev_python = source_harness / "bin" / "python"
    dev_python.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' {chain._BOOTSTRAP_PROBE_TOKEN}\n",
        encoding="utf-8",
    )
    dev_python.chmod(0o755)
    conda_base = tmp_path / "miniforge"
    (conda_base / "bin").mkdir(parents=True)
    (conda_base / "lib" / "python3.12" / "site-packages" / "conda").mkdir(
        parents=True
    )
    conda_python = conda_base / "bin" / "python3.12"
    conda_python.write_bytes(b"fixture conda python runtime\n")
    conda_python.chmod(0o755)
    (conda_base / "lib" / "python3.12" / "site-packages" / "conda" / "__init__.py").write_text(
        "__version__ = 'fixture'\n",
        encoding="utf-8",
    )
    conda = conda_base / "bin" / "conda"
    conda.write_text(f"#!{conda_python}\n# fixture entrypoint\n", encoding="utf-8")
    conda.chmod(0o755)
    pilot_root = recovery / "materialization_pilots" / chain.CHAIN_NAMESPACE
    canary_root = recovery / "slurm_canaries" / chain.CHAIN_NAMESPACE
    _json(pilot_root / chain.MATERIALIZATION_PILOT_MARKER, {"complete": True})
    _json(canary_root / chain.SLURM_CANARY_MARKER, {"complete": True})
    pilot_harness = pilot_root / "materialization" / "harness-environment"
    (pilot_harness / "bin").mkdir(parents=True)
    (pilot_harness / "lib").mkdir()
    pilot_python = pilot_harness / "bin" / "python"
    pilot_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pilot_python.chmod(0o555)
    pilot_manifest = (
        pilot_root
        / "materialization"
        / "release"
        / "harness_environment.schema5-v1.json"
    )
    _json(
        pilot_manifest,
        {
            "prefix": str(pilot_harness),
            "sealed_read_only": True,
            "directory_inventory": {"inventory_sha256": "8" * 64},
        },
    )
    (pilot_harness / "bin").chmod(0o555)
    (pilot_harness / "lib").chmod(0o555)
    pilot_harness.chmod(0o555)
    sealed_conda_toolchain = (
        schema5_conda_runtime_identity.conda_runtime_identity(conda)
    )

    marker = recovery / "pre_repair" / "SNAPSHOT_COMPLETE.json"
    _json(
        marker,
        {
            "file_count": chain.SEALED_SNAPSHOT_CONTRACT["file_count"],
            "total_bytes": chain.SEALED_SNAPSHOT_CONTRACT["total_bytes"],
            "snapshot_inventory_sha256": chain.SEALED_SNAPSHOT_CONTRACT[
                "snapshot_inventory_sha256"
            ],
        },
    )
    attestation = recovery / "pre_repair.attestation.json"
    _json(attestation, {"passed": True})
    monkeypatch.setitem(
        chain.SEALED_SNAPSHOT_CONTRACT,
        "completion_sha256",
        hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    monkeypatch.setitem(
        chain.SEALED_SNAPSHOT_CONTRACT,
        "attestation_sha256",
        hashlib.sha256(attestation.read_bytes()).hexdigest(),
    )
    for name, relative in chain.R1_EVIDENCE_RELATIVE_PATHS.items():
        if name in {
            "quarantine_idempotency_receipt",
            "zero_result_mutation_receipt",
        }:
            continue
        path = recovery / relative
        payload = (
            {
                "classification": "requires_superseding_release",
                "retry_same_generation": False,
                "superseded_by": chain.RELEASE_ID,
            }
            if name == "failure_envelope"
            else {"passed": True, "name": name}
        )
        _json(path, payload)
    idempotency_identity = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r2-r1-quarantine-idempotency-receipt-v1"
        ),
        "passed": True,
        "r1_renderer": {
            "git_commit": "a5cd9305e8fd741ba59bc14d9d296e3ecb5c5f96"
        },
        "first_repeat": {"status": "already_quarantined", "passed": True},
        "second_repeat": {"status": "already_quarantined", "passed": True},
        "sealed_quarantine": {
            "seal": {
                "sha256": hashlib.sha256(
                    (
                        recovery
                        / chain.R1_EVIDENCE_RELATIVE_PATHS["quarantine_seal"]
                    ).read_bytes()
                ).hexdigest()
            },
            "inventory": {
                "sha256": hashlib.sha256(
                    (
                        recovery
                        / chain.R1_EVIDENCE_RELATIVE_PATHS[
                            "quarantine_inventory"
                        ]
                    ).read_bytes()
                ).hexdigest()
            },
            "completion": {
                "sha256": hashlib.sha256(
                    (
                        recovery
                        / chain.R1_EVIDENCE_RELATIVE_PATHS[
                            "quarantine_completion"
                        ]
                    ).read_bytes()
                ).hexdigest()
            },
        },
        "sealed_tree_mutated": False,
    }
    idempotency_identity["receipt_id"] = chain._sha256_bytes(
        chain._canonical_json(idempotency_identity)
    )
    _json(
        recovery
        / chain.R1_EVIDENCE_RELATIVE_PATHS[
            "quarantine_idempotency_receipt"
        ],
        idempotency_identity,
    )
    zero_identity = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1"
        ),
        "passed": True,
        "r1_renderer": {
            "git_commit": "a5cd9305e8fd741ba59bc14d9d296e3ecb5c5f96"
        },
        "r1_manifest": {
            "sha256": hashlib.sha256(
                (
                    recovery
                    / chain.R1_EVIDENCE_RELATIVE_PATHS["chain_manifest"]
                ).read_bytes()
            ).hexdigest()
        },
        "r1_submission_receipt": {
            "sha256": hashlib.sha256(
                (
                    recovery
                    / chain.R1_EVIDENCE_RELATIVE_PATHS["submission_receipt"]
                ).read_bytes()
            ).hexdigest()
        },
        "r1_failure_envelope": {
            "sha256": hashlib.sha256(
                (
                    recovery
                    / chain.R1_EVIDENCE_RELATIVE_PATHS["failure_envelope"]
                ).read_bytes()
            ).hexdigest()
        },
        "failed_stage": "release_materialize",
        "first_result_mutating_stage": "legacy_consolidate",
        "started_result_mutating_stages": [],
        "legacy_result_mutation_count": 0,
        "schema5_result_mutation_count": 0,
    }
    zero_identity["receipt_id"] = chain._sha256_bytes(
        chain._canonical_json(zero_identity)
    )
    _json(
        recovery
        / chain.R1_EVIDENCE_RELATIVE_PATHS["zero_result_mutation_receipt"],
        zero_identity,
    )

    paths = chain.recovery_paths(
        repository=repository,
        results_root=results,
        recovery_root=recovery,
        hf_home=hf_home,
        dev_python=dev_python,
        source_harness=source_harness,
        source_serving=source_serving,
        conda_executable=conda,
        materialization_pilot_root=pilot_root,
        slurm_canary_root=canary_root,
    )

    def prerequisite_reports(
        received_paths,
        *,
        checkout,
        expected_commit,
        expected_tag_object,
        verifier_python=None,
        verifier_library=None,
    ):
        del checkout
        del verifier_python, verifier_library
        assert expected_commit == COMMIT
        assert expected_tag_object == TAG_OBJECT
        records = {
            row["git_path"]: row
            for row in chain._prerequisite_code_records(
                received_paths.repository, commit=COMMIT
            )
        }
        return {
            "materialization_pilot": {
                "status": "verified",
                "schema_version": 4,
                "release_id": chain.RELEASE_ID,
                "expected_tag": chain.RELEASE_TAG,
                "expected_commit": COMMIT,
                "durable_git_release": durable_binding,
                "pilot_id": "4" * 64,
                "scheduler_acceptance": {
                    "acceptance_id": "a" * 64,
                    "marker": str(
                        received_paths.materialization_pilot_root
                        / "PILOT_SCHEDULER_ACCEPTED.json"
                    ),
                    "marker_sha256": "b" * 64,
                    "job_id": "24680",
                    "receipt": str(
                        received_paths.recovery_root
                        / "jobs"
                        / "schema5-v1.2-r2-materialization-pilot.sbatch.receipt.json"
                    ),
                    "receipt_sha256": "c" * 64,
                    "receipt_id": "d" * 64,
                    "effective_requeue": 0,
                    "spooled_script_sha256": "e" * 64,
                    "terminal_state": "COMPLETED",
                    "exit_code": "0:0",
                    "reason": "None",
                },
                "live_source_inventory_sha256": {
                    "harness": "6" * 64,
                    "serving": "7" * 64,
                },
                "ownership_policy_sha256": records[
                    chain.OWNERSHIP_POLICY_GIT_PATH
                ]["sha256"],
                "integrity_normalization_policy_sha256": records[
                    chain.INTEGRITY_NORMALIZATION_POLICY_GIT_PATH
                ]["sha256"],
                "conda_executable": {
                    "path": str(received_paths.conda_executable),
                    "sha256": hashlib.sha256(
                        received_paths.conda_executable.read_bytes()
                    ).hexdigest(),
                    "size": received_paths.conda_executable.stat().st_size,
                },
                "conda_runtime_toolchain": (
                    sealed_conda_toolchain
                ),
                "verifier_runtime": {
                    "harness_prefix": str(pilot_harness),
                    "environment_manifest_path": str(pilot_manifest),
                    "environment_manifest_sha256": hashlib.sha256(
                        pilot_manifest.read_bytes()
                    ).hexdigest(),
                    "directory_inventory_sha256": "8" * 64,
                    "python_path": str(pilot_python),
                    "python_sha256": hashlib.sha256(
                        pilot_python.read_bytes()
                    ).hexdigest(),
                    "python_size": pilot_python.stat().st_size,
                    "library_path": str(pilot_harness / "lib"),
                },
            },
            "slurm_canary": {
                "schema_version": 4,
                "kind": "schema5_slurm_fleet_composite_canary_complete",
                "canary_root": str(received_paths.slurm_canary_root),
                "canary_id": "5" * 64,
                "transaction_root": str(
                    received_paths.slurm_canary_root / "transaction"
                ),
                "transaction_canary_id": "6" * 64,
                "transaction_marker_sha256": "7" * 64,
                "dependency_root": str(
                    received_paths.slurm_canary_root / "dependency-cascade"
                ),
                "dependency_canary_id": "a" * 64,
                "dependency_marker_sha256": "b" * 64,
                "dependency_parameters": [
                    "disable_remote_singleton_jobs",
                    "kill_invalid_depend",
                ],
                "dependency_kill_invalid_depend": True,
                "dependency_root_initial_hold": True,
                "dependency_child_never_started": True,
                "dependency_alert_latency_seconds": 10.0,
                "dependency_alert_latency_bound_seconds": 180.0,
                "turnover_root": str(
                    received_paths.slurm_canary_root / "turnover"
                ),
                "turnover_canary_id": "8" * 64,
                "turnover_marker_sha256": "9" * 64,
                "turnover_cycles_completed": 2,
                "turnover_allocations_submitted": 3,
                "turnover_maximum_physical_allocations_observed": 2,
                "turnover_maximum_extra_gpus_observed": 0,
                "turnover_all_effective_requeue": 0,
                "turnover_continuous_routed_endpoint_evidence": True,
                "production_overlap_gpu_ceiling": 4,
                "code_identity": {
                    "release_tag": chain.RELEASE_TAG,
                    "release_git_commit": COMMIT,
                    "release_tag_object": TAG_OBJECT,
                    "canary_script": records[chain.SLURM_CANARY_GIT_PATH],
                    "fleet_transactions": records[
                        chain.FLEET_TRANSACTIONS_GIT_PATH
                    ],
                    "durable_git_publisher": records[
                        chain.DURABLE_GIT_RELEASE_GIT_PATH
                    ],
                    "durable_git_release": durable_binding,
                },
            },
        }

    monkeypatch.setattr(
        chain,
        "_invoke_tagged_prerequisite_verifiers",
        prerequisite_reports,
    )

    def native_r1_report(
        received_paths,
        **_kwargs,
    ):
        r1 = chain._r1_evidence_contract(received_paths)
        return {
            "passed": True,
            "schema_version": 1,
            "protocol": chain.R1_EVIDENCE_DISPATCH_PROTOCOL,
            "chain_protocol": chain.R1_CHAIN_PROTOCOL,
            "renderer": "render_schema5_recovery_chain",
            "manifest_path": r1["chain_manifest"]["path"],
            "manifest_sha256": r1["chain_manifest"]["sha256"],
            "chain_report": {"passed": True, "chain_id": "a" * 64},
            "submission_receipt_path": r1["submission_receipt"]["path"],
            "submission_receipt_sha256": r1["submission_receipt"]["sha256"],
        }

    monkeypatch.setattr(
        chain,
        "_invoke_protocol_aware_r1_verifier",
        native_r1_report,
    )
    monkeypatch.setattr(
        chain,
        "_invoke_conda_runtime_identity",
        lambda received_paths, **_kwargs: (
            schema5_conda_runtime_identity.conda_runtime_identity(
                received_paths.conda_executable
            )
        ),
    )
    monkeypatch.setattr(
        chain,
        "_verify_sealed_r1_protocol_contract",
        lambda _paths, contract, **_kwargs: json.loads(
            json.dumps(contract, sort_keys=True)
        ),
    )
    return paths


def _stub_tagged_release(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, bytes]:
    payloads = {
        git_path: (
            "#!/usr/bin/env python3\n"
            f"# exact tagged fixture: {git_path}\n"
        ).encode("utf-8")
        for git_path in {
            *chain.BUNDLED_TOOL_GIT_PATHS,
            *chain.PREREQUISITE_CODE_GIT_PATHS,
        }
    }
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: {
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
    )
    monkeypatch.setattr(
        chain,
        "_tagged_file_bytes",
        lambda _repository, commit, relative: (
            payloads[relative]
            if commit == COMMIT and relative in payloads
            else pytest.fail(
                f"unexpected tagged payload request: {commit}:{relative}"
            )
        ),
    )
    return payloads


def _render_applied(
    paths: chain.RecoveryPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, bytes], dict]:
    payloads = _stub_tagged_release(monkeypatch)
    result = chain.render_chain(
        paths,
        partition="mit_normal",
        slurm_user="tester",
        apply=True,
    )
    assert result["status"] == "complete"
    return payloads, json.loads(paths.chain_manifest.read_text(encoding="utf-8"))


def _publish_throughput_qualification(
    paths: chain.RecoveryPaths, manifest: dict
) -> dict[str, object]:
    prerequisite = manifest["prerequisite_evidence"]

    def compact(name: str, identity_field: str) -> dict[str, object]:
        record = prerequisite[name]
        return {
            "marker": record["marker"],
            "marker_sha256": record["marker_sha256"],
            identity_field: record[identity_field],
        }

    trusted_generation_path = (
        paths.readiness / "TRUSTED_GENERATION.json"
    ).resolve()
    _json(
        trusted_generation_path,
        {"schema_version": 1, "catalog_id": "7" * 64},
    )
    readiness = {
        "catalog_id": "7" * 64,
        "marker_path": str(trusted_generation_path),
        "marker_sha256": chain._sha256(trusted_generation_path),
        "inventory_sha256": "9" * 64,
        "catalog_payload_sha256": "a" * 64,
        "allowed_generation_tuple_count": 24,
        "release_fleet_contract_sha256": "b" * 64,
        "fleet_contract_sha256": "c" * 64,
        "capacity_generation": 1,
        "rollout_generation": 1,
    }
    attempt_id = chain._qualification_attempt_id(readiness)
    attempt_root = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_ATTEMPT_DIRECTORY
        / attempt_id
    )
    run_root = (
        paths.results_root
        / chain.THROUGHPUT_QUALIFICATION_RUN_DIRECTORY
        / attempt_id
        / chain.THROUGHPUT_QUALIFICATION_ROOT_NAME
    )
    run_root.mkdir(parents=True)
    pointer_path = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY
        / f"000001-{attempt_id}.json"
    )
    created_timestamp = 1_000.0
    pointer = _identified_json(
        pointer_path,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_ATTEMPT_POINTER_PROTOCOL
            ),
            "chain_id": manifest["chain_id"],
            "attempt_ordinal": 1,
            "attempt_id": attempt_id,
            "attempt_root": str(attempt_root),
            "run_root": str(run_root),
            "dispatcher_state": str(attempt_root / "dispatcher"),
            "readiness_generation": readiness,
            "predecessor": None,
            "additive_retry": None,
            "created_at": chain.datetime.fromtimestamp(
                created_timestamp, tz=chain.timezone.utc
            ).isoformat(),
            "created_timestamp": created_timestamp,
        },
        identity_field="pointer_id",
    )
    pointer_reference = chain._qualification_pointer_reference(
        pointer_path,
        pointer,
    )
    _identified_json(
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_NAME,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL
            ),
            "attempt_id": attempt_id,
            "pointer": str(pointer_path),
            "pointer_sha256": pointer_reference["sha256"],
            "pointer_id": pointer["pointer_id"],
        },
        identity_field="current_id",
    )
    marker = _identified_json(
        attempt_root / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME,
        {
            "schema_version": 1,
            "protocol": chain.THROUGHPUT_QUALIFICATION_PROTOCOL,
            "passed": True,
            "release_id": chain.RELEASE_ID,
            "release_tag": chain.RELEASE_TAG,
            "release_git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "chain_namespace": chain.CHAIN_NAMESPACE,
            "chain_id": manifest["chain_id"],
            "manifest": str(paths.chain_manifest),
            "manifest_sha256": chain._sha256(paths.chain_manifest),
            "protected_capacity": compact(
                "protected_capacity", "marker_id"
            ),
            "attempt": chain._qualification_attempt_binding(
                pointer_path,
                pointer,
            ),
            "cells": chain.THROUGHPUT_QUALIFICATION_CELLS,
            "qids": chain.THROUGHPUT_QUALIFICATION_QIDS,
            "ceilings": list(chain.THROUGHPUT_QUALIFICATION_CEILINGS),
            "steady_384_seconds": (
                chain.THROUGHPUT_QUALIFICATION_STEADY_SECONDS
            ),
            "throughput_qids_per_day": (
                chain.THROUGHPUT_QUALIFICATION_MIN_QIDS_PER_DAY
            ),
            "every_stratum_progress": True,
            "integrity_incidents": 0,
            "transport_censor_incidents": 0,
        },
        identity_field="qualification_id",
    )
    _json(paths.throughput_qualification_marker, marker)
    for root in (attempt_root, run_root):
        _seal_test_tree(root)
    return marker


def _seal_test_tree(root: Path) -> None:
    members = [root, *root.rglob("*")]
    for member in sorted(
        members,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        member.chmod(stat.S_IMODE(member.stat().st_mode) & ~0o222)


def _promote_throughput_qualification_to_additive_successor(
    paths: chain.RecoveryPaths,
    marker: dict[str, object],
) -> dict[str, object]:
    pointer_root = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY
    )
    pointer1_path = next(pointer_root.iterdir())
    pointer1 = json.loads(pointer1_path.read_text(encoding="utf-8"))
    attempt1_root = Path(pointer1["attempt_root"])
    attempt1_marker = (
        attempt1_root / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    attempt1_root.chmod(0o755)
    attempt1_marker.unlink()
    scaling = {
        "serving_profile": "8B",
        "server_pool_root": str(paths.pool),
        "backlog_fanout_work": 100,
        "live_replicas": 1,
        "backlog_work_per_replica": 100.0,
        "additional_replicas": 1,
        "tensor_parallel_size": 1,
        "additional_gpus": 1,
        "requirement": "add one replica (1 GPU)",
        "capacity_mutated": False,
    }
    failure = _identified_json(
        attempt1_root / chain.THROUGHPUT_QUALIFICATION_FAILURE_NAME,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL
            ),
            "passed": False,
            "intent_id": "d" * 64,
            "attempt": chain._qualification_attempt_binding(
                pointer1_path,
                pointer1,
            ),
            "readiness_generation": pointer1["readiness_generation"],
            "reason": "qualification throughput is below threshold",
            "additive_scaling_requirement": scaling,
            "scheduler_capacity_mutated": False,
            "rerun_requirement": (
                "publish a fresh serving/capacity and trusted-catalog "
                "rollout generation, then create a fresh qualification "
                "namespace and intent; this failed intent and its "
                "observations cannot be reused"
            ),
        },
        identity_field="failure_id",
    )
    _seal_test_tree(attempt1_root)

    readiness2 = {
        **pointer1["readiness_generation"],
        "catalog_id": "e" * 64,
        "marker_path": str(
            (paths.readiness / "TRUSTED_GENERATION_2.json").resolve()
        ),
        "marker_sha256": "f" * 64,
        "inventory_sha256": "0" * 64,
        "catalog_payload_sha256": "1" * 64,
        "fleet_contract_sha256": "2" * 64,
        "capacity_generation": 2,
        "rollout_generation": 2,
    }
    attempt2_id = chain._qualification_attempt_id(readiness2)
    attempt2_root = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_ATTEMPT_DIRECTORY
        / attempt2_id
    )
    run2_root = (
        paths.results_root
        / chain.THROUGHPUT_QUALIFICATION_RUN_DIRECTORY
        / attempt2_id
        / chain.THROUGHPUT_QUALIFICATION_ROOT_NAME
    )
    run2_root.mkdir(parents=True)
    pointer2_path = (
        pointer_root / f"000002-{attempt2_id}.json"
    )
    created_timestamp = 2_000.0
    pointer2 = _identified_json(
        pointer2_path,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_ATTEMPT_POINTER_PROTOCOL
            ),
            "chain_id": pointer1["chain_id"],
            "attempt_ordinal": 2,
            "attempt_id": attempt2_id,
            "attempt_root": str(attempt2_root),
            "run_root": str(run2_root),
            "dispatcher_state": str(attempt2_root / "dispatcher"),
            "readiness_generation": readiness2,
            "predecessor": chain._qualification_pointer_reference(
                pointer1_path,
                pointer1,
            ),
            "additive_retry": {
                "previous_failure_id": failure["failure_id"],
                "serving_profile": "8B",
                "additional_replicas": 1,
                "tensor_parallel_size": 1,
                "from_capacity_generation": 1,
                "to_capacity_generation": 2,
                "from_rollout_generation": 1,
                "to_rollout_generation": 2,
                "from_fleet_contract_sha256": (
                    pointer1["readiness_generation"][
                        "fleet_contract_sha256"
                    ]
                ),
                "to_fleet_contract_sha256": (
                    readiness2["fleet_contract_sha256"]
                ),
                "validation": (
                    "schema5_control._assert_additive_capacity_contract+"
                    "exact-required-profile-delta"
                ),
            },
            "created_at": chain.datetime.fromtimestamp(
                created_timestamp, tz=chain.timezone.utc
            ).isoformat(),
            "created_timestamp": created_timestamp,
        },
        identity_field="pointer_id",
    )
    pointer2_reference = chain._qualification_pointer_reference(
        pointer2_path,
        pointer2,
    )
    current_path = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_NAME
    )
    current_path.chmod(0o644)
    _identified_json(
        current_path,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL
            ),
            "attempt_id": attempt2_id,
            "pointer": str(pointer2_path),
            "pointer_sha256": pointer2_reference["sha256"],
            "pointer_id": pointer2["pointer_id"],
        },
        identity_field="current_id",
    )
    successor_identity = dict(marker)
    successor_identity.pop("qualification_id")
    successor_identity["attempt"] = chain._qualification_attempt_binding(
        pointer2_path,
        pointer2,
    )
    successor = _identified_json(
        attempt2_root / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME,
        successor_identity,
        identity_field="qualification_id",
    )
    paths.throughput_qualification_marker.chmod(0o644)
    _json(paths.throughput_qualification_marker, successor)
    _seal_test_tree(attempt2_root)
    _seal_test_tree(run2_root)
    return successor


def _replace_throughput_success_with_failure(
    paths: chain.RecoveryPaths,
    marker: dict[str, object],
) -> tuple[Path, dict[str, object], dict[str, object]]:
    pointer_root = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY
    )
    pointer_path = next(pointer_root.iterdir())
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    attempt_root = Path(pointer["attempt_root"])
    attempt_marker = (
        attempt_root / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    attempt_root.chmod(0o755)
    attempt_marker.unlink()
    paths.throughput_qualification_marker.unlink()
    scaling = {
        "serving_profile": "8B",
        "server_pool_root": str(paths.pool),
        "backlog_fanout_work": 100,
        "live_replicas": 1,
        "backlog_work_per_replica": 100.0,
        "additional_replicas": 1,
        "tensor_parallel_size": 1,
        "additional_gpus": 1,
        "requirement": "add one replica (1 GPU)",
        "capacity_mutated": False,
    }
    failure = _identified_json(
        attempt_root / chain.THROUGHPUT_QUALIFICATION_FAILURE_NAME,
        {
            "schema_version": 1,
            "protocol": (
                chain.THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL
            ),
            "passed": False,
            "intent_id": "d" * 64,
            "attempt": marker["attempt"],
            "readiness_generation": pointer["readiness_generation"],
            "reason": "qualification throughput is below threshold",
            "additive_scaling_requirement": scaling,
            "scheduler_capacity_mutated": False,
            "rerun_requirement": (
                "publish a fresh exact additive serving/readiness "
                "generation and rerun in a new immutable attempt"
            ),
        },
        identity_field="failure_id",
    )
    _seal_test_tree(attempt_root)
    return pointer_path, pointer, failure


def test_r2_render_adopts_snapshot_captures_seeds_and_has_afterany_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: {
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
    )
    sentinel_bytes = b"#!/usr/bin/env python3\nprint('sentinel')\n"
    monkeypatch.setattr(
        chain,
        "_tagged_file_bytes",
        lambda _repository, _commit, _relative: sentinel_bytes,
    )

    dry_run = chain.render_chain(
        paths, partition="mit_normal", slurm_user="tester", apply=False
    )
    assert dry_run["status"] == "dry_run"
    assert dry_run["job_count"] == len(chain.EXPECTED_JOB_ORDER) == 43
    assert not paths.jobs_root.exists()

    applied = chain.render_chain(
        paths, partition="mit_normal", slurm_user="tester", apply=True
    )
    assert applied["status"] == "complete"
    manifest = json.loads(paths.chain_manifest.read_text(encoding="utf-8"))
    jobs = {row["name"]: row for row in manifest["jobs"]}
    assert tuple(jobs) == chain.EXPECTED_JOB_ORDER
    assert jobs["failure_sentinel"]["dependency_type"] == "afterany"
    assert set(jobs["failure_sentinel"]["dependencies"]) == set(jobs) - {
        "failure_sentinel"
    }
    assert all(
        jobs[f"{chain.STAGE_SENTINEL_PREFIX}{stage}"]["dependency_type"]
        == "afterany"
        and jobs[f"{chain.STAGE_SENTINEL_PREFIX}{stage}"]["dependencies"]
        == [stage]
        for stage in chain.PRODUCTION_STAGE_NAMES
    )
    assert all(
        row["dependency_type"] == "afterok"
        for name, row in jobs.items()
        if name in chain.PRODUCTION_STAGE_NAMES
    )
    assert jobs["throughput_qualification"]["dependencies"] == [
        "smoke_readiness"
    ]
    assert jobs["controller_drill"]["dependencies"] == [
        "throughput_qualification",
        "email_readiness",
    ]
    assert jobs["production_resume"]["dependencies"] == [
        "controller_drill",
        "throughput_qualification",
    ]
    assert "protected_capacity" in manifest["prerequisite_evidence"]
    assert "external_watchdog_drill" not in manifest["prerequisite_evidence"]
    assert "watchdog_ready" not in manifest["prerequisite_evidence"]
    qualification = Path(
        jobs["throughput_qualification"]["script"]
    ).read_text(encoding="utf-8")
    assert "run_schema5_throughput_qualification.py" in qualification
    assert " execute " in qualification
    qualification_attestations = list(
        re.finditer(
            (
                rf"--state-dir\s+{re.escape(chain._q(paths.state))}\s+"
                r"attest\s+--gate\s+throughput_qualification\s+"
                rf"--chain-manifest\s+"
                f"{re.escape(chain._q(paths.chain_manifest))}\\s+"
                rf"--evidence\s+"
                f"{re.escape(chain._q(paths.throughput_qualification_marker))}"
            ),
            qualification,
        )
    )
    assert len(qualification_attestations) == 1
    qualification_execute_offset = qualification.index(
        '"$qualification_producer" execute'
    )
    qualification_verify_offset = qualification.index(
        "verify-throughput-qualification"
    )
    assert (
        qualification_execute_offset
        < qualification_verify_offset
        < qualification_attestations[0].start()
    )
    drill = Path(jobs["controller_drill"]["script"]).read_text(encoding="utf-8")
    assert "watchdog-drill arm" in drill
    assert "watchdog-drill cancel" in drill
    assert "verify-watchdog-readiness" in drill
    watchdog_attestations = list(
        re.finditer(
            (
                r'"\$\{control\[@\]\}"\s+attest\s+--gate\s+'
                r"external_watchdog\s+"
                rf"--chain-manifest\s+"
                f"{re.escape(chain._q(paths.chain_manifest))}\\s+"
                rf"--evidence\s+"
                f"{re.escape(chain._q(paths.watchdog_ready_marker))}"
            ),
            drill,
        )
    )
    watchdog_verifications = [
        match.start()
        for match in re.finditer(
            r"\bverify-watchdog-readiness\b", drill
        )
    ]
    assert len(watchdog_attestations) == 2
    assert len(watchdog_verifications) == 3
    completed_marker_offset = drill.index(
        "CONTROLLER_KILL_DRILL_COMPLETE.json"
    )
    early_guard_offset = drill.rfind(
        "if [[", 0, completed_marker_offset
    )
    early_exit_offset = drill.index(
        "  exit 0\nfi\n", completed_marker_offset
    )
    reconcile_offset = drill.index(
        '"${control[@]}" reconcile --all --no-admit',
        early_exit_offset,
    )
    publisher_offset = drill.index(
        f"-I {chain._q(paths.worktree / 'scripts' / 'publish_schema5_watchdog_ready.py')}",
        reconcile_offset,
    )
    dispatcher_kill_offset = drill.index(
        '"${control[@]}" drill kill --role dispatcher'
    )
    assert (
        early_guard_offset
        < watchdog_verifications[0]
        < watchdog_attestations[0].start()
        < early_exit_offset
        < reconcile_offset
        < publisher_offset
        < watchdog_verifications[1]
        < watchdog_attestations[1].start()
        < dispatcher_kill_offset
        < watchdog_verifications[2]
    )
    assert "exec env" not in drill[
        early_guard_offset : watchdog_attestations[0].end()
    ]
    materialize = Path(jobs["release_materialize"]["script"]).read_text(
        encoding="utf-8"
    )
    assert str(paths.captured_harness) in materialize
    assert str(paths.captured_serving) in materialize
    assert (
        f"--source-harness-prefix {paths.source_harness}" not in materialize
    )
    capture = Path(jobs["environment_capture"]["script"]).read_text(
        encoding="utf-8"
    )
    assert "capture_schema5_environments.py" in capture
    assert "--ownership-policy" in capture
    assert "--integrity-normalization-policy" in capture
    assert "--apply" in capture
    snapshot_adopt = Path(jobs["snapshot_adopt_verify"]["script"]).read_text(
        encoding="utf-8"
    )
    assert "verify_schema5_recovery_evidence.py" in snapshot_adopt
    assert "protocol-aware r1 evidence report differs from sealed contract" in (
        snapshot_adopt
    )
    assert snapshot_adopt.find(
        "bundled protocol-aware r1 evidence verifier hash drifted"
    ) < snapshot_adopt.find('"$protocol_verifier"')
    assert (
        manifest["r1_protocol_verification"]["native_report"]["chain_protocol"]
        == chain.R1_CHAIN_PROTOCOL
    )
    sentinel = Path(jobs["failure_sentinel"]["script"]).read_text(
        encoding="utf-8"
    )
    fleet_readiness = Path(jobs["fleet_readiness"]["script"]).read_text(
        encoding="utf-8"
    )
    qualification = Path(
        jobs["throughput_qualification"]["script"]
    ).read_text(encoding="utf-8")
    drill = Path(jobs["controller_drill"]["script"]).read_text(
        encoding="utf-8"
    )
    resume = Path(jobs["production_resume"]["script"]).read_text(
        encoding="utf-8"
    )
    assert "schema5_recovery_sentinel.py" in sentinel
    assert "--apply" in sentinel
    assert 'if [[ "$generation" == g0000 ]]' in sentinel
    assert chain.REPAIR_ROOT_NAME in sentinel
    assert '/"$generation"' in sentinel
    assert '--submission-receipt "$receipt"' in sentinel
    assert '--capacity-transient-receipt "$capacity_receipt"' in sentinel
    assert chain.CAPACITY_TRANSIENT_ROOT_NAME in sentinel
    stage_sentinel = Path(
        jobs[f"{chain.STAGE_SENTINEL_PREFIX}context_readiness"]["script"]
    ).read_text(encoding="utf-8")
    assert "--stage-name context_readiness" in stage_sentinel
    assert '--stage-sentinel-job-id "$SLURM_JOB_ID"' in stage_sentinel
    assert "--capacity-transient-receipt" not in stage_sentinel
    assert "fleet-capacity-transient" in fleet_readiness
    assert '--readiness-job-id "$SLURM_JOB_ID"' in fleet_readiness
    assert "exit 75" in fleet_readiness
    assert chain.CAPACITY_TRANSIENT_ROOT_NAME in fleet_readiness
    assert "run_schema5_throughput_qualification.py" in qualification
    assert '"$qualification_producer" execute' in qualification
    assert "verify-throughput-qualification" in qualification
    assert "verify-throughput-qualification" in drill
    assert "verify-throughput-qualification" in resume
    assert chain.PROTECTED_CAPACITY_MARKER_NAME in qualification
    assert chain.WATCHDOG_READY_MARKER_NAME not in qualification
    assert chain.EXTERNAL_WATCHDOG_DRILL_MARKER_NAME not in qualification
    assert (
        paths.jobs_root / chain.SENTINEL_TOOL_FILENAME
    ).read_bytes() == sentinel_bytes
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True


def test_full_capacity_marker_is_a_fail_closed_render_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    paths.protected_capacity_marker.chmod(0o644)

    with pytest.raises(
        chain.ChainError,
        match="protected-capacity completion marker remains writable",
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )


def test_watchdog_is_post_initialize_not_a_renderer_bootstrap_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    paths.external_watchdog_drill_marker.unlink()
    paths.watchdog_ready_marker.unlink()
    _, manifest = _render_applied(paths, monkeypatch)
    assert manifest["prerequisite_evidence"]["schema_version"] == 5
    assert "external_watchdog_drill" not in manifest["prerequisite_evidence"]
    assert "watchdog_ready" not in manifest["prerequisite_evidence"]
    assert len(manifest["jobs"]) == 43


def test_post_initialize_watchdog_binds_exact_paused_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    paths.state.mkdir(parents=True, exist_ok=True)
    _json(
        paths.state / "control.json",
        {
            "desired_state": "paused",
            "drain_requested": False,
            "immutable_sha256": "c" * 64,
            "immutable": {
                "release_id": chain.RELEASE_ID,
                "git_commit": COMMIT,
            },
        },
    )
    report = chain.verify_post_initialize_watchdog(paths.chain_manifest)
    assert report["passed"] is True
    assert report["control_sha256"] == "c" * 64

    control = json.loads(
        (paths.state / "control.json").read_text(encoding="utf-8")
    )
    control["immutable_sha256"] = "d" * 64
    (paths.state / "control.json").chmod(0o644)
    _json(paths.state / "control.json", control)
    with pytest.raises(chain.ChainError, match="initialized paused control"):
        chain.verify_post_initialize_watchdog(paths.chain_manifest)


@pytest.mark.parametrize(
    "mutation",
    ["invalid_source_hash", "placement_total_drift", "unsorted_placements"],
)
def test_protected_capacity_source_and_placement_bindings_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    marker = json.loads(
        paths.protected_capacity_marker.read_text(encoding="utf-8")
    )
    if mutation == "invalid_source_hash":
        marker["scheduler_evidence_sha256"] = "x" * 64
    elif mutation == "placement_total_drift":
        marker["scientific_server_placements"][0][
            "active_serving_gpus"
        ] = 25
    else:
        marker["scientific_server_placements"] = [
            {
                "partition": "z_gpu",
                "qos": "z_science",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "OFF",
                "active_serving_gpus": 24,
                "warm_headroom_gpus": 0,
            },
            {
                "partition": "a_gpu",
                "qos": "a_science",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "OFF",
                "active_serving_gpus": 0,
                "warm_headroom_gpus": 4,
            },
        ]
    marker.pop("marker_id")
    marker["marker_id"] = chain._sha256_bytes(chain._canonical_json(marker))
    paths.protected_capacity_marker.chmod(0o644)
    _json(paths.protected_capacity_marker, marker)

    with pytest.raises(
        chain.ChainError,
        match=(
            "non-preemptible scientific placement"
            if mutation != "unsorted_placements"
            else "not canonically sorted"
        ),
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )


def test_throughput_qualification_binds_full_384_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    marker = _publish_throughput_qualification(paths, manifest)

    verified = chain.verify_throughput_qualification(paths.chain_manifest)
    assert verified["passed"] is True
    assert verified["qualification_id"] == marker["qualification_id"]
    assert verified["cells"] == 768
    assert verified["qids"] == 15_360
    assert verified["throughput_qids_per_day"] == 201_994

    marker["throughput_qids_per_day"] = 201_993
    marker.pop("qualification_id")
    marker["qualification_id"] = chain._sha256_bytes(
        chain._canonical_json(marker)
    )
    attempt_marker = (
        Path(marker["attempt"]["attempt_root"])
        / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    for marker_path in (
        attempt_marker,
        paths.throughput_qualification_marker,
    ):
        marker_path.chmod(0o644)
        _json(marker_path, marker)
    with pytest.raises(
        chain.ChainError,
        match="full-384 protected-capacity contract",
    ):
        chain.verify_throughput_qualification(paths.chain_manifest)


def test_throughput_qualification_accepts_exact_additive_attempt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    initial = _publish_throughput_qualification(paths, manifest)
    successor = _promote_throughput_qualification_to_additive_successor(
        paths,
        initial,
    )

    verified = chain.verify_throughput_qualification(paths.chain_manifest)

    assert verified["passed"] is True
    assert verified["attempt_id"] == successor["attempt"]["attempt_id"]
    assert len(
        list(
            (
                paths.throughput_qualification_root
                / chain.THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY
            ).iterdir()
        )
    ) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("byte_mismatch", "not byte-identical"),
        ("attempt_binding", "exact current attempt"),
        ("writable_attempt", "recursively sealed read-only"),
        ("writable_run", "recursively sealed read-only"),
        ("hardlink_alias", "recursively sealed read-only"),
    ],
)
def test_throughput_qualification_attempt_seal_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    marker = _publish_throughput_qualification(paths, manifest)
    attempt = marker["attempt"]
    assert isinstance(attempt, dict)
    attempt_root = Path(attempt["attempt_root"])
    run_root = Path(attempt["run_root"])
    attempt_marker = (
        attempt_root / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME
    )

    if mutation == "byte_mismatch":
        attempt_marker.chmod(0o644)
        attempt_marker.write_bytes(chain._canonical_json(marker))
        attempt_marker.chmod(0o444)
    elif mutation == "attempt_binding":
        mutated = dict(marker)
        mutated_attempt = dict(attempt)
        mutated_attempt["capacity_generation"] = 2
        mutated["attempt"] = mutated_attempt
        mutated.pop("qualification_id")
        mutated["qualification_id"] = chain._sha256_bytes(
            chain._canonical_json(mutated)
        )
        for marker_path in (
            attempt_marker,
            paths.throughput_qualification_marker,
        ):
            marker_path.chmod(0o644)
            _json(marker_path, mutated)
    elif mutation == "writable_attempt":
        attempt_root.chmod(0o755)
    elif mutation == "writable_run":
        run_root.chmod(0o755)
    else:
        source = attempt_root / "sealed-evidence.json"
        attempt_root.chmod(0o755)
        _json(source, {"sealed": True})
        alias = tmp_path / "external-hardlink-alias.json"
        os.link(source, alias)
        attempt_root.chmod(0o555)

    with pytest.raises(chain.ChainError, match=message):
        chain.verify_throughput_qualification(paths.chain_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("predecessor", "attempt-pointer identity is invalid"),
        ("stale_current", "does not select the latest"),
        ("latest_failed", "terminally failed"),
        ("writable_prior", "recursively sealed read-only"),
    ],
)
def test_throughput_qualification_latest_attempt_chain_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    initial = _publish_throughput_qualification(paths, manifest)
    successor = _promote_throughput_qualification_to_additive_successor(
        paths,
        initial,
    )
    pointer_root = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY
    )
    pointer_paths = sorted(pointer_root.iterdir())
    pointer1 = json.loads(pointer_paths[0].read_text(encoding="utf-8"))
    pointer2 = json.loads(pointer_paths[1].read_text(encoding="utf-8"))
    current_path = (
        paths.throughput_qualification_root
        / chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_NAME
    )

    if mutation == "predecessor":
        pointer2["predecessor"]["sha256"] = "f" * 64
        pointer2.pop("pointer_id")
        pointer2["pointer_id"] = chain._sha256_bytes(
            chain._canonical_json(pointer2)
        )
        pointer_paths[1].chmod(0o644)
        _json(pointer_paths[1], pointer2)
    elif mutation == "stale_current":
        reference = chain._qualification_pointer_reference(
            pointer_paths[0],
            pointer1,
        )
        current_path.chmod(0o644)
        _identified_json(
            current_path,
            {
                "schema_version": 1,
                "protocol": (
                    chain.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL
                ),
                "attempt_id": pointer1["attempt_id"],
                "pointer": str(pointer_paths[0]),
                "pointer_sha256": reference["sha256"],
                "pointer_id": pointer1["pointer_id"],
            },
            identity_field="current_id",
        )
    elif mutation == "latest_failed":
        attempt_root = Path(successor["attempt"]["attempt_root"])
        attempt_root.chmod(0o755)
        (
            attempt_root
            / chain.THROUGHPUT_QUALIFICATION_MARKER_NAME
        ).unlink()
        _json(
            attempt_root / chain.THROUGHPUT_QUALIFICATION_FAILURE_NAME,
            {"terminal": True},
        )
        _seal_test_tree(attempt_root)
    else:
        Path(initial["attempt"]["attempt_root"]).chmod(0o755)

    with pytest.raises(chain.ChainError, match=message):
        chain.verify_throughput_qualification(paths.chain_manifest)


def test_prerequisites_are_hash_bound_and_reverified_at_every_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    payloads = _stub_tagged_release(monkeypatch)
    original_verifier = chain._invoke_tagged_prerequisite_verifiers
    verifier_checkouts: list[Path] = []
    verifier_pythons: list[Path | None] = []

    def counted_verifier(received_paths, **kwargs):
        verifier_checkouts.append(Path(kwargs["checkout"]))
        verifier_pythons.append(
            None
            if kwargs.get("verifier_python") is None
            else Path(kwargs["verifier_python"])
        )
        return original_verifier(received_paths, **kwargs)

    monkeypatch.setattr(
        chain, "_invoke_tagged_prerequisite_verifiers", counted_verifier
    )
    chain.render_chain(
        paths,
        partition="mit_normal",
        slurm_user="tester",
        apply=True,
    )
    # Creation-time render cross-checks the live and sealed-pilot runtimes.  The
    # marker-last chain verifier is deliberately self-contained and does not return
    # to mutable source.
    assert verifier_checkouts == [
        paths.repository,
        paths.repository,
    ]
    sealed_python = Path(
        json.loads(paths.chain_manifest.read_text(encoding="utf-8"))[
            "prerequisite_evidence"
        ]["materialization_pilot"]["verifier_runtime"]["python_path"]
    )
    assert verifier_pythons == [None, sealed_python]

    manifest = json.loads(paths.chain_manifest.read_text(encoding="utf-8"))
    prerequisite = manifest["prerequisite_evidence"]
    assert manifest["schema_version"] == chain.CHAIN_SCHEMA_VERSION
    assert manifest["release_tag"] == chain.RELEASE_TAG
    assert manifest["release_git_commit"] == COMMIT
    assert manifest["release_tag_object"] == TAG_OBJECT
    assert prerequisite["release_tag"] == chain.RELEASE_TAG
    assert prerequisite["release_git_commit"] == COMMIT
    assert prerequisite["release_tag_object"] == TAG_OBJECT
    assert prerequisite["materialization_pilot"]["root"] == str(
        paths.materialization_pilot_root
    )
    assert prerequisite["slurm_canary"]["root"] == str(paths.slurm_canary_root)
    assert prerequisite["materialization_pilot"]["pilot_id"] == "4" * 64
    assert prerequisite["materialization_pilot"][
        "ownership_policy_sha256"
    ] == prerequisite["materialization_pilot"]["verifier_report"][
        "ownership_policy_sha256"
    ]
    assert prerequisite["materialization_pilot"][
        "integrity_normalization_policy_sha256"
    ] == prerequisite["materialization_pilot"]["verifier_report"][
        "integrity_normalization_policy_sha256"
    ]
    assert prerequisite["slurm_canary"]["canary_id"] == "5" * 64
    assert prerequisite["slurm_canary"]["transaction_canary_id"] == "6" * 64
    assert prerequisite["slurm_canary"]["turnover_canary_id"] == "8" * 64
    assert prerequisite["slurm_canary"]["turnover_cycles_completed"] == 2
    assert prerequisite["slurm_canary"][
        "turnover_maximum_physical_allocations_observed"
    ] == 2
    assert prerequisite["slurm_canary"][
        "turnover_continuous_routed_endpoint_evidence"
    ] is True
    assert prerequisite["materialization_pilot"]["marker_sha256"] == chain._sha256(
        paths.materialization_pilot_root / chain.MATERIALIZATION_PILOT_MARKER
    )
    assert prerequisite["slurm_canary"]["marker_sha256"] == chain._sha256(
        paths.slurm_canary_root / chain.SLURM_CANARY_MARKER
    )
    assert prerequisite["evidence_id"] == chain._sha256_bytes(
        chain._canonical_json(
            {
                key: value
                for key, value in prerequisite.items()
                if key != "evidence_id"
            }
        )
    )
    assert prerequisite["tagged_code"] == [
        {
            "git_path": git_path,
            "sha256": hashlib.sha256(payloads[git_path]).hexdigest(),
            "size": len(payloads[git_path]),
        }
        for git_path in chain.PREREQUISITE_CODE_GIT_PATHS
    ]

    # Once source checkout exists, verification cannot fall back to mutable
    # developer source.  Sealed verification uses neither checkout; the explicit
    # creation-time verifier uses the exact target.
    (paths.source_checkout / ".git").mkdir(parents=True)
    before_sealed_verify = len(verifier_checkouts)
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True
    assert len(verifier_checkouts) == before_sealed_verify
    assert chain.verify_live_creation_prerequisites(
        paths.chain_manifest
    )["passed"] is True
    assert verifier_checkouts[-1] == paths.source_checkout

    before_submit = len(verifier_checkouts)

    def stop_before_scheduler(_runner, **_kwargs):
        raise chain.ChainError("scheduler boundary reached")

    monkeypatch.setattr(
        chain, "_dependency_config_allows_fail_closed", stop_before_scheduler
    )
    with pytest.raises(chain.ChainError, match="scheduler boundary reached"):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=lambda _argv: pytest.fail("scheduler must not be contacted"),
        )
    # Apply submission verifies once before and once inside its singleton lock.
    assert len(verifier_checkouts) == before_submit + 2
    assert verifier_checkouts[-2:] == [
        paths.source_checkout,
        paths.source_checkout,
    ]


class _RecoveryLaunchScheduler:
    def __init__(
        self,
        recovery_root: Path,
        *,
        crash_after_release: bool = False,
        crash_before_release: bool = False,
        dependency_drift: str | None = None,
        dependency_or_drift: str | None = None,
        requeue_drift: str | None = None,
        spool_drift: str | None = None,
        crash_after_spool_write: str | None = None,
        foreign_namespace: bool = False,
        missing_squeue: str | None = None,
        missing_sacct: str | None = None,
    ):
        self.recovery_root = recovery_root
        self.crash_after_release = crash_after_release
        self.crash_before_release = crash_before_release
        self.dependency_drift = dependency_drift
        self.dependency_or_drift = dependency_or_drift
        self.requeue_drift = requeue_drift
        self.spool_drift = spool_drift
        self.crash_after_spool_write = crash_after_spool_write
        self.foreign_namespace = foreign_namespace
        self.missing_squeue = missing_squeue
        self.missing_sacct = missing_sacct
        self.next_job_id = 770000
        self.jobs: dict[str, dict[str, str]] = {}
        self.sbatch_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []

    def __call__(self, argv):
        if argv == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "DependencyParameters = disable_remote_singleton_jobs,"
                    "kill_invalid_depend\n"
                ),
                stderr="",
            )
        if argv[0] == "sbatch":
            path = Path(argv[-1])
            script = path.read_text(encoding="utf-8")
            name = next(
                line.split("=", 1)[1]
                for line in script.splitlines()
                if line.startswith("#SBATCH --job-name=")
            )
            comment = next(
                value.split("=", 1)[1]
                for value in argv
                if value.startswith("--comment=")
            )
            dependency = next(
                (
                    value.split("=", 1)[1]
                    for value in argv
                    if value.startswith("--dependency=")
                ),
                "(null)",
            )
            job_id = str(self.next_job_id)
            self.next_job_id += 1
            is_root = "--hold" in argv
            if is_root:
                assert "--hold" in argv
                state = "PENDING"
                reason = "JobHeldUser"
            else:
                assert "--hold" not in argv
                state = "PENDING"
                reason = "Dependency"
            self.jobs[job_id] = {
                "name": name,
                "comment": comment,
                "state": state,
                "reason": reason,
                "path": str(path),
                "dependency": dependency,
                "script": script,
            }
            self.sbatch_calls.append(list(argv))
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{job_id};cluster\n", stderr=""
            )
        if argv[:4] == ["squeue", "-u", "tester", "-h"]:
            output_format = argv[-1]
            if output_format == "%i|%T|%k|%j":
                rows = "".join(
                    f"{job_id}|{job['state']}|{job['comment']}|{job['name']}\n"
                    for job_id, job in self.jobs.items()
                    if job["name"] != self.missing_squeue
                )
                if self.foreign_namespace and self.jobs:
                    first = next(iter(self.jobs.values()))
                    rows += (
                        f"999999|PENDING|{first['comment']}|foreign-job\n"
                    )
            else:
                rows = "".join(
                    f"{job_id}|{job['comment']}|{job['name']}|{job['state']}\n"
                    for job_id, job in self.jobs.items()
                )
            return subprocess.CompletedProcess(
                argv, 0, stdout=rows, stderr=""
            )
        if argv[0] == "sacct":
            output_format = next(
                value for value in argv if value.startswith("--format=")
            )
            if output_format.startswith("--format=JobIDRaw,State"):
                rows = "".join(
                    (
                        f"{job_id}|{job['state']}|0:0||{job['name']}|"
                        f"sbatch --comment={job['comment']} {job['path']}\n"
                    )
                    for job_id, job in self.jobs.items()
                    if job["name"] != self.missing_sacct
                )
            else:
                rows = "".join(
                    (
                        f"{job_id}||{job['name']}|{job['state']}|"
                        f"sbatch --comment={job['comment']} {job['path']}\n"
                    )
                    for job_id, job in self.jobs.items()
                )
            return subprocess.CompletedProcess(argv, 0, stdout=rows, stderr="")
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            job_id = argv[4]
            job = self.jobs[job_id]
            dependency = job["dependency"]
            if job["name"] == self.dependency_drift:
                dependency = "afterok:999999(unfulfilled)"
            else:
                dependency = ",".join(
                    (
                        part.split(":", 1)[0]
                        + ":"
                        + dependency_job_id
                        + "(unfulfilled)"
                    )
                    for part in dependency.split(",")
                    for dependency_job_id in part.split(":")[1:]
                ) if dependency != "(null)" else dependency
            if (
                job["name"] == self.dependency_or_drift
                and "," in dependency
            ):
                dependency = dependency.replace(",", "?", 1)
            requeue = 1 if job["name"] == self.requeue_drift else 0
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    f"JobId={job_id} JobName={job['name']} "
                    f"JobState={job['state']} Reason={job['reason']} "
                    f"Comment={job['comment']} Command={job['path']} "
                    f"Dependency={dependency} Requeue={requeue}\n"
                ),
                stderr="",
            )
        if argv[:3] == ["scontrol", "write", "batch_script"]:
            job_id = argv[3]
            job = self.jobs[job_id]
            payload = job["script"]
            if job["name"] == self.spool_drift:
                payload += "# scheduler drift\n"
            target = Path(argv[4])
            assert not target.exists()
            target.write_text(payload, encoding="utf-8")
            if job["name"] == self.crash_after_spool_write:
                self.crash_after_spool_write = None
                raise KeyboardInterrupt("crash after scheduler spool write")
            return subprocess.CompletedProcess(
                argv, 0, stdout="", stderr=""
            )
        if argv[:2] == ["scontrol", "release"]:
            receipt = self.recovery_root / chain.SUBMISSION_RECEIPT_NAME
            assert receipt.is_file()
            assert not stat.S_IMODE(receipt.stat().st_mode) & 0o222
            job_id = argv[2]
            self.release_calls.append(list(argv))
            if self.crash_before_release:
                self.crash_before_release = False
                raise KeyboardInterrupt("crash before exact root release")
            self.jobs[job_id]["state"] = "RUNNING"
            self.jobs[job_id]["reason"] = "None"
            if self.crash_after_release:
                self.crash_after_release = False
                raise KeyboardInterrupt("crash after exact root release")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected launch scheduler command: {argv}")


def test_chain_submission_holds_root_until_receipt_then_releases_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(paths.recovery_root)

    result = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_400.0,
    )

    assert result["status"] == "launched"
    assert len(scheduler.sbatch_calls) == len(chain.EXPECTED_JOB_ORDER)
    assert scheduler.release_calls == [["scontrol", "release", "770000"]]
    assert "--hold" in scheduler.sbatch_calls[0]
    assert all("--hold" not in argv for argv in scheduler.sbatch_calls[1:])
    receipt = json.loads(
        (paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["root_initial_hold"] is True
    assert receipt["stage_failure_sentinels"] == [
        {
            "stage": stage,
            "sentinel": f"{chain.STAGE_SENTINEL_PREFIX}{stage}",
        }
        for stage in chain.PRODUCTION_STAGE_NAMES
    ]
    assert receipt["dependency_canary"]["alert_latency_bound_seconds"] == 180.0
    assert receipt["dependency_policy"] == chain.DEPENDENCY_POLICY_CONTRACT
    assert (
        paths.recovery_root / chain.ROOT_RELEASE_COMPLETE_NAME
    ).is_file()
    assert (paths.recovery_root / chain.LAUNCH_COMPLETE_NAME).is_file()

    repeated = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_400.0,
    )
    assert repeated["status"] == "already_launched"
    assert len(scheduler.sbatch_calls) == len(chain.EXPECTED_JOB_ORDER)
    assert scheduler.release_calls == [["scontrol", "release", "770000"]]


@pytest.mark.parametrize(
    ("scheduler_kwargs", "message"),
    [
        (
            {"dependency_drift": "asys-s5v12r2-resume"},
            "scontrol acceptance provenance drifted",
        ),
        (
            {"dependency_or_drift": "asys-s5v12r2-resume"},
            "OR semantics",
        ),
        (
            {"requeue_drift": "asys-s5v12r2-resume"},
            "scontrol acceptance provenance drifted",
        ),
        (
            {"spool_drift": "asys-s5v12r2-resume"},
            "spooled sbatch differs",
        ),
        (
            {"foreign_namespace": True},
            "foreign squeue job collides",
        ),
        (
            {"missing_squeue": "asys-s5v12r2-resume"},
            "squeue set is incomplete",
        ),
        (
            {"missing_sacct": "asys-s5v12r2-resume"},
            "sacct set is incomplete",
        ),
    ],
)
def test_chain_scheduler_acceptance_rejects_live_or_spooled_drift_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheduler_kwargs: dict[str, object],
    message: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(
        paths.recovery_root, **scheduler_kwargs
    )

    with pytest.raises(chain.ChainError, match=message):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=scheduler,
            now=1_721_750_400.0,
        )

    assert scheduler.release_calls == []
    assert scheduler.jobs["770000"]["state"] == "PENDING"
    assert scheduler.jobs["770000"]["reason"] == "JobHeldUser"
    assert not (
        paths.recovery_root / chain.ROOT_RELEASE_COMPLETE_NAME
    ).exists()
    assert not (
        paths.recovery_root / chain.LAUNCH_COMPLETE_NAME
    ).exists()


def test_chain_scheduler_acceptance_crash_before_release_replays_sealed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(
        paths.recovery_root, crash_before_release=True
    )

    with pytest.raises(KeyboardInterrupt, match="before exact root release"):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=scheduler,
            now=1_721_750_400.0,
        )
    acceptance_path = (
        paths.recovery_root / chain.SCHEDULER_ACCEPTANCE_COMPLETE_NAME
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    artifact_root = Path(acceptance["artifact_root"])
    assert acceptance["job_count"] == 43
    assert len(acceptance["artifact_inventory"]) == 86
    assert acceptance_path.stat().st_nlink == 1
    assert not stat.S_IMODE(acceptance_path.stat().st_mode) & 0o222
    assert not stat.S_IMODE(artifact_root.stat().st_mode) & 0o222
    assert len(tuple(artifact_root.iterdir())) == 86
    assert scheduler.jobs["770000"]["state"] == "PENDING"

    recovered = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_401.0,
    )
    assert recovered["status"] == "already_launched"
    assert scheduler.jobs["770000"]["state"] == "RUNNING"
    assert scheduler.release_calls == [
        ["scontrol", "release", "770000"],
        ["scontrol", "release", "770000"],
    ]
    release = recovered["root_release"]
    assert release["scheduler_acceptance"] == str(acceptance_path)
    assert (
        release["scheduler_acceptance_id"]
        == acceptance["acceptance_id"]
    )


def test_chain_scheduler_acceptance_crash_after_direct_spool_write_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(
        paths.recovery_root,
        crash_after_spool_write="asys-s5v12r2-resume",
    )

    with pytest.raises(
        KeyboardInterrupt, match="after scheduler spool write"
    ):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=scheduler,
            now=1_721_750_400.0,
        )
    assert scheduler.release_calls == []
    assert not (
        paths.recovery_root / chain.SCHEDULER_ACCEPTANCE_COMPLETE_NAME
    ).exists()
    staging_root = (
        paths.recovery_root
        / chain.SCHEDULER_ACCEPTANCE_STAGING_ROOT_NAME
    )
    assert list(staging_root.iterdir()) == []

    recovered = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_401.0,
    )
    assert recovered["status"] == "already_launched"
    assert scheduler.release_calls == [
        ["scontrol", "release", "770000"]
    ]
    acceptance = json.loads(
        (
            paths.recovery_root
            / chain.SCHEDULER_ACCEPTANCE_COMPLETE_NAME
        ).read_text(encoding="utf-8")
    )
    assert acceptance["job_count"] == 43
    assert len(acceptance["artifact_inventory"]) == 86


def test_generated_launch_validator_executes_against_schema4_receipt_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the embedded stage gate, not merely its renderer-side validator."""

    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(paths.recovery_root)
    chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_400.0,
    )

    manifest = json.loads(paths.chain_manifest.read_text(encoding="utf-8"))
    receipt_path = paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == chain.SUBMISSION_SCHEMA_VERSION == 4
    assert receipt["dependency_policy"] == chain.DEPENDENCY_POLICY_CONTRACT
    stage_name = "environment_capture"
    manifest_stage = next(
        row for row in manifest["jobs"] if row["name"] == stage_name
    )
    receipt_stage = next(
        row for row in receipt["jobs"] if row["name"] == stage_name
    )
    rendered = Path(manifest_stage["script"]).read_text(encoding="utf-8")
    heredoc_anchor = '"$launch_gate_comment" <<\'PY\'\n'
    validator_start = rendered.index(heredoc_anchor) + len(heredoc_anchor)
    validator_end = rendered.index("\nPY\n", validator_start)
    validator = rendered[validator_start:validator_end] + "\n"

    process = subprocess.run(
        [
            sys.executable,
            "-",
            str(paths.chain_manifest),
            str(paths.recovery_root),
            str(receipt_path),
            str(paths.recovery_root / chain.ROOT_RELEASE_COMPLETE_NAME),
            str(paths.recovery_root / chain.LAUNCH_COMPLETE_NAME),
            manifest["chain_id"],
            "0",
            stage_name,
            receipt_stage["job_id"],
            receipt_stage["comment"],
        ],
        input=validator,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert (
        "launch authorization passed for generation 0, "
        f"stage {stage_name}, job {receipt_stage['job_id']}"
    ) in process.stdout


def test_chain_root_release_crash_reconciles_without_second_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(
        paths.recovery_root, crash_after_release=True
    )

    with pytest.raises(KeyboardInterrupt, match="crash after exact root release"):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=scheduler,
            now=1_721_750_400.0,
        )
    assert scheduler.release_calls == [["scontrol", "release", "770000"]]
    assert (
        paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    ).is_file()
    assert not (
        paths.recovery_root / chain.ROOT_RELEASE_COMPLETE_NAME
    ).exists()

    recovered = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_401.0,
    )
    assert recovered["status"] == "already_launched"
    assert scheduler.release_calls == [["scontrol", "release", "770000"]]
    assert recovered["root_release"]["release_reconciled"] is True
    assert (
        recovered["root_release"]["release_attempts"][0]["result"]
        is None
    )


def test_chain_root_remains_held_when_release_time_dependency_policy_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(paths.recovery_root)
    config_queries = [0]

    def one_drift(argv):
        if argv == ["scontrol", "show", "config"]:
            config_queries[0] += 1
            if config_queries[0] == 2:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="DependencyParameters = (null)\n",
                    stderr="",
                )
        return scheduler(argv)

    with pytest.raises(chain.ChainError, match="kill_invalid_depend"):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=one_drift,
            now=1_721_750_400.0,
        )
    assert len(scheduler.sbatch_calls) == len(chain.EXPECTED_JOB_ORDER)
    assert scheduler.release_calls == []
    assert scheduler.jobs["770000"]["state"] == "PENDING"
    assert scheduler.jobs["770000"]["reason"] == "JobHeldUser"
    assert (
        paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    ).is_file()
    assert not (paths.recovery_root / chain.LAUNCH_COMPLETE_NAME).exists()

    recovered = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=one_drift,
        now=1_721_750_401.0,
    )
    assert recovered["status"] == "already_launched"
    assert scheduler.release_calls == [["scontrol", "release", "770000"]]


def test_repair_generation_holds_first_suffix_job_until_repair_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    (paths.source_checkout / ".git").mkdir(parents=True)
    scheduler = _RecoveryLaunchScheduler(paths.recovery_root)
    chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_400.0,
    )
    repair_names = [
        "fleet_readiness",
        "smoke_readiness",
        "throughput_qualification",
        "controller_drill",
        "production_resume",
        f"{chain.STAGE_SENTINEL_PREFIX}fleet_readiness",
        f"{chain.STAGE_SENTINEL_PREFIX}smoke_readiness",
        f"{chain.STAGE_SENTINEL_PREFIX}throughput_qualification",
        f"{chain.STAGE_SENTINEL_PREFIX}controller_drill",
        f"{chain.STAGE_SENTINEL_PREFIX}production_resume",
        "failure_sentinel",
    ]
    states = {
        name: {
            "state": (
                "NODE_FAIL"
                if name == "fleet_readiness"
                else ("CANCELLED" if name in repair_names else "COMPLETED")
            ),
            "active": False,
        }
        for name in chain.EXPECTED_JOB_ORDER
    }
    monkeypatch.setattr(
        chain,
        "_query_receipt_job_states",
        lambda **_kwargs: states,
    )
    monkeypatch.setattr(
        chain,
        "_sentinel_repair_jobs",
        lambda **_kwargs: list(repair_names),
    )

    result = chain.repair_chain(
        paths.chain_manifest,
        apply=True,
        runner=scheduler,
        now=1_721_750_500.0,
    )

    assert result["status"] == "repair_launched"
    assert result["repair_jobs"] == repair_names
    repair_calls = scheduler.sbatch_calls[len(chain.EXPECTED_JOB_ORDER):]
    assert len(repair_calls) == len(repair_names)
    assert "--hold" in repair_calls[0]
    assert all("--hold" not in argv for argv in repair_calls[1:])
    assert any(
        value.startswith("--dependency=afterok:")
        for value in repair_calls[0]
    )
    generation_root = (
        paths.recovery_root / chain.REPAIR_ROOT_NAME / "g0001"
    )
    receipt = json.loads(
        (generation_root / chain.SUBMISSION_RECEIPT_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["root_initial_hold"] is True
    assert (
        next(
            record
            for record in receipt["jobs"]
            if record["name"] == "fleet_readiness"
        )["disposition"]
        == "resubmitted"
    )
    assert (generation_root / chain.ROOT_RELEASE_COMPLETE_NAME).is_file()
    assert (generation_root / chain.LAUNCH_COMPLETE_NAME).is_file()
    assert scheduler.release_calls == [
        ["scontrol", "release", "770000"],
        [
            "scontrol",
            "release",
            str(770000 + len(chain.EXPECTED_JOB_ORDER)),
        ],
    ]


def test_rebound_r1_hashes_cannot_replace_native_protocol_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    manifest_path = (
        paths.recovery_root / chain.R1_EVIDENCE_RELATIVE_PATHS["chain_manifest"]
    )
    before = chain._r1_evidence_contract(paths)
    manifest_path.chmod(0o644)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": chain.R1_CHAIN_PROTOCOL,
                # The hash-only layer has no knowledge of native r1 history.
                "history": "internally-rebound-but-invalid",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    zero_receipt = (
        paths.recovery_root
        / chain.R1_EVIDENCE_RELATIVE_PATHS["zero_result_mutation_receipt"]
    )
    zero_payload = json.loads(zero_receipt.read_text(encoding="utf-8"))
    zero_payload["r1_manifest"]["sha256"] = chain._sha256(manifest_path)
    zero_payload.pop("receipt_id")
    zero_payload["receipt_id"] = chain._sha256_bytes(
        chain._canonical_json(zero_payload)
    )
    zero_receipt.chmod(0o644)
    zero_receipt.write_text(
        json.dumps(zero_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zero_receipt.chmod(0o444)
    rebound = chain._r1_evidence_contract(paths)
    assert (
        rebound["chain_manifest"]["sha256"]
        != before["chain_manifest"]["sha256"]
    )
    assert rebound["chain_manifest"]["sha256"] == chain._sha256(manifest_path)

    def actual_protocol_dispatch(received_paths, **_kwargs):
        try:
            report = verify_schema5_recovery_evidence.verify_recovery_evidence(
                received_paths.recovery_root
                / chain.R1_EVIDENCE_RELATIVE_PATHS["chain_manifest"],
                received_paths.recovery_root
                / chain.R1_EVIDENCE_RELATIVE_PATHS["submission_receipt"],
            )
        except (
            verify_schema5_recovery_evidence.EvidenceVerificationError
        ) as exc:
            raise chain.ChainError(
                f"protocol-aware immutable r1 evidence verifier failed: {exc}"
            ) from exc
        report.pop("manifest", None)
        report.pop("submission_receipt", None)
        report["passed"] = True
        return report

    monkeypatch.setattr(
        chain,
        "_invoke_protocol_aware_r1_verifier",
        actual_protocol_dispatch,
    )
    pilot = chain._prerequisite_evidence_contract(
        paths,
        git_identity={
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
        verifier_checkout=paths.repository,
    )
    python, library = chain._verify_pilot_runtime_binding(
        paths,
        pilot["materialization_pilot"]["verifier_runtime"],
    )
    with pytest.raises(
        chain.ChainError,
        match="protocol-aware immutable r1 evidence verifier failed",
    ):
        chain._r1_protocol_verification_contract(
            paths,
            checkout=paths.repository,
            git_identity={
                "release_tag": chain.RELEASE_TAG,
                "git_commit": COMMIT,
                "tag_object": TAG_OBJECT,
            },
            verifier_python=python,
            verifier_library=library,
        )


def test_r2_requires_semantic_r1_idempotency_and_zero_mutation_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    receipt = (
        paths.recovery_root
        / chain.R1_EVIDENCE_RELATIVE_PATHS["zero_result_mutation_receipt"]
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["legacy_result_mutation_count"] = 1
    payload.pop("receipt_id")
    payload["receipt_id"] = chain._sha256_bytes(
        chain._canonical_json(payload)
    )
    receipt.chmod(0o644)
    receipt.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt.chmod(0o444)
    with pytest.raises(
        chain.ChainError,
        match="zero-result-mutation receipt is invalid",
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )


@pytest.mark.parametrize(
    ("artifact", "fault", "message"),
    [
        ("pilot", "missing", "missing or unsafe"),
        ("pilot", "writable", "remains writable"),
        ("canary", "tampered", "prerequisite binding drifted"),
    ],
)
def test_prerequisite_marker_faults_block_verify_and_submit_before_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    fault: str,
    message: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    marker = (
        paths.materialization_pilot_root / chain.MATERIALIZATION_PILOT_MARKER
        if artifact == "pilot"
        else paths.slurm_canary_root / chain.SLURM_CANARY_MARKER
    )
    if fault == "missing":
        marker.unlink()
    elif fault == "writable":
        marker.chmod(0o644)
    else:
        marker.chmod(0o644)
        marker.write_text('{"tampered":true}\n', encoding="utf-8")
        marker.chmod(0o444)

    with pytest.raises(chain.ChainError, match=message):
        chain.verify_chain(paths.chain_manifest)
    scheduler_called = False

    def scheduler(_argv):
        nonlocal scheduler_called
        scheduler_called = True
        return None

    with pytest.raises(chain.ChainError, match=message):
        chain.submit_chain(paths.chain_manifest, apply=True, runner=scheduler)
    assert scheduler_called is False


@pytest.mark.parametrize(
    "drift",
    (
        "pilot_tag",
        "pilot_commit",
        "pilot_schema3",
        "ownership_policy",
        "integrity_policy",
        "canary_commit",
        "canary_script",
        "canary_schema2",
    ),
)
def test_wrong_prerequisite_release_or_code_report_blocks_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    original = chain._invoke_tagged_prerequisite_verifiers

    def drifted_reports(*args, **kwargs):
        reports = json.loads(json.dumps(original(*args, **kwargs)))
        if drift == "pilot_tag":
            reports["materialization_pilot"]["expected_tag"] = "wrong-tag"
        elif drift == "pilot_commit":
            reports["materialization_pilot"]["expected_commit"] = "9" * 40
        elif drift == "pilot_schema3":
            reports["materialization_pilot"]["schema_version"] = 3
        elif drift == "ownership_policy":
            reports["materialization_pilot"][
                "ownership_policy_sha256"
            ] = "9" * 64
        elif drift == "integrity_policy":
            reports["materialization_pilot"][
                "integrity_normalization_policy_sha256"
            ] = "9" * 64
        elif drift == "canary_commit":
            reports["slurm_canary"]["code_identity"][
                "release_git_commit"
            ] = "9" * 40
        elif drift == "canary_script":
            reports["slurm_canary"]["code_identity"]["canary_script"][
                "sha256"
            ] = "9" * 64
        else:
            reports["slurm_canary"]["schema_version"] = 2
            reports["slurm_canary"][
                "kind"
            ] = "schema5_slurm_fleet_transaction_canary_complete"
        return reports

    monkeypatch.setattr(
        chain, "_invoke_tagged_prerequisite_verifiers", drifted_reports
    )
    with pytest.raises(
        chain.ChainError,
        match="does not belong to the exact r2 code",
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )
    assert not paths.jobs_root.exists()


def test_prerequisite_roots_are_explicit_and_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    wrong = paths.recovery_root / "materialization_pilots" / "wrong-generation"
    wrong.mkdir(parents=True)
    with pytest.raises(
        chain.ChainError, match="materialization pilot root must be the canonical"
    ):
        chain.recovery_paths(
            repository=paths.repository,
            results_root=paths.results_root,
            recovery_root=paths.recovery_root,
            hf_home=paths.hf_home,
            dev_python=paths.dev_python,
            source_harness=paths.source_harness,
            source_serving=paths.source_serving,
            conda_executable=paths.conda_executable,
            materialization_pilot_root=wrong,
            slurm_canary_root=paths.slurm_canary_root,
        )


@pytest.mark.parametrize(
    "git_path",
    (
        "scripts/render_schema5_recovery_chain_v12.py",
        "scripts/schema5_recovery_sentinel.py",
    ),
)
def test_tampered_bundled_bootstrap_tool_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_path: str,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    record = next(
        item
        for item in manifest["recovery_tool_bundle"]
        if item["git_path"] == git_path
    )
    tool = Path(record["path"])
    tool.chmod(0o644)
    tool.write_bytes(tool.read_bytes() + b"# tampered after render\n")
    tool.chmod(0o444)
    with pytest.raises(chain.ChainError, match="tool content drifted"):
        chain.verify_chain(paths.chain_manifest)


def test_spooled_bootstraps_authenticate_exact_tagged_tools_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    payloads, manifest = _render_applied(paths, monkeypatch)
    jobs = {item["name"]: Path(item["script"]) for item in manifest["jobs"]}
    renderer_payload = payloads["scripts/render_schema5_recovery_chain_v12.py"]
    sentinel_payload = payloads["scripts/schema5_recovery_sentinel.py"]
    source = jobs["source_checkout"].read_text(encoding="utf-8")
    sentinel = jobs["failure_sentinel"].read_text(encoding="utf-8")

    assert str(paths.dev_python) not in source
    assert str(paths.sentinel_bootstrap_python) in source
    assert source.find("sentinel bootstrap payload verification failed") < source.find(
        '"$prerequisite_verifier" verify-prerequisites'
    )
    renderer_hash = hashlib.sha256(renderer_payload).hexdigest()
    assert source.count(renderer_hash) == 2
    assert source.count(str(len(renderer_payload))) >= 2
    assert source.find("bundled prerequisite verifier hash drifted") < source.find(
        "--markers-only"
    )
    assert source.find(
        "tagged-checkout prerequisite verifier hash drifted"
    ) < source.find(
        '"$target_prerequisite_verifier"',
        source.find("tagged-checkout prerequisite verifier hash drifted"),
    )
    assert source.count("! -L") >= 2
    assert source.count("8#$tool_mode & 8#222") >= 2
    assert chain.SOURCE_CHECKOUT_SEAL_NAME in source
    assert source.find("verify_full_prerequisites") < source.rfind("seal_target")
    assert "find \"$target\" -depth ! -type l -exec chmod a-w" in source
    assert "git -C \"$sealed_source_checkout\" fsck --full --strict" in source

    for name in (
        "snapshot_adopt_verify",
        "environment_capture",
        "release_materialize",
        "release_freeze",
    ):
        stage = jobs[name].read_text(encoding="utf-8")
        assert chain.SOURCE_CHECKOUT_SEAL_NAME in stage
        assert "sealed source checkout contains writable state" in stage
        assert "diff --no-ext-diff --quiet HEAD --" in stage
        assert "diff --cached --no-ext-diff --quiet" in stage
        assert "fsck --full --strict" in stage

    sentinel_hash = hashlib.sha256(sentinel_payload).hexdigest()
    assert sentinel_hash in sentinel
    assert str(len(sentinel_payload)) in sentinel
    assert sentinel.find("bundled recovery sentinel hash drifted") < sentinel.find(
        'exec env LD_LIBRARY_PATH="$bootstrap_root/lib"'
    )
    assert '[[ -f "${sentinel_tool}" && ! -L "${sentinel_tool}" ]]' in sentinel


def test_live_source_inventory_and_conda_identity_flow_into_production_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _, manifest = _render_applied(paths, monkeypatch)
    jobs = {
        item["name"]: Path(item["script"]).read_text(encoding="utf-8")
        for item in manifest["jobs"]
    }
    pilot = manifest["prerequisite_evidence"]["materialization_pilot"]
    expected = pilot["live_source_inventory_sha256"]
    capture = jobs["environment_capture"]
    materialize = jobs["release_materialize"]

    assert expected == {"harness": "6" * 64, "serving": "7" * 64}
    assert expected["harness"] in capture
    assert expected["serving"] in capture
    assert pilot["ownership_policy_sha256"] in capture
    assert pilot["integrity_normalization_policy_sha256"] in capture
    assert "--ownership-policy" in capture
    assert "--integrity-normalization-policy" in capture
    assert "live {role} prefix differs from sealed pilot" in capture
    assert "production capture used bytes outside sealed pilot" in capture
    assert "production capture used policies outside sealed pilot" in capture
    assert "directory_inventory" in capture
    assert expected["harness"] in materialize
    assert expected["serving"] in materialize
    assert "captured production seeds are not pilot-bound" in materialize
    assert f"captured_harness={paths.captured_harness}" in materialize
    assert '"$captured_harness/bin/python"' in materialize
    assert str(paths.source_harness / "bin/python") not in materialize
    assert "pilot-bound Conda executable hash drifted" in materialize
    assert pilot["conda_executable"]["sha256"] in materialize
    assert str(pilot["conda_executable"]["size"]) in materialize

    # No pre-freeze production allocation executes the mutable developer Python.
    for name in (
        "source_checkout",
        "maintenance_preflight",
        "snapshot_adopt_verify",
        "environment_capture",
        "release_materialize",
        "release_freeze",
    ):
        assert str(paths.dev_python) not in jobs[name]


def test_maintenance_preflight_executes_with_stdlib_only_and_exact_scoping(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    _json(
        recovery / "MAINTENANCE_INTERLOCK.json",
        {
            "desired_state": "maintenance",
            "admission_enabled": False,
            "retired_run_ids": list(chain.LEGACY_RUN_IDS),
        },
    )
    cell = results / chain.LEGACY_RUN_IDS[0] / "cells" / "cell-a"
    cell.mkdir(parents=True)
    lock = cell / ".cell.lock"
    lock.touch()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    squeue = fake_bin / "squeue"
    squeue.write_text(
        "#!/bin/sh\n"
        '[ "$#" -eq 6 ] && [ "$1" = -u ] && [ "$2" = tester ] && '
        '[ "$3" = -h ] && [ "$4" = -r ] && [ "$5" = -o ] && '
        '[ "$6" = "%i|%j|%T|%o" ] || exit 91\n'
        'printf "%s" "${SQUEUE_ROWS:-}"\n',
        encoding="utf-8",
    )
    squeue.chmod(0o755)

    def run(rows: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(results),
                str(recovery),
                "tester",
            ],
            input=chain.MAINTENANCE_PREFLIGHT_PYTHON,
            text=True,
            capture_output=True,
            check=False,
            env={
                "PATH": str(fake_bin),
                "USER": "tester",
                "SQUEUE_ROWS": rows,
            },
        )

    safe = run("999|unrelated-job|RUNNING|python unrelated.py\n")
    assert safe.returncode == 0, safe.stderr
    report = json.loads(safe.stdout)
    assert report["scheduler_rows"] == 1
    assert report["inspected_cell_directories"] == 1
    assert report["held_cell_locks"] == 0

    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = run()
        assert held.returncode != 0
        assert "held legacy cell locks remain" in held.stderr
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    unsafe = run("123|asys-cells-live|RUNNING|worker\n")
    assert unsafe.returncode != 0
    assert "legacy jobs remain live during cleanup" in unsafe.stderr
    assert "openai" not in chain.MAINTENANCE_PREFLIGHT_PYTHON


def test_conda_byte_drift_leaves_sealed_verify_available_but_blocks_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    paths.conda_executable.write_bytes(
        paths.conda_executable.read_bytes() + b"# drift after pilot\n"
    )
    paths.conda_executable.chmod(0o755)

    assert chain.verify_chain(paths.chain_manifest)["passed"] is True
    called = False

    def scheduler(_argv):
        nonlocal called
        called = True
        return None

    with pytest.raises(
        chain.ChainError,
        match=(
            "Conda executable differs from sealed pilot provenance"
            "|Conda runtime toolchain differs from sealed pilot provenance"
        ),
    ):
        chain.submit_chain(paths.chain_manifest, apply=True, runner=scheduler)
    assert called is False


def test_pilot_inventory_hash_detects_post_pilot_source_byte_drift(
    tmp_path: Path,
) -> None:
    from scripts import capture_schema5_environments as capture

    prefix = tmp_path / "live-prefix"
    prefix.mkdir()
    payload = prefix / "runtime.bin"
    payload.write_bytes(b"piloted bytes")
    piloted = capture.directory_inventory(prefix)["inventory_sha256"]
    payload.write_bytes(b"drifted production bytes")
    current = capture.directory_inventory(prefix)["inventory_sha256"]
    assert current != piloted


def test_underlying_conda_module_drift_blocks_submission_with_same_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    entrypoint_sha256 = chain._sha256(paths.conda_executable)
    module = (
        paths.conda_executable.parent.parent
        / "lib"
        / "python3.12"
        / "site-packages"
        / "conda"
        / "__init__.py"
    )
    module.write_text("__version__ = 'drifted-under-entrypoint'\n", encoding="utf-8")
    assert chain._sha256(paths.conda_executable) == entrypoint_sha256
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True

    scheduler_called = False

    def scheduler(_argv):
        nonlocal scheduler_called
        scheduler_called = True
        return None

    with pytest.raises(
        chain.ChainError,
        match="Conda runtime toolchain differs from sealed pilot provenance",
    ):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=scheduler,
        )
    assert scheduler_called is False


def test_sealed_chain_verify_needs_neither_source_checkout_nor_external_conda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _render_applied(paths, monkeypatch)
    repository_gone = tmp_path / "repository.removed"
    conda_gone = tmp_path / "miniforge.removed"
    paths.repository.rename(repository_gone)
    paths.conda_executable.parent.parent.rename(conda_gone)

    report = chain.verify_chain(paths.chain_manifest)
    assert report["passed"] is True
    assert report["chain_id"] == json.loads(
        paths.chain_manifest.read_text(encoding="utf-8")
    )["chain_id"]


def test_renderer_rejects_execution_from_either_live_source_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    _stub_tagged_release(monkeypatch)
    monkeypatch.setattr(chain.sys, "prefix", str(paths.source_harness))
    monkeypatch.setattr(
        chain.sys,
        "executable",
        str(paths.source_harness / "bin" / "python"),
    )
    with pytest.raises(
        chain.ChainError,
        match="sealed pilot harness or another non-live runtime",
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )


def test_submission_argv_uses_manifest_dependency_type() -> None:
    record = {
        "script": "/recovery/20_failure_sentinel.sbatch",
        "dependency_type": "afterany",
    }
    assert chain.submission_argv(
        record,
        dependency_job_ids=["101", "202"],
        comment="asys:s5-recovery-v1.2-r2:abc:g0000:failure_sentinel",
    ) == [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--comment=asys:s5-recovery-v1.2-r2:abc:g0000:failure_sentinel",
        "--dependency=afterany:101:202",
        "/recovery/20_failure_sentinel.sbatch",
    ]


def test_repair_consumes_generation_scoped_sentinel_classification(
    tmp_path: Path,
) -> None:
    recovery = tmp_path / "recovery"
    manifest_path = recovery / chain.CHAIN_MANIFEST_NAME
    receipt_path = recovery / chain.SUBMISSION_RECEIPT_NAME
    manifest = {
        "chain_id": "a" * 64,
        "jobs": [
            {"name": "capture", "dependencies": []},
            {"name": "materialize", "dependencies": ["capture"]},
            {
                "name": "failure_sentinel",
                "dependencies": ["capture", "materialize"],
            },
        ],
    }
    _json(manifest_path, manifest)
    _json(receipt_path, {"receipt_id": "b" * 64})
    root = (
        recovery
        / "recovery_chain_sentinels"
        / chain.CHAIN_NAMESPACE
        / "g0000"
    )
    evidence_path = root / "SCHEDULER_EVIDENCE.json"
    outcome = {
        "classification": "transient_repairable",
        "same_generation_repair_allowed": True,
        "requires_superseding_release": False,
        "transient_roots": ["capture"],
        "dependency_cancelled_suffix": ["materialize"],
    }
    scheduler = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-recovery-scheduler-evidence",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": chain._sha256(manifest_path),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": chain._sha256(receipt_path),
        "outcome": outcome,
    }
    scheduler["evidence_id"] = chain._sha256_bytes(
        chain._canonical_json(scheduler)
    )
    _json(evidence_path, scheduler)
    marker = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-recovery-sentinel-outcome",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": chain._sha256(manifest_path),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": chain._sha256(receipt_path),
        "scheduler_evidence": str(evidence_path),
        "scheduler_evidence_sha256": chain._sha256(evidence_path),
        "scheduler_evidence_id": scheduler["evidence_id"],
        "classification": "transient_repairable",
        "same_generation_repair_allowed": True,
        "requires_superseding_release": False,
    }
    marker["marker_id"] = chain._sha256_bytes(chain._canonical_json(marker))
    _json(root / "RECOVERY_SENTINEL_COMPLETE.json", marker)
    states = {name: {"state": "COMPLETED"} for name in ("capture", "materialize")}
    states["failure_sentinel"] = {"state": "COMPLETED"}
    assert chain._sentinel_repair_jobs(
        manifest=manifest,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        generation=0,
        states=states,
    ) == ["capture", "materialize", "failure_sentinel"]


def test_repair_consumes_only_exact_capacity_transient_root_and_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    support = runpy.run_path(
        str(Path(__file__).with_name("test_schema5_recovery_sentinel.py"))
    )
    verified = support["_capacity_verified"](recovery)
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    manifest_path = Path(verified["manifest_path"])
    receipt_path = Path(verified["submission_receipt_path"])
    fleet_row = next(
        row for row in receipt["jobs"] if row["name"] == "fleet_readiness"
    )
    capacity_root = (
        manifest_path.parent
        / chain.CAPACITY_TRANSIENT_ROOT_NAME
        / chain.CHAIN_NAMESPACE
        / "g0000"
        / fleet_row["job_id"]
    )
    capacity_marker_path = support["_capacity_receipt"](
        recovery,
        verified,
        root=capacity_root,
    )
    scheduler = support["FakeScheduler"](
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
    root = (
        manifest_path.parent
        / "recovery_chain_sentinels"
        / chain.CHAIN_NAMESPACE
        / "g0000"
    )
    report = support["_run"](
        recovery,
        monkeypatch,
        verified,
        scheduler,
        apply=True,
        output=root,
        capacity_transient_receipt=capacity_marker_path,
        mail_runner=lambda argv, _body: subprocess.CompletedProcess(
            list(argv), 0, "", ""
        ),
    )
    assert report["classification"] == "transient_repairable"
    evidence_path = root / "SCHEDULER_EVIDENCE.json"
    sentinel_marker = root / "RECOVERY_SENTINEL_COMPLETE.json"
    by_manifest_name = {
        row["name"]: row for row in manifest["jobs"]
    }
    by_receipt_name = {
        row["name"]: row for row in receipt["jobs"]
    }
    states = {
        "fleet_readiness": {
            "job_id": by_receipt_name["fleet_readiness"]["job_id"],
            "job_name": by_manifest_name["fleet_readiness"]["job_name"],
            "comment": by_receipt_name["fleet_readiness"]["comment"],
            "state": "FAILED",
            "exit_code": "75:0",
            "active": False,
        },
        "second": {
            "job_id": by_receipt_name["second"]["job_id"],
            "job_name": by_manifest_name["second"]["job_name"],
            "comment": by_receipt_name["second"]["comment"],
            "state": "CANCELLED",
            "exit_code": "0:0",
            "active": False,
        },
        "failure_sentinel": {
            "job_id": by_receipt_name["failure_sentinel"]["job_id"],
            "job_name": by_manifest_name["failure_sentinel"]["job_name"],
            "comment": by_receipt_name["failure_sentinel"]["comment"],
            "state": "COMPLETED",
            "exit_code": "0:0",
            "active": False,
        },
    }
    assert chain._sentinel_repair_jobs(
        manifest=manifest,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        generation=0,
        states=states,
    ) == ["fleet_readiness", "second", "failure_sentinel"]
    for state, exit_code in (("COMPLETED", "0:0"), ("FAILED", "1:0")):
        fresh_drift = json.loads(json.dumps(states))
        fresh_drift["fleet_readiness"].update(
            {"state": state, "exit_code": exit_code}
        )
        with pytest.raises(chain.ChainError, match="exact parent"):
            chain._sentinel_repair_jobs(
                manifest=manifest,
                manifest_path=manifest_path,
                receipt_path=receipt_path,
                generation=0,
                states=fresh_drift,
            )

    # Even a fully rehashed, internally self-consistent receipt remains ineligible
    # when the pending cause leaves the two-value capacity-only allowlist.
    support["_rehash_capacity_receipt"](
        capacity_marker_path,
        lambda evidence: evidence["fleet"]["pending"][0].__setitem__(
            "reason", "QOSMaxGRESPerUser"
        ),
    )
    capacity_evidence_path = (
        capacity_marker_path.parent
        / "FLEET_CAPACITY_TRANSIENT_EVIDENCE.json"
    )
    capacity_evidence = json.loads(capacity_evidence_path.read_text())
    capacity_marker = json.loads(capacity_marker_path.read_text())
    sentinel_evidence = json.loads(evidence_path.read_text())
    sentinel_evidence["capacity_transient_receipt"].update(
        {
            "sha256": chain._sha256(capacity_marker_path),
            "receipt_id": capacity_marker["receipt_id"],
            "evidence_sha256": chain._sha256(capacity_evidence_path),
            "evidence_id": capacity_evidence["evidence_id"],
            "preimage_archive": capacity_marker["preimage_archive"],
        }
    )
    sentinel_evidence.pop("evidence_id")
    sentinel_evidence["evidence_id"] = chain._sha256_bytes(
        chain._canonical_json(sentinel_evidence)
    )
    evidence_path.chmod(0o644)
    _json(evidence_path, sentinel_evidence)
    marker = json.loads(sentinel_marker.read_text())
    marker["scheduler_evidence_sha256"] = chain._sha256(evidence_path)
    marker["scheduler_evidence_id"] = sentinel_evidence["evidence_id"]
    marker.pop("marker_id")
    marker["marker_id"] = chain._sha256_bytes(
        chain._canonical_json(marker)
    )
    sentinel_marker.chmod(0o644)
    _json(sentinel_marker, marker)
    with pytest.raises(chain.ChainError, match="capacity-transient repair binding"):
        chain._sentinel_repair_jobs(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            generation=0,
            states=states,
        )


def test_qualification_capacity_transition_authorizes_only_exact_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    protected_path = paths.protected_capacity_marker
    protected_marker = json.loads(
        protected_path.read_text(encoding="utf-8")
    )
    production_jobs: list[dict[str, object]] = []
    for index, name in enumerate(chain.PRODUCTION_STAGE_NAMES):
        production_jobs.append(
            {
                "name": name,
                "dependencies": (
                    []
                    if index == 0
                    else [chain.PRODUCTION_STAGE_NAMES[index - 1]]
                ),
                "dependency_type": "afterok",
                "job_name": f"asys-{name}",
            }
        )
    observer_jobs = [
        {
            "name": f"{chain.STAGE_SENTINEL_PREFIX}{stage}",
            "dependencies": [stage],
            "dependency_type": "afterany",
            "job_name": f"asys-observer-{stage}",
        }
        for stage in chain.PRODUCTION_STAGE_NAMES
    ]
    manifest_jobs = [
        *production_jobs,
        *observer_jobs,
        {
            "name": "failure_sentinel",
            "dependencies": [
                *chain.PRODUCTION_STAGE_NAMES,
                *chain.STAGE_SENTINEL_NAMES,
            ],
            "dependency_type": "afterany",
            "job_name": "asys-failure-sentinel",
        },
    ]
    manifest = {
        "chain_id": "a" * 64,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "results_root": str(paths.results_root),
        "readiness_root": str(paths.readiness),
        "server_pool_root": str(paths.pool),
        "prerequisite_evidence": {
            "protected_capacity": {
                "marker": str(protected_path),
                "marker_sha256": chain._sha256(protected_path),
                "marker_size": protected_path.stat().st_size,
                "protocol": chain.PROTECTED_CAPACITY_PROTOCOL,
                "marker_id": protected_marker["marker_id"],
            }
        },
        "jobs": manifest_jobs,
    }
    _json(paths.chain_manifest, manifest)
    qualification_marker = _publish_throughput_qualification(
        paths,
        manifest,
    )
    pointer_path, pointer, failure = (
        _replace_throughput_success_with_failure(
            paths,
            qualification_marker,
        )
    )
    receipt_jobs = [
        {
            "name": row["name"],
            "job_id": str(10_000 + index),
            "comment": f"comment-{row['name']}",
        }
        for index, row in enumerate(manifest_jobs)
    ]
    receipt_identity = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-recovery-chain-submission",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(paths.chain_manifest),
        "manifest_sha256": chain._sha256(paths.chain_manifest),
        "jobs": receipt_jobs,
    }
    receipt = dict(receipt_identity)
    receipt["receipt_id"] = chain._sha256_bytes(
        chain._canonical_json(receipt_identity)
    )
    receipt_path = (
        paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    )
    _json(receipt_path, receipt)
    receipt_by_name = {
        row["name"]: row for row in receipt_jobs
    }
    scheduler_rows: list[dict[str, object]] = []
    for row in production_jobs:
        name = str(row["name"])
        state = "COMPLETED"
        exit_code = "0:0"
        reason = "None"
        start = "2026-07-23T00:00:00"
        elapsed = "00:00:01"
        if name == "throughput_qualification":
            state = "FAILED"
            exit_code = "76:0"
            reason = "NonZeroExitCode"
            elapsed = "00:10:00"
        elif name in {"controller_drill", "production_resume"}:
            state = "CANCELLED"
            reason = "DependencyNeverSatisfied"
            start = "Unknown"
            elapsed = "00:00:00"
        scheduler_rows.append(
            {
                "name": name,
                "job_id": receipt_by_name[name]["job_id"],
                "dependencies": row["dependencies"],
                "dependency_type": row["dependency_type"],
                "comment": receipt_by_name[name]["comment"],
                "job_name": row["job_name"],
                "state": state,
                "active": False,
                "exit_code": exit_code,
                "reason": reason,
                "start": start,
                "elapsed": elapsed,
            }
        )
    for row in observer_jobs:
        name = str(row["name"])
        scheduler_rows.append(
            {
                "name": name,
                "job_id": receipt_by_name[name]["job_id"],
                "dependencies": row["dependencies"],
                "dependency_type": row["dependency_type"],
                "comment": receipt_by_name[name]["comment"],
                "job_name": row["job_name"],
                "state": "COMPLETED",
                "active": False,
                "exit_code": "0:0",
                "reason": "None",
                "start": "2026-07-23T00:10:01",
                "elapsed": "00:00:01",
            }
        )
    scheduler_rows.append(
        {
            "name": "failure_sentinel",
            "job_id": receipt_by_name["failure_sentinel"]["job_id"],
            "dependencies": [
                *chain.PRODUCTION_STAGE_NAMES,
                *chain.STAGE_SENTINEL_NAMES,
            ],
            "dependency_type": "afterany",
            "comment": receipt_by_name["failure_sentinel"]["comment"],
            "job_name": "asys-failure-sentinel",
            "state": "RUNNING",
            "active": True,
            "exit_code": None,
            "reason": "None",
            "start": "2026-07-23T00:10:02",
            "elapsed": "00:00:01",
        }
    )
    verified = {
        "manifest": manifest,
        "manifest_path": str(paths.chain_manifest),
        "manifest_sha256": chain._sha256(paths.chain_manifest),
        "submission_receipt": receipt,
        "submission_receipt_path": str(receipt_path),
        "submission_receipt_sha256": chain._sha256(receipt_path),
    }
    failure_binding = (
        recovery_sentinel._qualification_capacity_failure_binding(
            verified=verified,
            jobs=scheduler_rows,
        )
    )
    assert failure_binding is not None
    outcome = recovery_sentinel.classify_recovery_jobs(
        scheduler_rows,
        qualification_capacity_failure=failure_binding,
    )
    assert (
        outcome["classification"]
        == "qualification_capacity_transition_required"
    )
    sentinel_root = (
        paths.chain_manifest.parent
        / "recovery_chain_sentinels"
        / chain.CHAIN_NAMESPACE
        / "g0000"
    )
    evidence_path = sentinel_root / "SCHEDULER_EVIDENCE.json"
    evidence_identity = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-recovery-scheduler-evidence",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(paths.chain_manifest),
        "manifest_sha256": chain._sha256(paths.chain_manifest),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": chain._sha256(receipt_path),
        "jobs": scheduler_rows,
        "capacity_transient_receipt": None,
        "qualification_capacity_failure": failure_binding,
        "outcome": outcome,
    }
    evidence = dict(evidence_identity)
    evidence["evidence_id"] = chain._sha256_bytes(
        chain._canonical_json(evidence_identity)
    )
    _json(evidence_path, evidence)
    completion_identity = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-recovery-sentinel-outcome",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(paths.chain_manifest),
        "manifest_sha256": chain._sha256(paths.chain_manifest),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": chain._sha256(receipt_path),
        "scheduler_evidence": str(evidence_path),
        "scheduler_evidence_sha256": chain._sha256(evidence_path),
        "scheduler_evidence_id": evidence["evidence_id"],
        "classification": outcome["classification"],
        "same_generation_repair_allowed": False,
        "requires_superseding_release": False,
    }
    completion = dict(completion_identity)
    completion["marker_id"] = chain._sha256_bytes(
        chain._canonical_json(completion_identity)
    )
    _json(
        sentinel_root / "RECOVERY_SENTINEL_COMPLETE.json",
        completion,
    )
    readiness2_path = (
        paths.readiness / "TRUSTED_GENERATION_2.json"
    ).resolve()
    _json(
        readiness2_path,
        {"schema_version": 1, "catalog_id": "e" * 64},
    )
    readiness2 = {
        **pointer["readiness_generation"],
        "catalog_id": "e" * 64,
        "marker_path": str(readiness2_path),
        "marker_sha256": chain._sha256(readiness2_path),
        "inventory_sha256": "f" * 64,
        "catalog_payload_sha256": "0" * 64,
        "fleet_contract_sha256": "1" * 64,
        "capacity_generation": 2,
        "rollout_generation": 2,
    }
    paused_identity = {
        "immutable_sha256": "2" * 64,
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": 1,
        "production_run_ids": list(chain.PRODUCTION_RUN_IDS),
        "admission": {"current_ceiling": 24},
        "admission_ramp": {"current_ceiling": 24},
        "admission_safety_hold": {"active": False},
    }
    paused = dict(paused_identity)
    paused["guard_sha256"] = chain._sha256_bytes(
        chain._canonical_json(paused_identity)
    )
    transition_identity = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r2-throughput-qualification-"
            "capacity-transition-v1"
        ),
        "passed": True,
        "release_id": chain.RELEASE_ID,
        "release_tag": chain.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": chain.CHAIN_NAMESPACE,
        "chain_id": manifest["chain_id"],
        "manifest": str(paths.chain_manifest),
        "manifest_sha256": chain._sha256(paths.chain_manifest),
        "protected_capacity": {
            "marker": str(protected_path),
            "marker_sha256": chain._sha256(protected_path),
            "marker_id": protected_marker["marker_id"],
        },
        "failed_attempt": qualification_marker["attempt"],
        "failure": {
            "path": str(
                Path(pointer["attempt_root"])
                / chain.THROUGHPUT_QUALIFICATION_FAILURE_NAME
            ),
            "sha256": chain._sha256(
                Path(pointer["attempt_root"])
                / chain.THROUGHPUT_QUALIFICATION_FAILURE_NAME
            ),
            "failure_id": failure["failure_id"],
        },
        "submission_receipt": {
            "path": str(receipt_path),
            "sha256": chain._sha256(receipt_path),
            "receipt_id": receipt["receipt_id"],
        },
        "failed_stage": {
            "name": "throughput_qualification",
            "job_id": receipt_by_name["throughput_qualification"][
                "job_id"
            ],
            "comment": receipt_by_name["throughput_qualification"][
                "comment"
            ],
        },
        "paused_control": paused,
        "additive_transition": {
            "previous_failure_id": failure["failure_id"],
            "serving_profile": "8B",
            "additional_replicas": 1,
            "tensor_parallel_size": 1,
            "from_capacity_generation": 1,
            "to_capacity_generation": 2,
            "from_rollout_generation": 1,
            "to_rollout_generation": 2,
            "from_fleet_contract_sha256": pointer[
                "readiness_generation"
            ]["fleet_contract_sha256"],
            "to_fleet_contract_sha256": readiness2[
                "fleet_contract_sha256"
            ],
            "validation": (
                "schema5_control._assert_additive_capacity_contract+"
                "exact-required-profile-delta"
            ),
        },
        "readiness_generation": readiness2,
        "created_at": chain.datetime.fromtimestamp(
            2_000.0, tz=chain.timezone.utc
        ).isoformat(),
        "created_timestamp": 2_000.0,
    }
    transition_path = (
        paths.throughput_qualification_root
        / "QUALIFICATION_CAPACITY_TRANSITION_COMPLETE.json"
    )
    states = {
        str(row["name"]): {
            "job_id": row["job_id"],
            "job_name": row["job_name"],
            "comment": row["comment"],
            "state": (
                "COMPLETED"
                if row["name"] == "failure_sentinel"
                else row["state"]
            ),
            "exit_code": (
                "0:0"
                if row["name"] == "failure_sentinel"
                else row["exit_code"]
            ),
            "active": False,
        }
        for row in scheduler_rows
    }
    with pytest.raises(chain.ChainError):
        chain._sentinel_repair_jobs(
            manifest=manifest,
            manifest_path=paths.chain_manifest,
            receipt_path=receipt_path,
            generation=0,
            states=states,
        )
    transition = _identified_json(
        transition_path,
        transition_identity,
        identity_field="transition_id",
    )
    assert chain._sentinel_repair_jobs(
        manifest=manifest,
        manifest_path=paths.chain_manifest,
        receipt_path=receipt_path,
        generation=0,
        states=states,
    ) == [
        "fleet_readiness",
        "smoke_readiness",
        "throughput_qualification",
        "controller_drill",
        "production_resume",
        f"{chain.STAGE_SENTINEL_PREFIX}fleet_readiness",
        f"{chain.STAGE_SENTINEL_PREFIX}smoke_readiness",
        f"{chain.STAGE_SENTINEL_PREFIX}throughput_qualification",
        f"{chain.STAGE_SENTINEL_PREFIX}controller_drill",
        f"{chain.STAGE_SENTINEL_PREFIX}production_resume",
        "failure_sentinel",
    ]

    for field, replacement in (
        ("receipt_id", "f" * 64),
        ("serving_profile", "32B-long"),
    ):
        drifted = json.loads(json.dumps(transition))
        drifted.pop("transition_id")
        if field == "receipt_id":
            drifted["submission_receipt"][field] = replacement
        else:
            drifted["additive_transition"][field] = replacement
        drifted["transition_id"] = chain._sha256_bytes(
            chain._canonical_json(drifted)
        )
        transition_path.chmod(0o644)
        _json(transition_path, drifted)
    with pytest.raises(
        chain.ChainError,
        match="qualification capacity-transition repair binding",
    ):
        chain._sentinel_repair_jobs(
            manifest=manifest,
            manifest_path=paths.chain_manifest,
            receipt_path=receipt_path,
            generation=0,
            states=states,
        )
    transition_path.chmod(0o644)
    _json(transition_path, transition)
    fresh_drift = json.loads(json.dumps(states))
    fresh_drift["throughput_qualification"]["exit_code"] = "2:0"
    with pytest.raises(chain.ChainError, match="exact parent stage-18"):
        chain._sentinel_repair_jobs(
            manifest=manifest,
            manifest_path=paths.chain_manifest,
            receipt_path=receipt_path,
            generation=0,
            states=fresh_drift,
        )


def test_renderer_smoke_verifier_selects_latest_sealed_generation_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    readiness = results / "recovery" / "schema5-v1" / "readiness"
    state = results / ".dispatcher-schema5-v1"
    chain_manifest = results / "recovery" / "schema5-v1" / "CHAIN.json"
    protected = results / "recovery" / "schema5-v1" / "PROTECTED.json"
    _json(protected, {"marker_id": "1" * 64})
    manifest = {
        "state_root": str(state),
        "results_root": str(results),
        "readiness_root": str(readiness),
        "prerequisite_evidence": {
            "protected_capacity": {
                "marker": str(protected),
                "marker_sha256": chain._sha256(protected),
                "marker_size": protected.stat().st_size,
                "protocol": chain.PROTECTED_CAPACITY_PROTOCOL,
                "marker_id": "1" * 64,
            }
        },
    }
    _json(chain_manifest, manifest)
    monkeypatch.setattr(
        chain,
        "verify_chain",
        lambda path: {"passed": True, "manifest": str(path)},
    )
    monkeypatch.setattr(
        smoke_runner,
        "initialize_all",
        lambda *, results_root, **_kwargs: (
            results_root.mkdir(parents=True, exist_ok=True)
            or {"status": "initialized", "total_cells": 41}
        ),
    )
    immutable = {
        "release_id": chain.RELEASE_ID,
        "git_commit": COMMIT,
        "source_tree_sha256": "2" * 64,
        "harness_environment_sha256": "3" * 64,
        "serving_environment_sha256": "4" * 64,
        "model_contract_sha256": "5" * 64,
        "fleet_contract_path": "/sealed/fleet.json",
        "fleet_contract_sha256": "6" * 64,
        "release_fleet_contract_sha256": "7" * 64,
        "capacity_generation": 2,
        "protected_capacity_marker_path": str(protected),
        "protected_capacity_marker_sha256": chain._sha256(protected),
        "protected_capacity_marker_id": "1" * 64,
    }
    immutable_sha = "8" * 64
    generation = {
        "catalog_id": "9" * 64,
        "marker_path": "/sealed/TRUSTED_GENERATION.json",
        "marker_sha256": "a" * 64,
        "inventory_sha256": "b" * 64,
        "catalog_payload_sha256": "c" * 64,
        "allowed_generation_tuple_count": 1,
        "release_fleet_contract_sha256": "7" * 64,
        "fleet_contract_sha256": "6" * 64,
        "capacity_generation": 2,
        "rollout_generation": 3,
    }
    trusted_catalog = chain.TrustedGenerationCatalog(
        marker_path=Path(generation["marker_path"]),
        marker_sha256=generation["marker_sha256"],
        inventory_sha256=generation["inventory_sha256"],
        catalog_sha256=generation["catalog_payload_sha256"],
        catalog_id=generation["catalog_id"],
        entries=(),
        generations=(),
        allowed_generation_tuples=frozenset(
            {
                (
                    generation["release_fleet_contract_sha256"],
                    generation["fleet_contract_sha256"],
                    generation["capacity_generation"],
                    generation["rollout_generation"],
                    "endpoint-generation",
                )
            }
        ),
    )
    monkeypatch.setattr(
        chain,
        "validate_trusted_generation_catalog",
        lambda *_args, **_kwargs: trusted_catalog,
    )
    monkeypatch.setattr(
        chain, "_verify_smoke_result_catalog", lambda **_kwargs: None
    )
    base = readiness / smoke_runner.SMOKE_ATTEMPT_BASE_NAME
    pointer_path, pointer = smoke_runner._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke_runner.__file__).resolve().parents[1],
        immutable_sha256=immutable_sha,
        immutable=immutable,
        readiness_generation=generation,
        now=1_000.0,
    )
    suite_cells = (
        ("schema5_smoke_32b_long_v1", "long_32b_smoke", 15),
        (
            "schema5_smoke_selective_long_v1",
            "selective_long_smoke",
            20,
        ),
        (
            "schema5_smoke_standard_canaries_v1",
            "standard_canary_smoke",
            6,
        ),
    )
    suites = [
        {
            "schema_version": 1,
            "kind": kind,
            "passed": True,
            "immutable_sha256": immutable_sha,
            "run_id": run_id,
            "expected_cells": cells,
            "schema5_complete_cells": cells,
            "context_incidents": 0,
            "protocol_incidents": 0,
            "transport_incidents": 0,
            "truncation_incidents": 0,
            "provenance_failures": 0,
            "trusted_generation_failures": 0,
        }
        for run_id, kind, cells in suite_cells
    ]
    smoke_runner._write_smoke_evidence(
        state_root=state,
        immutable_sha256=immutable_sha,
        suites=suites,
        output_root=Path(pointer["attempt_root"]),
    )
    provenance = smoke_runner._attempt_provenance(
        immutable_sha256=immutable_sha,
        immutable=immutable,
        readiness_generation=generation,
        rollout_generation=3,
    )
    selector = smoke_runner._publish_success(
        base=base,
        pointer_path=pointer_path,
        pointer=pointer,
        provenance=provenance,
    )
    _json(
        state / "control.json",
        {
            "desired_state": "paused",
            "rollout_generation": 2,
            "immutable_sha256": immutable_sha,
            "immutable": {
                "release_id": chain.RELEASE_ID,
                "git_commit": COMMIT,
                "server_pool_root": str((results / "server-pool").resolve()),
            },
        },
    )

    verified = chain.verify_smoke_attempt(chain_manifest)

    assert verified == {
        "status": "verified",
        "passed": True,
        "attempt_id": pointer["attempt_id"],
        "pointer_id": pointer["pointer_id"],
        "selector_id": selector["selector_id"],
        "completion_id": selector["completion_id"],
        "evidence": selector["evidence"],
        "evidence_sha256": selector["evidence_sha256"],
        "capacity_generation": 2,
        "rollout_generation": 3,
        "trusted_catalog_id": "9" * 64,
    }
    selected_artifact = (
        Path(pointer["attempt_root"])
        / "artifacts"
        / "long_32b_smoke.json"
    )
    original_artifact = selected_artifact.read_bytes()
    selected_artifact.chmod(0o644)
    selected_artifact.write_bytes(original_artifact + b"\n")
    selected_artifact.chmod(0o444)
    with pytest.raises(
        chain.ChainError,
        match="binding is invalid|inventory",
    ):
        chain.verify_smoke_attempt(chain_manifest)
    selected_artifact.chmod(0o644)
    selected_artifact.write_bytes(original_artifact)
    selected_artifact.chmod(0o444)
    current = base / smoke_runner.CURRENT_SELECTOR_NAME
    current.chmod(0o644)
    with pytest.raises(chain.ChainError, match="current smoke selector is not sealed"):
        chain.verify_smoke_attempt(chain_manifest)


def test_renderer_smoke_result_verifier_rejects_out_of_catalog_joint_tuple(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    attempt_id = "a000001-g000001-c000001-" + "9" * 16
    trusted_tuple = ("7" * 64, "6" * 64, 1, 1, "trusted-endpoint")
    historical_tuple = (
        "7" * 64,
        "d" * 64,
        2,
        2,
        "historical-endpoint",
    )
    catalog = chain.TrustedGenerationCatalog(
        marker_path=tmp_path / "catalog" / "TRUSTED.json",
        marker_sha256="a" * 64,
        inventory_sha256="b" * 64,
        catalog_sha256="c" * 64,
        catalog_id="9" * 64,
        entries=(),
        generations=(),
        allowed_generation_tuples=frozenset(
            {trusted_tuple, historical_tuple}
        ),
    )
    suites = {
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
    suite_artifacts = {}
    first_result = None
    first_cell = None
    for name, (run_id, count) in suites.items():
        cells = []
        for index in range(count):
            cell_id = f"{run_id}-{index:03d}"
            result = (
                results
                / chain.SMOKE_ATTEMPT_RUNS_NAME
                / attempt_id
                / run_id
                / "cells"
                / cell_id
                / "results.jsonl"
            )
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(
                json.dumps(
                    {
                        "coordinate_provenance_identity_counts": [
                            {
                                "release_fleet_contract_sha256": trusted_tuple[0],
                                "fleet_contract_sha256": trusted_tuple[1],
                                "capacity_generation": trusted_tuple[2],
                                "rollout_generation": trusted_tuple[3],
                                "endpoint_generation": trusted_tuple[4],
                                "count": 1,
                            }
                        ]
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result.chmod(0o444)
            cell = {
                "cell_id": cell_id,
                "status": "complete",
                "valid_qids": 1,
                "expected_qids": 1,
                "trusted_generation_errors": [],
                "results_artifact": {
                    "path": str(result.resolve()),
                    "sha256": chain._sha256(result),
                    "row_count": 1,
                },
            }
            cells.append(cell)
            if first_result is None:
                first_result = result
                first_cell = cell
        suite = {
            "run_id": run_id,
            "expected_cells": count,
            "schema5_complete_cells": count,
            "trusted_generation_failures": 0,
            "suite_identity": {
                "smoke_attempt": {
                    "protocol": chain.SMOKE_ATTEMPT_BINDING_PROTOCOL,
                    "attempt_id": attempt_id,
                    "attempt_ordinal": 1,
                    "immutable_sha256": "8" * 64,
                    "capacity_generation": trusted_tuple[2],
                    "rollout_generation": trusted_tuple[3],
                    "fleet_contract_sha256": trusted_tuple[1],
                    "release_fleet_contract_sha256": trusted_tuple[0],
                    "trusted_catalog_id": catalog.catalog_id,
                }
            },
            "provenance": {
                "capacity_generation": trusted_tuple[2],
                "rollout_generation": trusted_tuple[3],
                "fleet_contract_sha256": trusted_tuple[1],
                "release_fleet_contract_sha256": trusted_tuple[0],
                "trusted_generation_catalog": chain._smoke_catalog_binding(
                    catalog
                )
            },
            "cells": cells,
        }
        artifact_path = (
            tmp_path
            / "readiness"
            / chain.SMOKE_ATTEMPT_BASE_NAME
            / "attempts"
            / attempt_id
            / "artifacts"
            / f"{name}.json"
        )
        suite_artifacts[name] = (artifact_path, suite)

    chain._verify_smoke_result_catalog(
        suite_artifacts=suite_artifacts,
        results_root=results,
        attempt_id=attempt_id,
        catalog=catalog,
    )
    assert first_result is not None and first_cell is not None
    row = json.loads(first_result.read_text(encoding="utf-8"))
    identity = row["coordinate_provenance_identity_counts"][0]
    (
        identity["release_fleet_contract_sha256"],
        identity["fleet_contract_sha256"],
        identity["capacity_generation"],
        identity["rollout_generation"],
        identity["endpoint_generation"],
    ) = historical_tuple
    first_result.chmod(0o644)
    first_result.write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    first_result.chmod(0o444)
    first_cell["results_artifact"]["sha256"] = chain._sha256(first_result)
    with pytest.raises(chain.ChainError, match="out-of-catalog endpoint provenance"):
        chain._verify_smoke_result_catalog(
            suite_artifacts=suite_artifacts,
            results_root=results,
            attempt_id=attempt_id,
            catalog=catalog,
        )


def test_partial_materialization_quarantine_is_marker_first_sealed_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    release = recovery / "releases" / chain.RELEASE_ID
    nested = release / "environments" / "harness"
    nested.mkdir(parents=True)
    payload = nested / "runtime.bin"
    payload.write_bytes(b"immutable runtime bytes")
    payload.chmod(0o664)
    nested.chmod(0o775)
    release.chmod(0o775)

    manifest_path = recovery / chain.CHAIN_MANIFEST_NAME
    _json(
        manifest_path,
        {
            "chain_id": "a" * 64,
            "release_root": str(release),
        },
    )
    receipt_path = recovery / chain.SUBMISSION_RECEIPT_NAME
    _json(receipt_path, {"receipt_id": "b" * 64})
    receipt = {"jobs": []}
    states = {
        "release_materialize": {
            "active": False,
            "job_id": "24680",
            "state": "FAILED",
        }
    }
    monkeypatch.setattr(chain, "verify_chain", lambda _path: {"passed": True})
    monkeypatch.setattr(
        chain,
        "_latest_chain_receipt",
        lambda **_kwargs: (receipt, receipt_path, 0, None),
    )
    monkeypatch.setattr(
        chain,
        "_query_receipt_job_states",
        lambda **_kwargs: states,
    )

    evidence = recovery / chain.QUARANTINE_EVIDENCE_ROOT_NAME
    quarantine_intent = evidence / "partial-job-24680.intent.json"
    seal_intent = evidence / "partial-job-24680.seal-intent.json"
    destination = (
        recovery
        / "releases"
        / chain.QUARANTINE_ROOT_NAME
        / f"{chain.RELEASE_ID}.partial-job-24680"
    )
    events: list[str] = []
    real_rename = os.rename

    def guarded_rename(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(source) == release:
            assert quarantine_intent.is_file()
            assert quarantine_intent.stat().st_mode & 0o222 == 0
            events.append("quarantine-intent-before-rename")
        real_rename(source, target)

    real_remove_write_bits = seal_recovery_evidence._remove_write_bits

    def guarded_remove_write_bits(root: Path) -> None:
        assert root == destination
        assert seal_intent.is_file()
        assert seal_intent.stat().st_mode & 0o222 == 0
        assert (root / "environments" / "harness" / "runtime.bin").stat().st_mode & 0o222
        events.append("seal-intent-before-read-only")
        real_remove_write_bits(root)

    monkeypatch.setattr(chain.os, "rename", guarded_rename)
    monkeypatch.setattr(
        seal_recovery_evidence,
        "_remove_write_bits",
        guarded_remove_write_bits,
    )

    first = chain.quarantine_partial_materialization(
        manifest_path,
        apply=True,
    )
    assert first["status"] == "quarantined_and_sealed"
    assert first["quarantine_seal"]["status"] == "sealed"
    assert events == [
        "quarantine-intent-before-rename",
        "seal-intent-before-read-only",
    ]
    assert not release.exists()
    assert destination.is_dir()
    for entry in (destination, *destination.rglob("*")):
        if not entry.is_symlink():
            assert entry.stat().st_mode & 0o222 == 0
    seal_marker = evidence / "partial-job-24680.sealed.json"
    seal_inventory = evidence / "partial-job-24680.sealed-inventory.txt"
    assert seal_marker.is_file()
    assert seal_marker.stat().st_mode & 0o222 == 0
    assert "environments/harness/runtime.bin" in seal_inventory.read_text(
        encoding="utf-8"
    )
    stable_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            quarantine_intent,
            seal_intent,
            seal_inventory,
            seal_marker,
        )
    }

    second = chain.quarantine_partial_materialization(
        manifest_path,
        apply=True,
    )
    assert second["status"] == "already_quarantined_and_sealed"
    assert second["quarantine_seal"]["status"] == "already_sealed"
    assert events == [
        "quarantine-intent-before-rename",
        "seal-intent-before-read-only",
    ]
    assert stable_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            quarantine_intent,
            seal_intent,
            seal_inventory,
            seal_marker,
        )
    }


def test_renderer_refuses_stale_canonical_server_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    paths.pool.mkdir(parents=True)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: {
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
    )
    monkeypatch.setattr(
        chain,
        "_tagged_file_bytes",
        lambda _repository, _commit, _relative: b"#!/usr/bin/env python3\n",
    )

    with pytest.raises(
        chain.ChainError,
        match="schema-5 canonical server pool must be fresh and absent",
    ):
        chain.render_chain(
            paths,
            partition="mit_normal",
            slurm_user="tester",
            apply=False,
        )


def test_failure_sentinel_uses_sealed_real_copy_bootstrap_not_live_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: {
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
    )
    monkeypatch.setattr(
        chain,
        "_tagged_file_bytes",
        lambda _repository, _commit, _relative: b"#!/usr/bin/env python3\n",
    )
    expected_sha256 = hashlib.sha256(paths.dev_python.read_bytes()).hexdigest()

    chain.render_chain(
        paths,
        partition="mit_normal",
        slurm_user="tester",
        apply=True,
    )
    manifest = json.loads(paths.chain_manifest.read_text(encoding="utf-8"))
    assert manifest["dev_python_sha256"] == expected_sha256
    assert (
        manifest["sentinel_bootstrap"]["protocol"]
        == chain.SENTINEL_BOOTSTRAP_PROTOCOL
    )
    assert paths.sentinel_bootstrap_marker.is_file()
    assert paths.sentinel_bootstrap_inventory.is_file()
    assert paths.sentinel_bootstrap_python.read_bytes() == paths.dev_python.read_bytes()
    assert not paths.sentinel_bootstrap_python.samefile(paths.dev_python)
    assert not any(
        path.stat().st_mode & 0o222
        for path in paths.sentinel_bootstrap_root.rglob("*")
    )
    sentinel_record = next(
        row for row in manifest["jobs"] if row["name"] == "failure_sentinel"
    )
    sentinel_path = Path(sentinel_record["script"])
    sentinel = sentinel_path.read_text(encoding="utf-8")
    assert str(paths.sentinel_bootstrap_python) in sentinel
    assert str(paths.dev_python) not in sentinel
    assert "sha256sum --check --strict" in sentinel
    assert "find \"$bootstrap_root\" -type f ! -links 1" in sentinel
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True

    # The sentinel and native chain verifier remain usable after the mutable source
    # interpreter changes; the sealed bootstrap is the runtime trust anchor.
    paths.dev_python.write_text(
        "#!/bin/sh\n# drift after immutable rendering\nexit 0\n",
        encoding="utf-8",
    )
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True
    paths.source_harness.rename(tmp_path / "retired-source-harness")
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True

    paths.sentinel_bootstrap_python.chmod(0o750)
    paths.sentinel_bootstrap_python.write_text(
        "#!/bin/sh\nexit 7\n", encoding="utf-8"
    )
    paths.sentinel_bootstrap_python.chmod(0o550)
    with pytest.raises(chain.ChainError, match="payload hash drifted"):
        chain.verify_chain(paths.chain_manifest)


def test_every_production_stage_is_fenced_by_marker_last_launch_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: {
            "release_tag": chain.RELEASE_TAG,
            "git_commit": COMMIT,
            "tag_object": TAG_OBJECT,
        },
    )
    monkeypatch.setattr(
        chain,
        "_tagged_file_bytes",
        lambda _repository, _commit, _relative: b"#!/usr/bin/env python3\n",
    )

    chain.render_chain(
        paths,
        partition="mit_normal",
        slurm_user="tester",
        apply=True,
    )
    manifest = json.loads(paths.chain_manifest.read_text(encoding="utf-8"))
    token = (
        "No recovery-stage mutation may occur above this "
        "launch-authorization gate."
    )
    specs = chain.job_specs(
        paths,
        commit=COMMIT,
        tag_object=TAG_OBJECT,
        slurm_user="tester",
        bundled_tools={
            Path(git_path).name: b"#!/usr/bin/env python3\n"
            for git_path in chain.BUNDLED_TOOL_GIT_PATHS
        },
        prerequisite_evidence=manifest["prerequisite_evidence"],
        r1_protocol_verification=manifest["r1_protocol_verification"],
    )
    bodies = {spec.name: spec.body.rstrip() for spec in specs}
    for record in manifest["jobs"]:
        text = Path(record["script"]).read_text(encoding="utf-8")
        if (
            record["name"] == "failure_sentinel"
            or record["name"].startswith(chain.STAGE_SENTINEL_PREFIX)
        ):
            # Observers must still run and alert if launch publication or a stage
            # launch gate itself fails.
            assert token not in text
            continue
        assert token in text
        assert f"SECONDS + {chain.LAUNCH_GATE_TIMEOUT_SECONDS}" in text
        assert chain.SUBMISSION_RECEIPT_NAME in text
        assert chain.ROOT_RELEASE_COMPLETE_NAME in text
        assert chain.LAUNCH_COMPLETE_NAME in text
        assert "current allocation is not the receipt-bound stage" in text
        assert "root-release dependency policy is not fail closed" in text
        assert chain.PROTECTED_CAPACITY_MARKER_NAME in text
        assert "protected capacity does not prove full non-preemptible placement" in text
        assert '"cell_ceiling": 384' in text
        assert '"submit_headroom": 448' in text
        assert '"memory_mib": 1572864' in text
        assert text.index(token) < text.index(bodies[record["name"]])
    controller_drill = (
        paths.jobs_root / "19_controller_drill.sbatch"
    ).read_text(encoding="utf-8")
    production_resume = (
        paths.jobs_root / "20_production_resume.sbatch"
    ).read_text(encoding="utf-8")
    assert chain.WATCHDOG_READY_MARKER_NAME in controller_drill
    assert chain.EXTERNAL_WATCHDOG_DRILL_MARKER_NAME in controller_drill
    assert "verify-watchdog-readiness" in controller_drill
    assert "verify-watchdog-readiness" in production_resume
    assert chain.verify_chain(paths.chain_manifest)["passed"] is True


def test_sentinel_bootstrap_dry_run_is_nonmutating_and_capture_is_marker_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    prospective, _, _ = chain._prospective_sentinel_bootstrap(paths)
    assert prospective["source_interpreter_sha256"] == chain._sha256(
        paths.dev_python
    )
    assert not paths.sentinel_bootstrap_root.exists()

    original_copy = chain._copy_bootstrap_file
    calls = 0

    def fail_during_copy(source: Path, destination: Path, *, mode: int) -> None:
        nonlocal calls
        calls += 1
        original_copy(source, destination, mode=mode)
        raise RuntimeError("simulated capture crash")

    monkeypatch.setattr(chain, "_copy_bootstrap_file", fail_during_copy)
    with pytest.raises(RuntimeError, match="simulated capture crash"):
        chain._publish_sentinel_bootstrap(paths)
    assert calls == 1
    assert not paths.sentinel_bootstrap_root.exists()
    assert not paths.sentinel_bootstrap_marker.exists()

    monkeypatch.setattr(chain, "_copy_bootstrap_file", original_copy)
    stale = (
        paths.sentinel_bootstrap_root.parent
        / f".{chain.CHAIN_NAMESPACE}.bootstrap.999.{'a' * 32}"
    )
    (stale / "partial").mkdir(parents=True)
    (stale / "partial" / "payload").write_text("partial", encoding="utf-8")
    (stale / "partial" / "payload").chmod(0o440)
    (stale / "partial").chmod(0o550)
    stale.chmod(0o550)
    marker = chain._publish_sentinel_bootstrap(paths)
    assert not stale.exists()
    assert marker == prospective
    assert paths.sentinel_bootstrap_marker.is_file()
    assert chain._publish_sentinel_bootstrap(paths) == prospective


def test_sentinel_bootstrap_rejects_stdlib_link_escaping_source_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError\n", encoding="utf-8")
    escape = paths.source_harness / "lib" / "python3.11" / "escape.py"
    escape.symlink_to(outside)

    with pytest.raises(chain.ChainError, match="source is unsafe"):
        chain._prospective_sentinel_bootstrap(paths)
    assert not paths.sentinel_bootstrap_root.exists()


def test_email_job_contains_no_token_and_binds_runtime_scheduler_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    body = chain._email_body(paths)
    old_deterministic_token = hashlib.sha256(
        (
            f"{chain.RELEASE_TAG}\0"
            f"{chain._r1_evidence_contract(paths)['failure_envelope']['sha256']}"
            "\0email-ack"
        ).encode("utf-8")
    ).hexdigest()[:24]

    assert old_deterministic_token not in body
    assert "--ack-token" not in body
    assert "--ack-command" not in body
    assert "--token" not in body
    assert "One-time token" not in body
    assert "email_challenges/CURRENT.json" in body
    assert 'scontrol show job -o "$SLURM_JOB_ID"' in body
    assert 'squeue -h -j "$SLURM_JOB_ID" -o \'%k\'' in body
    assert ":g([0-9]{4}):email_readiness$" in body
    assert '--chain-id "$chain_id"' in body
    assert '--challenge-generation "$challenge_generation"' in body
    assert '--release-git-commit "$release_git_commit"' in body
