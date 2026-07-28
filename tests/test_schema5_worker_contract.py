"""Focused schema-5 worker provenance, endpoint, signal, and Slurm contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import completion, runner
from agents_scaling.experiment import io
from agents_scaling.experiment.artifact_policy import (
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.qid_checkpoint import CoordinateAdmissionClosed
from agents_scaling.experiment.transport_censor import (
    TRANSPORT_CENSOR_PROTOCOL_HASH,
    TRANSPORT_CENSOR_PROTOCOL_VERSION,
)
from agents_scaling.experiment.runner import WorkerDrainController
from agents_scaling.serving.model_contracts import load_model_contracts
from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving.launch_server import render_sbatch
from agents_scaling.serving.registry import (
    ServerEntry,
    endpoint_instance_id,
    entry_matches_frozen_provenance,
    promoted_entry_path,
    server_pool_id,
)
from agents_scaling.serving import registry as serving_registry
from agents_scaling.serving import client as serving_client
from scripts import clone_schema5_manifests as clone


def _cell() -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": "0.6B",
            "context_share_level": "artifact_only",
            "prompt_complexity_level": 0,
            "reasoning_level": "off",
            "topology": "single_agent",
            "benchmark": "gpqa",
            "n_agents": 1,
            "rounds": 1,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 1,
            "seed": 7,
        }
    )


def _policy_root(tmp_path: Path):
    root = tmp_path / "full_sweep_schema5_v1"
    root.mkdir()
    pins = clone.ReleasePins(
        release_id="sweep-recovery-schema5-v1.2",
        git_commit="a" * 40,
        source_tree_sha256="b" * 64,
        harness_sha256="c" * 64,
        serving_sha256="d" * 64,
    )
    contracts = load_model_contracts()
    payload = clone._policy(
        run_id=root.name,
        manifest_sha256="e" * 64,
        benchmark_sha256="f" * 64,
        model_contracts=contracts,
        pins=pins,
    )
    encoded = clone._json_bytes(payload)
    (root / clone.POLICY_FILENAME).write_bytes(encoded)
    (root / clone.POLICY_CHECKSUM_FILENAME).write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {clone.POLICY_FILENAME}\n",
        encoding="utf-8",
    )
    policy = load_artifact_policy(root, required=True)
    assert policy is not None
    return root, policy, pins, contracts


def _frozen_serving_environment(tmp_path: Path) -> dict[str, str]:
    import time
    from agents_scaling import runtime_integrity

    records = {}
    for role in ("harness", "serving"):
        prefix = tmp_path / f"{role}-env"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        if role == "serving":
            (prefix / "bin" / "vllm").write_text(
                f"#!{prefix / 'bin' / 'python'}\n", encoding="utf-8"
            )
            (prefix / "bin" / "vllm").chmod(0o555)
        (prefix / "bin" / "python").chmod(0o555)
        (prefix / "bin").chmod(0o555)
        prefix.chmod(0o555)
        inventory = runtime_integrity.directory_inventory(prefix)
        locks = {
            "conda_explicit": [
                "@EXPLICIT",
                f"https://repo.example.invalid/{role}-runtime.conda#{'1' * 64}",
            ],
            "pip_freeze_all": [f"{role}-runtime==1.0.0"],
        }
        provenance = {
            "conda_creation_tool": {
                "path": f"/sealed/build-tools/{role}/conda",
                "sha256": "2" * 64,
            },
            "conda_toolchain": {
                "protocol": "schema5-v1.2-r10-offline-conda-toolchain-v1",
                "binding_id": "b" * 64,
            },
            "environment_seed": {
                "capture_id": "3" * 64,
                "capture_marker_sha256": "4" * 64,
                "prefix": f"/retired/build-inputs/{role}-seed",
                "normalized_content_inventory_sha256": "5" * 64,
            },
            "ownership_policy": {
                "path": "/retired/build-inputs/environment_ownership_policy.v1.json",
                "sha256": "6" * 64,
            },
            "integrity_normalization_policy": {
                "path": (
                    "/retired/build-inputs/"
                    "environment_integrity_normalization_policy.v1.json"
                ),
                "sha256": "9" * 64,
            },
            "normalization_receipt": {"id": "7" * 64},
            "conda_package_cache_sha256": "8" * 64,
            "conda_package_cache_seed_sha256": "a" * 64,
        }
        payload = {
            "schema_version": 4,
            "release_id": "sweep-recovery-schema5-v1.2",
            "role": role,
            "prefix": str(prefix.resolve()),
            "sealed_read_only": True,
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "runtime": {
                "python_version": "3.11.15",
                "python_implementation": "CPython",
                "cuda_version": "13.0" if role == "serving" else None,
                "packages": {
                    "torch": "2.11.0" if role == "serving" else None,
                    "vllm": "0.21.0" if role == "serving" else None,
                    "transformers": "5.9.0",
                    "tokenizers": "0.22.2",
                },
            },
            "locks": locks,
            "release_package": None,
            **provenance,
            "installed_files": {
                "inventory_sha256": inventory["inventory_sha256"],
                "entry_count": inventory["entry_count"],
                "file_count": inventory["file_count"],
                "total_file_bytes": inventory["total_file_bytes"],
            },
            "directory_inventory": inventory,
        }
        payload["environment_content_sha256"] = hashlib.sha256(
            runtime_integrity.canonical_bytes(
                {
                    "runtime": payload["runtime"],
                    "locks": locks,
                    "release_package": None,
                    "environment_seed": provenance["environment_seed"],
                    "ownership_policy": provenance["ownership_policy"],
                    "integrity_normalization_policy": provenance[
                        "integrity_normalization_policy"
                    ],
                    "normalization_receipt": provenance["normalization_receipt"],
                    "conda_creation_tool": provenance["conda_creation_tool"],
                    "conda_toolchain": provenance["conda_toolchain"],
                    "conda_package_cache_sha256": provenance[
                        "conda_package_cache_sha256"
                    ],
                    "conda_package_cache_seed_sha256": provenance[
                        "conda_package_cache_seed_sha256"
                    ],
                    "inventory_sha256": inventory["inventory_sha256"],
                }
            )
        ).hexdigest()
        manifest = tmp_path / f"{role}-environment.json"
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        manifest.write_bytes(raw)
        records[role] = (prefix, manifest, hashlib.sha256(raw).hexdigest())
    harness, serving = records["harness"], records["serving"]
    release_worktree = Path(__file__).resolve().parents[1]
    model_contract_path = release_worktree / "configs" / "model_contracts.v1.json"
    fleet_path = release_worktree / "configs" / "schema5_fleet.v1.json"
    environment_pins = {
        role: {
            "prefix": str(records[role][0]),
            "manifest_path": str(records[role][1]),
            "manifest_sha256": records[role][2],
        }
        for role in ("harness", "serving")
    }
    immutable_sha = "9" * 64
    state_dir = tmp_path / "runtime-state"
    state_dir.mkdir()
    attestation = runtime_integrity.ensure_generation_attestation(
        state_dir=state_dir,
        generation=1,
        release_id="sweep-recovery-schema5-v1.2",
        release_bundle_id="8" * 64,
        immutable_pins_sha256=immutable_sha,
        environment_pins=environment_pins,
    )
    lease = runtime_integrity.refresh_generation_lease(
        state_dir=state_dir,
        attestation_path=Path(attestation["path"]),
        attestation_sha256=attestation["sha256"],
        generation=1,
        release_id="sweep-recovery-schema5-v1.2",
        immutable_pins_sha256=immutable_sha,
        expected_environment_hashes={
            role: environment_pins[role]["manifest_sha256"]
            for role in ("harness", "serving")
        },
        expected_prefixes={
            role: environment_pins[role]["prefix"]
            for role in ("harness", "serving")
        },
        now=time.time(),
    )
    return {
        "release_worktree": str(release_worktree),
        "model_contract_path": str(model_contract_path),
        "harness_environment_prefix": str(harness[0]),
        "serving_environment_prefix": str(serving[0]),
        "harness_environment_manifest_path": str(harness[1]),
        "serving_environment_manifest_path": str(serving[1]),
        "harness_environment_hash": harness[2],
        "environment_hash": serving[2],
        "fleet_contract_path": str(fleet_path),
        "fleet_contract_sha256": hashlib.sha256(fleet_path.read_bytes()).hexdigest(),
        "release_fleet_contract_sha256": hashlib.sha256(
            fleet_path.read_bytes()
        ).hexdigest(),
        "capacity_generation": 1,
        "runtime_attestation": attestation["path"],
        "runtime_attestation_sha256": attestation["sha256"],
        "runtime_integrity_lease": lease["path"],
        "immutable_pins_sha256": immutable_sha,
        "rollout_generation": 1,
    }


def test_artifact_policy_loader_matches_clone_contract_and_detects_drift(tmp_path):
    root, policy, pins, contracts = _policy_root(tmp_path)
    assert policy.release.release_id == pins.release_id
    assert policy.environment.harness_sha256 == pins.harness_sha256
    assert policy.accepted_model_contract_sha256 == contracts.sha256
    assert policy.required_artifact_schema_version == 5

    checksum = root / clone.POLICY_CHECKSUM_FILENAME
    checksum.write_text(f"{'0' * 64}  {clone.POLICY_FILENAME}\n", encoding="utf-8")
    with pytest.raises(ArtifactPolicyError, match="checksum mismatch"):
        load_artifact_policy(root, required=True)


def test_schema5_policy_provenance_accepts_exact_payload_and_rejects_drift(
    tmp_path, monkeypatch
):
    _root, policy, pins, contracts = _policy_root(tmp_path)
    cell = _cell()
    identity = contracts.for_size(cell.model_size)
    endpoint = "123@node:8000#100"
    fleet_hash = "1" * 64
    release_fleet_hash = "2" * 64
    coordinate_counts = {
        "capacity_generation": {"1": 1},
        "endpoint_generation": {endpoint: 1},
        "fleet_contract_sha256": {fleet_hash: 1},
        "release_fleet_contract_sha256": {release_fleet_hash: 1},
        "rollout_generation": {"3": 1},
    }
    coordinate_identities = [
        {
            "release_fleet_contract_sha256": release_fleet_hash,
            "fleet_contract_sha256": fleet_hash,
            "capacity_generation": 1,
            "rollout_generation": 3,
            "endpoint_generation": endpoint,
            "count": 1,
        }
    ]
    record = {
        "schema_version": 5,
        "termination_status": "completed",
        "serving_profile": "0.6B",
        "effective_context_limit": 32768,
        "tensor_parallel_size": 1,
        "per_agent": [{"endpoint_generation": endpoint}],
        "efficiency_raw": {"n_turns": 1},
        "self_consistency": {},
        "release_id": pins.release_id,
        "environment_hash": pins.harness_sha256,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "model_contract_sha256": contracts.sha256,
        "fleet_contract_sha256": fleet_hash,
        "release_fleet_contract_sha256": release_fleet_hash,
        "capacity_generation": 1,
        "endpoint_generation": endpoint,
        "coordinate_provenance_counts": coordinate_counts,
        "coordinate_provenance_identity_counts": coordinate_identities,
        "effective_context": 32768,
        "rollout_generation": 3,
    }
    meta = {
        "schema_version": 5,
        "release_id": pins.release_id,
        "environment_hash": pins.harness_sha256,
        "serving_environment_hash": pins.serving_sha256,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "model_contract_sha256": contracts.sha256,
        "fleet_contract_sha256": fleet_hash,
        "release_fleet_contract_sha256": release_fleet_hash,
        "capacity_generation": 1,
        "artifact_policy_sha256": policy.file_sha256,
        "serving_profile": "0.6B",
        "endpoint_generation": endpoint,
        "coordinate_provenance_counts": coordinate_counts,
        "coordinate_provenance_identity_counts": coordinate_identities,
        "effective_context": 32768,
        "rollout_generation": 3,
        "git_commit": pins.git_commit,
        "transport_censor_protocol_version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
        "transport_censor_protocol_hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
    }
    # An installed package may have no package-relative configs.  Explicit validator
    # authority must win even if the process fallback is unusable.
    monkeypatch.setenv("ASYS_RELEASE_WORKTREE", "relative-invalid-release")
    assert completion._artifact_policy_payload_errors(
        cell,
        policy,
        [record],
        meta,
        model_contract_path=contracts.path,
    ) == []

    forged_marginals = json.loads(json.dumps(record))
    forged_marginals["coordinate_provenance_counts"][
        "endpoint_generation"
    ] = {"invented-endpoint": 1}
    errors = completion._artifact_policy_payload_errors(
        cell,
        policy,
        [forged_marginals],
        None,
        model_contract_path=contracts.path,
    )
    assert any(
        "marginals are not the derivation of joint identities" in error
        for error in errors
    ), errors

    assert completion._artifact_policy_payload_errors(
        cell,
        policy,
        [record],
        None,
        model_contract_path=contracts.path,
    ) == []
    wrong_partial = dict(record, release_id="another-release")
    partial_errors = completion._artifact_policy_payload_errors(
        cell,
        policy,
        [wrong_partial],
        None,
        model_contract_path=contracts.path,
    )
    assert any("release_id does not match policy" in error for error in partial_errors)

    wrong_meta = dict(meta, tokenizer_revision="0" * 40)
    errors = completion._artifact_policy_payload_errors(
        cell,
        policy,
        [record],
        wrong_meta,
        model_contract_path=contracts.path,
    )
    assert any("tokenizer_revision" in error for error in errors)

    wrong_record = dict(record, schema_version=4)
    errors = completion._artifact_policy_payload_errors(
        cell,
        policy,
        [wrong_record],
        meta,
        model_contract_path=contracts.path,
    )
    assert any("requires artifact schema 5" in error for error in errors)


def test_schema5_policy_truthfully_accepts_cross_generation_rows_and_mid_qid_counts(
    tmp_path,
):
    _root, policy, pins, contracts = _policy_root(tmp_path)
    cell = _cell()
    identity = contracts.for_size(cell.model_size)
    base_hash = "2" * 64
    g1_hash = "3" * 64
    g2_hash = "4" * 64

    def fixed() -> dict:
        return {
            "schema_version": 5,
            "termination_status": "completed",
            "serving_profile": "0.6B",
            "effective_context_limit": 32768,
            "tensor_parallel_size": 1,
            "self_consistency": {},
            "release_id": pins.release_id,
            "environment_hash": pins.harness_sha256,
            "model_revision": identity.model_revision,
            "tokenizer_revision": identity.tokenizer_revision,
            "model_contract_sha256": contracts.sha256,
            "release_fleet_contract_sha256": base_hash,
            "effective_context": 32768,
        }

    first = {
        **fixed(),
        "per_agent": [{"endpoint_generation": "endpoint-g1"}],
        "efficiency_raw": {"n_turns": 1},
        "fleet_contract_sha256": g1_hash,
        "capacity_generation": 1,
        "rollout_generation": 1,
        "endpoint_generation": "endpoint-g1",
        "coordinate_provenance_counts": {
            "capacity_generation": {"1": 1},
            "endpoint_generation": {"endpoint-g1": 1},
            "fleet_contract_sha256": {g1_hash: 1},
            "release_fleet_contract_sha256": {base_hash: 1},
            "rollout_generation": {"1": 1},
        },
        "coordinate_provenance_identity_counts": [
            {
                "release_fleet_contract_sha256": base_hash,
                "fleet_contract_sha256": g1_hash,
                "capacity_generation": 1,
                "rollout_generation": 1,
                "endpoint_generation": "endpoint-g1",
                "count": 1,
            }
        ],
    }
    resumed_mid_qid = {
        **fixed(),
        # Coordinated rows may retain only terminal-round outputs; the authenticated
        # checkpoint count map still covers the earlier g1 coordinate.
        "per_agent": [{"endpoint_generation": "endpoint-g2"}],
        "efficiency_raw": {"n_turns": 2},
        "fleet_contract_sha256": None,
        "capacity_generation": None,
        "rollout_generation": None,
        "endpoint_generation": "mixed",
        "coordinate_provenance_counts": {
            "capacity_generation": {"1": 1, "2": 1},
            "endpoint_generation": {"endpoint-g1": 1, "endpoint-g2": 1},
            "fleet_contract_sha256": {g1_hash: 1, g2_hash: 1},
            "release_fleet_contract_sha256": {base_hash: 2},
            "rollout_generation": {"1": 1, "2": 1},
        },
        "coordinate_provenance_identity_counts": [
            {
                "release_fleet_contract_sha256": base_hash,
                "fleet_contract_sha256": g1_hash,
                "capacity_generation": 1,
                "rollout_generation": 1,
                "endpoint_generation": "endpoint-g1",
                "count": 1,
            },
            {
                "release_fleet_contract_sha256": base_hash,
                "fleet_contract_sha256": g2_hash,
                "capacity_generation": 2,
                "rollout_generation": 2,
                "endpoint_generation": "endpoint-g2",
                "count": 1,
            },
        ],
    }
    aggregate = completion.aggregate_coordinate_provenance_counts(
        [first, resumed_mid_qid]
    )
    aggregate_identities = (
        completion.aggregate_coordinate_provenance_identity_counts(
            [first, resumed_mid_qid]
        )
    )
    meta = {
        "schema_version": 5,
        "release_id": pins.release_id,
        "environment_hash": pins.harness_sha256,
        "serving_environment_hash": pins.serving_sha256,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "model_contract_sha256": contracts.sha256,
        "fleet_contract_sha256": None,
        "release_fleet_contract_sha256": base_hash,
        "capacity_generation": None,
        "artifact_policy_sha256": policy.file_sha256,
        "serving_profile": "0.6B",
        "endpoint_generation": "mixed",
        "coordinate_provenance_counts": aggregate,
        "coordinate_provenance_identity_counts": aggregate_identities,
        "effective_context": 32768,
        "rollout_generation": None,
        "git_commit": pins.git_commit,
        "transport_censor_protocol_version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
        "transport_censor_protocol_hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
    }
    assert completion._artifact_policy_payload_errors(
        cell,
        policy,
        [first, resumed_mid_qid],
        meta,
        model_contract_path=contracts.path,
    ) == []


def test_endpoint_selection_contract_rejects_any_frozen_identity_drift():
    contracts = load_model_contracts()
    identity = contracts.for_size("0.6B")
    exact = ServerEntry(
        model_size="0.6B",
        hf_id=identity.hf_id,
        host="node",
        port=8000,
        slurm_job_id="123",
        started_at=100.0,
        serving_profile="0.6B",
        served_model_name="0.6B",
        max_model_len=32768,
        tp_size=1,
        release_id="release",
        environment_hash="a" * 64,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256="f" * 64,
    )
    expected = dict(
        release_id="release",
        environment_hash="a" * 64,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256="f" * 64,
    )
    assert entry_matches_frozen_provenance(exact, "0.6B", **expected)
    exact.tokenizer_revision = "0" * 40
    assert not entry_matches_frozen_provenance(exact, "0.6B", **expected)


def test_worker_rejects_forged_capacity_and_release_fleet_flags():
    release_worktree = Path(__file__).resolve().parents[1]
    contracts = load_model_contracts(
        release_worktree / "configs" / "model_contracts.v1.json"
    )
    fleet_path = release_worktree / "configs" / "schema5_fleet.v1.json"
    fleet_sha256 = hashlib.sha256(fleet_path.read_bytes()).hexdigest()

    def runtime(**changes):
        values = {
            "capacity_generation": 1,
            "release_fleet_contract_sha256": fleet_sha256,
            "fleet_contract_sha256": fleet_sha256,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    frozen = runner._validate_runtime_fleet_lineage(
        runtime=runtime(),
        release_worktree=release_worktree,
        model_contracts=contracts,
    )
    assert frozen.sha256 == fleet_sha256
    for forged_capacity in (True, 0, -1, "1"):
        with pytest.raises(
            completion.ExperimentConfigurationError,
            match="capacity_generation",
        ):
            runner._validate_runtime_fleet_lineage(
                runtime=runtime(capacity_generation=forged_capacity),
                release_worktree=release_worktree,
                model_contracts=contracts,
            )
    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="release-fleet pin",
    ):
        runner._validate_runtime_fleet_lineage(
            runtime=runtime(release_fleet_contract_sha256="0" * 64),
            release_worktree=release_worktree,
            model_contracts=contracts,
        )
    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="generation 1",
    ):
        runner._validate_runtime_fleet_lineage(
            runtime=runtime(fleet_contract_sha256="0" * 64),
            release_worktree=release_worktree,
            model_contracts=contracts,
        )


def test_schema5_runner_rejects_an_exact_endpoint_from_another_pool(tmp_path):
    contracts = load_model_contracts()
    identity = contracts.for_size("0.6B")
    canonical_pool = tmp_path / "server_pools" / "schema5-v1"
    other_pool = tmp_path / "server_pools" / "other"
    fleet_hash = hashlib.sha256(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "schema5_fleet.v1.json"
        ).read_bytes()
    ).hexdigest()
    runtime = SimpleNamespace(
        release_id="sweep-recovery-schema5-v1.2",
        serving_environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=fleet_hash,
        release_fleet_contract_sha256=fleet_hash,
        capacity_generation=1,
        rollout_generation=1,
    )
    fleet = load_fleet_contract(
        None,
        model_contracts=contracts,
        expected_sha256=runtime.fleet_contract_sha256,
    )
    production = SimpleNamespace(
        runtime=runtime,
        tokenizer_id=identity.tokenizer_id,
        fleet=fleet,
    )
    from agents_scaling.serving.launch_server import _port_for

    entry = ServerEntry(
        model_size="0.6B",
        hf_id=identity.hf_id,
        host="node",
        port=_port_for("0.6B", 0),
        slurm_job_id="123",
        started_at=100.0,
        serving_profile="0.6B",
        served_model_name="0.6B",
        max_model_len=32768,
        tp_size=1,
        release_id=runtime.release_id,
        environment_hash=runtime.serving_environment_hash,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=runtime.fleet_contract_sha256,
        server_pool_id=server_pool_id(other_pool),
        replica_id=0,
        replica_index=0,
        release_fleet_contract_sha256=runtime.release_fleet_contract_sha256,
        capacity_generation=runtime.capacity_generation,
        rollout_generation=runtime.rollout_generation,
    )

    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    entry.server_pool_id = server_pool_id(canonical_pool)
    entry.replica_id = "schema5-v1--0p6b--standard--r00"
    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    _seal_and_promote_worker_endpoint(canonical_pool, entry)
    assert runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    entry.replica_id = "schema5-v1--0p6b--standard--r01"
    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    entry.replica_id = "schema5-v1--0p6b--standard--r00"
    entry.fleet_contract_sha256 = "0" * 64
    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    entry.fleet_contract_sha256 = runtime.fleet_contract_sha256
    entry.replica_index = None
    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )


def _promoted_pointer_fixture(tmp_path: Path):
    contracts = load_model_contracts()
    identity = contracts.for_size("0.6B")
    pool_root = tmp_path / "server_pools" / "schema5-v1"
    fleet_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "schema5_fleet.v1.json"
    )
    fleet_sha256 = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
    runtime = SimpleNamespace(
        release_id="sweep-recovery-schema5-v1.2",
        serving_environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=fleet_sha256,
        release_fleet_contract_sha256=fleet_sha256,
        capacity_generation=1,
        rollout_generation=1,
    )
    fleet = load_fleet_contract(
        fleet_path,
        model_contracts=contracts,
        expected_sha256=fleet_sha256,
    )
    production = SimpleNamespace(
        runtime=runtime,
        tokenizer_id=identity.tokenizer_id,
        fleet=fleet,
    )
    replica = fleet.for_replica("0.6B", 0)
    from agents_scaling.serving.launch_server import _port_for

    def entry(*, job_id: str, host: str, started_at: float) -> ServerEntry:
        return ServerEntry(
            model_size="0.6B",
            hf_id=identity.hf_id,
            host=host,
            port=_port_for("0.6B", 0),
            slurm_job_id=job_id,
            started_at=started_at,
            serving_profile="0.6B",
            served_model_name="0.6B",
            max_model_len=32768,
            tp_size=1,
            release_id=runtime.release_id,
            environment_hash=runtime.serving_environment_hash,
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
            model_contract_sha256=contracts.sha256,
            fleet_contract_sha256=fleet_sha256,
            server_pool_id="schema5-v1",
            replica_id=replica.replica_id,
            replica_index=replica.replica_index,
            release_fleet_contract_sha256=(
                runtime.release_fleet_contract_sha256
            ),
            capacity_generation=runtime.capacity_generation,
            rollout_generation=runtime.rollout_generation,
        )

    return pool_root, production, entry


def _seal_and_promote_worker_endpoint(
    pool_root: Path,
    entry: ServerEntry,
):
    """Publish the minimum real immutable archive used by worker-routing tests."""

    assert isinstance(entry.replica_id, str)
    assert isinstance(entry.serving_profile, str)
    assert isinstance(entry.slurm_job_id, str)
    token = hashlib.sha256(
        f"{entry.replica_id}:{entry.slurm_job_id}".encode("utf-8")
    ).hexdigest()[:32]
    script_path = (
        pool_root
        / ".fleet-transactions-v1"
        / "sbatch"
        / f"g{int(entry.rollout_generation):06d}"
        / f"{entry.replica_id}.{token}.sbatch"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "#!/bin/bash\n"
        "#SBATCH --no-requeue\n"
        f"# endpoint {entry.replica_id} job {entry.slurm_job_id}\n"
    ).encode("utf-8")
    script_path.write_bytes(script)
    script_path.chmod(0o444)
    serving_registry.write_standby_entry(pool_root, entry)
    script_sha256 = hashlib.sha256(script).hexdigest()
    spooled_provenance = {
        "run_root": str(pool_root.resolve()),
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
    comment = (
        f"asys-s5-fleet:pool={entry.server_pool_id};"
        f"profile={entry.serving_profile};replica={entry.replica_id};"
        f"generation={entry.rollout_generation};intent={token};"
        f"fleet={entry.fleet_contract_sha256}"
    )
    record = serving_registry.seal_endpoint_history(
        pool_root,
        entry,
        release_fleet_contract_sha256=str(
            entry.release_fleet_contract_sha256
        ),
        capacity_generation=int(entry.capacity_generation),
        rollout_generation=int(entry.rollout_generation),
        ledger_generation=int(entry.rollout_generation),
        intent_token=token,
        intent_state="committed",
        committed_at=float(entry.started_at),
        local_script_path=script_path,
        local_script_sha256=script_sha256,
        spooled_script=script,
        spooled_provenance=spooled_provenance,
        scheduler_job_name="asys-s5-serve-worker-test",
        scheduler_comment=comment,
        sealed_at=float(entry.started_at) + 1.0,
    )
    serving_registry.promote_standby_entry(pool_root, entry)
    return record


def test_worker_never_selects_staging_archive_free_or_stale_generation(
    tmp_path,
    monkeypatch,
):
    pool_root, production, make_entry = _promoted_pointer_fixture(tmp_path)
    promoted = make_entry(job_id="101", host="node-live", started_at=1.0)
    _seal_and_promote_worker_endpoint(pool_root, promoted)

    staged = make_entry(job_id="202", host="node-standby", started_at=2.0)
    serving_registry.write_standby_entry(pool_root, staged)
    monkeypatch.setattr(runner, "list_live_servers", lambda *_a, **_k: [staged])
    assert (
        runner._pick_endpoint(
            pool_root,
            "0.6B",
            0,
            exclude=set(),
            production=production,
        )
        is None
    )
    monkeypatch.setattr(
        runner, "list_live_servers", lambda *_a, **_k: [staged, promoted]
    )
    assert runner._pick_endpoint(
        pool_root,
        "0.6B",
        0,
        exclude=set(),
        production=production,
    ) == promoted

    production.runtime.capacity_generation = 2
    assert not runner._entry_matches_production(
        promoted,
        "0.6B",
        production,
        server_root=pool_root,
    )

    incomplete = replace(
        staged,
        slurm_job_id="303",
        release_fleet_contract_sha256=None,
    )
    serving_registry.write_standby_entry(pool_root, incomplete)
    with pytest.raises(ValueError, match="incomplete schema-5 production"):
        serving_registry.promote_standby_entry(pool_root, incomplete)


def test_worker_rejects_symlinked_endpoint_history_ancestor(tmp_path):
    pool_root, production, make_entry = _promoted_pointer_fixture(tmp_path)
    entry = make_entry(job_id="101", host="node-live", started_at=1.0)
    history = _seal_and_promote_worker_endpoint(pool_root, entry)
    profile_directory = history.marker_path.parent.parent.parent
    moved = pool_root / "moved-endpoint-history-profile"
    profile_directory.rename(moved)
    profile_directory.symlink_to(moved, target_is_directory=True)

    assert not runner._entry_matches_production(
        entry,
        "0.6B",
        production,
        server_root=pool_root,
    )


def test_worker_refreshes_fixed_promoted_pointer_and_fails_closed_on_drift(
    tmp_path,
):
    pool_root, production, make_entry = _promoted_pointer_fixture(tmp_path)
    generation_one = make_entry(job_id="101", host="node-g1", started_at=1.0)
    _seal_and_promote_worker_endpoint(pool_root, generation_one)
    pointer = promoted_entry_path(pool_root, "0.6B", generation_one.replica_id)

    class RefreshClient:
        def __init__(self):
            self.base_url = generation_one.base_url
            self.endpoint_generation = endpoint_instance_id(generation_one)
            self.calls: list[tuple[str, str]] = []

        def refresh_endpoint(self, *, base_url, endpoint_generation):
            self.calls.append((base_url, endpoint_generation))
            self.base_url = base_url
            self.endpoint_generation = endpoint_generation
            return True

    client = RefreshClient()
    binding = runner._PromotedEndpointBinding(
        server_root=pool_root,
        profile_name="0.6B",
        replica_id=str(generation_one.replica_id),
        production=production,
        client=client,
        current_entry=generation_one,
    )
    binding.refresh()
    assert client.calls[-1] == (
        generation_one.base_url,
        endpoint_instance_id(generation_one),
    )

    generation_two = make_entry(job_id="202", host="node-g2", started_at=2.0)
    second_history = _seal_and_promote_worker_endpoint(pool_root, generation_two)
    binding.refresh()
    assert client.base_url == generation_two.base_url
    assert client.endpoint_generation == endpoint_instance_id(generation_two)
    assert binding.current_entry == generation_two
    assert not runner._entry_matches_production(
        generation_one,
        "0.6B",
        production,
        server_root=pool_root,
    )

    trusted_state = (client.base_url, client.endpoint_generation, binding.current_entry)
    binding_path = second_history.marker_path.parent / "BINDING.json"
    binding_path.chmod(0o600)
    binding_path.write_bytes(binding_path.read_bytes() + b" ")
    binding_path.chmod(0o400)
    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="does not match frozen provenance",
    ):
        binding.refresh()
    assert (client.base_url, client.endpoint_generation, binding.current_entry) == (
        trusted_state
    )

    untrusted = make_entry(job_id="303", host="node-bad", started_at=3.0)
    untrusted.tokenizer_revision = "0" * 40
    io.write_json(pointer, asdict(untrusted))
    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="does not match frozen provenance",
    ):
        binding.refresh()
    assert (client.base_url, client.endpoint_generation, binding.current_entry) == (
        trusted_state
    )

    pointer.unlink()
    with pytest.raises(
        runner.EndpointPointerUnavailableError,
        match="promoted endpoint pointer is missing",
    ):
        binding.refresh()
    assert (client.base_url, client.endpoint_generation, binding.current_entry) == (
        trusted_state
    )


def test_logprob_client_refresh_publishes_url_and_generation_together(monkeypatch):
    transports: list[SimpleNamespace] = []

    def fake_openai(*, base_url, api_key, timeout, max_retries):
        transport = SimpleNamespace(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        transports.append(transport)
        return transport

    monkeypatch.setattr(serving_client, "OpenAI", fake_openai)
    client = serving_client.LogprobClient(
        base_url="http://node-g1:8000/v1",
        model="0.6B",
        endpoint_generation="endpoint-g1",
    )
    assert client.refresh_endpoint(
        base_url="http://node-g2:8000/v1",
        endpoint_generation="endpoint-g2",
    )
    assert client.base_url == "http://node-g2:8000/v1"
    assert client.endpoint_generation == "endpoint-g2"
    assert client._endpoint_snapshot() == (transports[-1], "endpoint-g2")

    with pytest.raises(
        ValueError,
        match="cannot identify two base URLs",
    ):
        client.refresh_endpoint(
            base_url="http://node-g3:8000/v1",
            endpoint_generation="endpoint-g2",
        )
    assert client._endpoint_snapshot() == (transports[-1], "endpoint-g2")


def test_server_launch_inherits_frozen_production_pins(tmp_path, monkeypatch):
    contracts = load_model_contracts()
    environments = _frozen_serving_environment(tmp_path)
    monkeypatch.setattr(
        "agents_scaling.serving.launch_server.TEMPLATE",
        tmp_path / "installed-package-without-slurm-template",
    )
    monkeypatch.setenv("ASYS_RELEASE_ID", "sweep-recovery-schema5-v1.2")
    monkeypatch.setenv(
        "ASYS_SERVING_ENVIRONMENT_SHA256", environments["environment_hash"]
    )
    monkeypatch.setenv("ASYS_MODEL_CONTRACT_SHA256", contracts.sha256)

    text = render_sbatch(
        "0.6B",
        "/results/server_pools/schema5-v1",
        "ou_bcs_normal",
        "a100",
        "1-00:00:00",
        "/logs",
        **environments,
    )

    assert '--release-id "sweep-recovery-schema5-v1.2"' in text
    assert f'--environment-hash "{environments["environment_hash"]}"' in text
    assert (
        f'export ASYS_HARNESS_ENVIRONMENT_SHA256="'
        f'{environments["harness_environment_hash"]}"' in text
    )
    assert (
        f'export ASYS_SERVING_ENVIRONMENT_SHA256="'
        f'{environments["environment_hash"]}"' in text
    )
    assert f'--model-contract-sha256 "{contracts.sha256}"' in text
    assert (
        f'--release-worktree "{environments["release_worktree"]}"' in text
    )
    assert f'--model-contract "{environments["model_contract_path"]}"' in text
    assert (
        f'export ASYS_RELEASE_WORKTREE="{environments["release_worktree"]}"'
        in text
    )
    assert "#SBATCH --job-name=asys-s5-serve-0p6b-s-r00" in text
    assert '--server-pool-id "schema5-v1"' in text
    assert '--replica-id "schema5-v1--0p6b--standard--r00"' in text
    assert "--replica-index 0" in text
    assert (
        f'--fleet-contract-sha256 "{environments["fleet_contract_sha256"]}"'
        in text
    )
    assert (
        f'--release-fleet-contract-sha256 "'
        f'{environments["release_fleet_contract_sha256"]}"' in text
    )
    assert "--capacity-generation 1" in text
    assert "--rollout-generation 1" in text
    assert "mamba activate" not in text
    assert "source " not in text
    assert (
        f'LD_LIBRARY_PATH="{environments["serving_environment_prefix"]}/lib" \\\n'
        f'"{environments["serving_environment_prefix"]}/bin/vllm" serve'
        in text
    )
    assert (
        f'LD_LIBRARY_PATH="{environments["harness_environment_prefix"]}/lib" \\\n'
        f'"{environments["harness_environment_prefix"]}/bin/python" -I -m'
        in text
    )
    assert "${LD_LIBRARY_PATH" not in text
    assert (
        "unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV "
        "LD_LIBRARY_PATH LD_PRELOAD"
    ) in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text


def test_usr1_controller_closes_admission_without_failure_semantics():
    controller = WorkerDrainController()
    controller.handle_signal(10, None)
    assert controller.requested.is_set()
    with pytest.raises(CoordinateAdmissionClosed, match="graceful drain"):
        controller.admit_coordinate()


def test_usr1_during_runtime_guard_closes_admission_before_coordinate_start():
    controller: WorkerDrainController | None = None

    def guard() -> None:
        assert controller is not None
        controller.handle_signal(runner.signal.SIGUSR1, None)

    controller = WorkerDrainController(admission_guard=guard)
    with pytest.raises(
        CoordinateAdmissionClosed,
        match="graceful drain requested by USR1",
    ):
        controller.admit_coordinate()
    assert controller.requested.is_set()


def test_runner_drain_handoff_never_creates_failure_ledger(tmp_path, monkeypatch):
    cell = _cell()
    run_root = tmp_path / "legacy-compatible-run"
    run_root.mkdir()
    snapshot = SimpleNamespace(
        path=run_root / "cells.json",
        cells=(cell,),
        sha256="e" * 64,
    )
    frozen = SimpleNamespace(sidecar_sha256="f" * 64)
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(runner, "load_manifest", lambda _root: snapshot)
    monkeypatch.setattr(
        runner,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: frozen,
    )
    monkeypatch.setattr(
        runner,
        "_current_server_pool_generation",
        lambda *_args, **_kwargs: "test-pool-generation",
    )

    def drain(**_kwargs):
        raise CoordinateAdmissionClosed("USR1 handoff")

    monkeypatch.setattr(runner, "_run_cell_locked", drain)
    runner.run_cell(cell, run_root.name)

    cdir = run_root / "cells" / cell.cell_id
    assert not (cdir / "failure.json").exists()


def test_cell_sbatch_templates_signal_early_and_disable_requeue():
    repo = Path(__file__).resolve().parents[1]
    for relative in (
        "slurm/run_dispatch_batch.sbatch.tmpl",
        "slurm/run_cell_array.sbatch.tmpl",
    ):
        text = (repo / relative).read_text(encoding="utf-8")
        assert "#SBATCH --signal=B:USR1@1200" in text
        assert "#SBATCH --no-requeue" in text
        if relative == "slurm/run_dispatch_batch.sbatch.tmpl":
            assert "#SBATCH --export=NONE" in text
            assert "#SBATCH --export=ALL" not in text
    legacy = (repo / "slurm/run_cell_array.sbatch.tmpl").read_text(encoding="utf-8")
    assert "exec python -m agents_scaling.experiment.run_one" in legacy
    serving = (repo / "slurm/serve_qwen.sbatch.tmpl").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --mail-user=mabdel03@mit.edu" in serving
    assert "#SBATCH --mail-type=FAIL" in serving
