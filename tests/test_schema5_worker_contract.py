"""Focused schema-5 worker provenance, endpoint, signal, and Slurm contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import completion, runner
from agents_scaling.experiment.artifact_policy import (
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.qid_checkpoint import CoordinateAdmissionClosed
from agents_scaling.experiment.runner import WorkerDrainController
from agents_scaling.serving.model_contracts import load_model_contracts
from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving.launch_server import render_sbatch
from agents_scaling.serving.registry import (
    ServerEntry,
    entry_matches_frozen_provenance,
    server_pool_id,
)
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
        release_id="sweep-recovery-schema5-v1.1",
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
        locks = {"conda_explicit": [], "pip_freeze_all": []}
        payload = {
            "schema_version": 1,
            "release_id": "sweep-recovery-schema5-v1.1",
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
            "directory_inventory": inventory,
        }
        payload["environment_content_sha256"] = hashlib.sha256(
            runtime_integrity.canonical_bytes(
                {
                    "runtime": payload["runtime"],
                    "locks": locks,
                    "release_package": None,
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
        release_id="sweep-recovery-schema5-v1.1",
        release_bundle_id="8" * 64,
        immutable_pins_sha256=immutable_sha,
        environment_pins=environment_pins,
    )
    lease = runtime_integrity.refresh_generation_lease(
        state_dir=state_dir,
        attestation_path=Path(attestation["path"]),
        attestation_sha256=attestation["sha256"],
        generation=1,
        release_id="sweep-recovery-schema5-v1.1",
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
    record = {
        "schema_version": 5,
        "serving_profile": "0.6B",
        "effective_context_limit": 32768,
        "tensor_parallel_size": 1,
        "per_agent": [{"endpoint_generation": endpoint}],
        "release_id": pins.release_id,
        "environment_hash": pins.harness_sha256,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "model_contract_sha256": contracts.sha256,
        "endpoint_generation": endpoint,
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
        "artifact_policy_sha256": policy.file_sha256,
        "serving_profile": "0.6B",
        "endpoint_generation": endpoint,
        "effective_context": 32768,
        "rollout_generation": 3,
        "git_commit": pins.git_commit,
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
        release_id="sweep-recovery-schema5-v1.1",
        serving_environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=fleet_hash,
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
    )

    assert not runner._entry_matches_production(
        entry, "0.6B", production, server_root=canonical_pool
    )
    entry.server_pool_id = server_pool_id(canonical_pool)
    entry.replica_id = "schema5-v1--0p6b--standard--r00"
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


def test_server_launch_inherits_frozen_production_pins(tmp_path, monkeypatch):
    contracts = load_model_contracts()
    environments = _frozen_serving_environment(tmp_path)
    monkeypatch.setattr(
        "agents_scaling.serving.launch_server.TEMPLATE",
        tmp_path / "installed-package-without-slurm-template",
    )
    monkeypatch.setenv("ASYS_RELEASE_ID", "sweep-recovery-schema5-v1.1")
    monkeypatch.setenv(
        "ASYS_SERVING_ENVIRONMENT_SHA256", environments["environment_hash"]
    )
    monkeypatch.setenv("ASYS_MODEL_CONTRACT_SHA256", contracts.sha256)

    text = render_sbatch(
        "0.6B",
        "/results/server_pools/schema5-v1",
        "ou_bcs_low",
        "a100",
        "1-00:00:00",
        "/logs",
        **environments,
    )

    assert '--release-id "sweep-recovery-schema5-v1.1"' in text
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
    legacy = (repo / "slurm/run_cell_array.sbatch.tmpl").read_text(encoding="utf-8")
    assert "exec python -m agents_scaling.experiment.run_one" in legacy
