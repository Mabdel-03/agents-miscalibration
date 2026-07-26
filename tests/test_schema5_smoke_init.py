from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.experiment.artifact_policy import (
    POLICY_FILENAME,
    POLICY_CHECKSUM_FILENAME,
    load_artifact_policy,
)
from scripts import init_schema5_smokes as smoke


def test_smoke_policy_payload_is_accepted_by_schema5_loader(tmp_path):
    root = tmp_path / "schema5_smoke_standard_canaries_v1"
    root.mkdir()
    payload = smoke._policy_payload(
        run_id=root.name,
        manifest_sha256="a" * 64,
        benchmark_sha256="b" * 64,
        model_contract_sha256="c" * 64,
        release_id="sweep-recovery-schema5-v1.2",
        git_commit="d" * 40,
        source_tree_sha256="e" * 64,
        harness_environment_sha256="f" * 64,
        serving_environment_sha256="1" * 64,
    )
    encoded = smoke._json_bytes(payload)
    (root / POLICY_FILENAME).write_bytes(encoded)
    (root / POLICY_CHECKSUM_FILENAME).write_bytes(
        smoke._checksum(POLICY_FILENAME, encoded)
    )

    policy = load_artifact_policy(root, required=True)

    assert policy is not None
    assert policy.required_artifact_schema_version == 5
    assert policy.release.release_id == "sweep-recovery-schema5-v1.2"
    assert policy.environment.harness_sha256 == "f" * 64


def test_smoke_dry_run_is_exactly_15_20_6_and_estimand_excluded(tmp_path):
    repo = Path(smoke.__file__).resolve().parent.parent
    report = smoke.initialize_all(
        results_root=tmp_path / "results",
        release_worktree=repo,
        model_contract_path=repo / "configs/model_contracts.v1.json",
        release_id="sweep-recovery-schema5-v1.2",
        git_commit="a" * 40,
        source_tree_sha256="b" * 64,
        harness_environment_sha256="c" * 64,
        serving_environment_sha256="d" * 64,
        apply=False,
    )

    assert report["status"] == "dry_run"
    assert report["total_cells"] == 41
    assert [row["cell_count"] for row in report["runs"]] == [15, 20, 6]
    assert all(row["estimand_excluded"] for row in report["runs"])
    assert report["runs"][0]["serving_profile_counts"] == {"32B-long": 15}
    assert report["runs"][2]["serving_profile_counts"] == {
        "0.6B": 1,
        "1.7B": 1,
        "4B": 1,
        "8B": 1,
        "14B": 1,
        "32B": 1,
    }
    assert not (tmp_path / "results").exists()


def test_smoke_attempt_binding_is_exact_and_rejects_partial_provenance(tmp_path):
    repo = Path(smoke.__file__).resolve().parent.parent
    binding = {
        "protocol": smoke.ATTEMPT_BINDING_PROTOCOL,
        "attempt_id": "a000001-g000001-c000001-" + "a" * 16,
        "attempt_ordinal": 1,
        "immutable_sha256": "b" * 64,
        "capacity_generation": 1,
        "rollout_generation": 1,
        "fleet_contract_sha256": "c" * 64,
        "release_fleet_contract_sha256": "d" * 64,
        "trusted_catalog_id": "e" * 64,
    }
    report = smoke.initialize_all(
        results_root=tmp_path / "attempt-runs",
        release_worktree=repo,
        model_contract_path=repo / "configs/model_contracts.v1.json",
        release_id="sweep-recovery-schema5-v1.2",
        git_commit="a" * 40,
        source_tree_sha256="b" * 64,
        harness_environment_sha256="c" * 64,
        serving_environment_sha256="d" * 64,
        apply=False,
        attempt_binding=binding,
    )
    assert report["total_cells"] == 41
    with pytest.raises(
        smoke.SmokeInitializationError, match="attempt binding is malformed"
    ):
        smoke.initialize_all(
            results_root=tmp_path / "invalid-attempt-runs",
            release_worktree=repo,
            model_contract_path=repo / "configs/model_contracts.v1.json",
            release_id="sweep-recovery-schema5-v1.2",
            git_commit="a" * 40,
            source_tree_sha256="b" * 64,
            harness_environment_sha256="c" * 64,
            serving_environment_sha256="d" * 64,
            apply=False,
            attempt_binding={
                key: value
                for key, value in binding.items()
                if key != "trusted_catalog_id"
            },
        )
