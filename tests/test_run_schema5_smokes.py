from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_schema5_smokes as smoke
from slurm import dispatch_sweeps
from slurm import schema5_control


IMMUTABLE_SHA = "a" * 64


def _immutable(tmp_path: Path) -> dict:
    return {
        "release_id": "sweep-recovery-schema5-v1",
        "git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "harness_environment_sha256": "3" * 64,
        "serving_environment_sha256": "4" * 64,
        "model_contract_sha256": "5" * 64,
        "fleet_contract_sha256": "6" * 64,
        "server_pool_root": str((tmp_path / "server-pool").resolve()),
    }


def _suite(run_id: str, kind: str, cells: int) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "immutable_sha256": IMMUTABLE_SHA,
        "run_id": run_id,
        "expected_cells": cells,
        "schema5_complete_cells": cells,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
    }


def _typed_suite(
    run_id: str, kind: str, cells: int, immutable: dict
) -> dict:
    profiles = {
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
    suite = _suite(run_id, kind, cells)
    suite.update(
        {
            "semantic_validation_failures": 0,
            "top_level_length_censored_qids": 0,
            "top_level_protocol_censored_qids": 0,
            "auxiliary_length_censored_draws": 0,
            "auxiliary_protocol_censored_draws": 0,
            "concurrency": 1,
            "resumable": True,
            "execution_halted": False,
            "execution_halt_reason": None,
            "provenance": {
                **immutable,
                "server_pool_root": str(Path(immutable["server_pool_root"]).resolve()),
                "rollout_generation": 1,
            },
            "suite_identity": {
                "run_id": run_id,
                "cell_count": cells,
                "manifest_sha256": "7" * 64,
                "benchmark_contracts_sha256": "8" * 64,
                "artifact_policy_sha256": "9" * 64,
                "lineage_sha256": "b" * 64,
                "serving_profile_counts": profiles[kind],
                "estimand_excluded": True,
            },
            "cells": [
                {
                    "cell_id": f"cell-{index:03d}",
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
                for index in range(cells)
            ],
        }
    )
    return suite


def test_runtime_environment_contains_exact_production_fleet_pin():
    policy = SimpleNamespace(
        release=SimpleNamespace(release_id="sweep-recovery-schema5-v1"),
        accepted_model_contract_sha256="b" * 64,
        environment=SimpleNamespace(harness_sha256="c" * 64, serving_sha256="d" * 64),
        file_sha256="e" * 64,
    )
    observed = smoke._runtime_environment(
        policy=policy,
        immutable_pins_sha256=IMMUTABLE_SHA,
        fleet_contract_sha256="f" * 64,
        rollout_generation=7,
    )
    assert set(observed) == set(dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS)
    assert observed["ASYS_FLEET_CONTRACT_SHA256"] == "f" * 64
    assert observed["ASYS_ROLLOUT_GENERATION"] == "7"


def test_incident_accounting_includes_top_level_auxiliary_and_current_failure():
    rows = [
        {
            "termination_status": "protocol_censored",
            "self_consistency": {
                "sample_count": 3,
                "completed_sample_count": 1,
                "length_censored_sample_count": 1,
                "protocol_censored_sample_count": 1,
            },
        },
        {"termination_status": "length_censored"},
    ]
    failure = SimpleNamespace(
        classification="context_capacity",
        last_error={"type": "GenerationTruncationError", "message": "capacity"},
    )
    status = SimpleNamespace(failure=failure, errors=())
    assert smoke._incident_accounting(rows, status) == {
        "top_level_length_censored_qids": 1,
        "top_level_protocol_censored_qids": 1,
        "auxiliary_length_censored_draws": 1,
        "auxiliary_protocol_censored_draws": 1,
        "context_incidents": 1,
        "protocol_incidents": 2,
        "truncation_incidents": 3,
    }


def test_smoke_evidence_exactly_satisfies_typed_control_contract(tmp_path: Path):
    immutable = _immutable(tmp_path)
    suites = [
        _typed_suite("schema5_smoke_32b_long_v1", "long_32b_smoke", 15, immutable),
        _typed_suite(
            "schema5_smoke_selective_long_v1", "selective_long_smoke", 20,
            immutable,
        ),
        _typed_suite(
            "schema5_smoke_standard_canaries_v1", "standard_canary_smoke", 6,
            immutable,
        ),
    ]
    envelope = smoke._write_smoke_evidence(
        state_root=tmp_path,
        immutable_sha256=IMMUTABLE_SHA,
        suites=suites,
    )
    assert set(envelope) == {
        "schema_version",
        "gate",
        "passed",
        "immutable_sha256",
        "metrics",
        "artifacts",
    }
    assert envelope["passed"] is True
    assert envelope["metrics"] == {
        "long_32b_cells": 15,
        "selective_long_cells": 20,
        "standard_canary_cells": 6,
        "schema5_complete_cells": 41,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
    }
    evidence = tmp_path / "readiness" / "smoke_runs.json"
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert (tmp_path / "readiness" / "smoke_runs.sha256").read_text() == (
        f"{digest}  smoke_runs.json\n"
    )
    schema5_control._validate_attestation(
        {"immutable_sha256": IMMUTABLE_SHA, "immutable": immutable},
        "smoke_runs",
        evidence,
        digest,
    )


def test_smoke_evidence_fails_when_auxiliary_protocol_censor_is_present(
    tmp_path: Path,
):
    suites = [
        _suite("schema5_smoke_32b_long_v1", "long_32b_smoke", 15),
        _suite(
            "schema5_smoke_selective_long_v1", "selective_long_smoke", 20
        ),
        _suite(
            "schema5_smoke_standard_canaries_v1", "standard_canary_smoke", 6
        ),
    ]
    suites[1]["protocol_incidents"] = 1
    suites[1]["passed"] = False
    envelope = smoke._write_smoke_evidence(
        state_root=tmp_path,
        immutable_sha256=IMMUTABLE_SHA,
        suites=suites,
    )
    assert envelope["passed"] is False
    assert envelope["metrics"]["protocol_incidents"] == 1


def test_concurrency_one_lock_fails_nonblocking(tmp_path: Path):
    with smoke._exclusive_smoke_lock(tmp_path):
        with pytest.raises(smoke.SmokeRunError, match="concurrency-one lock"):
            with smoke._exclusive_smoke_lock(tmp_path):
                pass


@pytest.mark.parametrize(
    ("state", "eligible", "expected"),
    [
        (smoke.CompletionState.MISSING, True, True),
        (smoke.CompletionState.PARTIAL, True, True),
        (smoke.CompletionState.RETRYABLE, True, True),
        (smoke.CompletionState.RETRYABLE, False, False),
        (smoke.CompletionState.ACTIVE, True, False),
        (smoke.CompletionState.PERMANENT, True, False),
        (smoke.CompletionState.CORRUPT, True, False),
        (smoke.CompletionState.COMPLETE, True, False),
    ],
)
def test_resume_only_starts_safe_eligible_states(state, eligible, expected):
    status = SimpleNamespace(status=state, eligible_for_retry=eligible)
    assert smoke._runnable(status) is expected


def test_suite_artifact_hashes_bind_exact_bytes(tmp_path: Path):
    suites = [
        _suite("schema5_smoke_32b_long_v1", "long_32b_smoke", 15),
        _suite(
            "schema5_smoke_selective_long_v1", "selective_long_smoke", 20
        ),
        _suite(
            "schema5_smoke_standard_canaries_v1", "standard_canary_smoke", 6
        ),
    ]
    envelope = smoke._write_smoke_evidence(
        state_root=tmp_path,
        immutable_sha256=IMMUTABLE_SHA,
        suites=suites,
    )
    for reference in envelope["artifacts"]:
        path = Path(reference["path"])
        assert json.loads(path.read_text())["kind"] == reference["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
