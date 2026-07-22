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
        "release_id": "sweep-recovery-schema5-v1.1",
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
        release=SimpleNamespace(release_id="sweep-recovery-schema5-v1.1"),
        accepted_model_contract_sha256="b" * 64,
        environment=SimpleNamespace(harness_sha256="c" * 64, serving_sha256="d" * 64),
        file_sha256="e" * 64,
    )
    observed = smoke._runtime_environment(
        policy=policy,
        immutable_pins_sha256=IMMUTABLE_SHA,
        fleet_contract_sha256="f" * 64,
        rollout_generation=7,
        runtime_attestation={
            "path": "/state/runtime.g000007.json",
            "sha256": "1" * 64,
            "lease_path": "/state/lease.g000007.json",
        },
    )
    assert set(observed) == set(dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS)
    assert observed["ASYS_FLEET_CONTRACT_SHA256"] == "f" * 64
    assert observed["ASYS_ROLLOUT_GENERATION"] == "7"
    assert observed["ASYS_RUNTIME_ATTESTATION_SHA256"] == "1" * 64
    assert observed["ASYS_RUNTIME_INTEGRITY_LEASE"].endswith(
        "lease.g000007.json"
    )


def test_long_smoke_worker_renews_g_plus_one_lease_every_minute(
    tmp_path: Path, monkeypatch
):
    refreshes = []
    monkeypatch.setattr(
        smoke.runtime_integrity,
        "refresh_generation_lease",
        lambda **kwargs: refreshes.append(kwargs) or {"cached": True},
    )

    class Process:
        returncode = 0

        def __init__(self):
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise smoke.subprocess.TimeoutExpired("worker", timeout)
            return "done\n", ""

        def poll(self):
            return None

        def send_signal(self, _signal):
            raise AssertionError("healthy lease renewal must not drain the worker")

    process = Process()
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *_a, **_k: process)
    runtime_attestation = {
        "generation": 1,
        "path": str(tmp_path / "runtime.g000001.json"),
        "sha256": "1" * 64,
        "lease_path": str(tmp_path / "lease.g000001.json"),
    }
    immutable = {
        "release_id": "sweep-recovery-schema5-v1.1",
        "harness_environment_sha256": "2" * 64,
        "serving_environment_sha256": "3" * 64,
        "harness_environment_prefix": str(tmp_path / "harness"),
        "serving_environment_prefix": str(tmp_path / "serving"),
    }
    task = {
        "run_id": "smoke",
        "cell_id": "cell",
        "runtime_environment": {"ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_SHA},
    }

    report = smoke._run_task(
        task,
        state_root=tmp_path / "state",
        release_worktree=tmp_path / "release",
        harness_prefix=tmp_path / "harness",
        hf_home=tmp_path / "hf",
        runtime_attestation=runtime_attestation,
        immutable=immutable,
    )
    assert report["returncode"] == 0
    assert len(refreshes) == 2
    assert all(item["generation"] == 1 for item in refreshes)


def test_smoke_lease_refresh_failure_gracefully_drains_and_fails(tmp_path, monkeypatch):
    refresh_count = 0

    def refresh(**_kwargs):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count > 1:
            raise smoke.runtime_integrity.RuntimeIntegrityError("nested drift")
        return {"cached": True}

    monkeypatch.setattr(smoke.runtime_integrity, "refresh_generation_lease", refresh)

    class Process:
        returncode = 0

        def __init__(self):
            self.communications = 0
            self.signals = []

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise smoke.subprocess.TimeoutExpired("worker", timeout)
            return "drained\n", ""

        def poll(self):
            return None

        def send_signal(self, sent):
            self.signals.append(sent)

    process = Process()
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *_a, **_k: process)
    runtime_attestation = {
        "generation": 1,
        "path": str(tmp_path / "runtime.g000001.json"),
        "sha256": "1" * 64,
        "lease_path": str(tmp_path / "lease.g000001.json"),
    }
    immutable = {
        "release_id": "sweep-recovery-schema5-v1.1",
        "harness_environment_sha256": "2" * 64,
        "serving_environment_sha256": "3" * 64,
        "harness_environment_prefix": str(tmp_path / "harness"),
        "serving_environment_prefix": str(tmp_path / "serving"),
    }
    task = {
        "run_id": "smoke",
        "cell_id": "cell",
        "runtime_environment": {"ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_SHA},
    }

    with pytest.raises(smoke.SmokeRunError, match="lease refresh failed"):
        smoke._run_task(
            task,
            state_root=tmp_path / "state",
            release_worktree=tmp_path / "release",
            harness_prefix=tmp_path / "harness",
            hf_home=tmp_path / "hf",
            runtime_attestation=runtime_attestation,
            immutable=immutable,
        )
    assert process.signals == [smoke.signal.SIGUSR1]


def test_fresh_paused_control_binds_smoke_to_first_rollout_generation():
    # ``initialize_control`` publishes exactly this generation/state boundary.  Smoke
    # provenance is intentionally the generation that the first resume will publish.
    initialized = {
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": 0,
    }
    smoke._validate_preproduction_rollout_generation(initialized, 1)


@pytest.mark.parametrize("rollout_generation", [0, 2, 7])
def test_preproduction_smoke_rejects_arbitrary_rollout_generation(
    rollout_generation: int,
):
    initialized = {
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": 0,
    }
    with pytest.raises(smoke.SmokeRunError, match=r"paused control.*\+ 1"):
        smoke._validate_preproduction_rollout_generation(
            initialized, rollout_generation
        )


@pytest.mark.parametrize(
    ("desired_state", "drain_requested", "error"),
    [
        ("running", False, "desired_state=paused"),
        ("resuming", False, "desired_state=paused"),
        ("paused", True, "forbidden while draining"),
    ],
)
def test_preproduction_smoke_rejects_live_or_draining_control(
    desired_state: str, drain_requested: bool, error: str
):
    control = {
        "desired_state": desired_state,
        "drain_requested": drain_requested,
        "rollout_generation": 0,
    }
    with pytest.raises(smoke.SmokeRunError, match=error):
        smoke._validate_preproduction_rollout_generation(control, 1)


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
