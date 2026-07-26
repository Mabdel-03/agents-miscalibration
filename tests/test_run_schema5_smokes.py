from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_schema5_smokes as smoke
from slurm import dispatch_sweeps
from slurm import schema5_control
from agents_scaling.serving import registry
from agents_scaling.serving.generation_catalog import (
    GenerationEvidence,
    TrustedGenerationCatalog,
    publish_trusted_generation_catalog,
)


IMMUTABLE_SHA = "a" * 64


def _capacity_readiness_control(tmp_path: Path):
    contract = {
        "capacity_generation": 2,
        "path": str((tmp_path / "overlay.json").resolve()),
        "sha256": "b" * 64,
        "marker_path": str((tmp_path / "overlay.complete.json").resolve()),
        "marker_sha256": "c" * 64,
        "fleet_id": "schema5-v1",
        "logical_replicas": 23,
        "allocated_gpus": 25,
        "profile_replicas": {
            **schema5_control.EXPECTED_FLEET_PROFILES,
            "0.6B": schema5_control.EXPECTED_FLEET_PROFILES["0.6B"] + 1,
        },
        "activated_at": "1970-01-01T00:00:01Z",
        "activated_timestamp": 1.0,
    }
    transition_id = "capacity-g000001-to-g000002-fixture"
    return {
        "immutable": {},
        "desired_state": "paused",
        "drain_requested": True,
        "rollout_generation": 3,
        "capacity": {
            "current_generation": 2,
            "current_contract": contract,
            "active_transition": {
                "transition_id": transition_id,
                "phase": "readiness_pending",
                "to_generation": 2,
                "old_fleet_job_ids": ["900"],
                "last_retirement_error": None,
                "published_timestamp": 2.0,
                "last_fleet_launch_error": None,
                "fleet_launch_completed_timestamp": 3.0,
                "new_contract": contract,
            },
        },
        "admission_safety_hold": {
            "active": True,
            "mode": "operator",
            "reasons": [f"capacity-transition:{transition_id}"],
        },
        "readiness": {
            "fleet": {
                "passed": True,
                "capacity_generation": 2,
            }
        },
    }


def _immutable(tmp_path: Path) -> dict:
    fleet_path = (
        Path(__file__).resolve().parents[1] / "configs" / "schema5_fleet.v1.json"
    ).resolve()
    fleet_sha256 = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
    return {
        "release_id": "sweep-recovery-schema5-v1.2",
        "git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "harness_environment_sha256": "3" * 64,
        "serving_environment_sha256": "4" * 64,
        "model_contract_sha256": "5" * 64,
        "fleet_contract_path": str(fleet_path),
        "fleet_contract_sha256": fleet_sha256,
        "release_fleet_contract_sha256": fleet_sha256,
        "capacity_generation": 1,
        "server_pool_root": str((tmp_path / "server-pool").resolve()),
        "results_root": str((tmp_path / "results").resolve()),
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
        "transport_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
        "trusted_generation_failures": 0,
    }


def _trusted_catalog(
    tmp_path: Path, immutable: dict
) -> TrustedGenerationCatalog:
    pool = Path(immutable["server_pool_root"]).resolve()
    pool.mkdir(parents=True, exist_ok=True)
    entry = registry.ServerEntry(
        model_size="4B",
        hf_id="model",
        host="node001",
        port=18000,
        slurm_job_id="700",
        started_at=100.0,
        serving_profile="4B",
        served_model_name="served-model",
        max_model_len=16_384,
        tp_size=1,
        release_id=immutable["release_id"],
        environment_hash=immutable["serving_environment_sha256"],
        model_revision="model-revision",
        tokenizer_id="tokenizer",
        tokenizer_revision="tokenizer-revision",
        model_contract_sha256=immutable["model_contract_sha256"],
        fleet_contract_sha256=immutable["fleet_contract_sha256"],
        server_pool_id="schema5-v1",
        replica_id="schema5-v1--4b--standard--r00",
        replica_index=0,
        release_fleet_contract_sha256=immutable[
            "release_fleet_contract_sha256"
        ],
        capacity_generation=1,
        rollout_generation=1,
    )
    registry.write_standby_entry(pool, entry)
    intent_token = "7" * 32
    script = (
        pool
        / ".fleet-transactions-v1"
        / "sbatch"
        / "g000001"
        / f"{entry.replica_id}.{intent_token}.sbatch"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    script.chmod(0o444)
    script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    provenance = {
        "run_root": str(pool),
        "server_pool_id": "schema5-v1",
        "replica_id": entry.replica_id,
        "replica_index": 0,
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
        "capacity_generation": 1,
        "rollout_generation": 1,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": script_sha256,
    }
    history = registry.seal_endpoint_history(
        pool,
        entry,
        release_fleet_contract_sha256=str(
            entry.release_fleet_contract_sha256
        ),
        capacity_generation=1,
        rollout_generation=1,
        ledger_generation=1,
        intent_token=intent_token,
        intent_state="committed",
        committed_at=99.0,
        local_script_path=script,
        local_script_sha256=script_sha256,
        spooled_script=script.read_bytes(),
        spooled_provenance=provenance,
        scheduler_job_name="asys-s5-serve-4b-s-r00",
        scheduler_comment=(
            "asys-s5-fleet:pool=schema5-v1;profile=4B;"
            f"replica={entry.replica_id};generation=1;"
            f"intent={intent_token};fleet={entry.fleet_contract_sha256}"
        ),
        sealed_at=101.0,
    )
    evidence = tmp_path / "generation-evidence.json"
    evidence.write_text('{"passed":true}\n', encoding="utf-8")
    evidence.chmod(0o444)
    return publish_trusted_generation_catalog(
        tmp_path / "catalog-state",
        server_pool_root=pool,
        server_pool_id="schema5-v1",
        endpoint_records=[history],
        release_fleet_contract_sha256=immutable[
            "release_fleet_contract_sha256"
        ],
        fleet_contract_sha256=immutable["fleet_contract_sha256"],
        capacity_generation=1,
        rollout_generation=1,
        generation_evidence=[
            GenerationEvidence(
                "smoke_generation",
                evidence,
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
            )
        ],
        now=102.0,
    )


def _catalog_binding(catalog: TrustedGenerationCatalog) -> dict:
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


def _typed_suite(
    run_id: str,
    kind: str,
    cells: int,
    immutable: dict,
    *,
    catalog: TrustedGenerationCatalog,
    attempt_binding: dict,
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
            "top_level_transport_censored_qids": 0,
            "transport_affected_qids": 0,
            "transport_censored_coordinates": 0,
            "auxiliary_length_censored_draws": 0,
            "auxiliary_protocol_censored_draws": 0,
            "auxiliary_transport_censored_draws": 0,
            "transport_censor_protocol_version": (
                smoke.TRANSPORT_CENSOR_PROTOCOL_VERSION
            ),
            "transport_censor_protocol_hash": (
                smoke.TRANSPORT_CENSOR_PROTOCOL_HASH
            ),
            "concurrency": 1,
            "resumable": True,
            "execution_halted": False,
            "execution_halt_reason": None,
            "provenance": {
                "release_id": immutable["release_id"],
                "git_commit": immutable["git_commit"],
                "source_tree_sha256": immutable["source_tree_sha256"],
                "harness_environment_sha256": immutable[
                    "harness_environment_sha256"
                ],
                "serving_environment_sha256": immutable[
                    "serving_environment_sha256"
                ],
                "model_contract_sha256": immutable["model_contract_sha256"],
                "fleet_contract_sha256": immutable["fleet_contract_sha256"],
                "release_fleet_contract_sha256": immutable[
                    "release_fleet_contract_sha256"
                ],
                "capacity_generation": immutable["capacity_generation"],
                "server_pool_root": str(
                    Path(immutable["server_pool_root"]).resolve()
                ),
                "rollout_generation": 1,
                "trusted_generation_catalog": _catalog_binding(catalog),
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
                "smoke_attempt": attempt_binding,
            },
            "cells": [],
        }
    )
    trusted = next(iter(catalog.allowed_generation_tuples))
    for index in range(cells):
        cell_id = f"cell-{index:03d}"
        result_path = (
            Path(immutable["results_root"])
            / smoke.SMOKE_ATTEMPT_RUNS_NAME
            / attempt_binding["attempt_id"]
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
                            "release_fleet_contract_sha256": trusted[0],
                            "fleet_contract_sha256": trusted[1],
                            "capacity_generation": trusted[2],
                            "rollout_generation": trusted[3],
                            "endpoint_generation": trusted[4],
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
        suite["cells"].append(
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
                        "sha256": hashlib.sha256(
                            result_path.read_bytes()
                        ).hexdigest(),
                        "row_count": 1,
                    },
                    "semantic_errors": [],
                    "failure": None,
                }
        )
    return suite


def test_runtime_environment_contains_exact_production_fleet_pin():
    policy = SimpleNamespace(
        release=SimpleNamespace(release_id="sweep-recovery-schema5-v1.2"),
        accepted_model_contract_sha256="b" * 64,
        environment=SimpleNamespace(harness_sha256="c" * 64, serving_sha256="d" * 64),
        file_sha256="e" * 64,
    )
    observed = smoke._runtime_environment(
        policy=policy,
        release_git_commit="1" * 40,
        protected_capacity_marker_path="/sealed/PROTECTED_CAPACITY_COMPLETE.json",
        protected_capacity_marker_sha256="2" * 64,
        protected_capacity_marker_id="3" * 64,
        immutable_pins_sha256=IMMUTABLE_SHA,
        fleet_contract_path="/sealed/capacity/fleet.json",
        fleet_contract_sha256="f" * 64,
        release_fleet_contract_sha256="0" * 64,
        capacity_generation=3,
        rollout_generation=7,
        runtime_attestation={
            "path": "/state/runtime.g000007.json",
            "sha256": "1" * 64,
            "lease_path": "/state/lease.g000007.json",
        },
    )
    assert set(observed) == set(dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS)
    assert observed["ASYS_RELEASE_GIT_COMMIT"] == "1" * 40
    assert observed["ASYS_PROTECTED_CAPACITY_MARKER"].endswith(
        "PROTECTED_CAPACITY_COMPLETE.json"
    )
    assert observed["ASYS_PROTECTED_CAPACITY_MARKER_SHA256"] == "2" * 64
    assert observed["ASYS_PROTECTED_CAPACITY_MARKER_ID"] == "3" * 64
    assert observed["ASYS_FLEET_CONTRACT_SHA256"] == "f" * 64
    assert observed["ASYS_FLEET_CONTRACT_PATH"] == "/sealed/capacity/fleet.json"
    assert observed["ASYS_RELEASE_FLEET_CONTRACT_SHA256"] == "0" * 64
    assert observed["ASYS_CAPACITY_GENERATION"] == "3"
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
        "release_id": "sweep-recovery-schema5-v1.2",
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
        "release_id": "sweep-recovery-schema5-v1.2",
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


def test_preproduction_smoke_allows_exact_capacity_readiness_drain(tmp_path):
    control = _capacity_readiness_control(tmp_path)
    smoke._validate_preproduction_rollout_generation(control, 4)
    control["readiness"]["fleet"]["capacity_generation"] = 1
    with pytest.raises(smoke.SmokeRunError, match="fresh generation fleet gate"):
        smoke._validate_preproduction_rollout_generation(control, 4)


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
        "top_level_transport_censored_qids": 0,
        "transport_affected_qids": 0,
        "transport_censored_coordinates": 0,
        "auxiliary_length_censored_draws": 1,
        "auxiliary_protocol_censored_draws": 1,
        "auxiliary_transport_censored_draws": 0,
        "context_incidents": 1,
        "protocol_incidents": 2,
        "transport_incidents": 0,
        "truncation_incidents": 3,
    }


def test_smoke_status_passes_exact_trusted_tuple_allowlist_to_semantic_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cell = SimpleNamespace(cell_id="cell-001")
    snapshot = SimpleNamespace(cells=(cell,))
    question = SimpleNamespace(qid="qid-001")
    catalog = SimpleNamespace(
        questions_for=lambda _cell: (question,),
        frozen=object(),
        snapshot=snapshot,
    )
    monkeypatch.setattr(smoke, "load_manifest", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        smoke, "VerifiedQuestionCatalog", lambda *_a, **_k: catalog
    )
    trusted = frozenset(
        {("a" * 64, "b" * 64, 1, 1, "trusted-endpoint")}
    )
    observed = {}

    def completion_status(*_args, **kwargs):
        observed["trusted"] = kwargs["trusted_generation_tuples"]
        return SimpleNamespace(valid_count=0)

    monkeypatch.setattr(smoke, "get_completion_status", completion_status)
    status, rows = smoke._status(
        tmp_path,
        0,
        model_contract_path=tmp_path / "model-contract.json",
        trusted_generation_tuples=trusted,
    )
    assert status.valid_count == 0
    assert rows == []
    assert observed["trusted"] is trusted


def test_smoke_evidence_exactly_satisfies_typed_control_contract(tmp_path: Path):
    immutable = _immutable(tmp_path)
    catalog = _trusted_catalog(tmp_path, immutable)
    attempt_binding = {
        "protocol": smoke.ATTEMPT_BINDING_PROTOCOL,
        "attempt_id": (
            "a000001-g000001-c000001-" f"{catalog.catalog_id[:16]}"
        ),
        "attempt_ordinal": 1,
        "immutable_sha256": IMMUTABLE_SHA,
        "capacity_generation": 1,
        "rollout_generation": 1,
        "fleet_contract_sha256": immutable["fleet_contract_sha256"],
        "release_fleet_contract_sha256": immutable[
            "release_fleet_contract_sha256"
        ],
        "trusted_catalog_id": catalog.catalog_id,
    }
    suites = [
        _typed_suite(
            "schema5_smoke_32b_long_v1",
            "long_32b_smoke",
            15,
            immutable,
            catalog=catalog,
            attempt_binding=attempt_binding,
        ),
        _typed_suite(
            "schema5_smoke_selective_long_v1", "selective_long_smoke", 20,
            immutable,
            catalog=catalog,
            attempt_binding=attempt_binding,
        ),
        _typed_suite(
            "schema5_smoke_standard_canaries_v1", "standard_canary_smoke", 6,
            immutable,
            catalog=catalog,
            attempt_binding=attempt_binding,
        ),
    ]
    attempt_root = (
        tmp_path
        / "readiness"
        / smoke.SMOKE_ATTEMPT_BASE_NAME
        / "attempts"
        / attempt_binding["attempt_id"]
    )
    envelope = smoke._write_smoke_evidence(
        state_root=tmp_path,
        immutable_sha256=IMMUTABLE_SHA,
        suites=suites,
        output_root=attempt_root,
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
        "transport_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
        "trusted_generation_failures": 0,
    }
    evidence = attempt_root / "smoke_runs.json"
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert (attempt_root / "smoke_runs.sha256").read_text() == (
        f"{digest}  smoke_runs.json\n"
    )
    schema5_control._validate_attestation(
        {"immutable_sha256": IMMUTABLE_SHA, "immutable": immutable},
        "smoke_runs",
        evidence,
        digest,
    )

    # Marginally valid provenance is insufficient: after recomputing every wrapper
    # hash, the control boundary must still reopen the row and reject its unknown
    # joint endpoint-generation tuple.
    result = Path(suites[0]["cells"][0]["results_artifact"]["path"])
    row = json.loads(result.read_text(encoding="utf-8"))
    row["coordinate_provenance_identity_counts"][0][
        "endpoint_generation"
    ] = "foreign-endpoint-generation"
    result.chmod(0o644)
    result.write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )
    result.chmod(0o444)
    suites[0]["cells"][0]["results_artifact"]["sha256"] = hashlib.sha256(
        result.read_bytes()
    ).hexdigest()
    long_reference = next(
        item for item in envelope["artifacts"]
        if item["name"] == "long_32b_smoke"
    )
    long_reference["sha256"] = smoke._write_json_with_checksum(
        Path(long_reference["path"]), suites[0]
    )
    smoke._write_json_with_checksum(evidence, envelope)
    with pytest.raises(
        schema5_control.ReadinessError, match="out-of-catalog endpoint provenance"
    ):
        schema5_control._validate_attestation(
            {"immutable_sha256": IMMUTABLE_SHA, "immutable": immutable},
            "smoke_runs",
            evidence,
            hashlib.sha256(evidence.read_bytes()).hexdigest(),
        )


def test_smoke_evidence_rejects_missing_semantic_payload_without_keyerror(
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
    del suites[0]["trusted_generation_failures"]
    with pytest.raises(
        smoke.SmokeRunError, match="missing required semantic fields"
    ):
        smoke._write_smoke_evidence(
            state_root=tmp_path,
            immutable_sha256=IMMUTABLE_SHA,
            suites=suites,
        )
    assert not (tmp_path / "readiness" / "smoke_runs.json").exists()


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


def test_smoke_evidence_fails_when_transport_censor_is_present(
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
    suites[0]["transport_incidents"] = 1
    suites[0]["passed"] = False
    envelope = smoke._write_smoke_evidence(
        state_root=tmp_path,
        immutable_sha256=IMMUTABLE_SHA,
        suites=suites,
    )
    assert envelope["passed"] is False
    assert envelope["metrics"]["transport_incidents"] == 1


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


def _attempt_generation(
    *,
    capacity: int = 1,
    rollout: int = 1,
    catalog: str = "a",
    fleet: str = "b",
) -> dict:
    return {
        "catalog_id": catalog * 64,
        "marker_path": "/sealed/TRUSTED_GENERATION.json",
        "marker_sha256": "c" * 64,
        "inventory_sha256": "d" * 64,
        "catalog_payload_sha256": "e" * 64,
        "allowed_generation_tuple_count": 22,
        "release_fleet_contract_sha256": "f" * 64,
        "fleet_contract_sha256": fleet * 64,
        "capacity_generation": capacity,
        "rollout_generation": rollout,
    }


def _attempt_immutable(tmp_path: Path) -> dict:
    value = _immutable(tmp_path)
    value.update(
        {
            "protected_capacity_marker_path": "/sealed/PROTECTED_CAPACITY_COMPLETE.json",
            "protected_capacity_marker_sha256": "1" * 64,
            "protected_capacity_marker_id": "2" * 64,
        }
    )
    return value


def _fake_attempt_initializer(*, results_root: Path, **_kwargs) -> dict:
    results_root.mkdir(parents=True, exist_ok=True)
    return {"status": "initialized", "total_cells": 41}


def test_killed_and_transport_censored_attempts_are_sealed_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "initialize_all", _fake_attempt_initializer)
    base = tmp_path / "state" / "readiness" / smoke.SMOKE_ATTEMPT_BASE_NAME
    results = tmp_path / "results"
    immutable = _attempt_immutable(tmp_path)
    generation = _attempt_generation()

    first_path, first = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=generation,
        now=1_000.0,
    )
    partial = Path(first["runs_root"]) / "partial-response.jsonl"
    partial.write_text('{"qid":"retained"}\n', encoding="utf-8")

    second_path, second = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=generation,
        now=2_000.0,
    )
    first_failure = smoke._read_sealed_json(
        Path(first["attempt_root"]) / smoke.ATTEMPT_FAILURE_NAME,
        description="first failure",
    )
    assert first_failure["reason"] == "interrupted_before_terminal_publication"
    assert first_failure["retryable"] is True
    assert first_failure["inventory"]["run_artifacts"] == [
        {
            "path": "partial-response.jsonl",
            "bytes": partial.stat().st_size,
            "sha256": hashlib.sha256(partial.read_bytes()).hexdigest(),
        }
    ]
    smoke._assert_tree_read_only(
        Path(first["runs_root"]), description="killed smoke runs"
    )

    smoke._publish_failure(
        second_path,
        second,
        reason="transport_censor_observed",
        retryable=True,
    )
    third_path, third = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=generation,
        now=3_000.0,
    )

    assert [first["attempt_ordinal"], second["attempt_ordinal"], third["attempt_ordinal"]] == [
        1,
        2,
        3,
    ]
    assert third["predecessor"] == smoke._pointer_reference(second_path, second)
    assert first_path != second_path != third_path
    assert Path(first["runs_root"]) != Path(second["runs_root"]) != Path(third["runs_root"])


def _publish_minimal_success(
    *,
    base: Path,
    pointer_path: Path,
    pointer: dict,
    provenance: dict,
) -> dict:
    evidence = Path(pointer["attempt_root"]) / smoke.ATTEMPT_EVIDENCE_NAME
    smoke._atomic_json(
        evidence,
        {
            "schema_version": 2,
            "gate": "smoke_runs",
            "passed": True,
            "immutable_sha256": pointer["immutable_sha256"],
            "metrics": {"schema5_complete_cells": 41},
            "artifacts": [],
        },
    )
    return smoke._publish_success(
        base=base,
        pointer_path=pointer_path,
        pointer=pointer,
        provenance=provenance,
    )


def test_current_selector_replay_and_attempt_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "initialize_all", _fake_attempt_initializer)
    state = tmp_path / "state"
    base = state / "readiness" / smoke.SMOKE_ATTEMPT_BASE_NAME
    results = tmp_path / "results"
    immutable = _attempt_immutable(tmp_path)
    generation1 = _attempt_generation()
    first_path, first = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=generation1,
        now=1_000.0,
    )
    first_selector = _publish_minimal_success(
        base=base,
        pointer_path=first_path,
        pointer=first,
        provenance={"capacity_generation": 1},
    )

    generation2 = _attempt_generation(
        capacity=2, rollout=2, catalog="3", fleet="4"
    )
    second_path, second = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=generation2,
        now=2_000.0,
    )
    second_selector = _publish_minimal_success(
        base=base,
        pointer_path=second_path,
        pointer=second,
        provenance={"capacity_generation": 2},
    )

    smoke._replace_sealed_json(base / smoke.CURRENT_SELECTOR_NAME, first_selector)
    with pytest.raises(smoke.SmokeRunError, match="replays a stale attempt"):
        smoke._load_current_attempt(
            state_root=state,
            results_root=results,
        )

    smoke._replace_sealed_json(
        base / smoke.CURRENT_SELECTOR_NAME, second_selector
    )
    completion = Path(second["attempt_root"]) / smoke.ATTEMPT_COMPLETE_NAME
    completion.chmod(0o644)
    with pytest.raises(smoke.SmokeRunError, match="canonical sealed file"):
        smoke._load_current_attempt(
            state_root=state,
            results_root=results,
            require_latest=False,
        )


def test_capacity_generation_creates_fresh_attempt_and_preserves_prior_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "initialize_all", _fake_attempt_initializer)
    base = tmp_path / "state" / "readiness" / smoke.SMOKE_ATTEMPT_BASE_NAME
    results = tmp_path / "results"
    immutable = _attempt_immutable(tmp_path)
    first_path, first = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=_attempt_generation(),
        now=1_000.0,
    )
    _publish_minimal_success(
        base=base,
        pointer_path=first_path,
        pointer=first,
        provenance={"capacity_generation": 1},
    )
    second_generation = _attempt_generation(
        capacity=2, rollout=3, catalog="5", fleet="6"
    )
    second_path, second = smoke._create_attempt(
        base=base,
        results_root=results,
        release_worktree=Path(smoke.__file__).resolve().parents[1],
        immutable_sha256=IMMUTABLE_SHA,
        immutable=immutable,
        readiness_generation=second_generation,
        now=2_000.0,
    )

    assert second["readiness_generation"] == second_generation
    assert second["predecessor"] == smoke._pointer_reference(first_path, first)
    assert Path(second["runs_root"]) != Path(first["runs_root"])
    smoke._assert_tree_read_only(
        Path(first["attempt_root"]), description="prior successful smoke"
    )
