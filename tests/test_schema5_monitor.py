"""Schema-5 monitoring is QID-based, generation-scoped, and censor-complete."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import schema5_monitor as monitor
from slurm import dispatch_sweeps as dispatcher


def _config() -> dict:
    return copy.deepcopy(monitor.load_monitor_config())


def test_monitoring_contract_pins_cadences_and_cardinality():
    config = _config()
    assert config["cadence_seconds"] == {
        "health": 300,
        "semantic": 21_600,
        "daily": 86_400,
    }
    assert config["expected_total_cells"] == 22_680
    assert config["expected_total_qids"] == 4_524_660
    assert config["throughput"]["minimum_qids_per_day"] == 161_595
    assert config["throughput"]["material_capacity_layout_generation"] == 1
    assert config["execution_deadline_seconds"] == {
        "health": 240,
        "semantic": 18_000,
        "daily": 18_000,
        "terminate_grace": 30,
    }
    assert config["alerts"]["dispatcher_ledger_stale_seconds"] == 360


def test_material_fleet_generation_ignores_endpoint_replacement_but_tracks_policy():
    fleet_hash = "a" * 64
    before = monitor._material_fleet_generation(
        fleet_contract_sha256=fleet_hash,
        material_capacity_layout_generation=1,
    )
    # Volatile endpoint allocation/process generations are intentionally not inputs.
    endpoint_allocations_before = {"32B-long": "allocation-100-start-1"}
    endpoint_allocations_after = {"32B-long": "allocation-200-start-2"}
    assert endpoint_allocations_before != endpoint_allocations_after
    after = monitor._material_fleet_generation(
        fleet_contract_sha256=fleet_hash,
        material_capacity_layout_generation=1,
    )
    assert after == before
    assert monitor._material_fleet_generation(
        fleet_contract_sha256="b" * 64,
        material_capacity_layout_generation=1,
    ) != before
    assert monitor._material_fleet_generation(
        fleet_contract_sha256=fleet_hash,
        material_capacity_layout_generation=2,
    ) != before


def _fleet_state_and_contract(tmp_path, config):
    source = (
        Path(__file__).resolve().parents[1] / "configs" / "schema5_fleet.v1.json"
    )
    fleet = json.loads(source.read_text(encoding="utf-8"))
    fleet_path = tmp_path / "fleet.json"
    fleet_path.write_text(json.dumps(fleet, sort_keys=True) + "\n", encoding="utf-8")
    fleet_hash = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
    fleet_path.with_suffix(".sha256").write_text(
        f"{fleet_hash}  {fleet_path.name}\n", encoding="utf-8"
    )
    model_path = (
        Path(__file__).resolve().parents[1] / "configs" / "model_contracts.v1.json"
    ).resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    server_pool = tmp_path / "pool"
    server_pool.mkdir()
    return {
        "immutable": {
            "fleet_contract_path": str(fleet_path),
            "fleet_contract_sha256": fleet_hash,
            "model_contract_path": str(model_path),
            "model_contract_sha256": model_hash,
            "server_pool_root": str(server_pool),
        },
        "capacity": {"current_generation": 1, "current_contract": None},
    }


def _base_fleet_binding(state, config):
    immutable = state["immutable"]
    return {
        "capacity_generation": 1,
        "path": immutable["fleet_contract_path"],
        "sha256": immutable["fleet_contract_sha256"],
        "fleet_id": "schema5-v1",
        "logical_replicas": sum(config["fleet_replicas"].values()),
        "allocated_gpus": 24,
        "profile_replicas": dict(config["fleet_replicas"]),
        "is_capacity_overlay": False,
    }


def test_collect_health_keeps_epoch_identity_across_server_allocation_replacement(
    monkeypatch, tmp_path
):
    config = _config()
    state = _fleet_state_and_contract(tmp_path, config)
    endpoints = {
        profile: {"live": count, "http_healthy": count}
        for profile, count in config["fleet_replicas"].items()
    }
    allocation = {"value": "allocation-a"}
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "effective_fleet_contract_binding",
        lambda *_a, **_kw: _base_fleet_binding(state, config),
    )
    monkeypatch.setattr(monitor.control, "query_scheduler", lambda **_kw: object())
    monkeypatch.setattr(
        monitor.control,
        "live_status",
        lambda *_a, **_kw: {
            "desired_state": "running",
            "open_throughput_epoch": None,
        },
    )
    monkeypatch.setattr(
        monitor.monitor_run,
        "_endpoint_stats",
        lambda *_a, **_kw: (
            endpoints,
            {
                profile: f"{allocation['value']}:{profile}"
                for profile in config["fleet_replicas"]
            },
        ),
    )
    monkeypatch.setattr(
        monitor.monitor_run, "_scheduler_stats", lambda: {"query_ok": True}
    )

    first = monitor.collect_health_state(
        results_root=tmp_path,
        state_dir=tmp_path / "state",
        config=config,
        now=1.0,
        probe_endpoints=True,
    )
    allocation["value"] = "allocation-b"
    second = monitor.collect_health_state(
        results_root=tmp_path,
        state_dir=tmp_path / "state",
        config=config,
        now=2.0,
        probe_endpoints=True,
    )
    assert first["server_pool_generations"] != second["server_pool_generations"]
    assert first["fleet_generation"] == second["fleet_generation"]


def test_collect_health_uses_additive_capacity_overlay_counts_and_identity(
    monkeypatch, tmp_path
):
    config = _config()
    state = _fleet_state_and_contract(tmp_path, config)
    state["capacity"]["current_generation"] = 2
    base_hash = state["immutable"]["fleet_contract_sha256"]
    overlay_hash = "b" * 64
    overlay_path = (tmp_path / "capacity-g000002" / "fleet.json").resolve()
    expected = dict(config["fleet_replicas"])
    expected["32B-long"] += 1
    binding = {
        "capacity_generation": 2,
        "path": str(overlay_path),
        "sha256": overlay_hash,
        "fleet_id": "schema5-v1",
        "logical_replicas": sum(expected.values()),
        "allocated_gpus": 26,
        "profile_replicas": expected,
        "is_capacity_overlay": True,
    }
    effective_fleet = SimpleNamespace(
        by_profile={
            profile: tuple(range(count)) for profile, count in expected.items()
        }
    )
    endpoints = {
        profile: {"live": count, "http_healthy": count}
        for profile, count in expected.items()
    }
    server_pool = Path(state["immutable"]["server_pool_root"])
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "effective_fleet_contract_binding",
        lambda *_a, **_kw: binding,
    )
    monkeypatch.setattr(
        monitor.control,
        "load_effective_fleet_contract",
        lambda *_a, **_kw: effective_fleet,
    )
    monkeypatch.setattr(monitor.control, "query_scheduler", lambda **_kw: object())
    monkeypatch.setattr(
        monitor.control,
        "live_status",
        lambda *_a, **_kw: {
            "desired_state": "running",
            "open_throughput_epoch": None,
        },
    )
    monkeypatch.setattr(
        monitor.monitor_run,
        "_endpoint_stats",
        lambda pool, profiles, **_kw: (
            endpoints,
            {profile: f"allocation-g2:{profile}" for profile in profiles},
        ),
    )
    monkeypatch.setattr(
        monitor.monitor_run, "_scheduler_stats", lambda: {"query_ok": True}
    )
    monkeypatch.setattr(
        monitor.fleet_transactions,
        "read_health_summary",
        lambda pool: {
            "available": True,
            "current_generation": 2,
            "active_hung_allocations": [],
            "historical_alert_count": 0,
            "alerts_path": str(pool / "alerts.jsonl"),
        },
    )

    health = monitor.collect_health_state(
        results_root=tmp_path,
        state_dir=tmp_path / "state",
        config=config,
        now=10.0,
        probe_endpoints=True,
    )
    assert health["fleet_mismatches"] == {}
    assert health["http_fleet_mismatches"] == {}
    assert health["material_fleet_identity"] == {
        "fleet_contract_path": str(overlay_path),
        "fleet_contract_sha256": overlay_hash,
        "release_fleet_contract_sha256": base_hash,
        "is_capacity_overlay": True,
        "capacity_generation": 2,
        "material_capacity_layout_generation": 2,
        "profile_replicas": dict(sorted(expected.items())),
    }
    assert health["fleet_generation"] == monitor._material_fleet_generation(
        fleet_contract_sha256=overlay_hash,
        material_capacity_layout_generation=2,
    )
    assert server_pool.is_dir()


def test_material_fleet_identity_fails_closed_on_contract_or_layout_drift(
    tmp_path, monkeypatch
):
    config = _config()
    state = _fleet_state_and_contract(tmp_path, config)
    monkeypatch.setattr(
        monitor.control,
        "effective_fleet_contract_binding",
        lambda *_a, **_kw: _base_fleet_binding(state, config),
    )
    contract_path = tmp_path / "fleet.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["note"] = "unattested mutation"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(monitor.MonitorError, match="effective fleet contract"):
        monitor._verified_material_fleet_identity(state, config)

    state["immutable"]["fleet_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    contract["profiles"][0]["replicas"].pop()
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    state["immutable"]["fleet_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    with pytest.raises(monitor.MonitorError, match="effective fleet contract"):
        monitor._verified_material_fleet_identity(state, config)


def test_projection_uses_qid_timestamps_and_requires_every_unfinished_stratum():
    config = _config()
    config["throughput"]["minimum_qids_per_day"] = 1
    epoch = 1_000.0
    now = epoch + 48 * 3600
    progress = {
        '{"agent_count":1,"model_size":"4B","reasoning_level":"off","topology":"single_agent"}': {
            "expected": 100,
            "validated": 20,
            "timestamps": [epoch + 1_000] * 20,
        },
        '{"agent_count":3,"model_size":"4B","reasoning_level":"off","topology":"independent"}': {
            "expected": 100,
            "validated": 0,
            "timestamps": [],
        },
    }
    report = monitor.throughput_projection(
        progress,
        progress,
        epoch_started_at=epoch,
        now=now,
        config=config,
    )
    assert report["overall"]["validated_qids"] == 20
    assert report["overall"]["window_qids"] == 20
    assert report["overall"]["qids_per_day"] == 10
    assert report["acceptance"]["observation_ready"] is True
    assert report["acceptance"]["strata_without_observed_throughput"] == 1
    assert report["acceptance"]["projected_completion_within_target"] is False


def test_projection_accepts_joint_28_day_gate_after_48_hours():
    config = _config()
    config["throughput"]["minimum_qids_per_day"] = 1
    epoch = 1_000.0
    now = epoch + 48 * 3600
    progress = {
        '{"agent_count":1,"model_size":"4B","reasoning_level":"off","topology":"single_agent"}': {
            "expected": 100,
            "validated": 60,
            "timestamps": [epoch + 1_000] * 60,
        },
        '{"agent_count":3,"model_size":"4B","reasoning_level":"off","topology":"independent"}': {
            "expected": 100,
            "validated": 60,
            "timestamps": [epoch + 2_000] * 60,
        },
    }
    report = monitor.throughput_projection(
        progress,
        progress,
        epoch_started_at=epoch,
        now=now,
        config=config,
    )
    assert report["acceptance"]["strata_without_observed_throughput"] == 0
    assert report["acceptance"]["max_stratum_eta_days"] < 28
    assert report["acceptance"]["projected_completion_within_target"] is True


def test_projection_requires_every_rotation_stratum_even_when_primary_strata_pass():
    config = _config()
    config["throughput"]["minimum_qids_per_day"] = 1
    epoch = 1_000.0
    now = epoch + 48 * 3600
    progress = {
        '{"agent_count":1,"model_size":"4B","reasoning_level":"off","topology":"single_agent"}': {
            "expected": 100,
            "validated": 60,
            "timestamps": [epoch + 1_000] * 60,
        }
    }
    rotation = {
        "observed": {"expected": 50, "validated": 30, "timestamps": [epoch + 1_000]},
        "missing": {"expected": 50, "validated": 30, "timestamps": []},
    }
    report = monitor.throughput_projection(
        progress, rotation, epoch_started_at=epoch, now=now, config=config
    )
    assert report["rotation_coverage"]["all_strata_observed"] is False
    assert report["acceptance"]["projected_completion_within_target"] is False


def test_fleet_generation_change_starts_new_prospective_epoch():
    state = {
        "desired_state": "running",
        "throughput_epochs": [
            {
                "fleet_generation": "old",
                "started_timestamp": 10.0,
                "closed_at": None,
            }
        ],
    }
    assert monitor._epoch_start_for_generation(
        state, "old", prospective_now=100.0
    ) == 10.0
    assert monitor._epoch_start_for_generation(
        state, "new", prospective_now=100.0
    ) == 100.0
    assert monitor._epoch_start_for_generation(
        {**state, "desired_state": "paused"},
        "new",
        prospective_now=100.0,
    ) is None


def test_policy_check_rejects_schema_less_or_revision_drift():
    cell = SimpleNamespace(model_size="4B")
    release_fleet_hash = "c" * 64
    active_fleet_hash = "d" * 64
    policy = SimpleNamespace(
        release=SimpleNamespace(release_id="release"),
        environment=SimpleNamespace(harness_sha256="a" * 64),
        accepted_model_contract_sha256="b" * 64,
    )
    contracts = SimpleNamespace(
        for_size=lambda _size: SimpleNamespace(
            model_revision="model-rev", tokenizer_revision="tok-rev"
        )
    )
    record = {
        "schema_version": 5,
        "release_id": "release",
        "environment_hash": "a" * 64,
        "model_revision": "model-rev",
        "tokenizer_revision": "tok-rev",
        "model_contract_sha256": "b" * 64,
        "fleet_contract_sha256": active_fleet_hash,
        "release_fleet_contract_sha256": release_fleet_hash,
        "capacity_generation": 1,
        "rollout_generation": 1,
        "effective_context": 32_768,
        "effective_context_limit": 32_768,
        "endpoint_generation": "endpoint",
        "termination_status": "completed",
        "per_agent": [{"endpoint_generation": "endpoint"}],
        "efficiency_raw": {"n_turns": 1},
        "coordinate_provenance_counts": {
            "capacity_generation": {"1": 1},
            "endpoint_generation": {"endpoint": 1},
            "fleet_contract_sha256": {active_fleet_hash: 1},
            "release_fleet_contract_sha256": {release_fleet_hash: 1},
            "rollout_generation": {"1": 1},
        },
        "coordinate_provenance_identity_counts": [
            {
                "release_fleet_contract_sha256": release_fleet_hash,
                "fleet_contract_sha256": active_fleet_hash,
                "capacity_generation": 1,
                "rollout_generation": 1,
                "endpoint_generation": "endpoint",
                "count": 1,
            }
        ],
    }
    allowed = frozenset(
        {(release_fleet_hash, active_fleet_hash, 1, 1, "endpoint")}
    )
    assert monitor._record_matches_policy(
        record,
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256=release_fleet_hash,
        trusted_generation_tuples=allowed,
    )
    assert not monitor._record_matches_policy(
        {key: value for key, value in record.items() if key != "schema_version"},
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256=release_fleet_hash,
        trusted_generation_tuples=allowed,
    )
    assert not monitor._record_matches_policy(
        {**record, "tokenizer_revision": "drift"},
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256=release_fleet_hash,
        trusted_generation_tuples=allowed,
    )


def test_policy_check_trusts_exact_cross_generation_qid_provenance():
    cell = SimpleNamespace(model_size="4B")
    release_fleet_hash = "c" * 64
    g1_fleet_hash = "d" * 64
    g2_fleet_hash = "e" * 64
    policy = SimpleNamespace(
        release=SimpleNamespace(release_id="release"),
        environment=SimpleNamespace(harness_sha256="a" * 64),
        accepted_model_contract_sha256="b" * 64,
    )
    contracts = SimpleNamespace(
        for_size=lambda _size: SimpleNamespace(
            model_revision="model-rev", tokenizer_revision="tok-rev"
        )
    )
    record = {
        "schema_version": 5,
        "release_id": "release",
        "environment_hash": "a" * 64,
        "model_revision": "model-rev",
        "tokenizer_revision": "tok-rev",
        "model_contract_sha256": "b" * 64,
        "fleet_contract_sha256": None,
        "release_fleet_contract_sha256": release_fleet_hash,
        "capacity_generation": None,
        "rollout_generation": None,
        "effective_context": 32_768,
        "effective_context_limit": 32_768,
        "endpoint_generation": "mixed",
        "termination_status": "completed",
        "efficiency_raw": {"n_turns": 2},
        "per_agent": [{"endpoint_generation": "endpoint-g2"}],
        "coordinate_provenance_counts": {
            "capacity_generation": {"1": 1, "2": 1},
            "endpoint_generation": {"endpoint-g1": 1, "endpoint-g2": 1},
            "fleet_contract_sha256": {
                g1_fleet_hash: 1,
                g2_fleet_hash: 1,
            },
            "release_fleet_contract_sha256": {release_fleet_hash: 2},
            "rollout_generation": {"1": 1, "2": 1},
        },
        "coordinate_provenance_identity_counts": [
            {
                "release_fleet_contract_sha256": release_fleet_hash,
                "fleet_contract_sha256": g1_fleet_hash,
                "capacity_generation": 1,
                "rollout_generation": 1,
                "endpoint_generation": "endpoint-g1",
                "count": 1,
            },
            {
                "release_fleet_contract_sha256": release_fleet_hash,
                "fleet_contract_sha256": g2_fleet_hash,
                "capacity_generation": 2,
                "rollout_generation": 2,
                "endpoint_generation": "endpoint-g2",
                "count": 1,
            },
        ],
    }
    allowed = frozenset(
        {
            (release_fleet_hash, g1_fleet_hash, 1, 1, "endpoint-g1"),
            (release_fleet_hash, g2_fleet_hash, 2, 2, "endpoint-g2"),
        }
    )
    assert monitor._record_matches_policy(
        record,
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256=release_fleet_hash,
        trusted_generation_tuples=allowed,
    )
    assert not monitor._record_matches_policy(
        {**record, "rollout_generation": 2},
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256=release_fleet_hash,
        trusted_generation_tuples=allowed,
    )
    assert not monitor._record_matches_policy(
        record,
        cell,
        policy,
        contracts,
        release_fleet_contract_sha256="f" * 64,
        trusted_generation_tuples=allowed,
    )


def test_semantic_scan_counts_cross_generation_qid_as_trusted(
    monkeypatch, tmp_path
):
    run_id = "mixed-generation-run"
    cell = SimpleNamespace(
        cell_id="cell-1",
        model_size="4B",
        reasoning_level=SimpleNamespace(value="off"),
        topology=SimpleNamespace(value="independent"),
        n_agents=2,
        benchmark="gpqa",
        context_share_level=SimpleNamespace(value="artifact_only"),
        prompt_complexity_level=0,
        seed=7,
    )
    question = SimpleNamespace(qid="q1")
    manifest_hash = "1" * 64
    benchmark_hash = "2" * 64
    policy_hash = "3" * 64
    model_hash = "4" * 64
    release_fleet_hash = "5" * 64
    g1_fleet_hash = "6" * 64
    g2_fleet_hash = "7" * 64
    snapshot = SimpleNamespace(
        cells=(cell,),
        ids=(cell.cell_id,),
        sha256=manifest_hash,
    )
    policy = SimpleNamespace(
        release=SimpleNamespace(release_id="release"),
        environment=SimpleNamespace(harness_sha256="8" * 64),
        accepted_manifest_sha256=manifest_hash,
        accepted_benchmark_contracts_sha256=benchmark_hash,
        accepted_model_contract_sha256=model_hash,
        file_sha256=policy_hash,
        policy_id="policy",
    )
    contracts = SimpleNamespace(
        sha256=model_hash,
        for_size=lambda _size: SimpleNamespace(
            model_revision="model-rev", tokenizer_revision="tok-rev"
        ),
    )
    state = {
        "immutable": {
            "runs": [
                {
                    "run_id": run_id,
                    "manifest_sha256": manifest_hash,
                    "benchmark_contract_sha256": benchmark_hash,
                    "policy_sha256": policy_hash,
                }
            ],
            "model_contract_path": str(tmp_path / "models.json"),
            "model_contract_sha256": model_hash,
            "fleet_contract_sha256": release_fleet_hash,
        },
        "rollout_generation": 2,
    }
    config = {
        "runs": [{"run_id": run_id, "expected_cells": 1, "expected_qids": 1}],
        "throughput": {
            "stratum_dimensions": ["model_size"],
            "rotation_dimensions": ["model_size"],
        },
    }
    record = {
        "schema_version": 5,
        "release_id": "release",
        "environment_hash": "8" * 64,
        "model_revision": "model-rev",
        "tokenizer_revision": "tok-rev",
        "model_contract_sha256": model_hash,
        "fleet_contract_sha256": None,
        "release_fleet_contract_sha256": release_fleet_hash,
        "capacity_generation": None,
        "rollout_generation": None,
        "effective_context": 32_768,
        "effective_context_limit": 32_768,
        "endpoint_generation": "mixed",
        "termination_status": monitor.TERMINATION_COMPLETED,
        "timestamp": 100.0,
        "efficiency_raw": {"n_turns": 2},
        "per_agent": [{"endpoint_generation": "endpoint-g2"}],
        "coordinate_provenance_counts": {
            "capacity_generation": {"1": 1, "2": 1},
            "endpoint_generation": {"endpoint-g1": 1, "endpoint-g2": 1},
            "fleet_contract_sha256": {
                g1_fleet_hash: 1,
                g2_fleet_hash: 1,
            },
            "release_fleet_contract_sha256": {release_fleet_hash: 2},
            "rollout_generation": {"1": 1, "2": 1},
        },
        "coordinate_provenance_identity_counts": [
            {
                "release_fleet_contract_sha256": release_fleet_hash,
                "fleet_contract_sha256": g1_fleet_hash,
                "capacity_generation": 1,
                "rollout_generation": 1,
                "endpoint_generation": "endpoint-g1",
                "count": 1,
            },
            {
                "release_fleet_contract_sha256": release_fleet_hash,
                "fleet_contract_sha256": g2_fleet_hash,
                "capacity_generation": 2,
                "rollout_generation": 2,
                "endpoint_generation": "endpoint-g2",
                "count": 1,
            },
        ],
    }

    class Catalog:
        def __init__(self, _run_root, *, snapshot):
            self.snapshot = snapshot
            self.frozen = object()
            self.sidecar_sha256 = benchmark_hash

        def questions_for(self, observed_cell):
            assert observed_cell is cell
            return (question,)

    run_root = tmp_path / run_id
    (run_root / "cells" / cell.cell_id).mkdir(parents=True)
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    allowed = frozenset(
        {
            (release_fleet_hash, g1_fleet_hash, 1, 1, "endpoint-g1"),
            (release_fleet_hash, g2_fleet_hash, 2, 2, "endpoint-g2"),
        }
    )
    monkeypatch.setattr(
        monitor.control,
        "refresh_trusted_generation_catalog",
        lambda *_a, **_kw: SimpleNamespace(
            allowed_generation_tuples=allowed,
            catalog_id="a" * 64,
            marker_path=(tmp_path / "catalog" / "COMPLETE.json").resolve(),
            marker_sha256="b" * 64,
            inventory_sha256="c" * 64,
            catalog_sha256="d" * 64,
        ),
    )
    monkeypatch.setattr(monitor, "load_manifest", lambda *_a, **_kw: snapshot)
    monkeypatch.setattr(monitor, "VerifiedQuestionCatalog", Catalog)
    monkeypatch.setattr(
        monitor, "load_artifact_policy", lambda *_a, **_kw: policy
    )
    monkeypatch.setattr(
        monitor, "load_model_contracts", lambda *_a, **_kw: contracts
    )
    monkeypatch.setattr(
        monitor,
        "get_completion_status",
        lambda *_a, **_kw: SimpleNamespace(
            status=monitor.CompletionState.PARTIAL,
            valid_count=1,
            malformed_lines=0,
            duplicate_qids=(),
            unexpected_qids=(),
            invalid_rows=0,
        ),
    )
    monkeypatch.setattr(
        monitor,
        "read_canonical_results",
        lambda *_a, **_kw: SimpleNamespace(records=(record,)),
    )

    semantic, progress, rotation = monitor.collect_semantic_state(
        results_root=tmp_path,
        state_dir=tmp_path / "state",
        config=config,
        now=200.0,
    )
    outcomes = semantic["runs"][run_id]["outcomes"]
    assert outcomes["validated_qids"] == 1
    assert outcomes.get("untrusted_valid_rows", 0) == 0
    assert set(outcomes) == set(monitor.control.FINAL_RUN_OUTCOME_FIELDS)
    assert set(semantic["outcomes"]) == set(
        monitor.control.FINAL_AGGREGATE_OUTCOME_FIELDS
    )
    assert semantic["artifact_schema_counts"] == {"5": 1}
    assert next(iter(progress.values()))["validated"] == 1
    assert next(iter(rotation.values()))["validated"] == 1


def test_zero_censor_generator_schema_passes_final_control_acceptance(tmp_path):
    runs = {}
    aggregate_counts: Counter[str] = Counter()
    for run_id, expected_cells in monitor.control.REQUIRED_RUNS.items():
        expected_qids = monitor.control.REQUIRED_RUN_QIDS[run_id]
        outcomes = monitor._closed_semantic_outcomes(
            {
                "validated_qids": expected_qids,
                "useful_qids": expected_qids,
                "completed_qids": expected_qids,
            },
            aggregate=False,
        )
        assert set(outcomes) == set(
            monitor.control.FINAL_RUN_OUTCOME_FIELDS
        )
        aggregate_counts.update(outcomes)
        runs[run_id] = {
            "run_root": str((tmp_path / run_id).resolve()),
            "manifest_cells": expected_cells,
            "expected_qids": expected_qids,
            "manifest_sha256": "1" * 64,
            "benchmark_contract_sha256": "2" * 64,
            "artifact_policy_sha256": "3" * 64,
            "artifact_policy_id": "4" * 64,
            "contract_errors": [],
            "states": {"complete": expected_cells},
            "outcomes": outcomes,
            "artifact_schema_counts": {"5": expected_qids},
            "stale_unmanifested_dirs": 0,
            "stale_unmanifested_examples": [],
        }
    aggregate = monitor._closed_semantic_outcomes(
        aggregate_counts,
        aggregate=True,
    )
    assert set(aggregate) == set(
        monitor.control.FINAL_AGGREGATE_OUTCOME_FIELDS
    )
    semantic = {
        "scan_successful": True,
        "scan_errors": [],
        "runs": runs,
        "states": {"complete": monitor.control.EXPECTED_TOTAL_CELLS},
        "outcomes": aggregate,
        "artifact_schema_counts": {
            "5": monitor.control.EXPECTED_TOTAL_QIDS
        },
        "trusted_generation_catalog": {
            "catalog_id": "5" * 64,
            "marker_path": str((tmp_path / "catalog.json").resolve()),
            "marker_sha256": "6" * 64,
            "inventory_sha256": "7" * 64,
            "catalog_payload_sha256": "8" * 64,
            "allowed_generation_tuple_count": 1,
        },
        "transport_censor_protocol": {
            "version": monitor.TRANSPORT_CENSOR_PROTOCOL_VERSION,
            "hash": monitor.TRANSPORT_CENSOR_PROTOCOL_HASH,
        },
    }
    acceptance = monitor.final_acceptance(
        semantic,
        {
            "acceptance": {
                "projected_completion_within_target": False
            }
        },
        _config(),
    )
    assert set(acceptance["checks"]) == set(
        monitor.control.FINAL_ACCEPTANCE_CHECK_NAMES
    )
    assert acceptance["passed"] is True

    monitor.control._validate_final_semantic_report(
        {
            "semantic": semantic,
            "final_acceptance": acceptance,
        }
    )


def test_final_acceptance_counts_censors_once_under_top_level_denominator():
    config = _config()
    semantic = {
        "states": {"complete": 22_680},
        "outcomes": {
            "validated_qids": 4_524_660,
            "useful_qids": 4_524_660,
            "completed_qids": 4_524_650,
            "length_censored_qids": 6,
            "protocol_censored_qids": 4,
            "transport_censored_qids": 0,
            "transport_affected_qids": 0,
            "topology_transport_censored_coordinates": 0,
            "transport_censored_coordinates": 0,
            "auxiliary_outcomes": 12,
            "auxiliary_completed": 9,
            "auxiliary_length_censored": 2,
            "auxiliary_protocol_censored": 1,
            "auxiliary_transport_censored": 0,
        },
        "artifact_schema_counts": {"5": 4_524_660},
        "transport_censor_protocol": {
            "version": monitor.TRANSPORT_CENSOR_PROTOCOL_VERSION,
            "hash": monitor.TRANSPORT_CENSOR_PROTOCOL_HASH,
        },
    }
    throughput = {
        "acceptance": {"projected_completion_within_target": False}
    }
    acceptance = monitor.final_acceptance(semantic, throughput, config)
    assert acceptance["checks"]["top_level_partition_exact"] is True
    assert acceptance["checks"]["auxiliary_partition_exact"] is True
    assert acceptance["passed"] is True


def test_final_acceptance_counts_concurrent_topology_transport_attempts_once():
    config = _config()
    semantic = {
        "states": {"complete": 22_680},
        "outcomes": {
            "validated_qids": 4_524_660,
            "useful_qids": 4_524_659,
            "completed_qids": 4_524_659,
            "length_censored_qids": 0,
            "protocol_censored_qids": 0,
            "transport_censored_qids": 1,
            "transport_affected_qids": 1,
            # Two siblings in one concurrent terminal wave can each end
            # ambiguously, while only one map-order exception terminates the QID.
            "topology_transport_censored_coordinates": 2,
            "transport_censored_coordinates": 2,
            "auxiliary_outcomes": 0,
            "auxiliary_completed": 0,
            "auxiliary_length_censored": 0,
            "auxiliary_protocol_censored": 0,
            "auxiliary_transport_censored": 0,
        },
        "artifact_schema_counts": {"5": 4_524_660},
        "transport_censor_protocol": {
            "version": monitor.TRANSPORT_CENSOR_PROTOCOL_VERSION,
            "hash": monitor.TRANSPORT_CENSOR_PROTOCOL_HASH,
        },
    }
    throughput = {
        "acceptance": {"projected_completion_within_target": False}
    }

    acceptance = monitor.final_acceptance(semantic, throughput, config)

    assert acceptance["checks"]["transport_coordinate_partition_exact"] is True
    assert (
        acceptance["checks"]["topology_transport_coordinate_qid_bound"] is True
    )
    assert acceptance["passed"] is True


def test_persist_does_not_start_throughput_epoch_from_untrusted_scan(
    monkeypatch, tmp_path
):
    calls: list[dict] = []
    state = _monitor_control_state()
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    report = _healthy_production_poll_report()
    report["semantic"]["outcomes"]["contract_errors"] = 1
    monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
    )
    assert calls == []
    assert report["production_poll_recorded"] is False


def _healthy_production_poll_report() -> dict:
    per_run = {
        "full_sweep_schema5_v1": 4,
        "full_sweep_agent_counts_schema5_v1": 3,
        "full_sweep_agent_count_7_schema5_v1": 3,
    }
    zero_run_outcomes = {
        field: 0 for field in monitor.control.FINAL_RUN_OUTCOME_FIELDS
    }
    zero_aggregate_outcomes = {
        field: 0
        for field in monitor.control.FINAL_AGGREGATE_OUTCOME_FIELDS
    }
    return {
        "semantic": {
            "scan_successful": True,
            "scan_errors": [],
            "states": {},
            "outcomes": {
                **zero_aggregate_outcomes,
                "validated_qids": 10,
                "useful_qids": 10,
            },
            "runs": {
                run_id: {
                    "run_root": f"/results/{run_id}",
                    "manifest_cells": monitor.control.REQUIRED_RUNS[run_id],
                    "expected_qids": monitor.control.REQUIRED_RUN_QIDS[run_id],
                    "manifest_sha256": "1" * 64,
                    "benchmark_contract_sha256": "2" * 64,
                    "artifact_policy_sha256": "3" * 64,
                    "artifact_policy_id": "schema5-policy",
                    "contract_errors": [],
                    "states": {},
                    "outcomes": {
                        **zero_run_outcomes,
                        "validated_qids": qids,
                        "useful_qids": qids,
                    },
                    "artifact_schema_counts": {},
                    "stale_unmanifested_dirs": 0,
                    "stale_unmanifested_examples": [],
                }
                for run_id, qids in per_run.items()
            },
            "artifact_schema_counts": {},
            "trusted_generation_catalog": {"catalog_id": "4" * 64},
            "transport_censor_protocol": {
                "version": monitor.TRANSPORT_CENSOR_PROTOCOL_VERSION,
                "hash": monitor.TRANSPORT_CENSOR_PROTOCOL_HASH,
            },
        },
        "health": {
            "fleet_generation": "fleet",
            "fleet_mismatches": {},
            "http_probes_performed": False,
            "http_fleet_mismatches": None,
            "scheduler": {
                "query_ok": True,
                "qos_max_memory_per_user_cell_holds": 0,
            },
            "ledger": {"starved_runs": [], "cached_state_counts": {}},
            "fleet_transactions": {
                "available": True,
                "current_generation": 1,
                "active_hung_allocations": [],
                "historical_alert_count": 0,
                "alerts_path": "/pool/.fleet-transactions-v1/alerts.jsonl",
            },
            "fleet_scheduler_policy": {
                "available": True,
                "drift": False,
                "scheduler_evidence_id": "a" * 64,
                "scheduler_policy_id": "b" * 64,
                "scheduler_policy_contract_id": "c" * 64,
                "attested_scheduler_policy_contract_id": "c" * 64,
                "transport_uncertainty_binding_sha256": "d" * 64,
                "preemptible_partitions": ["ou_bcs_low"],
                "partition_time_requirements_seconds": {
                    "ou_bcs_low": 86_400,
                    "ou_bcs_normal": 86_400,
                },
                "error": None,
            },
            "disk": {
                "free_bytes": 10**15,
                "free_fraction": 0.9,
            },
            "latest_semantic_report_age_seconds": 0,
            "control": {
                "desired_state": "running",
                "rollout_generation": 1,
                "admission": {"current_ceiling": 24},
                "external_watchdog_mirror": {
                    "required": True,
                    "healthy": True,
                    "status": "fresh",
                    "receipt_id": "e" * 64,
                    "sequence": 1,
                    "age_seconds": 1.0,
                },
                "scheduler": {"squeue_ok": True, "sacct_ok": True},
                "controllers": {
                    "dispatcher": {
                        "live_active": True,
                        "heartbeat_stale": False,
                    },
                    "fleet_supervisor": {
                        "live_active": True,
                        "heartbeat_stale": False,
                    },
                },
            },
        },
        "throughput": {
            "acceptance": {
                "observation_ready": False,
                "unfinished_strata": 1,
                "strata_with_observed_throughput": 1,
                "strata_without_observed_throughput": 0,
                "all_rotation_strata_observed": True,
                "max_stratum_eta_days": None,
                "overall_rate_meets_floor": True,
                "projected_completion_within_target": True,
            }
        },
    }


def test_missing_generation_catalog_is_a_data_trust_alert():
    report = _healthy_production_poll_report()
    report["health"]["disk"].update(
        {"free_inodes": 10**9, "free_inode_fraction": 0.9}
    )
    report["health"]["ledger"]["healthy"] = True
    report["semantic"].update(
        {
            "states": {},
            "scan_errors": [
                "trusted-generation-catalog: current marker is missing"
            ],
            "trusted_generation_catalog": None,
        }
    )
    report["throughput"]["acceptance"].update(
        {
            "observation_ready": False,
            "projected_completion_within_target": True,
        }
    )
    report["progress_watch"] = {
        "consecutive_no_progress_scans": 0,
        "validated_qids": 10,
    }
    findings = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )
    untrusted = [
        finding
        for finding in findings
        if finding.dedupe_key == "monitor:untrusted"
    ]
    assert len(untrusted) == 1
    assert untrusted[0].kind == "untrusted-responses"
    assert untrusted[0].severity == "critical"
    assert "catalog errors=" in untrusted[0].message


def test_useful_qids_exclude_mixed_topology_and_auxiliary_transport_censors():
    clean = {
        "termination_status": "completed",
        "observed_topology_coordinates": [],
        "self_consistency": {"samples": []},
    }
    mixed_topology = {
        "termination_status": "completed",
        "observed_topology_coordinates": [
            {
                "outcome": {
                    "termination_status": "completed",
                }
            },
            {
                "outcome": {
                    "termination_status": "transport_censored",
                }
            },
        ],
        "self_consistency": {"samples": []},
    }
    auxiliary = {
        "termination_status": "completed",
        "observed_topology_coordinates": [],
        "self_consistency": {
            "samples": [
                {"termination_status": "completed"},
                {"termination_status": "transport_censored"},
            ]
        },
    }
    top_level = {
        "termination_status": "transport_censored",
        "observed_topology_coordinates": [],
        "self_consistency": {"samples": []},
    }
    assert monitor._record_is_useful(clean) is True
    assert monitor._record_is_useful(mixed_topology) is False
    assert monitor._record_is_useful(auxiliary) is False
    assert monitor._record_is_useful(top_level) is False


def _monitor_control_state() -> dict:
    return {
        "desired_state": "running",
        "rollout_generation": 1,
        "immutable_sha256": "a" * 64,
        "admission": {"current_ceiling": 24},
        "alerts": [],
    }


def _stub_ramp_observation(*_args, **_kwargs) -> dict:
    return {
        "admission": {"current_ceiling": 24},
        "admission_ramp": {
            "last_action": {"action": "window_started", "timestamp": 1_000.0}
        },
    }


def test_complete_semantic_report_requests_autonomous_finalization(
    monkeypatch, tmp_path
):
    state = _monitor_control_state()
    requested: list[dict] = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control, "record_successful_poll", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    monkeypatch.setattr(
        monitor.control,
        "request_autonomous_finalization",
        lambda state_dir, **kwargs: requested.append(
            {"state_dir": state_dir, **kwargs}
        ),
    )
    report = _healthy_production_poll_report()
    for run_id, expected_qids in monitor.control.REQUIRED_RUN_QIDS.items():
        report["semantic"]["runs"][run_id]["outcomes"].update(
            {
                "validated_qids": expected_qids,
                "useful_qids": expected_qids,
            }
        )
    report["semantic"]["states"] = {
        "complete": monitor.control.EXPECTED_TOTAL_CELLS
    }
    report["semantic"]["outcomes"].update(
        {
            "validated_qids": monitor.control.EXPECTED_TOTAL_QIDS,
            "useful_qids": monitor.control.EXPECTED_TOTAL_QIDS,
        }
    )
    report["final_acceptance"] = {"passed": True}

    result = monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
        committed_at=1_001.0,
    )

    assert result["finalization_requested"] is True
    assert len(requested) == 1
    evidence = requested[0]["semantic_report_path"]
    assert evidence.is_file()
    assert evidence.stat().st_mode & 0o222 == 0
    assert requested[0]["now"] == 1_001.0


def test_incomplete_semantic_report_never_requests_finalization(
    monkeypatch, tmp_path
):
    state = _monitor_control_state()
    requested: list[dict] = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control, "record_successful_poll", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    monkeypatch.setattr(
        monitor.control,
        "request_autonomous_finalization",
        lambda *_args, **kwargs: requested.append(kwargs),
    )
    report = _healthy_production_poll_report()
    for run_id, expected_qids in monitor.control.REQUIRED_RUN_QIDS.items():
        report["semantic"]["runs"][run_id]["outcomes"].update(
            {
                "validated_qids": expected_qids,
                "useful_qids": expected_qids,
            }
        )
    last_run = tuple(monitor.control.REQUIRED_RUNS)[-1]
    report["semantic"]["runs"][last_run]["outcomes"].update(
        {
            "validated_qids": (
                monitor.control.REQUIRED_RUN_QIDS[last_run] - 1
            ),
            "useful_qids": (
                monitor.control.REQUIRED_RUN_QIDS[last_run] - 1
            ),
        }
    )
    report["semantic"]["states"] = {
        "complete": monitor.control.EXPECTED_TOTAL_CELLS - 1,
        "partial": 1,
    }
    report["semantic"]["outcomes"].update(
        {
            "validated_qids": monitor.control.EXPECTED_TOTAL_QIDS - 1,
            "useful_qids": monitor.control.EXPECTED_TOTAL_QIDS - 1,
        }
    )
    report["final_acceptance"] = {"passed": False}

    result = monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
        committed_at=1_001.0,
    )

    assert result["finalization_requested"] is False
    assert requested == []


def test_persist_starts_epoch_only_from_healthy_production_poll(monkeypatch, tmp_path):
    calls: list[dict] = []
    ramp_calls: list[dict] = []
    state = _monitor_control_state()
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    def record_ramp(_state_dir, **kwargs):
        evidence = kwargs["evidence_path"]
        assert evidence.stat().st_mode & 0o222 == 0
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == kwargs[
            "evidence_sha256"
        ]
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        ramp_calls.append(payload["ramp_observation"])
        return _stub_ramp_observation()

    monkeypatch.setattr(
        monitor.control, "record_admission_ramp_observation", record_ramp
    )
    report = _healthy_production_poll_report()

    monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
    )

    assert len(calls) == 1
    assert report["production_poll_recorded"] is True
    assert len(ramp_calls) == 1
    assert ramp_calls[0]["run_validated_qids"] == {
        "full_sweep_schema5_v1": 4,
        "full_sweep_agent_counts_schema5_v1": 3,
        "full_sweep_agent_count_7_schema5_v1": 3,
    }
    assert ramp_calls[0]["run_useful_qids"] == {
        "full_sweep_schema5_v1": 4,
        "full_sweep_agent_counts_schema5_v1": 3,
        "full_sweep_agent_count_7_schema5_v1": 3,
    }
    assert ramp_calls[0]["production_health_clean"] is True
    assert ramp_calls[0]["semantic_integrity_clean"] is True
    assert ramp_calls[0]["critical_finding_keys"] == []
    assert ramp_calls[0]["promotion_blocking_finding_keys"] == []


def test_only_health_cadence_advances_transient_hold_clean_polls(
    monkeypatch, tmp_path
):
    calls: list[dict] = []
    state = _monitor_control_state()
    state["admission_safety_hold"] = {"active": True}
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(monitor.control, "record_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "update_admission_safety_hold",
        lambda *_a, **kwargs: calls.append(kwargs) or state,
    )
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )

    monitor.persist_report(
        _healthy_production_poll_report(),
        cadence="semantic",
        state_dir=tmp_path / "semantic",
        findings=[],
        send_email=False,
        now=1_000.0,
        committed_at=1_000.0,
    )
    monitor.persist_report(
        _healthy_production_poll_report(),
        cadence="health",
        state_dir=tmp_path / "health",
        findings=[],
        send_email=False,
        now=1_001.0,
        committed_at=1_001.0,
    )

    assert calls[0]["clean_poll"] is False
    assert calls[0]["semantic_scan_clean"] is True
    assert calls[1]["clean_poll"] is True
    assert calls[1]["semantic_scan_clean"] is None


def test_ramp_evidence_blocks_qos_and_starvation_warnings_without_reclassifying_them():
    report = _healthy_production_poll_report()
    findings = [
        monitor.AlertFinding(
            "monitor:qos-memory", "qos-hold", "warning", "held cells"
        ),
        monitor.AlertFinding(
            "monitor:starvation", "run-starvation", "warning", "starved run"
        ),
    ]
    observation = monitor._admission_ramp_observation(
        report,
        cadence="semantic",
        state=_monitor_control_state(),
        findings=findings,
        captured_at=1_000.0,
        committed_at=1_001.0,
    )
    assert observation["critical_finding_keys"] == []
    assert observation["promotion_blocking_finding_keys"] == [
        "monitor:qos-memory",
        "monitor:starvation",
    ]


def test_persist_rejects_clean_scan_when_controller_or_fleet_is_unhealthy(
    monkeypatch, tmp_path
):
    calls: list[dict] = []
    state = _monitor_control_state()
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    report = _healthy_production_poll_report()
    report["health"]["control"]["controllers"]["dispatcher"][
        "heartbeat_stale"
    ] = True
    report["health"]["fleet_mismatches"] = {
        "32B-long": {"expected": 2, "live": 1}
    }

    monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
    )

    assert calls == []
    assert report["production_poll_recorded"] is False


def test_http_hung_endpoint_is_critical_and_blocks_successful_poll(
    monkeypatch, tmp_path
):
    calls: list[dict] = []
    state = _monitor_control_state()
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    report = _healthy_production_poll_report()
    report["health"]["http_probes_performed"] = True
    report["health"]["http_fleet_mismatches"] = {
        "32B-long": {"expected": 2, "http_healthy": 1}
    }
    findings = monitor.evaluate_alerts(report, cadence="health", config=_config())
    fleet_alert = next(item for item in findings if item.dedupe_key == "monitor:fleet")
    assert fleet_alert.severity == "critical"

    monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
    )
    assert calls == []
    assert report["production_poll_recorded"] is False


def test_scheduler_policy_drift_is_critical():
    report = _healthy_production_poll_report()
    report["health"]["fleet_scheduler_policy"] = {
        "available": True,
        "drift": True,
        "scheduler_evidence_id": "a" * 64,
        "scheduler_policy_id": "b" * 64,
        "scheduler_policy_contract_id": "c" * 64,
        "attested_scheduler_policy_contract_id": "d" * 64,
        "transport_uncertainty_binding_sha256": "e" * 64,
        "preemptible_partitions": ["ou_bcs_low"],
        "partition_time_requirements_seconds": {
            "ou_bcs_low": 86_400,
            "ou_bcs_normal": 86_400,
        },
        "error": None,
    }
    findings = monitor.evaluate_alerts(
        report, cadence="health", config=_config()
    )
    finding = next(
        item
        for item in findings
        if item.kind == "scheduler-policy-drift"
    )
    assert finding.dedupe_key == "monitor:scheduler"
    assert finding.severity == "critical"


def test_health_alerts_cover_scheduler_fleet_qos_disk_and_starvation():
    config = _config()
    report = {
        "health": {
            "control": {
                "desired_state": "running",
                "scheduler": {"squeue_ok": False, "sacct_ok": True},
                "controllers": {
                    "dispatcher": {"heartbeat_stale": True, "live_active": False},
                    "fleet_supervisor": {"heartbeat_stale": False, "live_active": True},
                },
            },
            "scheduler": {
                "query_ok": False,
                "qos_max_memory_per_user_cell_holds": 2,
            },
            "fleet_mismatches": {"32B-long": {"expected": 2, "live": 1}},
            "http_probes_performed": True,
            "http_fleet_mismatches": {
                "32B-long": {"expected": 2, "http_healthy": 1}
            },
            "ledger": {"starved_runs": ["full_sweep_schema5_v1"]},
            "disk": {"free_bytes": 1, "free_fraction": 0.01},
            "latest_semantic_report_age_seconds": 8 * 3600,
        }
    }
    keys = {
        finding.dedupe_key
        for finding in monitor.evaluate_alerts(report, cadence="health", config=config)
    }
    assert {
        "monitor:scheduler",
        "monitor:controllers",
        "monitor:fleet",
        "monitor:qos-memory",
        "monitor:disk",
        "monitor:starvation",
        "monitor:semantic-stale",
    } <= keys


def test_ledger_health_is_fail_closed_for_missing_invalid_and_stale(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    missing = monitor._ledger_health(
        state_dir, now=1_000.0, stale_seconds=360.0
    )
    assert missing["status"] == "missing"
    assert missing["healthy"] is False

    (state_dir / "ledger.json").write_text("{", encoding="utf-8")
    invalid = monitor._ledger_health(
        state_dir, now=1_000.0, stale_seconds=360.0
    )
    assert invalid["status"] == "invalid"

    # Freshness alone must not bless an object that the production dispatcher cannot
    # load.  In particular, a monitor must reject missing schema/mapping fields.
    (state_dir / "ledger.json").write_text(
        json.dumps(
            {
                "updated_at": 999.0,
                "poll_number": 2,
                "runs": {},
                "cells": {},
            }
        ),
        encoding="utf-8",
    )
    structurally_invalid = monitor._ledger_health(
        state_dir, now=1_000.0, stale_seconds=360.0
    )
    assert structurally_invalid["status"] == "invalid"
    assert structurally_invalid["healthy"] is False

    nested_invalid = {
        "schema_version": 1,
        "updated_at": 999.0,
        "poll_number": 2,
        "runs": {"full_sweep_schema5_v1": "not-a-run-record"},
        "jobs": {},
        "intents": {},
        "cells": {},
        "fairness": {},
        "validation_fairness": {},
    }
    (state_dir / "ledger.json").write_text(
        json.dumps(nested_invalid), encoding="utf-8"
    )
    assert (
        monitor._ledger_health(
            state_dir, now=1_000.0, stale_seconds=360.0
        )["status"]
        == "invalid"
    )

    stale_ledger = dispatcher._empty_ledger(now=600.0)
    stale_ledger["poll_number"] = 2
    (state_dir / "ledger.json").write_text(
        json.dumps(stale_ledger),
        encoding="utf-8",
    )
    stale = monitor._ledger_health(
        state_dir, now=1_000.0, stale_seconds=360.0
    )
    assert stale["status"] == "stale"
    assert stale["age_seconds"] == 400.0


def test_health_persistence_cannot_resolve_semantic_no_progress(
    monkeypatch, tmp_path
):
    state = _monitor_control_state()
    resolved = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(monitor.control, "record_alert", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        monitor.control,
        "resolve_alert",
        lambda _state_dir, **kwargs: resolved.append(kwargs["dedupe_key"]),
    )
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )

    monitor.persist_report(
        _healthy_production_poll_report(),
        cadence="health",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
        committed_at=1_000.0,
    )
    assert "monitor:no-progress" not in resolved
    assert "monitor:no-progress" not in monitor.HEALTH_ALERT_KEYS
    assert "monitor:no-progress" in monitor.SEMANTIC_ALERT_KEYS


def test_inode_ledger_no_progress_and_ramp_stall_are_critical(tmp_path):
    config = _config()
    assert config["alerts"]["ramp_stall_seconds"] == {
        "24": 28_800,
        "96": 46_800,
        "192": 68_400,
    }
    report = _healthy_production_poll_report()
    report["captured_timestamp"] = 40_000.0
    report["health"]["disk"].update(
        {
            "free_inodes": 10,
            "free_inode_fraction": 0.001,
        }
    )
    report["health"]["ledger"].update(
        {"healthy": False, "status": "stale", "age_seconds": 400.0}
    )
    report["health"]["control"].update(
        {
            "admission_safety_hold": {"active": False},
            "admission_ramp": {
                "window": None,
                "last_action": {"timestamp": 1_000.0},
            },
        }
    )
    report["progress_watch"] = {
        "useful_qids": 10,
        "previous_useful_qids": 10,
        "consecutive_no_progress_scans": 2,
    }
    report["semantic"]["states"] = {}
    report["throughput"]["acceptance"].update(
        {
            "observation_ready": False,
            "projected_completion_within_target": False,
        }
    )
    keys = {
        finding.dedupe_key
        for finding in monitor.evaluate_alerts(
            report, cadence="semantic", config=config
        )
        if finding.severity == "critical"
    }
    assert {
        "monitor:inodes",
        "monitor:ledger",
        "monitor:no-progress",
        "monitor:ramp-stall",
    } <= keys


@pytest.mark.parametrize(
    ("dedupe_key", "cadence"),
    [
        ("monitor:ramp-stall", "health"),
        ("monitor:throughput", "semantic"),
    ],
)
def test_capacity_gate_findings_remain_latched_until_transition(
    dedupe_key, cadence
):
    report = _healthy_production_poll_report()
    report["health"]["control"]["desired_state"] = "paused"
    report["health"]["control"]["admission_safety_hold"] = {
        "active": True,
        "mode": "integrity",
        "reasons": [dedupe_key],
    }
    report["throughput"]["acceptance"].update(
        {
            "observation_ready": False,
            "projected_completion_within_target": False,
        }
    )
    findings = monitor.evaluate_alerts(
        report, cadence=cadence, config=_config()
    )
    assert dedupe_key in {
        finding.dedupe_key
        for finding in findings
        if finding.severity == "critical"
    }


def test_exact_terminal_completion_stops_relatching_capacity_only_findings():
    report = _healthy_production_poll_report()
    report["health"]["control"].update(
        {
            "desired_state": "paused",
            "admission_safety_hold": {
                "active": True,
                "mode": "integrity",
                "reasons": [
                    "monitor:ramp-stall",
                    "monitor:throughput",
                ],
            },
            "safety_hold_drain_intent": {"state": "complete"},
        }
    )
    report["semantic"]["states"] = {
        "complete": monitor.control.EXPECTED_TOTAL_CELLS
    }
    report["semantic"]["scan_errors"] = []
    report["semantic"]["outcomes"].update(
        {
            "validated_qids": monitor.control.EXPECTED_TOTAL_QIDS,
            "useful_qids": monitor.control.EXPECTED_TOTAL_QIDS,
            "untrusted_valid_rows": 0,
            "contract_errors": 0,
            "protocol_censored_qids": 0,
        }
    )
    for run_id, expected_qids in monitor.control.REQUIRED_RUN_QIDS.items():
        report["semantic"]["runs"][run_id]["outcomes"].update(
            {
                "validated_qids": expected_qids,
                "useful_qids": expected_qids,
            }
        )
    report["throughput"]["acceptance"].update(
        {
            "observation_ready": True,
            "projected_completion_within_target": False,
        }
    )
    report["final_acceptance"] = {"passed": True}
    report["progress_watch"] = {
        "consecutive_no_progress_scans": 9,
        "useful_qids": monitor.control.EXPECTED_TOTAL_QIDS,
    }

    findings = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )
    critical_keys = {
        finding.dedupe_key
        for finding in findings
        if finding.severity == "critical"
    }
    assert "monitor:ramp-stall" not in critical_keys
    assert "monitor:throughput" not in critical_keys

    # Cardinality alone is insufficient: failed terminal acceptance keeps both
    # fail-closed capacity latches intact.
    report["final_acceptance"] = {"passed": False}
    blocked = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )
    blocked_keys = {
        finding.dedupe_key
        for finding in blocked
        if finding.severity == "critical"
    }
    assert {
        "monitor:ramp-stall",
        "monitor:throughput",
    } <= blocked_keys


def test_paused_latches_and_execution_alert_survive_missing_semantic_payload():
    report = _healthy_production_poll_report()
    report["health"]["disk"].update(
        {"free_inodes": 10**9, "free_inode_fraction": 0.9}
    )
    report["health"]["control"].update(
        {
            "desired_state": "paused",
            "admission_safety_hold": {
                "active": True,
                "mode": "integrity",
                "reasons": [
                    "monitor:ramp-stall",
                    "monitor:throughput",
                ],
            },
            "safety_hold_drain_intent": {"state": "complete"},
        }
    )
    report.pop("semantic")
    report.pop("throughput")

    findings = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )
    critical = {
        finding.dedupe_key: finding
        for finding in findings
        if finding.severity == "critical"
    }

    assert {
        "monitor:ramp-stall",
        "monitor:throughput",
        "monitor:execution:semantic",
    } <= set(critical)
    assert critical["monitor:execution:semantic"].kind == (
        "monitor-execution-failure"
    )
    assert "semantic must be an object" in critical[
        "monitor:execution:semantic"
    ].message
    assert "throughput must be an object" in critical[
        "monitor:execution:semantic"
    ].message


def test_nested_partial_semantic_payload_preserves_latches_and_reports_all_issues():
    report = _healthy_production_poll_report()
    report["health"]["disk"].update(
        {"free_inodes": 10**9, "free_inode_fraction": 0.9}
    )
    report["health"]["control"].update(
        {
            "desired_state": "paused",
            "admission_safety_hold": {
                "active": True,
                "mode": "integrity",
                "reasons": [
                    "monitor:ramp-stall",
                    "monitor:throughput",
                ],
            },
            "safety_hold_drain_intent": {"state": "complete"},
        }
    )
    missing_run = "full_sweep_agent_count_7_schema5_v1"
    report["semantic"]["runs"].pop(missing_run)
    report["throughput"]["acceptance"].pop(
        "strata_with_observed_throughput"
    )

    findings = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )
    critical = {
        finding.dedupe_key: finding
        for finding in findings
        if finding.severity == "critical"
    }

    assert {
        "monitor:ramp-stall",
        "monitor:throughput",
        "monitor:execution:semantic",
    } <= set(critical)
    message = critical["monitor:execution:semantic"].message
    assert "semantic.runs must contain exactly the production run IDs" in message
    assert missing_run in message
    assert (
        "throughput.acceptance is missing fields "
        "['strata_with_observed_throughput']"
    ) in message


def test_nested_partial_semantic_persistence_is_non_actionable_and_fail_closed(
    monkeypatch, tmp_path
):
    state = _monitor_control_state()
    state["desired_state"] = "paused"
    state["alerts"] = [
        {
            "dedupe_key": "monitor:ramp-stall",
            "severity": "critical",
            "resolved_at": None,
        },
        {
            "dedupe_key": "monitor:throughput",
            "severity": "critical",
            "resolved_at": None,
        },
    ]
    state["admission_safety_hold"] = {
        "active": True,
        "mode": "integrity",
        "reasons": ["monitor:ramp-stall", "monitor:throughput"],
    }
    report = _healthy_production_poll_report()
    report["health"]["control"]["desired_state"] = "paused"
    run_id = "full_sweep_schema5_v1"
    report["semantic"]["runs"][run_id]["outcomes"].pop("useful_qids")
    report["throughput"]["acceptance"][
        "strata_with_observed_throughput"
    ] = True
    report["final_acceptance"] = {"passed": True}

    recorded: list[str] = []
    resolved: list[str] = []
    hold_updates: list[dict] = []
    forbidden_calls: list[str] = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_alert",
        lambda *_a, **kwargs: recorded.append(kwargs["dedupe_key"]),
    )
    monkeypatch.setattr(
        monitor.control,
        "resolve_alert",
        lambda *_a, **kwargs: resolved.append(kwargs["dedupe_key"]),
    )
    monkeypatch.setattr(
        monitor.control,
        "update_admission_safety_hold",
        lambda *_a, **kwargs: hold_updates.append(kwargs) or state,
    )
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **_kw: forbidden_calls.append("progress"),
    )
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        lambda *_a, **_kw: forbidden_calls.append("ramp"),
    )
    monkeypatch.setattr(
        monitor.control,
        "request_autonomous_finalization",
        lambda *_a, **_kw: forbidden_calls.append("finalization"),
    )

    result = monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=[],
        send_email=False,
        now=1_000.0,
        committed_at=1_001.0,
    )

    assert recorded == ["monitor:execution:semantic"]
    assert resolved == []
    assert forbidden_calls == []
    assert result["finalization_requested"] is False
    assert report["production_poll_recorded"] is False
    assert report["admission_ramp_recorded"] is False
    assert report["ramp_observation"]["evidence_usable"] is False
    assert report["ramp_observation"]["semantic_integrity_clean"] is False
    assert {
        "monitor:ramp-stall",
        "monitor:throughput",
        "monitor:execution:semantic",
    } <= set(hold_updates[0]["active_critical_keys"])
    persisted = json.loads(
        Path(result["history"]).read_text(encoding="utf-8")
    )
    execution = {
        row["dedupe_key"]: row for row in persisted["alert_findings"]
    }["monitor:execution:semantic"]
    assert (
        f"semantic.runs.{run_id}.outcomes is missing fields ['useful_qids']"
        in execution["message"]
    )
    assert (
        "throughput.acceptance.strata_with_observed_throughput must be a "
        "non-negative integer"
    ) in execution["message"]


def test_incomplete_semantic_report_cannot_resolve_monitor_alerts(
    monkeypatch, tmp_path
):
    report = _healthy_production_poll_report()
    report["health"]["disk"].update(
        {"free_inodes": 10**9, "free_inode_fraction": 0.9}
    )
    report["health"]["control"].update(
        {
            "desired_state": "paused",
            "admission_safety_hold": {
                "active": True,
                "mode": "integrity",
                "reasons": [
                    "monitor:ramp-stall",
                    "monitor:throughput",
                ],
            },
            "safety_hold_drain_intent": {"state": "complete"},
        }
    )
    report.pop("semantic")
    report.pop("throughput")
    findings = monitor.evaluate_alerts(
        report, cadence="semantic", config=_config()
    )

    state = _monitor_control_state()
    recorded: list[str] = []
    resolved: list[str] = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_alert",
        lambda *_a, **kwargs: recorded.append(kwargs["dedupe_key"]),
    )
    monkeypatch.setattr(
        monitor.control,
        "resolve_alert",
        lambda *_a, **kwargs: resolved.append(kwargs["dedupe_key"]),
    )
    monkeypatch.setattr(
        monitor,
        "_admission_ramp_observation",
        lambda *_a, **_kw: {
            "production_health_clean": False,
            "semantic_integrity_clean": False,
            "critical_finding_keys": sorted(
                finding.dedupe_key
                for finding in findings
                if finding.severity == "critical"
            ),
        },
    )
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )

    monitor.persist_report(
        report,
        cadence="semantic",
        state_dir=tmp_path,
        findings=findings,
        send_email=False,
        now=1_000.0,
        committed_at=1_000.0,
    )

    assert {
        "monitor:ramp-stall",
        "monitor:throughput",
        "monitor:execution:semantic",
    } <= set(recorded)
    assert resolved == []


def test_capacity_transition_request_is_a_critical_monitor_owned_finding():
    report = _healthy_production_poll_report()
    report["health"]["control"]["admission"].update(
        {"current_ceiling": 96, "maximum_cell_tasks": 384}
    )
    report["health"]["control"]["admission_ramp"] = {
        "window": None,
        "last_action": {"timestamp": 1_000.0},
        "capacity_gate": {
            "state": "capacity_transition_required",
            "from_ceiling": 96,
            "target_ceiling": 192,
            "capacity_generation": 1,
            "control_immutable_sha256": "a" * 64,
            "requested_at": "1970-01-01T00:16:40Z",
            "requested_timestamp": 1_000.0,
            "reason": "sealed_client_capacity_authorization_missing",
        },
    }
    report["health"]["control"]["admission_safety_hold"] = {
        "active": True
    }
    report["health"]["control"]["safety_hold_drain_intent"] = {
        "state": "complete"
    }
    findings = monitor.evaluate_alerts(
        report, cadence="health", config=_config()
    )
    capacity = [
        finding
        for finding in findings
        if finding.dedupe_key == "monitor:capacity-gate"
    ]
    assert len(capacity) == 1
    assert capacity[0].severity == "critical"
    assert "controlled non-preemptible" in capacity[0].message
    assert "monitor:capacity-gate" in monitor.HEALTH_ALERT_KEYS


def test_live_client_capacity_drift_is_a_critical_monitor_owned_finding():
    report = _healthy_production_poll_report()
    report["health"]["control"]["admission"] = {
        "current_ceiling": 192,
        "maximum_cell_tasks": 384,
    }
    report["health"]["control"]["admission_ramp"] = {
        "capacity_gate": None
    }
    report["health"]["client_capacity_health"] = {
        "required": True,
        "available": False,
        "drift": True,
        "configured_ceiling": 192,
        "capacity_generation": 2,
        "authorization_sha256": "a" * 64,
        "error": "authorized QOS MaxSubmitJobsPerUser drifted",
    }
    findings = monitor.evaluate_alerts(
        report, cadence="health", config=_config()
    )
    capacity = [
        finding
        for finding in findings
        if finding.dedupe_key == "monitor:capacity-gate"
    ]
    assert len(capacity) == 1
    assert capacity[0].severity == "critical"
    assert "no longer matches live scheduler truth" in capacity[0].message


def test_semantic_progress_watch_counts_only_repeated_no_progress(tmp_path):
    state_dir = tmp_path / "state"
    root = state_dir / monitor.MONITORING_DIRNAME
    root.mkdir(parents=True)

    def publish(prior: dict, committed: float) -> None:
        prior["cadence"] = "semantic"
        history = (
            root
            / "semantic"
            / f"{int(committed * 1_000_000):020d}.json"
        )
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text(
            json.dumps(prior, sort_keys=True) + "\n", encoding="utf-8"
        )
        history.chmod(0o444)
        digest = hashlib.sha256(history.read_bytes()).hexdigest()
        pointer = monitor._latest_pointer(
            cadence="semantic",
            history=history,
            history_sha256=digest,
            captured_timestamp=prior["captured_timestamp"],
            committed_timestamp=committed,
        )
        (root / "semantic.latest.json").write_text(
            json.dumps(pointer, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    first = monitor._semantic_progress_watch(
        state_dir, useful_qids=100, expected_qids=1_000, now=10.0
    )
    assert first["consecutive_no_progress_scans"] == 0
    prior = {
        "captured_timestamp": 10.0,
        "semantic": {
            "outcomes": {"validated_qids": 100, "useful_qids": 100}
        },
        "progress_watch": first,
    }
    publish(prior, 11.0)
    second = monitor._semantic_progress_watch(
        state_dir, useful_qids=100, expected_qids=1_000, now=20.0
    )
    assert second["consecutive_no_progress_scans"] == 1
    prior["captured_timestamp"] = 20.0
    prior["progress_watch"] = second
    publish(prior, 21.0)
    third = monitor._semantic_progress_watch(
        state_dir, useful_qids=100, expected_qids=1_000, now=30.0
    )
    assert third["consecutive_no_progress_scans"] == 2
    progressed = monitor._semantic_progress_watch(
        state_dir, useful_qids=101, expected_qids=1_000, now=31.0
    )
    assert progressed["consecutive_no_progress_scans"] == 0


def test_semantic_progress_watch_rejects_forged_mutable_latest_pointer(
    tmp_path,
):
    state_dir = tmp_path / "state"
    root = state_dir / monitor.MONITORING_DIRNAME
    history = root / "semantic" / f"{11_000_000:020d}.json"
    history.parent.mkdir(parents=True)
    report = {
        "cadence": "semantic",
        "captured_timestamp": 10.0,
        "semantic": {
            "outcomes": {"validated_qids": 100, "useful_qids": 100}
        },
        "progress_watch": {"consecutive_no_progress_scans": 0},
    }
    history.write_text(json.dumps(report) + "\n", encoding="utf-8")
    history.chmod(0o444)
    pointer = monitor._latest_pointer(
        cadence="semantic",
        history=history,
        history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
        captured_timestamp=10.0,
        committed_timestamp=11.0,
    )
    pointer["history_sha256"] = "f" * 64
    identity = dict(pointer)
    identity.pop("pointer_id")
    pointer["pointer_id"] = monitor._sha256_value(identity)
    latest = root / "semantic.latest.json"
    latest.write_text(json.dumps(pointer) + "\n", encoding="utf-8")

    with pytest.raises(monitor.MonitorError, match="immutable history binding"):
        monitor._semantic_progress_watch(
            state_dir,
            useful_qids=100,
            expected_qids=1_000,
            now=20.0,
        )


def test_durable_hung_fleet_state_becomes_deduplicated_email_alert(
    monkeypatch, tmp_path
):
    report = _healthy_production_poll_report()
    report["health"]["fleet_transactions"]["active_hung_allocations"] = [
        {
            "replica_id": "schema5-v1--32b--long--r00",
            "ledger_generation": 2,
            "job_id": "700",
            "endpoint": "node001:8123",
            "cancel_state": "retryable",
            "cancel_attempts": 1,
            "first_failure_at": 0.0,
            "last_failure_at": 600.0,
            "alert_id": "fleet-hung-abc",
        }
    ]
    finding = next(
        item
        for item in monitor.evaluate_alerts(
            report, cadence="health", config=_config()
        )
        if item.dedupe_key == "monitor:fleet-hung"
    )
    assert finding.severity == "critical"
    assert monitor._production_poll_health_clean(report) is False

    state = _monitor_control_state()
    recorded = []
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_k: state)
    monkeypatch.setattr(
        monitor.control,
        "record_alert",
        lambda *_a, **kwargs: recorded.append(kwargs),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_k: None)
    monkeypatch.setattr(
        monitor.control,
        "record_admission_ramp_observation",
        _stub_ramp_observation,
    )
    monitor.persist_report(
        report,
        cadence="health",
        state_dir=tmp_path,
        findings=[finding],
        send_email=True,
        now=1_000.0,
        committed_at=1_000.0,
    )
    assert recorded == [
        {
            "kind": "hung-serving-allocation",
            "severity": "critical",
            "message": finding.message,
            "dedupe_key": "monitor:fleet-hung",
            "send_email": True,
            "now": 1_000.0,
        }
    ]


def test_handoff_violation_sets_critical_fleet_alert_and_blocks_clean_poll():
    report = _healthy_production_poll_report()
    report["health"]["fleet_transactions"].update(
        {
            "active_handoffs": [
                {
                    "replica_id": "schema5-v1--32b--long--r00",
                    "job_id": "701",
                    "lifecycle": "standby",
                }
            ],
            "handoff_overlap_gpus": 2,
            "handoff_violations": [
                {
                    "replica_id": "schema5-v1--32b--long--r00",
                    "job_id": "701",
                    "kind": "handoff_critical_lead",
                }
            ],
        }
    )
    findings = monitor.evaluate_alerts(
        report, cadence="health", config=_config()
    )
    finding = next(
        item for item in findings if item.dedupe_key == "monitor:fleet"
    )
    assert finding.severity == "critical"
    assert "handoff_violations" in finding.message
    assert monitor._production_poll_health_clean(report) is False

    report["health"]["fleet_transactions"]["handoff_violations"] = []
    assert monitor._production_poll_health_clean(report) is True


@pytest.mark.parametrize("status", ["missing", "stale", "invalid"])
def test_watchdog_mirror_failure_is_monitor_owned_critical_and_blocks_poll(
    status,
):
    report = _healthy_production_poll_report()
    report["health"]["control"]["external_watchdog_mirror"].update(
        {
            "healthy": False,
            "status": status,
            "error": f"synthetic {status}",
        }
    )
    findings = monitor.evaluate_alerts(
        report, cadence="health", config=_config()
    )
    mirror = [
        finding
        for finding in findings
        if finding.dedupe_key == "monitor:watchdog-mirror"
    ]
    assert len(mirror) == 1
    assert mirror[0].severity == "critical"
    assert "monitor:watchdog-mirror" in monitor.HEALTH_ALERT_KEYS
    assert monitor._production_poll_health_clean(report) is False
