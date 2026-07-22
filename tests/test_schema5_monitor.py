"""Schema-5 monitoring is QID-based, generation-scoped, and censor-complete."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import schema5_monitor as monitor


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
    fleet = {
        "schema_version": 1,
        "profiles": [
            {
                "serving_profile": profile,
                "replicas": [
                    {"replica_id": f"{profile}-{index}"}
                    for index in range(count)
                ],
            }
            for profile, count in config["fleet_replicas"].items()
        ],
    }
    fleet_path = tmp_path / "fleet.json"
    fleet_path.write_text(json.dumps(fleet, sort_keys=True) + "\n", encoding="utf-8")
    fleet_hash = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
    server_pool = tmp_path / "pool"
    server_pool.mkdir()
    return {
        "immutable": {
            "fleet_contract_path": str(fleet_path),
            "fleet_contract_sha256": fleet_hash,
            "server_pool_root": str(server_pool),
        }
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


def test_material_fleet_identity_fails_closed_on_contract_or_layout_drift(tmp_path):
    config = _config()
    state = _fleet_state_and_contract(tmp_path, config)
    contract_path = tmp_path / "fleet.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["note"] = "unattested mutation"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(monitor.MonitorError, match="hash drift"):
        monitor._verified_material_fleet_identity(state, config)

    state["immutable"]["fleet_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    contract["profiles"][0]["replicas"].pop()
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
    state["immutable"]["fleet_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    with pytest.raises(monitor.MonitorError, match="capacity layout differ"):
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
        "rollout_generation": 1,
        "effective_context": 32_768,
        "effective_context_limit": 32_768,
        "endpoint_generation": "endpoint",
    }
    assert monitor._record_matches_policy(record, cell, policy, contracts)
    assert not monitor._record_matches_policy(
        {key: value for key, value in record.items() if key != "schema_version"},
        cell,
        policy,
        contracts,
    )
    assert not monitor._record_matches_policy(
        {**record, "tokenizer_revision": "drift"}, cell, policy, contracts
    )


def test_final_acceptance_counts_censors_once_under_top_level_denominator():
    config = _config()
    semantic = {
        "states": {"complete": 22_680},
        "outcomes": {
            "validated_qids": 4_524_660,
            "completed_qids": 4_524_650,
            "length_censored_qids": 6,
            "protocol_censored_qids": 4,
            "auxiliary_outcomes": 12,
            "auxiliary_completed": 9,
            "auxiliary_length_censored": 2,
            "auxiliary_protocol_censored": 1,
        },
        "artifact_schema_counts": {"5": 4_524_660},
    }
    throughput = {
        "acceptance": {"projected_completion_within_target": False}
    }
    acceptance = monitor.final_acceptance(semantic, throughput, config)
    assert acceptance["checks"]["top_level_partition_exact"] is True
    assert acceptance["checks"]["auxiliary_partition_exact"] is True
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
    return {
        "semantic": {
            "scan_successful": True,
            "outcomes": {"validated_qids": 10},
            "runs": {
                run_id: {"outcomes": {"validated_qids": qids}}
                for run_id, qids in per_run.items()
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
            "disk": {
                "free_bytes": 10**15,
                "free_fraction": 0.9,
            },
            "latest_semantic_report_age_seconds": 0,
            "control": {
                "desired_state": "running",
                "rollout_generation": 1,
                "admission": {"current_ceiling": 24},
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
            "acceptance": {"strata_with_observed_throughput": 1}
        },
    }


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
    assert ramp_calls[0]["production_health_clean"] is True
    assert ramp_calls[0]["semantic_integrity_clean"] is True
    assert ramp_calls[0]["critical_finding_keys"] == []
    assert ramp_calls[0]["promotion_blocking_finding_keys"] == []


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
