"""Schema-5 monitoring is QID-based, generation-scoped, and censor-complete."""

from __future__ import annotations

import copy
from types import SimpleNamespace

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
    state = {"desired_state": "running", "alerts": []}
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
    report = {
        "semantic": {
            "scan_successful": True,
            "outcomes": {"validated_qids": 10, "contract_errors": 1},
        },
        "health": {"fleet_generation": "fleet"},
        "throughput": {"acceptance": {"strata_with_observed_throughput": 1}},
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


def _healthy_production_poll_report() -> dict:
    return {
        "semantic": {
            "scan_successful": True,
            "outcomes": {"validated_qids": 10},
        },
        "health": {
            "fleet_generation": "fleet",
            "fleet_mismatches": {},
            "scheduler": {"query_ok": True},
            "control": {
                "desired_state": "running",
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


def test_persist_starts_epoch_only_from_healthy_production_poll(monkeypatch, tmp_path):
    calls: list[dict] = []
    state = {"desired_state": "running", "alerts": []}
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
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


def test_persist_rejects_clean_scan_when_controller_or_fleet_is_unhealthy(
    monkeypatch, tmp_path
):
    calls: list[dict] = []
    state = {"desired_state": "running", "alerts": []}
    monkeypatch.setattr(monitor.control, "load_control", lambda *_a, **_kw: state)
    monkeypatch.setattr(
        monitor.control,
        "record_successful_poll",
        lambda *_a, **kw: calls.append(kw),
    )
    monkeypatch.setattr(monitor.control, "resolve_alert", lambda *_a, **_kw: None)
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
