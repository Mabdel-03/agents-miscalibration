"""Acceptance-monitor endpoint, failure, scheduler, and ETA regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.completion import CompletionState, CompletionStatus
from agents_scaling.experiment.manifest import ManifestSnapshot
from agents_scaling.serving.registry import ServerEntry, server_pool_generation

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import monitor_run as monitor  # noqa: E402


def _cell(
    *,
    model: str = "8B",
    reasoning: str = "off",
    topology: str = "single_agent",
    n_agents: int = 3,
) -> ExperimentCell:
    return ExperimentCell(
        model_size=model,
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level=reasoning,
        topology=topology,
        benchmark="gpqa",
        n_agents=n_agents,
        n_questions=1,
    )


def test_server_pool_mapping_matches_dispatcher_style(tmp_path):
    results = tmp_path / "results"
    run_root = results / "seven"
    assert monitor._resolve_server_pool(
        "seven",
        "seven=agent-count-pool",
        results_root=results,
        run_root=run_root,
    ) == (results / "agent-count-pool").resolve()

    absolute = tmp_path / "external-pool"
    assert monitor._resolve_server_pool(
        "seven",
        f"seven={absolute}",
        results_root=results,
        run_root=run_root,
    ) == absolute.resolve()

    with pytest.raises(ValueError, match="must target"):
        monitor._resolve_server_pool(
            "seven",
            "other=agent-count-pool",
            results_root=results,
            run_root=run_root,
        )


def test_cell_status_receives_code_and_current_server_pool_generation(
    tmp_path, monkeypatch
):
    cell = _cell()
    snapshot = ManifestSnapshot(tmp_path / "cells.json", (cell,), "digest")
    captured = {}

    def status(_cell, _directory, **kwargs):
        captured.update(kwargs)
        return CompletionStatus(
            status=CompletionState.MISSING,
            cell_id=cell.cell_id,
            expected_count=1,
            valid_count=0,
        )

    monkeypatch.setattr(monitor, "get_completion_status", status)
    report, observations = monitor._cell_stats(
        tmp_path,
        snapshot,
            question_catalog=SimpleNamespace(
                sidecar_sha256="f" * 64,
                frozen=SimpleNamespace(),
                snapshot=snapshot,
                questions_for=lambda _cell: (SimpleNamespace(qid="frozen-qid"),)
            ),
        code_version="commit+source.abc",
        server_pool_generations={"8B": "fleet-generation"},
        now=1234.0,
    )

    assert captured["code_version"] == "commit+source.abc"
    assert captured["server_pool_generation"] == "fleet-generation"
    assert captured["serving_profile"] == "8B"
    assert captured["now"] == 1234.0
    assert captured["expected_qids"] == ("frozen-qid",)
    assert tuple(q.qid for q in captured["expected_questions"]) == ("frozen-qid",)
    assert report["benchmark_contracts_sha256"] == "f" * 64
    assert report["missing"] == 1
    assert observations[0].state == "missing"


def test_endpoint_stats_uses_process_aware_server_pool_generation(
    tmp_path, monkeypatch
):
    entry = ServerEntry(
        model_size="8B",
        hf_id="Qwen/Qwen3-8B",
        host="gpu001",
        port=8000,
        slurm_job_id="123",
        started_at=100.25,
        serving_profile="8B",
        served_model_name="8B",
        max_model_len=32768,
        tp_size=1,
    )
    monkeypatch.setattr(monitor.registry, "list_servers", lambda *_args: [entry])
    monkeypatch.setattr(monitor.registry, "list_live_servers", lambda *_args, **_kwargs: [entry])

    report, generations = monitor._endpoint_stats(
        tmp_path, ["8B"], probe_http=False, timeout=0.01
    )

    expected = server_pool_generation([entry], profile_name="8B")
    assert generations == {"8B": expected}
    assert report["8B"]["server_pool_generation"] == expected
    assert report["8B"]["current_provenance_valid_registered"] == 1


def test_log_stats_classifies_rollout_blockers(tmp_path):
    legacy = tmp_path / "legacy"
    dispatcher = tmp_path / "dispatcher"
    legacy.mkdir()
    dispatcher.mkdir()
    (legacy / "cell_1_0.out").write_text(
        "APIConnectionError\nContextCapacityError\nBadRequestError: status 400\n"
        "retained length_censored\nretained length-censored\n"
        "retained protocol_censored\nretained protocol-censored\n"
    )
    (dispatcher / "dispatch_2_0.out").write_text(
        "GenerationTruncationError\nThinkingBudgetProtocolError\n"
        "ServerResponseProtocolError\n"
        "TokenizerInitializationError\n"
        "cannot import name 'AutoTokenizer' from 'transformers'\n"
    )

    report = monitor._log_stats((legacy, dispatcher))

    assert report["cell_or_dispatch_log_files"] == 2
    assert report["failure_signatures"]["connection_or_timeout"] == {
        "occurrences": 1,
        "affected_files": 1,
    }
    assert report["failure_signatures"]["context_capacity"]["occurrences"] == 1
    assert report["failure_signatures"]["generation_truncation"]["occurrences"] == 1
    assert report["failure_signatures"]["retained_length_censor"] == {
        "occurrences": 2,
        "affected_files": 1,
    }
    assert report["failure_signatures"]["retained_protocol_censor"] == {
        "occurrences": 2,
        "affected_files": 1,
    }
    assert report["failure_signatures"]["thinking_budget_protocol"]["occurrences"] == 2
    assert report["failure_signatures"]["tokenizer_initialization"] == {
        "occurrences": 2,
        "affected_files": 1,
    }
    assert report["failure_signatures"]["bad_request"]["affected_files"] == 1


def test_cell_stats_reports_auxiliary_censors_under_separate_denominator(
    tmp_path, monkeypatch
):
    cell = _cell()
    snapshot = ManifestSnapshot(tmp_path / "cells.json", (cell,), "digest")
    status = CompletionStatus(
        status=CompletionState.COMPLETE,
        cell_id=cell.cell_id,
        expected_count=1,
        valid_count=1,
        completed_question_count=1,
    )
    monkeypatch.setattr(monitor, "get_completion_status", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(
        monitor,
        "read_canonical_results",
        lambda *_args, **_kwargs: SimpleNamespace(
            records=(
                {
                    "termination_status": "completed",
                    "self_consistency": {
                        "sample_count": 5,
                        "completed_sample_count": 3,
                        "length_censored_sample_count": 1,
                        "protocol_censored_sample_count": 1,
                        "samples": [{}, {}, {}, {}, {}],
                    },
                },
            )
        ),
    )

    report, _ = monitor._cell_stats(
        tmp_path,
        snapshot,
        question_catalog=SimpleNamespace(
            sidecar_sha256="f" * 64,
            frozen=SimpleNamespace(),
            snapshot=snapshot,
            questions_for=lambda _cell: (SimpleNamespace(qid="frozen-qid"),),
        ),
        code_version="commit+source.abc",
        server_pool_generations={"8B": "fleet-generation"},
        now=1234.0,
    )

    assert report["completed_question_outcomes"] == 1
    assert report["length_censored_question_outcomes"] == 0
    assert report["protocol_censored_question_outcomes"] == 0
    assert report["auxiliary_sample_outcomes"] == 5
    assert report["auxiliary_completed_sample_outcomes"] == 3
    assert report["auxiliary_length_censored_sample_outcomes"] == 1
    assert report["auxiliary_protocol_censored_sample_outcomes"] == 1
    assert report["auxiliary_any_censored_sample_outcomes"] == 2
    assert report["auxiliary_censor_affected_cells"] == 1
    assert report["top_level_any_censor_rate"] == 0.0
    assert report["auxiliary_length_censor_rate"] == 0.2
    assert report["auxiliary_protocol_censor_rate"] == 0.2
    assert report["auxiliary_any_censor_rate"] == 0.4
    assert report["all_generation_length_censor_rate"] == 1 / 6
    assert report["all_generation_protocol_censor_rate"] == 1 / 6


def test_dispatcher_logs_are_scoped_to_the_requested_run(tmp_path):
    state = tmp_path / ".dispatcher"
    logs = state / "logs"
    logs.mkdir(parents=True)
    (logs / "dispatch_99_0.out").write_text("GenerationTruncationError\n")
    (logs / "dispatch_99_1.out").write_text("TokenizerInitializationError\n")
    (state / "ledger.json").write_text(
        json.dumps(
            {
                "jobs": {
                    "99": {
                        "tasks": [
                            {"run_id": "target"},
                            {"run_id": "different"},
                        ]
                    }
                }
            }
        )
    )

    selected = monitor._dispatcher_log_paths(state, "target")
    report = monitor._log_stats((), extra_paths=selected)

    assert selected == {logs / "dispatch_99_0.out"}
    assert report["failure_signatures"]["generation_truncation"]["occurrences"] == 1
    assert report["failure_signatures"]["tokenizer_initialization"]["occurrences"] == 0


def test_scheduler_surfaces_cell_qos_hold_reasons(monkeypatch):
    monkeypatch.setenv("USER", "scientist")
    stdout = "\n".join(
        [
            "10_0|PENDING|(QOSMaxMemoryPerUser)|asys-cells-run|16G|4",
            "11_0|PENDING|(Priority)|asys-dispatch-batch|2G|1",
            "12|PENDING|(QOSMaxGRESPerUser)|asys-serve-32B|120G|8",
            "13_0|RUNNING|node1|asys-dispatch-batch|2G|1",
        ]
    )
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=stdout, stderr=""
        ),
    )

    report = monitor._scheduler_stats()

    assert report["query_ok"]
    assert report["cell_jobs"] == 3
    assert report["pending_cell_jobs"] == 2
    assert report["pending_cell_hold_reasons"] == {
        "Priority": 1,
        "QOSMaxMemoryPerUser": 1,
    }
    assert report["qos_max_memory_per_user_cell_holds"] == 1


def test_eta_is_jointly_stratified_and_fails_closed_without_throughput():
    now = 10_000_000.0
    started = now - 72 * 3600.0
    fast = _cell(model="8B", reasoning="off", topology="single_agent", n_agents=3)
    stalled = _cell(
        model="32B", reasoning="b8192", topology="decentralized", n_agents=7
    )
    observations = [
        monitor.CellObservation(fast, "complete", now - 10 * 3600.0, 2.0),
        monitor.CellObservation(fast, "complete", now - 30 * 3600.0, 3.0),
        monitor.CellObservation(fast, "missing", None, None),
        monitor.CellObservation(stalled, "missing", None, None),
    ]

    report = monitor._eta_report(
        observations,
        now=now,
        started_at=started,
        window_hours=48.0,
        min_observation_hours=48.0,
        target_days=28.0,
    )

    assert report["acceptance"]["observation_ready"]
    assert report["acceptance"]["strata_without_observed_throughput"] == 1
    assert not report["acceptance"]["projected_completion_within_target"]
    rows = {
        (row["model_size"], row["reasoning_level"], row["topology"], row["agent_count"]): row
        for row in report["strata"]
    }
    # single_agent is scientifically one agent even if a legacy config stores n_agents=3.
    assert rows[("8B", "off", "single_agent", 1)]["eta_days"] == 1.0
    assert rows[("32B", "b8192", "decentralized", 7)]["eta_days"] is None


def test_eta_start_defaults_to_atomic_dispatcher_ledger(tmp_path):
    state = tmp_path / ".dispatcher"
    state.mkdir()
    (state / "ledger.json").write_text(json.dumps({"created_at": 1234.5}))
    assert monitor._eta_started_at(None, state) == 1234.5
    assert monitor._eta_started_at("1970-01-01T01:00:00Z", state) == 3600.0


def test_eta_prefers_durable_post_rollout_epoch_and_default_state_is_v3(tmp_path):
    results_root = tmp_path / "results"
    state = results_root / ".dispatcher-v3"
    state.mkdir(parents=True)
    (state / "ledger.json").write_text(
        json.dumps(
            {
                "created_at": 100.0,
                "throughput_observation_started_at": 900.0,
            }
        )
    )

    assert monitor._dispatcher_state_dir(None, results_root) == state
    assert monitor._eta_started_at(None, state) == 900.0

    # A malformed explicit rollout field must not silently fall back to the older
    # creation time and prematurely satisfy the 48-hour acceptance gate.
    (state / "ledger.json").write_text(
        json.dumps(
            {
                "created_at": 100.0,
                "throughput_observation_started_at": None,
            }
        )
    )
    assert monitor._eta_started_at(None, state) is None
