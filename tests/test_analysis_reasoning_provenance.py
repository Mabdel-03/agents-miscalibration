"""Analysis ingestion never labels legacy whitespace counts as exact tokens."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.nb_lib.ingest import (
    add_efficiency,
    agent_records,
    cell_record,
    item_record,
)
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import analyze as canonical_analyze
from agents_scaling.experiment.analyze import _censor_accounting
from agents_scaling.experiment.result_schema import (
    ARTIFACT_SCHEMA_VERSION,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)


CFG = {
    "benchmark": "gpqa",
    "model_size": "4B",
    "topology": "single_agent",
    "context_share_level": "artifact_only",
    "prompt_complexity_level": 0,
    "reasoning_level": "b512",
    "seed": 0,
}


def _row(*, exact: bool) -> dict:
    agent = {
        "agent_id": "agent0",
        "round": 0,
        "answer": "A",
        "option_logprobs": {"A": 0.8},
        "verbalized_conf": 0.8,
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "reasoning_tokens": 17,
        "reasoning_text": "some reasoning",
        "cot_text": "some reasoning",
    }
    row = {
        "qid": "q",
        "correct": True,
        "final_answer": "A",
        "answer_key": "A",
        "per_agent": [agent],
        "efficiency_raw": {
            "n_agents": 1,
            "n_turns": 1,
            "n_messages": 0,
            "total_prompt_tokens": 20,
            "total_completion_tokens": 30,
            "total_reasoning_tokens": 17,
            "wall_ms": 1.0,
        },
    }
    if exact:
        row["schema_version"] = ARTIFACT_SCHEMA_VERSION
        agent["reasoning_token_source"] = "vllm_native_token_ids"
    return row


def test_item_and_agent_token_columns_are_exact_only():
    legacy = _row(exact=False)
    exact = _row(exact=True)

    legacy_item = item_record("run", "cell", CFG, legacy)
    exact_item = item_record("run", "cell", CFG, exact)
    assert math.isnan(legacy_item["total_reasoning_tokens"])
    assert legacy_item["total_reasoning_tokens_legacy_or_mixed"] == 17
    assert legacy_item["reasoning_tokens_exact"] is False
    assert exact_item["total_reasoning_tokens"] == 17
    assert math.isnan(exact_item["total_reasoning_tokens_legacy_or_mixed"])
    assert exact_item["reasoning_tokens_exact"] is True

    legacy_agent = agent_records("run", "cell", CFG, legacy, False)[0]
    exact_agent = agent_records("run", "cell", CFG, exact, False)[0]
    assert math.isnan(legacy_agent["reasoning_tokens"])
    assert legacy_agent["reasoning_tokens_legacy_nonexact"] == 17
    assert legacy_agent["reasoning_token_source"] == "legacy_word_count"
    assert exact_agent["reasoning_tokens"] == 17
    assert math.isnan(exact_agent["reasoning_tokens_legacy_nonexact"])
    assert exact_agent["reasoning_token_source"] == "vllm_native_token_ids"


def _censored_row() -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "qid": "q-censored",
        "correct": False,
        "final_answer": None,
        "answer_key": "A",
        "termination_status": TERMINATION_LENGTH_CENSORED,
        "per_agent": [],
        "system_conf": {},
        "self_consistency": {},
        "efficiency_raw": {
            "n_agents": 1,
            "n_turns": 0,
            "n_messages": 0,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 200,
            "total_reasoning_tokens": 0,
            "wall_ms": 2.0,
        },
    }


def test_censors_remain_in_accuracy_but_not_complete_trajectory_means():
    completed = _row(exact=True)
    censored = _censored_row()
    item = item_record("run", "cell", CFG, censored)
    assert item["length_censored"] is True
    assert item["correct"] is False
    assert item["final_answer"] is None

    meta = {
        "cell_id": "cell",
        "config": CFG,
        "started_at": 1.0,
        "finished_at": 2.0,
        "prompt_token_count": 10,
    }
    record = cell_record(
        "run",
        meta,
        [completed, censored],
        n_bad=0,
        n_dupes=0,
        n_raw=2,
        rng=np.random.default_rng(0),
    )
    assert record["accuracy"] == 0.5
    assert record["n_completed_questions"] == 1
    assert record["n_length_censored_questions"] == 1
    assert record["length_censor_rate"] == 0.5
    assert record["mean_turns"] == 1.0
    assert record["mean_messages"] == 0.0
    assert record["mean_total_tokens"] == 50.0
    assert math.isnan(record["mean_reasoning_tokens"])
    assert record["mean_reasoning_tokens_completed_only"] == 17.0
    assert record["reasoning_tokens_exact"] is False
    assert record["reasoning_tokens_completed_only_exact"] is True
    assert record["reasoning_metric_primary_defined"] is False
    assert record["calibration_primary_defined"] is False
    assert math.isnan(record["pa_ece"])
    assert math.isnan(record["delta_vote_prim"])
    assert math.isfinite(record["pa_ece_completed_only"])

    all_censored = cell_record(
        "run",
        meta,
        [censored],
        n_bad=0,
        n_dupes=0,
        n_raw=1,
        rng=np.random.default_rng(1),
    )
    assert all_censored["accuracy"] == 0.0
    assert all_censored["n_completed_questions"] == 0
    assert math.isnan(all_censored["pa_ece"])
    assert math.isnan(all_censored["mean_turns"])
    assert math.isnan(all_censored["mean_reasoning_tokens_completed_only"])


def test_auxiliary_censors_have_separate_rates_without_redefining_primary_reasoning():
    row = _row(exact=True)
    row["self_consistency"] = {
        "sample_count": 5,
        "completed_sample_count": 3,
        "length_censored_sample_count": 1,
        "protocol_censored_sample_count": 1,
        "samples": [
            {
                "termination_status": TERMINATION_LENGTH_CENSORED,
                "agent_output": None,
            },
            {
                "termination_status": TERMINATION_PROTOCOL_CENSORED,
                "agent_output": None,
            },
            *[
                {
                    "termination_status": "completed",
                    "agent_output": {"answer": "A"},
                }
                for _ in range(3)
            ],
        ],
    }

    accounting = _censor_accounting([row])
    assert accounting["n_length_censored_questions"] == 0
    assert accounting["n_protocol_censored_questions"] == 0
    assert accounting["n_auxiliary_samples"] == 5
    assert accounting["n_auxiliary_completed_samples"] == 3
    assert accounting["n_auxiliary_length_censors"] == 1
    assert accounting["n_auxiliary_protocol_censors"] == 1
    assert accounting["auxiliary_length_censor_rate"] == 0.2
    assert accounting["auxiliary_protocol_censor_rate"] == 0.2
    assert accounting["auxiliary_any_censor_rate"] == 0.4
    assert accounting["all_generation_length_censor_rate"] == 1 / 6
    assert accounting["all_generation_protocol_censor_rate"] == 1 / 6
    assert accounting["top_level_uncensored"] is True
    assert accounting["whole_cell_uncensored"] is False

    meta = {
        "cell_id": "cell",
        "config": CFG,
        "started_at": 1.0,
        "finished_at": 2.0,
        "prompt_token_count": 10,
    }
    record = cell_record(
        "run",
        meta,
        [row],
        n_bad=0,
        n_dupes=0,
        n_raw=1,
        rng=np.random.default_rng(2),
    )
    assert record["accuracy"] == 1.0
    assert record["n_completed_questions"] == 1
    assert record["n_auxiliary_length_censors"] == 1
    assert record["n_auxiliary_protocol_censors"] == 1
    assert record["mean_reasoning_tokens"] == 17.0
    assert record["mean_reasoning_tokens_completed_only"] == 17.0
    assert record["reasoning_tokens_exact"] is True
    assert record["reasoning_tokens_completed_only_exact"] is True


def test_canonical_aggregate_keeps_primary_reasoning_separate_from_auxiliary_censor(
    tmp_path, monkeypatch
):
    cell = ExperimentCell(
        model_size="4B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="b512",
        topology="single_agent",
        benchmark="gpqa",
        seed=0,
        n_agents=1,
        n_samples=5,
        n_questions=1,
    )
    row = _row(exact=True)
    row["self_consistency"] = {
        "sample_count": 5,
        "completed_sample_count": 4,
        "length_censored_sample_count": 0,
        "protocol_censored_sample_count": 1,
        "samples": [{}, {}, {}, {}, {}],
    }
    run_root = tmp_path / "run"
    cell_root = run_root / "cells" / cell.cell_id
    cell_root.mkdir(parents=True)
    (cell_root / "meta.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "config": cell.to_dict(),
                "prompt_token_count": 10,
                "prompt_quality": {},
            }
        )
    )
    snapshot = SimpleNamespace(cells=(cell,))
    catalog = SimpleNamespace(
        questions_for=lambda _cell: (SimpleNamespace(qid="q"),),
        frozen=SimpleNamespace(),
        snapshot=snapshot,
        sidecar_sha256="f" * 64,
    )
    monkeypatch.setattr(canonical_analyze.io, "results_root", lambda: tmp_path)
    monkeypatch.setattr(canonical_analyze, "load_manifest", lambda _root: snapshot)
    monkeypatch.setattr(
        canonical_analyze,
        "VerifiedQuestionCatalog",
        lambda _root, snapshot: catalog,
    )
    monkeypatch.setattr(
        canonical_analyze,
        "get_completion_status",
        lambda *_args, **_kwargs: SimpleNamespace(is_complete=True),
    )
    monkeypatch.setattr(
        canonical_analyze,
        "read_canonical_results",
        lambda *_args, **_kwargs: SimpleNamespace(records=(row,)),
    )
    monkeypatch.setattr(
        canonical_analyze,
        "coordination_metrics",
        lambda **_kwargs: SimpleNamespace(to_dict=lambda: {}),
    )

    [record] = canonical_analyze.aggregate_run("run")
    assert record["n_protocol_censored_questions"] == 0
    assert record["n_auxiliary_protocol_censors"] == 1
    assert record["auxiliary_protocol_censor_rate"] == 0.2
    assert record["mean_reasoning_tokens"] == 17.0
    assert record["mean_reasoning_tokens_completed_only"] == 17.0
    assert record["reasoning_metric_primary_defined"] is True
    assert record["top_level_uncensored"] is True
    assert record["whole_cell_uncensored"] is False


def test_efficiency_is_withheld_when_target_or_baseline_has_a_censor():
    cells = pd.DataFrame(
        [
            {
                "model_size": "4B",
                "benchmark": "gpqa",
                "seed": 0,
                "reasoning_level": "b512",
                "prompt_complexity_level": 0,
                "topology": "single_agent",
                "n_length_censored_questions": 0,
                "accuracy": 1.0,
                "error_rate": 0.0,
                "mean_turns": 1.0,
                "mean_messages": 0.0,
                "mean_total_tokens": 50.0,
            },
            {
                "model_size": "4B",
                "benchmark": "gpqa",
                "seed": 0,
                "reasoning_level": "b512",
                "prompt_complexity_level": 0,
                "topology": "centralized",
                "n_length_censored_questions": 1,
                "accuracy": 0.5,
                "error_rate": 0.5,
                "mean_turns": 2.0,
                "mean_messages": 1.0,
                "mean_total_tokens": 100.0,
            },
        ]
    )
    result = add_efficiency(cells)
    assert math.isnan(result.loc[1, "Ec"])
    assert math.isnan(result.loc[1, "Ae"])
    assert math.isnan(result.loc[1, "Opct"])
