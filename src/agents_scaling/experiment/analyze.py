"""Aggregate per-cell JSONL into tidy per-cell metrics (performance, efficiency, calibration).

Walks ``<run>/cells/*/results.jsonl`` + ``meta.json``, then for each cell computes:
  * performance: accuracy.
  * calibration: ECE/MCE/Brier for each (confidence-signal x scope) where data exists —
    per-agent AND system-level, under each system-confidence definition.
  * efficiency: Kim metrics vs the matched single-agent baseline for that (model, benchmark).

The HEADLINE quantity (system ECE - mean per-agent ECE) per axis is computed downstream
in the analysis notebooks from this tidy table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents_scaling.calibration.metrics import compute_calibration
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog
from agents_scaling.efficiency.metrics import coordination_metrics
from agents_scaling.experiment import io
from agents_scaling.experiment.completion import (
    get_completion_status,
    read_canonical_results,
    reasoning_token_summary,
)
from agents_scaling.experiment.manifest import load_manifest
from agents_scaling.experiment.result_schema import (
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)
from agents_scaling.serving.profiles import serving_profile_for_cell
from agents_scaling.models import get_model


def _read_cell(cell_path: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((cell_path / "meta.json").read_text())
    rows = [json.loads(line) for line in (cell_path / "results.jsonl").read_text().splitlines() if line.strip()]
    return meta, rows


def _per_agent_calibration(rows: list[dict], n_bins: int = 15) -> dict[str, Any] | None:
    """ECE over individual agent answers, using each agent's option-logprob confidence."""
    confs, corrects = [], []
    for r in rows:
        key = r["answer_key"]
        for a in r["per_agent"]:
            ans = a.get("answer")
            lp = a.get("option_logprobs") or {}
            if ans is None or not lp:
                continue
            confs.append(float(lp.get(ans, 0.0)))     # confidence in its OWN choice
            corrects.append(ans == key)
    if not confs:
        return None
    return compute_calibration(confs, corrects, n_bins=n_bins).to_dict()


def _system_calibration(rows: list[dict], conf_key: str, n_bins: int = 15) -> dict[str, Any] | None:
    confs, corrects = [], []
    for r in rows:
        sc = r.get("system_conf") or {}
        if conf_key not in sc:
            continue
        confs.append(float(sc[conf_key]))
        corrects.append(bool(r["correct"]))
    if not confs:
        return None
    return compute_calibration(confs, corrects, n_bins=n_bins).to_dict()


def _actual_n_agents(cfg: dict, rows: list[dict]) -> int:
    """Prefer the executed topology count over legacy config defaults."""
    for r in rows:
        eff = r.get("efficiency_raw") or {}
        if eff.get("n_agents") is not None:
            return int(eff["n_agents"])
    if cfg.get("topology") == "single_agent":
        return 1
    return int(cfg.get("n_agents", 3))


def _nonnegative_count(value: Any, *, fallback: int = 0) -> int:
    """Read a validated count while remaining compatible with schema-less rows."""

    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else fallback
    )


def _censor_accounting(rows: list[dict]) -> dict[str, Any]:
    """Return denominator-explicit primary and auxiliary censor accounting.

    Current-schema rows have already passed semantic completion validation before this
    helper is called.  The fallbacks exist only for historical schema-less rows, whose
    self-consistency payloads represented completed samples without explicit counters.
    Top-level QID outcomes and auxiliary self-consistency draws are intentionally kept
    separate because they have different experimental denominators.
    """

    n_questions = len(rows)
    n_completed = sum(
        row.get("termination_status", TERMINATION_COMPLETED)
        == TERMINATION_COMPLETED
        for row in rows
    )
    n_length = sum(
        row.get("termination_status") == TERMINATION_LENGTH_CENSORED
        for row in rows
    )
    n_protocol = sum(
        row.get("termination_status") == TERMINATION_PROTOCOL_CENSORED
        for row in rows
    )

    auxiliary_samples = 0
    auxiliary_completed = 0
    auxiliary_length = 0
    auxiliary_protocol = 0
    for row in rows:
        value = row.get("self_consistency") or {}
        if not isinstance(value, dict) or not value:
            continue
        samples = value.get("samples")
        legacy_sample_count = len(samples) if isinstance(samples, list) else 0
        sample_count = _nonnegative_count(
            value.get("sample_count"), fallback=legacy_sample_count
        )
        length_count = _nonnegative_count(
            value.get("length_censored_sample_count")
        )
        protocol_count = _nonnegative_count(
            value.get("protocol_censored_sample_count")
        )
        completed_count = _nonnegative_count(
            value.get("completed_sample_count"),
            fallback=max(0, sample_count - length_count - protocol_count),
        )
        auxiliary_samples += sample_count
        auxiliary_completed += completed_count
        auxiliary_length += length_count
        auxiliary_protocol += protocol_count

    top_level_any = n_length + n_protocol
    auxiliary_any = auxiliary_length + auxiliary_protocol
    all_generation_outcomes = n_questions + auxiliary_samples
    return {
        "n_questions": n_questions,
        "n_completed_questions": n_completed,
        "n_length_censored_questions": n_length,
        "n_protocol_censored_questions": n_protocol,
        "length_censor_rate": n_length / n_questions if n_questions else 0.0,
        "protocol_censor_rate": n_protocol / n_questions if n_questions else 0.0,
        # Compatibility: ``any_censor_rate`` has always meant top-level QIDs only.
        "any_censor_rate": top_level_any / n_questions if n_questions else 0.0,
        "n_auxiliary_samples": auxiliary_samples,
        "n_auxiliary_completed_samples": auxiliary_completed,
        "n_auxiliary_length_censors": auxiliary_length,
        "n_auxiliary_protocol_censors": auxiliary_protocol,
        "auxiliary_length_censor_rate": (
            auxiliary_length / auxiliary_samples if auxiliary_samples else None
        ),
        "auxiliary_protocol_censor_rate": (
            auxiliary_protocol / auxiliary_samples if auxiliary_samples else None
        ),
        "auxiliary_any_censor_rate": (
            auxiliary_any / auxiliary_samples if auxiliary_samples else None
        ),
        "all_generation_length_censor_rate": (
            (n_length + auxiliary_length) / all_generation_outcomes
            if all_generation_outcomes
            else 0.0
        ),
        "all_generation_protocol_censor_rate": (
            (n_protocol + auxiliary_protocol) / all_generation_outcomes
            if all_generation_outcomes
            else 0.0
        ),
        "top_level_uncensored": top_level_any == 0,
        "whole_cell_uncensored": top_level_any + auxiliary_any == 0,
    }


def aggregate_run(run_id: str) -> list[dict[str, Any]]:
    """Return one tidy dict per cell. Also computes Kim efficiency vs SAS baselines."""
    run_root = io.results_root() / run_id
    cells_root = run_root / "cells"
    manifest = load_manifest(run_root)
    question_catalog = VerifiedQuestionCatalog(run_root, snapshot=manifest)
    cell_records: dict[str, dict] = {}
    # First pass: per-cell raw aggregates.
    for cell in sorted(manifest.cells, key=lambda value: value.cell_id):
        cell_path = cells_root / cell.cell_id
        # Need BOTH files: results.jsonl alone (no meta) means a cell is partially run
        # (resume in progress) and shouldn't be aggregated as if complete.
        profile = serving_profile_for_cell(cell)
        questions = question_catalog.questions_for(cell)
        status = get_completion_status(
            cell,
            cell_path,
            expected_qids=tuple(question.qid for question in questions),
            expected_questions=questions,
            verified_benchmark_contracts=question_catalog.frozen,
            verified_manifest=question_catalog.snapshot,
            check_active=False,
            serving_profile=profile.name,
        )
        if not status.is_complete:
            continue
        meta = json.loads((cell_path / "meta.json").read_text())
        rows = list(
            read_canonical_results(
                cell,
                cell_path,
                expected_qids=tuple(question.qid for question in questions),
                expected_questions=questions,
                verified_benchmark_contracts=question_catalog.frozen,
                verified_manifest=question_catalog.snapshot,
            ).records
        )
        cfg = meta["config"]
        censor_accounting = _censor_accounting(rows)
        n = censor_accounting["n_questions"]
        completed_rows = [
            row
            for row in rows
            if row.get("termination_status", TERMINATION_COMPLETED)
            == TERMINATION_COMPLETED
        ]
        n_completed = censor_accounting["n_completed_questions"]
        n_censored = censor_accounting["n_length_censored_questions"]
        n_protocol_censored = censor_accounting[
            "n_protocol_censored_questions"
        ]
        n_any_censored = n_censored + n_protocol_censored
        acc = sum(1 for r in rows if r["correct"]) / n if n else 0.0
        # Topology efficiency is undefined for a trajectory censored before the system
        # produced an answer.  Keep the consumed censor tokens in the raw row, but never
        # dilute complete-topology means with a synthetic zero-turn observation.
        mean_turns = (
            sum(r["efficiency_raw"]["n_turns"] for r in completed_rows) / n_completed
            if n_completed
            else 0.0
        )
        mean_msgs = (
            sum(r["efficiency_raw"]["n_messages"] for r in completed_rows) / n_completed
            if n_completed
            else 0.0
        )
        mean_tokens = (
            sum(
                r["efficiency_raw"]["total_prompt_tokens"]
                + r["efficiency_raw"]["total_completion_tokens"]
                for r in completed_rows
            )
            / n_completed
            if n_completed
            else 0.0
        )
        completed_reasoning = reasoning_token_summary(completed_rows)
        observed_mean_reasoning_tokens = (
            sum(
                r["efficiency_raw"].get("total_reasoning_tokens", 0)
                for r in completed_rows
            )
            / n_completed
            if n_completed
            else None
        )
        reasoning_token_sources = sorted(
            {
                str(agent.get("reasoning_token_source") or "legacy_word_count")
                for row in completed_rows
                for agent in row.get("per_agent", [])
            }
        )
        reasoning_tokens_completed_only_exact = completed_reasoning.all_exact
        reasoning_tokens_exact = bool(
            reasoning_tokens_completed_only_exact
            and censor_accounting["top_level_uncensored"]
            and n_completed == n
        )
        # calibration: per-agent + system-level under each confidence definition.
        conditional_cal: dict[str, Any] = {
            "per_agent": _per_agent_calibration(completed_rows)
        }
        conf_keys = set()
        for r in completed_rows:
            conf_keys.update((r.get("system_conf") or {}).keys())
        conditional_cal["system"] = {
            ck: _system_calibration(completed_rows, ck) for ck in sorted(conf_keys)
        }
        if n_any_censored:
            # Confidence is structurally absent for censored trajectories.  Computing
            # the headline ECE on only terminating rows would silently change its
            # estimand and can induce outcome-dependent selection bias.
            cal: dict[str, Any] = {
                "per_agent": None,
                "system": {ck: None for ck in sorted(conf_keys)},
                "status": "undefined_due_to_censor",
            }
            calibration_completed_only: dict[str, Any] | None = conditional_cal
        else:
            cal = conditional_cal | {"status": "defined_full_qid_coverage"}
            calibration_completed_only = None

        cell_records[meta["cell_id"]] = {
            "cell_id": meta["cell_id"],
            "benchmark_contracts_sha256": question_catalog.sidecar_sha256,
            "model_size": cfg["model_size"],
            "param_count": get_model(cfg["model_size"]).param_count,
            "topology": cfg["topology"],
            "n_agents": _actual_n_agents(cfg, rows),
            "context_share_level": cfg["context_share_level"],
            "prompt_complexity_level": cfg["prompt_complexity_level"],
            "reasoning_level": cfg.get("reasoning_level", "off"),
            "prompt_token_count": meta["prompt_token_count"],
            "prompt_quality": meta["prompt_quality"],
            # Never pool historical whitespace counts with exact token-ID metrics.
            # The compatibility column remains the primary all-question estimand.  It
            # is null whenever a primary topology draw censored; auxiliary sampling
            # costs have a separate estimand.  The observed terminating-row statistic
            # lives only under an explicit completed-only name so downstream fits cannot
            # silently condition on primary termination.
            "mean_reasoning_tokens": (
                completed_reasoning.mean_reasoning_tokens
                if reasoning_tokens_exact
                else None
            ),
            "mean_reasoning_tokens_completed_only": (
                completed_reasoning.mean_reasoning_tokens
                if reasoning_tokens_completed_only_exact
                else None
            ),
            "mean_reasoning_tokens_legacy_or_mixed": (
                observed_mean_reasoning_tokens
                if (
                    censor_accounting["top_level_uncensored"]
                    and not reasoning_tokens_completed_only_exact
                )
                else None
            ),
            "mean_reasoning_tokens_legacy_or_mixed_completed_only": (
                observed_mean_reasoning_tokens
                if not reasoning_tokens_completed_only_exact
                else None
            ),
            "reasoning_tokens_exact": reasoning_tokens_exact,
            "reasoning_tokens_completed_only_exact": (
                reasoning_tokens_completed_only_exact
            ),
            "reasoning_metric_primary_defined": reasoning_tokens_exact,
            "reasoning_token_sources": reasoning_token_sources,
            "benchmark": cfg["benchmark"],
            "seed": cfg["seed"],
            **censor_accounting,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "mean_turns": mean_turns,
            "mean_messages": mean_msgs,
            "mean_total_tokens": mean_tokens,
            "calibration": cal,
            "calibration_completed_only": calibration_completed_only,
        }

    # Baseline lookup: SAS per (model_size, benchmark, seed, reasoning_level). Efficiency
    # is reasoning-conditioned, so each reasoning level has its own single-agent baseline.
    def _base_key(rec: dict) -> tuple:
        return (rec["model_size"], rec["benchmark"], rec["seed"], rec["reasoning_level"])

    sas: dict[tuple, dict] = {}
    for rec in cell_records.values():
        if rec["topology"] == "single_agent":
            sas[_base_key(rec)] = rec

    # Second pass: efficiency vs the matched baseline.
    for rec in cell_records.values():
        base = sas.get(_base_key(rec))
        if (
            base is None
            or rec["n_length_censored_questions"]
            or rec["n_protocol_censored_questions"]
            or base["n_length_censored_questions"]
            or base["n_protocol_censored_questions"]
        ):
            # Missing baselines and censored trajectories both make the coordination
            # efficiency ratio non-comparable; the censor rate remains explicit above.
            rec["efficiency"] = None
            continue
        eff = coordination_metrics(
            success_rate=rec["accuracy"],
            error_rate=rec["error_rate"],
            mean_turns=rec["mean_turns"],
            mean_messages=rec["mean_messages"],
            sas_turns=base["mean_turns"],
            sas_error_rate=base["error_rate"],
            mean_total_tokens=rec["mean_total_tokens"],
        )
        rec["efficiency"] = eff.to_dict()

    return list(cell_records.values())
