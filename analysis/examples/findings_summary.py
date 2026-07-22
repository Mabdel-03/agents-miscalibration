#!/usr/bin/env python
"""Supplementary findings summary over completed legacy full_sweep_v1 cells.

Uses the project's own analyze.aggregate_run() so every number matches the
documented definitions (per-agent option-logprob ECE; system vote_fraction and
final_producer_logprob ECE). Tolerates the partial grid (missing SAS baselines
just leave efficiency=None instead of hard-failing).

This utility describes the partial, mixed-protocol legacy grid and must not be used for
the primary three-run schema-5 analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog
from agents_scaling.experiment import io
from agents_scaling.experiment.analyze import _per_agent_calibration, _system_calibration
from agents_scaling.experiment.completion import get_completion_status, read_canonical_results
from agents_scaling.experiment.manifest import load_manifest
from agents_scaling.serving.profiles import serving_profile_for_cell
from agents_scaling.efficiency.metrics import coordination_metrics
from agents_scaling.models import get_model


# A partial-run findings pass is restricted to the frozen manifest and the same semantic
# completion contract used by the runner/dispatcher. Corrupt, duplicate, and incomplete
# cells are excluded rather than being silently coerced into analysis-ready observations.
_n_bad_lines = 0
_n_bad_cells = 0


def _actual_n_agents(cfg: dict, rows: list[dict]) -> int:
    for r in rows:
        eff = r.get("efficiency_raw") or {}
        if eff.get("n_agents") is not None:
            return int(eff["n_agents"])
    if cfg.get("topology") == "single_agent":
        return 1
    return int(cfg.get("n_agents", 3))


def _robust_aggregate_run(run_id: str) -> list[dict]:
    global _n_bad_lines, _n_bad_cells
    run_root = io.results_root() / run_id
    cells_root = run_root / "cells"
    manifest = load_manifest(run_root)
    question_catalog = VerifiedQuestionCatalog(run_root, snapshot=manifest)
    recs: dict[str, dict] = {}
    for cell in sorted(manifest.cells, key=lambda value: value.cell_id):
        cell_path = cells_root / cell.cell_id
        questions = question_catalog.questions_for(cell)
        status = get_completion_status(
            cell,
            cell_path,
            expected_qids=tuple(question.qid for question in questions),
            expected_questions=questions,
            verified_benchmark_contracts=question_catalog.frozen,
            verified_manifest=question_catalog.snapshot,
            check_active=False,
            serving_profile=serving_profile_for_cell(cell).name,
        )
        if not status.is_complete:
            _n_bad_cells += status.status.value == "corrupt"
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
        n = len(rows)
        acc = sum(1 for r in rows if r["correct"]) / n
        mturns = sum(r["efficiency_raw"]["n_turns"] for r in rows) / n
        mmsgs = sum(r["efficiency_raw"]["n_messages"] for r in rows) / n
        mtok = sum(r["efficiency_raw"]["total_prompt_tokens"] + r["efficiency_raw"]["total_completion_tokens"] for r in rows) / n
        mrt = sum(r["efficiency_raw"].get("total_reasoning_tokens", 0) for r in rows) / n
        cal = {"per_agent": _per_agent_calibration(rows)}
        conf_keys = set()
        for r in rows:
            conf_keys.update((r.get("system_conf") or {}).keys())
        cal["system"] = {ck: _system_calibration(rows, ck) for ck in sorted(conf_keys)}
        recs[meta["cell_id"]] = {
            "cell_id": meta["cell_id"],
            "benchmark_contracts_sha256": question_catalog.sidecar_sha256,
            "model_size": cfg["model_size"],
            "param_count": get_model(cfg["model_size"]).param_count, "topology": cfg["topology"],
            "n_agents": _actual_n_agents(cfg, rows),
            "context_share_level": cfg["context_share_level"],
            "prompt_complexity_level": cfg["prompt_complexity_level"],
            "reasoning_level": cfg.get("reasoning_level", "off"),
            "mean_reasoning_tokens": mrt, "benchmark": cfg["benchmark"], "seed": cfg["seed"],
            "n_questions": n, "accuracy": acc, "error_rate": 1.0 - acc,
            "mean_turns": mturns, "mean_messages": mmsgs, "mean_total_tokens": mtok,
            "calibration": cal,
        }
    # Efficiency vs matched single-agent baseline (same keying as the package).
    sas = {(r["model_size"], r["benchmark"], r["seed"], r["reasoning_level"]): r
           for r in recs.values() if r["topology"] == "single_agent"}
    for r in recs.values():
        base = sas.get((r["model_size"], r["benchmark"], r["seed"], r["reasoning_level"]))
        if base is None:
            r["efficiency"] = None
            continue
        r["efficiency"] = coordination_metrics(
            success_rate=r["accuracy"], error_rate=r["error_rate"], mean_turns=r["mean_turns"],
            mean_messages=r["mean_messages"], sas_turns=base["mean_turns"],
            sas_error_rate=base["error_rate"], mean_total_tokens=r["mean_total_tokens"],
        ).to_dict()
    return list(recs.values())


aggregate_run = _robust_aggregate_run

PA = "per_agent"
VOTE = "vote_fraction"
FP = "final_producer_logprob"


def ece(cal, scope, key=None):
    if not isinstance(cal, dict):
        return None
    if scope == PA:
        r = cal.get("per_agent")
        return float(r["ece"]) if r else None
    sysd = cal.get("system") or {}
    r = sysd.get(key)
    return float(r["ece"]) if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full_sweep_v1")
    args = ap.parse_args()

    recs = aggregate_run(args.run_id)
    rows = []
    for r in recs:
        cal = r["calibration"]
        rows.append({
            "cell_id": r["cell_id"],
            "model_size": r["model_size"],
            "param_count": r["param_count"],
            "topology": r["topology"],
            "n_agents": r["n_agents"],
            "context_share_level": r["context_share_level"],
            "prompt_complexity_level": r["prompt_complexity_level"],
            "reasoning_level": r["reasoning_level"],
            "benchmark": r["benchmark"],
            "seed": r["seed"],
            "n_questions": r["n_questions"],
            "accuracy": r["accuracy"],
            "mean_reasoning_tokens": r["mean_reasoning_tokens"],
            "mean_total_tokens": r["mean_total_tokens"],
            "pa_ece": ece(cal, PA),
            "sys_ece_vote": ece(cal, "system", VOTE),
            "sys_ece_fp": ece(cal, "system", FP),
            "Ec": (r["efficiency"] or {}).get("coordination_efficiency"),
            "Ae": (r["efficiency"] or {}).get("error_amplification"),
            "Opct": (r["efficiency"] or {}).get("overhead_pct"),
        })
    df = pd.DataFrame(rows)
    # ΔECE headline measures (multi-agent cells only have system confidence)
    df["delta_vote"] = df["sys_ece_vote"] - df["pa_ece"]
    df["delta_fp"] = df["sys_ece_fp"] - df["pa_ece"]
    df["is_mas"] = df["topology"] != "single_agent"

    out = f"analysis/{args.run_id}_findings.parquet"
    df.to_parquet(out)

    SIZE_ORDER = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
    df["model_size"] = pd.Categorical(df["model_size"], SIZE_ORDER, ordered=True)
    REAS_ORDER = ["off", "b512", "b2048", "b8192", "unlimited"]
    df["reasoning_level"] = pd.Categorical(df["reasoning_level"], REAS_ORDER, ordered=True)

    P = print
    P("=" * 78)
    if _n_bad_lines or _n_bad_cells:
        P(f"[data integrity] skipped {_n_bad_lines} malformed result line(s) across "
          f"{_n_bad_cells} cell(s) with no valid rows (preemption partial-writes).")
    P(f"FINDINGS — {args.run_id}  ({len(df)} completed cells, {int(df['n_questions'].sum()):,} graded questions)")
    P("=" * 78)
    P(f"\nOverall mean accuracy: {df['accuracy'].mean():.3f}")
    P(f"Mean per-agent ECE:    {df['pa_ece'].mean():.3f}")
    mas = df[df.is_mas]
    P(f"\nHEADLINE  (multi-agent cells, n={len(mas)}):")
    P(f"  mean ΔECE  [vote_fraction system conf]   = {mas['delta_vote'].mean():+.3f}")
    P(f"  mean ΔECE  [final_producer system conf]  = {mas['delta_fp'].mean():+.3f}")
    P(f"  (positive = coordination AMPLIFIES miscalibration; negative = corrects it)")

    def tbl(by, cols=("accuracy", "pa_ece", "sys_ece_vote", "delta_vote", "delta_fp")):
        g = df.groupby(by, observed=True)[list(cols)].mean(numeric_only=True)
        c = df.groupby(by, observed=True).size().rename("n_cells")
        return pd.concat([g, c], axis=1).round(3)

    P("\n--- AXIS 1: MODEL CAPACITY ---")
    P(tbl("model_size").to_string())
    P("\n--- AXIS 2: CONTEXT SHARING (MAS cells) ---")
    P(df[df.is_mas].groupby("context_share_level", observed=True)[
        ["accuracy", "pa_ece", "sys_ece_vote", "delta_vote", "delta_fp"]].mean().round(3).to_string())
    P("\n--- AXIS 3: PROMPT COMPLEXITY ---")
    P(tbl("prompt_complexity_level").to_string())
    P("\n--- AXIS 4: REASONING LEVEL ---")
    P(tbl("reasoning_level").to_string())
    P("\n--- TOPOLOGY ---")
    P(df.groupby("topology")[["accuracy", "pa_ece", "sys_ece_vote", "sys_ece_fp", "delta_vote", "Ec", "Ae", "Opct"]]
      .mean(numeric_only=True).round(3).to_string())
    P("\n--- BENCHMARK ---")
    P(tbl("benchmark").to_string())
    P(f"\n[written] {out}")


if __name__ == "__main__":
    main()
