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
from agents_scaling.efficiency.metrics import coordination_metrics
from agents_scaling.experiment import io
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


def aggregate_run(run_id: str) -> list[dict[str, Any]]:
    """Return one tidy dict per cell. Also computes Kim efficiency vs SAS baselines."""
    cells_root = io.run_dir(run_id) / "cells"
    cell_records: dict[str, dict] = {}
    # First pass: per-cell raw aggregates.
    for cell_path in sorted(cells_root.glob("*")):
        # Need BOTH files: results.jsonl alone (no meta) means a cell is partially run
        # (resume in progress) and shouldn't be aggregated as if complete.
        if not (cell_path / "results.jsonl").exists():
            continue
        if not (cell_path / "meta.json").exists():
            continue
        meta, rows = _read_cell(cell_path)
        cfg = meta["config"]
        n = len(rows)
        acc = sum(1 for r in rows if r["correct"]) / n if n else 0.0
        mean_turns = sum(r["efficiency_raw"]["n_turns"] for r in rows) / n if n else 0.0
        mean_msgs = sum(r["efficiency_raw"]["n_messages"] for r in rows) / n if n else 0.0
        mean_tokens = (
            sum(r["efficiency_raw"]["total_prompt_tokens"] + r["efficiency_raw"]["total_completion_tokens"] for r in rows) / n
            if n else 0.0
        )
        mean_reasoning_tokens = (
            sum(r["efficiency_raw"].get("total_reasoning_tokens", 0) for r in rows) / n if n else 0.0
        )
        # calibration: per-agent + system-level under each confidence definition.
        cal: dict[str, Any] = {"per_agent": _per_agent_calibration(rows)}
        conf_keys = set()
        for r in rows:
            conf_keys.update((r.get("system_conf") or {}).keys())
        cal["system"] = {ck: _system_calibration(rows, ck) for ck in sorted(conf_keys)}

        cell_records[meta["cell_id"]] = {
            "cell_id": meta["cell_id"],
            "model_size": cfg["model_size"],
            "param_count": get_model(cfg["model_size"]).param_count,
            "topology": cfg["topology"],
            "context_share_level": cfg["context_share_level"],
            "prompt_complexity_level": cfg["prompt_complexity_level"],
            "reasoning_level": cfg.get("reasoning_level", "off"),
            "prompt_token_count": meta["prompt_token_count"],
            "prompt_quality": meta["prompt_quality"],
            "mean_reasoning_tokens": mean_reasoning_tokens,
            "benchmark": cfg["benchmark"],
            "seed": cfg["seed"],
            "n_questions": n,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "mean_turns": mean_turns,
            "mean_messages": mean_msgs,
            "mean_total_tokens": mean_tokens,
            "calibration": cal,
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
        if base is None:
            rec["efficiency"] = None  # missing baseline -> flagged by aggregate script
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
