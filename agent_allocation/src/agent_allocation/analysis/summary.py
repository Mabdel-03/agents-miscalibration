"""Fixed-denominator accuracy summaries with equal-benchmark and equal-BBH-subtask weighting.

Weights (config ``statistics``): every declared benchmark present in a population gets equal weight; inside
``bbh_logic4`` each subtask stratum gets an equal share of the BBH weight. A stratum is ``benchmark`` or
``bbh_logic4/<subtask>``. Summaries are computed per stratum and then weighted, so the BBH aggregate is the
mean of its four subtask means even when stratum sizes differ.

For every group the table reports counts by state, the fixed-denominator bounds ``acc_lower`` (unknowns
wrong) and ``acc_upper`` (unknowns correct), ``acc`` (only when nothing is unknown), and ``acc_known_only``
(explicitly partial).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from agent_allocation.analysis.outcomes import ADMINISTRATIVE_STATES, KNOWN_STATES
from agent_allocation.contracts import BBH_SUBTASKS, BENCHMARKS

MACRO = "macro"


class WeightingError(ValueError):
    """The configured weighting cannot be honoured (unsupported knob or unknown stratum)."""


def check_statistics_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read every ``statistics`` knob; refuse settings this module does not implement (R9: read, not ignore)."""
    stats = dict(cfg.get("statistics", {}))
    required = {"bootstrap_replicates", "cluster_unit", "macro_weight_benchmarks_equally",
                "bbh_macro_weight_subtasks_equally", "test_grid_frontier_is_descriptive",
                "no_outcome_based_main_sampling"}
    missing = sorted(required - set(stats))
    if missing:
        raise WeightingError(f"config.statistics lacks {missing}")
    if stats["cluster_unit"] != "benchmark_and_item":
        raise WeightingError(f"unsupported statistics.cluster_unit {stats['cluster_unit']!r}")
    if stats["macro_weight_benchmarks_equally"] is not True:
        raise WeightingError("only equal-benchmark macro weighting is implemented")
    if stats["bbh_macro_weight_subtasks_equally"] is not True:
        raise WeightingError("only equal BBH-subtask weighting is implemented")
    if stats["test_grid_frontier_is_descriptive"] is not True:
        raise WeightingError("the test-grid frontier is descriptive by specification; refusing another setting")
    if int(stats["bootstrap_replicates"]) < 1:
        raise WeightingError("bootstrap_replicates must be positive")
    return stats


def benchmark_of_stratum(stratum: str) -> str:
    return stratum.split("/", 1)[0]


def strata_order(strata: Iterable[str]) -> list[str]:
    order: list[str] = []
    for bench in BENCHMARKS:
        if bench == "bbh_logic4":
            order.extend(f"bbh_logic4/{s}" for s in BBH_SUBTASKS)
        else:
            order.append(bench)
    present = set(strata)
    unknown = sorted(present - set(order))
    if unknown:
        raise WeightingError(f"strata outside the declared benchmark/subtask axis: {unknown}")
    return [s for s in order if s in present]


def stratum_weights(strata: Sequence[str], *, within: str | None = None) -> dict[str, float]:
    """Weights over the given strata: equal per benchmark, equal per BBH subtask inside BBH.

    ``within`` restricts to one benchmark (the BBH aggregate uses equal subtask weights).
    """
    ordered = strata_order(strata)
    if within is not None:
        ordered = [s for s in ordered if benchmark_of_stratum(s) == within]
    benches = sorted({benchmark_of_stratum(s) for s in ordered}, key=BENCHMARKS.index)
    if not benches:
        return {}
    weights: dict[str, float] = {}
    for bench in benches:
        members = [s for s in ordered if benchmark_of_stratum(s) == bench]
        for s in members:
            weights[s] = 1.0 / len(benches) / len(members)
    return weights


def _stratum_counts(group: pd.DataFrame) -> dict[str, Any]:
    states = group["delivered_state"].value_counts().to_dict()
    n = int(len(group))
    n_correct = int(states.get("correct", 0))
    n_wrong = int(states.get("wrong", 0))
    n_op = int(sum(states.get(s, 0) for s in KNOWN_STATES if s not in ("correct", "wrong")))
    n_admin = int(sum(states.get(s, 0) for s in ADMINISTRATIVE_STATES))
    n_unknown = n - n_correct - n_wrong - n_op - n_admin
    return {"n_planned": n, "n_correct": n_correct, "n_wrong": n_wrong, "n_op_failure": n_op,
            "n_unknown": n_unknown, "n_not_run": n_admin}


def _weighted(parts: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float]) -> dict[str, Any]:
    out = {k: int(sum(parts[s][k] for s in weights)) for k in
           ("n_planned", "n_correct", "n_wrong", "n_op_failure", "n_unknown", "n_not_run")}
    lower = math.fsum(weights[s] * parts[s]["n_correct"] / parts[s]["n_planned"] for s in weights)
    upper = math.fsum(weights[s] * (parts[s]["n_planned"] - parts[s]["n_wrong"] - parts[s]["n_op_failure"])
                      / parts[s]["n_planned"] for s in weights)
    complete = all(parts[s]["n_unknown"] == 0 and parts[s]["n_not_run"] == 0 for s in weights)
    known_ok = all(parts[s]["n_correct"] + parts[s]["n_wrong"] + parts[s]["n_op_failure"] > 0 for s in weights)
    known_only = (math.fsum(weights[s] * parts[s]["n_correct"]
                            / (parts[s]["n_correct"] + parts[s]["n_wrong"] + parts[s]["n_op_failure"])
                            for s in weights) if known_ok else None)
    out.update({"acc_lower": lower, "acc_upper": upper, "acc": lower if complete else None,
                "acc_known_only": known_only, "complete": complete, "n_strata": len(weights)})
    return out


def accuracy_summary(outcomes: pd.DataFrame, *, by: Sequence[str] = ("partition_id", "name", "view")
                     ) -> pd.DataFrame:
    """Rows at level ``stratum``, ``benchmark`` and ``macro`` for each ``by`` group."""
    rows: list[dict[str, Any]] = []
    if outcomes.empty:
        return pd.DataFrame(rows)
    for keys, group in outcomes.groupby(list(by), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys))
        parts = {stratum: _stratum_counts(g) for stratum, g in group.groupby("stratum", sort=False)}
        ordered = strata_order(parts)
        for stratum in ordered:
            rows.append({**base, "level": "stratum", "benchmark": benchmark_of_stratum(stratum),
                         "stratum": stratum, **_weighted(parts, {stratum: 1.0})})
        benches = sorted({benchmark_of_stratum(s) for s in ordered}, key=BENCHMARKS.index)
        for bench in benches:
            rows.append({**base, "level": "benchmark", "benchmark": bench, "stratum": bench,
                         **_weighted(parts, stratum_weights(ordered, within=bench))})
        macro = _weighted(parts, stratum_weights(ordered))
        macro["macro_benchmarks"] = ",".join(benches)
        rows.append({**base, "level": MACRO, "benchmark": MACRO, "stratum": MACRO, **macro})
    frame = pd.DataFrame(rows)
    if "macro_benchmarks" not in frame:
        frame["macro_benchmarks"] = None
    return frame


def weighted_means(frame: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    """Equal-benchmark / equal-BBH-subtask weighted means of item-level columns at benchmark and macro level.

    Each value uses the items where it is known; ``n_<col>_unknown`` counts the rest and ``<col>_complete`` says
    whether the fixed population was fully known.
    """
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(rows)
    for keys, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        strata = strata_order(set(group["stratum"].dropna()))
        levels = [("macro", stratum_weights(strata))] + [
            (b, stratum_weights(strata, within=b)) for b in sorted({benchmark_of_stratum(s) for s in strata},
                                                                   key=BENCHMARKS.index)]
        by_stratum = {s: group[group["stratum"] == s] for s in strata}
        for level, weights in levels:
            row = {**base, "level": "macro" if level == "macro" else "benchmark", "benchmark": level,
                   "n_items": int(sum(len(by_stratum[s]) for s in weights))}
            for col in value_cols:
                parts = []
                unknown = 0
                for s, w in weights.items():
                    values = pd.to_numeric(by_stratum[s][col], errors="coerce").astype(float)
                    unknown += int(values.isna().sum())
                    parts.append((w, float(values.mean()) if values.notna().any() else math.nan))
                row[col] = None if any(math.isnan(v) for _, v in parts) else math.fsum(w * v for w, v in parts)
                row[f"n_{col}_unknown"] = unknown
                row[f"{col}_complete"] = unknown == 0
            rows.append(row)
    return pd.DataFrame(rows)
