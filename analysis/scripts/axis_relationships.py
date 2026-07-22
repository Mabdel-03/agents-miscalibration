#!/usr/bin/env python3
"""Supplementary axis EDA for the partial, mixed-protocol v1 evidence.

This is not a primary schema-5 analysis. The script reads only
``analysis/cache/supplementary_legacy/analysis_view_v1.parquet`` and writes:

* analysis/tables/axis_relationships/axis_summary_long.csv
* analysis/tables/axis_relationships/axis_matched_contrasts.csv
* analysis/tables/axis_relationships/axis_stratified_contrasts.csv
* analysis/figures/axis_relationships/*.png

Estimates are means over design-cell seed means. This prevents strata with more
completed seeds from receiving more weight in the headline summaries. CIs are
nonparametric bootstraps over design cells or matched design pairs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.nb_lib import data as nb_data  # noqa: E402
from analysis.nb_lib import plots  # noqa: E402


OUT_FIG = ROOT / "analysis" / "figures" / "axis_relationships"
OUT_TAB = ROOT / "analysis" / "tables" / "axis_relationships"

BENCHES = nb_data.BENCHMARKS
SIZE_ORDER = nb_data.SIZE_ORDER
REAS_ORDER = nb_data.REAS_ORDER
TOPO_ORDER = nb_data.TOPO_ORDER
CTX_ORDER = ["artifact_only", "plus_cot"]
PROMPT_ORDER = [0, 3]
AGENT_ORDER = nb_data.AGENT_ORDER

AXIS_ORDERS = {
    "model_size": SIZE_ORDER,
    "reasoning_level": REAS_ORDER,
    "topology": TOPO_ORDER,
    "prompt_complexity_level": PROMPT_ORDER,
    "context_share_level": CTX_ORDER,
    "n_agents": AGENT_ORDER,
}

AXIS_LABELS = {
    "model_size": "Model size",
    "reasoning_level": "Reasoning budget",
    "topology": "Topology",
    "prompt_complexity_level": "Prompt complexity",
    "context_share_level": "Context sharing",
    "n_agents": "Agent count",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "pa_ece_prim": "Per-agent ECE",
    "vote_ece_prim": "Vote ECE",
    "fp_ece_prim": "Final-producer ECE",
    "delta_vote_prim": "Delta ECE: vote - agent",
    "delta_fp_prim": "Delta ECE: final - agent",
    "pa_signed_gap": "Agent signed gap",
    "vote_signed_gap": "Vote signed gap",
    "cost": "Cost proxy",
    "mean_total_tokens": "Total tokens",
    "mean_reasoning_tokens": "Reasoning tokens",
    "Ec": "Coordination efficiency Ec",
    "Ae": "Error amplification Ae",
}

SUMMARY_METRICS = [
    "accuracy",
    "pa_ece_prim",
    "vote_ece_prim",
    "fp_ece_prim",
    "delta_vote_prim",
    "delta_fp_prim",
    "pa_signed_gap",
    "vote_signed_gap",
    "cost",
    "mean_total_tokens",
    "mean_reasoning_tokens",
    "Ec",
    "Ae",
]

CONTRAST_METRICS = [
    "accuracy",
    "pa_ece_prim",
    "vote_ece_prim",
    "fp_ece_prim",
    "delta_vote_prim",
    "delta_fp_prim",
    "pa_signed_gap",
    "vote_signed_gap",
    "cost",
    "Ec",
    "Ae",
]

STRATIFIED_METRICS = [
    "accuracy",
    "pa_ece_prim",
    "vote_ece_prim",
    "fp_ece_prim",
    "delta_vote_prim",
    "delta_fp_prim",
    "cost",
    "Ec",
    "Ae",
]


def _fmt_level(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _ordered_levels(axis: str, observed: pd.Series) -> list:
    order = AXIS_ORDERS.get(axis)
    vals = set(observed.dropna().astype(str))
    if order is None:
        return sorted(vals)
    return [x for x in order if str(x) in vals]


def _bootstrap_mean(values: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    dist = values[idx].mean(axis=1)
    return (
        float(np.nanmean(values)),
        float(np.nanpercentile(dist, 2.5)),
        float(np.nanpercentile(dist, 97.5)),
    )


def design_cell_means(
    df: pd.DataFrame,
    axis: str,
    metric: str,
    *,
    axis_scope: str,
    B: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return pd.DataFrame()
    for bench in BENCHES:
        db = d[d["benchmark"].astype(str) == bench]
        for level in _ordered_levels(axis, db[axis]):
            dl = db[db[axis].astype(str) == str(level)]
            if dl.empty:
                continue
            design = dl.groupby("design_cell", observed=True).agg(
                value=(metric, "mean"),
                n_seed_cells=(metric, "size"),
            )
            mean, lo, hi = _bootstrap_mean(
                design["value"].to_numpy(float),
                B=B,
                seed=seed + len(rows) * 17 + 11,
            )
            rows.append({
                "axis": axis,
                "axis_scope": axis_scope,
                "level": _fmt_level(level),
                "benchmark": bench,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_designs": int(len(design)),
                "n_seed_cells": int(design["n_seed_cells"].sum()),
            })
    return pd.DataFrame(rows)


def matched_contrast(
    df: pd.DataFrame,
    *,
    axis: str,
    low,
    high,
    metric: str,
    match_cols: list[str],
    label: str,
    axis_scope: str,
    filt: pd.Series | None = None,
    B: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    d = df.copy()
    if filt is not None:
        d = d[filt].copy()
    d = d.dropna(subset=[metric])
    d = d[d[axis].astype(str).isin([str(low), str(high)])].copy()
    rows = []
    for bench in BENCHES:
        db = d[d["benchmark"].astype(str) == bench]
        if db.empty:
            continue
        left = db[db[axis].astype(str) == str(low)]
        right = db[db[axis].astype(str) == str(high)]
        keep = match_cols + [metric]
        left = left[keep].rename(columns={metric: "low"})
        right = right[keep].rename(columns={metric: "high"})
        pair = left.merge(right, on=match_cols, how="inner")
        if pair.empty:
            continue
        pair["delta"] = pair["high"] - pair["low"]
        design_cols = [c for c in match_cols if c != "seed"]
        design_delta = pair.groupby(design_cols, observed=True)["delta"].mean().to_numpy(float)
        mean, lo, hi = _bootstrap_mean(
            design_delta,
            B=B,
            seed=seed + len(rows) * 23 + 19,
        )
        rows.append({
            "axis": axis,
            "axis_scope": axis_scope,
            "contrast": label,
            "low": _fmt_level(low),
            "high": _fmt_level(high),
            "benchmark": bench,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_matched_seed_pairs": int(len(pair)),
            "n_matched_design_pairs": int(len(design_delta)),
        })
    return pd.DataFrame(rows)


def agent_family_vs_single(
    df: pd.DataFrame,
    *,
    high: int,
    metric: str,
    B: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    d = df[
        (df["context_share_level"].astype(str) == "artifact_only")
        & df["topology"].astype(str).isin(TOPO_ORDER)
    ].dropna(subset=[metric]).copy()
    rows = []
    key = ["benchmark", "model_size", "reasoning_level", "prompt_complexity_level", "seed"]
    for bench in BENCHES:
        db = d[d["benchmark"].astype(str) == bench]
        sas = db[db["n_agents"] == 1][key + [metric]].rename(columns={metric: "single"})
        mas = db[db["n_agents"] == high].groupby(key, observed=True)[metric].mean().reset_index(
            name="agent_family"
        )
        pair = sas.merge(mas, on=key, how="inner")
        if pair.empty:
            continue
        pair["delta"] = pair["agent_family"] - pair["single"]
        design_cols = [c for c in key if c != "seed"]
        design_delta = pair.groupby(design_cols, observed=True)["delta"].mean().to_numpy(float)
        mean, lo, hi = _bootstrap_mean(design_delta, B=B, seed=seed + len(rows) * 29 + 7)
        rows.append({
            "axis": "n_agents",
            "axis_scope": "artifact_only; MAS rows averaged over available topologies at fixed agent count",
            "contrast": f"{high}-agent family - 1-agent single",
            "low": "1",
            "high": str(high),
            "benchmark": bench,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_matched_seed_pairs": int(len(pair)),
            "n_matched_design_pairs": int(len(design_delta)),
        })
    return pd.DataFrame(rows)


def matched_contrast_strata(
    df: pd.DataFrame,
    *,
    axis: str,
    low,
    high,
    metric: str,
    match_cols: list[str],
    group_cols: list[str],
    label: str,
    axis_scope: str,
    filt: pd.Series | None = None,
    B: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Matched high-low contrast estimated separately within group_cols strata."""
    d = df.copy()
    if filt is not None:
        d = d[filt].copy()
    d = d.dropna(subset=[metric])
    d = d[d[axis].astype(str).isin([str(low), str(high)])].copy()
    if d.empty:
        return pd.DataFrame()

    left = d[d[axis].astype(str) == str(low)][match_cols + [metric]].rename(
        columns={metric: "low"}
    )
    right = d[d[axis].astype(str) == str(high)][match_cols + [metric]].rename(
        columns={metric: "high"}
    )
    pair = left.merge(right, on=match_cols, how="inner")
    if pair.empty:
        return pd.DataFrame()
    pair["delta"] = pair["high"] - pair["low"]

    rows = []
    design_cols = [c for c in match_cols if c != "seed"]
    for group_key, dg in pair.groupby(group_cols, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        design_delta = dg.groupby(design_cols, observed=True)["delta"].mean().to_numpy(float)
        mean, lo, hi = _bootstrap_mean(
            design_delta,
            B=B,
            seed=seed + len(rows) * 31 + 13,
        )
        row = {
            "axis": axis,
            "axis_scope": axis_scope,
            "contrast": label,
            "low": _fmt_level(low),
            "high": _fmt_level(high),
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_matched_seed_pairs": int(len(dg)),
            "n_matched_design_pairs": int(len(design_delta)),
        }
        row.update({c: _fmt_level(v) for c, v in zip(group_cols, group_key)})
        rows.append(row)
    return pd.DataFrame(rows)


def agent_family_vs_single_strata(
    df: pd.DataFrame,
    *,
    high: int,
    metric: str,
    group_cols: list[str],
    B: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """N-agent family vs single-agent, separately by benchmark/model_size/reasoning."""
    d = df[
        (df["context_share_level"].astype(str) == "artifact_only")
        & df["topology"].astype(str).isin(TOPO_ORDER)
    ].dropna(subset=[metric]).copy()
    key = ["benchmark", "model_size", "reasoning_level", "prompt_complexity_level", "seed"]
    sas = d[d["n_agents"] == 1][key + [metric]].rename(columns={metric: "single"})
    mas = d[d["n_agents"] == high].groupby(key, observed=True)[metric].mean().reset_index(
        name="agent_family"
    )
    pair = sas.merge(mas, on=key, how="inner")
    if pair.empty:
        return pd.DataFrame()
    pair["delta"] = pair["agent_family"] - pair["single"]

    rows = []
    design_cols = [c for c in key if c != "seed"]
    for group_key, dg in pair.groupby(group_cols, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        design_delta = dg.groupby(design_cols, observed=True)["delta"].mean().to_numpy(float)
        mean, lo, hi = _bootstrap_mean(
            design_delta,
            B=B,
            seed=seed + len(rows) * 37 + 5,
        )
        row = {
            "axis": "n_agents",
            "axis_scope": "artifact_only; MAS rows averaged over available topologies at fixed agent count",
            "contrast": f"{high}-agent family - 1-agent single",
            "low": "1",
            "high": str(high),
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n_matched_seed_pairs": int(len(dg)),
            "n_matched_design_pairs": int(len(design_delta)),
        }
        row.update({c: _fmt_level(v) for c, v in zip(group_cols, group_key)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    axis_specs = [
        ("model_size", "all core rows", df.index == df.index),
        ("reasoning_level", "all core rows", df.index == df.index),
        ("topology", "all core rows", df.index == df.index),
        ("prompt_complexity_level", "all core rows", df.index == df.index),
        ("n_agents", "all core rows; agent count is confounded with topology", df.index == df.index),
        (
            "context_share_level",
            "communication topologies only: centralized/decentralized",
            df["topology"].astype(str).isin(["centralized", "decentralized"]),
        ),
    ]
    for axis, scope, filt in axis_specs:
        for metric in SUMMARY_METRICS:
            frames.append(design_cell_means(df[filt], axis, metric, axis_scope=scope))
    return pd.concat([x for x in frames if not x.empty], ignore_index=True)


def build_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    other = ["benchmark", "topology", "n_agents", "context_share_level",
             "prompt_complexity_level", "reasoning_level", "seed"]
    size_pairs = list(zip(SIZE_ORDER[:-1], SIZE_ORDER[1:])) + [
        ("0.6B", "14B"),
        ("0.6B", "32B"),
    ]
    for low, high in size_pairs:
        for metric in CONTRAST_METRICS:
            frames.append(matched_contrast(
                df,
                axis="model_size",
                low=low,
                high=high,
                metric=metric,
                match_cols=other,
                label=f"{high} - {low}",
                axis_scope="adjacent sizes matched on topology/agent_count/context/prompt/reasoning/seed",
            ))

    other = ["benchmark", "model_size", "topology", "n_agents", "context_share_level",
             "prompt_complexity_level", "seed"]
    reasoning_pairs = list(zip(REAS_ORDER[:-1], REAS_ORDER[1:])) + [
        ("off", "b2048"),
        ("off", "b8192"),
        ("off", "unlimited"),
    ]
    for low, high in reasoning_pairs:
        for metric in CONTRAST_METRICS:
            frames.append(matched_contrast(
                df,
                axis="reasoning_level",
                low=low,
                high=high,
                metric=metric,
                match_cols=other,
                label=f"{high} - {low}",
                axis_scope="reasoning levels matched on size/topology/agent_count/context/prompt/seed",
            ))

    other = ["benchmark", "model_size", "topology", "n_agents", "context_share_level",
             "reasoning_level", "seed"]
    for metric in CONTRAST_METRICS:
        frames.append(matched_contrast(
            df,
            axis="prompt_complexity_level",
            low=0,
            high=3,
            metric=metric,
            match_cols=other,
            label="L3 - L0",
            axis_scope="prompt levels matched on size/topology/agent_count/context/reasoning/seed",
        ))

    comm = df["topology"].astype(str).isin(["centralized", "decentralized"])
    other = ["benchmark", "model_size", "topology", "n_agents", "prompt_complexity_level",
             "reasoning_level", "seed"]
    for metric in CONTRAST_METRICS:
        frames.append(matched_contrast(
            df,
            axis="context_share_level",
            low="artifact_only",
            high="plus_cot",
            metric=metric,
            match_cols=other,
            label="plus_cot - artifact_only",
            axis_scope="centralized/decentralized only, matched within topology and agent count",
            filt=comm,
        ))

    artifact = df["context_share_level"].astype(str) == "artifact_only"
    single_topo_pairs = [
        ("single_agent", "independent"),
        ("single_agent", "decentralized"),
        ("single_agent", "centralized"),
    ]
    other = ["benchmark", "model_size", "context_share_level", "prompt_complexity_level",
             "reasoning_level", "seed"]
    single_vs_three = artifact & (
        (df["topology"].astype(str) == "single_agent") | (df["n_agents"] == 3)
    )
    for low, high in single_topo_pairs:
        for metric in CONTRAST_METRICS:
            frames.append(matched_contrast(
                df,
                axis="topology",
                low=low,
                high=high,
                metric=metric,
                match_cols=other,
                label=f"{high} - {low}",
                axis_scope="artifact_only; single-agent vs 3-agent topology matched on size/prompt/reasoning/seed",
                filt=single_vs_three,
            ))

    mas_topo_pairs = [
        ("independent", "decentralized"),
        ("independent", "centralized"),
        ("decentralized", "centralized"),
    ]
    other = ["benchmark", "model_size", "n_agents", "context_share_level",
             "prompt_complexity_level", "reasoning_level", "seed"]
    for low, high in mas_topo_pairs:
        for metric in CONTRAST_METRICS:
            frames.append(matched_contrast(
                df,
                axis="topology",
                low=low,
                high=high,
                metric=metric,
                match_cols=other,
                label=f"{high} - {low}",
                axis_scope="artifact_only MAS topology contrasts matched on size/agent_count/prompt/reasoning/seed",
                filt=artifact,
            ))

    for high in AGENT_ORDER:
        if high == 1:
            continue
        for metric in CONTRAST_METRICS:
            frames.append(agent_family_vs_single(df, high=high, metric=metric))

    return pd.concat([x for x in frames if not x.empty], ignore_index=True)


def build_stratified_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Contrasts for non-size/non-reasoning axes within model_size x reasoning strata."""
    frames = []
    group = ["benchmark", "model_size", "reasoning_level"]

    comm = df["topology"].astype(str).isin(["centralized", "decentralized"])
    match = ["benchmark", "model_size", "reasoning_level", "topology", "n_agents",
             "prompt_complexity_level", "seed"]
    for metric in STRATIFIED_METRICS:
        frames.append(matched_contrast_strata(
            df,
            axis="context_share_level",
            low="artifact_only",
            high="plus_cot",
            metric=metric,
            match_cols=match,
            group_cols=group,
            label="plus_cot - artifact_only",
            axis_scope="controlled within benchmark/model_size/reasoning/topology/agent_count/prompt/seed",
            filt=comm,
        ))

    match = ["benchmark", "model_size", "reasoning_level", "topology", "n_agents",
             "context_share_level", "seed"]
    for metric in STRATIFIED_METRICS:
        frames.append(matched_contrast_strata(
            df,
            axis="prompt_complexity_level",
            low=0,
            high=3,
            metric=metric,
            match_cols=match,
            group_cols=group,
            label="L3 - L0",
            axis_scope="controlled within benchmark/model_size/reasoning/topology/agent_count/context/seed",
        ))

    artifact = df["context_share_level"].astype(str) == "artifact_only"
    match = ["benchmark", "model_size", "reasoning_level", "n_agents", "context_share_level",
             "prompt_complexity_level", "seed"]
    topo_pairs = [
        ("independent", "decentralized"),
        ("independent", "centralized"),
        ("decentralized", "centralized"),
    ]
    for low, high in topo_pairs:
        for metric in STRATIFIED_METRICS:
            frames.append(matched_contrast_strata(
                df,
                axis="topology",
                low=low,
                high=high,
                metric=metric,
                match_cols=match,
                group_cols=group,
                label=f"{high} - {low}",
                axis_scope="artifact_only, controlled within benchmark/model_size/reasoning/agent_count/prompt/seed",
                filt=artifact,
            ))

    for high in AGENT_ORDER:
        if high == 1:
            continue
        for metric in STRATIFIED_METRICS:
            frames.append(agent_family_vs_single_strata(
                df, high=high, metric=metric, group_cols=group
            ))

    return pd.concat([x for x in frames if not x.empty], ignore_index=True)


def _summary_slice(summary: pd.DataFrame, axis: str, metric: str, scope_contains: str | None = None) -> pd.DataFrame:
    d = summary[(summary["axis"] == axis) & (summary["metric"] == metric)].copy()
    if scope_contains is not None:
        d = d[d["axis_scope"].str.contains(scope_contains, regex=False)]
    return d


def _level_stats(summary: pd.DataFrame, axis: str, metric: str, scope_contains: str | None = None) -> pd.DataFrame:
    d = _summary_slice(summary, axis, metric, scope_contains)
    levels = _ordered_levels(axis, d["level"])
    d["level"] = pd.Categorical(d["level"].astype(str), [str(x) for x in levels], ordered=True)
    d["benchmark"] = pd.Categorical(d["benchmark"], BENCHES, ordered=True)
    return d.sort_values(["benchmark", "level"])


def _plot_line_axis(
    summary: pd.DataFrame,
    *,
    axis: str,
    metrics: list[str],
    filename: str,
    title: str,
    scope_contains: str | None = None,
) -> None:
    plots.style()
    fig, axes = plt.subplots(
        len(metrics),
        len(BENCHES),
        figsize=(3.4 * len(BENCHES), 2.45 * len(metrics)),
        sharex=False,
        squeeze=False,
    )
    for r, metric in enumerate(metrics):
        d = _level_stats(summary, axis, metric, scope_contains)
        levels = [str(x) for x in _ordered_levels(axis, d["level"])]
        xpos = np.arange(len(levels))
        for c, bench in enumerate(BENCHES):
            ax = axes[r, c]
            db = d[d["benchmark"].astype(str) == bench]
            if db.empty:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            y = []
            lo = []
            hi = []
            ns = []
            for lev in levels:
                row = db[db["level"].astype(str) == lev]
                if row.empty:
                    y.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    ns.append(0)
                else:
                    rr = row.iloc[0]
                    y.append(rr["mean"])
                    lo.append(rr["ci_low"])
                    hi.append(rr["ci_high"])
                    ns.append(int(rr["n_designs"]))
            y = np.array(y, float)
            lo = np.array(lo, float)
            hi = np.array(hi, float)
            ax.errorbar(
                xpos,
                y,
                yerr=[y - lo, hi - y],
                color=plots.BENCH_COLORS.get(bench, plots.PRIMARY),
                marker="o",
                linewidth=1.8,
                capsize=2.5,
            )
            if metric.startswith("delta") or metric.endswith("signed_gap"):
                ax.axhline(0, color=plots.REF_GRAY, linestyle="--", linewidth=1)
            if metric == "cost":
                ax.set_yscale("log")
            ax.set_xticks(xpos)
            ax.set_xticklabels(levels, rotation=30, ha="right")
            if r == 0:
                ax.set_title(bench)
            if c == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.text(
                0.98,
                0.02,
                "designs: " + ",".join(str(n) for n in ns if n),
                transform=ax.transAxes,
                fontsize=6.5,
                color="0.4",
                ha="right",
                va="bottom",
            )
    fig.suptitle(title, y=1.002, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_axis(
    summary: pd.DataFrame,
    *,
    axis: str,
    metrics: list[str],
    filename: str,
    title: str,
    scope_contains: str | None = None,
) -> None:
    plots.style()
    fig, axes = plt.subplots(
        len(metrics),
        len(BENCHES),
        figsize=(3.4 * len(BENCHES), 2.45 * len(metrics)),
        sharex=False,
        squeeze=False,
    )
    colors = {
        "single_agent": "#64748b",
        "independent": "#2563eb",
        "decentralized": "#d97706",
        "centralized": "#dc2626",
        "1": "#64748b",
        "3": "#2563eb",
        "0": "#64748b",
        "3_prompt": "#2563eb",
        "artifact_only": "#64748b",
        "plus_cot": "#d97706",
        "2": "#0891b2",
        "4": "#16a34a",
        "5": "#ca8a04",
        "6": "#dc2626",
    }
    for r, metric in enumerate(metrics):
        d = _level_stats(summary, axis, metric, scope_contains)
        levels = [str(x) for x in _ordered_levels(axis, d["level"])]
        xpos = np.arange(len(levels))
        for c, bench in enumerate(BENCHES):
            ax = axes[r, c]
            db = d[d["benchmark"].astype(str) == bench]
            if db.empty:
                ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            y = []
            lo = []
            hi = []
            ns = []
            for lev in levels:
                row = db[db["level"].astype(str) == lev]
                if row.empty:
                    y.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    ns.append(0)
                else:
                    rr = row.iloc[0]
                    y.append(rr["mean"])
                    lo.append(rr["ci_low"])
                    hi.append(rr["ci_high"])
                    ns.append(int(rr["n_designs"]))
            y = np.array(y, float)
            lo = np.array(lo, float)
            hi = np.array(hi, float)
            bar_colors = []
            for lev in levels:
                if axis == "prompt_complexity_level" and lev == "3":
                    bar_colors.append(colors["3_prompt"])
                else:
                    bar_colors.append(colors.get(lev, plots.PRIMARY))
            ax.bar(xpos, y, color=bar_colors, alpha=0.9)
            ax.errorbar(xpos, y, yerr=[y - lo, hi - y], fmt="none", ecolor="0.2", capsize=2)
            if metric.startswith("delta") or metric.endswith("signed_gap"):
                ax.axhline(0, color=plots.REF_GRAY, linestyle="--", linewidth=1)
            if metric == "cost":
                ax.set_yscale("log")
            ax.set_xticks(xpos)
            ax.set_xticklabels(levels, rotation=30, ha="right")
            if r == 0:
                ax.set_title(bench)
            if c == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.text(
                0.98,
                0.02,
                "designs: " + ",".join(str(n) for n in ns if n),
                transform=ax.transAxes,
                fontsize=6.5,
                color="0.4",
                ha="right",
                va="bottom",
            )
    fig.suptitle(title, y=1.002, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FIG / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    plots.style()
    design = df.dropna(subset=[metric]).groupby(
        ["benchmark", "model_size", "reasoning_level", "design_cell"], observed=True
    )[metric].mean().reset_index()
    mean = design.groupby(["benchmark", "model_size", "reasoning_level"], observed=True)[metric].mean()
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(3.4 * len(BENCHES), 3.0), sharey=True)
    for ax, bench in zip(axes, BENCHES):
        mat = np.full((len(SIZE_ORDER), len(REAS_ORDER)), np.nan)
        for i, size in enumerate(SIZE_ORDER):
            for j, reas in enumerate(REAS_ORDER):
                key = (bench, size, reas)
                if key in mean.index:
                    mat[i, j] = mean.loc[key]
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(bench)
        ax.set_xticks(np.arange(len(REAS_ORDER)))
        ax.set_xticklabels(REAS_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(SIZE_ORDER)))
        ax.set_yticklabels(SIZE_ORDER)
        for i in range(len(SIZE_ORDER)):
            for j in range(len(REAS_ORDER)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        if bench == "gpqa":
            ax.set_ylabel("Model size")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label=METRIC_LABELS[metric])
    fig.suptitle(title, y=1.03, fontsize=13)
    fig.savefig(OUT_FIG / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stratified_contrast_heatmap(
    stratified: pd.DataFrame,
    *,
    axis: str,
    contrast: str,
    metric: str,
    filename: str,
    title: str,
    sparse_threshold: int = 3,
) -> None:
    """Heatmap of a controlled contrast over model_size x reasoning, by benchmark.

    Each cell contains the contrast delta and the number of matched design pairs.
    Cells with fewer than sparse_threshold design pairs get a star marker.
    """
    plots.style()
    d = stratified[
        (stratified["axis"] == axis)
        & (stratified["contrast"] == contrast)
        & (stratified["metric"] == metric)
    ].copy()
    if d.empty:
        return

    finite = d["delta"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(finite) == 0:
        return
    lim = float(np.nanpercentile(np.abs(finite), 95))
    lim = max(lim, float(np.nanmax(np.abs(finite))), 1e-6)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(1, len(BENCHES), figsize=(3.95 * len(BENCHES), 4.0), sharey=True)
    for ax, bench in zip(axes, BENCHES):
        mat = np.full((len(SIZE_ORDER), len(REAS_ORDER)), np.nan)
        nmat = np.zeros((len(SIZE_ORDER), len(REAS_ORDER)), dtype=int)
        db = d[d["benchmark"].astype(str) == bench]
        for i, size in enumerate(SIZE_ORDER):
            for j, reas in enumerate(REAS_ORDER):
                row = db[
                    (db["model_size"].astype(str) == size)
                    & (db["reasoning_level"].astype(str) == reas)
                ]
                if row.empty:
                    continue
                rr = row.iloc[0]
                mat[i, j] = rr["delta"]
                nmat[i, j] = int(rr["n_matched_design_pairs"])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.axhline(4.5, color="#dc2626", linewidth=0.8, alpha=0.45)
        ax.set_title(bench)
        ax.set_xticks(np.arange(len(REAS_ORDER)))
        ax.set_xticklabels(REAS_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(SIZE_ORDER)))
        ax.set_yticklabels(SIZE_ORDER)
        for i in range(len(SIZE_ORDER)):
            for j in range(len(REAS_ORDER)):
                if not np.isfinite(mat[i, j]):
                    continue
                sparse = "*" if 0 < nmat[i, j] < sparse_threshold else ""
                color = "white" if abs(mat[i, j]) > 0.55 * lim else "black"
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:+.2f}\n({nmat[i, j]}){sparse}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=color,
                )
        if bench == "gpqa":
            ax.set_ylabel("Model size")
    fig.subplots_adjust(left=0.065, right=0.875, bottom=0.27, top=0.80, wspace=0.28)
    cax = fig.add_axes([0.905, 0.33, 0.014, 0.42])
    fig.colorbar(im, cax=cax, label=f"{METRIC_LABELS[metric]} contrast")
    fig.suptitle(title, y=0.955, fontsize=13)
    fig.text(
        0.5,
        0.08,
        f"Cell text: delta (matched design pairs); * means n < {sparse_threshold}. "
        "Red guide separates 14B from the 32B row; 32B long-reasoning cells are context-capped.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.savefig(OUT_FIG / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cost_frontier(df: pd.DataFrame) -> None:
    plots.style()
    d = df.dropna(subset=["accuracy", "cost"]).copy()
    d["design_accuracy"] = d.groupby("design_cell", observed=True)["accuracy"].transform("mean")
    d["design_cost"] = d.groupby("design_cell", observed=True)["cost"].transform("mean")
    d = d.drop_duplicates("design_cell")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, bench in zip(axes, BENCHES):
        db = d[d["benchmark"].astype(str) == bench].copy()
        for topo in TOPO_ORDER:
            dt = db[db["topology"].astype(str) == topo]
            if dt.empty:
                continue
            ax.scatter(
                dt["design_cost"],
                dt["design_accuracy"],
                s=18,
                alpha=0.55,
                color=plots.TOPO_COLORS.get(topo, plots.PRIMARY),
                label=topo,
            )
        frontier = []
        for _, row in db.sort_values("design_cost").iterrows():
            if not frontier or row["design_accuracy"] > frontier[-1][1]:
                frontier.append((row["design_cost"], row["design_accuracy"]))
        if frontier:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color="black", linewidth=1.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_title(bench)
        ax.set_xlabel("Cost proxy: B params x total tokens")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Observed accuracy-cost cloud and empirical frontier", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUT_FIG / "axis_cost_frontier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(df: pd.DataFrame, summary: pd.DataFrame, stratified: pd.DataFrame) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    common_metrics = ["accuracy", "vote_ece_prim", "pa_ece_prim", "delta_vote_prim"]
    _plot_line_axis(
        summary,
        axis="model_size",
        metrics=common_metrics,
        filename="axis_model_size_outcomes.png",
        title="Outcome relationships by model size",
    )
    _plot_line_axis(
        summary,
        axis="reasoning_level",
        metrics=common_metrics,
        filename="axis_reasoning_outcomes.png",
        title="Outcome relationships by reasoning budget",
    )
    _plot_bar_axis(
        summary,
        axis="topology",
        metrics=["accuracy", "vote_ece_prim", "Ec", "Ae"],
        filename="axis_topology_outcomes.png",
        title="Outcome relationships by topology",
    )
    _plot_bar_axis(
        summary,
        axis="n_agents",
        metrics=["accuracy", "vote_ece_prim", "cost", "Ae"],
        filename="axis_agent_count_outcomes.png",
        title="Outcome relationships by observed agent count",
    )
    _plot_bar_axis(
        summary,
        axis="prompt_complexity_level",
        metrics=common_metrics,
        filename="axis_prompt_outcomes.png",
        title="Outcome relationships by prompt complexity",
    )
    _plot_bar_axis(
        summary,
        axis="context_share_level",
        metrics=common_metrics,
        filename="axis_context_outcomes.png",
        title="Outcome relationships by context sharing",
        scope_contains="communication topologies only",
    )
    plot_heatmaps(
        df,
        metric="accuracy",
        filename="axis_size_reasoning_accuracy_heatmap.png",
        title="Accuracy over model size x reasoning budget",
    )
    plot_cost_frontier(df)
    plot_stratified_contrast_heatmap(
        stratified,
        axis="context_share_level",
        contrast="plus_cot - artifact_only",
        metric="accuracy",
        filename="axis_controlled_context_accuracy.png",
        title="Controlled context-sharing effect on accuracy",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="context_share_level",
        contrast="plus_cot - artifact_only",
        metric="vote_ece_prim",
        filename="axis_controlled_context_vote_ece.png",
        title="Controlled context-sharing effect on vote ECE",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="context_share_level",
        contrast="plus_cot - artifact_only",
        metric="delta_vote_prim",
        filename="axis_controlled_context_delta_vote.png",
        title="Controlled context-sharing effect on Delta vote ECE",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="prompt_complexity_level",
        contrast="L3 - L0",
        metric="accuracy",
        filename="axis_controlled_prompt_accuracy.png",
        title="Controlled prompt-complexity effect on accuracy",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="topology",
        contrast="decentralized - independent",
        metric="accuracy",
        filename="axis_controlled_decentralized_vs_independent_accuracy.png",
        title="Controlled decentralized-vs-independent effect on accuracy",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="topology",
        contrast="centralized - decentralized",
        metric="accuracy",
        filename="axis_controlled_centralized_vs_decentralized_accuracy.png",
        title="Controlled centralized-vs-decentralized effect on accuracy",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="n_agents",
        contrast="3-agent family - 1-agent single",
        metric="accuracy",
        filename="axis_controlled_agent_family_accuracy.png",
        title="Controlled 3-agent-family effect on accuracy",
    )
    plot_stratified_contrast_heatmap(
        stratified,
        axis="n_agents",
        contrast="6-agent family - 1-agent single",
        metric="accuracy",
        filename="axis_controlled_six_agent_family_accuracy.png",
        title="Controlled 6-agent-family effect on accuracy",
    )


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    plots.style()

    df = pd.read_parquet(nb_data.CACHE / "analysis_view_v1.parquet")
    df = nb_data.add_derived(df)
    df = df[df["is_core"]].copy()

    summary = build_summary(df)
    contrasts = build_contrasts(df)
    stratified = build_stratified_contrasts(df)

    summary.to_csv(OUT_TAB / "axis_summary_long.csv", index=False)
    contrasts.to_csv(OUT_TAB / "axis_matched_contrasts.csv", index=False)
    stratified.to_csv(OUT_TAB / "axis_stratified_contrasts.csv", index=False)
    make_figures(df, summary, stratified)

    manifest = nb_data.manifest()
    print(f"loaded_rows={len(df)}")
    if manifest:
        print(
            "cache_manifest="
            f"{manifest.get('n_cells_complete')} complete cells, "
            f"{manifest.get('n_bad_lines')} bad lines, "
            f"{manifest.get('n_dupes_dropped')} duplicate rows dropped"
        )
    print(f"wrote {OUT_TAB / 'axis_summary_long.csv'}")
    print(f"wrote {OUT_TAB / 'axis_matched_contrasts.csv'}")
    print(f"wrote {OUT_TAB / 'axis_stratified_contrasts.csv'}")
    print(f"wrote figures to {OUT_FIG}")


if __name__ == "__main__":
    main()
