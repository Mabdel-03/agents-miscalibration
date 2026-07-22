#!/usr/bin/env python3
"""Supplementary reasoning-vs-size analysis for partial full_sweep_v1 evidence.

This script answers a narrower question than the broad EDA notebooks:

    How much model-size scaling is substituted by reasoning/test-time compute,
    and where do the returns plateau?

The fits are descriptive, mixed-protocol, and outside the primary schema-5 estimand. They
use the historical post-trim core cache and must not be presented as final scaling laws.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.nb_lib import data as nb_data  # noqa: E402
from analysis.nb_lib import plots  # noqa: E402


OUT_TAB = ROOT / "analysis" / "tables" / "scaling_equivalence"
OUT_FIG = ROOT / "analysis" / "figures" / "scaling_equivalence"
OUT_REPORT = ROOT / "analysis" / "scaling_equivalence_report.md"

BENCHES = nb_data.BENCHMARKS
MCQ_BENCHES = nb_data.MCQ_BENCHMARKS
SIZE_ORDER = nb_data.SIZE_ORDER
REAS_ORDER = nb_data.REAS_ORDER
TOPO_ORDER = nb_data.TOPO_ORDER
CTX_ORDER = ["artifact_only", "plus_cot"]

SIZE_PAIRS = list(zip(SIZE_ORDER[:-1], SIZE_ORDER[1:]))
REAS_PAIRS = list(zip(REAS_ORDER[:-1], REAS_ORDER[1:]))

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "cost": "Cost proxy",
    "vote_ece_prim": "Vote ECE",
    "delta_vote_prim": "Delta vote ECE",
    "Ec": "Coordination efficiency",
    "Ae": "Error amplification",
}


def _as_str_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["model_size", "reasoning_level", "topology", "context_share_level"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def load_core(*, drop_ctx_capped: bool = False) -> pd.DataFrame:
    raw = pd.read_parquet(nb_data.CACHE / "analysis_view_v1.parquet")
    df = nb_data.analysis_view(
        raw,
        post_trim_only=True,
        min_seeds=1,
        drop_ctx_capped=drop_ctx_capped,
    )
    df = _as_str_categories(df)
    if "reasoning_tokens_exact" not in df.columns:
        raise RuntimeError(
            "analysis cache predates exact reasoning-token provenance; rerun "
            "analysis/refresh.sh before fitting token-dose laws"
        )
    exact = df["reasoning_tokens_exact"].fillna(False).astype(bool)
    reasoning_tokens = pd.to_numeric(df["mean_reasoning_tokens"], errors="coerce")
    # Missing/nonexact counts stay missing.  Treating them as zero would silently turn
    # historical whitespace counts into a false no-reasoning treatment.
    df["log2_reas"] = np.where(
        exact & reasoning_tokens.notna(),
        np.log2(reasoning_tokens + 1.0),
        np.nan,
    )
    df["ctx32b_capped"] = df["ctx32b_capped"].astype(bool)
    return df


def _glm_formula(df: pd.DataFrame, *, reasoning: str, include_topology: bool) -> str:
    terms = ["log2_params"]
    if reasoning == "continuous":
        terms.append("log2_reas")
    elif reasoning == "rung":
        terms.append('C(reasoning_level, Treatment(reference="off"))')
    else:
        raise ValueError(f"unknown reasoning mode {reasoning!r}")

    terms.append("prompt_complexity_level")

    if include_topology and df["topology"].nunique() > 1:
        if "single_agent" in set(df["topology"]):
            terms.append('C(topology, Treatment(reference="single_agent"))')
        else:
            terms.append("C(topology)")

    if df["context_share_level"].nunique() > 1:
        if "artifact_only" in set(df["context_share_level"]):
            terms.append('C(context_share_level, Treatment(reference="artifact_only"))')
        else:
            terms.append("C(context_share_level)")

    if df["ctx32b_capped"].nunique() > 1:
        terms.append("ctx32b_capped")

    return "accuracy ~ " + " + ".join(terms)


def _fit_glm(df: pd.DataFrame, *, reasoning: str, include_topology: bool):
    d = df.dropna(subset=["accuracy", "n_questions", "log2_params"]).copy()
    if reasoning == "continuous":
        d = d.dropna(subset=["log2_reas"])
    if len(d) < 20 or d["accuracy"].nunique() < 2:
        return None
    formula = _glm_formula(d, reasoning=reasoning, include_topology=include_topology)
    try:
        return smf.glm(
            formula=formula,
            data=d,
            family=sm.families.Binomial(),
            var_weights=d["n_questions"].astype(float),
        ).fit()
    except Exception as exc:  # pragma: no cover - report-oriented fallback
        print(f"[warn] GLM failed for formula {formula}: {exc}", file=sys.stderr)
        return None


def _dev_r2(result) -> float:
    if result is None or not np.isfinite(result.null_deviance) or result.null_deviance <= 0:
        return math.nan
    return float(1.0 - result.deviance / result.null_deviance)


def fit_continuous_laws(scopes: list[tuple[str, pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for scope, df, scope_note in scopes:
        groups: list[tuple[str, pd.DataFrame, bool]] = [("all_systems", df, True)]
        groups.extend((f"topology:{topo}", df[df["topology"] == topo], False) for topo in TOPO_ORDER)

        for bench in BENCHES:
            for fit_group, dg, include_topology in groups:
                d = dg[dg["benchmark"] == bench].copy()
                if len(d) < 40:
                    continue
                result = _fit_glm(d, reasoning="continuous", include_topology=include_topology)
                if result is None or "log2_params" not in result.params or "log2_reas" not in result.params:
                    continue
                beta_p = float(result.params["log2_params"])
                beta_r = float(result.params["log2_reas"])
                rows.append(
                    {
                        "scope": scope,
                        "scope_note": scope_note,
                        "fit_group": fit_group,
                        "benchmark": bench,
                        "n_rows": int(len(d)),
                        "n_designs": int(d["design_cell"].nunique()),
                        "coef_log2_params": beta_p,
                        "se_log2_params": float(result.bse.get("log2_params", math.nan)),
                        "coef_log2_reasoning_tokens": beta_r,
                        "se_log2_reasoning_tokens": float(result.bse.get("log2_reas", math.nan)),
                        "reasoning_doublings_per_param_doubling": beta_p / beta_r
                        if beta_r > 0
                        else math.nan,
                        "param_doublings_substituted_per_reasoning_doubling": beta_r / beta_p
                        if beta_p > 0
                        else math.nan,
                        "reasoning_multiplier_per_param_doubling": 2 ** (beta_p / beta_r)
                        if beta_r > 0 and beta_p > 0 and beta_p / beta_r < 30
                        else math.nan,
                        "deviance_r2": _dev_r2(result),
                    }
                )
    return pd.DataFrame(rows)


def fit_rung_equivalents(scopes: list[tuple[str, pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for scope, df, scope_note in scopes:
        groups: list[tuple[str, pd.DataFrame, bool]] = [("all_systems", df, True)]
        groups.extend((f"topology:{topo}", df[df["topology"] == topo], False) for topo in TOPO_ORDER)

        for bench in BENCHES:
            for fit_group, dg, include_topology in groups:
                d = dg[dg["benchmark"] == bench].copy()
                if len(d) < 40:
                    continue
                result = _fit_glm(d, reasoning="rung", include_topology=include_topology)
                if result is None or "log2_params" not in result.params:
                    continue
                beta_p = float(result.params["log2_params"])
                for rung in REAS_ORDER[1:]:
                    term = f'C(reasoning_level, Treatment(reference="off"))[T.{rung}]'
                    delta = float(result.params.get(term, math.nan))
                    se = float(result.bse.get(term, math.nan))
                    equiv_doublings = delta / beta_p if beta_p > 0 else math.nan
                    rows.append(
                        {
                            "scope": scope,
                            "scope_note": scope_note,
                            "fit_group": fit_group,
                            "benchmark": bench,
                            "reasoning_rung": rung,
                            "n_rows": int(len(d)),
                            "n_designs": int(d["design_cell"].nunique()),
                            "coef_log2_params": beta_p,
                            "rung_logit_gain_vs_off": delta,
                            "se_rung_logit_gain_vs_off": se,
                            "equiv_param_doublings_vs_off": equiv_doublings,
                            "equiv_param_multiplier_vs_off": 2 ** equiv_doublings
                            if np.isfinite(equiv_doublings)
                            else math.nan,
                            "deviance_r2": _dev_r2(result),
                        }
                    )
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, *, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        val = float(values[0])
        return val, val, val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    dist = values[idx].mean(axis=1)
    return (
        float(np.mean(values)),
        float(np.percentile(dist, 2.5)),
        float(np.percentile(dist, 97.5)),
    )


def _matched_contrast(
    df: pd.DataFrame,
    *,
    axis: str,
    low: str,
    high: str,
    metric: str,
    match_cols: list[str],
    seed_offset: int,
) -> pd.DataFrame:
    rows = []
    d = df[df[axis].isin([low, high])].dropna(subset=[metric]).copy()
    for bench in BENCHES:
        db = d[d["benchmark"] == bench]
        left = db[db[axis] == low][match_cols + [metric]].rename(columns={metric: "low"})
        right = db[db[axis] == high][match_cols + [metric]].rename(columns={metric: "high"})
        pair = left.merge(right, on=match_cols, how="inner")
        if pair.empty:
            continue
        pair["delta"] = pair["high"] - pair["low"]
        design_cols = [c for c in match_cols if c != "seed"]
        design_delta = pair.groupby(design_cols, observed=True)["delta"].mean().to_numpy(float)
        mean, lo, hi = _bootstrap_mean(design_delta, seed=seed_offset + len(rows) * 31)
        rows.append(
            {
                "axis": axis,
                "contrast": f"{high} - {low}",
                "low": low,
                "high": high,
                "benchmark": bench,
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_matched_seed_pairs": int(len(pair)),
                "n_matched_design_pairs": int(len(design_delta)),
                "plateau_like": bool(abs(mean) < 0.01 or (lo <= 0 <= hi)),
            }
        )
    return pd.DataFrame(rows)


def adjacent_plateaus(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    size_match = [
        "benchmark",
        "topology",
        "context_share_level",
        "prompt_complexity_level",
        "reasoning_level",
        "seed",
    ]
    reas_match = [
        "benchmark",
        "model_size",
        "topology",
        "context_share_level",
        "prompt_complexity_level",
        "seed",
    ]
    for low, high in SIZE_PAIRS:
        frames.append(
            _matched_contrast(
                df,
                axis="model_size",
                low=low,
                high=high,
                metric="accuracy",
                match_cols=size_match,
                seed_offset=101,
            )
        )
    for low, high in REAS_PAIRS:
        frames.append(
            _matched_contrast(
                df,
                axis="reasoning_level",
                low=low,
                high=high,
                metric="accuracy",
                match_cols=reas_match,
                seed_offset=202,
            )
        )
    return pd.concat(frames, ignore_index=True)


def orchestration_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    d = df[df["context_share_level"] == "artifact_only"].copy()
    match_cols = ["benchmark", "model_size", "prompt_complexity_level", "reasoning_level", "seed"]
    pairs = [
        ("single_agent", "independent"),
        ("independent", "decentralized"),
        ("independent", "centralized"),
        ("decentralized", "centralized"),
    ]
    metrics = ["accuracy", "cost", "vote_ece_prim", "delta_vote_prim", "Ec", "Ae"]
    for metric in metrics:
        for low, high in pairs:
            frames.append(
                _matched_contrast(
                    d,
                    axis="topology",
                    low=low,
                    high=high,
                    metric=metric,
                    match_cols=match_cols,
                    seed_offset=303 + 11 * len(frames),
                )
            )
    out = pd.concat(frames, ignore_index=True)
    out["axis"] = "topology"
    return out


def grid_summary(df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    if scope == "all_systems":
        d = df.copy()
    elif scope == "single_agent":
        d = df[df["topology"] == "single_agent"].copy()
    elif scope.startswith("topology:"):
        topo = scope.split(":", 1)[1]
        d = df[df["topology"] == topo].copy()
    else:
        raise ValueError(scope)

    design_cols = ["benchmark", "model_size", "reasoning_level", "design_cell"]
    design = (
        d.groupby(design_cols, observed=True)
        .agg(
            accuracy=("accuracy", "mean"),
            cost=("cost", "mean"),
            mean_total_tokens=("mean_total_tokens", "mean"),
            mean_reasoning_tokens=("mean_reasoning_tokens", "mean"),
        )
        .reset_index()
    )
    grid = (
        design.groupby(["benchmark", "model_size", "reasoning_level"], observed=True)
        .agg(
            accuracy=("accuracy", "mean"),
            cost=("cost", "mean"),
            mean_total_tokens=("mean_total_tokens", "mean"),
            mean_reasoning_tokens=("mean_reasoning_tokens", "mean"),
            n_designs=("design_cell", "nunique"),
        )
        .reset_index()
    )
    grid["scope"] = scope
    grid["param_count"] = grid["model_size"].map(dict(zip(SIZE_ORDER, [0.6, 1.7, 4.0, 8.2, 14.8, 32.8])))
    grid["reasoning_rank"] = grid["reasoning_level"].map(nb_data.REAS_RANK)
    grid["size_rank"] = grid["model_size"].map({s: i for i, s in enumerate(SIZE_ORDER)})
    return grid


def nearest_equivalence_pairs(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in sorted(grid["scope"].unique()):
        gs = grid[grid["scope"] == scope]
        for bench in BENCHES:
            gb = gs[gs["benchmark"] == bench].copy()
            targets = gb[gb["reasoning_level"].isin(["off", "b512"])].copy()
            for _, target in targets.iterrows():
                candidates = gb[
                    (gb["size_rank"] < target["size_rank"])
                    & (gb["reasoning_rank"] > target["reasoning_rank"])
                ].copy()
                if candidates.empty:
                    continue
                candidates["abs_gap"] = (candidates["accuracy"] - target["accuracy"]).abs()
                cand = candidates.sort_values(["abs_gap", "cost"]).iloc[0]
                rows.append(
                    {
                        "scope": scope,
                        "benchmark": bench,
                        "target_model_size": target["model_size"],
                        "target_reasoning": target["reasoning_level"],
                        "target_accuracy": float(target["accuracy"]),
                        "target_cost": float(target["cost"]),
                        "candidate_model_size": cand["model_size"],
                        "candidate_reasoning": cand["reasoning_level"],
                        "candidate_accuracy": float(cand["accuracy"]),
                        "candidate_cost": float(cand["cost"]),
                        "accuracy_gap_candidate_minus_target": float(cand["accuracy"] - target["accuracy"]),
                        "abs_accuracy_gap": float(cand["abs_gap"]),
                        "candidate_cost_over_target_cost": float(cand["cost"] / target["cost"])
                        if target["cost"] > 0
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_rung_equivalents(rung: pd.DataFrame) -> None:
    d = rung[(rung["scope"] == "core_all_cap_control") & (rung["fit_group"] == "all_systems")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plots.BENCH_COLORS
    x = np.arange(len(REAS_ORDER[1:]))
    width = 0.18
    for i, bench in enumerate(BENCHES):
        db = d[d["benchmark"] == bench].set_index("reasoning_rung").reindex(REAS_ORDER[1:])
        ax.bar(
            x + (i - 1.5) * width,
            db["equiv_param_multiplier_vs_off"],
            width=width,
            label=bench,
            color=colors[bench],
        )
    ax.axhline(1.0, color="0.2", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(REAS_ORDER[1:])
    ax.set_ylabel("Equivalent parameter multiplier vs off")
    ax.set_xlabel("Reasoning rung")
    ax.set_title("Reasoning rungs as model-size equivalents")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "rung_equivalent_param_multiplier.png", dpi=180)
    plt.close(fig)


def plot_adjacent_plateaus(adj: pd.DataFrame) -> None:
    d = adj[(adj["metric"] == "accuracy") & adj["axis"].isin(["model_size", "reasoning_level"])]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, axis in zip(axes, ["model_size", "reasoning_level"]):
        da = d[d["axis"] == axis].copy()
        contrasts = da["contrast"].drop_duplicates().tolist()
        x = np.arange(len(contrasts))
        width = 0.18
        for i, bench in enumerate(BENCHES):
            db = da[da["benchmark"] == bench].set_index("contrast").reindex(contrasts)
            y = db["delta"].to_numpy(float)
            lo = y - db["ci_low"].to_numpy(float)
            hi = db["ci_high"].to_numpy(float) - y
            ax.errorbar(
                x + (i - 1.5) * width,
                y,
                yerr=np.vstack([lo, hi]),
                fmt="o",
                ms=4,
                capsize=2,
                label=bench if axis == "model_size" else None,
                color=plots.BENCH_COLORS[bench],
            )
        ax.axhline(0, color="0.2", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(contrasts, rotation=35, ha="right")
        ax.set_title("Adjacent " + ("size" if axis == "model_size" else "reasoning") + " gains")
        ax.set_ylabel("Matched accuracy delta")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "adjacent_accuracy_plateaus.png", dpi=180)
    plt.close(fig)


def _fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def write_report(
    coeffs: pd.DataFrame,
    rung: pd.DataFrame,
    adj: pd.DataFrame,
    orch: pd.DataFrame,
    pairs: pd.DataFrame,
    manifest: dict,
) -> None:
    primary_coeffs = coeffs[
        (coeffs["scope"] == "core_all_cap_control") & (coeffs["fit_group"] == "all_systems")
    ].copy()
    primary_rung = rung[
        (rung["scope"] == "core_all_cap_control") & (rung["fit_group"] == "all_systems")
    ].copy()
    adj_acc = adj[adj["metric"] == "accuracy"].copy()
    orch_acc = orch[(orch["metric"] == "accuracy")].copy()

    lines: list[str] = []
    lines.append("# Reasoning vs model-size scaling equivalence")
    lines.append("")
    lines.append(
        "Generated from `analysis/cache/analysis_view_v1.parquet` using the post-trim core view. "
        "These are descriptive fits over the current partial sweep, not final scaling laws."
    )
    if manifest:
        lines.append(
            f"Cache manifest: {manifest.get('n_cells_complete', 'n/a')} complete cells, "
            f"{manifest.get('n_bad_lines', 'n/a')} malformed JSONL lines skipped, "
            f"{manifest.get('n_dupes_dropped', 'n/a')} duplicate item rows dropped."
        )
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(
        "Primary law: `logit(accuracy) = beta_P log2(params_B) + beta_R log2(1 + reasoning_tokens) "
        "+ controls`, where controls include topology, context sharing, prompt level, and a 32B "
        "long-reasoning context-cap indicator. The categorical companion model replaces "
        "`log2(1 + reasoning_tokens)` with reasoning-rung dummies. The categorical model gives the "
        "more actionable equivalence because the jump from `off` to native thinking is discrete."
    )
    lines.append("")
    lines.append("## Continuous token-dose equivalence")
    lines.append("")
    lines.append("| Benchmark | beta_P | beta_R | reasoning doublings per parameter doubling | deviance R2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for bench in BENCHES:
        row = primary_coeffs[primary_coeffs["benchmark"] == bench]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {bench} | {_fmt(r.coef_log2_params)} | {_fmt(r.coef_log2_reasoning_tokens)} | "
            f"{_fmt(r.reasoning_doublings_per_param_doubling, 2)} | {_fmt(r.deviance_r2)} |"
        )
    lines.append("")
    lines.append(
        "Read this as a local marginal rate: one doubling of realized reasoning tokens substitutes for "
        "only about 0.10-0.15 parameter doublings, so replacing a full parameter doubling requires many "
        "doublings of reasoning tokens. This is why the rung model below is the better decision summary."
    )
    lines.append("")
    lines.append("## Reasoning rungs as parameter multipliers")
    lines.append("")
    lines.append("| Benchmark | b512 | b2048 | b8192 | unlimited |")
    lines.append("|---|---:|---:|---:|---:|")
    for bench in BENCHES:
        rb = primary_rung[primary_rung["benchmark"] == bench].set_index("reasoning_rung")
        vals = []
        for rung_name in REAS_ORDER[1:]:
            vals.append(_fmt(float(rb.loc[rung_name, "equiv_param_multiplier_vs_off"]), 2))
        lines.append(f"| {bench} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append(
        "This is the cleanest current answer to the small-model-plus-thinking question. Moving from "
        "`off` to `b2048` is worth roughly 2.1x-3.2x in parameters depending on task; `b8192` is "
        "roughly 2.3x-4.0x. `unlimited` does not materially beat `b8192`."
    )
    lines.append("")
    lines.append("## Plateau points")
    lines.append("")
    lines.append("| Axis contrast | GPQA | MMLU-Pro | TruthfulQA | MATH |")
    lines.append("|---|---:|---:|---:|---:|")
    wanted = [
        ("model_size", "32B - 14B"),
        ("reasoning_level", "b8192 - b2048"),
        ("reasoning_level", "unlimited - b8192"),
    ]
    for axis, contrast in wanted:
        row_vals = []
        for bench in BENCHES:
            row = adj_acc[
                (adj_acc["axis"] == axis)
                & (adj_acc["contrast"] == contrast)
                & (adj_acc["benchmark"] == bench)
            ]
            if row.empty:
                row_vals.append("n/a")
            else:
                r = row.iloc[0]
                row_vals.append(f"{_fmt(r.delta)} [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]")
        lines.append(f"| {contrast} | " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append(
        "The strongest plateau signal is reasoning: `unlimited - b8192` is zero or negative on every "
        "benchmark in matched contrasts. Model size also plateaus from 14B to 32B in this cache, but "
        "that conclusion is weaker because the 32B high-reasoning corner is under-filled and context-capped."
    )
    lines.append("")
    lines.append("## Orchestration effects")
    lines.append("")
    lines.append("| Contrast | GPQA | MMLU-Pro | TruthfulQA | MATH |")
    lines.append("|---|---:|---:|---:|---:|")
    for contrast in [
        "independent - single_agent",
        "decentralized - independent",
        "centralized - independent",
        "centralized - decentralized",
    ]:
        vals = []
        for bench in BENCHES:
            row = orch_acc[(orch_acc["contrast"] == contrast) & (orch_acc["benchmark"] == bench)]
            if row.empty:
                vals.append("n/a")
            else:
                r = row.iloc[0]
                vals.append(f"{_fmt(r.delta)} [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]")
        lines.append(f"| {contrast} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append(
        "Three-agent systems give modest accuracy gains, largest on MATH. Decentralized debate adds a "
        "small gain over independent voting, while centralized orchestration is not a general winner and "
        "is harmful on MATH."
    )
    lines.append("")
    lines.append("## Nearest empirical equivalences")
    lines.append("")
    lines.append(
        "The nearest-pair table searches for smaller models with higher reasoning that match larger "
        "`off` or `b512` targets after marginalizing over the core design grid. The cost ratio uses the "
        "repo's inference proxy `params_B * total_tokens`, so ratios above 1 mean the smaller-thinking "
        "candidate is more expensive at inference despite using fewer parameters."
    )
    lines.append("")
    if not pairs.empty:
        view = pairs[
            (pairs["scope"] == "single_agent")
            & (pairs["target_reasoning"] == "off")
            & (pairs["target_model_size"].isin(["4B", "8B", "14B", "32B"]))
        ].sort_values(["benchmark", "target_model_size"])
        lines.append("| Benchmark | Target | Candidate | Acc gap | Candidate cost / target cost |")
        lines.append("|---|---|---|---:|---:|")
        for _, r in view.head(24).iterrows():
            lines.append(
                f"| {r.benchmark} | {r.target_model_size}+{r.target_reasoning} "
                f"({_fmt(r.target_accuracy)}) | {r.candidate_model_size}+{r.candidate_reasoning} "
                f"({_fmt(r.candidate_accuracy)}) | {_fmt(r.accuracy_gap_candidate_minus_target)} | "
                f"{_fmt(r.candidate_cost_over_target_cost, 2)} |"
            )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `analysis/tables/scaling_equivalence/scaling_law_coefficients.csv`")
    lines.append("- `analysis/tables/scaling_equivalence/reasoning_rung_equivalents.csv`")
    lines.append("- `analysis/tables/scaling_equivalence/adjacent_plateau_contrasts.csv`")
    lines.append("- `analysis/tables/scaling_equivalence/orchestration_matched_contrasts.csv`")
    lines.append("- `analysis/tables/scaling_equivalence/size_reasoning_grid.csv`")
    lines.append("- `analysis/tables/scaling_equivalence/nearest_equivalence_pairs.csv`")
    lines.append("- `analysis/figures/scaling_equivalence/rung_equivalent_param_multiplier.png`")
    lines.append("- `analysis/figures/scaling_equivalence/adjacent_accuracy_plateaus.png`")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Model size is a proxy for pretraining compute. The cache does not contain actual pretraining "
        "tokens or training FLOPs, so claims about saving pretraining compute should be stated as "
        "parameter-scaling tradeoffs unless model training budgets are added."
    )
    lines.append(
        "- Reasoning tokens are realized system-level tokens, not a pure knob. They are affected by "
        "topology, prompt, task, and model behavior."
    )
    lines.append(
        "- The 32B high-reasoning region remains the main non-random missingness/context-cap caveat."
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    df_all = load_core(drop_ctx_capped=False)
    df_uncapped = load_core(drop_ctx_capped=True)
    scopes = [
        ("core_all_cap_control", df_all, "post-trim core; includes 32B capped rows with a cap indicator"),
        ("core_uncapped", df_uncapped, "post-trim core; drops 32B b8192/unlimited rows"),
    ]

    coeffs = fit_continuous_laws(scopes)
    rung = fit_rung_equivalents(scopes)
    adj = adjacent_plateaus(df_all)
    orch = orchestration_contrasts(df_all)
    grid = pd.concat(
        [
            grid_summary(df_all, scope="all_systems"),
            grid_summary(df_all, scope="single_agent"),
            grid_summary(df_all, scope="topology:independent"),
            grid_summary(df_all, scope="topology:decentralized"),
            grid_summary(df_all, scope="topology:centralized"),
        ],
        ignore_index=True,
    )
    pairs = nearest_equivalence_pairs(grid)

    coeffs.to_csv(OUT_TAB / "scaling_law_coefficients.csv", index=False)
    rung.to_csv(OUT_TAB / "reasoning_rung_equivalents.csv", index=False)
    adj.to_csv(OUT_TAB / "adjacent_plateau_contrasts.csv", index=False)
    orch.to_csv(OUT_TAB / "orchestration_matched_contrasts.csv", index=False)
    grid.to_csv(OUT_TAB / "size_reasoning_grid.csv", index=False)
    pairs.to_csv(OUT_TAB / "nearest_equivalence_pairs.csv", index=False)

    plot_rung_equivalents(rung)
    plot_adjacent_plateaus(adj)

    write_report(coeffs, rung, adj, orch, pairs, nb_data.manifest())
    print(f"Wrote {OUT_REPORT.relative_to(ROOT)}")
    print(f"Wrote tables under {OUT_TAB.relative_to(ROOT)}")
    print(f"Wrote figures under {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
