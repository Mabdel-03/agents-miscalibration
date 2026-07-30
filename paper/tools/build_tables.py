#!/usr/bin/env python
"""Generate every LaTeX table for the full_sweep_v1 report.

Two families of tables are produced:

  T*  derived from the nine tidy CSVs already written by
      ``analysis/scripts/{axis_relationships,scaling_equivalence}.py``
  P*  derived directly from the cached parquets, covering material the CSVs do
      not carry (attrition, coverage, seed replication, sensitivity analyses,
      item difficulty, round dynamics, the column glossary)

Usage:  build_tables.py --out tables [--only GLOB]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from fnmatch import fnmatch as _fn
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperdata as D  # noqa: E402
import texlib as T  # noqa: E402

sys.path.insert(0, str(D.REPO))
from analysis.nb_lib import boot as _boot  # noqa: E402

OUT = Path("tables")
BUILDERS: dict[str, callable] = {}
REL_AXIS = "axis_relationships"
REL_SCAL = "scaling_equivalence"
SRC_AXIS = "analysis/tables/axis_relationships"
SRC_SCAL = "analysis/tables/scaling_equivalence"


def table(name: str):
    def deco(fn):
        BUILDERS[name] = fn
        return fn

    return deco


def _bench_col(df, col="benchmark"):
    return df[col].map(lambda b: D.BENCH_LABEL.get(str(b), str(b)))


def _order(df, col, order):
    rank = {v: i for i, v in enumerate(order)}
    return df.assign(__k=df[col].map(lambda v: rank.get(str(v), 99))).sort_values(
        "__k", kind="stable"
    ).drop(columns="__k")


LEVEL_ORDER = (
    D.SIZE_ORDER
    + D.REAS_ORDER
    + D.TOPO_ORDER
    + D.CTX_ORDER
    + ["0", "1", "3", "L0", "L3"]
)


# =========================================================================== #
# T1-T2  marginal means
# =========================================================================== #
def _marginals_frame(metric: str) -> pd.DataFrame:
    s = D.csv(f"{REL_AXIS}/axis_summary_long.csv")
    s = s[s.metric == metric].copy()
    rank = {v: i for i, v in enumerate(LEVEL_ORDER)}
    axrank = {a: i for i, a in enumerate(D.AXIS_LABEL)}
    s["__a"] = s.axis.map(lambda a: axrank.get(a, 99))
    s["__l"] = s.level.astype(str).map(lambda v: rank.get(v, 99))
    s["__b"] = s.benchmark.map({b: i for i, b in enumerate(D.BENCHMARKS)})
    s = s.sort_values(["__a", "__l", "__b"], kind="stable")
    d = 0 if metric in ("cost", "mean_total_tokens", "mean_reasoning_tokens") else 3
    body = pd.DataFrame(
        {
            "Axis": [T.pretty(a) for a in s.axis],
            "Level": [T.pretty(v) for v in s.level],
            "Benchmark": [T.pretty(b) for b in s.benchmark],
            "Mean [95\\% CI]": T.ci(s["mean"], s.ci_low, s.ci_high, d=d),
            "$n_{\\mathrm{des}}$": [T.intfmt(v) for v in s.n_designs],
            "$n_{\\mathrm{cell}}$": [T.intfmt(v) for v in s.n_seed_cells],
        }
    )
    return body


@table("tab_marginals_accuracy")
def _():
    body = _marginals_frame("accuracy")
    T.emit(
        body,
        OUT / "tab_marginals_accuracy.tex",
        caption=(
            "Marginal accuracy by axis level and benchmark over the post-trim core "
            "view. Intervals are hierarchical qid-then-seed bootstrap percentile "
            "intervals. $n_{\\mathrm{des}}$ counts distinct design cells, "
            "$n_{\\mathrm{cell}}$ counts seed-level cells."
        ),
        label="tab:marginals-accuracy",
        column_format="llrlrr",
        longtable=True,
        source=f"{SRC_AXIS}/axis_summary_long.csv",
    )


for _m in [
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
]:

    def _mk(metric=_m):
        @table(f"tab_marginals_{metric}")
        def _():
            body = _marginals_frame(metric)
            lbl = D.METRIC_LABEL.get(metric, metric)
            T.emit(
                body,
                OUT / f"tab_marginals_{metric}.tex",
                caption=(
                    f"Marginal {lbl} by axis level and benchmark, post-trim core view, "
                    "with hierarchical bootstrap percentile intervals."
                ),
                label=f"tab:marginals-{metric.replace('_', '-')}",
                column_format="llrlrr",
                longtable=True,
                source=f"{SRC_AXIS}/axis_summary_long.csv",
            )

    _mk()


# =========================================================================== #
# T3-T4  matched adjacent contrasts
# =========================================================================== #
def _contrast_frame(df: pd.DataFrame, d: int = 3) -> pd.DataFrame:
    axrank = {a: i for i, a in enumerate(D.AXIS_LABEL)}
    df = df.assign(__a=df.axis.map(lambda a: axrank.get(a, 99))).sort_values(
        ["__a", "contrast", "benchmark"], kind="stable"
    )
    return pd.DataFrame(
        {
            "Axis": [T.pretty(a) for a in df.axis],
            "Contrast": [T.pretty(c) for c in df.contrast],
            "Benchmark": [T.pretty(b) for b in df.benchmark],
            "$\\Delta$ [95\\% CI]": T.ci(df.delta, df.ci_low, df.ci_high, d=d, signed=True),
            "": T.sig(df.ci_low, df.ci_high),
            "$n_{\\mathrm{pair}}$": [T.intfmt(v) for v in df.n_matched_seed_pairs],
            "$n_{\\mathrm{des}}$": [T.intfmt(v) for v in df.n_matched_design_pairs],
        }
    )


@table("tab_contrasts_accuracy")
def _():
    c = D.csv(f"{REL_AXIS}/axis_matched_contrasts.csv")
    body = _contrast_frame(c[c.metric == "accuracy"])
    T.emit(
        body,
        OUT / "tab_contrasts_accuracy.tex",
        caption=(
            "Matched adjacent-level accuracy contrasts. Each contrast is formed only "
            "from cell pairs that agree on every other axis and on seed, so it is a "
            "within-design difference rather than a marginal comparison. "
            "$\\ast$ marks an interval excluding zero."
        ),
        label="tab:contrasts-accuracy",
        column_format="lllrlrr",
        longtable=True,
        source=f"{SRC_AXIS}/axis_matched_contrasts.csv",
    )


for _m in [
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
]:

    def _mk(metric=_m):
        @table(f"tab_contrasts_{metric}")
        def _():
            c = D.csv(f"{REL_AXIS}/axis_matched_contrasts.csv")
            sub = c[c.metric == metric]
            body = _contrast_frame(sub, d=0 if metric == "cost" else 3)
            T.emit(
                body,
                OUT / f"tab_contrasts_{metric}.tex",
                caption=(
                    f"Matched adjacent-level contrasts for {D.METRIC_LABEL.get(metric, metric)}. "
                    "$\\ast$ marks an interval excluding zero."
                ),
                label=f"tab:contrasts-{metric.replace('_', '-')}",
                column_format="lllrlrr",
                longtable=True,
                source=f"{SRC_AXIS}/axis_matched_contrasts.csv",
            )

    _mk()


# =========================================================================== #
# T5  stratum-consistency digest
# =========================================================================== #
@table("tab_stratum_digest")
def _():
    s = D.csv(f"{REL_AXIS}/axis_stratified_contrasts.csv")
    pooled = D.csv(f"{REL_AXIS}/axis_matched_contrasts.csv")
    pooled_map = {
        (r.axis, r.contrast, r.metric, r.benchmark): r.delta for r in pooled.itertuples()
    }
    keep = ["accuracy", "vote_ece_prim", "delta_vote_prim", "cost"]
    s = s[s.metric.isin(keep)].copy()
    s["sig"] = (s.ci_low > 0) | (s.ci_high < 0)
    s["pooled"] = [
        pooled_map.get((r.axis, r.contrast, r.metric, r.benchmark), np.nan)
        for r in s.itertuples()
    ]
    s["agrees"] = np.sign(s.delta) == np.sign(s.pooled)
    g = (
        s.groupby(["axis", "contrast", "metric"], observed=True)
        .agg(
            k=("delta", "size"),
            med=("delta", "median"),
            q25=("delta", lambda x: x.quantile(0.25)),
            q75=("delta", lambda x: x.quantile(0.75)),
            frac_sig=("sig", "mean"),
            frac_agree=("agrees", "mean"),
            min_pairs=("n_matched_seed_pairs", "min"),
        )
        .reset_index()
    )
    mrank = {m: i for i, m in enumerate(keep)}
    g = g.assign(__m=g.metric.map(mrank)).sort_values(
        ["__m", "axis", "contrast"], kind="stable"
    )
    body = pd.DataFrame(
        {
            "Metric": [T.pretty(m) for m in g.metric],
            "Axis": [T.pretty(a) for a in g.axis],
            "Contrast": [T.pretty(c) for c in g.contrast],
            "$k$": [T.intfmt(v) for v in g.k],
            "Median [IQR]": [
                f"{T.num(m, 3, signed=True)} [{T.num(a)}, {T.num(b)}]"
                for m, a, b in zip(g.med, g.q25, g.q75)
            ],
            "\\% CI excl.\\ 0": [T.num(100 * v, 0) for v in g.frac_sig],
            "\\% sign of pooled": [T.num(100 * v, 0) for v in g.frac_agree],
            "min $n_{\\mathrm{pair}}$": [T.intfmt(v) for v in g.min_pairs],
        }
    )
    T.emit(
        body,
        OUT / "tab_stratum_digest.tex",
        caption=(
            "Stratum-consistency digest. Every contrast in "
            "\\texttt{axis\\_stratified\\_contrasts.csv} is recomputed inside each "
            "benchmark $\\times$ model size $\\times$ reasoning-budget stratum; the table "
            "reports how many strata exist ($k$), the median and interquartile range of "
            "the stratum-level effect, the share of strata whose interval excludes zero, "
            "and the share whose sign matches the pooled matched contrast."
        ),
        label="tab:stratum-digest",
        column_format="lllrlrrr",
        source=f"{SRC_AXIS}/axis_stratified_contrasts.csv",
    )


# =========================================================================== #
# T6  full stratified contrasts, landscape longtables
# =========================================================================== #
def _strat_table(axis: str, metric: str, stem: str, caption: str):
    s = D.csv(f"{REL_AXIS}/axis_stratified_contrasts.csv")
    sub = s[(s.axis == axis) & (s.metric == metric)].copy()
    if sub.empty:
        return
    sub = D.sort_axis(sub, "benchmark", "model_size", "reasoning_level")
    body = pd.DataFrame(
        {
            "Contrast": [T.pretty(c) for c in sub.contrast],
            "Benchmark": [T.pretty(b) for b in sub.benchmark],
            "Size": [T.pretty(v) for v in sub.model_size],
            "Reasoning": [T.pretty(v) for v in sub.reasoning_level],
            "$\\Delta$ [95\\% CI]": T.ci(
                sub.delta, sub.ci_low, sub.ci_high, d=1 if metric == "cost" else 3, signed=True
            ),
            "": T.sig(sub.ci_low, sub.ci_high),
            "$n_{\\mathrm{pair}}$": [T.intfmt(v) for v in sub.n_matched_seed_pairs],
            "$n_{\\mathrm{des}}$": [T.intfmt(v) for v in sub.n_matched_design_pairs],
        }
    )
    T.emit(
        body,
        OUT / f"{stem}.tex",
        caption=caption,
        label="tab:" + stem.replace("tab_", "").replace("_", "-"),
        column_format="llllrlrr",
        longtable=True,
        fontsize=r"\footnotesize",
        source=f"{SRC_AXIS}/axis_stratified_contrasts.csv",
    )


for _axis in ["topology", "context_share_level", "prompt_complexity_level", "n_agents"]:
    for _metric in ["accuracy", "vote_ece_prim", "cost"]:

        def _mk(axis=_axis, metric=_metric):
            stem = f"tab_strat_{axis}_{metric}".replace("__", "_")

            @table(stem)
            def _():
                _strat_table(
                    axis,
                    metric,
                    stem,
                    caption=(
                        f"Fully stratified {D.METRIC_LABEL.get(metric, metric)} contrasts on the "
                        f"{D.AXIS_LABEL.get(axis, axis)} axis, one row per benchmark "
                        "$\\times$ model size $\\times$ reasoning-budget stratum. "
                        "$\\ast$ marks an interval excluding zero."
                    ),
                )

        _mk()


# =========================================================================== #
# T7-T12  scaling equivalence
# =========================================================================== #
@table("tab_scaling_coefficients")
def _():
    c = D.csv(f"{REL_SCAL}/scaling_law_coefficients.csv")
    c = D.sort_axis(c, "benchmark")
    c = c.sort_values(["scope", "benchmark", "fit_group"], kind="stable")
    body = pd.DataFrame(
        {
            "Scope": [T.esc(s.replace("core_", "").replace("_", " ")) for s in c.scope],
            "Fit group": [T.pretty(f.replace("topology:", "")) for f in c.fit_group],
            "Bench.": [T.pretty(b) for b in c.benchmark],
            "$\\beta_P$ (SE)": [
                f"{T.num(a)} ({T.num(b)})" for a, b in zip(c.coef_log2_params, c.se_log2_params)
            ],
            "$\\beta_R$ (SE)": [
                f"{T.num(a)} ({T.num(b)})"
                for a, b in zip(c.coef_log2_reasoning_tokens, c.se_log2_reasoning_tokens)
            ],
            "$\\beta_P/\\beta_R$": [T.num(v, 2) for v in c.reasoning_doublings_per_param_doubling],
            "$n_{\\mathrm{row}}$": [T.intfmt(v) for v in c.n_rows],
            "dev.\\ $R^2$": [T.num(v) for v in c.deviance_r2],
        }
    )
    T.emit(
        body,
        OUT / "tab_scaling_coefficients.tex",
        caption=(
            "Fitted coefficients of $\\operatorname{logit}(\\mathrm{accuracy}) = "
            "\\beta_P \\log_2(\\mathrm{params}_B) + \\beta_R \\log_2(1 + \\mathrm{reasoning "
            "tokens}) + \\mathrm{controls}$, where the controls are topology, context sharing, "
            "prompt level and a 32B context-cap indicator. $\\beta_P/\\beta_R$ is the number of "
            "reasoning-token doublings that substitute for one parameter doubling. The "
            "per-topology fit groups are reported here for the first time; the source memo "
            "printed only the pooled \\emph{all systems} row."
        ),
        label="tab:scaling-coefficients",
        column_format="lllllrrr",
        longtable=True,
        source=f"{SRC_SCAL}/scaling_law_coefficients.csv",
    )


@table("tab_rung_equivalents")
def _():
    c = D.csv(f"{REL_SCAL}/reasoning_rung_equivalents.csv")
    c = D.sort_axis(c, "benchmark", "reasoning_level")
    rr = {v: i for i, v in enumerate(D.REAS_ORDER)}
    c = c.assign(__r=c.reasoning_rung.map(lambda v: rr.get(v, 9))).sort_values(
        ["scope", "fit_group", "benchmark", "__r"], kind="stable"
    )
    body = pd.DataFrame(
        {
            "Scope": [T.esc(s.replace("core_", "").replace("_", " ")) for s in c.scope],
            "Fit group": [T.pretty(f.replace("topology:", "")) for f in c.fit_group],
            "Bench.": [T.pretty(b) for b in c.benchmark],
            "Rung": [T.pretty(v) for v in c.reasoning_rung],
            "Gain vs off (SE)": [
                f"{T.num(a, 3, signed=True)} ({T.num(b)})"
                for a, b in zip(c.rung_logit_gain_vs_off, c.se_rung_logit_gain_vs_off)
            ],
            "Doublings": [T.num(v, 2) for v in c.equiv_param_doublings_vs_off],
            "Multiplier": [T.num(v, 2) for v in c.equiv_param_multiplier_vs_off],
        }
    )
    T.emit(
        body,
        OUT / "tab_rung_equivalents.tex",
        caption=(
            "Each reasoning rung expressed as an equivalent parameter multiplier, obtained by "
            "dividing the rung's logit gain over \\texttt{off} by the fitted $\\beta_P$ of the "
            "same model."
        ),
        label="tab:rung-equivalents",
        column_format="lllllrr",
        longtable=True,
        fontsize=r"\footnotesize",
        source=f"{SRC_SCAL}/reasoning_rung_equivalents.csv",
    )


@table("tab_plateaus")
def _():
    c = D.csv(f"{REL_SCAL}/adjacent_plateau_contrasts.csv")
    c = D.sort_axis(c, "benchmark")
    c = c.sort_values(["axis", "contrast", "benchmark"], kind="stable")
    body = pd.DataFrame(
        {
            "Axis": [T.pretty(a) for a in c.axis],
            "Contrast": [T.pretty(v) for v in c.contrast],
            "Benchmark": [T.pretty(b) for b in c.benchmark],
            "$\\Delta$ accuracy [95\\% CI]": T.ci(c.delta, c.ci_low, c.ci_high, signed=True),
            "Plateau-like": ["yes" if bool(v) else "no" for v in c.plateau_like],
            "$n_{\\mathrm{pair}}$": [T.intfmt(v) for v in c.n_matched_seed_pairs],
        }
    )
    T.emit(
        body,
        OUT / "tab_plateaus.tex",
        caption=(
            "Adjacent-rung accuracy contrasts on the capacity and reasoning axes, with the "
            "plateau flag set when the interval contains zero."
        ),
        label="tab:plateaus",
        column_format="lllrlr",
        longtable=True,
        source=f"{SRC_SCAL}/adjacent_plateau_contrasts.csv",
    )


@table("tab_orchestration")
def _():
    c = D.csv(f"{REL_SCAL}/orchestration_matched_contrasts.csv")
    c = D.sort_axis(c, "benchmark")
    c = c.sort_values(["metric", "contrast", "benchmark"], kind="stable")
    body = pd.DataFrame(
        {
            "Metric": [T.pretty(m) for m in c.metric],
            "Contrast": [T.pretty(v) for v in c.contrast],
            "Benchmark": [T.pretty(b) for b in c.benchmark],
            "$\\Delta$ [95\\% CI]": [
                f"{T.num(a, 1 if m == 'cost' else 3, signed=True)} "
                f"[{T.num(b, 1 if m == 'cost' else 3)}, {T.num(cc, 1 if m == 'cost' else 3)}]"
                for a, b, cc, m in zip(c.delta, c.ci_low, c.ci_high, c.metric)
            ],
            "": T.sig(c.ci_low, c.ci_high),
            "$n_{\\mathrm{pair}}$": [T.intfmt(v) for v in c.n_matched_seed_pairs],
        }
    )
    T.emit(
        body,
        OUT / "tab_orchestration.tex",
        caption=(
            "Matched topology contrasts across all six reported outcome metrics. "
            "$\\ast$ marks an interval excluding zero."
        ),
        label="tab:orchestration",
        column_format="lllrlr",
        longtable=True,
        source=f"{SRC_SCAL}/orchestration_matched_contrasts.csv",
    )


@table("tab_size_reasoning_grid")
def _():
    c = D.csv(f"{REL_SCAL}/size_reasoning_grid.csv")
    c = c.sort_values(["scope", "benchmark", "size_rank", "reasoning_rank"], kind="stable")
    body = pd.DataFrame(
        {
            "Scope": [T.pretty(s) for s in c.scope],
            "Benchmark": [T.pretty(b) for b in c.benchmark],
            "Size": [T.pretty(v) for v in c.model_size],
            "Reasoning": [T.pretty(v) for v in c.reasoning_level],
            "Accuracy": [T.num(v) for v in c.accuracy],
            "Cost proxy": [T.num(v, 1) for v in c.cost],
            "Total tok.": [T.num(v, 1) for v in c.mean_total_tokens],
            "Reas.\\ tok.": [T.num(v, 1) for v in c.mean_reasoning_tokens],
            "$n_{\\mathrm{des}}$": [T.intfmt(v) for v in c.n_designs],
        }
    )
    T.emit(
        body,
        OUT / "tab_size_reasoning_grid.tex",
        caption=(
            "Complete capacity $\\times$ reasoning grid of accuracy, realized token counts and "
            "the cost proxy $\\mathrm{params} \\times \\mathrm{mean\\ total\\ tokens}$, "
            "reported for the pooled fit and for each topology separately."
        ),
        label="tab:size-reasoning-grid",
        column_format="llllrrrrr",
        longtable=True,
        fontsize=r"\footnotesize",
        source=f"{SRC_SCAL}/size_reasoning_grid.csv",
    )


@table("tab_nearest_equivalences")
def _():
    c = D.csv(f"{REL_SCAL}/nearest_equivalence_pairs.csv")
    c = c.sort_values(["scope", "benchmark", "abs_accuracy_gap"], kind="stable")
    body = pd.DataFrame(
        {
            "Scope": [T.pretty(s) for s in c.scope],
            "Benchmark": [T.pretty(b) for b in c.benchmark],
            "Target": [
                f"{T.pretty(a)}, {T.pretty(b)}"
                for a, b in zip(c.target_model_size, c.target_reasoning)
            ],
            "Target acc.": [T.num(v) for v in c.target_accuracy],
            "Candidate": [
                f"{T.pretty(a)}, {T.pretty(b)}"
                for a, b in zip(c.candidate_model_size, c.candidate_reasoning)
            ],
            "Cand.\\ acc.": [T.num(v) for v in c.candidate_accuracy],
            "Gap": [T.num(v, 3, signed=True) for v in c.accuracy_gap_candidate_minus_target],
            "Cost ratio": [T.num(v, 2) for v in c.candidate_cost_over_target_cost],
        }
    )
    T.emit(
        body,
        OUT / "tab_nearest_equivalences.tex",
        caption=(
            "Nearest empirical equivalences: for each low-reasoning target configuration, the "
            "smaller model with a larger reasoning budget whose accuracy is closest, together "
            "with the ratio of their cost proxies."
        ),
        label="tab:nearest-equivalences",
        column_format="lllrlrrr",
        longtable=True,
        fontsize=r"\footnotesize",
        source=f"{SRC_SCAL}/nearest_equivalence_pairs.csv",
    )


# =========================================================================== #
# P1  attrition ledger
# =========================================================================== #
@table("tab_attrition")
def _():
    m = D.manifest()
    c, v = D.cells(), D.view()
    dropped = c[~c.cell_id.isin(set(v.cell_id))]
    rows = [
        ("Cell directories on disk", m["n_cell_dirs"], "includes partial, pre-trim and touched directories"),
        ("Semantically complete cells", m["n_cells_complete"], "cell has a \\texttt{meta.json} and a parseable result stream"),
        ("Cells in the post-trim core view", len(v), "\\texttt{is\\_core}: prompt level in $\\{0,3\\}$ and context in $\\{$artifact only, plus CoT$\\}$"),
        ("Cells dropped by the core filter", len(dropped), "retained in \\texttt{cells\\_dedup\\_v1} but outside the analysis population"),
        ("Distinct design cells in the view", v.design_cell.nunique(), "configuration without seed"),
        ("Item rows after deduplication", 858089, "one row per graded question"),
        ("Agent-round rows after deduplication", 3952802, "one row per agent per round"),
        ("Duplicate qid rows dropped", m["n_dupes_dropped"], f"across {m['n_cells_with_dupes']} cells"),
        ("Malformed JSONL lines skipped", m["n_bad_lines"], "consistent with preemption partial writes"),
        ("Unrecoverable cells", m["n_bad_cells"], ""),
        ("Cells with unexpected question counts", m["n_unexpected_n"], "after deduplication"),
        ("Bootstrap replicates per cell", m["boot_B"], "fixed-bin paired resample"),
    ]
    body = pd.DataFrame(
        {
            "Stage": [r[0] for r in rows],
            "Count": [T.intfmt(r[1]) for r in rows],
            "Definition": [r[2] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_attrition.tex",
        caption=(
            "Attrition ledger from raw cell directories to the analysis population. "
            "Counts come from \\texttt{analysis/cache/ingest\\_manifest\\_v1.json} and from "
            "the cached tables themselves."
        ),
        label="tab:attrition",
        column_format="lrp{0.50\\linewidth}",
        source="analysis/cache/ingest_manifest_v1.json",
    )


@table("tab_attrition_breakdown")
def _():
    c, v = D.cells(), D.view()
    dropped = c[~c.cell_id.isin(set(v.cell_id))]
    parts = []
    for col, label in [
        ("prompt_complexity_level", "prompt complexity"),
        ("context_share_level", "context sharing"),
        ("topology", "topology"),
        ("model_size", "model size"),
        ("reasoning_level", "reasoning budget"),
        ("benchmark", "benchmark"),
    ]:
        tot = c[col].astype(str).value_counts()
        drp = dropped[col].astype(str).value_counts()
        kept = v[col].astype(str).value_counts()
        for lvl in tot.index:
            parts.append(
                {
                    "Factor": label,
                    "Level": lvl,
                    "Complete": int(tot.get(lvl, 0)),
                    "In view": int(kept.get(lvl, 0)),
                    "Dropped": int(drp.get(lvl, 0)),
                }
            )
    df = pd.DataFrame(parts)
    rank = {v_: i for i, v_ in enumerate(LEVEL_ORDER + D.BENCHMARKS)}
    df = df.assign(__k=df.Level.map(lambda x: rank.get(str(x), 99))).sort_values(
        ["Factor", "__k"], kind="stable"
    )
    body = pd.DataFrame(
        {
            "Factor": [T.esc(x) for x in df.Factor],
            "Level": [T.pretty(x) for x in df.Level],
            "Complete cells": [T.intfmt(x) for x in df.Complete],
            "In core view": [T.intfmt(x) for x in df["In view"]],
            "Dropped": [T.intfmt(x) for x in df.Dropped],
            "\\% dropped": [
                T.num(100 * d / t if t else np.nan, 1)
                for d, t in zip(df.Dropped, df.Complete)
            ],
        }
    )
    T.emit(
        body,
        OUT / "tab_attrition_breakdown.tex",
        caption=(
            "Where the 1{,}100 cells outside the core view come from. Prompt levels 1 and 2 "
            "and the \\texttt{plus\\_intermediate} context level were removed from the grid by "
            "the 2026-06-10 trim, and the prompt-level-1 single-agent baselines are excluded "
            "by the \\texttt{is\\_core} definition even though the sweep still generates them."
        ),
        label="tab:attrition-breakdown",
        column_format="llrrrr",
        longtable=True,
        source="analysis/cache/{cells_dedup_v1,analysis_view_v1}.parquet",
    )


# =========================================================================== #
# P3  coverage
# =========================================================================== #
@table("tab_coverage")
def _():
    v = D.view()
    g = (
        v.groupby(["model_size", "reasoning_level", "topology"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            n_designs=("design_cell", "nunique"),
            n_seeds=("seed", "nunique"),
            n_questions=("n_questions", "sum"),
        )
        .reset_index()
    )
    g = D.sort_axis(g, "model_size", "reasoning_level", "topology")
    body = pd.DataFrame(
        {
            "Size": [T.pretty(x) for x in g.model_size],
            "Reasoning": [T.pretty(x) for x in g.reasoning_level],
            "Topology": [T.pretty(x) for x in g.topology],
            "Cells": [T.intfmt(x) for x in g.n_cells],
            "Designs": [T.intfmt(x) for x in g.n_designs],
            "Seeds": [T.intfmt(x) for x in g.n_seeds],
            "Questions": [T.intfmt(x) for x in g.n_questions],
        }
    )
    T.emit(
        body,
        OUT / "tab_coverage.tex",
        caption=(
            "Coverage of the post-trim core view by capacity, reasoning budget and topology. "
            "The 32B long-reasoning corner is the thinnest region of the design space and "
            "carries the context-cap flag."
        ),
        label="tab:coverage",
        column_format="lllrrrr",
        longtable=True,
        source="analysis/cache/analysis_view_v1.parquet",
    )


@table("tab_coverage_benchmark")
def _():
    v = D.view()
    g = (
        v.groupby(["benchmark", "topology", "context_share_level", "prompt_complexity_level"], observed=True)
        .agg(n_cells=("cell_id", "size"), n_designs=("design_cell", "nunique"),
             acc=("accuracy", "mean"))
        .reset_index()
    )
    g = D.sort_axis(g, "benchmark", "topology", "context_share_level")
    body = pd.DataFrame(
        {
            "Benchmark": [T.pretty(x) for x in g.benchmark],
            "Topology": [T.pretty(x) for x in g.topology],
            "Context": [T.pretty(x) for x in g.context_share_level],
            "Prompt": [f"L{int(x)}" for x in g.prompt_complexity_level],
            "Cells": [T.intfmt(x) for x in g.n_cells],
            "Designs": [T.intfmt(x) for x in g.n_designs],
            "Unweighted accuracy": [T.num(x) for x in g.acc],
        }
    )
    T.emit(
        body,
        OUT / "tab_coverage_benchmark.tex",
        caption=(
            "Cell counts and unweighted mean accuracy by benchmark, topology, context-sharing "
            "level and prompt level. Unweighted means are shown for completeness; every "
            "inferential statement in this report uses the design-weighted bootstrap instead."
        ),
        label="tab:coverage-benchmark",
        column_format="llllrrr",
        longtable=True,
        source="analysis/cache/analysis_view_v1.parquet",
    )


# =========================================================================== #
# P4  seed replication
# =========================================================================== #
@table("tab_seed_replication")
def _():
    v = D.view()
    g = v.groupby("design_cell", observed=True).agg(
        k=("accuracy", "size"), sd=("accuracy", "std"), n=("n_questions", "mean"),
        bench=("benchmark", "first"), size=("model_size", "first"),
    )
    rows = []
    for nseed in sorted(g.k.unique()):
        sub = g[g.k == nseed]
        rows.append(
            {
                "Seeds per design": int(nseed),
                "Design cells": len(sub),
                "Median SD": sub.sd.median(),
                "Mean SD": sub.sd.mean(),
                "P90 SD": sub.sd.quantile(0.9),
                "Binomial expectation": float(
                    np.sqrt(0.6 * 0.4 / sub.n.mean()) if len(sub) else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Seeds per design": [T.intfmt(x) for x in df["Seeds per design"]],
            "Design cells": [T.intfmt(x) for x in df["Design cells"]],
            "Median SD": [T.num(x, 4) for x in df["Median SD"]],
            "Mean SD": [T.num(x, 4) for x in df["Mean SD"]],
            "P90 SD": [T.num(x, 4) for x in df["P90 SD"]],
            "Binomial SD at $p{=}0.6$": [T.num(x, 4) for x in df["Binomial expectation"]],
        }
    )
    T.emit(
        body,
        OUT / "tab_seed_replication.tex",
        caption=(
            "Across-seed accuracy dispersion within design cells, compared with the binomial "
            "standard deviation expected from question sampling alone at the observed mean "
            "accuracy. Seed variation exceeding the binomial expectation indicates genuine "
            "sampling-temperature variance rather than question-set noise."
        ),
        label="tab:seed-replication",
        column_format="rrrrrr",
        source="analysis/cache/analysis_view_v1.parquet",
    )


# =========================================================================== #
# P5-P6  sensitivity analyses
# =========================================================================== #
def _sens(flag_col: str, stem: str, caption: str, label: str, note: str):
    v = D.view()
    metrics = ["accuracy", "pa_ece_prim", "vote_ece_prim", "delta_vote_prim", "delta_fp_prim"]
    rows = []
    for bench in D.BENCHMARKS:
        for topo in D.TOPO_ORDER:
            sub = v[(v.benchmark == bench) & (v.topology == topo)]
            if sub.empty:
                continue
            excl = sub[~sub[flag_col].astype(bool)]
            for m in metrics:
                a, b = sub[m].mean(), excl[m].mean()
                if not np.isfinite(a) and not np.isfinite(b):
                    continue
                rows.append(
                    {
                        "benchmark": bench, "topology": topo, "metric": m,
                        "n_all": len(sub), "n_excl": len(excl),
                        "all": a, "excl": b, "diff": b - a,
                    }
                )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Benchmark": [T.pretty(x) for x in df.benchmark],
            "Topology": [T.pretty(x) for x in df.topology],
            "Metric": [D.METRIC_LABEL.get(x, x) for x in df.metric],
            "All cells": [T.num(x) for x in df["all"]],
            "Flagged excluded": [T.num(x) for x in df.excl],
            "Difference": [T.num(x, 4, signed=True) for x in df["diff"]],
            "$n$ all": [T.intfmt(x) for x in df.n_all],
            "$n$ excl.": [T.intfmt(x) for x in df.n_excl],
        }
    )
    T.emit(
        body,
        OUT / f"{stem}.tex",
        caption=caption,
        label=label,
        column_format="lllrrrrr",
        longtable=True,
        fontsize=r"\footnotesize",
        note=note,
        source="analysis/cache/analysis_view_v1.parquet",
    )


@table("tab_sens_ctxcap")
def _():
    _sens(
        "ctx32b_capped",
        "tab_sens_ctxcap",
        caption=(
            "Sensitivity of the headline outcome means to the 32B context cap. The "
            "\\emph{excl.} columns drop every cell carrying \\texttt{ctx32b\\_capped}, that is "
            "every 32B cell served at $\\texttt{max\\_model\\_len}=16384$ under the "
            "\\texttt{b8192} or \\texttt{unlimited} reasoning budget."
        ),
        label="tab:sens-ctxcap",
        note=(
            "106 rows of the core view carry the flag. Where a column pair differs materially, "
            "the corresponding 32B claim in the main text should be read as capacity-limited "
            "rather than capability-limited."
        ),
    )


@table("tab_sens_era")
def _():
    v = D.view().copy()
    v["__pre"] = ~v.era_post_trim.astype(bool)
    metrics = ["accuracy", "pa_ece_prim", "vote_ece_prim", "delta_vote_prim"]
    rows = []
    for bench in D.BENCHMARKS:
        for era, name in [(False, "pre-trim"), (True, "post-trim")]:
            sub = v[(v.benchmark == bench) & (v.era_post_trim.astype(bool) == era)]
            if sub.empty:
                continue
            rec = {"benchmark": bench, "era": name, "n": len(sub)}
            for m in metrics:
                rec[m] = sub[m].mean()
            rows.append(rec)
    df = pd.DataFrame(rows)
    cols = {
        "Benchmark": [T.pretty(x) for x in df.benchmark],
        "Execution era": [T.esc(x) for x in df.era],
        "$n$ cells": [T.intfmt(x) for x in df.n],
    }
    for m in metrics:
        cols[D.METRIC_LABEL[m]] = [T.num(x) for x in df[m]]
    T.emit(
        pd.DataFrame(cols),
        OUT / "tab_sens_era.tex",
        caption=(
            "Sensitivity to execution era. \\texttt{era\\_post\\_trim} is a calendar flag "
            "derived from each cell's \\texttt{started\\_at} timestamp against the trim commit's "
            "epoch, and is distinct from the factor-level \\texttt{is\\_core} membership. A "
            "material pre/post difference would indicate that serving-stack changes confound "
            "the design."
        ),
        label="tab:sens-era",
        column_format="llr" + "r" * len(metrics),
        source="analysis/cache/analysis_view_v1.parquet",
    )


# =========================================================================== #
# P7  column glossary
# =========================================================================== #
GLOSS = {
    "cell_id": "Deterministic configuration identifier; see Section 2.1.",
    "model_size": "Capacity axis level.",
    "param_count": "Parameter count in billions, from the model registry.",
    "topology": "Coordination structure.",
    "context_share_level": "What peers see of one another.",
    "prompt_complexity_level": "System-prompt level.",
    "reasoning_level": "Thinking-token budget rung.",
    "benchmark": "Evaluation suite.",
    "seed": "Replication seed; also drives option shuffling.",
    "n_questions": "Graded questions in the cell after deduplication.",
    "accuracy": "Fraction of questions answered correctly by the system.",
    "error_rate": "$1 - \\mathrm{accuracy}$.",
    "mean_reasoning_tokens": "Mean realized thinking tokens per question, summed over agents.",
    "mean_total_tokens": "Mean total generated tokens per question, summed over agents.",
    "mean_turns": "Mean generation turns per question.",
    "mean_messages": "Mean inter-agent messages per question.",
    "pa_ece": "Per-agent ECE, 15-bin equal-width plug-in (package parity column).",
    "sys_ece_vote": "System ECE on vote fraction, 15-bin plug-in.",
    "sys_ece_fp": "System ECE on final-producer logprob, 15-bin plug-in.",
    "delta_vote": "$\\texttt{sys\\_ece\\_vote} - \\texttt{pa\\_ece}$, plug-in estimators.",
    "delta_fp": "$\\texttt{sys\\_ece\\_fp} - \\texttt{pa\\_ece}$, plug-in estimators.",
    "is_mas": "True when topology is not single agent.",
    "n_raw_rows": "Rows read before deduplication.",
    "n_dupes_dropped": "Duplicate-qid rows removed.",
    "n_bad_lines": "Malformed JSONL lines skipped.",
    "started_at": "Unix epoch of cell start.",
    "finished_at": "Unix epoch of cell completion.",
    "prompt_token_count": "Exact token count of the system prompt under the profile tokenizer.",
    "acc_first_sample": "Accuracy using only the first self-consistency sample.",
    "pa_n": "Per-agent calibration items available.",
    "pa_ece_prim": "Per-agent ECE, pre-registered equal-mass 10-bin estimator.",
    "pa_ece_db": "Per-agent debiased $\\ell_2$ ECE.",
    "pa_ece_plugin15": "Per-agent 15-bin equal-width plug-in ECE.",
    "pa_signed_gap": "Mean per-agent confidence minus accuracy; positive is overconfident.",
    "pa_brier": "Per-agent Brier score.",
    "pa_brier_rel": "Reliability component of the Murphy decomposition.",
    "pa_cox_intercept": "Intercept of the Cox recalibration logistic fit.",
    "pa_cox_slope": "Slope of the Cox recalibration logistic fit.",
    "vote_n": "Vote-fraction calibration items available.",
    "vote_ece_prim": "Vote-fraction ECE, exact-atom estimator.",
    "fp_n": "Final-producer calibration items available.",
    "fp_ece_prim": "Final-producer ECE, equal-mass 10-bin estimator.",
    "delta_vote_prim": "Pre-registered $\\Delta$ECE for the vote signal.",
    "delta_fp_prim": "Pre-registered $\\Delta$ECE for the final-producer signal.",
    "se_pa_ece": "Bootstrap standard error, per-agent ECE.",
    "se_vote_ece": "Bootstrap standard error, vote ECE.",
    "se_fp_ece": "Bootstrap standard error, final-producer ECE.",
    "se_delta_vote": "Paired bootstrap standard error of the vote delta.",
    "se_delta_fp": "Paired bootstrap standard error of the final-producer delta.",
    "Ec": "Coordination efficiency, $\\mathrm{success\\ rate} / \\mathrm{relative\\ turns}$.",
    "Ae": "Error amplification relative to the matched single-agent baseline.",
    "Opct": "Turn overhead over the matched single-agent baseline, in percent.",
    "log2_params": "$\\log_2(\\texttt{param\\_count})$.",
    "log_reas_tok": "$\\log(1 + \\texttt{mean\\_reasoning\\_tokens})$.",
    "reas_rank": "Ordinal rank of the reasoning rung, 0 to 4.",
    "reas_budget": "Nominal thinking-token budget; NaN for unlimited.",
    "ctx_rank": "Ordinal rank of the context-sharing level.",
    "design_cell": "Configuration key excluding seed.",
    "is_core": "Factor-level membership of the post-trim grid.",
    "n_agents": "Participating agents; 1 for single agent, 3 otherwise.",
    "era_post_trim": "Calendar flag: cell started after the trim commit epoch.",
    "ctx32b_capped": "32B cell served at 16384 context under a large reasoning budget.",
    "cost": "$\\texttt{param\\_count} \\times \\texttt{mean\\_total\\_tokens}$, an inference-FLOPs proxy.",
}


@table("tab_glossary")
def _():
    v = D.view()
    rows = []
    for c in v.columns:
        base = c
        gloss = GLOSS.get(c)
        if gloss is None:
            for pre, fam in [("pa_", "per-agent"), ("vote_", "vote fraction"), ("fp_", "final producer")]:
                if c.startswith(pre):
                    stat = c[len(pre):]
                    gloss = {
                        "ece_db": f"Debiased $\\ell_2$ ECE, {fam} signal.",
                        "ece_plugin15": f"15-bin equal-width plug-in ECE, {fam} signal.",
                        "signed_gap": f"Mean confidence minus accuracy, {fam} signal.",
                        "brier": f"Brier score, {fam} signal.",
                        "brier_rel": f"Murphy reliability component, {fam} signal.",
                        "cox_intercept": f"Cox recalibration intercept, {fam} signal.",
                        "cox_slope": f"Cox recalibration slope, {fam} signal.",
                        "n": f"Calibration items available, {fam} signal.",
                    }.get(stat)
                    if gloss:
                        break
        rows.append(
            {
                "col": base,
                "dtype": str(v[c].dtype),
                "nonnull": int(v[c].notna().sum()),
                "gloss": gloss or "",
            }
        )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Column": [f"\\texttt{{{T.esc(x)}}}" for x in df.col],
            "Type": [T.esc(x) for x in df.dtype],
            "Non-null": [T.intfmt(x) for x in df.nonnull],
            "Definition": df.gloss.tolist(),
        }
    )
    T.emit(
        body,
        OUT / "tab_glossary.tex",
        caption=(
            "Complete column glossary for \\texttt{analysis\\_view\\_v1.parquet}, the analysis "
            "population used throughout this report. Non-null counts are out of 3{,}201 rows; "
            "the calibration columns are null by construction on MATH."
        ),
        label="tab:glossary",
        column_format="llrp{0.44\\linewidth}",
        longtable=True,
        fontsize=r"\footnotesize",
        source="analysis/cache/analysis_view_v1.parquet",
    )


# =========================================================================== #
# P8  item difficulty
# =========================================================================== #
@table("tab_item_difficulty")
def _():
    it = D.items(["benchmark", "qid", "correct"])
    g = it.groupby(["benchmark", "qid"], observed=True).correct.mean().reset_index()
    rows = []
    for b in D.BENCHMARKS:
        sub = g[g.benchmark == b]
        if sub.empty:
            continue
        obs = it[it.benchmark == b]
        rows.append(
            {
                "benchmark": b,
                "n_items": sub.qid.nunique(),
                "n_obs": len(obs),
                "mean": sub.correct.mean(),
                "p10": sub.correct.quantile(0.10),
                "p50": sub.correct.median(),
                "p90": sub.correct.quantile(0.90),
                "never": (sub.correct == 0).mean(),
                "always": (sub.correct == 1).mean(),
            }
        )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Benchmark": [T.pretty(x) for x in df.benchmark],
            "Items": [T.intfmt(x) for x in df.n_items],
            "Observations": [T.intfmt(x) for x in df.n_obs],
            "Mean item accuracy": [T.num(x) for x in df["mean"]],
            "P10": [T.num(x) for x in df.p10],
            "Median": [T.num(x) for x in df.p50],
            "P90": [T.num(x) for x in df.p90],
            "\\% never solved": [T.num(100 * x, 1) for x in df.never],
            "\\% always solved": [T.num(100 * x, 1) for x in df.always],
        }
    )
    T.emit(
        body,
        OUT / "tab_item_difficulty.tex",
        caption=(
            "Item-difficulty spectrum. Item accuracy is computed over every system "
            "configuration that answered the question, so an item never solved by any of the "
            "roughly 3{,}200 configurations is a hard floor and an item always solved is a "
            "ceiling; both carry no discriminative information."
        ),
        label="tab:item-difficulty",
        column_format="lrrrrrrrr",
        source="analysis/cache/items_v1.parquet",
    )


# =========================================================================== #
# P9  round dynamics
# =========================================================================== #
@table("tab_round_accuracy")
def _():
    ag = D.agents(["topology", "model_size", "round", "correct"])
    g = (
        ag.groupby(["topology", "round"], observed=True)
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    g = D.sort_axis(g, "topology").sort_values(["topology", "round"], kind="stable")
    piv = g.pivot_table(index="topology", columns="round", values="acc", observed=True)
    cnt = g.pivot_table(index="topology", columns="round", values="n", observed=True)
    piv = piv.reindex([t for t in D.TOPO_ORDER if t in piv.index])
    cnt = cnt.reindex(piv.index)
    body = pd.DataFrame(
        {
            "Topology": [T.pretty(x) for x in piv.index],
            "Round 0 accuracy": [T.num(x) for x in piv.get(0, pd.Series(index=piv.index))],
            "Round 1 accuracy": [T.num(x) for x in piv.get(1, pd.Series(index=piv.index))],
            "$\\Delta$ round": [
                T.num(b - a, 4, signed=True) if pd.notna(a) and pd.notna(b) else "--"
                for a, b in zip(
                    piv.get(0, pd.Series(index=piv.index)),
                    piv.get(1, pd.Series(index=piv.index)),
                )
            ],
            "Rows round 0": [T.intfmt(x) for x in cnt.get(0, pd.Series(index=piv.index))],
            "Rows round 1": [T.intfmt(x) for x in cnt.get(1, pd.Series(index=piv.index))],
        }
    )
    T.emit(
        body,
        OUT / "tab_round_accuracy.tex",
        caption=(
            "Per-agent accuracy by topology and debate round, pooled over the whole sweep. "
            "Single-agent and independent cells run one round by construction; the two "
            "communicating topologies gain accuracy between rounds 0 and 1."
        ),
        label="tab:round-accuracy",
        column_format="lrrrrr",
        source="analysis/cache/agents_v1.parquet",
    )


@table("tab_round_accuracy_size")
def _():
    ag = D.agents(["topology", "model_size", "round", "correct"])
    ag = ag[ag.topology.isin(["decentralized", "centralized"])]
    g = (
        ag.groupby(["topology", "model_size", "round"], observed=True)
        .agg(acc=("correct", "mean"), n=("correct", "size"))
        .reset_index()
    )
    piv = g.pivot_table(index=["topology", "model_size"], columns="round",
                        values="acc", observed=True).reset_index()
    piv = D.sort_axis(piv, "topology", "model_size")
    body = pd.DataFrame(
        {
            "Topology": [T.pretty(x) for x in piv.topology],
            "Size": [T.pretty(x) for x in piv.model_size],
            "Round 0": [T.num(x) for x in piv[0]],
            "Round 1": [T.num(x) for x in piv[1]],
            "$\\Delta$ round": [T.num(b - a, 4, signed=True) for a, b in zip(piv[0], piv[1])],
        }
    )
    T.emit(
        body,
        OUT / "tab_round_accuracy_size.tex",
        caption=(
            "Round-over-round per-agent accuracy gain in the two communicating topologies, "
            "resolved by capacity. This slice is not available in the existing figure set."
        ),
        label="tab:round-accuracy-size",
        column_format="llrrr",
        longtable=True,
        source="analysis/cache/agents_v1.parquet",
    )


# =========================================================================== #
# P2  design-weighted marginals with bootstrap intervals, from the parquet
# =========================================================================== #
@table("tab_topology_means")
def _():
    v = D.view()
    rows = []
    for topo in D.TOPO_ORDER:
        sub = v[v.topology == topo]
        if sub.empty:
            continue
        rec = {"topology": topo, "n": len(sub), "designs": sub.design_cell.nunique()}
        for m in ["accuracy", "pa_ece_prim", "vote_ece_prim", "fp_ece_prim",
                  "delta_vote_prim", "delta_fp_prim", "Ec", "Ae", "Opct",
                  "mean_turns", "mean_messages", "cost"]:
            rec[m] = sub[m].mean()
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = {
        "Topology": [T.pretty(x) for x in df.topology],
        "Cells": [T.intfmt(x) for x in df.n],
        "Designs": [T.intfmt(x) for x in df.designs],
    }
    for m, lab, d in [
        ("accuracy", "Accuracy", 3),
        ("pa_ece_prim", "Per-agent ECE", 3),
        ("vote_ece_prim", "Vote ECE", 3),
        ("fp_ece_prim", "FP ECE", 3),
        ("delta_vote_prim", r"$\Delta$ECE vote", 3),
        ("delta_fp_prim", r"$\Delta$ECE fp", 3),
        ("Ec", r"$E_c$", 3),
        ("Ae", r"$A_e$", 3),
        ("Opct", r"$O\%$", 1),
        ("mean_turns", "Turns", 2),
        ("mean_messages", "Messages", 2),
        ("cost", "Cost proxy", 0),
    ]:
        cols[lab] = [T.num(x, d) for x in df[m]]
    T.emit(
        pd.DataFrame(cols),
        OUT / "tab_topology_means.tex",
        caption=(
            "Complete outcome profile of the four topologies over the post-trim core view. "
            "$E_c$, $A_e$ and $O\\%$ are defined against the matched single-agent baseline, "
            "so the single-agent row is 1 by construction for $A_e$ and 0 for $O\\%$."
        ),
        label="tab:topology-means",
        column_format="lrr" + "r" * 12,
        fontsize=r"\footnotesize",
        source="analysis/cache/analysis_view_v1.parquet",
    )


@table("tab_headline_axes")
def _():
    v = D.view()
    rows = []
    for axis, order in [
        ("model_size", D.SIZE_ORDER),
        ("reasoning_level", D.REAS_ORDER),
        ("prompt_complexity_level", [0, 3]),
        ("context_share_level", D.CTX_ORDER),
        ("n_agents", [1, 3]),
    ]:
        for lvl in order:
            sub = v[v[axis].astype(str) == str(lvl)]
            if sub.empty:
                continue
            rows.append(
                {
                    "axis": axis,
                    "level": lvl,
                    "n": len(sub),
                    "accuracy": sub.accuracy.mean(),
                    "pa": sub.pa_ece_prim.mean(),
                    "vote": sub.vote_ece_prim.mean(),
                    "dv": sub.delta_vote_prim.mean(),
                    "dfp": sub.delta_fp_prim.mean(),
                    "tok": sub.mean_total_tokens.mean(),
                    "cost": sub.cost.mean(),
                }
            )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Axis": [T.pretty(x) for x in df.axis],
            "Level": [T.pretty(x) for x in df.level],
            "Cells": [T.intfmt(x) for x in df.n],
            "Accuracy": [T.num(x) for x in df.accuracy],
            "PA ECE": [T.num(x) for x in df.pa],
            "Vote ECE": [T.num(x) for x in df.vote],
            "$\\Delta$vote": [T.num(x, 3, signed=True) for x in df.dv],
            "$\\Delta$fp": [T.num(x, 3, signed=True) for x in df.dfp],
            "Tokens": [T.intfmt(x) for x in df.tok],
            "Cost": [T.intfmt(x) for x in df.cost],
        }
    )
    T.emit(
        body,
        OUT / "tab_headline_axes.tex",
        caption=(
            "Pooled outcome means for every level of every design axis, over the post-trim "
            "core view and over all four benchmarks. These are unweighted cell means; the "
            "benchmark-resolved bootstrap version is Table~\\ref{tab:marginals-accuracy} and "
            "its companions."
        ),
        label="tab:headline-axes",
        column_format="llrrrrrrrr",
        longtable=True,
        fontsize=r"\footnotesize",
        source="analysis/cache/analysis_view_v1.parquet",
    )


@table("tab_estimator_comparison")
def _():
    v = D.view()
    mcq = v[v.benchmark.isin(D.MCQ)]
    rows = []
    for sig_name, pre in [("per-agent", "pa"), ("vote fraction", "vote"), ("final producer", "fp")]:
        rec = {"signal": sig_name}
        for stat, col in [
            ("Primary", f"{pre}_ece_prim"),
            ("Plug-in", f"{pre}_ece_plugin15"),
            ("Debiased", f"{pre}_ece_db"),
            ("Brier", f"{pre}_brier"),
            ("Reliab.", f"{pre}_brier_rel"),
            ("Gap", f"{pre}_signed_gap"),
            ("Cox int.", f"{pre}_cox_intercept"),
            ("Cox slope", f"{pre}_cox_slope"),
        ]:
            rec[stat] = mcq[col].mean() if col in mcq else np.nan
        rec["$n$ cells"] = int(mcq[f"{pre}_ece_prim"].notna().sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = {"Signal": [T.esc(x) for x in df.signal], "$n$ cells": [T.intfmt(x) for x in df["$n$ cells"]]}
    for c in df.columns:
        if c in ("signal", "$n$ cells"):
            continue
        cols[c] = [T.num(x, 3, signed=c in ("Gap", "Cox int.")) for x in df[c]]
    T.emit(
        pd.DataFrame(cols),
        OUT / "tab_estimator_comparison.tex",
        caption=(
            "Every calibration estimator computed in the pipeline, averaged over MCQ cells, "
            "for each of the three confidence signals. The primary column uses the "
            "pre-registered estimator for that signal: exact-atom for vote fraction because "
            "its support is $\\{1/3, 2/3, 1\\}$, equal-mass 10-bin otherwise. The debiased "
            "$\\ell_2$ statistic is undefined for a discrete signal and is reported as missing."
        ),
        label="tab:estimator-comparison",
        column_format="lr" + "r" * 8,
        fontsize=r"\footnotesize",
        source="analysis/cache/analysis_view_v1.parquet",
    )


@table("tab_bootstrap_ses")
def _():
    v = D.view()
    mcq = v[v.benchmark.isin(D.MCQ)]
    rows = []
    for col, lab in [
        ("se_pa_ece", "per-agent ECE"),
        ("se_vote_ece", "vote ECE"),
        ("se_fp_ece", "final-producer ECE"),
        ("se_delta_vote", r"$\Delta$ECE vote"),
        ("se_delta_fp", r"$\Delta$ECE final producer"),
    ]:
        s = mcq[col].dropna()
        rows.append(
            {
                "stat": lab,
                "n": len(s),
                "median": s.median(),
                "mean": s.mean(),
                "p90": s.quantile(0.9),
                "max": s.max(),
            }
        )
    df = pd.DataFrame(rows)
    body = pd.DataFrame(
        {
            "Statistic": df.stat.tolist(),
            "$n$ cells": [T.intfmt(x) for x in df.n],
            "Median SE": [T.num(x, 4) for x in df["median"]],
            "Mean SE": [T.num(x, 4) for x in df["mean"]],
            "P90 SE": [T.num(x, 4) for x in df.p90],
            "Max SE": [T.num(x, 4) for x in df["max"]],
        }
    )
    T.emit(
        body,
        OUT / "tab_bootstrap_ses.tex",
        caption=(
            "Distribution of the per-cell bootstrap standard errors across MCQ cells. The "
            "delta standard errors use paired replicates, so they are smaller than the naive "
            "sum of the two component errors."
        ),
        label="tab:bootstrap-ses",
        column_format="lrrrrr",
        source="analysis/cache/analysis_view_v1.parquet",
    )


# =========================================================================== #
# Methods reference tables
# =========================================================================== #
@table("tab_grid")
def _():
    rows = [
        ("model size", "6", "0.6B, 1.7B, 4B, 8B, 14B, 32B", "Qwen3 dense family; capacity axis"),
        ("topology", "4", "single agent, independent, decentralized, centralized",
         "single agent collapses to 1 agent and 1 round"),
        ("context sharing", "2", "artifact only, plus CoT",
         "plus intermediate removed by the 2026-06-10 trim"),
        ("prompt complexity", "2", "L0, L3",
         "L1 and L2 removed by the trim; L1 survives only as the single-agent baseline prompt"),
        ("reasoning budget", "5", "off, b512, b2048, b8192, unlimited",
         "vLLM \\texttt{thinking\\_token\\_budget}; unlimited allows 8192 output tokens for thinking"),
        ("benchmark", "4", "GPQA Diamond, MMLU-Pro, MATH-500, TruthfulQA MC1",
         "198 questions on GPQA, 200 elsewhere"),
        ("seed", "3", "0, 1, 2", "also drives the deterministic per-question option shuffle"),
    ]
    body = pd.DataFrame(
        {
            "Axis": [r[0] for r in rows],
            "Levels": [r[1] for r in rows],
            "Values": [T.esc(r[2]) for r in rows],
            "Notes": [r[3] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_grid.tex",
        caption=(
            "The post-trim sweep axes. Fixed across every cell: 3 agents in multi-agent "
            "conditions, 2 debate rounds, 5 self-consistency samples for single-agent cells, "
            "sampling temperature 0.7 when thinking is disabled, and 200 requested questions."
        ),
        label="tab:grid",
        column_format="lrp{0.34\\linewidth}p{0.34\\linewidth}",
        source="configs/full_sweep.yaml",
    )


@table("tab_models")
def _():
    rows = [
        ("0.6B", "Qwen/Qwen3-0.6B", "0.6", "1", "32768", "standard"),
        ("1.7B", "Qwen/Qwen3-1.7B", "1.7", "1", "32768", "standard"),
        ("4B", "Qwen/Qwen3-4B", "4.0", "1", "32768", "standard"),
        ("8B", "Qwen/Qwen3-8B", "8.2", "1", "32768", "standard"),
        ("14B", "Qwen/Qwen3-14B", "14.8", "1", "32768", "standard"),
        ("32B", "Qwen/Qwen3-32B", "32.8", "1", "16384", "capped"),
    ]
    body = pd.DataFrame(
        {
            "Size": [r[0] for r in rows],
            "Hugging Face identifier": [f"\\texttt{{{T.esc(r[1])}}}" for r in rows],
            "Params (B)": [r[2] for r in rows],
            "TP": [r[3] for r in rows],
            "\\texttt{max\\_model\\_len}": [T.intfmt(r[4]) for r in rows],
            "Context status": [r[5] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_models.tex",
        caption=(
            "Served models. All six are dense Qwen3 checkpoints pinned to explicit commit "
            "revisions in \\texttt{configs/model\\_contracts.v1.json} and served in bfloat16 on "
            "A100-80GB. The 32B context reduction is a memory constraint: 32B weights leave "
            "insufficient KV-cache room for a 32768-token context on one A100."
        ),
        label="tab:models",
        column_format="llrrrl",
        source="src/agents_scaling/models.py",
    )


@table("tab_benchmarks")
def _():
    rows = [
        ("GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train", "198", "198", "MCQ, 4 options"),
        ("MMLU-Pro", "TIGER-Lab/MMLU-Pro", None, "test", "12032", "200", "MCQ, up to 10 options"),
        ("MATH", "HuggingFaceH4/MATH-500", None, "test", "500", "200", "free-form numeric"),
        ("TruthfulQA", "truthfulqa/truthful_qa", "multiple_choice", "validation", "817", "200", "MC1, single correct"),
    ]
    body = pd.DataFrame(
        {
            "Benchmark": [r[0] for r in rows],
            "Repository": [f"\\texttt{{{T.esc(r[1])}}}" for r in rows],
            "Config": ["--" if r[2] is None else f"\\texttt{{{T.esc(r[2])}}}" for r in rows],
            "Split": [r[3] for r in rows],
            "Split size": [T.intfmt(r[4]) for r in rows],
            "Used per cell": [r[5] for r in rows],
            "Answer type": [r[6] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_benchmarks.tex",
        caption=(
            "Evaluation suites. Each is pinned to a 40-character dataset commit revision and "
            "loaded cache-only with the revision verified against the on-disk Arrow path. "
            "The per-cell question count is $\\min(200, |\\mathrm{split}|)$, which is why GPQA "
            "contributes 198 rather than 200 questions."
        ),
        label="tab:benchmarks",
        column_format="lllrrrl",
        source="src/agents_scaling/benchmarks/loaders.py",
    )


@table("tab_sampling")
def _():
    rows = [
        ("chat, thinking disabled", "0.7", "0.8", "20", "envelope", "deterministic per agent and round"),
        ("chat, thinking enabled", "0.6", "0.95", "20", "envelope", "deterministic per agent and round"),
        ("forced-option probe", "0.0", "--", "--", "1", "not applicable"),
    ]
    body = pd.DataFrame(
        {
            "Request": [r[0] for r in rows],
            "Temperature": [r[1] for r in rows],
            "top-$p$": [r[2] for r in rows],
            "top-$k$": [r[3] for r in rows],
            "\\texttt{max\\_tokens}": [r[4] for r in rows],
            "Seed": [r[5] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_sampling.tex",
        caption=(
            "Sampling configuration. \\emph{Envelope} means the request asks for the entire "
            "remaining context: $\\texttt{max\\_model\\_len} - |\\mathrm{prompt}| - 128$. The "
            "forced-option probe is a separate deterministic single-token completion request "
            "used for confidence measurement, issued before any stochastic generation."
        ),
        label="tab:sampling",
        column_format="lrrrll",
        source="src/agents_scaling/serving/client.py",
    )


@table("tab_reasoning_protocol")
def _():
    rows = [
        ("off", "--", "4096", "thinking disabled in the chat template"),
        ("b512", "512", "4608", "budget enforced server-side by vLLM"),
        ("b2048", "2048", "6144", "budget enforced server-side by vLLM"),
        ("b8192", "8192", "12288", "budget enforced server-side by vLLM"),
        ("unlimited", "none", "12288", "no budget field; 8192 tokens of thinking allowance"),
    ]
    body = pd.DataFrame(
        {
            "Rung": [r[0] for r in rows],
            "\\texttt{thinking\\_token\\_budget}": [r[1] for r in rows],
            "Output floor (tokens)": [T.intfmt(r[2]) for r in rows],
            "Mechanism": [r[3] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_reasoning_protocol.tex",
        caption=(
            "Reasoning rungs under generation protocol version 4. The output floor is the "
            "minimum generation capacity a request must be able to obtain: 4096 tokens for the "
            "answer plus the thinking allowance. A cell is routed to a long serving profile "
            "when its exact rendered prompt cannot leave that floor within the served context."
        ),
        label="tab:reasoning-protocol",
        column_format="lrrp{0.42\\linewidth}",
        source="src/agents_scaling/serving/client.py",
    )


@table("tab_estimators")
def _():
    rows = [
        ("Equal-mass 10-bin $\\ell_1$ ECE",
         "$\\sum_g (W_g/W)\\,|\\bar{c}_g - \\bar{y}_g|$ with quantile bin edges",
         "primary for continuous signals"),
        ("Exact-atom ECE",
         "same weighted form with one group per distinct confidence value",
         "primary for vote fraction, whose support is $\\{1/3, 2/3, 1\\}$"),
        ("15-bin equal-width plug-in",
         "fixed edges on $[0,1]$",
         "reported for comparability with the prior literature only"),
        ("Debiased $\\ell_2$ ECE",
         "$\\sqrt{\\max(0, \\sum_g w_g[(\\bar{c}_g-\\bar{y}_g)^2 - \\bar{y}_g(1-\\bar{y}_g)/(n_g-1)])}$",
         "robustness; undefined for discrete signals"),
        ("Brier score",
         "$\\frac{1}{n}\\sum_i (c_i - y_i)^2$",
         "proper scoring rule"),
        ("Murphy reliability",
         "$\\sum_g w_g (\\bar{c}_g - \\bar{y}_g)^2$",
         "calibration component of the Brier decomposition"),
        ("Signed gap",
         "$\\bar{c} - \\bar{y}$",
         "positive indicates overconfidence"),
        ("Cox recalibration",
         "logistic regression of $y$ on $\\operatorname{logit}(c)$",
         "intercept and slope diagnose bias and sharpness"),
        ("AUROC",
         "rank discrimination of $c$ against $y$",
         "separates discrimination from calibration"),
        ("Temperature scaling",
         "scalar $T$ minimising negative log-likelihood",
         "post-hoc recalibration ceiling"),
    ]
    body = pd.DataFrame(
        {
            "Estimator": [r[0] for r in rows],
            "Definition": [r[1] for r in rows],
            "Role": [r[2] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_estimators.tex",
        caption=(
            "Calibration estimators. $c$ denotes a confidence, $y$ the binary correctness "
            "indicator, $g$ a bin or atom, $n_g$ its count and $w_g = n_g/n$ its weight. The "
            "estimator applied to a signal is fixed in advance by the signal's support, not "
            "chosen after inspecting the result."
        ),
        label="tab:estimators",
        column_format="p{0.24\\linewidth}p{0.42\\linewidth}p{0.28\\linewidth}",
        source="analysis/nb_lib/calib.py",
    )


@table("tab_signals")
def _():
    rows = [
        ("per-agent", "\\texttt{option\\_logprobs[chosen]}",
         "every agent-round in the cell", "equal-mass 10-bin",
         "the $\\Delta$ECE baseline"),
        ("vote fraction", "share of agents whose answer equals the system answer",
         "one value per question", "exact atom",
         "degenerate for single agent, where it is always 1"),
        ("final producer", "\\texttt{option\\_logprobs} of the agent that produced the final answer",
         "one value per question", "equal-mass 10-bin",
         "the orchestrator in centralized cells, the decisive winning-side agent otherwise"),
        ("mean agreeing logprob", "mean logprob confidence over the winning side",
         "one value per question", "equal-mass 10-bin",
         "tournament entrant only"),
        ("verbalized", "the percentage the model states in its answer",
         "per agent", "equal-mass 10-bin",
         "elicited by the shared answer-format instruction"),
        ("self-consistency", "modal-answer share over 5 samples",
         "single-agent cells only", "equal-mass 10-bin",
         "the only confidence family available on MATH"),
    ]
    body = pd.DataFrame(
        {
            "Signal": [r[0] for r in rows],
            "Definition": [r[1] for r in rows],
            "Grain": [r[2] for r in rows],
            "Estimator": [r[3] for r in rows],
            "Notes": [r[4] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_signals.tex",
        caption=(
            "Confidence signals measured in the sweep. The headline estimand contrasts the "
            "first signal against the second and third: $\\Delta\\mathrm{ECE} = "
            "\\mathrm{ECE}(\\text{system signal}) - \\mathrm{ECE}(\\text{per agent})$."
        ),
        label="tab:signals",
        column_format="p{0.13\\linewidth}p{0.26\\linewidth}p{0.15\\linewidth}p{0.13\\linewidth}p{0.24\\linewidth}",
        source="src/agents_scaling/agents/aggregate.py",
    )


@table("tab_topology_protocol")
def _():
    rows = [
        ("single agent", "1", "1", "1", "0", "the agent's own answer",
         "collapses context sharing to artifact only; runs 5 self-consistency samples"),
        ("independent", "3", "1", "3", "0", "majority vote, ties broken by summed option probability",
         "agents never see one another"),
        ("decentralized", "3", "2", "6", "6", "majority vote over the final round",
         "all-to-all: each agent sees every peer's previous-round output"),
        ("centralized", "3", "2", "6", "8", "the orchestrator's final synthesis",
         "1 orchestrator and 2 sub-agents; sub-agents see only the orchestrator's synthesis"),
    ]
    body = pd.DataFrame(
        {
            "Topology": [r[0] for r in rows],
            "Agents": [r[1] for r in rows],
            "Rounds": [r[2] for r in rows],
            "Turns": [r[3] for r in rows],
            "Messages": [r[4] for r in rows],
            "Final answer": [r[5] for r in rows],
            "Visibility": [r[6] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_topology_protocol.tex",
        caption=(
            "Coordination protocols at the sweep's fixed setting of 3 agents and 2 rounds. "
            "Turns count generation calls; messages count inter-agent transmissions, which are "
            "accrued only from round 1 onward because round 0 has no peer context."
        ),
        label="tab:topology-protocol",
        column_format="lrrrrp{0.20\\linewidth}p{0.26\\linewidth}",
        fontsize=r"\footnotesize",
        source="src/agents_scaling/agents/topologies/",
    )


# =========================================================================== #
# Provenance
# =========================================================================== #
@table("tab_provenance")
def _():
    import hashlib

    inputs = [
        ("analysis/cache/analysis_view_v1.parquet", "analysis population, 3201 x 77"),
        ("analysis/cache/cells_dedup_v1.parquet", "canonical per-cell table, 4301 x 66"),
        ("analysis/cache/items_v1.parquet", "per-question rows"),
        ("analysis/cache/agents_v1.parquet", "per-agent-round rows"),
        ("analysis/cache/ingest_manifest_v1.json", "ingest counters"),
        ("analysis/tables/axis_relationships/axis_summary_long.csv", "marginal means"),
        ("analysis/tables/axis_relationships/axis_matched_contrasts.csv", "matched contrasts"),
        ("analysis/tables/axis_relationships/axis_stratified_contrasts.csv", "stratified contrasts"),
        ("analysis/tables/scaling_equivalence/scaling_law_coefficients.csv", "scaling fits"),
        ("analysis/tables/scaling_equivalence/reasoning_rung_equivalents.csv", "rung equivalents"),
        ("analysis/tables/scaling_equivalence/adjacent_plateau_contrasts.csv", "plateau contrasts"),
        ("analysis/tables/scaling_equivalence/orchestration_matched_contrasts.csv", "topology contrasts"),
        ("analysis/tables/scaling_equivalence/size_reasoning_grid.csv", "capacity by reasoning grid"),
        ("analysis/tables/scaling_equivalence/nearest_equivalence_pairs.csv", "nearest equivalences"),
        ("configs/full_sweep.yaml", "sweep grid definition"),
        ("configs/model_contracts.v1.json", "pinned model revisions"),
    ]
    rows = []
    for rel, note in inputs:
        p = D.REPO / rel
        if not p.exists():
            rows.append((rel, note, "--", "missing", "--"))
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        import datetime as _dt

        mt = _dt.datetime.utcfromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
        rows.append((rel, note, f"{p.stat().st_size / 1e6:.2f}", h.hexdigest()[:16], mt))
    body = pd.DataFrame(
        {
            # Long paths are single unbreakable tokens in \texttt; allow a break after
            # every separator so the column wraps instead of running into its neighbour.
            "Input": [
                "\\texttt{"
                + T.esc(r[0]).replace("/", "/\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}")
                + "}"
                for r in rows
            ],
            "Contents": [r[1] for r in rows],
            "MB": [r[2] for r in rows],
            "SHA-256 (first 16)": [f"\\texttt{{\\scriptsize {r[3]}}}" for r in rows],
            "Modified": [r[4] for r in rows],
        }
    )
    T.emit(
        body,
        OUT / "tab_provenance.tex",
        caption=(
            "Every input consumed by this document, with a truncated content hash. The build "
            "is read-only over these files; it does not regenerate the caches."
        ),
        label="tab:provenance",
        column_format=">{\\raggedright\\arraybackslash}p{0.36\\linewidth}>{\\raggedright\\arraybackslash}p{0.24\\linewidth}rll",
        longtable=True,
        fontsize=r"\scriptsize",
        source="paper build inputs",
    )


# =========================================================================== #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables")
    ap.add_argument("--only", default="*")
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    names = [n for n in BUILDERS if _fn(n, args.only)]
    if not names:
        print(f"no builders match {args.only!r}", file=sys.stderr)
        return 1
    for n in sorted(names):
        BUILDERS[n]()
    (OUT / "_manifest.json").write_text(json.dumps(T.MANIFEST, indent=2, sort_keys=True))

    problems = T.lint(OUT)
    if problems:
        print("LaTeX lint failures:", file=sys.stderr)
        for p in problems[:40]:
            print("  " + p, file=sys.stderr)
        return 1
    total = sum(v["rows"] for v in T.MANIFEST.values())
    print(f"wrote {len(names)} tables ({total:,} data rows) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
