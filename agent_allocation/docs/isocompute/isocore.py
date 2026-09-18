"""Iso-compute re-keying of the sealed openrouter_agent_allocation_v1 tables.

Every checkpoint is a triple (B agents, R rounds, A per-agent-per-round reasoning allowance).
The compute budget is C = B * R * A, uniformly across every protocol family: the sealed
``policy_outcomes`` table carries ``team_size`` (B), ``total_cycles`` (R) and ``reasoning_budget``
(A), and for the ``deep`` family A is already the scaled allowance. Nothing here recomputes a
sealed number; the run's own analysis library supplies the estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_allocation.analysis import stats as astats

# ---------------------------------------------------------------- constants

BENCH_ORDER = ["math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"]
BENCH_LABEL = {
    "aime": "AIME 2025+2026",
    "hmmt": "HMMT Feb 2025+2026",
    "omni_math_hard": "Omni-MATH (difficulty >= 7)",
    "gpqa_diamond": "GPQA-Diamond",
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "bbh_logic4": "BBH-logic4",
    "mbpp_plus": "MBPP+",
    "macro4": "Macro (4 benchmarks)",
    "macro3": "Macro (3 informative)",
}
# BBH is saturated at 1.000 in every configuration and carries no signal.
INFORMATIVE = ["math500", "mmlu_pro", "mbpp_plus"]
SATURATED = ["bbh_logic4"]

FAMILY_LABEL = {
    "sample": "Sample + vote",
    "private_single": "Private single",
    "private_team": "Private team",
    "peer": "Peer",
    "deep": "Deep single call",
    "cp": "Communicate-then-private",
    "pooled": "Pooled synthesis",
}
# Colour-blind-safe series palette (Okabe-Ito derived), stable across every figure.
FAMILY_COLOR = {
    "sample": "#666666",
    "private_single": "#0072B2",
    "private_team": "#009E73",
    "peer": "#D55E00",
    "deep": "#CC79A7",
    "cp": "#E69F00",
    "pooled": "#56B4E9",
}
ORACLE_COLOR = "#000000"

REPLICATES = 10_000
STUDY_SEED = 20260917
A0 = {"main": 1024, "high": 4096, "pilot": 1024, "replication": 1024}
# Families that live on the 400-item population_0 ladder (cp/pooled use a separate 192-item panel).
LADDER_FAMILIES = ["sample", "private_single", "private_team", "peer", "deep"]
# forced_budget_v2: deep multipliers {2,4,8,16,25}, so the deep curve reaches the 25-generation rung
# (sample/25, private_single/25) and every rung C/A0 in LADDER_RUNGS has a depth-only point where one exists.
LADDER_RUNGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25]
DEEP_RUNGS_FORCED = [1, 2, 4, 8, 16, 25]


def tables_dir(run_dir: str) -> str:
    return str(run_dir).rstrip("/") + "/tables/"


def load(run_dir: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(tables_dir(run_dir) + name + ".parquet")


# ---------------------------------------------------------------- accounting

def add_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Attach B, R, A, C and the rung C/A0 to a frame carrying the sealed design columns."""
    out = df.copy()
    out["B"] = out["team_size"].astype(int)
    out["R"] = out["total_cycles"].astype(int)
    out["A"] = out["reasoning_budget"].astype(int)
    out["C"] = out["B"] * out["R"] * out["A"]
    out["rung"] = out.apply(lambda r: r["C"] // A0[r["partition_id"]], axis=1)
    return out


def outcomes(run_dir: str, partition: str = "main", *, ladder_only: bool = True) -> pd.DataFrame:
    """Current-view per-item delivered outcomes for one partition, with the budget columns."""
    po = load(run_dir, "policy_outcomes")
    po = po[(po["partition_id"] == partition) & (po["view"] == "current")]
    if ladder_only:
        po = po[po["family"].isin(LADDER_FAMILIES)]
    return add_budget(po)


def checkpoint_index(oc: pd.DataFrame) -> pd.DataFrame:
    """One row per checkpoint: name, family, B, R, A, C, rung."""
    cols = ["name", "family", "B", "R", "A", "C", "rung", "partition_id"]
    idx = oc.drop_duplicates("name")[cols].reset_index(drop=True)
    return idx.sort_values(["rung", "family", "B"]).reset_index(drop=True)


def has_table(run_dir: str, name: str) -> bool:
    import os
    return os.path.exists(tables_dir(run_dir) + name + ".parquet")


# ---------------------------------------------------------------- effective depth (forced runs)

def effective_depth(run_dir: str, partition: str = "main") -> pd.DataFrame:
    """Per checkpoint: requested C, mean effective tokens (reasoning before loop onset, summed over the checkpoint's
    own generations; ``analysis.effective_depth``), the effective fraction and the effective rung C_eff/A0.

    Loop onset per generation = first 3,000-char window (step 500) of the reasoning whose zlib ratio < 0.1, as a
    fraction of the reasoning length x reasoning_budget; a generation without a loop counts its whole realized
    reasoning. Requires the ``effective_depth_summary`` table that ``allocation analyze`` writes for forced runs."""
    eff = load(run_dir, "effective_depth_summary")
    eff = eff[eff["partition_id"] == partition]
    idx = checkpoint_index(outcomes(run_dir, partition, ladder_only=False))
    out = idx.merge(eff[["name", "items", "items_known", "mean_effective_tokens", "mean_effective_fraction",
                         "mean_looped_generations"]], on="name", how="left")
    out["effective_rung"] = out["mean_effective_tokens"] / A0[partition]
    return out.sort_values(["rung", "family", "B", "name"]).reset_index(drop=True)


def realized_compute(run_dir: str, partition: str = "main") -> pd.DataFrame:
    """Per checkpoint: requested C next to the realized tokens of its own ancestor set (policy ledger, macro rows).
    In a forced run realized reasoning equals requested, so ``realized_reasoning / C`` must be 1 everywhere."""
    cs = load(run_dir, "cost_summary")
    cs = cs[(cs["partition_id"] == partition) & (cs["benchmark"] == "macro")]
    idx = checkpoint_index(outcomes(run_dir, partition, ladder_only=False))
    cols = [c for c in ("mean_requested_reasoning_tokens", "mean_reasoning_tokens", "mean_realized_tokens",
                        "mean_total_charged_usd", "acc") if c in cs.columns]
    out = idx.merge(cs[["name", *cols]], on="name", how="left")
    if "mean_reasoning_tokens" in out:
        out["realized_over_requested"] = out["mean_reasoning_tokens"] / out["C"]
    return out.sort_values(["rung", "family", "B", "name"]).reset_index(drop=True)


# ---------------------------------------------------------------- self-check

def self_check(run_dir: str, partition: str = "main") -> list[str]:
    """Assert the derived accounting against the sealed cost table. Returns failure messages."""
    fails: list[str] = []
    oc = outcomes(run_dir, partition)
    pc = load(run_dir, "policy_costs")
    pc = pc[(pc["partition_id"] == partition) & (pc["view"] == "current")]

    # C must equal the sealed requested reasoning tokens of the checkpoint's own ancestor set.
    req = pc.groupby("name")["requested_reasoning_tokens"].agg(["min", "max"]).reset_index()
    idx = checkpoint_index(oc).merge(req, on="name", how="inner")
    bad = idx[(idx["C"] != idx["min"]) | (idx["C"] != idx["max"])]
    if len(bad):
        fails.append(
            "derived C != sealed requested_reasoning_tokens for "
            + ", ".join(f"{r['name']} (C={r['C']} sealed={r['min']}..{r['max']})"
                        for _, r in bad.head(8).iterrows())
        )

    # G = B*R must equal nominal_generations, except deep (one call carrying m*A of reasoning).
    gen = pc.groupby("name")["nominal_generations"].max().reset_index()
    idx2 = checkpoint_index(oc).merge(gen, on="name", how="inner")
    idx2["G"] = idx2["B"] * idx2["R"]
    nd = idx2[idx2["family"] != "deep"]
    bad2 = nd[nd["G"] != nd["nominal_generations"]]
    if len(bad2):
        fails.append("derived G != nominal_generations for " + ", ".join(bad2["name"].head(8)))
    dp = idx2[idx2["family"] == "deep"]
    if len(dp) and not (dp["nominal_generations"] == 1).all():
        fails.append("deep family nominal_generations is not 1")
    return fails


# ---------------------------------------------------------------- estimation

def _columns(partition: str, names) -> list[tuple[str, str]]:
    return [(partition, n) for n in names]


def accuracy_table(oc_all: pd.DataFrame, partition: str, names, *, items=None,
                   label: str = "iso") -> pd.DataFrame:
    """Delivered accuracy with paired item-clustered bootstrap intervals, macro and per benchmark.

    All columns share resample indices, so column differences taken from the same call are paired.
    """
    cols = _columns(partition, list(names))
    pop = astats.build_population(oc_all, cols, view="current", label=label, items=items)
    mat = astats.value_matrix(oc_all, pop, cols, view="current", column="y")
    rs = pop.resampler(REPLICATES, STUDY_SEED)
    tab = astats.column_intervals(mat, rs, list(names))
    return tab, pop, mat, rs


def macro3_weights(strata) -> dict:
    """Equal weight on the three informative benchmarks, BBH subtasks excluded entirely."""
    keep = [s for s in strata if s.split("/")[0] in INFORMATIVE]
    per_bench: dict[str, list[str]] = {}
    for s in keep:
        per_bench.setdefault(s.split("/")[0], []).append(s)
    w = {}
    for b, ss in per_bench.items():
        for s in ss:
            w[s] = 1.0 / (len(per_bench) * len(ss))
    return w


def weighted_point(mat: dict, weights: dict, j: int) -> float:
    est = 0.0
    for s, wt in weights.items():
        if s in mat and mat[s].shape[0]:
            est += wt * float(np.nanmean(mat[s][:, j]))
    return est


def paired_diff(mat: dict, rs, weights: dict, j: int, i: int, *, alpha: float = 0.05):
    """Weighted mean of (column j minus column i), with a paired percentile interval."""
    est = weighted_point(mat, weights, j) - weighted_point(mat, weights, i)
    acc = np.zeros(rs.n_resamples)
    for s, wt in weights.items():
        if s not in mat or mat[s].shape[0] == 0:
            continue
        d = mat[s][:, j] - mat[s][:, i]
        acc += wt * np.nanmean(d[rs.idx[s]], axis=1)
    lo, hi = np.nanquantile(acc, [alpha / 2, 1 - alpha / 2])
    # Two-sided bootstrap p-value for H0: difference = 0.
    centred = acc - np.nanmean(acc)
    p = float(np.mean(np.abs(centred) >= abs(est))) if rs.n_resamples else np.nan
    return est, float(lo), float(hi), p


def bh_fdr(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg. Returns a boolean array of rejections, NaN p-values never rejected."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    rej = np.zeros(p.shape, dtype=bool)
    if not ok.any():
        return rej
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    if below.any():
        kmax = np.max(np.where(below)[0])
        rej[order[: kmax + 1]] = True
    return rej
