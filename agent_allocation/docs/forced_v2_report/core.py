"""Loading, accounting and the paired bootstrap frame for the forced_budget_v2 report.

Every generation in this study spent exactly its reasoning budget (A = 1024 per call, m * A for deep/m), so the
compute budget of a checkpoint, C = B * R * A, is both requested and realized. Nothing here recomputes a sealed
number: accuracies, costs and effective depth come from the merged run tables, and every interval is a paired
item-clustered bootstrap from the run's own ``agent_allocation.analysis.stats``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from agent_allocation.analysis import stats as astats

BENCH_ORDER = ["aime", "hmmt", "omni_math_hard", "gpqa_diamond"]
MATH = ["aime", "hmmt", "omni_math_hard"]
BENCH_LABEL = {
    "aime": "AIME 25+26",
    "hmmt": "HMMT Feb 25+26",
    "omni_math_hard": "Omni-MATH $\\geq$7",
    "gpqa_diamond": "GPQA-Diamond",
    "macro": "Macro (4)",
}
BENCH_PLAIN = {"aime": "AIME 2025+2026", "hmmt": "HMMT Feb 2025+2026", "omni_math_hard": "Omni-MATH (difficulty >= 7)",
               "gpqa_diamond": "GPQA-Diamond", "macro": "Macro, 4 benchmarks"}
FAMILY_LABEL = {
    "sample": "Sample + vote",
    "private_single": "Private single",
    "private_team": "Private team",
    "peer": "Peer",
    "deep": "Deep single call",
    "cp": "Communicate-then-private",
    "pooled": "Pooled synthesis",
}
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
LADDER_FAMILIES = ["sample", "private_single", "private_team", "peer", "deep"]
A0 = 1024
REPLICATES = 10_000
SEED = 20260918
CHUNK = 500


def T(run_dir: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(run_dir, "tables", name + ".parquet"))


def outcomes(run_dir: str, *, families=None) -> pd.DataFrame:
    po = T(run_dir, "policy_outcomes")
    po = po[(po["partition_id"] == "main") & (po["view"] == "current")].copy()
    if families is not None:
        po = po[po["family"].isin(families)]
    po["B"] = po["team_size"].astype(int)
    po["R"] = po["total_cycles"].astype(int)
    po["A"] = po["reasoning_budget"].astype(int)
    po["C"] = po["B"] * po["R"] * po["A"]
    po["rung"] = po["C"] // A0
    return po


def checkpoint_index(oc: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "family", "B", "R", "A", "C", "rung"]
    return oc.drop_duplicates("name")[cols].sort_values(["rung", "family", "B", "name"]).reset_index(drop=True)


# ---------------------------------------------------------------- bootstrap helpers

def boot_means(mat: dict, rs) -> dict:
    out = {}
    for s, Y in mat.items():
        if Y.shape[0] == 0:
            continue
        acc = np.empty((rs.n_resamples, Y.shape[1]))
        for start in range(0, rs.n_resamples, CHUNK):
            sl = slice(start, min(rs.n_resamples, start + CHUNK))
            acc[sl] = np.nanmean(Y[rs.idx[s][sl]], axis=1)
        out[s] = acc
    return out


def levels_of(strata) -> dict:
    by = {}
    for s in strata:
        by.setdefault(s.split("/")[0], []).append(s)
    bench = [b for b in BENCH_ORDER if b in by]
    lv = {"macro": {s: 1.0 / (len(bench) * len(by[b])) for b in bench for s in by[b]}}
    for b in bench:
        lv[b] = {s: 1.0 / len(by[b]) for s in by[b]}
    return lv


def wpoint(mat, w, j):
    return float(sum(wt * np.nanmean(mat[s][:, j]) for s, wt in w.items() if s in mat and mat[s].shape[0]))


def wboot(boot, w, j):
    acc = None
    for s, wt in w.items():
        if s in boot:
            v = wt * boot[s][:, j]
            acc = v if acc is None else acc + v
    return acc


def bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    rej = np.zeros(p.shape, bool)
    ok = np.where(~np.isnan(p))[0]
    if not len(ok):
        return rej
    order = ok[np.argsort(p[ok])]
    m = len(order)
    below = p[order] <= alpha * np.arange(1, m + 1) / m
    if below.any():
        rej[order[: np.max(np.where(below)[0]) + 1]] = True
    return rej


class Frame:
    """One population over every checkpoint of the given families; all columns share resample indices."""

    def __init__(self, run_dir: str, families=None, *, label="v2"):
        self.oc = outcomes(run_dir, families=families)
        self.idx = checkpoint_index(self.oc)
        self.names = list(self.idx["name"])
        cols = [("main", n) for n in self.names]
        self.pop = astats.build_population(self.oc, cols, view="current", label=label)
        self.mat = astats.value_matrix(self.oc, self.pop, cols, view="current", column="y")
        self.rs = self.pop.resampler(REPLICATES, SEED)
        self.boot = boot_means(self.mat, self.rs)
        self.levels = levels_of(self.pop.strata)
        self.col = {n: j for j, n in enumerate(self.names)}
        self.n_items = self.pop.n_items

    def acc(self, name, level="macro", alpha=0.05):
        j, w = self.col[name], self.levels[level]
        b = wboot(self.boot, w, j)
        lo, hi = np.nanquantile(b, [alpha / 2, 1 - alpha / 2])
        return wpoint(self.mat, w, j), float(lo), float(hi)

    def diff(self, a, b, level="macro", alpha=0.05):
        w = self.levels[level]
        ja, jb = self.col[a], self.col[b]
        est = wpoint(self.mat, w, ja) - wpoint(self.mat, w, jb)
        d = wboot(self.boot, w, ja) - wboot(self.boot, w, jb)
        lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
        p = float(np.mean(np.abs(d - np.nanmean(d)) >= abs(est)))
        return est, float(lo), float(hi), p

    def unknown_share(self, name):
        sub = self.oc[self.oc["name"] == name]
        return float(sub["y"].isna().mean())


def bank_curve(run_dir: str) -> pd.DataFrame:
    """Oracle pass@k, realized prefix coverage and the vote readout per k, per level (macro = equal benchmarks)."""
    sb = T(run_dir, "sample_bank_items")
    sb = sb[sb["partition_id"] == "main"]
    lv = levels_of(sorted(set(sb["stratum"])))
    out = []
    for level, w in lv.items():
        for k, g in sb.groupby("k"):
            row = dict(level=level, k=int(k))
            for col in ("oracle_pass_at_k", "prefix_coverage", "readout", "mean_draw_correct"):
                row[col] = float(sum(wt * g[g["stratum"] == s][col].mean() for s, wt in w.items()))
            out.append(row)
    return pd.DataFrame(out)


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=conv)
