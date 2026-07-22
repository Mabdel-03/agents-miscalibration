"""Hierarchical bootstrap — the single uncertainty engine for pooled/paired claims.

Design facts this respects (analysis plan §3):
  * every config within a benchmark answers the SAME ~200 questions (verified), so
    cell-level statistics are cross-correlated through question difficulty — the qid
    resample is drawn ONCE per replicate and applied jointly to every config;
  * 1-3 seeds per design cell is too few for seed random effects — seeds are resampled
    with replacement within design cell as the second bootstrap stage;
  * DerSimonian-Laird / independence-based pooling is NOT valid here (shared questions);
    use these bootstrap CIs for headline numbers, DL only as a heterogeneity descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def multinomial_counts(n_units: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B x n_units counts from resampling n_units units with replacement."""
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=B)


def _seed_unit_counts(design_cells: np.ndarray, seed_units: np.ndarray, B: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Resample seeds with replacement within each design cell.

    design_cells/seed_units: arrays aligned over UNIQUE seed units (one entry per
    (design_cell, seed) pair). Returns B x n_units integer counts.
    """
    n_units = len(seed_units)
    counts = np.zeros((B, n_units), dtype=np.int64)
    order = np.argsort(design_cells, kind="stable")
    sorted_cells = design_cells[order]
    boundaries = np.flatnonzero(np.r_[1, sorted_cells[1:] != sorted_cells[:-1], 1])
    # Group design cells by seed multiplicity, derived from the data — a hard-coded k
    # range would silently zero-weight any cell with more seeds (follow-up sweeps).
    for k in np.unique(np.diff(boundaries)):
        k = int(k)
        starts = [boundaries[i] for i in range(len(boundaries) - 1)
                  if boundaries[i + 1] - boundaries[i] == k]
        if not starts:
            continue
        idx = np.array([order[s:s + k] for s in starts])  # (n_cells_k, k)
        if k == 1:
            counts[:, idx[:, 0]] = 1
            continue
        draw = rng.multinomial(k, np.full(k, 1.0 / k), size=(B, len(starts)))
        for j in range(k):
            counts[:, idx[:, j]] += draw[:, :, j]
    return counts


class HierBoot:
    """Reusable hierarchical (qid -> seed) resampler over an item-level frame.

    Precomputes index structures once; per-replicate row weights are
    qid_count[qid(row)] * seed_count[(design_cell, seed)(row)]. Restrict the frame to a
    single benchmark before constructing (qids are only shared within benchmark).
    """

    def __init__(self, df: pd.DataFrame, B: int = 1000, seed: int = 0,
                 qid_level: bool = True, seed_level: bool = True):
        if qid_level and df["benchmark"].nunique() > 1:
            raise ValueError("qid resampling is per-benchmark; filter the frame first")
        rng = _rng(seed)
        self.df = df
        self.B = B
        uq, self._qinv = np.unique(df["qid"].to_numpy(), return_inverse=True)
        su = (df["design_cell"].astype(str) + "#" + df["seed"].astype(str)).to_numpy()
        uu, self._uinv = np.unique(su, return_inverse=True)
        self._qid_counts = (multinomial_counts(len(uq), B, rng) if qid_level
                            else np.ones((B, len(uq)), dtype=np.int64))
        if seed_level:
            unit_cells = np.array([u.split("#")[0] for u in uu])
            self._unit_counts = _seed_unit_counts(unit_cells, uu, B, rng)
        else:
            self._unit_counts = np.ones((B, len(uu)), dtype=np.int64)

    def weights(self, b: int) -> np.ndarray:
        return (self._qid_counts[b][self._qinv]
                * self._unit_counts[b][self._uinv]).astype(float)

    def mean_matrix(self, value_col: str, by: str | list[str]):
        """(labels, observed means, counts, B x G replicate-mean matrix) per group."""
        by = [by] if isinstance(by, str) else list(by)
        val = self.df[value_col].to_numpy(float)
        ok = np.isfinite(val)
        key = self.df[by].astype(str).agg("\x1f".join, axis=1).to_numpy()
        labels, ginv = np.unique(key[ok], return_inverse=True)
        v, q, u = val[ok], self._qinv[ok], self._uinv[ok]
        est = np.full((self.B, len(labels)), np.nan)
        for b in range(self.B):
            w = (self._qid_counts[b][q] * self._unit_counts[b][u]).astype(float)
            sums = np.bincount(ginv, w * v, minlength=len(labels))
            wts = np.bincount(ginv, w, minlength=len(labels))
            with np.errstate(invalid="ignore"):
                est[b] = sums / wts
        obs = np.bincount(ginv, v, minlength=len(labels)) / np.bincount(
            ginv, minlength=len(labels))
        return labels, obs, np.bincount(ginv, minlength=len(labels)), est

    def mean_by(self, value_col: str, by: str | list[str]) -> pd.DataFrame:
        """Bootstrap distribution of weighted group means of value_col.

        Returns one row per group: observed mean, se, lo/hi (2.5/97.5 percentiles), n.
        """
        by = [by] if isinstance(by, str) else list(by)
        labels, obs, n, est = self.mean_matrix(value_col, by)
        out = pd.DataFrame({
            "group": labels, "mean": obs, "n": n,
            "se": np.nanstd(est, axis=0),
            "lo": np.nanpercentile(est, 2.5, axis=0),
            "hi": np.nanpercentile(est, 97.5, axis=0),
        })
        if len(by) > 1:
            out[by] = out["group"].str.split("\x1f", expand=True).to_numpy()
        else:
            out[by[0]] = out["group"]
        return out.drop(columns="group")

    def stat(self, stat_fn) -> np.ndarray:
        """Bootstrap distribution of an arbitrary statistic: stat_fn(df, weights)->float/array."""
        vals = [np.asarray(stat_fn(self.df, self.weights(b)), float) for b in range(self.B)]
        return np.stack(vals)


def ci(dist: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (float(np.nanpercentile(dist, 100 * alpha / 2)),
            float(np.nanpercentile(dist, 100 * (1 - alpha / 2))))


def paired_accuracy_test(correct_a: np.ndarray, correct_b: np.ndarray,
                         B: int = 2000, seed: int = 0) -> dict:
    """Paired comparison over shared questions: bootstrap CI on delta + exact McNemar.

    Inputs are aligned correctness vectors over the SAME qids (assert alignment upstream).
    """
    from scipy.stats import binomtest

    a = np.asarray(correct_a, float)
    b = np.asarray(correct_b, float)
    assert a.shape == b.shape
    n = len(a)
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    dist = (a[idx] - b[idx]).mean(axis=1)
    delta = float(a.mean() - b.mean())
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    # binomtest is already two-sided — this IS the exact McNemar p, do not double it.
    mcnemar_p = (binomtest(min(n10, n01), n10 + n01, 0.5).pvalue
                 if (n10 + n01) > 0 else 1.0)
    lo, hi = ci(dist)
    p_boot = float(2 * min((dist <= 0).mean(), (dist >= 0).mean()))
    return {"delta": delta, "lo": lo, "hi": hi, "p_boot": min(1.0, p_boot),
            "p_mcnemar": min(1.0, mcnemar_p), "n": n, "n10": n10, "n01": n01}


def matched_n_ece(conf: np.ndarray, correct: np.ndarray, signal: str, n0: int,
                  draws: int = 20, seed: int = 0) -> float:
    """ECE at a matched sample size: subsample n0 without replacement, average draws.

    Robustness check for comparing ECE across cells with unequal n (plan §3).
    """
    from . import calib

    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(conf) & np.isfinite(correct)
    conf, correct = conf[ok], correct[ok]
    if len(conf) < n0:
        return np.nan
    rng = _rng(seed)
    vals = []
    for _ in range(draws):
        pick = rng.choice(len(conf), size=n0, replace=False)
        vals.append(calib.ece_primary(conf[pick], correct[pick], signal))
    return float(np.nanmean(vals))


def cluster_boot_mean(cells: pd.DataFrame, value_col: str, by: str | list[str],
                      B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cell-level fallback: cluster bootstrap resampling design cells (not rows).

    For cell-level frames where item data is unnecessary. Seeds of one design cell move
    together (the design cell is the resampled unit).
    """
    by = [by] if isinstance(by, str) else list(by)
    rng = _rng(seed)
    d = cells.dropna(subset=[value_col])
    clusters, cinv = np.unique(d["design_cell"].to_numpy(), return_inverse=True)
    key = d[by].astype(str).agg("\x1f".join, axis=1).to_numpy()
    labels, ginv = np.unique(key, return_inverse=True)
    val = d[value_col].to_numpy(float)
    counts = multinomial_counts(len(clusters), B, rng)
    est = np.full((B, len(labels)), np.nan)
    for b in range(B):
        w = counts[b][cinv].astype(float)
        sums = np.bincount(ginv, w * val, minlength=len(labels))
        wts = np.bincount(ginv, w, minlength=len(labels))
        with np.errstate(invalid="ignore"):
            est[b] = sums / wts
    obs = pd.Series(val).groupby(ginv).mean().to_numpy()
    out = pd.DataFrame({
        "group": labels, "mean": obs,
        "n": np.bincount(ginv, minlength=len(labels)),
        "se": np.nanstd(est, axis=0),
        "lo": np.nanpercentile(est, 2.5, axis=0),
        "hi": np.nanpercentile(est, 97.5, axis=0),
    })
    if len(by) > 1:
        out[by] = out["group"].str.split("\x1f", expand=True).to_numpy()
    else:
        out[by[0]] = out["group"]
    return out.drop(columns="group")


def ipw_weights(cells: pd.DataFrame, present_model, truncate: float = 10.0) -> np.ndarray:
    """Inverse-probability weights from a fitted missingness model (plan §3).

    present_model: fitted statsmodels result with .predict on the cells frame.
    """
    p = np.asarray(present_model.predict(cells), float)
    w = 1.0 / np.clip(p, 1e-3, 1.0)
    return np.minimum(w, truncate)
