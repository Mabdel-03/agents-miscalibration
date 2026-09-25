"""E2 depth-response models (numpy/scipy only).

Every response model yields, per item, posterior draws ``P`` of shape (S, len(DEPTHS)): the per-call correctness
probability at each depth a in DEPTHS = (1, 2, 4, 8, 16)·A. Repeated calls at one depth reuse the same p (they are
conditionally independent given p); a mixed allocation {a: n_a} is covered with probability
1 - prod_a (1 - p_a)^(n_a), evaluated per posterior draw and then averaged ("products per draw").

* ``beta``: independent Beta(1/2, 1/2) prior per item-depth cell; posterior Beta(1/2 + c, 1/2 + m - c).
  Closed form: E[(1 - p)^k] = B(alpha, beta + k) / B(alpha, beta).
* ``glm``: pooled logistic, benchmark x categorical depth + Gauss-Hermite item intercept (``glm.fit_glm``), ridge by
  problem-grouped 5-fold CV; posterior draws combine Laplace draws of the fixed effects with the grid posterior of
  the item's intercept. The monotone variant (cumulative depth steps bounded >= 0) is compared on development CV only.
* ``shallow``: ablation -- the item's own information comes only from its A-level draws (Beta at A); deeper depths
  shift it by the pooled glm benchmark x depth contrasts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special

from agent_allocation.passk.models import glm as glm_mod
from agent_allocation.passk.models.inputs import BENCHMARKS, DEPTHS, label_value

A0 = 0.5
B0 = 0.5


# ---------------------------------------------------------------------- Beta
def beta_posterior(c: float, m: float, a0: float = A0, b0: float = B0) -> tuple[float, float]:
    return a0 + c, b0 + (m - c)


def beta_miss_moment(alpha, beta, k) -> np.ndarray:
    """E[(1 - p)^k] under Beta(alpha, beta): B(alpha, beta + k) / B(alpha, beta)."""
    return np.exp(special.betaln(alpha, np.asarray(beta, float) + k) - special.betaln(alpha, beta))


def beta_coverage(alpha, beta, k) -> np.ndarray:
    return 1.0 - beta_miss_moment(alpha, beta, k)


def coverage_from_draws(P: np.ndarray, alloc: dict[int, int]) -> np.ndarray:
    """Per-draw coverage of a (possibly mixed) allocation {depth: n calls}; P is (S, len(DEPTHS))."""
    miss = np.ones(P.shape[0])
    for a, n in alloc.items():
        miss = miss * (1.0 - P[:, DEPTHS.index(a)]) ** int(n)
    return 1.0 - miss


def cell_counts(fit_draws: pd.DataFrame) -> pd.DataFrame:
    """(item_id, benchmark, allowance_a) -> c (correct), m (known labels)."""
    d = fit_draws.copy()
    d["y"] = [label_value(v) for v in d["correct"]]
    d = d[~d.y.isna()]
    g = d.groupby(["item_id", "benchmark", "allowance_a"]).y.agg(["sum", "count"]).reset_index()
    return g.rename(columns={"sum": "c", "count": "m"})


def beta_draws(counts: pd.DataFrame, items: list[str], S: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    cc = {(r.item_id, int(r.allowance_a)): (r.c, r.m) for r in counts.itertuples(index=False)}
    for item in items:
        P = np.empty((S, len(DEPTHS)))
        for j, a in enumerate(DEPTHS):
            c, m = cc.get((item, a), (0.0, 0.0))
            al, be = beta_posterior(c, m)
            P[:, j] = rng.beta(al, be, size=S)
        out[item] = P
    return out


# ---------------------------------------------------------------------- pooled glm
def _design(benchmarks: np.ndarray, depths: np.ndarray, monotone: bool) -> tuple[np.ndarray, list[str], list]:
    cols, names, bounds = [], [], []
    for b in BENCHMARKS:
        isb = (benchmarks == b).astype(float)
        if monotone:
            cols.append(isb)
            names.append(f"{b}")
            bounds.append((None, None))
            for a in DEPTHS[1:]:
                cols.append(isb * (depths >= a))
                names.append(f"{b}:ge{a}")
                bounds.append((0.0, None))
        else:
            for a in DEPTHS:
                cols.append(isb * (depths == a))
                names.append(f"{b}:d{a}")
                bounds.append((None, None))
    return np.stack(cols, 1).astype(float), names, bounds


@dataclass
class PooledResponse:
    fit: glm_mod.GLMFit
    monotone: bool
    ridge: float
    cv: dict = field(default_factory=dict)

    def design(self, benchmark: str, depths=DEPTHS) -> np.ndarray:
        X, _, _ = _design(np.array([benchmark] * len(depths)), np.array(depths), self.monotone)
        return X

    def draws(self, item: str, benchmark: str, S: int, rng: np.random.Generator, *,
              use_item_posterior: bool = True) -> np.ndarray:
        X = self.design(benchmark)
        b0, coef, _ = self.fit.sample_params(S, rng)
        eta = b0[:, None] + coef @ X.T                                  # (S, D)
        b = (self.fit.posterior_item_effects(item, S, rng) if use_item_posterior
             else self.fit.sample_item_effects(S, rng))
        return special.expit(eta + b[:, None])

    def depth_contrast(self, benchmark: str) -> np.ndarray:
        """Fixed-effect logit shift of each depth relative to A for a benchmark."""
        eta = self.fit.linear_predictor(self.design(benchmark))
        return eta - eta[0]


def fit_pooled(fit_draws: pd.DataFrame, *, ridges=(0.1, 0.3, 1.0, 3.0, 10.0), seed: int = 20260925,
               compare_monotone: bool = True) -> PooledResponse:
    d = fit_draws.copy()
    d["y"] = [label_value(v) for v in d["correct"]]
    d = d[~d.y.isna() & d.allowance_a.isin(DEPTHS)]
    y = d.y.to_numpy(float)
    groups = d.item_id.to_numpy(object)
    bench = d.benchmark.to_numpy(object)
    depth = d.allowance_a.to_numpy(int)
    results = {}
    for mono in ((False, True) if compare_monotone else (False,)):
        X, names, bounds = _design(bench, depth, mono)
        best, table = glm_mod.cv_ridge(X, y, groups, item_intercept=True, ridges=ridges, seed=seed,
                                       bounds=bounds if mono else None, gh_nodes=15)
        results[mono] = (best, table, X, names, bounds)
    mono = min(results, key=lambda k: min(r["log_loss"] for r in results[k][1])) if compare_monotone else False
    best, table, X, names, bounds = results[mono]
    fit = glm_mod.fit_glm(X, y, groups, item_intercept=True, ridge=best, bounds=bounds if mono else None,
                          columns=names, laplace=True, gh_nodes=20)
    cv = {("monotone" if k else "categorical"): {"best_ridge": v[0], "table": v[1]} for k, v in results.items()}
    cv["chosen"] = "monotone" if mono else "categorical"
    return PooledResponse(fit=fit, monotone=mono, ridge=best, cv=cv)


def shallow_draws(counts: pd.DataFrame, pooled: PooledResponse, item: str, benchmark: str, S: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Ablation: item info from A draws only; deeper depths = logit(p_A) + pooled benchmark depth contrast."""
    row = counts[(counts.item_id == item) & (counts.allowance_a == 1)]
    c, m = (float(row.c.iloc[0]), float(row.m.iloc[0])) if len(row) else (0.0, 0.0)
    al, be = beta_posterior(c, m)
    pA = np.clip(rng.beta(al, be, size=S), 1e-6, 1 - 1e-6)
    return special.expit(special.logit(pA)[:, None] + pooled.depth_contrast(benchmark)[None, :])
