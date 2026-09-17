"""Item-clustered bootstrap engine, vendored from ``agents_scaling.ard.stats`` (origin commit e4b4037).

Origin: ``src/agents_scaling/ard/stats.py`` (sha256 e27ad9ae...). Provenance row in ``PROVENANCE.json``.

Changes from the origin (and nothing else):

* The ARD36 module constants (``DOMAINS``, ``WEIGHTS``, ``N_RESAMPLES``, ``N_PRIMARY`` from
  ``agents_scaling.ard.contract``) are removed. ``domains``, ``weights`` and ``n_primary`` are now
  **required keyword arguments** with no default, so an analysis can never fall back to another study's axis.
* ``alpha`` keeps the conventional default 0.05.
* ``noninferiority_bound`` and ``max_t_family`` are not copied (unused here).

Everything else (resample indices, studentized statistic, +1/+1 Monte-Carlo p-value, percentile intervals,
Holm step-down, the loud axis checks) is the origin line for line.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ALPHA: float = 0.05


class StatsError(RuntimeError):
    """The engine could not form the estimand it was asked for."""


class StatsAxisError(StatsError):
    """The supplied per-domain mapping does not match the declared domain axis."""


def analysis_seed(study_seed_hex: str, label: str) -> int:
    """Deterministic 63-bit seed derived from the frozen study seed and an analysis label."""
    digest = hashlib.sha256(f"{label}:{study_seed_hex}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class Resampler:
    """Source-cluster resample indices, drawn independently within each domain.

    ``domains`` fixes the draw order, so a given ``(n_by_domain, n_resamples, seed, domains)`` always
    reproduces the same indices.
    """

    def __init__(
        self,
        n_by_domain: Mapping[str, int],
        n_resamples: int,
        seed: int,
        *,
        domains: Sequence[str],
    ) -> None:
        self.domains = tuple(domains)
        if len(set(self.domains)) != len(self.domains):
            raise StatsAxisError(f"duplicate domain in {self.domains!r}")
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)
        self.n_by_domain = {d: int(n_by_domain.get(d, 0)) for d in self.domains}
        rng = np.random.default_rng(self.seed)
        self.idx: dict[str, np.ndarray] = {}
        for d in self.domains:
            n = self.n_by_domain[d]
            self.idx[d] = (
                rng.integers(0, n, size=(self.n_resamples, n), dtype=np.int32)
                if n > 0
                else np.zeros((self.n_resamples, 0), dtype=np.int32)
            )


def _norm_weights(domains: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    missing = [d for d in domains if d not in weights]
    if missing:
        raise StatsAxisError(f"no weight for domain(s) {missing}")
    total = math.fsum(weights[d] for d in domains)
    if total <= 0:
        raise StatsAxisError(f"domain weights sum to {total}")
    return {d: weights[d] / total for d in domains}


def _check_axis(by_domain: Mapping[str, Any], domains: Sequence[str], what: str) -> None:
    """An unknown or missing key is a bug, not p = 1."""
    if set(by_domain) != set(domains):
        raise StatsAxisError(
            f"{what}: domain axis mismatch - got {sorted(by_domain)}, declared {sorted(domains)}."
        )


def paired_contrast(
    diff_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    domains: Sequence[str],
    weights: Mapping[str, float],
    n_primary: int,
    alternative: str = "two-sided",
    components: int = 1,
    null_value: float = 0.0,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Null-centred studentised cluster bootstrap for a pooled mean of per-item contrasts.

    One scalar per item (rows are averaged to the item before they get here).
    """
    _check_axis(diff_by_domain, domains, "paired_contrast")
    ds = [d for d in domains if len(diff_by_domain[d]) >= 2]
    if not ds:
        raise StatsError(
            "paired_contrast: no domain has >= 2 clusters; the estimand is undefined "
            f"(sizes {{{', '.join(f'{d}: {len(diff_by_domain[d])}' for d in domains)}}})"
        )
    w = _norm_weights(ds, weights)
    vals = {d: np.asarray(diff_by_domain[d], dtype=float) for d in ds}
    n_d = {d: int(vals[d].shape[0]) for d in ds}
    est_d = {d: float(vals[d].mean()) for d in ds}
    var_d = {d: float(vals[d].var(ddof=1)) for d in ds}
    est = math.fsum(w[d] * est_d[d] for d in ds)
    se = math.sqrt(math.fsum(w[d] ** 2 * var_d[d] / n_d[d] for d in ds))
    B = rs.n_resamples
    theta = np.zeros(B)
    var_acc = np.zeros(B)
    for d in ds:
        v = vals[d][rs.idx[d]]  # (B, n_d)
        theta += w[d] * v.mean(axis=1)
        var_acc += w[d] ** 2 * v.var(axis=1, ddof=1) / n_d[d]
    se_b = np.sqrt(var_acc)
    ok = se_b > 0
    t_b = np.where(ok, (theta - est) / np.where(ok, se_b, 1.0), 0.0)
    t_obs = (est - null_value) / se if se > 0 else 0.0
    if se <= 0:
        p = 1.0
    elif alternative == "two-sided":
        p = (1 + int((np.abs(t_b) >= abs(t_obs)).sum())) / (B + 1)
    elif alternative == "greater":
        p = (1 + int((t_b >= t_obs).sum())) / (B + 1)
    elif alternative == "less":
        p = (1 + int((t_b <= t_obs).sum())) / (B + 1)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    q_lo, q_hi = (float(x) for x in np.quantile(t_b, [alpha / 2, 1 - alpha / 2]))
    p_lo, p_hi = (float(x) for x in np.quantile(theta, [alpha / 2, 1 - alpha / 2]))
    a_pri = alpha / (max(1, int(n_primary)) * max(1, int(components)))
    b_lo, b_hi = (float(x) for x in np.quantile(t_b, [a_pri / 2, 1 - a_pri / 2]))
    t_one = float(np.quantile(t_b, alpha))
    result = {
        "estimate": est,
        "by_domain": est_d,
        "n": n_d,
        "weights": w,
        "se": se,
        "t": t_obs,
        "p": float(p),
        "alternative": alternative,
        "null_value": float(null_value),
        "ci95_studentized": [est - q_hi * se, est - q_lo * se],
        "ci95_percentile": [p_lo, p_hi],
        "ci_primary_bonferroni": [est - b_hi * se, est - b_lo * se],
        "ci_one_sided_upper_percentile": float(np.quantile(theta, 1 - alpha)),
        "ci_one_sided_upper_studentized": est - t_one * se,
        "primary_alpha": a_pri,
        "components": int(components),
        "n_primary": int(n_primary),
        "n_resamples": B,
        "domains": list(domains),
        "alpha": float(alpha),
    }
    return result


def mean_ci(
    values_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    domains: Sequence[str],
    weights: Mapping[str, float],
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Pooled mean with an ordinary percentile interval (secondary/descriptive tables)."""
    _check_axis(values_by_domain, domains, "mean_ci")
    ds = [d for d in domains if len(values_by_domain[d]) >= 1]
    if not ds:
        raise StatsError("mean_ci: every domain is empty")
    w = _norm_weights(ds, weights)
    vals = {d: np.asarray(values_by_domain[d], dtype=float) for d in ds}
    est_d = {d: float(vals[d].mean()) for d in ds}
    est = math.fsum(w[d] * est_d[d] for d in ds)
    theta = np.zeros(rs.n_resamples)
    for d in ds:
        theta += w[d] * vals[d][rs.idx[d]].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(theta, [alpha / 2, 1 - alpha / 2]))
    return {"mean": est, "by_domain": est_d, "n": {d: int(vals[d].shape[0]) for d in ds},
            "ci95_percentile": [lo, hi]}


def holm(
    pvalues: Mapping[str, float],
    alpha: float = ALPHA,
    *,
    n_primary: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Holm step-down; adjusted p-values are monotone and capped at 1.

    ``n_primary`` asserts the family size, so a contrast slipping into or out of the family fails loudly.
    """
    if n_primary is not None and len(pvalues) != int(n_primary):
        raise StatsAxisError(
            f"holm: expected exactly {int(n_primary)} contrasts in this family, "
            f"got {len(pvalues)}: {sorted(pvalues)}"
        )
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = {"p": float(p), "p_holm": float(running), "reject": bool(running <= alpha),
                     "rank": rank + 1}
    return out


__all__ = [
    "ALPHA", "Resampler", "StatsAxisError", "StatsError", "analysis_seed", "holm", "mean_ci", "paired_contrast",
]
