"""ARD36 confirmatory statistics: ``study/stats.py``'s engine, vendored with the domain axis
as an argument (spec §15.1-§15.3; plan §8, §4.1 "vendor ... anything whose frozen vocabulary
would have to change").

Why vendor rather than import.  ``agents_scaling.study.stats`` hard-wires
``DOMAINS = ("hle", "bcb")`` and ``WEIGHTS = {"hle": .5, "bcb": .5}`` at module scope, and every
engine function filters its input with ``[d for d in DOMAINS if d in ...]``.  Handed ARD36's
family axis ``("math500", "mmlu_pro")`` it selects **no** domain and returns
``{"estimate": None, "p": 1.0}`` — silently, with no error.  All three primary contrasts would
be reported as "no rejection" and nothing in the pipeline would notice.  The Bonferroni divisor
``len(FAMILIES)`` (six study_v4 families) is likewise wrong here, so it becomes an explicit
``n_primary``.  The functions below are otherwise the study engine line for line;
``tests/ard/test_wp7_stats.py`` pins them to it bit-for-bit on an ``("hle", "bcb")`` fixture.

Contents

* :class:`Resampler` - deterministic source-cluster resample indices, drawn independently
  within each domain (spec §15.1: the cluster is the original task).
* :func:`paired_contrast` - null-centred studentised paired cluster bootstrap with a
  Monte-Carlo +1/+1 p-value, extended with a **shifted null** (``null_value``) and one-sided
  percentile bounds for the noninferiority endpoint (plan §8 safety endpoint).
* :func:`noninferiority_bound` - the pre-registered "does not hurt correct majorities"
  readout: a one-sided 95% bound against margin ``δ`` that reports "harm not excluded"
  whenever the bound fails to clear ``δ`` (spec §15.3 forbids reading a nonsignificant harm
  test as safety).
* :func:`max_t_family` - all-pair max-T over the centred resampling distribution.
* :func:`holm` - Holm step-down, with ``n_primary`` asserted against the number of contrasts.
* :func:`analysis_seed`, :func:`mean_ci` - vendored verbatim (no domain constants).

Loud where the original was quiet: :func:`paired_contrast` and :func:`max_t_family` raise
:class:`StatsAxisError` when the supplied ``diff_by_domain``/``y_by_domain`` key set is not
exactly ``domains``, and :class:`StatsError` when no estimate can be formed.  A statistics
engine that cannot compute the estimand must fail the analysis, not report ``p = 1``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from agents_scaling.ard import contract as K

#: Default family axis and weights for ARD36 (spec §15.1 prespecified equal-family macro
#: average).  Every engine function still takes ``domains``/``weights`` explicitly.
DOMAINS: tuple[str, ...] = tuple(f.value for f in K.FAMILIES)
WEIGHTS: dict[str, float] = dict(K.FAMILY_WEIGHTS)
N_RESAMPLES: int = K.N_RESAMPLES
ALPHA: float = K.ALPHA
N_PRIMARY: int = K.N_PRIMARY


class StatsError(RuntimeError):
    """The engine could not form the estimand it was asked for."""


class StatsAxisError(StatsError):
    """The supplied per-domain mapping does not match the declared domain axis."""


def analysis_seed(study_seed_hex: str, label: str) -> int:
    """Deterministic 63-bit seed derived from the frozen study seed and an analysis label.

    Vendored verbatim from ``study/stats.py`` (no domain constants).
    """
    digest = hashlib.sha256(f"{label}:{study_seed_hex}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class Resampler:
    """Source-cluster resample indices, drawn independently within each domain.

    ``domains`` fixes the draw order, so a given ``(n_by_domain, n_resamples, seed, domains)``
    reproduces the study engine's indices exactly when ``domains == ("hle", "bcb")``.
    """

    def __init__(
        self,
        n_by_domain: Mapping[str, int],
        n_resamples: int,
        seed: int,
        *,
        domains: Sequence[str] = DOMAINS,
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
    """The loud assertion the study engine lacks: an unknown key is a bug, not p = 1."""
    if set(by_domain) != set(domains):
        raise StatsAxisError(
            f"{what}: domain axis mismatch - got {sorted(by_domain)}, "
            f"declared {sorted(domains)}.  The study engine would have silently returned "
            f"estimate=None, p=1.0 here."
        )


def paired_contrast(
    diff_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    domains: Sequence[str] = DOMAINS,
    weights: Mapping[str, float] = WEIGHTS,
    n_primary: int = N_PRIMARY,
    alternative: str = "two-sided",
    components: int = 1,
    null_value: float = 0.0,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Null-centred studentised cluster bootstrap for a pooled mean of per-item contrasts.

    One scalar per item (spec §15.1: rows are averaged to the item before they get here).
    ``components`` is the family's number of required components: the conservative primary
    interval uses ``alpha / (n_primary * components)``.

    ``null_value`` shifts the null (H0: θ = ``null_value``), which with
    ``alternative="less"`` is the noninferiority test of plan §8; the returned
    ``ci_one_sided_upper_percentile`` / ``ci_one_sided_upper_studentized`` are the matching
    one-sided 1-``alpha`` bounds.
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
    # Conservative simultaneous PRIMARY bound: a scalar family with ``components`` required
    # components uses per-component alpha .05/(n_primary*c) (Bonferroni across the primaries).
    a_pri = alpha / (max(1, int(n_primary)) * max(1, int(components)))
    b_lo, b_hi = (float(x) for x in np.quantile(t_b, [a_pri / 2, 1 - a_pri / 2]))
    # One-sided 1-alpha bounds for the noninferiority endpoint (plan §8).
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
    if result["estimate"] is None:  # pragma: no cover - unreachable; the guard is the point
        raise StatsError("paired_contrast produced estimate=None")
    return result


def noninferiority_bound(
    diff_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    margin: float = K.NONINFERIORITY_MARGIN,
    domains: Sequence[str] = DOMAINS,
    weights: Mapping[str, float] = WEIGHTS,
    n_primary: int = N_PRIMARY,
    alpha: float = ALPHA,
    label: str = "GATE - PEER destruction on the C0=4 challenge cohort",
) -> dict[str, Any]:
    """The pre-registered safety endpoint (spec §15.3; plan §8): shifted-null one-sided test.

    ``diff_by_domain`` holds per-item **harm** contrasts (positive = the treatment destroys
    more correct majorities than the control).  Noninferiority is declared only when the
    one-sided 95% upper bound lies strictly below ``margin``; otherwise the readout is
    ``"harm not excluded"``, never "noninferior" and never "no significant harm".
    """
    res = paired_contrast(
        diff_by_domain, rs, domains=domains, weights=weights, n_primary=n_primary,
        alternative="less", null_value=float(margin), alpha=alpha,
    )
    upper = float(res["ci_one_sided_upper_studentized"])
    cleared = bool(upper < float(margin))
    res.update({
        "endpoint": "noninferiority",
        "label": label,
        "margin": float(margin),
        "upper_bound_one_sided": upper,
        "upper_bound_one_sided_percentile": float(res["ci_one_sided_upper_percentile"]),
        "noninferior": cleared,
        "readout": ("noninferiority met at the pre-registered margin" if cleared
                    else "harm not excluded"),
        "note": ("A one-sided bound above the margin is reported as 'harm not excluded'; "
                 "a nonsignificant harm test never establishes safety (spec §15.3)."),
    })
    return res


def max_t_family(
    y_by_domain: Mapping[str, np.ndarray],
    labels: Sequence[str],
    rs: Resampler,
    *,
    domains: Sequence[str] = DOMAINS,
    weights: Mapping[str, float] = WEIGHTS,
    n_primary: int = N_PRIMARY,
    alpha: float = ALPHA,
    chunk: int = 2_000,
) -> dict[str, Any]:
    """All-pair max-T over the centred resampling distribution.

    ``y_by_domain[d]`` is an ``(n_d, k)`` matrix of per-item outcomes, columns = ``labels``.
    Returns the global p-value, per-pair estimates with max-T-adjusted p-values and
    simultaneous 95% intervals, and per-arm pooled means with percentile intervals.
    """
    _check_axis(y_by_domain, domains, "max_t_family")
    k = len(labels)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    ds = [d for d in domains if y_by_domain[d] is not None and len(y_by_domain[d]) >= 2]
    if not ds or not pairs:
        raise StatsError(
            f"max_t_family: insufficient panel (domains with >= 2 clusters: {ds}, "
            f"pairs: {len(pairs)})"
        )
    w = _norm_weights(ds, weights)
    Y = {d: np.asarray(y_by_domain[d], dtype=float) for d in ds}
    n_d = {d: int(Y[d].shape[0]) for d in ds}
    P = len(pairs)
    est = np.zeros(P)
    var_sum = np.zeros(P)
    mean_m = np.zeros(k)
    for d in ds:
        D = Y[d][:, ii] - Y[d][:, jj]  # (n, P)
        est += w[d] * D.mean(axis=0)
        var_sum += w[d] ** 2 * D.var(axis=0, ddof=1) / n_d[d]
        mean_m += w[d] * Y[d].mean(axis=0)
    se = np.sqrt(var_sum)
    t = np.where(se > 0, est / np.where(se > 0, se, 1.0), 0.0)
    B = rs.n_resamples
    maxabs = np.zeros(B)
    means_b = np.zeros((B, k))
    for start in range(0, B, chunk):
        sl = slice(start, min(B, start + chunk))
        b = sl.stop - sl.start
        th = np.zeros((b, P))
        va = np.zeros((b, P))
        for d in ds:
            Yb = Y[d][rs.idx[d][sl]]  # (b, n_d, k)
            Db = Yb[:, :, ii] - Yb[:, :, jj]  # (b, n_d, P)
            th += w[d] * Db.mean(axis=1)
            va += w[d] ** 2 * Db.var(axis=1, ddof=1) / n_d[d]
            means_b[sl] += w[d] * Yb.mean(axis=1)
        seb = np.sqrt(va)
        tb = np.where(seb > 0, (th - est) / np.where(seb > 0, seb, 1.0), 0.0)
        maxabs[sl] = np.abs(tb).max(axis=1)
    obs_max = float(np.abs(t).max())
    p_global = (1 + int((maxabs >= obs_max).sum())) / (B + 1)
    q95 = float(np.quantile(maxabs, 1 - alpha))
    # The ordinary all-pair max-T interval controls only WITHIN this family (secondary);
    # a primary pairwise claim uses the same max-T distribution at alpha/n_primary.
    a_pri = alpha / max(1, int(n_primary))
    q_pri = float(np.quantile(maxabs, 1 - a_pri))
    pair_rows = []
    for p_idx, (i, j) in enumerate(pairs):
        pair_rows.append(
            {
                "contrast": f"{labels[i]} - {labels[j]}",
                "estimate": float(est[p_idx]),
                "se": float(se[p_idx]),
                "t": float(t[p_idx]),
                "p_maxT_adjusted": (1 + int((maxabs >= abs(t[p_idx])).sum())) / (B + 1) if se[p_idx] > 0 else 1.0,
                "ci95_simultaneous_within_family": [float(est[p_idx] - q95 * se[p_idx]), float(est[p_idx] + q95 * se[p_idx])],
                "ci_primary_bonferroni": [float(est[p_idx] - q_pri * se[p_idx]), float(est[p_idx] + q_pri * se[p_idx])],
            }
        )
    method_rows = []
    for m in range(k):
        lo, hi = (float(x) for x in np.quantile(means_b[:, m], [alpha / 2, 1 - alpha / 2]))
        method_rows.append(
            {
                "method": labels[m],
                "mean": float(mean_m[m]),
                "by_domain": {d: float(Y[d][:, m].mean()) for d in ds},
                "ci95_percentile": [lo, hi],
            }
        )
    return {
        "p_global": float(p_global),
        "max_abs_t_observed": obs_max,
        "max_abs_t_q95": q95,
        "max_abs_t_q_primary": q_pri,
        "primary_alpha": a_pri,
        "interval_note": ("ci95_simultaneous_within_family controls only within this family and is "
                          "SECONDARY; ci_primary_bonferroni is the conservative primary bound at "
                          f"alpha/{max(1, int(n_primary))}"),
        "n": n_d,
        "weights": w,
        "n_resamples": B,
        "n_primary": int(n_primary),
        "domains": list(domains),
        "methods": method_rows,
        "pairs": pair_rows,
    }


def mean_ci(
    values_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    domains: Sequence[str] = DOMAINS,
    weights: Mapping[str, float] = WEIGHTS,
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

    ``n_primary`` asserts the family size.  The primary family is frozen at exactly the three
    contrasts of ``contract.PRIMARY_CONTRASTS``: a fourth contrast slipping into the family
    (or a dropped one) changes every adjusted p-value, so it must fail here rather than be
    absorbed by ``m = len(pvalues)``.
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
    "ALPHA", "DOMAINS", "N_PRIMARY", "N_RESAMPLES", "WEIGHTS",
    "Resampler", "StatsAxisError", "StatsError",
    "analysis_seed", "holm", "max_t_family", "mean_ci", "noninferiority_bound",
    "paired_contrast",
]
