"""The confirmatory statistics engine: ``ard.stats`` imported, plus the tests nextdyn adds
(spec 14.1-14.3; plan D13, §4.2).

**Nothing is re-vendored.**  ``ard/stats.py`` already carries the study engine with the domain
axis as an argument, a loud axis assertion, a null-centred studentised cluster bootstrap and a
Monte-Carlo ``(1 + #extreme)/(B + 1)`` p-value.  Those are imported by name from the
``nextdyn.ARD_IMPORTS`` allowlist and used for **intervals** (``paired_contrast`` percentile-t),
so the numbers in this study and in ARD36 come from one implementation and the E0 leg can
reproduce an ARD36 claim through the original code path.

What this module adds, and why each addition exists rather than reusing what is there:

* :func:`wild_cluster_test` -- the **test of record** (plan D13).  ``paired_contrast``
  resamples clusters *with replacement*, which is the right interval engine but is known to be
  liberal with few clusters; the wild bootstrap instead re-signs the within-family residuals,
  which keeps the null exactly centred at zero and is what the pre-registration names.  Webb's
  six-point weights replace Rademacher below ``webb_below_clusters`` clusters, where the two
  Rademacher signs give only ``2^G`` distinct draws.
* :func:`joint_contrast` -- H2 is a **heterogeneity** hypothesis: it must reject when the three
  per-family effects differ from each other and *not* when they are equal and large.  No
  contrast in ``ard.stats`` does that; ``max_t_family`` compares arms, not families.
* :func:`holm_with_not_estimable` -- ``ard.stats.holm`` refuses a family whose size is not
  ``n_primary``, and has nowhere to put a hypothesis whose estimand could not be formed.  Here
  a ``None`` p-value keeps its slot with ``p = 1`` and the verdict ``NOT_ESTIMABLE``, so
  dropping a primary can never make the other seven easier to reject.
* :func:`mcnemar_exact` -- vendored from ``analysis/nb_lib/boot.py:137`` together with the one
  comment that matters: ``binomtest`` is **already two-sided**, so the p-value is not doubled.
* :func:`bh_fdr` -- vendored from ``analysis/nb_lib/stats.py:297``.  The original calls
  ``statsmodels``, which is not installed in the harness environment; the step-up rule is
  fifteen lines and is implemented here with a parity test against ``statsmodels`` that skips
  when it is absent.
* :func:`tost_equivalence` -- the pre-declared +/-5 pp two-one-sided-test readout, which is what
  a NARROWED verdict must invoke instead of reading a nonsignificant difference as "the same".

Every value comes from the compiled configuration: ``n_resamples``,
``bootstrap_seed_label_rule``, ``equivalence_margin_pp``, ``secondary_fdr_q``,
``min_clusters_for_test`` and ``webb_below_clusters`` from ``analysis.yaml``'s profile, and
Holm's family size and alpha computed from ``study.analysis`` (R1: ``analysis.yaml`` never
re-types them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from agents_scaling.ard.stats import (  # noqa: F401  (re-exported: one engine, one implementation)
    Resampler,
    analysis_seed,
    mean_ci,
    noninferiority_bound,
    paired_contrast,
)
from agents_scaling.nextdyn.errors import ContractViolation

#: The seed rule this module implements.  ``analysis.yaml`` states it as a string so a reader of
#: the frozen config can see it; :func:`settings` holds the string to the implementation, which
#: is what stops the declared rule and the coded rule from drifting apart.
SEED_LABEL_RULE: Final[str] = "ard.stats.analysis_seed(study_seed_hex, hypothesis_id)"

RADEMACHER: Final[str] = "rademacher"
WEBB: Final[str] = "webb"

#: Webb (2014) six-point weights: sqrt(3/2), sqrt(1), sqrt(1/2) with both signs, equiprobable.
#: They give 6^G distinct draws instead of 2^G, which matters exactly where the cluster count is
#: small enough for the Rademacher lattice to quantise the p-value.
_WEBB_POINTS: Final[tuple[float, ...]] = (
    -math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5),
)

NOT_ESTIMABLE: Final[str] = "NOT_ESTIMABLE"
ESTIMATION_ONLY: Final[str] = "estimation_only"


@dataclass(frozen=True)
class StatsSettings:
    """Every statistical knob, resolved once from the two configs (R1, R2)."""

    n_resamples: int
    seed_label_rule: str
    equivalence_margin_pp: float
    secondary_fdr_q: float
    min_clusters_for_test: int
    webb_below_clusters: int
    n_primary: int
    alpha: float

    @property
    def equivalence_margin(self) -> float:
        """The TOST margin as a proportion (the config states percentage points)."""
        return float(self.equivalence_margin_pp) / 100.0


def settings(profile: Any, analysis_block: Any) -> StatsSettings:
    """Compile ``analysis.yaml``'s profile and ``study.yaml``'s analysis block into knobs.

    ``profile`` is a ``config.AnalysisProfile`` and ``analysis_block`` a ``config.AnalysisBlock``.
    Holm's family size is ``len(primary_hypotheses)`` and its alpha is ``familywise_alpha``:
    both are computed here and never re-typed in ``analysis.yaml`` (contract §0 rule 9).
    """
    rule = str(profile.bootstrap_seed_label_rule)
    if rule != SEED_LABEL_RULE:
        raise ContractViolation(
            f"analysis.profiles.*.bootstrap_seed_label_rule is {rule!r} but stats.py implements "
            f"{SEED_LABEL_RULE!r}; the declared rule and the coded rule must be the same fact")
    return StatsSettings(
        n_resamples=int(profile.n_resamples),
        seed_label_rule=rule,
        equivalence_margin_pp=float(profile.equivalence_margin_pp),
        secondary_fdr_q=float(profile.secondary_fdr_q),
        min_clusters_for_test=int(profile.min_clusters_for_test),
        webb_below_clusters=int(profile.webb_below_clusters),
        n_primary=len(tuple(analysis_block.primary_hypotheses)),
        alpha=float(analysis_block.familywise_alpha),
    )


# --------------------------------------------------------------------------- wild cluster


def _normalized_weights(families: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    missing = [f for f in families if f not in weights]
    if missing:
        raise ContractViolation(f"no task weight for {missing}")
    total = math.fsum(weights[f] for f in families)
    if total <= 0:
        raise ContractViolation(f"task weights sum to {total}")
    return {f: weights[f] / total for f in families}


def _draw_weights(rng: np.random.Generator, shape: tuple[int, ...], dist: str) -> np.ndarray:
    if dist == RADEMACHER:
        return rng.choice(np.array([-1.0, 1.0]), size=shape)
    if dist == WEBB:
        return rng.choice(np.array(_WEBB_POINTS), size=shape)
    raise ContractViolation(f"unknown wild-bootstrap weight distribution {dist!r}")


def wild_cluster_test(
    diff_by_family: Mapping[str, Sequence[float]],
    *,
    weights: Mapping[str, float],
    n_resamples: int,
    seed: int,
    webb_below_clusters: int = 10,
    min_clusters_for_test: int | None = None,
    alternative: str = "two-sided",
    null_value: float = 0.0,
    label: str = "",
) -> dict[str, Any]:
    """Studentised, **null-centred** wild cluster bootstrap over source items (plan §4.2).

    One scalar ``d_i`` per source item (replicates already averaged, spec 14.2).  With
    ``w_f`` the normalised task weights::

        theta_hat = sum_f w_f * mean_f(d)
        se        = sqrt(sum_f w_f^2 * var_f(d) / n_f)
        t_obs     = theta_hat / se
        d*_i      = (d_i - mean_f(d)) * omega_i          # residual re-signed, mean removed
        t*        = theta* / se*
        p         = (1 + #{|t*| >= |t_obs|}) / (B + 1)

    Subtracting the within-family mean is what makes the resampling distribution a sample from
    the **null**: ``theta*`` has expectation zero whatever the true effect is, so the test has
    its nominal size even where the effect is large.  ``omega`` is Rademacher, or Webb's
    six-point distribution when any family has fewer than ``webb_below_clusters`` clusters.
    """
    families = sorted(diff_by_family)
    values = {f: np.asarray(diff_by_family[f], dtype=float) for f in families}
    usable = [f for f in families if values[f].size >= 2]
    if not usable:
        return {"estimate": None, "p": None, "verdict": NOT_ESTIMABLE, "label": label,
                "n_clusters": int(sum(v.size for v in values.values())),
                "reason": "no task has at least two clusters", "by_family": {},
                "n": {f: int(values[f].size) for f in families}}
    w = _normalized_weights(usable, weights)
    n_f = {f: int(values[f].size) for f in usable}
    mean_f = {f: float(values[f].mean()) for f in usable}
    var_f = {f: float(values[f].var(ddof=1)) for f in usable}
    estimate = math.fsum(w[f] * mean_f[f] for f in usable)
    se = math.sqrt(math.fsum(w[f] ** 2 * var_f[f] / n_f[f] for f in usable))
    n_clusters = sum(n_f.values())
    dist = WEBB if min(n_f.values()) < int(webb_below_clusters) else RADEMACHER

    rng = np.random.default_rng(int(seed))
    B = int(n_resamples)
    theta = np.zeros(B)
    var_acc = np.zeros(B)
    for f in usable:
        residual = values[f] - mean_f[f]
        omega = _draw_weights(rng, (B, n_f[f]), dist)
        starred = residual[None, :] * omega
        theta += w[f] * starred.mean(axis=1)
        var_acc += w[f] ** 2 * starred.var(axis=1, ddof=1) / n_f[f]
    se_star = np.sqrt(var_acc)
    ok = se_star > 0
    t_star = np.where(ok, theta / np.where(ok, se_star, 1.0), 0.0)
    t_obs = ((estimate - float(null_value)) / se) if se > 0 else 0.0
    if se <= 0:
        p = 1.0
    elif alternative == "two-sided":
        p = (1 + int((np.abs(t_star) >= abs(t_obs)).sum())) / (B + 1)
    elif alternative == "greater":
        p = (1 + int((t_star >= t_obs).sum())) / (B + 1)
    elif alternative == "less":
        p = (1 + int((t_star <= t_obs).sum())) / (B + 1)
    else:
        raise ContractViolation(f"unknown alternative {alternative!r}")
    verdict = None
    if min_clusters_for_test is not None and n_clusters < int(min_clusters_for_test):
        verdict = ESTIMATION_ONLY
    return {
        "estimate": float(estimate), "se": float(se), "t": float(t_obs), "p": float(p),
        "n": n_f, "by_family": mean_f, "weights": w, "n_clusters": int(n_clusters),
        "n_resamples": B, "weights_dist": dist, "seed": int(seed), "label": label,
        "alternative": alternative, "null_value": float(null_value),
        "verdict": verdict, "families": usable,
        "families_dropped": [f for f in families if f not in usable],
    }


# --------------------------------------------------------------------------- H2 joint contrast


def joint_contrast(
    diff_by_family: Mapping[str, Sequence[float]],
    *,
    n_resamples: int,
    seed: int,
    min_items_per_family: int = 30,
    min_families: int = 2,
    label: str = "H2",
) -> dict[str, Any]:
    """Wald-type joint test that the per-family effects are **not all equal** (H2, spec 14.1).

    With ``theta = (D_f)_f`` and the successive-difference contrast matrix
    ``L = [[1,-1,0],[0,1,-1]]`` (one row fewer than families), ``Sigma = diag(s_f^2 / n_f)``
    (families are disjoint item sets, so the covariance is diagonal by construction) and
    ``T = (L theta)' (L Sigma L')^-1 (L theta)``.  The reference distribution is built by
    resampling items **within** each family and centring each family's mean at its observed
    value, so ``T*`` is drawn under "all families share one effect" -- which is why a large
    homogeneous effect does not reject and a small heterogeneous one does
    (``test_joint_contrast_rejects_only_heterogeneous_task_effects``).
    """
    families = sorted(diff_by_family)
    values = {f: np.asarray(diff_by_family[f], dtype=float) for f in families}
    supported = [f for f in families if values[f].size >= int(min_items_per_family)]
    if len(supported) < int(min_families):
        return {"verdict": NOT_ESTIMABLE, "p": None, "statistic": None, "label": label,
                "reason": (f"fewer than {min_families} tasks have >= {min_items_per_family} "
                           f"paired items (have {[(f, int(values[f].size)) for f in families]})"),
                "by_family": {f: float(values[f].mean()) if values[f].size else None for f in families},
                "families": supported}
    k = len(supported)
    theta = np.array([values[f].mean() for f in supported])
    n_f = np.array([values[f].size for f in supported], dtype=float)
    var_f = np.array([values[f].var(ddof=1) for f in supported])
    contrast = np.zeros((k - 1, k))
    for i in range(k - 1):
        contrast[i, i], contrast[i, i + 1] = 1.0, -1.0

    def statistic(th: np.ndarray, sigma_diag: np.ndarray) -> float:
        middle = contrast @ np.diag(sigma_diag) @ contrast.T
        gap = contrast @ th
        try:
            solved = np.linalg.solve(middle, gap)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(gap @ solved)

    observed = statistic(theta, var_f / n_f)
    rng = np.random.default_rng(int(seed))
    B = int(n_resamples)
    exceed = 0
    finite = 0
    for _ in range(B):
        th = np.empty(k)
        sig = np.empty(k)
        for j, f in enumerate(supported):
            draw = values[f][rng.integers(0, values[f].size, values[f].size)]
            th[j] = draw.mean() - theta[j]       # centred: every family shares the null effect 0
            sig[j] = draw.var(ddof=1) / draw.size
        value = statistic(th, sig)
        if math.isfinite(value):
            finite += 1
            exceed += int(value >= observed)
    p = (1 + exceed) / (finite + 1) if finite else None
    return {
        "verdict": None, "statistic": float(observed), "p": None if p is None else float(p),
        "by_family": {f: float(values[f].mean()) for f in supported},
        "se_by_family": {f: float(math.sqrt(values[f].var(ddof=1) / values[f].size)) for f in supported},
        "n": {f: int(values[f].size) for f in supported},
        "families": supported, "n_resamples": B, "n_finite": finite, "seed": int(seed),
        "label": label, "contrast": contrast.tolist(),
    }


# --------------------------------------------------------------------------- multiplicity


def holm_with_not_estimable(
    pvalues: Mapping[str, float | None],
    *,
    alpha: float,
    n_primary: int,
) -> dict[str, dict[str, Any]]:
    """Holm step-down over exactly ``n_primary`` hypotheses; ``None`` keeps its slot at ``p = 1``.

    A hypothesis whose estimand could not be formed is ``NOT_ESTIMABLE``, carries ``p = 1``, and
    stays **in** the family.  Dropping it would shrink ``m`` and make every other adjusted
    p-value smaller, which is exactly the way a primary family silently becomes easier to
    reject (spec 14.1; ``test_holm_over_eight_with_not_estimable_slot``).
    """
    if len(pvalues) != int(n_primary):
        raise ContractViolation(
            f"the primary family has exactly {int(n_primary)} hypotheses, got {len(pvalues)}: "
            f"{sorted(pvalues)}")
    items = sorted(((name, 1.0 if p is None else float(p)) for name, p in pvalues.items()),
                   key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[name] = {
            "p": p,
            "p_holm": float(running),
            "reject": bool(running <= float(alpha) and pvalues[name] is not None),
            "rank": rank + 1,
            "estimable": pvalues[name] is not None,
            "verdict": None if pvalues[name] is not None else NOT_ESTIMABLE,
        }
    return out


def bh_fdr(pvalues: Mapping[str, float | None], *, q: float) -> dict[str, dict[str, Any]]:
    """Benjamini-Hochberg step-up (vendored from ``analysis/nb_lib/stats.py:297``).

    The original delegates to ``statsmodels.stats.multitest.multipletests(method='fdr_bh')``,
    which is not installed in the harness environment; the rule is reproduced here and pinned
    to it by a parity test that skips when statsmodels is absent.  Non-finite p-values are kept
    out of ``m`` and reported as not estimable rather than counted as ones.
    """
    finite = sorted(((name, float(p)) for name, p in pvalues.items()
                     if p is not None and math.isfinite(float(p))), key=lambda kv: (kv[1], kv[0]))
    m = len(finite)
    out: dict[str, dict[str, Any]] = {
        name: {"p": None, "p_adj": None, "reject": False, "rank": None, "estimable": False,
               "verdict": NOT_ESTIMABLE}
        for name, p in pvalues.items() if p is None or not math.isfinite(float(p))
    }
    running = 1.0
    adjusted: dict[str, float] = {}
    for rank in range(m, 0, -1):
        name, p = finite[rank - 1]
        running = min(running, p * m / rank)
        adjusted[name] = running
    for rank, (name, p) in enumerate(finite, start=1):
        out[name] = {"p": p, "p_adj": float(adjusted[name]),
                     "reject": bool(adjusted[name] <= float(q)), "rank": rank,
                     "estimable": True, "verdict": None}
    return out


# --------------------------------------------------------------------------- paired binary


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict[str, Any]:
    """Exact McNemar on aligned binary vectors (vendored, ``analysis/nb_lib/boot.py:137``).

    ``scipy.stats.binomtest`` is **already two-sided**: this IS the exact McNemar p, and it is
    not doubled.  Only the discordant pairs carry information, so ``n01``/``n10`` are reported
    beside the p-value and an all-concordant comparison returns ``p = 1`` with ``n_discordant``
    zero rather than an undefined statistic.
    """
    from scipy.stats import binomtest

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ContractViolation(f"McNemar needs aligned vectors, got {left.shape} and {right.shape}")
    n01 = int(np.sum((left == 0) & (right == 1)))
    n10 = int(np.sum((left == 1) & (right == 0)))
    discordant = n10 + n01
    p = binomtest(min(n10, n01), discordant, 0.5).pvalue if discordant else 1.0
    return {"n": int(left.size), "n01": n01, "n10": n10, "n_discordant": discordant,
            "delta": float(left.mean() - right.mean()) if left.size else None,
            "p": float(min(1.0, p)), "two_sided": True,
            "note": "binomtest is already two-sided; this is the exact McNemar p, never doubled"}


def tost_equivalence(
    diff_by_family: Mapping[str, Sequence[float]],
    *,
    weights: Mapping[str, float],
    margin: float,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
    label: str = "",
) -> dict[str, Any]:
    """Two one-sided tests against a pre-declared +/- ``margin`` (plan D13; spec 14.1).

    Equivalence is declared only when **both** one-sided nulls are rejected, i.e. the two-sided
    ``1 - 2*alpha`` interval lies strictly inside ``(-margin, +margin)``.  A nonsignificant
    difference is never read as equivalence: the ``equivalent`` flag and the ``readout`` string
    say which of the two it is.
    """
    lower = wild_cluster_test(
        diff_by_family, weights=weights, n_resamples=n_resamples, seed=seed,
        alternative="greater", null_value=-float(margin), label=f"{label}:lower")
    upper = wild_cluster_test(
        diff_by_family, weights=weights, n_resamples=n_resamples, seed=seed,
        alternative="less", null_value=float(margin), label=f"{label}:upper")
    if lower.get("estimate") is None or upper.get("estimate") is None:
        return {"verdict": NOT_ESTIMABLE, "equivalent": None, "margin": float(margin),
                "label": label, "reason": lower.get("reason") or upper.get("reason")}
    p = max(float(lower["p"]), float(upper["p"]))
    equivalent = bool(p <= float(alpha))
    return {
        "verdict": None, "equivalent": equivalent, "p": float(p),
        "p_lower": float(lower["p"]), "p_upper": float(upper["p"]),
        "estimate": float(lower["estimate"]), "margin": float(margin),
        "alpha": float(alpha), "label": label,
        "readout": ("equivalent within the pre-declared margin" if equivalent
                    else "equivalence not established (this is not evidence of a difference)"),
    }


__all__ = [
    "ESTIMATION_ONLY", "NOT_ESTIMABLE", "RADEMACHER", "Resampler", "SEED_LABEL_RULE",
    "StatsSettings", "WEBB", "analysis_seed", "bh_fdr", "holm_with_not_estimable",
    "joint_contrast", "mcnemar_exact", "mean_ci", "noninferiority_bound", "paired_contrast",
    "settings", "tost_equivalence", "wild_cluster_test",
]
