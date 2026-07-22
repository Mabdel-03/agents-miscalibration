"""Calibration estimators beyond the package's 15-bin equal-width plug-in ECE.

Pre-registered estimator policy (analysis plan §3):
  * continuous confidence signals (logprobs, verbal)  -> equal-mass 10-bin L1 ECE
  * discrete signals (vote_fraction: atoms {1/3, 2/3, 1}) -> exact-atom ECE (no binning)
  * robustness: debiased l2-ECE (Kumar-Liang-Ma 2019) and Brier reliability (Murphy)
  * the 15-bin equal-width plug-in is reported once, only for Kim et al. comparability
Never compare plug-in ECEs across cells with different n; use the debiased estimator or
matched-n subsampling (boot.matched_n_ece).
"""

from __future__ import annotations

import numpy as np

# Signals that are discrete by construction (finite atoms), not merely coarse.
DISCRETE_SIGNALS = {"vote_fraction"}

_EPS = 1e-6


def _clean(conf, correct, weights=None):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(conf) & np.isfinite(correct)
    if weights is None:
        return conf[ok], correct[ok], None
    weights = np.asarray(weights, float)
    if len(weights) != len(conf):
        raise ValueError("weights must align with conf/correct BEFORE NaN filtering")
    return conf[ok], correct[ok], weights[ok]


def equal_mass_edges(conf: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges (deduplicated). Fewer unique edges => effectively fewer bins."""
    conf = np.asarray(conf, float)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # every point falls in some bin
    return np.unique(edges)


def bin_index(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)


def grouped_ece(conf: np.ndarray, correct: np.ndarray, groups: np.ndarray,
                weights: np.ndarray | None = None) -> float:
    """Weighted L1 ECE over an arbitrary grouping (bins or exact atoms).

    ECE = sum_g (W_g / W) * |mean_conf_g - mean_acc_g| with per-point weights (all-ones
    by default). The weighted form is what the fixed-bin bootstrap re-weights.
    """
    if weights is None:
        weights = np.ones_like(conf)
    w_g = np.bincount(groups, weights=weights)
    keep = w_g > 0
    conf_g = np.bincount(groups, weights=weights * conf)[keep] / w_g[keep]
    acc_g = np.bincount(groups, weights=weights * correct)[keep] / w_g[keep]
    w_g = w_g[keep]
    return float(np.sum(w_g / w_g.sum() * np.abs(conf_g - acc_g)))


def ece_equal_mass(conf, correct, n_bins: int = 10,
                   weights: np.ndarray | None = None) -> float:
    """Primary estimator for continuous signals: equal-mass (quantile) L1 ECE."""
    conf, correct, weights = _clean(conf, correct, weights)
    if len(conf) == 0:
        return np.nan
    groups = bin_index(conf, equal_mass_edges(conf, n_bins))
    return grouped_ece(conf, correct, groups, weights)


def ece_discrete(conf, correct, weights: np.ndarray | None = None) -> float:
    """Primary estimator for discrete signals: exact ECE over support atoms."""
    conf, correct, weights = _clean(conf, correct, weights)
    if len(conf) == 0:
        return np.nan
    _, groups = np.unique(np.round(conf, 6), return_inverse=True)
    return grouped_ece(conf, correct, groups, weights)


def ece_primary(conf, correct, signal: str, n_bins: int = 10) -> float:
    """Dispatch on the pre-registered signal-type policy."""
    if signal in DISCRETE_SIGNALS:
        return ece_discrete(conf, correct)
    return ece_equal_mass(conf, correct, n_bins=n_bins)


def ece_plugin_15(conf, correct) -> float:
    """15-bin equal-width plug-in (package/Kim-comparable). Biased; comparability only."""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, 16)
    groups = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, 14)
    return grouped_ece(conf, correct, groups)


def ece_debiased_l2(conf, correct, n_bins: int = 10) -> float:
    """Debiased l2-ECE (Kumar-Liang-Ma 2019) on equal-mass bins; sqrt(max(0, .)).

    Subtracts each bin's accuracy sampling variance from the squared gap, removing the
    n-dependent upward bias of the plug-in — the estimator to use when comparing cells
    with different n_questions. Degenerate for discrete signals (use ece_discrete).
    """
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return np.nan
    groups = bin_index(conf, equal_mass_edges(conf, n_bins))
    n_g = np.bincount(groups).astype(float)
    keep = n_g > 1  # bias term needs n_b - 1
    conf_g = np.bincount(groups, weights=conf)[keep] / n_g[keep]
    acc_g = np.bincount(groups, weights=correct)[keep] / n_g[keep]
    n_g = n_g[keep]
    sq = (conf_g - acc_g) ** 2 - acc_g * (1 - acc_g) / (n_g - 1)
    val = np.sum(n_g / n_g.sum() * sq)
    return float(np.sqrt(max(0.0, val)))


def brier_decomposition(conf, correct, n_bins: int = 10) -> dict:
    """Brier score + Murphy decomposition on equal-mass bins.

    brier ~= reliability - resolution + uncertainty (up to within-bin confidence
    variance). `reliability` (lower = better calibrated) is the Brier-based co-primary
    calibration outcome; it is a mean of iid item terms, so CLT-valid, no binning
    pathology in the score itself.
    """
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return {"brier": np.nan, "reliability": np.nan, "resolution": np.nan,
                "uncertainty": np.nan}
    brier = float(np.mean((conf - correct) ** 2))
    groups = bin_index(conf, equal_mass_edges(conf, n_bins))
    n_g = np.bincount(groups).astype(float)
    keep = n_g > 0
    conf_g = np.bincount(groups, weights=conf)[keep] / n_g[keep]
    acc_g = np.bincount(groups, weights=correct)[keep] / n_g[keep]
    n_g = n_g[keep]
    base = correct.mean()
    rel = float(np.sum(n_g / n_g.sum() * (conf_g - acc_g) ** 2))
    res = float(np.sum(n_g / n_g.sum() * (acc_g - base) ** 2))
    unc = float(base * (1 - base))
    return {"brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc}


def signed_gap(conf, correct) -> float:
    """mean(confidence) - accuracy. Positive = overconfident. ECE hides this direction."""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return np.nan
    return float(conf.mean() - correct.mean())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def cox_recalibration(conf, correct) -> dict:
    """Cox (1958) recalibration: logistic of correct on logit(conf).

    intercept != 0 -> systematic over/under-confidence; slope < 1 -> confidence is
    overdispersed. Smooth alternatives to ECE that regress cleanly on design axes.
    """
    conf, correct, _ = _clean(conf, correct)
    if len(conf) < 10 or len(np.unique(correct)) < 2:
        return {"cox_intercept": np.nan, "cox_slope": np.nan}
    from sklearn.linear_model import LogisticRegression

    x = _logit(conf).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, correct)
    except Exception:
        return {"cox_intercept": np.nan, "cox_slope": np.nan}
    return {"cox_intercept": float(m.intercept_[0]), "cox_slope": float(m.coef_[0][0])}


def auroc(conf, correct) -> float:
    """Discrimination: can the signal rank correct above incorrect answers at all?"""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0 or len(np.unique(correct)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(correct, conf))


def fit_temperature(conf_train, correct_train) -> float:
    """Fit a scalar temperature on the logit of the confidence, minimizing NLL."""
    conf, correct, _ = _clean(conf_train, correct_train)
    if len(conf) < 20 or len(np.unique(correct)) < 2:
        return np.nan
    from scipy.optimize import minimize_scalar

    z = _logit(conf)

    def nll(log_t):
        p = 1 / (1 + np.exp(-z / np.exp(log_t)))
        p = np.clip(p, _EPS, 1 - _EPS)
        return -np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(conf, t: float) -> np.ndarray:
    if not np.isfinite(t):
        return np.asarray(conf, float)
    return 1 / (1 + np.exp(-_logit(np.asarray(conf, float)) / t))


def cell_calibration(conf, correct, signal: str) -> dict:
    """Full per-cell calibration summary under the pre-registered estimator policy."""
    conf, correct, _ = _clean(conf, correct)
    out = {"n": int(len(conf))}
    if len(conf) == 0:
        return out | {"ece": np.nan, "ece_db": np.nan, "ece_plugin15": np.nan,
                      "signed_gap": np.nan, "brier": np.nan, "brier_rel": np.nan,
                      "cox_intercept": np.nan, "cox_slope": np.nan}
    out["ece"] = ece_primary(conf, correct, signal)
    out["ece_db"] = (np.nan if signal in DISCRETE_SIGNALS
                     else ece_debiased_l2(conf, correct))
    out["ece_plugin15"] = ece_plugin_15(conf, correct)
    out["signed_gap"] = signed_gap(conf, correct)
    bd = brier_decomposition(conf, correct)
    out["brier"], out["brier_rel"] = bd["brier"], bd["reliability"]
    out |= cox_recalibration(conf, correct)
    return out
