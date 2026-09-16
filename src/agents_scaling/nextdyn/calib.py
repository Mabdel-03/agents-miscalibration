"""Calibration estimators (spec 13.1; plan §1.11 ``VENDORED``, §4.3).

The six pure-numpy estimators of ``analysis/nb_lib/calib.py`` are **vendored** here rather than
imported.  ``analysis/`` is a notebook library outside the installed package: ``ard`` reaches it
through a ``sys.path`` bootstrap (``ard/calibration_panel.py:105-112``) that works from the repo
checkout and from nowhere else, and a confirmatory analysis may not depend on an import that
resolves differently on a compute node than on a login node.  Each vendored function carries a
parity test against the original in ``tests/nextdyn/test_calib.py``.

Two of the seven are *not* pure numpy upstream and are reimplemented here so the whole module
is dependency-free:

* :func:`auroc` upstream calls ``sklearn.metrics.roc_auc_score``; here it is the rank statistic
  with explicit half-credit for ties, which matters because confidence on this family of models
  lives on a handful of atoms and the tie fraction is most of the signal;
* :func:`cox_recalibration` upstream fits ``sklearn.linear_model.LogisticRegression(C=1e6)``;
  here it is Newton-Raphson IRLS on the same design (``correct ~ logit(conf)``), which converges
  to the same unpenalised maximum-likelihood fit.

Both are parity-tested against their upstream versions, and the parity test skips (rather than
silently passing) when sklearn is absent.

Two rules travel with every number here, and they are the same two that ARD36's calibration
panel violated:

* **Missing confidence is missing.**  A decision whose output carried no ``confidence`` is
  excluded and counted, never imputed to 0.5 (spec 18.3
  ``test_confidence_not_imputed_as_model_statement``).  :func:`confidence_signal` is the only
  way rows become arrays here, and it refuses an imputed value.
* **Within-item AUC needs mixed groups.**  An item whose candidates are all correct or all wrong
  carries no ranking information; it is excluded and counted, so a panel row's AUC denominator
  can never equal its row count (``test_within_item_auc_requires_mixed_candidate_groups``).
  That computation is ``ard.calibration_panel.within_item_auc``, imported, together with
  ``tie_rate``: those two already carry the repair and are on the ``ARD_IMPORTS`` allowlist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from agents_scaling.ard.calibration_panel import tie_rate, within_item_auc  # noqa: F401
from agents_scaling.nextdyn.errors import ContractViolation

#: Signals that are discrete **by construction** (finite atoms), so the exact-atom estimator
#: applies and the debiased-L2 variance correction is degenerate by design, not by failure.
DISCRETE_SIGNALS: Final[frozenset[str]] = frozenset({"vote_fraction", "system_vote_fraction"})
CONTINUOUS: Final[str] = "continuous"
DISCRETE: Final[str] = "discrete"
DEFAULT_BINS: Final[int] = 10
_EPS: Final[float] = 1e-6

#: The value a missing confidence would be imputed to if anyone did it.  It exists only so
#: :func:`confidence_signal` can refuse it by name.
FORBIDDEN_IMPUTATION: Final[float] = 0.5


def dev_bin_edges(profile: Any) -> tuple[float, ...] | None:
    """The frozen dev reliability bins of a compiled ``config.AnalysisProfile``.

    ``analysis.profiles.*.reliability_bins`` is null until the pilot fixes the edges; this is
    the one reader of that leaf, and it returns ``None`` rather than choosing a bin count, so a
    fixed-bin ECE is reported only against edges a human froze (spec 13.1).
    """
    edges = profile.reliability_bins
    return None if edges is None else tuple(float(e) for e in edges)


def is_discrete(signal: str, signal_type: str | None = None) -> bool:
    """Whether the exact-atom estimator applies; an explicit ``signal_type`` wins."""
    if signal_type is None:
        return signal in DISCRETE_SIGNALS
    if signal_type not in (CONTINUOUS, DISCRETE):
        raise ContractViolation(f"signal_type must be {CONTINUOUS!r} or {DISCRETE!r}, got {signal_type!r}")
    return signal_type == DISCRETE


def _clean(conf, correct, weights=None):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(conf) & np.isfinite(correct)
    if weights is None:
        return conf[ok], correct[ok], None
    weights = np.asarray(weights, float)
    if len(weights) != len(conf):
        raise ContractViolation("weights must align with conf/correct BEFORE NaN filtering")
    return conf[ok], correct[ok], weights[ok]


# --------------------------------------------------------------------------- missingness


@dataclass(frozen=True)
class ConfidenceSignal:
    """Aligned ``(conf, correct)`` arrays plus the missingness that produced them."""

    conf: np.ndarray
    correct: np.ndarray
    n_total: int
    n_missing_conf: int
    n_missing_label: int

    @property
    def n(self) -> int:
        return int(self.conf.size)

    @property
    def missing_rate(self) -> float | None:
        return (self.n_missing_conf / self.n_total) if self.n_total else None


def confidence_signal(rows: Sequence[Mapping[str, Any]], *, signal: str = "confidence",
                      label: str = "correct") -> ConfidenceSignal:
    """Build the estimator's arrays from rows, **dropping** missing values and counting them.

    Imputation is refused rather than avoided by convention: a row that already carries an
    ``imputed``/``confidence_imputed`` marker raises, because a confidence the model never
    stated is not a model statement and must not enter a calibration estimate (spec 13.1).
    """
    conf: list[float] = []
    correct: list[float] = []
    n_missing_conf = n_missing_label = 0
    for row in rows:
        if row.get("confidence_imputed") or row.get("imputed"):
            raise ContractViolation(
                "a row carries an imputed confidence; missing confidence is missing, never "
                f"{FORBIDDEN_IMPUTATION} (spec 13.1)")
        value = row.get(signal)
        outcome = row.get(label)
        if value is None:
            n_missing_conf += 1
            continue
        if outcome is None:
            n_missing_label += 1
            continue
        conf.append(float(value))
        correct.append(float(outcome))
    return ConfidenceSignal(np.asarray(conf, dtype=float), np.asarray(correct, dtype=float),
                            len(rows), n_missing_conf, n_missing_label)


# --------------------------------------------------------------------------- vendored ECE


def equal_mass_edges(conf, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Quantile bin edges (deduplicated); fewer unique edges means effectively fewer bins."""
    conf = np.asarray(conf, float)
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def bin_index(conf, edges) -> np.ndarray:
    return np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)


def grouped_ece(conf, correct, groups, weights=None) -> float:
    """Weighted L1 ECE over an arbitrary grouping (bins or exact atoms)."""
    if weights is None:
        weights = np.ones_like(conf)
    w_g = np.bincount(groups, weights=weights)
    keep = w_g > 0
    conf_g = np.bincount(groups, weights=weights * conf)[keep] / w_g[keep]
    acc_g = np.bincount(groups, weights=weights * correct)[keep] / w_g[keep]
    w_g = w_g[keep]
    return float(np.sum(w_g / w_g.sum() * np.abs(conf_g - acc_g)))


def ece_equal_mass(conf, correct, n_bins: int = DEFAULT_BINS, weights=None) -> float:
    """Primary estimator for continuous signals: equal-mass (quantile) L1 ECE."""
    conf, correct, weights = _clean(conf, correct, weights)
    if len(conf) == 0:
        return float("nan")
    return grouped_ece(conf, correct, bin_index(conf, equal_mass_edges(conf, n_bins)), weights)


def ece_discrete(conf, correct, weights=None) -> float:
    """Primary estimator for discrete signals: exact ECE over the support atoms."""
    conf, correct, weights = _clean(conf, correct, weights)
    if len(conf) == 0:
        return float("nan")
    _, groups = np.unique(np.round(conf, 6), return_inverse=True)
    return grouped_ece(conf, correct, groups, weights)


def ece_fixed_bins(conf, correct, edges: Sequence[float] | None) -> float | None:
    """ECE on the **frozen dev bin edges** (``analysis.profiles.*.reliability_bins``).

    ``None`` edges mean the pilot has not fixed them yet; the estimator returns ``None`` rather
    than silently choosing a bin count, because a plug-in ECE is a property of its grid.
    """
    if edges is None:
        return None
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return float("nan")
    grid = np.asarray(list(edges), dtype=float)
    if grid.size < 2:
        raise ContractViolation("reliability_bins needs at least two edges")
    grid = np.concatenate(([-np.inf], grid[1:-1], [np.inf])) if grid.size > 2 else np.array([-np.inf, np.inf])
    return grouped_ece(conf, correct, bin_index(conf, grid))


def ece_debiased_l2(conf, correct, n_bins: int = DEFAULT_BINS) -> float:
    """Debiased l2-ECE (Kumar-Liang-Ma 2019) on equal-mass bins; ``sqrt(max(0, .))``."""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return float("nan")
    groups = bin_index(conf, equal_mass_edges(conf, n_bins))
    n_g = np.bincount(groups).astype(float)
    keep = n_g > 1
    conf_g = np.bincount(groups, weights=conf)[keep] / n_g[keep]
    acc_g = np.bincount(groups, weights=correct)[keep] / n_g[keep]
    n_g = n_g[keep]
    sq = (conf_g - acc_g) ** 2 - acc_g * (1 - acc_g) / (n_g - 1)
    return float(np.sqrt(max(0.0, float(np.sum(n_g / n_g.sum() * sq)))))


def brier_decomposition(conf, correct, n_bins: int = DEFAULT_BINS) -> dict[str, float]:
    """Brier score with the Murphy decomposition on equal-mass bins."""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}
    brier = float(np.mean((conf - correct) ** 2))
    groups = bin_index(conf, equal_mass_edges(conf, n_bins))
    n_g = np.bincount(groups).astype(float)
    keep = n_g > 0
    conf_g = np.bincount(groups, weights=conf)[keep] / n_g[keep]
    acc_g = np.bincount(groups, weights=correct)[keep] / n_g[keep]
    n_g = n_g[keep]
    base = float(correct.mean())
    return {
        "brier": brier,
        "reliability": float(np.sum(n_g / n_g.sum() * (conf_g - acc_g) ** 2)),
        "resolution": float(np.sum(n_g / n_g.sum() * (acc_g - base) ** 2)),
        "uncertainty": float(base * (1 - base)),
    }


def signed_gap(conf, correct) -> float:
    """``mean(confidence) - accuracy``.  Positive = overconfident; an ECE hides the direction."""
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0:
        return float("nan")
    return float(conf.mean() - correct.mean())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def cox_recalibration(conf, correct, *, max_iter: int = 100, tol: float = 1e-10) -> dict[str, float]:
    """Cox (1958) recalibration: unpenalised logistic of ``correct`` on ``logit(conf)``.

    ``intercept != 0`` is systematic over/under-confidence, ``slope < 1`` is overdispersion.
    Newton-Raphson IRLS on the two-column design, which is the unpenalised limit of the
    ``C = 1e6`` fit the upstream function performs with sklearn.
    """
    conf, correct, _ = _clean(conf, correct)
    nan = {"cox_intercept": float("nan"), "cox_slope": float("nan")}
    if len(conf) < 10 or len(np.unique(correct)) < 2:
        return nan
    design = np.column_stack([np.ones(conf.size), _logit(conf)])
    beta = np.zeros(2)
    for _ in range(int(max_iter)):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(mu * (1 - mu), 1e-12, None)
        gradient = design.T @ (correct - mu)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return nan
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return nan
        if float(np.max(np.abs(step))) < tol:
            break
    return {"cox_intercept": float(beta[0]), "cox_slope": float(beta[1])}


def auroc(conf, correct) -> float:
    """Discrimination: can the signal rank correct above incorrect answers at all?

    The Mann-Whitney rank statistic with explicit half-credit for ties.  Ties are the whole
    story for a self-reported confidence that lives on a handful of atoms, so they are counted
    at 0.5 rather than broken arbitrarily.
    """
    conf, correct, _ = _clean(conf, correct)
    if len(conf) == 0 or len(np.unique(correct)) < 2:
        return float("nan")
    positive = conf[correct > 0.5]
    negative = conf[correct <= 0.5]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = float(np.sum(positive[:, None] > negative[None, :]))
    equal = float(np.sum(positive[:, None] == negative[None, :]))
    return float((greater + 0.5 * equal) / (positive.size * negative.size))


def fit_temperature(conf_train, correct_train, *, bounds: tuple[float, float] = (0.05, 20.0),
                    steps: int = 400) -> float:
    """Scalar temperature on the logit of the confidence, minimising NLL on the dev split.

    Fitted on the **pilot** only and always reported with that label beside the recalibrated
    Brier, so a recalibrated number is never mistaken for the deployed one.
    """
    conf, correct, _ = _clean(conf_train, correct_train)
    if len(conf) < 20 or len(np.unique(correct)) < 2:
        return float("nan")
    z = _logit(conf)
    grid = np.exp(np.linspace(math.log(bounds[0]), math.log(bounds[1]), int(steps)))
    best_t, best_nll = float("nan"), math.inf
    for t in grid:
        p = np.clip(1.0 / (1.0 + np.exp(-z / t)), _EPS, 1 - _EPS)
        nll = float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(conf, t: float) -> np.ndarray:
    if not np.isfinite(t):
        return np.asarray(conf, float)
    return 1.0 / (1.0 + np.exp(-_logit(np.asarray(conf, float)) / t))


def cell_calibration(signal_rows: Sequence[Mapping[str, Any]], *, signal: str,
                     signal_type: str | None = None, reliability_bins: Sequence[float] | None = None,
                     n_bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Every estimator for one panel cell, with its missingness census attached.

    ``reliability_bins`` are the frozen dev bin edges from ``analysis.yaml``; when they are
    still null the fixed-bin ECE is ``None`` and the equal-mass estimator carries the row.
    """
    sig = confidence_signal(signal_rows, signal=signal)
    discrete = is_discrete(signal, signal_type)
    out: dict[str, Any] = {
        "signal": signal, "n": sig.n, "n_total": sig.n_total,
        "n_missing_conf": sig.n_missing_conf, "n_missing_label": sig.n_missing_label,
        "conf_missing_rate": sig.missing_rate, "signal_type": DISCRETE if discrete else CONTINUOUS,
    }
    if sig.n == 0:
        return out | {"ece": None, "ece_dev_bins": None, "ece_debiased_l2": None,
                      "signed_bias": None, "brier": None, "brier_reliability": None,
                      "brier_resolution": None, "cox_intercept": None, "cox_slope": None,
                      "auroc": None, "tie_rate": None}
    ece = ece_discrete(sig.conf, sig.correct) if discrete else ece_equal_mass(sig.conf, sig.correct, n_bins)
    decomposition = brier_decomposition(sig.conf, sig.correct, n_bins)
    ties = tie_rate(list(sig.conf))
    return out | {
        "ece": float(ece),
        "ece_dev_bins": ece_fixed_bins(sig.conf, sig.correct, reliability_bins),
        # The variance correction subtracts a per-bin term equal-mass bins cannot form on
        # discrete support, so it is None by design on the atom signals, not by failure.
        "ece_debiased_l2": None if discrete else float(ece_debiased_l2(sig.conf, sig.correct, n_bins)),
        "signed_bias": float(signed_gap(sig.conf, sig.correct)),
        "brier": decomposition["brier"],
        "brier_reliability": decomposition["reliability"],
        "brier_resolution": decomposition["resolution"],
        "auroc": float(auroc(sig.conf, sig.correct)),
        "tie_rate": ties["tie_rate"], "n_atoms": ties["n_atoms"],
        **cox_recalibration(sig.conf, sig.correct),
    }


__all__ = [
    "CONTINUOUS", "dev_bin_edges", "DEFAULT_BINS", "DISCRETE", "DISCRETE_SIGNALS", "ConfidenceSignal",
    "FORBIDDEN_IMPUTATION", "apply_temperature", "auroc", "bin_index", "brier_decomposition",
    "cell_calibration", "confidence_signal", "cox_recalibration", "ece_debiased_l2",
    "ece_discrete", "ece_equal_mass", "ece_fixed_bins", "equal_mass_edges", "fit_temperature",
    "grouped_ece", "is_discrete", "signed_gap", "tie_rate", "within_item_auc",
]
