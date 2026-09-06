"""Family C — system confidence recalibrators and calibration diagnostics (N3; spec §8.7, §9.2, §9.5).

* Common logistic recalibrator: intercept + clipped probability logit + missingness flag.
* Architecture-conditioned recalibrator: the same plus fixed per-method intercepts.  It is
  the primary map iff every method has at least :data:`MIN_DEV_OUTCOMES` development
  successes and failures; otherwise the common fit is primary and ``threshold_failure`` is
  reported (§8.7).
* Both are fitted on development only, separately per target scope, with source-item
  grouped folds over the common penalty grid and the one-SE rule (readout.PENALTY_GRID).
* Missing confidence / invalid forecasts use the development marginal prior of the scope
  plus the missingness flag (§8.7 "development-fitted scope-specific marginal prior").
* Diagnostics: Brier (source-item weighted, 0.5/0.5 superdomains), five development-fixed
  reliability bins with source-item-weighted counts, signed bias ``E[q - y]``, Cox
  intercept/slope, missingness, Murphy decomposition (``analysis/nb_lib/calib.py`` is reused
  when importable; the backend is recorded).
* Contrast C = mean(Brier(recalibrated selected PERSONAL_FINAL) - Brier(recalibrated explicit
  TEAM_SELECTED)) with the same §9.3 source-cluster bootstrap as family G.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents_scaling.study.neural import readout as R

MIN_DEV_OUTCOMES = 10
N_BINS = 5
SCOPE_PERSONAL = "PERSONAL_FINAL"
SCOPE_TEAM = "TEAM_SELECTED"
SCOPES: tuple[str, ...] = (SCOPE_PERSONAL, SCOPE_TEAM)
C_FROZEN_JSON = "C_frozen.json"
CLIP = 1e-3
_EPS = 1e-6


class CalibrationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- nb_lib reuse


def _nb_calib() -> Any | None:
    """``analysis/nb_lib/calib.py`` when importable (added to ``sys.path`` from the repo)."""
    try:
        from nb_lib import calib  # type: ignore

        return calib
    except Exception:
        pass
    try:
        import agents_scaling

        analysis_dir = Path(agents_scaling.__file__).resolve().parents[2] / "analysis"
        if analysis_dir.is_dir() and str(analysis_dir) not in sys.path:
            sys.path.append(str(analysis_dir))
        from nb_lib import calib  # type: ignore

        return calib
    except Exception:
        return None


# --------------------------------------------------------------------------- helpers


def clipped_logit(q: np.ndarray, clip: float = CLIP) -> np.ndarray:
    p = np.clip(np.asarray(q, dtype=np.float64), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def fill_missing(q: Sequence[Any], prior: float) -> tuple[np.ndarray, np.ndarray]:
    """``(q_filled, missing_flag)``: None/NaN/out-of-range → the development prior + flag."""
    out = np.empty(len(q), dtype=np.float64)
    flag = np.zeros(len(q), dtype=np.float64)
    for i, v in enumerate(q):
        try:
            x = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            x = float("nan")
        if not math.isfinite(x) or x < 0.0 or x > 1.0:
            out[i], flag[i] = prior, 1.0
        else:
            out[i] = x
    return out, flag


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    return float(np.sum(w * v) / np.sum(w)) if np.sum(w) > 0 else float("nan")


def threshold_rule(y: Sequence[int], methods: Sequence[str], *, minimum: int = MIN_DEV_OUTCOMES) -> tuple[bool, dict[str, dict[str, int]]]:
    """Every method has ≥ ``minimum`` development successes and failures?"""
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray([str(v) for v in methods])
    counts: dict[str, dict[str, int]] = {}
    ok = True
    for name in sorted(set(m)):
        s = int(np.sum(y[m == name] == 1))
        f = int(np.sum(y[m == name] == 0))
        counts[name] = {"successes": s, "failures": f}
        ok = ok and s >= minimum and f >= minimum
    return bool(ok and len(counts) > 0), counts


# --------------------------------------------------------------------------- recalibrator


def _design(q_filled: np.ndarray, flag: np.ndarray, methods: Sequence[str] | None, method_vocab: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    cols = [clipped_logit(q_filled), np.asarray(flag, dtype=np.float64)]
    names = ["logit_q", "missing"]
    if method_vocab:
        m = [str(v) for v in (methods or [])]
        for name in method_vocab:
            cols.append(np.asarray([1.0 if v == name else 0.0 for v in m], dtype=np.float64))
            names.append(f"method={name}")
    return np.column_stack(cols), names


def _fit_logistic(X: np.ndarray, y: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        p = float(np.clip((y.sum() + 0.5) / (len(y) + 1.0), _EPS, 1 - _EPS))
        return np.zeros(X.shape[1]), math.log(p / (1 - p))
    clf = LogisticRegression(C=1.0 / float(penalty), solver="lbfgs", max_iter=5000, tol=1e-8)  # L2 (sklearn default; `penalty=` is deprecated in 1.8)
    clf.fit(X, y)
    return np.asarray(clf.coef_[0], dtype=np.float64), float(clf.intercept_[0])


@dataclass
class Recalibrator:
    """One fitted logistic recalibration map (common or architecture-conditioned)."""

    scope: str
    conditioned: bool
    prior: float
    method_vocab: tuple[str, ...] = ()
    penalty: float = 1.0
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    penalty_search: dict[str, Any] = field(default_factory=dict)

    def design(self, q: Sequence[Any], methods: Sequence[str] | None) -> np.ndarray:
        q_filled, flag = fill_missing(q, self.prior)
        X, names = _design(q_filled, flag, methods, self.method_vocab if self.conditioned else None)
        self.feature_names = names
        return X

    def fit(self, q: Sequence[Any], y: Sequence[int], methods: Sequence[str] | None, *, penalty: float) -> "Recalibrator":
        X = self.design(q, methods)
        self.coef_, self.intercept_ = _fit_logistic(X, np.asarray(y, dtype=np.int64), penalty)
        self.penalty = float(penalty)
        return self

    def apply(self, q: Sequence[Any], methods: Sequence[str] | None = None) -> np.ndarray:
        if self.coef_ is None:
            raise CalibrationError("recalibrator is not fitted")
        X = self.design(q, methods)
        z = X @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    @property
    def slope(self) -> float:
        return float(self.coef_[0]) if self.coef_ is not None else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "conditioned": self.conditioned, "prior": self.prior, "method_vocab": list(self.method_vocab),
                "penalty": self.penalty, "coef": None if self.coef_ is None else self.coef_.tolist(), "intercept": self.intercept_,
                "feature_names": list(self.feature_names), "penalty_search": dict(self.penalty_search)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Recalibrator":
        r = cls(scope=str(data["scope"]), conditioned=bool(data["conditioned"]), prior=float(data["prior"]),
                method_vocab=tuple(data.get("method_vocab") or ()), penalty=float(data["penalty"]), intercept_=float(data["intercept"]),
                feature_names=list(data.get("feature_names") or []), penalty_search=dict(data.get("penalty_search") or {}))
        r.coef_ = None if data.get("coef") is None else np.asarray(data["coef"], dtype=np.float64)
        return r


def select_penalty(
    q: Sequence[Any], y: Sequence[int], methods: Sequence[str] | None, clusters: Sequence[str], superdomains: Sequence[str],
    *, scope: str, conditioned: bool, prior: float, method_vocab: Sequence[str], seed: bytes,
    penalties: Sequence[float] = R.PENALTY_GRID, n_folds: int = R.N_INNER_FOLDS,
) -> tuple[float, dict[str, Any]]:
    """Source-item grouped CV over the common penalty grid; one-SE rule (stronger penalty)."""
    y_arr = np.asarray(y, dtype=np.int64)
    folds = R.grouped_stratified_folds(clusters, superdomains, n_folds, seed, namespace=f"C_FOLDS:{scope}:{int(conditioned)}")
    per: dict[float, list[float]] = {float(lam): [] for lam in penalties}
    q_list = list(q)
    m_list = None if methods is None else [str(v) for v in methods]
    for j in range(n_folds):
        tr, va = np.flatnonzero(folds != j), np.flatnonzero(folds == j)
        if len(va) == 0 or len(tr) < 3:
            continue
        for lam in penalties:
            r = Recalibrator(scope, conditioned, prior, tuple(method_vocab))
            r.fit([q_list[i] for i in tr], y_arr[tr], None if m_list is None else [m_list[i] for i in tr], penalty=lam)
            p = r.apply([q_list[i] for i in va], None if m_list is None else [m_list[i] for i in va])
            per[float(lam)].append(float(R.brier(p, y_arr[va]).mean()))
    cands = [((None, None, lam), *R._mean_se(v)) for lam, v in per.items() if v]
    sel = R.one_se_select(cands, block_order=())
    return float(sel.penalty), {"grid": {str(k): v for k, v in per.items()}, "selected": sel.penalty, "inner_mean": sel.inner_mean, "inner_se": sel.inner_se}


@dataclass
class ScopeCalibration:
    """Both recalibrators of one scope plus the frozen primary choice."""

    scope: str
    prior: float
    common: Recalibrator
    conditioned: Recalibrator | None
    primary: str  # "conditioned" | "common"
    threshold_failure: bool
    outcome_counts: dict[str, dict[str, int]]
    bin_edges: tuple[float, ...]
    n_dev: int
    missing_rate_dev: float

    @property
    def primary_map(self) -> Recalibrator:
        return self.conditioned if (self.primary == "conditioned" and self.conditioned is not None) else self.common

    def apply(self, q: Sequence[Any], methods: Sequence[str] | None, *, which: str = "primary") -> np.ndarray:
        if which == "primary":
            return self.primary_map.apply(q, methods)
        if which == "common":
            return self.common.apply(q, methods)
        if which == "conditioned":
            if self.conditioned is None:
                raise CalibrationError("no conditioned recalibrator was fitted")
            return self.conditioned.apply(q, methods)
        raise ValueError(which)

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "prior": self.prior, "common": self.common.to_dict(),
                "conditioned": None if self.conditioned is None else self.conditioned.to_dict(), "primary": self.primary,
                "threshold_failure": self.threshold_failure, "outcome_counts": self.outcome_counts, "bin_edges": list(self.bin_edges),
                "n_dev": self.n_dev, "missing_rate_dev": self.missing_rate_dev}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScopeCalibration":
        return cls(scope=str(data["scope"]), prior=float(data["prior"]), common=Recalibrator.from_dict(data["common"]),
                   conditioned=None if data.get("conditioned") is None else Recalibrator.from_dict(data["conditioned"]),
                   primary=str(data["primary"]), threshold_failure=bool(data["threshold_failure"]),
                   outcome_counts={k: dict(v) for k, v in data["outcome_counts"].items()}, bin_edges=tuple(float(v) for v in data["bin_edges"]),
                   n_dev=int(data["n_dev"]), missing_rate_dev=float(data["missing_rate_dev"]))


def dev_bin_edges(q: Sequence[Any], *, n_bins: int = N_BINS) -> tuple[float, ...]:
    """Five development-fixed reliability bins: interior quantile boundaries of the
    observed (non-missing) development probabilities, deduplicated; outer edges 0 and 1."""
    vals = np.asarray([float(v) for v in q if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        inner = np.linspace(0, 1, n_bins + 1)[1:-1]
    else:
        inner = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(np.concatenate([[0.0], np.clip(inner, 0.0, 1.0), [1.0]]))
    return tuple(float(e) for e in edges)


def fit_scope(
    q: Sequence[Any], y: Sequence[int], methods: Sequence[str], clusters: Sequence[str], superdomains: Sequence[str],
    *, scope: str, seed: bytes, penalties: Sequence[float] = R.PENALTY_GRID, minimum: int = MIN_DEV_OUTCOMES,
) -> ScopeCalibration:
    """Fit the common and (when the threshold holds) conditioned recalibrators on development."""
    y_arr = np.asarray(y, dtype=np.int64)
    if len(y_arr) == 0:
        raise CalibrationError(f"{scope}: no development rows")
    w = R.item_weights(superdomains)
    prior = weighted_mean(y_arr, w)  # source-item weighted development marginal
    vocab = tuple(sorted({str(m) for m in methods}))
    _, flag = fill_missing(q, prior)
    lam_c, search_c = select_penalty(q, y_arr, None, clusters, superdomains, scope=scope, conditioned=False, prior=prior, method_vocab=vocab, seed=seed, penalties=penalties)
    common = Recalibrator(scope, False, prior, vocab).fit(q, y_arr, None, penalty=lam_c)
    common.penalty_search = search_c
    ok, counts = threshold_rule(y_arr, methods, minimum=minimum)
    conditioned = None
    if ok:
        lam_k, search_k = select_penalty(q, y_arr, methods, clusters, superdomains, scope=scope, conditioned=True, prior=prior, method_vocab=vocab, seed=seed, penalties=penalties)
        conditioned = Recalibrator(scope, True, prior, vocab).fit(q, y_arr, methods, penalty=lam_k)
        conditioned.penalty_search = search_k
    return ScopeCalibration(scope=scope, prior=float(prior), common=common, conditioned=conditioned, primary="conditioned" if ok else "common",
                            threshold_failure=not ok, outcome_counts=counts, bin_edges=dev_bin_edges(q), n_dev=int(len(y_arr)),
                            missing_rate_dev=float(flag.mean()))


# --------------------------------------------------------------------------- diagnostics


def reliability_bins(q: np.ndarray, y: np.ndarray, weights: np.ndarray, edges: Sequence[float]) -> list[dict[str, Any]]:
    """Weighted reliability table on fixed edges (source-item weights, §9.5)."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    idx = np.clip(np.searchsorted(e, q, side="right") - 1, 0, len(e) - 2)
    out = []
    for b in range(len(e) - 1):
        m = idx == b
        wb = float(w[m].sum())
        out.append({
            "bin": b, "low": float(e[b]), "high": float(e[b + 1]), "n": int(m.sum()), "weight": wb,
            "mean_confidence": weighted_mean(q[m], w[m]) if wb > 0 else float("nan"),
            "mean_outcome": weighted_mean(y[m], w[m]) if wb > 0 else float("nan"),
            "gap": (weighted_mean(q[m], w[m]) - weighted_mean(y[m], w[m])) if wb > 0 else float("nan"),
        })
    return out


def cox_intercept_slope(q: np.ndarray, y: np.ndarray) -> dict[str, float]:
    calib = _nb_calib()
    if calib is not None:
        out = calib.cox_recalibration(q, y)
        return {"cox_intercept": float(out["cox_intercept"]), "cox_slope": float(out["cox_slope"]), "backend": "nb_lib.calib"}
    y = np.asarray(y, dtype=np.int64)
    if len(q) < 10 or len(np.unique(y)) < 2:
        return {"cox_intercept": float("nan"), "cox_slope": float("nan"), "backend": "neural.calibration"}
    coef, intercept = _fit_logistic(clipped_logit(np.asarray(q), 1e-6).reshape(-1, 1), y, 1e-6)
    return {"cox_intercept": float(intercept), "cox_slope": float(coef[0]), "backend": "neural.calibration"}


def calibration_diagnostics(q: np.ndarray, y: np.ndarray, *, weights: np.ndarray, missing: np.ndarray | None, edges: Sequence[float]) -> dict[str, Any]:
    """Brier, reliability bins, signed bias, Cox, missingness, discrimination, decomposition."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    out: dict[str, Any] = {
        "n": int(len(q)),
        "brier_weighted": weighted_mean((q - y) ** 2, w),
        "brier_unweighted": float(np.mean((q - y) ** 2)) if len(q) else float("nan"),
        "signed_bias_weighted": weighted_mean(q - y, w),
        "log_loss_clipped": float(-np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1)))) if len(q) else float("nan"),
        "missing_rate": float(np.mean(missing)) if missing is not None and len(missing) else 0.0,
        "reliability_bins": reliability_bins(q, y, w, edges),
        "bin_edges": [float(e) for e in edges],
        **cox_intercept_slope(q, y),
    }
    calib = _nb_calib()
    if calib is not None and len(q) >= 2:
        out["signed_gap_unweighted"] = float(calib.signed_gap(q, y))
        out["decomposition"] = {k: (None if (isinstance(v, float) and math.isnan(v)) else float(v)) for k, v in calib.brier_decomposition(q, y, n_bins=N_BINS).items()}
        try:
            out["auroc"] = float(calib.auroc(q, y))
        except Exception:
            out["auroc"] = None
        out["diagnostics_backend"] = "nb_lib.calib"
    else:
        out["diagnostics_backend"] = "neural.calibration"
    return out


# --------------------------------------------------------------------------- the C pipeline


@dataclass
class FrozenC:
    scopes: dict[str, ScopeCalibration]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | os.PathLike) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "kind": "C_FROZEN", "scopes": {k: v.to_dict() for k, v in self.scopes.items()}, "meta": dict(self.meta)}
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        path = directory / C_FROZEN_JSON
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "FrozenC":
        path = Path(directory) / C_FROZEN_JSON
        if not path.is_file():
            raise CalibrationError(f"{path} is missing (fit the recalibrators on development first)")
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("sha256", None)
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != claimed:
            raise CalibrationError(f"{path}: sha256 mismatch (frozen artifact was modified)")
        return cls(scopes={k: ScopeCalibration.from_dict(v) for k, v in payload["scopes"].items()}, meta=dict(payload.get("meta") or {}))


def scope_inputs(rows: Any, scope: str) -> list[Any]:
    """The raw probability of a scope per row: selected PERSONAL_FINAL confidence or the
    explicit TEAM_SELECTED ``q_team_now`` (None when missing / invalid)."""
    if scope == SCOPE_PERSONAL:
        return [None if bool(m) else v for v, m in zip(rows["personal_conf"].tolist(), rows["personal_missing"].tolist())]
    if scope == SCOPE_TEAM:
        return [v if bool(ok) else None for v, ok in zip(rows["q_team_now"].tolist(), rows["forecast_valid"].tolist())]
    raise ValueError(scope)


def fit_recalibrators(dev_rows: Any, *, seed: bytes, penalties: Sequence[float] = R.PENALTY_GRID, minimum: int = MIN_DEV_OUTCOMES, meta: Mapping[str, Any] | None = None) -> FrozenC:
    """Development-only fit of both scopes' recalibrators (§8.7, §9.2 C)."""
    y = dev_rows["y"].to_numpy(dtype=np.int64)
    methods = [str(m) for m in dev_rows["method"].tolist()]
    clusters = [str(c) for c in dev_rows["cluster"].tolist()]
    sds = [str(s) for s in dev_rows["superdomain"].tolist()]
    scopes = {scope: fit_scope(scope_inputs(dev_rows, scope), y, methods, clusters, sds, scope=scope, seed=seed, penalties=penalties, minimum=minimum) for scope in SCOPES}
    info = {"n_dev_rows": int(len(dev_rows)), "n_dev_items": int(dev_rows["source_id"].nunique()), "penalties": list(penalties), "minimum_outcomes": minimum, **dict(meta or {})}
    return FrozenC(scopes=scopes, meta=info)


def score_rows(rows: Any, frozen: FrozenC) -> Any:
    """Per-row raw and recalibrated probabilities and Brier losses for both scopes."""
    out = rows[["source_id", "method", "superdomain", "cluster", "report_id", "selection_id", "y"]].copy()
    methods = [str(m) for m in rows["method"].tolist()]
    y = rows["y"].to_numpy(dtype=np.float64)
    for scope, col in ((SCOPE_PERSONAL, "baseline"), (SCOPE_TEAM, "explicit")):
        cal = frozen.scopes[scope]
        raw = scope_inputs(rows, scope)
        filled, flag = fill_missing(raw, cal.prior)
        out[f"q_raw_{col}"] = filled
        out[f"missing_{col}"] = flag
        out[f"q_{col}"] = cal.apply(raw, methods)
        out[f"q_common_{col}"] = cal.apply(raw, methods, which="common")
        out[f"brier_raw_{col}"] = R.brier(filled, y)
        out[f"brier_{col}"] = R.brier(out[f"q_{col}"].to_numpy(), y)
        out[f"brier_common_{col}"] = R.brier(out[f"q_common_{col}"].to_numpy(), y)
    return out


def contrast_C(scored: Any, *, seed: int, n_resamples: int = R.DEFAULT_BOOTSTRAP, frozen: FrozenC | None = None) -> dict[str, Any]:
    """Primary C: source-item-weighted mean Brier(baseline) - Brier(explicit), positive =
    the explicit TEAM_SELECTED forecast is the better proper score; same bootstrap as G."""
    coverage = R.method_coverage(scored)
    items = R.per_item_losses(scored, ["brier_baseline", "brier_explicit", "brier_raw_baseline", "brier_raw_explicit", "brier_common_baseline", "brier_common_explicit"])
    diff = (items["brier_baseline"] - items["brier_explicit"]).to_numpy(dtype=np.float64)
    boot = R.paired_cluster_bootstrap(diff, items["cluster"].tolist(), items["superdomain"].tolist(), n_resamples=n_resamples, seed=seed, alternative="greater")
    raw_diff = (items["brier_raw_baseline"] - items["brier_raw_explicit"]).to_numpy(dtype=np.float64)
    raw_boot = R.paired_cluster_bootstrap(raw_diff, items["cluster"].tolist(), items["superdomain"].tolist(), n_resamples=min(n_resamples, 2000), seed=seed + 1, alternative="greater")
    w = R.item_weights(items["superdomain"].tolist())
    weights_rows = R.item_weights(scored["superdomain"].tolist()) * 1.0
    out: dict[str, Any] = {
        "contrast": "brier_baseline_minus_explicit", "bootstrap": boot.to_dict(), "raw_bootstrap": raw_boot.to_dict(),
        "n_items": int(len(items)), "n_rows": int(len(scored)),
        "n_items_complete": coverage["n_items_complete"], "n_items_dropped": coverage["n_items_dropped"], "dropped_items": coverage["dropped_items"], "methods": coverage["methods"],
        "weighted_brier": {c: float(np.sum(w * items[f"brier_{c}"].to_numpy())) for c in ("baseline", "explicit", "raw_baseline", "raw_explicit", "common_baseline", "common_explicit")},
        "by_method": {str(m): {"contrast": float((sub["brier_baseline"] - sub["brier_explicit"]).mean()), "n": int(len(sub))} for m, sub in scored.groupby("method")},
        "diagnostics": {},
    }
    y = scored["y"].to_numpy(dtype=np.float64)
    for col, scope in (("baseline", SCOPE_PERSONAL), ("explicit", SCOPE_TEAM)):
        edges = frozen.scopes[scope].bin_edges if frozen is not None else dev_bin_edges(scored[f"q_raw_{col}"].tolist())
        out["diagnostics"][col] = {
            "recalibrated": calibration_diagnostics(scored[f"q_{col}"].to_numpy(), y, weights=weights_rows, missing=scored[f"missing_{col}"].to_numpy(), edges=edges),
            "raw": calibration_diagnostics(scored[f"q_raw_{col}"].to_numpy(), y, weights=weights_rows, missing=scored[f"missing_{col}"].to_numpy(), edges=edges),
        }
        for m in sorted(scored["method"].unique()):
            sub = scored[scored["method"] == m]
            out["diagnostics"][col][f"by_method:{m}"] = {
                "signed_bias": float(np.mean(sub[f"q_{col}"].to_numpy() - sub["y"].to_numpy(dtype=np.float64))),
                "brier": float(sub[f"brier_{col}"].mean()), "missing_rate": float(sub[f"missing_{col}"].mean()), "n": int(len(sub)),
            }
    return out


__all__ = [
    "C_FROZEN_JSON",
    "MIN_DEV_OUTCOMES",
    "N_BINS",
    "SCOPES",
    "SCOPE_PERSONAL",
    "SCOPE_TEAM",
    "CalibrationError",
    "FrozenC",
    "Recalibrator",
    "ScopeCalibration",
    "calibration_diagnostics",
    "clipped_logit",
    "contrast_C",
    "cox_intercept_slope",
    "dev_bin_edges",
    "fill_missing",
    "fit_recalibrators",
    "fit_scope",
    "reliability_bins",
    "scope_inputs",
    "score_rows",
    "select_penalty",
    "threshold_rule",
    "weighted_mean",
]
