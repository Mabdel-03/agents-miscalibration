"""Scaling-law fits over the tidy per-cell table.

Provides:
  * ``power_law_fit`` — fit y = a * x^b (+ c) via log-log least squares, for e.g. accuracy
    or ECE vs parameter count.
  * ``kim_regression`` — a standardized OLS in the spirit of Kim et al. Eq. 1, regressing
    a system metric on capacity (+ capacity^2), context level, prompt complexity, and
    topology indicators, with cross-validated R^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def power_law_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit log10(y) = b*log10(x) + log10(a). Returns a, b, and R^2 (drops nonpositive)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": int(mask.sum())}
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    b, log_a = np.polyfit(lx, ly, 1)
    pred = b * lx + log_a
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(10**log_a), "b": float(b), "r2": r2, "n": int(mask.sum())}


def kim_regression(df: pd.DataFrame, target: str = "accuracy") -> dict:
    """OLS: target ~ log(params) + log(params)^2 + context_rank + prompt_level + topology dummies.

    Returns coefficients, R^2, and 5-fold CV R^2. Standardizes continuous predictors.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=[target]).copy()
    if len(d) < 10:
        return {"error": f"too few rows ({len(d)}) to fit"}

    ctx_rank = {"artifact_only": 0, "plus_intermediate": 1, "plus_cot": 2}
    d["log_params"] = np.log1p(d["param_count"])
    d["log_params_sq"] = d["log_params"] ** 2
    d["ctx_rank"] = d["context_share_level"].map(ctx_rank).fillna(0)

    cont = StandardScaler().fit_transform(
        d[["log_params", "log_params_sq", "ctx_rank", "prompt_complexity_level"]]
    )
    topo_dummies = pd.get_dummies(d["topology"], prefix="topo", drop_first=True).to_numpy(float)
    X = np.hstack([cont, topo_dummies])
    y = d[target].to_numpy(float)

    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    cv = cross_val_score(LinearRegression(), X, y, cv=min(5, len(d) // 2), scoring="r2")
    names = ["log_params", "log_params_sq", "ctx_rank", "prompt_complexity_level"] + [
        c for c in pd.get_dummies(d["topology"], prefix="topo", drop_first=True).columns
    ]
    return {
        "target": target,
        "coefficients": dict(zip(names, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "r2": r2,
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "n": len(d),
    }
