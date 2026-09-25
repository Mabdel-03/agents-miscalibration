"""Penalized logistic regression with an optional Gauss-Hermite item random intercept (contract §5.7).

One kernel for every E2/E5 model (numpy/scipy only).

Model, for row i of problem (group) g:

    logit P(y_i = 1 | u_g) = b0 + x_i . beta + sigma * u_g,      u_g ~ N(0, 1)

* ``item_intercept=False`` is plain ridge logistic regression (sigma = 0).
* ``item_intercept=True`` integrates u_g out of the group likelihood with ``gh_nodes``-point Gauss-Hermite
  quadrature (probabilists' nodes), and estimates ``log sigma`` jointly with ``beta`` by L-BFGS-B.
* The ridge penalty ``ridge/2 * ||beta||^2`` applies to the standardized non-intercept coefficients; the intercept and
  sigma are unpenalized. Columns are standardized internally (constant columns are left alone and penalized as is);
  coefficients are reported on the original scale.
* ``y`` may be fractional in [0, 1] (quasi-Bernoulli), and rows may carry ``sample_weight``.
* ``bounds`` (per original column; positive scaling preserves sign constraints) lets a caller build the monotone
  depth variant: cumulative indicator columns with lower bound 0.

``cv_ridge`` chooses the ridge by K-fold CV whose folds are whole problems (groups), seeded; held-out problems are
scored with marginal predictions (their item effect is unknown), which is exactly the prediction a new problem gets.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import optimize, special

__all__ = [
    "GLMFit", "fit_glm", "cv_ridge", "grouped_folds", "gh_standard_normal", "log_loss", "brier",
]

_LOG_SIGMA_BOUNDS = (np.log(1e-4), np.log(20.0))


def gh_standard_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights (summing to 1) integrating against the standard normal density."""
    x, w = np.polynomial.hermite_e.hermegauss(int(n))
    return x, w / w.sum()


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _group_index(groups: Sequence) -> tuple[np.ndarray, list]:
    keys: dict = {}
    idx = np.empty(len(groups), dtype=np.int64)
    for i, g in enumerate(groups):
        idx[i] = keys.setdefault(g, len(keys))
    return idx, list(keys)


@dataclass
class GLMFit:
    """A fitted model. ``coef``/``intercept`` are on the original column scale."""

    coef: np.ndarray
    intercept: float
    sigma: float
    item_intercept: bool
    ridge: float
    gh_nodes: int
    columns: list[str] | None
    x_mean: np.ndarray
    x_scale: np.ndarray
    coef_std: np.ndarray                   # standardized-scale beta (optimizer space, without intercept)
    group_keys: list = field(default_factory=list)
    group_post_mean: np.ndarray | None = None   # posterior mean of b_g = sigma*u_g for training groups
    _post_grid: np.ndarray | None = None        # (G, M) posterior probabilities over _u_grid
    _u_grid: np.ndarray | None = None
    cov_std: np.ndarray | None = None           # Laplace covariance over [b0, beta_std, log sigma?]
    converged: bool = True
    n_iter: int = 0
    objective: float = float("nan")

    # ------------------------------------------------------------------ prediction
    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return self.intercept + X @ self.coef

    def predict_proba(self, X: np.ndarray, item_effect: np.ndarray | float | None = None) -> np.ndarray:
        """P(y=1). ``item_effect`` is b = sigma*u on the logit scale (scalar, per row, or (S, n) for S draws).

        With ``item_effect=None`` and a random intercept, returns the marginal probability over b ~ N(0, sigma^2)
        (the prediction for a new problem); without a random intercept it is the plain logistic prediction.
        """
        eta = self.linear_predictor(X)
        if item_effect is not None:
            return special.expit(eta + np.asarray(item_effect, float))
        if not self.item_intercept or self.sigma <= 0:
            return special.expit(eta)
        u, w = gh_standard_normal(max(self.gh_nodes, 20))
        return special.expit(eta[:, None] + self.sigma * u[None, :]) @ w

    def sample_item_effects(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` prior draws of b = sigma*u for a new problem (zeros without a random intercept)."""
        if not self.item_intercept:
            return np.zeros(int(n))
        return self.sigma * rng.standard_normal(int(n))

    def posterior_item_effects(self, group, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` posterior draws of b_g for a *training* group (grid posterior); prior draws for an unseen one."""
        if not self.item_intercept or self._post_grid is None:
            return np.zeros(int(n))
        try:
            gi = self.group_keys.index(group)
        except ValueError:
            return self.sample_item_effects(n, rng)
        probs = self._post_grid[gi]
        du = self._u_grid[1] - self._u_grid[0]
        pick = rng.choice(len(self._u_grid), size=int(n), p=probs)
        return self.sigma * (self._u_grid[pick] + rng.uniform(-du / 2, du / 2, size=int(n)))

    def sample_params(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Laplace draws of (intercept (n,), coef (n,p), sigma (n,)) on the original scale."""
        p = len(self.coef)
        theta0 = np.concatenate([[self._b0_std()], self.coef_std])
        k = len(theta0)
        if self.cov_std is None:
            draws = np.tile(theta0, (int(n), 1))
            sig = np.full(int(n), self.sigma)
        else:
            full0 = np.concatenate([theta0, [np.log(max(self.sigma, 1e-4))]]) if self.cov_std.shape[0] == k + 1 \
                else theta0
            d = rng.multivariate_normal(full0, self.cov_std, size=int(n), method="eigh")
            draws = d[:, :k]
            sig = np.exp(d[:, k]) if d.shape[1] == k + 1 else np.full(int(n), self.sigma)
        beta_std = draws[:, 1:]
        coef = beta_std / self.x_scale[None, :]
        intercept = draws[:, 0] - (beta_std * (self.x_mean / self.x_scale)[None, :]).sum(1)
        assert coef.shape[1] == p
        return intercept, coef, sig

    def _b0_std(self) -> float:
        return float(self.intercept + np.sum(self.coef_std * self.x_mean / self.x_scale))

    def to_dict(self) -> dict:
        return {
            "coef": self.coef.tolist(), "intercept": self.intercept, "sigma": self.sigma,
            "item_intercept": self.item_intercept, "ridge": self.ridge, "gh_nodes": self.gh_nodes,
            "columns": self.columns, "x_mean": self.x_mean.tolist(), "x_scale": self.x_scale.tolist(),
            "converged": self.converged, "n_iter": self.n_iter, "objective": self.objective,
        }


# ---------------------------------------------------------------------- objective
def _objective(theta, Xs, y, sw, gidx, n_groups, u, logw, ridge, n_beta, item_intercept, sigma_fixed):
    b0 = theta[0]
    beta = theta[1:1 + n_beta]
    eta0 = b0 + Xs @ beta
    pen = 0.5 * ridge * float(beta @ beta)
    if not item_intercept:
        nll = np.sum(sw * (np.logaddexp(0.0, eta0) - y * eta0))
        r = sw * (y - special.expit(eta0))
        g = np.empty_like(theta)
        g[0] = -r.sum()
        g[1:1 + n_beta] = -(Xs.T @ r) + ridge * beta
        return float(nll + pen), g
    if sigma_fixed is None:
        s = theta[1 + n_beta]
        sigma = np.exp(s)
    else:
        sigma = float(sigma_fixed)
    eta = eta0[:, None] + sigma * u[None, :]                           # (n, Q)
    ell = sw[:, None] * (y[:, None] * eta - np.logaddexp(0.0, eta))     # (n, Q)
    Lgq = np.zeros((n_groups, len(u)))
    np.add.at(Lgq, gidx, ell)
    a = Lgq + logw[None, :]
    lse = special.logsumexp(a, axis=1)
    post = np.exp(a - lse[:, None])                                     # (G, Q)
    r = sw[:, None] * (y[:, None] - special.expit(eta))                 # (n, Q)
    pr = post[gidx] * r                                                 # (n, Q)
    rs = pr.sum(1)
    g = np.empty_like(theta)
    g[0] = -rs.sum()
    g[1:1 + n_beta] = -(Xs.T @ rs) + ridge * beta
    if sigma_fixed is None:
        g[1 + n_beta] = -sigma * float((pr * u[None, :]).sum())
    return float(-lse.sum() + pen), g


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(0) if len(X) else np.zeros(X.shape[1])
    scale = X.std(0) if len(X) else np.ones(X.shape[1])
    const = scale < 1e-12
    mean = np.where(const, 0.0, mean)
    scale = np.where(const, 1.0, scale)
    return (X - mean) / scale, mean, scale


def fit_glm(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence | None = None,
    *,
    item_intercept: bool,
    ridge: float,
    gh_nodes: int = 20,
    sample_weight: np.ndarray | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    sigma_fixed: float | None = None,
    sigma_init: float = 0.5,
    columns: list[str] | None = None,
    laplace: bool = False,
    maxiter: int = 2000,
) -> GLMFit:
    """Fit by L-BFGS-B. ``groups`` (one key per row) is required with ``item_intercept=True``."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    n, p = X.shape
    if len(y) != n:
        raise ValueError("X and y lengths differ")
    if np.any((y < 0) | (y > 1)) or not np.all(np.isfinite(y)):
        raise ValueError("y must be finite and in [0, 1]")
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    if item_intercept and groups is None:
        raise ValueError("item_intercept=True requires groups")
    Xs, mean, scale = _standardize(X)
    if groups is not None:
        gidx, gkeys = _group_index(list(groups))
    else:
        gidx, gkeys = np.zeros(n, dtype=np.int64), [None]
    u, w = gh_standard_normal(gh_nodes)
    logw = np.log(w)
    free_sigma = item_intercept and sigma_fixed is None
    k = 1 + p + (1 if free_sigma else 0)
    theta0 = np.zeros(k)
    ybar = float(np.clip(np.average(y, weights=sw) if sw.sum() > 0 else 0.5, 1e-3, 1 - 1e-3))
    theta0[0] = np.log(ybar / (1 - ybar))
    if free_sigma:
        theta0[-1] = np.log(max(sigma_init, 1e-3))
    bnds: list[tuple[float | None, float | None]] = [(None, None)]
    if bounds is None:
        bnds += [(None, None)] * p
    else:
        if len(bounds) != p:
            raise ValueError("bounds must have one entry per column")
        for j, (lo, hi) in enumerate(bounds):
            if scale[j] == 1.0 and mean[j] == 0.0:
                bnds.append((lo, hi))
            else:
                bnds.append((None if lo is None else lo * scale[j], None if hi is None else hi * scale[j]))
        theta0[1:1 + p] = np.clip(theta0[1:1 + p], [b[0] if b[0] is not None else -np.inf for b in bnds[1:1 + p]],
                                  [b[1] if b[1] is not None else np.inf for b in bnds[1:1 + p]])
    if free_sigma:
        bnds.append(_LOG_SIGMA_BOUNDS)
    args = (Xs, y, sw, gidx, len(gkeys), u, logw, float(ridge), p, bool(item_intercept),
            None if not item_intercept else sigma_fixed)
    res = optimize.minimize(_objective, theta0, args=args, jac=True, method="L-BFGS-B", bounds=bnds,
                            options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-12, "gtol": 1e-7})
    theta = res.x
    beta_std = theta[1:1 + p]
    sigma = float(np.exp(theta[-1])) if free_sigma else (float(sigma_fixed) if (item_intercept and sigma_fixed)
                                                         else 0.0)
    coef = beta_std / scale
    intercept = float(theta[0] - np.sum(beta_std * mean / scale))
    fit = GLMFit(coef=coef, intercept=intercept, sigma=sigma, item_intercept=bool(item_intercept),
                 ridge=float(ridge), gh_nodes=int(gh_nodes), columns=columns, x_mean=mean, x_scale=scale,
                 coef_std=beta_std.copy(), group_keys=gkeys if groups is not None else [],
                 converged=bool(res.success), n_iter=int(res.nit), objective=float(res.fun))
    if item_intercept and sigma > 0 and groups is not None:
        _attach_posterior(fit, X, y, sw, gidx, len(gkeys))
    if laplace:
        fit.cov_std = _laplace_cov(theta, args)
    return fit


def _attach_posterior(fit: GLMFit, X, y, sw, gidx, n_groups, m: int = 241) -> None:
    grid = np.linspace(-6.0, 6.0, m)
    eta = fit.linear_predictor(X)[:, None] + fit.sigma * grid[None, :]
    ell = sw[:, None] * (y[:, None] * eta - np.logaddexp(0.0, eta))
    L = np.zeros((n_groups, m))
    np.add.at(L, gidx, ell)
    a = L - 0.5 * grid[None, :] ** 2
    post = np.exp(a - special.logsumexp(a, axis=1)[:, None])
    fit._u_grid = grid
    fit._post_grid = post
    fit.group_post_mean = fit.sigma * (post @ grid)


def _laplace_cov(theta: np.ndarray, args: tuple, h: float = 1e-5) -> np.ndarray:
    k = len(theta)
    H = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h
        H[:, j] = (_objective(theta + e, *args)[1] - _objective(theta - e, *args)[1]) / (2 * h)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H + 1e-8 * np.eye(k))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    return (vecs * np.clip(vals, 0, None)) @ vecs.T


# ---------------------------------------------------------------------- grouped CV
def grouped_folds(groups: Sequence, folds: int = 5, seed: int = 20260925, strata: Sequence | None = None) -> np.ndarray:
    """Fold id per row; whole problems go to one fold. Deterministic: problems are ordered by a salted sha256 and
    dealt round-robin (within stratum when ``strata`` -- one label per row, constant within a problem -- is given)."""
    gidx, gkeys = _group_index(list(groups))
    gstrat: dict[int, str] = {}
    if strata is not None:
        for gi, s in zip(gidx, strata):
            gstrat.setdefault(int(gi), str(s))
    by_stratum: dict[str, list[int]] = {}
    for gi, key in enumerate(gkeys):
        by_stratum.setdefault(gstrat.get(gi, ""), []).append(gi)
    gfold = np.empty(len(gkeys), dtype=np.int64)
    offset = 0
    for s in sorted(by_stratum):
        members = sorted(by_stratum[s], key=lambda gi: hashlib.sha256(
            f"passk_cont_v1/cv|{seed}|{gkeys[gi]}".encode()).hexdigest())
        for rank, gi in enumerate(members):
            gfold[gi] = (rank + offset) % folds
        offset += len(members)
    return gfold[gidx]


def cv_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    *,
    item_intercept: bool,
    ridges: Sequence[float] = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0),
    folds: int = 5,
    seed: int = 20260925,
    strata: Sequence | None = None,
    metric: str = "log_loss",
    fold_ids: np.ndarray | None = None,
    **fit_kwargs,
) -> tuple[float, list[dict]]:
    """Choose ``ridge`` by problem-grouped K-fold CV. Returns (best ridge, per-ridge table)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    fid = grouped_folds(groups, folds, seed, strata) if fold_ids is None else np.asarray(fold_ids)
    garr = np.asarray(list(groups), dtype=object)
    sw = fit_kwargs.pop("sample_weight", None)
    table = []
    for lam in ridges:
        pred = np.full(len(y), np.nan)
        for f in np.unique(fid):
            tr, te = fid != f, fid == f
            fit = fit_glm(X[tr], y[tr], garr[tr], item_intercept=item_intercept, ridge=lam,
                          sample_weight=None if sw is None else np.asarray(sw)[tr], **fit_kwargs)
            pred[te] = fit.predict_proba(X[te])
        score = log_loss(y, pred) if metric == "log_loss" else brier(y, pred)
        table.append({"ridge": float(lam), metric: score, "brier": brier(y, pred), "log_loss": log_loss(y, pred)})
    best = min(table, key=lambda r: (r[metric], -r["ridge"]))
    return best["ridge"], table
