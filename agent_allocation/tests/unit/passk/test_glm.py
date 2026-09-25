"""passk.models.glm: coefficient recovery, sigma=0 == plain ridge, grouped CV, posteriors (simulated data only)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from agent_allocation.passk.models import glm


def _sim(n_groups=300, per=12, beta=(1.0, -0.7), b0=-0.3, sigma=1.2, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    X = rng.normal(size=(len(g), len(beta)))
    u = rng.normal(size=n_groups) * sigma
    eta = b0 + X @ np.asarray(beta) + u[g]
    y = (rng.uniform(size=len(g)) < special.expit(eta)).astype(float)
    return X, y, g, u


def test_plain_ridge_recovers_coefficients():
    X, y, g, _ = _sim(sigma=0.0, n_groups=800, per=10)
    fit = glm.fit_glm(X, y, g, item_intercept=False, ridge=1e-6)
    assert fit.converged
    np.testing.assert_allclose(fit.coef, [1.0, -0.7], atol=0.08)
    assert abs(fit.intercept + 0.3) < 0.08


def test_random_intercept_recovers_coefficients_and_sigma():
    X, y, g, u = _sim(n_groups=400, per=12, sigma=1.2, seed=1)
    fit = glm.fit_glm(X, y, g, item_intercept=True, ridge=1e-6)
    np.testing.assert_allclose(fit.coef, [1.0, -0.7], atol=0.12)
    assert abs(fit.intercept + 0.3) < 0.15
    assert abs(fit.sigma - 1.2) < 0.2
    # posterior means of item effects track the truth
    assert np.corrcoef(fit.group_post_mean, u)[0, 1] > 0.7
    # the naive (no intercept) fit is attenuated toward zero
    naive = glm.fit_glm(X, y, g, item_intercept=False, ridge=1e-6)
    assert abs(naive.coef[0]) < abs(fit.coef[0])


def test_sigma_zero_equals_plain_ridge():
    X, y, g, _ = _sim(n_groups=100, per=8, seed=2)
    for lam in (0.0, 0.5, 5.0):
        plain = glm.fit_glm(X, y, g, item_intercept=False, ridge=lam)
        zero = glm.fit_glm(X, y, g, item_intercept=True, ridge=lam, sigma_fixed=0.0)
        np.testing.assert_allclose(zero.coef, plain.coef, atol=1e-6)
        assert abs(zero.intercept - plain.intercept) < 1e-6
        np.testing.assert_allclose(zero.predict_proba(X), plain.predict_proba(X), atol=1e-7)


def test_ridge_shrinks():
    X, y, g, _ = _sim(n_groups=60, per=5, seed=3, sigma=0.0)
    small = glm.fit_glm(X, y, g, item_intercept=False, ridge=1e-3)
    big = glm.fit_glm(X, y, g, item_intercept=False, ridge=1e3)
    assert np.linalg.norm(big.coef) < 0.1 * np.linalg.norm(small.coef)


def test_gradient_matches_finite_difference():
    X, y, g, _ = _sim(n_groups=20, per=5, seed=4)
    Xs, _, _ = glm._standardize(X)
    gidx, keys = glm._group_index(list(g))
    u, w = glm.gh_standard_normal(15)
    args = (Xs, y, np.ones(len(y)), gidx, len(keys), u, np.log(w), 0.7, 2, True, None)
    theta = np.array([0.1, 0.4, -0.2, np.log(0.8)])
    f, grad = glm._objective(theta, *args)
    num = np.array([(glm._objective(theta + e, *args)[0] - glm._objective(theta - e, *args)[0]) / 2e-6
                    for e in np.eye(4) * 1e-6])
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-5)


def test_marginal_prediction_integrates_item_effect():
    X, y, g, _ = _sim(n_groups=300, per=10, seed=5, sigma=1.5)
    fit = glm.fit_glm(X, y, g, item_intercept=True, ridge=1e-4)
    rng = np.random.default_rng(0)
    b = fit.sample_item_effects(200_000, rng)
    x0 = np.array([[0.3, -0.2]])
    mc = special.expit(fit.linear_predictor(x0)[0] + b).mean()
    assert abs(fit.predict_proba(x0)[0] - mc) < 3e-3
    # conditional prediction with explicit effect
    assert fit.predict_proba(x0, item_effect=0.0)[0] == pytest.approx(special.expit(fit.linear_predictor(x0)[0]))
    # posterior draws for a training group centre on its posterior mean
    d = fit.posterior_item_effects(fit.group_keys[0], 20_000, rng)
    assert abs(d.mean() - fit.group_post_mean[0]) < 0.05


def test_grouped_folds_keep_problems_whole_and_are_deterministic():
    g = np.repeat(np.arange(37), 3)
    f1 = glm.grouped_folds(g, 5, seed=7)
    f2 = glm.grouped_folds(g, 5, seed=7)
    assert (f1 == f2).all()
    for gi in np.unique(g):
        assert len(set(f1[g == gi])) == 1
    counts = np.bincount(f1) // 3
    assert counts.max() - counts.min() <= 1
    assert not (glm.grouped_folds(g, 5, seed=8) == f1).all()


def test_cv_ridge_prefers_small_ridge_with_strong_signal_and_large_with_noise():
    X, y, g, _ = _sim(n_groups=150, per=6, seed=6, sigma=0.0, beta=(2.0, -1.5))
    best, table = glm.cv_ridge(X, y, g, item_intercept=False, ridges=(0.01, 1.0, 1000.0))
    assert best < 1000.0 and len(table) == 3
    rng = np.random.default_rng(9)
    Xn = rng.normal(size=(300, 10))
    yn = (rng.uniform(size=300) < 0.5).astype(float)
    gn = np.repeat(np.arange(60), 5)
    best_n, _ = glm.cv_ridge(Xn, yn, gn, item_intercept=False, ridges=(0.01, 1000.0))
    assert best_n == 1000.0


def test_monotone_bounds_and_laplace():
    rng = np.random.default_rng(10)
    n = 2000
    depth = rng.integers(0, 4, size=n)
    cum = np.stack([(depth >= k).astype(float) for k in (1, 2, 3)], 1)
    true = np.array([0.8, -0.4, 0.6])          # the middle step truly decreases
    y = (rng.uniform(size=n) < special.expit(-0.5 + cum @ true)).astype(float)
    fit = glm.fit_glm(cum, y, None, item_intercept=False, ridge=1e-4, bounds=[(0, None)] * 3, laplace=True)
    assert (fit.coef >= -1e-9).all()
    free = glm.fit_glm(cum, y, None, item_intercept=False, ridge=1e-4)
    assert free.coef[1] < 0
    b0, coef, sig = fit.sample_params(500, rng)
    assert coef.shape == (500, 3) and np.isfinite(b0).all()
    assert abs(b0.mean() - fit.intercept) < 0.1


def test_fractional_targets_and_weights():
    X, y, g, _ = _sim(n_groups=50, per=4, seed=11, sigma=0.0)
    w = np.full(len(y), 2.0)
    a = glm.fit_glm(X, y, g, item_intercept=False, ridge=2.0, sample_weight=w)
    b = glm.fit_glm(np.vstack([X, X]), np.concatenate([y, y]), None, item_intercept=False, ridge=2.0)
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-5)
    frac = glm.fit_glm(X, np.clip(y * 0.8 + 0.1, 0, 1), g, item_intercept=True, ridge=1.0)
    assert frac.converged or frac.n_iter > 0
    with pytest.raises(ValueError):
        glm.fit_glm(X, y + 2, g, item_intercept=False, ridge=1.0)
