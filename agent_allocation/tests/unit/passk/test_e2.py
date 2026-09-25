"""E2: Beta closed form vs Monte Carlo, pool simulator limits, fit guard, synthetic end-to-end fit/predict/score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk.models import e2, response
from agent_allocation.passk.models.answers import ItemClasses, simulate_pools
from agent_allocation.passk.models.inputs import DEPTHS, PREDICT_CONFIGS, pool_outcome


def test_beta_coverage_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for c, m, k in ((0, 16, 16), (3, 16, 4), (1, 1, 2), (7, 8, 8)):
        al, be = response.beta_posterior(c, m)
        p = rng.beta(al, be, size=400_000)
        mc = 1 - np.mean((1 - p) ** k)
        assert abs(response.beta_coverage(al, be, k) - mc) < 3e-3


def test_mixed_coverage_is_product_per_draw():
    P = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0, 1.0]])
    cov = response.coverage_from_draws(P, {1: 2, 4: 1})
    np.testing.assert_allclose(cov, [1 - 0.9 ** 2 * 0.7, 0.0])
    assert response.coverage_from_draws(P, {16: 1})[1] == 1.0


def _ic(correct=(10,), wrong=(), novote=0):
    return ItemClasses("x", np.array(correct, float), np.array(wrong, float), novote, int(sum(wrong)))


def test_simulator_limits_and_treatments():
    rng = np.random.default_rng(1)
    S = 300
    plur, cov = simulate_pools(np.ones(S), _ic(), 8, rng)
    assert np.allclose(plur, 1) and np.allclose(cov, 1)
    plur, cov = simulate_pools(np.zeros(S), _ic(), 8, rng)
    assert np.allclose(plur, 0) and np.allclose(cov, 0)
    # single call: plurality == correctness probability
    p = np.full(S, 0.3)
    plur1, _ = simulate_pools(p, _ic(wrong=(5, 3)), 1, rng, inner=400)
    assert abs(plur1.mean() - 0.3) < 0.02
    # many calls, p = 0.3, wrong mass on one strong class -> competition hurts; diffuse residual is kinder
    strong = _ic(correct=(6,), wrong=(14,))
    comp, _ = simulate_pools(p, strong, 16, rng, inner=200)
    nocomp, _ = simulate_pools(p, strong, 16, rng, inner=200, competition=False)
    assert nocomp.mean() > comp.mean() + 0.2
    empty = _ic(correct=(3,), wrong=())
    diff, _ = simulate_pools(p, empty, 16, rng, inner=200, treatment="diffuse")
    conc, _ = simulate_pools(p, empty, 16, rng, inner=200, treatment="concentrated")
    assert diff.mean() > conc.mean() + 0.2


def test_pool_outcome_tie_average_and_tristate():
    classes = {"a": "k1", "b": "k2", "c": "k3", "d": None}
    labels = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    out = pool_outcome(["a", "b", "c", "d"], classes, labels)
    assert out["coverage"] == 1.0 and out["plurality"] == pytest.approx(1 / 3)
    out = pool_outcome(["b", "d"], classes, {"b": np.nan, "d": 0.0})
    assert np.isnan(out["coverage"]) and np.isnan(out["plurality"])


# ---------------------------------------------------------------------- synthetic end-to-end
BENCH = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
V48_BANK = {1: 16, 2: 8, 4: 4, 8: 2, 16: 1}
FIT_BANK = {1: 16, 2: 8, 4: 8, 8: 8, 16: 8}


def synth_draws(n_per_bench=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    items = []
    for b in BENCH:
        for j in range(n_per_bench):
            item = f"{b}/{j}"
            items.append(item)
            base = rng.normal(-0.5, 1.5)
            wrong_pool = [f"w{t}" for t in range(3)]
            for part, bank in (("legacy_main", FIT_BANK), ("v48_bank", V48_BANK)):
                for a, m in bank.items():
                    p = 1 / (1 + np.exp(-(base + 0.6 * np.log2(a))))
                    for r in range(m):
                        ok = rng.uniform() < p
                        key = "gold" if ok else rng.choice(wrong_pool, p=[0.6, 0.3, 0.1])
                        rows.append(dict(candidate_id=f"{part}|{item}|{a}|{r}", source="x", partition=part,
                                         item_id=item, benchmark=b, kind="root" if a == 1 else "deep",
                                         allowance_a=a, replicate=r, vote_exact_key=key, option=None,
                                         correct=bool(ok), parse_status="valid"))
    return pd.DataFrame(rows), items


def test_e2_fit_predict_score_end_to_end(tmp_path):
    draws, items = synth_draws()
    res = e2.fit_and_predict(draws=draws, c48=items, out_dir=tmp_path / "v48", S=100, inner=16)
    pred = res["predictions"]
    assert set(pred.model) == set(e2.MODELS)
    menu = pred[pred.in_menu]
    assert len(menu) == len(items) * len(PREDICT_CONFIGS) * len(e2.MODELS)
    # a posterior mean may sit outside a skewed 5-95% band; the band itself must be ordered and in [0, 1]
    assert ((0 <= menu.pi_lo) & (menu.pi_lo <= menu.pi_hi) & (menu.pi_hi <= 1)).all()
    assert menu.pred_plurality.between(0, 1).all() and menu.pred_coverage.between(0, 1).all()
    assert (tmp_path / "v48" / "predictions.parquet").exists()
    assert (tmp_path / "v48" / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        e2.fit_and_predict(draws=draws, c48=items, out_dir=tmp_path / "v48", S=50, inner=8)
    sc = e2.score(draws=draws, v48=items, pred_dir=tmp_path / "v48", out_dir=tmp_path / "score")
    ov = sc["overall"].set_index("model")
    assert ov.plurality_brier.between(0, 1).all()
    assert ov.ranking_accuracy.dropna().between(0, 1).all()
    assert {"residual", "sign_agree"} <= set(sc["crossover"].columns)


def test_e2_guard_refuses_fenced_rows():
    draws, items = synth_draws(n_per_bench=1)
    fit = draws[draws.partition == "v48_bank"]
    with pytest.raises(RuntimeError):
        e2._guard(fit)
