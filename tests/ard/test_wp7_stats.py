"""WP7 statistics: the vendored engine reproduces ``study/stats.py`` and fails loudly.

Plan §10: "vendored ``ard/stats.py`` reproduces ``study/stats.py`` bit-for-bit on an
('hle','bcb') fixture and fails loudly on a domain-axis mismatch"; plan §8 (Holm over exactly
the three primaries, the shifted-null / one-sided bound for the noninferiority endpoint).
"""

from __future__ import annotations

import numpy as np
import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import stats as S
from agents_scaling.study import stats as SS

LEGACY_DOMAINS = ("hle", "bcb")
LEGACY_WEIGHTS = {"hle": 0.5, "bcb": 0.5}
B = 400


def _fixture(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"hle": rng.normal(0.04, 0.3, size=41), "bcb": rng.normal(-0.02, 0.5, size=29)}


def _matrix_fixture(seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"hle": rng.integers(0, 2, size=(37, 4)).astype(float),
            "bcb": rng.integers(0, 2, size=(23, 4)).astype(float)}


def _pair(n_by_domain, seed=5):
    """The vendored and the study resampler, constructed identically."""
    return (S.Resampler(n_by_domain, B, seed, domains=LEGACY_DOMAINS),
            SS.Resampler(n_by_domain, B, seed))


# --------------------------------------------------------------------------- bit-for-bit


def test_resampler_indices_are_identical():
    mine, theirs = _pair({"hle": 41, "bcb": 29})
    assert mine.n_by_domain == theirs.n_by_domain
    for d in LEGACY_DOMAINS:
        assert np.array_equal(mine.idx[d], theirs.idx[d])


def test_paired_contrast_reproduces_study_stats_bit_for_bit():
    diffs = _fixture()
    mine_rs, their_rs = _pair({d: len(v) for d, v in diffs.items()})
    mine = S.paired_contrast(diffs, mine_rs, domains=LEGACY_DOMAINS, weights=LEGACY_WEIGHTS,
                             n_primary=len(SS.FAMILIES))
    theirs = SS.paired_contrast(diffs, their_rs)
    for key in ("estimate", "se", "t", "p", "primary_alpha"):
        assert mine[key] == theirs[key], key
    assert mine["by_domain"] == theirs["by_domain"]
    assert mine["n"] == theirs["n"]
    assert mine["weights"] == theirs["weights"]
    for key in ("ci95_studentized", "ci95_percentile", "ci_primary_bonferroni"):
        assert mine[key] == theirs[key], key


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_every_alternative_reproduces_study_stats(alternative):
    diffs = _fixture(3)
    mine_rs, their_rs = _pair({d: len(v) for d, v in diffs.items()}, seed=9)
    mine = S.paired_contrast(diffs, mine_rs, domains=LEGACY_DOMAINS, weights=LEGACY_WEIGHTS,
                             n_primary=len(SS.FAMILIES), alternative=alternative)
    theirs = SS.paired_contrast(diffs, their_rs, alternative=alternative)
    assert mine["p"] == theirs["p"]
    assert mine["t"] == theirs["t"]


def test_max_t_family_reproduces_study_stats_bit_for_bit():
    Y = _matrix_fixture()
    labels = ["FRESH", "PRIVATE", "PEER", "BALANCED"]
    mine_rs, their_rs = _pair({d: len(v) for d, v in Y.items()}, seed=13)
    mine = S.max_t_family(Y, labels, mine_rs, domains=LEGACY_DOMAINS, weights=LEGACY_WEIGHTS,
                          n_primary=len(SS.FAMILIES))
    theirs = SS.max_t_family(Y, labels, their_rs)
    assert mine["p_global"] == theirs["p_global"]
    assert mine["max_abs_t_observed"] == theirs["max_abs_t_observed"]
    assert mine["max_abs_t_q95"] == theirs["max_abs_t_q95"]
    assert mine["max_abs_t_q_primary"] == theirs["max_abs_t_q_primary"]
    for a, b in zip(mine["pairs"], theirs["pairs"]):
        assert a["contrast"] == b["contrast"]
        assert (a["estimate"], a["se"], a["t"], a["p_maxT_adjusted"]) == (
            b["estimate"], b["se"], b["t"], b["p_maxT_adjusted"])
        assert a["ci95_simultaneous_within_family"] == b["ci95_simultaneous_within_O"]
        assert a["ci_primary_bonferroni"] == b["ci_primary_bonferroni"]
    assert mine["methods"] == theirs["methods"]


def test_mean_ci_and_analysis_seed_reproduce_study_stats():
    vals = _fixture(17)
    mine_rs, their_rs = _pair({d: len(v) for d, v in vals.items()}, seed=21)
    assert S.mean_ci(vals, mine_rs, domains=LEGACY_DOMAINS, weights=LEGACY_WEIGHTS) == \
        SS.mean_ci(vals, their_rs)
    assert S.analysis_seed("deadbeef", "GATE_minus_PEER") == \
        SS.analysis_seed("deadbeef", "GATE_minus_PEER")


def test_holm_reproduces_study_stats():
    p = {"A": 0.004, "B": 0.03, "C": 0.4}
    assert S.holm(p) == SS.holm(p)


# --------------------------------------------------------------------------- loud failures


def test_domain_axis_mismatch_raises_instead_of_reporting_no_rejection():
    """The study engine returns estimate=None, p=1.0 for an unknown domain key; we raise."""
    diffs = {"math500": np.array([0.1, 0.2, 0.3]), "mmlu_pro": np.array([0.0, -0.1, 0.2])}
    rs = S.Resampler({d: 3 for d in diffs}, 50, 1, domains=tuple(diffs))
    silent = SS.paired_contrast(diffs, SS.Resampler({d: 3 for d in diffs}, 50, 1))
    assert silent["estimate"] is None and silent["p"] == 1.0   # the defect this vendoring fixes
    with pytest.raises(S.StatsAxisError) as excinfo:
        S.paired_contrast(diffs, rs, domains=LEGACY_DOMAINS, weights=LEGACY_WEIGHTS)
    assert "domain axis mismatch" in str(excinfo.value)


def test_extra_domain_key_is_a_loud_failure():
    diffs = {"math500": np.zeros(4), "mmlu_pro": np.ones(4), "surprise": np.ones(4)}
    rs = S.Resampler({d: 4 for d in diffs}, 20, 1, domains=tuple(diffs))
    with pytest.raises(S.StatsAxisError):
        S.paired_contrast(diffs, rs)


def test_estimate_is_never_none_a_degenerate_panel_raises():
    diffs = {"math500": np.array([0.1]), "mmlu_pro": np.array([])}
    rs = S.Resampler({"math500": 1, "mmlu_pro": 0}, 20, 1)
    with pytest.raises(S.StatsError):
        S.paired_contrast(diffs, rs)


def test_max_t_family_axis_is_checked_too():
    Y = {"math500": np.zeros((5, 3))}
    rs = S.Resampler({"math500": 5, "mmlu_pro": 5}, 20, 1)
    with pytest.raises(S.StatsAxisError):
        S.max_t_family(Y, ["a", "b", "c"], rs)


def test_missing_weight_for_a_declared_domain_raises():
    diffs = {"math500": np.zeros(4) + 0.1, "mmlu_pro": np.zeros(4)}
    rs = S.Resampler({d: 4 for d in diffs}, 20, 1)
    with pytest.raises(S.StatsAxisError):
        S.paired_contrast(diffs, rs, weights={"math500": 0.5})


# --------------------------------------------------------------------------- primary family


def test_holm_over_exactly_three_primary_contrasts():
    p = dict(zip(K.PRIMARY_CONTRASTS, (0.01, 0.02, 0.9)))
    out = S.holm(p, n_primary=K.N_PRIMARY)
    assert set(out) == set(K.PRIMARY_CONTRASTS)
    assert out[K.PRIMARY_CONTRASTS[0]]["p_holm"] == pytest.approx(0.03)
    assert out[K.PRIMARY_CONTRASTS[1]]["p_holm"] == pytest.approx(0.04)
    assert out[K.PRIMARY_CONTRASTS[2]]["p_holm"] == pytest.approx(0.9)
    assert [out[c]["reject"] for c in K.PRIMARY_CONTRASTS] == [True, True, False]


def test_holm_refuses_a_family_that_is_not_exactly_the_three_primaries():
    with pytest.raises(S.StatsAxisError):
        S.holm({"a": 0.01, "b": 0.02}, n_primary=K.N_PRIMARY)
    with pytest.raises(S.StatsAxisError):
        S.holm(dict(zip((*K.PRIMARY_CONTRASTS, "sneaked_in"), (0.01,) * 4)),
               n_primary=K.N_PRIMARY)


def test_primary_bonferroni_divisor_is_n_primary_not_the_study_six():
    diffs = {"math500": np.linspace(-0.2, 0.4, 30), "mmlu_pro": np.linspace(-0.1, 0.3, 24)}
    rs = S.Resampler({"math500": 30, "mmlu_pro": 24}, 200, 3)
    out = S.paired_contrast(diffs, rs)
    assert out["primary_alpha"] == pytest.approx(K.ALPHA / K.N_PRIMARY)
    assert out["n_primary"] == 3


# --------------------------------------------------------------------------- noninferiority


def test_noninferiority_clears_the_margin_when_harm_is_small():
    rng = np.random.default_rng(5)
    diffs = {"math500": rng.normal(0.0, 0.05, size=120),
             "mmlu_pro": rng.normal(0.0, 0.05, size=120)}
    rs = S.Resampler({d: 120 for d in diffs}, 2_000, 2)
    out = S.noninferiority_bound(diffs, rs, margin=K.NONINFERIORITY_MARGIN)
    assert out["margin"] == K.NONINFERIORITY_MARGIN
    assert out["null_value"] == K.NONINFERIORITY_MARGIN
    assert out["alternative"] == "less"
    assert out["upper_bound_one_sided"] < K.NONINFERIORITY_MARGIN
    assert out["noninferior"] is True
    assert out["p"] < 0.05


def test_noninferiority_reports_harm_not_excluded_when_the_bound_fails():
    rng = np.random.default_rng(6)
    diffs = {"math500": rng.normal(0.06, 0.30, size=40),
             "mmlu_pro": rng.normal(0.06, 0.30, size=40)}
    rs = S.Resampler({d: 40 for d in diffs}, 2_000, 4)
    out = S.noninferiority_bound(diffs, rs)
    assert out["upper_bound_one_sided"] >= K.NONINFERIORITY_MARGIN
    assert out["noninferior"] is False
    assert out["readout"] == "harm not excluded"


def test_shifted_null_changes_the_p_value_but_not_the_estimate():
    diffs = {"math500": np.full(30, 0.05), "mmlu_pro": np.full(30, 0.05) + 1e-9}
    rs = S.Resampler({d: 30 for d in diffs}, 500, 8)
    plain = S.paired_contrast(diffs, rs, alternative="less")
    shifted = S.paired_contrast(diffs, rs, alternative="less", null_value=0.08)
    assert plain["estimate"] == shifted["estimate"]
    assert shifted["t"] < plain["t"]
