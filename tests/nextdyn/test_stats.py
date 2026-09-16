"""The confirmatory engine: parity with ``ard.stats`` and the four additions (plan D13, §4.2).

The parity tests exist because ``nextdyn.stats`` re-exports ``ard.stats``'s interval engine
rather than copying it: they assert that the re-export is the same object and that it still
produces ARD36's numbers on an ``n_primary = 3`` fixture, so a future edit to ``ard/stats.py``
that changed the estimator would fail here instead of silently changing this study's intervals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents_scaling.nextdyn import stats
from agents_scaling.nextdyn.errors import ContractViolation

FAMILIES = ("MATH", "MMLU_PRO", "TRUTHFULQA_LEGACY_MC1_GENERATED_CHOICE")
EQUAL = {f: 1.0 for f in FAMILIES}


def frame(rng, effects, n=60):
    return {f: rng.normal(effect, 1.0, n) for f, effect in zip(FAMILIES, effects)}


# --------------------------------------------------------------------------- parity


def test_the_interval_engine_is_ard_stats_itself_not_a_copy():
    from agents_scaling.ard import stats as ard_stats

    assert stats.paired_contrast is ard_stats.paired_contrast
    assert stats.Resampler is ard_stats.Resampler
    assert stats.analysis_seed is ard_stats.analysis_seed
    assert stats.mean_ci is ard_stats.mean_ci
    assert stats.noninferiority_bound is ard_stats.noninferiority_bound


def test_paired_contrast_parity_with_ard_stats_on_an_n_primary_3_fixture():
    from agents_scaling.ard import stats as ard_stats

    rng = np.random.default_rng(11)
    domains = ("math500", "mmlu_pro", "tqa")
    diff = {d: rng.normal(-0.04, 0.3, 90) for d in domains}
    seed = stats.analysis_seed("ARD36_v1", "PEER_minus_PRIVATE_root_to_round3_coverage_change")
    weights = {d: 1 / 3 for d in domains}
    ours = stats.paired_contrast(
        diff, stats.Resampler({d: 90 for d in domains}, 2000, seed, domains=domains),
        domains=domains, weights=weights, n_primary=3)
    theirs = ard_stats.paired_contrast(
        diff, ard_stats.Resampler({d: 90 for d in domains}, 2000, seed, domains=domains),
        domains=domains, weights=weights, n_primary=3)
    assert ours["estimate"] == theirs["estimate"] == pytest.approx(
        float(np.mean([v.mean() for v in diff.values()])), abs=1e-12)
    assert ours["p"] == theirs["p"]
    assert ours["ci95_studentized"] == theirs["ci95_studentized"]
    assert ours["n_primary"] == 3


def test_analysis_seed_is_a_pure_function_of_the_study_seed_and_the_label():
    a = stats.analysis_seed("deadbeef", "H1")
    assert a == stats.analysis_seed("deadbeef", "H1")
    assert a != stats.analysis_seed("deadbeef", "H2")
    assert a != stats.analysis_seed("cafebabe", "H1")


# --------------------------------------------------------------------------- wild cluster


def test_wild_cluster_test_is_null_centred():
    """Under the null the test has its nominal size; under a large true effect it still does.

    The second half is the property that distinguishes a null-centred wild bootstrap from a
    naive one: subtracting the within-family mean makes the resampling distribution a sample
    from H0 whatever the true effect is, so the *size* of the test does not depend on the
    effect that happens to be present.
    """
    rng = np.random.default_rng(3)
    rejections = 0
    trials = 60
    for _ in range(trials):
        result = stats.wild_cluster_test(frame(rng, (0.0, 0.0, 0.0), n=40), weights=EQUAL,
                                         n_resamples=199, seed=int(rng.integers(1 << 30)))
        rejections += int(result["p"] <= 0.05)
    assert rejections <= 8, f"size under the null is {rejections}/{trials}, far above 5%"

    # the same test on data with a large common effect: the resampling distribution is still
    # centred at zero, so t* is not dragged along with the estimate
    shifted = stats.wild_cluster_test(frame(np.random.default_rng(7), (5.0, 5.0, 5.0), n=40),
                                      weights=EQUAL, n_resamples=999, seed=5)
    assert shifted["estimate"] == pytest.approx(5.0, abs=0.4)
    assert shifted["p"] < 0.01
    from_null = stats.wild_cluster_test(frame(np.random.default_rng(7), (5.0, 5.0, 5.0), n=40),
                                        weights=EQUAL, n_resamples=999, seed=5,
                                        null_value=5.0)
    assert from_null["p"] > 0.05, "a null AT the true effect must not reject"


def test_wild_cluster_test_uses_webb_weights_below_the_declared_cluster_count():
    rng = np.random.default_rng(1)
    few = {f: rng.normal(0.1, 1.0, 6) for f in FAMILIES}
    many = {f: rng.normal(0.1, 1.0, 40) for f in FAMILIES}
    assert stats.wild_cluster_test(few, weights=EQUAL, n_resamples=99, seed=1,
                                   webb_below_clusters=10)["weights_dist"] == stats.WEBB
    assert stats.wild_cluster_test(many, weights=EQUAL, n_resamples=99, seed=1,
                                   webb_below_clusters=10)["weights_dist"] == stats.RADEMACHER


def test_wild_cluster_test_reports_estimation_only_below_the_cluster_floor():
    rng = np.random.default_rng(2)
    small = {f: rng.normal(0.2, 1.0, 5) for f in FAMILIES}
    result = stats.wild_cluster_test(small, weights=EQUAL, n_resamples=99, seed=1,
                                     min_clusters_for_test=30)
    assert result["verdict"] == stats.ESTIMATION_ONLY
    assert result["n_clusters"] == 15


def test_wild_cluster_test_is_not_estimable_without_two_clusters_anywhere():
    result = stats.wild_cluster_test({f: np.array([0.3]) for f in FAMILIES}, weights=EQUAL,
                                     n_resamples=99, seed=1)
    assert result["verdict"] == stats.NOT_ESTIMABLE and result["p"] is None
    assert "at least two clusters" in result["reason"]


# --------------------------------------------------------------------------- H2 joint


def test_joint_contrast_rejects_only_heterogeneous_task_effects():
    rng = np.random.default_rng(19)
    homogeneous = {f: rng.normal(0.5, 0.2, 60) for f in FAMILIES}       # big, but identical
    heterogeneous = {f: rng.normal(effect, 0.2, 60)
                     for f, effect in zip(FAMILIES, (0.6, 0.0, -0.6))}  # same size, different signs
    flat = stats.joint_contrast(homogeneous, n_resamples=400, seed=4, min_items_per_family=30)
    split = stats.joint_contrast(heterogeneous, n_resamples=400, seed=4, min_items_per_family=30)
    assert flat["p"] > 0.10, "a large but homogeneous effect is not heterogeneity"
    assert split["p"] < 0.01, "three clearly different per-task effects must reject"
    assert set(split["by_family"]) == set(FAMILIES)
    assert len(split["contrast"]) == len(FAMILIES) - 1


def test_joint_contrast_is_not_estimable_below_two_supported_families():
    rng = np.random.default_rng(5)
    thin = {FAMILIES[0]: rng.normal(0, 1, 40), FAMILIES[1]: rng.normal(0, 1, 5),
            FAMILIES[2]: rng.normal(0, 1, 3)}
    result = stats.joint_contrast(thin, n_resamples=50, seed=1, min_items_per_family=30)
    assert result["verdict"] == stats.NOT_ESTIMABLE and result["p"] is None
    assert result["families"] == [FAMILIES[0]]


# --------------------------------------------------------------------------- multiplicity


def test_holm_over_eight_with_not_estimable_slot():
    pvalues = {"H1": 0.001, "H2": 0.02, "H3": None, "H4": 0.04, "H5": 0.5,
               "H6": 0.006, "H7": 0.9, "H8": 0.3}
    result = stats.holm_with_not_estimable(pvalues, alpha=0.05, n_primary=8)
    assert set(result) == set(pvalues)
    assert result["H3"]["p"] == 1.0 and result["H3"]["verdict"] == stats.NOT_ESTIMABLE
    assert result["H3"]["reject"] is False and result["H3"]["estimable"] is False
    # Holm over 8: the smallest p is multiplied by 8, and the adjusted values are monotone
    assert result["H1"]["p_holm"] == pytest.approx(0.008)
    ordered = sorted(result.values(), key=lambda r: r["rank"])
    assert all(a["p_holm"] <= b["p_holm"] + 1e-12 for a, b in zip(ordered, ordered[1:]))
    # Holm is step-down, not Bonferroni: H6 clears alpha/(m-j+1) = 0.05/7 at rank 2, so it
    # rejects even though 0.006 > 0.05/8.  H2 at rank 3 faces 0.05/6 and stops the descent.
    assert result["H1"]["reject"] is True and result["H6"]["reject"] is True
    assert result["H2"]["reject"] is False and result["H4"]["reject"] is False

    # dropping the NOT_ESTIMABLE hypothesis would make the family smaller -- and is refused
    with pytest.raises(ContractViolation, match="exactly 8"):
        stats.holm_with_not_estimable({k: v for k, v in pvalues.items() if k != "H3"},
                                      alpha=0.05, n_primary=8)


def test_a_not_estimable_slot_never_makes_the_others_easier_to_reject():
    full = {"H1": 0.006, "H2": 0.02, "H3": None, "H4": 0.04, "H5": 0.5, "H6": 0.06,
            "H7": 0.9, "H8": 0.3}
    with_value = dict(full, H3=0.5)
    a = stats.holm_with_not_estimable(full, alpha=0.05, n_primary=8)
    b = stats.holm_with_not_estimable(with_value, alpha=0.05, n_primary=8)
    assert a["H1"]["p_holm"] == b["H1"]["p_holm"] == pytest.approx(0.048)


def test_bh_fdr_matches_statsmodels_when_it_is_installed():
    pvalues = {f"s{i}": p for i, p in enumerate([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9])}
    ours = stats.bh_fdr(pvalues, q=0.10)
    assert [ours[k]["rank"] for k in sorted(pvalues)] == [1, 2, 3, 4, 5, 6, 7]
    adjusted = [ours[k]["p_adj"] for k in sorted(pvalues)]
    assert all(a <= b + 1e-12 for a, b in zip(adjusted, adjusted[1:])), "BH is monotone"
    statsmodels = pytest.importorskip("statsmodels.stats.multitest",
                                      reason="statsmodels is not installed in the harness env")
    reject, padj, _, _ = statsmodels.multipletests(
        [pvalues[k] for k in sorted(pvalues)], alpha=0.10, method="fdr_bh")
    assert adjusted == pytest.approx(list(padj))
    assert [ours[k]["reject"] for k in sorted(pvalues)] == list(reject)


def test_bh_fdr_keeps_a_not_estimable_secondary_out_of_m():
    with_none = stats.bh_fdr({"a": 0.01, "b": 0.02, "c": None}, q=0.10)
    assert with_none["c"]["verdict"] == stats.NOT_ESTIMABLE and with_none["c"]["p_adj"] is None
    assert with_none["a"]["p_adj"] == pytest.approx(0.02)   # m = 2, not 3


# --------------------------------------------------------------------------- paired binary


def test_mcnemar_exact_is_not_doubled():
    from scipy.stats import binomtest

    a = [1] * 12 + [0] * 1 + [1] * 50 + [0] * 50
    b = [0] * 12 + [1] * 1 + [1] * 50 + [0] * 50
    result = stats.mcnemar_exact(a, b)
    assert (result["n10"], result["n01"]) == (12, 1)
    assert result["p"] == pytest.approx(binomtest(1, 13, 0.5).pvalue)
    assert result["p"] == pytest.approx(0.003417, abs=1e-5)
    assert result["two_sided"] is True


def test_mcnemar_exact_on_an_all_concordant_comparison_is_one():
    result = stats.mcnemar_exact([1, 0, 1, 0], [1, 0, 1, 0])
    assert result["n_discordant"] == 0 and result["p"] == 1.0


def test_tost_equivalence_distinguishes_equivalence_from_a_null_result():
    rng = np.random.default_rng(23)
    tight = {f: rng.normal(0.0, 0.005, 80) for f in FAMILIES}
    noisy = {f: rng.normal(0.0, 0.5, 20) for f in FAMILIES}
    equivalent = stats.tost_equivalence(tight, weights=EQUAL, margin=0.05, n_resamples=999, seed=2)
    inconclusive = stats.tost_equivalence(noisy, weights=EQUAL, margin=0.05, n_resamples=999, seed=2)
    assert equivalent["equivalent"] is True
    assert inconclusive["equivalent"] is False
    assert "not evidence of a difference" in inconclusive["readout"]


# --------------------------------------------------------------------------- settings


def test_settings_holds_the_declared_seed_rule_to_the_implementation(analysis_cfg, study_cfg):
    settings = stats.settings(analysis_cfg.profile, study_cfg.analysis)
    assert settings.seed_label_rule == stats.SEED_LABEL_RULE
    assert settings.n_primary == len(study_cfg.analysis.primary_hypotheses) == 8
    assert settings.alpha == study_cfg.analysis.familywise_alpha
    assert settings.equivalence_margin == pytest.approx(0.05)
    assert settings.n_resamples == analysis_cfg.profile.n_resamples

    class Drifted:
        bootstrap_seed_label_rule = "some other rule"
        n_resamples = 10
        equivalence_margin_pp = 5
        secondary_fdr_q = 0.1
        min_clusters_for_test = 30
        webb_below_clusters = 10

    with pytest.raises(ContractViolation, match="same fact"):
        stats.settings(Drifted(), study_cfg.analysis)
