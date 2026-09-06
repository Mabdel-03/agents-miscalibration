"""N3 family-C tests (CPU): the threshold rule, missing-prior handling, recalibrator fit /
apply / freeze, source-item-weighted reliability bins and the C contrast."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents_scaling.study.neural import calibration as C
from agents_scaling.study.neural import readout as R
from tests.study_neural.n3_support import synthetic_frames

SEED = bytes(range(32))


def test_threshold_rule_requires_ten_successes_and_failures_per_method():
    y = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    m = ["IND_VOTE"] * 20 + ["DEC"] * 20
    ok, counts = C.threshold_rule(y, m)
    assert ok and counts == {"DEC": {"successes": 10, "failures": 10}, "IND_VOTE": {"successes": 10, "failures": 10}}
    y[0] = 0  # IND_VOTE drops to 9 successes
    ok, counts = C.threshold_rule(y, m)
    assert not ok and counts["IND_VOTE"] == {"successes": 9, "failures": 11}
    assert C.threshold_rule([], [])[0] is False


def test_fill_missing_uses_the_dev_prior_and_flags():
    q, flag = C.fill_missing([0.3, None, float("nan"), 1.5, "x", 0.9], prior=0.42)
    assert q.tolist() == pytest.approx([0.3, 0.42, 0.42, 0.42, 0.42, 0.9])
    assert flag.tolist() == [0, 1, 1, 1, 1, 0]
    z = C.clipped_logit(np.asarray([0.0, 0.5, 1.0]))
    assert z[1] == 0.0 and z[0] == pytest.approx(-z[2]) and math.isfinite(z[0])


def test_reliability_bins_are_source_item_weighted():
    q = np.asarray([0.1, 0.2, 0.7, 0.8])
    y = np.asarray([0.0, 1.0, 1.0, 0.0])
    w = np.asarray([1.0, 3.0, 1.0, 1.0])
    bins = C.reliability_bins(q, y, w, edges=[0.0, 0.5, 1.0])
    assert bins[0]["n"] == 2 and bins[0]["weight"] == 4.0
    assert bins[0]["mean_confidence"] == pytest.approx((0.1 * 1 + 0.2 * 3) / 4)
    assert bins[0]["mean_outcome"] == pytest.approx(3 / 4)  # not the unweighted 1/2
    assert bins[1]["mean_outcome"] == pytest.approx(0.5) and bins[1]["gap"] == pytest.approx(0.75 - 0.5)
    edges = C.dev_bin_edges([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None], n_bins=5)
    assert edges[0] == 0.0 and edges[-1] == 1.0 and len(edges) == 6


def _dev_rows(n: int, seed: int, *, drop_method_successes: str | None = None, split: str = "dev", sharp_team: bool = False):
    frames = synthetic_frames(n, seed=seed, neural_signal=0.0, obs_signal=1.0, split=split)
    rows = frames.rows.copy()
    if drop_method_successes:
        rows.loc[rows["method"] == drop_method_successes, "y"] = 0
    if sharp_team:  # an explicit forecast that tracks the sealed outcome closely
        rng = np.random.default_rng(seed + 100)
        rows["q_team_now"] = np.clip(0.15 + 0.7 * rows["y"].to_numpy(dtype=float) + rng.normal(scale=0.1, size=len(rows)), 0.01, 0.99)
    return rows


def test_fit_recalibrators_primary_choice_and_apply(tmp_path):
    rows = _dev_rows(20, seed=21)
    frozen = C.fit_recalibrators(rows, seed=SEED, penalties=(1.0, 100.0))
    personal = frozen.scopes[C.SCOPE_PERSONAL]
    assert personal.primary == "conditioned" and not personal.threshold_failure and personal.conditioned is not None
    assert set(personal.outcome_counts) == {"IND_VOTE", "DEC", "CEN_FLAT"}
    assert 0 < personal.prior < 1 and personal.missing_rate_dev == 0.0
    assert personal.conditioned.feature_names == ["logit_q", "missing", "method=CEN_FLAT", "method=DEC", "method=IND_VOTE"]
    assert personal.common.feature_names == ["logit_q", "missing"]
    p = personal.apply(rows["personal_conf"].tolist(), rows["method"].tolist())
    assert p.shape == (len(rows),) and np.all((p > 0) & (p < 1))
    # confidence carries signal here (obs_signal=1): the map keeps a positive slope
    assert personal.common.slope > 0
    # a missing probability maps through the prior + flag, not to 0
    pm = personal.apply([None], ["DEC"])
    assert 0 < pm[0] < 1
    # freeze round trip and tamper guard
    path = frozen.save(tmp_path)
    loaded = C.FrozenC.load(tmp_path)
    assert np.allclose(loaded.scopes[C.SCOPE_TEAM].apply(rows["q_team_now"].tolist(), rows["method"].tolist()),
                       frozen.scopes[C.SCOPE_TEAM].apply(rows["q_team_now"].tolist(), rows["method"].tolist()))
    text = path.read_text()
    path.write_text(text.replace('"primary": "conditioned"', '"primary": "common"', 1))
    with pytest.raises(C.CalibrationError):
        C.FrozenC.load(tmp_path)


def test_threshold_failure_falls_back_to_the_common_map():
    rows = _dev_rows(20, seed=22, drop_method_successes="CEN_FLAT")
    frozen = C.fit_recalibrators(rows, seed=SEED, penalties=(1.0, 100.0))
    for scope in C.SCOPES:
        cal = frozen.scopes[scope]
        assert cal.threshold_failure and cal.primary == "common" and cal.conditioned is None
        assert cal.outcome_counts["CEN_FLAT"]["successes"] == 0
        assert np.allclose(cal.apply([0.5], ["DEC"]), cal.apply([0.5], ["DEC"], which="common"))
        with pytest.raises(C.CalibrationError):
            cal.apply([0.5], ["DEC"], which="conditioned")
    # a lower frozen minimum would have admitted the conditioned map for the other scope's data
    ok, _ = C.threshold_rule(rows["y"].tolist(), rows["method"].tolist(), minimum=0)
    assert ok


def test_score_rows_and_contrast_C_report_raw_and_recalibrated_losses():
    dev = _dev_rows(20, seed=23, sharp_team=True)
    frozen = C.fit_recalibrators(dev, seed=SEED, penalties=(1.0, 100.0))
    conf = _dev_rows(20, seed=24, split="main", sharp_team=True)
    conf.loc[conf.index[:5], "forecast_valid"] = False  # five invalid forecasts → prior + flag
    scored = C.score_rows(conf, frozen)
    for col in ("q_baseline", "q_explicit", "brier_baseline", "brier_explicit", "brier_raw_baseline", "missing_explicit"):
        assert col in scored
    assert scored["missing_explicit"].sum() == 5 and scored["missing_baseline"].sum() == 0
    assert np.allclose(scored.loc[scored["missing_explicit"] == 1, "q_raw_explicit"], frozen.scopes[C.SCOPE_TEAM].prior)
    out = C.contrast_C(scored, seed=5, n_resamples=300, frozen=frozen)
    boot = out["bootstrap"]
    assert set(boot) >= {"estimate", "p_one_sided", "ci_low", "ci_high", "n_clusters", "weights"}
    assert boot["n_clusters"] == {"hle": 20, "bcb": 20} and boot["weights"] == {"bcb": 0.5, "hle": 0.5}
    assert set(out["by_method"]) == {"IND_VOTE", "DEC", "CEN_FLAT"}
    diag = out["diagnostics"]["explicit"]["recalibrated"]
    assert len(diag["reliability_bins"]) == len(frozen.scopes[C.SCOPE_TEAM].bin_edges) - 1
    assert diag["missing_rate"] == pytest.approx(5 / len(conf))
    assert "cox_slope" in diag and "signed_bias_weighted" in diag and diag["diagnostics_backend"] in ("nb_lib.calib", "neural.calibration")
    # the explicit q_team_now was generated from the true logit → it should beat personal confidence
    assert out["weighted_brier"]["explicit"] < out["weighted_brier"]["baseline"]
    assert boot["estimate"] > 0


def test_recalibration_improves_a_miscalibrated_signal():
    rng = np.random.default_rng(3)
    n = 300
    truth = rng.uniform(0.05, 0.95, size=n)
    y = (rng.random(n) < truth).astype(int)
    q = np.clip(truth * 0.5 + 0.45, 0.01, 0.99)  # squashed, overconfident on the low end
    methods = ["IND_VOTE" if i % 2 else "DEC" for i in range(n)]
    clusters = [f"hle:{i}" if i < n // 2 else f"bcb:{i}" for i in range(n)]
    sds = ["hle" if i < n // 2 else "bcb" for i in range(n)]
    cal = C.fit_scope(list(q), y, methods, clusters, sds, scope=C.SCOPE_PERSONAL, seed=SEED)
    p = cal.apply(list(q), methods)
    assert R.brier(p, y).mean() < R.brier(q, y).mean() - 0.01
    assert cal.common.slope > 0
    # the systematic overconfidence is removed: signed bias and the Cox intercept go to ~0
    # (the slope stays shrunk by the one-SE-selected penalty; that is the intended map, §8.7)
    assert abs(float(np.mean(p - y))) < 0.02 < abs(float(np.mean(q - y)))
    raw, fixed = C.cox_intercept_slope(q, y), C.cox_intercept_slope(p, y)
    assert abs(fixed["cox_intercept"]) < 0.2 < abs(raw["cox_intercept"])
