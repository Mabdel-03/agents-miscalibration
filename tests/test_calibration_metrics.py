"""ECE/MCE/Brier on synthetic data with known calibration properties."""

from agents_scaling.calibration.metrics import compute_calibration


def test_perfect_calibration_zero_ece():
    # Confidence exactly equals empirical accuracy within each bin.
    # 10 items at conf 0.9, 9 correct -> bin accuracy 0.9 == confidence.
    confs = [0.9] * 10
    corrects = [True] * 9 + [False]
    rep = compute_calibration(confs, corrects, n_bins=10)
    assert abs(rep.ece) < 1e-9
    assert abs(rep.mce) < 1e-9


def test_overconfident_has_positive_ece():
    # Always 100% confident but only 50% correct -> ECE ~ 0.5.
    confs = [1.0] * 10
    corrects = [True] * 5 + [False] * 5
    rep = compute_calibration(confs, corrects, n_bins=10)
    assert abs(rep.ece - 0.5) < 1e-6
    assert abs(rep.mce - 0.5) < 1e-6


def test_brier_bounds():
    rep = compute_calibration([1.0, 0.0], [False, True], n_bins=10)
    assert abs(rep.brier - 1.0) < 1e-9  # worst case
    rep2 = compute_calibration([1.0, 0.0], [True, False], n_bins=10)
    assert abs(rep2.brier - 0.0) < 1e-9  # best case


def test_empty_input():
    rep = compute_calibration([], [], n_bins=15)
    assert rep.n == 0 and rep.ece == 0.0


def test_confidence_one_lands_in_last_bin():
    # A confidence of exactly 1.0 must be counted (last-bin right edge inclusive).
    rep = compute_calibration([1.0], [True], n_bins=15)
    assert rep.n == 1
    assert sum(b.count for b in rep.bins) == 1
