"""Spec 13.3 geometry: the estimators are imported, never re-derived, and their two refusals.

Spec 13.3 fixes two things this file pins:

* "Rank is at most N-1" -- the centered covariance of ``s`` states has at most ``s-1``
  nonzero eigenvalues however large the hidden dimension is, so every spectral summary is
  computed through the small ``s x s`` Gram matrix with a ``rank_ceiling = min(d, s-1)``.
* "When total variance is zero, spectral ratios are **undefined**, not evidence of a
  one-dimensional collapse."  Identical states at a byte-identical root prompt are the
  *expected* case under deterministic forward evaluation (spec 13.2), so a participation
  ratio of 1.0 there would be a manufactured finding.

Nothing in ``nextdyn`` re-implements these: :func:`capture.team_geometry` /
:func:`capture.agent_geometry` are pass-throughs to ``ard.neural.geometry.team_metrics`` /
``agent_metrics`` (both on the ``ARD_IMPORTS`` allowlist), and this file asserts the
pass-through as well as the behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents_scaling.ard.neural import geometry as ARDG
from agents_scaling.nextdyn import capture as CAP

N_AGENTS = 5
HIDDEN = 64


def _states(seed: int = 0, n: int = N_AGENTS, hidden: int = HIDDEN) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, hidden))


def _same(mine: dict, theirs: dict) -> None:
    """Equality that treats NaN as NaN (an undefined ratio is a value, not a mismatch)."""
    assert set(mine) == set(theirs)
    for key, value in mine.items():
        other = theirs[key]
        if isinstance(value, float) and np.isnan(value):
            assert isinstance(other, float) and np.isnan(other), key
        else:
            assert value == other, key


def test_zero_variance_spectral_metrics_are_undefined():
    """Five identical states: every scale-free ratio is NaN and the run says so explicitly."""
    identical = np.repeat(_states(0, n=1), N_AGENTS, axis=0)
    metrics = CAP.team_geometry(identical)

    assert np.isnan(metrics["total_variance"])
    for name in CAP.SPECTRAL_RATIOS:
        assert np.isnan(metrics[name]), f"{name} must be undefined at zero total variance"
    assert CAP.spectral_ratios_defined(metrics) is False
    # and it is not merely "small": the eigenvalue list is empty, so nothing downstream can
    # read a 1.0 participation ratio as a one-dimensional collapse
    assert metrics["n_nonzero_eigenvalues"] == 0
    assert metrics["mean_pairwise"] == pytest.approx(0.0)

    spread = CAP.team_geometry(_states(1))
    assert spread["total_variance"] > 0 and CAP.spectral_ratios_defined(spread) is True


def test_geometry_rank_not_above_n_minus_one():
    """Rank ceiling is ``min(d, s-1)`` for every team size, and the spectrum obeys it."""
    for n in (2, 3, 5, 9):
        metrics = CAP.team_geometry(_states(n, n=n))
        assert metrics["rank_ceiling"] == min(HIDDEN, n - 1) == n - 1
        assert metrics["n_nonzero_eigenvalues"] <= n - 1
        assert len(metrics["eigenvalues"]) <= n - 1
        if metrics["spectrum_testable"]:        # s >= 3; two states carry no spectrum at all
            assert metrics["participation_ratio"] <= n - 1 + 1e-9
            assert metrics["entropy_rank"] <= n - 1 + 1e-9
    # a hidden size smaller than n-1 binds instead
    narrow = CAP.team_geometry(_states(3, n=9, hidden=4))
    assert narrow["rank_ceiling"] == 4 and narrow["n_nonzero_eigenvalues"] <= 4


def test_the_geometry_entry_points_are_the_ard_estimators_not_a_second_derivation():
    """R1/D2: one estimator, imported.  A re-derivation here would drift from ARD36's."""
    X = _states(7)
    _same(CAP.team_geometry(X), ARDG.team_metrics(X, standardizer=None, normalised=False))
    _same(CAP.team_geometry(X, normalised=True), ARDG.team_metrics(X, normalised=True))
    for mine, theirs in zip(CAP.agent_geometry(X), ARDG.agent_metrics(X)):
        _same(mine, theirs)
    for name in ("agents_scaling.ard.neural.geometry.team_metrics",
                 "agents_scaling.ard.neural.geometry.agent_metrics"):
        from agents_scaling.nextdyn import ARD_IMPORTS
        assert name in ARD_IMPORTS


def test_raw_and_normalised_team_rows_are_produced_side_by_side():
    """Spec 13.3: "use both raw and development-normalized scale-carrying metrics"."""
    X = _states(3) * 10.0
    raw, unit = CAP.team_geometry(X), CAP.team_geometry(X, normalised=True)
    assert raw["normalised"] is False and unit["normalised"] is True
    assert raw["total_variance"] > unit["total_variance"]
    # the unit-normalised row is the scale-free view: its states all have norm one, while the
    # raw row keeps the scale-carrying norms spec 13.3 asks to be shown beside it
    assert unit["state_norm_mean"] == pytest.approx(1.0) and raw["state_norm_mean"] > 1.0
    assert raw["representation"].startswith("raw+") and unit["representation"].startswith("unit_normalised+")


def test_per_agent_displacement_alignment_records_near_zero_denominators():
    """Spec 13.3: a near-zero denominator is recorded, never divided through."""
    previous = _states(11)
    moved = previous + np.random.default_rng(12).standard_normal(previous.shape) * 0.5
    rows = CAP.agent_geometry(moved, previous=previous, agent_ids=[0, 1, 2, 3, 4])
    assert [r["agent_id"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(np.isfinite(r["own_displacement"]) for r in rows)
    assert all(not r["near_zero_denominator"] for r in rows)

    frozen = CAP.agent_geometry(previous, previous=previous)
    assert all(r["near_zero_denominator"] for r in frozen)
    assert all(np.isnan(r["displacement_alignment"]) for r in frozen)
    assert all(r["own_displacement"] == pytest.approx(0.0) for r in frozen)


def test_a_single_state_has_no_team_geometry_at_all():
    """One agent is not a team: the spectrum is refused rather than reported as zero spread."""
    metrics = CAP.team_geometry(_states(0, n=1))
    assert metrics["n_states"] == 1 and metrics["spectrum_testable"] is False
    assert np.isnan(metrics["participation_ratio"]) and CAP.spectral_ratios_defined(metrics) is False
    with pytest.raises(ARDG.GeometryError):
        CAP.agent_geometry(_states(0, n=1))
