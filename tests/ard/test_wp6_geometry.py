"""WP6 geometry: the ``neural_geometry`` / ``neural_agent`` tables on constructed states.

Every case here has a hand-computable answer — identical states, one outlier, a pure
translation, an agent sitting exactly on its peer centroid — so the estimator is checked
against arithmetic rather than against itself.  CPU only; no model is loaded.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as A
from agents_scaling.ard.neural import geometry as G

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ANCHOR = A.ANCHOR_ANSWER_END


def state_set(X, *, round_=1, layer=17, anchor=ANCHOR, item_id="math500:0001",
              protocol="PEER", agent_ids=None, **kwargs):
    X = np.asarray(X, dtype=np.float64)
    ids = tuple(range(X.shape[0])) if agent_ids is None else tuple(agent_ids)
    return G.StateSet(item_id=item_id, protocol=protocol, round_=round_, layer=layer,
                      anchor=anchor, states=X, agent_ids=ids, **kwargs)


def random_states(seed=0, s=K.N_AGENTS, d=16):
    return np.random.default_rng(seed).standard_normal((s, d)) + 3.0


# --------------------------------------------------------------------------- the N-1 ceiling


def test_rank_statistics_respect_the_n_minus_one_ceiling_after_centering():
    X = random_states(seed=1, s=K.N_AGENTS, d=16)
    assert G.rank_ceiling(16, K.N_AGENTS) == K.N_AGENTS - 1 == 4
    for normalised in (False, True):
        m = G.team_metrics(X, normalised=normalised)
        assert m["n_states"] == K.N_AGENTS and m["hidden_size"] == 16
        assert m["rank_ceiling"] == 4
        assert m["n_nonzero_eigenvalues"] <= 4
        assert len(m["eigenvalues"]) <= 4
        assert 1.0 <= m["participation_ratio"] <= 4.0 + 1e-9
        assert 1.0 <= m["entropy_rank"] <= 4.0 + 1e-9
        assert m["participation_ratio_unit"] <= 4.0 + 1e-9
        assert m["entropy_rank_unit"] <= 4.0 + 1e-9
        assert m["spectrum_testable"] is True


def test_the_ceiling_is_the_hidden_size_when_it_is_the_smaller_of_the_two():
    X = random_states(seed=2, s=K.N_AGENTS, d=2)
    m = G.team_metrics(X)
    assert m["rank_ceiling"] == 2
    assert m["n_nonzero_eigenvalues"] <= 2
    assert m["participation_ratio"] <= 2.0 + 1e-9


# --------------------------------------------------------------------------- degenerate teams


def test_identical_states_have_no_spectrum_and_zero_distances():
    base = np.linspace(-1.0, 1.0, 8)
    X = np.tile(base, (K.N_AGENTS, 1))
    m = G.team_metrics(X)

    assert m["eigenvalues"] == []
    assert math.isnan(m["participation_ratio"]) and math.isnan(m["entropy_rank"])
    assert math.isnan(m["leading_eigenvalue_share"]) and math.isnan(m["total_variance"])
    assert m["mean_pairwise"] == m["min_pairwise"] == m["max_pairwise"] == 0.0
    assert m["cosine_distance"] == pytest.approx(0.0, abs=1e-12)
    assert m["state_norm_std"] == pytest.approx(0.0)

    rows = G.agent_metrics(X, previous=X)
    for row in rows:
        assert row["distance_to_peer_centroid"] == pytest.approx(0.0)
        assert row["distance_to_nearest_peer"] == pytest.approx(0.0)
        # no movement at all: the denominator is recorded, the alignment is NOT 1.0
        assert row["own_displacement"] == pytest.approx(0.0)
        assert row["near_zero_denominator"] is True
        assert math.isnan(row["displacement_alignment"])
        assert row["displacement_denominator"] == pytest.approx(0.0)


def test_one_outlier_gives_an_exactly_rank_one_spectrum():
    X = np.asarray([[1.0, 0.0, 0.0]] * 4 + [[1.0, 0.0, 10.0]])
    m = G.team_metrics(X)

    assert m["n_nonzero_eigenvalues"] == 1
    assert m["eigenvalues"][0] == pytest.approx(20.0)        # (4*2^2 + 8^2)/(5-1)
    assert m["total_variance"] == pytest.approx(20.0)
    assert m["leading_eigenvalue_share"] == pytest.approx(1.0)
    assert m["participation_ratio"] == pytest.approx(1.0)
    assert m["entropy_rank"] == pytest.approx(1.0)
    assert m["min_pairwise"] == pytest.approx(0.0)
    assert m["max_pairwise"] == pytest.approx(10.0)
    assert m["mean_pairwise"] == pytest.approx(4.0)          # 6 pairs at 0, 4 pairs at 10

    rows = G.agent_metrics(X)
    outlier, inlier = rows[4], rows[0]
    assert outlier["distance_to_peer_centroid"] == pytest.approx(10.0)
    assert inlier["distance_to_peer_centroid"] == pytest.approx(2.5)   # centroid of 3 peers + outlier
    assert outlier["distance_to_nearest_peer"] == pytest.approx(10.0)
    assert inlier["distance_to_nearest_peer"] == pytest.approx(0.0)
    assert outlier["nearest_peer_agent"] == 0
    assert max(r["distance_to_peer_centroid"] for r in rows) == outlier["distance_to_peer_centroid"]


def test_leading_share_and_total_variance_on_a_known_two_axis_spectrum():
    X = np.asarray([[-2.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 1.0]])
    assert X.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert float(X[:, 0] @ X[:, 1]) == pytest.approx(0.0)    # the two axes are uncorrelated
    m = G.team_metrics(X)

    assert sorted(m["eigenvalues"], reverse=True) == pytest.approx([2.5, 1.0])
    assert m["total_variance"] == pytest.approx(3.5)
    assert m["leading_eigenvalue_share"] == pytest.approx(2.5 / 3.5)
    assert m["participation_ratio"] == pytest.approx(3.5 ** 2 / (2.5 ** 2 + 1.0 ** 2))
    p = np.asarray([2.5, 1.0]) / 3.5
    assert m["entropy_rank"] == pytest.approx(math.exp(-float((p * np.log(p)).sum())))


def test_leading_share_and_total_variance_of_an_empty_spectrum_are_nan():
    assert math.isnan(G.leading_eigenvalue_share([]))
    assert math.isnan(G.total_variance([]))
    assert math.isnan(G.leading_eigenvalue_share([0.0, -1e-30]))
    assert G.leading_eigenvalue_share([3.0, 1.0]) == pytest.approx(0.75)
    assert G.total_variance([3.0, 1.0]) == pytest.approx(4.0)


# --------------------------------------------------------------------------- per-agent axis


def test_peer_centroid_is_the_centroid_of_the_other_four():
    X = np.asarray([[0.0], [4.0], [8.0], [12.0], [16.0]])
    assert G.peer_centroid(X, 0) == pytest.approx([10.0])
    assert G.peer_centroid(X, 4) == pytest.approx([6.0])
    with pytest.raises(G.GeometryError):
        G.peer_centroid(np.asarray([[1.0]]), 0)

    rows = G.agent_metrics(X, agent_ids=[10, 11, 12, 13, 14])
    assert [r["agent_id"] for r in rows] == [10, 11, 12, 13, 14]
    assert rows[0]["distance_to_peer_centroid"] == pytest.approx(10.0)
    assert rows[0]["distance_to_nearest_peer"] == pytest.approx(4.0)
    assert rows[0]["nearest_peer_agent"] == 11
    assert rows[2]["nearest_peer_agent"] == 11          # ties break to the lower slot
    assert rows[0]["state_norm"] == pytest.approx(0.0)
    assert rows[4]["state_norm"] == pytest.approx(16.0)
    assert all(r["has_previous_state"] is False for r in rows)
    assert all(math.isnan(r["own_displacement"]) for r in rows)


def test_a_pure_translation_moves_every_agent_by_the_same_amount():
    X0 = random_states(seed=3)
    v = np.full(X0.shape[1], 0.3)
    X1 = X0 + v

    before, after = G.team_metrics(X0), G.team_metrics(X1)
    # the centered spectrum and the pairwise distances are translation invariant
    assert after["eigenvalues"] == pytest.approx(before["eigenvalues"])
    assert after["total_variance"] == pytest.approx(before["total_variance"])
    assert after["leading_eigenvalue_share"] == pytest.approx(before["leading_eigenvalue_share"])
    assert after["participation_ratio"] == pytest.approx(before["participation_ratio"])
    assert after["mean_pairwise"] == pytest.approx(before["mean_pairwise"])
    # the scale-carrying norms are not
    assert after["state_norm_mean"] != pytest.approx(before["state_norm_mean"])

    rows = G.agent_metrics(X1, previous=X0)
    for i, row in enumerate(rows):
        assert row["has_previous_state"] is True
        assert row["own_displacement"] == pytest.approx(float(np.linalg.norm(v)))
        assert row["near_zero_denominator"] is False
        direction = G.peer_centroid(X0, i) - X0[i]
        expected = float(v @ direction / (np.linalg.norm(v) * np.linalg.norm(direction)))
        assert row["displacement_alignment"] == pytest.approx(expected)
        assert -1.0 - 1e-9 <= row["displacement_alignment"] <= 1.0 + 1e-9


def test_displacement_alignment_signs_are_towards_away_and_orthogonal():
    previous = np.asarray([[0.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    assert G.peer_centroid(previous, 0) == pytest.approx([4.0, 0.0])   # the pre-update direction

    def alignment_of(step):
        X = previous.copy()
        X[0] = previous[0] + step
        return G.agent_metrics(X, previous=previous)[0]

    assert alignment_of(np.asarray([1.0, 0.0]))["displacement_alignment"] == pytest.approx(1.0)
    assert alignment_of(np.asarray([-1.0, 0.0]))["displacement_alignment"] == pytest.approx(-1.0)
    assert alignment_of(np.asarray([0.0, 1.0]))["displacement_alignment"] == pytest.approx(0.0, abs=1e-12)
    assert alignment_of(np.asarray([3.0, 3.0]))["displacement_alignment"] == pytest.approx(1 / math.sqrt(2))
    # the direction is the PRE-update one: overshooting past the old centroid is still +1
    assert alignment_of(np.asarray([9.0, 0.0]))["displacement_alignment"] == pytest.approx(1.0)


def test_near_zero_denominators_are_recorded_not_divided_through():
    # (a) the agent sits exactly on its peer centroid at t-1: the direction has no length
    previous = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert G.peer_centroid(previous, 0) == pytest.approx([0.0, 0.0])
    X = previous.copy()
    X[0] = [1.0, 1.0]
    row = G.agent_metrics(X, previous=previous)[0]
    assert row["near_zero_denominator"] is True
    assert row["direction_denominator"] == pytest.approx(0.0)
    assert row["own_displacement"] == pytest.approx(math.sqrt(2.0))   # still measured
    assert math.isnan(row["displacement_alignment"])                  # never 0.0 and never 1.0

    # (b) the agent did not move: the displacement has no length
    stuck = G.agent_metrics(previous, previous=previous)[1]
    assert stuck["near_zero_denominator"] is True
    assert math.isnan(stuck["displacement_alignment"])

    # (c) an agent that really moved, away from a real peer direction, keeps a finite
    # alignment — so a NaN in this column always means "denominator recorded", not "no data"
    moved = previous.copy()
    moved[1] = [1.5, 0.5]
    healthy = G.agent_metrics(moved, previous=previous)[1]
    assert healthy["near_zero_denominator"] is False
    assert healthy["direction_denominator"] == pytest.approx(1.25)
    assert math.isfinite(healthy["displacement_alignment"])

    # the eps is an argument, not a hidden constant
    loose = G.agent_metrics(moved, previous=previous, eps=10.0)[1]
    assert loose["near_zero_denominator"] is True and math.isnan(loose["displacement_alignment"])


def test_agent_metrics_guards_its_inputs():
    X = random_states(seed=4)
    with pytest.raises(G.GeometryError):
        G.agent_metrics(X[:1])
    with pytest.raises(G.GeometryError):
        G.agent_metrics(X, previous=X[:4])
    with pytest.raises(G.GeometryError):
        G.agent_metrics(X, agent_ids=[0, 1, 2])
    with pytest.raises(G.GeometryError):
        G.team_metrics(np.zeros((0, 4)))


# --------------------------------------------------------------------------- raw vs normalised


def test_a_zero_norm_state_is_refused_by_the_normalised_variant():
    """A missing anchor must never arrive here as a zero vector (storage stores no vector)."""
    X = random_states(seed=5)
    X[2] = 0.0
    with pytest.raises(G.GeometryError, match="zero-norm"):
        G.team_metrics(X, normalised=True)
    raw = G.team_metrics(X, normalised=False)
    assert math.isnan(raw["cosine_distance"])           # reported as missing, not as 1.0


def test_raw_and_normalised_metrics_are_emitted_together_and_differ():
    X = random_states(seed=6)
    X[0] *= 25.0                                        # one very long state
    raw = G.team_metrics(X, normalised=False)
    unit = G.team_metrics(X, normalised=True)

    assert raw["normalised"] is False and unit["normalised"] is True
    assert raw["representation"].startswith("raw+")
    assert unit["representation"].startswith("unit_normalised+")
    assert unit["max_pairwise"] < raw["max_pairwise"]
    assert unit["total_variance"] != pytest.approx(raw["total_variance"])
    assert unit["state_norm_mean"] == pytest.approx(1.0)
    assert set(raw) == set(unit)


# --------------------------------------------------------------------------- tables


def test_neural_tables_emit_both_representations_and_one_row_per_agent():
    group = state_set(random_states(seed=7), round_=2)
    tables = G.neural_tables([group])

    assert sorted(tables) == ["neural_agent", "neural_geometry"]
    geometry, agents = tables["neural_geometry"], tables["neural_agent"]
    assert len(geometry) == 2
    assert [row["normalised"] for row in geometry] == [False, True]
    assert len(agents) == K.N_AGENTS
    assert [row["agent_id"] for row in agents] == list(range(K.N_AGENTS))

    for row in geometry:
        # the StateSet path produces the spec statistics; the cell/episode join tail is
        # added by the shards->tables adapter, which is the only path that knows a cell id
        assert set(G.GEOMETRY_SPEC_COLUMNS) <= set(row)
        assert (row["item_id"], row["protocol"], row["round"], row["layer"], row["anchor"]) == \
               ("math500:0001", "PEER", 2, 17, ANCHOR)
        assert row["feature_time"] == A.feature_time_of(ANCHOR) == A.FEATURE_TIME_POST
        assert row["last_allowed_round"] == A.last_allowed_round(ANCHOR, 2) == 2
    for row in agents:
        assert set(G.AGENT_SPEC_COLUMNS) <= set(row)
        assert row["feature_time"] == A.FEATURE_TIME_POST and row["last_allowed_round"] == 2


def test_a_pre_answer_anchor_carries_the_earlier_horizon():
    group = state_set(random_states(seed=8), round_=2, anchor=A.ANCHOR_PROMPT_END)
    rows = G.neural_tables([group])["neural_geometry"]
    assert all(r["feature_time"] == A.FEATURE_TIME_PRE for r in rows)
    assert all(r["last_allowed_round"] == 1 for r in rows)


def test_the_previous_round_is_joined_by_key_when_it_is_present():
    X0 = random_states(seed=9)
    X1 = X0 + 0.25
    groups = [state_set(X0, round_=0), state_set(X1, round_=1)]
    agents = G.neural_tables(groups)["neural_agent"]

    round0 = [r for r in agents if r["round"] == 0]
    round1 = [r for r in agents if r["round"] == 1]
    assert all(r["has_previous_state"] is False for r in round0)
    assert all(math.isnan(r["own_displacement"]) for r in round0)
    assert all(r["has_previous_state"] is True for r in round1)
    expected = float(np.linalg.norm(np.full(X0.shape[1], 0.25)))
    assert all(r["own_displacement"] == pytest.approx(expected) for r in round1)

    # a different team in the previous round is not a previous state of this one
    mismatched = [state_set(X0, round_=0, agent_ids=(5, 6, 7, 8, 9)), state_set(X1, round_=1)]
    assert all(r["has_previous_state"] is False for r in G.neural_tables(mismatched)["neural_agent"]
               if r["round"] == 1)
    # neither is the same round of another protocol / layer / anchor
    other = [state_set(X0, round_=0, protocol="PRIVATE"), state_set(X1, round_=1, protocol="PEER")]
    assert all(r["has_previous_state"] is False for r in G.neural_tables(other)["neural_agent"]
               if r["round"] == 1)


def test_duplicate_group_keys_are_refused():
    X = random_states(seed=10)
    with pytest.raises(G.GeometryError, match="duplicate"):
        G.neural_tables([state_set(X), state_set(X + 1.0)])
    # the checkpoint is part of the key: 8B and 32B coordinates are not comparable (spec 14.1)
    tables = G.neural_tables([state_set(X, checkpoint="8B"), state_set(X + 1.0, checkpoint="32B")])
    assert len(tables["neural_geometry"]) == 4


def test_state_set_guards_shape_and_anchor():
    X = random_states(seed=11)
    with pytest.raises(G.GeometryError):
        state_set(X, agent_ids=(0, 1, 2))
    with pytest.raises(G.GeometryError, match="CAPTURE_ANCHORS"):
        state_set(X, anchor="SOME_OTHER_ANCHOR")


def test_ordered_columns_puts_the_frozen_compact_columns_first():
    from agents_scaling.ard import compact as CP

    assert G.GEOMETRY_COLUMNS == CP.SCHEMAS["neural_geometry"]
    assert G.AGENT_COLUMNS == CP.SCHEMAS["neural_agent"]
    tables = G.neural_tables([state_set(random_states(seed=12))])
    for table, frozen in (("neural_geometry", G.GEOMETRY_COLUMNS), ("neural_agent", G.AGENT_COLUMNS)):
        columns = G.ordered_columns(table, tables[table])
        assert columns[: len(frozen)] == list(frozen)
        assert set(columns) >= set().union(*(set(r) for r in tables[table]))
        assert len(columns) == len(set(columns))


def test_the_team_statistic_nonclaim_is_stated():
    assert "permutation-invariant" in G.TEAM_STATISTIC_NONCLAIM
    assert "neural_agent" in G.TEAM_STATISTIC_NONCLAIM
