"""WP6 probes: the spec §14.4 evaluation contract (CPU, synthetic states).

The estimator itself is study_v4's ``readout.LogisticModel``; what is tested here is the
contract around it — the timing rule, identical folds and rows across arms, the
shuffled-within-item control, the within-item AUC denominator, and the refusal to choose a
layer or anchor with test items in hand.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as A
from agents_scaling.ard.neural import probes as P
from tests.ard import wp6_support as W

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = b"\x11" * 32
SIGNAL_LAYER, SIGNAL_ANCHOR = W.SIGNAL_SITE
NOISE_LAYER, NOISE_ANCHOR = W.NOISE_SITE


def frame(target=P.TARGET_CURRENT, *, n_folds=4, **kwargs):
    rows, H = W.probe_dataset(**kwargs)
    return P.build_frame(rows, H, target=target, study_seed=SEED, n_folds=n_folds)


def row(anchor, round_, targets, target_round=None, **kwargs):
    return P.ProbeRow(row_id="r", item_id="i", episode_id="e", agent_id=0, protocol="PEER",
                      round_=round_, layer=17, anchor=anchor, vector_index=0,
                      targets=targets, target_round=target_round or {}, **kwargs)


# --------------------------------------------------------------------------- feature_time


def test_a_post_answer_anchor_may_predict_the_answer_it_contains():
    """Spec §14.4 claim 2: post-answer features can predict *that* answer's correctness."""
    for anchor in (A.ANCHOR_ANSWER_END, A.ANCHOR_REPORT_END):
        r = row(anchor, 2, {P.TARGET_CURRENT: 1.0})
        assert r.feature_time == A.FEATURE_TIME_POST
        assert r.last_allowed_round == 2 == r.round_of(P.TARGET_CURRENT)
        P.check_timing(r, P.TARGET_CURRENT)          # permitted, at exactly the horizon


def test_a_post_answer_anchor_may_predict_a_round_t_plus_one_target():
    r = row(A.ANCHOR_ANSWER_END, 2, {P.TARGET_EXPOSURE_VALUE: 0.3})
    assert r.round_of(P.TARGET_EXPOSURE_VALUE) == 3 > r.last_allowed_round
    P.check_timing(r, P.TARGET_EXPOSURE_VALUE)
    P.check_timing(row(A.ANCHOR_REPORT_END, 1, {P.TARGET_CURRENT: 1.0},
                       {P.TARGET_CURRENT: 2}), P.TARGET_CURRENT)


def test_a_post_answer_anchor_is_refused_for_the_upcoming_answer_at_the_same_round():
    """The exemption is the *emitted* answer's correctness only.  A value target at the
    same round would make an already-emitted answer explain the update it precedes."""
    r = row(A.ANCHOR_ANSWER_END, 2, {P.TARGET_EXPOSURE_VALUE: 0.3}, {P.TARGET_EXPOSURE_VALUE: 2})
    with pytest.raises(P.TimingViolation) as excinfo:
        P.check_timing(r, P.TARGET_EXPOSURE_VALUE)
    assert "post-answer" in str(excinfo.value) and "14.4" in str(excinfo.value)
    # and anything strictly before the horizon is refused as well
    with pytest.raises(P.TimingViolation):
        P.check_timing(row(A.ANCHOR_REPORT_END, 3, {P.TARGET_CURRENT: 1.0},
                           {P.TARGET_CURRENT: 1}), P.TARGET_CURRENT)


def test_a_pre_answer_anchor_is_prospective_for_the_upcoming_answer_but_not_the_past():
    """Spec §14.4 claim 1: pre-answer features predicting the upcoming answer are genuinely
    prospective — and that is the only claim a pre-answer state gets."""
    for anchor in K.PROSPECTIVE_ANCHORS:
        r = row(anchor, 2, {P.TARGET_CURRENT: 1.0})
        assert r.feature_time == A.FEATURE_TIME_PRE and r.last_allowed_round == 1
        P.check_timing(r, P.TARGET_CURRENT)                     # the round-2 answer: upcoming
        with pytest.raises(P.TimingViolation):
            P.check_timing(row(anchor, 2, {P.TARGET_CURRENT: 1.0}, {P.TARGET_CURRENT: 1}),
                           P.TARGET_CURRENT)
        with pytest.raises(P.TimingViolation):
            P.check_timing(row(anchor, 2, {P.TARGET_EXPOSURE_VALUE: 0.2},
                               {P.TARGET_EXPOSURE_VALUE: 0}), P.TARGET_EXPOSURE_VALUE)


def test_the_frame_builder_is_the_enforcement_point():
    rows, H = W.probe_dataset(n_items=2)
    bad = [P.ProbeRow(**{**r.__dict__, "target_round": {P.TARGET_CURRENT: 0}}) for r in rows]
    with pytest.raises(P.TimingViolation):
        P.build_frame(bad, H, target=P.TARGET_CURRENT, study_seed=SEED, n_folds=2)
    # the good frame is built from exactly the rows that carry the target
    partial = [r for r in rows]
    partial[0] = P.ProbeRow(**{**rows[0].__dict__, "targets": {}})
    built = P.build_frame(partial, H, target=P.TARGET_CURRENT, study_seed=SEED, n_folds=2)
    assert built.n == len(rows) - 1                             # missing target: dropped, not imputed
    with pytest.raises(P.ReadoutError):
        P.build_frame(rows, H, target="SOMETHING_ELSE", study_seed=SEED)


def test_exposure_value_is_kept_continuous_and_binarised_at_zero():
    f = frame(P.TARGET_EXPOSURE_VALUE)
    assert P.TARGET_KIND[P.TARGET_EXPOSURE_VALUE] == "value"
    assert np.array_equal(f.y, (f.value > 0.0).astype(np.int64))
    assert f.value.min() < 0.0 < f.value.max()
    assert not np.array_equal(f.value, f.y)                     # the ranking keeps the magnitude
    # a value target is about the round the protection acts on: t+1
    assert all(g.endswith("|3") for g in f.groups)


# --------------------------------------------------------------------------- identical folds


def test_every_arm_is_fitted_on_identical_folds_and_identical_rows(monkeypatch):
    f = frame()
    calls: dict[str, list] = {}
    real = P.LogisticModel

    class Recording:
        def __init__(self, penalty):
            self.inner = real(penalty=penalty)

        def fit(self, X, y):
            calls[Recording.arm].append((int(X.shape[0]), tuple(int(v) for v in np.asarray(y))))
            self.inner.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.inner.predict_proba(X)

    monkeypatch.setattr(P, "LogisticModel", Recording)
    for arm in P.ARMS:
        Recording.arm = arm
        calls[arm] = []
        P.run_probes(f, layer=SIGNAL_LAYER, anchor=SIGNAL_ANCHOR, arms=(arm,))

    reference = calls[P.ARM_NEURAL]
    assert len(reference) > 4
    for arm in P.ARMS:
        # identical training-row counts and identical label vectors, fold by fold and in
        # the same order: one fold vector and one row index, built once in build_frame
        assert calls[arm] == reference, arm


def test_every_arm_scores_exactly_the_same_rows():
    f = frame()
    report = P.run_probes(f, layer=SIGNAL_LAYER, anchor=SIGNAL_ANCHOR)
    index = f.index_of_site(SIGNAL_LAYER, SIGNAL_ANCHOR)
    scored = None
    for arm in P.ARMS:
        p = np.asarray(report.arms[arm]["probabilities"], dtype=float)
        assert p.size == index.size == report.n_rows
        mask = np.isfinite(p)
        scored = mask if scored is None else scored
        assert np.array_equal(mask, scored), arm
        assert report.arms[arm]["n_scored"] == int(mask.sum())
    assert scored.all()
    assert report.folds == len(set(int(v) for v in f.folds[index]))
    assert report.n_items == len(set(f.items[i] for i in index))


def test_folds_keep_an_item_and_all_its_agents_together():
    f = frame()
    by_item: dict[str, set] = {}
    for item, fold in zip(f.items, f.folds):
        by_item.setdefault(item, set()).add(int(fold))
    assert all(len(folds) == 1 for folds in by_item.values())
    assert len(set(int(v) for v in f.folds)) == 4


# --------------------------------------------------------------------------- controls


def test_the_shuffled_within_item_control_degrades_to_chance():
    f = frame(n_items=32)
    report = P.run_probes(f, layer=SIGNAL_LAYER, anchor=SIGNAL_ANCHOR,
                          arms=(P.ARM_NEURAL, P.ARM_SHUFFLED, P.ARM_BEHAVIOUR, P.ARM_ANSWER_CLASS))

    neural = report.arms[P.ARM_NEURAL]["within_item_auc"]
    shuffled = report.arms[P.ARM_SHUFFLED]["within_item_auc"]
    behaviour = report.arms[P.ARM_BEHAVIOUR]["within_item_auc"]

    assert neural > 0.9
    assert 0.30 < shuffled < 0.70                       # chance, within sampling error
    assert neural - shuffled > 0.25
    assert behaviour < neural                           # the weak behavioural channel
    assert report.arms[P.ARM_SHUFFLED]["brier"] > report.arms[P.ARM_NEURAL]["brier"]
    assert report.arms[P.ARM_NEURAL]["paired_ranking_accuracy"] > 0.9

    incremental = report.incremental[f"{P.ARM_NEURAL}_minus_{P.ARM_SHUFFLED}"]
    assert incremental["within_item_auc"] == pytest.approx(neural - shuffled)
    assert incremental["brier"] < 0
    assert set(report.incremental) == {f"{P.ARM_NEURAL}_minus_{b}"
                                       for b in (P.ARM_BEHAVIOUR, P.ARM_ANSWER_CLASS, P.ARM_SHUFFLED)}


def test_the_shuffle_never_moves_a_state_across_items():
    rows, H = W.probe_dataset(n_items=6)
    f = P.build_frame(rows, H, target=P.TARGET_CURRENT, study_seed=SEED, n_folds=3)
    shuffled = P.shuffle_within_item(f.H, f.items, study_seed=SEED)

    assert shuffled.shape == f.H.shape
    assert not np.allclose(shuffled, f.H)                       # the pairing really is broken
    for item in sorted(set(f.items)):
        mask = np.asarray([i == item for i in f.items], dtype=bool)
        before = sorted(tuple(np.round(v, 9)) for v in f.H[mask])
        after = sorted(tuple(np.round(v, 9)) for v in shuffled[mask])
        assert before == after                                  # a permutation, inside the item
    assert np.allclose(np.sort(shuffled, axis=0), np.sort(f.H, axis=0))
    # deterministic in the study seed
    assert np.array_equal(shuffled, P.shuffle_within_item(f.H, f.items, study_seed=SEED))
    assert not np.array_equal(shuffled, P.shuffle_within_item(f.H, f.items, study_seed=b"\x22" * 32))


def test_the_random_projection_is_fitted_on_training_rows_only():
    rows, H = W.probe_dataset(n_items=8)
    f = P.build_frame(rows, H, target=P.TARGET_CURRENT, study_seed=SEED, n_folds=2)
    train = np.flatnonzero(f.folds == 0)
    held = np.flatnonzero(f.folds == 1)

    projection = P.RandomProjection(dim=8, tag="t").fit(f.H[train], study_seed=SEED)
    assert projection.mean_ == pytest.approx(f.H[train].mean(axis=0))
    assert projection.mean_ != pytest.approx(f.H.mean(axis=0))      # no held-out row entered it
    assert projection.matrix_.shape == (f.H.shape[1], 8)
    assert projection.transform(f.H[held]).shape == (held.size, 8)

    # the matrix is fixed in (seed, tag, dim, hidden), not refit per fold
    again = P.RandomProjection(dim=8, tag="t").fit(f.H[held], study_seed=SEED)
    assert np.array_equal(again.matrix_, projection.matrix_)
    assert not np.array_equal(again.mean_, projection.mean_)
    other = P.RandomProjection(dim=8, tag="other").fit(f.H[train], study_seed=SEED)
    assert not np.array_equal(other.matrix_, projection.matrix_)
    assert P.RandomProjection(dim=P.PROJECTION_DIM).fit(f.H[train], study_seed=SEED).matrix_.shape[1] \
        == min(P.PROJECTION_DIM, f.H.shape[1])
    with pytest.raises(P.ReadoutError):
        P.RandomProjection().transform(f.H)


# --------------------------------------------------------------------------- within-item AUC


def test_within_item_auc_returns_none_on_a_single_class_item_and_counts_it():
    items = ["a"] * 4 + ["b"] * 4
    y = np.asarray([1, 1, 0, 0, 1, 1, 1, 1])              # item "b" is all-correct
    p = np.asarray([0.9, 0.8, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2])
    out = P.within_item_auc(p, y, items)

    assert out["within_item_auc"] == pytest.approx(1.0)   # item "a" alone
    assert out["eligible_items"] == 1
    assert out["excluded_items"] == 1
    assert out["eligible_items"] + out["excluded_items"] == len(set(items)) < len(items)
    assert out["pooled_auc"] is not None and out["pooled_auc"] != out["within_item_auc"]
    assert "not minority recognition" in out["pooled_auc_note"]


def test_within_item_auc_is_none_when_no_item_is_mixed():
    # a perfectly ordered *pooled* problem in which no single item is mixed at all
    out = P.within_item_auc(np.asarray([0.9, 0.8, 0.2, 0.1]), np.asarray([1, 1, 0, 0]), ["a", "a", "b", "b"])
    assert out["within_item_auc"] is None                 # never silently the pooled value
    assert out["eligible_items"] == 0 and out["excluded_items"] == 2
    assert out["pooled_auc"] == pytest.approx(1.0)


def test_the_auc_denominator_never_equals_the_row_count():
    f = frame(n_items=12)
    report = P.run_probes(f, layer=SIGNAL_LAYER, anchor=SIGNAL_ANCHOR, arms=(P.ARM_NEURAL,))
    metrics = report.arms[P.ARM_NEURAL]
    assert metrics["eligible_items"] + metrics["excluded_items"] == report.n_items
    assert metrics["eligible_items"] < report.n_rows
    assert report.n_rows == report.n_items * K.N_AGENTS


def test_within_item_auc_uses_midranks_for_ties():
    out = P.within_item_auc(np.asarray([0.5, 0.5]), np.asarray([1, 0]), ["a", "a"])
    assert out["within_item_auc"] == pytest.approx(0.5)
    pairs = P.paired_ranking_accuracy(np.asarray([0.5, 0.5]), np.asarray([1, 0]), ["a", "a"])
    assert pairs == {"paired_ranking_accuracy": 0.5, "pairs": 1}
    assert P.paired_ranking_accuracy(np.asarray([0.9, 0.8]), np.asarray([1, 1]), ["a", "a"]) == \
        {"paired_ranking_accuracy": None, "pairs": 0}


def test_policy_relevance_excludes_undecidable_groups():
    p = np.asarray([0.9, 0.1, 0.2, 0.8, 0.5, 0.5])
    value = np.asarray([1.0, 0.0, 0.0, 1.0, 0.3, 0.3])
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    out = P.policy_relevance(p, value, groups)
    assert out["decisions"] == 2 and out["excluded_groups"] == 1      # g3 has an all-tied u
    assert out["policy_accuracy"] == pytest.approx(1.0)
    assert out["mean_captured_value"] == pytest.approx(out["mean_oracle_value"])
    assert out["chance_accuracy"] == pytest.approx(0.5)

    single = P.policy_relevance(np.asarray([0.4]), np.asarray([1.0]), ["g"])
    assert single["policy_accuracy"] is None and single["excluded_groups"] == 1

    wrong = P.policy_relevance(np.asarray([0.1, 0.9]), np.asarray([1.0, 0.0]), ["g", "g"])
    assert wrong["policy_accuracy"] == pytest.approx(0.0)
    assert wrong["mean_captured_value"] < wrong["mean_oracle_value"]


def test_brier_score_is_the_mean_squared_error():
    assert P.brier_score(np.asarray([1.0, 0.0]), np.asarray([1, 0])) == pytest.approx(0.0)
    assert P.brier_score(np.asarray([0.5, 0.5]), np.asarray([1, 0])) == pytest.approx(0.25)


# --------------------------------------------------------------------------- no test sweep


def test_the_site_is_selected_on_development_rows_and_locked_for_test():
    sites = (W.SIGNAL_SITE, W.NOISE_SITE)
    dev = frame(n_items=24, sites=sites, signal_sites=(W.SIGNAL_SITE,), split="train",
                item_prefix="dev", seed=1)
    # the *test* half carries its signal at the other site: a sweep on test would move the lock
    test = frame(n_items=24, sites=sites, signal_sites=(W.NOISE_SITE,), split="test",
                 item_prefix="tst", seed=2)
    assert not (set(dev.items) & set(test.items))

    lock = P.select_site(dev, projection_dim=8)
    assert (lock.layer, lock.anchor) == W.SIGNAL_SITE
    assert lock.target == P.TARGET_CURRENT
    assert lock.selected_on == "train+validation"
    assert set(lock.candidates) == set(dev.sites()) == set(sites)
    assert lock.within_item_auc > 0.9

    report = P.evaluate_locked(test, lock, arms=(P.ARM_NEURAL,), projection_dim=8)
    assert (report.layer, report.anchor) == W.SIGNAL_SITE      # the locked site, not the better one
    assert report.feature_time == A.feature_time_of(SIGNAL_ANCHOR)
    # the site that happens to win on test is refused outright
    with pytest.raises(P.ReadoutError, match="no layer sweep on test"):
        P.run_probes(test, layer=NOISE_LAYER, anchor=NOISE_ANCHOR, site_lock=lock, arms=(P.ARM_NEURAL,))
    lock.check(*W.SIGNAL_SITE)
    with pytest.raises(P.ReadoutError):
        lock.check(NOISE_LAYER, SIGNAL_ANCHOR)


def test_a_lock_belongs_to_one_target():
    dev = frame(n_items=8)
    lock = P.select_site(dev, projection_dim=4)
    other = frame(P.TARGET_EXPOSURE_VALUE, n_items=8)
    with pytest.raises(P.ReadoutError, match="lock is for target"):
        P.evaluate_locked(other, lock, arms=(P.ARM_NEURAL,), projection_dim=4)


def test_site_selection_is_deterministic_and_refuses_an_empty_or_absent_site():
    dev = frame(n_items=8, sites=(W.SIGNAL_SITE, W.NOISE_SITE), signal_sites=(W.SIGNAL_SITE,))
    first = P.select_site(dev, projection_dim=4)
    second = P.select_site(dev, projection_dim=4)
    assert (first.layer, first.anchor) == (second.layer, second.anchor)
    assert first.within_item_auc == pytest.approx(second.within_item_auc)
    with pytest.raises(P.ReadoutError, match="no rows at site"):
        P.run_probes(dev, layer=99, anchor=A.ANCHOR_PROMPT_END, arms=(P.ARM_NEURAL,))
    with pytest.raises(P.ReadoutError, match="no candidate sites"):
        P.select_site(dev, sites=())


def test_the_report_round_trips_and_names_its_arms():
    f = frame(n_items=8)
    report = P.run_probes(f, layer=SIGNAL_LAYER, anchor=SIGNAL_ANCHOR, projection_dim=4)
    payload = report.to_dict()
    assert payload["target"] == P.TARGET_CURRENT
    assert set(payload["arms"]) == set(P.ARMS)
    assert payload["feature_time"] == A.FEATURE_TIME_POST
    assert P.ARM_BEHAVIOUR in P.ARMS and P.ARM_SHUFFLED in P.ARMS
    assert f.behaviour_names == ("confidence", "prompt_tokens")
    assert f.class_names == ("A", "B", "C", "D")
    assert f.B.shape == (f.n, 2) and f.C.shape == (f.n, 4)
