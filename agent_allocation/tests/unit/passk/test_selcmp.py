"""selcmp: the population LEX ranking helper and the cross-fit guard (no item is predicted by a model trained on it)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk import features as feat_mod
from agent_allocation.passk import selcmp
from agent_allocation.passk.models.inputs import ClassSource

from .test_e5 import synth


# ---------------------------------------------------------------------- lex_order
def _table(rows):
    return pd.DataFrame(rows, columns=["selector", "acc", "C", "lat"])


def test_lex_order_accuracy_then_consistency_then_latency():
    t = _table([
        ("deep", 0.80, 0.90, 300.0),        # best accuracy, slow
        ("fast_tied", 0.79, 0.89, 100.0),   # within 0.02 on accuracy and consistency -> wins on latency
        ("fast_inconsistent", 0.79, 0.85, 50.0),   # accuracy-tied but 0.05 below on consistency
        ("fastest_wrong", 0.76, 0.95, 10.0),       # 0.04 below on accuracy: never tied with the top
    ])
    assert selcmp.lex_order(t, acc="acc", cons="C", lat="lat") == [
        "fast_tied", "deep", "fast_inconsistent", "fastest_wrong"]
    # without the consistency stage the inconsistent-but-fast option wins the accuracy tie
    assert selcmp.lex_order(t, acc="acc", lat="lat")[0] == "fast_inconsistent"


def test_lex_order_delta_boundary_ties_and_missing():
    t = _table([("a", 0.80, 0.9, 200.0), ("b", 0.78, 0.9, 100.0), ("c", 0.7799, 0.9, 1.0),
                ("x", np.nan, 0.9, 1.0)])
    order = selcmp.lex_order(t, acc="acc", cons="C", lat="lat")
    assert order[:2] == ["b", "a"]          # 0.78 is exactly delta below the best: tied
    assert order[2] == "c" and "x" not in order
    same = _table([("z", 0.8, 0.9, 100.0), ("y", 0.8, 0.9, 100.0)])
    assert selcmp.lex_order(same, acc="acc", cons="C", lat="lat") == ["y", "z"]   # exact tie -> name
    assert selcmp.lex_order(t, acc="acc", cons="C", lat="lat", delta=0.0)[0] == "a"


def test_lex_order_matches_winner_share_on_the_point_estimate():
    """With every bootstrap replicate equal to the sample itself, the winner share puts 1.0 on rank 1."""
    items = pd.DataFrame({"item_id": [f"{b}/{i}" for b in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
                                      for i in range(3)]})
    items["benchmark"] = items.item_id.str.split("/").str[0]
    vals = {"acc": {"s1": 0.8, "s2": 0.79, "s3": 0.5}, "C": {"s1": 0.9, "s2": 0.9, "s3": 0.9},
            "lat": {"s1": 300.0, "s2": 100.0, "s3": 10.0}}
    wide = {m: items.assign(**{s: v for s, v in d.items()}) for m, d in vals.items()}
    share = selcmp.winner_share(wide, ["s1", "s2", "s3"], {b: 0.25 for b in ("aime", "hmmt", "omni_math_hard",
                                                                              "gpqa_diamond")},
                                cons=True, n_resamples=50, seed=1)
    order = selcmp.lex_order(_table([(s, vals["acc"][s], vals["C"][s], vals["lat"][s]) for s in ("s1", "s2", "s3")]),
                             acc="acc", cons="C", lat="lat")
    assert order[0] == "s2" and share == {"s1": 0.0, "s2": 1.0, "s3": 0.0}


# ---------------------------------------------------------------------- cross-fit guard
def test_guard_rejects_training_frames_and_predictions_that_touch_the_item():
    with pytest.raises(selcmp.CrossFitLeakError):
        selcmp.check_training_frames(["x"], pool_frame=pd.DataFrame({"item_id": ["a", "x"]}))
    selcmp.check_training_frames(["x"], pool_frame=pd.DataFrame({"item_id": ["a", "b"]}))
    pred = pd.DataFrame({"item_id": ["x", "y"], "fold": [0, 1]})
    selcmp.assert_crossfit(pred, {0: ["y", "z"], 1: ["x", "z"]})
    with pytest.raises(selcmp.CrossFitLeakError):
        selcmp.assert_crossfit(pred, {0: ["x", "z"], 1: ["x", "z"]})
    with pytest.raises(selcmp.CrossFitLeakError):
        selcmp.assert_crossfit(pred.drop(columns="fold"), {0: []})
    with pytest.raises(selcmp.CrossFitLeakError):
        selcmp.assert_crossfit(pred, {0: ["y"]})          # fold 1 has no recorded training set


def test_fit_predict_refuses_overlapping_train_and_test():
    draws, panels = synth(n_dev=4, n_t200=1)
    cls = ClassSource(draws=draws)
    feats = feat_mod.diag_features(draws, cls)
    dev = panels["L96"]
    with pytest.raises(selcmp.CrossFitLeakError):
        selcmp.fit_predict(draws, None, cls, feats, dev, dev[:2], panels["C48"], selcmp.M15, seed=1, n_sim=20)


@pytest.fixture(scope="module")
def crossfit_pair():
    draws, panels = synth(n_dev=6, n_t200=1)
    items = panels["L96"]
    folds = selcmp.outer_folds(items)
    target = sorted(i for i in items if folds[i] == 0)[0]
    flipped = draws.copy()
    sel = (flipped.item_id == target) & flipped.partition.isin(["legacy_main", "e1_fill", "c48_extra"])
    flipped.loc[sel, "correct"] = ~flipped.loc[sel, "correct"].astype(bool)
    flipped.loc[sel, "grade_state"] = np.where(flipped.loc[sel, "correct"], "correct", "wrong")
    out = []
    for d in (draws, flipped):
        cls = ClassSource(draws=d)
        feats = feat_mod.diag_features(d, cls)
        oof, info = selcmp.crossfit(d, None, cls, feats, items, panels["C48"], selcmp.M15, seed=7, n_sim=50,
                                    folds=folds, log=lambda _m: None)
        out.append((oof, info))
    return out, folds, target, items


def test_crossfit_predicts_every_item_from_a_model_that_never_saw_it(crossfit_pair):
    runs, folds, _, items = crossfit_pair
    oof, info = runs[0]
    assert set(oof.item_id) == set(items)
    assert set(oof.config_id) == set(selcmp.M15)                      # withheld cells are predicted, never trained
    assert (oof.groupby(["item_id", "config_id", "predictor"]).size() == 1).all()
    for f, train in info["train_sets"].items():
        test = set(oof[oof.fold == f].item_id)
        assert test == {i for i in items if folds[i] == f} and not test & set(train)
    assert oof.p_delivered.between(0, 1).all()


def test_an_items_own_labels_never_reach_its_predictions(crossfit_pair):
    """Flipping every development label of one item changes the fold models that train on it (other items'
    predictions move) but leaves that item's own out-of-fold predictions bit-identical."""
    (base, _), (flip, _) = crossfit_pair[0]
    target = crossfit_pair[2]
    key = ["item_id", "config_id", "predictor"]
    a = base.set_index(key).sort_index()
    b = flip.set_index(key).sort_index()
    own = a.index.get_level_values("item_id") == target
    np.testing.assert_array_equal(a[own].p_delivered.to_numpy(), b[own].p_delivered.to_numpy())
    assert not np.allclose(a[~own].p_delivered.to_numpy(), b[~own].p_delivered.to_numpy())
