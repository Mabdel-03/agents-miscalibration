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


# ---------------------------------------------------------------------- headline facts and the T200 table
def test_ci_side_counts_a_bound_at_zero_as_touching_not_excluding():
    assert selcmp.ci_side([0.01, 0.08]) == 1 and selcmp.ci_side([-0.08, -0.01]) == -1
    assert selcmp.ci_side([0.0, 0.08]) == 0                      # discrete bootstrap landing exactly on 0
    assert selcmp.ci_side([3.469446951953614e-18, 0.08]) == 0    # ... or on float noise around 0
    assert selcmp.ci_side([-0.08, -3.469446951953614e-18]) == 0
    assert selcmp.ci_side([-0.02, 0.03]) == 0
    assert selcmp._fmt_diff({"estimate": -0.04, "ci95": [-0.08, -3.469446951953614e-18]}, 100) == \
        "-4.0 [-8.0, +0.0]"


def _lvl(m):
    return {"mean": m, "ci95": [m - 0.1, m + 0.1], "n_items": 10}


def _d(est, lo, hi, p=0.5):
    return {"estimate": est, "ci95": [lo, hi], "p": p, "n_items": 10}


def _c48_row(sel, kind, menu, acc, rank, share, d=None):
    r = {"selector": sel, "kind": kind, "menu": menu, "overhead_kind": "none", "acc": _lvl(acc),
         "lex_rank_deployable": rank, "winner_share_deployable": share}
    if d is not None:
        r["minus_1x16"] = {"acc": d}
    return r


def _t_row(sel, kind, menu, mac, mix, rk_mac, rk_mix, sh_mac, sh_mix, d_mac=None, d_mix=None):
    r = {"selector": sel, "kind": kind, "menu": menu, "overhead_kind": "none", "acc_macro": _lvl(mac),
         "acc_mixture": _lvl(mix), "lat_macro": _lvl(200.0), "lex_rank_macro": rk_mac, "lex_rank_mixture": rk_mix,
         "winner_share_macro": sh_mac, "winner_share_mixture": sh_mix}
    if d_mac is not None:
        r["minus_1x16"] = {"acc_macro": d_mac, "acc_mixture": d_mix, "lat_macro": _d(-10.0, -20.0, 0.0)}
    return r


def _synthetic_report():
    """C48: P3 inside the tie band, P2 outside, both above the LLMs. T200: P2 ranks above both LLMs under macro
    weights but below them under mixture weights, and the LLM wins far more bootstrap replicates than any model."""
    c48 = [_c48_row("fixed:1x16", "fixed", "1x16", 0.80, 1, 0.50),
           _c48_row("stat:P3/M5", "stat", "M5", 0.79, 2, 0.30, _d(-0.01, -0.03, 0.01)),
           _c48_row("stat:P2/M5", "stat", "M5", 0.77, 3, 0.10, _d(-0.03, -0.06, 0.0)),
           _c48_row("heuristic/M5", "heuristic", "M5", 0.76, 4, 0.05, _d(-0.04, -0.08, 0.0)),
           _c48_row("LLM:A0", "llm", "M5", 0.74, 5, 0.04, _d(-0.06, -0.10, -0.02, 0.04)),
           _c48_row("LLM:A1", "llm", "M5", 0.72, 6, 0.01, _d(-0.08, -0.13, -0.03, 0.03)),
           _c48_row("oracle/M5", "ceiling", "M5", 0.85, None, None, _d(0.05, 0.01, 0.08))]
    c48_vs = {"stat:P3/M5": {"A0": {"acc": _d(0.05, 0.01, 0.09, 0.04)}, "A1": {"acc": _d(0.07, 0.02, 0.12, 0.03)}},
              "stat:P2/M5": {"A0": {"acc": _d(0.03, -0.01, 0.07)}, "A1": {"acc": _d(0.05, 0.01, 0.10, 0.04)}},
              "heuristic/M5": {"A0": {"acc": _d(0.02, -0.02, 0.06)}, "A1": {"acc": _d(0.04, -0.01, 0.09)}}}
    t = [_t_row("fixed:1x16", "fixed", "1x16", 0.79, 0.70, 2, 1, 0.05, 0.10),
         _t_row("sealed:P3/M5", "stat-sealed", "M5sealed", 0.79, 0.70, 1, 2, 0.01, 0.0,
                _d(-0.004, -0.012, 0.0), _d(-0.005, -0.015, 0.0)),
         _t_row("stat:P3/M15", "stat", "M15", 0.79, 0.70, 3, 3, 0.0, 0.0,
                _d(-0.004, -0.012, 0.0), _d(-0.005, -0.015, 0.0)),
         _t_row("sealed:P2/M5", "stat-sealed", "M5sealed", 0.776, 0.675, 4, 7, 0.006, 0.0,
                _d(-0.018, -0.038, -0.001), _d(-0.03, -0.06, -0.005)),
         _t_row("heuristic/M5", "heuristic", "M5", 0.79, 0.70, 5, 4, 0.30, 0.30,
                _d(-0.004, -0.012, 0.0), _d(-0.005, -0.015, 0.0)),
         _t_row("LLM:A0", "llm", "M5", 0.762, 0.680, 6, 5, 0.21, 0.29,
                _d(-0.032, -0.076, 0.003, 0.15), _d(-0.025, -0.06, 0.01, 0.17)),
         _t_row("LLM:A1", "llm", "M5", 0.731, 0.665, 8, 6, 0.01, 0.04,
                _d(-0.063, -0.118, -0.017, 0.047), _d(-0.04, -0.08, -3.469446951953614e-18, 0.049)),
         _t_row("oracle/M5", "ceiling", "M5", 0.83, 0.765, None, None, None, None,
                _d(0.035, 0.017, 0.056), _d(0.06, 0.03, 0.095))]
    t_vs = {s: {arm: {"acc_macro": dm, "acc_mixture": dx, "lat_macro": _d(60.0, 40.0, 80.0)}
                for arm, (dm, dx) in arms.items()}
            for s, arms in {
                "sealed:P3/M5": {"A0": (_d(0.028, -0.008, 0.072, 0.21), _d(0.02, -0.015, 0.06, 0.30)),
                                 "A1": (_d(0.059, 0.011, 0.115, 0.06), _d(0.035, 3.469446951953614e-18, 0.075))},
                "sealed:P2/M5": {"A0": (_d(0.014, -0.021, 0.058, 0.51), _d(-0.005, -0.04, 0.03, 0.79)),
                                 "A1": (_d(0.045, -0.002, 0.101, 0.12), _d(0.01, -0.03, 0.05, 0.63))},
                "stat:P3/M15": {"A0": (_d(0.028, -0.008, 0.072), _d(0.02, -0.015, 0.06)),
                                "A1": (_d(0.059, 0.011, 0.115), _d(0.035, -0.005, 0.075))},
                "heuristic/M5": {"A0": (_d(0.028, -0.008, 0.072), _d(0.02, -0.015, 0.06)),
                                 "A1": (_d(0.059, 0.011, 0.115), _d(0.035, -0.005, 0.075))}}.items()}
    return {"c48": {"selectors": c48, "stat_minus_llm": c48_vs}, "t200": {"selectors": t, "stat_minus_llm": t_vs}}


def test_headline_facts_state_each_panel_and_weighting_separately():
    f = selcmp.headline_facts(_synthetic_report())
    assert set(f) == {"C48_macro", "T200_macro", "T200_mixture"}
    c = f["C48_macro"]
    assert c["band_floor"] == pytest.approx(0.78)
    assert c["stat_outside_top_band"] == ["stat:P2/M5"]          # -3 pts: outside the band although above the LLMs
    assert c["stat_ranked_below_best_llm"] == []
    assert {s: e["minus_1x16"]["ci_side"] for s, e in c["llm"].items()} == {"LLM:A0": -1, "LLM:A1": -1}
    vs = c["stat_minus_llm_like_for_like"]
    assert vs["LLM:A0"]["n_ci_above_zero"] == 1 and vs["LLM:A0"]["ci_includes_or_touches_zero"] == ["stat:P2/M5"]
    assert vs["LLM:A1"]["n_ci_above_zero"] == 2 and vs["LLM:A0"]["estimate_range"] == [0.03, 0.05]
    assert set(c["reference_rows"]) == {"fixed:1x16", "heuristic/M5"}
    # like-for-like on T200 = the sealed M5 rows only (not the post-hoc M15 refit)
    assert f["T200_macro"]["stat_like_for_like"] == ["sealed:P3/M5", "sealed:P2/M5"]
    # the ranking verdict flips with the weighting
    assert f["T200_macro"]["stat_ranked_below_best_llm"] == []
    assert f["T200_mixture"]["stat_ranked_below_best_llm"] == ["sealed:P2/M5"]
    assert f["T200_mixture"]["llm"]["LLM:A0"]["minus_1x16"]["ci_side"] == 0      # A0's loss is not settled
    assert f["T200_mixture"]["llm"]["LLM:A1"]["minus_1x16"]["ci_side"] == 0      # upper bound touches 0
    assert f["T200_macro"]["llm"]["LLM:A1"]["minus_1x16"]["ci_side"] == -1
    mix_vs = f["T200_mixture"]["stat_minus_llm_like_for_like"]
    assert mix_vs["LLM:A0"]["n_ci_above_zero"] == 0 and mix_vs["LLM:A0"]["estimate_range"] == [-0.005, 0.02]
    assert mix_vs["LLM:A1"]["n_ci_above_zero"] == 0                 # a lower bound at 3e-18 is not "above 0"
    assert f["T200_mixture"]["max_stat_winner_share"]["share"] == 0.0
    assert f["T200_mixture"]["llm"]["LLM:A0"]["winner_share"] == 0.29
    # within 0.02 of 1x16 is not the same as inside the top band when a model beats 1x16
    p2 = next(e for e in f["T200_macro"]["stat_rows"] if e["selector"] == "sealed:P2/M5")
    assert p2["within_delta_of_1x16"] and p2["in_top_band"]
    p2_mix = next(e for e in f["T200_mixture"]["stat_rows"] if e["selector"] == "sealed:P2/M5")
    assert not p2_mix["within_delta_of_1x16"] and not p2_mix["in_top_band"]

    text = "\n".join(selcmp.render_findings(f))
    assert "statistical rows ranked below it: sealed:P2/M5." in text
    assert "statistical rows ranked below it: none." in text
    assert "CI touches 0" in text and "-0.0]" not in text and "[-0.0," not in text
    assert "LLM:A0 0.680, vs 1x16 -2.5 pts [-6.0, +1.0], CI includes 0, p = 0.170" in text


def test_t200_table_shows_both_weightings_and_sorts_by_either_rank():
    rep = _synthetic_report()
    table = selcmp.t200_table(rep)
    header = table.splitlines()[0]
    for col in ("Δacc mixture vs 1x16", "Δacc macro vs 1x16", "rank mixture", "rank macro", "P(win) mixture",
                "P(win) macro"):
        assert col in header
    a0 = next(line for line in table.splitlines() if line.startswith("| LLM:A0 "))
    assert "| 0.29 | 0.21 |" in a0 and "-2.5 [-6.0, +1.0]" in a0 and "-3.2 [-7.6, +0.3]" in a0
    body = lambda t: [line.split("|")[1].strip() for line in t.splitlines()[2:]]
    assert body(table)[:2] == ["sealed:P3/M5", "fixed:1x16"]
    assert body(selcmp.t200_table(rep, order="mixture"))[:2] == ["fixed:1x16", "sealed:P3/M5"]
    assert body(table)[-1] == "oracle/M5"                            # reference rows last
    with pytest.raises(ValueError):
        selcmp.t200_table(rep, order="nope")
