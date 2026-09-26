"""selcmp: the population LEX ranking helper, the cross-fit guard (no item is predicted by a model trained on it),
the E8b K-sample LLM scoring and the no-diagnostic C48 bank."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk import features as feat_mod
from agent_allocation.passk import lexfront
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


# ---------------------------------------------------------------------- E8b: K-sample LLM selectors
def _call(item, sample, final, choice, lat, arm="B0", panel="C48"):
    return {"arm": arm, "panel": panel, "item_id": item, "sample": sample, "final": final, "choice": choice,
            "latency_s": lat}


def _calls():
    return pd.DataFrame([
        _call("a", 0, True, "1x16", 10.0),                                      # one attempt
        _call("a", 1, False, None, 12.0), _call("a", 1, True, "4x4", 8.0),      # invalid parse, then a valid retry
        _call("b", 0, False, None, 11.0), _call("b", 0, True, None, 13.0),      # invalid twice -> fallback 1x16
        _call("b", 1, False, None, None), _call("b", 1, True, "1x1", 9.0),      # a failed transport, no latency
    ])


def test_e8b_decisions_sum_latency_over_attempts_and_fall_back_to_1x16():
    dec = selcmp.e8b_decisions(_calls(), {"C48": ["a", "b"]}, arms=("B0",), k=2).set_index(["item_id", "sample"])
    assert list(dec.option) == ["1x16", "4x4", "1x16", "1x1"]
    assert list(dec.fallback) == [False, False, True, False]
    assert pd.isna(dec.loc[("b", 0), "choice"])
    assert list(dec.call_latency) == [10.0, 20.0, 24.0, 9.0]              # every attempt counts, retries included
    assert list(dec.attempts) == [1, 2, 2, 2] and dec.attempts_missing_latency.tolist() == [0, 0, 0, 1]
    summary = selcmp.e8b_choice_summary(dec.reset_index().assign(arm="B0", panel="C48"))["B0/C48"]
    assert summary["n_invalid_fallback"] == 1 and summary["n_decisions_retried"] == 3
    assert summary["raw_choice_counts"]["invalid"] == 1 and summary["option_counts"]["1x16"] == 2
    assert summary["choice_stability"] == pytest.approx(0.5)             # both items split 1-1 across 2 decisions


def test_e8b_decisions_refuse_incomplete_or_ambiguous_logs():
    calls = _calls()
    with pytest.raises(RuntimeError, match="missing"):
        selcmp.e8b_decisions(calls[calls.item_id != "b"], {"C48": ["a", "b"]}, arms=("B0",), k=2)
    with pytest.raises(RuntimeError, match="missing"):
        selcmp.e8b_decisions(calls, {"C48": ["a", "b"]}, arms=("B0",), k=3)
    dup = pd.concat([calls, pd.DataFrame([_call("a", 0, True, "2x8", 5.0)])], ignore_index=True)
    with pytest.raises(RuntimeError, match="final attempts"):
        selcmp.e8b_decisions(dup, {"C48": ["a", "b"]}, arms=("B0",), k=2)
    off = calls.copy()
    off.loc[0, "choice"] = "3x5"
    with pytest.raises(RuntimeError, match="not on the menu"):
        selcmp.e8b_decisions(off, {"C48": ["a", "b"]}, arms=("B0",), k=2)


def _value(table):
    def value(i, c):
        p, lat = table[(i, c)]
        return p, p * p + (1 - p) * (1 - p), lat
    return value


def test_llm_k_frame_scores_the_mean_over_decisions_with_summed_overheads():
    dec = selcmp.e8b_decisions(_calls(), {"C48": ["a", "b"]}, arms=("B0",), k=2)
    items = pd.DataFrame({"item_id": ["a", "b"], "benchmark": ["aime", "hmmt"]})
    value = _value({("a", "1x16"): (0.9, 300.0), ("a", "4x4"): (0.5, 100.0),
                    ("b", "1x16"): (0.8, 280.0), ("b", "1x1"): (0.2, 30.0)})
    f = selcmp.llm_k_frame(items, dec, value=value, extra_overhead=None, name="LLM:B0", menu="M15",
                           overhead_kind="llm", consistency=True).set_index("item_id")
    assert f.loc["a", "acc"] == pytest.approx(0.7)                       # mean over the two decisions
    assert f.loc["a", "C"] == pytest.approx(0.7 ** 2 + 0.3 ** 2)         # of the mean p, not the mean of C (0.66)
    assert f.loc["a", "lat"] == pytest.approx(((300 + 10) + (100 + 20)) / 2)
    assert f.loc["a", "overhead"] == pytest.approx(15.0)
    assert f.loc["b", "acc"] == pytest.approx(0.5) and f.loc["b", "n_fallback"] == 1   # the fallback is scored 1x16
    assert f.loc["b", "lat"] == pytest.approx(((280 + 24) + (30 + 9)) / 2)
    assert f.loc["a", "stability"] == 0.5 and f.loc["a", "choice"] == "4x4"            # modal tie -> menu order
    assert f.loc["a", "choices_k"] == "1x16|4x4" and f.loc["a", "solver_A"] == pytest.approx(16.0)
    assert set(f.kind) == {"llm"} and set(f.menu) == {"M15"}
    # B1: the diagnostics' wall-clock is added to every decision; T200-style scoring has no consistency
    g = selcmp.llm_k_frame(items, dec, value=value, extra_overhead={"a": 40.0, "b": 0.0}, name="LLM:B1",
                           menu="M15", overhead_kind="diag+llm", consistency=False).set_index("item_id")
    assert g.loc["a", "lat"] == pytest.approx(f.loc["a", "lat"] + 40.0) and g.C.isna().all()
    # a decision whose option outcome is unknown is skipped, not counted as wrong
    value_unknown = _value({("a", "1x16"): (np.nan, 300.0), ("a", "4x4"): (0.5, 100.0),
                            ("b", "1x16"): (0.8, 280.0), ("b", "1x1"): (0.2, 30.0)})
    h = selcmp.llm_k_frame(items, dec, value=value_unknown, extra_overhead=None, name="LLM:B0", menu="M15",
                           overhead_kind="llm", consistency=True).set_index("item_id")
    assert h.loc["a", "acc"] == pytest.approx(0.5) and h.loc["a", "n_unknown_decisions"] == 1
    with pytest.raises(RuntimeError, match="no decisions"):
        selcmp.llm_k_frame(pd.DataFrame({"item_id": ["a", "z"], "benchmark": ["aime", "aime"]}), dec, value=value,
                           extra_overhead=None, name="LLM:B0", menu="M15", overhead_kind="llm", consistency=True)


# ---------------------------------------------------------------------- the no-diagnostic C48 bank
def _bank():
    rows = []

    def add(partition, a, rep, correct):
        rows.append({"candidate_id": f"{partition}/{a}/{rep}", "item_id": "aime/x", "benchmark": "aime",
                     "partition": partition, "allowance_a": a, "replicate": rep, "correct": float(correct),
                     "cls": "right" if correct else "wrong", "latency_s": 100.0 + rep if correct else 10.0 + rep})
    for r in range(25):
        add("legacy_main", 1, r, r >= 20)             # slots 20-24 = the diagnostics, all correct here
    for r in range(16):
        add("v48_bank", 1, r, False)
    add("v48_bank", 1, 20, False)                     # replicate 20 outside the legacy partition: not a diagnostic
    add("legacy_main", 2, 20, True)                   # slot 20 at 2A: not a diagnostic
    for r in range(3):
        add("e1_fill", 2, r, False)
    return pd.DataFrame(rows)


def test_nodiag_bank_excludes_exactly_the_legacy_A_slots_20_to_24():
    bank = _bank()
    mask = selcmp.is_diagnostic_draw(bank)
    assert int(mask.sum()) == 5
    assert set(bank[mask].candidate_id) == {f"legacy_main/1/{r}" for r in range(20, 25)}
    full = lexfront.estimate(bank).set_index("config")
    nd = selcmp.nodiag_estimates(bank).set_index("config")
    assert full.loc["1x1", "m"] == 42 and nd.loc["1x1", "m"] == 37
    assert full.loc["1x1", "p"] == pytest.approx(5 / 42) and nd.loc["1x1", "p"] == 0.0     # the correct slots are gone
    assert nd.loc["1x1", "latency"] == pytest.approx(bank[~mask & (bank.allowance_a == 1)].latency_s.mean())
    for c in ("1x2", "2x2"):                                                               # other allowances: untouched
        assert nd.loc[c, "m"] == full.loc[c, "m"] == 4
        assert nd.loc[c, "p"] == full.loc[c, "p"] and nd.loc[c, "latency"] == full.loc[c, "latency"]


def test_same_partition_ignores_label_names_but_not_grouping():
    assert selcmp.same_partition(["x", "x", "y", None], ["b", "b", "a", None])
    assert not selcmp.same_partition(["x", "x", "y"], ["b", "c", "a"])
    assert not selcmp.same_partition(["x", "y", "y"], ["a", "a", "a"])


# ---------------------------------------------------------------------- headline with the E8b arms
def test_headline_pairs_each_llm_with_the_statistical_rows_on_its_own_menu():
    rep = _synthetic_report()
    rep["c48"]["selectors"] += [
        _c48_row("stat:P3/M15", "stat", "M15", 0.785, 7, 0.02, _d(-0.005, -0.02, 0.01)),
        {**_c48_row("LLM:B0", "llm", "M15", 0.70, 8, 0.0, _d(-0.10, -0.15, -0.05, 0.005)),
         "choice_stability": 0.46, "n_invalid_fallback": 0}]
    rep["c48"]["stat_minus_llm"]["stat:P3/M15"] = {"A0": {"acc": _d(0.045, 0.0, 0.09)},
                                                   "A1": {"acc": _d(0.065, 0.02, 0.11)},
                                                   "B0": {"acc": _d(0.085, 0.04, 0.13, 0.005)}}
    rep["c48"]["llm_objective_effect"] = {"LLM:B0": {"LLM:A0": {"acc": _d(-0.04, -0.09, 0.01, 0.1),
                                                                "C": _d(-0.07, -0.12, -0.02, 0.02),
                                                                "lat": _d(-50.0, -70.0, -30.0, 0.005)}}}
    f = selcmp.headline_facts(rep)["C48_macro"]
    assert f["stat_like_for_like_by_menu"] == {"M5": ["stat:P3/M5", "stat:P2/M5"], "M15": ["stat:P3/M15"]}
    vs = f["stat_minus_llm_like_for_like"]
    assert [r["selector"] for r in vs["LLM:A0"]["rows"]] == ["stat:P3/M5", "stat:P2/M5"]    # A arms: M5 rows only
    assert [r["selector"] for r in vs["LLM:B0"]["rows"]] == ["stat:P3/M15"] and vs["LLM:B0"]["menu"] == "M15"
    assert vs["LLM:B0"]["n_ci_above_zero"] == 1
    assert f["stat_ranked_below_llm"]["LLM:B0"] == [] and f["stat_ranked_below_llm"]["LLM:A0"] == ["stat:P3/M15"]
    assert f["best_llm"] == "LLM:A0" and f["llm"]["LLM:B0"]["choice_stability"] == 0.46
    assert f["llm_objective_effect"]["LLM:B0 - LLM:A0"]["acc"]["ci_side"] == 0
    text = "\n".join(selcmp.render_findings({"C48_macro": f}))
    assert "on E8b's 15-option menu (stat:P3/M15): vs LLM:B0 +8.5 to +8.5 pts, CI above 0 for 1 of 1" in text
    assert "LLM:B0 - LLM:A0: accuracy -4.0 pts [-9.0, +1.0], CI includes 0" in text
    assert "choice stability 0.46, 0 invalid -> 1x16" in text


def test_tiny_bootstrap_p_values_print_as_below_0001():
    assert selcmp._p_eq(0.0) == "p < 0.001" and selcmp._p_eq(0.17) == "p = 0.170"
    assert selcmp._p_range(0.0, 0.0) == "p <0.001" and selcmp._p_range(0.0, 0.002) == "p <0.001-0.002"
    c = selcmp._contrast({"estimate": 0.1, "ci95": [0.05, 0.15], "p": 0.0})
    assert selcmp._fmt_c(c).endswith("CI above 0, p < 0.001")
