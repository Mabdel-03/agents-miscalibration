"""E5: withheld/fence guards, rank([1, log n, log a]) = 3, label-free features, synthetic end-to-end fit/predict/score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk import features as feat_mod
from agent_allocation.passk.models import e5
from agent_allocation.passk.models.inputs import ClassSource, PREDICT_CONFIGS
from agent_allocation.passk.models.predictors import (
    PREDICTORS, WithheldTargetError, config_rank, guard_training_frame,
)

BENCH = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")


def _row(part, item, b, a, r, ok, rng, label=True):
    key = "gold" if ok else rng.choice(["w0", "w1", "w2"], p=[0.6, 0.3, 0.1])
    return dict(candidate_id=f"{part}|{item}|{a}|{r}", source="x", partition=part, item_id=item, benchmark=b,
                shard="mancer" if hash(item) % 2 else "parasail", kind="root" if a == 1 else "deep", allowance_a=a,
                replicate=r, vote_exact_key=key, option=None, parse_status="valid",
                answer_len_chars=int(rng.integers(200, 900)), answer_finish_reason="stop",
                latency_s=float(rng.uniform(5, 50) * a), correct=bool(ok) if label else None,
                grade_state=("correct" if ok else "wrong") if label else None)


def synth(n_dev=5, n_t200=3, seed=0, t200_labels=False):
    rng = np.random.default_rng(seed)
    rows, dev, c48, t200 = [], [], [], []
    for b in BENCH:
        for j in range(n_dev + n_t200):
            item = f"{b}/{j}"
            base = rng.normal(-0.3, 1.5)
            p = lambda a: 1 / (1 + np.exp(-(base + 0.5 * np.log2(a))))
            if j < n_dev:
                dev.append(item)
                if j < 3:
                    c48.append(item)
                for r in range(25):
                    rows.append(_row("legacy_main", item, b, 1, r, rng.uniform() < p(1), rng))
                for a in (2, 4, 8, 16):
                    rows.append(_row("legacy_main", item, b, a, 0, rng.uniform() < p(a), rng))
                for a, reps in ((2, range(1, 8)), (4, range(1, 4)), (8, range(1, 2))):
                    rows += [_row("e1_fill", item, b, a, r, rng.uniform() < p(a), rng) for r in reps]
                if item in c48:
                    for a, reps in ((4, range(4, 8)), (8, range(2, 8)), (16, range(1, 8))):
                        rows += [_row("c48_extra", item, b, a, r, rng.uniform() < p(a), rng) for r in reps]
            else:
                t200.append(item)
                rows += [_row("t200_diag", item, b, 1, r, rng.uniform() < p(1), rng, label=t200_labels)
                         for r in range(5)]
                for a, m in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
                    rows += [_row("t200_bank", item, b, a, r, rng.uniform() < p(a), rng, label=t200_labels)
                             for r in range(m)]
    return pd.DataFrame(rows), {"L96": dev, "C48": c48, "T200": t200}


def test_guard_raises_on_withheld_and_fenced():
    ok = pd.DataFrame({"n": [1, 2, 16], "a": [1, 1, 1], "item_id": ["x"] * 3})
    guard_training_frame(ok)
    for n, a in ((4, 4), (8, 1), (4, 2), (2, 4)):
        with pytest.raises(WithheldTargetError):
            guard_training_frame(pd.DataFrame({"n": [n], "a": [a]}))
    with pytest.raises(WithheldTargetError):
        guard_training_frame(pd.DataFrame({"n": [1], "a": [1], "member_partitions": [["legacy_main", "v48_bank"]]}))
    with pytest.raises(WithheldTargetError):
        guard_training_frame(pd.DataFrame({"partition": ["t200_bank"], "y": [1.0]}), draw_level=True)
    with pytest.raises(WithheldTargetError):
        guard_training_frame(pd.DataFrame({"partition": ["t200_diag"], "correct": [True]}), draw_level=True)
    with pytest.raises(WithheldTargetError):
        guard_training_frame(pd.DataFrame({"n": [1], "a": [1], "t200_diag_correct": [1.0]}))


def test_training_frame_rank_and_no_withheld():
    draws, panels = synth()
    cls = ClassSource(draws=draws)
    feats = feat_mod.diag_features(draws, cls)
    pf = e5.dev_pool_frame(draws, None, cls, panels["L96"], panels["C48"], feats[feats.item_id.isin(panels["L96"])])
    assert config_rank(pf) == 3
    assert not set(pf.config_id) & set(e5.WITHHELD_IDS)
    assert {"8x2", "2x8", "2x16", "4x8", "8x4"} <= set(pf.config_id)
    assert set(pf[pf.config_id.isin(e5.C48_ONLY_CONFIGS)].item_id) <= set(panels["C48"])
    # a frame with only one configuration shape is rank-deficient
    assert config_rank(pd.DataFrame({"n": [2, 4], "a": [1, 1]})) == 2


def test_features_are_label_free_and_cover_both_collections():
    draws, panels = synth()
    cls = ClassSource(draws=draws)
    f1 = feat_mod.diag_features(draws, cls)
    flipped = draws.assign(correct=~draws.correct.astype(bool))
    f2 = feat_mod.diag_features(flipped, cls)
    pd.testing.assert_frame_equal(f1, f2)
    assert set(f1.collection) == {"legacy", "t200_diag"}
    assert set(f1.item_id) == set(panels["L96"]) | set(panels["T200"])
    r = f1.iloc[0]
    assert r.f1 >= r.f2 >= r.f3 and 0 <= r.entropy <= 1
    assert set(feat_mod.FEATURES) <= set(f1.columns)


def test_e5_fit_predict_score_end_to_end(tmp_path):
    draws, panels = synth()
    res = e5.fit_and_predict(draws=draws, panels=panels, out_dir=tmp_path / "t200", n_sim=200)
    pred = res["predictions"]
    assert set(pred.predictor) == set(PREDICTORS)
    assert len(pred) == len(panels["T200"]) * len(PREDICT_CONFIGS) * len(PREDICTORS)
    assert pred.p_delivered.between(0, 1).all() and pred.p_coverage.between(0, 1).all()
    p2 = pred[pred.predictor == "P2"]
    assert (p2.p_delivered <= p2.p_coverage + 1e-12).all()
    for f in ("predictions.parquet", "features.parquet", "manifest.json", "models/P2.json", "seeds.json"):
        assert (tmp_path / "t200" / f).exists()
    with pytest.raises(FileExistsError):
        e5.fit_and_predict(draws=draws, panels=panels, out_dir=tmp_path / "t200")
    labelled, _ = synth(t200_labels=True)
    with pytest.raises(RuntimeError):
        e5.fit_and_predict(draws=labelled, panels=panels, out_dir=tmp_path / "x", write=False)
    sc = e5.score(draws=labelled, t200=panels["T200"], pred_dir=tmp_path / "t200", out_dir=tmp_path / "score")
    s = sc["summary"].set_index("predictor")
    assert s.primary_brier_mixture.between(0, 1).all()
    assert s.n_items_primary.min() == len(panels["T200"])
    assert {"P2-P1", "P1-conc-P1", "P2-conc-P2"} <= set(sc["contrasts"].contrast)
    assert (sc["policy"].policy == "select_per_item").sum() == len(PREDICTORS)


def test_items_without_diagnostics_are_imputed_and_predicted(tmp_path):
    draws, panels = synth()
    missing = panels["T200"][0]
    draws = draws[~((draws.item_id == missing) & (draws.partition == "t200_diag"))]
    res = e5.fit_and_predict(draws=draws, panels=panels, write=False, n_sim=100)
    assert missing in set(res["predictions"].item_id)
    assert bool(res["features"].set_index("item_id").loc[missing, "imputed"])


def test_holdout_folds_are_disjoint_and_cover():
    items = {f"{b}/{j}": b for b in BENCH for j in range(24)}
    seen = []
    for k in range(4):
        train, test = e5.holdout_split(items, 6, fold=k)
        assert not set(train) & set(test) and len(test) == 24 and len(train) == 72
        seen += test
    assert sorted(seen) == sorted(items)


def test_e5_is_deterministic():
    draws, panels = synth(n_dev=4, n_t200=2)
    a = e5.fit_and_predict(draws=draws, panels=panels, write=False, n_sim=100)["predictions"]
    b = e5.fit_and_predict(draws=draws, panels=panels, write=False, n_sim=100)["predictions"]
    pd.testing.assert_frame_equal(a, b)


def test_class_table_is_used_when_complete_and_falls_back_otherwise():
    draws, panels = synth(n_dev=2, n_t200=1)
    item = panels["L96"][0]
    ids = list(draws[draws.item_id == item].candidate_id[:3])
    table = pd.DataFrame({"scope_id": ["s1"] * 3, "candidate_id": ids, "class_id": ["c", "c", None]})
    cls = ClassSource(draws=draws, table=table)
    assert cls.classes("s1", ids) == {ids[0]: "c", ids[1]: "c", ids[2]: None}
    cls.classes("s2", ids)
    assert cls.stats() == {"source": "analysis_inputs/classes.parquet", "scopes_from_table": 1,
                           "scopes_exact_fallback": 1}


def test_class_source_reads_wpa_class_rows_and_aliases():
    draws, panels = synth(n_dev=2, n_t200=1)
    item = panels["C48"][0]
    ids = list(draws[(draws.item_id == item) & (draws.allowance_a == 1)].sort_values("replicate").candidate_id[:4])
    pools = pd.DataFrame({"pool_id": ["P-fit", "P-diag"], "item_id": [item, item], "config_id": ["c48fit", "diag5"]})
    table = pd.DataFrame({"pool_id": ["P-fit", "P-fit"], "class_id": ["math:1", "math:2"],
                          "member_ids": [[ids[0], ids[2]], [ids[1]]]})
    cls = ClassSource(draws=draws, table=table, pools=pools)
    got = cls.classes(f"e2fit:{item}", ids)
    assert got == {ids[0]: "math:1", ids[1]: "math:2", ids[2]: "math:1", ids[3]: None}   # ids[3]: no vote
    assert cls.stats()["scopes_from_table"] == 1
    cls.classes(f"diag5:{item}", ids)                     # alias exists but no class rows -> exact fallback
    assert cls.stats()["scopes_exact_fallback"] == 1


def test_observed_outcomes_use_seeded_readouts_and_absent_members():
    from agent_allocation.passk.models.inputs import observed_pool_table
    draws = pd.DataFrame({"candidate_id": ["a", "b", "c"], "item_id": ["x"] * 3, "vote_exact_key": ["1", "2", None],
                          "option": [None] * 3, "correct": [True, False, None], "absent": [False, False, True]})
    pools = pd.DataFrame({"pool_id": ["p", "q"], "item_id": ["x", "x"], "config_id": ["2x1", "1x1"], "n": [2, 1],
                          "a": [1, 1], "member_ids": [["a", "b"], ["c"]]})
    cls = ClassSource(draws=draws)
    obs = observed_pool_table(pools, draws, cls).set_index("pool_id")
    assert obs.loc["p", "plurality"] == 0.5 and obs.loc["q", "coverage"] == 0.0
    ro = pd.DataFrame({"pool_id": ["p"], "delivered_correct": [True]})
    obs = observed_pool_table(pools, draws, cls, ro).set_index("pool_id")
    assert obs.loc["p", "plurality"] == 1.0 and obs.loc["p", "plurality_tieavg"] == 0.5
    assert obs.loc["p", "plurality_source"] == "readouts_seeded"
