from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from agent_allocation.passk.extract import extract, load_draws
from agent_allocation.passk.pairs import PairCache, enumerate_pairs, math_votes_from_frame
from agent_allocation.passk.pools import B16_MENU
from agent_allocation.passk.score import score

from . import fakeroot


def _eval(task):
    item, a, b, _ = task
    return item, a, b, False, 0.0


def _seal(run_root, rels):
    arts = {rel: hashlib.sha256((run_root / rel).read_bytes()).hexdigest() for rel in rels}
    (run_root / "sealed" / "SEAL_predictions.json").write_text(json.dumps(
        {"code_commit": "abc", "reviewed_commit": "abc", "artifacts": arts}))


def test_release_then_score_primary_family(tmp_path):
    items = fakeroot.build_t200(tmp_path, n_per_bench=4)
    extract(tmp_path, include_legacy=False)
    pre = load_draws(tmp_path)
    assert not (pre.partition == "t200_bank").any()
    assert pre.loc[pre.partition == "t200_diag", "correct"].isna().all()
    sealed = tmp_path / "sealed"
    (sealed / "t200").mkdir(parents=True)
    (sealed / "e1_selection.json").write_text(json.dumps({"selected": "4x4"}))
    rows = [{"item_id": i, "config_id": c, "predictor": p, "p_delivered": 0.5 if p == "P1" else 0.6,
             "p_coverage": 0.7} for i in items for c in B16_MENU for p in ("P0", "P1", "P2")]
    pd.DataFrame(rows).to_parquet(sealed / "t200" / "predictions.parquet")
    with pytest.raises(PermissionError):
        score(tmp_path)
    _seal(tmp_path, ["sealed/e1_selection.json", "sealed/t200/predictions.parquet"])
    extract(tmp_path, include_legacy=False, release=["t200"])
    draws = load_draws(tmp_path)
    assert draws.loc[draws.partition == "t200_bank", "correct"].notna().all()
    PairCache(tmp_path / "catalog" / "pair_verdicts.parquet", evaluator=_eval).fill(
        enumerate_pairs(math_votes_from_frame(draws)))
    primary = score(tmp_path, n_resamples=300)
    assert primary["locked"] == "4x4"
    assert primary["mixed_vs_deep"]["n_items"] == len(items)
    assert set(primary["holm"]) == {"mixed_vs_deep", "p2_vs_p1"}
    assert (tmp_path / "scored" / "menus.csv").exists()
    menus = pd.read_csv(tmp_path / "scored" / "menus.csv")
    assert set(menus[menus.panel == "T200"].config_id) >= set(B16_MENU)
