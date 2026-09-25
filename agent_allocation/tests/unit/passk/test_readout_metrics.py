from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd
import pytest

from agent_allocation.contracts import OutputKind
from agent_allocation.passk.metrics import decomposition, entropy, finite_bank_pass_at_k, mixed_coverage, \
    pool_metrics
from agent_allocation.passk.pairs import MissingPair, PairCache
from agent_allocation.passk.pools import CONFIGS, WITHHELD, build_pools, pool_id
from agent_allocation.passk.readout import passk_tie_key, plurality, readout_pools, tie_averaged
from agent_allocation.selection.keys import VoteKey


def _draws(item="aime/2026/1", bench="aime", answers=None, correct=None):
    answers = answers or {}
    rows = []
    for (a, r), text in answers.items():
        rows.append({"candidate_id": f"{item}|{a}|{r}", "item_id": item, "benchmark": bench,
                     "partition": "legacy_main" if (a == 1 or r == 0) else "e1_fill", "panel_roles": ["L96"],
                     "draw_role": "pool", "kind": "root" if a == 1 else "deep", "allowance_a": a, "replicate": r,
                     "absent": text is None, "vote_text": text, "vote_exact_key": text,
                     "correct": (correct or {}).get((a, r))})
    return pd.DataFrame(rows)


def test_passk_tie_is_seeded_per_pool_and_uniformish():
    votes = [("c0", VoteKey("1", "1")), ("c1", VoteKey("2", "2"))]
    wins = {plurality(f"pool{i}", "i", ["c0", "c1"], votes, OutputKind.OPTION, present={"c0", "c1"}).delivered_id
            for i in range(40)}
    assert wins == {"c0", "c1"}
    one = plurality("poolX", "i", ["c0", "c1"], votes, OutputKind.OPTION, present={"c0", "c1"})
    again = plurality("poolX", "i", ["c1", "c0"], list(reversed(votes)), OutputKind.OPTION, present={"c0", "c1"})
    assert one.winner_class == again.winner_class
    first = min(("option:1", "option:2"), key=lambda c: passk_tie_key("poolX", c))
    assert one.winner_class == first
    lo, hi = tie_averaged(one, {"c0": True, "c1": False})
    assert lo == hi == 0.5


def test_single_member_and_absent_rules():
    r = plurality("p", "i", ["c0"], [], OutputKind.MATH, present={"c0"})
    assert r.status == "single" and r.delivered_id == "c0"      # an invalid lone member is still delivered
    r = plurality("p", "i", ["c0"], [], OutputKind.MATH, present=set())
    assert r.status == "abstained" and r.delivered_id is None
    r = plurality("p", "i", ["c0", "c1"], [("c1", VoteKey("5", "5"))], OutputKind.MATH, present={"c1"})
    assert r.delivered_id == "c1"


def test_cache_served_equivalence_and_missing_pair(tmp_path):
    def evaluator(task):
        item, a, b, _ = task
        return item, a, b, {a, b} == {"0.5", "\\frac{1}{2}"}, 0.0
    answers = {(1, 0): "0.5", (1, 1): "\\frac{1}{2}", (1, 2): "3"}
    draws = _draws(answers=answers, correct={(1, 0): True, (1, 1): True, (1, 2): False})
    pools = build_pools(draws, panels=("L96",), configs=("2x1",))
    cache = PairCache(tmp_path / "c.parquet", evaluator=evaluator)
    with pytest.raises(MissingPair):
        readout_pools(pools, draws, cache=cache)
    from agent_allocation.passk.pairs import enumerate_pairs, math_votes_from_frame
    cache.fill(enumerate_pairs(math_votes_from_frame(draws)))
    out = readout_pools(pools, draws, cache=cache)
    assert out.iloc[0]["n_classes"] == 1 and out.iloc[0]["support"] == 2


def test_pools_prefix_rule_and_configs():
    answers = {(1, r): str(r % 3) for r in range(25)}
    answers.update({(2, r): "1" for r in range(8)})
    draws = _draws(answers=answers)
    pools = build_pools(draws, panels=("L96",), configs=("16x1", "8x2", "4x4", "diag5"))
    by = pools.set_index("config_id")
    assert list(by.loc["16x1", "member_ids"]) == [f"aime/2026/1|1|{r}" for r in range(16)]
    assert list(by.loc["8x2", "member_ids"]) == [f"aime/2026/1|2|{r}" for r in range(8)]
    assert "4x4" not in by.index                                     # 4A bank incomplete -> no pool
    assert list(by.loc["diag5", "member_ids"]) == [f"aime/2026/1|1|{r}" for r in range(20, 25)]
    assert by.loc["8x2", "B"] == 16
    assert pools["pool_id"].is_unique
    assert pool_id("i", "1x1", "current", ["a"]) != pool_id("i", "1x1", "current", ["b"])
    assert {(n, a) for n, a in CONFIGS.values()} >= WITHHELD


def test_metrics_decomposition_exact_and_entropy():
    answers = {(1, r): ("7" if r < 3 else "9") for r in range(4)}
    draws = _draws(answers=answers, correct={(1, r): r < 3 for r in range(4)})
    pools = build_pools(draws, panels=("L96",), configs=("4x1", "1x1"))
    m = pool_metrics(readout_pools(pools, draws), draws)
    for row in m.itertuples():
        d = decomposition(row.H_known, row.H_known, row.S_known)
        assert d["generation"] + d["availability"] + d["selection"] == pytest.approx(d["error"], abs=0)
    four = m[m.config_id == "4x1"].iloc[0]
    assert four.S_known == 1.0 and four.top_wrong_mass == 0.25
    assert four.entropy == pytest.approx(entropy([3, 1]))
    with pytest.raises(ValueError):
        decomposition(0.5, 0.6, 0.4)


def test_pass_at_k_and_mixed_coverage():
    assert finite_bank_pass_at_k([True, False, None, False], 2) == (0.5, pytest.approx(5 / 6))
    with pytest.raises(ValueError):
        finite_bank_pass_at_k([True, False], 3)
    g = mixed_coverage({1: 20, 2: 8}, {1: 4, 2: 2}, {1: 3, 2: 2})
    expected = 1 - Fraction(comb(16, 3), comb(20, 3)) * Fraction(comb(6, 2), comb(8, 2))
    assert g == expected
    rng = np.random.default_rng(0)
    bank1 = np.array([1] * 4 + [0] * 16)
    bank2 = np.array([1] * 2 + [0] * 6)
    hits = sum((rng.choice(bank1, 3, replace=False).any() or rng.choice(bank2, 2, replace=False).any())
               for _ in range(20000))
    assert abs(hits / 20000 - float(g)) < 0.015


def test_dev_training_table_has_no_withheld_configs(tmp_path):
    from agent_allocation.passk.extract import _derived_tables
    answers = {(1, r): str(r % 2) for r in range(25)}
    answers.update({(a, r): "1" for a, n in ((2, 8), (4, 4), (8, 2), (16, 1)) for r in range(n)})
    draws = _draws(answers=answers, correct={k: True for k in answers})
    from agent_allocation.passk.pairs import PairCache, enumerate_pairs, math_votes_from_frame

    def evaluator(task):
        item, a, b, _ = task
        return item, a, b, False, 0.0
    PairCache(tmp_path / "catalog" / "pair_verdicts.parquet", evaluator=evaluator).fill(
        enumerate_pairs(math_votes_from_frame(draws)))
    out = tmp_path / "analysis_inputs"
    out.mkdir()
    result = _derived_tables(tmp_path, draws, out)
    assert result["missing"] == 0
    dev = pd.read_parquet(out / "dev_training_readouts.parquet")
    allr = pd.read_parquet(out / "readouts.parquet")
    assert {"4x4", "8x1", "4x2", "2x4"} <= set(allr.config_id)
    assert not {(int(n), int(a)) for n, a in zip(dev.n, dev.a)} & WITHHELD
    assert {"16x1", "8x2", "2x8", "1x16", "1x8"} <= set(dev.config_id)
    classes = pd.read_parquet(out / "classes.parquet")
    assert set(classes.pool_id) <= set(pd.read_parquet(out / "pools.parquet").pool_id)
