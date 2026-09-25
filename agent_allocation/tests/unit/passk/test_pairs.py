from __future__ import annotations

import pandas as pd
import pytest

from agent_allocation.passk.pairs import MissingPair, Pair, PairCache, canonical_pair, enumerate_pairs, pair_sha

CALLS: list = []


def _none_eval(task):
    CALLS.append(task)
    item, a, b, _t = task
    return item, a, b, None, 0.0


def _true_eval(task):
    CALLS.append(task)
    item, a, b, _t = task
    return item, a, b, True, 0.0


def _eq_eval(task):
    item, a, b, _t = task
    return item, a, b, a.replace(" ", "") == b.replace(" ", ""), 0.01


def test_enumerate_pairs_skips_same_exact_key_and_gpqa_free():
    votes = [("m/1", "1/2", "1/2"), ("m/1", "0.5", "0.5"), ("m/1", "1/2", "1/2"), ("m/1", "3", "3"),
             ("m/2", "x", "x")]
    pairs = enumerate_pairs(votes)
    assert pairs == sorted([Pair("m/1", "0.5", "1/2"), Pair("m/1", "0.5", "3"), Pair("m/1", "1/2", "3")])


def test_enumerate_pairs_rejects_one_text_two_keys():
    with pytest.raises(ValueError):
        enumerate_pairs([("m/1", "a", "t"), ("m/1", "b", "t")])


def test_order_invariant_lookup_and_missing(tmp_path):
    cache = PairCache(tmp_path / "c.parquet", evaluator=_eq_eval)
    cache.fill([canonical_pair("m/1", "b", "a"), canonical_pair("m/1", "x y", "xy")])
    assert cache.lookup("m/1", "a", "b") is False
    assert cache.lookup("m/1", "b", "a") is False
    assert cache.lookup("m/1", "xy", "x y") is True
    assert cache.lookup("m/1", "q", "q") is True
    with pytest.raises(MissingPair):
        cache.lookup("m/1", "a", "zz")
    with pytest.raises(MissingPair):
        cache.lookup("m/2", "a", "b")
    eq = cache.equivalence("m/1")
    assert eq("a", "b") is False and eq("b", "a") is False


def test_stored_none_is_final_and_append_only(tmp_path):
    path = tmp_path / "c.parquet"
    CALLS.clear()
    cache = PairCache(path, evaluator=_none_eval)
    pairs = [Pair("m/1", "1", "2"), Pair("m/1", "1", "3")]
    summary = cache.fill(pairs)
    assert summary["none"] == 2 and len(CALLS) == 2
    before = pd.read_parquet(path)
    again = PairCache(path, evaluator=_true_eval)
    CALLS.clear()
    summary = again.fill(pairs + [Pair("m/1", "2", "3")])
    assert len(CALLS) == 1 and summary["requested_new"] == 1
    assert again.lookup("m/1", "2", "1") is None
    after = pd.read_parquet(path)
    assert len(after) == 3
    pd.testing.assert_frame_equal(after.iloc[:2].reset_index(drop=True), before.reset_index(drop=True))
    assert list(after["pair_sha"]) == [pair_sha(i, a, b) for i, a, b in zip(after.item_id, after.text_a,
                                                                            after.text_b)]
    assert set(after["timeout_s"]) == {5.0}


def test_real_math_verify_in_spawned_workers(tmp_path):
    pytest.importorskip("math_verify")
    cache = PairCache(tmp_path / "c.parquet")
    pairs = [canonical_pair("m/1", "\\frac{1}{2}", "0.5"), canonical_pair("m/1", "2", "3")]
    summary = cache.fill(pairs, workers=2)
    assert summary["requested_new"] == 2
    assert cache.lookup("m/1", "0.5", "\\frac{1}{2}") is True
    assert cache.lookup("m/1", "3", "2") is False


def test_isolated_mode_in_spawned_workers(tmp_path):
    pytest.importorskip("math_verify")
    cache = PairCache(tmp_path / "c.parquet")
    summary = cache.fill([canonical_pair("m/1", "\\frac{1}{2}", "0.5"), canonical_pair("m/1", "2", "3")], workers=2,
                         isolate=True)
    assert summary["requested_new"] == 2
    assert cache.lookup("m/1", "0.5", "\\frac{1}{2}") is True and cache.lookup("m/1", "2", "3") is False
