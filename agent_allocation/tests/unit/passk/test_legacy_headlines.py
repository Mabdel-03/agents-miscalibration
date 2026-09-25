"""E0 headline reproduction on the real legacy L96 banks (skipped when the legacy run root is absent).

The pair cache for the plurality checks comes from ``$PASSK_PAIR_CACHE`` or the run root's
``catalog/pair_verdicts.parquet`` (filled by ``passk-analyze --stage pairs``); those checks skip without it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.passk.extract import RUN_ROOT, legacy_draws
from agent_allocation.passk.metrics import config_summary, pool_metrics
from agent_allocation.passk.pairs import PairCache
from agent_allocation.passk.pools import build_pools
from agent_allocation.passk.readout import readout_pools

EXCLUDED_94 = {"hmmt/2025/18", "omni_math_hard/473"}


@pytest.fixture(scope="module")
def draws(legacy_root):
    cached = os.environ.get("PASSK_LEGACY_DRAWS")
    if cached and Path(cached).exists():
        return pd.read_parquet(cached)
    return legacy_draws(legacy_root, read_raw=False)


@pytest.fixture(scope="module")
def cache():
    path = Path(os.environ.get("PASSK_PAIR_CACHE") or RUN_ROOT / "catalog" / "pair_verdicts.parquet")
    if not path.exists():
        pytest.skip("no filled pair cache (run passk-analyze --stage pairs)")
    return PairCache(path)


def _macro(draws, configs, cache, items=None):
    pools = build_pools(draws, panels=("L96",), configs=configs)
    m = pool_metrics(readout_pools(pools, draws, cache=cache, mode="legacy"), draws)
    return config_summary(m, items=items).set_index("config_id")


def test_legacy_bank_shape(draws):
    main = draws[draws.partition == "legacy_main"]
    assert main.item_id.nunique() == 96
    assert (main[main.kind == "root"].groupby("item_id").size() == 25).all()
    assert (main[main.kind == "deep"].groupby("item_id").allowance_a.apply(sorted).map(tuple)
            == (2, 4, 8, 16, 25)).all()
    assert draws[draws.partition == "legacy_pilot"].item_id.nunique() == 32


def test_deep16_macro_96_and_94(draws):
    s96 = _macro(draws, ("1x16",), None)
    items94 = set(draws.item_id[draws.partition == "legacy_main"]) - EXCLUDED_94
    s94 = _macro(draws, ("1x16",), None, items=items94)
    assert round(s96.loc["1x16", "S_known_macro"], 4) == 0.7917
    assert round(s94.loc["1x16", "S_known_macro"], 4) == 0.7853


def test_sample16_and_vote25_legacy_tie_mode(draws, cache):
    items94 = set(draws.item_id[draws.partition == "legacy_main"]) - EXCLUDED_94
    s96 = _macro(draws, ("16x1", "25x1"), cache)
    s94 = _macro(draws, ("25x1",), cache, items=items94)
    assert round(s96.loc["16x1", "S_known_macro"], 4) == 0.4167
    assert round(s96.loc["25x1", "S_known_macro"], 4) == 0.4271
    assert round(s94.loc["25x1", "S_known_macro"], 4) == 0.4330


def test_legacy_replay_matches_sealed_selections(legacy_root, draws, cache):
    from agent_allocation.passk.e0.replay import replay
    out = replay(legacy_root, draws, cache)
    clean = out[~out.drift_eligible]
    assert len(clean) > 0 and clean.match.all()
