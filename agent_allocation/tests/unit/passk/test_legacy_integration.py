"""WP-M models on the REAL legacy L96 data through the dev shim (read-only; skipped when the legacy root is absent)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.unit.passk import devshim

pytestmark = pytest.mark.skipif(not devshim.available(), reason="legacy forced_v2 run root not available")


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    cache = None if "PASSK_DEVSHIM_CACHE" not in os.environ else devshim.CACHE
    return devshim.legacy_draws(cache=cache)


def test_devshim_shape(legacy):
    assert legacy.item_id.nunique() == 96
    counts = legacy.groupby(["kind", "allowance_a"]).size().to_dict()
    assert counts[("root", 1)] == 2400 and counts[("deep", 16)] == 96


def test_e5_validate_split_on_legacy(legacy):
    from agent_allocation.passk.models import e5
    items = sorted(legacy.item_id.unique())
    r = e5.validate_split(legacy, items, c48=devshim.pseudo_c48(legacy), n_sim=200, min_known=2)
    s = r["score"]["summary"].set_index("predictor")
    assert len(r["test"]) == 24 and not set(r["test"]) & set(r["train"])
    assert s.primary_brier_mixture.between(0, 1).all()
    assert np.isfinite(s.loc["P2", "primary_brier_mixture"])


def test_e2_fit_on_legacy_pseudo_c48(legacy, tmp_path):
    from agent_allocation.passk.models import e2
    c48 = devshim.pseudo_c48(legacy)
    res = e2.fit_and_predict(draws=legacy, c48=c48, out_dir=tmp_path / "v48", S=60, inner=8)
    pred = res["predictions"]
    assert pred.item_id.nunique() == 48
    assert pred.pred_plurality.between(0, 1).all()
