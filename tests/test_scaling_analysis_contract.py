"""Token-dose fits fail closed on caches without exact-token provenance."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from analysis.scripts import scaling_equivalence


def _frame(*, include_exact: bool) -> pd.DataFrame:
    data = {
        "model_size": ["4B", "4B"],
        "reasoning_level": ["b512", "b512"],
        "topology": ["single_agent", "single_agent"],
        "context_share_level": ["artifact_only", "artifact_only"],
        "mean_reasoning_tokens": [512.0, np.nan],
        "ctx32b_capped": [False, False],
    }
    if include_exact:
        data["reasoning_tokens_exact"] = [True, False]
    return pd.DataFrame(data)


def test_load_core_masks_nonexact_reasoning_counts(monkeypatch):
    monkeypatch.setattr(
        scaling_equivalence.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _frame(include_exact=True),
    )
    monkeypatch.setattr(
        scaling_equivalence.nb_data,
        "analysis_view",
        lambda frame, **_kwargs: frame,
    )
    loaded = scaling_equivalence.load_core()
    assert loaded.loc[0, "log2_reas"] == pytest.approx(np.log2(513.0))
    assert np.isnan(loaded.loc[1, "log2_reas"])


def test_load_core_rejects_pre_provenance_cache(monkeypatch):
    monkeypatch.setattr(
        scaling_equivalence.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _frame(include_exact=False),
    )
    monkeypatch.setattr(
        scaling_equivalence.nb_data,
        "analysis_view",
        lambda frame, **_kwargs: frame,
    )
    with pytest.raises(RuntimeError, match="predates exact reasoning-token provenance"):
        scaling_equivalence.load_core()
