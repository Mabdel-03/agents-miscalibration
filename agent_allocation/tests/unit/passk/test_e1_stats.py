from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk.analyze import e1_lock
from agent_allocation.passk.stats import MIXTURE, contrast, holm2, level

BENCH = {"aime": "aime/1", "hmmt": "hmmt/1", "omni_math_hard": "omni_math_hard/1", "gpqa_diamond": "gpqa_diamond/r"}


def _bank(correct_at: dict[int, bool]) -> pd.DataFrame:
    """Every L96 item gets 2A r0-7, 4A r0-3, 8A r0-1, 16A r0; all members of an allowance answer alike."""
    rows = []
    for bench, item in BENCH.items():
        for a, n in ((2, 8), (4, 4), (8, 2), (16, 1)):
            for r in range(n):
                ok = correct_at[a]
                text = "A" if bench == "gpqa_diamond" and ok else ("B" if bench == "gpqa_diamond" else
                                                                     ("1" if ok else "2"))
                rows.append({"candidate_id": f"{item}|{a}|{r}", "item_id": item, "benchmark": bench,
                             "partition": "legacy_main" if r == 0 else "e1_fill", "panel_roles": ["L96"],
                             "draw_role": "deep_r0", "kind": "deep", "allowance_a": a, "replicate": r,
                             "absent": False, "vote_text": text, "vote_exact_key": text, "correct": ok})
    return pd.DataFrame(rows)


def test_e1_lock_picks_argmax_and_is_write_once(tmp_path):
    draws = _bank({2: False, 4: True, 8: False, 16: True})
    payload = e1_lock(tmp_path, draws=draws, cache=None)
    assert payload["selected"] == "4x4"
    saved = json.loads((tmp_path / "sealed" / "e1_selection.json").read_text())
    assert saved["selected"] == "4x4" and saved["table"]["4x4"]["mixture"] == 1.0
    with pytest.raises(RuntimeError):
        e1_lock(tmp_path, draws=_bank({2: True, 4: False, 8: False, 16: True}), cache=None)


def test_e1_lock_tie_goes_to_fewer_calls(tmp_path):
    payload = e1_lock(tmp_path, draws=_bank({2: True, 4: True, 8: True, 16: True}), cache=None)
    assert payload["selected"] == "2x8"


def test_e1_lock_refuses_incomplete_fill(tmp_path):
    draws = _bank({2: True, 4: True, 8: True, 16: True})
    draws = draws[~((draws.allowance_a == 2) & (draws.replicate == 7) & (draws.benchmark == "aime"))]
    with pytest.raises(RuntimeError):
        e1_lock(tmp_path, draws=draws, cache=None)


def test_contrast_mixture_and_holm():
    rng = np.random.default_rng(1)
    rows = []
    for bench, n in (("aime", 20), ("hmmt", 20), ("omni_math_hard", 100), ("gpqa_diamond", 60)):
        for i in range(n):
            a = float(rng.random() < 0.7)
            rows.append({"item_id": f"{bench}/{i}", "benchmark": bench, "a": a, "b": float(rng.random() < 0.5),
                         "c": a})
    items = pd.DataFrame(rows)
    res = contrast(items, "a", "b", weights=MIXTURE, n_resamples=2000)
    assert res["weights"] == pytest.approx(MIXTURE) and res["n_items"] == 200
    assert res["ci95_studentized"][0] < res["estimate"] < res["ci95_studentized"][1]
    same = contrast(items, "a", "c", n_resamples=500)
    assert same["estimate"] == 0.0 and same["p"] == 1.0
    lv = level(items, "a", n_resamples=500)
    assert 0 < lv["mean"] < 1
    h = holm2(0.01, 0.04)
    assert h["mixed_vs_deep"]["p_holm"] == pytest.approx(0.02) and h["p2_vs_p1"]["p_holm"] == pytest.approx(0.04)


def test_e1_lock_waits_for_pending_or_ungraded(tmp_path):
    from agent_allocation.passk.analyze import NotReady
    draws = _bank({2: True, 4: True, 8: True, 16: True})
    draws["pending"] = False
    draws["graded"] = True
    draws.loc[3, "pending"] = True
    with pytest.raises(NotReady):
        e1_lock(tmp_path, draws=draws, cache=None)
    draws.loc[3, "pending"] = False
    draws.loc[4, "graded"] = False
    with pytest.raises(NotReady):
        e1_lock(tmp_path, draws=draws, cache=None)
