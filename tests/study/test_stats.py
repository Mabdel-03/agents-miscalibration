"""stats.py: cluster bootstrap, max-T, Holm, prefix rule (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pytest

from agents_scaling.study import stats as S


def _rs(n_by_domain, B=2000, seed=7):
    return S.Resampler(n_by_domain, B, seed)


def test_paired_contrast_detects_effect_and_null_is_centred():
    rng = np.random.default_rng(1)
    eff = {"hle": rng.normal(0.15, 0.3, 200), "bcb": rng.normal(0.15, 0.3, 200)}
    r = S.paired_contrast(eff, _rs({"hle": 200, "bcb": 200}))
    assert r["p"] < 0.001 and r["ci95_studentized"][0] > 0 and abs(r["estimate"] - 0.15) < 0.06
    assert set(r["n"]) == {"hle", "bcb"} and abs(r["weights"]["hle"] - 0.5) < 1e-12
    null = {"hle": rng.normal(0.0, 0.3, 200), "bcb": rng.normal(0.0, 0.3, 200)}
    r0 = S.paired_contrast(null, _rs({"hle": 200, "bcb": 200}))
    assert r0["p"] > 0.05 and r0["ci95_studentized"][0] < 0 < r0["ci95_studentized"][1]
    one = S.paired_contrast(eff, _rs({"hle": 200, "bcb": 200}), alternative="greater")
    assert one["p"] <= r["p"]


def test_paired_contrast_single_domain_and_degenerate():
    r = S.paired_contrast({"hle": np.array([0.1, 0.2, 0.3, 0.4])}, _rs({"hle": 4}))
    assert r["weights"] == {"hle": 1.0} and r["n"] == {"hle": 4}
    d = S.paired_contrast({"hle": np.zeros(10), "bcb": np.zeros(10)}, _rs({"hle": 10, "bcb": 10}))
    assert d["p"] == 1.0 and d["estimate"] == 0.0


def test_max_t_global_p_is_at_most_min_adjusted_pair_p_and_cis_cover_estimates():
    rng = np.random.default_rng(3)
    k = 4
    Y = {d: (rng.random((120, k)) < np.array([0.3, 0.3, 0.55, 0.3])).astype(float) for d in ("hle", "bcb")}
    r = S.max_t_family(Y, ["a", "b", "c", "d"], _rs({"hle": 120, "bcb": 120}))
    assert r["p_global"] < 0.01
    assert r["p_global"] <= min(p["p_maxT_adjusted"] for p in r["pairs"]) + 1e-12
    for p in r["pairs"]:
        lo, hi = p["ci95_simultaneous"]
        assert lo <= p["estimate"] <= hi
    c_minus_a = next(p for p in r["pairs"] if p["contrast"] == "a - c")
    assert c_minus_a["ci95_simultaneous"][1] < 0
    assert len(r["methods"]) == k and all(m["ci95_percentile"][0] <= m["mean"] <= m["ci95_percentile"][1] for m in r["methods"])


def test_holm_known_example():
    out = S.holm({"A": 0.01, "O": 0.04, "R": 1.0, "C": 0.005, "G": 1.0, "M": 1.0})
    assert out["C"]["p_holm"] == pytest.approx(0.03) and out["C"]["reject"]
    assert out["A"]["p_holm"] == pytest.approx(0.05) and out["A"]["reject"]
    assert out["O"]["p_holm"] == pytest.approx(0.16) and not out["O"]["reject"]
    assert all(out[f]["p_holm"] == 1.0 for f in ("R", "G", "M"))


def test_build_panel_prefix_rule_and_duplicates():
    rows = []
    for d in ("hle", "bcb"):
        for rank in range(60):
            for cfg in ("00", "01", "10", "11"):
                if d == "bcb" and rank == 52 and cfg == "11":
                    continue  # hole at rank 52 → prefix floors to 50
                rows.append({"domain": d, "rank": rank, "source_id": f"{d}:{rank}", "framing": cfg, "y": 1.0 if (rank + int(cfg, 2)) % 3 == 0 else 0.0})
    rows.append(dict(rows[0]))  # duplicate row must be ignored, not counted twice
    panel = S.build_panel(rows, configs=S.FRAMINGS, config_of=lambda r: r["framing"], value_of=lambda r: r["y"])
    assert panel["n_prefix"] == 50 and panel["duplicate_rows_ignored"] == 1
    assert panel["prefix"]["Y"]["hle"].shape == (50, 4) and panel["prefix"]["Y"]["bcb"].shape == (50, 4)
    assert panel["complete"]["Y"]["hle"].shape == (60, 4) and panel["complete"]["Y"]["bcb"].shape == (59, 4)
    assert panel["prefix"]["ids"]["hle"][:2] == ["hle:0", "hle:1"]


def test_family_A_and_O_on_synthetic_tables():
    rng = np.random.default_rng(11)
    sel, eps = [], []
    for d in ("hle", "bcb"):
        for rank in range(75):
            sid = f"{d}:{rank}"
            for f in S.FRAMINGS:
                p = 0.3 + (0.2 if f[1] == "1" else 0.0)
                sel.append({"module": "F", "pool_kind": "bank_prefix", "prefix_k": 5, "selector_id": "VOTE", "checkpoint": "32B", "split": "main", "framing": f, "domain": d, "rank": rank, "source_id": sid, "selected_correct": bool(rng.random() < p)})
            for m in ("S_FRESH", "IND_VOTE", "DEC"):
                eps.append({"module": "A", "checkpoint": "32B", "B": 4, "episode_rep": 0, "split": "main", "method": m, "domain": d, "rank": rank, "source_id": sid, "native_final_correct": bool(rng.random() < (0.6 if m == "DEC" else 0.3))})
    A = S.family_A(sel, 123, 1000)
    assert A["executed"] and A["n_prefix"] == 75 and A["p"] < 0.01 and 0.1 < A["estimate"] < 0.3
    O = S.family_O(eps, 123, 1000)
    assert O["executed"] and O["methods"] == ["S_FRESH", "IND_VOTE", "DEC"] and O["p"] < 0.01
    assert "CEN_FLAT" in O["unexecuted_methods"]
    assert S.family_A([], 1, 100)["p"] == 1.0 and not S.family_A([], 1, 100)["executed"]
    assert S.family_O([], 1, 100)["p"] == 1.0
