"""Per-pool and per-configuration metrics (plan "Per configuration"; spec §15.5 item 5).

Per pool (from a ``readout.readout_pools`` frame):

* coverage ``H`` (some member correct), plurality accuracy ``S`` (delivered correct) and the analytic
  tie-averaged ``S_tie`` -- each as ``(lo, hi)`` bounds: an unknown grade is not-correct in ``lo`` and correct in
  ``hi``; absent members are not correct;
* answer entropy over the pool's classes (nats, over votes) and top-wrong-class mass (the largest class whose
  first member is known wrong, over votes);
* the three-term decomposition ``1 − S = (1 − H_arc) + (H_arc − H_cur) + (H_cur − S)`` (generation failure,
  availability loss, selection failure). Independent pools have ``H_arc = H_cur``.

Finite-bank pass@k (``metrics.passk``) never takes ``k > m``; ``mixed_coverage`` is
``G_x = 1 − Π_a C(m_a−c_a, k_a)/C(m_a, k_a)`` exactly (``Fraction``).
"""
from __future__ import annotations

import math
from fractions import Fraction
from math import comb
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation.metrics.passk import bank_pass_at_k

__all__ = ["BENCH_ORDER", "MACRO_WEIGHTS", "MIXTURE_WEIGHTS", "config_summary", "decomposition", "entropy",
           "family_means", "finite_bank_pass_at_k", "mixed_coverage", "pool_metrics", "weighted_mean"]

BENCH_ORDER = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
MIXTURE_WEIGHTS = {"aime": 0.10, "hmmt": 0.10, "omni_math_hard": 0.50, "gpqa_diamond": 0.30}
MACRO_WEIGHTS = {b: 0.25 for b in BENCH_ORDER}


def entropy(sizes: Sequence[int]) -> float:
    total = sum(sizes)
    if total <= 0:
        return 0.0
    return float(-sum((s / total) * math.log(s / total) for s in sizes if s > 0))


def _tri(value) -> bool | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return bool(value)


def decomposition(h_arc: float, h_cur: float, s: float, *, tol: float = 1e-12) -> dict[str, float]:
    if not (h_arc + tol >= h_cur >= s - tol):
        raise ValueError(f"decomposition needs H_arc >= H_cur >= S, got {h_arc}, {h_cur}, {s}")
    out = {"generation": 1.0 - h_arc, "availability": h_arc - h_cur, "selection": h_cur - s, "error": 1.0 - s}
    if abs(out["generation"] + out["availability"] + out["selection"] - out["error"]) > 1e-12:
        raise AssertionError("decomposition is not exact")
    return out


def pool_metrics(readouts: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Add H/S/S_tie bounds, entropy, top-wrong mass and the decomposition to a readout frame."""
    correct = {cid: _tri(v) for cid, v in zip(draws["candidate_id"], draws["correct"])}
    out = readouts.copy()
    n = out["n"].astype(int)
    out["H_lo"] = (out["n_member_correct"] > 0).astype(float)
    out["H_hi"] = ((out["n_member_correct"] + out["n_member_unknown"]) > 0).astype(float)
    delivered = out["delivered_correct"].map(_tri)
    out["S_lo"] = delivered.map(lambda v: float(v is True))
    out["S_hi"] = delivered.map(lambda v: float(v is not False))
    out["S_known"] = delivered.map(lambda v: np.nan if v is None else float(v))
    out["H_known"] = np.where(out["H_lo"] == out["H_hi"], out["H_lo"], np.nan)
    out["S_tie"] = np.where(out["tie_avg_lo"] == out["tie_avg_hi"], out["tie_avg_lo"], np.nan)
    out["entropy"] = out["class_sizes"].map(lambda s: entropy(list(s)))

    def top_wrong(row) -> float:
        votes = row.n_votes
        if not votes:
            return 0.0
        wrong = [size for size, first in zip(row.class_sizes, row.class_first_ids) if correct.get(first) is False]
        return max(wrong, default=0) / votes
    out["top_wrong_mass"] = [top_wrong(r) for r in out.itertuples(index=False)]
    out["gen_fail"] = 1.0 - out["H_known"]
    out["avail_loss"] = 0.0
    out["sel_fail"] = out["H_known"] - out["S_known"]
    out["n_calls"] = n
    return out


def finite_bank_pass_at_k(bank: Sequence[bool | None], k: int) -> tuple[float, float]:
    """(lo, hi) pass@k of one item's bank of m exchangeable draws; raises when k > m."""
    if k > len(bank):
        raise ValueError(f"pass@k with k={k} > m={len(bank)} is not estimable from this bank")
    result = bank_pass_at_k([_tri(v) for v in bank], k)
    return float(result.lo), float(result.hi)


def mixed_coverage(m: Mapping[int, int], c: Mapping[int, int], k: Mapping[int, int]) -> Fraction:
    """``G_x = 1 − Π_a C(m_a − c_a, k_a) / C(m_a, k_a)`` over allowances ``a`` (exact)."""
    miss = Fraction(1)
    for a, ka in k.items():
        if ka == 0:
            continue
        ma, ca = int(m[a]), int(c[a])
        if ka > ma:
            raise ValueError(f"k_{a}={ka} > m_{a}={ma}")
        if not 0 <= ca <= ma:
            raise ValueError(f"c_{a}={ca} outside [0, {ma}]")
        miss *= Fraction(comb(ma - ca, ka), comb(ma, ka))
    return 1 - miss


def family_means(frame: pd.DataFrame, value: str, *, by: Iterable[str] = ()) -> pd.DataFrame:
    keys = list(by) + ["benchmark"]
    return frame.groupby(keys, dropna=False)[value].mean().unstack("benchmark")


def weighted_mean(per_family: Mapping[str, float], weights: Mapping[str, float]) -> float:
    present = [b for b in weights if b in per_family and per_family[b] == per_family[b]]
    total = sum(weights[b] for b in present)
    return float(sum(weights[b] * per_family[b] for b in present) / total) if total else float("nan")


def config_summary(metrics: pd.DataFrame, *, items: Iterable[str] | None = None,
                   values: Sequence[str] = ("H_known", "S_known", "S_tie", "S_lo", "S_hi", "H_lo", "H_hi",
                                            "entropy", "top_wrong_mass")) -> pd.DataFrame:
    """Per (panel, config): macro and mixture means of each value, plus micro and reliability S/H."""
    frame = metrics if items is None else metrics[metrics["item_id"].isin(set(items))]
    rows = []
    for (panel, config_id), sub in frame.groupby(["panel", "config_id"], sort=False):
        row = {"panel": panel, "config_id": config_id, "n_items": sub["item_id"].nunique()}
        for value in values:
            fam = sub.groupby("benchmark")[value].mean().to_dict()
            row[f"{value}_macro"] = weighted_mean(fam, MACRO_WEIGHTS)
            row[f"{value}_mixture"] = weighted_mean(fam, MIXTURE_WEIGHTS)
            row[f"{value}_micro"] = float(sub[value].mean())
        h, s = row.get("H_known_macro"), row.get("S_known_macro")
        row["reliability_macro"] = s / h if h else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
