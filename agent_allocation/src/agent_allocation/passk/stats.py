"""Bootstrap statistics for passk_cont_v1 (PREREG "Primary family"; contract §1).

Benchmark-stratified, item-clustered paired bootstrap through ``_vendor/ard_stats`` (``Resampler``,
``paired_contrast``, ``mean_ci``, ``holm``): 10,000 replicates, seed 20260925, domains = the four benchmarks.
Estimands: the T200 mixture (aime .10, hmmt .10, omni_math_hard .50, gpqa_diamond .30) and the equal macro.
The primary family is exactly two tests (Holm, α = 0.05). One scalar per item enters a contrast: every draw, pool
and configuration of a sampled item moves together.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from agent_allocation._vendor.ard_stats import Resampler, holm, mean_ci, paired_contrast

__all__ = ["DOMAINS", "MACRO", "MIXTURE", "N_PRIMARY", "N_REPLICATES", "SEED", "contrast", "holm2", "level",
           "menu_contrasts", "resampler"]

DOMAINS = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
MIXTURE = {"aime": 0.10, "hmmt": 0.10, "omni_math_hard": 0.50, "gpqa_diamond": 0.30}
MACRO = {d: 0.25 for d in DOMAINS}
N_REPLICATES = 10_000
SEED = 20260925
N_PRIMARY = 2


def _by_domain(frame: pd.DataFrame, value: str) -> dict[str, np.ndarray]:
    out = {}
    for d in DOMAINS:
        sub = frame[frame["benchmark"] == d].sort_values("item_id")
        out[d] = sub[value].to_numpy(dtype=float)
    return out


def resampler(frame: pd.DataFrame, *, n_resamples: int = N_REPLICATES, seed: int = SEED) -> Resampler:
    counts = frame.groupby("benchmark")["item_id"].nunique().to_dict()
    return Resampler({d: counts.get(d, 0) for d in DOMAINS}, n_resamples, seed, domains=DOMAINS)


def contrast(items: pd.DataFrame, a: str, b: str, *, weights: Mapping[str, float] = MIXTURE,
             n_resamples: int = N_REPLICATES, seed: int = SEED, n_family: int = N_PRIMARY) -> dict:
    """Paired contrast mean(a − b) over items where both are known; one row per item (``item_id, benchmark``)."""
    known = items.dropna(subset=[a, b]).copy()
    known["_diff"] = known[a].astype(float) - known[b].astype(float)
    rs = resampler(known, n_resamples=n_resamples, seed=seed)
    result = paired_contrast(_by_domain(known, "_diff"), rs, domains=DOMAINS, weights=dict(weights),
                             n_primary=n_family)
    result["n_items"] = int(len(known))
    result["n_dropped_unknown"] = int(len(items) - len(known))
    return result


def level(items: pd.DataFrame, value: str, *, weights: Mapping[str, float] = MACRO,
          n_resamples: int = N_REPLICATES, seed: int = SEED) -> dict:
    known = items.dropna(subset=[value])
    rs = resampler(known, n_resamples=n_resamples, seed=seed)
    return mean_ci(_by_domain(known, value), rs, domains=DOMAINS, weights=dict(weights))


def menu_contrasts(metrics: pd.DataFrame, menu: tuple[str, ...], reference: str, *, value: str = "S_known",
                   weights: Mapping[str, float] = MACRO, n_resamples: int = N_REPLICATES,
                   seed: int = SEED) -> pd.DataFrame:
    """Secondary: every menu configuration vs ``reference`` per panel, with Bonferroni-simultaneous intervals
    (``ci_primary_bonferroni`` over the ``len(menu) - 1`` comparisons of the menu)."""
    rows = []
    others = [c for c in menu if c != reference]
    for panel, sub in metrics[metrics["config_id"].isin(menu)].groupby("panel"):
        wide = sub.pivot_table(index=["item_id", "benchmark"], columns="config_id", values=value, aggfunc="first")
        if reference not in wide:
            continue
        items = wide.reset_index()
        for config_id in others:
            if config_id not in items or items[[config_id, reference]].dropna().empty:
                continue
            try:
                r = contrast(items, config_id, reference, weights=weights, n_resamples=n_resamples, seed=seed,
                             n_family=len(others))
            except Exception as exc:  # noqa: BLE001 - too few items in a panel: reported, not fatal
                rows.append({"panel": panel, "config_id": config_id, "reference": reference, "error": str(exc)})
                continue
            rows.append({"panel": panel, "config_id": config_id, "reference": reference, "estimate": r["estimate"],
                         "ci95_lo": r["ci95_studentized"][0], "ci95_hi": r["ci95_studentized"][1],
                         "simul_lo": r["ci_primary_bonferroni"][0], "simul_hi": r["ci_primary_bonferroni"][1],
                         "p": r["p"], "n_items": r["n_items"]})
    return pd.DataFrame(rows)


def holm2(p_mixed_vs_deep: float, p_p2_vs_p1: float) -> dict:
    return holm({"mixed_vs_deep": p_mixed_vs_deep, "p2_vs_p1": p_p2_vs_p1}, n_primary=N_PRIMARY)
