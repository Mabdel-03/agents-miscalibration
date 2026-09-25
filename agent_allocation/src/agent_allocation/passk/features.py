"""E5 label-free five-draw features (WP-M; contract §5.7, PREREG E5).

One row per item from its 5-draw diagnostic scope: L96 = legacy A slots 20-24 (``draw_role == "diag"``, or
``allowance_a == 1 & 20 <= replicate <= 24`` on the legacy partition); T200 = partition ``t200_diag`` r0-4. Classes
are built within those five draws only (``ClassSource`` scope = the ``diag5`` pool id when WP-A pools exist, else
``diag5:<item>``).

Groups:

* CONC: sorted class frequencies f1..f5 (support / 5), plurality share (= f1), top-two gap (f1 - f2), number of
  classes / 5, normalized entropy of the voter class distribution;
* invalid fraction (draws casting no vote), mean log answer length, fraction of ANSWER finish reason ``length``;
* latency: mean within-group z-score, groups = provider x budget x collection;
* benchmark and provider (shard) are carried for the predictors.

Label-free by construction: the label columns are dropped before anything is computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_allocation.passk.models.inputs import ClassSource

CONC = ["f1", "f2", "f3", "f4", "f5", "top_gap", "n_classes", "entropy"]   # f1 = plurality share
OTHER = ["invalid_frac", "log_answer_len", "finish_length_frac"]
LATENCY = ["latency_z"]
FEATURES = CONC + OTHER + LATENCY
LABEL_COLUMNS = ("correct", "grade_state")


def diag_draws(draws: pd.DataFrame, items=None) -> pd.DataFrame:
    """The 5-draw diagnostic rows (labels stripped)."""
    d = draws.drop(columns=[c for c in LABEL_COLUMNS if c in draws.columns])
    legacy = d.partition.astype(str).str.startswith("legacy") & (d.allowance_a == 1) & d.replicate.between(20, 24)
    t200 = (d.partition == "t200_diag") & d.replicate.between(0, 4)
    out = d[legacy | t200].copy()
    out["collection"] = np.where(out.partition == "t200_diag", "t200_diag", "legacy")
    if items is not None:
        out = out[out.item_id.isin(set(items))]
    return out


def _entropy(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    p = counts / tot
    return float(-(p * np.log(p)).sum() / np.log(5))


def diag_features(draws: pd.DataFrame, cls: ClassSource, *, items=None, pools: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    d = diag_draws(draws, items)
    prov = d.get("shard", d.get("observed_provider")).astype(str)
    d["_prov"] = prov
    lat = d["latency_s"].astype(float)
    grp = d.groupby(["_prov", "allowance_a", "collection"])["latency_s"]
    mu, sd = grp.transform("mean"), grp.transform("std").replace(0, np.nan)
    d["_latz"] = ((lat - mu) / sd).fillna(0.0)
    diag_pool = {}
    if pools is not None and "config_id" in pools:
        for r in pools[pools.config_id == "diag5"].itertuples(index=False):
            diag_pool[r.item_id] = r.pool_id
    rows = []
    for item, g in d.groupby("item_id"):
        g = g.sort_values("replicate")
        mem = list(g.candidate_id)
        classes = cls.classes(diag_pool.get(item, f"diag5:{item}"), mem)
        support: dict[str, int] = {}
        for c in mem:
            k = classes.get(c)
            if k is not None:
                support[k] = support.get(k, 0) + 1
        counts = np.array(sorted(support.values(), reverse=True), float)
        f = np.zeros(5)
        f[:min(5, len(counts))] = counts[:5] / 5.0
        alen = g["answer_len_chars"].astype(float)
        fin = g["answer_finish_reason"].astype(str)
        rows.append({
            "item_id": item, "benchmark": g.benchmark.iloc[0], "provider": g._prov.iloc[0],
            "collection": g.collection.iloc[0], "n_diag": int(len(g)),
            **{f"f{i + 1}": f[i] for i in range(5)},
            "top_gap": f[0] - f[1], "n_classes": len(counts) / 5.0, "entropy": _entropy(counts),
            "invalid_frac": 1.0 - counts.sum() / max(len(g), 1),
            "log_answer_len": float(np.log1p(alen.fillna(0)).mean()),
            "finish_length_frac": float((fin == "length").mean()),
            "latency_z": float(g._latz.mean()),
        })
    return pd.DataFrame(rows)
