"""Physical and logical cost ledgers (spec §15.5 item 8; plan "full usage").

* **Physical**: one row per request (candidate node) -- each generated call is counted once, however many pools
  it serves. Summed by source/partition/shard/allowance.
* **Logical**: one row per evaluated pool -- every member is charged to the pool (a draw shared by several pools is
  charged to each), with nominal compute ``C = n · R · a · A`` (R = 1 for independent calls).
"""
from __future__ import annotations

import pandas as pd

__all__ = ["A", "USAGE", "logical_ledger", "physical_ledger", "physical_summary"]

A = 1024
USAGE = ("think_in", "think_out", "answer_in", "answer_out", "cost_usd", "latency_s")


def physical_ledger(draws: pd.DataFrame) -> pd.DataFrame:
    present = draws[~draws["absent"].astype(bool)].drop_duplicates("candidate_id")
    cols = ["candidate_id", "source", "partition", "shard", "item_id", "benchmark", "kind", "allowance_a",
            "replicate", *USAGE]
    out = present[cols].copy()
    for col in USAGE:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["nominal_think_tokens"] = out["allowance_a"].astype(int) * A
    return out


def physical_summary(ledger: pd.DataFrame, by=("source", "partition", "shard", "allowance_a")) -> pd.DataFrame:
    g = ledger.groupby(list(by), dropna=False)
    out = g[list(USAGE)].sum(min_count=1)
    out["requests"] = g.size()
    out["nominal_think_tokens"] = g["nominal_think_tokens"].sum()
    return out.reset_index()


def logical_ledger(pools: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    usage = draws.set_index("candidate_id")
    rows = []
    for pool in pools.itertuples(index=False):
        members = list(pool.member_ids)
        sub = usage.loc[members]
        row = {"pool_id": pool.pool_id, "item_id": pool.item_id, "panel": pool.panel, "config_id": pool.config_id,
               "n": int(pool.n), "R": int(pool.R), "a": int(pool.a),
               "nominal_C_tokens": int(pool.n) * int(pool.R) * int(pool.a) * A,
               "members_charged": len(members), "members_absent": int(sub["absent"].astype(bool).sum())}
        billed = sub[~sub["absent"].astype(bool)]              # an absent member has no billed usage record
        for col in USAGE:
            values = pd.to_numeric(billed[col], errors="coerce")
            row[col] = float(values.sum()) if col != "latency_s" else float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)
