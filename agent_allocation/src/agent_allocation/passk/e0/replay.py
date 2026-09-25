"""Legacy tie replay: re-run the forced_v2 plurality from the pair cache under the legacy tie key.

For every legacy ``main`` current-view checkpoint of family ``sample`` or ``deep`` the replay rebuilds the pool from
``manifests/policies/main.jsonl`` (the legacy member order), reads out under ``mode="legacy"`` and compares the
delivered id with the sealed ``selected_candidate_id`` in ``merged/tables/policy_outcomes``. Pools whose class build
met an unknown equivalence (a stored None) are *drift-eligible*: the legacy memoized verdict may have differed. The
gate is a 100% match on pools with no unknown equivalence; drift pools are listed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_allocation.passk.pools import pool_id
from agent_allocation.passk.readout import readout_pools

__all__ = ["legacy_policies", "replay"]

SHARDS = ("mancer", "parasail")


def legacy_policies(legacy_root: Path, families=("sample", "deep")) -> pd.DataFrame:
    rows = []
    for shard in SHARDS:
        path = Path(legacy_root) / "shards" / shard / "manifests" / "policies" / "main.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cp = json.loads(line)
                if cp["family"] not in families or cp["view"] != "current":
                    continue
                rows.append({"policy_id": cp["policy_id"], "item_id": cp["item_id"], "name": cp["name"],
                             "family": cp["family"], "member_ids": list(cp["candidate_ids"])})
    return pd.DataFrame(rows)


def replay(legacy_root: Path, draws: pd.DataFrame, cache, *, families=("sample", "deep")) -> pd.DataFrame:
    """One row per legacy checkpoint: our delivered id vs the sealed one, and whether the pool can drift."""
    policies = legacy_policies(legacy_root, families)
    sealed = pd.read_parquet(Path(legacy_root) / "merged" / "tables" / "policy_outcomes.parquet",
                             columns=["policy_id", "selected_candidate_id", "y", "delivered_state", "benchmark"])
    policies = policies.merge(sealed, on="policy_id", how="left", validate="one_to_one")
    pools = pd.DataFrame({
        "pool_id": [pool_id(i, n, "current", m) for i, n, m in zip(policies.item_id, policies.name,
                                                                     policies.member_ids)],
        "item_id": policies.item_id, "benchmark": policies.benchmark, "panel": "L96", "config_id": policies.name,
        "n": policies.member_ids.map(len), "a": 1, "member_ids": policies.member_ids})
    got = readout_pools(pools, draws, cache=cache, mode="legacy")
    out = policies.reset_index(drop=True).join(got[["delivered_id", "delivered_correct", "equivalence_unknown",
                                                    "n_tied", "status"]])
    out["match"] = out["delivered_id"].fillna("<none>") == out["selected_candidate_id"].fillna("<none>")
    out["drift_eligible"] = out["equivalence_unknown"] > 0
    return out
