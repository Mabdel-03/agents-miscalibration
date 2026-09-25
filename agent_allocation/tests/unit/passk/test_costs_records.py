from __future__ import annotations

import json

import pandas as pd

from agent_allocation.passk.costs import logical_ledger, physical_ledger, physical_summary
from agent_allocation.passk.metrics import bank_counts, item_mixed_coverage
from agent_allocation.passk.pools import build_pools
from agent_allocation.passk.records import write_records


def _draws():
    rows = []
    for r in range(25):
        rows.append({"candidate_id": f"i|1|{r}", "source": "legacy", "partition": "legacy_main", "shard": "mancer",
                     "item_id": "aime/1", "benchmark": "aime", "kind": "root", "allowance_a": 1, "replicate": r,
                     "panel_roles": ["L96"], "draw_role": "pool", "absent": r == 3, "think_in": 10,
                     "think_out": 1024, "answer_in": 20, "answer_out": 5, "cost_usd": 0.004, "latency_s": 1.0 + r,
                     "correct": r % 4 == 0, "vote_text": "1", "vote_exact_key": "1"})
    return pd.DataFrame(rows)


def test_logical_charges_every_member_physical_counts_once():
    draws = _draws()
    pools = build_pools(draws, panels=("L96",), configs=("1x1", "2x1", "4x1", "16x1"))
    phys = physical_ledger(draws)
    assert len(phys) == 24                                    # the absent node was never billed
    assert abs(phys["cost_usd"].sum() - 24 * 0.004) < 1e-12
    logical = logical_ledger(pools, draws).set_index("config_id")
    assert logical.loc["16x1", "members_charged"] == 16 and logical.loc["16x1", "members_absent"] == 1
    assert abs(logical.loc["4x1", "cost_usd"] - 3 * 0.004) < 1e-12    # member r3 is absent
    assert logical["cost_usd"].sum() > phys.loc[phys.replicate < 16, "cost_usd"].sum()   # shared draws re-charged
    for row in logical.itertuples():
        assert row.nominal_C_tokens == row.n * row.R * row.a * 1024
    summary = physical_summary(phys)
    assert int(summary["requests"].sum()) == 24


def test_bank_coverage_uses_first_20_slots():
    draws = _draws()
    banks = bank_counts(draws, ("legacy_main",))
    assert int(banks.loc[0, "m"]) == 20 and int(banks.loc[0, "c_lo"]) == 5
    g = item_mixed_coverage(banks, {1: 20})
    assert g.loc[0, "G_lo"] == 1.0
    assert item_mixed_coverage(banks, {1: 21}).empty


def test_records_written_with_hashes(tmp_path):
    draws = _draws()
    pools = build_pools(draws, panels=("L96",), configs=("1x1",))
    manifest = write_records(tmp_path, draws, pools, pd.DataFrame({"pool_id": pools.pool_id}))
    saved = json.loads((tmp_path / "records" / "run_manifest.json").read_text())
    assert saved["sha256"] == manifest["sha256"] and set(saved["sha256"]) == {
        "candidates.parquet", "pools.parquet", "selections.parquet"}
    assert saved["prompt_version"] == "prompts-v1-e998b556294c"
