"""WP-C: dynorch/tables.py against the sealed forced_budget_v2 schemas (contract section 7).

Rules under test, each with a test that fails if the rule is violated: exactly the sealed 36 / 39 columns and
dtypes; ``C == cap`` (``team_size * total_cycles * reasoning_budget``); one row per planned cell with pending rows
``y = NaN``; the B7 outcome vocabulary; cost identities (``completion_tokens = completion_total + grace``,
standardized cost at planning prices, retries summed over superseded attempts); loud failure on malformed input;
and the merged Frame reproducing the sealed ladder estimates with harness columns present.
"""
from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from agent_allocation.analysis.costs import standardized_cost
from agent_allocation.dynorch import tables as T

ROOT = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "configs" / "dynorch_v1.json"
CFG = json.loads(CFG_PATH.read_text())
BASE = Path(CFG["base_run_root"])
SEALED = BASE / "merged" / "tables"
sealed_available = pytest.mark.skipif(not (SEALED / "policy_outcomes.parquet").is_file(),
                                      reason="sealed forced_budget_v2 tables not mounted")


def _items(per_bench: int | None) -> dict[str, dict[str, str]]:
    panels = json.loads((BASE / "manifests" / "panels.json").read_text())
    assign = json.loads((BASE / "shards" / "assignment.json").read_text())["assignment"]
    items = {}
    for bench in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond"):
        ids = panels[bench]["main_item_ids"]
        for it in (ids if per_bench is None else ids[:per_bench]):
            items[it] = {"benchmark": bench, "shard_id": assign[it], "endpoint_tag": f"{assign[it]}/fp8"}
    return items


def _synthetic_items(per_bench: int) -> dict[str, dict[str, str]]:
    return {f"{b}/item{i}": {"benchmark": b, "shard_id": "parasail", "endpoint_tag": "parasail/fp8"}
            for b in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond") for i in range(per_bench)}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """8 items x {H1, H3, H0} x rungs {4, 16}: 48 planned cells, 44 records, one cell with a superseded attempt."""
    root = tmp_path_factory.mktemp("small_run")
    plan = T.synthetic_plan(CFG, _synthetic_items(2), arms=["H1", "H3", "H0"], rungs=[4, 16])
    T.write_plan(T.plan_manifest_path(root, "main"), plan)
    records = []
    for i, cell in enumerate(plan):
        if i >= len(plan) - 4:                       # four planned cells never ran: pending
            continue
        kwargs = dict(correct=(i % 3 == 0), spawns=(i % 2 if cell.arm != "H0" else 0), valid=(i % 5 != 0),
                      grace=(300 if i % 4 == 0 else 0), forced_close_kind=("mid_call" if i % 4 == 0 else "none"))
        if i == 1:                                    # first attempt failed at transport, rerun as attempt 2
            first = T.synthetic_run_record(cell, status="transport_failed", correct=None, cost_usd="0.0100")
            records.append(first)
            rec = T.synthetic_run_record(cell, attempt=2, run_key=cell.run_key + "b", **kwargs)
        elif i == 2:
            rec = T.synthetic_run_record(cell, status="transport_failed", correct=None)
        elif i == 3:
            rec = T.synthetic_run_record(cell, status="running", correct=None)
        elif i == 4:
            rec = T.synthetic_run_record(cell, correct=None, valid=True)     # evaluator unknown
        else:
            rec = T.synthetic_run_record(cell, **kwargs)
        records.append(rec)
    for rec in records:
        T.write_run_record(root, rec)
    out = T.build_tables(root, CFG, stage="main")
    return root, plan, records, out


# ------------------------------------------------------------------ schema

@sealed_available
@pytest.mark.parametrize("table, sealed", [("policy_outcomes_dynorch", "policy_outcomes"),
                                           ("policy_costs_dynorch", "policy_costs")])
def test_exactly_the_sealed_columns_and_dtypes(small_run, table, sealed):
    root, _, _, _ = small_run
    ours = pd.read_parquet(root / "tables" / f"{table}.parquet")
    theirs = pd.read_parquet(SEALED / f"{sealed}.parquet")
    assert list(ours.columns) == list(theirs.columns)
    assert len(ours.columns) == {"policy_outcomes": 36, "policy_costs": 39}[sealed]
    assert [str(d) for d in ours.dtypes] == [str(d) for d in theirs.dtypes]
    s_ours = pq.read_schema(root / "tables" / f"{table}.parquet")
    s_theirs = pq.read_schema(SEALED / f"{sealed}.parquet")
    assert [(f.name, str(f.type)) for f in s_ours] == [(f.name, str(f.type)) for f in s_theirs]


def test_column_lists_match_the_contract():
    assert T.OUTCOME_COLUMNS[:3] == ("partition_id", "policy_id", "item_id") and len(T.OUTCOME_COLUMNS) == 36
    assert T.COST_COLUMNS[0] == "policy_id" and T.COST_COLUMNS[-1] == "endpoint_tag" and len(T.COST_COLUMNS) == 39


def test_enforce_schema_refuses_extra_or_missing_columns(small_run):
    _, _, _, out = small_run
    po = out["policy_outcomes_dynorch"]
    with pytest.raises(T.TablesError, match="extra"):
        T.enforce_schema(po.assign(extra=1), "policy_outcomes")
    with pytest.raises(T.TablesError, match="missing"):
        T.enforce_schema(po.drop(columns=["y"]), "policy_outcomes")


# ------------------------------------------------------------------ rows and values

def test_one_row_per_planned_cell_and_c_equals_cap(small_run):
    _, plan, _, out = small_run
    po = out["policy_outcomes_dynorch"]
    assert len(po) == len(plan) == 48
    assert set(po["policy_id"]) == {c.policy_id for c in plan}
    C = po["team_size"] * po["total_cycles"] * po["reasoning_budget"]
    caps = pd.Series({c.policy_id: c.cap_tokens for c in plan})
    assert (C.to_numpy() == po["policy_id"].map(caps).to_numpy()).all()
    assert ((C // 1024).to_numpy() == po["name"].map(lambda n: int(n.split("/")[1])).to_numpy()).all()
    assert (po["family"] == "harness").all() and (po["view"] == "current").all() and (po["partition_id"] == "main").all()
    assert (po["selector_name"] == "last_message").all() and (po["selector_version"] == "dynorch-v1").all()
    assert (po["selector_concrete"] == "LAST_ANSWER").all() and (po["n_pool"] == 1).all()


def test_pending_rows_have_nan_y_and_pending_state(small_run):
    _, plan, _, out = small_run
    po = out["policy_outcomes_dynorch"].set_index("policy_id")
    for cell in plan[-4:]:
        row = po.loc[cell.policy_id]
        assert row["delivered_state"] == "pending" and math.isnan(row["y"]) and not row["sealed"]
        assert row["n_required_terminal"] == 0 and row["n_pool_unknown"] == 1
    assert set(po["delivered_state"]) <= set(T.DELIVERED_STATES)


def test_outcome_rule_b7():
    assert T.outcome_of(None) == (T.outcome_of(None)[0], "pending") and math.isnan(T.outcome_of(None)[0])
    base = {"status": "done", "parse_status": "valid", "correct": True}
    assert T.outcome_of(base) == (1.0, "delivered")
    assert T.outcome_of({**base, "correct": False}) == (0.0, "delivered")
    y, s = T.outcome_of({**base, "correct": None})
    assert math.isnan(y) and s == "unknown"
    assert T.outcome_of({**base, "parse_status": "missing_final"}) == (0.0, "op_no_output")
    assert T.outcome_of({**base, "parse_status": "MALFORMED", "correct": True}) == (0.0, "op_no_output")
    y, s = T.outcome_of({**base, "status": "transport_failed", "parse_status": None, "correct": None})
    assert math.isnan(y) and s == "unknown"
    assert T.outcome_of({**base, "status": "running"})[1] == "pending"
    assert T.outcome_of({**base, "status": "timeout", "parse_status": "valid", "correct": True}) == (1.0, "delivered")
    with pytest.raises(T.TablesError):
        T.outcome_of({**base, "correct": 1})


def test_cell_names_and_policy_ids():
    assert T.cell_name("H1", 16, 1) == "H1/16" and T.cell_name("H3", 4, 2) == "H3/4/r2"
    assert T.policy_id("main", "aime/2026/11", "H1/16") == "main/aime/2026/11/H1/16/current"
    with pytest.raises(T.TablesError):
        T.cell_name("H1", 16, 0)
    assert len(T.run_key_of("openrouter_dynorch_v1", "H1", "aime/2026/11", 16, 1, 1)) == 24


def test_costs_identities(small_run):
    _, plan, records, out = small_run
    pc = out["policy_costs_dynorch"].set_index("policy_id")
    price = T.price_of(CFG)
    current, superseded = T.current_records(plan, records)
    for cell in plan:
        row = pc.loc[cell.policy_id]
        rec = current.get(cell.cell_key)
        assert row["requested_reasoning_tokens"] == cell.cap_tokens
        assert row["requested_completion_cap_tokens"] == cell.cap_tokens + CFG["dynorch"]["grace_tokens"]
        assert row["cost_source"] == "dynorch_ledger" and row["nominal_generations"] == 1
        if rec is None or rec["status"] not in T.TERMINAL_STATUSES:
            assert row["completion_tokens"] == 0 and row["results_missing"] == 1 and row["success_cost_usd"] == "0"
            continue
        assert row["completion_tokens"] == rec["completion_total"] + rec["grace_tokens"]
        assert row["realized_tokens"] == row["completion_tokens"]
        assert row["prompt_tokens"] == rec["prompt_total"]
        assert row["answer_tokens"] == rec["grace_tokens"]
        assert Decimal(row["standardized_uncached_cost_usd"]) == standardized_cost(
            rec["prompt_total"], rec["completion_total"] + rec["grace_tokens"], price)
        retry = sum((Decimal(r["cost_usd"]) for r in superseded.get(cell.cell_key, [])), Decimal(0))
        assert Decimal(row["retry_cost_usd"]) == retry
        assert Decimal(row["total_charged_usd"]) == Decimal(rec["cost_usd"]) + retry
    # the rerun cell carries its first attempt's cost as retry
    rerun = plan[1]
    assert Decimal(pc.loc[rerun.policy_id, "retry_cost_usd"]) == Decimal("0.0100")


def test_runs_and_calls_tables_carry_every_attempt(small_run):
    _, _, records, out = small_run
    runs, calls = out["dynorch_runs"], out["dynorch_calls"]
    assert len(runs) == len(records) == 45
    assert runs["run_key"].is_unique
    assert set(runs["attempt"]) == {1, 2}
    assert len(calls) == sum(len(r["calls_detail"]) for r in records)
    assert (calls.groupby("run_key").size().reindex(runs["run_key"]).fillna(0).to_numpy()
            == runs.set_index("run_key").loc[runs["run_key"], "n_calls_detail"].to_numpy()).all()
    assert {"realized_share", "cap_bound_calls", "prompt_tokens_first_root", "calls_root", "calls_solver",
            "calls_other", "delivered_state", "y"} <= set(runs.columns)
    # counted over the CURRENT attempt of each cell only (the superseded transport-failed attempt is not a cost row)
    current, _ = T.current_records(small_run[1], records)
    assert out["reasoning_unknown_rows"] == sum(1 for r in current.values() if r["status"] in T.TERMINAL_STATUSES)
    assert out["reasoning_unknown_rows"] == int(runs["status"].isin(T.TERMINAL_STATUSES).sum()) - 1


# ------------------------------------------------------------------ loud failure

def test_malformed_inputs_raise(tmp_path):
    plan = T.synthetic_plan(CFG, _synthetic_items(1), arms=["H1"], rungs=[4])
    cell = plan[0]
    rec = T.synthetic_run_record(cell, correct=True)
    bad = dict(rec)
    del bad["completion_total"]
    with pytest.raises(T.TablesError, match="lacks"):
        T.validate_record(bad)
    with pytest.raises(T.TablesError, match="status"):
        T.validate_record({**rec, "status": "weird"})
    with pytest.raises(T.TablesError, match="tagged"):
        T.validate_record({**rec, "__record__": "CallRecord"})
    call = dict(rec["calls_detail"][0])
    del call["granted_max_tokens"]
    with pytest.raises(T.TablesError, match="calls_detail"):
        T.validate_record({**rec, "calls_detail": [call]})
    # a record for a cell outside the plan
    other = T.synthetic_plan(CFG, _synthetic_items(1), arms=["H3"], rungs=[4])[0]
    with pytest.raises(T.TablesError, match="not a planned cell"):
        T.policy_outcomes_table(plan, [T.synthetic_run_record(other, correct=True)])
    # duplicate attempt numbers for one cell
    with pytest.raises(T.TablesError, match="duplicate attempt"):
        T.policy_outcomes_table(plan, [rec, T.synthetic_run_record(cell, correct=False, run_key="x")])
    # plan manifest problems
    p = tmp_path / "plan.jsonl"
    p.write_text(json.dumps({"run_key": "k", "arm": "H1"}) + "\n")
    with pytest.raises(T.TablesError, match="lacks"):
        T.read_plan(p)
    p.write_text("not json\n")
    with pytest.raises(T.TablesError, match="not JSON"):
        T.read_plan(p)
    T.write_plan(p, [cell, cell])
    with pytest.raises(T.TablesError, match="duplicate planned cell"):
        T.read_plan(p)
    with pytest.raises(T.TablesError, match="not found"):
        T.read_plan(tmp_path / "missing.jsonl")
    p.write_text("")
    with pytest.raises(T.TablesError, match="empty"):
        T.read_plan(p)


def test_plan_reader_accepts_spec_wrapper(tmp_path):
    cell = T.synthetic_plan(CFG, _synthetic_items(1), arms=["H1"], rungs=[16])[0]
    p = tmp_path / "plan.jsonl"
    p.write_text(json.dumps({"spec": {**cell.__dict__, "__record__": "RunSpec"}}) + "\n")
    assert T.read_plan(p) == [cell]


def test_sealed_tables_are_not_written(small_run):
    """merged_* only reads the sealed dir; nothing under it may be created by the tables module."""
    root, _, _, _ = small_run
    assert not (root / "merged").exists()
    with pytest.raises(T.TablesError, match="not found"):
        T.merged_outcomes(root, root)


# ------------------------------------------------------------------ merged frame regression (contract section 7)

@sealed_available
def test_merged_frame_reproduces_sealed_ladder_estimates_with_harness_columns(tmp_path):
    sys.path.insert(0, str(ROOT / "docs" / "dynorch_report"))
    import core as rc  # noqa: E402

    plan = T.synthetic_plan(CFG, _items(None), arms=["H1", "H3", "H0"], rungs=[1, 2, 4, 8, 16, 25])
    assert len(plan) == 96 * 3 * 6
    T.write_plan(T.plan_manifest_path(tmp_path, "main"), plan)
    for i, cell in enumerate(plan):
        if cell.arm == "H0" and cell.rung == 25 and (i // 6) % 10 == 0:
            continue                                            # pending cells stay in the population
        T.write_run_record(tmp_path, T.synthetic_run_record(cell, correct=(i * 7) % 5 < 3, spawns=i % 3 if cell.arm != "H0" else 0))
    out = T.build_tables(tmp_path, CFG, stage="main")
    merged = T.merged_outcomes(BASE / "merged", tmp_path)
    assert len(merged) == len(pd.read_parquet(SEALED / "policy_outcomes.parquet")) + len(plan)
    assert list(merged.columns) == list(T.OUTCOME_COLUMNS)
    F = rc.Frame(BASE / "merged", out["policy_outcomes_dynorch"], items=rc.main_items(BASE / "manifests" / "panels.json"),
                 replicates=200)
    assert F.n_items == 96 and F.families == rc.LADDER_FAMILIES + ["harness"]
    assert {"H1/16", "H3/16", "H0/16", "H0/25", "deep/16", "sample/16"} <= set(F.names)
    assert round(F.acc("deep/16")[0], 4) == 0.7917
    assert round(F.acc("sample/16")[0], 4) == 0.4167
    # one population per contrast, no mismatches even with pending rows in H0/25
    est, lo, hi, _ = F.diff("H0/25", "deep/25")
    assert lo <= est <= hi
    assert F.state_counts("H0/25")["pending"] > 0
    # C == cap holds through the sealed core.py derivation
    h = F.oc[F.oc["family"] == "harness"]
    assert (h["C"] == h["reasoning_budget"]).all() and (h["rung"] == h["name"].map(rc.rung_of)).all()
