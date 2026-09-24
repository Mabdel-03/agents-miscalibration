"""WP-D: every Gate 1/2/4 criterion flips on a synthetic table; the reconciliation document; exit codes."""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.config import load_config
from agent_allocation.contracts import GateResult, load_record
from agent_allocation.dynorch import gates, reconcile
from agent_allocation.paths import RunPaths

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"
BENCHES = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


# ------------------------------------------------------------------------------------------------------------------
# Synthetic tables (contract §5 field names)
# ------------------------------------------------------------------------------------------------------------------

class Tables:
    """Builds runs/calls rows; each ``run`` call appends a run and its calls (one per ``calls`` entry)."""

    def __init__(self, partition="canary"):
        self.partition = partition
        self.runs: list[dict] = []
        self.calls: list[dict] = []
        self.n = 0

    def run(self, arm, item, rung, *, calls=(), spawns=0, forced_close_kind="none", grace=0, final="<final>1</final>",
            parse="valid", grade="graded", correct=True, status="done", guard=False, hits=0, misses=0, repeat=1,
            completion_total=None, cost="0.01", bench=None, error=None):
        self.n += 1
        key = f"rk{self.n:04d}"
        cap = rung * 1024
        spent = 0
        rows = []
        for i, c in enumerate(calls, start=1):
            c = dict(c)
            kind = c.get("forced_close_kind", "none")
            granted = c.get("granted", (1024 if kind == "terminal" else cap - spent))
            completion = c.get("completion", 100)
            rows.append({"run_key": key, "call_no": i, "conversation_class": c.get("cls", "root"), "granted_max_tokens": granted,
                         "completion_tokens": completion, "prompt_tokens": c.get("prompt", 5000), "grace_tokens": c.get("grace", 0),
                         "forced_close_kind": kind, "tool_calls": c.get("tool_calls", 0), "spawn_calls": c.get("spawn_calls", 0),
                         "tool_parse": c.get("tool_parse", "none"), "reinjection": c.get("reinjection", "n/a"),
                         "wire": c.get("wire", "responses" if arm.startswith("H3") else "chat"),
                         "duplicate_served": c.get("duplicate", False), "generation_ids": c.get("gen", [f"gen-{key}-{i}"]),
                         "cost_usd": c.get("cost", "0.001")})
            if not c.get("duplicate", False) and kind != "terminal":
                spent += completion
        self.calls.extend(rows)
        self.runs.append({"run_key": key, "arm": arm, "harness": {"H1": "openclaude", "H3": "codex", "H0": "null"}.get(arm, "x"),
                          "item_id": item, "benchmark": bench or item.split("/")[0], "rung": rung, "cap_tokens": cap, "repeat": repeat,
                          "attempt": 1, "partition": self.partition, "status": status, "calls": len(rows), "spawns": spawns,
                          "completion_total": spent if completion_total is None else completion_total, "grace_tokens": grace,
                          "forced_close_kind": forced_close_kind, "exhausted": forced_close_kind != "none", "guard_triggered": guard,
                          "tool_parse_failures": sum(1 for r in rows if r["tool_parse"] == "failed"), "reinjection_hits": hits,
                          "reinjection_misses": misses, "final_message": final, "parse_status": parse, "grade_status": grade,
                          "correct": correct, "wall_seconds": 10.0, "cost_usd": cost, "shard_id": "parasail",
                          "endpoint_tag": "parasail/fp8", "error": error})
        return key

    def frames(self):
        return pd.DataFrame(self.runs), pd.DataFrame(self.calls)


def clean_canary(cfg):
    """Two dev items x {H1, H3, H0} at k=4, plus the k=16 probe for H1/H3 — every Gate 1 criterion satisfied."""
    t = Tables("canary")
    for item in ("aime/2025/1", "gpqa_diamond/r1"):
        t.run("H0", item, 4, calls=[{"completion": 4096, "prompt": 1000, "forced_close_kind": "mid_call", "grace": 200}],
              forced_close_kind="mid_call", grace=200)
        t.run("H1", item, 4, calls=[{"completion": 4096, "prompt": 3000, "forced_close_kind": "mid_call", "grace": 300}],
              forced_close_kind="mid_call", grace=300)
        t.run("H3", item, 4, calls=[{"completion": 2000, "prompt": 4000, "tool_calls": 1, "tool_parse": "ok"},
                                    {"completion": 1500, "cls": "solver", "prompt": 2000},
                                    {"completion": 596, "forced_close_kind": "mid_call", "grace": 250, "reinjection": "hit"}],
              spawns=1, forced_close_kind="mid_call", grace=250, hits=1)
        t.run("H0", item, 16, calls=[{"completion": 9000, "prompt": 1000}])
        t.run("H1", item, 16, calls=[{"completion": 16384, "prompt": 3000, "forced_close_kind": "mid_call", "grace": 300}],
              forced_close_kind="mid_call", grace=300, spawns=0)
        t.run("H3", item, 16, calls=[{"completion": 3000, "prompt": 4000, "tool_calls": 1, "tool_parse": "ok"},
                                     {"completion": 5000, "cls": "solver"}, {"completion": 8384, "forced_close_kind": "mid_call", "grace": 100}],
              spawns=1, forced_close_kind="mid_call", grace=100, hits=1)
    return t


def recon(*, diff="0.00", rows=None, upstream_402=False, open_usd="0", ledger_delta="1.00"):
    rows = [] if rows is None else rows
    return {"layout": reconcile.LAYOUT, "stage": "canary", "phase": "after", "key_delta_usd": str(Decimal(ledger_delta) + Decimal(diff)),
            "ledger_delta_usd": ledger_delta, "key_minus_ledger_usd": diff, "ledger_after": {"open_reservations_usd": open_usd},
            "generation_sample": {"rows": rows, "sampled": len(rows), "population": max(len(rows), 1)}, "upstream_402": upstream_402,
            "decision": "clean"}


def sample_row(**over):
    row = {"run_key": "rk0001", "call_no": 1, "generation_ids": ["g"], "fetched": True, "journaled_completion_tokens": 100,
           "metadata_completion_tokens": 100, "tokens_match": True, "journaled_cost_usd": "0.001", "metadata_cost_usd": "0.001",
           "cost_within_tolerance": True, "provider_names": ["Parasail"], "expected_provider": "Parasail", "provider_ok": True}
    row.update(over)
    return row


def failing(result: GateResult) -> set[str]:
    return {c.name for c in result.criteria if not c.passed}


LOCK = {"action": "keep", "shard_to_tag": {"mancer": "mancer/fp8", "parasail": "parasail/fp8"}}


def gate1(cfg, tables, *, reconciliation=None, lock=LOCK, key_hits=(), home=()):
    runs, calls = tables.frames()
    rec = recon(rows=[sample_row()]) if reconciliation is None else (None if reconciliation == "absent" else reconciliation)
    return gates.evaluate_gate1(cfg, runs, calls, rec, lock_decision=lock, key_hits=key_hits, home_new_files=home)


# ------------------------------------------------------------------------------------------------------------------
# Gate 1
# ------------------------------------------------------------------------------------------------------------------

def test_gate1_clean_canary_advances(cfg):
    result, overheads = gate1(cfg, clean_canary(cfg))
    assert result.decision == gates.ADVANCE and result.passed, failing(result)
    assert gates.exit_code(result) == 0
    assert overheads["H1"]["n"] == 4 and overheads["H1"]["mean"] == 2000.0 and overheads["H3"]["mean"] == 3000.0


def test_gate1_completion_over_grant_is_a_hard_stop(cfg):
    t = clean_canary(cfg)
    t.run("H1", "hmmt/2025/1", 4, calls=[{"completion": 5000}])          # 5000 > granted 4096
    result, _ = gate1(cfg, t)
    assert "completion_within_grant" in failing(result) and result.decision == gates.STOP
    assert gates.exit_code(result) == 1


def test_gate1_grant_must_equal_remaining(cfg):
    t = clean_canary(cfg)
    t.run("H1", "hmmt/2025/1", 4, calls=[{"completion": 1000}, {"completion": 1000, "granted": 4096}])   # second grant should be 3096
    result, _ = gate1(cfg, t)
    assert "grant_equals_remaining" in failing(result) and result.decision == gates.STOP
    # a duplicate served from the cache and a terminal grace call are exempt from the equality
    t = clean_canary(cfg)
    t.run("H1", "hmmt/2025/2", 4, calls=[{"completion": 1000}, {"completion": 1000, "granted": 4096, "duplicate": True},
                                          {"completion": 3000, "granted": 3096}, {"completion": 50, "forced_close_kind": "terminal", "granted": 1024}],
          forced_close_kind="terminal", grace=50)
    result, _ = gate1(cfg, t)
    assert "grant_equals_remaining" not in failing(result)


def test_gate1_usage_missing_and_endpoint_rule(cfg):
    t = clean_canary(cfg)
    t.run("H1", "hmmt/2025/1", 4, calls=[{"completion": 1000, "prompt": float("nan")}])
    result, _ = gate1(cfg, t)
    assert "usage_present" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), lock={"action": "stop", "shard_to_tag": {}})
    assert "endpoint_rule_locked" in failing(result) and result.decision == gates.STOP_NOTIFY
    result, _ = gate1(cfg, clean_canary(cfg), lock=None)
    assert "endpoint_rule_locked" in failing(result)


def test_gate1_reconciliation_criteria(cfg):
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(diff="0.03", rows=[sample_row()]))
    assert "key_delta_within_tolerance" in failing(result) and result.decision == gates.STOP_NOTIFY
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(diff="-0.02", rows=[sample_row()]))
    assert "key_delta_within_tolerance" not in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row(tokens_match=False)]))
    assert "generation_sample_matches" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row(cost_within_tolerance=False)]))
    assert "generation_sample_matches" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row(fetched=False, tokens_match=None)]))
    assert "generation_sample_matches" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[]))
    assert "generation_sample_matches" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row(provider_ok=False, provider_names=["Mancer"])]))
    assert "observed_provider_pinned" in failing(result) and result.decision == gates.STOP
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row()], upstream_402=True))
    assert "no_upstream_402" in failing(result) and result.decision == gates.STOP_NOTIFY
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation=recon(rows=[sample_row()], open_usd="0.5"))
    assert "open_reservations_zero" in failing(result)
    result, _ = gate1(cfg, clean_canary(cfg), reconciliation="absent")
    assert "reconciliation_present" in failing(result)


def test_gate1_grace_truncated_share(cfg):
    t = Tables("canary")
    for i in range(50):
        t.run("H1", f"aime/2025/{i}", 4, calls=[{"completion": 4096, "forced_close_kind": "mid_call", "grace": 100}],
              forced_close_kind="mid_call", grace=100)
    t.run("H1", "aime/2025/x", 4, calls=[{"completion": 4096, "forced_close_kind": "mid_call", "grace": 1024}],
          forced_close_kind="mid_call", grace=1024)          # 1/51 = 0.0196 <= 0.02
    result, _ = gate1(cfg, t)
    assert "grace_truncated_share" not in failing(result)
    t.run("H1", "aime/2025/y", 4, calls=[{"completion": 4096, "forced_close_kind": "mid_call", "grace": 1024}],
          forced_close_kind="mid_call", grace=1024)          # 2/52 = 0.038 > 0.02
    result, _ = gate1(cfg, t)
    assert "grace_truncated_share" in failing(result)


def test_gate1_forced_close_text_and_codex_loop_and_probe(cfg):
    t = clean_canary(cfg)
    runs, calls = t.frames()
    runs.loc[runs["arm"] == "H3", "final_message"] = ""
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[], home_new_files=[])
    assert "H3_forced_close_nonempty" in failing(result) and "H1_forced_close_nonempty" not in failing(result)
    runs, calls = t.frames()
    calls.loc[calls["run_key"].isin(set(runs[runs["arm"] == "H3"]["run_key"])), "tool_calls"] = 0
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[], home_new_files=[])
    assert "codex_responses_tool_loop" in failing(result)
    runs, calls = t.frames()
    runs.loc[runs["rung"] == 16, "spawns"] = 0
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[], home_new_files=[])
    assert "ceiling_probe_spawns" in failing(result) and result.decision == gates.STOP_NOTIFY
    runs, calls = t.frames()
    runs = runs[runs["rung"] != 16]
    calls = calls[calls["run_key"].isin(set(runs["run_key"]))]
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[], home_new_files=[])
    assert "ceiling_probe_recorded" in failing(result)


def test_gate1_harness_prompts_need_an_h0_anchor(cfg):
    t = clean_canary(cfg)
    runs, calls = t.frames()
    runs = runs[runs["arm"] != "H0"]
    calls = calls[calls["run_key"].isin(set(runs["run_key"]))]
    result, overheads = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[], home_new_files=[])
    assert "harness_prompts_measured" in failing(result) and overheads == {}


def test_gate1_key_string_and_home_scans(cfg):
    result, _ = gate1(cfg, clean_canary(cfg), key_hits=["raw/x.json.gz"])
    assert "no_key_string_in_run_tree" in failing(result) and result.decision == gates.STOP
    result, _ = gate1(cfg, clean_canary(cfg), home=["/home/u/.npm/x"])
    assert "nothing_written_under_home" in failing(result)
    runs, calls = clean_canary(cfg).frames()
    # review minor 21: a scan that could not run is "not scanned" -- a NOTIFY of its own, not a claim that files
    # WERE written (an lstat walk of ~995k NFS inodes routinely exceeds its budget)
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[],
                                     home_new_files=None, home_detail={"source": "not_scanned"})
    assert failing(result) == {"home_scan_completed"} and result.decision == gates.STOP_NOTIFY
    observed = next(c for c in result.criteria if c.name == "nothing_written_under_home")
    assert json.loads(observed.observed)["detail"] == {"source": "not_scanned"}
    # the quota inode delta is the cheap path: a positive delta fails the criterion with the count in evidence
    result, _ = gates.evaluate_gate1(cfg, runs, calls, recon(rows=[sample_row()]), lock_decision=LOCK, key_hits=[],
                                     home_new_files=["$HOME gained 12 inode(s) during the stage"],
                                     home_detail={"source": "quota", "before": 900, "after": 912, "delta": 12})
    assert "nothing_written_under_home" in failing(result) and "home_scan_completed" not in failing(result)


def test_gate1_missing_column_raises(cfg):
    runs, calls = clean_canary(cfg).frames()
    with pytest.raises(gates.GateInputError):
        gates.evaluate_gate1(cfg, runs.drop(columns=["spawns"]), calls, recon(), lock_decision=LOCK, key_hits=[], home_new_files=[])
    with pytest.raises(gates.GateInputError):
        gates.evaluate_gate1(cfg, runs, calls.drop(columns=["wire"]), recon(), lock_decision=LOCK, key_hits=[], home_new_files=[])


# ------------------------------------------------------------------------------------------------------------------
# Gate 2
# ------------------------------------------------------------------------------------------------------------------

def clean_pilot(cfg, *, n_per_bench=8, valid_share=1.0, spawn_share=0.5):
    t = Tables("pilot")
    for arm in ("H1", "H3"):
        idx = 0
        for bench in BENCHES:
            for i in range(n_per_bench):
                for rung in (2, 16):
                    idx += 1
                    valid = (idx % 20) < valid_share * 20
                    spawn = 1 if (i % 2 == 0 and spawn_share >= 0.5) else (1 if spawn_share >= 1.0 else 0)
                    calls = [{"completion": 500, "tool_calls": spawn, "tool_parse": "ok" if spawn else "none"}]
                    if spawn:
                        calls += [{"completion": 300, "cls": "solver"}, {"completion": 200, "reinjection": "hit"}]
                    t.run(arm, f"{bench}/{i}", rung, calls=calls, spawns=spawn, hits=spawn, parse="valid" if valid else "missing_final",
                          grade="graded" if valid else None, correct=valid)
    for bench in BENCHES:
        for i in range(4):
            t.run("H0", f"{bench}/{i}", 16, calls=[{"completion": 9000, "prompt": 1000}])
    return t


FORECAST = {"core_forecast_usd": "60", "spent_usd": "5", "cuts_applied": []}


def gate2(cfg, t, *, reconciliation=None, forecast=FORECAST):
    runs, calls = t.frames()
    rec = recon(rows=[sample_row()]) if reconciliation is None else reconciliation
    return gates.evaluate_gate2(cfg, runs, calls, rec, forecast=forecast)


def test_gate2_clean_pilot_advances(cfg):
    result = gate2(cfg, clean_pilot(cfg))
    assert result.decision == gates.ADVANCE, failing(result)


def test_gate2_valid_final_share_bands(cfg):
    result = gate2(cfg, clean_pilot(cfg, valid_share=0.85))
    assert "H1_valid_final_share" in failing(result) and result.decision == gates.STOP_NOTIFY
    result = gate2(cfg, clean_pilot(cfg, valid_share=0.75))
    assert result.decision == gates.STOP
    assert gates.exit_code(result) == 1


def test_gate2_tool_parse_failure_share(cfg):
    t = clean_pilot(cfg)
    for i in range(6):
        t.run("H1", f"aime/{i}", 16, calls=[{"completion": 500, "tool_parse": "failed"}, {"completion": 100}])
    result = gate2(cfg, t)
    assert "H1_tool_call_parse_failure_share" in failing(result) and "H3_tool_call_parse_failure_share" not in failing(result)


def test_gate2_cap_and_grace_overrun_are_hard_stops(cfg):
    t = clean_pilot(cfg)
    t.run("H3", "aime/9", 2, calls=[{"completion": 2048}], completion_total=2049)
    result = gate2(cfg, t)
    assert "H3_cap_overrun_runs" in failing(result) and result.decision == gates.STOP
    t = clean_pilot(cfg)
    t.run("H3", "aime/9", 2, calls=[{"completion": 2048, "forced_close_kind": "mid_call", "grace": 1025}], forced_close_kind="mid_call", grace=1025)
    result = gate2(cfg, t)
    assert "H3_grace_overrun_runs" in failing(result) and result.decision == gates.STOP


def test_gate2_spawn_share_k16(cfg):
    result = gate2(cfg, clean_pilot(cfg, spawn_share=0.0))
    assert {"H1_spawn_share_k16", "H3_spawn_share_k16"} <= failing(result) and result.decision == gates.STOP_NOTIFY
    # no rung-16 runs at all also fails (the criterion needs the cell)
    t = clean_pilot(cfg)
    runs, calls = t.frames()
    runs = runs[runs["rung"] != 16]
    result = gates.evaluate_gate2(cfg, runs, calls, recon(rows=[sample_row()]), forecast=FORECAST)
    assert "H1_spawn_share_k16" in failing(result)


def test_gate2_guard_evaluator_reinjection_timeout(cfg):
    t = clean_pilot(cfg)
    for i in range(8):
        t.run("H1", f"hmmt/{i+10}", 2, calls=[{"completion": 100}], guard=True)
    assert "H1_guard_triggered_share" in failing(gate2(cfg, t))
    t = clean_pilot(cfg)
    for i in range(2):
        t.run("H3", f"hmmt/{i+10}", 2, calls=[{"completion": 100}], grade="evaluator_error", correct=None)
    assert "H3_evaluator_error_share" in failing(gate2(cfg, t))
    t = clean_pilot(cfg)
    t.run("H1", "hmmt/50", 16, calls=[{"completion": 100}], hits=1, misses=5)
    assert "H1_reinjection_hit_rate" in failing(gate2(cfg, t))
    t = clean_pilot(cfg)
    for i in range(3):
        t.run("H3", f"hmmt/{i+10}", 2, calls=[{"completion": 100}], status="timeout", final=None, parse=None, correct=None)
    result = gate2(cfg, t)
    assert "H3_timeout_share" in failing(result)


def test_gate2_reinjection_vacuous_without_tool_turns(cfg):
    t = clean_pilot(cfg, spawn_share=0.0)
    result = gate2(cfg, t)
    assert "H1_reinjection_hit_rate" not in failing(result)


def test_gate2_forecast_criterion(cfg):
    t = clean_pilot(cfg)
    assert "core_forecast_within_soft_ceiling" in failing(gate2(cfg, t, forecast=None))
    assert "core_forecast_within_soft_ceiling" in failing(gate2(cfg, t, forecast={"core_forecast_usd": "86", "spent_usd": "5", "cuts_applied": []}))
    assert "core_forecast_within_soft_ceiling" not in failing(gate2(cfg, t, forecast={"core_forecast_usd": "85", "spent_usd": "5", "cuts_applied": ["C1", "C2"]}))
    assert "core_forecast_within_soft_ceiling" in failing(gate2(cfg, t, forecast={"core_forecast_usd": "50", "spent_usd": "5", "cuts_applied": ["C4 owner"]}))


def test_pilot_report_pools_per_benchmark_only(cfg):
    t = clean_pilot(cfg, valid_share=0.9)
    runs, calls = t.frames()
    result = gates.evaluate_gate2(cfg, runs, calls, recon(rows=[sample_row()]), forecast=FORECAST)
    text = gates.render_pilot_report(cfg, runs, calls, result, recon(rows=[sample_row()]), FORECAST)
    assert "| H1 | 16 | aime |" in text and "| H1 | 16 | macro |" in text
    assert "aime/3" not in text and "rk00" not in text          # no per-item rows, no run keys
    assert "Owner go/no-go" in text and "stop_notify" in text
    rows = gates.pooled_accuracies(runs)
    assert {r["benchmark"] for r in rows} == set(BENCHES) | {"macro"}


# ------------------------------------------------------------------------------------------------------------------
# Gate 4
# ------------------------------------------------------------------------------------------------------------------

def planned_cells(items, arms=("H1", "H3", "H0"), rungs=(1, 2, 4, 8, 16, 25)):
    return [{"arm": a, "rung": r, "item_id": i, "repeat": 1} for a in arms for r in rungs for i in items]


def outcomes_for(cells, *, y=1.0, state="delivered"):
    return pd.DataFrame([{"name": gates.cell_name(c["arm"], c["rung"], c["repeat"]), "item_id": c["item_id"], "y": y,
                          "delivered_state": state, "benchmark": c["item_id"].split("/")[0]} for c in cells])


ITEMS = [f"{b}/{i}" for b in BENCHES for i in range(6)]
LADDER = {"deep/16": 0.7917, "sample/16": 0.4167}
SEAL = {f"{n}.parquet": "0" * 64 for n in ("dynorch_runs", "dynorch_calls", "policy_outcomes_dynorch", "policy_costs_dynorch")}


def gate4(cfg, cells, outcomes, *, rec=None, mismatches=0, ladder=LADDER, seal=SEAL):
    rec = recon(rows=[sample_row()], ledger_delta="40") if rec is None else rec
    return gates.evaluate_gate4(cfg, cells, outcomes, rec, grader_rerun_mismatches=mismatches, ladder_estimates=ladder, sealed_tables=seal)


def test_gate4_clean_seal_advances(cfg):
    cells = planned_cells(ITEMS)
    result = gate4(cfg, cells, outcomes_for(cells))
    assert result.decision == gates.ADVANCE, failing(result)


def test_gate4_missing_or_pending_cell_is_a_hard_stop(cfg):
    cells = planned_cells(ITEMS)
    result = gate4(cfg, cells, outcomes_for(cells[:-1]))
    assert "every_planned_cell_has_row" in failing(result) and result.decision == gates.STOP
    out = outcomes_for(cells)
    out.loc[0, "delivered_state"] = "pending"
    assert "every_planned_cell_has_row" in failing(gate4(cfg, cells, out))
    assert "every_planned_cell_has_row" in failing(gate4(cfg, [], out))


def test_gate4_completeness_partial_primary_cell_notifies(cfg):
    cells = planned_cells(ITEMS)
    out = outcomes_for(cells)
    h1_4 = out[(out["name"] == "H1/4")].index[:3]
    out.loc[h1_4, "y"] = float("nan")                    # 21/24 = 0.875 < 0.95: partial, but not primary
    result = gate4(cfg, cells, out)
    assert result.decision == gates.ADVANCE
    observed = json.loads(next(c.observed for c in result.criteria if c.name == "completeness_per_cell"))
    assert observed["partial"] == ["H1/4"]
    h3_16 = out[(out["name"] == "H3/16")].index[:3]
    out.loc[h3_16, "y"] = float("nan")
    result = gate4(cfg, cells, out)
    assert "completeness_per_cell" in failing(result) and result.decision == gates.STOP_NOTIFY


def test_gate4_key_delta_tolerance_is_50c_or_2pct(cfg):
    cells = planned_cells(ITEMS)
    out = outcomes_for(cells)
    assert "key_delta_within_tolerance" not in failing(gate4(cfg, cells, out, rec=recon(rows=[sample_row()], ledger_delta="20", diff="0.49")))
    assert "key_delta_within_tolerance" in failing(gate4(cfg, cells, out, rec=recon(rows=[sample_row()], ledger_delta="20", diff="0.51")))
    assert "key_delta_within_tolerance" not in failing(gate4(cfg, cells, out, rec=recon(rows=[sample_row()], ledger_delta="60", diff="1.10")))
    assert "key_delta_within_tolerance" in failing(gate4(cfg, cells, out, rec=recon(rows=[sample_row()], ledger_delta="60", diff="1.30")))
    assert "reconciliation_present" in failing(gates.evaluate_gate4(cfg, cells, out, None, grader_rerun_mismatches=0, ladder_estimates=LADDER, sealed_tables=SEAL))


def test_gate4_grader_ladder_and_seal(cfg):
    cells = planned_cells(ITEMS)
    out = outcomes_for(cells)
    assert "grader_rerun_identical" in failing(gate4(cfg, cells, out, mismatches=1))
    assert "grader_rerun_identical" in failing(gate4(cfg, cells, out, mismatches=None))
    result = gate4(cfg, cells, out, ladder={"deep/16": 0.79, "sample/16": 0.4167})
    assert "sealed_ladder_reproduced" in failing(result) and result.decision == gates.STOP
    assert "sealed_ladder_reproduced" in failing(gate4(cfg, cells, out, ladder=None))
    assert "tables_sealed" in failing(gate4(cfg, cells, out, seal={"dynorch_runs.parquet": "x"}))
    assert "tables_sealed" in failing(gate4(cfg, cells, out, seal=None))


def test_ladder_estimates_from_the_sealed_base(cfg):
    base = Path(cfg["base_run_root"])
    if not (base / "merged" / "tables" / "policy_outcomes.parquet").exists():
        pytest.skip("base run root not mounted")
    est = gates.ladder_estimates_from_base(base)
    assert abs(est["deep/16"] - 0.7917) < 1e-4 and abs(est["sample/16"] - 0.4167) < 1e-4


def test_seal_tables_write_once(tmp_path):
    for n in ("dynorch_runs", "dynorch_calls", "policy_outcomes_dynorch", "policy_costs_dynorch"):
        (tmp_path / f"{n}.parquet").write_bytes(n.encode())
    first = gates.seal_tables(tmp_path)
    assert set(first) == set(SEAL)
    assert gates.seal_tables(tmp_path) == first
    (tmp_path / "dynorch_runs.parquet").write_bytes(b"changed")
    with pytest.raises(gates.GateInputError):
        gates.seal_tables(tmp_path)


def test_gate_record_round_trips(cfg, tmp_path):
    result, _ = gate1(cfg, clean_canary(cfg))
    paths = RunPaths.for_run(tmp_path, local_state_dir=tmp_path / "local")
    target = gates.record(paths, result)
    assert target == tmp_path / "gates" / "gate1_canary.json"
    loaded = load_record(GateResult, json.loads(target.read_text()))
    assert loaded.decision == gates.ADVANCE and len(loaded.criteria) == len(result.criteria)


def test_gate0_criteria():
    checks = [{"check": "a", "ok": True, "blocking": True}, {"check": "b", "ok": False, "blocking": False}]
    result = gates.evaluate_gate0(checks, pytest_returncode=0, pytest_summary="12 passed", smoke_report={"passed": True, "failures": []})
    assert result.decision == gates.ADVANCE
    result = gates.evaluate_gate0(checks + [{"check": "c", "ok": False, "blocking": True}], pytest_returncode=0, pytest_summary="", smoke_report={"passed": True})
    assert "doctor_preconditions" in failing(result) and result.decision == gates.STOP
    result = gates.evaluate_gate0(checks, pytest_returncode=1, pytest_summary="1 failed", smoke_report={"passed": True})
    assert "unit_suite_green" in failing(result) and result.decision == gates.STOP
    result = gates.evaluate_gate0(checks, pytest_returncode=0, pytest_summary="", smoke_report=None)
    assert "real_cli_smoke" in failing(result) and result.decision == gates.STOP_NOTIFY
    result = gates.evaluate_gate0(checks, pytest_returncode=0, pytest_summary="", smoke_report={"passed": False, "failures": ["codex: no spawn"]})
    assert "real_cli_smoke" in failing(result)


# ------------------------------------------------------------------------------------------------------------------
# Reconciliation
# ------------------------------------------------------------------------------------------------------------------

class StubClient:
    """Stands in for ``OpenRouterClient``: /key, /credits and /generation from dicts; optional 402 on /key."""

    def __init__(self, *, usage, limit=200, credits=("100", "20"), generations=None, payment_required=False, lag=0):
        self.usage = Decimal(usage)
        self.limit = Decimal(limit)
        self.credits_data = credits
        self.generations = generations or {}
        self.payment_required = payment_required
        self.lag = lag
        self.asked: dict[str, int] = {}

    async def key_info(self):
        if self.payment_required:
            from agent_allocation.api.client import OpenRouterAccountError
            from agent_allocation.contracts import TransportStatus
            raise OpenRouterAccountError("GET /key: http 402", transport_status=TransportStatus.PAYMENT_REQUIRED, http_status=402)
        return {"usage": self.usage, "limit": self.limit, "limit_remaining": self.limit - self.usage, "is_free_tier": False}

    async def credits(self):
        return {"total_credits": Decimal(self.credits_data[0]), "total_usage": Decimal(self.credits_data[1])}

    async def generation_metadata(self, gid):
        self.asked[gid] = self.asked.get(gid, 0) + 1
        if self.asked[gid] <= self.lag:
            return None
        return self.generations.get(gid)

    async def aclose(self):
        pass


async def _no_wait(_seconds):
    return None


class LaggingKeyClient(StubClient):
    """``/key`` usage that catches up over successive reads (the canary's lag: a read the second the stage ended
    showed $0.1139 of the ledger's $0.253105; the key reached it exactly within ~45 s)."""

    def __init__(self, usages, **kw):
        super().__init__(usage=usages[0], **kw)
        self.usages = [Decimal(u) for u in usages]
        self.reads = 0

    async def key_info(self):
        self.usage = self.usages[min(self.reads, len(self.usages) - 1)]
        self.reads += 1
        return await super().key_info()


def write_journal(paths: RunPaths, events):
    from agent_allocation.storage.journal import LedgerJournal
    from agent_allocation.contracts import LedgerEvent, LedgerEventKind
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    journal = LedgerJournal(paths.ledger_journal)
    seq = sum(1 for _ in journal.replay())
    for kind, aid, amount in events:
        seq += 1
        journal.append_sync(LedgerEvent(seq=seq, kind=LedgerEventKind(kind), attempt_id=aid, node_id=aid.split(":")[0],
                                        partition_id="canary", amount_usd=Decimal(amount), recorded_at="2026-09-23T00:00:00Z"))


def test_snapshot_never_records_the_label_and_402_is_flagged():
    snap = asyncio.run(reconcile.snapshot_account(StubClient(usage="10.5")))
    assert snap.key_usage_usd == Decimal("10.5") and snap.key_limit_remaining_usd == Decimal("189.5") and not snap.payment_required
    assert "label" not in json.dumps(snap.to_json())
    snap = asyncio.run(reconcile.snapshot_account(StubClient(usage="10", payment_required=True)))
    assert snap.payment_required and snap.key_usage_usd == 0


def test_sample_is_seeded_and_two_percent():
    population = [reconcile.CallGenerations(f"rk{i:03d}", 1, (f"g{i}",), 100, Decimal("0.001"), "Parasail") for i in range(250)]
    a = reconcile.sample_calls(population, seed=20260923)
    b = reconcile.sample_calls(list(reversed(population)), seed=20260923)
    assert len(a) == 5 and a == b
    assert reconcile.sample_calls(population[:10], seed=1) and len(reconcile.sample_calls(population[:10], seed=1)) == 1
    assert reconcile.sample_calls([], seed=1) == []


def test_compare_call_counts_a_forced_close_pair_once():
    call = reconcile.CallGenerations("rk1", 3, ("g-think", "g-answer"), 4096 + 300, Decimal("0.0110"), "Parasail")
    meta = {"g-think": {"tokens_completion": 4096, "native_tokens_completion": 4096, "total_cost": "0.010",
                        "provider_name": "Parasail"},
            "g-answer": {"tokens_completion": 300, "native_tokens_completion": 300, "total_cost": "0.0008",
                         "provider_name": "Parasail"}}
    row = reconcile.compare_call(call, meta)
    assert row["fetched"] and row["tokens_match"] and row["cost_within_tolerance"] and row["provider_ok"]
    row = reconcile.compare_call(call, {**meta, "g-answer": {**meta["g-answer"], "provider_name": "Mancer 2"}})
    assert row["provider_ok"] is False
    row = reconcile.compare_call(call, {"g-think": meta["g-think"]})
    assert not row["fetched"] and row["tokens_match"] is None
    off = reconcile.compare_call(reconcile.CallGenerations("rk1", 3, ("g-think",), 4096, Decimal("0.5"), None), meta)
    assert off["cost_within_tolerance"] is False and off["provider_ok"] is None


def test_cost_tolerance_is_a_cent_or_five_percent():
    assert reconcile.cost_within(Decimal("0.005"), Decimal("0.014"))
    assert not reconcile.cost_within(Decimal("0.005"), Decimal("0.016"))
    assert reconcile.cost_within(Decimal("1.00"), Decimal("1.05"))
    assert not reconcile.cost_within(Decimal("1.00"), Decimal("1.06"))
    assert reconcile.cost_within(None, Decimal("1")) is None


def test_before_after_report_clean_and_mismatch(tmp_path, cfg):
    paths = RunPaths.for_run(tmp_path, local_state_dir=tmp_path / "local")
    write_journal(paths, [("reserve", "old:1:think:1", "1.0"), ("sent", "old:1:think:1", "0"), ("settle", "old:1:think:1", "0.50")])
    before = asyncio.run(reconcile.snapshot_account(StubClient(usage="10.00")))
    reconcile.write_before(paths, "canary", before)
    doc = reconcile.read_before(paths, "canary")
    assert doc["ledger_charged_usd"] == "0.50" and doc["account"]["key"]["usage"] == "10.00"
    # the stage spends $0.30 settled + $0.10 charged at maximum; the key moves by $0.40
    write_journal(paths, [("reserve", "rk1:g1:1:think:1", "0.5"), ("sent", "rk1:g1:1:think:1", "0"), ("settle", "rk1:g1:1:think:1", "0.30"),
                          ("reserve", "rk1:g1:2:think:1", "0.2"), ("sent", "rk1:g1:2:think:1", "0"), ("charge_max", "rk1:g1:2:think:1", "0.10")])
    population = [reconcile.CallGenerations("rk1", 1, ("g1",), 4396, Decimal("0.30"), "Parasail")]
    client = StubClient(usage="10.40", generations={"g1": {"tokens_completion": 4396, "native_tokens_completion": 4396,
                                                                  "total_cost": "0.30", "provider_name": "Parasail"}}, lag=1)
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=client, population=population, population_source="tables",
                                                   upstream_402=False, key_tolerance_usd=Decimal("0.02")))
    assert report["decision"] == reconcile.CLEAN and report["problems"] == []
    assert report["key_delta_usd"] == "0.40" and report["ledger_delta_usd"] == "0.40"
    assert report["generation_sample"]["sampled"] == 1 and report["generation_sample"]["rows"][0]["tokens_match"]
    assert client.asked["g1"] == 2            # retried once after the 404 lag
    assert reconcile.read_report(paths, "canary")["decision"] == reconcile.CLEAN
    # a drifted key is a mismatch (once it has settled: two reads 30 s apart agree); an upstream 402 is stop_notify
    client = StubClient(usage="10.50", generations=client.generations)
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=client, population=population, population_source="tables",
                                                   upstream_402=False, key_tolerance_usd=Decimal("0.02"), sleep=_no_wait))
    assert report["decision"] == reconcile.MISMATCH and any("key delta" in p for p in report["problems"])
    assert report["key_settle"]["settled"] and len(report["key_settle"]["polls"]) == 2
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=StubClient(usage="10.40", generations=client.generations),
                                                   population=population, population_source="tables", upstream_402=True,
                                                   key_tolerance_usd=Decimal("0.02")))
    assert report["decision"] == reconcile.STOP_NOTIFY
    # an open reservation is a problem
    write_journal(paths, [("reserve", "rk1:g1:3:think:1", "0.2"), ("sent", "rk1:g1:3:think:1", "0")])
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=StubClient(usage="10.40", generations=client.generations),
                                                   population=population, population_source="tables", upstream_402=False,
                                                   key_tolerance_usd=Decimal("0.02")))
    assert any("open reservations" in p for p in report["problems"])
    with pytest.raises(reconcile.ReconcileError):
        reconcile.read_before(paths, "pilot")


def test_generation_tokens_are_compared_on_the_native_count():
    """Review item 7a: the journal holds the provider's own usage.completion_tokens, i.e. /generation's
    ``native_tokens_completion``; ``tokens_completion`` is OpenRouter's normalised re-count (canary: 4,323 native
    vs 2,388 normalised on one call with an identical cost). Comparing against the normalised count reported a
    false token mismatch."""
    call = reconcile.CallGenerations("a2e99fad9322c6758c6d11ea", 1, ("g1",), 4323, Decimal("0.0132775"), "Parasail")
    meta = {"g1": {"tokens_completion": 2388, "native_tokens_completion": 4323, "total_cost": "0.0132775",
                   "provider_name": "Parasail"}}
    row = reconcile.compare_call(call, meta)
    assert row["metadata_completion_tokens"] == 4323 and row["tokens_match"] is True
    assert row["metadata_token_field"] == "native_tokens_completion"
    # a metadata document without the native field is never silently compared on the normalised one
    row = reconcile.compare_call(call, {"g1": {"tokens_completion": 4323, "total_cost": "0.0132775"}})
    assert row["tokens_match"] is None and row["native_tokens_missing"] is True


def test_dollar_verdict_waits_for_the_key_to_settle(tmp_path, cfg):
    """Review item 7b: /key lags the ledger. The after-read is re-polled every 30 s until two reads agree (bounded at
    5 minutes) before a dollar mismatch is declared; a key that catches up to the ledger is clean."""
    paths = RunPaths.for_run(tmp_path, local_state_dir=tmp_path / "local")
    reconcile.write_before(paths, "canary", asyncio.run(reconcile.snapshot_account(StubClient(usage="10.00"))))
    write_journal(paths, [("reserve", "rk1:g1:1:think:1", "0.5"), ("sent", "rk1:g1:1:think:1", "0"),
                          ("settle", "rk1:g1:1:think:1", "0.253105")])
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    client = LaggingKeyClient(["10.1138925", "10.200", "10.253105", "10.253105"])
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=client, population=[],
                                                   population_source="journal", upstream_402=False,
                                                   key_tolerance_usd=Decimal("0.02"), sleep=sleep))
    assert report["decision"] == reconcile.CLEAN and report["problems"] == []
    assert report["key_delta_usd"] == "0.253105" and slept == [30.0, 30.0]
    assert [p["key_usage_usd"] for p in report["key_settle"]["polls"]] == ["10.1138925", "10.200", "10.253105"]
    # a key that keeps moving is polled for at most 5 minutes, then the last read decides and says so
    slept.clear()
    moving = LaggingKeyClient([f"10.{i:02d}" for i in range(1, 30)])
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=moving, population=[],
                                                   population_source="journal", upstream_402=False,
                                                   key_tolerance_usd=Decimal("0.02"), sleep=sleep))
    assert sum(slept) == 300.0 and report["decision"] == reconcile.MISMATCH
    assert report["key_settle"]["settled"] is False and any("did not settle" in p for p in report["problems"])
    # a settled key that disagrees is a mismatch after exactly one 30 s re-read
    slept.clear()
    report = asyncio.run(reconcile.reconcile_after(cfg, paths, "canary", client=StubClient(usage="10.40"), population=[],
                                                   population_source="journal", upstream_402=False,
                                                   key_tolerance_usd=Decimal("0.02"), sleep=sleep))
    assert slept == [30.0] and report["decision"] == reconcile.MISMATCH and report["key_settle"]["settled"] is True


def test_population_from_tables_and_upstream_402(tmp_path, cfg):
    t = clean_canary(cfg)
    runs, calls = t.frames()
    from agent_allocation.api.fake import fake_forced_endpoint_lock
    population = reconcile.calls_from_table(calls, runs, {"parasail": fake_forced_endpoint_lock(tag="parasail/fp8")})
    assert len(population) == len(calls) and population[0].expected_provider == "Parasail"
    forced = next(c for c in population if c.completion_tokens == 4096 + 300)
    assert forced.cost_usd == Decimal("0.001")
    assert not reconcile.upstream_402_seen(runs)
    runs.loc[0, "error"] = "upstream 402 payment_required"
    assert reconcile.upstream_402_seen(runs)
    with pytest.raises(reconcile.ReconcileError):
        reconcile.calls_from_table(calls.drop(columns=["generation_ids"]), runs, {})


def test_population_from_journal(tmp_path):
    import gzip
    raw = tmp_path / "raw" / "H1" / "4" / "1" / "aime_1" / "a1"
    raw.mkdir(parents=True)
    doc = {"layout": "dynorch/v1", "run_key": "rk1", "call_no": 1, "phases": [{"phase": "think", "generation_id": "g1"}, {"phase": "answer", "generation_id": "g2"}],
           "budget": {"granted": 4096, "realized": 4096, "remaining_after": 0}, "reply": {"grace_tokens": 200}}
    with gzip.open(raw / "call_0001.json.gz", "wb") as h:
        h.write(json.dumps(doc).encode())
    population = reconcile.calls_from_journal(tmp_path)
    assert population == [reconcile.CallGenerations("rk1", 1, ("g1", "g2"), 4296, None, None)]
    with gzip.open(raw / "call_0002.json.gz", "wb") as h:
        h.write(json.dumps({"layout": "other"}).encode())
    with pytest.raises(reconcile.ReconcileError):
        reconcile.calls_from_journal(tmp_path)


def test_key_tolerance_by_stage():
    assert reconcile.key_tolerance_for("canary") == Decimal("0.02") and reconcile.key_tolerance_for("pilot") == Decimal("0.02")
    assert reconcile.key_tolerance_for("main", Decimal("10")) == Decimal("0.50")
    assert reconcile.key_tolerance_for("main", Decimal("60")) == Decimal("1.20")
