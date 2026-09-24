"""WP-C: dynorch/forecast.py reproduces the plan section 8 table from the config (contract section 7)."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.dynorch import forecast as F
from agent_allocation.dynorch import tables as T

ROOT = Path(__file__).resolve().parents[3]
CFG = json.loads((ROOT / "configs" / "dynorch_v1.json").read_text())


def _row(table: pd.DataFrame, block: str) -> pd.Series:
    return table.set_index("block").loc[block]


def test_named_assumptions_are_the_plans():
    assert F.PARASAIL_INPUT_USD_PER_M == Decimal("0.24") and F.PARASAIL_OUTPUT_USD_PER_M == Decimal("2.20")
    assert F.BASE_PROMPT_TOKENS == 5_500
    assert F.CALLS_PER_RUN == {1: 2, 2: 3, 4: 4, 8: 6, 16: 8, 25: 10}
    assert F.contingency_share(CFG) == Decimal("0.25")


def test_per_run_forecast_uses_cap_plus_grace_and_reinjection():
    rf = F.run_forecast("harness", 16, CFG)
    assert rf.output_tokens == 16 * 1024 + 1024 and rf.calls == 8
    assert rf.input_tokens_ceiling == 8 * (5_500 + rf.output_tokens)
    assert rf.input_tokens_expected == 8 * 5_500
    h0 = F.run_forecast("null", 16, CFG)
    assert h0.calls == 1 and h0.input_tokens_ceiling == h0.input_tokens_expected == 5_500
    with pytest.raises(F.ForecastError):
        F.run_forecast("harness", 3, CFG)
    with pytest.raises(F.ForecastError):
        F.run_forecast("weird", 16, CFG)


def test_per_item_ceilings_match_section_8():
    per = F.per_item_table(CFG).set_index("kind")
    assert per.loc["harness", "output_usd"] == pytest.approx(0.14, abs=0.005)
    assert per.loc["harness", "input_usd_ceiling"] == pytest.approx(0.16, abs=0.005)
    assert per.loc["harness", "total_usd_ceiling"] == pytest.approx(0.30, abs=0.005)
    assert per.loc["null", "total_usd_ceiling"] == pytest.approx(0.15, abs=0.005)


def test_block_runs_and_core_ceilings_reproduce_section_8():
    tab = F.plan_table(CFG)
    for block, (runs, ceiling, lo, hi) in F.PLAN_SECTION_8.items():
        assert int(_row(tab, block)["runs"]) == runs, block
    # ceilings of the core rows within rounding of the plan's stated numbers
    for block, tol in (("core_h1_h3", 1.0), ("core_h0", 0.5), ("pilot", 0.5), ("core_total", 1.0), ("e1_repeat", 1.0)):
        assert abs(_row(tab, block)["ceiling_usd"] - F.PLAN_SECTION_8[block][1]) <= tol, block
    # expected (flat-input) figures land in the plan's expected ranges where the plan gives a range
    assert 35.0 - 0.5 <= _row(tab, "core_h1_h3")["expected_usd"] <= 45.0
    assert 50.0 <= _row(tab, "core_total")["expected_usd"] <= 60.0
    assert abs(_row(tab, "pilot")["expected_usd"] - 4.0) <= 0.5
    # the canary row is an allowance: forecast under it and under the config's canary allowance
    assert _row(tab, "canary_gate1")["ceiling_usd"] <= 1.5 <= CFG["budget"]["canary_allowance_usd"]
    # E0 / E2: the plan's $12 is the flat-input figure; the re-injection ceiling is higher (request R5)
    for block in ("e0_capped", "e2_spawn_disabled"):
        assert abs(_row(tab, block)["expected_usd"] - 12.0) <= 0.6
        assert _row(tab, block)["ceiling_usd"] > 12.0
    # every block: expected <= ceiling, contingency = 25 % of the ceiling
    assert (tab["expected_usd"] <= tab["ceiling_usd"] + 1e-9).all()
    assert (tab["contingency_usd"] == (tab["ceiling_usd"] * 0.25).round(4)).all()
    assert _row(tab, "core_total")["ceiling_usd"] <= CFG["budget"]["credit_ceiling_usd"]


def test_arm_kind_from_config():
    assert F.arm_kind("H0", CFG) == "null"
    assert F.arm_kind("H1", CFG) == F.arm_kind("H1c", CFG) == F.arm_kind("H2", CFG) == "harness"
    with pytest.raises(F.ForecastError):
        F.arm_kind("H9", CFG)


def test_refit_from_measured_runs():
    items = {f"aime/x{i}": {"benchmark": "aime", "shard_id": "parasail", "endpoint_tag": "parasail/fp8"} for i in range(4)}
    plan = T.synthetic_plan(CFG, items, arms=["H1", "H0"], rungs=[4, 16])
    records = [T.synthetic_run_record(c, correct=True, spawns=1 if c.arm == "H1" else 0, realized_share=0.5,
                                      grace=100) for c in plan]
    runs = T.runs_table(records)
    out = F.refit(runs, CFG, planned_runs={("H1", 4): 96, ("H1", 16): 96, ("H0", 4): 96, ("H0", 16): 96})
    row = out.set_index(["arm", "rung"]).loc[("H1", 16)]
    g = runs[(runs["arm"] == "H1") & (runs["rung"] == 16)]
    prompt = g["prompt_total"].mean()
    completion = (g["completion_total"] + g["grace_tokens"]).mean()
    assert row["mean_completion_tokens"] == pytest.approx(completion)
    assert row["mean_realized_share"] == pytest.approx(0.5, abs=1e-6)
    expected = (prompt * 0.24 + completion * 2.20) / 1e6
    assert row["cost_usd_per_run_refit"] == pytest.approx(expected, rel=1e-9)
    assert row["remaining_runs"] == 92
    assert row["forecast_remaining_usd"] == pytest.approx(92 * expected * 1.25, rel=1e-9)
    with pytest.raises(F.ForecastError, match="lacks"):
        F.refit(runs.drop(columns=["prompt_total"]), CFG)
    with pytest.raises(F.ForecastError, match="empty"):
        F.refit(runs.iloc[0:0], CFG)


def test_cli_prints_the_table(capsys):
    assert F.main(["--config", str(ROOT / "configs" / "dynorch_v1.json")]) == 0
    out = capsys.readouterr().out
    assert "core_h1_h3" in out and "core_total" in out
