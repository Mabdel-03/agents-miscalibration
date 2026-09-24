"""Integration lead: the seams between the work packages (contract §6 ``plan_runs`` vs WP-D's cell enumeration,
``runner.launch_stage`` as ``dynorch-run`` calls it, ``forecast.gate2_refit`` as Gate 2 calls it).

The launch test drives the real verb in rehearsal (``--rehearsal``: the fake OpenRouter transport, no key, no
network) for the canary's H0 cells: endpoint rule frozen, proxy bound and announced through ``state/proxy.json``
before the first run, two H0 runs through the real proxy stack, tables built, the stage reconciled. Scratch and
run root live under ``tmp_path`` (never the study scratch root, never ``/home``).
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.config import load_config
from agent_allocation.dynorch import cli as dcli
from agent_allocation.dynorch import forecast as F
from agent_allocation.dynorch import tables as T
from agent_allocation.dynorch.plan import BaseInputs, plan_stage
from agent_allocation.paths import RunPaths

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


@pytest.fixture(scope="module")
def base(cfg) -> BaseInputs:
    root = Path(cfg["base_run_root"])
    if not (root / "manifests" / "panels.json").is_file():
        pytest.skip("base run root not readable here")
    return BaseInputs.load(root, model=cfg["primary_model"])


# ---- plan_stage == enumerate_cells (Gate 4 checks planned cells against rows) -----------------------------------

@pytest.mark.parametrize("stage", ["canary", "pilot", "main", "ext_e0", "ext_e1", "ext_e2", "ext_e3"])
def test_plan_stage_and_enumerate_cells_agree_cell_for_cell(cfg, base, stage):
    planned = {(s.arm, s.rung, s.item_id, s.repeat) for s in plan_stage(cfg, stage, base)}
    enumerated = {(c["arm"], c["rung"], c["item_id"], c["repeat"]) for c in dcli.enumerate_cells(cfg, stage)}
    assert planned == enumerated, (stage, sorted(planned ^ enumerated)[:5])
    assert {s.partition for s in plan_stage(cfg, stage, base)} == {stage}
    expected = {"canary": 10, "pilot": 144, "main": 1728, "ext_e0": 384, "ext_e1": 1152, "ext_e2": 384, "ext_e3": 192}
    assert len(planned) == expected[stage]


# ---- gate2_refit ---------------------------------------------------------------------------------------------------

def _pilot_runs(cfg, *, realized_share: float) -> pd.DataFrame:
    items = {f"aime/p{i}": {"benchmark": "aime", "shard_id": "parasail", "endpoint_tag": "parasail/fp8"} for i in range(4)}
    plan = T.synthetic_plan(cfg, items, arms=["H1", "H3"], rungs=[2, 16])
    records = [T.synthetic_run_record(c, correct=True, spawns=1, realized_share=realized_share, grace=100)
               for c in plan]
    return T.runs_table(records)


def test_gate2_refit_measured_cells_and_section_8_fallback(cfg):
    runs = _pilot_runs(cfg, realized_share=0.5)
    out = F.gate2_refit(cfg, runs, spent_usd="3")
    assert out["spent_usd"] == "3" and out["soft_ceiling_usd"] == "90" and out["room_usd"] == "87"
    assert out["cuts_applied"] == [] and out["fits"] is True
    per = out["per_cell"]
    assert per["H1/16"]["source"] == "pilot_refit" and per["H1/2"]["source"] == "pilot_refit"
    assert per["H1/4"]["source"] == "section_8_ceiling" and per["H0/16"]["source"] == "section_8_ceiling"
    assert per["H1/16"]["remaining"] == 96 - 4 and per["H0/1"]["remaining"] == 96
    refit = F.refit(runs, cfg).set_index(["arm", "rung"])
    assert Decimal(per["H1/16"]["usd_per_run"]) == Decimal(str(refit.loc[("H1", 16), "cost_usd_per_run_refit"]))
    assert Decimal(per["H0/16"]["usd_per_run"]) == F.run_forecast("null", 16, cfg).usd_ceiling
    total = sum(Decimal(c["usd_with_contingency"]) for c in per.values())
    assert Decimal(out["core_forecast_usd"]) == total
    assert Decimal(per["H1/16"]["usd_with_contingency"]) == 92 * Decimal(per["H1/16"]["usd_per_run"]) * Decimal("1.25")


def test_gate2_refit_applies_cuts_in_order_until_it_fits(cfg):
    runs = _pilot_runs(cfg, realized_share=1.0)
    uncut = Decimal(F.gate2_refit(cfg, runs, spent_usd="0")["core_forecast_usd"])
    # room just below the uncut forecast: C1 alone must do
    c1 = F.gate2_refit(cfg, runs, spent_usd=str(Decimal("90") - uncut + Decimal("0.01")))
    assert c1["cuts_applied"] == ["C1 drop H0 at rungs {1,2,8}"] and c1["fits"] is True
    assert Decimal(c1["core_forecast_usd"]) < uncut
    # no room at all: every automatic cut applies and the forecast still does not fit (C4 is the owner's)
    none = F.gate2_refit(cfg, runs, spent_usd="90")
    assert none["cuts_applied"] == [name for name, _ in F.CUTS] and none["fits"] is False
    assert Decimal(none["core_forecast_usd"]) < Decimal(c1["core_forecast_usd"])
    dropped = {("H0", 1), ("H0", 2), ("H0", 8), ("H1", 1), ("H3", 1), ("H1", 2), ("H3", 2)}
    kept = sum(Decimal(c["usd_with_contingency"]) for c in none["per_cell"].values()
               if (c["arm"], c["rung"]) not in dropped)
    assert Decimal(none["core_forecast_usd"]) == kept


# ---- launch_stage through the verb, in rehearsal --------------------------------------------------------------------

def _rehearsal_config(cfg, tmp_path: Path) -> Path:
    doc = json.loads(CFG_PATH.read_text())
    doc["dynorch"]["scratch_root"] = str(tmp_path / "scratch")
    path = tmp_path / "dynorch_rehearsal.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


@pytest.mark.timeout(600)
def test_dynorch_run_canary_h0_in_rehearsal_runs_the_real_stack(cfg, base, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALLOCATION_LOCAL_STATE", str(tmp_path / "local"))
    monkeypatch.delenv(dcli.REHEARSAL_ENV, raising=False)
    config_path = _rehearsal_config(cfg, tmp_path)
    home_marker = tmp_path / "marker"
    home_marker.write_text("x")
    rc = dcli.cmd_dynorch_run(["--config", str(config_path), "--run-dir", str(tmp_path / "run"), "--stage", "canary",
                               "--max-usd", "1", "--allow-paid", "--rehearsal", "--arms", "H0", "--concurrency", "2"])
    err = capsys.readouterr()
    assert rc == 0, err.err
    paths = RunPaths.for_run(tmp_path / "run")
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "finished" and status["cells"] == 2 and status["endpoint_rule"] == "keep"
    runner = status["runner"]
    assert runner["runs"] == 2 and runner["done"] == 2 and runner["executed"] == 2 and runner["upstream_402"] is False
    assert Decimal(runner["settled_usd"]) > 0 and Decimal(runner["unresolved_usd"]) == 0 and Decimal(runner["in_flight_usd"]) == 0
    proxy = json.loads((paths.state_dir / "proxy.json").read_text())
    assert proxy["host"] == "127.0.0.1" and proxy["port"] > 0 and proxy["stage"] == "canary"
    # the two H0 runs went through the real proxy: records, raw call journals, the pre-registration file
    from agent_allocation.dynorch.journal import RunStore
    records = RunStore(paths.run_dir).scan()
    assert len(records) == 2 and all(r.status == "done" and r.calls == 1 for r in records.values())
    assert all(r.spec.arm == "H0" and r.spec.rung == 4 and r.spec.partition == "canary" for r in records.values())
    assert all(r.parse_status == "valid" and r.correct in (True, False) for r in records.values())
    assert all(r.spec.proxy_version == "dynorch-proxy-1.0.0" for r in records.values())
    assert {r.spec.item_id for r in records.values()} == {base.panels["aime"].dev_item_ids[0],
                                                          base.panels["gpqa_diamond"].dev_item_ids[0]}
    assert len(list((paths.run_dir / "raw").rglob("call_*.json.gz"))) == 2
    assert (paths.manifests_dir / "dynorch_plan_canary.jsonl").is_file()
    assert (paths.run_dir / "state" / "ledger_journal.jsonl").is_file()
    # tables over the stage, and the reconciliation of the stage from them
    assert status["tables"] == {"n_planned": 2, "n_records": 2, "reasoning_unknown_rows": 0}
    outcomes = pd.read_parquet(paths.tables_dir / "policy_outcomes_dynorch.parquet")
    assert len(outcomes) == 2 and set(outcomes["delivered_state"]) == {"delivered"} and set(outcomes["name"]) == {"H0/4"}
    runs = pd.read_parquet(paths.tables_dir / "dynorch_runs.parquet")
    assert set(runs["partition"]) == {"canary"} and list(runs["calls"]) == [1, 1]
    assert status["reconciliation"]["decision"] == "clean", status["reconciliation"]
    # work dirs under the rehearsal scratch root, the harness env carries the run key only
    work = sorted((tmp_path / "scratch" / "work" / "H0" / "4" / "1").glob("*/a1/launch.json"))
    assert len(work) == 2
    for launch in work:
        doc = json.loads(launch.read_text())
        assert doc["env"]["OPENAI_API_KEY"] == doc["run_key"] and "sk-or-" not in launch.read_text()
    assert dcli.key_hits_in_run_tree(paths.run_dir) == []
    assert not any(p.is_relative_to(Path.home()) for p in (paths.run_dir, tmp_path / "scratch"))
