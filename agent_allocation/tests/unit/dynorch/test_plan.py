"""WP-B: plan enumeration against the sealed base run root (contract §6; frozen table)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from agent_allocation.dynorch import plan as planmod
from agent_allocation.dynorch.plan import (
    ALLOWED_TOOLS_BY_HARNESS, BaseInputs, cell_rung_order, plan_runs, plan_stage, read_plan, run_key_for, write_plan,
)

PROJECT = Path(__file__).resolve().parents[3]
CFG_PATH = PROJECT / "configs" / "dynorch_v1.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return json.loads(CFG_PATH.read_text("utf-8"))


@pytest.fixture(scope="module")
def base(cfg) -> BaseInputs:
    root = Path(cfg["base_run_root"])
    if not (root / "manifests" / "panels.json").is_file():
        pytest.skip(f"base run root {root} not readable here")
    return BaseInputs.load(root, model=cfg["primary_model"])


def test_base_inputs_load_the_sealed_panels_tasks_and_shards(base: BaseInputs):
    assert len(base.main_items()) == 96 and len(base.dev_items()) == 32
    assert all(i in base.tasks and i in base.privates for i in base.main_items())
    assert Counter(base.assignment[i] for i in base.main_items()) == {"mancer": 58, "parasail": 38}
    assert base.endpoint_tags == {"mancer": "mancer/fp8", "parasail": "parasail/fp8"}


def test_main_stage_is_96_items_by_core_arms_by_rungs(cfg, base):
    specs = plan_stage(cfg, "main", base)
    assert len(specs) == 96 * 3 * 6
    assert Counter(s.arm for s in specs) == {"H1": 576, "H3": 576, "H0": 576}
    assert Counter(s.rung for s in specs) == {r: 288 for r in (1, 2, 4, 8, 16, 25)}
    assert len({s.run_key for s in specs}) == len(specs)
    assert all(s.cap_tokens == s.rung * 1024 for s in specs)
    assert all(s.partition == "main" and s.repeat == 1 and s.attempt == 1 and s.generation == 1 for s in specs)
    assert all(s.prompt_version == "dynorch-v1" for s in specs)
    by_arm = {s.arm: s for s in specs}
    assert by_arm["H1"].harness == "openclaude" and by_arm["H1"].allowed_tools == ("Agent",) and by_arm["H1"].spawn_tools == ("Agent",)
    assert by_arm["H3"].harness == "codex" and by_arm["H3"].spawn_tools == ("spawn_agent",)
    assert set(by_arm["H3"].allowed_tools) >= {"spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent", "update_plan"}
    assert by_arm["H0"].harness == "null" and by_arm["H0"].allowed_tools == () and by_arm["H0"].spawn_tools == ()
    assert all(s.root_call_ceiling is None for s in specs)
    sample = specs[0]
    assert sample.shard_id == base.assignment[sample.item_id]
    assert sample.endpoint_tag == base.endpoint_tags[sample.shard_id]
    assert sample.stratum == sample.benchmark == base.tasks[sample.item_id].benchmark


def test_run_key_is_the_contract_hash_and_plan_is_deterministic(cfg, base):
    a = plan_stage(cfg, "main", base)
    b = plan_stage(cfg, "main", base)
    assert [s.run_key for s in a] == [s.run_key for s in b]
    s = a[0]
    expected = hashlib.sha256(f"{cfg['study_id']}|{s.arm}|{s.item_id}|{s.rung}|{s.repeat}|{s.attempt}".encode()).hexdigest()[:24]
    assert s.run_key == expected == run_key_for(cfg["study_id"], s.arm, s.item_id, s.rung, s.repeat, s.attempt)
    assert run_key_for(cfg["study_id"], s.arm, s.item_id, s.rung, s.repeat, 2) != s.run_key


def test_order_interleaves_items_and_uses_cell_rung_order_inside_each_item(cfg, base):
    specs = plan_stage(cfg, "main", base)
    assert cell_rung_order(cfg) == (16, 4, 8, 25, 2, 1)
    per_item = 3 * 6
    first_item = specs[:per_item]
    assert len({s.item_id for s in first_item}) == 1
    assert [s.rung for s in first_item[::3]] == [16, 4, 8, 25, 2, 1]
    assert [s.arm for s in first_item[:3]] == ["H1", "H3", "H0"]
    # the first eight items mix benchmarks (seeded interleave, not panel order)
    first_items = []
    for s in specs:
        if s.item_id not in first_items:
            first_items.append(s.item_id)
        if len(first_items) == 8:
            break
    assert len({base.tasks[i].benchmark for i in first_items}) >= 2


def test_pilot_stage_is_dev_items_with_h0_subset(cfg, base):
    specs = plan_stage(cfg, "pilot", base)
    dev = set(base.dev_items())
    assert all(s.item_id in dev for s in specs)
    assert Counter((s.arm, s.rung) for s in specs) == {("H1", 2): 32, ("H1", 16): 32, ("H3", 2): 32, ("H3", 16): 32,
                                                         ("H0", 16): 16}
    h0_items = {s.item_id for s in specs if s.arm == "H0"}
    assert Counter(base.tasks[i].benchmark for i in h0_items) == {b: 4 for b in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")}
    assert all(s.partition == "pilot" for s in specs)


def test_canary_stage(cfg, base):
    specs = plan_stage(cfg, "canary", base)
    cells = Counter((s.arm, s.rung) for s in specs)
    assert cells == {("H1", 4): 2, ("H3", 4): 2, ("H0", 4): 2, ("H1", 16): 2, ("H3", 16): 2}
    canary_items = {s.item_id for s in specs if s.rung == 4}
    assert canary_items == {base.panels["aime"].dev_item_ids[0], base.panels["gpqa_diamond"].dev_item_ids[0]}


def test_extension_stages(cfg, base):
    e0 = plan_stage(cfg, "ext_e0", base)
    assert Counter((s.arm, s.rung) for s in e0) == {("H1c", 4): 96, ("H1c", 16): 96, ("H3c", 4): 96, ("H3c", 16): 96}
    assert {s.root_call_ceiling for s in e0 if s.rung == 4} == {1024} and {s.root_call_ceiling for s in e0 if s.rung == 16} == {4096}
    e1 = plan_stage(cfg, "ext_e1", base)
    assert len(e1) == 96 * 2 * 6 and {s.repeat for s in e1} == {2}
    assert not ({s.run_key for s in e1} & {s.run_key for s in plan_stage(cfg, "main", base)})
    e2 = plan_stage(cfg, "ext_e2", base)
    assert Counter(s.arm for s in e2) == {"H1n": 192, "H3n": 192}
    assert all(s.spawn_tools == () and s.allowed_tools == () for s in e2)
    e3 = plan_stage(cfg, "ext_e3", base)
    assert Counter(s.arm for s in e3) == {"H2": 192} and e3[0].harness == "openclaw"
    assert e3[0].spawn_tools == ("sessions_spawn",) and e3[0].allowed_tools == ALLOWED_TOOLS_BY_HARNESS["openclaw"]


def test_stage_filters_by_arms_and_rungs(cfg, base):
    specs = plan_stage(cfg, "main", base, arms=["H1"], rungs=[16])
    assert len(specs) == 96 and {s.arm for s in specs} == {"H1"} and {s.rung for s in specs} == {16}


def test_plan_runs_rejects_unknown_rungs_arms_items_and_ceilings(cfg, base):
    items = base.main_items()[:2]
    with pytest.raises(ValueError):
        plan_runs(cfg, stage="main", arms=["H1"], rungs=[3], repeats=[1], items=items, base=base)
    with pytest.raises(KeyError):
        plan_runs(cfg, stage="main", arms=["H9"], rungs=[1], repeats=[1], items=items, base=base)
    with pytest.raises(KeyError):
        plan_runs(cfg, stage="main", arms=["H1"], rungs=[1], repeats=[1], items=["aime/9999/1"], base=base)
    with pytest.raises(ValueError):
        plan_runs(cfg, stage="nope", arms=["H1"], rungs=[1], repeats=[1], items=items, base=base)
    bad = json.loads(json.dumps(cfg))
    bad["dynorch"]["extension_arms"]["H1c"]["root_call_ceiling"] = "C/2"
    with pytest.raises(ValueError):
        plan_runs(bad, stage="ext_e0", arms=["H1c"], rungs=[4], repeats=[1], items=items, base=base)
    bad["dynorch"]["cell_rung_order"] = [16, 4]
    with pytest.raises(ValueError):
        cell_rung_order(bad)
    no_surface = json.loads(json.dumps(cfg))
    del no_surface["dynorch"]["arms"]["H1"]["allowed_tools"]
    with pytest.raises(KeyError, match="allowed_tools"):
        plan_runs(no_surface, stage="main", arms=["H1"], rungs=[1], repeats=[1], items=items, base=base)
    widened = json.loads(json.dumps(cfg))
    widened["dynorch"]["extension_arms"]["H1n"]["allowed_tools"] = ["Agent"]
    with pytest.raises(ValueError, match="spawning_disabled"):
        plan_runs(widened, stage="ext_e2", arms=["H1n"], rungs=[4], repeats=[1], items=items, base=base)


def test_allowed_tools_come_from_the_frozen_config_arm(cfg, base):
    specs = plan_stage(cfg, "main", base, rungs=[16])
    by_arm = {s.arm: s for s in specs}
    assert by_arm["H3"].allowed_tools == tuple(cfg["dynorch"]["arms"]["H3"]["allowed_tools"])
    assert set(by_arm["H3"].allowed_tools) == {"update_plan", "spawn_agent", "send_input", "wait_agent", "resume_agent",
                                               "close_agent"}
    assert by_arm["H1"].allowed_tools == ("Agent",)


def test_write_and_read_plan_round_trip(cfg, base, tmp_path):
    specs = plan_stage(cfg, "canary", base)
    path = write_plan(specs, tmp_path, "canary")
    assert path == tmp_path / "manifests" / "dynorch_plan_canary.jsonl"
    assert read_plan(path) == specs
    lines = path.read_text().splitlines()
    assert len(lines) == len(specs) and json.loads(lines[0])["run_key"] == specs[0].run_key


def test_versions_are_stamped(cfg, base):
    specs = plan_runs(cfg, stage="main", arms=["H1", "H3"], rungs=[1], repeats=[1], items=base.main_items()[:1],
                      base=base, proxy_version="proxy-1", harness_versions={"openclaude": "0.13.0"})
    by_arm = {s.arm: s for s in specs}
    assert by_arm["H1"].proxy_version == "proxy-1" and by_arm["H1"].harness_version == "0.13.0"
    assert by_arm["H3"].harness_version == "unrecorded"
    assert planmod.PROMPT_VERSION == "dynorch-v1"
