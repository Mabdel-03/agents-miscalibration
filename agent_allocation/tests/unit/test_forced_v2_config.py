"""WP-3: the forced27b preset validates, old presets still do, and its grid (deep/25, no high/replication) builds."""
from __future__ import annotations

import copy
import importlib.util
import sys
from decimal import Decimal

import pytest

from agent_allocation.config import (
    PRESET_CEILINGS_USD, forced_mode, grids_from_config, high_enabled, load_config, replication_enabled,
    validate_config,
)
from agent_allocation.contracts import OutputKind, PublicTask
from agent_allocation.plan import generation_plan
from agent_allocation.protocols.grid import build_item_graph, checkpoint_by_name, expected_node_count
from tests.forced_cfg import PROJECT, forced_test_config, needs_benchmark_swap, raw_forced_config


@pytest.fixture()
def forced():
    return forced_test_config()


def test_old_presets_still_validate():
    for name in ("balanced150", "lean100"):
        assert validate_config(load_config(PROJECT / "configs" / f"{name}.json")) == []


def test_forced_preset_validates(forced):
    assert PRESET_CEILINGS_USD["forced27b"] == Decimal("95")
    assert validate_config(forced) == []
    assert forced_mode(forced) and not high_enabled(forced) and not replication_enabled(forced)


def test_raw_forced_config_only_fails_on_the_benchmark_block():
    problems = validate_config(raw_forced_config())
    if not needs_benchmark_swap():
        assert problems == []                      # after the WP-2 merge
    else:
        assert problems and all("benchmark" in p.lower() or "BBH" in p for p in problems), problems


@pytest.mark.parametrize("mutate, fragment", [
    (lambda c: c["budget"].update(credit_ceiling_usd=100), "credit ceiling"),
    (lambda c: c["reasoning"].update(requested_reasoning_max=25599), "exceeds requested_reasoning_max"),
    (lambda c: c["panels"].update(high_items=16), "high_items"),
    (lambda c: c["panels"].update(replication_items=16), "replication_items"),
    (lambda c: c["protocols"].update(deep_high_multipliers=[2]), "deep_high_multipliers"),
    (lambda c: c["reasoning"].update(budget_semantics="soft_requested_allowance"), "forced_exact"),
    (lambda c: c["reasoning"].update(budget_semantics="whatever"), "budget_semantics"),
    (lambda c: c["forced"].update(ban_token_ids=[]), "ban_token_ids"),
    (lambda c: c["forced"].update(ban_bias=-5), "ban_bias"),
    (lambda c: c["forced"].update(endpoint="chat"), "forced.endpoint"),
    (lambda c: c["forced"].update(think_close_text="\n\n"), "think_close_text"),
    (lambda c: c["forced"].update(answer_attempts=0), "answer_attempts"),
    (lambda c: c["shards"].append({"id": "parasail", "endpoint_tag": "x/y", "share": 1}), "shard ids must be unique"),
    (lambda c: c["shards"].append({"id": "other", "endpoint_tag": "mancer/fp8", "share": 1}), "tags must be unique"),
    (lambda c: c["shards"][0].update(share=0), "share must be a number > 0"),
    (lambda c: c["shards"][0].update(id="Bad Id"), "id must match"),
    (lambda c: c["provider"].update(endpoint_tag="parasail/fp8"), "leave provider.endpoint_tag null"),
    (lambda c: c.update(shards=[]), "non-empty list"),
    (lambda c: c["calibration"].update(forced_pass_at_1_min=0.9), "calibration gate"),
    (lambda c: c["sampling"].update(top_k=0), "top_k"),
    (lambda c: c["panels"].update(extension_items=112), "extension panel"),
    (lambda c: c.update(study_id="openrouter_agent_allocation_v1"), "study_id"),
])
def test_forced_validation_rejects(forced, mutate, fragment):
    cfg = copy.deepcopy(forced)
    mutate(cfg)
    problems = validate_config(cfg)
    assert any(fragment in p for p in problems), problems


def test_grids_skip_disabled_partitions(forced):
    grids = grids_from_config(forced)
    assert "high" not in grids and "replication" not in grids
    assert grids["main"].deep_multipliers == (2, 4, 8, 16, 25)
    lean = grids_from_config(load_config(PROJECT / "configs" / "lean100.json"))
    assert "high" in lean and "replication" in lean           # old presets unchanged


def test_deep25_grid_counts_and_checkpoints(forced):
    grids = grids_from_config(forced)
    main, ext = grids["main"], grids["main_extension"]
    assert expected_node_count(main) == 126 and expected_node_count(ext) == 139
    task = PublicTask(item_id="x/1", benchmark=forced["benchmarks"][0]["id"], source_revision="r", prompt="p",
                      output_kind=OutputKind.MATH)
    graph = build_item_graph(ext, run_id=forced["study_id"], partition_id="main", task=task, model="m",
                             base_budget=1024, headroom=4096, endpoint_lock_hash="l", decoding_config_hash="d",
                             prompt_version="v")
    assert len(graph.nodes) == 139
    deep25 = checkpoint_by_name(graph, "deep/25")
    assert deep25.reasoning_budget == 25600 and len(deep25.required_node_ids) == 1
    budgets = sorted(n.reasoning_budget for n in graph.nodes if n.kind.value == "deep")
    assert budgets == [2048, 4096, 8192, 16384, 25600]
    assert sum(n.reasoning_budget for n in graph.nodes) == 189 * 1024      # the plan's 189 A per item
    names = {c.name for c in graph.checkpoints}
    assert {"deep/1", "deep/25", "sample/25", "private_single/25", "cp/3", "pooled/4"} <= names


def test_plan_dry_run_counts(forced):
    plan = generation_plan(forced)
    assert plan["blocks"]["main"] == 96 * 126
    assert plan["blocks"]["extension_cp_pooled"] == 96 * 13
    assert plan["blocks"]["high"] == 0 and plan["blocks"]["replication"] == 0
    assert plan["per_item"]["main_extension"] == 139
    assert [s["shard_id"] for s in plan["shards"]] == ["parasail", "mancer", "dekallm"]
    assert all(s["credit_ceiling_usd"] == "31.66" for s in plan["shards"])


def test_waves_cover_all_main_items_with_extension(forced):
    from agent_allocation.contracts import PanelDraw
    from agent_allocation.manifest import build_waves
    draws = {b["id"]: PanelDraw(b["id"], (), tuple(f"{b['id']}/{i}" for i in range(24)), (), 0, "t")
             for b in forced["benchmarks"]}
    waves = build_waves(forced, draws)
    assert len(waves) == 6 == forced["scheduler"]["max_active_waves"]
    assert all(w.panels == ("main", "extension") for w in waves)


def test_forecast_without_high(forced):
    from agent_allocation.api.fake import fake_endpoint_lock
    from agent_allocation.contracts import BENCHMARKS
    from agent_allocation.forecast import BenchmarkModel, core_forecast, per_item_costs
    models = {b: BenchmarkModel(500.0, 1000.0, 300.0) for b in BENCHMARKS}
    costs = per_item_costs(models, grids_from_config(forced), lock=fake_endpoint_lock(forced["primary_model"]),
                           headroom_by_benchmark={b: 4096 for b in BENCHMARKS}, main_a=1024, high_a=None)
    assert all(c["high"] == 0 and c["main"] > 0 for c in costs.values())
    per = {k: v // 4 for k, v in forced["panels"].items()}
    assert core_forecast(costs, per) > 0


def _estimate_module():
    path = PROJECT / "tools" / "estimate_budget.py"
    spec = importlib.util.spec_from_file_location("estimate_budget_forced", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_estimate_budget_forced_preset():
    est = _estimate_module().estimate_forced("forced27b", avg_input_tokens=1000, answer_tokens=400, natural_tokens=0)
    assert est["forced_nodes_per_item"] == 139 and est["nodes_per_item"] == 140
    assert est["reasoning_a_units_per_item"] == 189
    budgets = 189 * 1024
    assert est["input_tokens_per_item"] == 2 * 1000 * 139 + budgets + 1000
    assert est["output_tokens_per_item"] == budgets + 400 * 139 + 400
    assert [r["shard_id"] for r in est["shards"]] == ["parasail", "mancer", "dekallm"]
    assert all(r["credit_ceiling_usd"] == "31.66" for r in est["shards"])
    saturated = _estimate_module().estimate_forced("forced27b")
    assert saturated["assumptions"]["answer_tokens_per_call"] == 4096
