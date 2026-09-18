"""WP-3 mocked end to end: a forced27b-shaped study split over provider shards, each shard dispatching its own
disjoint items straight through main+extension (deep/25 included), then merge-shards; and the calibration command.

Uses the fake OpenRouter transport (``api.fake``), so after the WP-1 merge the same test exercises whichever
request path the forced client takes against the fake. No network, no key.
"""
from __future__ import annotations

import asyncio
import copy
import json
from decimal import Decimal

import pandas as pd
import pytest

from agent_allocation import live
from agent_allocation.api.fake import fake_endpoint_lock, fake_openrouter_transport
from agent_allocation.calibrate import (
    NATURAL_PARTITION, NaturalRoutingClient, calibration_summary, collect, evaluate,
)
from agent_allocation.config import validate_config
from agent_allocation.contracts import InventoryItem, dump_record
from agent_allocation.paths import RunPaths
from agent_allocation.shards import items_of, materialize_shard, root_assignment, shard_config
from tests.forced_cfg import forced_test_config, request_ready
from tests.integration import adversarial
from tests.integration.dispatch_harness import fast_config

pytestmark = [pytest.mark.timeout(600)]
TEXT_ITEMS = ("math500/adv/1", "math500/adv/2", "mmlu_pro/adv/1", "mmlu_pro/adv/2", "bbh_logic4/adv/1",
              "bbh_logic4/adv/2")


def _study_config() -> dict:
    cfg = request_ready(fast_config(forced_test_config()))
    cfg["shards"] = [{"id": "alpha", "endpoint_tag": "prov-a/fp8", "share": 1},
                     {"id": "beta", "endpoint_tag": "prov-b/fp8", "share": 1}]
    assert validate_config(cfg) == [], validate_config(cfg)
    return cfg


def _inventory():
    items = {t.item_id: (t, p) for t, p, _ in adversarial.load_items()}
    return [InventoryItem(public=items[i][0], private=items[i][1]) for i in TEXT_ITEMS]


def _write_lock(paths: RunPaths, model: str) -> None:
    lock = fake_endpoint_lock(model)
    paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
    paths.endpoint_lock.write_text(json.dumps({"locks": {model: dump_record(lock)}}, sort_keys=True))


def _run_shard(cfg, paths: RunPaths, inventory) -> dict:
    tasks = live.load_tasks(paths)
    locks = live.read_endpoint_locks(paths)
    by_wave = live.graphs_by_wave(cfg, paths, tasks, locks, {})
    assert by_wave["high"] == [] and by_wave["replication"] == []
    graphs = [g for wave in by_wave["main"] for g in wave]
    live.write_jobs(paths, "main", graphs)
    ceiling = Decimal(str(cfg["budget"]["credit_ceiling_usd"]))

    async def main():
        runtime = live.Runtime(cfg, paths, lock=locks[cfg["primary_model"]], tasks=tasks, ceiling=ceiling,
                               transport=fake_openrouter_transport())
        try:
            return await live._run_main_phases(
                cfg, paths, ceiling, runtime=runtime, by_wave=by_wave, tasks=tasks, locks=locks, amendments={},
                api_key=None, soft=Decimal(str(cfg["budget"]["soft_admission_ceiling_usd"])),
                reserve=Decimal(str(cfg["budget"]["contingency_usd"])), ceiling=ceiling)
        finally:
            await runtime.aclose()
    rc = asyncio.run(main())
    privates = {i.public.item_id: i.private for i in inventory}
    live._grade_offline(paths, graphs, tasks, privates, None, None)
    live.select_and_seal(cfg, paths, "main", graphs, tasks)
    decisions = [json.loads(l) for l in paths.run_decisions.read_text().splitlines()]
    return {"rc": rc, "graphs": graphs, "decisions": decisions}


def test_sharded_forced_study_runs_straight_through_and_merges(tmp_path):
    cfg = _study_config()
    inventory = _inventory()
    root = RunPaths.for_run(tmp_path / "study", local_state_dir=tmp_path / "local_root")
    live._write_offline_manifests(root, inventory, extension_ids=set(TEXT_ITEMS))
    assignment = root_assignment(cfg, root)
    seen: set[str] = set()
    for shard in ("alpha", "beta"):
        info = materialize_shard(cfg, root, shard, assignment, local_state_dir=tmp_path / f"local_{shard}")
        assert info["problems"] == []
        derived = shard_config(cfg, shard)
        paths = RunPaths.for_run(root.run_dir / "shards" / shard, local_state_dir=tmp_path / f"local_{shard}")
        _write_lock(paths, cfg["primary_model"])
        out = _run_shard(derived, paths, inventory)
        items = {g.item_id for g in out["graphs"]}
        assert items == items_of(assignment["assignment"], shard) and not items & seen
        seen |= items
        assert all(len(g.nodes) == 139 for g in out["graphs"])
        assert out["rc"] == 0
        assert [d["phase"] for d in out["decisions"]] == ["main_all_waves"]
        assert all(paths.parsed(n.node_id).exists() for g in out["graphs"] for n in g.nodes)
    assert seen == set(TEXT_ITEMS)

    from agent_allocation.analysis.merge import merge_shards
    base = copy.deepcopy(cfg)
    analysis, manifest = merge_shards(root.run_dir, config=base, replicates=40, figures=False)
    assert manifest["items"] == len(TEXT_ITEMS)
    outcomes = pd.read_parquet(root.run_dir / "merged" / "tables" / "policy_outcomes.parquet")
    assert set(outcomes["item_id"]) == set(TEXT_ITEMS)
    assert set(outcomes["shard_id"]) == {"alpha", "beta"}
    deep = outcomes[(outcomes["name"] == "deep/25") & (outcomes["view"] == "current")]
    assert len(deep) == len(TEXT_ITEMS) and (deep["reasoning_budget"] == 25600).all()
    for table in ("effective_depth", "effective_depth_summary", "forced_budget_check", "cost_summary"):
        assert (root.run_dir / "merged" / "tables" / f"{table}.parquet").is_file(), table
    assert (root.run_dir / "merged" / "reports" / "RUN_REPORT.md").read_text().count("forced to exactly") == 1

    # The iso-compute ladder reads the merged tables: C = B*R*A holds, deep/25 sits on rung 25, effective depth joins.
    import sys
    from tests.forced_cfg import PROJECT
    sys.path.insert(0, str(PROJECT / "docs" / "isocompute"))
    try:
        import isocore
        merged_dir = str(root.run_dir / "merged")
        assert isocore.self_check(merged_dir, "main") == []
        eff = isocore.effective_depth(merged_dir, "main").set_index("name")
        assert eff.loc["deep/25", "rung"] == 25 and eff.loc["deep/25", "C"] == 25600
        assert eff["mean_effective_tokens"].notna().all()
        assert (eff["mean_effective_tokens"] <= eff["C"] + 1e-9).all()
    finally:
        sys.path.remove(str(PROJECT / "docs" / "isocompute"))


def test_calibration_collects_forced_and_natural_and_gates(tmp_path):
    cfg = request_ready(fast_config(forced_test_config()))
    cfg.pop("shards")
    cfg.pop("shards_note", None)
    cfg["provider"]["endpoint_tag"] = None
    inventory = _inventory()
    paths = RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")
    live._write_offline_manifests(paths, inventory, extension_ids=set())
    panels = json.loads(paths.panels.read_text())
    for draw in panels.values():                                   # the offline writer puts items in main; use dev
        draw["dev_item_ids"], draw["main_item_ids"] = draw["main_item_ids"], []
    paths.panels.write_text(json.dumps(panels, sort_keys=True))
    _write_lock(paths, cfg["primary_model"])
    summary = asyncio.run(collect(cfg, paths, max_usd=Decimal("95"), roots=2, transport=fake_openrouter_transport(),
                                  allow_paid=False))
    assert summary["stop_reason"] == "completed", summary
    result = evaluate(cfg, [(None, paths)], paths.reports_dir / "calibration.json")
    assert result["rows"] == len(TEXT_ITEMS) * 3
    for bench, row in result["benchmarks"].items():
        assert row["forced_generations"] == 2 * row["items"] and row["natural_generations"] == row["items"]
        assert row["gate"]["verdict"] in ("pass", "too_hard", "too_easy")
        assert set(row["natural_share_at_or_above_deep_budget"]) == {"deep/1", "deep/2", "deep/4", "deep/8",
                                                                     "deep/16", "deep/25"}
    assert json.loads((paths.reports_dir / "calibration.json").read_text())["decision"] == result["decision"]
    jobs = [json.loads(l)["record"] for l in paths.jobs(NATURAL_PARTITION).read_text().splitlines()]
    assert {j["reasoning_budget"] for j in jobs} == {25600}


def test_calibration_gate_verdicts():
    cfg = forced_test_config()
    rows = []
    for bench, correct in (("aime", [0, 0, 0, 1]), ("hmmt", [1, 0, 1, 0]), ("gpqa_diamond", [1, 1, 1, 1])):
        for k, c in enumerate(correct):
            rows.append({"mode": "forced", "benchmark": bench, "item_id": f"{bench}/{k // 2}", "correct": bool(c),
                         "reasoning_tokens": 1024, "reasoning_budget": 1024})
        rows.append({"mode": "natural", "benchmark": bench, "item_id": f"{bench}/0", "correct": True,
                     "reasoning_tokens": 5000, "reasoning_budget": 25600})
    out = calibration_summary(cfg, rows)
    b = out["benchmarks"]
    assert b["aime"]["forced_pass_at_1"] == 0.25 and b["aime"]["gate"]["verdict"] == "pass"
    assert b["hmmt"]["gate"]["verdict"] == "pass"
    assert b["gpqa_diamond"]["gate"] == {"verdict": "too_easy", "min": 0.15, "max": 0.85, "fallback": None}
    assert b["aime"]["natural_share_at_or_above_deep_budget"]["deep/4"] == 1.0
    assert b["aime"]["natural_share_at_or_above_deep_budget"]["deep/8"] == 0.0
    assert out["decision"] == "swap_benchmarks"
    hard = [dict(r, correct=False) for r in rows if r["benchmark"] == "hmmt"]
    verdict = calibration_summary(cfg, hard)["benchmarks"]["hmmt"]["gate"]
    assert verdict["verdict"] == "too_hard" and verdict["fallback"] == "omni_math_mid"


def test_natural_routing_client_forwards_forced_false_only_to_natural_nodes():
    calls = []

    class Forced:
        async def generate(self, spec, messages, *, attempt_id, lock, sampling, forced=True):
            calls.append((spec.partition_id, forced))

    class Plain:
        async def generate(self, spec, messages, *, attempt_id, lock, sampling):
            calls.append((spec.partition_id, "plain"))

    class Spec:
        def __init__(self, partition_id):
            self.partition_id = partition_id

    async def go(client):
        await client.generate(Spec(NATURAL_PARTITION), [], attempt_id="a", lock=None, sampling={})
        await client.generate(Spec("calibration"), [], attempt_id="b", lock=None, sampling={})
    asyncio.run(go(NaturalRoutingClient(Forced())))
    asyncio.run(go(NaturalRoutingClient(Plain())))
    assert calls == [(NATURAL_PARTITION, False), ("calibration", True), (NATURAL_PARTITION, "plain"),
                     ("calibration", "plain")]
