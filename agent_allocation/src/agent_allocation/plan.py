"""``allocation plan --dry-run`` (lead-owned): exact generation counts per partition, spending nothing.

Uses agent 03's grid builder on synthetic tasks so the printed counts are the counts the dispatcher would register.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from agent_allocation.config import grids_from_config, load_config, validate_config
from agent_allocation.contracts import BENCHMARKS, OutputKind, PublicTask

KIND_BY_BENCHMARK = {"aime": OutputKind.MATH, "hmmt": OutputKind.MATH, "omni_math_hard": OutputKind.MATH,
                     "gpqa_diamond": OutputKind.OPTION}


def _kind(benchmark: str) -> OutputKind:
    """Output kind for a dry-run task; node counts do not depend on it, so an unknown id counts as MATH."""
    return KIND_BY_BENCHMARK.get(benchmark, OutputKind.MATH)


def count_nodes(grid, benchmark: str, base_budget: int, cfg: Mapping[str, Any]) -> int:
    from agent_allocation.protocols.grid import build_item_graph
    task = PublicTask(item_id=f"{benchmark}/dry-run", benchmark=benchmark, source_revision="dry-run", prompt="dry-run",
                      output_kind=_kind(benchmark))
    graph = build_item_graph(grid, run_id=cfg["study_id"], partition_id="dry-run", task=task, model="dry-run",
                             base_budget=base_budget, headroom=1024, endpoint_lock_hash="dry-run",
                             decoding_config_hash="dry-run", prompt_version="dry-run")
    return len(graph.nodes)


def generation_plan(cfg: Mapping[str, Any]) -> dict[str, Any]:
    grids = grids_from_config(cfg)
    panels = cfg["panels"]
    per_benchmark = {name: panels[name] // len(BENCHMARKS) for name in panels}
    main_a, high_a = cfg["reasoning"]["main_a"], cfg["reasoning"]["high_a"]
    probe = BENCHMARKS[0]
    blocks: dict[str, int] = {"main": 0, "high": 0, "extension_cp_pooled": 0, "replication": 0}
    for benchmark in BENCHMARKS:
        main_nodes = count_nodes(grids["main"], benchmark, main_a, cfg)
        ext_nodes = count_nodes(grids["main_extension"], benchmark, main_a, cfg)
        blocks["main"] += per_benchmark["main_items"] * main_nodes
        blocks["extension_cp_pooled"] += per_benchmark["extension_items"] * (ext_nodes - main_nodes)
        if "high" in grids and per_benchmark.get("high_items"):
            blocks["high"] += per_benchmark["high_items"] * count_nodes(grids["high"], benchmark, high_a, cfg)
        if "replication" in grids and per_benchmark.get("replication_items"):
            blocks["replication"] += per_benchmark["replication_items"] * count_nodes(grids["replication"], benchmark,
                                                                                       main_a, cfg)
    pilot_r1 = count_nodes(grids["pilot_r1"], probe, main_a, cfg)
    pilot_r2 = count_nodes(grids["pilot_r2"], probe, main_a, cfg)
    pilot = 16 * pilot_r1 + 16 * pilot_r2 + 4 + 16
    per_item = {"main": count_nodes(grids["main"], probe, main_a, cfg),
                "main_extension": count_nodes(grids["main_extension"], probe, main_a, cfg),
                "pilot_r1": pilot_r1, "pilot_r2": pilot_r2}
    if "high" in grids:
        per_item["high"] = count_nodes(grids["high"], probe, high_a, cfg)
    if "replication" in grids:
        per_item["replication"] = count_nodes(grids["replication"], probe, main_a, cfg)
    out = {"blocks": blocks, "total_generations": sum(blocks.values()), "per_item": per_item,
           "pilot_generations": pilot}
    if cfg.get("shards"):
        from agent_allocation.shards import shard_budget_table
        out["shards"] = shard_budget_table(cfg)
    return out


def cmd_plan(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation plan")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    problems = validate_config(cfg)
    if problems:
        print(json.dumps({"problems": problems}))
        return 1
    print(json.dumps(generation_plan(cfg), indent=1, sort_keys=True))
    return 0
