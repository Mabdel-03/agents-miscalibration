"""``allocation plan --dry-run`` (lead-owned): exact generation counts per partition, spending nothing.

Uses agent 03's grid builder on synthetic tasks so the printed counts are the counts the dispatcher would register.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from agent_allocation.config import grids_from_config, load_config, validate_config
from agent_allocation.contracts import BENCHMARKS, OutputKind, PublicTask

KIND_BY_BENCHMARK = {"math500": OutputKind.MATH, "mmlu_pro": OutputKind.OPTION, "bbh_logic4": OutputKind.OPTION,
                     "mbpp_plus": OutputKind.PYTHON}


def count_nodes(grid, benchmark: str, base_budget: int, cfg: Mapping[str, Any]) -> int:
    from agent_allocation.protocols.grid import build_item_graph
    task = PublicTask(item_id=f"{benchmark}/dry-run", benchmark=benchmark, source_revision="dry-run", prompt="dry-run",
                      output_kind=KIND_BY_BENCHMARK[benchmark])
    graph = build_item_graph(grid, run_id=cfg["study_id"], partition_id="dry-run", task=task, model="dry-run",
                             base_budget=base_budget, headroom=1024, endpoint_lock_hash="dry-run",
                             decoding_config_hash="dry-run", prompt_version="dry-run")
    return len(graph.nodes)


def generation_plan(cfg: Mapping[str, Any]) -> dict[str, Any]:
    grids = grids_from_config(cfg)
    panels = cfg["panels"]
    per_benchmark = {name: panels[name] // len(BENCHMARKS) for name in panels}
    main_a, high_a = cfg["reasoning"]["main_a"], cfg["reasoning"]["high_a"]
    blocks: dict[str, int] = {"main": 0, "high": 0, "extension_cp_pooled": 0, "replication": 0}
    for benchmark in BENCHMARKS:
        main_nodes = count_nodes(grids["main"], benchmark, main_a, cfg)
        ext_nodes = count_nodes(grids["main_extension"], benchmark, main_a, cfg)
        blocks["main"] += per_benchmark["main_items"] * main_nodes
        blocks["extension_cp_pooled"] += per_benchmark["extension_items"] * (ext_nodes - main_nodes)
        blocks["high"] += per_benchmark["high_items"] * count_nodes(grids["high"], benchmark, high_a, cfg)
        blocks["replication"] += per_benchmark["replication_items"] * count_nodes(grids["replication"], benchmark,
                                                                                   main_a, cfg)
    pilot_r1 = count_nodes(grids["pilot_r1"], "math500", main_a, cfg)
    pilot_r2 = count_nodes(grids["pilot_r2"], "math500", main_a, cfg)
    pilot = 16 * pilot_r1 + 16 * pilot_r2 + 4 + 16
    return {"blocks": blocks, "total_generations": sum(blocks.values()),
            "per_item": {"main": count_nodes(grids["main"], "math500", main_a, cfg),
                         "main_extension": count_nodes(grids["main_extension"], "math500", main_a, cfg),
                         "high": count_nodes(grids["high"], "math500", high_a, cfg),
                         "replication": count_nodes(grids["replication"], "math500", main_a, cfg),
                         "pilot_r1": pilot_r1, "pilot_r2": pilot_r2},
            "pilot_generations": pilot}


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
