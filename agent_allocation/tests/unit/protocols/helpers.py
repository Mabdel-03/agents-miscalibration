"""Shared builders for agent 03's protocol tests (tasks, graphs, synthetic candidates)."""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from agent_allocation.config import grids_from_config, load_config
from agent_allocation.contracts import (
    STUDY_ID, Candidate, GenerationKind, GenerationSpec, ItemGraph, OutputKind, ParseStatus, PublicTask,
)
from agent_allocation.prompts.render import prompt_version
from agent_allocation.protocols.grid import build_item_graph

ROOT = Path(__file__).resolve().parents[3]
SEED = 20260917
MATH_TASK = PublicTask(item_id="aime/2025/1", benchmark="aime",
                       source_revision="c94da77eb22bbd6439e62a323bec18493a421302",
                       prompt="Compute $\\frac{1}{2}+\\frac{1}{2}$ and give {braces} as-is.", output_kind=OutputKind.MATH)
CODE_TASK = PublicTask(item_id="mbpp_plus/Mbpp/2", benchmark="mbpp_plus", source_revision="evalplus-26d6d00b",
                       prompt="Write a function similar_elements(a, b) returning the shared elements as a set.\n"
                              "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))",
                       output_kind=OutputKind.PYTHON, public_examples=(
                           "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))",))


@lru_cache(maxsize=1)
def grids():
    return grids_from_config(load_config(ROOT / "configs" / "balanced150.json"))


def graph(grid_name: str = "main", task: PublicTask = MATH_TASK, *, partition_id: str | None = None,
          base_budget: int | None = None, headroom: int = 1024, version: str | None = None) -> ItemGraph:
    partition = partition_id or {"main_extension": "main", "pilot_r1": "pilot", "pilot_r2": "pilot"}.get(grid_name, grid_name)
    budget = base_budget or (4096 if grid_name == "high" else 1024)
    return build_item_graph(grids()[grid_name], run_id=STUDY_ID, partition_id=partition, task=task,
                            model="google/gemini-2.5-flash-lite", base_budget=budget, headroom=headroom,
                            endpoint_lock_hash="lock-hash", decoding_config_hash="decoding-hash",
                            prompt_version=version or prompt_version())


def reference_dag():
    path = ROOT / "tools" / "reference_dag.py"
    spec = importlib.util.spec_from_file_location("reference_dag_for_protocols", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_name(node: GenerationSpec, base: int = 1024) -> str:
    """The reference_dag.py name of a production node (coordinate table -> readable name)."""
    k = node.kind
    if k is GenerationKind.ROOT:
        return f"root/{node.root_slot}"
    if k is GenerationKind.PRIVATE:
        return f"private/{node.root_slot}/{node.update_round}"
    if k is GenerationKind.PEER:
        return f"peer/{node.team_size}/{node.update_round}/{node.root_slot}"
    if k is GenerationKind.DEEP:
        return f"deep/{node.reasoning_budget // base}"
    if k is GenerationKind.CP:
        return f"cp/{node.update_round}/{node.root_slot}"
    return "pooled/4"


def candidate(node: GenerationSpec, content: str | None = None, *, status: ParseStatus = ParseStatus.VALID,
              output_kind: OutputKind = OutputKind.MATH) -> Candidate:
    text = content if content is not None else (
        f"<rationale>\nWorked the decisive steps for draft {node.node_id[:8]}.\n</rationale>\n<final>\n1\n</final>")
    return Candidate(candidate_id=node.node_id, node_id=node.node_id, item_id=node.item_id, visible_content=text,
                     final_answer="1" if status is ParseStatus.VALID else None, rationale=None, parse_status=status,
                     finish_reason="stop", output_kind=output_kind)


def parents_for(g: ItemGraph, node: GenerationSpec, candidates: dict[str, Candidate | None]):
    index = {n.node_id: n for n in g.nodes}
    return [(index[p], candidates.get(p)) for p in node.parent_ids]
