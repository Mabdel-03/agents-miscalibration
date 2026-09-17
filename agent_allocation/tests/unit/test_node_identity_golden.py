"""Golden node identities (lead-owned). Pins the hashing scheme and the coordinate table in GridSpec's docstring.

Agent 03's builder must reproduce these node_ids for the fixture item: once ``protocols.grid`` exists the second
test stops skipping and becomes the cross-module identity check.
"""
from __future__ import annotations

import pytest

from agent_allocation.contracts import STUDY_ID, GenerationKind, GenerationSpec, OutputKind, PublicTask, public_task_hash

TASK = PublicTask(item_id="math500/test/algebra/1", benchmark="math500", source_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                  prompt="Compute $1+1$.", output_kind=OutputKind.MATH)
COMMON = dict(run_id=STUDY_ID, partition_id="main", item_id=TASK.item_id, benchmark=TASK.benchmark,
              model="google/gemini-2.5-flash-lite", endpoint_lock_hash="golden-endpoint-lock",
              input_manifest_hash=public_task_hash(TASK), decoding_config_hash="golden-decoding",
              prompt_version="prompts-v1-golden", completion_headroom=1024, draw_id=0)


def spec(kind, *, root_slot=0, team_size=0, update_round=0, budget=1024, parents=()):
    return GenerationSpec(kind=kind, reasoning_budget=budget, root_slot=root_slot, team_size=team_size,
                          update_round=update_round, parent_ids=tuple(parents), **COMMON)


def golden_nodes():
    roots = [spec(GenerationKind.ROOT, root_slot=j) for j in range(5)]
    priv1 = spec(GenerationKind.PRIVATE, root_slot=0, update_round=1, parents=(roots[0].node_id,))
    peer3 = [spec(GenerationKind.PEER, root_slot=j, team_size=3, update_round=1,
                  parents=[r.node_id for r in roots[:3]]) for j in range(3)]
    deep16 = spec(GenerationKind.DEEP, budget=1024 * 16)
    return {"root0": roots[0], "root4": roots[4], "private_0_1": priv1, "peer_3_1_2": peer3[2], "deep_16": deep16}


def test_golden_ids_are_pinned():
    ids = {name: node.node_id for name, node in golden_nodes().items()}
    assert len(set(ids.values())) == len(ids)
    assert ids == PINNED


def test_grid_builder_reproduces_golden_ids():
    grid_module = pytest.importorskip("agent_allocation.protocols.grid")
    from agent_allocation.config import grids_from_config, load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "balanced150.json")
    graph = grid_module.build_item_graph(grids_from_config(cfg)["main"], run_id=STUDY_ID, partition_id="main",
                                         task=TASK, model=COMMON["model"], base_budget=1024, headroom=1024,
                                         endpoint_lock_hash=COMMON["endpoint_lock_hash"],
                                         decoding_config_hash=COMMON["decoding_config_hash"],
                                         prompt_version=COMMON["prompt_version"])
    built = {n.node_id for n in graph.nodes}
    for name, node_id in PINNED.items():
        assert node_id in built, f"grid builder does not produce golden node {name}"
    assert len(graph.nodes) == 125


PINNED = {
    "root0": "b2b136242d40bfbd32f60222c40e23931b6cdf195ee71d53977297a5863e95b0",
    "root4": "0229ee82021e8be9834f6bd5cb3bdc6dd74cdab0a5b2b8d175322ae7b828f9a8",
    "private_0_1": "53ae3bbcdb91f64dad46076ba6911229179058aa98b982311c541fa140a43ec0",
    "peer_3_1_2": "e8f9df9b54bf3942bfab78d7bc701def9b6721492879d50eaff6270e57901318",
    "deep_16": "59074ae4e250d27c8005c9f6a827a1a0de7e192015b3fd564f0eafeed0e61703",
}
