"""Golden node identities (lead-owned). Pins the hashing scheme and the coordinate table in GridSpec's docstring.

Agent 03's builder must reproduce these node_ids for the fixture item: once ``protocols.grid`` exists the second
test stops skipping and becomes the cross-module identity check.
"""
from __future__ import annotations

import pytest

from agent_allocation.contracts import STUDY_ID, GenerationKind, GenerationSpec, OutputKind, PublicTask, public_task_hash

TASK = PublicTask(item_id="aime/2025/1", benchmark="aime", source_revision="c94da77eb22bbd6439e62a323bec18493a421302",
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
    "root0": "07d5ea3b1c0893ff58f24089fdd3bda3b9a6e6911b862e5c35be4cd7e1c552e9",
    "root4": "130c76f0c1d951ec635be441db3f3d55b07780e1d39e7a11c2598c614196fa5e",
    "private_0_1": "1ab95dcc7ec5de861e63f1559cdaec65827f3322a99c6340397d3144a70205b3",
    "peer_3_1_2": "126fb27b313d533738af27a2465c967f75942ab9f3687c9bd23b53d0cf1c0b51",
    "deep_16": "a32d359e5733b5cc99abdd43afdd5cfbe48d406d1cbad42bfbaddf5da7239f4b",
}
