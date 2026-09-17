"""Protocol DAG builder: counts, reference-oracle structure, reuse, ancestry, views, budgets (agent 03)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from agent_allocation.config import load_config
from agent_allocation.contracts import CheckpointView, GenerationKind, GridSpec
from agent_allocation.protocols.grid import (
    CODE_SELECTOR, TEXT_SELECTOR, BudgetUnsupported, GridError, build_item_graph, check_budgets,
    check_config_constants, checkpoint_by_name, expected_node_count, node_index, validate_grid,
)

from .helpers import CODE_TASK, MATH_TASK, ROOT, graph, grids, reference_dag, reference_name

COUNTS = {"main": 125, "main_extension": 138, "high": 123, "replication": 57, "pilot_r1": 12, "pilot_r2": 15}
REFERENCE_PROFILES = {"main": dict(), "high": dict(a=4096, high=True), "replication": dict(reduced=True),
                      "main_extension": dict(extension=True)}


@pytest.mark.parametrize("name", sorted(COUNTS))
def test_exact_node_counts_and_topological_order(name):
    g = graph(name)
    assert len(g.nodes) == COUNTS[name] == expected_node_count(grids()[name])
    assert len({n.node_id for n in g.nodes}) == len(g.nodes)
    seen: set[str] = set()
    for node in g.nodes:
        assert set(node.parent_ids) <= seen, "parent emitted after child"
        seen.add(node.node_id)
        assert node.draw_id == 0 and node.repetition_id == 0
        assert node.item_id == MATH_TASK.item_id


@pytest.mark.parametrize("name", sorted(REFERENCE_PROFILES))
def test_structure_equals_reference_dag(name):
    """Every node, parent edge, checkpoint pool and ancestor set equals tools/reference_dag.py (plus deep/1)."""
    base = 4096 if name == "high" else 1024
    g = graph(name, base_budget=base)
    ref = reference_dag().build_plan(**REFERENCE_PROFILES[name])
    names = {n.node_id: reference_name(n, base) for n in g.nodes}
    assert sorted(names.values()) == sorted(ref.nodes)
    for node in g.nodes:
        r = ref.nodes[names[node.node_id]]
        assert tuple(names[p] for p in node.parent_ids) == r.parents
        assert node.reasoning_budget == r.reasoning_budget
    for view in CheckpointView:
        mine = {cp.name: cp for cp in g.checkpoints if cp.view is view}
        assert set(mine) == set(ref.checkpoints) | {"deep/1"}
        for cp_name, pool in ref.checkpoints.items():
            cp = mine[cp_name]
            assert {names[i] for i in cp.required_node_ids} == ref.ancestors(cp_name)
            if view is CheckpointView.CURRENT:
                assert tuple(names[i] for i in cp.candidate_ids) == pool
            else:
                assert cp.candidate_ids == cp.required_node_ids


def test_reference_ancestry_counts_and_deep_one():
    g = graph("main_extension")
    for name, count in [("sample/25", 25), ("private_single/25", 25), ("peer/5/4", 25), ("peer/4/4", 20),
                        ("private_team/4/4", 20), ("cp/3", 20), ("pooled/4", 5), ("cp/0", 8), ("deep/16", 1)]:
        for view in CheckpointView:
            assert len(checkpoint_by_name(g, name, view).required_node_ids) == count, (name, view)
    deep1 = checkpoint_by_name(g, "deep/1")
    root0 = g.nodes[0]
    assert root0.kind is GenerationKind.ROOT and root0.root_slot == 0
    assert deep1.candidate_ids == deep1.required_node_ids == (root0.node_id,)
    assert deep1.family == "deep" and deep1.reasoning_budget == 1024


def test_root_round_zero_equality_across_families():
    g = graph("main")
    for n in (2, 3, 4, 5):
        sample = checkpoint_by_name(g, f"sample/{n}").candidate_ids
        assert checkpoint_by_name(g, f"private_team/{n}/0").candidate_ids == sample
        assert checkpoint_by_name(g, f"peer/{n}/0").candidate_ids == sample
    assert checkpoint_by_name(g, "private_single/1").candidate_ids == checkpoint_by_name(g, "sample/1").candidate_ids


def test_no_future_or_same_round_peer_parents():
    g = graph("main")
    index = node_index(g)
    for n in (2, 3, 4, 5):
        for r in range(1, 5):
            previous = checkpoint_by_name(g, f"peer/{n}/{r - 1}").candidate_ids
            for j, node_id in enumerate(checkpoint_by_name(g, f"peer/{n}/{r}").candidate_ids):
                node = index[node_id]
                assert (node.kind, node.team_size, node.update_round, node.root_slot) == (GenerationKind.PEER, n, r, j)
                assert node.parent_ids == previous
                assert all(index[p].update_round == r - 1 for p in node.parent_ids)


def test_private_team_reuses_private_single_chain_and_team_slots_are_team_independent():
    g = graph("main")
    index = node_index(g)
    for r in range(5):
        chain = checkpoint_by_name(g, f"private_single/{r + 1}").candidate_ids[0]
        pools = [checkpoint_by_name(g, f"private_team/{n}/{r}").candidate_ids for n in (2, 3, 4, 5)]
        assert all(pool[0] == chain for pool in pools)
        for small, large in zip(pools, pools[1:]):
            assert large[:len(small)] == small       # one trajectory per slot, shared by every n
    privates = [n for n in g.nodes if n.kind is GenerationKind.PRIVATE]
    assert len(privates) == 24 + 16
    assert all(n.team_size == 0 for n in privates)
    assert {(n.root_slot, n.update_round) for n in privates if n.root_slot} == {(j, r) for j in range(1, 5)
                                                                               for r in range(1, 5)}
    assert sum(1 for n in privates if index[n.parent_ids[0]].root_slot != n.root_slot) == 0


def test_peer_team_sizes_never_alias():
    g = graph("main")
    by_round: dict[tuple[int, int], set[str]] = {}
    for n in (2, 3, 4, 5):
        for r in range(1, 5):
            by_round[n, r] = set(checkpoint_by_name(g, f"peer/{n}/{r}").candidate_ids)
    keys = list(by_round)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not by_round[a] & by_round[b], (a, b)
    assert sum(len(v) for v in by_round.values()) == 56


def test_extension_cp_and_pooled_shape():
    g = graph("main_extension")
    index = node_index(g)
    roots = checkpoint_by_name(g, "sample/4").candidate_ids
    assert checkpoint_by_name(g, "cp/0").candidate_ids == checkpoint_by_name(g, "peer/4/1").candidate_ids
    for h in (1, 2, 3):
        for j, node_id in enumerate(checkpoint_by_name(g, f"cp/{h}").candidate_ids):
            node = index[node_id]
            assert (node.kind, node.root_slot, node.team_size, node.update_round) == (GenerationKind.CP, j, 4, h)
            assert node.parent_ids == (checkpoint_by_name(g, f"cp/{h - 1}").candidate_ids[j],)
    pooled = index[checkpoint_by_name(g, "pooled/4").candidate_ids[0]]
    assert (pooled.kind, pooled.root_slot, pooled.team_size, pooled.update_round) == (GenerationKind.POOLED, 0, 4, 0)
    assert pooled.parent_ids == roots
    main_ids = {n.node_id for n in graph("main").nodes}
    extra = [n for n in g.nodes if n.node_id not in main_ids]
    assert len(extra) == 13 and {n.kind for n in extra} == {GenerationKind.CP, GenerationKind.POOLED}


def test_deep_budgets_are_a_times_m_and_never_clipped():
    main, high, rep = graph("main"), graph("high"), graph("replication")
    assert sorted(n.reasoning_budget for n in main.nodes if n.kind is GenerationKind.DEEP) == [2048, 4096, 8192, 16384]
    assert sorted(n.reasoning_budget for n in high.nodes if n.kind is GenerationKind.DEEP) == [8192, 16384]
    assert sorted(n.reasoning_budget for n in rep.nodes if n.kind is GenerationKind.DEEP) == [4096, 16384]
    for g in (main, high, rep):
        for m in g.grid.deep_multipliers:
            cp = checkpoint_by_name(g, f"deep/{m}")
            node = node_index(g)[cp.candidate_ids[0]]
            assert node.reasoning_budget == cp.reasoning_budget == m * (4096 if g is high else 1024)
            assert node.max_completion_tokens == node.reasoning_budget + node.completion_headroom
    check_budgets(high, max_completion_tokens=65535, requested_reasoning_max=24576)
    with pytest.raises(BudgetUnsupported):
        check_budgets(high, max_completion_tokens=16384, requested_reasoning_max=24576)
    with pytest.raises(BudgetUnsupported):
        check_budgets(main, max_completion_tokens=65535, requested_reasoning_max=8192)


def test_checkpoint_views_ids_and_metadata():
    g = graph("main_extension")
    names = [cp.name for cp in g.checkpoints if cp.view is CheckpointView.CURRENT]
    assert len(names) == len(set(names)) == 25 + 25 + 20 + 20 + 5 + 4 + 1
    assert len(g.checkpoints) == 2 * len(names)
    for cp in g.checkpoints:
        assert cp.policy_id == f"main/{MATH_TASK.item_id}/{cp.name}/{cp.view.value}"
        assert cp.partition_id == "main" and cp.item_id == MATH_TASK.item_id
        assert cp.family == cp.name.split("/")[0]
        assert cp.selector_name == TEXT_SELECTOR
        assert len(set(cp.required_node_ids)) == len(cp.required_node_ids)
        assert set(cp.candidate_ids) <= set(cp.required_node_ids)
        width = len(cp.required_node_ids)
        if cp.family == "pooled":
            assert (cp.team_size, cp.total_cycles, cp.peer_rounds, width) == (4, 2, 0, 5)
        else:
            assert width == cp.team_size * cp.total_cycles, cp.name
        if cp.view is CheckpointView.CURRENT and cp.family != "pooled":
            assert len(cp.candidate_ids) == cp.team_size
        if cp.family == "peer":
            assert cp.total_cycles == 1 + cp.peer_rounds
        if cp.family == "cp":
            assert cp.peer_rounds == 1 and cp.total_cycles == 2 + int(cp.name.split("/")[1])
    assert {cp.selector_name for cp in graph("main", CODE_TASK).checkpoints} == {CODE_SELECTOR}


def test_pilot_grids_checkpoints():
    r1, r2 = graph("pilot_r1"), graph("pilot_r2")
    assert {n.node_id for n in r1.nodes} <= {n.node_id for n in r2.nodes}
    names = {cp.name for cp in r2.checkpoints}
    assert names == ({f"sample/{k}" for k in (1, 2, 3)} | {f"private_single/{t}" for t in (1, 2, 3)}
                     | {f"private_team/3/{r}" for r in (0, 1, 2)} | {f"peer/3/{r}" for r in (0, 1, 2)})
    assert len(checkpoint_by_name(r2, "peer/3/2").required_node_ids) == 9
    assert len(checkpoint_by_name(r1, "private_team/3/2").required_node_ids) == 9


def test_identity_depends_on_partition_budget_and_prompt_version():
    a = {n.node_id for n in graph("main").nodes}
    assert not a & {n.node_id for n in graph("main", partition_id="high").nodes}
    assert not a & {n.node_id for n in graph("main", version="prompts-v1-other").nodes}
    assert not a & {n.node_id for n in graph("main", headroom=2048).nodes}
    assert not a & {n.node_id for n in graph("main", CODE_TASK).nodes}


def test_invalid_grids_raise():
    main = grids()["main"]
    for bad in (replace(main, team_sizes=()), replace(main, team_sizes=(1, 2)), replace(main, team_sizes=(2, 2)),
                replace(main, bank_size=4), replace(main, private_single_cycles=4),
                replace(main, deep_multipliers=(1, 2)), replace(main, deep_multipliers=(2, 2)),
                replace(main, team_sizes=(2, 3, 5), extension=True), replace(main, peer_rounds=0, extension=True)):
        with pytest.raises(GridError):
            validate_grid(bad)
    with pytest.raises(GridError):
        graph("main", base_budget=-1)


def test_config_constants_match_both_presets():
    for preset in ("balanced150", "lean100"):
        check_config_constants(load_config(ROOT / "configs" / f"{preset}.json"))
    cfg = load_config(ROOT / "configs" / "balanced150.json")
    cfg["protocols"]["cp_private_rounds"] = 2
    with pytest.raises(GridError):
        check_config_constants(cfg)


def test_grid_spec_is_the_only_shape_input():
    custom = GridSpec(name="tiny", bank_size=4, private_single_cycles=3, private_team_rounds=1, team_sizes=(2, 4),
                      peer_rounds=1, deep_multipliers=(2,), extension=True)
    g = build_item_graph(custom, run_id="r", partition_id="dev", task=MATH_TASK, model="m", base_budget=512,
                         headroom=256, endpoint_lock_hash="e", decoding_config_hash="d", prompt_version="p")
    assert len(g.nodes) == expected_node_count(custom) == 4 + 2 + 3 + 6 + 1 + 13
