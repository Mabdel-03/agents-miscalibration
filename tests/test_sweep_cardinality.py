"""Sweep generation: validity collapsing + SAS-baseline enforcement + dedup."""

from collections import Counter

from agents_scaling.experiment.sweep import load_sweep
from agents_scaling.experiment.sweep import generate_cells


def _spec(**over):
    spec = {
        "axes": {
            "model_size": ["1.7B", "8B"],
            "topology": ["single_agent", "decentralized"],
            "context_share_level": ["artifact_only", "plus_cot"],
            "prompt_complexity_level": [1],
            "benchmark": ["truthfulqa"],
            "seed": [0],
        },
        "fixed": {"n_agents": 3, "rounds": 2, "n_questions": 50},
    }
    spec["axes"].update(over)
    return spec


def test_single_agent_collapses_context_levels():
    cells = generate_cells(_spec())
    # single_agent cells must not duplicate across the two context levels.
    sa = [c for c in cells if c.topology.value == "single_agent"]
    sizes = {(c.model_size, c.context_share_level.value) for c in sa}
    # each model_size has exactly one single-agent cell (canonical context).
    assert len(sa) == 2  # 1.7B, 8B
    assert all(ctx == "artifact_only" for _, ctx in sizes)
    assert all(c.n_agents == 1 for c in sa)


def test_decentralized_keeps_both_context_levels():
    cells = generate_cells(_spec())
    dec = [c for c in cells if c.topology.value == "decentralized"]
    ctxs = {c.context_share_level.value for c in dec}
    assert ctxs == {"artifact_only", "plus_cot"}


def test_sas_baseline_present_for_every_model_benchmark():
    cells = generate_cells(_spec())
    pairs_needed = {(c.model_size, c.benchmark) for c in cells}
    sas_pairs = {(c.model_size, c.benchmark) for c in cells if c.topology.value == "single_agent"}
    assert pairs_needed.issubset(
        sas_pairs | {(c.model_size, c.benchmark) for c in cells}
    )
    # explicitly: each (size, benchmark) has a SAS cell
    for size in ["1.7B", "8B"]:
        assert (size, "truthfulqa") in sas_pairs


def test_no_duplicate_cell_ids():
    cells = generate_cells(_spec())
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))


def test_independent_collapses_context():
    cells = generate_cells(_spec(topology=["single_agent", "independent"]))
    ind = [c for c in cells if c.topology.value == "independent"]
    # independent never shares context -> one cell per (size), canonical context.
    assert all(c.context_share_level.value == "artifact_only" for c in ind)
    assert len(ind) == 2


# --- Axis 4: reasoning ---

def test_reasoning_axis_not_collapsed():
    """Reasoning is per-agent; every rung survives for all topologies (unlike context)."""
    cells = generate_cells(_spec(reasoning_level=["off", "b2048", "unlimited"]))
    # decentralized cells exist at each reasoning level (x 2 context levels).
    dec = [c for c in cells if c.topology.value == "decentralized"]
    rlevels = {c.reasoning_level.value for c in dec}
    assert rlevels == {"off", "b2048", "unlimited"}
    # single-agent (context collapsed) still keeps all 3 reasoning levels.
    sa = [c for c in cells if c.topology.value == "single_agent"]
    assert {c.reasoning_level.value for c in sa} == {"off", "b2048", "unlimited"}


def test_sas_baseline_per_reasoning_level():
    """Every (size, benchmark, seed, reasoning_level) must have its SAS baseline."""
    cells = generate_cells(_spec(reasoning_level=["off", "b2048", "unlimited"]))
    needed = {(c.model_size, c.benchmark, c.seed, c.reasoning_level.value) for c in cells}
    sas = {
        (c.model_size, c.benchmark, c.seed, c.reasoning_level.value)
        for c in cells
        if c.topology.value == "single_agent"
    }
    assert needed == sas  # every reasoning-conditioned cell has a matching SAS baseline


def test_reasoning_axis_absent_defaults_off():
    """3-axis configs (no reasoning_level key) still work, defaulting to off."""
    cells = generate_cells(_spec())  # _spec has no reasoning_level
    assert all(c.reasoning_level.value == "off" for c in cells)


def test_no_duplicate_cell_ids_with_reasoning():
    cells = generate_cells(_spec(reasoning_level=["off", "b512", "b2048", "b8192", "unlimited"]))
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))


# --- Agent-count follow-up axis ---

def test_n_agents_axis_expands_mas_only_without_sas_baselines():
    spec = _spec(
        topology=["independent", "decentralized", "centralized"],
        n_agents=[2, 4],
    )
    spec["include_sas_baselines"] = False

    cells = generate_cells(spec)
    assert {c.topology.value for c in cells} == {"independent", "decentralized", "centralized"}
    assert {c.n_agents for c in cells} == {2, 4}
    assert len(cells) == 20  # per count: 2 independent + 4 decentralized + 4 centralized
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))


def test_three_agent_ids_remain_backward_compatible():
    cell = generate_cells(_spec(topology=["decentralized"], reasoning_level=["off"]))[0]
    assert cell.n_agents == 3
    assert "_n3_" not in cell.cell_id


def test_non_default_mas_ids_include_agent_count():
    spec = _spec(topology=["decentralized"], n_agents=[2, 4], reasoning_level=["off"])
    spec["include_sas_baselines"] = False
    cells = generate_cells(spec)
    ids = {c.n_agents: c.cell_id for c in cells}
    assert "_n2_" in ids[2]
    assert "_n4_" in ids[4]


def test_single_agent_canonicalizes_to_one_agent_with_n_agents_axis():
    cells = generate_cells(_spec(topology=["single_agent"], n_agents=[2, 6]))
    assert {c.topology.value for c in cells} == {"single_agent"}
    assert {c.n_agents for c in cells} == {1}
    assert all("_n2_" not in c.cell_id and "_n6_" not in c.cell_id for c in cells)


def test_full_agent_count_followup_config_cardinality():
    cells = load_sweep("configs/full_sweep_agent_counts.yaml")
    assert len(cells) == 14400
    assert Counter(c.n_agents for c in cells) == {2: 3600, 4: 3600, 5: 3600, 6: 3600}
    assert Counter(c.topology.value for c in cells) == {
        "independent": 2880,
        "decentralized": 5760,
        "centralized": 5760,
    }
    assert "single_agent" not in {c.topology.value for c in cells}
    assert len({c.cell_id for c in cells}) == len(cells)


def test_seven_agent_tranche_and_full_union_cardinality():
    base = load_sweep("configs/full_sweep.yaml")
    followup = load_sweep("configs/full_sweep_agent_counts.yaml")
    seven = load_sweep("configs/full_sweep_agent_count_7.yaml")

    assert len(seven) == 3600
    assert Counter(c.n_agents for c in seven) == {7: 3600}
    assert "single_agent" not in {c.topology.value for c in seven}

    all_ids = [c.cell_id for c in [*base, *followup, *seven]]
    assert len(all_ids) == 22680
    assert len(set(all_ids)) == 22680


def test_schema5_smoke_suite_cardinalities_and_routes():
    long_32b = load_sweep("configs/long_context_protocol_smoke.yaml")
    selective_long = load_sweep("configs/selective_long_profiles_smoke.yaml")
    standard = load_sweep("configs/standard_profile_canaries.yaml")

    assert len(long_32b) == 15
    assert len(selective_long) == 20
    assert len(standard) == 6
    assert {cell.model_size for cell in standard} == {
        "0.6B",
        "1.7B",
        "4B",
        "8B",
        "14B",
        "32B",
    }
    assert all(cell.topology.value == "single_agent" for cell in standard)
    assert all(cell.n_agents == 1 for cell in standard)
    assert all(cell.reasoning_level.value == "off" for cell in standard)
