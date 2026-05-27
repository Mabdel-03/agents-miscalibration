"""Sweep generation: validity collapsing + SAS-baseline enforcement + dedup."""

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
