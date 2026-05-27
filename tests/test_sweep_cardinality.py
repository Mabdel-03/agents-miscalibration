"""Sweep generation: validity collapsing + SAS-baseline enforcement + dedup."""

from agents_scaling.experiment.sweep import generate_cells


def _spec(**over):
    spec = {
        "axes": {
            "model_size": ["1.5B", "7B"],
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
    assert len(sa) == 2  # 1.5B, 7B
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
    for size in ["1.5B", "7B"]:
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
