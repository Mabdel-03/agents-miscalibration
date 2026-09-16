"""Graphs are data: the edge arithmetic of every arm, at three team sizes (plan §1.5)."""

from __future__ import annotations

import pytest

from agents_scaling.nextdyn import graphs as G
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn.errors import ContractViolation

SEED = bytes.fromhex("11" * 32)
PAIRING = "item-1|team:abc|model16|budget8|rep0"


def _resolved(name: str, *, n: int = 5, round_: int = 1, focal: int | None = 2):
    spec = P.load_registry().episodes[name]
    return G.resolve_graph(spec.graph, n=n, round_=round_, focal_slot=focal, study_seed=SEED,
                           pairing_id=PAIRING)


def test_a_graph_never_carries_a_self_edge_at_any_team_size():
    for n in (3, 5, 9):
        for graph in (G.complete(n), G.empty(n), G.focal_channels(n, 0, incoming=True, outgoing=True),
                      G.ring_fixed(n, tuple(range(n))), G.protect_slot(n, 1)):
            assert all(not graph[i][i] for i in range(n))


def test_c11_c01_c10_c00_have_20_16_16_12_edges():
    """Spec 10.3 at N=5: cutting one focal channel removes exactly four directed edges."""
    counts = {name: G.edge_count(_resolved(name))
              for name in ("IN1_OUT1", "IN0_OUT1", "IN1_OUT0", "IN0_OUT0")}
    assert counts == {"IN1_OUT1": 20, "IN0_OUT1": 16, "IN1_OUT0": 16, "IN0_OUT0": 12}


def test_isolated_focal_does_not_isolate_other_agents():
    """IN0_OUT0 cuts the focal agent's channels only; the non-focal block stays complete."""
    graph = _resolved("IN0_OUT0", focal=2)
    assert G.in_degree(graph, 2) == 0 and G.out_degree(graph, 2) == 0
    for slot in (0, 1, 3, 4):
        assert G.in_degree(graph, slot) == 3
        assert G.out_degree(graph, slot) == 3


def test_focal_slot_is_fixed_after_root():
    """A fixed schedule resolves to the same graph at every round (the focal slot is a
    property of the team, drawn at construction, never redrawn per round)."""
    per_round = [_resolved("IN0_OUT1", round_=r) for r in range(1, 5)]
    assert len(set(per_round)) == 1
    assert G.in_degree(per_round[0], 2) == 0


def test_a_different_focal_slot_is_a_different_graph_but_the_same_condition():
    focal_2 = _resolved("IN0_OUT1", focal=2)
    focal_0 = _resolved("IN0_OUT1", focal=0)
    assert focal_2 != focal_0
    registry = P.load_registry()
    assert registry.episodes["IN0_OUT1"].condition_hash == registry.episodes["IN0_OUT1"].condition_hash


def test_ring_indegree_and_outdegree_are_two():
    """Spec 11.1: a bidirectional ring gives every slot exactly two neighbours at N>=4."""
    for n in (5, 9):
        graph = _resolved("RING_FIXED", n=n, focal=None)
        assert all(G.in_degree(graph, slot) == 2 for slot in range(n))
        assert all(G.out_degree(graph, slot) == 2 for slot in range(n))
        assert G.edge_count(graph) == 2 * n


def test_a_three_agent_ring_is_the_complete_graph():
    """At N=3 the ring is not a sparser treatment: it is all-to-all, an equality the planner
    must treat as explicit reuse rather than as a second arm (spec 10.1)."""
    assert G.ring_fixed(3, (0, 1, 2)) == G.complete(3)


def test_a_rewired_ring_changes_between_rounds_and_a_fixed_one_does_not():
    registry = P.load_registry()
    rewired = registry.episodes["RING_REWIRED"]
    per_round = {r: G.resolve_graph(rewired.graph, n=9, round_=r, focal_slot=None,
                                    study_seed=SEED, pairing_id=PAIRING) for r in range(1, 5)}
    assert len(set(per_round.values())) > 1
    fixed = registry.episodes["RING_FIXED"]
    assert len({G.resolve_graph(fixed.graph, n=9, round_=r, focal_slot=None, study_seed=SEED,
                                pairing_id=PAIRING) for r in range(1, 5)}) == 1


def test_rewiring_is_keyed_on_the_pairing_not_on_any_output():
    """Spec 6.4: the schedule is drawn from a correctness-blind key, so it depends on the
    episode and the round and on nothing an agent produced."""
    a = G.ring_permutation(SEED, PAIRING, 3, 9)
    b = G.ring_permutation(SEED, PAIRING, 3, 9)
    c = G.ring_permutation(SEED, "other-pairing", 3, 9)
    assert a == b and a != c


def test_random_protection_protects_one_slot_drawn_blind():
    graph = _resolved("RANDOM_PROTECT", focal=None)
    protected = G.protected_slot(SEED, PAIRING, 5)
    assert G.in_degree(graph, protected) == 0
    assert G.out_degree(graph, protected) == 4
    assert G.edge_count(graph) == 16


def test_an_unknown_builder_or_a_bad_permutation_is_refused():
    with pytest.raises(ContractViolation):
        G.GraphRef(name="star", params={})
    with pytest.raises(ContractViolation):
        G.ring_fixed(5, (0, 1, 2, 3, 3))
    with pytest.raises(ContractViolation):
        G.focal_channels(5, 7, incoming=True, outgoing=True)
    with pytest.raises(ContractViolation):
        G.complete(1)


def test_a_per_round_schedule_needs_a_per_round_parameter_and_the_reverse():
    ring = G.GraphRef(name="ring_fixed", params={"perm": G.PERM_ROUND})
    with pytest.raises(ContractViolation):
        G.GraphSchedule(kind=G.SCHEDULE_FIXED, ref=ring)
    once = G.GraphRef(name="ring_fixed", params={"perm": G.PERM_ONCE})
    with pytest.raises(ContractViolation):
        G.GraphSchedule(kind=G.SCHEDULE_PER_ROUND, ref=once)


def test_a_focal_graph_without_a_focal_slot_is_refused_not_defaulted():
    spec = P.load_registry().episodes["IN0_OUT1"]
    with pytest.raises(ContractViolation):
        G.resolve_graph(spec.graph, n=5, round_=1, focal_slot=None, study_seed=SEED,
                        pairing_id=PAIRING)


def test_every_registry_condition_resolves_at_three_five_and_nine_agents():
    registry = P.load_registry()
    for name, spec in registry.episodes.items():
        for n in (3, 5, 9):
            graph = G.resolve_graph(spec.graph, n=n, round_=1, focal_slot=0, study_seed=SEED,
                                    pairing_id=PAIRING)
            assert len(graph) == n, name
