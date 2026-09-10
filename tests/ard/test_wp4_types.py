"""WP4 types: the ARD cell record, its projection onto the frozen CellSpec, lanes, deps.

Contract §0.1 (``study/`` is frozen and imported, never edited) and §1 (WP4 owns
``ard/types.py``).  Plan §4.2 (the additive lane change), §5.8 (the episode/decision/state
grains ``ard/compact.py`` consumes).
"""

from __future__ import annotations

import dataclasses

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import types as T
from agents_scaling.study import types as ST


# --------------------------------------------------------------------------- lanes


def test_the_ard_lane_tuple_appends_the_three_new_lanes_without_editing_study():
    assert T.LANES is K.LANES
    assert T.LANES == (K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE, "grade", "analysis", "capture")
    assert T.LANES[-3:] == ("grade", "analysis", "capture")
    from agents_scaling.study.cells import LANES as STUDY_LANES

    assert STUDY_LANES == ("32B", "14B", "8B", "4B", "eval")   # frozen, untouched


def test_the_frozen_study_vocabulary_is_untouched():
    assert [d.value for d in ST.Domain] == ["hle", "bcb"]
    assert [m.value for m in ST.Method] == ["BANK", "S_FRESH", "S_HISTORY", "IND_VOTE", "DEC",
                                            "CEN_FLAT", "IND_PRIVATE_REVISION", "DEC_ONE_ROUND",
                                            "DEGREE"]
    assert {f.name for f in dataclasses.fields(ST.PublicTask)} == {
        "source_id", "domain", "split", "task_text", "answer_format", "stratum", "rank",
        "task_tokens", "entry_point", "category"}


# --------------------------------------------------------------------------- cell spec


def _cell(**overrides):
    base = dict(
        cell_id="EPISODE.natural.PEER.8B.T1024.s000", kind=T.ArdCellKind.EPISODE,
        protocol=K.Protocol.PEER, round=None, cohort=K.Cohort.NATURAL,
        checkpoint=K.CHECKPOINT_SMALL, reasoning_allowance=K.THINKING_TOKENS_LOW,
        final_allowance=K.FINAL_TOKENS, items=("i1", "i2"), team_id=None, parent_ref=None,
        leader_seed=None, depends_on=(), max_inflight=5, parallel_items=1, lane="8B")
    base.update(overrides)
    return T.ArdCellSpec(**base)


def test_the_cell_record_round_trips_through_json_with_optional_enums():
    cell = _cell(protocol=None, kind=T.ArdCellKind.ROOTS, options={"draws": 16})
    data = cell.to_dict()
    assert data["protocol"] is None and data["kind"] == "ROOTS"
    assert T.ArdCellSpec.from_dict(data) == cell


@pytest.mark.parametrize("overrides,match", [
    ({"lane": "14B"}, "lane"),
    ({"checkpoint": "4B"}, "checkpoint"),
    ({"split": "main"}, "split"),
    ({"reasoning_allowance": 2048}, "reasoning_allowance"),
    ({"final_allowance": 512}, "final_allowance"),
    ({"items": ()}, "no items"),
    ({"items": ("i1", "i1")}, "duplicate"),
    ({"max_inflight": 0}, "concurrency"),
    ({"round": 9}, "round"),
    ({"instance": "not a tag!"}, "instance"),
    ({"options": {"is_correct": True}}, "label-conditioned"),
])
def test_the_cell_record_refuses_what_the_frozen_design_does_not_admit(overrides, match):
    with pytest.raises(T.ArdManifestError, match=match):
        _cell(**overrides)


def test_the_projection_onto_the_frozen_cellspec_keeps_what_the_reused_runner_reads():
    cell = _cell(depends_on=("controller:" + "ab" * 32,))
    projected = cell.as_study_cell()
    assert isinstance(projected, ST.CellSpec)
    assert projected.cell_id == cell.cell_id and projected.items == cell.items
    assert projected.lane == cell.lane and projected.split == cell.split
    assert projected.depends_on == cell.depends_on
    assert projected.max_inflight == cell.max_inflight
    assert projected.parallel_items == cell.parallel_items
    # documented placeholders for the v4 factorial fields
    assert projected.method is ST.Method.BANK and projected.framing is ST.Framing.NATIVE
    assert projected.B == 0 and projected.N == K.N_AGENTS
    assert projected.kind is ST.CellKind.GENERATE


def test_the_seed_namespace_of_a_cell_follows_its_protocol():
    assert _cell(protocol=K.Protocol.GATE).namespace == K.NS_GATE
    assert _cell(protocol=None, kind=T.ArdCellKind.ROOTS).namespace == K.NS_STATELESS_BANK


# --------------------------------------------------------------------------- dependencies


def test_dependency_helpers_round_trip_and_classify():
    controller = T.controller_dependency("ab" * 32)
    state = T.state_dependency("cd" * 32)
    cell = T.cell_dependency("EPISODE.natural.PEER.8B.T1024.s000")
    assert T.parse_dependency(controller) == ("controller", "ab" * 32)
    assert T.parse_dependency(state) == ("state", "cd" * 32)
    assert T.parse_dependency(cell)[0] == "cell"
    assert T.parse_dependency("bare.cell.id") == ("cell", "bare.cell.id")
    assert T.hard_dependencies([controller, state, cell]) == [controller, state]


@pytest.mark.parametrize("sha", ["", "xyz", "ab", "zz" * 32, "ab" * 40])
def test_a_dependency_needs_a_real_hex_digest(sha):
    with pytest.raises(T.ArdManifestError):
        T.controller_dependency(sha)


def test_a_dependency_digest_is_normalised_to_lower_case():
    assert T.controller_dependency("AB" * 32) == "controller:" + "ab" * 32


# --------------------------------------------------------------------------- run records


def test_the_episode_record_carries_the_spec_18_fields_and_never_constructed_c0():
    fields = {f.name for f in dataclasses.fields(T.EpisodeRecord)}
    assert {"episode_id", "item_id", "cohort", "protocol", "checkpoint_revision",
            "root_candidate_ids", "root_seed_ids", "N", "horizon", "reasoning_allowance",
            "final_allowance", "packet_policy", "memory_policy", "leader_id",
            "condition_manifest_hash"} <= fields
    assert "constructed_C0" not in fields and "is_correct" not in fields


def test_decision_and_state_records_round_trip_and_expose_validity():
    decision = T.DecisionRef(decision_id="d", episode_id="e", request_id="r", agent_id=0,
                             round=1, phase="revise", protocol="PEER", checkpoint="8B",
                             answer_class="A")
    assert decision.valid is True
    assert T.DecisionRef.from_dict(decision.to_dict()) == decision
    invalid = dataclasses.replace(decision, parse_status=K.ParseStatus.MISSING_FINAL.value)
    assert invalid.valid is False
    unkeyed = dataclasses.replace(decision, answer_class=None)
    assert unkeyed.valid is False
    state = T.RoundState(state_id="s", episode_id="e", round=1, decision_ids=("d",),
                         answer_classes=("A",), valid=(True,))
    assert state.n_valid == 1 and T.RoundState.from_dict(state.to_dict()) == state


def test_episode_and_decision_ids_are_derived_from_the_cell_so_two_waves_differ():
    first = _cell()
    second = _cell(cell_id="EPISODE.natural.PEER.8B.T1024.pdeadbeef.ww2.s000", instance="w2",
                   parent_ref="controller:" + "de" * 32)
    assert T.episode_id_for(first, "i1") != T.episode_id_for(second, "i1")
    assert T.decision_id_for(T.episode_id_for(first, "i1"), round_=1, agent_id=2,
                             phase="revise").endswith("|r1|a2|revise")


def test_the_item_result_key_is_the_field_the_reused_resume_check_reads():
    fields = {f.name for f in dataclasses.fields(T.ArdItemResult)}
    assert "source_id" in fields and "cell" in fields and "status" in fields
