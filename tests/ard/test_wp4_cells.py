"""WP4 manifests: freezing, sha sidecars, cell-id uniqueness, sizes, label discipline.

Spec §5.3 (the challenge manifest lists root candidate ids only), §10 (module → cells),
§18 (manifest hashes).  Plan §5.3, §6 (cell sizes), §7 (train+validation first), §10
("manifest freezing and sha sidecar; cell-id uniqueness across waves and parents").
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import cells as ard_cells
from agents_scaling.ard import contract as K
from agents_scaling.ard import kinds as KINDS
from agents_scaling.ard.types import ArdCellKind, ArdCellSpec, ArdManifestError
from agents_scaling.study.types import ProtocolError
from tests.ard.wp4_support import synthetic_tasks


@pytest.fixture
def tasks():
    return synthetic_tasks(2, splits=("train", "validation", "test"))


@pytest.fixture
def test_tasks(tasks):
    return [task for task in tasks if task.split == "test"]


# --------------------------------------------------------------------------- freezing


def test_manifest_write_creates_the_sha256_sidecar_and_round_trips(tmp_path, tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e1_roots(builder, tasks, checkpoint=K.CHECKPOINT_SMALL)
    path = tmp_path / "cells_E1_8B.json"
    digest = ard_cells.write_ard_cells_file(path, builder.cells, ard_cells.manifest_meta(
        module="E1", lane="8B", cells=builder.cells))
    sidecar = path.with_name(path.name + ".sha256")
    assert sidecar.exists() and sidecar.read_text().split()[0] == digest
    assert ard_cells.load_ard_cells_file(path) == builder.cells


def test_rewriting_the_same_manifest_is_a_no_op_but_a_different_one_is_refused(tmp_path, tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e1_roots(builder, tasks, checkpoint=K.CHECKPOINT_SMALL)
    path = tmp_path / "cells.json"
    meta = ard_cells.manifest_meta(module="E1", lane=None, cells=builder.cells)
    first = ard_cells.write_ard_cells_file(path, builder.cells, meta)
    assert ard_cells.write_ard_cells_file(path, builder.cells, meta) == first
    with pytest.raises(ProtocolError):
        ard_cells.write_ard_cells_file(path, builder.cells[:1], meta)


def test_an_edited_manifest_fails_its_sidecar(tmp_path, tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e1_roots(builder, tasks, checkpoint=K.CHECKPOINT_SMALL)
    path = tmp_path / "cells.json"
    ard_cells.write_ard_cells_file(path, builder.cells, ard_cells.manifest_meta(
        module="E1", lane=None, cells=builder.cells))
    payload = json.loads(path.read_text())
    payload["cells"][0]["items"] = payload["cells"][0]["items"][:1]
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    with pytest.raises(ProtocolError, match="sha256"):
        ard_cells.load_ard_cells_file(path)


# --------------------------------------------------------------------------- uniqueness


def test_cell_ids_are_unique_within_a_manifest(tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2_natural(builder, tasks, checkpoint=K.CHECKPOINT_SMALL)
    ids = [cell.cell_id for cell in builder.cells]
    assert len(set(ids)) == len(ids)


def test_two_waves_of_a_dependent_stage_never_share_a_cell_directory(tasks):
    """A new sealed state (or a new wave tag) is a new directory, so no wave resumes onto
    another wave's item files (plan §5.3)."""
    first = ard_cells.ArdBuilder()
    ard_cells.build_e4_forks(first, tasks, checkpoint=K.CHECKPOINT_SMALL, state_sha="aa" * 32)
    second = ard_cells.ArdBuilder()
    ard_cells.build_e4_forks(second, tasks, checkpoint=K.CHECKPOINT_SMALL, state_sha="bb" * 32)
    third = ard_cells.ArdBuilder()
    ard_cells.build_e4_forks(third, tasks, checkpoint=K.CHECKPOINT_SMALL, state_sha="aa" * 32,
                             instance="w2")
    ids = [{cell.cell_id for cell in builder.cells} for builder in (first, second, third)]
    assert ids[0] and not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_a_dependent_cell_id_carries_its_parent_hash_and_instance_tag(test_tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e4_gate(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                            controller_sha="cd" * 32, protocols=(K.Protocol.GATE,),
                            instance="w1")
    cell = builder.cells[0]
    assert f".p{('cd' * 32)[:8]}" in cell.cell_id and ".ww1." in cell.cell_id
    assert cell.depends_on == (f"controller:{'cd' * 32}",)
    assert cell.needs_controller and cell.hard_dependencies == list(cell.depends_on)


def test_gate_without_a_frozen_controller_is_refused_at_manifest_time(test_tasks):
    builder = ard_cells.ArdBuilder()
    with pytest.raises(ArdManifestError, match="controller"):
        ard_cells.build_e2_natural(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                                   protocols=(K.Protocol.GATE,))


# --------------------------------------------------------------------------- sizes / order


@pytest.mark.parametrize(
    "checkpoint,budget,protocol,expected",
    [
        (K.CHECKPOINT_SMALL, K.THINKING_TOKENS_LOW, K.Protocol.PEER, 8),
        (K.CHECKPOINT_SMALL, K.THINKING_TOKENS_LOW, K.Protocol.HUB5, 4),
        (K.CHECKPOINT_LARGE, K.THINKING_TOKENS_LOW, K.Protocol.PEER, 4),
        (K.CHECKPOINT_SMALL, K.THINKING_TOKENS_HIGH, K.Protocol.PEER, 3),
        (K.CHECKPOINT_LARGE, K.THINKING_TOKENS_HIGH, K.Protocol.PEER, 2),
    ],
)
def test_cell_sizes_follow_the_plan_per_kind_rule(checkpoint, budget, protocol, expected):
    assert KINDS.items_per_cell(ArdCellKind.EPISODE, checkpoint=checkpoint,
                                reasoning_allowance=budget, protocol=protocol) == expected


def test_no_cell_mixes_splits_and_train_validation_come_first(tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2_natural(builder, tasks, checkpoint=K.CHECKPOINT_SMALL,
                               protocols=(K.Protocol.PEER,))
    splits = [cell.split for cell in builder.cells]
    assert set(splits) <= set(K.SPLITS)
    order = {split: index for index, split in enumerate(ard_cells.SPLIT_ORDER)}
    assert splits == sorted(splits, key=lambda split: order[split])


def test_dispatch_order_interleaves_families_inside_a_split(tasks):
    ordered = ard_cells.dispatch_order(tasks)
    train = [task.family for task in ordered if task.split == "train"]
    assert train[:2] == [K.Family.MATH500.value, K.Family.MMLUPRO.value]


# --------------------------------------------------------------------------- label discipline


def test_challenge_manifests_refuse_non_test_items(tasks):
    builder = ard_cells.ArdBuilder()
    with pytest.raises(ArdManifestError, match="TEST"):
        ard_cells.build_e2c_challenge(builder, tasks, checkpoint=K.CHECKPOINT_SMALL,
                                      team_ids=("teamA",), teams_manifest_sha="ab" * 32)


@pytest.mark.parametrize("team_id", ["C0", "c1", "correctfour", "wrong1"])
def test_team_tags_that_encode_their_composition_are_refused(team_id):
    with pytest.raises(ArdManifestError):
        ard_cells.assert_opaque_team_id(team_id)


def test_a_cell_may_not_carry_constructed_c0_or_any_label_key(test_tasks):
    builder = ard_cells.ArdBuilder()
    with pytest.raises(ArdManifestError, match="label-conditioned"):
        builder.emit(ArdCellKind.CHALLENGE, test_tasks, cohort=K.Cohort.CHALLENGE,
                     checkpoint=K.CHECKPOINT_SMALL, protocol=K.Protocol.PRIVATE,
                     team_id="teamA", options={"constructed_C0": 4})
    assert "constructed_C0" not in {field for field in ArdCellSpec.__dataclass_fields__}


def test_challenge_cells_carry_only_the_team_tag_and_the_frozen_manifest_hash(test_tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2c_challenge(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                                  team_ids=("teamA", "teamB"), teams_manifest_sha="ab" * 32,
                                  protocols=(K.Protocol.PRIVATE,))
    for cell in builder.cells:
        assert cell.team_id in ("teamA", "teamB")
        assert set(cell.options) == {"teams_manifest_sha256"}
        assert cell.split == "test"


def test_e2d_runs_the_high_budget_arms_with_roots_by_reference(test_tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2d_high_budget(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL)
    assert {cell.protocol for cell in builder.cells} == set(K.HIGH_BUDGET_PROTOCOLS)
    for cell in builder.cells:
        assert cell.reasoning_allowance == K.THINKING_TOKENS_HIGH
        assert cell.options["root_budget"] == K.ROOT_THINKING_TOKENS
        assert cell.options["roots_by_reference"] is True


def test_every_manifest_cell_lands_on_a_declared_lane(tasks, test_tasks):
    builder = ard_cells.ArdBuilder()
    ard_cells.build_e1_roots(builder, tasks, checkpoint=K.CHECKPOINT_LARGE)
    ard_cells.build_e2_natural(builder, tasks, checkpoint=K.CHECKPOINT_SMALL,
                               protocols=(K.Protocol.PEER,))
    ard_cells.build_grade_cells(builder, tasks)
    ard_cells.build_analysis_cells(builder)
    ard_cells.build_e6_capture(builder, test_tasks, decision_manifest_sha="ef" * 32)
    lanes = {cell.lane for cell in builder.cells}
    assert lanes <= set(K.LANES)
    assert {"32B", "8B", "grade", "analysis", "capture"} <= lanes
