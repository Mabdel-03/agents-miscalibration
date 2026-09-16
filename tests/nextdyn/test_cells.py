"""Cell ids, options and block manifests (plan §3.3, §3.4; spec 17.2).

The three claims this file exists for are the three ARD36 defects ``cells.py`` designs out: a
cell id that can collide across kinds or across manifests (D-11 / D-04), a manifest that can
carry the label-conditioned construction, and a dispatch order that can hand a partial run one
arm's worth of data.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import deps, ids
from agents_scaling.nextdyn.cells import Block, Bucket, CellSpec, ConditionRef, PlanState
from agents_scaling.nextdyn.errors import ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool

SEED = bytes.fromhex("aa" * 32)


def cell(cell_id: str, *, kind: CellKind = CellKind.EPISODE, label: str = "PRIVATE",
         items: tuple[str, ...] = ("item0",), **kwargs) -> CellSpec:
    condition = None if kind not in (CellKind.EPISODE, CellKind.RECEIVER) else ConditionRef(
        protocol_hash="0" * 12, display_label=label)
    return CellSpec(cell_id=cell_id, kind=kind, lane=Lane.SMALL, pool=Pool.DEADLINE,
                    model="primary", family="MATH", tokens_hash="deadbeef", condition=condition,
                    items=items, replicate=0, split_role="pilot", block=Block.E1_NATURAL,
                    **kwargs)


# --------------------------------------------------------------------------- ids


def test_bank_cell_ids_can_never_equal_an_episode_id():
    """D-11: the kind is the first component, so no two kinds share a directory.

    ARD36 let a bank draw and an episode share an id, which put two cohorts in one directory
    and made 14,336 decision keys ambiguous.  Here the id starts with the kind and a cell
    whose id does not is refused at construction.
    """
    common = dict(family="MATH", model="primary", tokens_hash8="deadbeef",
                  condition_hash12=None, split_role="pilot", replicate=0, shard=0)
    bank = CE.cell_id_for(CellKind.DIFFICULTY_BANK, **common)
    episode = CE.cell_id_for(CellKind.EPISODE, **common)
    assert bank != episode
    assert bank.startswith(CellKind.DIFFICULTY_BANK.value + CE.ID_SEP)
    assert episode.startswith(CellKind.EPISODE.value + CE.ID_SEP)
    # and no episode *id* (ids.episode_id) can be mistaken for a cell id: different alphabet
    episode_id = ids.episode_id(CellKind.EPISODE, "team:" + "a" * 32, "b" * 12, "c" * 16,
                                "d" * 8, 0)
    assert not deps._CELL_ID_RE.match(episode_id)
    with pytest.raises(ManifestError):
        cell("episode_without_a_kind_prefix")


def test_every_cell_id_component_is_a_bounded_token():
    with pytest.raises(ManifestError):
        CE.cell_id_for(CellKind.EPISODE, family="MATH/500", model="primary", tokens_hash8="x",
                       condition_hash12=None, split_role="pilot", replicate=0, shard=0)
    long_id = CE.cell_id_for(CellKind.EPISODE, family="MATH", model="primary",
                             tokens_hash8="deadbeefdeadbeef", condition_hash12="c" * 40,
                             split_role="confirmation", replicate=3, shard=12,
                             extra=("constructed", "b0007", "m1"))
    assert deps._CELL_ID_RE.match(long_id)
    assert ".deadbeef." in long_id and ".cccccccccccc." in long_id      # truncated, not dropped


# --------------------------------------------------------------------------- manifests


def test_two_manifests_of_the_same_module_cannot_share_a_cell_id(tmp_run_root):
    """D-04: one builder, one namespace -- a duplicate id is refused, never merged."""
    builder = CE.ManifestBuilder(study_seed=SEED)
    builder.add(cell(CE.cell_id_for(CellKind.EPISODE, family="MATH", model="primary",
                                    tokens_hash8="deadbeef", condition_hash12="a" * 12,
                                    split_role="pilot", replicate=0, shard=0)))
    with pytest.raises(ManifestError, match="duplicate cell id"):
        builder.add(cell(CE.cell_id_for(CellKind.EPISODE, family="MATH", model="primary",
                                        tokens_hash8="deadbeef", condition_hash12="a" * 12,
                                        split_role="pilot", replicate=0, shard=0)),
                    block=Block.E1_CONSTRUCTED)


def test_a_manifest_is_written_once_and_verified_by_its_sidecar(tmp_run_root):
    path = CE.manifest_path(tmp_run_root, Block.E1_NATURAL)
    one = cell(CE.cell_id_for(CellKind.EPISODE, family="MATH", model="primary",
                              tokens_hash8="deadbeef", condition_hash12="a" * 12,
                              split_role="pilot", replicate=0, shard=0))
    sha = CE.write_manifest(path, [one], block=Block.E1_NATURAL)
    assert CE.write_manifest(path, [one], block=Block.E1_NATURAL) == sha      # idempotent
    assert [c.cell_id for c in CE.load_manifest(path)] == [one.cell_id]
    with pytest.raises(ManifestError, match="different content"):
        CE.write_manifest(path, [one, cell(one.cell_id.replace(".s0", ".s1"))],
                          block=Block.E1_NATURAL)
    path.with_name(path.name + CE.SHA_SUFFIX).write_text("00" * 32 + "  x\n")
    with pytest.raises(ManifestError, match="sha256"):
        CE.load_manifest(path)


def test_a_manifest_without_its_sidecar_is_not_frozen(tmp_run_root):
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, [cell(CE.cell_id_for(CellKind.DIFFICULTY_BANK, family="MATH",
                                                 model="primary", tokens_hash8="deadbeef",
                                                 condition_hash12=None, split_role="pilot",
                                                 replicate=0, shard=0),
                                  kind=CellKind.DIFFICULTY_BANK)], block=Block.BANKS)
    path.with_name(path.name + CE.SHA_SUFFIX).unlink()
    with pytest.raises(ManifestError, match="sidecar"):
        CE.load_manifest(path)


# --------------------------------------------------------------------------- options


def test_a_manifest_option_can_never_name_the_construction():
    """spec 5.3: the composition, the C0 and the correctness live in ``root_teams_private``."""
    for key in ("composition", "c0", "correct_slots", "eligibility", "w_star", "gold_answer"):
        with pytest.raises(ManifestError):
            CE.assert_option_is_label_free(key, "x")
    for value in ("C1_COHERENT", "c1_fragmented", "w_star", "semantic_correct"):
        with pytest.raises(ManifestError):
            CE.assert_option_is_label_free("bucket", value)
    # a published *condition* name may say what the receiver's assigned draft was (spec 9.1)
    CE.assert_option_is_label_free("arm", "E2B_1_4_CHECK_CORRECT")


def test_an_option_outside_its_kinds_allowlist_is_refused():
    with pytest.raises(ManifestError, match="not one of the episode options"):
        cell("episode.MATH.primary.deadbeef.x.pilot.r0.s0", options={"pool": "sha"})
    with pytest.raises(ManifestError, match="may depend on"):
        cell("episode.MATH.primary.deadbeef.x.pilot.r0.s0",
             depends_on=(deps.register_dependency("pool", "ab" * 8),))


# --------------------------------------------------------------------------- order


def test_any_chunk_of_a_block_manifest_carries_every_condition():
    """spec 17.2: a partial run is never one arm's worth of data.

    The cells are blind-ordered inside each condition and round-robined across conditions, so
    every window the size of the condition count -- and therefore every chunk a lane driver
    admits -- holds them all.
    """
    conditions = ("PRIVATE", "PEER", "FRESH", "IN0_OUT1")
    cells = [cell(CE.cell_id_for(CellKind.EPISODE, family="MATH", model="primary",
                                 tokens_hash8="deadbeef", condition_hash12=f"{index:012d}",
                                 split_role="pilot", replicate=0, shard=item),
                  label=label, items=(f"item{item}",))
             for index, label in enumerate(conditions) for item in range(24)]
    ordered = CE.order_cells(cells, study_seed=SEED)
    assert len(ordered) == len(cells)
    chunk = 24
    for start in range(0, len(ordered) - chunk + 1):
        window = ordered[start:start + chunk]
        assert {c.display_label for c in window} == set(conditions)
    # and the order is blind, not the input order
    assert [c.cell_id for c in ordered] != [c.cell_id for c in cells]


def test_the_order_is_a_function_of_the_study_seed_alone():
    cells = [cell(f"episode.MATH.primary.deadbeef.x.pilot.r0.s{i}", items=(f"item{i}",))
             for i in range(8)]
    assert CE.order_cells(cells, study_seed=SEED) == CE.order_cells(cells, study_seed=SEED)
    assert (CE.order_cells(cells, study_seed=SEED)
            != CE.order_cells(cells, study_seed=bytes.fromhex("bb" * 32)))


# --------------------------------------------------------------------------- blocks


@pytest.fixture
def state(study_cfg, compute_cfg):
    from agents_scaling.nextdyn import plan as PLAN

    families = tuple(study_cfg.cohorts.families)
    items = {f: tuple(f"{f[:3].lower()}item{i:04d}" for i in range(4)) for f in families}
    inventory = PLAN.Inventory(
        families=families, items=items, natural_items=items,
        panels={p: {f: items[f] for f in families} for p in CE.PANELS},
        buckets={f: tuple(Bucket(family=f, index=i, source_item_id=item, pair="primary")
                          for i, item in enumerate(items[f])) for f in families},
        problems=tuple(f"mod97{i:03d}" for i in range(2)))
    built = PLAN.plan_state(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    # what the planner has by the time GRADE and CONSTRUCT are emitted: the bank cells to grade
    # and the natural-root cells the natural episodes wait for
    cells = {f: (f"donor_bank.{f}.primary.deadbeef.x.confirmation.r0.s0",) for f in families}
    return dataclasses.replace(built, grade_targets=cells, root_cells=cells)


def test_every_block_has_a_builder_and_builds(state):
    assert set(CE.BUILDERS) == set(CE.BLOCKS)
    for block in (Block.BANKS, Block.NATURAL_ROOTS, Block.E1_NATURAL, Block.E1_CONSTRUCTED,
                  Block.E1_FRAGMENTED, Block.E1_BRIDGE, Block.E3A, Block.E3A_PROTECTION,
                  Block.E3A_MATCHED_SLOT, Block.E2B, Block.GRADE, Block.CONSTRUCT, Block.E0,
                  Block.MATERIALIZE):
        cells = CE.build_block(block, state)
        assert cells, f"{block} built nothing"
        assert all(c.block == block or c.block == "" for c in cells)


def test_e3b_ring_and_n9_cells_build_with_zero_engine_changes(state):
    """plan D11: the extension blocks are expressible -- config only, no engine change.

    ``plan --block E3B_RING --dry-run`` is the release gate; this is that gate at the unit
    level.  A ring is a ``protocols.yaml`` entry, N=9 is a cell option, and both compose the
    same EPISODE kind that E1 uses.
    """
    ring = CE.build_block(Block.E3B_RING, state)
    assert ring and all(c.kind is CellKind.EPISODE for c in ring)
    assert {c.option("revisions") for c in ring} == {state.cfg.experiments.E3b.horizon_rounds}
    sizes = CE.build_block(Block.E3B_SIZE, state)
    assert {c.option("n_agents") for c in sizes} == set(state.cfg.experiments.E3b.team_sizes)
    assert all(c.condition is not None for c in sizes)
    replay = CE.build_block(Block.E3C_REPLAY, state)
    assert {c.model for c in replay} == {state.cfg.experiments.E3c.model}
    family = CE.build_block(Block.E3C_FAMILY, state)
    assert {c.model for c in family} == set(state.cfg.experiments.E3C_FAMILY.models)


def test_a_constructed_cell_waits_for_its_bucket_bundle(state):
    cells = CE.build_block(Block.E1_CONSTRUCTED, state)
    for one in cells:
        assert len(one.items) == 1
        assert one.depends_on == (deps.bundle_dependency(one.family, one.option("bucket")),)
        assert one.condition.composition in state.cfg.cohorts.paired_compositions
        # the id carries the member index, never the composition name (it travels into the
        # capture panel's ``cells`` option, where a composition would be a label in a manifest)
        assert one.condition.composition not in one.cell_id


def test_a_bank_cell_is_sized_by_compute_yaml(state, compute_cfg):
    cells = CE.build_block(Block.BANKS, state)
    per_cell = compute_cfg.cells.kinds["difficulty_bank"].items_per_cell
    assert all(len(c.items) <= per_cell for c in cells)
    assert {c.kind for c in cells} == {CellKind.DIFFICULTY_BANK, CellKind.DONOR_BANK}
    assert all(c.option("extension_stage") == 0 for c in cells)


def test_a_manifest_round_trips_through_json(tmp_run_root, state):
    cells = CE.build_block(Block.E2B, state)[:4]
    path = CE.manifest_path(tmp_run_root, Block.E2B)
    CE.write_manifest(path, cells, block=Block.E2B, meta={"stage": "test"})
    loaded = CE.load_manifest(path)
    assert [c.to_dict() for c in loaded] == [c.to_dict() for c in cells]
    assert json.loads(path.read_text())["block"] == Block.E2B
    assert CE.load_manifest_meta(path) == {"stage": "test"}
