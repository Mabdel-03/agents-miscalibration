"""Prior use, near duplicates and split roles (spec 5.2, 7.1; plan §2.2, D19).

The synthetic panel of ``test_items_build`` is reused so the real split policy runs: a
180-rank pilot block, a 200-item natural frame under the D19 prior-use rule, the constructed
scan order and every ``sub_*`` panel as a rank prefix.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents_scaling.config import DEFAULT_HF_HOME
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import datasets as D
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import inventory as INV
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import items_build as IB
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study.data.splits import rank_key
from tests.nextdyn.conftest import CONFIG_DIR
from tests.nextdyn.test_items_build import (
    EMPTY_REGISTRY,
    PER_FAMILY,
    SALT,
    SEED,
    draft_of,
    math_row,
    mc_row,
    synthetic_drafts,
)

HF_HOME = Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
CACHE_READY = (HF_HOME / "datasets").is_dir()


@pytest.fixture(scope="module")
def cfg():
    return C.load_study(CONFIG_DIR / "study.yaml", mode="test")


@pytest.fixture(scope="module")
def panel():
    panels = synthetic_drafts()
    return [draft for drafts in panels.values() for draft in drafts]


@pytest.fixture(scope="module")
def frames(panel, cfg):
    return INV.assign_split_roles(panel, [], [], cfg, SALT)


@pytest.fixture(scope="module")
def built(tmp_path_factory, cfg):
    run_root = tmp_path_factory.mktemp("nd_inventory")
    panels = synthetic_drafts()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(IB, "build_drafts", lambda cfg_, home: panels)
    try:
        IB.build_items(run_root, cfg, split_salt=SALT, study_seed=SEED, hf_home=run_root,
                       registry=EMPTY_REGISTRY)
    finally:
        monkey.undo()
    return run_root


# --------------------------------------------------------------------------- prior use


def test_prior_use_inventory_marks_unknown_not_untouched(panel):
    """Spec 18.1 / 5.2: no root can say "untouched"; silence is ``unknown``."""
    assert INV.prior_use_status([]) == INV.STATUS_UNKNOWN
    assert "untouched" not in INV.STATUS_ORDER
    records = INV.prior_use_inventory(panel[:5], HF_HOME, registry=EMPTY_REGISTRY)
    assert records == []
    frames = INV.assign_split_roles(panel[:400], [], [], _test_cfg(), SALT)
    assert {frame.prior_use_status for frame in frames.values()} == {INV.STATUS_UNKNOWN}
    assert all(frame.prior_use_sources == () for frame in frames.values())


def _test_cfg():
    return C.load_study(CONFIG_DIR / "study.yaml", mode="test")


def test_prior_use_precedence_prefers_the_strongest_evidence():
    def record(status):
        return INV.PriorUse("item0001", F.MATH500, "root", status, INV.MATCH_SOURCE_ID, "")

    assert INV.prior_use_status([record(INV.STATUS_RESERVED_UNUSED),
                                 record(INV.STATUS_UNKNOWN)]) == INV.STATUS_UNKNOWN
    assert INV.prior_use_status([record(INV.STATUS_UNKNOWN),
                                 record(INV.STATUS_USED_INFERRED)]) == INV.STATUS_USED_INFERRED
    assert INV.prior_use_status([record(INV.STATUS_USED_INFERRED),
                                 record(INV.STATUS_USED_VERIFIED)]) == INV.STATUS_USED_VERIFIED
    with pytest.raises(ContractViolation):
        INV.PriorUse("i", "f", "r", "untouched", INV.MATCH_NONE, "")


def test_a_prior_root_without_a_sidecar_can_only_say_unknown(tmp_path, panel):
    root = tmp_path / "long_context_protocol_smoke_v9"
    root.mkdir()
    registry = D.LegacyRegistry(path=None, roots={}, prior_roots=(str(root),))
    records = INV.prior_use_inventory(panel[:3], HF_HOME, registry=registry)
    assert {r.status for r in records} == {INV.STATUS_UNKNOWN}
    assert {r.match_method for r in records} == {INV.MATCH_NONE}
    assert {r.root for r in records} == {root.name}


def test_a_prior_root_whose_cells_name_a_benchmark_only_touches_that_family(tmp_path, panel):
    root = tmp_path / "smoke_math_only"
    root.mkdir()
    (root / "cells.json").write_text(json.dumps({"cells": [{"benchmark": "math", "seed": 0}]}))
    registry = D.LegacyRegistry(path=None, roots={}, prior_roots=(str(root),))
    records = INV.prior_use_inventory(panel, HF_HOME, registry=registry)
    assert {r.family for r in records} == {F.MATH500}


@pytest.mark.needs_hf_cache
@pytest.mark.skipif(not CACHE_READY, reason="the pinned datasets are not cached")
def test_run0_contracts_replay_and_verify_the_canonical_question_hashes():
    """Spec 7.1: the Run0 permutation is replayed and checked, never assumed."""
    registry = D.load_legacy_sources()
    root = registry.roots.get(INV.ROOT_FULL_SWEEP)
    contracts = Path(root) / INV.RUN0_CONTRACTS
    if not contracts.exists():
        pytest.skip("the Run0 contract file is not on this filesystem")
    drafts = []
    for benchmark, family in INV.BENCHMARK_FAMILY_MAP.items():
        index = INV._run0_index(benchmark, HF_HOME)
        spec = F.family(family)
        for row_index in list(index)[:5]:
            row = D.SourceRow(family=family, original_id=index[row_index], question="q",
                              options=(), gold={}, subject=None, level=None)
            item_id = IB.source_item_id("s", family, spec.loader.revision, index[row_index])
            drafts.append(IB.ItemDraft(source_item_id=item_id, family=family, row=row,
                                       variants=()))
    records = INV._run0_prior_use(contracts, {(d.family, d.row.original_id): d.source_item_id
                                              for d in drafts}, HF_HOME)
    assert records, "the replay matched no pinned row"
    assert {r.status for r in records} == {INV.STATUS_USED_VERIFIED}
    assert {r.match_method for r in records} == {INV.MATCH_REPLAY_VERIFIED}


@pytest.mark.needs_hf_cache
@pytest.mark.skipif(not CACHE_READY, reason="the pinned datasets are not cached")
def test_the_ard36_panel_is_recognised_as_used_and_its_reserve_as_reserved():
    registry = D.load_legacy_sources()
    root = Path(registry.roots[INV.ROOT_ARD36])
    if not (root / INV.ARD36_ITEMS).exists():
        pytest.skip("the ARD36 run root is not on this filesystem")
    rows = D.load_panel(F.MATH500, HF_HOME).rows[:500]
    spec = F.family(F.MATH500).loader
    by_id = {(F.MATH500, row.original_id):
             IB.source_item_id("s", F.MATH500, spec.revision, row.original_id) for row in rows}
    records = INV._ard36_prior_use(root, by_id, INV.BENCHMARK_FAMILY_MAP)
    statuses = {r.status for r in records}
    assert INV.STATUS_USED_VERIFIED in statuses
    assert INV.STATUS_RESERVED_UNUSED in statuses


# --------------------------------------------------------------------------- near duplicates


def test_the_near_duplicate_audit_flags_a_planted_pair_and_never_merges_it():
    text = ("The train leaves the station at noon and travels for three hours at a constant "
            "speed of sixty kilometres per hour before stopping")
    drafts = [draft_of(math_row(i)) for i in range(3)]
    twin_a = draft_of(D.SourceRow(family=F.MATH500, original_id="dup/a.json", question=text,
                                  options=(), gold={"answer": "1", "solution": ""},
                                  subject=None, level=None))
    twin_b = draft_of(D.SourceRow(family=F.MATH500, original_id="dup/b.json",
                                  question=text + " at the platform", options=(),
                                  gold={"answer": "1", "solution": ""}, subject=None, level=None))
    found = INV.near_duplicate_audit([*drafts, twin_a, twin_b])
    assert len(found) == 1
    pair = found[0]
    assert {pair.source_item_id, pair.near_duplicate_of} == {twin_a.source_item_id,
                                                             twin_b.source_item_id}
    assert pair.jaccard >= INV.SHINGLE_JACCARD
    assert INV.near_duplicate_audit(drafts) == []


def test_a_near_duplicate_loser_is_excluded_from_every_frame(panel, cfg):
    victim = panel[7]
    duplicate = INV.NearDuplicate(source_item_id=victim.source_item_id,
                                  near_duplicate_of=panel[8].source_item_id,
                                  family=victim.family, jaccard=0.95)
    frames = INV.assign_split_roles(panel, [], [duplicate], cfg, SALT)
    frame = frames[victim.source_item_id]
    assert frame.split_role == INV.SPLIT_EXCLUDED_DUPLICATE
    assert frame.near_duplicate_of == panel[8].source_item_id
    assert frame.sampling_weight == 0.0 and frame.frame_size is None


# --------------------------------------------------------------------------- split roles


def test_split_roles_are_hash_order_and_the_pilot_block_comes_first(frames, panel, cfg):
    by_family: dict[str, list] = {}
    for draft in panel:
        by_family.setdefault(draft.task_family, []).append(draft)
    for cohort, group in by_family.items():
        ranked = sorted(group, key=lambda d: rank_key(SALT, d.source_item_id))
        depth = max([cfg.cohorts.pilot_items_per_family, *cfg.cohorts.pilot_eligibility_extensions])
        pilot = [frames[d.source_item_id] for d in ranked[:depth]]
        rest = [frames[d.source_item_id] for d in ranked[depth:]]
        assert {frame.split_role for frame in pilot} == {INV.SPLIT_PILOT}
        assert INV.SPLIT_PILOT not in {frame.split_role for frame in rest}
        stages = {frame.split_row.pilot_stage for frame in pilot}
        assert stages == {INV.PILOT_STAGE_BASE, "pilot_ext_120", "pilot_ext_180"}
        assert len(rest) == PER_FAMILY - depth


def test_the_natural_frame_is_the_d19_prior_use_rule(panel, cfg):
    used = [INV.PriorUse(draft.source_item_id, draft.family, "ard36", INV.STATUS_USED_VERIFIED,
                         INV.MATCH_SOURCE_ID, "") for draft in panel[::10]]
    frames = INV.assign_split_roles(panel, used, [], cfg, SALT)
    natural = [f for f in frames.values() if f.split_role in (INV.SPLIT_NATURAL, INV.SPLIT_BOTH)]
    assert natural, "no natural panel was drawn"
    assert all(f.prior_use_status in cfg.cohorts.natural_frame_prior_use for f in natural)
    assert all("natural" in f.frame_rule for f in natural)
    verified = [f for f in frames.values() if f.prior_use_status == INV.STATUS_USED_VERIFIED]
    assert verified and all(f.split_role != INV.SPLIT_NATURAL for f in verified)


def test_every_sub_panel_is_a_rank_prefix_of_its_frame(frames, cfg):
    rows = [frame.split_row for frame in frames.values()]
    for panel_spec in INV.sub_panels(cfg):
        column = f"sub_{panel_spec.name}"
        members = [row.source_item_id for row in rows if getattr(row, column)]
        assert len(members) == panel_spec.size * len(cfg.cohorts.families[:2]) or members
        ordered = sorted(members, key=lambda item_id: rank_key(SALT, f"sub:{panel_spec.name}:{item_id}"))
        assert members and set(ordered) == set(members)


def test_the_planner_refuses_a_sub_panel_bigger_than_its_frame(panel, cfg):
    with pytest.raises(ContractViolation, match="sub-panel"):
        INV.assign_split_roles(panel[:200], [], [], cfg, SALT)


def test_all_variants_of_source_item_share_split_and_cluster(built):
    """Spec 18.1: a variant is a presentation, never a second frame member."""
    run_root = built
    items = I.load_items(run_root)
    splits = {row["source_item_id"]: row for row in I.load_splits(run_root)}
    for item_id, item in items.items():
        assert splits[item_id]["split_role"] == item.split_role
        for variant in item.variants:
            assert variant.source_item_id == item_id          # cluster_id == source_item_id
        assert len({variant.source_item_id for variant in item.variants}) == 1
    assert len(splits) == len(items)


def test_the_split_table_is_written_before_any_outcome_and_covers_every_item(built):
    run_root = built
    rows = I.load_splits(run_root)
    assert {row["split_role"] for row in rows} <= set(INV.SPLIT_ROLES)
    counts = INV.split_counts([tables.row_from_dict(tables.ItemSplitRow, row) for row in rows])
    assert counts[INV.SPLIT_PILOT] == 2 * 180
    assert sum(counts.values()) == len(rows) == 2 * PER_FAMILY
    assert all(row["assigned_at_manifest_hash"] for row in rows)


def test_the_inventory_hashes_are_stable_and_change_with_the_table(built):
    run_root = built
    summary = IB.load_summary(run_root)
    rows = [tables.row_from_dict(tables.SourceItemRow, row)
            for row in tables.read_table(I.public_dir(run_root), "source_items")]
    assert INV.source_inventory_hash(rows) == summary["source_inventory_hash"]
    assert INV.source_inventory_hash(list(reversed(rows))) == summary["source_inventory_hash"]
    assert INV.prior_use_inventory_hash([]) == summary["prior_use_inventory_hash"]
    changed = [tables.row_from_dict(tables.SourceItemRow,
                                    dict(tables.row_to_dict(rows[0]), split_role="reserve"))]
    assert INV.source_inventory_hash(changed) != INV.source_inventory_hash(rows[:1])
