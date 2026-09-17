"""Panel draws: seeded, outcome-blind, disjoint, proportional, and BBH wave blocks of one per subtask."""
from __future__ import annotations

import collections
from dataclasses import replace
from fractions import Fraction

import pytest

from agent_allocation.benchmarks.text_manifest import largest_remainder, select_panels, validate_panel_draw
from agent_allocation.contracts import (
    BBH_SUBTASKS,
    WAVE_ITEMS_PER_BENCHMARK,
    EvalKind,
    InventoryItem,
    OutputKind,
    PrivateEvalSpec,
    PublicTask,
)

SIZES = {"n_dev": 8, "n_main": 100, "n_reserve": 8}


def _item(benchmark: str, item_id: str, strata, *, subtask=None, eligible=True, reason=None) -> InventoryItem:
    public = PublicTask(item_id=item_id, benchmark=benchmark, source_revision="rev", prompt=f"prompt {item_id}",
                        output_kind=OutputKind.OPTION, allowed_choices=("A", "B"), subtask=subtask)
    private = PrivateEvalSpec(item_id=item_id, benchmark=benchmark, source_revision="rev",
                              eval_kind=EvalKind.OPTION_EXACT, gold_answer="A", private_record_ref=None,
                              private_hash="h", subtask=subtask)
    return InventoryItem(public=public, private=private, strata=strata, eligible=eligible, exclusion_reason=reason)


def _math_like(n_per=(3, 40, 7, 90, 12, 1, 60, 5)):
    items = []
    for s, count in enumerate(n_per):
        for k in range(count):
            items.append(_item("math500", f"math500/s{s}/{k}", (("level", str(s % 5)), ("subject", f"S{s}"))))
    return items


def _bbh_like(per=30):
    return [_item("bbh_logic4", f"bbh_logic4/{sub}/{k}", (("subtask", sub),), subtask=sub)
            for sub in BBH_SUBTASKS for k in range(per)]


def test_largest_remainder_is_exact_and_proportional():
    sizes = {"a": 3, "b": 40, "c": 7, "d": 90}
    alloc = largest_remainder(sizes, 50, tie_key=lambda name: name)
    assert sum(alloc.values()) == 50
    population = sum(sizes.values())
    for name, size in sizes.items():
        quota = Fraction(50 * size, population)
        assert int(quota) <= alloc[name] <= int(quota) + 1
    with pytest.raises(ValueError):
        largest_remainder(sizes, population + 1, tie_key=lambda name: name)


def test_stratified_draw_is_disjoint_proportional_and_deterministic():
    inventory = _math_like()
    draw = select_panels(inventory, benchmark="math500", seed=20260917, **SIZES)
    assert (len(draw.dev_item_ids), len(draw.main_item_ids), len(draw.reserve_item_ids)) == (8, 100, 8)
    assert not set(draw.dev_item_ids) & set(draw.main_item_ids)
    assert not set(draw.reserve_item_ids) & (set(draw.dev_item_ids) | set(draw.main_item_ids))
    assert draw == select_panels(list(reversed(inventory)), benchmark="math500", seed=20260917, **SIZES)
    assert draw.selection_seed == 20260917 and draw.procedure.startswith("text-panels-v1")
    strata = {item.public.item_id: item.strata for item in inventory}
    sizes = collections.Counter(item.strata for item in inventory)
    main = collections.Counter(strata[item_id] for item_id in draw.main_item_ids)
    for stratum, size in sizes.items():
        quota = Fraction(100 * size, len(inventory))
        assert int(quota) <= main[stratum] <= int(quota) + 1


def test_seed_changes_the_draw_but_not_its_shape():
    inventory = _math_like()
    a = select_panels(inventory, benchmark="math500", seed=1, **SIZES)
    b = select_panels(inventory, benchmark="math500", seed=2, **SIZES)
    assert a.main_item_ids != b.main_item_ids
    assert len(a.main_item_ids) == len(b.main_item_ids) == 100


def test_main_panel_does_not_depend_on_dev_size():
    inventory = _math_like()
    a = select_panels(inventory, benchmark="math500", seed=5, n_dev=8, n_main=100, n_reserve=8)
    b = select_panels(inventory, benchmark="math500", seed=5, n_dev=16, n_main=100, n_reserve=0)
    assert a.main_item_ids == b.main_item_ids


def test_ineligible_items_are_never_drawn():
    inventory = _math_like()
    excluded = {item.public.item_id for item in inventory[::3]}
    marked = [replace(item, eligible=False, exclusion_reason="engineering") if item.public.item_id in excluded
              else item for item in inventory]
    draw = select_panels(marked, benchmark="math500", seed=3, **SIZES)
    drawn = set(draw.dev_item_ids) | set(draw.main_item_ids) | set(draw.reserve_item_ids)
    assert not drawn & excluded


def test_prefixes_of_the_main_order_stay_roughly_proportional():
    inventory = _math_like()
    draw = select_panels(inventory, benchmark="math500", seed=11, **SIZES)
    strata = {item.public.item_id: item.strata for item in inventory}
    main = collections.Counter(strata[item_id] for item_id in draw.main_item_ids)
    prefix = collections.Counter(strata[item_id] for item_id in draw.main_item_ids[:24])
    for stratum, count in main.items():
        assert abs(prefix[stratum] - count * 24 / 100) <= 1.0 + 1e-9


def test_bbh_blocks_hold_one_item_per_subtask_in_contract_order():
    inventory = _bbh_like()
    draw = select_panels(inventory, benchmark="bbh_logic4", seed=20260917, **SIZES)
    subtask = {item.public.item_id: item.public.subtask for item in inventory}
    width = WAVE_ITEMS_PER_BENCHMARK
    assert width == len(BBH_SUBTASKS)
    for ids in (draw.main_item_ids, draw.dev_item_ids, draw.reserve_item_ids):
        for w in range(len(ids) // width):
            assert tuple(subtask[i] for i in ids[w * width:(w + 1) * width]) == BBH_SUBTASKS
    assert collections.Counter(subtask[i] for i in draw.main_item_ids) == {s: 25 for s in BBH_SUBTASKS}
    assert collections.Counter(subtask[i] for i in draw.dev_item_ids) == {s: 2 for s in BBH_SUBTASKS}
    for high in (24, 48, 12):          # nested panels are prefixes: equal subtasks in every one
        assert collections.Counter(subtask[i] for i in draw.main_item_ids[:high]) == {s: high // 4 for s in BBH_SUBTASKS}


def test_bbh_sizes_must_divide_by_the_subtask_count():
    with pytest.raises(ValueError):
        select_panels(_bbh_like(), benchmark="bbh_logic4", seed=1, n_dev=6, n_main=100, n_reserve=8)


def test_empty_or_short_inventory_is_a_blocker():
    with pytest.raises(ValueError):
        select_panels([], benchmark="math500", seed=1, **SIZES)
    with pytest.raises(ValueError):
        select_panels(_math_like((5, 5)), benchmark="math500", seed=1, **SIZES)


def test_validator_rejects_overlap():
    inventory = _bbh_like()
    draw = select_panels(inventory, benchmark="bbh_logic4", seed=1, **SIZES)
    bad = replace(draw, dev_item_ids=draw.main_item_ids[:8])
    with pytest.raises(ValueError):
        validate_panel_draw(bad, inventory)


def test_real_inventories_draw_config_sized_panels(cfg, math_inventory, mmlu_inventory, bbh_inventory):
    for benchmark, inventory in (("math500", math_inventory), ("mmlu_pro", mmlu_inventory),
                                 ("bbh_logic4", bbh_inventory)):
        per_benchmark_main = cfg["panels"]["main_items"] // 4
        per_benchmark_dev = cfg["panels"]["pilot_items"] // 4
        draw = select_panels(inventory, benchmark=benchmark, seed=cfg["seed"], n_dev=per_benchmark_dev,
                             n_main=per_benchmark_main, n_reserve=8)
        assert len(draw.main_item_ids) == 100 and len(draw.dev_item_ids) == 8
        assert not set(draw.dev_item_ids) & set(draw.main_item_ids)
        assert draw == select_panels(inventory, benchmark=benchmark, seed=cfg["seed"], n_dev=8, n_main=100,
                                     n_reserve=8)


def test_panel_audit_counts_match_the_draw():
    from agent_allocation.benchmarks.text_manifest import panel_audit
    inventory = _math_like()
    draw = select_panels(inventory, benchmark="math500", seed=13, **SIZES)
    audit = panel_audit(inventory, draw)
    assert audit["population"] == {"items": len(inventory), "eligible": len(inventory), "excluded": 0,
                                   "strata": audit["population"]["strata"]}
    for name, ids in (("dev", draw.dev_item_ids), ("main", draw.main_item_ids), ("reserve", draw.reserve_item_ids)):
        assert audit["panels"][name]["size"] == len(ids) == sum(audit["panels"][name]["strata"].values())
