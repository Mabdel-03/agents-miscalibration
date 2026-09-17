"""MBPP+ inventory: pinned sources, public/private separation, one public assert, seeded panels."""
from __future__ import annotations

import json

import pytest

from agent_allocation._vendor.evalplus_external import decode_evalplus_record
from agent_allocation.benchmarks import mbpp_plus as mp
from agent_allocation.contracts import EvalKind, OutputKind, canonical_json, dump_record

pytestmark = pytest.mark.needs_sources


def test_inventory_matches_frozen_manifest_and_separates_private_fields(run_paths):
    manifest, public_rows, restricted_rows, _ = mp.read_sources()
    items, plan = mp.build_inventory(run_paths)
    assert len(items) == manifest["count"] == len(restricted_rows)      # size read, never hardcoded
    assert len({item.public.item_id for item in items}) == len(items)
    for item in items:
        public, private = item.public, item.private
        native = decode_evalplus_record(plan[public.item_id]["restricted"])
        assert public.item_id == "mbpp_plus/" + native["task_id"].replace("/", "_")
        assert public.output_kind is OutputKind.PYTHON and public.prompt == native["prompt"]
        assert len(public.public_examples) == 1 and public.public_examples[0].startswith("assert")
        assert public.public_examples[0] in public.prompt
        text = canonical_json(dump_record(public))
        assert native["canonical_solution"].strip() not in text
        for field in ("canonical_solution", "base_input", "plus_input", "assertion", "contract", "atol"):
            assert field not in json.loads(text)
        private_asserts = [line.strip() for line in native["assertion"].splitlines() if line.strip()]
        leaked = [line for line in private_asserts if line in text and line != public.public_examples[0]]
        assert not leaked, (public.item_id, leaked)
        assert private.eval_kind is EvalKind.EVALPLUS_FULL and private.gold_answer is None
        assert private.private_record_ref.startswith("private/evalplus_oracles/")
        assert item.eligible, item.exclusion_reason


def test_parse_public_assert_rejects_zero_or_two_asserts():
    assert mp.parse_public_assert('"""\nDo it.\nassert f(1) == 2\n"""\n') == "assert f(1) == 2"
    with pytest.raises(ValueError):
        mp.parse_public_assert('"""\nDo it.\n"""\n')
    with pytest.raises(ValueError):
        mp.parse_public_assert('"""\nassert f(1) == 2\nassert f(2) == 3\n"""\n')


def test_install_private_record_is_idempotent_and_private(small_inventory, run_paths):
    chosen, plan = small_inventory
    item = chosen["Mbpp/2"]
    entry = plan[item.public.item_id]
    mp.install_private_record(entry)                                     # second call: verify only
    directory = run_paths.run_dir / item.private.private_record_ref
    assert (directory / "oracle.json").exists() and (directory / "restricted.json").exists()
    assert oct(directory.stat().st_mode & 0o777) == "0o700"


def test_select_panels_seeded_disjoint_and_order_stable(run_paths):
    items, _ = mp.build_inventory(run_paths)
    draw = mp.select_panels(items, seed=20260917, n_dev=8, n_main=100, n_reserve=8)
    again = mp.select_panels(list(reversed(items)), seed=20260917, n_dev=8, n_main=100, n_reserve=8)
    assert draw == again
    assert (len(draw.dev_item_ids), len(draw.main_item_ids), len(draw.reserve_item_ids)) == (8, 100, 8)
    everything = draw.dev_item_ids + draw.main_item_ids + draw.reserve_item_ids
    assert len(set(everything)) == len(everything)
    assert draw.selection_seed == 20260917 and draw.benchmark == "mbpp_plus"
    other = mp.select_panels(items, seed=1, n_dev=8, n_main=100, n_reserve=8)
    assert other.main_item_ids != draw.main_item_ids


def test_select_panels_skips_ineligible_and_blocks_when_short(run_paths):
    import dataclasses
    items, _ = mp.build_inventory(run_paths)
    excluded = [dataclasses.replace(items[0], eligible=False, exclusion_reason="test")] + items[1:]
    draw = mp.select_panels(excluded, seed=3, n_dev=8, n_main=100, n_reserve=8)
    assert items[0].public.item_id not in draw.dev_item_ids + draw.main_item_ids + draw.reserve_item_ids
    with pytest.raises(ValueError):
        mp.select_panels(items[:10], seed=3, n_dev=8, n_main=100, n_reserve=8)
