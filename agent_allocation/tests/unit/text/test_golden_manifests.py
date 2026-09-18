"""Golden hashes of the pinned text inventories and their forced27b panel draws (24 main, 8 dev, 8 reserve).

A failure here means preprocessing, exclusions, private records or the panel procedure changed. That is only
legitimate before the dataset lock is written and must come with a new preprocessing/procedure version string and a
note to the lead; update the hashes deliberately, never to make a run pass. The draw hash also depends on
``contracts.STUDY_ID`` (it salts the panel ranks), so a study-id change moves every draw hash.
"""
from __future__ import annotations

import pytest

from agent_allocation.benchmarks.text_manifest import select_panels
from agent_allocation.contracts import canonical_hash, dump_record, public_task_hash

GOLDEN = {
    "aime": ("51d8e62d18e043a4a2ce9e84cacd4691339cc55730aa8875e85cf0aac8a87670",
             "87cad34ce42a18ea73fae4e31337d33c02bc9e68e363af9c0115bc0be9dc8fa2"),
    "hmmt": ("617321aee55d093eff11de52e2b007baa464abaf996e0b2c4c584901b62ce871",
             "81d4e73b75d99ac6dd411e554e69d6c76f0191bc272ecc736b241c96dbc7dfaf"),
    "omni_math_hard": ("48917802bf6cc5459611cb912e86356ed2ab489f79e7201a28753e7ad19a388c",
                       "8095e5b0a8e8f67e022fe7b51517b7d3ac65dc8da1a51e08097f0b11d1d70257"),
    "gpqa_diamond": ("66392f369dd543257905579b36133de08c48d02ec33f4656a918f6e06bc8d3e3",
                     "5e9e43ae0deef32e6d7d4105e712456a222152e1042d01d4a44195494daad346"),
}


def _inventory_hash(inventory) -> str:
    return canonical_hash([[item.public.item_id, public_task_hash(item.public), item.private.private_hash,
                            item.eligible, item.exclusion_reason] for item in inventory])


@pytest.mark.parametrize("benchmark, fixture", [
    ("aime", "aime_inventory"), ("hmmt", "hmmt_inventory"), ("omni_math_hard", "omni_inventory"),
    ("gpqa_diamond", "gpqa_inventory"),
])
def test_inventory_and_forced27b_draw_match_golden(benchmark, fixture, cfg, request):
    inventory = request.getfixturevalue(fixture)
    inventory_hash, draw_hash = GOLDEN[benchmark]
    assert _inventory_hash(inventory) == inventory_hash, f"{benchmark} inventory changed"
    per_benchmark_main = cfg["panels"]["main_items"] // 4
    draw = select_panels(inventory, benchmark=benchmark, seed=cfg["seed"], n_dev=8, n_main=per_benchmark_main,
                         n_reserve=8)
    assert canonical_hash(dump_record(draw)) == draw_hash, f"{benchmark} panel draw changed"
