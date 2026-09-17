"""Golden hashes of the pinned text inventories and their balanced150 panel draws.

A failure here means preprocessing, exclusions, private records or the panel procedure changed. That is only
legitimate before the dataset lock is written and must come with a new preprocessing/procedure version string and a
note to the lead; update the hashes deliberately, never to make a run pass.
"""
from __future__ import annotations

import pytest

from agent_allocation.benchmarks.text_manifest import select_panels
from agent_allocation.contracts import canonical_hash, dump_record, public_task_hash

GOLDEN = {
    "math500": ("b6c9f3ec34a05e612bd426916bcb7fa18653b42441d4a731c317f11389091446",
                "f8fe0c9d48b96812b3f938978627284cbf7800939604478270a810e9df1cc53b"),
    "mmlu_pro": ("c44c4bebbe7e5679c738e98ba04cc3cdb5f0cb99dcd6d3e46a1e1346d1e0ec1a",
                 "29a797869f0845d334a780c289b9852711944f7d31589632267c1d4f462c2fe1"),
    "bbh_logic4": ("7549866470136554a4e42b041d22ea8fcaf7a476e7eae61f054de87ebc410c05",
                   "ddd2991f0e9e2df586eb5813d6f24179d098d102cb242653e1c06ba4f45b76a2"),
}


def _inventory_hash(inventory) -> str:
    return canonical_hash([[item.public.item_id, public_task_hash(item.public), item.private.private_hash,
                            item.eligible, item.exclusion_reason] for item in inventory])


@pytest.mark.parametrize("benchmark, fixture", [
    ("math500", "math_inventory"), ("mmlu_pro", "mmlu_inventory"), ("bbh_logic4", "bbh_inventory"),
])
def test_inventory_and_balanced150_draw_match_golden(benchmark, fixture, cfg, request):
    inventory = request.getfixturevalue(fixture)
    inventory_hash, draw_hash = GOLDEN[benchmark]
    assert _inventory_hash(inventory) == inventory_hash, f"{benchmark} inventory changed"
    draw = select_panels(inventory, benchmark=benchmark, seed=cfg["seed"], n_dev=8, n_main=100, n_reserve=8)
    assert canonical_hash(dump_record(draw)) == draw_hash, f"{benchmark} panel draw changed"
