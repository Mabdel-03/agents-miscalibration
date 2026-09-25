"""WP-G P6: the new THINK body equals the legacy raw request_body.think (bar max_tokens) on every L96/D32 item."""
from __future__ import annotations

import copy

import pytest

from agent_allocation.config import load_config
from agent_allocation.passk.gates import slices_match, think_parity
from tests.forced_cfg import PROJECT
from tests.unit.passk.wpg_fixtures import LEGACY_ROOT, legacy_available, legacy_panels, legacy_tasks

pytestmark = pytest.mark.skipif(not legacy_available(), reason="legacy run root not readable")


def _cfg():
    return load_config(PROJECT / "configs" / "passk_cont_v1.json")


def test_think_body_parity_all_l96_and_d32():
    panels = legacy_panels()
    items = [i for b in panels for i in panels[b]["main_item_ids"] + panels[b]["dev_item_ids"]]
    assert len(items) == 128
    result = think_parity(_cfg(), LEGACY_ROOT, items, legacy_tasks())
    assert result["passed"], (result["mismatches"][:3], result["missing"][:3])
    assert result["checked"] == 128
    assert result["provider_only"] == {"mancer": ["mancer/fp8"], "parasail": ["parasail/fp8"]}


def test_parity_detects_a_changed_decoding_parameter():
    cfg = copy.deepcopy(_cfg())
    cfg["sampling"]["top_k"] = 21
    item = legacy_panels()["aime"]["main_item_ids"][0]
    result = think_parity(cfg, LEGACY_ROOT, [item], legacy_tasks())
    assert not result["passed"] and result["mismatches"][0]["differing_keys"] == ["top_k"]


def test_parity_detects_a_wrong_provider_pin():
    import json
    item = legacy_panels()["aime"]["main_item_ids"][0]
    legacy_shard = json.loads((LEGACY_ROOT / "shards" / "assignment.json").read_text())["assignment"][item]
    other = "mancer" if legacy_shard == "parasail" else "parasail"
    # the item's legacy raw (its own shard's pin) against a body built on the other shard's lock
    result = think_parity(_cfg(), LEGACY_ROOT, [item], legacy_tasks(), assignment={item: legacy_shard},
                          locks={legacy_shard: _other_lock(other)})
    assert not result["passed"]


def _other_lock(shard):
    import json
    from agent_allocation.contracts import EndpointLock, load_record
    data = json.loads((LEGACY_ROOT / "shards" / shard / "locks" / "openrouter_endpoints.json").read_text())
    return load_record(EndpointLock, data["locks"]["qwen/qwen3.8-27b"])


def test_budget_slices_match_the_config():
    out = slices_match(_cfg())
    assert out["passed"]
    assert out["slices"] == {"mancer": {"credit_ceiling_usd": "87.00", "soft_admission_ceiling_usd": "82.00"},
                             "parasail": {"credit_ceiling_usd": "63.00", "soft_admission_ceiling_usd": "60.00"}}
