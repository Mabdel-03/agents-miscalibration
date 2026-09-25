"""WP-G P7: the offline E0 gate subset over the read-only forced_v2 run root."""
from __future__ import annotations

import pytest

from agent_allocation.passk import e0_gate
from tests.unit.passk.wpg_fixtures import LEGACY_ROOT, fixture_splits, legacy_available


def test_classify_collisions_allows_only_context_free_banks():
    specs = {"a": {"item_id": "x", "kind": "root", "parent_ids": []},
             "b": {"item_id": "x", "kind": "root", "parent_ids": []},
             "c": {"item_id": "x", "kind": "private", "parent_ids": ["a"]},
             "d": {"item_id": "x", "kind": "private", "parent_ids": ["c"]},
             "e": {"item_id": "y", "kind": "root", "parent_ids": []}}
    unexpected, by_design = e0_gate.classify_collisions({"h1": {"a", "b"}, "h2": {"c", "d"}, "h3": {"a", "e"},
                                                          "h4": {"a"}}, specs)
    assert by_design == 1
    assert sorted(tuple(g["nodes"]) for g in unexpected) == [("a", "e"), ("c", "d")]


def test_split_overlap_check():
    splits, _tasks = fixture_splits(use_legacy=False)
    assert e0_gate.check_splits(splits)["passed"] is True
    bad = {**splits, "panels": {**splits["panels"], "T200": [*splits["panels"]["T200"], splits["panels"]["L96"][0]]}}
    assert e0_gate.check_splits(bad)["passed"] is False
    assert e0_gate.check_splits(None)["passed"] is None


@pytest.mark.skipif(not legacy_available(), reason="legacy run root not readable")
@pytest.mark.timeout(900)
def test_e0_gate_passes_on_the_legacy_run_root():
    splits, _tasks = fixture_splits(use_legacy=True)
    result = e0_gate.e0_gate(LEGACY_ROOT, splits=splits)
    failing = {k: v for k, v in result["checks"].items() if v["passed"] is False}
    assert result["passed"], failing
    checks = result["checks"]
    assert checks["generation_metadata"]["records"] == 1054
    assert checks["observed_model_provider"]["parsed"] >= 14_000
    collisions = checks["request_body_collisions"]["unexpected_groups"]
    assert sorted(i for g in collisions for i in g["items"]) == ["hmmt/2026/24", "omni_math_hard/3907"]
    assert all(g["kinds"] == ["private"] for g in collisions)
