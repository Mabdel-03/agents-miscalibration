from __future__ import annotations

import dataclasses
import json

import pytest

from agent_allocation.api.canary import CANARY_REPORT_SCHEMA, canary_report
from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.fake import OMIT, FakeReply, FakeScript, Fault, fake_openrouter_transport
from tests.unit.api.conftest import MESSAGES, SAMPLING, attempt


async def collect(make_spec, lock, script, plan, *, with_metadata=True):
    """plan: list of (reasoning_budget, root_slot). Returns canary rows."""
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    rows = []
    for budget, slot in plan:
        spec = make_spec(partition_id="canary", reasoning_budget=budget, root_slot=slot)
        outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
        metadata = None
        if with_metadata and outcome.result.generation_id:
            metadata = await client.generation_metadata(outcome.result.generation_id)
        rows.append((spec, outcome.result, metadata))
    await client.aclose()
    return rows


def half_budget(call):
    return FakeReply(reasoning_tokens=call.coordinates["reasoning_budget"] // 2)


async def test_report_per_budget_distributions_and_identity(make_spec, lock):
    script = FakeScript(hook=half_budget)
    plan = [(budget, slot) for budget in (512, 1024, 4096) for slot in range(3)]
    report = canary_report(await collect(make_spec, lock, script, plan))
    json.dumps(report)  # evidence files must be plain JSON
    assert report["schema"] == CANARY_REPORT_SCHEMA and report["row_count"] == 9
    assert [b["requested_reasoning_budget"] for b in report["budgets"]] == [512, 1024, 4096]
    a1024 = report["budgets"][1]
    assert a1024["count"] == 3 and a1024["transport_statuses"] == {"ok": 3}
    assert a1024["reasoning_tokens"] == {"n": 3, "n_none": 0, "share_none": 0.0, "min": 512, "median": 512,
                                         "max": 512}
    assert a1024["reasoning_budget_ratio"]["median"] == 0.5
    assert a1024["max_completion_tokens"] == [2048]
    assert 0 < a1024["headroom_utilization"]["max"] < 1
    assert a1024["finish_reasons"] == {"stop": 3} and a1024["content_empty_share"] == 0.0
    assert a1024["observed_providers"] == {"Google AI Studio": 3}
    assert a1024["cache_statuses"] == {"MISS": 3}
    assert a1024["metadata"]["service_tiers"] == {"standard": 3}
    assert a1024["metadata"]["provider_names"] == {"Google AI Studio": 3}
    assert a1024["metadata"]["reasoning_usage_disagreements"] == 0
    assert a1024["metadata"]["cost_disagreements"] == 0
    assert a1024["cost_usd"]["unknown_count"] == 0 and float(a1024["cost_usd"]["total_known"]) > 0
    assert report["cross_budget"] == {"median_reasoning_tokens_by_budget": {"512": 256, "1024": 512, "4096": 2048},
                                      "median_nondecreasing": True, "distinct_medians": 3}
    assert report["flags"] == []
    assert len(report["rows"]) == 9 and report["rows"][0]["metadata_service_tier"] == "standard"


async def test_flags_surface_blockers_without_deciding(make_spec, lock):
    script = FakeScript(hook=lambda call: FakeReply(reasoning_tokens=OMIT) if call.coordinates["root_slot"] == 0
                        else None, service_tier="flex")
    script.queue.extend([FakeReply(fault=Fault.CACHE_HIT), FakeReply(fault=Fault.EMPTY_LENGTH)])
    rows = await collect(make_spec, lock, script, [(512, 1), (512, 2), (1024, 0), (1024, 3)])
    report = canary_report(rows)
    flags = set(report["flags"])
    assert "A=512:transport_cache_hit=1" in flags and "overall:cache_hit_observed" in flags
    assert "A=1024:reasoning_tokens_unaudited" not in flags   # one of two rows still reports reasoning
    assert "A=1024:reasoning_tokens_partially_missing=1" in flags
    assert "A=512:empty_content=1" in flags and "A=512:length_finishes=1" in flags
    assert "overall:service_tier_not_standard:flex" in flags
    a512 = report["budgets"][0]
    assert a512["transport_ok_count"] == 1 and a512["length_share"] == 1.0
    assert a512["at_completion_cap_count"] == 1 and a512["reasoning_over_budget_count"] == 1


async def test_unaudited_reasoning_missing_metadata_and_ignored_budgets(make_spec, lock):
    script = FakeScript(default=FakeReply(reasoning_tokens=OMIT, reasoning=OMIT))
    rows = await collect(make_spec, lock, script, [(512, 0), (4096, 0)], with_metadata=False)
    flags = set(canary_report(rows)["flags"])
    assert {"A=512:reasoning_tokens_unaudited", "A=4096:reasoning_tokens_unaudited",
            "overall:generation_metadata_missing=2"} <= flags
    same = FakeScript(default=FakeReply(reasoning_tokens=300))
    report = canary_report(await collect(make_spec, lock, same, [(512, 0), (1024, 0), (4096, 0)]))
    assert "cross_budget:reasoning_budget_possibly_ignored" in report["flags"]
    assert report["cross_budget"]["distinct_medians"] == 1


async def test_disagreements_with_metadata_are_counted(make_spec, lock):
    script = FakeScript()
    rows = await collect(make_spec, lock, script, [(1024, 0)])
    spec, result, metadata = rows[0]
    edited = dict(metadata, native_tokens_reasoning=result.usage.reasoning_tokens + 1, provider_name="Google")
    report = canary_report([(spec, result, edited)])
    assert report["overall"]["metadata"]["reasoning_usage_disagreements"] == 1
    assert "overall:reasoning_usage_disagrees_with_metadata=1" in report["flags"]
    two = canary_report([(spec, result, edited), (spec, dataclasses.replace(result, attempt_id=attempt(spec, 2)),
                                                  metadata)])
    assert "overall:multiple_metadata_providers" in two["flags"]


def test_empty_rows_and_mismatched_pairs(make_spec):
    report = canary_report([])
    assert report["row_count"] == 0 and report["budgets"] == [] and report["cross_budget"]["median_nondecreasing"] \
        is None
    json.dumps(report)


async def test_rows_must_pair_results_with_their_specs(make_spec, lock):
    rows = await collect(make_spec, lock, FakeScript(), [(1024, 0)])
    spec, result, metadata = rows[0]
    with pytest.raises(ValueError, match="does not belong"):
        canary_report([(make_spec(root_slot=5), result, metadata)])
