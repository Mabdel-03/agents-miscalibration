from __future__ import annotations

import dataclasses

import pytest

from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_openrouter_transport
from agent_allocation.api.reconcile import (
    CONSISTENT,
    DISCREPANT,
    PENDING,
    UNRECONCILABLE,
    metadata_identity_problems,
    reconcile_generation,
)
from tests.unit.api.conftest import MESSAGES, SAMPLING, attempt


async def one(make_spec, lock, script):
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    return client, outcome.result


async def test_consistent_after_metadata_lag(make_spec, lock):
    client, result = await one(make_spec, lock, FakeScript(metadata_lag_polls=1))
    first = reconcile_generation(result, await client.generation_metadata(result.generation_id), lock)
    assert first.state == PENDING and first.metadata_cost_usd is None
    second = reconcile_generation(result, await client.generation_metadata(result.generation_id), lock)
    await client.aclose()
    assert second.state == CONSISTENT and second.problems == ()
    assert second.metadata_cost_usd == second.result_cost_usd == result.usage.actual_cost_usd
    assert second.native_reasoning_tokens == result.usage.reasoning_tokens


async def test_tier_drift_is_visible_only_in_metadata(make_spec, lock):
    client, result = await one(make_spec, lock, FakeScript(service_tier="flex"))
    metadata = await client.generation_metadata(result.generation_id)
    await client.aclose()
    assert result.observed_provider == lock.provider_name  # the response itself looks fine
    reconciliation = reconcile_generation(result, metadata, lock)
    assert reconciliation.state == DISCREPANT
    assert any("service_tier 'flex'" in p for p in reconciliation.problems)


async def test_ambiguous_timeouts_cannot_be_reconciled_by_id(make_spec, lock):
    client, result = await one(make_spec, lock, FakeScript(default=FakeReply(fault=Fault.READ_TIMEOUT)))
    await client.aclose()
    reconciliation = reconcile_generation(result, None, lock)
    assert reconciliation.state == UNRECONCILABLE and reconciliation.generation_id is None


async def test_cost_and_identity_discrepancies(make_spec, lock):
    client, result = await one(make_spec, lock, FakeScript())
    metadata = dict(await client.generation_metadata(result.generation_id))
    await client.aclose()
    doubled = dict(metadata, total_cost=metadata["total_cost"] * 2, response_cache_source_id="gen-older",
                   model="google/gemini-2.5-flash", provider_name=None, service_tier=None)
    problems = reconcile_generation(result, doubled, lock).problems
    assert any("charged" in p for p in problems)
    assert any("response cache" in p for p in problems)
    assert any("metadata model" in p for p in problems)
    assert "metadata provider_name unknown" in problems and "metadata service_tier unknown" in problems
    assert metadata_identity_problems(metadata, lock) == []
    with pytest.raises(ValueError):
        reconcile_generation(dataclasses.replace(result, generation_id="gen-other"), metadata, lock)
