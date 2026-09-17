from __future__ import annotations

import json
from decimal import Decimal

import pytest

from agent_allocation.api.client import OpenRouterAccountError, OpenRouterClient
from agent_allocation.api.fake import OMIT, FakeReply, FakeScript, Fault, fake_openrouter_transport
from agent_allocation.contracts import (
    CompletionStatus,
    GenerationOutcome,
    GenerationResult,
    TransportStatus,
    dump_record,
    load_record,
)
from tests.unit.api.conftest import DUMMY_KEY, MESSAGES, SAMPLING, attempt


def make_client(script: FakeScript, **kwargs) -> OpenRouterClient:
    return OpenRouterClient(api_key=kwargs.pop("api_key", None), transport=fake_openrouter_transport(script),
                            **kwargs)


async def generate_one(make_spec, lock, reply: FakeReply | None = None, **spec_overrides) -> GenerationOutcome:
    script = FakeScript(default=reply or FakeReply())
    client = make_client(script)
    spec = make_spec(**spec_overrides)
    try:
        return await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    finally:
        await client.aclose()


async def test_success_with_reasoning_fields(make_spec, lock):
    outcome = await generate_one(make_spec, lock)
    result = outcome.result
    assert result.transport_status is TransportStatus.OK
    assert result.completion_status is CompletionStatus.COMPLETED
    assert result.reasoning_returned is True
    assert result.generation_id and result.observed_provider == "Google AI Studio"
    assert result.observed_model == lock.model
    assert result.finish_reason == "stop" and result.native_finish_reason == "STOP"
    usage = result.usage
    assert usage.reasoning_tokens is not None and usage.completion_tokens >= usage.reasoning_tokens
    assert isinstance(usage.actual_cost_usd, Decimal) and usage.actual_cost_usd > 0
    assert usage.cached_prompt_tokens == 0 and usage.raw_usage["cost"] == float(usage.actual_cost_usd)
    assert result.raw_response_path is None and result.raw_response_sha256 is None
    assert result.http_status == 200 and result.latency_seconds >= 0
    assert result.cache_status == "MISS"
    assert outcome.request_body["stream"] is False
    assert json.loads(outcome.response_body)["id"] == result.generation_id
    # the result survives the one codec unchanged
    assert load_record(GenerationResult, json.loads(json.dumps(dump_record(result)))) == result


async def test_success_without_reasoning_fields_keeps_unknowns_none(make_spec, lock):
    reply = FakeReply(reasoning=OMIT, reasoning_details=OMIT, reasoning_tokens=OMIT)
    outcome = await generate_one(make_spec, lock, reply)
    result = outcome.result
    assert result.transport_status is TransportStatus.OK
    assert result.completion_status is CompletionStatus.COMPLETED
    assert result.reasoning_returned is False
    assert result.usage.reasoning_tokens is None          # absent is unknown, never 0
    assert result.usage.completion_tokens is not None
    message = json.loads(outcome.response_body)["choices"][0]["message"]
    assert "reasoning" not in message and "reasoning_details" not in message


async def test_one_persistent_http_client_per_instance(make_spec, lock):
    script = FakeScript()
    client = make_client(script)
    inner = client._client
    for slot in range(3):
        spec = make_spec(root_slot=slot)
        await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    assert client._client is inner and not inner.is_closed
    await client.aclose()
    await client.aclose()  # idempotent
    assert inner.is_closed


async def test_attempt_id_must_belong_to_the_node(make_spec, lock):
    client = make_client(FakeScript())
    spec = make_spec()
    for bad in ("1", f"{spec.node_id}:", f"{make_spec(root_slot=9).node_id}:1"):
        with pytest.raises(ValueError, match="attempt_id"):
            await client.generate(spec, MESSAGES, attempt_id=bad, lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert client  # nothing was sent


async def test_key_info_drops_label_and_credits_are_decimal():
    script = FakeScript(expected_api_key=DUMMY_KEY)
    client = make_client(script, api_key=DUMMY_KEY)
    info = await client.key_info()
    credits = await client.credits()
    await client.aclose()
    assert "label" not in info and info["limit"] == Decimal("135.0")
    assert isinstance(credits["total_credits"], Decimal)
    assert all(DUMMY_KEY not in json.dumps(c.headers) for c in script.calls)


async def test_generation_metadata_is_none_until_ready(make_spec, lock):
    script = FakeScript(metadata_lag_polls=2)
    client = make_client(script)
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    generation_id = outcome.result.generation_id
    assert await client.generation_metadata(generation_id) is None
    assert await client.generation_metadata(generation_id) is None
    metadata = await client.generation_metadata(generation_id)
    await client.aclose()
    assert metadata["provider_name"] == "Google AI Studio" and metadata["service_tier"] == "standard"
    assert metadata["total_cost"] == outcome.result.usage.actual_cost_usd
    assert metadata["native_tokens_reasoning"] == outcome.result.usage.reasoning_tokens


async def test_account_errors_raise_typed_errors():
    client = make_client(FakeScript(expected_api_key="sk-or-v1-somethingelse0000"), api_key=DUMMY_KEY)
    with pytest.raises(OpenRouterAccountError) as info:
        await client.key_info()
    assert info.value.transport_status is TransportStatus.AUTH_FAILED and info.value.http_status == 401
    assert DUMMY_KEY not in str(info.value)
    await client.aclose()
    client = make_client(FakeScript())
    with pytest.raises(OpenRouterAccountError) as info:
        await client.list_endpoints("google/not-a-model")
    assert info.value.transport_status is TransportStatus.BAD_REQUEST and info.value.http_status == 404
    with pytest.raises(ValueError):
        await client.list_endpoints("../../key")
    await client.aclose()


async def test_list_endpoints_returns_raw_bytes_and_parsed():
    raw_bytes = b'{"data":{"id":"google/gemini-2.5-flash-lite","endpoints":[]}}'
    client = make_client(FakeScript(endpoints={"google/gemini-2.5-flash-lite": raw_bytes}))
    raw, parsed = await client.list_endpoints("google/gemini-2.5-flash-lite")
    await client.aclose()
    assert raw == raw_bytes and parsed["data"]["endpoints"] == []


async def test_live_construction_has_zero_transport_retries_and_study_timeouts():
    client = OpenRouterClient(api_key=DUMMY_KEY, connect_timeout=15.0, read_timeout=600.0)  # no request is made
    inner = client._client
    transport = inner._transport
    assert transport._pool._retries == 0
    assert inner.timeout.connect == 15.0 and inner.timeout.read == 600.0
    assert inner.follow_redirects is False
    assert DUMMY_KEY not in repr(client) and "authorization" not in {k.lower() for k in inner.headers}
    await client.aclose()
