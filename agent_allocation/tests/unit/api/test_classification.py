"""Required adapter behaviours (brief 01) exercised end to end: OpenRouterClient -> fake transport -> outcome."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_openrouter_transport
from agent_allocation.api.request import build_request_body
from agent_allocation.contracts import (
    CIRCUIT_BREAKING_TRANSPORT,
    RETRYABLE_TRANSPORT,
    CompletionStatus,
    TransportStatus,
    dump_record,
)
from tests.unit.api.conftest import DUMMY_KEY, MESSAGES, SAMPLING, attempt


async def run_reply(make_spec, lock, reply, *, api_key=None, read_timeout=600.0, **spec_overrides):
    script = FakeScript(default=reply)
    client = OpenRouterClient(api_key=api_key, transport=fake_openrouter_transport(script),
                              read_timeout=read_timeout)
    spec = make_spec(**spec_overrides)
    try:
        outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    finally:
        await client.aclose()
    return outcome, script


@pytest.mark.parametrize("fault, status, http_status", [
    (Fault.RATE_LIMITED, TransportStatus.RATE_LIMITED, 429),
    (Fault.AUTH_FAILED, TransportStatus.AUTH_FAILED, 401),
    (Fault.FORBIDDEN, TransportStatus.AUTH_FAILED, 403),
    (Fault.PAYMENT_REQUIRED, TransportStatus.PAYMENT_REQUIRED, 402),
    (Fault.BAD_REQUEST, TransportStatus.BAD_REQUEST, 400),
    (Fault.NOT_FOUND, TransportStatus.BAD_REQUEST, 404),
    (Fault.UNPROCESSABLE, TransportStatus.BAD_REQUEST, 422),
    (Fault.REQUEST_TIMEOUT, TransportStatus.AMBIGUOUS_TIMEOUT, 408),
    (Fault.UPSTREAM_ERROR, TransportStatus.UPSTREAM_ERROR, 502),
])
async def test_http_error_statuses(make_spec, lock, fault, status, http_status):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=fault))
    result = outcome.result
    assert result.transport_status is status and result.http_status == http_status
    assert result.completion_status is None and result.usage is None and result.generation_id is None
    assert result.error_detail.startswith(f"http {http_status}")
    assert json.loads(outcome.response_body)["error"]["code"] == http_status


async def test_upstream_status_code_is_configurable(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.UPSTREAM_ERROR, status_code=503,
                                                             headers={"Retry-After": "4"}))
    assert outcome.result.transport_status is TransportStatus.UPSTREAM_ERROR
    assert outcome.result.http_status == 503 and outcome.result.retry_after_seconds == 4.0


async def test_retry_hints_seconds_date_and_absent(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.RATE_LIMITED, retry_after="12"))
    assert outcome.result.transport_status in RETRYABLE_TRANSPORT
    assert outcome.result.retry_after_seconds == 12.0
    assert outcome.response_headers["retry-after"] == "12"
    when = (datetime.now(timezone.utc) + timedelta(seconds=120)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.RATE_LIMITED, retry_after=when))
    assert 100.0 <= outcome.result.retry_after_seconds <= 120.0
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.RATE_LIMITED, retry_after=None))
    assert outcome.result.retry_after_seconds is None


async def test_empty_length_limited_content_is_a_terminal_billed_outcome(make_spec, lock):
    outcome, script = await run_reply(make_spec, lock, FakeReply(fault=Fault.EMPTY_LENGTH))
    result = outcome.result
    assert result.transport_status is TransportStatus.OK
    assert result.completion_status is CompletionStatus.LENGTH
    assert result.visible_content == "" and result.finish_reason == "length"
    assert result.native_finish_reason == "MAX_TOKENS"
    spec = make_spec()
    assert result.usage.completion_tokens == spec.max_completion_tokens == result.usage.reasoning_tokens
    assert result.usage.actual_cost_usd > 0 and script.billed_usd == result.usage.actual_cost_usd
    assert result.transport_status not in RETRYABLE_TRANSPORT


@pytest.mark.parametrize("fault, expected, finish", [
    (Fault.EMPTY_STOP, CompletionStatus.EMPTY, "stop"),
    (Fault.BLOCKED, CompletionStatus.BLOCKED, "content_filter"),
])
async def test_empty_and_blocked_are_terminal_model_outcomes(make_spec, lock, fault, expected, finish):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=fault))
    assert outcome.result.transport_status is TransportStatus.OK
    assert outcome.result.completion_status is expected and outcome.result.finish_reason == finish


async def test_provider_mismatch_opens_the_circuit_and_keeps_usage(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.PROVIDER_MISMATCH))
    result = outcome.result
    assert result.transport_status is TransportStatus.PROVIDER_MISMATCH
    assert result.transport_status in CIRCUIT_BREAKING_TRANSPORT
    assert result.observed_provider == "Google" and "Google AI Studio" in result.error_detail
    assert result.completion_status is None
    assert result.usage.actual_cost_usd > 0 and result.generation_id  # billed: 02 must still settle it
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.MODEL_MISMATCH))
    assert outcome.result.transport_status is TransportStatus.PROVIDER_MISMATCH
    assert "preview" in outcome.result.observed_model


async def test_200_error_envelope_is_not_success(make_spec, lock):
    outcome, script = await run_reply(make_spec, lock, FakeReply(fault=Fault.ERROR_ENVELOPE))
    result = outcome.result
    assert outcome.result.http_status == 200
    assert result.transport_status is TransportStatus.ERROR_ENVELOPE
    assert result.transport_status in RETRYABLE_TRANSPORT             # contract v2 (R01-1, option b)
    assert result.transport_status not in CIRCUIT_BREAKING_TRANSPORT
    assert result.completion_status is None and result.visible_content is None
    assert "Provider returned error" in result.error_detail and "code=502" in result.error_detail
    assert script.billed_usd == 0


async def test_malformed_json_body_is_an_error_envelope(make_spec, lock):
    outcome, script = await run_reply(make_spec, lock, FakeReply(fault=Fault.MALFORMED_JSON))
    assert outcome.result.transport_status is TransportStatus.ERROR_ENVELOPE
    assert "malformed_body" in outcome.result.error_detail
    assert outcome.response_body and script.billed_usd > 0   # the raw bytes are still handed over for storage


async def test_malformed_usage_is_kept_raw_with_unknown_fields(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.MALFORMED_USAGE))
    result = outcome.result
    assert result.transport_status is TransportStatus.OK
    assert result.completion_status is CompletionStatus.COMPLETED
    usage = result.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.reasoning_tokens, usage.cached_prompt_tokens,
            usage.actual_cost_usd) == (None, None, None, None, None)
    assert usage.raw_usage["cost"] == "n/a"
    assert outcome.result.usage is not None
    missing, _ = await run_reply(make_spec, lock, FakeReply(usage=None))
    assert missing.result.usage is None and missing.result.transport_status is TransportStatus.OK


async def test_usage_overrides_zero_reasoning_is_a_real_zero(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(reasoning_tokens=0))
    assert outcome.result.usage.reasoning_tokens == 0
    outcome, _ = await run_reply(make_spec, lock, FakeReply(reasoning_tokens=None))
    assert outcome.result.usage.reasoning_tokens is None


async def test_streaming_is_disabled_everywhere(make_spec, lock):
    outcome, script = await run_reply(make_spec, lock, FakeReply())
    assert outcome.request_body["stream"] is False
    assert script.completion_calls()[0].body["stream"] is False
    assert script.violations == []
    with pytest.raises(ValueError, match="stream"):
        build_request_body(make_spec(), MESSAGES, lock, {**SAMPLING, "stream": True})


async def test_cache_hit_header_opens_the_circuit_even_with_a_good_body(make_spec, lock):
    outcome, _ = await run_reply(make_spec, lock, FakeReply(fault=Fault.CACHE_HIT))
    result = outcome.result
    assert result.transport_status is TransportStatus.CACHE_HIT
    assert result.transport_status in CIRCUIT_BREAKING_TRANSPORT
    assert result.cache_status == "HIT" and result.completion_status is None
    assert result.visible_content.startswith("<rationale>") and result.usage is not None


async def test_authorization_is_redacted_everywhere(make_spec, lock):
    leaky = FakeReply(headers={"Authorization": f"Bearer {DUMMY_KEY}", "Set-Cookie": "session=abc",
                               "X-Echo": f"key {DUMMY_KEY}"},
                      content=f"<rationale>echo {DUMMY_KEY}</rationale>\n<final>1</final>")
    script = FakeScript(default=leaky, expected_api_key=DUMMY_KEY)
    client = OpenRouterClient(api_key=DUMMY_KEY, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    denied = FakeScript(default=FakeReply(fault=Fault.AUTH_FAILED, error_message=f"bad key {DUMMY_KEY}"))
    denied_client = OpenRouterClient(api_key=DUMMY_KEY, transport=fake_openrouter_transport(denied))
    rejected = await denied_client.generate(spec, MESSAGES, attempt_id=attempt(spec, 2), lock=lock,
                                            sampling=SAMPLING)
    await client.aclose()
    await denied_client.aclose()
    assert outcome.result.transport_status is TransportStatus.OK  # the key was really sent
    assert script.calls[0].headers["authorization"] == "[REDACTED]"
    assert outcome.response_headers["authorization"] == "[REDACTED]"
    assert outcome.response_headers["set-cookie"] == "[REDACTED]"
    assert all(key == key.lower() for key in outcome.response_headers)
    for item in (outcome, rejected):
        serialized = json.dumps({"request": item.request_body, "headers": dict(item.response_headers),
                                 "body": item.response_body.decode(), "result": dump_record(item.result)})
        assert DUMMY_KEY not in serialized
    assert "[REDACTED_KEY]" in outcome.response_body.decode()
    assert DUMMY_KEY not in repr(client)
    assert rejected.result.transport_status is TransportStatus.AUTH_FAILED


async def test_identical_bodies_for_different_draws_are_distinct_attempts(make_spec, lock):
    script = FakeScript()
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    draws = [make_spec(partition_id="canary", draw_id=d) for d in (0, 1, 2)]
    outcomes = [await client.generate(s, MESSAGES, attempt_id=attempt(s), lock=lock, sampling=SAMPLING)
                for s in draws]
    repeat = await client.generate(draws[0], MESSAGES, attempt_id=attempt(draws[0], 2), lock=lock,
                                   sampling=SAMPLING)
    await client.aclose()
    bodies = {json.dumps(o.request_body, sort_keys=True) for o in outcomes + [repeat]}
    assert len(bodies) == 1                                            # prompts and bodies are identical
    assert len({s.node_id for s in draws}) == 3                        # ... but the nodes are distinct
    assert len({o.result.attempt_id for o in outcomes + [repeat]}) == 4
    assert len({o.result.generation_id for o in outcomes + [repeat]}) == 4
    assert len(script.completion_calls()) == 4                         # every attempt reached the API
    assert {c.coordinates["draw_id"] for c in script.completion_calls()} == {0, 1, 2}
    assert all(o.result.cache_status == "MISS" for o in outcomes + [repeat])


@pytest.mark.parametrize("fault, status, billed", [
    (Fault.READ_TIMEOUT, TransportStatus.AMBIGUOUS_TIMEOUT, True),
    (Fault.REMOTE_DISCONNECT, TransportStatus.AMBIGUOUS_TIMEOUT, True),
    (Fault.CONNECT_ERROR, TransportStatus.CONNECT_FAILED, False),
    (Fault.CONNECT_TIMEOUT, TransportStatus.CONNECT_FAILED, False),
])
async def test_transport_exceptions(make_spec, lock, fault, status, billed):
    outcome, script = await run_reply(make_spec, lock, FakeReply(fault=fault))
    result = outcome.result
    assert result.transport_status is status
    assert result.http_status is None and result.completion_status is None and result.usage is None
    assert outcome.response_body == b"" and outcome.response_headers == {}
    assert outcome.request_body["model"] == lock.model
    assert result.error_detail.startswith(status.value)
    assert (script.billed_usd > 0) is billed   # an ambiguous timeout may really have been charged


async def test_server_delay_beyond_read_timeout_is_ambiguous(make_spec, lock):
    outcome, script = await run_reply(make_spec, lock, FakeReply(delay_seconds=5.0), read_timeout=0.01)
    assert outcome.result.transport_status is TransportStatus.AMBIGUOUS_TIMEOUT
    assert script.billed_usd > 0 and script.calls[0].outcome == "read_timeout"


async def test_cancellation_propagates(make_spec, lock):
    script = FakeScript(default=FakeReply(delay_seconds=5.0))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    task = asyncio.create_task(client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock,
                                               sampling=SAMPLING))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()


async def test_client_does_not_serialize_concurrent_requests(make_spec, lock):
    script = FakeScript(default=FakeReply(delay_seconds=0.2))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    specs = [make_spec(root_slot=slot) for slot in range(16)]
    started = time.monotonic()
    outcomes = await asyncio.gather(*(client.generate(s, MESSAGES, attempt_id=attempt(s), lock=lock,
                                                      sampling=SAMPLING) for s in specs))
    elapsed = time.monotonic() - started
    await client.aclose()
    assert all(o.result.transport_status is TransportStatus.OK for o in outcomes)
    assert elapsed < 1.6  # serial would take 3.2 s


async def test_total_billing_matches_settled_costs(make_spec, lock):
    script = FakeScript()
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    total = Decimal("0")
    for slot in range(5):
        spec = make_spec(root_slot=slot)
        outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
        total += outcome.result.usage.actual_cost_usd
    key = await client.key_info()
    await client.aclose()
    assert total == script.billed_usd == key["usage"]
