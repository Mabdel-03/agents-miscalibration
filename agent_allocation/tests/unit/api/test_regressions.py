"""Regressions for defects found by an independent review of the adapter and the fake (2026-09-17)."""
from __future__ import annotations

import asyncio
import json
import ssl
from decimal import Decimal

import anyio
import httpx
import pytest

from agent_allocation.api.client import OpenRouterAccountError, OpenRouterClient
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_openrouter_transport
from agent_allocation.api.normalize import parse_completion, parse_retry_after
from agent_allocation.api.reconcile import DISCREPANT, reconcile_generation
from agent_allocation.contracts import RETRYABLE_TRANSPORT, TransportStatus
from tests.unit.api.conftest import MESSAGES, SAMPLING, attempt


# 1. Low-level errors that httpcore does not wrap must classify, never crash the dispatcher.
@pytest.mark.parametrize("exc", [ssl.SSLError(1, "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] bad record mac"),
                                 ConnectionResetError(104, "Connection reset by peer"),
                                 anyio.EndOfStream(), anyio.BrokenResourceError()])
async def test_unwrapped_transport_errors_are_ambiguous(make_spec, lock, exc):
    def handler(request):
        raise exc

    client = OpenRouterClient(api_key=None, transport=httpx.MockTransport(handler))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    assert outcome.result.transport_status is TransportStatus.AMBIGUOUS_TIMEOUT
    assert type(exc).__name__ in outcome.result.error_detail
    with pytest.raises(OpenRouterAccountError) as info:
        await client.credits()
    assert info.value.transport_status is TransportStatus.AMBIGUOUS_TIMEOUT
    await client.aclose()


# 2. A client cancellation does not cancel the provider: the fake still records and bills the generation.
async def test_cancelled_in_flight_request_is_recorded_and_billed(make_spec, lock):
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
    (call,) = script.completion_calls()
    assert call.outcome == "cancelled" and call.billed_usd > 0 and call.generation_id in script.generations
    assert script.billed_usd == call.billed_usd and script.in_flight == 0


async def test_cancelled_error_reply_is_recorded_but_not_billed(make_spec, lock):
    script = FakeScript(default=FakeReply(fault=Fault.RATE_LIMITED, delay_seconds=5.0))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    task = asyncio.create_task(client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock,
                                               sampling=SAMPLING))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    (call,) = script.completion_calls()
    assert call.outcome == "cancelled" and call.billed_usd == 0 and script.generations == {}


# 3. Refusals before scripting (capacity, account limits) never consume a scripted reply.
async def test_capacity_rejections_do_not_consume_scripted_replies(make_spec, lock):
    blocker = make_spec(root_slot=9)
    target = make_spec(root_slot=0)
    script = FakeScript(max_concurrent_requests=1)
    script.by_node[blocker.node_id] = FakeReply(delay_seconds=0.3)
    script.by_node[target.node_id] = [FakeReply(fault=Fault.UPSTREAM_ERROR), FakeReply()]
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    slow = asyncio.create_task(client.generate(blocker, MESSAGES, attempt_id=attempt(blocker), lock=lock,
                                               sampling=SAMPLING))
    await asyncio.sleep(0.05)
    rejected = await client.generate(target, MESSAGES, attempt_id=attempt(target, 1), lock=lock, sampling=SAMPLING)
    await slow
    second = await client.generate(target, MESSAGES, attempt_id=attempt(target, 2), lock=lock, sampling=SAMPLING)
    third = await client.generate(target, MESSAGES, attempt_id=attempt(target, 3), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert rejected.result.transport_status is TransportStatus.RATE_LIMITED
    assert second.result.transport_status is TransportStatus.UPSTREAM_ERROR   # the scripted error still fires
    assert third.result.transport_status is TransportStatus.OK
    calls = [c for c in script.completion_calls() if c.coordinates["node_id"] == target.node_id]
    assert [(c.node_call_no, c.served_no, c.outcome) for c in calls] == [
        (1, 0, "capacity_rate_limited"), (2, 1, "response"), (3, 2, "response")]
    assert calls[0].fault is None


async def test_queue_survives_capacity_rejection(make_spec, lock):
    script = FakeScript(max_concurrent_requests=0, queue=[FakeReply(fault=Fault.EMPTY_STOP)])
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert outcome.result.transport_status is TransportStatus.RATE_LIMITED and len(script.queue) == 1


# 4. A scripted non-2xx status without a fault is never billed or registered.
async def test_scripted_error_status_without_fault_is_not_billed(make_spec, lock):
    script = FakeScript(default=FakeReply(status_code=503))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert outcome.result.transport_status is TransportStatus.UPSTREAM_ERROR
    assert script.billed_usd == 0 and script.generations == {}
    assert script.completion_calls()[0].generation_id is None


# 5. Key limits and account credits are enforced like the real backstop.
@pytest.mark.parametrize("kwargs, status, http_status", [
    (dict(prior_usage_usd=Decimal("135")), TransportStatus.AUTH_FAILED, 403),
    (dict(prior_usage_usd=Decimal("150"), key_limit_usd=None), TransportStatus.PAYMENT_REQUIRED, 402),
])
async def test_account_limits_refuse_requests(make_spec, lock, kwargs, status, http_status):
    script = FakeScript(**kwargs)
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert outcome.result.transport_status is status and outcome.result.http_status == http_status
    assert script.billed_usd == 0 and script.completion_calls()[0].outcome == "account_limit"


async def test_spending_reaches_the_key_limit_then_refuses(make_spec, lock):
    script = FakeScript(key_limit_usd=Decimal("0.0003"))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    statuses = []
    for slot in range(4):
        spec = make_spec(root_slot=slot)
        statuses.append((await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock,
                                                sampling=SAMPLING)).result.transport_status)
    unlimited = FakeScript(prior_usage_usd=Decimal("500"), enforce_account_limits=False)
    free = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(unlimited))
    spec = make_spec()
    ok = await free.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    await free.aclose()
    assert statuses[0] is TransportStatus.OK and statuses[-1] is TransportStatus.AUTH_FAILED
    assert ok.result.transport_status is TransportStatus.OK


# 6/7. Error bodies are ERROR_ENVELOPE (retryable) even with slug-form identity or a bare finish_reason "error".
def test_error_body_identity_strings_do_not_open_the_circuit(lock):
    body = json.dumps({"id": "cmpl-abc123", "object": "chat.completion", "model": "gemini-2.5-flash-lite",
                       "provider": "google-ai-studio",
                       "error": {"code": "server_error", "message": "Provider disconnected"},
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                    "finish_reason": "error"}]}).encode()
    parsed = parse_completion(body, "MISS", lock)
    assert parsed.transport_status is TransportStatus.ERROR_ENVELOPE
    assert parsed.transport_status in RETRYABLE_TRANSPORT
    assert "identity in error body" in parsed.error_detail and "Provider disconnected" in parsed.error_detail


@pytest.mark.parametrize("content", ["", "partial text"])
def test_finish_reason_error_without_error_object_is_an_envelope(lock, content):
    body = json.dumps({"id": "gen-1", "provider": lock.provider_name, "model": lock.model,
                       "choices": [{"finish_reason": "error", "message": {"content": content}}]}).encode()
    parsed = parse_completion(body, "MISS", lock)
    assert parsed.transport_status is TransportStatus.ERROR_ENVELOPE and parsed.completion_status is None


# 8. Absurd Retry-After dates never crash classification.
@pytest.mark.parametrize("value", ["Thu, 17 Sep 2026 07:00:00 +99999999999999999999", "Thu, 99 Foo 99999999 99:99:99"])
def test_retry_after_overflow_is_unknown(value):
    assert parse_retry_after(value) is None


# 9. An injected cache HIT is visible in generation metadata, so reconciliation flags it.
async def test_cache_hit_metadata_names_a_cache_source(make_spec, lock):
    script = FakeScript(default=FakeReply(fault=Fault.CACHE_HIT))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    metadata = await client.generation_metadata(outcome.result.generation_id)
    await client.aclose()
    reconciliation = reconcile_generation(outcome.result, metadata, lock)
    assert reconciliation.state == DISCREPANT and any("response cache" in p for p in reconciliation.problems)


# 10. Connect failures are attempts, but never requests a provider could have billed.
async def test_connect_failures_did_not_reach_the_server(make_spec, lock):
    spec = make_spec()
    script = FakeScript()
    script.by_node[spec.node_id] = [FakeReply(fault=Fault.CONNECT_ERROR), FakeReply()]
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    first = await client.generate(spec, MESSAGES, attempt_id=attempt(spec, 1), lock=lock, sampling=SAMPLING)
    second = await client.generate(spec, MESSAGES, attempt_id=attempt(spec, 2), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert first.result.transport_status is TransportStatus.CONNECT_FAILED
    assert second.result.transport_status is TransportStatus.OK
    assert len(script.completion_calls()) == 2
    assert len(script.completion_calls(reached_server=True)) == 1
    assert [c.outcome for c in script.completion_calls(reached_server=False)] == ["connect_error"]
    assert script.in_flight == 0 and script.peak_in_flight == 1


# Gateway timeouts stay UPSTREAM_ERROR by rule but say they may have been billed.
@pytest.mark.parametrize("status", [504, 524])
async def test_gateway_timeouts_are_marked_possibly_billed(make_spec, lock, status):
    script = FakeScript(default=FakeReply(fault=Fault.UPSTREAM_ERROR, status_code=status))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert outcome.result.transport_status is TransportStatus.UPSTREAM_ERROR
    assert outcome.result.error_detail.startswith("gateway_timeout_may_be_billed")
