from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from agent_allocation.api.fake import fake_endpoint_lock
from agent_allocation.api.normalize import (
    classify_exception,
    classify_http_status,
    completion_status,
    normalize_usage,
    parse_completion,
    parse_money,
    parse_retry_after,
)
from agent_allocation.contracts import CompletionStatus, TransportStatus

LOCK = fake_endpoint_lock()


@pytest.mark.parametrize("status, expected", [
    (200, None), (201, None),
    (429, TransportStatus.RATE_LIMITED),
    (401, TransportStatus.AUTH_FAILED), (403, TransportStatus.AUTH_FAILED),
    (402, TransportStatus.PAYMENT_REQUIRED),
    (400, TransportStatus.BAD_REQUEST), (404, TransportStatus.BAD_REQUEST), (422, TransportStatus.BAD_REQUEST),
    (413, TransportStatus.BAD_REQUEST), (302, TransportStatus.BAD_REQUEST),
    (408, TransportStatus.AMBIGUOUS_TIMEOUT),
    (500, TransportStatus.UPSTREAM_ERROR), (502, TransportStatus.UPSTREAM_ERROR),
    (503, TransportStatus.UPSTREAM_ERROR), (524, TransportStatus.UPSTREAM_ERROR),
])
def test_http_status_classification(status, expected):
    assert classify_http_status(status) is expected


@pytest.mark.parametrize("exc, expected", [
    (httpx.ConnectError("x"), TransportStatus.CONNECT_FAILED),
    (httpx.ConnectTimeout("x"), TransportStatus.CONNECT_FAILED),
    (httpx.PoolTimeout("x"), TransportStatus.CONNECT_FAILED),
    (httpx.ProxyError("x"), TransportStatus.CONNECT_FAILED),
    (httpx.ReadTimeout("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.WriteTimeout("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.ReadError("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.WriteError("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.RemoteProtocolError("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.LocalProtocolError("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
    (httpx.DecodingError("x"), TransportStatus.AMBIGUOUS_TIMEOUT),
])
def test_exception_classification_is_conservative(exc, expected):
    assert classify_exception(exc) is expected


def test_retry_after_seconds_and_http_date():
    now = datetime(2026, 9, 17, 7, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("7", now=now) == 7.0
    assert parse_retry_after(" 1.5 ", now=now) == 1.5
    assert parse_retry_after("-3", now=now) == 0.0
    later = (now + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(later, now=now) == 30.0
    earlier = (now - timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(earlier, now=now) == 0.0
    for junk in (None, "", "soon", "nan", "inf"):
        assert parse_retry_after(junk, now=now) is None


@pytest.mark.parametrize("content, finish, native, refusal, expected", [
    ("<final>4</final>", "stop", "STOP", None, CompletionStatus.COMPLETED),
    ("<final>4</final>", None, "STOP", None, CompletionStatus.COMPLETED),
    ("", "length", "MAX_TOKENS", None, CompletionStatus.LENGTH),
    ("<rationale>partial", "length", "MAX_TOKENS", None, CompletionStatus.LENGTH),
    (None, None, "MAX_TOKENS", None, CompletionStatus.LENGTH),
    ("", "stop", "STOP", None, CompletionStatus.EMPTY),
    ("   \n", "stop", "STOP", None, CompletionStatus.EMPTY),
    (None, "stop", "STOP", None, CompletionStatus.EMPTY),
    ("", "stop", "STOP", "I can't help with that.", CompletionStatus.BLOCKED),
    ("", "content_filter", "SAFETY", None, CompletionStatus.BLOCKED),
    ("partial", "stop", "RECITATION", None, CompletionStatus.BLOCKED),
    ("", None, "BLOCKLIST", None, CompletionStatus.BLOCKED),
    ("", None, "PROHIBITED_CONTENT", None, CompletionStatus.BLOCKED),
    ("text", "tool_calls", None, None, CompletionStatus.OTHER),
    ("text", None, None, None, CompletionStatus.OTHER),
    ("text", "error", None, None, CompletionStatus.OTHER),
])
def test_completion_status_table(content, finish, native, refusal, expected):
    assert completion_status(content, finish, native, refusal) is expected


def test_usage_absent_fields_are_none_never_zero():
    usage = normalize_usage({"prompt_tokens": 10, "completion_tokens": 30})
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 30)
    assert usage.reasoning_tokens is None and usage.cached_prompt_tokens is None
    assert usage.actual_cost_usd is None and usage.upstream_inference_cost_usd is None
    assert normalize_usage(None) is None and normalize_usage([1, 2]) is None


def test_usage_reasoning_is_a_subset_never_added():
    raw = {"prompt_tokens": 100, "completion_tokens": 900, "total_tokens": 1000, "cost": 0.00037,
           "completion_tokens_details": {"reasoning_tokens": 800}, "prompt_tokens_details": {"cached_tokens": 64},
           "cost_details": {"upstream_inference_cost": 0.0003}}
    usage = normalize_usage(raw)
    assert usage.completion_tokens == 900 and usage.reasoning_tokens == 800
    assert usage.cached_prompt_tokens == 64
    assert usage.actual_cost_usd == Decimal("0.00037") and usage.upstream_inference_cost_usd == Decimal("0.0003")
    assert usage.raw_usage == raw and usage.raw_usage is not raw


def test_malformed_usage_values_become_none():
    raw = {"prompt_tokens": "12", "completion_tokens": -3, "total_tokens": 1.5, "cost": "n/a",
           "prompt_tokens_details": [], "completion_tokens_details": {"reasoning_tokens": True},
           "cost_details": "none"}
    usage = normalize_usage(raw)
    assert usage.prompt_tokens is None and usage.completion_tokens is None
    assert usage.reasoning_tokens is None and usage.cached_prompt_tokens is None
    assert usage.actual_cost_usd is None and usage.upstream_inference_cost_usd is None
    assert usage.raw_usage == raw


@pytest.mark.parametrize("value, expected", [
    (0.00012, Decimal("0.00012")), (1e-07, Decimal("1E-7")), (0, Decimal("0")), ("0.5", Decimal("0.5")),
    (Decimal("0.25"), Decimal("0.25")), (-0.1, None), (float("nan"), None), (float("inf"), None), (True, None),
    ("abc", None), (None, None), ([1], None),
])
def test_money_parsing(value, expected):
    assert parse_money(value) == expected


def _body(**overrides):
    payload = {"id": "gen-1", "provider": "Google AI Studio", "model": "google/gemini-2.5-flash-lite",
               "choices": [{"finish_reason": "stop", "native_finish_reason": "STOP",
                            "message": {"role": "assistant", "content": "<final>1</final>"}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1}}
    payload.update(overrides)
    return json.dumps(payload).encode()


def test_cache_hit_wins_over_a_perfect_body():
    parsed = parse_completion(_body(), "HIT", LOCK)
    assert parsed.transport_status is TransportStatus.CACHE_HIT and parsed.completion_status is None
    assert parsed.usage is not None and parsed.visible_content == "<final>1</final>"  # kept for audit
    assert parse_completion(_body(), "hit", LOCK).transport_status is TransportStatus.CACHE_HIT
    assert parse_completion(_body(), "MISS", LOCK).transport_status is TransportStatus.OK
    assert parse_completion(_body(), None, LOCK).transport_status is TransportStatus.OK


def test_provider_and_model_mismatch():
    parsed = parse_completion(_body(provider="Google"), "MISS", LOCK)
    assert parsed.transport_status is TransportStatus.PROVIDER_MISMATCH and "provider" in parsed.error_detail
    parsed = parse_completion(_body(model="google/gemini-2.5-flash"), "MISS", LOCK)
    assert parsed.transport_status is TransportStatus.PROVIDER_MISMATCH and "model" in parsed.error_detail
    no_identity = json.loads(_body())
    del no_identity["provider"], no_identity["model"]
    parsed = parse_completion(json.dumps(no_identity).encode(), "MISS", LOCK)
    assert parsed.transport_status is TransportStatus.OK and parsed.observed_provider is None


@pytest.mark.parametrize("body, fragment", [
    (json.dumps({"error": {"code": 502, "message": "Provider returned error"}}).encode(), "code=502"),
    (_body(choices=[{"finish_reason": "error", "error": {"code": 500, "message": "boom"},
                     "message": {"role": "assistant", "content": ""}}]), "choice"),
    (_body(choices=[{"finish_reason": "error", "error": {"code": 500, "message": "boom"}}]), "choice"),
    (b"{\"id\": \"gen-1\", \"choi", "not a JSON object"),
    (b"[]", "not a JSON object"),
    (_body(choices=[]), "exactly one choice"),
    (_body(choices=[{"message": {}}, {"message": {}}]), "exactly one choice"),
    (_body(choices=["x"]), "choice is not an object"),
    (_body(choices=[{"finish_reason": "stop"}]), "no message"),
])
def test_200_bodies_that_are_not_single_completions_are_error_envelopes(body, fragment):
    parsed = parse_completion(body, "MISS", LOCK)
    assert parsed.transport_status is TransportStatus.ERROR_ENVELOPE
    assert parsed.completion_status is None
    assert fragment in parsed.error_detail


def test_reasoning_returned_detection():
    def message(**fields):
        return _body(choices=[{"finish_reason": "stop", "message": {"content": "x", **fields}}])
    assert parse_completion(message(reasoning="thinking"), None, LOCK).reasoning_returned is True
    assert parse_completion(message(reasoning_details=[{"type": "reasoning.encrypted", "data": "..."}]), None,
                            LOCK).reasoning_returned is True
    assert parse_completion(message(reasoning="", reasoning_details=[]), None, LOCK).reasoning_returned is False
    assert parse_completion(message(reasoning=None), None, LOCK).reasoning_returned is False
    assert parse_completion(message(), None, LOCK).reasoning_returned is False


def test_multipart_content_is_concatenated_and_non_text_is_none():
    parts = _body(choices=[{"finish_reason": "stop", "message": {"content": [
        {"type": "text", "text": "<final>"}, {"type": "image_url"}, {"type": "text", "text": "2</final>"}]}}])
    assert parse_completion(parts, None, LOCK).visible_content == "<final>2</final>"
    numeric = _body(choices=[{"finish_reason": "stop", "message": {"content": 5}}])
    parsed = parse_completion(numeric, None, LOCK)
    assert parsed.visible_content is None and parsed.completion_status is CompletionStatus.EMPTY
