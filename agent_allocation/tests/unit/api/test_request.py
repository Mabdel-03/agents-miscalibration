from __future__ import annotations

import dataclasses

import pytest

from agent_allocation.api.fake import request_contract_violations
from agent_allocation.api.request import REQUEST_HEADERS, build_request_body, request_coordinates
from tests.unit.api.conftest import MESSAGES, SAMPLING


def test_headers_disable_the_response_cache():
    assert REQUEST_HEADERS["X-OpenRouter-Cache"] == "false"
    assert REQUEST_HEADERS["Content-Type"] == "application/json"
    assert REQUEST_HEADERS["X-Title"] == "openrouter_agent_allocation_v1"
    assert not any(k.lower() == "authorization" for k in REQUEST_HEADERS)
    with pytest.raises(TypeError):
        REQUEST_HEADERS["X-OpenRouter-Cache"] = "true"  # type: ignore[index]


def test_body_is_exactly_the_contract(make_spec, lock):
    spec = make_spec(reasoning_budget=4096, completion_headroom=2048)
    body = build_request_body(spec, MESSAGES, lock, SAMPLING)
    assert body == {
        "model": "google/gemini-2.5-flash-lite",
        "messages": MESSAGES,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 6144,
        "reasoning": {"max_tokens": 4096, "exclude": False},
        "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True},
        "stream": False,
    }
    assert request_contract_violations(body) == []
    assert body["messages"] is not MESSAGES  # copied, never aliased to the caller's list


def test_body_accepts_only_temperature_and_top_p(make_spec, lock):
    body = build_request_body(make_spec(), MESSAGES, lock, {"temperature": 0.7, "top_p": 0.95})
    assert "seed" not in body and "effort" not in body["reasoning"]


@pytest.mark.parametrize("sampling, message", [
    ({**SAMPLING, "api_seed_enabled": True}, "api_seed_enabled"),
    ({**SAMPLING, "response_cache": True}, "response_cache"),
    ({**SAMPLING, "n_per_request": 2}, "n_per_request"),
    ({**SAMPLING, "n_per_request": True}, "n_per_request"),
    ({**SAMPLING, "stream": True}, "stream"),
    ({**SAMPLING, "seed": 7}, "unknown sampling keys"),
    ({"top_p": 0.95}, "temperature is required"),
    ({**SAMPLING, "temperature": "0.7"}, "finite number"),
    ({**SAMPLING, "top_p": 1.5}, "outside"),
])
def test_bad_sampling_is_refused(make_spec, lock, sampling, message):
    with pytest.raises(ValueError, match=message):
        build_request_body(make_spec(), MESSAGES, lock, sampling)


def test_lock_and_spec_mismatches_are_refused(make_spec, lock):
    with pytest.raises(ValueError, match="differs from the endpoint lock model"):
        build_request_body(make_spec(model="google/gemini-2.5-flash"), MESSAGES, lock, SAMPLING)
    with pytest.raises(ValueError, match="tier variant"):
        build_request_body(make_spec(), MESSAGES, dataclasses.replace(lock, provider_tag="google-ai-studio/flex"),
                           SAMPLING)
    with pytest.raises(ValueError, match="required parameters"):
        build_request_body(make_spec(), MESSAGES, dataclasses.replace(lock, supported_parameters=("max_tokens",)),
                           SAMPLING)
    with pytest.raises(ValueError, match="exceeds the locked endpoint cap"):
        build_request_body(make_spec(reasoning_budget=65000, completion_headroom=1024), MESSAGES, lock, SAMPLING)


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "tool", "content": "x"}],
    [{"role": "user", "content": "x", "name": "peer"}],
    [{"role": "user", "content": None}],
])
def test_messages_outside_the_contract_are_refused(make_spec, lock, messages):
    with pytest.raises(ValueError):
        build_request_body(make_spec(), messages, lock, SAMPLING)


def test_coordinates_identify_the_node_without_entering_the_body(make_spec, lock):
    spec = make_spec(draw_id=3)
    coordinates = request_coordinates(spec, f"{spec.node_id}:1")
    assert coordinates["node_id"] == spec.node_id and coordinates["draw_id"] == 3 and coordinates["kind"] == "root"
    body = build_request_body(spec, MESSAGES, lock, SAMPLING)
    assert spec.node_id not in str(body) and "draw" not in str(body)
