"""The normalized response fixture (deliverable for downstream agents) stays truthful: replaying each stored raw
response through the real client reproduces the stored ``GenerationResult`` record exactly (latency aside)."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_allocation.api.client import OpenRouterClient
from agent_allocation.contracts import EndpointLock, GenerationResult, GenerationSpec, dump_record, load_record

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "normalized_responses.json"
DATA = json.loads(FIXTURE.read_text())
CASES = {case["case"]: case for case in DATA["cases"]}


def replay_transport(case):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(case["http_status"], headers=case["response_headers"],
                              content=case["response_body_text"].encode())
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("name", sorted(CASES))
async def test_stored_raw_response_normalizes_to_the_stored_record(name):
    case = CASES[name]
    lock = load_record(EndpointLock, DATA["lock"])
    spec = load_record(GenerationSpec, case["spec"])
    client = OpenRouterClient(api_key=None, transport=replay_transport(case))
    outcome = await client.generate(spec, case["messages"], attempt_id=f"{spec.node_id}:1", lock=lock,
                                    sampling={"temperature": 0.7, "top_p": 0.95})
    await client.aclose()
    replayed = dump_record(outcome.result)
    replayed["latency_seconds"] = None
    assert replayed == case["result"]
    assert outcome.request_body == case["request_body"]
    stored = load_record(GenerationResult, case["result"])
    assert stored.node_id == spec.node_id


def test_fixture_covers_the_statuses_downstream_agents_handle():
    statuses = {(c["result"]["transport_status"], c["result"]["completion_status"]) for c in DATA["cases"]}
    assert {("ok", "completed"), ("ok", "length"), ("ok", "empty"), ("ok", "blocked"), ("cache_hit", None),
            ("provider_mismatch", None), ("error_envelope", None), ("rate_limited", None),
            ("payment_required", None), ("upstream_error", None)} <= statuses
    no_reasoning = CASES["ok_without_reasoning_fields"]["result"]
    assert no_reasoning["usage"]["reasoning_tokens"] is None and no_reasoning["reasoning_returned"] is False
    assert CASES["rate_limited_429"]["result"]["retry_after_seconds"] == 7.0
