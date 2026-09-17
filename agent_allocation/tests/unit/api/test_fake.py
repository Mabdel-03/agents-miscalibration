from __future__ import annotations

import json
import re
from decimal import Decimal

import httpx
import pytest

from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.endpoints import select_standard_endpoint
from agent_allocation.api.fake import (
    OMIT,
    FakeReply,
    FakeScript,
    Fault,
    default_final_answer,
    fake_endpoint_lock,
    fake_endpoints_listing,
    fake_openrouter_transport,
)
from agent_allocation.api.request import REQUEST_HEADERS
from agent_allocation.contracts import CompletionStatus, GenerationKind, TransportStatus
from tests.unit.api.conftest import FLASH, FLASH_LITE, MESSAGES, SAMPLING, attempt, load_snapshot

TAGGED = re.compile(r"\A<rationale>.+</rationale>\n<final>(.+)</final>\Z", re.S)


@pytest.mark.parametrize("model", [FLASH_LITE, FLASH])
def test_fake_lock_is_the_lock_selected_from_the_saved_snapshot(model):
    selected = select_standard_endpoint(load_snapshot(model), model)
    fake = fake_endpoint_lock(model)
    assert fake == selected
    assert fake.lock_hash == selected.lock_hash
    assert fake.provider_tag == "google-ai-studio"


def test_fake_endpoint_listing_selects_like_the_saved_snapshot():
    for model in (FLASH_LITE, FLASH):
        assert select_standard_endpoint(fake_endpoints_listing(model), model) == fake_endpoint_lock(model)


async def test_default_reply_is_openrouter_shaped(make_spec, lock):
    script = FakeScript()
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    payload = json.loads(outcome.response_body)
    assert payload["id"].startswith("gen-")
    assert payload["model"] == lock.model and payload["provider"] == lock.provider_name
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["native_finish_reason"] == "STOP"
    assert choice["message"]["reasoning"] and choice["message"]["reasoning_details"]
    usage = payload["usage"]
    assert isinstance(usage["cost"], float)
    assert usage["completion_tokens_details"]["reasoning_tokens"] <= usage["completion_tokens"]
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert outcome.response_headers["x-openrouter-cache-status"] == "MISS"
    assert TAGGED.match(choice["message"]["content"])
    assert outcome.result.transport_status is TransportStatus.OK
    assert outcome.result.completion_status is CompletionStatus.COMPLETED
    assert script.billed_usd == outcome.result.usage.actual_cost_usd > 0
    assert script.generations[outcome.result.generation_id]["native_tokens_reasoning"] == \
        outcome.result.usage.reasoning_tokens


async def test_default_final_is_derived_from_the_item_and_valid_per_benchmark(make_spec, lock):
    script = FakeScript()
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    finals = {}
    cases = [("math500/test/1", "math500", r"\d+"), ("mmlu_pro/test/7", "mmlu_pro", r"[ABC]"),
             ("bbh_logic4/boolean_expressions/3", "bbh_logic4", r"True|False"),
             ("bbh_logic4/temporal_sequences/3", "bbh_logic4", r"[ABC]")]
    for item_id, benchmark, pattern in cases:
        for slot in (0, 1):
            spec = make_spec(item_id=item_id, benchmark=benchmark, root_slot=slot)
            outcome = await client.generate(spec, MESSAGES, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
            final = TAGGED.match(outcome.result.visible_content).group(1)
            assert re.fullmatch(pattern, final), (item_id, final)
            finals.setdefault(item_id, set()).add(final)
    code_messages = [{"role": "user", "content": "Write it.\nassert similar_elements((3, 4), (4, 5)) == (4,)"}]
    spec = make_spec(item_id="mbpp_plus/Mbpp/2", benchmark="mbpp_plus", completion_headroom=2048)
    outcome = await client.generate(spec, code_messages, attempt_id=attempt(spec), lock=lock, sampling=SAMPLING)
    await client.aclose()
    assert "def similar_elements(" in TAGGED.match(outcome.result.visible_content).group(1)
    assert all(len(values) == 1 for values in finals.values())  # same item -> same default final


def test_vary_by_node_changes_the_identity():
    coordinates = {"item_id": "math500/test/1", "benchmark": "math500", "node_id": "n1"}
    other = {**coordinates, "node_id": "n2"}
    assert default_final_answer(coordinates, None) == default_final_answer(other, None)
    answers = {default_final_answer({**coordinates, "node_id": f"n{i}"}, None, vary_by_node=True) for i in range(20)}
    assert len(answers) > 1


async def test_keyed_replies_wildcards_sequences_and_node_overrides(make_spec, lock):
    root0 = make_spec(root_slot=0)
    root1 = make_spec(root_slot=1)
    private = make_spec(kind=GenerationKind.PRIVATE, update_round=1, parent_ids=(root0.node_id,))
    script = FakeScript()
    script.set_reply("math500/test/algebra/1", None, None, None, FakeReply(content="<final>item</final>"))
    script.set_reply("math500/test/algebra/1", GenerationKind.ROOT, 1, 0,
                     [FakeReply(fault=Fault.RATE_LIMITED), FakeReply(content="<final>second</final>")])
    script.by_node[private.node_id] = FakeReply(content="<final>node</final>")
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))

    async def run(spec, number):
        return (await client.generate(spec, MESSAGES, attempt_id=attempt(spec, number), lock=lock,
                                      sampling=SAMPLING)).result

    assert (await run(root0, 1)).visible_content == "<final>item</final>"
    first = await run(root1, 1)
    assert first.transport_status is TransportStatus.RATE_LIMITED
    assert (await run(root1, 2)).visible_content == "<final>second</final>"
    assert (await run(root1, 3)).visible_content == "<final>second</final>"   # last element repeats
    assert (await run(private, 1)).visible_content == "<final>node</final>"
    await client.aclose()
    assert [c.node_call_no for c in script.completion_calls()] == [1, 1, 2, 3, 1]


async def test_queue_and_hook_take_precedence(make_spec, lock):
    seen = []

    def hook(call):
        seen.append(call.coordinates["attempt_id"])
        return FakeReply(content="<final>hook</final>") if call.coordinates["root_slot"] == 2 else None

    script = FakeScript(hook=hook, queue=[FakeReply(fault=Fault.UPSTREAM_ERROR, status_code=503)])
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    specs = [make_spec(root_slot=slot) for slot in (0, 1, 2)]
    results = [(await client.generate(s, MESSAGES, attempt_id=attempt(s), lock=lock, sampling=SAMPLING)).result
               for s in specs]
    await client.aclose()
    assert results[0].transport_status is TransportStatus.UPSTREAM_ERROR and results[0].http_status == 503
    assert results[1].transport_status is TransportStatus.OK
    assert results[2].visible_content == "<final>hook</final>"
    assert len(seen) == 3


@pytest.mark.parametrize("mutation, message", [
    (lambda b: b.update(stream=True), "stream must be false"),
    (lambda b: b.update(seed=1), "'seed' is outside"),
    (lambda b: b["provider"].update(allow_fallbacks=True), "allow_fallbacks"),
    (lambda b: b["reasoning"].update(effort="high"), "reasoning must be exactly"),
    (lambda b: b["provider"].update(only=["google-ai-studio/flex"]), "tier variant"),
])
async def test_fake_refuses_requests_outside_the_contract(mutation, message):
    body = {"model": FLASH_LITE, "messages": MESSAGES, "temperature": 0.7, "top_p": 0.95, "max_tokens": 2048,
            "reasoning": {"max_tokens": 1024, "exclude": False},
            "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True},
            "stream": False}
    mutation(body)
    script = FakeScript()
    async with httpx.AsyncClient(transport=fake_openrouter_transport(script)) as raw:
        response = await raw.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                  headers=dict(REQUEST_HEADERS))
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert script.violations and script.billed_usd == 0


async def test_cache_emulation_turns_on_only_without_the_disable_header():
    body = {"model": FLASH_LITE, "messages": MESSAGES, "temperature": 0.7, "top_p": 0.95, "max_tokens": 2048,
            "reasoning": {"max_tokens": 1024, "exclude": False},
            "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True},
            "stream": False}
    script = FakeScript()
    url = "https://openrouter.ai/api/v1/chat/completions"
    async with httpx.AsyncClient(transport=fake_openrouter_transport(script)) as raw:
        disabled = [await raw.post(url, json=body, headers={"X-OpenRouter-Cache": "false"}) for _ in range(2)]
        enabled = [await raw.post(url, json=body) for _ in range(2)]
    assert [r.headers["x-openrouter-cache-status"] for r in disabled] == ["MISS", "MISS"]
    assert disabled[0].json()["id"] != disabled[1].json()["id"]
    assert [r.headers["x-openrouter-cache-status"] for r in enabled] == ["MISS", "HIT"]
    assert enabled[0].content == enabled[1].content


async def test_account_routes_and_billing_state():
    script = FakeScript(prior_usage_usd=Decimal("1.5"), metadata_lag_polls=1)
    transport = fake_openrouter_transport(script)
    assert transport.script is script
    client = OpenRouterClient(api_key=None, transport=transport)
    key = await client.key_info()
    assert "label" not in key and key["usage"] == Decimal("1.5") and key["limit_remaining"] == Decimal("133.5")
    credits = await client.credits()
    assert credits == {"total_credits": Decimal("150.0"), "total_usage": Decimal("1.5")}
    assert await client.generation_metadata("gen-unknown") is None
    raw, parsed = await client.list_endpoints(FLASH_LITE)
    assert json.loads(raw) == parsed and parsed["data"]["id"] == FLASH_LITE
    await client.aclose()


def test_omit_sentinel_survives_copy():
    import copy
    reply = FakeReply(reasoning=OMIT)
    assert copy.deepcopy(reply).reasoning is OMIT
