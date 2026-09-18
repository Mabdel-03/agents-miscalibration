"""The forced request contract: THINK/ANSWER bodies, required endpoint parameters, top_k only in forced mode,
forced endpoint selection, the fake's /completions contract checker, and raw-completion parsing."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_allocation.api.endpoints import EndpointBlocked, endpoint_candidates, select_standard_endpoint
from agent_allocation.api.fake import (
    FAKE_FORCED_MODEL, completions_contract_violations, fake_forced_endpoint_lock, fake_forced_endpoints_listing,
)
from agent_allocation.api.forced import endpoint_selection, forced_watchdog_seconds
from agent_allocation.api.normalize import answer_visible_text, parse_text_completion, split_reasoning
from agent_allocation.api.request import (
    FORCED_REQUIRED_ENDPOINT_PARAMETERS, RequestContractError, build_forced_answer_body, build_forced_think_body,
    build_request_body, spec_lock_problems,
)
from agent_allocation.config import load_config
from agent_allocation.contracts import GenerationKind, GenerationSpec, TransportStatus

ROOT = Path(__file__).resolve().parents[3]
FORCED_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "api_seed_enabled": False, "response_cache": False,
                   "n_per_request": 1}
BAN = (248069, 248046, 248044)


def forced_spec(lock, **overrides) -> GenerationSpec:
    fields = dict(run_id="unit", partition_id="pilot", item_id="aime/2025/1", benchmark="aime",
                  kind=GenerationKind.ROOT, model=lock.model, endpoint_lock_hash=lock.lock_hash,
                  input_manifest_hash="a" * 64, decoding_config_hash="b" * 64, prompt_version="p",
                  reasoning_budget=1024, completion_headroom=4096, root_slot=0, team_size=0, update_round=0,
                  draw_id=0)
    fields.update(overrides)
    return GenerationSpec(**fields)


def test_think_body_is_exactly_the_forced_contract():
    lock = fake_forced_endpoint_lock()
    spec = forced_spec(lock)
    body = build_forced_think_body(spec, "P<think>\n", lock, FORCED_SAMPLING, ban_token_ids=BAN, ban_bias=-100)
    assert body == {"model": FAKE_FORCED_MODEL, "prompt": "P<think>\n", "temperature": 1.0, "top_p": 0.95,
                    "top_k": 20, "max_tokens": 1024,
                    "logit_bias": {"248069": -100, "248046": -100, "248044": -100},
                    "provider": {"only": ["parasail/fp8"], "allow_fallbacks": False, "require_parameters": True},
                    "usage": {"include": True}, "stream": False}
    assert "reasoning" not in body and "seed" not in body
    assert completions_contract_violations(body) == []


def test_answer_body_has_headroom_and_no_bias():
    lock = fake_forced_endpoint_lock()
    body = build_forced_answer_body(forced_spec(lock), "P<think>\nR\n</think>\n\n", lock, FORCED_SAMPLING)
    assert body["max_tokens"] == 4096 and "logit_bias" not in body and body["usage"] == {"include": True}
    assert completions_contract_violations(body) == []


def test_top_k_is_forced_only_and_needs_endpoint_support():
    lock = fake_forced_endpoint_lock()
    spec = forced_spec(lock)
    with pytest.raises(RequestContractError, match="unknown sampling keys"):
        build_request_body(spec, [{"role": "user", "content": "q"}], lock, FORCED_SAMPLING)   # chat: unchanged
    no_top_k = replace(lock, supported_parameters=tuple(p for p in lock.supported_parameters if p != "top_k"))
    with pytest.raises(RequestContractError, match="top_k"):
        build_forced_think_body(forced_spec(no_top_k), "p", no_top_k, FORCED_SAMPLING, ban_token_ids=BAN,
                                ban_bias=-100)
    sampling = {k: v for k, v in FORCED_SAMPLING.items() if k != "top_k"}
    assert "top_k" not in build_forced_think_body(forced_spec(no_top_k), "p", no_top_k, sampling,
                                                  ban_token_ids=BAN, ban_bias=-100)
    with pytest.raises(RequestContractError, match="top_k"):
        build_forced_think_body(spec, "p", lock, {**FORCED_SAMPLING, "top_k": 0}, ban_token_ids=BAN, ban_bias=-100)


@pytest.mark.parametrize("change,match", [
    (dict(supported_parameters=("max_tokens", "temperature", "top_k", "top_p")), "logit_bias"),
    (dict(provider_tag="parasail/fp8/flex"), "tier variant"),
    (dict(max_completion_tokens=1000), "cap"),
])
def test_think_body_refusals(change, match):
    lock = replace(fake_forced_endpoint_lock(), **change)
    with pytest.raises(RequestContractError, match=match):
        build_forced_think_body(forced_spec(lock), "p", lock, FORCED_SAMPLING, ban_token_ids=BAN, ban_bias=-100)


@pytest.mark.parametrize("ids,bias", [((), -100), ((True,), -100), (BAN, 5), (BAN, -101)])
def test_ban_configuration_is_validated(ids, bias):
    lock = fake_forced_endpoint_lock()
    with pytest.raises(RequestContractError):
        build_forced_think_body(forced_spec(lock), "p", lock, FORCED_SAMPLING, ban_token_ids=ids, ban_bias=bias)


def test_spec_lock_problems_forced_checks_each_call_against_the_cap():
    lock = replace(fake_forced_endpoint_lock(), max_completion_tokens=8192)
    spec = forced_spec(lock, reasoning_budget=8192, completion_headroom=4096)     # sum > cap, each call fits
    assert spec_lock_problems(spec, lock, forced=True) == []
    assert any("max_tokens" in p for p in spec_lock_problems(spec, lock))           # chat semantics: sum must fit


def test_forced_endpoint_selection_uses_the_configured_tag_and_allows_no_reasoning_price():
    listing = fake_forced_endpoints_listing()
    lock = select_standard_endpoint(listing, FAKE_FORCED_MODEL, preferred_tag="mancer/fp8", forced=True)
    assert lock.provider_tag == "mancer/fp8" and lock.provider_name == "Mancer"
    assert lock.internal_reasoning_price_per_token == "0"
    assert select_standard_endpoint(listing, FAKE_FORCED_MODEL, preferred_tag="parasail/fp8",
                                    forced=True) == fake_forced_endpoint_lock()
    with pytest.raises(EndpointBlocked):            # chat contract: needs `reasoning` and a reasoning price
        select_standard_endpoint(listing, FAKE_FORCED_MODEL, preferred_tag="parasail/fp8")
    rows = endpoint_candidates(listing, FAKE_FORCED_MODEL, forced=True)
    assert all(r["standard_selectable"] for r in rows)
    for p in FORCED_REQUIRED_ENDPOINT_PARAMETERS:
        assert p in lock.supported_parameters


def test_endpoint_selection_from_config():
    cfg = load_config(ROOT / "configs" / "forced27b.json")
    with pytest.raises(ValueError, match="endpoint_tag"):
        endpoint_selection(cfg)                      # forced mode needs the shard's tag
    cfg["provider"]["endpoint_tag"] = "parasail/fp8"
    assert endpoint_selection(cfg) == {"preferred_tag": "parasail/fp8", "forced": True}
    assert endpoint_selection(load_config(ROOT / "configs" / "balanced150.json")) == {}
    assert endpoint_selection(None) == {}


def test_forced_watchdog_covers_think_and_every_answer_attempt():
    assert forced_watchdog_seconds(15, 900, 3, 30) >= 4 * 915 + 30
    assert forced_watchdog_seconds(15, 900, 1, 30) == 2 * 915 + 30


@pytest.mark.parametrize("body,needle", [
    ({"messages": []}, "outside the forced request contract"),
    ({"stream": True}, "stream"),
    ({"logit_bias": {"x": -100}}, "logit_bias"),
    ({"logit_bias": {"1": -101}}, "logit_bias"),
    ({"usage": {"include": False}}, "usage"),
    ({"top_k": 0}, "top_k"),
    ({"prompt": ""}, "prompt"),
])
def test_fake_completions_contract_refusals(body, needle):
    lock = fake_forced_endpoint_lock()
    good = build_forced_think_body(forced_spec(lock), "p", lock, FORCED_SAMPLING, ban_token_ids=BAN, ban_bias=-100)
    problems = completions_contract_violations({**good, **body})
    assert any(needle in p for p in problems), problems


def _text_body(**choice) -> bytes:
    return json.dumps({"id": "gen-1", "provider": "Parasail", "model": FAKE_FORCED_MODEL,
                       "choices": [{"text": "", "finish_reason": "length", **choice}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}).encode()


def test_parse_text_completion_keeps_the_chat_classification():
    lock = fake_forced_endpoint_lock()
    parsed = parse_text_completion(_text_body(text="abc", reasoning="r"), "MISS", lock)
    assert parsed.transport_status is TransportStatus.OK and parsed.text == "abc" and parsed.reasoning == "r"
    assert parsed.usage.completion_tokens == 5 and str(parsed.usage.actual_cost_usd) == "0.001"
    assert parse_text_completion(_text_body(text="x"), "HIT", lock).transport_status is TransportStatus.CACHE_HIT
    wrong = json.loads(_text_body(text="x"))
    wrong["provider"] = "DeepInfra"
    assert parse_text_completion(json.dumps(wrong).encode(), None, lock).transport_status is \
        TransportStatus.PROVIDER_MISMATCH
    assert parse_text_completion(b"{", None, lock).transport_status is TransportStatus.ERROR_ENVELOPE
    assert parse_text_completion(json.dumps({"error": {"code": 502}}).encode(), None, lock).transport_status is \
        TransportStatus.ERROR_ENVELOPE
    no_text = json.loads(_text_body())
    del no_text["choices"][0]["text"]
    assert parse_text_completion(json.dumps(no_text).encode(), None, lock).transport_status is \
        TransportStatus.ERROR_ENVELOPE


def test_split_reasoning_and_answer_visible_text_both_server_styles():
    assert split_reasoning("", "think</think>\n\nanswer") == ("think", "\n\nanswer", True)
    assert split_reasoning("think", "answer") == ("think", "answer", True)
    assert split_reasoning("", "still thinking") == ("still thinking", "", False)
    assert answer_visible_text("", "<final>1</final>") == "<final>1</final>"
    assert answer_visible_text("<final>1</final>", "") == "<final>1</final>"      # parser filed the answer as reasoning
    assert answer_visible_text("", "more</think><final>1</final>") == "<final>1</final>"
