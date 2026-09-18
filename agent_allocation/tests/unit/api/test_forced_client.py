"""ForcedClient against the fake /completions route: exact budgets, both server styles, short-THINK retryability,
ANSWER retries that never re-pay THINK, cost accounting on failure, raw payload round trip, metadata merging and
Gate 0."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent_allocation.api import fake
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_forced_endpoint_lock
from agent_allocation.api.forced import (
    ForcedClient, ForcedSettings, RAW_LAYOUT, forced_endpoint_gate, forced_generation_ids, forced_usage_from_raw,
    is_forced_usage, make_generation_client, reasoning_text_from_raw, render_chat_prompt,
)
from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.reconcile import CONSISTENT, DISCREPANT, PENDING, reconcile_forced_generation
from agent_allocation.api.request import RequestContractError
from agent_allocation.config import load_config
from agent_allocation.contracts import (
    RETRYABLE_TRANSPORT, CompletionStatus, GenerationKind, GenerationResult, GenerationSpec, LedgerEventKind,
    TransportStatus, load_record,
)
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.ledger import CollectionLedger, confirmed_cost, journal_attempts
from agent_allocation.storage.journal import LedgerJournal
from agent_allocation.storage.raw import read_raw_response, result_from_raw, write_raw_response

ROOT = Path(__file__).resolve().parents[3]
MESSAGES = [{"role": "system", "content": "Solve the supplied task."},
            {"role": "user", "content": "TASK\nWhat is 6 * 7?\n"}]


def forced_cfg(**forced) -> dict:
    cfg = load_config(ROOT / "configs" / "forced27b.json")
    cfg["provider"]["endpoint_tag"] = "parasail/fp8"
    cfg["forced"].update(forced)
    return cfg


def spec_for(lock, budget: int = 1024, headroom: int = 4096, **overrides) -> GenerationSpec:
    fields = dict(run_id="unit", partition_id="pilot", item_id="aime/2025/1", benchmark="aime",
                  kind=GenerationKind.ROOT, model=lock.model, endpoint_lock_hash=lock.lock_hash,
                  input_manifest_hash="a" * 64, decoding_config_hash="b" * 64, prompt_version="p",
                  reasoning_budget=budget, completion_headroom=headroom, root_slot=0, team_size=0, update_round=0,
                  draw_id=0)
    fields.update(overrides)
    return GenerationSpec(**fields)


def client_for(script: FakeScript, cfg: dict | None = None) -> ForcedClient:
    return ForcedClient(settings=ForcedSettings.from_config(cfg or forced_cfg()), api_key=None,
                        transport=fake.fake_openrouter_transport(script), answer_backoff_base_seconds=0.0)


async def run_one(script: FakeScript, budget: int = 1024, cfg: dict | None = None, attempt: int = 1):
    lock = fake_forced_endpoint_lock()
    spec = spec_for(lock, budget)
    cfg = cfg or forced_cfg()
    client = client_for(script, cfg)
    try:
        outcome = await client.generate(spec, MESSAGES, attempt_id=f"{spec.node_id}:{attempt}", lock=lock,
                                        sampling=cfg["sampling"])
    finally:
        await client.aclose()
    return spec, outcome


@pytest.mark.parametrize("style", ["inline", "parser", "parser_answer_in_reasoning"])
@pytest.mark.parametrize("budget", [512, 1024, 2048])
async def test_honoured_bias_realizes_exactly_the_budget_in_one_outcome(style, budget):
    script = FakeScript(forced_reasoning_style=style)
    spec, outcome = await run_one(script, budget)
    result = outcome.result
    assert result.transport_status is TransportStatus.OK and result.completion_status is CompletionStatus.COMPLETED
    assert result.usage.reasoning_tokens == budget
    assert "<final>" in result.visible_content and "</think>" not in result.visible_content
    assert result.reasoning_returned
    calls = script.text_completion_calls()
    assert [c.coordinates["phase"] for c in calls] == ["think", "answer"]
    assert script.violations == []
    think_body, answer_body = calls[0].body, calls[1].body
    assert outcome.request_body == {"think": think_body, "answer": answer_body}
    assert think_body["max_tokens"] == budget and think_body["logit_bias"] == {"248069": -100, "248046": -100,
                                                                               "248044": -100}
    assert think_body["prompt"] == render_chat_prompt(MESSAGES)
    reasoning = reasoning_text_from_raw(outcome.response_body)
    assert len(reasoning) == 4 * budget
    assert answer_body["prompt"] == think_body["prompt"] + reasoning + "\n</think>\n\n"
    assert answer_body["max_tokens"] == spec.completion_headroom and "logit_bias" not in answer_body
    assert calls[0].headers["x-openrouter-cache"] == "false" and calls[1].headers["x-openrouter-cache"] == "false"
    # money and tokens: summed over both calls, raw usage split per call
    assert result.usage.actual_cost_usd == script.billed_usd == sum(c.billed_usd for c in calls)
    usage = forced_usage_from_raw(outcome.response_body)
    assert result.usage.prompt_tokens == usage["think_prompt_tokens"] + usage["answer_prompt_tokens"]
    assert result.usage.completion_tokens == budget + usage["answer_completion_tokens"]
    assert usage["retokenization_delta"] in (2, 3)
    assert usage["think_finish_reason"] == "length" and usage["reasoning_tokens"] == budget
    assert set(result.usage.raw_usage) == {"think", "answer"} and is_forced_usage(result.usage)
    assert result.generation_id == calls[1].generation_id
    assert forced_generation_ids(outcome.response_body) == (calls[0].generation_id, calls[1].generation_id)


async def test_short_think_is_retryable_and_never_completed():
    script = FakeScript(hook=lambda call: FakeReply(fault=Fault.SHORT_THINK)
                        if call.coordinates.get("phase") == "think" else None)
    spec, outcome = await run_one(script, 1024)
    result = outcome.result
    assert result.transport_status is TransportStatus.ERROR_ENVELOPE and result.transport_status in RETRYABLE_TRANSPORT
    assert result.completion_status is None and result.visible_content is None
    assert "forced_think_short" in result.error_detail
    assert [c.coordinates["phase"] for c in script.text_completion_calls()] == ["think"]     # no ANSWER paid
    assert confirmed_cost(result) == script.billed_usd > 0                  # the cut THINK is settled, not held
    assert result.generation_id == script.text_completion_calls()[0].generation_id
    assert json.loads(outcome.response_body)["answer"] is None


async def test_ignored_bias_think_closes_on_its_own_and_is_rejected():
    spec, outcome = await run_one(FakeScript(forced_bias_honoured=False), 1024)
    assert outcome.result.transport_status is TransportStatus.ERROR_ENVELOPE
    assert "forced_think" in outcome.result.error_detail


async def test_answer_429_is_retried_without_resending_think():
    seen = {"answer": 0}

    def hook(call):
        if call.coordinates.get("phase") == "answer":
            seen["answer"] += 1
            if seen["answer"] == 1:
                return FakeReply(fault=Fault.RATE_LIMITED, retry_after="0")
        return None

    script = FakeScript(hook=hook)
    spec, outcome = await run_one(script)
    phases = [c.coordinates["phase"] for c in script.text_completion_calls()]
    assert phases == ["think", "answer", "answer"]
    result = outcome.result
    assert result.transport_status is TransportStatus.OK and result.usage.reasoning_tokens == 1024
    assert result.usage.actual_cost_usd == script.billed_usd
    raw = json.loads(outcome.response_body)
    assert raw["layout"] == RAW_LAYOUT and len(raw["answer_failures"]) == 1
    assert raw["answer_failures"][0]["status"] == "rate_limited" and forced_usage_from_raw(raw)["answer_failures"] == 1


async def test_answer_failure_still_charges_think(tmp_path):
    script = FakeScript(hook=lambda call: FakeReply(fault=Fault.UPSTREAM_ERROR, status_code=503)
                        if call.coordinates.get("phase") == "answer" else None)
    cfg = forced_cfg(answer_attempts=2)
    spec, outcome = await run_one(script, cfg=cfg)
    result = outcome.result
    phases = [c.coordinates["phase"] for c in script.text_completion_calls()]
    assert phases == ["think", "answer", "answer"]
    assert result.transport_status is TransportStatus.UPSTREAM_ERROR and result.transport_status in RETRYABLE_TRANSPORT
    think_cost = script.text_completion_calls()[0].billed_usd
    assert think_cost > 0 and result.usage.actual_cost_usd == think_cost == script.billed_usd
    assert result.usage.reasoning_tokens == 1024 and result.visible_content is None
    # the ledger is charged THINK's cost for this failed attempt
    ledger = CollectionLedger(LedgerJournal(tmp_path / "journal.jsonl"), ceiling_usd=Decimal("5"),
                              max_unresolved_usd=Decimal("2"))
    aid = f"{spec.node_id}:1"
    assert await ledger.reserve(attempt_id=aid, node_id=spec.node_id, partition_id="pilot", amount_usd=Decimal("1"))
    await ledger.mark_sent(aid)
    await ledger.settle(aid, confirmed_cost(result))
    info = journal_attempts(tmp_path / "journal.jsonl")[aid]
    assert info.resolution is LedgerEventKind.SETTLE and info.resolved_usd == think_cost


async def test_answer_with_unknown_cost_is_not_retried_and_leaves_the_attempt_unresolved():
    script = FakeScript(hook=lambda call: FakeReply(fault=Fault.REMOTE_DISCONNECT)
                        if call.coordinates.get("phase") == "answer" else None)
    spec, outcome = await run_one(script)
    result = outcome.result
    assert [c.coordinates["phase"] for c in script.text_completion_calls()] == ["think", "answer"]
    assert result.transport_status is TransportStatus.AMBIGUOUS_TIMEOUT
    assert confirmed_cost(result) is None and result.generation_id is None     # charge_max, never under-reconciled


async def test_circuit_breaking_think_is_returned_as_is():
    script = FakeScript(forced_provider_name="DeepInfra")
    spec, outcome = await run_one(script)
    assert outcome.result.transport_status is TransportStatus.PROVIDER_MISMATCH
    assert len(script.text_completion_calls()) == 1


async def test_contract_errors_raise_before_any_byte_is_sent():
    script = FakeScript()
    lock = fake_forced_endpoint_lock()
    spec = spec_for(lock, headroom=lock.max_completion_tokens + 1)
    client = client_for(script)
    with pytest.raises(RequestContractError):
        await client.generate(spec, MESSAGES, attempt_id=f"{spec.node_id}:1", lock=lock,
                              sampling=forced_cfg()["sampling"])
    with pytest.raises(RequestContractError):
        client.validate_request(spec, MESSAGES, lock, forced_cfg()["sampling"])
    with pytest.raises(RequestContractError):
        await client.generate(spec_for(lock), MESSAGES, attempt_id="bad", lock=lock, sampling=forced_cfg()["sampling"])
    await client.aclose()
    assert script.calls == []


async def test_raw_payload_round_trips_through_storage(tmp_path):
    script = FakeScript(forced_reasoning_style="parser")
    spec, outcome = await run_one(script)
    paths = RunPaths.for_run(run_dir=tmp_path / "run", local_state_dir=tmp_path / "local")
    paths.ensure()
    relative, digest = write_raw_response(paths, outcome.result.attempt_id, outcome)
    payload = read_raw_response(paths, relative, expected_sha256=digest)
    restored = result_from_raw(payload, relative_path=relative, sha256=digest)       # strict load
    assert restored.usage == outcome.result.usage and restored.visible_content == outcome.result.visible_content
    assert payload["request_body"]["think"]["max_tokens"] == 1024
    assert payload["request_body"]["answer"]["max_tokens"] == spec.completion_headroom
    assert reasoning_text_from_raw(payload) == reasoning_text_from_raw(outcome.response_body)
    assert len(reasoning_text_from_raw(payload)) == 4 * 1024
    assert forced_usage_from_raw(payload)["reasoning_tokens"] == 1024
    assert load_record(GenerationResult, payload["result"]).usage.reasoning_tokens == 1024
    assert reasoning_text_from_raw({"response_body_text": "{}"}) == "" and forced_usage_from_raw(None)["think"] is None


async def test_generation_metadata_merges_think_and_answer():
    script = FakeScript()
    lock = fake_forced_endpoint_lock()
    spec = spec_for(lock)
    client = client_for(script)
    try:
        outcome = await client.generate(spec, MESSAGES, attempt_id=f"{spec.node_id}:1", lock=lock,
                                        sampling=forced_cfg()["sampling"])
        merged = await client.generation_metadata(outcome.result.generation_id)
        think_id, answer_id = forced_generation_ids(outcome.response_body)
        think_meta = await client.generation_metadata(think_id)
        answer_meta = await OpenRouterClient.generation_metadata(client, answer_id)
    finally:
        await client.aclose()
    assert Decimal(str(merged["total_cost"])) == outcome.result.usage.actual_cost_usd
    assert merged["provider_name"] == lock.provider_name and merged["id"] == answer_id
    ok = reconcile_forced_generation(outcome.result, think_meta, answer_meta, lock)
    assert ok.state == CONSISTENT, ok.problems
    assert reconcile_forced_generation(outcome.result, None, answer_meta, lock).state == PENDING
    bad = dict(think_meta, provider_name="DeepInfra")
    assert reconcile_forced_generation(outcome.result, bad, answer_meta, lock).state == DISCREPANT


def test_make_generation_client_selects_by_config():
    assert isinstance(make_generation_client(forced_cfg(), api_key=None), ForcedClient)
    chat = make_generation_client(load_config(ROOT / "configs" / "balanced150.json"), api_key=None)
    assert type(chat) is OpenRouterClient


@pytest.mark.parametrize("style", ["inline", "parser"])
async def test_gate0_passes_on_an_honouring_endpoint(style, tmp_path):
    script = FakeScript(forced_reasoning_style=style)
    lock = fake_forced_endpoint_lock()
    client = client_for(script)
    ledger = CollectionLedger(LedgerJournal(tmp_path / "journal.jsonl"), ceiling_usd=Decimal("5"),
                              max_unresolved_usd=Decimal("2"))
    try:
        result = await forced_endpoint_gate(client, lock, forced_cfg(), ledger=ledger,
                                            evidence_path=tmp_path / "gate0.json")
    finally:
        await client.aclose()
    assert result.gate == "gate0" and result.passed, [(c.name, c.observed) for c in result.criteria]
    assert [c.name for c in result.criteria] == ["logit_bias_honoured_banana", "think_close_token_id",
                                                 "forced_think_exact_length", "answer_returns_final"]
    think = [c for c in script.text_completion_calls() if c.coordinates["phase"] == "gate0_forced_think"][0]
    assert think.body["max_tokens"] == 2048
    attempts = journal_attempts(tmp_path / "journal.jsonl")
    assert len(attempts) == 4 and all(a.resolution is LedgerEventKind.SETTLE for a in attempts.values())
    assert sum(a.resolved_usd for a in attempts.values()) == script.billed_usd
    assert json.loads((tmp_path / "gate0.json").read_text())["passed"] is True


async def test_gate0_fails_when_the_endpoint_ignores_logit_bias():
    script = FakeScript(forced_bias_honoured=False)
    client = client_for(script)
    try:
        result = await forced_endpoint_gate(client, fake_forced_endpoint_lock(), forced_cfg())
    finally:
        await client.aclose()
    failed = {c.name for c in result.criteria if not c.passed}
    assert not result.passed and result.decision == "stop_notify"
    assert {"logit_bias_honoured_banana", "think_close_token_id", "forced_think_exact_length"} <= failed


async def test_gate0_fails_on_a_short_think():
    script = FakeScript(hook=lambda call: FakeReply(fault=Fault.SHORT_THINK)
                        if call.coordinates.get("phase") == "gate0_forced_think" else None)
    client = client_for(script)
    try:
        result = await forced_endpoint_gate(client, fake_forced_endpoint_lock(), forced_cfg())
    finally:
        await client.aclose()
    assert not result.passed
    assert [c.name for c in result.criteria if not c.passed] == ["forced_think_exact_length"]
