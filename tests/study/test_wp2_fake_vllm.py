"""fake_vllm: wire shape pinned to the live vLLM 0.21 observation, determinism, modes, faults."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from agents_scaling.serving import registry
from tests.study import fake_vllm as F
from tests.study.wp2_support import fake_server, solver_messages

OBSERVED_TOP_KEYS = {
    "choices", "created", "id", "kv_transfer_params", "model", "object", "prompt_logprobs",
    "prompt_routed_experts", "prompt_text", "prompt_token_ids", "service_tier", "system_fingerprint", "usage",
}
OBSERVED_CHOICE_KEYS = {"finish_reason", "index", "logprobs", "message", "routed_experts", "stop_reason", "token_ids"}
OBSERVED_MESSAGE_KEYS = {"annotations", "audio", "content", "function_call", "reasoning", "refusal", "role", "tool_calls"}
OBSERVED_USAGE_KEYS = {"completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"}


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _chat_body(messages, seed=1, thinking=True, max_tokens=8192, **extra):
    return {
        "model": "32B", "messages": list(messages), "temperature": 0.6, "max_tokens": max_tokens, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": thinking}, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0, "return_token_ids": True, **extra,
    }


def test_health_models_tokenize(tmp_run_root):
    with fake_server(tmp_run_root) as srv:
        with urllib.request.urlopen(f"http://{srv.host}:{srv.port}/health", timeout=5) as r:
            assert r.status == 200
        with urllib.request.urlopen(f"http://{srv.host}:{srv.port}/v1/models", timeout=5) as r:
            models = json.loads(r.read().decode())
        assert models["object"] == "list" and models["data"][0]["id"] == "32B"
        assert models["data"][0]["max_model_len"] == 40960
        msgs = list(solver_messages())
        status, tok = _post(f"http://{srv.host}:{srv.port}/tokenize", {"model": "32B", "messages": msgs,
                                                                          "add_generation_prompt": True,
                                                                          "chat_template_kwargs": {"enable_thinking": True}})
        assert status == 200 and set(tok) == {"count", "max_model_len", "tokens", "token_strs"}
        expected = srv.tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=True, return_dict=False)
        assert tok["tokens"] == list(expected) and tok["count"] == len(expected)
        # Registered entry is accepted by the real registry and found live via /health.
        live = registry.list_live_servers(tmp_run_root, "32B-long", probe_timeout=2.0, probe_attempts=1)
        assert [(e.host, e.port) for e in live] == [(srv.host, srv.port)]
        assert srv.request_count >= 4  # /health probes count as requests too


def test_response_shape_matches_live_observation(tmp_run_root):
    with fake_server(tmp_run_root) as srv:
        status, body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=42))
    assert status == 200
    assert set(body) == OBSERVED_TOP_KEYS
    choice = body["choices"][0]
    assert set(choice) == OBSERVED_CHOICE_KEYS
    assert set(choice["message"]) == OBSERVED_MESSAGE_KEYS
    assert set(body["usage"]) == OBSERVED_USAGE_KEYS
    assert "reasoning_content" not in choice["message"]
    assert body["usage"]["prompt_tokens"] == len(body["prompt_token_ids"])
    assert body["usage"]["completion_tokens"] == len(choice["token_ids"])
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert choice["finish_reason"] == "stop" and choice["stop_reason"] is None
    # Thinking on: token_ids = <think> ... </think> ... <|im_end|>; content starts with "\n\n".
    assert choice["token_ids"][0] == F.THINK_START_ID and choice["token_ids"][-1] == F.IM_END_ID
    assert F.THINK_END_ID in choice["token_ids"]
    assert choice["message"]["content"].startswith("\n\n")
    assert isinstance(choice["message"]["reasoning"], str) and choice["message"]["reasoning"].startswith("\n")
    candidate = json.loads(choice["message"]["content"])
    assert list(candidate) == ["approach", "evidence", "alternatives_considered", "failure_checks", "final_answer", "confidence"]
    assert candidate["final_answer"] in F.DEFAULT_ANSWER_POOL and 0 <= candidate["confidence"] <= 1


def test_thinking_off_reasoning_null_and_fence(tmp_run_root):
    with fake_server(tmp_run_root, fence_rate=1.0) as srv:
        status, body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=3, thinking=False))
    choice = body["choices"][0]
    assert choice["message"]["reasoning"] is None
    assert choice["token_ids"][0] != F.THINK_START_ID
    content = choice["message"]["content"]
    assert content.startswith("```json\n") and content.endswith("\n```")
    # The empty think block is part of the rendered prompt when thinking is off.
    assert F.THINK_END_ID in body["prompt_token_ids"]


def test_determinism_and_seed_sensitivity(tmp_run_root):
    with fake_server(tmp_run_root) as srv:
        a = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=7))[1]
        b = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=7))[1]
        c = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=8))[1]
    assert a["choices"][0]["message"] == b["choices"][0]["message"]
    assert a["choices"][0]["token_ids"] == b["choices"][0]["token_ids"]
    assert a["choices"][0]["message"]["content"] != c["choices"][0]["message"]["content"]
    assert a["id"] != b["id"]  # response ids are per call, content is per key


def test_votes_non_degenerate_per_item(tmp_run_root):
    from collections import Counter

    items = ("item alpha?", "item beta?", "item delta?")  # pool rotations 1, 2, 0
    with fake_server(tmp_run_root) as srv:
        for item in items:
            draws = []
            for seed in range(8):
                body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(item), seed=seed))[1]
                draws.append(json.loads(body["choices"][0]["message"]["content"])["final_answer"])
            assert set(draws) <= set(F.DEFAULT_ANSWER_POOL)
            assert max(draws.count(x) for x in set(draws)) >= 2  # a plurality exists
            assert len(set(draws)) >= 2  # ... but not unanimity across 8 draws
        # Distribution (pure method, no HTTP): the favoured answer is the item's pool rotation.
        favourites = []
        for item in items:
            msgs = solver_messages(item)
            counts = Counter(srv.candidate(msgs, F.request_key(msgs, seed))["final_answer"] for seed in range(300))
            rotation = int(F.item_key(msgs)[:8], 16) % len(F.DEFAULT_ANSWER_POOL)
            assert counts.most_common(1)[0][0] == F.DEFAULT_ANSWER_POOL[rotation]
            assert 0.4 < counts.most_common(1)[0][1] / 300 < 0.7
            favourites.append(counts.most_common(1)[0][0])
    assert favourites == ["B", "C", "A"]


def test_invalid_rate_and_modes(tmp_run_root):
    with fake_server(tmp_run_root, invalid_rate=1.0, hub_mode="delegate", worker_mode="partial") as srv:
        url = srv.base_url + "/chat/completions"
        solver = _post(url, _chat_body(solver_messages(), seed=1, thinking=False))[1]["choices"][0]["message"]["content"]
        with pytest.raises(ValueError):
            json.loads(solver)
        hub_prompt = ({"role": "user", "content": "You have 5 total role slots, including yourself, and 4 available worker slots.\n"
                                                  "Return exactly one coordinator_action object.\nTask: q?"},)
        hub = json.loads(_post(url, _chat_body(hub_prompt, seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert hub["action"] == "delegate" and [a["worker_slot"] for a in hub["assignments"]] == [1, 2]
        assert set(hub["assignments"][0]) == {"worker_slot", "subtask_id", "question", "source_handles", "required_output_type", "return_contract"}
        worker_prompt = ({"role": "user", "content": 'Return subtask_result JSON. Assignment: {"subtask_id": "sub-9"}'},)
        worker = json.loads(_post(url, _chat_body(worker_prompt, seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert worker["subtask_id"] == "sub-9" and worker["status"] == "partial"
        assert set(worker) == {"subtask_id", "contract", "status", "result", "assumptions", "evidence_handles", "confidence"}
        judge = json.loads(_post(url, _chat_body(({"role": "user", "content": "Return JSON with quality_score ..."},), seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert set(judge) == {"quality_score", "requirement_coverage", "reasoning_support", "unresolved_risks"}
        hle = json.loads(_post(url, _chat_body(({"role": "user", "content": "extracted_final_answer: ..."},), seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert hle["correct"] in ("yes", "no")
        fc = json.loads(_post(url, _chat_body(({"role": "user", "content": "For PERSONAL_FINAL, assess ..."},), seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert set(fc) == {"q_personal", "q_child_contract", "q_team_now", "q_recover", "q_preserve"}
    # hub with zero workers → final even in delegate mode; N=1 truthful (P1-3)
    with fake_server(tmp_run_root, hub_mode="delegate") as srv:
        one = ({"role": "user", "content": "1 total role slots and 0 available worker slots. coordinator_action"},)
        hub = json.loads(_post(srv.base_url + "/chat/completions", _chat_body(one, seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        assert hub["action"] == "final" and set(hub["candidate"]) == {"approach", "evidence", "alternatives_considered", "failure_checks", "final_answer", "confidence"}


def test_hub_alternate_mode(tmp_run_root):
    with fake_server(tmp_run_root, hub_mode="alternate") as srv:
        p = ({"role": "user", "content": "4 available worker slots. coordinator_action\nTask: x?"},)
        first = json.loads(_post(srv.base_url + "/chat/completions", _chat_body(p, seed=1, thinking=False))[1]["choices"][0]["message"]["content"])
        second = json.loads(_post(srv.base_url + "/chat/completions", _chat_body(p, seed=2, thinking=False))[1]["choices"][0]["message"]["content"])
    assert (first["action"], second["action"]) == ("delegate", "final")


def test_length_truncation(tmp_run_root):
    with fake_server(tmp_run_root) as srv:
        body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=1, max_tokens=5))[1]
        choice = body["choices"][0]
        assert choice["finish_reason"] == "length" and len(choice["token_ids"]) == 5
        assert body["usage"]["completion_tokens"] == 5
        assert choice["message"]["content"] is None  # cut inside the thinking channel
        assert isinstance(choice["message"]["reasoning"], str)
        body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=1, thinking=False, max_tokens=3))[1]
        assert body["choices"][0]["finish_reason"] == "length" and body["choices"][0]["message"]["content"]


def test_guided_json_recorded_not_enforced_by_default(tmp_run_root):
    schema = {"type": "object", "required": ["answer", "confidence"], "additionalProperties": False,
              "properties": {"answer": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
    with fake_server(tmp_run_root, fence_rate=1.0) as srv:
        body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=1, thinking=False, guided_json=schema))[1]
        assert srv.requests[-1]["guided_json"] == schema
        assert body["choices"][0]["message"]["content"].startswith("```json")  # ignored, as on the real stack
    with fake_server(tmp_run_root, fence_rate=1.0, guided_json_mode="enforce") as srv:
        body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=1, thinking=False, guided_json=schema))[1]
        assert set(json.loads(body["choices"][0]["message"]["content"])) == {"answer", "confidence"}


def test_fault_injection_and_model_check(tmp_run_root):
    with fake_server(tmp_run_root) as srv:
        srv.fail_next(1, status=500)
        srv.fail_next(1, status=429)
        url = srv.base_url + "/chat/completions"
        assert _post(url, _chat_body(solver_messages()))[0] == 500
        assert _post(url, _chat_body(solver_messages()))[0] == 429
        assert _post(url, _chat_body(solver_messages()))[0] == 200
        assert _post(url, dict(_chat_body(solver_messages()), model="nope"))[0] == 404
        assert srv.chat_count == 4
        srv.kill()
        with pytest.raises(urllib.error.URLError):
            _post(url, _chat_body(solver_messages()))


def test_fake_tokenizer_is_instance_independent(stub_tokenizer):
    a, b = F.FakeTokenizer(), F.FakeTokenizer()
    msgs = list(solver_messages())
    assert a.apply_chat_template(msgs, return_dict=False) == b.apply_chat_template(msgs, return_dict=False)
    assert a.apply_chat_template(msgs)["input_ids"] == b.apply_chat_template(msgs, return_dict=False)
    assert a.convert_tokens_to_ids("</think>") == F.THINK_END_ID
    assert a.decode(a.encode("hello world")) == "hello world"
    # Same rendering rules as the WP0 stub (only the id assignment differs).
    assert a.render_chat(msgs, enable_thinking=False) == stub_tokenizer.render_chat(msgs, enable_thinking=False)


def test_responder_hook_and_detect_mode(tmp_run_root):
    seen = []

    def responder(ctx):
        seen.append(ctx["mode"])
        return '{"custom": true}'

    with fake_server(tmp_run_root, responder=responder) as srv:
        body = _post(srv.base_url + "/chat/completions", _chat_body(solver_messages(), seed=1, thinking=False))[1]
    assert body["choices"][0]["message"]["content"] == '{"custom": true}' and seen == ["solver"]
    assert F.detect_mode(({"role": "user", "content": "coordinator_action"},)) == "hub"
    assert F.item_key(solver_messages("q1")) != F.item_key(solver_messages("q2"))
    assert F.item_key(solver_messages("q1")) == F.item_key(({"role": "user", "content": "other wrapper\nTask: q1\nOutput contract: x"},))
