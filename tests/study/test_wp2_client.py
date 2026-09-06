"""inference/client.py: round trip, §4.3 prompt-id check, §10.4 retry policy, failover, pool."""

from __future__ import annotations

import json
import time

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.inference import client as C
from tests.study import fake_vllm as F
from tests.study.wp2_support import fake_server, make_spec, solver_messages


def _pool(run_root, **kw):
    kw.setdefault("refresh_min_interval_s", 0.0)
    kw.setdefault("probe_timeout", 2.0)
    return C.EndpointPool(run_root, "32B-long", shard=0, **kw)


# --------------------------------------------------------------------------- round trip


def test_round_trip_with_reasoning(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer, run_id="study_v4")
        record = client.generate(spec, cell_id="F.BANK.32B.N1.B0.F00.e0.s000")
        sent = srv.requests[-1]["body"]
    # Wire contract (05_vllm_response_shape.md / P0-5): the exact create() payload.
    assert sent["model"] == "32B" and sent["seed"] == spec.engine_seed and sent["max_tokens"] == 8192
    assert sent["temperature"] == 0.6 and sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["top_k"] == 20 and sent["top_p"] == 0.95 and sent["min_p"] == 0.0
    assert sent["presence_penalty"] == 0.0 and sent["repetition_penalty"] == 1.0 and sent["return_token_ids"] is True
    assert "guided_json" not in sent
    # Record fields (architecture §2.1).
    assert record.schema_version == 1 and record.request_id == spec.request_id
    assert record.identity == spec.identity_fields() and record.engine_seed == spec.engine_seed
    assert record.seed_key == spec.seed_key and record.sampling == T.SOLVER_DECODING.as_strings()
    assert record.model["served_model_name"] == "32B" and record.model["profile"] == "32B-long" and record.model["tp_size"] == 2
    assert record.prompt_tokens == len(record.prompt_token_ids) > 0
    assert list(record.messages) == [dict(m) for m in spec.messages]
    assert record.chat_template_kwargs == {"enable_thinking": True} and len(record.chat_template_hash) == 64
    resp = record.response
    assert resp["reasoning_field_name"] == "reasoning" and resp["reasoning"].startswith("\n")
    assert resp["finish_reason"] == "stop" and resp["completion_tokens"] == len(resp["token_ids"])
    assert resp["reasoning_tokens"] == resp["token_ids"].index(F.THINK_END_ID) + 1
    assert resp["reasoning_tokens_source"] == "think_end_id"
    assert resp["usage"]["prompt_tokens"] == record.prompt_tokens
    assert json.loads(resp["content"])["final_answer"] in F.DEFAULT_ANSWER_POOL
    assert record.flops == dict(C.FLOPS_PLACEHOLDER)
    assert record.attempts == 1 and len(record.timing["attempts"]) == 1 and record.timing["attempts"][0]["error"] is None
    assert record.timing["latency_s"] >= 0 and record.timing["completed_at"] >= record.timing["submitted_at"]
    assert record.endpoint["host"] == srv.host and record.endpoint["port"] == srv.port and record.endpoint["slurm_job_id"] is None
    assert record.producer["cell_id"] == "F.BANK.32B.N1.B0.F00.e0.s000" and record.producer["run_id"] == "study_v4"
    # Integrity hash set and JSON round trip is the identity.
    record.verify()
    back = T.RequestRecord.from_dict(json.loads(record.to_json()))
    assert back == record
    back.verify()


def test_thinking_off_no_reasoning_and_cost_hook(tmp_run_root, study_config):
    spec = make_spec(study_config, decoding=T.JUDGE_DECODING, purpose=T.PURPOSE_JUDGE_BEST, namespace=T.NS_JUDGE, role="judge")

    def cost_fn(ckpt, prompt_tokens, completion_tokens):
        assert ckpt.size == "32B"
        return {"prefill": 1.0 * prompt_tokens, "decode": 2.0 * completion_tokens, "total": 3.0, "oracle": "test"}

    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer, cost_fn=cost_fn)
        record = client.generate(spec)
        assert srv.requests[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert srv.requests[-1]["body"]["max_tokens"] == 1024 and srv.requests[-1]["body"]["temperature"] == 0.0
    assert record.response["reasoning"] is None and record.response["reasoning_field_name"] is None
    assert record.response["reasoning_tokens"] == 0 and record.response["reasoning_tokens_source"] == "none"
    assert record.flops["prefill"] == float(record.prompt_tokens) and record.flops["oracle"] == "test"
    assert record.producer["cell_id"] is None


def test_prompt_token_id_mismatch_is_protocol_error(tmp_run_root, study_config):
    class DriftingTokenizer(F.FakeTokenizer):
        def render_chat(self, messages, add_generation_prompt=True, enable_thinking=None):
            return super().render_chat(messages, add_generation_prompt, enable_thinking) + " drift"

    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), DriftingTokenizer())
        with pytest.raises(T.ProtocolError, match="prompt_token_ids disagree"):
            client.generate(spec)
        assert srv.chat_count == 1  # never retried: the same request would fail identically


def test_usage_count_mismatch_is_protocol_error(tmp_run_root, study_config, monkeypatch):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        original = srv.completion_payload

        def tampered(body):
            payload = original(body)
            payload["usage"]["completion_tokens"] += 1
            return payload

        monkeypatch.setattr(srv, "completion_payload", tampered)
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.ProtocolError, match="usage.completion_tokens"):
            client.generate(spec)


def test_missing_token_ids_is_protocol_error(tmp_run_root, study_config, monkeypatch):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        original = srv.completion_payload

        def without_ids(body):
            payload = original(body)
            del payload["choices"][0]["token_ids"]
            return payload

        monkeypatch.setattr(srv, "completion_payload", without_ids)
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.ProtocolError, match="token_ids"):
            client.generate(spec)


def test_length_finish_is_a_completed_outcome(tmp_run_root, study_config):
    spec = make_spec(study_config, decoding=T.Decoding(0.6, 0.95, 20, 0.0, 0.0, 1.0, 6, True))
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        record = client.generate(spec)
        assert srv.chat_count == 1
    assert record.response["finish_reason"] == "length" and record.response["completion_tokens"] == 6
    assert record.response["content"] is None  # cut inside thinking: parser (WP3) marks it invalid
    assert record.attempts == 1


# --------------------------------------------------------------------------- retry policy


def test_failover_to_second_endpoint_after_kill(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as a, fake_server(tmp_run_root) as b:
        pool = _pool(tmp_run_root)
        first, second = sorted((a, b), key=lambda s: s.port)  # registry order: (host, port)
        assert pool.pick().port == first.port
        first.kill()
        client = C.VllmChatClient(pool, a.tokenizer)
        record = client.generate(spec)
        assert record.endpoint["port"] == second.port
        assert record.attempts == 3  # two consecutive failures rotate the pool (§3 step 9)
        errors = [att["error"]["type"] for att in record.timing["attempts"] if att["error"]]
        assert errors == ["APIConnectionError", "APIConnectionError"]
        assert [att["endpoint"]["port"] for att in record.timing["attempts"]] == [first.port, first.port, second.port]
        assert pool.rotation == 1
        # The forced refresh dropped the dead endpoint, so the next pick is direct.
        assert [e.port for e in pool.endpoints()] == [second.port]
        assert client.generate(make_spec(study_config, step_slot=1)).attempts == 1


def test_retry_cap_then_infra_failure(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        srv.fail_next(3, status=500)
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.InfraFailure, match="after 3 attempts") as info:
            client.generate(spec)
        assert srv.chat_count == 3
        assert [a["error"]["type"] for a in info.value.attempts] == ["InternalServerError"] * 3
        assert all(a["endpoint"]["port"] == srv.port for a in info.value.attempts)
        # 429 is exogenous too; one failure then success → attempts == 2.
        srv.fail_next(1, status=429)
        record = client.generate(spec)
        assert record.attempts == 2 and record.timing["attempts"][0]["error"]["type"] == "RateLimitError"


def test_non_exogenous_error_is_not_retried(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        srv.fail_next(1, status=400)
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.InfraFailure, match="non-exogenous") as info:
            client.generate(spec)
        assert srv.chat_count == 1 and len(info.value.attempts) == 1
        assert info.value.attempts[0]["error"]["type"] == "BadRequestError"


def test_timeout_is_exogenous_and_capped(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        srv.hang(3.0)
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer, request_timeout_s=0.3, max_exogenous_retries=1)
        t0 = time.monotonic()
        with pytest.raises(T.InfraFailure) as info:
            client.generate(spec)
        assert time.monotonic() - t0 < 3.0
        assert [a["error"]["type"] for a in info.value.attempts] == ["APITimeoutError"] * 2
        srv.reset_faults()


def test_aborted_finish_reason_is_exogenous(tmp_run_root, study_config):
    spec = make_spec(study_config)
    with fake_server(tmp_run_root) as srv:
        srv.finish_reason_override = "abort"
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.InfraFailure) as info:
            client.generate(spec)
        assert srv.chat_count == 3 and info.value.attempts[0]["error"]["type"] == "_AbortedCompletion"


def test_guided_json_flag_and_schema_must_agree(tmp_run_root, study_config):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    plain = make_spec(study_config, decoding=T.JUDGE_DECODING)
    guided = make_spec(study_config, decoding=T.Decoding(0.0, 1.0, 1, 0.0, 0.0, 1.0, 64, False, True))
    assert plain.request_id != guided.request_id  # the flag enters the identity (amendment E2)
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer)
        with pytest.raises(T.ProtocolError, match="guided_json"):
            client.generate(plain, guided_json_schema=schema)
        with pytest.raises(T.ProtocolError, match="guided_json"):
            client.generate(guided)
        assert srv.chat_count == 0
        client.generate(guided, guided_json_schema=schema)
        assert srv.requests[-1]["guided_json"] == schema


def test_sdk_client_has_no_auto_retry(tmp_run_root, study_config):
    with fake_server(tmp_run_root) as srv:
        client = C.VllmChatClient(_pool(tmp_run_root), srv.tokenizer, request_timeout_s=123.0)
        sdk = client._sdk_client(client.pool.pick())
        assert sdk.max_retries == 0 and float(sdk.timeout) == 123.0 and sdk.api_key == "EMPTY"
        assert str(sdk.base_url).rstrip("/") == srv.base_url


# --------------------------------------------------------------------------- endpoint pool


def test_pool_shard_round_robin_and_rotation(tmp_run_root):
    with fake_server(tmp_run_root) as a, fake_server(tmp_run_root) as b:
        ports = sorted((a.port, b.port))
        p0, p1 = _pool(tmp_run_root), C.EndpointPool(tmp_run_root, "32B-long", shard=1, refresh_min_interval_s=0.0, probe_timeout=2.0)
        assert [e.port for e in p0.endpoints()] == ports
        assert p0.pick().port == ports[0] and p1.pick().port == ports[1]
        entry = p0.pick()
        assert p0.report_failure(entry) is False  # one failure: no rotation yet
        assert p0.pick().port == ports[0]
        p0.report_success(entry)  # success resets the consecutive counter
        assert p0.report_failure(entry) is False
        assert p0.report_failure(entry) is True and p0.rotation == 1
        assert p0.pick().port == ports[1]
        assert p0.refresh_count == 2


def test_pool_cache_refresh_and_rate_limit(tmp_run_root):
    now = [1000.0]
    calls = []

    def fake_list(run_root, profile, probe_timeout):
        calls.append((str(profile), probe_timeout))
        from agents_scaling.serving.registry import ServerEntry

        return [ServerEntry(model_size="32B", hf_id="Qwen/Qwen3-32B", host="h", port=1),
                ServerEntry(model_size="32B", hf_id="Qwen/Qwen3-32B", host="h", port=2)]

    pool = C.EndpointPool(tmp_run_root, "32B-long", shard=0, refresh_s=300.0, refresh_min_interval_s=60.0,
                          clock=lambda: now[0], list_live=fake_list)
    pool.endpoints(); pool.endpoints()
    assert len(calls) == 1 and calls[0] == ("32B-long", 3.0)
    now[0] += 299.0
    pool.endpoints()
    assert len(calls) == 1
    now[0] += 2.0
    pool.endpoints()
    assert len(calls) == 2
    # Rotation-triggered refresh is rate limited to once per 60 s.
    entry = pool.pick()
    pool.report_failure(entry); pool.report_failure(entry)
    assert pool.rotation == 1 and len(calls) == 2
    now[0] += 61.0
    pool.report_failure(entry); pool.report_failure(entry)
    assert pool.rotation == 2 and len(calls) == 3
    pool.endpoints(force=True)
    assert len(calls) == 4


def test_pool_empty_and_wait_timeout(tmp_run_root):
    pool = _pool(tmp_run_root)
    assert pool.endpoints() == []
    with pytest.raises(T.InfraFailure, match="no live"):
        pool.pick()
    with pytest.raises(TimeoutError):
        pool.wait(timeout_s=0.2, poll_s=0.05)
    with fake_server(tmp_run_root) as srv:
        entry = pool.wait(timeout_s=5.0, poll_s=0.05)
        assert entry.port == srv.port and pool.pick().port == srv.port


def test_pool_rejects_bad_arguments(tmp_run_root):
    with pytest.raises(ValueError):
        C.EndpointPool(tmp_run_root, "32B-long", shard=-1)
    with pytest.raises(ValueError):
        C.EndpointPool(tmp_run_root, "32B-long", failure_threshold=0)
    with pytest.raises(ValueError):
        C.VllmChatClient(_pool(tmp_run_root), F.FakeTokenizer(), max_exogenous_retries=-1)
    with pytest.raises(ValueError):
        C.VllmChatClient(_pool(tmp_run_root), F.FakeTokenizer(), request_timeout_s=0)


def test_request_kwargs_are_the_frozen_recipe(study_config):
    spec = make_spec(study_config)
    kwargs = C.VllmChatClient.request_kwargs(spec)
    assert set(kwargs) == {"model", "messages", "temperature", "max_tokens", "seed", "extra_body"}
    assert set(kwargs["extra_body"]) == {"chat_template_kwargs", "top_k", "top_p", "min_p", "presence_penalty",
                                         "repetition_penalty", "return_token_ids"}
    assert kwargs["messages"] == [dict(m) for m in solver_messages()]
    assert 0 <= kwargs["seed"] < 2**63


# --------------------------------------------------------------------------- real tokenizer


def test_real_tokenizer_render_matches_live_observation(real_tokenizer_32b, study_config):
    """Pinned to WP2's live smoke request (2026-09-05, node3807:8829, vLLM 0.21.0): the server
    tokenized this exact prompt to 112 ids (== /tokenize count) and prompt_token_ids equalled
    the local render; ``</think>`` is id 151668 and the response token_ids started with 151667."""
    schema_line = ('{"approach":string,"evidence":[{"claim":string,"support":string,"uncertainty":"low"|"medium"|"high"|"unknown"}],'
                   '"alternatives_considered":[string],"failure_checks":[string],"final_answer":string,"confidence":number in [0,1]}')
    messages = ({"role": "user", "content": (
        "Solve the supplied task independently and return one complete answer as JSON only.\n\n"
        "Task: What is the smallest prime p such that p^2 + 1 is divisible by 5? Answer with the integer.\n"
        f"Output contract: {schema_line}")},)
    spec = make_spec(study_config, messages=messages)
    pool = C.EndpointPool("/nonexistent", "32B-long", shard=0, list_live=lambda *a, **k: [])
    client = C.VllmChatClient(pool, real_tokenizer_32b)
    ids = client.render_prompt_ids(spec)
    assert len(ids) == 112 and ids[:3] == (151644, 872, 198)
    assert client._think_end_id == F.THINK_END_ID == 151668
    assert len(client.chat_template_hash(spec)) == 64
    off = make_spec(study_config, messages=messages, decoding=T.JUDGE_DECODING)
    ids_off = client.render_prompt_ids(off)
    # The empty think block is appended when thinking is off (real Qwen3 template).
    assert F.THINK_END_ID in ids_off and F.THINK_END_ID not in ids
