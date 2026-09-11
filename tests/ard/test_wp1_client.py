"""ARD36 WP1: the inference path against the fake vLLM server.

Spec §6.2 (split thinking/report budget through ``thinking_token_budget``;
``max_tokens = 1 + T + 1 + 256``), §6.3 (the report span and its failure statuses),
§10.2 (the published FLOP oracle), §18 (cache identity).  Plan §5.2, §4.2.
Contract: docs/ard36/00_contract.md §2 (WP1) and §3 (verified backend facts).
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import decoding as D
from agents_scaling.ard.client import (
    ArdChatClient,
    ContextTooLong,
    DecisionRequest,
    bind_context,
    cost_fn_for,
    generate_decision,
    oracle_for,
    oracle_tables,
    request_spec,
    top_logprobs_for,
    verify_oracle_hash,
)
from agents_scaling.serving import launch_server as L
from agents_scaling.serving.profiles import get_serving_profile
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import ProtocolError
from tests.ard import wp1_support as S
from tests.study import fake_vllm as F

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def cfg():
    return S.ard_config()


@pytest.fixture
def run_root(tmp_path):
    return S.make_run_root(tmp_path)


@pytest.fixture
def tokenizer():
    return F.FakeTokenizer()


@pytest.fixture
def context(cfg, tokenizer):
    return S.make_context(cfg, tokenizer)


@pytest.fixture
def cost_fn(cfg):
    """The real analytic oracle over the pinned snapshots (contract §2: never a placeholder)."""
    return cost_fn_for(cfg.checkpoints)


@pytest.fixture
def serve(run_root, tokenizer):
    """``with serve(**knobs) as srv:`` — a fake endpoint sharing the client's tokenizer.

    ``FakeTokenizer.decode`` only knows the tokens its own instance encoded, so the
    server and the client under test must hold the same instance for ``final_text`` to
    decode to the report the server actually emitted.
    """

    def _serve(profile=K.PROFILE_SMALL, **kwargs):
        kwargs.setdefault("tokenizer", tokenizer)
        return S.fake_server(run_root, profile, **kwargs)

    return _serve


def _run(req, run_root, context, cost_fn, server, **kwargs):
    return generate_decision(
        req,
        store=S.make_store(run_root),
        pool=S.make_pool(run_root, context.checkpoint(req.checkpoint).profile),
        cost_fn=cost_fn,
        context=context,
        **kwargs,
    )


# --------------------------------------------------------------------------- decoding


def test_decoding_is_derived_from_the_frozen_contract():
    assert D.LOW_DECODING.thinking_tokens == K.THINKING_TOKENS_LOW == 1024
    assert D.HIGH_DECODING.thinking_tokens == K.THINKING_TOKENS_HIGH == 4096
    for decoding in (D.LOW_DECODING, D.HIGH_DECODING, D.ROOT_DECODING):
        assert decoding.final_tokens == K.FINAL_TOKENS == 256
        # §6.2: the two delimiters are sampled tokens and must fit inside max_tokens, or a
        # budget cap would eat into the 256-token report.
        assert decoding.max_tokens == 1 + decoding.thinking_tokens + 1 + K.FINAL_TOKENS
        assert decoding.max_tokens == K.max_tokens_for(decoding.thinking_tokens)
        assert (decoding.temperature, decoding.top_p, decoding.top_k) == (0.6, 0.95, 20)
        assert decoding.enable_thinking is True and decoding.guided_json is False
    assert D.LOW_DECODING.max_tokens == 1282 and D.HIGH_DECODING.max_tokens == 4354
    assert D.ROOT_DECODING == D.LOW_DECODING  # ROOT_THINKING_TOKENS is the low arm
    with pytest.raises(ValueError):
        D.decoding_for(2048)


def test_budgets_enter_the_decoding_hash_as_ints():
    # Contract §0.3: anything sent to vLLM enters the identity; JCS forbids floats.
    low, high = D.LOW_DECODING.as_strings(), D.HIGH_DECODING.as_strings()
    assert low["thinking_tokens"] == 1024 and low["final_tokens"] == 256
    assert isinstance(low["thinking_tokens"], int) and isinstance(low["final_tokens"], int)
    assert D.LOW_DECODING.hash != D.HIGH_DECODING.hash
    # A budget change alone must move the hash even at equal max_tokens.
    same_cap = type(D.LOW_DECODING)(
        **{**D.LOW_DECODING.to_dict(), "thinking_tokens": K.THINKING_TOKENS_HIGH}
    )
    assert same_cap.max_tokens == D.LOW_DECODING.max_tokens
    assert same_cap.hash != D.LOW_DECODING.hash


def test_study_decodings_are_untouched_by_the_additive_fields():
    from agents_scaling.study.types import JUDGE_DECODING, SOLVER_DECODING

    for decoding in (SOLVER_DECODING, JUDGE_DECODING):
        assert decoding.thinking_tokens == 0 and decoding.final_tokens == 0


# --------------------------------------------------------------------------- wire contract


def test_max_tokens_arithmetic_and_budget_on_the_wire(cfg, run_root, context, cost_fn, serve):
    with serve() as srv:
        for budget in (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH):
            req = S.make_request(
                cfg,
                thinking_tokens=budget,
                purpose=K.PURPOSE_REVISE,
                namespace=K.NS_PEER,
                actor_slot=2,
                step_slot=1,
            )
            _run(req, run_root, context, cost_fn, srv)
            sent = srv.requests[-1]["body"]
            assert sent["max_tokens"] == 1 + budget + 1 + 256
            assert sent["thinking_token_budget"] == budget
            assert sent["return_token_ids"] is True
            assert sent["logprobs"] is True and sent["top_logprobs"] == 1
            assert sent["model"] == "8B" and sent["temperature"] == 0.6
            assert sent["top_k"] == 20 and sent["top_p"] == 0.95
            assert sent["chat_template_kwargs"] == {"enable_thinking": True}
            assert isinstance(sent["seed"], int)


def test_receiver_purpose_asks_for_twenty_top_logprobs(cfg, run_root, context, cost_fn, serve):
    assert top_logprobs_for(K.PURPOSE_RECEIVER) == K.RECEIVER_TOP_LOGPROBS == 20
    assert top_logprobs_for(K.PURPOSE_REVISE) == 1
    req = S.make_request(
        cfg,
        purpose=K.PURPOSE_RECEIVER,
        namespace=K.NS_RECEIVER,
        source_id="receiver:prog-0007",
    )
    with serve() as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    assert srv.requests[-1]["body"]["top_logprobs"] == 20
    assert record.answer_logprob is not None


def test_a_decoding_without_a_budget_is_refused_on_the_wire(cfg):
    from agents_scaling.study.types import SOLVER_DECODING, RequestSpec

    spec = RequestSpec(
        messages=S.decision_messages(),
        decoding=SOLVER_DECODING,
        checkpoint=cfg.checkpoints[K.CHECKPOINT_SMALL],
        seed_key=S.seed_key(cfg),
        role="ard_decision",
        study_id=cfg.study_id,
        study_seed_hex=cfg.study_seed_hex,
    )
    with pytest.raises(ProtocolError, match="thinking budget"):
        ArdChatClient.request_kwargs(spec)


# --------------------------------------------------------------------------- budget accounting


def test_natural_close_under_budget(cfg, run_root, context, cost_fn, serve):
    """The model closes ``</think>`` on its own: no injection, a report span follows."""
    req = S.make_request(cfg)
    with serve(thinking_words=40) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    ids = record.completion_token_ids
    assert ids[0] == K.THINK_START_ID
    assert record.thinking_tokens == ids.index(K.THINK_END_ID) - 1
    assert 0 < record.thinking_tokens < K.THINKING_TOKENS_LOW
    assert record.boundary_injected is False and record.cap_hit is False
    assert record.think_status is None and record.valid_delimiters
    assert record.final_tokens > 0 and record.final_overrun is False
    assert record.finish_reason == "stop"
    assert K.FINAL_OPEN in record.final_text and K.CONFIDENCE_OPEN in record.final_text
    # The report span never contains the thinking span (§6.3).
    assert "think-" not in record.final_text


def test_forced_close_at_the_budget_is_boundary_injected(cfg, run_root, context, cost_fn, serve):
    """vLLM forces ``</think>`` once the thinking count reaches T (contract §3).

    A forced boundary is also the phase-2 trigger, so this draw is delivered as two
    physical requests (``tests/ard/test_two_phase.py`` owns that behaviour).  What is
    asserted here is the thinking-side accounting, which the continuation must leave
    exactly as it was.
    """
    budget = K.THINKING_TOKENS_LOW
    req = S.make_request(cfg)
    with serve(thinking_words=budget + 500) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
        assert (srv.chat_count, srv.completion_count) == (1, 1)
    ids = record.completion_token_ids
    assert record.thinking_tokens == budget == record.budget_tokens
    assert ids[budget + 1] == K.THINK_END_ID
    assert record.boundary_injected is True
    assert record.think_status is None
    # §6.2: the 256-token report allowance survives a cap hit intact.
    assert record.final_tokens > 0 and record.final_text.startswith(K.FINAL_OPEN)
    assert record.finish_reason == "stop"
    assert record.continued is True and len(record.physical_request_ids) == 2


def test_no_think_open_maps_to_its_parse_status(cfg, run_root, context, cost_fn, serve):
    req = S.make_request(cfg)
    with serve(think_open=False) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    assert record.completion_token_ids[0] != K.THINK_START_ID
    assert record.think_status is K.ParseStatus.NO_THINK_OPEN
    assert record.valid_delimiters is False
    assert record.thinking_tokens == 0 and record.final_tokens == 0
    assert record.final_text == "" and record.answer_logprob is None
    assert record.boundary_injected is False


def test_premature_reasoning_eos_maps_to_its_parse_status(cfg, run_root, context, cost_fn, serve):
    req = S.make_request(cfg)
    with serve(think_close=False, thinking_words=30) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    ids = record.completion_token_ids
    assert ids[0] == K.THINK_START_ID and K.THINK_END_ID not in ids
    assert record.think_status is K.ParseStatus.PREMATURE_REASONING_EOS
    assert record.thinking_tokens == len(ids) - 1
    assert record.final_tokens == 0 and record.final_text == ""
    assert record.cap_hit is True and record.finish_reason == "length"
    assert record.boundary_injected is False


def test_final_overrun_keeps_only_the_first_256_report_tokens(cfg, run_root, context, cost_fn, serve):
    long_report = (
        f"{K.FINAL_OPEN} 4 {K.FINAL_CLOSE} {K.EVIDENCE_OPEN} "
        + " ".join(f"w{i}" for i in range(400))
        + f" {K.EVIDENCE_CLOSE}"
    )
    req = S.make_request(cfg)
    with serve(responder=lambda _r: long_report, thinking_words=8) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    assert record.final_tokens > K.FINAL_TOKENS
    assert record.final_overrun is True
    # The parser sees exactly the first 256 final ids; the surplus is only wasted compute.
    assert len(record.final_text.split()) == K.FINAL_TOKENS
    assert record.final_text.startswith(K.FINAL_OPEN)


def test_answer_logprob_is_the_mean_over_the_report_span(cfg, run_root, context, cost_fn, serve):
    req = S.make_request(cfg)
    with serve(thinking_words=10) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
        payload = srv.completion_payload(srv.requests[-1]["body"])
    entries = payload["choices"][0]["logprobs"]["content"]
    assert len(entries) == len(record.completion_token_ids)
    start = len(record.completion_token_ids) - record.final_tokens
    span = [e["logprob"] for e in entries[start : start + record.final_tokens]]
    assert record.answer_logprob == pytest.approx(sum(span) / len(span))
    assert record.answer_logprob < 0.0


def test_cached_prefill_tokens_are_captured(cfg, run_root, context, cost_fn, serve):
    """usage.prompt_tokens_details.cached_tokens: PEER prompts share the task prefix."""
    messages = ({"role": "user", "content": " ".join(f"tok{i}" for i in range(400))},)
    req = S.make_request(cfg, messages=messages)
    with serve(cached_prefill_ratio=0.5) as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    blocks = int(record.prompt_tokens * 0.5) // F.PREFIX_CACHE_BLOCK
    assert record.cached_prefill_tokens == blocks * F.PREFIX_CACHE_BLOCK > 0
    assert record.cached_prefill_tokens <= record.prompt_tokens


def test_cached_prefill_defaults_to_zero_when_the_server_omits_the_block(cfg, run_root, context, cost_fn, serve):
    req = S.make_request(cfg)
    with serve() as srv:
        record = _run(req, run_root, context, cost_fn, srv)
    assert record.cached_prefill_tokens == 0


def test_invalid_and_partial_report_variants_reach_the_client_untouched(cfg, run_root, context, cost_fn, serve):
    """The client never parses tags; it hands WP2 whatever the report span holds."""
    seen = {}
    for mode, needle in (
        ("no_confidence", K.CONFIDENCE_OPEN),
        ("unclosed_final", K.FINAL_CLOSE),
        ("invalid", K.FINAL_OPEN),
    ):
        with serve(report_mode=mode, thinking_words=6) as srv:
            req = S.make_request(cfg, source_id=f"math500:{mode}")
            record = _run(req, run_root, context, cost_fn, srv)
        seen[mode] = record.final_text
        assert record.think_status is None  # the delimiters are fine; the report is not
        assert needle not in record.final_text
    assert K.FINAL_OPEN in seen["no_confidence"] and K.FINAL_OPEN in seen["unclosed_final"]


# --------------------------------------------------------------------------- FLOP oracle


def test_flops_are_populated_with_the_frozen_oracle_hash(cfg, run_root, context, cost_fn, serve):
    req = S.make_request(cfg)
    with serve() as srv:
        record = _run(req, run_root, context, cost_fn, srv)
        stored = S.make_store(run_root).get(record.request_id)
    oracle = oracle_for(cfg.checkpoints[K.CHECKPOINT_SMALL])
    expected = oracle.flops_record(record.prompt_tokens, len(record.completion_token_ids))
    assert record.flops == expected == dict(stored.flops)
    assert record.flops["oracle"] == oracle.oracle_hash
    assert all(isinstance(record.flops[k], int) for k in ("prefill", "decode", "total"))
    assert record.flops["total"] == record.flops["prefill"] + record.flops["decode"]
    # Contract §2: the placeholder must never survive to a record.
    assert None not in record.flops.values()


def test_oracle_tables_carry_the_hash_for_frozen_yaml(cfg):
    tables = oracle_tables(cfg.checkpoints)
    assert set(tables) == set(cfg.checkpoints)
    for size, block in tables.items():
        oracle = oracle_for(cfg.checkpoints[size])
        assert block["oracle_hash"] == oracle.oracle_hash
        assert block["convention"]["size"] == size
        verify_oracle_hash(tables, oracle, size)
    other = FlopOracle.from_table(K.CHECKPOINT_SMALL)  # no config_sha256 -> other hash
    with pytest.raises(ProtocolError, match="differs from the frozen one"):
        verify_oracle_hash(tables, other, K.CHECKPOINT_SMALL)


def test_a_record_without_flops_is_refused_before_it_is_committed(cfg, run_root, context, serve):
    """Without the oracle the run would produce no FLOP number at all (contract §2)."""
    req = S.make_request(cfg)
    with serve() as srv:
        with pytest.raises(ProtocolError, match="FLOP placeholder"):
            _run(req, run_root, context, None, srv)
        assert srv.chat_count == 1
    # Nothing was committed, so a later run with the oracle attached still generates.
    spec = request_spec(req, context)
    assert S.make_store(run_root).get(spec.request_id) is None


# --------------------------------------------------------------------------- context preflight


def test_context_assertion_fires_before_any_request(cfg, run_root, context, cost_fn, serve):
    profile = get_serving_profile(K.PROFILE_SMALL)
    assert profile.max_model_len == K.MAX_MODEL_LEN == 16384
    over = profile.max_model_len - D.LOW_DECODING.max_tokens - K.CONTEXT_RESERVE_TOKENS + 64
    messages = ({"role": "user", "content": " ".join(f"tok{i}" for i in range(over))},)
    req = S.make_request(cfg, messages=messages)
    with serve() as srv:
        with pytest.raises(ContextTooLong) as excinfo:
            _run(req, run_root, context, cost_fn, srv)
        assert srv.chat_count == 0  # no HTTP request was ever made
    error = excinfo.value
    assert error.parse_status is K.ParseStatus.CONTEXT_FAILURE
    assert error.preflight.profile_name == K.PROFILE_SMALL
    assert error.preflight.reserve_tokens == K.CONTEXT_RESERVE_TOKENS == 128
    assert error.preflight.requested_output_tokens == D.LOW_DECODING.max_tokens
    assert error.preflight.required_tokens > profile.max_model_len
    assert not error.preflight.fits


def test_a_prompt_at_the_frozen_cap_fits_both_budget_arms(cfg, run_root, context, cost_fn, serve):
    """Plan §5.2: prompt <= ~3.3k + 4,354 output + 128 reserve fits the 16,384 profile."""
    messages = (
        {"role": "user", "content": " ".join(f"tok{i}" for i in range(K.PROMPT_TOKENS_CAP))},
    )
    with serve() as srv:
        for budget in (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH):
            req = S.make_request(
                cfg,
                messages=messages,
                thinking_tokens=budget,
                purpose=K.PURPOSE_REVISE,
                namespace=K.NS_PEER,
                step_slot=budget,
            )
            record = _run(req, run_root, context, cost_fn, srv)
            assert record.prompt_tokens > K.PROMPT_TOKENS_CAP
            assert (
                record.prompt_tokens + K.max_tokens_for(budget) + K.CONTEXT_RESERVE_TOKENS
                <= K.MAX_MODEL_LEN
            )


# --------------------------------------------------------------------------- identity


def test_two_thinking_budgets_give_different_request_ids(cfg, context):
    kwargs = dict(purpose=K.PURPOSE_REVISE, namespace=K.NS_PEER, actor_slot=1, step_slot=2)
    low = request_spec(S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_LOW, **kwargs), context)
    high = request_spec(S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_HIGH, **kwargs), context)
    assert low.seed_key == high.seed_key and low.messages == high.messages
    assert low.decoding_hash != high.decoding_hash
    assert low.request_id != high.request_id
    # ... and the local caps differ too, so the collision is impossible twice over.
    assert low.local_caps["max_tokens"] != high.local_caps["max_tokens"]


def test_the_same_root_spec_has_one_id_under_both_budget_arms(cfg, context):
    """Spec §10 E2d: no root request is ever issued at T = 4,096."""
    low = request_spec(S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_LOW), context)
    high = request_spec(S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_HIGH), context)
    assert low.decoding == high.decoding == D.ROOT_DECODING
    assert low.request_id == high.request_id


def test_a_root_bank_draw_aliases_instead_of_regenerating(cfg, run_root, context, cost_fn, serve):
    req_low = S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_LOW, step_slot=3)
    req_high = S.make_request(cfg, thinking_tokens=K.THINKING_TOKENS_HIGH, step_slot=3)
    with serve() as srv:
        first = _run(req_low, run_root, context, cost_fn, srv)
        second = _run(req_high, run_root, context, cost_fn, srv)
        assert srv.chat_count == 1  # the high arm spent no GPU work
    assert first.request_id == second.request_id
    assert first.aliased is False and second.aliased is True
    # A resumed cell measures exactly what the original run measured.
    assert second.to_dict() | {"aliased": False} == first.to_dict()


def test_the_seed_key_must_name_the_requested_checkpoint(cfg, context):
    req = DecisionRequest(
        messages=S.decision_messages(),
        checkpoint=K.CHECKPOINT_LARGE,
        thinking_tokens=K.THINKING_TOKENS_LOW,
        seed_key=S.seed_key(cfg, size=K.CHECKPOINT_SMALL),
        purpose=K.PURPOSE_ROOT,
    )
    with pytest.raises(ProtocolError, match="model_cell"):
        request_spec(req, context)


def test_purpose_and_seed_key_must_agree(cfg):
    with pytest.raises(ValueError, match="disagrees with seed_key"):
        DecisionRequest(
            messages=S.decision_messages(),
            checkpoint=K.CHECKPOINT_SMALL,
            thinking_tokens=K.THINKING_TOKENS_LOW,
            seed_key=S.seed_key(cfg, purpose=K.PURPOSE_ROOT),
            purpose=K.PURPOSE_REVISE,
        )


def test_bound_context_supports_the_frozen_four_argument_signature(cfg, run_root, context, cost_fn, serve):
    previous = bind_context(context)
    try:
        req = S.make_request(cfg, source_id="math500:bound-0001")
        with serve() as srv:
            record = generate_decision(
                req,
                store=S.make_store(run_root),
                pool=S.make_pool(run_root),
                cost_fn=cost_fn,
            )
        assert record.request_id == request_spec(req, context).request_id
    finally:
        bind_context(previous)


# --------------------------------------------------------------------------- serving


def test_ard36_profiles_are_registered_with_the_frozen_layout():
    small, large = get_serving_profile(K.PROFILE_SMALL), get_serving_profile(K.PROFILE_LARGE)
    assert (small.model_size, small.tp_size, small.served_model_name) == ("8B", 1, "8B")
    assert (large.model_size, large.tp_size, large.served_model_name) == ("32B", 2, "32B")
    assert small.max_model_len == large.max_model_len == K.MAX_MODEL_LEN


def test_port_bases_are_clear_of_every_existing_profile():
    bases = L.assert_port_base_clearance([K.PROFILE_SMALL, K.PROFILE_LARGE])
    assert bases == {K.PROFILE_SMALL: 8925, K.PROFILE_LARGE: 8992}
    table = L.port_base_table()
    for name, base in bases.items():
        gaps = [abs(base - b) for other, b in table.items() if other not in bases]
        assert min(gaps) >= L.MIN_PORT_BASE_CLEARANCE
    # The naive name would land four ports from 32B-long (8826) — a SO_REUSEPORT split.
    assert L.port_base("32B-ard36") == 8830
    with pytest.raises(ValueError, match="port base"):
        L.assert_port_base_clearance(["32B-ard36"])


def test_serve_template_placeholders_render_for_both_profiles(tmp_path):
    for profile, size, seqs, tp in (
        (K.PROFILE_SMALL, "8B", K.MAX_NUM_SEQS_SMALL, 1),
        (K.PROFILE_LARGE, "32B", K.MAX_NUM_SEQS_LARGE, 2),
    ):
        text = L.render_sbatch(
            size,
            str(tmp_path),
            "pi_tpoggio",
            "a100",
            "1-00:00:00",
            str(tmp_path / "logs"),
            serving_profile=profile,
        )
        assert f"--max-num-seqs {seqs}" in text
        assert f"--dtype {K.DTYPE}" in text  # §6.1: one precision throughout
        assert "--gpu-memory-utilization 0.90" in text
        assert f"--max-model-len {K.MAX_MODEL_LEN}" in text
        assert f"#SBATCH --gres=gpu:a100:{tp}" in text
        assert "#SBATCH --exclude=node3804" in text
        assert "--reasoning-parser qwen3" in text  # creates the reasoning_config
        assert f"PORT={L.port_base(profile)}" in text
        assert "{" + "MAX_NUM_SEQS}" not in text and "{DTYPE}" not in text


def test_serve_template_engine_knobs_are_overridable(tmp_path):
    text = L.render_sbatch(
        "8B",
        str(tmp_path),
        "pi_tpoggio",
        "a100",
        "1-00:00:00",
        str(tmp_path / "logs"),
        serving_profile=K.PROFILE_SMALL,
        max_num_seqs=32,
        dtype="float16",
        gpu_memory_utilization=0.85,
        exclude_nodes=(),
    )
    assert "--max-num-seqs 32" in text and "--dtype float16" in text
    assert "--gpu-memory-utilization 0.85" in text
    assert "#SBATCH --exclude" not in text


def test_study_v4_profiles_keep_their_serve_command(tmp_path):
    """The promoted placeholders must not change what a study_v4 replica is served with."""
    text = L.render_sbatch(
        "32B",
        str(tmp_path),
        "pi_tpoggio",
        "a100",
        "2-00:00:00",
        str(tmp_path / "logs"),
        serving_profile="32B-long",
    )
    assert "--max-num-seqs 64" in text and "--gpu-memory-utilization 0.90" in text
    assert "--max-model-len 40960" in text and "#SBATCH --exclude=node3804" in text


# --------------------------------------------------------------------------- config


def test_ard36_config_agrees_with_the_frozen_contract(cfg):
    assert cfg.study_id == K.STUDY_ID
    assert set(cfg.checkpoints) == {K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE}
    for size, profile in ((K.CHECKPOINT_SMALL, K.PROFILE_SMALL), (K.CHECKPOINT_LARGE, K.PROFILE_LARGE)):
        assert cfg.checkpoints[size].profile == profile
        assert cfg.checkpoints[size].model_revision == K.REVISIONS[size]
        assert cfg.checkpoints[size].tokenizer_revision == K.REVISIONS[size]
    block = cfg.raw["ard"]
    generation = block["generation"]
    assert generation["thinking_tokens_low"] == K.THINKING_TOKENS_LOW
    assert generation["thinking_tokens_high"] == K.THINKING_TOKENS_HIGH
    assert generation["root_thinking_tokens"] == K.ROOT_THINKING_TOKENS
    assert generation["final_tokens"] == K.FINAL_TOKENS
    assert generation["max_tokens_low"] == K.max_tokens_for(K.THINKING_TOKENS_LOW)
    assert generation["max_tokens_high"] == K.max_tokens_for(K.THINKING_TOKENS_HIGH)
    assert generation["think_start_id"] == K.THINK_START_ID
    assert generation["think_end_id"] == K.THINK_END_ID
    assert generation["receiver_top_logprobs"] == K.RECEIVER_TOP_LOGPROBS
    assert block["prompt_budget"]["prompt_tokens_cap"] == K.PROMPT_TOKENS_CAP
    assert block["prompt_budget"]["context_reserve"] == K.CONTEXT_RESERVE_TOKENS
    assert block["serving"]["max_model_len"] == K.MAX_MODEL_LEN
    assert block["serving"]["small"]["max_num_seqs"] == K.MAX_NUM_SEQS_SMALL
    assert block["serving"]["large"]["max_num_seqs"] == K.MAX_NUM_SEQS_LARGE
    assert block["serving"]["dtype"] == K.DTYPE
    assert block["team"]["n_agents"] == K.N_AGENTS
    assert block["team"]["update_rounds"] == K.UPDATE_ROUNDS
    assert block["team"]["early_stop_on_unanimity"] is K.EARLY_STOP_ON_UNANIMITY


def test_model_contract_already_covers_both_ard36_profiles(cfg):
    """``configs/model_contracts.v1.json`` needs no ARD36 entry and must not get one.

    The contract is keyed by *model size*, not by serving profile
    (``launch_server.render_sbatch`` resolves ``contracts.for_size(profile.model_size)``),
    and its loader rejects any key set other than the ordered dense ladder while the file
    is pinned by ``configs/model_contracts.v1.sha256``.  Both ARD36 profiles serve sizes
    that are already in it, at the revisions ARD36 pins.
    """
    from agents_scaling.serving.model_contracts import MODEL_SIZES, load_model_contracts

    contracts = load_model_contracts(S.REPO_ROOT / "configs" / "model_contracts.v1.json")
    for profile_name in (K.PROFILE_SMALL, K.PROFILE_LARGE):
        size = get_serving_profile(profile_name).model_size
        assert size in MODEL_SIZES
        identity = contracts.for_size(size)
        assert identity.model_revision == K.REVISIONS[size]
        assert identity.tokenizer_revision == K.REVISIONS[size]
        assert identity.hf_id == cfg.checkpoints[size].hf_id


def test_ard36_seeds_are_not_study_v4_seeds(cfg):
    from agents_scaling.study.config import load_config

    v4 = load_config(S.REPO_ROOT / "configs" / "study_v4.yaml")
    assert cfg.study_seed_hex != v4.study_seed_hex
    assert cfg.split_salt_hex != v4.split_salt_hex
    assert cfg.study_seed_hex != cfg.split_salt_hex
