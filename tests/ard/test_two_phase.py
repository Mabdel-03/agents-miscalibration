"""ARD36 WP1: the two-phase report continuation (spec §6.2/§6.3, plan §5.2 contingency).

Why this path exists, measured on 224 real 8B decisions on the pilot roots cell: vLLM's
forced ``</think>`` takes the delimiter out of the stream but does not make the model
switch into report mode.  53% of draws hit the thinking cap; about a third of *those*
produce no report at all (``final_tokens == 256``, ``finish_reason == "length"``, not one
``<final>`` among them) because the model is mid-derivation when the boundary lands.
Valid reports floor at 0.82 against the 0.95 gate, and since cap hits track item
difficulty the loss is not noise — it biases every arm.  A prompt-wording repair moved it
by less than a point, because a model mid-derivation is not attending to the instruction.

So a cap-hit decision is finished by a second request: the exact phase-1 token prefix
through ``</think>``, with ``<final>`` prefilled, continued through ``/v1/completions``.
The trigger is ``boundary_injected`` alone — never whether phase 1's span happened to
parse — which is both correctness-blind and the faithful reading of §6.2's independent
256-token report allowance under a forced boundary.

Contract: docs/ard36/00_contract.md §2 (WP1 ``generate_decision`` is unchanged from the
caller's side) and §3 (``boundary_injected == (thinking_tokens == budget)`` is exact;
prefix caching is on by default, block 16).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import decoding as D
from agents_scaling.ard.client import (
    ArdChatClient,
    ArdCompletionClient,
    ContextTooLong,
    ContinuationSpec,
    continuation_prefill_ids,
    continuation_seed_key,
    cost_fn_for,
    oracle_for,
    preflight_continuation,
    request_spec,
)
from agents_scaling.ard.report import parse_decision
from agents_scaling.study.types import ProtocolError
from tests.ard import wp1_support as S
from tests.study import fake_vllm as F

CHAT, COMPLETIONS = "/v1/chat/completions", "/v1/completions"


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
    return cost_fn_for(cfg.checkpoints)


@pytest.fixture
def serve(run_root, tokenizer):
    """``with serve(**knobs) as srv:`` — a cap-hitting fake endpoint by default.

    The server and the client under test must share the tokenizer instance, or a decoded
    report comes back as ``<unk>``.
    """

    def _serve(profile=K.PROFILE_SMALL, **kwargs):
        kwargs.setdefault("tokenizer", tokenizer)
        kwargs.setdefault("thinking_words", S.cap_hit_words())
        return S.fake_server(run_root, profile, **kwargs)

    return _serve


# --------------------------------------------------------------------------- the trigger


def test_a_forced_boundary_is_continued_and_a_natural_close_is_not(cfg, run_root, context, cost_fn, serve):
    """The rule is ``boundary_injected``, and nothing else (spec §6.2)."""
    with serve() as srv:
        capped = S.run_decision(S.make_request(cfg, source_id="math500:capped"), run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 1)
    with serve(thinking_words=40) as srv:
        natural = S.run_decision(S.make_request(cfg, source_id="math500:natural"), run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 0)

    assert capped.boundary_injected is True and capped.continued is True
    assert capped.thinking_tokens == capped.budget_tokens == K.THINKING_TOKENS_LOW
    assert natural.boundary_injected is False and natural.continued is False
    assert natural.continuation is None
    assert len(natural.physical_request_ids) == 1
    assert natural.physical_request_ids == (natural.request_id,)


def test_a_naturally_closed_draw_keeps_the_single_call_record_exactly(cfg, run_root, context, cost_fn, serve):
    """Nothing about the untouched path may move: it is 47% of the run."""
    req = S.make_request(cfg)
    with serve(thinking_words=40) as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert srv.completion_count == 0
    stored = S.stored_decision(run_root, req, context)
    assert record.to_dict() | {"aliased": True} == stored.to_dict()
    assert record.final_text.startswith(K.FINAL_OPEN)
    assert parse_decision(record).parse_status is K.ParseStatus.OK


def test_the_trigger_never_looks_at_whether_phase_one_parsed(cfg, run_root, context, cost_fn, serve):
    """A cap-hit draw whose span *did* parse is continued all the same (uniform rule).

    Conditioning on the parse would be a per-draw judgement correlated with difficulty;
    under a forced boundary the genuinely independent report is the continuation.
    """
    req = S.make_request(cfg)
    with serve() as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert srv.completion_count == 1
    phase1 = S.stored_decision(run_root, req, context)
    assert parse_decision(phase1).parse_status is K.ParseStatus.OK   # phase 1 was fine...
    assert record.continued is True                                   # ... and continued anyway
    assert record.final_text != phase1.final_text


def test_a_report_phase_one_never_started_now_parses(cfg, run_root, context, cost_fn, serve):
    """The defect itself: phase 1 keeps deriving, phase 2 delivers the report."""
    req = S.make_request(cfg)
    with serve(responder=S.unreported_derivation()) as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 1)

    phase1 = S.stored_decision(run_root, req, context)
    assert K.FINAL_OPEN not in phase1.final_text          # not even an opening tag
    assert parse_decision(phase1).parse_status is K.ParseStatus.MISSING_FINAL

    assert record.final_text.startswith(K.FINAL_OPEN)
    report = parse_decision(record)
    assert report.parse_status is K.ParseStatus.OK
    assert report.final_answer and report.confidence is not None


# --------------------------------------------------------------------------- the prefix


def test_the_phase_two_prompt_is_the_exact_phase_one_prefix_plus_the_prefill(
    cfg, run_root, context, cost_fn, serve, tokenizer
):
    """Token ids, byte for byte — never a decoded/re-encoded round trip (spec §6.2)."""
    req = S.make_request(cfg)
    with serve(responder=S.unreported_derivation()) as srv:
        S.run_decision(req, run_root, context, cost_fn)
        body = S.sent(srv, COMPLETIONS)[-1]

    phase1 = S.stored_decision(run_root, req, context)
    ids = phase1.completion_token_ids
    end = ids.index(K.THINK_END_ID)
    prefill = continuation_prefill_ids(tokenizer)
    assert prefill == tuple(tokenizer.encode(D.CONTINUATION_PREFILL_TEXT, add_special_tokens=False))
    assert K.FINAL_OPEN in tokenizer.decode(list(prefill))

    expected = tuple(phase1.prompt_token_ids) + ids[: end + 1] + prefill
    assert tuple(body["prompt"]) == expected
    # The prefix ends at the forced boundary: phase 1's derivation *after* it is dropped,
    # and the last prompt tokens are </think> then the prefilled opening.
    assert expected[-len(prefill) - 1] == K.THINK_END_ID
    assert len(ids) > end + 1   # phase 1 did write a span there; it is not in the prefix
    assert K.THINK_END_ID not in expected[-len(prefill):]


def test_the_continuation_body_is_a_completions_body(cfg, run_root, context, cost_fn, serve):
    """vLLM 0.21 completions: ``prompt`` as token ids, ``logprobs`` as a width."""
    req = S.make_request(cfg)
    with serve() as srv:
        S.run_decision(req, run_root, context, cost_fn)
        body = S.sent(srv, COMPLETIONS)[-1]
        chat = S.sent(srv, CHAT)[-1]

    assert body["model"] == chat["model"] == "8B"
    assert isinstance(body["prompt"], list) and all(isinstance(i, int) for i in body["prompt"])
    assert "messages" not in body and "chat_template_kwargs" not in body
    # §6.3: the report allowance, on its own, with no thinking channel to pay for.
    assert body["max_tokens"] == K.FINAL_TOKENS == 256
    assert "thinking_token_budget" not in body
    # The same §6.2 sampler as the decision it continues.
    for field in ("temperature", "top_k", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
        assert body[field] == chat[field]
    assert body["return_token_ids"] is True
    assert body["logprobs"] == 1            # a width, not a flag: see /v1/completions
    assert isinstance(body["seed"], int) and body["seed"] != chat["seed"]


def test_the_receiver_purpose_keeps_its_logprob_width_in_phase_two(cfg, run_root, context, cost_fn, serve):
    """Spec §11's answer-logit margin must survive the continuation (spec §14)."""
    req = S.make_request(
        cfg, purpose=K.PURPOSE_RECEIVER, namespace=K.NS_RECEIVER, source_id="receiver:prog-0007"
    )
    with serve() as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert S.sent(srv, COMPLETIONS)[-1]["logprobs"] == K.RECEIVER_TOP_LOGPROBS == 20

    assert record.answer_logprob is not None
    rows = record.final_top_logprobs
    assert rows is not None
    n_prefill = len(record.continuation.prefill_token_ids)
    # Rows are indexed by the final span, so the harness-prefilled ids lead with empty
    # rows: they were prompt, not generation.
    assert all(row == () for row in rows[:n_prefill])
    assert len(rows[n_prefill]) == K.RECEIVER_TOP_LOGPROBS
    assert [alt["logprob"] for alt in rows[n_prefill]] == sorted(
        (alt["logprob"] for alt in rows[n_prefill]), reverse=True
    )


# --------------------------------------------------------------------------- accounting


def test_two_physical_requests_with_summed_flops(cfg, run_root, context, cost_fn, serve):
    """Cost accounting must never count one request where two were executed (plan §5.8)."""
    req = S.make_request(cfg)
    with serve() as srv:
        record = S.run_decision(req, run_root, context, cost_fn)

    store = S.make_store(run_root)
    first_id, second_id = record.physical_request_ids
    assert len(record.physical_request_ids) == 2
    assert first_id == record.request_id != second_id
    phase1, phase2 = store.get(first_id), store.get(second_id)
    assert phase1 is not None and phase2 is not None

    oracle = oracle_for(cfg.checkpoints[K.CHECKPOINT_SMALL])
    one = oracle.flops_record(phase1.prompt_tokens, len(phase1.response["token_ids"]))
    two = oracle.flops_record(phase2.prompt_tokens, len(phase2.response["token_ids"]))
    assert dict(phase1.flops) == one and dict(phase2.flops) == two
    for key in ("prefill", "decode", "total"):
        assert record.flops[key] == one[key] + two[key] > one[key]
    assert record.flops["oracle"] == oracle.oracle_hash
    assert record.flops["total"] == record.flops["prefill"] + record.flops["decode"]
    # The phase-2 half is kept whole so the split can be recovered from one row.
    assert record.continuation.flops == two
    assert record.continuation.request_id == second_id


def test_the_merged_record_takes_its_thinking_from_one_phase_and_its_report_from_the_other(
    cfg, run_root, context, cost_fn, serve
):
    req = S.make_request(cfg)
    with serve(responder=S.unreported_derivation()) as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        completion = srv.text_completion_payload(S.sent(srv, COMPLETIONS)[-1])
    phase1 = S.stored_decision(run_root, req, context)
    generated = completion["choices"][0]["token_ids"]
    generated = generated[:-1] if generated[-1] == K.IM_END_ID else generated

    assert record.request_id == phase1.request_id
    assert record.prompt_token_ids == phase1.prompt_token_ids
    assert record.prompt_tokens == phase1.prompt_tokens
    assert record.thinking_tokens == phase1.thinking_tokens
    assert record.budget_tokens == phase1.budget_tokens
    assert record.boundary_injected is True and record.think_status is None
    # ``final_tokens`` counts what phase 2 generated; the prefilled opening was prompt.
    assert record.final_tokens == len(generated) != phase1.final_tokens
    assert record.finish_reason == completion["choices"][0]["finish_reason"]
    assert record.cap_hit is (record.finish_reason == "length")
    assert record.latency_s >= phase1.latency_s          # the decision is both calls
    # ``cached_prefill_tokens`` stays phase 1's, so it keeps describing ``prompt_tokens``;
    # phase 2's near-total prefix-cache hit is its own number.
    assert record.cached_prefill_tokens == phase1.cached_prefill_tokens <= record.prompt_tokens
    # The logical completion: phase 1 through the boundary, the prefill, then the report.
    ids = record.completion_token_ids
    assert ids[0] == K.THINK_START_ID
    assert ids.index(K.THINK_END_ID) == record.thinking_tokens + 1
    assert ids[-len(generated):] == tuple(generated)
    assert len(ids) == record.thinking_tokens + 2 + len(record.continuation.prefill_token_ids) + len(generated)


def test_the_continuation_is_its_own_committed_record(cfg, run_root, context, cost_fn, serve):
    """Two records, two ids, no transaction spanning them (store §3.6)."""
    req = S.make_request(cfg)
    with serve() as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
    store = S.make_store(run_root)
    stored = store.get(record.continuation.request_id)
    assert stored is not None and store.get(record.request_id) is not None
    identity = stored.identity
    # The prompt is a token prefix, not chat messages: the prefix hash carries the prompt.
    assert list(stored.messages) == []
    assert identity["prefix_hash"] and len(identity["prefix_hash"]) == 64
    assert tuple(stored.prompt_token_ids)[: len(record.prompt_token_ids)] == record.prompt_token_ids
    assert stored.response["route"] == COMPLETIONS
    assert stored.sampling["max_tokens"] == K.FINAL_TOKENS
    assert stored.sampling["thinking_tokens"] == 0
    assert stored.seed_key.namespace.endswith("|CONTINUATION")
    assert stored.seed_key.purpose == req.seed_key.purpose


# --------------------------------------------------------------------------- endpoints


def test_the_prefix_goes_back_to_the_endpoint_that_prefilled_it(cfg, run_root, context, cost_fn, tokenizer):
    """~1.3k tokens of prefix: the prefix cache is worth aiming at (contract §3)."""
    with S.fake_server(run_root, tokenizer=tokenizer, thinking_words=S.cap_hit_words()) as one, \
         S.fake_server(run_root, tokenizer=tokenizer, thinking_words=S.cap_hit_words()) as two:
        # Without affinity the pool would rotate: it round-robins per request.
        pool = S.make_pool(run_root)
        assert {(e.host, e.port) for e in (pool.pick(), pool.pick())} == {
            (one.host, one.port), (two.host, two.port)
        }
        record = S.run_decision(S.make_request(cfg), run_root, context, cost_fn)
        served = [srv for srv in (one, two) if srv.chat_count == 1]
        idle = [srv for srv in (one, two) if srv not in served]
        assert len(served) == 1
        assert served[0].completion_count == 1 and idle[0].completion_count == 0
        assert record.endpoint_id == f"{served[0].host}:{served[0].port}"

    assert record.continuation.endpoint_pinned is True
    assert record.continuation.reprefilled is False
    assert record.continuation.endpoint_id == record.endpoint_id


def test_the_fallback_is_recorded_when_the_phase_one_endpoint_is_gone(
    cfg, run_root, context, cost_fn, tokenizer
):
    """The pin is a preference, never a dependency: phase 2 completes anyway, and says so.

    This is also the resume shape: phase 1 committed on an endpoint that is gone by the
    time the cell comes back, so the prefix has to be paid again somewhere else.
    """
    req = S.make_request(cfg)
    spec = request_spec(req, context)
    with S.fake_server(run_root, tokenizer=tokenizer, thinking_words=S.cap_hit_words()) as gone:
        # Commit phase 1 alone, through the chat client, then take its endpoint away.
        S.make_store(run_root).get_or_generate(
            spec,
            ArdChatClient(S.make_pool(run_root), tokenizer, cost_fn=cost_fn),
            "phase-one-only",
        )
        assert (gone.chat_count, gone.completion_count) == (1, 0)
        host, port = gone.host, gone.port
    F.unregister_fake(run_root, K.PROFILE_SMALL, host, port)

    with S.fake_server(run_root, tokenizer=tokenizer, thinking_words=S.cap_hit_words()) as fresh:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert (fresh.chat_count, fresh.completion_count) == (0, 1)   # phase 1 aliased

    assert record.continued is True
    assert record.continuation.endpoint_pinned is False
    assert record.continuation.reprefilled is True
    assert record.continuation.endpoint_id == f"{fresh.host}:{fresh.port}"
    assert record.endpoint_id == f"{host}:{port}" != record.continuation.endpoint_id
    assert parse_decision(record).parse_status is K.ParseStatus.OK


# --------------------------------------------------------------------------- resume


def test_resume_issues_no_new_requests(cfg, run_root, context, cost_fn, serve):
    """A resumed cell measures exactly what the original run measured (spec §3.6)."""
    req = S.make_request(cfg)
    with serve(responder=S.unreported_derivation()) as srv:
        first = S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 1)
        second = S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 1)   # nothing was regenerated

    assert first.aliased is False and second.aliased is True
    assert second.continuation.aliased is True
    # Nothing was issued on the resume, so there is no endpoint decision to report.
    assert second.continuation.endpoint_pinned is None
    assert second.continuation.reprefilled is False

    def measured(record):
        row = record.to_dict()
        row["aliased"] = None
        row["continuation"] = dict(row["continuation"], aliased=None, endpoint_pinned=None)
        return row

    assert measured(first) == measured(second)


# --------------------------------------------------------------------------- identity


def test_the_prefix_is_the_continuation_identity(cfg, context):
    """Everything sent to the engine is inside the id (contract §0.3)."""
    base = request_spec(S.make_request(cfg), context)
    one = ContinuationSpec(
        messages=(),
        decoding=D.CONTINUATION_DECODING,
        checkpoint=base.checkpoint,
        seed_key=continuation_seed_key(base.seed_key),
        role=D.CONTINUATION_ROLE,
        study_id=base.study_id,
        study_seed_hex=base.study_seed_hex,
        prompt_ids=(1, 2, 3, K.THINK_END_ID, 7),
    )
    two = replace(one, prompt_ids=(1, 2, 3, K.THINK_END_ID, 8))
    assert one.prefix_hash != two.prefix_hash
    assert one.request_id != two.request_id
    assert len(one.request_id) == 64 and int(one.request_id, 16) >= 0
    assert one.identity_fields()["prefix_hash"] == one.prefix_hash
    # A continuation can never alias the decision it continues.
    assert one.request_id != base.request_id
    assert one.seed_key != base.seed_key and one.engine_seed != base.engine_seed
    with pytest.raises(ProtocolError, match="token prefix"):
        ContinuationSpec(
            messages=(), decoding=D.CONTINUATION_DECODING, checkpoint=base.checkpoint,
            seed_key=one.seed_key, role=D.CONTINUATION_ROLE, study_id=base.study_id,
            study_seed_hex=base.study_seed_hex, prompt_ids=(),
        )


def test_the_continuation_decoding_asks_for_the_report_allowance_only(cfg, context):
    decoding = D.CONTINUATION_DECODING
    assert decoding.max_tokens == decoding.final_tokens == K.FINAL_TOKENS
    assert decoding.thinking_tokens == 0 and decoding.enable_thinking is False
    assert (decoding.temperature, decoding.top_p, decoding.top_k) == (0.6, 0.95, 20)
    assert D.is_continuation(decoding) and not D.is_continuation(D.LOW_DECODING)
    base = request_spec(S.make_request(cfg), context)
    wrong = ContinuationSpec(
        messages=(), decoding=D.LOW_DECODING, checkpoint=base.checkpoint,
        seed_key=continuation_seed_key(base.seed_key), role=D.CONTINUATION_ROLE,
        study_id=base.study_id, study_seed_hex=base.study_seed_hex, prompt_ids=(1, 2, 3),
    )
    with pytest.raises(ProtocolError, match="report allowance"):
        ArdCompletionClient.request_kwargs(wrong)


def test_an_oversized_prefix_is_refused_before_the_request(cfg, context):
    """The prefix is a whole prompt *plus* a whole thinking span: check it on its own."""
    base = request_spec(S.make_request(cfg), context)
    spec = ContinuationSpec(
        messages=(), decoding=D.CONTINUATION_DECODING, checkpoint=base.checkpoint,
        seed_key=continuation_seed_key(base.seed_key), role=D.CONTINUATION_ROLE,
        study_id=base.study_id, study_seed_hex=base.study_seed_hex,
        prompt_ids=tuple(range(1000, 1000 + K.MAX_MODEL_LEN)),
    )
    with pytest.raises(ContextTooLong) as excinfo:
        preflight_continuation(spec, context, size=K.CHECKPOINT_SMALL)
    preflight = excinfo.value.preflight
    assert preflight.requested_output_tokens == K.FINAL_TOKENS
    assert preflight.reserve_tokens == K.CONTEXT_RESERVE_TOKENS
    assert preflight.required_tokens > K.MAX_MODEL_LEN and not preflight.fits
    assert excinfo.value.parse_status is K.ParseStatus.CONTEXT_FAILURE
    short = replace(spec, prompt_ids=(1, 2, 3))
    assert preflight_continuation(short, context, size=K.CHECKPOINT_SMALL).fits


# --------------------------------------------------------------------------- failure modes


def test_a_report_that_runs_out_of_allowance_is_truncated_not_silently_kept(
    cfg, run_root, context, cost_fn, serve
):
    """Phase 2 has the §6.3 allowance and no more; a cut span is ``TRUNCATED``, never OK."""

    def responder(request):
        if request["mode"] != "ard_continuation":
            return None
        return " ".join(f"w{i}" for i in range(400))   # a report tail far past the allowance

    req = S.make_request(cfg)
    with serve(responder=responder) as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert srv.completion_count == 1

    assert record.final_tokens == K.FINAL_TOKENS == 256
    assert record.finish_reason == "length" and record.cap_hit is True
    assert parse_decision(record).parse_status is K.ParseStatus.TRUNCATED


def test_the_frozen_retry_policy_still_governs_phase_two(cfg, run_root, context, cost_fn, serve):
    """§10.4: an exogenous fault on the continuation is retried, and the pin is given up."""
    req = S.make_request(cfg)
    with serve() as srv:
        srv.fail_next(1, status=500, path=COMPLETIONS)
        record = S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 2)

    assert record.continued is True
    assert record.continuation.endpoint_pinned is False   # the pinned endpoint had failed
    assert record.continuation.reprefilled is True
    assert parse_decision(record).parse_status is K.ParseStatus.OK


def test_a_failed_continuation_leaves_phase_one_committed(cfg, run_root, context, cost_fn, serve):
    """The two records commit independently, so a resume redoes only the missing half."""
    from agents_scaling.study.types import InfraFailure

    req = S.make_request(cfg)
    with serve() as srv:
        srv.fail_next(3, status=500, path=COMPLETIONS)   # exhausts phase 2, leaves phase 1 alone
        with pytest.raises(InfraFailure):
            S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (1, 3)   # 1 + the two §10.4 retries
    store = S.make_store(run_root)
    assert store.get(request_spec(req, context).request_id) is not None

    with serve() as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert (srv.chat_count, srv.completion_count) == (0, 1)   # only the missing half
    assert record.continued is True
    assert store.get(record.continuation.request_id) is not None


def test_the_prefix_cache_hit_on_the_continuation_is_recorded(cfg, run_root, context, cost_fn, serve):
    """What the endpoint pin is for: the re-sent prefix comes back as a cache hit.

    vLLM reports hits in whole 16-token blocks (contract §3), and they are the phase-2
    prefix's, not the decision prompt's — which is why they are kept on the continuation.
    """
    req = S.make_request(cfg)
    with serve(cached_prefill_ratio=1.0) as srv:
        record = S.run_decision(req, run_root, context, cost_fn)
        assert srv.completion_count == 1
    blocks = record.continuation.prefix_tokens // F.PREFIX_CACHE_BLOCK
    assert record.continuation.cached_prefill_tokens == blocks * F.PREFIX_CACHE_BLOCK > 0
    assert record.continuation.cached_prefill_tokens > record.prompt_tokens
    assert record.cached_prefill_tokens <= record.prompt_tokens
