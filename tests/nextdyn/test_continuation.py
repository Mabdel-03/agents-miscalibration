"""WP1: the two-phase report continuation, and its parity with the ARD36 original.

Plan D5 / amendment A3.  A forced ``</think>`` takes the delimiter out of the stream but does
not put the model into report mode, so a cap-hit draw is finished by a second physical request
from the exact phase-1 token prefix.  The trigger is ``boundary_injected`` **alone**: a rule
that continued only the draws whose report failed to parse would be a per-draw judgement
correlated with item difficulty, and every arm would be biased by it.

``agents_scaling.ard.client`` is not on the ``nextdyn.ARD_IMPORTS`` allowlist, so
``continuation.py`` is a vendored parametric copy of ``ard/client.py:978-1358``.  The vendoring
discipline of this repo is that a copy carries a parity test:
:func:`test_vendored_continuation_matches_ard_on_the_ard_contract` runs the ARD36 original and
the copy on the *same* fake-server phase-1 record under the ARD36 constants and asserts the two
agree down to the phase-2 request id, the merged token ids and the summed FLOPs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import continuation as CON
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn import inference as INF
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.resources.oracle import FlopOracle, make_cost_fn
from tests.nextdyn.conftest import (
    NDJSON_OPEN,
    WP1_PROFILE,
    nd_json_responder,
    nd_request,
    nd_unreported_derivation,
)

CHAT, COMPLETIONS = "/v1/chat/completions", "/v1/completions"
REPO_ROOT = Path(__file__).resolve().parents[2]


def sent(server, path):
    return [r["body"] for r in server.requests if r["path"] == path]


# --------------------------------------------------------------------------- the trigger


def test_continuation_fires_only_on_boundary_injected_and_ignores_phase1_content(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """The rule is the forced boundary, and nothing else (plan D5).

    Two cap-hit draws: one whose phase-1 span happened to be a perfectly good JSON object, one
    that never opened the object at all.  Both are continued, because conditioning on the parse
    would correlate the treatment with difficulty.  A naturally closed draw is untouched.
    """
    ctx = nd_ctx()
    capped = nd_request(nd_contract, nd_tokens, draw=0)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        parsed_anyway = INF.generate_decision(capped, ctx=ctx)
        assert (srv.chat_count, srv.completion_count) == (1, 1)
    assert parsed_anyway.boundary_injected is True and parsed_anyway.continued is True

    ctx2 = nd_ctx()
    never_reported = nd_request(nd_contract, nd_tokens, draw=1)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64,
                  responder=nd_unreported_derivation()) as srv:
        record = INF.generate_decision(never_reported, ctx=ctx2)
        assert (srv.chat_count, srv.completion_count) == (1, 1)

    # Phase 1 kept deriving and never opened the object; phase 2 delivered one.
    phase1 = ctx2.store.get(record.physical_request_ids[0])
    assert "{" not in "".join(phase1.response["content"] or "")
    assert record.final_text.strip().startswith("{")
    assert record.continued is True and record.n_physical_requests == 2


def test_naturally_closed_draw_uses_single_call(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Nothing about the untouched path may move: it is about half of the run."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4) as srv:
        record = INF.generate_decision(req, ctx=ctx)
        assert (srv.chat_count, srv.completion_count) == (1, 0)

    assert record.boundary_injected is False and record.natural_stop is True
    assert record.continuation is None and record.continued is False
    assert record.n_physical_requests == 1
    assert record.discarded_span_tokens == 0
    assert record.request_ids == record.physical_request_ids


def test_the_envelope_contract_is_never_continued(nd_contract, nd_tokens):
    """Spec 5.6's companion has no thinking channel, so it has no boundary to force."""
    adapter = A.adapter_for(nd_contract.model("qwen_nonthinking"))
    assert adapter.continuation_supported() is False


def test_continuing_a_naturally_closed_draw_is_refused(nd_serve, nd_ctx, nd_contract, nd_tokens):
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4):
        record = INF.generate_decision(req, ctx=ctx)
    spec = INF.request_spec(req, ctx)
    with pytest.raises(ContractViolation, match="naturally"):
        CON.continue_from(record, ctx.store.get(spec.request_id), spec=spec,
                          cc=CON.ContinuationContract.of(nd_tokens, D.prefill_text()),
                          sampling=nd_contract.sampling, tokenizer=ctx.tokenizer(req.model),
                          store=ctx.store, pool=ctx.pool(req.model), cost_fn=ctx.cost_fn,
                          failure_policy=nd_contract.failure_policy, top_logprobs=1,
                          profile_name=WP1_PROFILE, cell_id="c", run_id="r")


# --------------------------------------------------------------------------- the prefix


def test_the_phase_two_prompt_is_the_exact_phase_one_prefix_plus_the_prefill(
        nd_serve, nd_ctx, nd_contract, nd_tokens, fake_tokenizer):
    """Token ids, byte for byte -- never a decoded/re-encoded round trip (plan D5)."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        record = INF.generate_decision(req, ctx=ctx)
        body = sent(srv, COMPLETIONS)[0]

    phase1 = ctx.store.get(record.physical_request_ids[0])
    ids = list(phase1.response["token_ids"])
    _, think_end = A.delimiter_ids(fake_tokenizer)
    cut = ids.index(think_end) + 1
    prefill = list(D.continuation_prefill_ids(fake_tokenizer))
    assert body["prompt"] == list(phase1.prompt_token_ids) + ids[:cut] + prefill
    # The phase-1 span after the forced boundary is NOT re-sent; its length is recorded so the
    # GPU seconds that produced it stay attributable (defect D-02).
    assert record.discarded_span_tokens == len(ids) - cut - 1      # minus the turn token
    assert record.discarded_span_tokens > 0


def test_the_merged_record_opens_with_the_prefilled_bytes(nd_serve, nd_ctx, nd_contract,
                                                          nd_tokens):
    """The prefilled opening was *prompt*, so it is part of the delivered object's text but not
    part of the tokens phase 2 generated."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    assert record.final_text.startswith(NDJSON_OPEN.strip()[0])
    assert '"reasoning"' in record.final_text
    assert record.continuation["prefill_token_ids"]
    assert record.continuation["generated_tokens"] == record.final_tokens


def test_one_sample_one_generation_uid_over_both_physical_requests(nd_serve, nd_ctx, nd_contract,
                                                                   nd_tokens):
    """Plan §1.3: a continued draw is ONE sample; both ids go in, one uid comes out."""
    from agents_scaling.nextdyn import ids as I

    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)

    assert len(set(record.physical_request_ids)) == 2
    assert record.generation_uid == I.generation_uid(record.physical_request_ids)
    assert record.request_ids == record.physical_request_ids
    assert record.flops["total"] > 0


def test_the_flops_of_a_continued_decision_are_the_sum_over_both_requests(nd_serve, nd_ctx,
                                                                          nd_contract, nd_tokens):
    """Cost accounting must never count one request where two were issued (defect D-02)."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    phase1 = ctx.store.get(record.physical_request_ids[0])
    phase2 = ctx.store.get(record.physical_request_ids[1])
    for key in ("prefill", "decode", "total"):
        assert record.flops[key] == phase1.flops[key] + phase2.flops[key]
    with pytest.raises(Exception, match="oracle"):
        CON.sum_flops({"oracle": "a", "prefill": 1, "decode": 1, "total": 2},
                      {"oracle": "b", "prefill": 1, "decode": 1, "total": 2})


def test_a_cell_that_died_between_the_phases_resumes_by_aliasing_phase_one(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Phase 2 commits as its own store record: the store has no transaction spanning two."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        first = INF.generate_decision(req, ctx=ctx)
        assert (srv.chat_count, srv.completion_count) == (1, 1)
        again = INF.generate_decision(req, ctx=nd_ctx(cell_id="episode/resumed"))
        assert (srv.chat_count, srv.completion_count) == (1, 1)     # nothing re-issued

    assert again.aliased is True and first.aliased is False
    assert again.physical_request_ids == first.physical_request_ids
    assert again.completion_token_ids == first.completion_token_ids
    assert again.final_text == first.final_text


def test_the_continuation_prefers_the_endpoint_that_just_prefilled_the_prefix(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """A continuation re-sends the whole thinking span; a prefix-cache miss pays it twice."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    assert record.continuation["endpoint_pinned"] is True
    assert record.continuation["reprefilled"] is False


# --------------------------------------------------------------------------- vendoring parity


def test_vendored_continuation_matches_ard_on_the_ard_contract(tmp_path, fake_tokenizer):
    """The vendoring parity test (plan §1.11 VENDORED table).

    ``continuation.py`` is ``ard/client.py:978-1358`` with every ARD36 constant turned into a
    parameter.  Handed those constants back -- the 256-token report allowance, the
    ``\\n\\n<final>`` prefill, ARD36's continuation id tag and namespace suffix -- the copy must
    behave as the original: the same phase-2 request id (so it would alias in a shared store),
    the same body on the wire, the same merged token ids and the same summed FLOPs.
    """
    from agents_scaling.ard import contract as KA
    from agents_scaling.ard import decoding as DA
    from agents_scaling.ard.client import (
        ArdChatClient,
        CONTINUATION_ID_TAG,
        CONTINUATION_NAMESPACE_SUFFIX,
        DecisionContext,
        continue_decision,
        decision_from_record,
    )
    from agents_scaling.serving.profiles import get_serving_profile
    from agents_scaling.study.config import load_config
    from agents_scaling.study.inference.client import EndpointPool
    from agents_scaling.study.types import RequestSpec, SeedKey
    from tests.study.fake_vllm import FakeVllmServer

    budget = 16                     # the ARD36 shape at a size a whitespace tokenizer can reach
    cfg = load_config(REPO_ROOT / "configs" / "ard36.yaml")
    checkpoint = cfg.checkpoints[KA.CHECKPOINT_SMALL]
    profile = get_serving_profile(checkpoint.profile)
    run_root = tmp_path / "run"
    (run_root / "servers").mkdir(parents=True)
    cost_fn = make_cost_fn({size: FlopOracle.from_table(size) for size in cfg.checkpoints})

    spec = RequestSpec(
        messages=({"role": "user", "content": "Solve the task . Task: what is 2 + 2 ?"},),
        decoding=DA.ard_decoding(budget),
        checkpoint=checkpoint,
        seed_key=SeedKey(source_id="math500:parity", split="dev",
                         model_cell=checkpoint.model_cell, episode_rep=0, actor_slot=0,
                         purpose=KA.PURPOSE_ROOT, step_slot=0, namespace=KA.NS_STATELESS_BANK),
        role=DA.ARD_DECODING_ROLE, study_id=cfg.study_id, study_seed_hex=cfg.study_seed_hex)

    def report_tail(payload):
        """One ARD36 report minus its opening tag (the prompt already carries it)."""
        if "prompt_token_ids" in payload:
            return "4 </final> <evidence> because 2 + 2 = 4 </evidence> <confidence> 0.9 </confidence>"
        return "still deriving , no report yet"

    server = FakeVllmServer(tokenizer=fake_tokenizer,
                            served_model_name=checkpoint.served_model_name,
                            max_model_len=profile.max_model_len, responder=report_tail,
                            thinking_words=budget + 32).start()
    server.register(run_root, checkpoint.profile)
    try:
        pool = EndpointPool(run_root, checkpoint.profile, shard=0, refresh_min_interval_s=0.0,
                            probe_timeout=2.0)
        client = ArdChatClient(pool, fake_tokenizer, cost_fn=cost_fn, run_id="parity")
        shared = RequestStore(run_root / "requests")
        phase1, _ = shared.get_or_generate(spec, client, "parity")
        assert server.completion_count == 0

        # ---- the ARD36 original -------------------------------------------------------
        context = DecisionContext(checkpoints=dict(cfg.checkpoints),
                                  tokenizers={s: fake_tokenizer for s in cfg.checkpoints},
                                  study_seed_hex=cfg.study_seed_hex, study_id=cfg.study_id,
                                  run_id="parity")
        ard_first = decision_from_record(phase1, budget_tokens=budget, tokenizer=fake_tokenizer)
        assert ard_first.boundary_injected is True
        ard_merged = continue_decision(
            ard_first, phase1, spec=spec, size=KA.CHECKPOINT_SMALL, context=context,
            store=RequestStore(run_root / "requests_ard"), pool=pool, cost_fn=cost_fn,
            client=client, cell_id="parity")
        ard_body = sent(server, COMPLETIONS)[-1]

        # ---- the vendored copy, handed the ARD36 constants ----------------------------
        ard_tokens = K.TokenContract(thinking_cap=budget, final_cap=KA.FINAL_TOKENS,
                                     served_window=profile.max_model_len,
                                     reserve=KA.CONTEXT_RESERVE_TOKENS, packet_cap=512,
                                     rationale_cap=256, claims_max=4, claim_cap=64)
        cc = CON.ContinuationContract(tokens=ard_tokens,
                                      prefill_text=DA.CONTINUATION_PREFILL_TEXT,
                                      id_tag=CONTINUATION_ID_TAG,
                                      namespace_suffix=CONTINUATION_NAMESPACE_SUFFIX,
                                      role=DA.CONTINUATION_ROLE)
        sampling = K.Sampling.from_floats(temperature=KA.TEMPERATURE, top_p=KA.TOP_P,
                                          top_k=KA.TOP_K, min_p=KA.MIN_P,
                                          repetition_penalty=KA.REPETITION_PENALTY,
                                          presence_penalty=KA.PRESENCE_PENALTY,
                                          frequency_penalty=0.0)
        split = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING].split(phase1, fake_tokenizer)
        nd_first = INF.decision_from_record(phase1, split, spec=spec, tokenizer=fake_tokenizer,
                                            replica=INF.replica_of(pool, phase1), aliased=False)
        assert nd_first.boundary_injected is True
        nd_merged = CON.continue_from(
            nd_first, phase1, spec=spec, cc=cc, sampling=sampling, tokenizer=fake_tokenizer,
            store=RequestStore(run_root / "requests_nd"), pool=pool, cost_fn=cost_fn,
            failure_policy=_ard_shaped_policy(), top_logprobs=1, profile_name=profile.name,
            cell_id="parity", run_id="parity")
        nd_body = sent(server, COMPLETIONS)[-1]
    finally:
        server.stop()

    # The identity: same prefix, same decoding, same seed, same domain tag -> the same id, so
    # the two would alias each other in one store.
    assert nd_merged.physical_request_ids[1] == ard_merged.physical_request_ids[1]
    assert nd_body["prompt"] == ard_body["prompt"]
    assert nd_body["seed"] == ard_body["seed"]
    assert nd_body["max_tokens"] == ard_body["max_tokens"] == KA.FINAL_TOKENS
    # The merge itself.
    assert nd_merged.completion_token_ids == ard_merged.completion_token_ids
    assert nd_merged.final_text == ard_merged.final_text
    assert nd_merged.thinking_tokens == ard_merged.thinking_tokens == budget
    assert dict(nd_merged.flops) == dict(ard_merged.flops)
    assert nd_merged.flops["total"] > 0


def _ard_shaped_policy() -> K.FailurePolicy:
    """ARD36's §10.4 retry policy, expressed as this study's compiled failure policy."""
    return K.FailurePolicy(transient_retries=2, retry_same_request_and_seed=True,
                           model_output_repair_calls=0, wrong_answer_retry=False,
                           context_overflow="typed_failure", unresolved_semantics="unknown",
                           null_item_ids="fatal", duplicate_logical_keys="fatal",
                           hidden_label_leakage="fatal")
