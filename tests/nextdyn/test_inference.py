"""WP1: the generation front door, offline against the fake vLLM 0.21 server.

Spec 18.5's serving names live here.  Each one is a rule that ARD36 or study_v4 learned the
hard way: the reasoning cap is the engine's, not the harness's, so it must be *observed* at the
declared boundary; the final allowance must survive a cap hit; an infrastructure retry must be
the same request with the same seed, or it is a second sample; and a malformed object is an
observation, never a reason to sample again.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn import inference as INF
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study.inference.client import COMPLETED_FINISH_REASONS
from agents_scaling.study.types import InfraFailure
from tests.nextdyn.conftest import (
    WP1_PROFILE,
    nd_json_responder,
    nd_key,
    nd_messages,
    nd_request,
)

CHAT, COMPLETIONS = "/v1/chat/completions", "/v1/completions"
GUIDED_FIELDS = ("guided_json", "guided_regex", "guided_grammar", "guided_choice",
                 "response_format", "structural_tag")


def sent(server, path=CHAT):
    return [r["body"] for r in server.requests if r["path"] == path]


# --------------------------------------------------------------------------- spec 18.5


def test_reasoning_cap_is_enforced_at_declared_boundary(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Spec 18.5.  The budget goes on the wire as ``extra_body.thinking_token_budget`` and the
    engine forces ``</think>`` the moment the thinking count reaches it, so the observed span
    is exactly the declared cap -- never one token more, and never a harness-side trim."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        record = INF.generate_decision(req, ctx=ctx)
        body = sent(srv)[0]

    assert body["thinking_token_budget"] == nd_tokens.thinking_cap
    assert record.thinking_tokens == nd_tokens.thinking_cap
    assert record.boundary_injected is True and record.natural_stop is False


def test_final_phase_has_its_reserved_cap(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Spec 18.5.  A forced boundary is the *thinking* budget; the final allowance is
    independent and is still there (amendment A1 buys the two delimiters separately)."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        record = INF.generate_decision(req, ctx=ctx)
        chat = sent(srv)[0]
        completion = sent(srv, COMPLETIONS)[0]

    assert chat["max_tokens"] == 1 + nd_tokens.thinking_cap + 1 + nd_tokens.final_cap
    # The continuation asks for the bare final allowance: no delimiters left to pay for.
    assert completion["max_tokens"] == nd_tokens.final_cap
    assert 0 < record.final_tokens <= nd_tokens.final_cap
    assert record.final_text.strip().startswith("{")


def test_pinned_backend_response_fields_supported(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Spec 18.5.  vLLM 0.21's shape, re-verified live on 2026-09-05: the thinking channel is
    ``choices[0].message.reasoning`` (``reasoning_content`` is the fallback name, absent here),
    ``prompt_token_ids`` is top-level, ``token_ids`` is per choice, and the prefix-cache hit
    count is ``usage.prompt_tokens_details.cached_tokens``."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(cached_prefill_ratio=0.5):
        INF.generate_decision(req, ctx=ctx)
    stored = ctx.store.get(INF.request_spec(req, ctx).request_id)

    assert stored is not None
    assert stored.response["reasoning_field_name"] == "reasoning"
    assert stored.response["reasoning"]
    assert stored.prompt_token_ids and stored.response["token_ids"]
    assert stored.response["finish_reason"] in COMPLETED_FINISH_REASONS
    assert stored.response["cached_prefill_tokens"] > 0
    assert stored.response["token_logprobs"] is not None


def test_infrastructure_retry_keeps_exact_request_and_seed(nd_serve, nd_ctx, nd_contract,
                                                           nd_tokens):
    """Spec 18.5 / spec 15.3.  A retry that changed anything would be a second *sample* of the
    same logical decision, so the two bodies must be byte-identical -- seed included."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    assert ctx.contract.failure_policy.retry_same_request_and_seed is True
    with nd_serve() as srv:
        srv.fail_next(1, 500, path=CHAT)
        record = INF.generate_decision(req, ctx=ctx)
        bodies = sent(srv)

    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert bodies[0]["seed"] == INF.request_spec(req, ctx).engine_seed
    assert len(record.attempts) == 2
    assert record.attempts[0]["error"]["type"] and record.attempts[1]["error"] is None
    assert record.physical_request_ids == (INF.request_spec(req, ctx).request_id,)


def test_malformed_model_output_is_not_stochastically_retried(nd_serve, nd_ctx, nd_contract,
                                                              nd_tokens):
    """Spec 18.5 / spec 4.6.  ``stop``/``length`` are terminal whatever the text turned out to
    be: a refusal or a malformed object scores zero delivery success, it does not buy a redraw
    (``failure_policy.model_output_repair_calls == 0``)."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(responder=lambda p: "not json at all , not even close") as srv:
        record = INF.generate_decision(req, ctx=ctx)
        assert (srv.chat_count, srv.completion_count) == (1, 0)

    assert ctx.contract.failure_policy.model_output_repair_calls == 0
    assert ctx.contract.failure_policy.wrong_answer_retry is False
    assert record.finish_reason in COMPLETED_FINISH_REASONS
    assert record.n_physical_requests == 1
    assert "not json" in record.final_text


def test_worst_case_context_fits_every_compared_model(nd_ctx, nd_contract, fake_tokenizer):
    """Spec 18.5.  Both compared cells are served at the *same* declared window (spec 5.5: never
    quietly give only 32B a smaller one), so the worst-case revision prompt must preflight on
    each of them under the real token contract."""
    ctx = nd_ctx()
    tokens = nd_contract.tokens
    worst_case = " ".join(f"w{i}" for i in range(3100))          # plan D5's worst-case estimate
    for name in ("primary", "scale"):
        cell = nd_contract.model(name)
        key = nd_key(nd_contract, cell, tokens)
        req = INF.DecisionRequest(
            messages=({"role": "user", "content": worst_case},), model=cell, adapter=cell.adapter,
            tokens=tokens, seed_key=I.seed_key_for(key, split_role="pilot"), purpose=key.phase,
            top_logprobs=1, key=key)
        result = INF.preflight(req, ctx)
        assert result.fits
        assert result.served_context == tokens.served_window == 32768
        assert result.requested_output_tokens == tokens.max_new_tokens
        assert result.reserve_tokens == tokens.reserve

    over = " ".join(f"w{i}" for i in range(40000))
    cell = nd_contract.model("primary")
    key = nd_key(nd_contract, cell, tokens)
    req = INF.DecisionRequest(
        messages=({"role": "user", "content": over},), model=cell, adapter=cell.adapter,
        tokens=tokens, seed_key=I.seed_key_for(key, split_role="pilot"), purpose=key.phase,
        top_logprobs=1, key=key)
    with pytest.raises(INF.ContextTooLong) as excinfo:
        INF.preflight(req, ctx)
    assert excinfo.value.preflight.required_tokens > tokens.served_window
    assert nd_contract.failure_policy.context_overflow == "typed_failure"


# --------------------------------------------------------------------------- the wire


def test_request_body_has_no_guided_decoding_fields(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Plan D6: guided decoding is accepted but NOT enforced on vLLM 0.21 (measured), so
    ``protocol.final_phase_constrained_decoding`` is frozen false and nothing guided is sent --
    on either route."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64) as srv:
        INF.generate_decision(req, ctx=ctx)
        bodies = sent(srv, CHAT) + sent(srv, COMPLETIONS)

    assert len(bodies) == 2
    for body in bodies:
        flat = json.dumps(body)
        for field in GUIDED_FIELDS:
            assert field not in flat, f"{field} reached the wire"
    client = ctx.client(req)
    with pytest.raises(ContractViolation, match="guided"):
        client.request_kwargs(INF.request_spec(req, ctx), {"type": "object"})


def test_the_request_body_carries_the_logprob_width_and_the_return_token_ids_flag(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Everything sent is inside the id or is a fixed per-phase argument; nothing is optional."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens, top_logprobs=20)
    with nd_serve() as srv:
        INF.generate_decision(req, ctx=ctx)
        body = sent(srv)[0]
    # The OpenAI SDK merges ``extra_body`` into the top-level body, so this is the shape vLLM
    # actually receives.
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["return_token_ids"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["model"] == nd_contract.model("primary").served_model_name


# --------------------------------------------------------------------------- the failure policy


def test_a_config_that_allowed_a_repair_call_cannot_build_a_client(nd_contract, fake_tokenizer,
                                                                   nd_pool):
    """Spec 15.3 asserted at construction, not documented in a comment."""
    policy = nd_contract.failure_policy
    adapter = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING]
    pool = nd_pool()
    assert INF.NextdynChatClient(pool, fake_tokenizer, adapter=adapter, failure_policy=policy,
                                 top_logprobs=1).max_exogenous_retries == policy.transient_retries
    for field, value, message in (("retry_same_request_and_seed", False, "second sample"),
                                  ("model_output_repair_calls", 1, "repair"),
                                  ("wrong_answer_retry", True, "observation"),
                                  ("context_overflow", "retry", "typed failure")):
        broken = object.__new__(K.FailurePolicy)
        for f in policy.__dataclass_fields__:
            object.__setattr__(broken, f, getattr(policy, f))
        object.__setattr__(broken, field, value)
        with pytest.raises(ContractViolation, match=message):
            INF.NextdynChatClient(pool, fake_tokenizer, adapter=adapter, failure_policy=broken,
                                  top_logprobs=1)


def test_a_retry_cap_exhausted_is_an_infra_failure_not_a_silent_null(nd_serve, nd_ctx,
                                                                     nd_contract, nd_tokens):
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve() as srv:
        srv.fail_next(1 + nd_contract.failure_policy.transient_retries, 500, path=CHAT)
        with pytest.raises(InfraFailure):
            INF.generate_decision(req, ctx=ctx)
    assert ctx.store.get(INF.request_spec(req, ctx).request_id) is None


def test_a_timeout_that_may_have_completed_is_marked_uncertain(nd_ctx):
    """Spec 15.3: the sample is well defined (same request, same seed) but the replica may have
    generated twice, and the GPU-second ledger must not pretend otherwise."""
    assert INF.uncertain_execution_of([{"error": {"type": "APITimeoutError"}}, {"error": None}])
    assert not INF.uncertain_execution_of([{"error": {"type": "InternalServerError"}},
                                           {"error": None}])
    assert not INF.uncertain_execution_of([{"error": None}])


# --------------------------------------------------------------------------- identity and cost


def test_a_committed_record_is_aliased_not_regenerated(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """R5 / spec 17.2: a second cell asking for the same request spends no GPU work."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve() as srv:
        first = INF.generate_decision(req, ctx=ctx)
        second = INF.generate_decision(req, ctx=nd_ctx(cell_id="episode/other"))
        assert srv.chat_count == 1

    assert first.aliased is False and second.aliased is True
    assert first.generation_uid == second.generation_uid
    assert first.completion_token_ids == second.completion_token_ids


def test_the_flop_placeholder_is_refused_before_it_can_be_committed(nd_serve, nd_ctx, nd_contract,
                                                                    nd_tokens, tmp_run_root,
                                                                    fake_tokenizer, nd_pool):
    """A committed placeholder would alias forever and the run would end with no FLOP number."""
    from agents_scaling.study.inference.store import RequestStore
    from agents_scaling.study.types import ProtocolError

    ctx = INF.DispatchContext(store=RequestStore(tmp_run_root / "requests"),
                              pool_for=lambda p: nd_pool(), cost_fn=None,
                              tokenizer_for=lambda n: fake_tokenizer, contract=nd_contract,
                              cell_id="donor_bank/test", run_id="t")
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve():
        with pytest.raises(ProtocolError, match="FLOP placeholder"):
            INF.generate_decision(req, ctx=ctx)


def test_the_producer_stamp_records_the_endpoint_the_request_actually_landed_on(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """ARD36 stamped ``{cell_id, run_id, slurm_job_id, host}`` and could not answer "which GPU
    type did this cost land on" without re-joining the registry by hand."""
    ctx = nd_ctx(cell_id="donor_bank/MATH/0")
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve() as srv:
        INF.generate_decision(req, ctx=ctx)
    stored = ctx.store.get(INF.request_spec(req, ctx).request_id)
    assert set(stored.producer) == {"cell_id", "run_id", "slurm_job_id", "host", "endpoint_id",
                                    "gpu_type", "partition"}
    assert stored.producer["cell_id"] == "donor_bank/MATH/0"
    assert stored.producer["endpoint_id"] == f"{srv.host}:{srv.port}"


def test_a_request_whose_seed_key_names_another_model_is_refused(nd_ctx, nd_contract, nd_tokens):
    """Spec 17.2: the generating model is in every key, and the seed key carries its hash."""
    ctx = nd_ctx()
    cell = nd_contract.model("primary")
    key = nd_key(nd_contract, cell, nd_tokens)
    other = I.seed_key_for(nd_key(nd_contract, nd_contract.model("scale"), nd_tokens),
                           split_role="pilot")
    req = INF.DecisionRequest(messages=nd_messages(), model=cell, adapter=cell.adapter,
                              tokens=nd_tokens, seed_key=other, purpose=key.phase, top_logprobs=1,
                              key=key)
    with pytest.raises(ContractViolation, match="model_cell"):
        INF.request_spec(req, ctx)


def test_a_request_whose_purpose_disagrees_with_its_key_is_refused(nd_contract, nd_tokens):
    cell = nd_contract.model("primary")
    key = nd_key(nd_contract, cell, nd_tokens)
    with pytest.raises(ContractViolation, match="purpose"):
        INF.DecisionRequest(messages=nd_messages(), model=cell, adapter=cell.adapter,
                            tokens=nd_tokens,
                            seed_key=I.seed_key_for(key, split_role="pilot"),
                            purpose=K.PHASE_REVISION, top_logprobs=1, key=key)


# --------------------------------------------------------------------------- the partition filter


def test_a_deadline_lane_never_sees_an_endpoint_on_a_preemptable_partition(tmp_run_root,
                                                                           register_fake_server):
    """Plan §1.7: a preemption on a deadline lane is not a slow cell, it is a cell that dies
    mid-episode.  An endpoint whose partition cannot be resolved is excluded, not assumed safe."""
    register_fake_server(tmp_run_root, WP1_PROFILE, host="ok0", port=8100, slurm_job_id="1")
    register_fake_server(tmp_run_root, WP1_PROFILE, host="spot0", port=8101, slurm_job_id="2")
    register_fake_server(tmp_run_root, WP1_PROFILE, host="mystery0", port=8102, slurm_job_id="3")
    partitions = {"1": "ou_bcs_low", "2": "mit_preemptable"}

    def job_info(job_id):
        return INF.JobInfo(partition=partitions.get(job_id), gpu_type="l40s")

    live = [_entry(h, p, j) for h, p, j in (("ok0", 8100, "1"), ("spot0", 8101, "2"),
                                            ("mystery0", 8102, "3"))]
    pool = INF.NextdynEndpointPool(tmp_run_root, WP1_PROFILE,
                                   allowed_partitions={"ou_bcs_low"}, job_info=job_info,
                                   list_live=lambda *a, **k: live)
    assert [e.host for e in pool.endpoints()] == ["ok0"]
    assert pool.pick().host == "ok0"
    assert pool.replica_stamp(live[0]) == INF.ReplicaStamp(
        endpoint_id="ok0:8100", replica_id="ok0:8100", host="ok0", gpu_type="l40s",
        partition="ou_bcs_low")

    unfiltered = INF.NextdynEndpointPool(tmp_run_root, WP1_PROFILE, job_info=job_info,
                                         list_live=lambda *a, **k: live)
    assert len(unfiltered.endpoints()) == 3


def _entry(host, port, job_id):
    from agents_scaling.serving.profiles import get_serving_profile
    from agents_scaling.serving.registry import ServerEntry

    spec = get_serving_profile(WP1_PROFILE)
    return ServerEntry(model_size=spec.model_size, hf_id=spec.hf_id, host=host, port=port,
                       slurm_job_id=job_id, serving_profile=spec.name,
                       served_model_name=spec.served_model_name, max_model_len=spec.max_model_len,
                       tp_size=spec.tp_size)
