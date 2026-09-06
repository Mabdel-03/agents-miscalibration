"""policies/*: every policy end-to-end on the fake vLLM server with the WP4 stub contracts.

Fixtures from docs/study_v4/02_spec_fidelity_audit.md §3: T3 (sentinel bytes), T4 (DEC
full-round stop at an exact boundary), T5 (DEC round-cap compiler), T6 (CEN 8 cycles =
41 calls, CYCLE_CAP), T7 (action validation consumes a hub call), T10 (final reserve),
T17 (alias table, extended per critic P0-1); plus IND indivisible optional draw,
S_HISTORY previous-candidate-only state, previous-round-only DEC packets, context
failures as used opportunities, and the §10.6 import firewall.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.policies import policy_for
from agents_scaling.study.policies.base import AdmissionRefused, CallResult, EpisodeContext, call_by_name
from agents_scaling.study.policies.dec import DecPolicy, dec_round_cap
from agents_scaling.study.policies.degree import root_permutation
from agents_scaling.study.prompts import load_template
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.resources.envelopes import TableE, measure_wrappers
from tests.study import wp4_stubs as S

STUDY_DIR = Path(__file__).resolve().parents[2] / "src" / "agents_scaling" / "study"


@contextmanager
def factory(run_root, cfg, **server_kwargs):
    with S.fake_server(run_root, **server_kwargs) as srv:
        f = S.ContextFactory(cfg, run_root, srv)
        try:
            yield f
        finally:
            f.close()


def run(f: S.ContextFactory, task, cell, B, **kw):
    ctx = f.context(task, cell, B, **kw)
    result = policy_for(cell).run(ctx)
    return ctx, result


def prompt_of(ctx: EpisodeContext, call: CallResult) -> str:
    return call.spec.messages[0]["content"]


def calls_by_owner(ctx: EpisodeContext, prefix: str) -> list[CallResult]:
    return [c for c in ctx.calls if c.owner.startswith(prefix)]


def is_root_prompt(messages) -> bool:
    return "=== SAVED STATE" not in messages[0]["content"]


# --------------------------------------------------------------------------- S_FRESH


def test_s_fresh_budget_then_call_cap(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.S_FRESH, N=1)
    with factory(tmp_run_root, study_config) as f:
        Rr = f.root_reservation(task)
        B = Rr + Rr // 4  # realized draws cost a few percent of a full-cap reservation
        ctx, res = run(f, task, cell, B)
        ep = res.episode
        assert ep["stop_reason"] == "BUDGET" and res.status == "complete"
        led = ep["ledger"]
        assert led["stop_reason"] == "BUDGET" and 1 <= led["calls_admitted"] < 64 and led["spent"] <= B and led["reserved_open"] == 0
        assert len(ep["candidates"]) == led["calls_admitted"] == len(ep["calls"])
        # admission of draw k used only earlier debits: the reject event leaves committed + R > B
        reject = [e for e in led["events"] if e["op"] == "reject"][-1]
        assert reject["committed"] + Rr > B and reject["detail"]["reason"] == "BUDGET"
        assert all(e["remaining"] >= 0 for e in led["events"])
        assert ep["selection"]["selector_id"] == "VOTE" and ep["selection"]["pool_kind"] == "archive"
        assert ep["native_final"]["candidate_id"] == ep["selection"]["selected_candidate_id"]
        assert ep["counters"]["assigned_roster"] == 1 and ep["counters"]["unique_actors_used"] == 1
        assert ep["counters"]["resets"] == led["calls_admitted"] - 1
        # F00 draws come back as store hits (logical cost still debited)
        ctx2, res2 = run(f, task, cell, 200 * Rr)
        ep2 = res2.episode
        assert ep2["stop_reason"] == "CALL_CAP" and ep2["ledger"]["calls_admitted"] == 64
        # store hits = the admitted prefix plus the batch pre-generated records the first
        # episode never admitted (physical telemetry, not episode content)
        pre = ep["counters"]["pregenerated_unadmitted"]
        assert 0 <= pre < 8 and ep2["counters"]["aliased_calls"] == led["calls_admitted"] + pre and ep2["ledger"]["spent"] > 0
        assert [c["request_id"] for c in ep2["calls"]][: led["calls_admitted"]] == [c["request_id"] for c in ep["calls"]]


def test_s_fresh_refuses_wrong_framing_and_unstartable_budget(tmp_run_root, study_config):
    task = S.make_task()
    with factory(tmp_run_root, study_config) as f:
        with pytest.raises(T.ProtocolError, match="neutral 00"):
            run(f, task, S.make_cell(T.Method.S_FRESH, N=1, framing=T.Framing.F11), 10**18)
        with pytest.raises(T.ProtocolError, match="cannot admit a single root"):
            run(f, task, S.make_cell(T.Method.S_FRESH, N=1), f.root_reservation(task) - 1)


# --------------------------------------------------------------------------- T17 alias table


def test_alias_table_T17(tmp_run_root, study_config):
    task = S.make_task()
    cfg = study_config
    f00 = [S.bank_request_id(cfg, task, T.Framing.F00, k) for k in range(12)]
    f11 = [S.bank_request_id(cfg, task, T.Framing.F11, k) for k in range(12)]
    with factory(tmp_run_root, cfg) as f:
        Rr = f.root_reservation(task)
        big = 200 * Rr
        ctx, _ = run(f, task, S.make_cell(T.Method.S_FRESH, N=1), big, call_cap=12)
        assert [c.spec.request_id for c in ctx.calls] == f00[:12]
        ctx, _ = run(f, task, S.make_cell(T.Method.S_HISTORY, N=1), big, call_cap=3)
        assert ctx.calls[0].spec.request_id == f00[0] and ctx.calls[1].spec.request_id not in f00
        ctx, _ = run(f, task, S.make_cell(T.Method.IND_VOTE, N=5, framing=T.Framing.F11), big, call_cap=12)
        assert [c.spec.request_id for c in ctx.calls] == f11[:12]
        ctx, _ = run(f, task, S.make_cell(T.Method.IND_VOTE, N=3, module="N", framing=T.Framing.F00), big, call_cap=6)
        assert [c.spec.request_id for c in ctx.calls] == f00[:6]
        ctx, _ = run(f, task, S.make_cell(T.Method.DEC, N=5, framing=T.Framing.NATIVE), big, call_cap=5)
        main_dec = [c.spec.request_id for c in ctx.calls]
        assert not set(main_dec) & set(f00) and not set(main_dec) & set(f11)
        ctx, _ = run(f, task, S.make_cell(T.Method.DEC, N=5, module="N", framing=T.Framing.F00), big, call_cap=5)
        assert [c.spec.request_id for c in ctx.calls] == f00[:5]
        ctx, _ = run(f, task, S.make_cell(T.Method.IND_PRIVATE_REVISION, N=5, module="N", framing=T.Framing.F00), big)
        assert [c.spec.request_id for c in ctx.calls[:5]] == f00[:5]
        ctx, _ = run(f, task, S.make_cell(T.Method.DEC_ONE_ROUND, N=5, module="N", framing=T.Framing.F00), big)
        assert [c.spec.request_id for c in ctx.calls[:5]] == f00[:5]
        ctx, _ = run(f, task, S.make_cell(T.Method.DEGREE, N=1, module="D", degree=2), big)
        assert [c.spec.request_id for c in ctx.calls[:9]] == f00[:9]
        ctx, _ = run(f, task, S.make_cell(T.Method.CEN_FLAT, N=5, framing=T.Framing.NATIVE), big)
        assert not {c.spec.request_id for c in ctx.calls} & set(f00)
        # E-module replication with a new episode_rep is fresh
        ctx, _ = run(f, task, S.make_cell(T.Method.S_FRESH, N=1, module="E", episode_rep=1), big, call_cap=3)
        assert not {c.spec.request_id for c in ctx.calls} & set(f00)


def test_dec_manifest_framing_rules(tmp_run_root, study_config):
    task = S.make_task()
    with factory(tmp_run_root, study_config) as f:
        big = 200 * f.root_reservation(task)
        with pytest.raises(T.ProtocolError, match="P0-1"):
            run(f, task, S.make_cell(T.Method.DEC, N=5, module="A", framing=T.Framing.F00), big)
        with pytest.raises(T.ProtocolError, match="P0-1"):
            run(f, task, S.make_cell(T.Method.DEC, N=5, module="N", framing=T.Framing.NATIVE), big)
        with pytest.raises(T.ProtocolError, match="N >= 2"):
            run(f, task, S.make_cell(T.Method.DEC, N=1, module="A", framing=T.Framing.NATIVE), big)
        with pytest.raises(T.ProtocolError, match="never NATIVE"):
            run(f, task, S.make_cell(T.Method.IND_VOTE, N=5, framing=T.Framing.NATIVE), big)


# --------------------------------------------------------------------------- S_HISTORY


def test_s_history_sentinel_and_previous_only_T3(tmp_run_root, study_config):
    task = S.make_task()
    seen = []

    def responder(info):
        if info["mode"] == "solver" and is_root_prompt(info["messages"]) and not seen:
            seen.append(info["key"])
            return '{"approach": "broken", "final_answer": '  # invalid root
        return None

    with factory(tmp_run_root, study_config, responder=responder) as f:
        ctx, res = run(f, task, S.make_cell(T.Method.S_HISTORY, N=1), 200 * f.root_reservation(task), call_cap=6)
        ep = res.episode
        cands = ep["candidates"]
        assert cands[0]["valid"] is False and cands[0]["failure_code"] == "NOT_JSON"
        p1 = prompt_of(ctx, ctx.calls[1])
        assert T.SENTINEL_JSON in p1 and "--- own_saved_candidate ---\n" + T.SENTINEL_JSON + "\n" in p1
        assert ep["ledger"]["calls_admitted"] == 6 and len(cands) == 6  # the invalid root stays a used opportunity
        records = [T.CandidateRecord.from_dict(c) for c in cands]
        for j in range(2, 6):
            pj = prompt_of(ctx, ctx.calls[j])
            assert R.saved_object_text(records[j - 1]) in pj
            for older in records[: j - 1]:
                if older.valid:
                    assert R.saved_object_text(older) not in pj
            assert pj.count("--- own_saved_candidate ---") == 1 and "--- messages: none ---" in pj
        assert ep["stop_reason"] == "CALL_CAP" and ep["selection"]["pool_kind"] == "archive"
        assert ep["selection"]["valid_count"] == 5


def test_context_failure_is_a_used_opportunity(tmp_run_root, study_config):
    """Table E: an own candidate above ``caps.own_candidate_tokens`` → the next revision is a
    0-FLOP context-failure record that keeps its call slot; the one after gets the sentinel
    (corrections §4 item 7).  The cap is lowered to 5 recipient tokens so the fake server's
    ordinary candidates (10 whitespace tokens) exceed it (the real cap equals the output cap and can only be
    exceeded through re-serialisation under the pinned tokenizer)."""
    import dataclasses

    task = S.make_task()
    cfg = dataclasses.replace(study_config, caps=dataclasses.replace(study_config.caps, own_candidate_tokens=5))
    with factory(tmp_run_root, cfg) as f:
        ctx, res = run(f, task, S.make_cell(T.Method.S_HISTORY, N=1), 200 * f.root_reservation(task), call_cap=4)
        ep = res.episode
        assert ep["candidates"][0]["valid"] is True and ctx.own_state_tokens(T.CandidateRecord.from_dict(ep["candidates"][0])) > 5
        assert ep["candidates"][1]["valid"] is False and ep["candidates"][1]["failure_code"] == "CONTEXT_FAILURE"
        assert ep["calls"][1]["context_failure"] and ep["calls"][1]["actual_flops"] == 0 and ep["calls"][1]["reserved_flops"] == 0
        assert len(ep["calls"][1]["request_id"]) == 64  # the opportunity keeps its request identity
        assert T.SENTINEL_JSON in prompt_of(ctx, ctx.calls[2]) and ep["candidates"][2]["valid"] is True
        assert ep["candidates"][3]["failure_code"] == "CONTEXT_FAILURE"  # a valid but oversized parent again
        assert ep["ledger"]["calls_admitted"] == 4 and ep["counters"]["context_failures_by_role"] == {"revise": 2}
        assert ep["counters"]["model_invocations"] == 2 and ep["counters"]["opportunities"] == 4
        assert ep["stop_reason"] == "CALL_CAP" and ep["selection"]["planned_count"] == 4 and ep["selection"]["valid_count"] == 2


# --------------------------------------------------------------------------- IND_VOTE


def test_ind_vote_indivisible_optional_draw(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.IND_VOTE, N=5, framing=T.Framing.F11)
    with factory(tmp_run_root, study_config) as f:
        Rr = f.root_reservation(task, T.Framing.F11)
        ctx, res = run(f, task, cell, 200 * Rr)
        ep = res.episode
        assert ep["stop_reason"] == "CALL_CAP" and ep["ledger"]["calls_admitted"] == 64 and ep["counters"]["optional_draws"] == 59
        assert [c["actor_slot"] for c in ep["calls"]] == [k % 5 for k in range(64)]  # blind round-robin metadata
        assert all(c["seed_key"][4] == 0 and c["seed_key"][6] == k for k, c in enumerate(ep["calls"]))  # stateless keys
        assert ep["counters"]["assigned_roster"] == 5 and ep["counters"]["unique_actors_used"] == 5
        roots = [e for e in ep["ledger"]["events"] if e["op"] == "reserve" and e["owner"] == "roots"]
        assert len(roots) == 1 and roots[0]["detail"]["calls"] == 5  # all N slots initialised as one group
        # the initialisation is one full-cap group: one FLOP short refuses the whole episode
        with pytest.raises(T.ProtocolError, match="initialization"):
            run(f, task, cell, 5 * Rr - 1)
        # indivisibility of one optional draw, at N=1 (00 wording, N panel) where the boundary is reachable
        cell1 = S.make_cell(T.Method.IND_VOTE, N=1, module="N", framing=T.Framing.F00)
        R0 = f.root_reservation(task, T.Framing.F00)
        _, big = run(f, task, cell1, 200 * R0)
        actual = [c["actual_flops"] for c in big.episode["calls"]]
        _, res1 = run(f, task, cell1, R0 + actual[0] + actual[1] - 1)  # draw 1 fits, draw 2 does not
        assert res1.episode["counters"]["optional_draws"] == 1 and res1.episode["stop_reason"] == "BUDGET"
        _, res0 = run(f, task, cell1, R0 + actual[0] - 1)  # "almost" a full draw of headroom admits nothing
        assert res0.episode["counters"]["optional_draws"] == 0 and res0.episode["stop_reason"] == "BUDGET"
        assert res0.episode["ledger"]["calls_admitted"] == 1


# --------------------------------------------------------------------------- DEC


def test_dec_round_cap_compiler_T5():
    assert dec_round_cap(9, 64, 8) == (6, "call_cap")
    assert dec_round_cap(5, 64, 8) == (8, "dec_max_rounds")
    assert dec_round_cap(1, 64, 8) == (8, "dec_max_rounds")
    assert dec_round_cap(32, 64, 8) == (1, "call_cap")
    assert dec_round_cap(33, 64, 8) == (0, "call_cap")
    for N in (1, 2, 3, 5, 9):
        r, _ = dec_round_cap(N, 64, 8)
        assert N * (1 + r) <= 64 and N * (2 + r) > 64 or r == 8
    with pytest.raises(T.ProtocolError):
        dec_round_cap(65, 64, 8)


def test_dec_full_rounds_then_round_cap(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.DEC, N=5, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config) as f:
        ctx, res = run(f, task, cell, 400 * f.root_reservation(task))
        ep = res.episode
        assert ep["stop_reason"] == "ROUND_CAP" and ep["counters"]["rounds_completed"] == 8 and ep["ledger"]["calls_admitted"] == 45
        assert ep["counters"]["r_max"] == 8 and ep["counters"]["r_max_binding"] == "dec_max_rounds"
        assert ep["selection"]["pool_kind"] == "latest_slots" and len(ep["selection"]["pool_candidate_ids"]) == 5
        assert ep["selection"]["pool_candidate_ids"] == ep["latest_candidate_ids"]
        assert len(ep["packets"]) == 8 * 5 * 4 and ep["counters"]["unique_actors_used"] == 5
        assert ep["counters"]["calls_by_role"] == {"root": 5, "revise": 40}
        # every round was reserved as one 5-call group and the truthful root wording was used
        for e in ep["ledger"]["events"]:
            if e["op"] == "reserve" and e["owner"].startswith("round"):
                assert e["detail"]["calls"] == 5
        assert R.DEC_ROOT_TRUTHFUL_CLAUSE in prompt_of(ctx, ctx.calls[0])
        assert "You are one of 5 solvers" in prompt_of(ctx, ctx.calls[5])


def test_dec_full_round_stop_at_exact_boundary_T4(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.DEC, N=5, module="N", framing=T.Framing.F00)
    with factory(tmp_run_root, study_config) as f:
        ctx, res = run(f, task, cell, 400 * f.root_reservation(task))
        events = res.episode["ledger"]["events"]
        roots_debit = next(e for e in events if e["op"] == "debit" and e["owner"] == "roots")
        round1 = next(e for e in events if e["op"] == "reserve" and e["owner"] == "round1")
        B = roots_debit["committed"] + round1["amount"]  # round 1 fits with zero headroom
        ctx2, res2 = run(f, task, cell, B)
        ep = res2.episode
        assert ep["counters"]["rounds_completed"] == 1 and ep["stop_reason"] == "BUDGET"
        assert ep["ledger"]["calls_admitted"] == 10 and ep["ledger"]["spent"] <= B
        reject = [e for e in ep["ledger"]["events"] if e["op"] == "reject"]
        assert len(reject) == 1 and reject[0]["owner"] == "round2" and reject[0]["detail"]["calls"] == 5
        # room for several single calls remained, yet no member was launched alone
        per_call = min(round1["detail"]["amounts"])
        assert reject[0]["remaining"] >= 3 * per_call
        assert not any(e["op"] == "reserve" and e["owner"] == "round2" for e in ep["ledger"]["events"])


def test_dec_sentinel_and_unavailable_packet_T3(tmp_run_root, study_config):
    task = S.make_task()
    broken: list[int] = []

    def responder(info):
        if info["mode"] == "solver" and is_root_prompt(info["messages"]) and not broken:
            broken.append(info["seed"])
            return "not json at all"
        return None

    cell = S.make_cell(T.Method.DEC, N=3, module="N", framing=T.Framing.F00)
    with factory(tmp_run_root, study_config, responder=responder) as f:
        ctx, res = run(f, task, cell, 400 * f.root_reservation(task), call_cap=6)
        ep = res.episode
        roots = ep["candidates"][:3]
        bad = [c["slot"] for c in roots if not c["valid"]]
        assert len(bad) == 1
        s = bad[0]
        round1 = calls_by_owner(ctx, "round1")
        own_prompt = prompt_of(ctx, round1[s])
        assert "--- own_saved_candidate ---\n" + T.SENTINEL_JSON + "\n" in own_prompt
        for p in range(3):
            if p != s:
                assert T.PACKET_UNAVAILABLE_JSON in prompt_of(ctx, round1[p])
        assert ep["counters"]["complete_candidates"] == 5 and ep["counters"]["candidate_opportunities"] == 6


def test_dec_packets_come_only_from_previous_round(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.DEC, N=4, module="N", framing=T.Framing.F00)
    with factory(tmp_run_root, study_config) as f:
        ctx = f.context(task, cell, 400 * f.root_reservation(task))
        res = DecPolicy(max_rounds=3).run(ctx)
        ep = res.episode
        assert ep["counters"]["rounds_completed"] == 3 and ep["stop_reason"] == "ROUND_CAP" and ep["counters"]["r_max_binding"] == "max_rounds"
        by_stage: dict[str, list[dict]] = {}
        for c in ep["candidates"]:
            by_stage.setdefault(c["stage"], []).append(c)
        stages = ["root", "round1", "round2", "round3"]
        for r in (1, 2, 3):
            prev = by_stage[stages[r - 1]]
            for member_call in calls_by_owner(ctx, f"round{r}"):
                s = member_call.actor_slot
                prompt = prompt_of(ctx, member_call)
                for c in prev:
                    if c["slot"] != s:
                        assert c["candidate_sha256"] in prompt  # the N-1 peers' immediately preceding candidates
                    else:
                        assert c["candidate_sha256"] not in prompt  # own state is rendered as JSON, never as a peer packet
                for other_stage in stages:
                    if other_stage == stages[r - 1]:
                        continue
                    for c in by_stage.get(other_stage, []):
                        assert c["candidate_sha256"] not in prompt
                assert prompt.count("[message ") == 3


def test_dec_n1_and_controls(tmp_run_root, study_config):
    task = S.make_task()
    with factory(tmp_run_root, study_config) as f:
        big = 400 * f.root_reservation(task)
        ctx, res = run(f, task, S.make_cell(T.Method.DEC, N=1, module="N", framing=T.Framing.F00), big, call_cap=4)
        p = prompt_of(ctx, ctx.calls[1])
        assert "You are the only solver" in p and "other" not in p.split("=== TASK ===")[0] and "--- messages: none ---" in p
        assert res.episode["counters"]["rounds_completed"] == 3 and res.episode["stop_reason"] == "BUDGET" or res.episode["stop_reason"] in ("CALL_CAP", "ROUND_CAP")
        ctx, res = run(f, task, S.make_cell(T.Method.DEC_ONE_ROUND, N=5, module="N", framing=T.Framing.F00), big)
        ep = res.episode
        assert ep["counters"]["rounds_completed"] == 1 and ep["stop_reason"] == "ROUND_CAP" and ep["ledger"]["calls_admitted"] == 10
        assert ep["method"] == "DEC_ONE_ROUND" and len(ep["packets"]) == 20
        ctx2, res2 = run(f, task, S.make_cell(T.Method.IND_PRIVATE_REVISION, N=5, module="N", framing=T.Framing.F00), big)
        ep2 = res2.episode
        assert ep2["counters"]["rounds_completed"] == 1 and ep2["ledger"]["calls_admitted"] == 10 and ep2["packets"] == []
        for call in calls_by_owner(ctx2, "round1"):
            pr = prompt_of(ctx2, call)
            assert "--- messages: none ---" in pr and "solvers" not in pr.split("=== TASK ===")[0]
        assert [c["request_id"] for c in ep2["calls"][:5]] == [c["request_id"] for c in ep["calls"][:5]]
        assert len(ep2["selection"]["pool_candidate_ids"]) == 5


# --------------------------------------------------------------------------- CEN_FLAT


def hub_delegates_until_final(text: str) -> str:
    return "final" if load_template("cen_final_instruction").strip() in text else "delegate"


def test_cen_eight_cycles_T6(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.CEN_FLAT, N=5, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config, hub_mode=hub_delegates_until_final, delegate_count=4) as f:
        ctx, res = run(f, task, cell, 2000 * f.root_reservation(task))
        ep = res.episode
        assert ep["stop_reason"] == "CYCLE_CAP"
        assert ep["counters"]["calls_by_role"] == {"hub": 9, "worker": 32} and ep["ledger"]["calls_admitted"] == 41
        assert ep["counters"]["delegation_cycles"] == 8 and ep["counters"]["realized_participation"] == 4
        assert ep["native_final"]["valid"] is True and ep["candidates"][0]["stage"] == "final"
        final_call = ctx.calls[-1]
        assert final_call.owner == "final" and load_template("cen_final_instruction") in prompt_of(ctx, final_call)
        assert final_call.spec.seed_key.step_slot == 8 and final_call.spec.seed_key.purpose == "hub"
        assert ep["ledger"]["final_reserve_released"] is True and ep["ledger"]["reserved_open"] == 0
        # the hub prompts carried the exact roster capacity and every cycle reserved 4 worker envelopes
        assert "5 total role slots" in prompt_of(ctx, ctx.calls[0]) and "4 available worker slots" in prompt_of(ctx, ctx.calls[0])
        hub_reserves = [e for e in ep["ledger"]["events"] if e["op"] == "reserve" and e["owner"].endswith(".hub")]
        assert len(hub_reserves) == 8 and all(e["detail"]["calls"] == 5 for e in hub_reserves)
        # workers returned in planned order and the next hub saw them
        assert [c["actor_slot"] for c in ep["calls"][1:5]] == [1, 2, 3, 4]
        assert "--- returned_results: 4 ---" in prompt_of(ctx, ctx.calls[5])
        assert len(ep["packets"]) == 32 and all(p["kind"] == "subtask_result" for p in ep["packets"])


def test_cen_action_validation_T7(tmp_run_root, study_config):
    task = S.make_task()
    calls: list[str] = []

    def assignment(slot, handles=("task",)):
        return {"worker_slot": slot, "subtask_id": f"s{slot}", "question": "q", "source_handles": list(handles), "required_output_type": "text", "return_contract": "c"}

    def responder(info):
        if info["mode"] != "hub":
            return None
        n = len(calls)
        calls.append(info["key"])
        if n == 0:
            return json.dumps({"action": "delegate", "assignments": [assignment(1), assignment(1)]})
        if n == 1:
            return json.dumps({"action": "delegate", "assignments": [assignment(s) for s in range(1, 6)]})
        if n == 2:
            return json.dumps({"action": "delegate", "assignments": [assignment(1, ("task", "secret"))]})
        if n == 3:
            return json.dumps({"action": "plan", "steps": ["x"]})
        if n == 4:
            return json.dumps({"action": "delegate", "assignments": [assignment(7)]})
        return None  # server default: final

    cell = S.make_cell(T.Method.CEN_FLAT, N=5, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config, hub_mode="final", responder=responder) as f:
        ctx, res = run(f, task, cell, 2000 * f.root_reservation(task))
        ep = res.episode
        assert [e["code"] for e in ep["hub_action_errors"]] == ["DUP_SLOT", "TOO_MANY", "HIDDEN_HANDLE", "INVALID", "BAD_SLOT"]
        assert ep["counters"]["calls_by_role"] == {"hub": 6} and ep["ledger"]["calls_admitted"] == 6
        assert ep["stop_reason"] == "VOLUNTARY_FINISH" and ep["native_final"]["valid"] is True
        assert [c["cycle"] for c in ep["cycles"]] == [0, 1, 2, 3, 4, 5] and ep["cycles"][-1]["action"] == "final"
        assert ep["counters"]["realized_participation"] == 0
        # each erroneous hub call was debited (never refunded) and consumed its cycle seed
        assert [c["seed_key"][6] for c in ep["calls"]] == [0, 1, 2, 3, 4, 5]
        assert all(c["actual_flops"] > 0 for c in ep["calls"])


def test_cen_final_reserve_always_available_T10(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.CEN_FLAT, N=5, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config, hub_mode=hub_delegates_until_final, delegate_count=4) as f:
        from agents_scaling.study.inference.tokens import count_tokens

        tok = f.server.tokenizer
        table = TableE(study_config.caps, measure_wrappers(tok, task, Ns=(5,)), task_tokens=count_tokens(task.task_text, tok))
        o = f.oracle
        B = o.reservation(table.hub_prompt_max(5, cycle0=True), 8192) + 4 * o.reservation(table.worker_prompt_max(cycle0=True), 8192) + o.reservation(
            table.hub_prompt_max(5, cycle0=False), 8192
        )
        ctx, res = run(f, task, cell, B)
        ep = res.episode
        assert ep["counters"]["delegation_cycles"] == 1 and ep["stop_reason"] == "BUDGET"
        assert ep["native_final"]["valid"] is True and ctx.calls[-1].owner == "final"
        assert ep["ledger"]["spent"] <= B and ep["ledger"]["calls_admitted"] == 6
        reject = [e for e in ep["ledger"]["events"] if e["op"] == "reject"]
        assert len(reject) == 1 and reject[0]["owner"] == "cycle1.hub"
        # one FLOP less cannot start the protocol
        with pytest.raises(T.ProtocolError):
            run(f, task, cell, o.reservation(table.hub_prompt_max(5, cycle0=False), 8192) - 1)


def test_cen_n1_hub_only_and_invalid_final(tmp_run_root, study_config):
    task = S.make_task()
    with factory(tmp_run_root, study_config, hub_mode="final") as f:
        ctx, res = run(f, task, S.make_cell(T.Method.CEN_FLAT, N=1, framing=T.Framing.NATIVE), 2000 * f.root_reservation(task))
        ep = res.episode
        assert "1 total role slots" in prompt_of(ctx, ctx.calls[0]) and "0 available worker slots" in prompt_of(ctx, ctx.calls[0])
        assert ep["ledger"]["calls_admitted"] == 1 and ep["stop_reason"] == "VOLUNTARY_FINISH" and ep["counters"]["workers_assigned"] == 0
    with factory(tmp_run_root, study_config, hub_mode="delegate", delegate_count=2) as f:
        ctx, res = run(f, task, S.make_cell(T.Method.CEN_FLAT, N=3, framing=T.Framing.NATIVE), 2000 * f.root_reservation(task))
        ep = res.episode
        assert ep["stop_reason"] == "CYCLE_CAP" and ep["native_final"]["valid"] is False and ep["candidates"][0]["failure_code"] == "NOT_FINAL"
        assert ep["ledger"]["calls_admitted"] == 8 + 16 + 1 and ep["counters"]["realized_participation"] == 2


# --------------------------------------------------------------------------- DEGREE


def test_degree_module(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.DEGREE, N=1, module="D", degree=4)
    with factory(tmp_run_root, study_config) as f:
        ctx, res = run(f, task, cell, 400 * f.root_reservation(task))
        ep = res.episode
        perm = root_permutation(study_config.study_seed, task.source_id)
        assert ep["root_permutation"] == perm and ep["focal_root_index"] == perm[0] and ep["peer_root_indices"] == perm[1:5]
        assert ep["stop_reason"] == "COMPLETED" and ep["ledger"]["calls_admitted"] == 14 and ep["selection"] is None
        stages = [c["stage"] for c in ep["candidates"]]
        assert stages == ["root"] * 9 + ["focal_revision"] + ["consumer"] * 4
        roots = ep["candidates"][:9]
        focal_prompt = prompt_of(ctx, ctx.calls[9])
        for i, c in enumerate(roots):
            assert (c["candidate_sha256"] in focal_prompt) == (i in perm[1:5])
        assert focal_prompt.count("[message ") == 4 and R.saved_object_text(T.CandidateRecord.from_dict(roots[perm[0]])) in focal_prompt
        revised = T.CandidateRecord.from_dict(ep["candidates"][9])
        for call in ctx.calls[10:]:
            assert R.saved_object_text(revised) in prompt_of(ctx, call) and load_template("common_consumer").strip() in prompt_of(ctx, call)
        assert [c["seed_key"][4:8] for c in ep["calls"][9:]] == [[0, "revise", 4, "DEGREE"]] + [[j, "consumer", 4, "DEGREE"] for j in range(1, 5)]
        assert ep["native_final"]["candidate_id"] == ep["focal_candidate_id"] and ep["counters"]["assigned_roster"] == 5
        with pytest.raises(T.ProtocolError, match="degree"):
            run(f, task, S.make_cell(T.Method.DEGREE, N=1, module="D", degree=3), 400 * f.root_reservation(task))


# --------------------------------------------------------------------------- machinery


def test_generate_group_settles_ledger_on_infra_failure(tmp_run_root, study_config):
    task = S.make_task()
    cell = S.make_cell(T.Method.S_FRESH, N=1)
    with factory(tmp_run_root, study_config) as f:
        ctx = f.context(task, cell, 100 * f.root_reservation(task), executor=False)  # sequential: one request absorbs the faults
        rendered = R.render_root(task, T.Framing.F00)
        specs = [ctx.spec(rendered, T.SOLVER_DECODING, ctx.seed(0, "root", k, "stateless_bank"), "root") for k in range(3)]
        f.server.fail_next(3)  # one request exhausts its two exogenous retries
        with pytest.raises(T.InfraFailure):
            ctx.generate_group(specs, "root", owner="g")
        assert ctx.ledger.reserved == 0 and 0 < ctx.ledger.spent
        assert ctx.ledger.calls_admitted() == 2  # the failed call's slot was released
        with pytest.raises(AdmissionRefused) as exc:
            ctx.generate_group([specs[0]] * 100, "root")
        assert exc.value.reason is StopReason.CALL_CAP


def test_episode_result_round_trip(tmp_run_root, study_config, episode_result_fixture):
    task = S.make_task()
    with factory(tmp_run_root, study_config) as f:
        _, res = run(f, task, S.make_cell(T.Method.S_FRESH, N=1), 3 * f.root_reservation(task))
    data = json.loads(res.to_json())
    back = T.EpisodeResult.from_dict(data)
    assert back.to_dict() == data and set(data) == set(episode_result_fixture)
    ep = data["episode"]
    for key in ("episode_id", "calls", "candidates", "packets", "ledger", "counters", "selection", "native_final"):
        assert key in ep
    assert len(ep["episode_id"]) == 64 and ep["ledger"]["B_flops"] == 3 * f.root_reservation(task)


def test_call_by_name_adapter():
    def fn(content, N_total, *, allowed_handles=("task",)):
        return content, N_total, allowed_handles

    assert call_by_name(fn, content="x", N_total=5, allowed_handles=("task", "s1"), extra=1) == ("x", 5, ("task", "s1"))
    with pytest.raises(T.ProtocolError, match="requires parameter"):
        call_by_name(fn, content="x")

    def var(**kw):
        return kw

    assert call_by_name(var, a=1)["a"] == 1


def test_no_policy_imports_evaluation_or_protected():
    pattern = re.compile(r"^\s*(from|import)\s+agents_scaling\.study\.(evaluation|data\.protected)", re.M)
    for sub in ("policies", "resources"):
        for path in (STUDY_DIR / sub).glob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path
            assert "/protected/" not in path.read_text(encoding="utf-8"), path


# --------------------------------------------------------------------------- real WP3 contracts


def test_policies_against_real_wp3_contracts(tmp_run_root, study_config):
    """The lazily resolved WP3 modules (parse/, packets.py, selection/) drive every policy
    end-to-end; skipped only while those modules are absent."""
    from agents_scaling.study.policies.base import ContractsUnavailable, default_contracts

    try:
        default_contracts()
    except ContractsUnavailable as exc:
        pytest.skip(str(exc))
    task = S.make_task()
    with factory(tmp_run_root, study_config, hub_mode=hub_delegates_until_final, delegate_count=3, fence_rate=0.5) as f:
        big = 400 * f.root_reservation(task)
        _, s_fresh = run(f, task, S.make_cell(T.Method.S_FRESH, N=1), big, contracts="real", call_cap=6)
        ep = s_fresh.episode
        assert ep["ledger"]["calls_admitted"] == 6 and ep["counters"]["complete_candidates"] == 6
        assert ep["selection"]["selector_id"] == "VOTE" and ep["selection"]["selected_candidate_id"] in ep["selection"]["pool_candidate_ids"]
        assert all(c["vote_key"] in ("A", "B", "C") and c["grouping_mode"] == "mc_letter" for c in ep["candidates"])
        ctx = f.context(task, S.make_cell(T.Method.DEC, N=3, module="N", framing=T.Framing.F00), big, contracts="real")
        dec = DecPolicy(max_rounds=2).run(ctx).episode
        assert dec["counters"]["rounds_completed"] == 2 and len(dec["packets"]) == 2 * 3 * 2
        assert all(0 < p["recipient_tokens"] <= study_config.caps.packet_tokens and p["kind"] == "peer" for p in dec["packets"])
        round1 = calls_by_owner(ctx, "round1")
        assert all(dec["candidates"][p]["candidate_sha256"] in prompt_of(ctx, round1[s]) for s in range(3) for p in range(3) if p != s)
        ctx, cen = run(f, task, S.make_cell(T.Method.CEN_FLAT, N=4, framing=T.Framing.NATIVE), big, contracts="real")
        ep = cen.episode
        assert ep["stop_reason"] == "CYCLE_CAP" and ep["counters"]["calls_by_role"] == {"hub": 9, "worker": 24}
        assert ep["native_final"]["valid"] is True and ep["hub_action_errors"] == []
        assert all(p["kind"] == "subtask_result" and p["recipient_tokens"] <= study_config.caps.subtask_result_tokens for p in ep["packets"])
        _, deg = run(f, task, S.make_cell(T.Method.DEGREE, N=1, module="D", degree=8), big, contracts="real")
        assert deg.episode["ledger"]["calls_admitted"] == 14 and deg.episode["counters"]["complete_candidates"] == 14
