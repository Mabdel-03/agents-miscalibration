"""WP5: the receiver assays, offline against the fake vLLM server (spec 9.1/9.4; 18.4 names).

The assay's whole claim is "only the cue changed".  So the tests below hold the evidence bytes
fixed and vary the arm: the ``OTHER DRAFTS`` block must come out byte-identical under 1:4, 4:1
and hidden, the working draft must be one of the two frozen packets, and the reasoning the
episode is allowed to spend must not grow just because the protocol uses two calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agents_scaling.nextdyn import checker as CH
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import inference as INF
from agents_scaling.nextdyn import mod97 as M
from agents_scaling.nextdyn import packets as PK
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn import receiver as RX
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.ids import CellKind
from tests.nextdyn.conftest import WP1_MODEL, nd_json_responder
from tests.nextdyn.test_mod97 import CONFIG_DIR, SEED

STUDY_SEED = b"nextdyn-receiver-study-seed"
SPLIT_ROLE = "pilot"
PANEL = "packets:abc123def456"

#: The committed E2b block: the generator shape, the check budget and the panel size all come
#: from ``study.yaml`` here exactly as they do in the runner (no number is invented by a test).
E2B = C.load_study(CONFIG_DIR / "study.yaml").experiments.E2b
E2A = C.load_study(CONFIG_DIR / "study.yaml").experiments.E2a
SPEC = M.spec_from_config(E2B)
PANEL_SIZE = RX.support_panel_size(E2A)


def counter(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class FakeItem:
    """``prompts.TaskItem``: the public question and nothing about the arm."""

    source_item_id: str = "nditem0000000001"
    item_variant_id: str = "nditem0000000001.v0"
    rendered_question: str = "What is the value of x24 ?"


@pytest.fixture(scope="module")
def registry():
    return P.load_registry()


@pytest.fixture(scope="module")
def e2b_block():
    return C.load_study(CONFIG_DIR / "study.yaml").experiments.E2b


@pytest.fixture
def pair():
    """Two frozen packets with distinct opaque labels and distinct bytes."""
    def make(pseudonym: str, answer: str, claim: str) -> PK.TransmittedPacket:
        out = schema.PublicOutput(
            reasoning=f"a rationale for {claim}", shareable_claims=(claim,), final_answer=answer,
            confidence=0.7, field_order=schema.DECLARED_FIELD_ORDER, order_violation=False,
            parse_status=schema.ParseStatus.OK, raw_text="{}", content_hash="c:x",
            over_limits={}, final_answer_span=None, confidence_span=None,
            confidence_emitted_before_answer=False)
        return PK.render_packet(out, pseudonym=pseudonym, counter=counter,
                                counter_tokenizer_hash="tok:test", cap=512,
                                content=P.PacketContent.FULL)

    return RX.FixedPacketPair(
        source_item_id=FakeItem().source_item_id, panel_id=PANEL,
        correct=make("agent-0", "41", "alpha"), wrong=make("agent-1", "17", "beta"),
        correct_generation_uid="g:correct", wrong_generation_uid="g:wrong")


def condition(nd_contract, registry, name: str, *, budget: int = 1024, replicate: int = 0,
              item=None, panel_id: str = PANEL) -> RX.ReceiverCondition:
    spec = registry.receivers[name]
    item = item or FakeItem()
    return RX.ReceiverCondition.from_contract(
        nd_contract, assay=RX.ASSAY_E2A if name.startswith("E2A") else RX.ASSAY_E2B, spec=spec,
        item=item, source_item_id=item.source_item_id, item_variant_id=item.item_variant_id,
        panel_id=panel_id, model=nd_contract.model(WP1_MODEL), budget=budget,
        replicate=replicate, split_role=SPLIT_ROLE, top_logprobs=1,
        support_panel_size=PANEL_SIZE)


def tool_for(pair, *, enabled: bool, index: int = 0) -> CH.CheckStepTool:
    """A ``CheckStepTool`` over the fixture pair -- **both** E2b arms are served by one.

    The disabled arm gets an object too (``enabled=False``), never ``None``: that is what makes
    the two arms spend the same request slots and write the same ``tool_events``.
    """
    problem = M.generate(SEED, SPEC.chain_depths[0], index, spec=SPEC)
    traces = {pair.correct.packet_id: M.correct_trace(problem),
              pair.wrong.packet_id: M.wrong_trace(problem, SEED)}
    return CH.CheckStepTool(traces, problem, enabled=enabled, max_checks=E2B.max_checks)


def receiver_responder(steps=((0, 1),)):
    """Plan phase -> a ``check_requests`` object; decision phase -> the spec-6.1 answer object.

    The two are told apart by the ``CHECK RESULTS`` marker of the decision template, not by the
    presence of the word ``check_requests`` -- the decision prompt quotes the plan transcript
    and would otherwise look like a plan.
    """
    answer = nd_json_responder()

    def respond(payload: dict):
        if "prompt_token_ids" in payload:
            return answer(payload)
        text = json.dumps(payload.get("messages", []))
        if "CHECK RESULTS" in text:
            return answer(payload)
        return json.dumps({"check_requests": [{"packet_id": p, "step_id": s} for p, s in steps]})

    return respond


def run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, *, tool=None, steps=((0, 1),)):
    ctx = nd_ctx(cell_id="receiver/test")
    with nd_serve(responder=receiver_responder(steps)):
        return RX.run_receiver_episode(
            cond, pair, lambda req: INF.generate_decision(req, ctx=ctx), tool,
            study_seed=STUDY_SEED, tokenizer=fake_tokenizer)


# --------------------------------------------------------------------------- spec 18.4


def test_e2a_episode_is_one_generation_call(nd_contract, registry, pair, nd_ctx, nd_serve,
                                            fake_tokenizer):
    """Spec 18.3/18.4 and plan D17.  E2a and its PRIVATE controls are **one** call: the
    decision phase only, at the full cap B, with no plan phase and no tool.

    Counting this is not pedantry -- the E2a and E2b budgets are only comparable if E2a spends
    B in one call where E2b spends B/2 twice.
    """
    cond = condition(nd_contract, registry, "E2A_1_4_CORRECT")
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer)
    assert cond.phases == 1
    assert episode.generation_calls == 1
    assert [d.phase for d in episode.decisions] == [K.PHASE_RECEIVER_DECISION]
    assert episode.decisions[0].key.budget_hash == cond.episode_tokens.budget_hash()
    assert cond.tokens_for(K.PHASE_RECEIVER_DECISION).thinking_cap == cond.budget == 1024
    assert episode.check_requests == () and episode.tool_events == ()

    control = condition(nd_contract, registry, "E2A_CONTROL_CORRECT")
    controlled = run(control, pair, nd_ctx, nd_serve, fake_tokenizer)
    assert controlled.generation_calls == 1 and controlled.displayed == ()


def test_two_phase_thinking_caps_sum_to_declared_episode_budget(nd_contract, registry, pair,
                                                                nd_ctx, nd_serve, fake_tokenizer):
    """Spec 18.4.  "Total reasoning cap per logical episode is B, not B per call."

    Both phases also carry the **episode-level** budget hash in their key, so one E2b episode
    is one ``episode_id`` with two phases rather than two episodes that happen to be adjacent.
    """
    for budget in (1024, 4096):
        cond = condition(nd_contract, registry, "E2B_1_4_CHECK_CORRECT", budget=budget)
        assert cond.plan_tokens.thinking_cap + cond.decision_tokens.thinking_cap == budget
        assert cond.episode_tokens.thinking_cap == budget
        assert cond.plan_tokens.final_cap == 256 and cond.decision_tokens.final_cap == 512
        RX.assert_budget_split(cond.episode_tokens, cond.plan_tokens, cond.decision_tokens)

    cond = condition(nd_contract, registry, "E2B_1_4_NOCHECK_CORRECT")
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer,
                  tool=tool_for(pair, enabled=False))
    assert [d.phase for d in episode.decisions] == [K.PHASE_RECEIVER_PLAN,
                                                    K.PHASE_RECEIVER_DECISION]
    assert len({d.key.budget_hash for d in episode.decisions}) == 1
    assert len({d.key.episode_id if hasattr(d.key, "episode_id") else episode.episode_id
                for d in episode.decisions}) == 1
    assert len({d.decision_id for d in episode.decisions}) == 2

    doubled = K.TokenContract(thinking_cap=2048, final_cap=256, served_window=4096, reserve=8,
                              packet_cap=64, rationale_cap=32, claims_max=4, claim_cap=16)
    with pytest.raises(ContractViolation):
        RX.assert_budget_split(cond.episode_tokens, doubled, doubled)


def test_support_conditions_have_identical_packet_bodies(nd_contract, registry, pair, nd_ctx,
                                                         nd_serve, fake_tokenizer):
    """Spec 18.4.  "The two displayed packet bodies are byte-identical under all support
    conditions for an item.  The 1:4 and 4:1 arms differ only in count fields."

    The display *order* is drawn from a blind key with no condition in it, so it too is fixed
    across the arms; if it were not, the bodies would be identical only as a set.
    """
    episodes = {}
    for name, support in (("E2A_1_4_CORRECT", P.Support.CORRECT_MINORITY),
                          ("E2A_4_1_CORRECT", P.Support.CORRECT_MAJORITY),
                          ("E2A_HIDDEN_CORRECT", P.Support.HIDDEN)):
        cond = condition(nd_contract, registry, name)
        episodes[support] = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer)

    RX.assert_support_invariant_bodies(episodes)
    bodies = {tuple(p.serialized for p in ep.displayed) for ep in episodes.values()}
    assert len(bodies) == 1

    minority = episodes[P.Support.CORRECT_MINORITY]
    majority = episodes[P.Support.CORRECT_MAJORITY]
    assert minority.support_lines != majority.support_lines
    assert episodes[P.Support.HIDDEN].support_lines == ()
    # the two count assignments are the same sentence with the digits swapped
    assert len(minority.support_lines) == len(majority.support_lines) == 3
    assert RX.support_counts(P.Support.CORRECT_MINORITY, PANEL_SIZE) == (1, 4)
    assert RX.support_counts(P.Support.CORRECT_MAJORITY, PANEL_SIZE) == (4, 1)

    # and the working drafts are the frozen packets, not a third text
    assert pair.draft_for(P.DraftRole.CORRECT) is pair.correct
    assert pair.draft_for(P.DraftRole.WRONG) is pair.wrong


def test_support_metadata_is_not_counted_as_independent_generations(nd_contract, registry, pair,
                                                                    nd_ctx, nd_serve,
                                                                    fake_tokenizer):
    """Spec 18.4.  The five advisers are randomized metadata over a constructed panel; the
    episode's independent generations are its model calls -- one for E2a, two for E2b.

    The prompt says so in as many words, because the published method has to (spec 9.1).
    """
    one = condition(nd_contract, registry, "E2A_4_1_CORRECT")
    two = condition(nd_contract, registry, "E2B_4_1_NOCHECK_CORRECT")
    first = run(one, pair, nd_ctx, nd_serve, fake_tokenizer)
    second = run(two, pair, nd_ctx, nd_serve, fake_tokenizer,
                 tool=tool_for(pair, enabled=False))

    assert first.independent_generations == first.generation_calls == 1
    assert second.independent_generations == second.generation_calls == 2
    assert PANEL_SIZE == 5
    assert first.independent_generations != PANEL_SIZE

    head = first.support_lines[0]
    assert "constructed panel" in head
    assert f"not {PANEL_SIZE} independently generated" in head
    # the fixed packets are replay material, not generations of this episode either
    assert len(first.displayed) == 2 and first.independent_generations == 1


def test_14400_receiver_episodes_are_28800_generation_calls_per_model(e2b_block, study_cfg,
                                                                      registry):
    """Spec 18.4.  The planner count over the declared grid, per receiver model.

    Depth is *not* a sixth factor: the three depths are already inside the 300 source problems
    (spec 9.5 "Do not multiply by the three depth levels again").  The controls are counted
    separately and run the same two-phase adapter, so they cost two calls each.
    """
    budgets = study_cfg.budgets.receiver_thinking_tokens
    counts = RX.e2b_grid_counts(e2b_block, budgets=budgets)
    assert (e2b_block.source_problems, len(e2b_block.endorsement), len(e2b_block.checker_access),
            len(e2b_block.initial_draft), len(budgets), e2b_block.replicates) == (300, 3, 2, 2, 2, 2)
    assert counts.episodes == 300 * 3 * 2 * 2 * 2 * 2 == 14_400
    assert counts.calls == 28_800
    assert counts.control_episodes == 2_400 and counts.control_calls == 4_800
    assert counts.total_calls == 33_600

    # every cell of the grid is a real registry entry, not an implied one
    graded = [name for name, spec in registry.receivers.items()
              if spec.phases == 2 and spec.packet_content is not P.PacketContent.NONE]
    assert len(graded) == 3 * 2 * 2

    e2a = study_cfg.experiments.E2a
    one_call = RX.e2a_grid_counts(e2a, supports=e2b_block.endorsement,
                                  drafts=e2b_block.initial_draft, budgets=budgets)
    assert one_call.episodes == 100 * 3 * 2 * 2 * 2 == 2_400
    assert one_call.calls == one_call.episodes           # one call per E2a episode
    assert one_call.control_episodes == 800


# --------------------------------------------------------------------------- the tool loop


def test_the_decision_phase_reads_the_plan_transcript_and_the_typed_responses(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.4 step 3.  The decision phase sees the same evidence plus the plan/check
    transcript, and the tool events are keyed on the *plan* decision -- the call that asked."""
    cond = condition(nd_contract, registry, "E2B_1_4_CHECK_CORRECT")
    tool = tool_for(pair, enabled=True)
    traces = tool.packets
    steps = ((pair.wrong.packet_id, traces[pair.wrong.packet_id].violated_step),
             (pair.correct.packet_id, 2))
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, tool=tool, steps=steps)

    assert episode.plan_parsed is True
    assert [r.status for r in episode.check_responses] == [CH.CheckStatus.INVALID,
                                                           CH.CheckStatus.VALID]
    assert {e.decision_id for e in episode.tool_events} == {episode.decisions[0].decision_id}
    body = "\n".join(m["content"] for m in episode.decisions[1].messages)
    assert "invalid" in body and "valid" in body
    for packet in episode.displayed:
        assert packet.serialized in body


def test_a_disabled_checker_arm_still_serves_the_same_response_schema(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.4 step 2: "when disabled, return the same response schema with status
    ``disabled``. ... This is tool access, not a placebo that secretly provides correctness
    feedback." """
    cond = condition(nd_contract, registry, "E2B_1_4_NOCHECK_CORRECT")
    off = tool_for(pair, enabled=False, index=1)
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, tool=off,
                  steps=((pair.correct.packet_id, 1),))
    assert [r.status for r in episode.check_responses] == [CH.CheckStatus.DISABLED]
    assert {e.access_condition for e in episode.tool_events} == {CH.ACCESS_DISABLED}
    plan_prompt = "\n".join(m["content"] for m in episode.decisions[0].messages)
    assert RX.CHECKER_UNAVAILABLE in plan_prompt


def test_an_arm_and_a_tool_that_disagree_about_the_checker_are_refused(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    cond = condition(nd_contract, registry, "E2B_1_4_NOCHECK_CORRECT")
    live = tool_for(pair, enabled=True, index=2)
    with pytest.raises(ContractViolation):
        run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, tool=live)


def test_a_two_phase_arm_without_a_tool_is_refused(nd_contract, registry, pair, nd_ctx,
                                                   nd_serve, fake_tokenizer):
    """``tool=None`` is not "the checker is off".

    Reading it that way is the quiet failure: the plan prompt would say CHECKER_UNAVAILABLE,
    every request would come back ``disabled``, no ``tool_events`` row would be written -- and
    the two decision rows would still be keyed under the **checker-enabled** ``protocol_hash``.
    H4's I_V contrast (spec 9.6) would then be estimated over rows where the treatment arm had
    silently received the control, which is worse than a missing arm because nothing says so.
    """
    for name in ("E2B_1_4_CHECK_CORRECT", "E2B_1_4_NOCHECK_CORRECT"):
        cond = condition(nd_contract, registry, name)
        assert cond.phases == 2
        with pytest.raises(ContractViolation) as raised:
            run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, tool=None)
        assert "CheckStepTool" in str(raised.value)

    # a one-call arm is the only one that legitimately has no tool
    one_call = condition(nd_contract, registry, "E2A_1_4_CORRECT")
    assert run(one_call, pair, nd_ctx, nd_serve, fake_tokenizer).tool_events == ()


def test_both_checker_arms_spend_the_same_slots_and_log_the_same_events(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.3 ("no adaptive exception grants extra checks") and 9.4 step 2.

    A plan that asks for more checks than the budget is truncated in *both* arms, so the two
    CHECK RESULTS blocks carry the same number of lines and the arms differ in what the
    responses say, not in how much text the decision phase reads.  Both arms write the same
    ``tool_events`` shape, which is where spec 9.6's checker-use report comes from.
    """
    greedy = tuple((pair.correct.packet_id, step) for step in range(1, E2B.max_checks + 4))
    episodes = {}
    for name, enabled in (("E2B_1_4_CHECK_CORRECT", True), ("E2B_1_4_NOCHECK_CORRECT", False)):
        cond = condition(nd_contract, registry, name)
        episodes[enabled] = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer,
                                tool=tool_for(pair, enabled=enabled), steps=greedy)

    on, off = episodes[True], episodes[False]
    assert len(greedy) > E2B.max_checks
    for episode in (on, off):
        assert len(episode.check_responses) == E2B.max_checks
        assert len(episode.tool_events) == len(greedy)
        over = [e for e in episode.tool_events if e.status == CH.OVER_BUDGET]
        assert len(over) == len(greedy) - E2B.max_checks
    assert {e.access_condition for e in on.tool_events} == {CH.ACCESS_ENABLED}
    assert {e.access_condition for e in off.tool_events} == {CH.ACCESS_DISABLED}
    # same shape, different content: that is the placebo symmetry spec 9.4 asks for
    assert [r.status for r in off.check_responses] == [CH.CheckStatus.DISABLED] * E2B.max_checks
    assert CH.CheckStatus.DISABLED not in {r.status for r in on.check_responses}


def test_invalid_plan_output_continues_to_the_decision_without_a_repair_call(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.4.  A plan that did not parse costs nothing and blocks nothing."""
    cond = condition(nd_contract, registry, "E2B_HIDDEN_NOCHECK_WRONG")
    ctx = nd_ctx(cell_id="receiver/test")

    def responder(payload):
        if "prompt_token_ids" in payload:
            return nd_json_responder()(payload)
        if "CHECK RESULTS" in json.dumps(payload.get("messages", [])):
            return nd_json_responder()(payload)
        return "I will check the third step."

    with nd_serve(responder=responder):
        episode = RX.run_receiver_episode(
            cond, pair, lambda req: INF.generate_decision(req, ctx=ctx),
            tool_for(pair, enabled=False, index=3),
            study_seed=STUDY_SEED, tokenizer=fake_tokenizer)
    assert episode.check_requests == () and episode.plan_parsed is False
    assert episode.generation_calls == 2
    assert episode.decision.output.final_answer == "4"


# --------------------------------------------------------------------------- identity & rows


def test_both_phases_share_one_episode_id_and_differ_only_in_phase(nd_contract, registry):
    cond = condition(nd_contract, registry, "E2B_4_1_CHECK_WRONG")
    plan = cond.key_for(K.PHASE_RECEIVER_PLAN)
    decision = cond.key_for(K.PHASE_RECEIVER_DECISION)
    assert cond.episode_id.startswith(f"{CellKind.RECEIVER.value}|")
    assert plan.decision_id() != decision.decision_id()
    fields = {f for f in plan.__dataclass_fields__ if getattr(plan, f) != getattr(decision, f)}
    assert fields == {"phase"}
    assert plan.root_team_id == cond.panel_id == PANEL
    assert (plan.round, plan.agent_slot, plan.replicate_id) == (0, 0, 0)


def test_the_seed_key_carries_no_condition_and_no_budget(nd_contract, registry):
    """R5/D4.  Two arms of one item share their seed key exactly, so their prompts alias in
    the store whenever the rendered bytes coincide."""
    from agents_scaling.nextdyn import ids as I

    a = condition(nd_contract, registry, "E2B_1_4_CHECK_CORRECT")
    b = condition(nd_contract, registry, "E2B_4_1_CHECK_CORRECT")
    key_a = I.seed_key_for(a.key_for(K.PHASE_RECEIVER_PLAN), split_role=SPLIT_ROLE)
    key_b = I.seed_key_for(b.key_for(K.PHASE_RECEIVER_PLAN), split_role=SPLIT_ROLE)
    assert key_a == key_b
    assert a.protocol_hash != b.protocol_hash


def test_transmission_rows_record_the_assigned_edge_and_the_error_position(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.2 asks for the token length and the error position of the displayed evidence;
    ``transmitted_tokens`` is the first and ``error_position_rel`` the second, on the wrong
    packet's row only."""
    located = RX.FixedPacketPair(
        source_item_id=pair.source_item_id, panel_id=pair.panel_id, correct=pair.correct,
        wrong=pair.wrong, correct_generation_uid=pair.correct_generation_uid,
        wrong_generation_uid=pair.wrong_generation_uid, error_position_rel=0.25)
    cond = condition(nd_contract, registry, "E2A_HIDDEN_WRONG")
    episode = run(cond, located, nd_ctx, nd_serve, fake_tokenizer)
    rows = RX.transmission_rows(episode, located, receiver_counter=counter)

    assert len(rows) == 2
    assert {r.declared_edge for r in rows} == {RX.DECLARED_EDGE_ASSIGNED}
    assert {r.receiver_decision_id for r in rows} == {episode.decision.decision_id}
    assert all(r.transmitted_tokens > 0 for r in rows)
    by_hash = {r.packet_hash: r for r in rows}
    assert by_hash[located.wrong.packet_hash].error_position_rel == 0.25
    assert by_hash[located.correct.packet_hash].error_position_rel is None
    assert len({(r.receiver_decision_id, r.sender_slot, r.source_round) for r in rows}) == 2


def test_a_pair_whose_packets_are_byte_identical_is_refused(pair):
    with pytest.raises(ContractViolation):
        RX.FixedPacketPair(source_item_id="i", panel_id=PANEL, correct=pair.correct,
                           wrong=pair.correct)


def test_generation_phases_agree_between_the_two_configs(e2b_block, registry):
    """R1: ``experiments.E2b.generation_phases`` and ``protocols.yaml``'s ``phases`` describe
    one fact, and a disagreement is a compile error rather than a silent extra call."""
    for name, spec in registry.receivers.items():
        if name.startswith("E2B"):
            RX.assert_generation_phases(e2b_block, spec)
    with pytest.raises(ContractViolation):
        RX.assert_generation_phases(e2b_block, registry.receivers["E2A_1_4_CORRECT"])


def test_the_template_list_of_an_arm_is_its_render_order(registry):
    """R1/R2: ``protocols.yaml``'s ``templates`` is the arm's render order and is checked, so a
    two-phase arm cannot quietly run two decision calls at B/2 -- a different experiment with
    the same name."""
    for name, spec in registry.receivers.items():
        RX.assert_templates(spec)
        assert tuple(spec.templates) == RX.PHASE_TEMPLATES[spec.phases]
    twisted = P.ReceiverSpec(
        packet_content=P.PacketContent.FULL, support=P.Support.HIDDEN, checker=False,
        draft_role=P.DraftRole.CORRECT, phases=2,
        templates=("receiver_decision", "receiver_decision"), display_label="TWISTED")
    with pytest.raises(ContractViolation):
        RX.assert_templates(twisted)


# --------------------------------------------------------------------------- E2b packets


def test_mod97_packets_are_rendered_and_carry_the_generated_error_position(nd_contract):
    """Blocking gap this closes: nothing turned a ``mod97.Trace`` into the fixed packets and
    the tool an E2b episode needs, so ``error_position_rel`` reached no table and the packets
    skipped the cap, clipping, denylist and tokenizer stamp every other packet goes through.

    The value asserted here is the **generator's**, read off the trace -- not a fixture
    literal.
    """
    problem = M.generate(SEED, SPEC.chain_depths[1], 7, spec=SPEC)
    pair, traces = RX.mod97_pair(problem, study_seed=STUDY_SEED, counter=counter,
                                 counter_tokenizer_hash="tok:test", tokens=nd_contract.tokens)

    wrong = M.wrong_trace(problem, STUDY_SEED)
    assert pair.error_position_rel == wrong.error_position_rel is not None
    assert pair.panel_id == problem.problem_id == problem.source_item_id
    # the trace map is keyed by the *displayed* packet id, which is what the tool serves
    assert set(traces) == {pair.correct.packet_id, pair.wrong.packet_id}
    assert traces[pair.correct.packet_id].violated_step is None
    assert traces[pair.wrong.packet_id].violated_step == wrong.violated_step
    # rendered through packets.render_packet: stamped, hashed, wrapped, un-clipped
    for packet in (pair.correct, pair.wrong):
        assert packet.counter_tokenizer_hash == "tok:test"
        assert packet.token_count == counter(packet.serialized)
        assert not packet.any_truncated
        PK.check_packet_bytes(packet.serialized)
    assert M.render_trace(traces[pair.wrong.packet_id]) in pair.wrong.serialized.replace("\\n", "\n")
    assert pair.correct.packet_hash != pair.wrong.packet_hash


def test_a_clipped_mod97_trace_is_refused(nd_contract):
    """A clipped trace would let the checker answer about rows the receiver never read."""
    problem = M.generate(SEED, SPEC.chain_depths[0], 8, spec=SPEC)
    tiny = K.TokenContract(thinking_cap=16, final_cap=32, served_window=4096, reserve=8,
                           packet_cap=8, rationale_cap=32, claims_max=4, claim_cap=16)
    with pytest.raises(ContractViolation):
        RX.mod97_pair(problem, study_seed=STUDY_SEED, counter=counter,
                      counter_tokenizer_hash="tok:test", tokens=tiny)


def test_an_e2b_episode_runs_end_to_end_on_generated_packets(nd_contract, registry, nd_ctx,
                                                             nd_serve, fake_tokenizer):
    """The whole E2b path with nothing hand-built: generator -> packets -> tool -> rows.

    The wrong packet's ``transmissions`` row carries the generator's error position, and every
    row carries the **receiver's** own token count of the replayed bytes.
    """
    problem = M.generate(SEED, SPEC.chain_depths[2], 11, spec=SPEC)
    pair, traces = RX.mod97_pair(problem, study_seed=STUDY_SEED, counter=counter,
                                 counter_tokenizer_hash="tok:donor", tokens=nd_contract.tokens)
    cond = condition(nd_contract, registry, "E2B_4_1_CHECK_WRONG", panel_id=problem.problem_id)
    tool = CH.CheckStepTool(traces, problem, enabled=True, max_checks=E2B.max_checks)
    violated = traces[pair.wrong.packet_id].violated_step
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer, tool=tool,
                  steps=((pair.wrong.packet_id, violated),))

    assert [r.status for r in episode.check_responses] == [CH.CheckStatus.INVALID]
    rows = RX.transmission_rows(episode, pair, receiver_counter=counter)
    by_hash = {r.packet_hash: r for r in rows}
    assert by_hash[pair.wrong.packet_hash].error_position_rel == pair.error_position_rel
    assert by_hash[pair.correct.packet_hash].error_position_rel is None
    for row in rows:
        assert row.counter_tokenizer_hash == "tok:donor"
        assert row.transmitted_tokens_receiver == counter(
            next(p.serialized for p in episode.displayed if p.packet_hash == row.packet_hash))


def test_the_receiver_token_count_is_the_reading_models_not_the_senders(
        nd_contract, registry, pair, nd_ctx, nd_serve, fake_tokenizer):
    """Spec 9.1's receiver-model factor: one frozen packet, two receivers, two counts.

    ``transmitted_tokens`` is the sender's (``counter_tokenizer_hash`` says whose) and
    ``transmitted_tokens_receiver`` is the reading model's; the bytes are replayed verbatim and
    never re-clipped, so the two columns are the same evidence measured by two tokenizers.
    """
    cond = condition(nd_contract, registry, "E2A_HIDDEN_CORRECT")
    episode = run(cond, pair, nd_ctx, nd_serve, fake_tokenizer)

    def chatty(text: str) -> int:                # a coarser tokenizer than the donor's
        return len(text)

    sender_rows = RX.transmission_rows(episode, pair, receiver_counter=counter)
    other_rows = RX.transmission_rows(episode, pair, receiver_counter=chatty)
    assert all(r.transmitted_tokens_receiver == r.transmitted_tokens for r in sender_rows)
    assert all(r.transmitted_tokens_receiver > r.transmitted_tokens for r in other_rows)
    assert {r.counter_tokenizer_hash for r in other_rows} == {"tok:test"}
    assert all(a.packet_hash == b.packet_hash for a, b in zip(sender_rows, other_rows))


def test_the_endorsement_panel_size_comes_from_study_yaml(nd_contract, registry, pair, nd_ctx,
                                                          nd_serve, fake_tokenizer):
    """Contract preamble: no experimental number is a module constant.  The 1:4 / 4:1 levels
    are *derived* from ``experiments.E2a.support_panel_size``, so a different panel is a config
    edit the consumer check sees -- not an edit to this module."""
    assert not hasattr(RX, "SUPPORT_PANEL_SIZE")
    assert RX.support_panel_size(E2A) == E2A.support_panel_size == PANEL_SIZE

    lines = RX.support_fields(P.Support.CORRECT_MINORITY, pair.labels, b"\x00",
                              panel_size=3)
    assert f"supported by 1 of 3 advisers" in lines[1]
    assert RX.support_counts(P.Support.CORRECT_MAJORITY, 3) == (2, 1)
    with pytest.raises(ContractViolation):
        RX.support_counts(P.Support.CORRECT_MINORITY, 1)
    with pytest.raises(ContractViolation):
        condition(nd_contract, registry, "E2A_1_4_CORRECT").__class__(
            **{**{f: getattr(condition(nd_contract, registry, "E2A_1_4_CORRECT"), f)
                  for f in RX.ReceiverCondition.__dataclass_fields__},
               "support_panel_size": 1})
