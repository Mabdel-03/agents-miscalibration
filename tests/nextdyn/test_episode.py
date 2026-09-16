"""The synchronous loop: one snapshot, one batch, n*R decisions, no early stop (spec 6.4, 8.4).

The fake ``generate`` here answers as a function of the **seed key** -- item, round, slot --
and never of the prompt bytes.  That is exactly the common-random-numbers property D4 buys by
keeping the condition out of the seed, and it is what makes the byte-equality tests between
paired arms meaningful rather than tautological.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import episode as E
from agents_scaling.nextdyn import ids, prompts
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.schema import ParseStatus, PublicOutput
from tests.study.conftest import StubTokenizer

SEED = bytes.fromhex("55" * 32)
STUDY_ID = "task_conditioned_dynamics_v1"
TOKENS = K.TokenContract(thinking_cap=64, final_cap=32, served_window=4096, reserve=8,
                         packet_cap=64, rationale_cap=32, claims_max=4, claim_cap=16)
MODEL = K.ModelCell(name="primary", hf_id="Qwen/Qwen3-8B", revision="rev-a",
                    tokenizer_revision="tok-a", adapter=K.ADAPTER_NATIVE_THINKING,
                    profile="nd1-8B", tp=1, served_model_name="nd1-8B", lane="8B",
                    total_output_tokens=None)


# ------------------------------------------------------------------ structural WP3 stand-ins


@dataclass(frozen=True)
class _Item:
    source_item_id: str = "item-1"
    item_variant_id: str = "item-1.v0"
    rendered_question: str = "What is 2 + 2 ?"


@dataclass(frozen=True)
class _Family:
    answer_instruction: str = "Answer with the integer ."


@dataclass(frozen=True)
class _Slot:
    decision_id: str
    generation_uid: str
    output: PublicOutput


@dataclass(frozen=True)
class _Bundle:
    root_team_id: str
    n: int
    slots: tuple[_Slot, ...]
    export_hash: str
    focal_slot: int | None
    source_item_id: str = "item-1"
    item_variant_id: str = "item-1.v0"


def _output(answer: str) -> PublicOutput:
    return PublicOutput(reasoning=f"rationale for {answer}", shareable_claims=(f"claim {answer}",),
                        final_answer=answer, confidence=0.5, field_order=(), order_violation=False,
                        parse_status=ParseStatus.OK, raw_text="{}",
                        content_hash=ids.content_hash(answer.encode()), over_limits={},
                        final_answer_span=None, confidence_span=None,
                        confidence_emitted_before_answer=False)


def bundle(n: int = 5, focal: int | None = 2, answers=None) -> _Bundle:
    answers = answers or [chr(ord("A") + slot) for slot in range(n)]
    return _Bundle(root_team_id="team:" + "ab" * 16, n=n,
                   slots=tuple(_Slot(decision_id=f"d:bank:{slot}", generation_uid=f"g:bank:{slot}",
                                     output=_output(answers[slot])) for slot in range(n)),
                   export_hash="x" * 32, focal_slot=focal)


def spec_for(condition: str, *, n: int = 5, revisions: int = 4, focal: int | None = 2,
             answers=None, replicate: int = 0) -> E.EpisodeSpec:
    return E.EpisodeSpec(
        cell_kind=CellKind.EPISODE, source_item=_Item(), roots=bundle(n, focal, answers),
        visibility=P.load_registry().episodes[condition], model=MODEL, tokens=TOKENS,
        replicate=replicate, revisions=revisions, n=n, split_role="pilot", study_id=STUDY_ID,
        root_round=0, top_logprobs=1, family=_Family())


# ------------------------------------------------------------------ the fake generation path


@dataclass
class _Record:
    final_text: str
    finish_reason: str = "stop"
    completion_token_ids: tuple[int, ...] = ()
    generation_uid: str = "g:x"
    aliased: bool = False


@dataclass
class _Generator:
    """Answers from the seed key alone (item, round, slot) -- never from the prompt bytes."""

    answer_of: Any = None
    calls: list[Any] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    barrier: threading.Barrier | None = None
    refuse: set[tuple[int, int]] = field(default_factory=set)
    peak: int = 0
    _live: int = 0

    def __call__(self, request):
        with self.lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
            self.calls.append(request)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=5)
            slot, round_ = request.seed_key.actor_slot, request.seed_key.step_slot
            if (round_, slot) in self.refuse:
                raise _Refusal()
            answer = (self.answer_of or (lambda s, r: chr(ord("A") + (s + r) % 4)))(slot, round_)
            text = ('{"reasoning": "r ' + f"{slot} {round_}" + '", "shareable_claims": [ "c" ], '
                    f'"final_answer": "{answer}", "confidence": 0.5 }}')
            return _Record(final_text=text, generation_uid=f"g:{round_}:{slot}")
        finally:
            with self.lock:
                self._live -= 1


class _Refusal(Exception):
    """A build-time context refusal, typed the way ``inference.ContextTooLong`` types itself."""

    parse_status = ParseStatus.CONTEXT_FAILURE


def run(condition: str, generator=None, **kwargs) -> E.Episode:
    generator = generator or _Generator()
    return E.run_episode(spec_for(condition, **kwargs), generator, tokenizer=StubTokenizer(),
                         study_seed=SEED)


def prompts_of(episode: E.Episode, round_: int, slot: int) -> str:
    for decision in episode.decisions:
        if decision.round_ == round_ and decision.agent_slot == slot:
            return "\n".join(m["content"] for m in decision.messages)
    raise AssertionError((round_, slot))


# ------------------------------------------------------------------ structure


def test_an_episode_produces_exactly_n_times_r_decisions_and_no_early_stop():
    for n, revisions in ((3, 2), (5, 4), (9, 1)):
        gen = _Generator()
        episode = run("PEER", gen, n=n, revisions=revisions, focal=0)
        assert len(episode.revisions) == n * revisions
        assert len(episode.references) == n
        assert len(gen.calls) == n * revisions            # round 0 generated nothing


def test_all_agents_update_after_unanimity():
    """Spec 8.4: unanimity is not a stopping rule; every agent still updates."""
    gen = _Generator(answer_of=lambda slot, round_: "A")
    episode = run("PEER", gen, revisions=4)
    assert episode.states[1].unanimous() == "A"
    assert len(episode.revisions) == 5 * 4
    for round_ in range(1, 5):
        assert len([d for d in episode.revisions if d.round_ == round_]) == 5


def test_four_revisions_produce_five_observed_states_in_the_episode():
    episode = run("PEER", revisions=4)
    assert len(episode.states) == 5
    assert [s.round_ for s in episode.states] == [0, 1, 2, 3, 4]


def test_historical_coverage_is_nondecreasing_over_the_episode():
    episode = run("PEER", revisions=4)
    sizes = [len(s.historical_classes()) for s in episode.states]
    assert sizes == sorted(sizes)


def test_a_round_goes_out_as_one_batch_of_n():
    """Concurrency is the contract, not an optimisation: the barrier only clears when all n
    calls of a round are in flight at once (``dispatch_batch(max_workers=n)``)."""
    for n in (3, 5):
        gen = _Generator(barrier=threading.Barrier(n))
        episode = run("PEER", gen, n=n, revisions=2, focal=0)
        assert gen.peak == n
        assert len(episode.revisions) == n * 2


def test_every_prompt_of_a_round_is_rendered_before_any_of_them_is_dispatched(monkeypatch):
    """Spec 6.4: "Freeze the entire input set before dispatch."."""
    gen = _Generator()
    renders: list[int] = []
    original = prompts.render_revision

    def spy(*args, **kwargs):
        renders.append(len(gen.calls))
        return original(*args, **kwargs)

    monkeypatch.setattr(prompts, "render_revision", spy)
    E.run_episode(spec_for("PEER", revisions=3), gen, tokenizer=StubTokenizer(), study_seed=SEED)
    # each round renders its five prompts while the generate count is still the round's start
    assert renders == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10]


def test_peer_messages_use_only_previous_round_in_a_running_episode():
    episode = run("PEER", revisions=3)
    for round_, plan in episode.plans.items():
        for intents in plan.values():
            assert {i.source_round for i in intents} == {round_ - 1}
            P.assert_packet_rounds(intents, round_=round_)


# ------------------------------------------------------------------ roots and pairing


def test_paired_protocols_have_byte_identical_roots():
    """Episodes never generate roots: every fork of one team references the same bank draws,
    so the round-0 rows agree in content hash and generation uid and the store sees no new
    root request (plan D14)."""
    episodes = {}
    for name in ("PRIVATE", "PEER", "FRESH"):
        gen = _Generator()
        episodes[name] = (run(name, gen, revisions=2), gen)
    first = episodes["PRIVATE"][0]
    for name, (episode, gen) in episodes.items():
        assert episode.spec.roots.export_hash == first.spec.roots.export_hash
        assert episode.spec.roots.root_team_id == first.spec.roots.root_team_id
        assert episode.spec.pairing_id == first.spec.pairing_id
        assert [d.generation_uid for d in episode.references] == \
            [d.generation_uid for d in first.references]
        assert [d.output.content_hash for d in episode.references] == \
            [d.output.content_hash for d in first.references]
        assert all(d.is_reference and d.reuse_of for d in episode.references)
        assert len(gen.calls) == 5 * 2, name       # no root was generated in any arm


def test_the_condition_is_in_the_key_and_never_in_the_seed():
    """R5 / D4: paired arms share seed keys byte for byte; only the composite key differs."""
    private, peer = spec_for("PRIVATE"), spec_for("PEER")
    assert private.protocol_hash != peer.protocol_hash
    assert private.episode_id != peer.episode_id
    for slot in range(5):
        a = ids.seed_key_for(private.key_for(round_=1, agent_slot=slot, phase=K.PHASE_REVISION),
                             split_role="pilot")
        b = ids.seed_key_for(peer.key_for(round_=1, agent_slot=slot, phase=K.PHASE_REVISION),
                             split_role="pilot")
        assert a == b


# ------------------------------------------------------------------ what each arm reads


def test_private_has_no_peer_content_in_any_prompt():
    episode = run("PRIVATE", revisions=3)
    for round_ in range(1, 4):
        for slot in range(5):
            text = prompts_of(episode, round_, slot)
            block = text.split("OTHER DRAFTS\n", 1)[1].split("\n\n", 1)[0]
            assert block == prompts.NONE_LITERAL
            for other in range(5):
                if other != slot:
                    assert episode.packets[round_][other].serialized not in text


def test_fresh_has_no_own_or_peer_history():
    """Spec 6.3: FRESH uses the root template and sees no own or peer history."""
    episode = run("FRESH", revisions=3)
    root = "\n".join(m["content"] for m in prompts.render_root(_Item(), _Family()))
    for round_ in range(1, 4):
        for slot in range(5):
            assert prompts_of(episode, round_, slot) == root
            assert "WORKING DRAFT" not in prompts_of(episode, round_, slot)


def test_r4_prefix_of_r8_is_identical_when_prompts_are_horizon_free():
    short = run("PEER", _Generator(), revisions=4)
    long = run("PEER", _Generator(), revisions=8)
    for round_ in range(1, 5):
        for slot in range(5):
            assert prompts_of(short, round_, slot) == prompts_of(long, round_, slot)
            assert short.decisions[0].key.decision_id() == long.decisions[0].key.decision_id()
    assert len(long.revisions) == 5 * 8


def test_the_in0_out1_focal_prompt_equals_the_private_prompt_at_every_round():
    """Plan §1.7 alias byte test: the isolated focal slot reads exactly what a PRIVATE agent
    reads, because both render the literal NONE from the same own packet (spec 10.1 reuse)."""
    gated = run("IN0_OUT1", _Generator(), revisions=4, focal=2)
    private = run("PRIVATE", _Generator(), revisions=4, focal=2)
    for round_ in range(1, 5):
        assert prompts_of(gated, round_, 2) == prompts_of(private, round_, 2)


def test_the_in0_out1_nonfocal_round_one_prompt_equals_the_peer_prompt():
    """Plan §1.7: at round 1 a non-focal agent's sender set and display order are PEER's, and
    the packets come from the same roots, so the bytes agree."""
    gated = run("IN0_OUT1", _Generator(), revisions=1, focal=2)
    peer = run("PEER", _Generator(), revisions=1, focal=2)
    for slot in (0, 1, 3, 4):
        assert prompts_of(gated, 1, slot) == prompts_of(peer, 1, slot)
    assert prompts_of(gated, 1, 2) != prompts_of(peer, 1, 2)


def test_a_filled_arm_shows_the_same_number_of_packet_slots_as_its_unfilled_twin():
    filled = run("IN0_OUT1_FILLED", _Generator(), revisions=2, focal=2)
    peer = run("PEER", _Generator(), revisions=2, focal=2)
    for round_ in (1, 2):
        assert len(filled.plans[round_][2]) == len(peer.plans[round_][2]) == 4
        assert all(i.filler for i in filled.plans[round_][2])
        text = prompts_of(filled, round_, 2)
        block = text.split("OTHER DRAFTS\n", 1)[1].split("\n\n", 1)[0]
        assert len(block.split("\n")) == 4
        assert "final_answer" not in block


def test_answer_only_transmits_no_rationale_between_agents():
    episode = run("PEER_ANSWER_ONLY", _Generator(), revisions=1)
    packet = episode.packets[1][0]
    assert "reasoning" not in packet.serialized
    assert "final_answer" in packet.serialized


# ------------------------------------------------------------------ failures


def test_a_context_refusal_is_a_typed_decision_and_never_a_retry():
    gen = _Generator(refuse={(1, 3)})
    episode = run("PEER", gen, revisions=2)
    refused = [d for d in episode.revisions if d.context_failure]
    assert len(refused) == 1
    assert refused[0].round_ == 1 and refused[0].agent_slot == 3
    assert refused[0].output.parse_status is ParseStatus.CONTEXT_FAILURE
    assert refused[0].record is None and refused[0].generation_uid is None
    assert len(episode.revisions) == 5 * 2              # the slot still spent its opportunity
    assert len(gen.calls) == 5 * 2                      # asked once, never re-asked


def test_a_refused_slot_does_not_cancel_the_rest_of_its_round():
    gen = _Generator(refuse={(1, 0)})
    episode = run("PEER", gen, revisions=1)
    ok = [d for d in episode.revisions if not d.context_failure]
    assert len(ok) == 4 and all(d.output.valid for d in ok)


def test_an_error_that_is_not_a_context_refusal_still_propagates():
    class _Boom(Exception):
        pass

    def generate(request):
        raise _Boom()

    with pytest.raises(_Boom):
        E.run_episode(spec_for("PEER", revisions=1), generate, tokenizer=StubTokenizer(),
                      study_seed=SEED)


def test_a_spec_whose_team_size_disagrees_with_its_bundle_is_refused():
    with pytest.raises(ContractViolation, match="root bundle"):
        E.EpisodeSpec(cell_kind=CellKind.EPISODE, source_item=_Item(), roots=bundle(5),
                      visibility=P.load_registry().episodes["PEER"], model=MODEL, tokens=TOKENS,
                      replicate=0, revisions=1, n=3, split_role="pilot", study_id=STUDY_ID,
                      root_round=0, top_logprobs=1, family=_Family())


def test_run_episode_drives_episode_cells_only():
    with pytest.raises(ContractViolation, match="episode"):
        E.EpisodeSpec(cell_kind=CellKind.RECEIVER, source_item=_Item(), roots=bundle(5),
                      visibility=P.load_registry().episodes["PEER"], model=MODEL, tokens=TOKENS,
                      replicate=0, revisions=1, n=5, split_role="pilot", study_id=STUDY_ID,
                      root_round=0, top_logprobs=1, family=_Family())


# ------------------------------------------------------------------ vendoring parity


def test_vendored_assert_packet_rounds_matches_ard_at_n5():
    """Parity with ``ard/episode.py:489`` minus ARD36's single HUB5 exception (nextdyn has no
    same-round arm): the same (round, source_round) pairs are accepted and refused."""
    from agents_scaling.ard import contract as ARD_K
    from agents_scaling.ard import episode as ARD_E
    from agents_scaling.ard.packets import ArdPacket
    from agents_scaling.study.types import ProtocolError

    for round_ in range(1, 6):
        for source_round in range(0, 6):
            intent = P.TransmissionIntent(sender_slot=1, source_round=source_round,
                                          source_decision_id="d:1", content=P.PacketContent.FULL,
                                          filler=False, display_position=0)
            packet = ArdPacket(packet_id="p", kind="report", pseudonym="agent-1",
                               content_sha256="c", fields={}, truncated={}, spans=(),
                               recipient_tokens=1, serialized="{}",
                               source_round=source_round, sender_slot=1)
            mine = _raises(lambda: P.assert_packet_rounds((intent,), round_=round_),
                           ContractViolation)
            theirs = _raises(lambda: ARD_E._assert_packet_rounds(
                ARD_K.Protocol.PEER, (packet,), round_=round_, phase=ARD_E.PHASE_UPDATE),
                ProtocolError)
            assert mine == theirs, (round_, source_round)


def test_vendored_execute_updates_matches_the_ard_batch_discipline_at_n5():
    """``ard/episode.py:504``'s two invariants, kept parametrically: every request is built
    from the snapshot before the first dispatch, and slot k is always results[k]."""
    gen = _Generator(barrier=threading.Barrier(5))
    episode = run("PEER", gen, n=5, revisions=1)
    by_slot = [d.agent_slot for d in episode.revisions]
    assert by_slot == list(range(5))
    assert [r.seed_key.actor_slot for r in gen.calls] == list(range(5))
    assert gen.peak == 5


def _raises(call, exc_type) -> bool:
    try:
        call()
    except exc_type:
        return True
    return False


def test_the_episode_shape_must_be_the_contracts_team_shape(study_cfg):
    """R1: N and R are declared once, in ``study.protocol``; an episode that disagrees is a
    planner defect rather than a silent per-cell override."""
    contract = study_cfg.contract
    ok = spec_for("PEER", n=contract.team.n_agents, revisions=contract.team.revision_rounds)
    E.assert_team_shape(ok, contract)
    with pytest.raises(ContractViolation, match="team shape"):
        E.assert_team_shape(spec_for("PEER", n=3, revisions=contract.team.revision_rounds,
                                     focal=0), contract)
    with pytest.raises(ContractViolation, match="revision rounds"):
        E.assert_team_shape(spec_for("PEER", n=contract.team.n_agents, revisions=1), contract)
