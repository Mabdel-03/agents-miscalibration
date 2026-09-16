"""The transmitted packet: the cap, the priority, the filler and the denylist (spec 6.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids, packets as PK
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.protocols import PacketContent, TransmissionIntent
from agents_scaling.nextdyn.schema import ParseStatus, PublicOutput

TOKENIZER_HASH = "0123456789abcdef"
SEED = bytes.fromhex("33" * 32)
TOKENS = K.TokenContract(thinking_cap=2048, final_cap=1024, served_window=32768, reserve=256,
                         packet_cap=512, rationale_cap=256, claims_max=4, claim_cap=64)


def counter(text: str) -> int:
    return len(text.split())


def _out(*, answer="B", reasoning="a public rationale", claims=("first claim", "second claim"),
         status=ParseStatus.OK) -> PublicOutput:
    return PublicOutput(reasoning=reasoning, shareable_claims=tuple(claims), final_answer=answer,
                        confidence=0.9, field_order=(), order_violation=False, parse_status=status,
                        raw_text="{}", content_hash=ids.content_hash(str(answer).encode()),
                        over_limits={}, final_answer_span=None, confidence_span=None,
                        confidence_emitted_before_answer=False)


def render(out, *, cap=512, content=PacketContent.FULL, pseudonym="agent-0"):
    return PK.render_packet(out, pseudonym=pseudonym, counter=counter,
                            counter_tokenizer_hash=TOKENIZER_HASH, cap=cap, content=content)


# ---------------------------------------------------------------------------- shape


def test_transmitted_packet_carries_no_confidence_and_no_slot():
    """Spec 6.1: confidence is hidden in both renderings, and no visible id encodes a slot,
    a source model size or a treatment."""
    packet = render(_out())
    wrapper = json.loads(packet.serialized)
    assert set(wrapper) <= PK.ALLOWED_KEYS
    assert "confidence" not in packet.serialized
    assert wrapper[PK.FROM_KEY] == "agent-0"
    assert not any(k in wrapper for k in PK.PACKET_METADATA_DENYLIST)
    assert str(packet.pseudonym).startswith("agent-")


def test_the_packet_records_the_tokenizer_that_counted_it():
    packet = render(_out())
    assert packet.counter_tokenizer_hash == TOKENIZER_HASH
    assert packet.token_count == counter(packet.serialized)
    assert packet.packet_hash and packet.original_content_hash == _out().content_hash


def test_tokenizer_hash_is_the_pinned_tokenizer_not_the_weights():
    cell = K.ModelCell(name="m", hf_id="org/model", revision="rev-a",
                       tokenizer_revision="tok-1", adapter=K.ADAPTER_NATIVE_THINKING,
                       profile="p", tp=1, served_model_name="m", lane="8B",
                       total_output_tokens=None)
    other_weights = K.ModelCell(**{**cell.__dict__, "revision": "rev-b"})
    other_tokenizer = K.ModelCell(**{**cell.__dict__, "tokenizer_revision": "tok-2"})
    assert PK.tokenizer_hash(cell) == PK.tokenizer_hash(other_weights)
    assert PK.tokenizer_hash(cell) != PK.tokenizer_hash(other_tokenizer)


def test_own_and_peer_rendering_are_identical_except_source_label():
    """Spec 6.1: "Use the same frozen rendering for an agent's own previous public draft and
    its peers, except for source-slot labels."."""
    out = _out()
    own = render(out, pseudonym="agent-0")
    peer = render(out, pseudonym="agent-3")
    assert json.loads(own.serialized) | {PK.FROM_KEY: "x"} == \
        json.loads(peer.serialized) | {PK.FROM_KEY: "x"}
    assert own.serialized != peer.serialized


def test_answer_only_transmits_the_answer_alone():
    packet = render(_out(), content=PacketContent.ANSWER_ONLY)
    wrapper = json.loads(packet.serialized)
    assert set(wrapper) == {PK.FROM_KEY, "final_answer", PK.TRUNCATED_KEY}
    assert wrapper["final_answer"] == "B"


def test_a_condition_that_transmits_nothing_never_renders_a_packet():
    with pytest.raises(ContractViolation):
        render(_out(), content=PacketContent.NONE)


# ---------------------------------------------------------------------------- the cap


def test_512_cap_clips_claims_before_reasoning_before_answer():
    """Spec 6.1 priority: "Preserve the final answer first, then the public rationale, then
    claims in order."."""
    out = _out(answer="the exact final answer", reasoning=" ".join(f"r{i}" for i in range(40)),
               claims=tuple(" ".join(f"c{j}{i}" for i in range(6)) for j in range(3)))
    assert PK.FIELD_PRIORITY == ("final_answer", "reasoning", "shareable_claims")

    tight = render(out, cap=12)
    fields = json.loads(tight.serialized)
    assert fields["final_answer"] == "the exact final answer"        # never clipped
    assert fields["shareable_claims"] == []                          # dropped first
    assert 0 < len(fields["reasoning"].split()) < 40                 # clipped second
    assert tight.truncated["shareable_claims"] is True
    assert tight.truncated["reasoning"] is True
    assert tight.truncated["final_answer"] is False
    assert tight.token_count <= 12

    roomy = render(out, cap=512)
    assert not roomy.any_truncated
    assert json.loads(roomy.serialized)["shareable_claims"] == list(out.shareable_claims)


def test_the_cap_is_counted_on_the_serialized_wrapper():
    out = _out(reasoning=" ".join("word" for _ in range(200)))
    for cap in (8, 20, 64, TOKENS.packet_cap):
        packet = render(out, cap=cap)
        assert counter(packet.serialized) <= cap


def test_a_cross_model_replay_transmits_the_stored_bytes_and_never_reclips():
    packet = render(_out(reasoning=" ".join("word" for _ in range(50))), cap=20)
    stored = packet.serialized
    assert PK.check_packet_bytes(stored) is None
    assert packet.serialized == stored


def test_an_ungradeable_output_becomes_the_typed_unparsed_packet():
    out = _out(answer=None, status=ParseStatus.INVALID_JSON)
    packet = render(out)
    wrapper = json.loads(packet.serialized)
    assert wrapper == {PK.FROM_KEY: "agent-0", PK.STATUS_KEY: PK.STATUS_UNPARSED}
    assert "B" not in packet.serialized
    other = PK.unparsed_packet("agent-0", counter, counter_tokenizer_hash=TOKENIZER_HASH)
    assert other.serialized == packet.serialized      # one failed slot looks like any other


# ---------------------------------------------------------------------------- filler


def test_missing_packet_padding_has_no_task_content():
    """Spec 10.3: a removed slot may be padded, and the padding carries no task content --
    no digits, no option letters, nothing from the question or from any answer."""
    real = render(_out(answer="42", reasoning="the quotient is forty two"), cap=64)
    filler = PK.inert_filler(real.token_count, SEED, counter, pseudonym="agent-2",
                             counter_tokenizer_hash=TOKENIZER_HASH)
    body = json.loads(filler.serialized)[PK.FILLER_KEY]
    vocabulary = set(__import__("agents_scaling.nextdyn.prompts", fromlist=["x"]).filler_words())
    assert set(body.split()) <= vocabulary
    assert not any(ch.isdigit() for ch in body)
    assert "42" not in filler.serialized and "quotient" not in filler.serialized
    assert filler.filler is True and filler.original_content_hash is None


def test_filler_is_length_matched_in_the_senders_counter():
    real = render(_out(reasoning=" ".join("word" for _ in range(30))), cap=64)
    filler = PK.inert_filler(real.token_count, SEED, counter, pseudonym="agent-2",
                             counter_tokenizer_hash=TOKENIZER_HASH)
    assert filler.token_count <= real.token_count
    assert filler.token_count >= real.token_count - 2


def test_filler_is_deterministic_in_the_seed():
    a = PK.inert_filler(20, SEED, counter, pseudonym="agent-2", counter_tokenizer_hash=TOKENIZER_HASH)
    b = PK.inert_filler(20, SEED, counter, pseudonym="agent-2", counter_tokenizer_hash=TOKENIZER_HASH)
    c = PK.inert_filler(20, bytes(32), counter, pseudonym="agent-2",
                        counter_tokenizer_hash=TOKENIZER_HASH)
    assert a.serialized == b.serialized and a.serialized != c.serialized


# ---------------------------------------------------------------------------- denylist


def test_the_denylist_refuses_treatment_metadata_but_not_model_bytes():
    with pytest.raises(LeakageError, match="undeclared key"):
        PK.check_packet_bytes(json.dumps({"from": "agent-0", "protocol": "PEER"}))
    with pytest.raises(LeakageError, match="not an opaque pseudonym"):
        PK.check_packet_bytes(json.dumps({"from": "slot3", "final_answer": "B"}))
    with pytest.raises(LeakageError):
        PK.check_packet_bytes("not json")
    # the model's own excerpt may say anything; it is an exact excerpt and is never rewritten
    PK.check_packet_bytes(json.dumps({"from": "agent-1",
                                      "final_answer": "the correct answer is my confidence"}))


def test_candidate_ids_and_slot_labels_do_not_encode_correctness():
    right = render(_out(answer="B"))
    wrong = render(_out(answer="A"))
    for packet in (right, wrong):
        assert packet.packet_id.startswith(PK.PACKET_ID_PREFIX)
        assert packet.pseudonym.startswith("agent-")
    assert right.packet_id != wrong.packet_id


def test_pseudonyms_are_keyed_on_the_pairing_and_are_n_parametric():
    for n in (3, 5, 9):
        names = PK.pseudonym_map(SEED, "pairing-1", n)
        assert len(set(names)) == n
        assert sorted(names) == sorted(f"agent-{i}" for i in range(n))
    assert PK.pseudonym_map(SEED, "pairing-1", 5) == PK.pseudonym_map(SEED, "pairing-1", 5)
    assert PK.pseudonym_map(SEED, "pairing-1", 5) != PK.pseudonym_map(SEED, "pairing-2", 5)


# ---------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class _Family:
    max_question_tokens = 1200
    template_tokens = 80


def test_worst_case_prompt_tokens_counts_own_plus_peers_plus_task(study_cfg):
    contract = study_cfg.contract
    model = contract.model(sorted(contract.models)[0])
    total = PK.worst_case_prompt_tokens(contract, _Family(), 5, model)
    assert total == 1200 + 80 + contract.tokens.packet_cap * 5
    assert total + contract.tokens.max_new_tokens + contract.tokens.reserve <= \
        contract.tokens.served_window
    with pytest.raises(ContractViolation):
        PK.worst_case_prompt_tokens(contract, _Family(), 1, model)


def test_transmissions_for_emits_one_row_per_slot_including_filler():
    packets = {slot: render(_out(answer=f"ans{slot}"), pseudonym=f"agent-{slot}")
               for slot in range(3)}
    packets[2] = PK.inert_filler(10, SEED, counter, pseudonym="agent-2",
                                 counter_tokenizer_hash=TOKENIZER_HASH)
    plan = {0: (
        TransmissionIntent(sender_slot=1, source_round=2, source_decision_id="d:1",
                           content=PacketContent.FULL, filler=False, display_position=0),
        TransmissionIntent(sender_slot=2, source_round=2, source_decision_id="",
                           content=PacketContent.FULL, filler=True, display_position=1),
    )}
    rows = PK.transmissions_for(plan, packets, "d:receiver", destination_round=3, receiver_slot=0,
                                episode_id="episode|x", cell_kind="episode", protocol_hash="ph")
    assert [r.sender_slot for r in rows] == [1, 2]
    assert [r.declared_edge for r in rows] == [PK.DECLARED_EDGE_GRAPH, PK.DECLARED_EDGE_FILLER]
    assert [r.filler_flag for r in rows] == [False, True]
    assert rows[0].source_decision_id == "d:1" and rows[1].source_decision_id is None
    assert rows[1].filler_tokens == packets[2].token_count
    assert all(r.source_round == 2 and r.destination_round == 3 for r in rows)
    assert len({r.transmission_id for r in rows}) == 2
    assert PK.transmission_ids("d:receiver", plan[0]) == tuple(r.transmission_id for r in rows)


def test_a_missing_rendered_packet_is_a_refusal_not_a_silent_gap():
    plan = {0: (TransmissionIntent(sender_slot=4, source_round=0, source_decision_id="d:4",
                                   content=PacketContent.FULL, filler=False, display_position=0),)}
    with pytest.raises(ContractViolation):
        PK.transmissions_for(plan, {}, "d:r", destination_round=1, receiver_slot=0,
                             episode_id="e", cell_kind="episode", protocol_hash="ph")
