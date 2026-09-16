"""WP5: the E2a fixed-packet writer (spec 9.1; plan §1.8).

Two things have to be true of this writer for E2a to mean anything.  The packets the receiver
replays must be the *same kind of object* an E1 agent transmits -- otherwise the spec-9.7 link
between the endorsement assay and the natural dynamics compares two different artifacts -- and
the sealed panel must carry no grade: no ``semantic_correct``, no canonical class, no donor
uid, nothing the grader produced about the source item.

What the panel *does* carry is each packet's design **role**, because the E2a arms are defined
over it (``draft_role``, the endorsement counts), and the tests below check the honest version
of the blinding claim: the packet entries and the audit slots are emitted in one blind order,
so over a set of items the correct packet is not always slot 0 and a position is not a label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn import natural_packets as NP
from agents_scaling.nextdyn import packets as PK
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn.errors import LeakageError

STUDY_SEED = b"nextdyn-e2a-writer-seed"
TOKENS = K.TokenContract(thinking_cap=16, final_cap=32, served_window=4096, reserve=8,
                         packet_cap=512, rationale_cap=32, claims_max=4, claim_cap=16)


def counter(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class FakeItem:
    source_item_id: str = "nditem0000000002"
    item_variant_id: str = "nditem0000000002.v0"
    rendered_question: str = "A question."


def output(answer: str, reasoning: str, *, valid: bool = True) -> schema.PublicOutput:
    return schema.PublicOutput(
        reasoning=reasoning, shareable_claims=("a claim",),
        final_answer=answer if valid else None, confidence=0.6,
        field_order=schema.DECLARED_FIELD_ORDER, order_violation=False,
        parse_status=schema.ParseStatus.OK if valid else schema.ParseStatus.NO_JSON,
        raw_text="{}", content_hash=f"c:{answer}", over_limits={}, final_answer_span=None,
        confidence_span=None, confidence_emitted_before_answer=False)


def grade(uid: str, state: str) -> T.GradePrivateRow:
    return T.GradePrivateRow(
        generation_uid=uid, grader_hash="grader:v1", decision_ids_json=json.dumps([f"d:{uid}"]),
        source_item_id=FakeItem().source_item_id, family="MATH", semantic_correct=state,
        grading_status="resolved", u_reason=None, extracted="x", canonical_class=f"cls-{state}",
        gold_ref_hash="gold:1", audit_status="none", grader_version="v1")


def draw(index: int, state: str, answer: str, *, valid: bool = True) -> NP.DonorDraw:
    uid = f"g:donor{index}"
    return NP.DonorDraw(generation_uid=uid, decision_id=f"d:donor{index}", draw_index=index,
                        output=output(answer, f"rationale {index}", valid=valid),
                        grade_row=grade(uid, state))


@pytest.fixture
def donors():
    return (draw(0, "W", "11"), draw(1, "C", "42"), draw(2, "C", "42"),
            draw(3, "U", "", valid=False), draw(4, "W", "13"))


@pytest.fixture
def pair(donors):
    return NP.fix_natural_pair(FakeItem(), donors, study_seed=STUDY_SEED, counter=counter,
                               counter_tokenizer_hash="tok:test", tokens=TOKENS)


# --------------------------------------------------------------------------- selection


def test_one_correct_and_one_wrong_donor_are_fixed_per_item(pair, donors):
    """Spec 9.1: "Recruit ... one correct and one wrong donor packet"; the U draws are never
    donors, because an ungradeable draw is not evidence of either kind."""
    states = {d.generation_uid: d.state for d in donors}
    assert states[pair.correct_generation_uid] == "C"
    assert states[pair.wrong_generation_uid] == "W"
    assert pair.correct.packet_hash != pair.wrong.packet_hash
    assert pair.correct.pseudonym != pair.wrong.pseudonym


def test_an_item_without_both_states_is_not_in_the_panel():
    """``None``, not an exception: an item that cannot supply the contrast is simply out, and
    the recruitment count records it (selection happens "without receiver outcomes")."""
    only_correct = (draw(0, "C", "42"), draw(1, "C", "42"))
    assert NP.fix_natural_pair(FakeItem(), only_correct, study_seed=STUDY_SEED, counter=counter,
                               counter_tokenizer_hash="tok:test", tokens=TOKENS) is None
    ungradeable = (draw(0, "C", "42"), draw(1, "U", "", valid=False))
    assert NP.fix_natural_pair(FakeItem(), ungradeable, study_seed=STUDY_SEED, counter=counter,
                               counter_tokenizer_hash="tok:test", tokens=TOKENS) is None


def test_the_donor_choice_is_deterministic_and_blind(donors):
    """The same seed picks the same two donors; a different seed may pick different ones, and
    neither choice reads the draw order."""
    first = NP.fix_natural_pair(FakeItem(), donors, study_seed=STUDY_SEED, counter=counter,
                                counter_tokenizer_hash="tok:test", tokens=TOKENS)
    again = NP.fix_natural_pair(FakeItem(), tuple(reversed(donors)), study_seed=STUDY_SEED,
                                counter=counter, counter_tokenizer_hash="tok:test", tokens=TOKENS)
    assert first.correct_generation_uid == again.correct_generation_uid
    assert first.wrong_generation_uid == again.wrong_generation_uid


# --------------------------------------------------------------------------- rendering


def test_packets_are_rendered_through_the_episode_packet_contract(pair):
    """Plan §1.8: E2a packets go **through** ``packets.render_packet``, so they carry the same
    cap, the same priority clipping, the same tokenizer stamp and the same denylist as an
    episode packet -- and a receiver packet is comparable with an E1 transmission."""
    for packet in (pair.correct, pair.wrong):
        assert isinstance(packet, PK.TransmittedPacket)
        assert packet.counter_tokenizer_hash == "tok:test"
        assert packet.content is P.PacketContent.FULL and packet.filler is False
        assert packet.token_count <= TOKENS.packet_cap
        PK.check_packet_bytes(packet.serialized)
        wrapper = json.loads(packet.serialized)
        assert set(wrapper) <= PK.ALLOWED_KEYS
        assert "confidence" not in wrapper


def test_a_long_rationale_is_clipped_by_the_shared_priority_rule():
    """The final answer survives the cap before the rationale does (spec 6.1 priority)."""
    tight = K.TokenContract(thinking_cap=16, final_cap=32, served_window=4096, reserve=8,
                            packet_cap=12, rationale_cap=32, claims_max=4, claim_cap=16)
    donors = (draw(0, "C", "42"), draw(1, "W", "11"))
    long_rationale = NP.DonorDraw(
        generation_uid="g:long", decision_id="d:long", draw_index=9,
        output=output("42", " ".join(["word"] * 200)), grade_row=grade("g:long", "C"))
    pair = NP.fix_natural_pair(FakeItem(), (long_rationale, donors[1]), study_seed=STUDY_SEED,
                               counter=counter, counter_tokenizer_hash="tok:test", tokens=tight)
    assert json.loads(pair.correct.serialized)["final_answer"] == "42"
    assert pair.correct.any_truncated


# --------------------------------------------------------------------------- the audit


def test_a_blinded_validity_audit_row_is_written_for_each_packet(pair):
    """Spec 9.1: "record audited validity separately and keep the unfiltered panel visible".

    The audit observes interpretability, not correctness, and the slot label is drawn blind, so
    the audit table cannot be read backwards into which packet was the correct donor.
    """
    audits = NP.audit_rows(pair, STUDY_SEED)
    assert [a.audit_slot for a in audits] == list(NP.AUDIT_SLOTS)
    assert {a.packet_hash for a in audits} == {pair.correct.packet_hash, pair.wrong.packet_hash}
    for entry in audits:
        assert entry.parsed is True and entry.has_rationale is True and entry.has_claims is True
        assert entry.explicit_correctness_tag is False
        assert entry.token_count > 0
        assert not set(entry.as_entry()) & {"semantic_correct", "gold", "grade"}


def test_the_audit_flags_an_explicit_correctness_tag_without_rewriting_the_packet():
    """A donor whose own rationale claims correctness is *recorded*, never edited or dropped:
    the panel stays unfiltered and the flag becomes a covariate."""
    tagged = NP.DonorDraw(generation_uid="g:tag", decision_id="d:tag", draw_index=0,
                          output=output("42", "This is correct, I verified it."),
                          grade_row=grade("g:tag", "C"))
    pair = NP.fix_natural_pair(FakeItem(), (tagged, draw(1, "W", "11")), study_seed=STUDY_SEED,
                              counter=counter, counter_tokenizer_hash="tok:test", tokens=TOKENS)
    flags = {a.packet_hash: a.explicit_correctness_tag for a in NP.audit_rows(pair, STUDY_SEED)}
    assert flags[pair.correct.packet_hash] is True
    assert "This is correct, I verified it." in pair.correct.serialized


# --------------------------------------------------------------------------- the register


def test_the_panel_is_sealed_through_a_content_addressed_packets_register(tmp_run_root, pair):
    """D16: the panel is a ``register:packets/<sha>`` dependency, so a RECEIVER cell cannot
    start before the packets it replays were frozen, and a re-run seals the same sha."""
    sha = NP.seal_panel(tmp_run_root, [pair], study_seed=STUDY_SEED, family="MATH",
                        model_hash="0123456789abcdef", budget_hash="budget00")
    assert NP.REGISTER_KIND == deps.REGISTER_PACKETS
    assert deps.verify_register(tmp_run_root, NP.REGISTER_KIND, sha).status == deps.STATUS_MET
    again = NP.seal_panel(tmp_run_root, [pair], study_seed=STUDY_SEED, family="MATH",
                          model_hash="0123456789abcdef", budget_hash="budget00")
    assert again == sha

    payload = deps.read_register(tmp_run_root, NP.REGISTER_KIND, sha)
    entries = payload[deps.REGISTER_ENTRIES_FIELD]
    assert len(entries) == 1 and entries[0]["source_item_id"] == pair.source_item_id
    bodies = [p["serialized"] for p in entries[0]["packets"]]
    assert sorted(bodies) == sorted([pair.correct.serialized, pair.wrong.serialized])
    assert deps.parse_dependency(deps.register_dependency(NP.REGISTER_KIND, sha)).kind == "register"


def test_the_sealed_panel_carries_no_grade_and_no_donor_uid(pair):
    """R7.  The register is gold-free: the grader's verdict on the source item, the canonical
    class and the donor generation uids stay writer-side."""
    entry = NP.panel_entry(pair, NP.audit_rows(pair, STUDY_SEED), study_seed=STUDY_SEED)
    blob = json.dumps(entry)
    assert pair.correct_generation_uid not in blob and pair.wrong_generation_uid not in blob
    assert "semantic_correct" not in blob and "canonical_class" not in blob
    with pytest.raises(LeakageError):
        NP._assert_label_free({**entry, "canonical_class": "cls-C"})


def test_the_packet_order_in_the_panel_is_not_the_correctness_order():
    """The channel this closes: an entry that always listed the correct donor first would let
    ``packets[0]`` be read as the label, and the audit rows -- which carry the packet hash --
    would join straight back to it, so blinding the ``audit_slot`` name alone bought nothing.

    One blind draw now orders both, so slot *i* is packet *i* and, over a set of items, the
    correct packet lands in either position.  The role each packet plays is stated outright
    (``ROLE_KEY``) because the arms are defined over it -- saying so beats hiding it badly.
    """
    positions = []
    for index in range(24):
        item = type(FakeItem())(source_item_id=f"nditem{index:010d}",
                                item_variant_id=f"nditem{index:010d}.v0")
        built = NP.fix_natural_pair(item, (draw(0, "C", "42"), draw(1, "W", "11")),
                                    study_seed=STUDY_SEED, counter=counter,
                                    counter_tokenizer_hash="tok:test", tokens=TOKENS)
        entry = NP.panel_entry(built, NP.audit_rows(built, STUDY_SEED), study_seed=STUDY_SEED)
        roles = [p[NP.ROLE_KEY] for p in entry["packets"]]
        assert sorted(roles) == [NP.ROLE_CORRECT, NP.ROLE_WRONG]
        positions.append(roles.index(NP.ROLE_CORRECT))
        # slot i is packet i: one draw orders both, so the audit cannot be un-blinded by index
        assert [a["packet_hash"] for a in entry["validity_audit"]] == \
               [p["packet_hash"] for p in entry["packets"]]
        assert [a["audit_slot"] for a in entry["validity_audit"]] == list(NP.AUDIT_SLOTS)
    assert set(positions) == {0, 1}, "the correct donor never moved off one index"


def test_an_empty_panel_is_refused(tmp_run_root):
    """An empty register would let a RECEIVER cell start against nothing."""
    with pytest.raises(Exception):
        NP.seal_panel(tmp_run_root, [], study_seed=STUDY_SEED, family="MATH",
                      model_hash="0123456789abcdef", budget_hash="budget00")
