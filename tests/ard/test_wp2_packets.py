"""ARD36 WP2: packet bytes and pseudonyms (spec §6.4, §7; plan §5.2).

"Peer confidence numbers are stripped in the base protocols" (§6.4); "For every peer packet
use stable pseudonymous agent IDs" (§7); contract §3: the study ``Packet`` cannot carry an
ARD report because its field priority contains ``confidence`` and its ``sender_slot`` is a
hashed non-negative int.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.packets import (
    ARD_FIELD_PRIORITY,
    UNPARSED_PEER_JSON,
    check_packet_bytes,
    make_counter,
    packet_cap,
    pseudonym_map,
    report_packet,
    unavailable_packet,
    unparsed_peer_packet,
)
from agents_scaling.study.types import PACKET_UNAVAILABLE_JSON, ProtocolError
from tests.ard import wp2_support as W
from tests.study import fake_vllm as F


@pytest.fixture(scope="module")
def count():
    return make_counter(F.FakeTokenizer())


def test_field_priority_has_no_confidence() -> None:
    assert ARD_FIELD_PRIORITY == ("final_answer", "evidence")
    assert "confidence" not in ARD_FIELD_PRIORITY


def test_report_packet_bytes_carry_no_confidence_and_no_true_slot(count) -> None:
    report = W.report("42", evidence="six times seven", confidence=0.93)
    packet = report_packet(report, pseudonym="agent-2", count=count, source_round=0, sender_slot=4)
    body = json.loads(packet.serialized)
    assert set(body) == {"from", "final_answer", "evidence", "truncated"}
    assert body["from"] == "agent-2"
    assert "confidence" not in packet.serialized
    assert "0.93" not in packet.serialized
    # the true slot lives on the object (private log) and never in the bytes
    assert packet.sender_slot == 4
    assert '"sender_slot"' not in packet.serialized


def test_sanitizer_refuses_a_confidence_key() -> None:
    check_packet_bytes('{"from":"agent-0","final_answer":"42"}')
    with pytest.raises(ProtocolError):
        check_packet_bytes('{"from":"agent-0","confidence":0.9}')
    with pytest.raises(ProtocolError):
        check_packet_bytes('{"sender_slot":3}')


def test_confidence_wording_inside_evidence_survives(count) -> None:
    """Spec §6.4: do not erase confidence *wording* embedded in the evidence field."""
    report = W.report("42", evidence="I am about 90 percent sure", confidence=0.9)
    packet = report_packet(report, pseudonym="agent-0", count=count)
    assert "90 percent sure" in packet.serialized


def test_a_whole_report_fits_without_clipping(count) -> None:
    report = W.report("42", evidence=" ".join(["reason"] * K.EVIDENCE_MAX_TOKENS), confidence=0.5)
    packet = report_packet(report, pseudonym="agent-1", count=count)
    assert packet.clipped is False
    assert packet.recipient_tokens <= packet_cap()


def test_the_clipper_is_the_safety_net_and_keeps_the_answer_first(count) -> None:
    report = W.report("42", evidence=" ".join(["overflow"] * 4000), confidence=0.5)
    packet = report_packet(report, pseudonym="agent-1", count=count)
    assert packet.recipient_tokens <= packet_cap()
    assert packet.fields["final_answer"] == "42"      # priority order held
    assert packet.truncated["evidence"] is True


def test_typed_entries_are_deterministic_bytes(count) -> None:
    a = unparsed_peer_packet(count=count, source_round=1, collapsed=3)
    b = unparsed_peer_packet(count=count, source_round=1, collapsed=1)
    assert a.serialized == b.serialized == UNPARSED_PEER_JSON
    assert "3" not in a.serialized and "1" not in a.serialized  # multiplicity is hidden
    assert a.pseudonym is None and a.sender_slot is None
    assert unavailable_packet(count=count).serialized == PACKET_UNAVAILABLE_JSON


def test_pseudonyms_are_a_stable_per_episode_permutation() -> None:
    seed = W.STUDY_SEED
    first = pseudonym_map(seed, "item-1|8B|natural|e0")
    assert sorted(first) == [f"agent-{i}" for i in range(K.N_AGENTS)]
    assert pseudonym_map(seed, "item-1|8B|natural|e0") == first          # stable within
    others = {pseudonym_map(seed, f"item-{i}|8B|natural|e0") for i in range(30)}
    assert len(others) > 1                                               # re-drawn across
    # a different study seed re-draws them as well
    assert pseudonym_map(bytes.fromhex("11" * 32), "item-1|8B|natural|e0") != first


def test_packet_id_is_content_addressed(count) -> None:
    report = W.report("42")
    a = report_packet(report, pseudonym="agent-0", count=count)
    b = report_packet(report, pseudonym="agent-0", count=count)
    c = report_packet(report, pseudonym="agent-1", count=count)
    assert a.packet_id == b.packet_id and a.bytes_sha256 == b.bytes_sha256
    assert c.packet_id != a.packet_id
