"""WP3 packet compiler tests: §4.3 bound, priority order, unavailable packet, Table E (fixture T13)."""

from __future__ import annotations

import json
import random

import pytest

from agents_scaling.study import packets as pk
from agents_scaling.study.parse.candidate import candidate_sha256, parse_candidate
from agents_scaling.study.parse.coordinator import subtask_result_sha256
from agents_scaling.study.types import (
    PACKET_FIELD_PRIORITY,
    PACKET_UNAVAILABLE_JSON,
    Candidate,
    CandidateRecord,
    Evidence,
    Packet,
    ProtocolError,
    SubtaskResult,
)

EMOJI = "😀🎉👍🏽❤️🚀"
CJK = "漢字仮名交じり文、日本語のテキストです。"
ESCAPES = 'quote " backslash \\ newline \n tab \t nul \x00 ctrl \x1f slash /'


def make_candidate(
    final_answer: str = "42",
    approach: str = "add",
    evidence=(("2+2=4", "arithmetic", "low"),),
    alternatives=("none",),
    checks=("parity",),
    confidence: float = 0.9,
) -> Candidate:
    return Candidate(
        approach=approach,
        evidence=tuple(Evidence(*e) for e in evidence),
        alternatives_considered=tuple(alternatives),
        failure_checks=tuple(checks),
        final_answer=final_answer,
        confidence=confidence,
    )


def record(candidate: Candidate | None, slot: int = 0, valid: bool = True) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=f"cand{slot}",
        request_id=f"req{slot}",
        slot=slot,
        stage="root",
        valid=valid,
        failure_code=None if valid else "SCHEMA",
        candidate=candidate if valid else None,
        candidate_sha256=candidate_sha256(candidate) if (valid and candidate) else "",
        raw_content_sha256="0" * 64,
    )


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def check_packet_invariants(packet: Packet, count, cap: int, final_cap: int, candidate: Candidate) -> None:
    assert not packet.unavailable
    assert packet.recipient_tokens == count(packet.serialized) <= cap
    wrapper = json.loads(packet.serialized)
    assert list(wrapper) == ["sender_slot", "candidate_sha256", "fields", "truncated", "final_partial"]
    assert list(wrapper["fields"]) == list(PACKET_FIELD_PRIORITY)
    assert wrapper["candidate_sha256"] == candidate_sha256(candidate) == packet.candidate_sha256
    assert wrapper["fields"] == packet.fields and wrapper["truncated"] == packet.truncated
    assert wrapper["final_partial"] == packet.final_partial
    fa = packet.fields["final_answer"]
    assert candidate.final_answer.startswith(fa)
    assert count(fa) <= final_cap
    assert packet.final_partial == (fa != candidate.final_answer) == packet.truncated["final_answer"]
    assert candidate.approach.startswith(packet.fields["approach"])
    for i, item in enumerate(packet.fields["evidence"]):
        src = candidate.evidence[i]
        assert src.claim.startswith(item["claim"]) and src.support.startswith(item["support"])
        assert item["uncertainty"] == src.uncertainty
    for name in ("failure_checks", "alternatives_considered"):
        for i, item in enumerate(packet.fields[name]):
            assert getattr(candidate, name)[i].startswith(item)
    assert packet.fields["confidence"] in (candidate.confidence, None)
    # spans are exact [0,end) code-point prefixes
    for span in packet.spans:
        assert span["start"] == 0 and 0 <= span["end"] <= span["source_len"]
        if span["field"] == "final_answer":
            assert span["end"] == len(fa) and span["source_len"] == len(candidate.final_answer)
    # no lone surrogates anywhere (the excerpt rule cannot split a pair)
    packet.serialized.encode("utf-8")
    # once a field is truncated every later field is empty and truncated
    seen_cut = False
    for name in PACKET_FIELD_PRIORITY:
        if seen_cut:
            assert packet.truncated[name]
            assert packet.fields[name] in ("", [], None)
        elif packet.truncated[name] and name != "final_answer":
            seen_cut = True
        elif name == "final_answer" and packet.truncated[name] and fa == candidate.final_answer[: len(fa)] and count(fa) < final_cap - 1:
            # a packet-level cut of the final display: everything after is empty
            seen_cut = True


# --------------------------------------------------------------------------- small / unavailable


def test_small_candidate_is_sent_whole(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    cand = make_candidate()
    packet = pk.compile_packet(record(cand), stub_tokenizer, sender_slot=3)
    check_packet_invariants(packet, count, 2048, 1024, cand)
    assert packet.sender_slot == 3
    assert not any(packet.truncated.values()) and not packet.final_partial
    assert packet.fields["confidence"] == 0.9
    assert packet.fields["evidence"] == [{"claim": "2+2=4", "support": "arithmetic", "uncertainty": "low"}]
    assert packet.bytes_sha256 == pk.identity.sha256_hex(packet.serialized)
    assert Packet.from_dict(packet.to_dict()).serialized == packet.serialized


def test_unavailable_packet_for_missing_and_invalid(stub_tokenizer):
    for source in (None, record(None, slot=1, valid=False)):
        packet = pk.compile_packet(source, stub_tokenizer, sender_slot=1)
        assert packet.unavailable
        assert packet.serialized == PACKET_UNAVAILABLE_JSON == '{"status":"unavailable"}'
        assert packet.fields == {} and packet.spans == () and packet.truncated == {}
        assert not packet.final_partial
        assert packet.recipient_tokens == 1
        assert packet.candidate_sha256 == ""
    a = pk.compile_packet(None, stub_tokenizer, 1)
    b = pk.compile_packet(None, stub_tokenizer, 2)
    assert a.packet_id != b.packet_id and a.serialized == b.serialized


def test_valid_record_without_candidate_is_unavailable(stub_tokenizer):
    broken = CandidateRecord("c", "r", 0, "root", True, None, None, "abc", "0" * 64)
    packet = pk.compile_packet(broken, stub_tokenizer, 0)
    assert packet.unavailable and packet.candidate_sha256 == "abc"


def test_compile_packet_type_and_slot_checks(stub_tokenizer):
    with pytest.raises(TypeError):
        pk.compile_packet(make_candidate(), stub_tokenizer, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pk.compile_packet(record(make_candidate()), stub_tokenizer, -1)
    with pytest.raises(TypeError):
        pk.make_counter(object())


def test_make_counter_accepts_callables_and_mapping_encoders(stub_tokenizer):
    assert pk.make_counter(lambda t: 7)("x") == 7

    class MappingTok:
        def encode(self, text, add_special_tokens=False):
            return {"input_ids": text.split(), "attention_mask": []}

    assert pk.make_counter(MappingTok())("a b c") == 3
    assert pk.make_counter(stub_tokenizer)("a b c") == 3


# --------------------------------------------------------------------------- priority allocation


def test_priority_allocation_order(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    cand = make_candidate(
        final_answer=words("f", 300),
        evidence=((words("c", 300), words("s", 300), "medium"), (words("c2", 300), words("s2", 300), "high")),
        checks=(words("k", 300), words("k2", 300)),
        alternatives=(words("a", 300),),
        approach=words("p", 300),
    )
    rec = record(cand)
    # generous cap: everything whole
    full = pk.compile_packet(rec, stub_tokenizer, 0, cap=100000, final_cap=100000)
    assert not any(full.truncated.values())
    # final display cap binds first: final_partial, rest intact
    fd = pk.compile_packet(rec, stub_tokenizer, 0, cap=100000, final_cap=100)
    assert fd.final_partial and fd.truncated["final_answer"]
    assert count(fd.fields["final_answer"]) <= 100
    assert not any(v for k, v in fd.truncated.items() if k != "final_answer")
    # shrinking packet caps cut fields in reverse priority (confidence goes first, final answer last)
    order = []
    for cap in range(full.recipient_tokens + 50, 100, -50):
        packet = pk.compile_packet(rec, stub_tokenizer, 0, cap=cap, final_cap=1024)
        check_packet_invariants(packet, count, cap, 1024, cand)
        cut = [name for name in PACKET_FIELD_PRIORITY if packet.truncated[name]]
        order.append(tuple(cut))
    reverse = list(reversed(PACKET_FIELD_PRIORITY))
    for cut in order:
        # the set of truncated fields is always a suffix of the priority list
        assert list(cut) == [name for name in PACKET_FIELD_PRIORITY if name in cut]
        assert cut == tuple(name for name in PACKET_FIELD_PRIORITY if name in set(reverse[: len(cut)]))
    assert order[0] == () or order[0] == ("confidence",)
    assert len(order[-1]) >= 4
    # evidence claim/support order: the first evidence item is admitted before the second
    mid = pk.compile_packet(rec, stub_tokenizer, 0, cap=900, final_cap=1024)
    assert mid.truncated["evidence"]
    ev = mid.fields["evidence"]
    assert ev and ev[0]["claim"] == cand.evidence[0].claim
    if len(ev) == 2:
        assert ev[0]["support"] == cand.evidence[0].support


def test_evidence_units_in_original_order_with_partial_support(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    cand = make_candidate(
        final_answer="x",
        evidence=(("claim one", words("s", 200), "low"), ("claim two", "support two", "low")),
        checks=(),
        alternatives=(),
        approach="",
    )
    packet = pk.compile_packet(record(cand), stub_tokenizer, 0, cap=120, final_cap=1024)
    check_packet_invariants(packet, count, 120, 1024, cand)
    ev = packet.fields["evidence"]
    assert ev[0]["claim"] == "claim one"
    assert 0 < len(ev[0]["support"]) < len(cand.evidence[0].support)
    assert len(ev) == 1
    assert packet.truncated["evidence"] and packet.truncated["failure_checks"]
    spans = [s for s in packet.spans if s["field"] == "evidence"]
    assert [(s["index"], s["part"]) for s in spans] == [(0, "claim"), (0, "support")]


def test_list_items_prefix_and_omission(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    cand = make_candidate(final_answer="x", evidence=(), checks=tuple(words(f"k{i}_", 40) for i in range(5)), alternatives=("alt",), approach="")
    packet = pk.compile_packet(record(cand), stub_tokenizer, 0, cap=110, final_cap=1024)
    check_packet_invariants(packet, count, 110, 1024, cand)
    items = packet.fields["failure_checks"]
    assert 1 <= len(items) < 5
    for i, item in enumerate(items[:-1]):
        assert item == cand.failure_checks[i]
    assert packet.fields["alternatives_considered"] == [] and packet.truncated["alternatives_considered"]


def test_skeleton_too_large_is_a_protocol_error():
    # a per-character counter: the empty wrapper alone is far more than 10 "tokens"
    with pytest.raises(ProtocolError):
        pk.compile_packet(record(make_candidate()), len, 0, cap=10)
    with pytest.raises(ValueError):
        pk.compile_packet(record(make_candidate()), len, 0, cap=0)


# --------------------------------------------------------------------------- adversarial bound (stub)


def adversarial_candidates(rng: random.Random, n: int) -> list[Candidate]:
    alphabet = [EMOJI, CJK, ESCAPES, "plain ascii words ", "\\u00e9 literal backslash-u ", '"""', "\\\\\\\\", "\n\n", "🇯🇵🇺🇸", "é"]
    out = []
    for i in range(n):
        def text(k: int) -> str:
            return "".join(rng.choice(alphabet) + (" " if rng.random() < 0.5 else "") for _ in range(k))
        # chunk sizes keep every field inside the §3.5 bounds (longest alphabet entry ≈ 63 chars)
        out.append(
            make_candidate(
                final_answer=text(rng.randint(1, 1500)),
                approach=text(rng.randint(0, 250)),
                evidence=tuple((text(rng.randint(0, 120)), text(rng.randint(0, 120)), rng.choice(["low", "medium", "high", "unknown"])) for _ in range(rng.randint(0, 12))),
                alternatives=tuple(text(rng.randint(0, 120)) for _ in range(rng.randint(0, 6))),
                checks=tuple(text(rng.randint(0, 120)) for _ in range(rng.randint(0, 6))),
                confidence=rng.choice([0.0, 1.0, 0.5, 0.123456789, 1e-7]),
            )
        )
    return out


def test_packet_bound_adversarial_stub(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    rng = random.Random(4)
    for cand in adversarial_candidates(rng, 40):
        for cap, final_cap in ((2048, 1024), (512, 256), (64, 16)):
            packet = pk.compile_packet(record(cand), stub_tokenizer, 1, cap=cap, final_cap=final_cap)
            check_packet_invariants(packet, count, cap, final_cap, cand)
            # round trip through the wire bytes is exact
            assert json.loads(packet.serialized)["fields"] == packet.fields


def test_packet_is_deterministic(stub_tokenizer):
    cand = adversarial_candidates(random.Random(9), 1)[0]
    a = pk.compile_packet(record(cand), stub_tokenizer, 2, cap=300)
    b = pk.compile_packet(record(cand), stub_tokenizer, 2, cap=300)
    assert a == b


def test_char_level_counter_never_splits_surrogates_or_exceeds():
    # a counter that charges per UTF-16 code unit (worst case for astral characters)
    def utf16_units(text: str) -> int:
        return len(text.encode("utf-16-le")) // 2

    cand = make_candidate(final_answer=EMOJI * 400, approach=CJK * 50, evidence=(), checks=(), alternatives=())
    packet = pk.compile_packet(record(cand), utf16_units, 0, cap=700, final_cap=300)
    assert utf16_units(packet.serialized) <= 700
    assert utf16_units(packet.fields["final_answer"]) <= 300
    packet.serialized.encode("utf-8")
    assert cand.final_answer.startswith(packet.fields["final_answer"])


def test_safe_prefix_end_guards_surrogate_pairs():
    text = "a\ud83d\ude00b"  # explicit surrogate code points, only reachable synthetically
    assert len(text) == 4
    assert pk.safe_prefix_end(text, 2) == 1  # would split the pair → back off
    assert pk.safe_prefix_end(text, 3) == 3
    assert pk.safe_prefix_end("a😀b", 2) == 2  # a real astral code point is atomic
    assert pk.safe_prefix_end("abc", 10) == 3 and pk.safe_prefix_end("abc", -1) == 0


def test_largest_prefix(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    text = words("w", 50)
    n = pk.largest_prefix(text, 10, count)
    assert count(text[:n]) <= 10 and count(text[: n + 2]) > 10
    assert pk.largest_prefix(text, 1000, count) == len(text)
    assert pk.largest_prefix(text, 0, count) == 0


# --------------------------------------------------------------------------- Table E: subtask results


def test_subtask_result_packet_clips_result_first(stub_tokenizer):
    count = pk.make_counter(stub_tokenizer)
    result = SubtaskResult("s1", words("contract", 50), "partial", words("r", 500), (words("a", 30),), ("task",), 0.4)
    packet = pk.render_subtask_result_packet(result, stub_tokenizer, sender_slot=2, cap=200)
    assert packet.recipient_tokens == count(packet.serialized) <= 200
    wrapper = json.loads(packet.serialized)
    assert list(wrapper) == ["sender_slot", "result_sha256", "fields", "truncated", "partial"]
    assert wrapper["result_sha256"] == subtask_result_sha256(result) == packet.candidate_sha256
    assert wrapper["fields"]["subtask_id"] == "s1" and wrapper["fields"]["status"] == "partial"
    assert wrapper["fields"]["confidence"] == 0.4
    assert result.result.startswith(wrapper["fields"]["result"]) and wrapper["fields"]["result"]
    assert wrapper["partial"] is True and packet.final_partial and packet.truncated["result"]
    assert wrapper["fields"]["evidence_handles"] == [] and wrapper["fields"]["assumptions"] == []
    assert wrapper["fields"]["contract"] == ""
    whole = pk.render_subtask_result_packet(result, stub_tokenizer, 2, cap=4096)
    assert not any(whole.truncated.values()) and not whole.final_partial
    assert json.loads(whole.serialized)["fields"]["contract"] == result.contract
    assert not whole.unavailable
    unavailable = pk.render_subtask_result_packet(None, stub_tokenizer, 2)
    assert unavailable.unavailable and unavailable.serialized == PACKET_UNAVAILABLE_JSON
    with pytest.raises(TypeError):
        pk.render_subtask_result_packet(make_candidate(), stub_tokenizer, 2)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- T13 real tokenizer


def test_packet_length_bound_real_tokenizer(real_tokenizer_32b):
    count = pk.make_counter(real_tokenizer_32b)
    rng = random.Random(13)
    code = "def task_func(x):\n    return [i ** 2 for i in range(x)]  # squares\n" * 60
    candidates = adversarial_candidates(rng, 44) + [
        make_candidate(final_answer="```python\n" + code + "```", approach="loop", evidence=(("c", "s", "low"),) * 32),
        make_candidate(final_answer=CJK * 800, approach=EMOJI * 100),
        make_candidate(final_answer=EMOJI * 1500),
        make_candidate(final_answer=ESCAPES * 600),
        make_candidate(final_answer="x", approach="y" * 16384, checks=("z" * 8192,) * 32),
        make_candidate(final_answer="f" * 131072),
    ]
    assert len(candidates) >= 50
    n_final_partial = 0
    for cand in candidates:
        parsed = parse_candidate(json.dumps(cand.to_dict(), ensure_ascii=False), "stop")
        assert parsed.valid, parsed.detail
        packet = pk.compile_packet(record(parsed.candidate), real_tokenizer_32b, 1)
        check_packet_invariants(packet, count, 2048, 1024, parsed.candidate)
        assert len(real_tokenizer_32b.encode(packet.serialized, add_special_tokens=False)) <= 2048
        n_final_partial += packet.final_partial
    assert n_final_partial >= 5
    # Table E subtask result under the real tokenizer
    result = SubtaskResult("s", "c" * 8000, "complete", (CJK + EMOJI + ESCAPES) * 300, ("a" * 8000,) * 4, ("task",), None)
    packet = pk.render_subtask_result_packet(result, real_tokenizer_32b, 1)
    assert count(packet.serialized) <= 4096 and packet.final_partial
