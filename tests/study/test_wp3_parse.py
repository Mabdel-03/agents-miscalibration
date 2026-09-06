"""WP3 parser tests: §3.5 candidate contract (fixture T2), sentinel, coordinator/subtask (T7 parse side)."""

from __future__ import annotations

import json

import pytest

from agents_scaling.study import identity
from agents_scaling.study.parse import candidate as pc
from agents_scaling.study.parse import coordinator as co
from agents_scaling.study.types import (
    SENTINEL,
    SENTINEL_JSON,
    Candidate,
    DelegateAction,
    FinalAction,
    SubtaskResult,
)

GOOD = {
    "approach": "brief method",
    "evidence": [{"claim": "checkable statement", "support": "basis", "uncertainty": "low"}],
    "alternatives_considered": [],
    "failure_checks": [],
    "final_answer": "answer or complete code",
    "confidence": 0.5,
}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def variant(**changes) -> str:
    return dumps({**GOOD, **changes})


def without(key: str) -> str:
    body = dict(GOOD)
    del body[key]
    return dumps(body)


# --------------------------------------------------------------------------- T2 strict-parse table

REJECTIONS = [
    ("length_finish", dumps(GOOD), "length", "TRUNCATED"),
    ("empty", "", "stop", "EMPTY"),
    ("whitespace_only", " \n\t ", "stop", "EMPTY"),
    ("empty_fence", "```json\n```", "stop", "EMPTY"),
    ("none_content", None, "stop", "EMPTY"),
    ("prose", "The answer is 4.", "stop", "NOT_JSON"),
    ("double_fence", "```json\n```json\n" + dumps(GOOD) + "\n```\n```", "stop", "NOT_JSON"),
    ("fence_no_newline", "```json" + dumps(GOOD) + "```", "stop", "NOT_JSON"),
    ("unterminated", dumps(GOOD)[:-3], "stop", "NOT_JSON"),
    ("single_quotes", dumps(GOOD).replace('"', "'"), "stop", "NOT_JSON"),
    ("duplicate_final_answer", dumps(GOOD)[:-1] + ',"final_answer":"other"}', "stop", "DUPLICATE_KEY"),
    ("duplicate_nested_key", variant(evidence=[{"claim": "a", "support": "b", "uncertainty": "low"}]).replace(
        '"claim": "a"', '"claim": "a", "claim": "z"'
    ), "stop", "DUPLICATE_KEY"),
    ("nan_confidence", dumps(GOOD).replace("0.5", "NaN"), "stop", "NONFINITE"),
    ("infinity_confidence", dumps(GOOD).replace("0.5", "Infinity"), "stop", "NONFINITE"),
    ("neg_infinity", dumps(GOOD).replace("0.5", "-Infinity"), "stop", "NONFINITE"),
    ("overflow_literal", dumps(GOOD).replace("0.5", "1e999"), "stop", "NONFINITE"),
    ("trailing_text", dumps(GOOD) + "\nI hope this helps!", "stop", "TRAILING_TEXT"),
    ("trailing_object", dumps(GOOD) + dumps(GOOD), "stop", "TRAILING_TEXT"),
    ("trailing_after_fence", "```json\n" + dumps(GOOD) + "\n```\nDone.", "stop", "NOT_JSON"),  # fence not enclosing
    ("extra_key", variant(notes="x"), "stop", "SCHEMA"),
    ("missing_failure_checks", without("failure_checks"), "stop", "SCHEMA"),
    ("missing_evidence", without("evidence"), "stop", "SCHEMA"),
    ("bool_confidence", variant(confidence=True), "stop", "SCHEMA"),
    ("string_confidence", variant(confidence="0.5"), "stop", "SCHEMA"),
    ("null_confidence", variant(confidence=None), "stop", "SCHEMA"),
    ("confidence_1_2", variant(confidence=1.2), "stop", "SCHEMA"),
    ("confidence_negative", variant(confidence=-0.1), "stop", "SCHEMA"),
    ("uncertainty_med", variant(evidence=[{"claim": "a", "support": "b", "uncertainty": "med"}]), "stop", "SCHEMA"),
    ("evidence_extra_key", variant(evidence=[{"claim": "a", "support": "b", "uncertainty": "low", "x": 1}]), "stop", "SCHEMA"),
    ("evidence_missing_support", variant(evidence=[{"claim": "a", "uncertainty": "low"}]), "stop", "SCHEMA"),
    ("evidence_not_object", variant(evidence=["a"]), "stop", "SCHEMA"),
    ("evidence_33", variant(evidence=[{"claim": "a", "support": "b", "uncertainty": "low"}] * 33), "stop", "SCHEMA"),
    ("alternatives_33", variant(alternatives_considered=["a"] * 33), "stop", "SCHEMA"),
    ("checks_non_string", variant(failure_checks=[1]), "stop", "SCHEMA"),
    ("approach_too_long", variant(approach="x" * 16385), "stop", "SCHEMA"),
    ("claim_too_long", variant(evidence=[{"claim": "x" * 8193, "support": "b", "uncertainty": "low"}]), "stop", "SCHEMA"),
    ("list_string_too_long", variant(failure_checks=["x" * 8193]), "stop", "SCHEMA"),
    ("final_answer_too_long", variant(final_answer="x" * 131073), "stop", "SCHEMA"),
    ("final_answer_not_string", variant(final_answer=4), "stop", "SCHEMA"),
    ("array_top_level", "[" + dumps(GOOD) + "]", "stop", "SCHEMA"),
    ("string_top_level", '"just a string"', "stop", "SCHEMA"),
    ("lone_surrogate", '{"approach":"\\ud83d","evidence":[],"alternatives_considered":[],"failure_checks":[],"final_answer":"a","confidence":0.5}', "stop", "SCHEMA"),
]


@pytest.mark.parametrize("name,content,finish,code", REJECTIONS, ids=[r[0] for r in REJECTIONS])
def test_candidate_parser_rejections(name, content, finish, code):
    parsed = pc.parse_candidate(content, finish)
    assert not parsed.valid
    assert parsed.failure_code == code
    assert parsed.candidate is None and parsed.canonical_sha256 is None
    assert parsed.raw_sha256 == identity.sha256_hex(content or "")
    assert parsed.detail


def test_at_least_25_distinct_strict_cases():
    assert len(REJECTIONS) >= 25
    assert {r[3] for r in REJECTIONS} >= {"TRUNCATED", "EMPTY", "NOT_JSON", "DUPLICATE_KEY", "NONFINITE", "TRAILING_TEXT", "SCHEMA"}


ACCEPTS = [
    ("bare", dumps(GOOD)),
    ("leading_newlines", "\n\n" + dumps(GOOD)),  # observed vLLM shape
    ("fence_json", "```json\n" + dumps(GOOD) + "\n```"),
    ("fence_plain", "```\n" + dumps(GOOD) + "\n```"),
    ("fence_whitespace", "  \n```json  \n  " + dumps(GOOD) + "  \n```  \n"),
    ("fence_close_same_line", "```json\n" + dumps(GOOD) + "```"),
    ("pretty", json.dumps(GOOD, indent=2)),
    ("confidence_int_1", variant(confidence=1)),
    ("confidence_int_0", variant(confidence=0)),
    ("unicode", variant(final_answer="漢字 😀 é", approach="naïve")),
    ("escaped_unicode", dumps(GOOD).replace("basis", "\\u00e9\\ud83d\\ude00")),
    ("bounds_exact", variant(approach="x" * 16384, evidence=[{"claim": "y" * 8192, "support": "", "uncertainty": "unknown"}] * 32, alternatives_considered=["a"] * 32, failure_checks=["z" * 8192] * 32, final_answer="f" * 131072)),
]


@pytest.mark.parametrize("name,content", ACCEPTS, ids=[a[0] for a in ACCEPTS])
def test_candidate_parser_accepts(name, content):
    parsed = pc.parse_candidate(content, "stop")
    assert parsed.valid and parsed.failure_code is None
    assert isinstance(parsed.candidate, Candidate)
    assert parsed.canonical_sha256 == identity.sha256_hex(parsed.canonical_json)
    # canonical bytes are RFC 8785: sorted keys, compact, UTF-8 preserved
    assert parsed.canonical_json.startswith('{"alternatives_considered":')
    assert json.loads(parsed.canonical_json) == parsed.candidate.to_dict()


def test_canonical_hash_independent_of_wire_formatting():
    a = pc.parse_candidate(dumps(GOOD), "stop")
    b = pc.parse_candidate("```json\n" + json.dumps(GOOD, indent=4, sort_keys=True) + "\n```", "stop")
    c = pc.parse_candidate(variant(confidence=1), "stop")
    d = pc.parse_candidate(variant(confidence=1.0), "stop")
    assert a.canonical_sha256 == b.canonical_sha256
    assert a.raw_sha256 != b.raw_sha256
    assert c.canonical_sha256 == d.canonical_sha256
    assert '"confidence":1,' in c.canonical_json


def test_canonical_preserves_utf8_and_escapes_controls():
    parsed = pc.parse_candidate(variant(final_answer='漢 "q" \\ \n \x01'), "stop")
    assert '漢 \\"q\\" \\\\ \\n \\u0001' in parsed.canonical_json


@pytest.mark.parametrize(
    "value,expected",
    [(0.5, "0.5"), (1, "1"), (1.0, "1"), (0.0, "0"), (-0.0, "0"), (1e-7, "1e-7"), (0.000001, "0.000001"),
     (0.1, "0.1"), (0.25, "0.25"), (1e21, "1e+21"), (1e20, "100000000000000000000"), (123456789.0, "123456789"),
     (5e-324, "5e-324"), (0.30000000000000004, "0.30000000000000004")],
)
def test_es6_number(value, expected):
    assert pc.es6_number(value) == expected


def test_es6_number_rejects_nonfinite_and_bool():
    with pytest.raises(ValueError):
        pc.es6_number(float("nan"))
    with pytest.raises(TypeError):
        pc.es6_number(True)


def test_jcs_with_numbers_matches_identity_jcs_when_float_free():
    obj = {"b": [1, "x", None, True], "a": {"z": "漢", "y": [2, 3]}, "é": 1, "E": 2}
    assert pc.jcs_with_numbers(obj) == identity.jcs(obj)


def test_ambiguous_channel_only_when_content_has_no_json():
    reasoning = "Let me think... " + dumps(GOOD) + " that should do."
    assert pc.parse_candidate("", "stop", reasoning=reasoning).failure_code == "AMBIGUOUS_CHANNEL"
    assert pc.parse_candidate("Sure, see above.", "stop", reasoning=reasoning).failure_code == "AMBIGUOUS_CHANNEL"
    # a JSON-ish reasoning without the six keys is not ambiguous
    assert pc.parse_candidate("", "stop", reasoning='{"a": 1} {"final_answer": "x"}').failure_code == "EMPTY"
    # content that *has* JSON (even invalid schema / trailing) keeps its own code
    assert pc.parse_candidate(variant(confidence=2), "stop", reasoning=reasoning).failure_code == "SCHEMA"
    assert pc.parse_candidate(dumps(GOOD) + "x", "stop", reasoning=reasoning).failure_code == "TRAILING_TEXT"
    # valid content wins regardless of reasoning
    assert pc.parse_candidate(dumps(GOOD), "stop", reasoning=reasoning).valid
    # length beats everything
    assert pc.parse_candidate("", "length", reasoning=reasoning).failure_code == "TRUNCATED"


def test_parse_candidate_rejects_unknown_finish_reason_and_types():
    with pytest.raises(ValueError):
        pc.parse_candidate(dumps(GOOD), "abort")
    with pytest.raises(TypeError):
        pc.parse_candidate(b"bytes", "stop")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pc.parse_candidate(dumps(GOOD), "stop", reasoning=1)  # type: ignore[arg-type]


def test_strip_one_fence_rule():
    assert pc.strip_one_fence("```json\n{}\n```") == "{}"
    assert pc.strip_one_fence("```python\nprint(1)\n```") == "print(1)"
    assert pc.strip_one_fence("  ```\n x \n```  ") == "x"
    assert pc.strip_one_fence("```json\n```json\n{}\n```\n```") == "```json\n{}\n```"
    assert pc.strip_one_fence("```{}```") == "```{}```"  # not a fence line
    assert pc.strip_one_fence("```json\n{}") == "```json\n{}"  # no closing fence: untouched
    assert pc.strip_one_fence("{}\n```") == "{}\n```"  # no opening fence: untouched
    assert pc.strip_one_fence("plain") == "plain"
    with pytest.raises(TypeError):
        pc.strip_one_fence(None)  # type: ignore[arg-type]


def test_load_strict_json_is_not_repairing():
    assert pc.load_strict_json('{"a": [1, 2.5, "x"]}') == {"a": [1, 2.5, "x"]}
    for text, code in [("", "EMPTY"), ("{", "NOT_JSON"), ('{"a":1,"a":2}', "DUPLICATE_KEY"), ("[NaN]", "NONFINITE"),
                       ("[1e999]", "NONFINITE"), ("{} {}", "TRAILING_TEXT"), ("1 x", "TRAILING_TEXT")]:
        with pytest.raises(pc.StrictJSONError) as info:
            pc.load_strict_json(text)
        assert info.value.code == code


# --------------------------------------------------------------------------- sentinel


def test_sentinel_is_exact_and_never_a_candidate():
    assert SENTINEL_JSON == '{"status":"unavailable","failure_code":"NO_VALID_PARENT"}'
    assert json.loads(SENTINEL_JSON) == SENTINEL
    parsed = pc.parse_candidate(SENTINEL_JSON, "stop")
    assert not parsed.valid and parsed.failure_code == "SCHEMA"
    with pytest.raises(pc.SchemaError):
        pc.validate_candidate_object(SENTINEL)
    # the unavailable packet body is not a candidate either
    assert pc.parse_candidate('{"status":"unavailable"}', "stop").failure_code == "SCHEMA"


def test_parsed_candidate_invariants():
    with pytest.raises(ValueError):
        pc.ParsedCandidate(True, None, None, "x", None)
    with pytest.raises(ValueError):
        pc.ParsedCandidate(False, "BOGUS", None, "x", None)


# --------------------------------------------------------------------------- coordinator (T7 parse side)


def assignment(slot: int, sid: str = "s", handles=("task",)) -> dict:
    return {
        "worker_slot": slot,
        "subtask_id": sid,
        "question": "q",
        "source_handles": list(handles),
        "required_output_type": "text",
        "return_contract": "r",
    }


def delegate(*assignments) -> str:
    return dumps({"action": "delegate", "assignments": list(assignments)})


def test_final_action_parses_and_validates_candidate():
    parsed = co.parse_coordinator_action("```json\n" + dumps({"action": "final", "candidate": GOOD}) + "\n```", 5)
    assert isinstance(parsed, FinalAction)
    assert parsed.candidate.final_answer == GOOD["final_answer"]
    bad = co.parse_coordinator_action(dumps({"action": "final", "candidate": {**GOOD, "confidence": 2}}), 5)
    assert isinstance(bad, co.ActionError) and bad.code == "INVALID" and bad.reason == "SCHEMA"
    extra = co.parse_coordinator_action(dumps({"action": "final", "candidate": GOOD, "assignments": []}), 5)
    assert isinstance(extra, co.ActionError) and extra.reason == "SCHEMA"


def test_delegate_action_happy_path_and_handles():
    parsed = co.parse_coordinator_action(
        delegate(assignment(1, "a", ["task"]), assignment(3, "b", ["task", "res:1"])), 5, allowed_handles={"task", "res:1"}
    )
    assert isinstance(parsed, DelegateAction)
    assert [a.worker_slot for a in parsed.assignments] == [1, 3]
    assert parsed.to_dict()["action"] == "delegate"


@pytest.mark.parametrize(
    "content,N,code",
    [
        (delegate(assignment(1, "a"), assignment(1, "b")), 5, "DUP_SLOT"),
        (delegate(*[assignment(i, f"s{i}") for i in range(1, 6)]), 5, "TOO_MANY"),
        (delegate(assignment(1, "a", ["task", "hidden"])), 5, "HIDDEN_HANDLE"),
        (delegate(assignment(5, "a")), 5, "BAD_SLOT"),
        (delegate(assignment(1, "a"), assignment(2, "a")), 5, "DUP_SUBTASK"),
        (delegate(assignment(1, "a")), 1, "TOO_MANY"),  # N=1: no workers at all (P1-3)
        (dumps({"action": "plan", "steps": []}), 5, "INVALID"),
        (dumps({"action": "delegate", "assignments": []}), 5, "INVALID"),
        (delegate(*[assignment(i, f"s{i}") for i in range(1, 10)]), 12, "INVALID"),  # 9 > schema max 8
        (delegate({**assignment(1), "extra": 1}), 5, "INVALID"),
        (delegate({**assignment(1), "worker_slot": "1"}), 5, "INVALID"),
        (delegate({**assignment(1), "worker_slot": 0}), 5, "INVALID"),  # schema minimum 1
        (delegate({**assignment(1), "subtask_id": "x" * 129}), 5, "INVALID"),
        (delegate({**assignment(1), "required_output_type": "x" * 257}), 5, "INVALID"),
        (delegate({**assignment(1), "source_handles": ["task"] * 33}), 5, "INVALID"),
        ("not json", 5, "INVALID"),
        ("", 5, "INVALID"),
        (delegate(assignment(1)) + " trailing", 5, "INVALID"),
        ("[1,2]", 5, "INVALID"),
    ],
)
def test_coordinator_cross_record_and_schema_errors(content, N, code):
    parsed = co.parse_coordinator_action(content, N, allowed_handles={"task"})
    assert isinstance(parsed, co.ActionError), parsed
    assert parsed.code == code
    assert parsed.raw_sha256 == identity.sha256_hex(content)
    if code == "INVALID":
        assert parsed.reason in pc.FAILURE_CODES


def test_coordinator_invalid_reason_subcodes():
    assert co.parse_coordinator_action("", 5).reason == "EMPTY"
    assert co.parse_coordinator_action("x", 5).reason == "NOT_JSON"
    assert co.parse_coordinator_action('{"action":"final","action":"final"}', 5).reason == "DUPLICATE_KEY"
    assert co.parse_coordinator_action(delegate(assignment(1)) + "{}", 5).reason == "TRAILING_TEXT"
    assert co.parse_coordinator_action(delegate(assignment(1)), 5, finish_reason="length").code == "TRUNCATED"
    with pytest.raises(ValueError):
        co.parse_coordinator_action(delegate(assignment(1)), 0)
    with pytest.raises(ValueError):
        co.parse_coordinator_action(delegate(assignment(1)), 5, finish_reason="abort")


def test_cross_record_order_too_many_before_bad_slot():
    # six assignments at N=5 with an out-of-range slot: TOO_MANY is reported first
    parsed = co.parse_coordinator_action(delegate(*[assignment(i, f"s{i}") for i in range(1, 7)]), 5)
    assert parsed.code == "TOO_MANY"


# --------------------------------------------------------------------------- subtask results


SUB = {
    "subtask_id": "s1",
    "contract": "return the count",
    "status": "complete",
    "result": "42",
    "assumptions": ["input is sorted"],
    "evidence_handles": ["task"],
    "confidence": 0.8,
}


def test_subtask_result_parses_and_is_not_a_candidate():
    parsed = co.parse_subtask_result("```json\n" + dumps(SUB) + "\n```")
    assert isinstance(parsed, SubtaskResult)
    assert parsed.assumptions == ("input is sorted",)
    assert co.parse_subtask_result(dumps({**SUB, "confidence": None})).confidence is None
    assert not isinstance(parsed, Candidate)
    assert pc.parse_candidate(dumps(SUB), "stop").failure_code == "SCHEMA"
    assert not hasattr(parsed, "to_candidate")
    assert co.subtask_result_sha256(parsed) == identity.sha256_hex(pc.jcs_with_numbers(parsed.to_dict()))


@pytest.mark.parametrize(
    "content,code",
    [
        (dumps({**SUB, "status": "done"}), "SCHEMA"),
        (dumps({**SUB, "confidence": 1.5}), "SCHEMA"),
        (dumps({**SUB, "confidence": True}), "SCHEMA"),
        (dumps({**SUB, "result": "x" * 32769}), "SCHEMA"),
        (dumps({**SUB, "assumptions": ["a"] * 33}), "SCHEMA"),
        (dumps({**SUB, "extra": 1}), "SCHEMA"),
        (dumps({k: v for k, v in SUB.items() if k != "contract"}), "SCHEMA"),
        (dumps(GOOD), "SCHEMA"),
        ("", "EMPTY"),
        ("nope", "NOT_JSON"),
        (dumps(SUB) + dumps(SUB), "TRAILING_TEXT"),
        ('{"subtask_id":"a","subtask_id":"b"}', "DUPLICATE_KEY"),
        (dumps(SUB).replace("0.8", "NaN"), "NONFINITE"),
    ],
)
def test_subtask_result_errors(content, code):
    parsed = co.parse_subtask_result(content)
    assert isinstance(parsed, co.SubtaskError) and parsed.code == code


def test_subtask_result_truncated():
    parsed = co.parse_subtask_result(dumps(SUB), finish_reason="length")
    assert isinstance(parsed, co.SubtaskError) and parsed.code == "TRUNCATED"
