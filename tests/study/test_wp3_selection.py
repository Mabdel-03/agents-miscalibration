"""WP3 selection tests: MC/exact/code keys (§5.3, S1, P2-2), VOTE multiplicity + blind ties (T12), JUDGE_BEST."""

from __future__ import annotations

import hashlib
import hmac
import random

import pytest

from agents_scaling.study import metrics_reference
from agents_scaling.study.selection import judge_best as jb
from agents_scaling.study.selection import normalize as nz
from agents_scaling.study.selection import vote as vt
from agents_scaling.study.types import (
    GROUPING_MODES,
    Candidate,
    CandidateRecord,
    Evidence,
    ProtocolError,
    SelectionRecord,
)

# --------------------------------------------------------------------------- MC keys (P2-2)

MC_TABLE = [
    ("B", "B"),
    ("b", "B"),
    ("(B)", "B"),
    ("B.", "B"),
    ("B)", "B"),
    ("B:", "B"),
    (" B \n", "B"),
    ("Answer: B", "B"),
    ("answer: (b).", "B"),
    ("The answer is B", "B"),
    ("The answer is (C).", "C"),
    ("The correct answer is D", "D"),
    ("Final answer: E", "E"),
    ("Option F", "F"),
    ("**G**", "G"),
    ("$J$", "J"),
    ("\\boxed{A}", "A"),
    ('"H"', "H"),
    ("Ｂ", "B"),  # fullwidth letter → NFKC
    # not unambiguous single letters → None (fall back to exact key)
    ("Paris", None),
    ("AB", None),
    ("B or C", None),
    ("B) Paris", None),
    ("K", None),
    ("42", None),
    ("", None),
    ("Answer: none of the above", None),
    ("1", None),
    ("x^2 + 1", None),
    ("The answer is B because A is wrong", None),
]


@pytest.mark.parametrize("text,expected", MC_TABLE, ids=[repr(t[0]) for t in MC_TABLE])
def test_hle_mc_key(text, expected):
    assert nz.hle_mc_key(text) == expected


def test_vote_key_mc_falls_back_to_exact_for_non_letter_answers():
    assert nz.vote_key("multipleChoice", "(B).") == ("B", "mc_letter")
    assert nz.vote_key("multipleChoice", "Paris.") == ("paris", "exact_norm")
    assert nz.vote_key("multipleChoice", "1,000") == ("1000", "exact_norm")
    assert nz.vote_key("multipleChoice", "") == (None, "exact_norm")
    # exactMatch never uses letters
    assert nz.vote_key("exactMatch", "B") == ("b", "exact_norm")
    with pytest.raises(ValueError):
        nz.vote_key("essay", "x")


# --------------------------------------------------------------------------- exact keys

EXACT_TABLE = [
    ("Paris", "paris"),
    ("  Paris  ", "paris"),
    ("PARIS.", "paris"),
    ("Paris..", "paris."),  # only one trailing period
    ("The answer is Paris.", "paris"),
    ("Answer: Paris", "paris"),
    ("answer = paris", "paris"),
    ("hello   world\n again", "hello world again"),
    ('"Paris"', "paris"),
    ("“Paris”", "paris"),
    ("'Paris'", "paris"),
    ("$x^2$", "x^2"),
    ("$$x^2$$", "x^2"),
    ("\\boxed{x^2}", "x^2"),
    ("$\\boxed{x^2}$", "x^2"),
    ("\\boxed{\\boxed{x}}", "\\boxed{x}"),  # only one of each wrapper kind
    ("42", "42"),
    ("+42", "42"),
    ("42.0", "42"),
    ("42.50", "42.5"),
    ("0.1", "0.1"),
    (".5", "0.5"),
    ("-0", "0"),
    ("-3.10", "-3.1"),
    ("1,000", "1000"),
    ("1,000.50", "1000.5"),
    ("$\\boxed{1,000.50}$", "1000.5"),
    ("1e3", "1000"),
    ("1.5E-3", "0.0015"),
    ("1/2", "1/2"),
    ("2/4", "1/2"),
    ("4/2", "2"),
    ("-6/4", "-3/2"),
    ("\\frac{2}{4}", "1/2"),
    ("\\dfrac{1}{3}", "1/3"),
    ("$\\frac{3}{6}$", "1/2"),
    ("3.", "3"),
    ("1,00", "1,00"),  # not a thousands grouping → string
    ("1/0", "1/0"),  # division by zero is not a number → string
    ("12 apples", "12 apples"),
    ("ﬁne", "fine"),  # NFKC ligature
    ("Ⅻ", "xii"),  # NFKC roman numeral then casefold
    ("", ""),
    ("$", "$"),
]


@pytest.mark.parametrize("text,expected", EXACT_TABLE, ids=[repr(t[0]) for t in EXACT_TABLE])
def test_hle_exact_key(text, expected):
    assert nz.hle_exact_key(text) == expected


def test_exact_key_is_idempotent_and_typed():
    for text, _ in EXACT_TABLE:
        key = nz.hle_exact_key(text)
        if key.endswith(".") or "\\boxed" in key or "$" in key:
            continue  # exactly one trailing period / one wrapper of each kind is stripped by design
        assert nz.hle_exact_key(key) == key
    with pytest.raises(TypeError):
        nz.hle_exact_key(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- code keys (S1, E3')

PROGRAM = "def task_func(x):\n    return x + 1\n"


def test_ast_grouping_equal_for_whitespace_and_comment_variants():
    variants = [
        PROGRAM,
        "def task_func(x):\n    # add one\n    return x + 1\n",
        "def task_func(x):\n\n\n    return   x+1   \n",
        "```python\n" + PROGRAM + "```",
        "```\n" + PROGRAM + "\n```",
        "def task_func(x):  return x + 1",
        "def task_func(x):\n    return (x + 1)\n",
    ]
    keys = {nz.code_key(v) for v in variants}
    assert len(keys) == 1
    key, mode = keys.pop()
    assert mode == "ast" and len(key) == 64


def test_ast_grouping_unequal_for_identifier_and_literal_changes():
    base = nz.code_key(PROGRAM)[0]
    assert nz.code_key("def task_func(y):\n    return y + 1\n")[0] != base
    assert nz.code_key("def task_func(x):\n    return x + 2\n")[0] != base
    assert nz.code_key("def task_func(x):\n    return 1 + x\n")[0] != base
    assert nz.code_key('def task_func(x):\n    """doc"""\n    return x + 1\n')[0] != base  # docstrings preserved
    assert nz.code_key("def task_func(x):\n    return x + 1\nprint(1)\n")[0] != base


def test_exact_source_mode_on_syntax_error_and_empty():
    key, mode = nz.code_key("def task_func(x:\n    return x")
    assert mode == "exact_source" and key == nz.identity.sha256_hex("def task_func(x:\n    return x")
    # exact source is byte-sensitive (whitespace variants differ)
    assert nz.code_key("def f(:\n  pass")[0] != nz.code_key("def f(:\n   pass")[0]
    # the fence is stripped *before* hashing the source (E3': same rule as evaluation)
    assert nz.code_key("```python\ndef f(:\n```")[0] == nz.identity.sha256_hex("def f(:")
    assert nz.code_key("")[1] == "empty" and nz.code_key("```python\n```") == (None, "empty")
    assert nz.code_key("x = 'a\x00b'")[1] == "exact_source"  # null byte → ValueError → exact source
    assert nz.vote_key("code", PROGRAM)[1] == "ast"


# --------------------------------------------------------------------------- helpers for pools


def candidate(answer: str) -> Candidate:
    return Candidate("a", (Evidence("c", "s", "low"),), (), (), answer, 0.5)


def rec(cid: str, answer: str | None, slot: int = 0) -> CandidateRecord:
    valid = answer is not None
    return CandidateRecord(cid, f"req-{cid}", slot, "root", valid, None if valid else "SCHEMA", candidate(answer) if valid else None, "h" if valid else "", "r")


STUDY_SEED = bytes(range(32))


# --------------------------------------------------------------------------- VOTE


def test_vote_multiplicity_is_preserved():
    pool = [rec("a", "B"), rec("b", "(B)"), rec("c", "Answer: B."), rec("d", "C"), rec("e", "C"), rec("f", None)]
    sel = vt.vote(pool, "hle:1", STUDY_SEED, "multipleChoice")
    assert isinstance(sel, SelectionRecord) and sel.selector_id == "VOTE"
    assert sel.selected_candidate_id in {"a", "b", "c"}
    assert sel.planned_count == 6 and sel.valid_count == 5
    assert sel.winning_count == 3 and sel.tied_classes == 1
    assert not sel.all_singleton and not sel.no_valid_candidate
    assert sel.grouping_mode == "mc_letter" and sel.grouping_mode_counts == {"mc_letter": 5}
    assert sel.vote_keys == {"a": "B", "b": "B", "c": "B", "d": "C", "e": "C", "f": None}
    assert sel.pool_candidate_ids == ("a", "b", "c", "d", "e", "f")
    assert SelectionRecord.from_dict(sel.to_dict()) == sel


def test_vote_tie_determinism_t12():
    pools = [
        [rec("p", "X"), rec("q", "X"), rec("r", "Y"), rec("s", "Y")],
        [rec("s", "Y"), rec("p", "X"), rec("r", "Y"), rec("q", "X")],
        [rec("r", "Y"), rec("s", "Y"), rec("q", "X"), rec("p", "X")],
    ]
    winners = {vt.vote(pool, "hle:9", STUDY_SEED, "exactMatch").selected_candidate_id for pool in pools}
    assert len(winners) == 1
    sel = vt.vote(pools[0], "hle:9", STUDY_SEED, "exactMatch")
    assert sel.tied_classes == 2 and sel.winning_count == 2
    # a different study seed may pick differently, but is itself deterministic
    other = {vt.vote(pool, "hle:9", b"\x01" * 32, "exactMatch").selected_candidate_id for pool in pools}
    assert len(other) == 1
    # relabelling the answers does not change the choice (blind to labels)
    relabelled = [rec("p", "Q"), rec("q", "Q"), rec("r", "P"), rec("s", "P")]
    assert vt.vote(relabelled, "hle:9", STUDY_SEED, "exactMatch").selected_candidate_id == sel.selected_candidate_id
    # the tie seed is the documented HMAC
    assert vt.vote_tie_seed(STUDY_SEED) == hmac.new(STUDY_SEED, b"VOTE_TIE", hashlib.sha256).digest()


def test_vote_matches_reference_helper_directly():
    pool = [rec("a", "1"), rec("b", "1.0"), rec("c", "2"), rec("d", None)]
    sel = vt.vote(pool, "hle:2", STUDY_SEED, "exactMatch")
    public = [
        metrics_reference.PublicCandidate("a", "1"),
        metrics_reference.PublicCandidate("b", "1"),
        metrics_reference.PublicCandidate("c", "2"),
        metrics_reference.PublicCandidate("d", None, False),
    ]
    ref = metrics_reference.plurality_vote(public, source_id="hle:2", tie_seed=vt.vote_tie_seed(STUDY_SEED))
    assert sel.selected_candidate_id == ref.candidate_id in {"a", "b"}
    assert (sel.winning_count, sel.tied_classes) == (2, 1)


def test_vote_all_singleton_and_no_valid():
    sel = vt.vote([rec("a", "1"), rec("b", "2"), rec("c", "3")], "hle:3", STUDY_SEED, "exactMatch")
    assert sel.all_singleton and sel.winning_count == 1 and sel.tied_classes == 3
    assert sel.selected_candidate_id in {"a", "b", "c"}
    none = vt.vote([rec("a", None), rec("b", None)], "hle:3", STUDY_SEED, "exactMatch")
    assert none.no_valid_candidate and none.selected_candidate_id is None
    assert none.valid_count == 0 and none.planned_count == 2 and not none.all_singleton
    assert none.grouping_mode is None and none.grouping_mode_counts == {}
    single = vt.vote([rec("a", "1"), rec("b", None)], "hle:3", STUDY_SEED, "exactMatch")
    assert single.selected_candidate_id == "a" and single.all_singleton


def test_vote_unkeyed_empty_answers_cast_no_vote():
    pool = [rec("a", ""), rec("b", ""), rec("c", "x")]
    sel = vt.vote(pool, "hle:4", STUDY_SEED, "exactMatch")
    assert sel.selected_candidate_id == "c" and sel.valid_count == 1
    assert sel.vote_keys == {"a": None, "b": None, "c": "x"}
    assert sel.grouping_mode_counts == {"exact_norm": 3}
    only_empty = vt.vote([rec("a", "")], "hle:4", STUDY_SEED, "exactMatch")
    assert only_empty.no_valid_candidate


def test_vote_code_pool_reports_modes():
    pool = [rec("a", PROGRAM), rec("b", "```python\n" + PROGRAM + "```"), rec("c", "def f(:"), rec("d", "def g(:")]
    sel = vt.vote(pool, "bcb:1", STUDY_SEED, "code")
    assert sel.selected_candidate_id in {"a", "b"} and sel.winning_count == 2
    assert sel.grouping_mode == "ast" and sel.grouping_mode_counts == {"ast": 2, "exact_source": 2}
    assert all(m in GROUPING_MODES for m in sel.grouping_mode_counts)


def test_vote_rejects_bad_pools_and_stale_keys():
    with pytest.raises(ValueError):
        vt.vote([], "hle:1", STUDY_SEED, "exactMatch")
    with pytest.raises(ValueError):
        vt.vote([rec("a", "1"), rec("a", "2")], "hle:1", STUDY_SEED, "exactMatch")
    with pytest.raises(ValueError):
        vt.vote([rec("a", "1")], "", STUDY_SEED, "exactMatch")
    with pytest.raises(ValueError):
        vt.vote([rec("a", "1")], "hle:1", b"", "exactMatch")
    import dataclasses

    stale = dataclasses.replace(rec("a", "1"), vote_key="zzz", grouping_mode="exact_norm")
    with pytest.raises(ProtocolError):
        vt.vote([stale], "hle:1", STUDY_SEED, "exactMatch")
    consistent = dataclasses.replace(rec("a", "1"), vote_key="1", grouping_mode="exact_norm")
    assert vt.vote([consistent], "hle:1", STUDY_SEED, "exactMatch").selected_candidate_id == "a"
    invalid_with_key = dataclasses.replace(rec("a", None), vote_key="1")
    with pytest.raises(ProtocolError):
        vt.keyed_record(invalid_with_key, "exactMatch")


def test_prefix_vote_over_bank():
    bank = [rec(f"d{i}", ans, i) for i, ans in enumerate(["A", "B", "B", "A", "A", "C", "C", "C", "C", "C"])]
    k5 = vt.prefix_vote(bank, 5, "hle:7", STUDY_SEED, "multipleChoice")
    assert k5.pool_kind == "bank_prefix" and k5.prefix_k == 5
    assert k5.pool_candidate_ids == ("d0", "d1", "d2", "d3", "d4")
    assert k5.vote_keys[k5.selected_candidate_id] == "A"
    k10 = vt.prefix_vote(bank, 10, "hle:7", STUDY_SEED, "multipleChoice")
    assert k10.vote_keys[k10.selected_candidate_id] == "C"
    assert vt.prefix_vote(bank, 1, "hle:7", STUDY_SEED, "multipleChoice").selected_candidate_id == "d0"
    with pytest.raises(ValueError):
        vt.prefix_vote(bank, 11, "hle:7", STUDY_SEED, "multipleChoice")
    with pytest.raises(ValueError):
        vt.prefix_vote(bank, 0, "hle:7", STUDY_SEED, "multipleChoice")


def test_vote_random_pools_agree_with_manual_count():
    rng = random.Random(1)
    for _ in range(30):
        pool = [rec(f"c{i}", rng.choice(["A", "B", "C", None]), i) for i in range(rng.randint(1, 12))]
        sel = vt.vote(pool, "hle:r", STUDY_SEED, "multipleChoice")
        counts: dict[str, int] = {}
        for r in pool:
            if r.valid:
                counts[r.candidate.final_answer] = counts.get(r.candidate.final_answer, 0) + 1
        if not counts:
            assert sel.no_valid_candidate
            continue
        top = max(counts.values())
        assert sel.winning_count == top
        assert sel.tied_classes == sum(1 for v in counts.values() if v == top)
        assert counts[next(r for r in pool if r.candidate_id == sel.selected_candidate_id).candidate.final_answer] == top


# --------------------------------------------------------------------------- JUDGE_BEST


def judge_json(score, **overrides) -> str:
    import json

    body = {"quality_score": score, "requirement_coverage": "ok", "reasoning_support": "ok", "unresolved_risks": "none"}
    body.update(overrides)
    return json.dumps(body)


def test_parse_judge_best_accepts_and_rejects():
    assert jb.parse_judge_best(judge_json(0.75)) == (0.75, {"quality_score": 0.75, "requirement_coverage": "ok", "reasoning_support": "ok", "unresolved_risks": "none"})
    assert jb.parse_judge_best("```json\n" + judge_json(1) + "\n```")[0] == 1.0
    assert jb.parse_judge_best("\n\n" + judge_json(0))[0] == 0.0
    for content, code in [
        (judge_json(1.2), "SCHEMA"),
        (judge_json(-0.1), "SCHEMA"),
        (judge_json(True), "SCHEMA"),
        (judge_json("0.5"), "SCHEMA"),
        (judge_json(0.5, extra=1), "SCHEMA"),
        (judge_json(0.5, requirement_coverage=3), "SCHEMA"),
        ('{"quality_score":0.5,"requirement_coverage":"a","reasoning_support":"b"}', "SCHEMA"),
        (judge_json(0.5).replace("0.5", "NaN"), "NONFINITE"),
        (judge_json(0.5) + " ok", "TRAILING_TEXT"),
        ('{"quality_score":0.5,"quality_score":0.6,"requirement_coverage":"a","reasoning_support":"b","unresolved_risks":"c"}', "DUPLICATE_KEY"),
        ("I rate this 0.5", "NOT_JSON"),
        ("", "EMPTY"),
        (None, "EMPTY"),
        ("[0.5]", "SCHEMA"),
    ]:
        score, fields = jb.parse_judge_best(content)
        assert score is None and fields["failure_code"] == code, (content, fields)
    assert jb.parse_judge_best(judge_json(0.5), finish_reason="length")[1]["failure_code"] == "TRUNCATED"
    with pytest.raises(ValueError):
        jb.parse_judge_best(judge_json(0.5), finish_reason="abort")


def test_judge_best_select_max_score_with_frozen_low_and_flags():
    pool = [rec("a", "1"), rec("b", "2"), rec("c", "3"), rec("d", None)]
    sel = jb.judge_best_select({"a": 0.2, "b": None, "c": 0.9}, pool, "hle:5", STUDY_SEED, "exactMatch")
    assert sel.selector_id == "JUDGE_BEST" and sel.selected_candidate_id == "c"
    assert sel.scores == {"a": 0.2, "b": 0.0, "c": 0.9}
    assert sel.score_failures == {"a": False, "b": True, "c": False}
    assert sel.planned_count == 4 and sel.valid_count == 3 and not sel.no_valid_candidate
    assert sel.winning_count is None and sel.tied_classes is None and not sel.all_singleton
    assert sel.vote_keys == {"a": "1", "b": "2", "c": "3", "d": None}
    assert SelectionRecord.from_dict(sel.to_dict()) == sel
    # all judge records invalid: still a selection among the frozen-low candidates, all flagged
    low = jb.judge_best_select({"a": None, "b": None, "c": None}, pool, "hle:5", STUDY_SEED, "exactMatch")
    assert low.selected_candidate_id in {"a", "b", "c"} and all(low.score_failures.values())
    # all candidates invalid: failure
    none = jb.judge_best_select({}, [rec("x", None)], "hle:5", STUDY_SEED, "exactMatch")
    assert none.no_valid_candidate and none.selected_candidate_id is None
    # unkeyed (empty answer) candidates are still scorable
    empty = jb.judge_best_select({"e": 0.4, "f": 0.3}, [rec("e", ""), rec("f", "x")], "hle:5", STUDY_SEED, "exactMatch")
    assert empty.selected_candidate_id == "e"


def test_judge_best_ties_are_blind_and_deterministic():
    pools = [[rec("a", "1"), rec("b", "2"), rec("c", "3")], [rec("c", "3"), rec("a", "1"), rec("b", "2")]]
    scores = {"a": 0.7, "b": 0.7, "c": 0.7}
    winners = {jb.judge_best_select(scores, pool, "hle:6", STUDY_SEED, "exactMatch").selected_candidate_id for pool in pools}
    assert len(winners) == 1
    ref = metrics_reference.judge_best(
        [metrics_reference.PublicCandidate(c, c) for c in ("a", "b", "c")],
        scores,
        source_id="hle:6",
        tie_seed=jb.judge_best_tie_seed(STUDY_SEED),
    )
    assert winners == {ref.candidate_id}
    assert jb.judge_best_tie_seed(STUDY_SEED) != vt.vote_tie_seed(STUDY_SEED)


def test_judge_best_select_fails_closed_on_score_coverage():
    pool = [rec("a", "1"), rec("b", "2"), rec("c", None)]
    with pytest.raises(ProtocolError):
        jb.judge_best_select({"a": 0.5}, pool, "hle:8", STUDY_SEED, "exactMatch")  # missing b
    with pytest.raises(ProtocolError):
        jb.judge_best_select({"a": 0.5, "b": 0.5, "c": 0.5}, pool, "hle:8", STUDY_SEED, "exactMatch")  # invalid scored
    with pytest.raises(ProtocolError):
        jb.judge_best_select({"a": "0.5", "b": 0.5}, pool, "hle:8", STUDY_SEED, "exactMatch")
    with pytest.raises(ValueError):
        jb.judge_best_select({"a": 1.5, "b": 0.5}, pool, "hle:8", STUDY_SEED, "exactMatch")
    with pytest.raises(ValueError):
        jb.judge_best_select({}, [], "hle:8", STUDY_SEED, "exactMatch")
