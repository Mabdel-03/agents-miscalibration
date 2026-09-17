"""Text grading: tri-state statuses, exact option identity, explicit booleans, Math-Verify order, reference round
trips, and evaluator failures that stay unknown."""
from __future__ import annotations

import pytest

from agent_allocation.contracts import (
    EvalKind,
    GradeResult,
    GradeStatus,
    OutputKind,
    ParseStatus,
    PrivateEvalSpec,
    dump_record,
    load_record,
)
from agent_allocation.grading.text import grade as grade_module
from agent_allocation.grading.text import math_equiv
from agent_allocation.grading.text.grade import grade_text
from agent_allocation.grading.text.normalize import boolean_identity, option_identity

from .conftest import make_candidate


def _spec(item_id: str, kind: EvalKind, gold: str | None, *, subtask: str | None = None) -> PrivateEvalSpec:
    benchmark = item_id.split("/")[0]
    return PrivateEvalSpec(item_id=item_id, benchmark=benchmark, source_revision="rev", eval_kind=kind,
                           gold_answer=gold, private_record_ref=None, private_hash="private-hash", subtask=subtask)


MMLU = _spec("mmlu_pro/1", EvalKind.OPTION_EXACT, "C")
BOOL = _spec("bbh_logic4/boolean_expressions/0", EvalKind.TASK_EXACT, "False", subtask="boolean_expressions")
TEMPORAL = _spec("bbh_logic4/temporal_sequences/0", EvalKind.TASK_EXACT, "A", subtask="temporal_sequences")
MATH = _spec("math500/test/algebra/1.json", EvalKind.MATH_VERIFY, "\\frac{1}{2}")


# ---------------------------------------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("final, expected", [
    ("C", "C"), ("(C)", "C"), ("  C\n", "C"), (" (D) ", "D"),
    ("c", None), ("C.", None), ("Answer: C", None), ("(C", None), ("C)", None), ("CD", None), ("K", None),
    ("The answer is (C)", None), ("", None), (None, None), ("**C**", None),
])
def test_option_identity_accepts_only_bare_or_parenthesized_legal_letters(final, expected):
    assert option_identity(final, ("A", "B", "C", "D")) == expected


@pytest.mark.parametrize("final, expected", [
    ("True", "True"), ("false", "False"), ("FALSE.", "False"), ("**True**", "True"), ("(True)", "True"),
    (" `False` ", "False"), ("True, False", None), ("not True", None), ("Yes", None), ("1", None), ("", None),
    ("True or False", None), (None, None),
])
def test_boolean_identity_is_explicit(final, expected):
    assert boolean_identity(final) == expected


# ---------------------------------------------------------------------------------------------------------
# Options and booleans
# ---------------------------------------------------------------------------------------------------------

def test_option_grading_known_right_and_wrong():
    right = grade_text(make_candidate("mmlu_pro/1", "(C)", OutputKind.OPTION), MMLU, allowed_choices="ABCD")
    wrong = grade_text(make_candidate("mmlu_pro/1", "B", OutputKind.OPTION), MMLU, allowed_choices="ABCD")
    assert (right.status, right.semantic_correct, right.operational_success) == (GradeStatus.GRADED, True, True)
    assert (wrong.status, wrong.semantic_correct, wrong.operational_success) == (GradeStatus.GRADED, False, True)
    assert right.private_eval_hash == "private-hash" and right.grader_version.startswith("text-grade-v1")


def test_illegal_option_is_invalid_output_not_a_wrong_answer():
    result = grade_text(make_candidate("mmlu_pro/1", "E", OutputKind.OPTION), MMLU, allowed_choices="ABCD")
    assert (result.status, result.semantic_correct, result.operational_success) == (
        GradeStatus.INVALID_OUTPUT, None, False)
    assert result.detail_code == "not_a_legal_option"
    prose = grade_text(make_candidate("mmlu_pro/1", "The answer is C", OutputKind.OPTION), MMLU,
                       allowed_choices="ABCD")
    assert prose.status is GradeStatus.INVALID_OUTPUT and prose.semantic_correct is None


@pytest.mark.parametrize("status", [s for s in ParseStatus if s is not ParseStatus.VALID])
def test_non_valid_parse_status_is_invalid_output(status):
    candidate = make_candidate("mmlu_pro/1", "C", OutputKind.OPTION, status=status)
    result = grade_text(candidate, MMLU, allowed_choices="ABCD")
    assert (result.status, result.semantic_correct, result.operational_success) == (
        GradeStatus.INVALID_OUTPUT, None, False)
    assert result.detail_code == f"parse_status:{status.value}"


def test_boolean_and_bbh_choice_grading():
    assert grade_text(make_candidate(BOOL.item_id, "false.", OutputKind.BOOLEAN), BOOL).semantic_correct is True
    assert grade_text(make_candidate(BOOL.item_id, "True", OutputKind.BOOLEAN), BOOL).semantic_correct is False
    maybe = grade_text(make_candidate(BOOL.item_id, "Unknown", OutputKind.BOOLEAN), BOOL)
    assert maybe.status is GradeStatus.INVALID_OUTPUT and maybe.semantic_correct is None
    choice = grade_text(make_candidate(TEMPORAL.item_id, "(A)", OutputKind.OPTION), TEMPORAL,
                        allowed_choices=("A", "B", "C", "D"))
    assert choice.status is GradeStatus.GRADED and choice.semantic_correct is True


def test_wiring_mismatches_raise():
    with pytest.raises(ValueError):
        grade_text(make_candidate("mmlu_pro/2", "C", OutputKind.OPTION), MMLU)
    with pytest.raises(ValueError):
        grade_text(make_candidate("mmlu_pro/1", "C", OutputKind.MATH), MMLU)
    code = _spec("mbpp_plus/Mbpp/2", EvalKind.EVALPLUS_FULL, None)
    with pytest.raises(ValueError):
        grade_text(make_candidate("mbpp_plus/Mbpp/2", "def f(): pass", OutputKind.PYTHON), code)


def test_grade_result_round_trips_through_the_codec():
    result = grade_text(make_candidate("mmlu_pro/1", "C", OutputKind.OPTION), MMLU, allowed_choices="ABCD")
    assert load_record(GradeResult, dump_record(result)) == result


def test_every_candidate_gets_exactly_one_grade_and_a_fixed_denominator():
    finals = ["C", "B", None, "(C)", "zzz"]
    statuses = [ParseStatus.VALID, ParseStatus.VALID, ParseStatus.MISSING, ParseStatus.VALID, ParseStatus.VALID]
    grades = [grade_text(make_candidate("mmlu_pro/1", final, OutputKind.OPTION, status=status, node=f"{i:064x}"),
                         MMLU, allowed_choices="ABCD")
              for i, (final, status) in enumerate(zip(finals, statuses))]
    assert len(grades) == len(finals) and len({g.candidate_id for g in grades}) == len(finals)
    delivered_correct = sum(g.semantic_correct is True and g.operational_success is True for g in grades)
    assert delivered_correct / len(grades) == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------------------------------------

def test_math_grading_right_wrong_and_boxed():
    for final, expected in (("0.5", True), ("\\boxed{\\frac12}", True), ("$\\frac{1}{2}$", True), ("1/3", False)):
        result = grade_text(make_candidate(MATH.item_id, final, OutputKind.MATH), MATH)
        assert result.status is GradeStatus.GRADED and result.semantic_correct is expected, final


def test_math_grading_uses_documented_gold_prediction_order():
    interval_gold = _spec("math500/i", EvalKind.MATH_VERIFY, "(1,2)")
    relation_gold = _spec("math500/r", EvalKind.MATH_VERIFY, "1<x<2")
    assert grade_text(make_candidate("math500/i", "1<x<2", OutputKind.MATH), interval_gold).semantic_correct is False
    assert grade_text(make_candidate("math500/r", "(1,2)", OutputKind.MATH), relation_gold).semantic_correct is True


def test_math_evaluator_exception_stays_unknown(monkeypatch):
    def broken(answer):
        raise RuntimeError("sympy exploded")
    monkeypatch.setattr(math_equiv, "parse_answer", broken)
    result = grade_text(make_candidate(MATH.item_id, "0.5", OutputKind.MATH), MATH)
    assert (result.status, result.semantic_correct) == (GradeStatus.EVALUATOR_ERROR, None)
    assert result.detail_code == "evaluator_exception:RuntimeError"


def test_math_evaluator_timeout_stays_unknown(monkeypatch):
    def spin(gold, prediction):
        while True:
            pass
    monkeypatch.setattr(math_equiv, "_verify", spin)
    result = grade_text(make_candidate(MATH.item_id, "0.5", OutputKind.MATH), MATH, timeout_s=0.2)
    assert (result.status, result.semantic_correct, result.detail_code) == (
        GradeStatus.EVALUATOR_ERROR, None, "timeout")


def test_missing_gold_is_evaluator_error():
    result = grade_text(make_candidate("mmlu_pro/1", "C", OutputKind.OPTION), _spec("mmlu_pro/1",
                        EvalKind.OPTION_EXACT, None), allowed_choices="ABCD")
    assert result.status is GradeStatus.EVALUATOR_ERROR and result.semantic_correct is None


# ---------------------------------------------------------------------------------------------------------
# Reference round trips on the pinned inventories
# ---------------------------------------------------------------------------------------------------------

def _round_trip(inventory, forms):
    failures = []
    for item in inventory:
        if not item.eligible:
            continue
        for form in forms:
            final = form(item.private.gold_answer)
            candidate = make_candidate(item.public.item_id, final, item.public.output_kind)
            result = grade_text(candidate, item.private, allowed_choices=item.public.allowed_choices)
            if not (result.status is GradeStatus.GRADED and result.semantic_correct is True):
                failures.append((item.public.item_id, final, result.status, result.detail_code))
    return failures


def test_every_mmlu_gold_grades_correct_against_itself(mmlu_inventory):
    assert _round_trip(mmlu_inventory, (lambda g: g, lambda g: f"({g})")) == []


def test_every_bbh_gold_grades_correct_against_itself(bbh_inventory):
    assert _round_trip(bbh_inventory, (lambda g: g,)) == []


def test_every_math_gold_grades_correct_against_itself(math_inventory):
    assert _round_trip(math_inventory, (lambda g: g, lambda g: f"\\boxed{{{g}}}")) == []


def test_frozen_math_exclusions_match_the_recomputed_audit(math_inventory):
    from agent_allocation.benchmarks import math500
    failing = {item_id for item_id, *_ in _round_trip([i for i in math_inventory], (lambda g: g,))}
    assert failing == set() and set(math500.ENGINEERING_EXCLUSIONS) == failing


def test_grade_module_exports_versions():
    assert "math-verify==0.9.0" in grade_module.grader_version()
