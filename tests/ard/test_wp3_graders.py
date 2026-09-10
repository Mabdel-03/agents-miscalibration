"""WP3 grader tests: MATH equivalence, MMLU option ids, E3 trace evaluation.

Plan §10: "MATH grader fixture (fractions, surds, intervals, tuples, ``\\pi``,
``\\text{}``, degrees, timeouts, ambiguity; the lark parse-failure list is a reported gate
metric) and MMLU permutation round-trip"; spec §5.1 (an ambiguous parse or a timeout is
never silently ``incorrect``), §5.4/§11 (a second independent evaluator validates the E3
gold output).
"""

from __future__ import annotations

import time

import pytest

from agents_scaling.ard.contract import RECEIVER_MODULUS, Family
from agents_scaling.ard import graders
from agents_scaling.ard.graders import math_equiv, mmlu, trace

CORRECT = math_equiv.STATUS_CORRECT
INCORRECT = math_equiv.STATUS_INCORRECT
AMBIGUOUS = math_equiv.STATUS_AMBIGUOUS
TIMEOUT = math_equiv.STATUS_TIMEOUT


# --------------------------------------------------------------------------- MATH: equal

EQUIVALENT = [
    pytest.param(r"\frac{1}{2}", "0.5", id="fraction-decimal"),
    pytest.param(r"\dfrac{14}{3}", r"\frac{14}{3}", id="dfrac"),
    pytest.param(r"\tfrac{14}{3}", r"\frac{14}{3}", id="tfrac"),
    pytest.param(r"\frac43", r"\frac{4}{3}", id="braceless-frac"),
    pytest.param(r"2\sqrt{3}", r"\sqrt{12}", id="surd"),
    pytest.param(r"11\sqrt2", r"11\sqrt{2}", id="braceless-surd"),
    pytest.param(r"\frac{3\sqrt{3}}{4}", r"0.75\sqrt{3}", id="surd-over-integer"),
    pytest.param(r"\left( 3, \frac{\pi}{2} \right)", r"(3,\pi/2)", id="ordered-pair-left-right"),
    pytest.param(r"(6,31,-1)", r"\left(6, 31, -1\right)", id="ordered-triple"),
    pytest.param(r"(3,4]", r"\left(3,4\right]", id="half-open-interval"),
    pytest.param(r"-\frac{\pi}{6}", r"-\pi/6", id="pi"),
    pytest.param(r"2\pi", r"\pi \cdot 2", id="pi-product"),
    pytest.param(r"\text{Evelyn}", "Evelyn", id="text-name"),
    pytest.param(r"\text{even}", r"\mbox{even}", id="text-vs-mbox"),
    pytest.param(r"90^\circ", "90", id="degrees"),
    pytest.param(r"145^{\circ}", "145", id="braced-degrees"),
    pytest.param(r"\$5", "5", id="dollars"),
    pytest.param("80\\%", "80", id="percent"),
    pytest.param("1{,}000", "1000", id="braced-thousands"),
    pytest.param("1,000", "1000", id="plain-thousands"),
    pytest.param(".35625", "0.35625", id="leading-dot"),
    pytest.param("x=5", "5", id="leading-equation"),
    pytest.param(r"x \in [-2,7]", "[-2,7]", id="set-membership"),
    pytest.param("1,-2", "-2,1", id="unordered-root-list"),
    pytest.param(r"1 \pm \sqrt{19}", r"1+\sqrt{19}, 1-\sqrt{19}", id="plus-minus"),
    pytest.param(r"(-\infty, 3) \cup (3, \infty)", r"(3,\infty) \cup (-\infty,3)", id="union"),
    pytest.param(r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
                 r"\begin{pmatrix} -\frac13 \\ \frac23 \\ \frac53 \end{pmatrix}", id="pmatrix"),
    pytest.param(r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}",
                 r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}", id="pmatrix-columns"),
    pytest.param("2k+2", "2(k+1)", id="symbolic"),
    pytest.param("6-5i", "6 - 5i", id="complex"),
    pytest.param(r"\boxed{\frac{1}{2}}", "0.5", id="boxed-candidate"),
    pytest.param(r"\frac{270}7\text{ degrees}", r"\frac{270}{7}", id="unit-suffix"),
    pytest.param("52_8", "52_{8}", id="base-numeral"),
]

DIFFERENT = [
    pytest.param("3", "4", id="integers"),
    pytest.param(r"\frac{1}{2}", r"\frac{1}{3}", id="fractions"),
    pytest.param("[0,2)", "[0,2]", id="interval-endpoint-kind"),
    pytest.param("(3,-1)", "(-1,3)", id="tuple-order"),
    pytest.param("(1,2)", "(1,2,3)", id="arity"),
    pytest.param(r"\text{even}", r"\text{odd}", id="text"),
    pytest.param("52_8", "42", id="base-versus-decimal"),
    pytest.param(r"\infty", "3", id="infinity"),
    pytest.param(r"2\sqrt{3}", r"2\sqrt{5}", id="surds"),
    pytest.param("x", "y", id="symbols"),
]


@pytest.mark.parametrize("candidate,reference", EQUIVALENT)
def test_equivalent_answers_grade_correct(candidate, reference):
    verdict = math_equiv.compare(candidate, reference)
    assert verdict.status == CORRECT, (verdict.status, verdict.detail)
    assert math_equiv.compare(reference, candidate).status == CORRECT


@pytest.mark.parametrize("candidate,reference", DIFFERENT)
def test_different_answers_grade_incorrect(candidate, reference):
    verdict = math_equiv.compare(candidate, reference)
    assert verdict.status == INCORRECT, (verdict.status, verdict.detail)


def test_empty_answer_is_incorrect_not_ambiguous():
    assert math_equiv.compare("", "3").status == INCORRECT
    assert math_equiv.compare(None, "3").status == INCORRECT
    assert math_equiv.compare("   ", "3").status == INCORRECT


def test_unparseable_candidate_is_ambiguous_never_silently_incorrect():
    """Spec §5.1: "Ambiguous parses/timeouts remain explicit"."""
    verdict = math_equiv.compare(r"\sigma_{\text{max}} \implies 3", "3")
    assert verdict.status == AMBIGUOUS
    assert not verdict.candidate_parsed
    assert "unparsed" in verdict.detail


def test_identical_unparsed_strings_still_grade_correct():
    weird = r"\bowtie\bowtie"
    assert math_equiv.compare(weird, weird).status == CORRECT


def test_container_notation_difference_is_ambiguous():
    verdict = math_equiv.compare("1,2", "(1,2)")
    assert verdict.status == AMBIGUOUS
    assert "notation" in verdict.detail


def test_timeout_is_reported_as_timeout(monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(3.0)
        raise AssertionError("should have been interrupted")

    monkeypatch.setattr(math_equiv.sympy, "simplify", slow)
    started = time.monotonic()
    verdict = math_equiv.compare("x^2-1", "(x-1)(x+1)", timeout_s=0.25)
    assert verdict.status == TIMEOUT
    assert time.monotonic() - started < 2.0


def test_status_is_always_one_of_the_four_contract_statuses():
    for candidate in ["", "3", r"\frac{1}{2}", r"\text{even}", r"\sigma", "(1,2)", "\\",
                      "the answer is 3", "0.5"]:
        assert math_equiv.compare(candidate, r"\frac{1}{2}").status in math_equiv.STATUSES


# --------------------------------------------------------------------------- MATH: classes


def test_canonical_class_is_gold_independent_and_merges_equivalent_forms():
    assert math_equiv.canonical_class(r"\frac{1}{2}") == math_equiv.canonical_class("0.5")
    assert math_equiv.canonical_class(r"2\sqrt{3}") == math_equiv.canonical_class(r"\sqrt{12}")
    assert math_equiv.canonical_class("2k+2") == math_equiv.canonical_class("2(k+1)")
    assert math_equiv.canonical_class(r"\frac{1}{2}") != math_equiv.canonical_class(r"\frac{1}{3}")
    assert math_equiv.canonical_class("1,-2") == math_equiv.canonical_class("-2,1")
    assert math_equiv.canonical_class(r"\text{even}") != math_equiv.canonical_class(r"\text{odd}")


def test_canonical_class_never_raises_and_degrades_to_the_string_class():
    for answer in ["", None, r"\sigma", "\\", "((((", r"\frac{", "🙂"]:
        result = math_equiv.canonical_class(answer)
        assert result is None or isinstance(result, str)
    assert math_equiv.canonical_class(r"\sigma").startswith("str:")
    assert math_equiv.canonical_class("") is None


def test_canonical_class_of_identical_strings_is_identical():
    assert math_equiv.canonical_class(r"\bowtie") == math_equiv.canonical_class(r"\bowtie")


def test_boxed_extraction_uses_the_repository_extractor():
    assert math_equiv.strip_answer_wrapper(r"so \boxed{42}.") == "42"
    assert math_equiv.strip_answer_wrapper("plain") == "plain"


# --------------------------------------------------------------------------- MMLU


@pytest.mark.parametrize("answer", ["B", "(B)", "B.", "**B**", "Answer: B", "The answer is (B)"])
def test_mmlu_letter_forms(answer):
    assert mmlu.grade(answer, "B").status == CORRECT
    assert mmlu.canonical_class(answer) == "B"


def test_mmlu_uses_the_full_ten_letter_alphabet():
    for position, letter in enumerate("ABCDEFGHIJ"):
        assert mmlu.letter_for_index(position) == letter
        assert mmlu.grade(letter, letter).status == CORRECT
    with pytest.raises(ValueError):
        mmlu.letter_for_index(10)
    with pytest.raises(ValueError):
        mmlu.grade("A", "K")


def test_mmlu_wrong_letter_is_incorrect_and_empty_is_incorrect():
    assert mmlu.grade("C", "B").status == INCORRECT
    assert mmlu.grade("", "B").status == INCORRECT
    assert mmlu.grade(None, "B").status == INCORRECT
    assert mmlu.canonical_class("") is None


def test_mmlu_option_text_resolves_to_its_letter():
    options = [f"option {index}" for index in range(10)]
    assert mmlu.grade("option 3", "D", options=options).status == CORRECT
    assert mmlu.grade("option 3", "E", options=options).status == INCORRECT


def test_mmlu_unresolvable_answer_is_ambiguous():
    verdict = mmlu.grade("it depends on the reactor design", "B", options=["a"] * 10)
    assert verdict.status == AMBIGUOUS
    assert verdict.chosen_letter is None


def test_mmlu_canonical_class_is_correctness_blind():
    assert mmlu.canonical_class("D") == mmlu.canonical_class("(D)") == "D"
    assert mmlu.canonical_class("Z") is None


# --------------------------------------------------------------------------- E3 traces

PROGRAM = """
x = 7
y = 3
x = (3*x + 5) % 97
y = (x + 2*y + 4) % 97
x = (x + y) % 97
"""


def test_trace_evaluator_tracks_sequential_assignment_semantics():
    evaluation = trace.evaluate_program(PROGRAM)
    assert evaluation.modulus == RECEIVER_MODULUS
    assert [step.value for step in evaluation.steps] == [7, 3, 26, 36, 62]
    assert evaluation.final_target == "x"
    assert evaluation.final_value == 62
    assert evaluation.steps[-1].state == (("x", 62), ("y", 36))
    assert all(0 <= step.value < RECEIVER_MODULUS for step in evaluation.steps)


def test_trace_evaluator_is_independent_of_python_evaluation():
    """The evaluator must not delegate to ``eval``/``ast`` (spec §5.4 second evaluator)."""
    import ast as _ast

    tree = _ast.parse(trace.__loader__.get_source(trace.__name__))
    imported = set()
    called = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
            called.add(node.func.id)
    assert not imported & {"ast", "sympy", "numpy", "pandas"}
    assert not called & {"eval", "exec", "compile", "__import__"}


@pytest.mark.parametrize("program", [
    "x = 1/2", "import os", "x = 2**3", "x = foo(3)", "x = z + 1",
    "for i in range(3): x = 1", "x = 1; y = 2", "x = y = 3", "x = 1 % 97 % 5", "",
])
def test_trace_evaluator_rejects_disallowed_programs(program):
    with pytest.raises((trace.TraceProgramError, ValueError)):
        trace.evaluate_program(program)


def test_trace_evaluator_rejects_a_foreign_modulus():
    with pytest.raises(trace.TraceProgramError):
        trace.evaluate_program("x = 5 % 13")


def test_trace_check_finds_the_corrupted_transition():
    evaluation = trace.evaluate_program(PROGRAM)
    claimed = [(step.target, step.value + (1 if step.index == 3 else 0))
               for step in evaluation.steps]
    checks = trace.check_trace(PROGRAM, claimed)
    divergence = trace.first_divergence(checks)
    assert divergence is not None and divergence.index == 3
    assert divergence.claimed == 37 and divergence.reference == 36
    assert trace.first_divergence(trace.check_trace(
        PROGRAM, [(step.target, step.value) for step in evaluation.steps])) is None


def test_trace_check_accepts_a_rendered_trace():
    evaluation = trace.evaluate_program(PROGRAM)
    rendered = "\n".join(f"{step.target} = (...) % 97 = {step.value}" for step in evaluation.steps)
    assert trace.first_divergence(trace.check_trace(PROGRAM, rendered)) is None
    assert trace.parse_claimed_trace(rendered)[0] == ("x", 7)


def test_trace_grading_and_classes():
    assert trace.grade("The final value is 62", 62).status == CORRECT
    assert trace.grade("61", 62).status == INCORRECT
    assert trace.grade("", 62).status == INCORRECT
    assert trace.grade("no idea", 62).status == AMBIGUOUS
    assert trace.canonical_class("62") == f"mod{RECEIVER_MODULUS}:62"
    assert trace.canonical_class("the value is 62") == trace.canonical_class("62")
    assert trace.canonical_class("nothing numeric") is None


# --------------------------------------------------------------------------- facade


def test_grader_hash_covers_the_module_and_the_pinned_tool(monkeypatch):
    before = graders.grader_hash(Family.MATH500)
    assert before == graders.grader_hashes()[Family.MATH500.value]
    monkeypatch.setattr(math_equiv, "PINNED_LARK_VERSION", "0.0.0-not-the-pin")
    assert graders.grader_hash(Family.MATH500) != before
    assert len(before) == 64


def test_grader_hashes_differ_per_family():
    hashes = graders.grader_hashes()
    assert len(set(hashes.values())) == len(hashes)
    with pytest.raises(ValueError):
        graders.grader_hash("hle")


def test_grading_without_a_bound_gold_store_fails_closed():
    graders.unbind_gold()
    with pytest.raises(graders.GoldUnavailable):
        graders.grade(Family.MATH500, "3", "math500-deadbeef")


def test_facade_canonical_class_dispatches_by_family():
    assert graders.canonical_class(Family.MMLUPRO, "(C)") == "C"
    assert graders.canonical_class(Family.MATH500, "0.5") == \
           graders.canonical_class(Family.MATH500, r"\frac12")
    assert graders.canonical_class(graders.TRACE_FAMILY, "62") == f"mod{RECEIVER_MODULUS}:62"
    with pytest.raises(ValueError):
        graders.canonical_class("bcb", "x")


def test_grade_result_row_shape():
    result = graders.GradeResult("i", Family.MATH500.value, CORRECT, "d", "expr:x", "h", 0.1)
    row = result.to_row()
    assert row["is_correct"] is True
    assert row["evaluator_status"] == CORRECT
    assert row["grader_version"] == "h"
    assert set(row) == {"item_id", "family", "is_correct", "evaluator_status",
                        "evaluator_detail", "answer_class", "grader_version", "grading_duration"}
