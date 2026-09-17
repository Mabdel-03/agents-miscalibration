"""selection/keys.py: one normalization for voting and grading (lead decision on 04-R2), and MATH hedges."""
from __future__ import annotations

import pytest

from agent_allocation.contracts import (
    Candidate, EvalKind, GradeStatus, OutputKind, ParseStatus, PrivateEvalSpec,
)
from agent_allocation.grading.text import normalize
from agent_allocation.grading.text.grade import grade_text
from agent_allocation.selection import keys
from tests.unit.selection import factories as f

LETTERS = ("A", "B", "C", "D")
OPTION_FORMS = ["B", "(B)", " B\n", "B.", "**B**", "Answer: B", "The answer is (B)", "b", "(b)", "B)", "(B", "BC",
                "K", "(K)", "\\boxed{B}", "$B$", "Option B", "Ｂ", "", "   "]
BOOLEAN_FORMS = ["true", "True", "True.", "**False**", " FALSE ", "(true)", "`True`", "$False$", "True or False",
                 "yes", "Truee", "not True", ""]


def _candidate(final: str, kind: OutputKind, item: str) -> Candidate:
    return Candidate(candidate_id=f"c-{abs(hash(final))}", node_id=f"c-{abs(hash(final))}", item_id=item,
                     visible_content=f"<final>{final}</final>", final_answer=final, rationale=None,
                     parse_status=ParseStatus.VALID, finish_reason="stop", output_kind=kind)


def _spec(item: str, kind: EvalKind, gold: str, choices: tuple[str, ...] = (), subtask: str | None = None):
    return PrivateEvalSpec(item_id=item, benchmark="bbh_logic4" if kind is EvalKind.TASK_EXACT else "mmlu_pro",
                           source_revision="r", eval_kind=kind, gold_answer=gold, private_record_ref=None,
                           private_hash="h", subtask=subtask, allowed_choices=choices)


def test_keys_are_the_grader_identities():
    assert keys.TEXT_NORMALIZER_VERSION == normalize.NORMALIZER_VERSION
    assert keys.option_key.__doc__ and keys.boolean_key.__doc__
    for form in OPTION_FORMS:
        assert keys.option_key(form, frozenset(LETTERS)) == normalize.option_identity(form, LETTERS), form
    for form in BOOLEAN_FORMS:
        assert keys.boolean_key(form) == normalize.boolean_identity(form), form


@pytest.mark.parametrize("form", OPTION_FORMS)
def test_option_vote_and_grade_agree(form):
    task = f.task(OutputKind.OPTION, item_id="mmlu", choices=LETTERS)
    key = keys.vote_key(_candidate(form, OutputKind.OPTION, "mmlu"), task)
    grade = grade_text(_candidate(form, OutputKind.OPTION, "mmlu"), _spec("mmlu", EvalKind.OPTION_EXACT, "B", LETTERS))
    if key is None:
        assert grade.status is GradeStatus.INVALID_OUTPUT, (form, grade)
    else:
        assert grade.status is GradeStatus.GRADED and grade.semantic_correct is (key.exact == "B"), (form, grade)
    assert (key is not None) == (form.strip() in ("B", "(B)")), form     # B., **B**, Answer: B are not votes


@pytest.mark.parametrize("form", BOOLEAN_FORMS)
def test_boolean_vote_and_grade_agree(form):
    task = f.task(OutputKind.BOOLEAN, item_id="bool", subtask="boolean_expressions")
    key = keys.vote_key(_candidate(form, OutputKind.BOOLEAN, "bool"), task)
    spec = _spec("bool", EvalKind.TASK_EXACT, "True", ("True", "False"), subtask="boolean_expressions")
    grade = grade_text(_candidate(form, OutputKind.BOOLEAN, "bool"), spec)
    if key is None:
        assert grade.status is GradeStatus.INVALID_OUTPUT, (form, grade)
    else:
        assert grade.status is GradeStatus.GRADED and grade.semantic_correct is (key.exact == "True"), (form, grade)


@pytest.mark.parametrize("final, expected", [
    ("\\boxed{2} or \\boxed{3}", None),                    # hedge: two different boxed answers
    ("\\boxed{2}, \\boxed{3}", None),
    ("$\\boxed{2}$ or $\\boxed{-2}$", None),
    ("\\boxed{\\dfrac{1}{2}} and \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),   # the same answer twice is not a hedge
    ("\\boxed{3} and \\boxed{3}", "3"),
    ("\\boxed{\\boxed{3}}", "3"),
    ("**\\boxed{3}**", "3"),
    ("The answer is \\boxed{3}.", "3"),
    ("\\boxed{3", None),                                   # unclosed box
    ("\\boxed{}", None),
    ("\\boxed{2} or \\boxed{}", None),
    ("$2$ or $3$", "$2$or$3$"),                            # no box: kept whole, never keyed as one of them
])
def test_math_boxes_and_hedges(final, expected):
    task = f.task(OutputKind.MATH, item_id="math")
    key = keys.vote_key(_candidate(final, OutputKind.MATH, "math"), task)
    assert (None if key is None else key.exact) == expected


def test_a_hedge_never_represents_a_class_the_grader_marks_it_wrong():
    task = f.task(OutputKind.MATH, item_id="math")
    spec = PrivateEvalSpec(item_id="math", benchmark="math500", source_revision="r", eval_kind=EvalKind.MATH_VERIFY,
                           gold_answer="3", private_record_ref=None, private_hash="h")
    hedge = _candidate("\\boxed{2} or \\boxed{3}", OutputKind.MATH, "math")
    assert keys.vote_key(hedge, task) is None
    assert grade_text(hedge, spec).semantic_correct is not True
    repeated = _candidate("\\boxed{3} and \\boxed{3}", OutputKind.MATH, "math")
    assert keys.vote_key(repeated, task).exact == "3" and grade_text(repeated, spec).semantic_correct is True


def test_math_text_and_exact_key():
    assert keys.math_text("The answer is \\boxed{\\dfrac{1}{2}}.") == "\\dfrac{1}{2}"
    assert keys.math_text("$x^2$") == "x^2" and keys.math_text("$a$ and $b$") == "$a$ and $b$"
    assert keys.boxed_contents("\\boxed{a{b}} x \\boxed {c}") == ["a{b}", "c"]
    assert keys.math_exact_key("\\dfrac {1}{2}") == keys.math_exact_key("\\frac{1}{2}") == "\\frac{1}{2}"
    assert keys.math_exact_key("\\left( 1, 2 \\right)") == "(1,2)"
    assert keys.math_exact_key("\\pi r^2") == "\\pi r^2" and keys.math_exact_key("\\rightarrow") == "\\rightarrow"
    assert keys.math_exact_key("50\\%") == "50\\%" and keys.math_exact_key("5.") == "5"
    # value-preserving only: no digit joining, no Unicode folding, no guessing about percent or tildes
    assert keys.math_exact_key("1 2") == "1 2" != keys.math_exact_key("12")
    assert keys.math_exact_key("x y") == "x y" and keys.math_exact_key("1 + 2") == "1+2"
    assert keys.math_text("x²") == "x²" and keys.math_exact_key("x²") != keys.math_exact_key("x2")
    assert keys.math_exact_key("5~\\text{cm}") == "5~\\text{cm}"


def test_option_letters_accept_bare_or_parenthesized_choices():
    assert keys.option_letters(f.task(OutputKind.OPTION, choices=("(A)", "(B)"))) == frozenset({"A", "B"})
    with pytest.raises(ValueError):
        keys.option_letters(f.task(OutputKind.OPTION, choices=("Paris", "Rome")))
