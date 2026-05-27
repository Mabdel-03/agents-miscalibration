"""Answer extraction + grading for MCQ and numeric questions."""

from agents_scaling.benchmarks.grading import (
    extract_boxed,
    extract_letter,
    grade,
)
from agents_scaling.benchmarks.schema import AnswerType, Question


def _mcq(answer_key="B"):
    return Question(
        qid="q", benchmark="t", prompt_stem="?",
        options=["w", "x", "y", "z"], answer_key=answer_key, answer_type=AnswerType.MCQ,
    )


def test_extract_letter_variants():
    valid = ["A", "B", "C", "D"]
    assert extract_letter("The answer is B.", valid) == "B"
    assert extract_letter("Final answer: (C)", valid) == "C"
    assert extract_letter("D", valid) == "D"
    assert extract_letter("I think option A is best", valid) == "A"
    assert extract_letter("none of these letters here", valid) is None


def test_extract_letter_ignores_out_of_range():
    # 'E' is not a valid option (only A-D) -> should not be returned.
    assert extract_letter("The answer is E", ["A", "B", "C", "D"]) is None


def test_grade_mcq():
    q = _mcq("B")
    assert grade(q, "The answer is B") is True
    assert grade(q, "The answer is A") is False


def test_extract_boxed_balanced():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_boxed("no box here") is None
    # last boxed wins
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_grade_numeric_normalization():
    q = Question(qid="q", benchmark="math", prompt_stem="?", answer_key="42", answer_type=AnswerType.NUMERIC)
    assert grade(q, r"the result is \boxed{42}") is True
    assert grade(q, r"\boxed{ 42 }") is True
    assert grade(q, r"\boxed{43}") is False
