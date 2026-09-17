"""selection/keys.py: vote identities agree exactly with the grader's frozen normalizer (text-normalize-v1)."""
from __future__ import annotations

import pytest

from agent_allocation.selection import keys

normalize = pytest.importorskip("agent_allocation.grading.text.normalize")

OPTION_CASES = ["A", "(B)", " C\n", "(c)", "c", "Answer: C", "**C**", "C.", "(C", "C)", "AB", "", "  ", "K", "(K)",
                "\\boxed{C}", "$C$", "Option C", "(C)", "Ｃ", " ( C ) "]
BOOLEAN_CASES = ["True", "false", " TRUE ", "**False**", "False.", "(true)", "`True`", "$False$", "True or False",
                 "yes", "", "Truee", "'false'!", "not True", "\ttrue\n"]


def test_normalizer_version_is_the_one_mirrored():
    assert normalize.NORMALIZER_VERSION == keys.TEXT_NORMALIZER_PARITY
    assert normalize.BOOLEAN_OUTER_CHARACTERS == keys.BOOLEAN_OUTER_CHARACTERS


@pytest.mark.parametrize("final", OPTION_CASES)
def test_option_key_parity(final):
    letters = ("A", "B", "C", "D")
    assert keys.option_key(final, frozenset(letters)) == normalize.option_identity(final, letters)


@pytest.mark.parametrize("final", BOOLEAN_CASES)
def test_boolean_key_parity(final):
    assert keys.boolean_key(final) == normalize.boolean_identity(final)


def test_math_text_and_exact_key():
    assert keys.math_text("The answer is \\boxed{\\dfrac{1}{2}}.") == "\\dfrac{1}{2}"
    assert keys.math_text("$x^2$") == "x^2" and keys.math_text("$a$ and $b$") == "$a$ and $b$"
    assert keys.math_text("\\boxed{}") is None
    assert keys.math_exact_key("\\dfrac {1}{2}") == keys.math_exact_key("\\frac{1}{2}") == "\\frac{1}{2}"
    assert keys.math_exact_key("\\left( 1, 2 \\right)") == "(1,2)"
    assert keys.math_exact_key("\\pi r^2") == "\\pi r^2" and keys.math_exact_key("\\rightarrow") == "\\rightarrow"
    assert keys.math_exact_key("50\\%") == "50%" and keys.math_exact_key("5.") == "5"


def test_option_letters_accept_bare_or_parenthesized_choices():
    from tests.unit.selection import factories as f
    from agent_allocation.contracts import OutputKind
    assert keys.option_letters(f.task(OutputKind.OPTION, choices=("(A)", "(B)"))) == frozenset({"A", "B"})
    with pytest.raises(ValueError):
        keys.option_letters(f.task(OutputKind.OPTION, choices=("Paris", "Rome")))
