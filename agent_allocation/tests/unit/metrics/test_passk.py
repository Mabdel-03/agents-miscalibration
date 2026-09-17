"""metrics/passk.py: exact finite-bank pass@k and its edges."""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations

import pytest

from agent_allocation.metrics.passk import bank_pass_at_k, pass_at_k


@pytest.mark.parametrize("c", range(0, 26))
def test_pass_at_1_is_the_mean_draw_correctness_and_pass_at_25_is_coverage(c):
    assert pass_at_k(25, c, 1) == Fraction(c, 25)
    assert pass_at_k(25, c, 25) == (1 if c else 0)


def test_formula_matches_subset_enumeration_exactly():
    bank = [True, False, False, True, False, False, False]
    n, c = len(bank), sum(bank)
    for k in range(1, n + 1):
        subsets = list(combinations(range(n), k))
        enumerated = Fraction(sum(any(bank[i] for i in s) for s in subsets), len(subsets))
        assert pass_at_k(n, c, k) == enumerated


def test_edges_and_exactness():
    assert pass_at_k(1, 0, 1) == 0 and pass_at_k(1, 1, 1) == 1
    assert pass_at_k(25, 24, 2) == 1                       # n - c < k
    assert pass_at_k(25, 1, 13) == Fraction(13, 25)
    value = pass_at_k(25, 3, 16)
    assert value == 1 - Fraction(math.comb(22, 16), math.comb(25, 16)) and isinstance(value, Fraction)
    for bad in [(0, 0, 1), (25, 26, 1), (25, 1, 0), (25, 1, 26), (25, True, 1), (25.0, 1, 1)]:
        with pytest.raises(ValueError):
            pass_at_k(*bad)


def test_bank_pass_at_k_keeps_unknown_as_bounds():
    bank = [True, None, False, False]
    result = bank_pass_at_k(bank, 2)
    assert (result.c_lo, result.c_hi) == (1, 2)
    assert result.lo == pass_at_k(4, 1, 2) and result.hi == pass_at_k(4, 2, 2) and result.known is None
    assert bank_pass_at_k([True, False], 1).known == Fraction(1, 2)
    with pytest.raises(TypeError):
        bank_pass_at_k([1, 0], 1)
