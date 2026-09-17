"""Finite-bank oracle pass@k for the independent root bank only (MASTER_SPEC §5.1, §12).

``pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)`` as an exact ``Fraction`` (integer combinatorics, no floats).
Edges: ``pass_at_k(n, c, 1) == c/n``; ``pass_at_k(n, c, n) == (c > 0)``; ``c == 0`` gives 0; ``n - c < k`` gives 1.

This estimator assumes exchangeable independent draws. It is applied only to a ``sample`` checkpoint's full root
bank (``bank_pass_at_k``); revision, peer, CP and pooled outputs are correlated and never enter it. Unknown grades
give bounds: ``lo`` treats unknown draws as not correct, ``hi`` as correct (pass@k is monotone in ``c``).

Config keys consumed: none.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from agent_allocation._vendor.metrics_reference import pass_at_k

__all__ = ["BankPassAtK", "bank_pass_at_k", "pass_at_k"]


@dataclass(frozen=True)
class BankPassAtK:
    n: int
    k: int
    c_lo: int
    c_hi: int
    lo: Fraction
    hi: Fraction

    @property
    def known(self) -> Fraction | None:
        return self.lo if self.lo == self.hi else None


def bank_pass_at_k(correct: Sequence[bool | None], k: int) -> BankPassAtK:
    """pass@k of one item's full independent bank; ``None`` entries are unknown grades (never coerced)."""
    if any(value is not None and not isinstance(value, bool) for value in correct):
        raise TypeError("bank correctness must be bool | None")
    n = len(correct)
    c_lo = sum(1 for value in correct if value is True)
    c_hi = c_lo + sum(1 for value in correct if value is None)
    return BankPassAtK(n=n, k=k, c_lo=c_lo, c_hi=c_hi, lo=pass_at_k(n, c_lo, k), hi=pass_at_k(n, c_hi, k))
