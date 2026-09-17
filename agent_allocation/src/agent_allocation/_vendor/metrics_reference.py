"""Exact finite-bank pass@k and binary transition counts (vendored for agent 06; see PROVENANCE.json).

Origins at ``agents_scaling@e4b4037``:

* ``pass_at_k`` -- ``src/agents_scaling/study/metrics_reference.py`` (``_integer``, ``pass_at_k``). Change: returns
  the exact ``Fraction`` instead of converting it to ``float``.
* ``transition_counts`` -- ``src/agents_scaling/ard/reference_metrics.py:57``. Change: raises ``ValueError`` instead
  of ``assert`` for the identity, so it survives ``python -O``.
"""
from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from typing import Sequence


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def pass_at_k(n: int, c: int, k: int) -> Fraction:
    """Unbiased finite-bank estimator ``1 - C(n-c, k) / C(n, k)`` for IID complete attempts on one task."""
    _integer("n", n, 1)
    _integer("c", c)
    _integer("k", k, 1)
    if c > n or k > n:
        raise ValueError("require c <= n and k <= n")
    if n - c < k:
        return Fraction(1)
    denominator = math.comb(n, k)
    return Fraction(denominator - math.comb(n - c, k), denominator)


def transition_counts(before: Sequence[int], after: Sequence[int]) -> dict[str, int]:
    if len(before) != len(after):
        raise ValueError("A transition must match the same agent slots.")
    if any(y not in (0, 1) for y in list(before) + list(after)):
        raise ValueError("Labels must be 0/1; retain invalidity in a separate field.")
    counts = Counter(zip(before, after))
    result = {"CC": counts[(1, 1)], "CW": counts[(1, 0)], "WC": counts[(0, 1)], "WW": counts[(0, 0)]}
    if sum(after) - sum(before) != result["WC"] - result["CW"]:
        raise ValueError("transition identity violated")
    return result
