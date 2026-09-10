"""Small, dependency-free analysis checks for ARD36.

These functions consume correctness labels only in the OFFLINE evaluator.
They must not be imported into runtime controller feature extraction with gold data.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Sequence


def _validate_bank(n: int, c: int, k: int) -> None:
    if any(not isinstance(x, int) for x in (n, c, k)):
        raise TypeError('n, c, and k must be integers.')
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Require n>=1, 0<=c<=n, and 1<=k<=n.')


def choose(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_at_k(n: int, c: int, k: int) -> float:
    _validate_bank(n, c, k)
    return 1.0 - choose(n-c, k) / choose(n, k)


def correct_count_pmf(n: int, c: int, k: int) -> list[float]:
    """Hypergeometric count distribution for random k-subsets of a fixed bank."""
    _validate_bank(n, c, k)
    denom = choose(n, k)
    return [choose(c, j)*choose(n-c, k-j)/denom for j in range(k+1)]


def iid_correct_count_pmf(p: float, k: int) -> list[float]:
    if not 0 <= p <= 1 or k < 1:
        raise ValueError('Require 0<=p<=1 and k>=1.')
    return [choose(k,j)*(p**j)*((1-p)**(k-j)) for j in range(k+1)]


def heterogeneous_correct_count_pmf(probabilities: Sequence[float]) -> list[float]:
    """Poisson-binomial PMF, conditional on independent heterogeneous draws."""
    pmf = [1.0]
    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError('Probabilities must lie in [0,1].')
        new = [0.0]*(len(pmf)+1)
        for j, mass in enumerate(pmf):
            new[j] += mass*(1-p)
            new[j+1] += mass*p
        pmf = new
    return pmf


def transition_counts(before: Sequence[int], after: Sequence[int]) -> dict[str, int]:
    if len(before) != len(after):
        raise ValueError('A transition must match the same agent slots.')
    if any(y not in (0,1) for y in list(before)+list(after)):
        raise ValueError('Labels must be 0/1; retain invalidity in a separate field.')
    counts = Counter(zip(before, after))
    result = {'CC': counts[(1,1)], 'CW': counts[(1,0)],
              'WC': counts[(0,1)], 'WW': counts[(0,0)]}
    assert sum(after)-sum(before) == result['WC']-result['CW']
    return result


def within_item_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError('Score and label lengths differ.')
    if any(not math.isfinite(s) for s in scores):
        raise ValueError('Missing/nonfinite scores require an explicit missingness policy.')
    if any(y not in (0,1) for y in labels):
        raise ValueError('Labels must be binary.')
    positive = [s for s,y in zip(scores,labels) if y==1]
    negative = [s for s,y in zip(scores,labels) if y==0]
    if not positive or not negative:
        return None
    return sum((p>q) + 0.5*(p==q) for p in positive for q in negative)/(len(positive)*len(negative))


def plurality_expected_correctness(classes: Sequence[str | None], labels: Sequence[int]) -> float:
    """Offline expected accuracy of a uniform tie break over winning valid classes.

    None denotes an invalid answer. Gold-inconsistent canonical classes raise an
    error instead of silently merging or splitting candidates by correctness.
    """
    if len(classes) != len(labels):
        raise ValueError('Classes and labels must align.')
    if any(y not in (0,1) for y in labels):
        raise ValueError('Labels must be 0/1.')
    counts = Counter(c for c in classes if c is not None)
    if not counts:
        return 0.0
    by_class: dict[str, set[int]] = defaultdict(set)
    for cls, y in zip(classes, labels):
        if cls is not None:
            by_class[cls].add(y)
        elif y != 0:
            raise ValueError('An invalid delivered answer cannot be labeled correct.')
    if any(len(ys)>1 for ys in by_class.values()):
        raise ValueError('Canonical-class/grade inconsistency: audit grouping or labels.')
    top = max(counts.values())
    winners = [cls for cls, count in counts.items() if count==top]
    return sum(next(iter(by_class[c])) for c in winners)/len(winners)


def request_digest(payload: dict[str, Any]) -> str:
    required = {'checkpoint_revision', 'tokenizer_hash', 'prompt_token_ids',
                'sampler', 'thinking_budget', 'final_budget', 'seed'}
    absent = required.difference(payload)
    if absent:
        raise ValueError(f'Missing cache-key fields: {sorted(absent)}')
    data = json.dumps(payload, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()
