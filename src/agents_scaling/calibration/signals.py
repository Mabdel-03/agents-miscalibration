"""Confidence signals: each maps a model's response(s) to p(chosen answer is correct).

Four signals, computed wherever the data permits, so calibration conclusions are robust
to how 'confidence' is defined:

1. ``option_logprob_conf`` — softmax mass on the chosen option letter (needs MCQ +
   ``score_options``). The cleanest, model-internal signal — the open-weight advantage.
2. ``verbalized_conf`` — the model's stated 'Confidence: X%' parsed to [0,1].
3. ``self_consistency_conf`` — fraction of n samples that agree with the chosen answer.
4. ``semantic_entropy_conf`` — mass of the largest meaning-cluster (see semantic_entropy).
"""

from __future__ import annotations

import re
from collections import Counter

_CONF_RE = re.compile(r"confidence\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)


def verbalized_conf(text: str) -> float | None:
    """Parse 'Confidence: X%' -> X/100 in [0,1]. Returns None if absent/unparseable."""
    matches = _CONF_RE.findall(text)
    if not matches:
        return None
    val = float(matches[-1])  # last stated confidence wins
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def option_logprob_conf(option_probs: dict[str, float], chosen: str) -> float:
    """p(chosen) from the normalized option distribution."""
    return float(option_probs.get(chosen, 0.0))


def self_consistency_conf(sampled_answers: list[str], chosen: str) -> float:
    """Fraction of samples agreeing with ``chosen`` (e.g. majority answer)."""
    answers = [a for a in sampled_answers if a is not None]
    if not answers:
        return 0.0
    return Counter(answers)[chosen] / len(answers)


def majority_answer(sampled_answers: list[str]) -> str | None:
    answers = [a for a in sampled_answers if a is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]
