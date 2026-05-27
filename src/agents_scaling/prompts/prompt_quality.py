"""Score prompt quality two ways (both logged per cell), decoupling length from quality.

* ``heuristic_quality`` — deterministic, dependency-light: readability + presence of
  role / strategy / format / examples / constraints + instruction density. Always
  available; good for sanity and for configs without GPU access.
* ``llm_judge_quality`` — a rubric-based score from a strong reference model (the largest
  Qwen or a fixed judge), 0-100 over clarity / specificity / alignment / format / bias.
  This is the primary ``prompt_quality_score`` reported; validated against a small
  human-rated set (report inter-rater agreement, mirroring Kim's annotation κ).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agents_scaling.serving.client import LogprobClient


@dataclass
class PromptQuality:
    heuristic: float            # 0-100 composite of the deterministic features
    features: dict[str, float]  # individual heuristic features (logged)
    llm_judge: float | None = None  # 0-100 rubric score (None if judge unavailable)


def _flesch_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z]+", text)
    n_words = max(1, len(words))
    syllables = sum(max(1, len(re.findall(r"[aeiouyAEIOUY]+", w))) for w in words)
    return 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words)


def heuristic_quality(text: str) -> PromptQuality:
    low = text.lower()
    has_role = bool(re.search(r"\byou are\b", low))
    has_strategy = bool(re.search(r"step|strategy|methodical|first|then|approach", low))
    has_format = bool(re.search(r"format|final answer|on (its|a) own line|respond with", low))
    has_examples = "example" in low
    has_constraints = bool(re.search(r"exactly one|must|do not|avoid|precise", low))
    n_steps = len(re.findall(r"^\s*\d+\.", text, flags=re.MULTILINE))

    flesch = _flesch_reading_ease(text)
    # Normalize Flesch (~0-100, higher=easier) and clip.
    readability = max(0.0, min(100.0, flesch))

    presence = sum([has_role, has_strategy, has_format, has_examples, has_constraints])
    structure = min(1.0, n_steps / 5.0)  # up to 5 explicit steps = full marks

    composite = (
        0.30 * (presence / 5.0) * 100
        + 0.30 * structure * 100
        + 0.20 * readability
        + 0.20 * min(1.0, len(text.split()) / 200.0) * 100  # mild length credit, capped
    )
    features = {
        "has_role": float(has_role),
        "has_strategy": float(has_strategy),
        "has_format": float(has_format),
        "has_examples": float(has_examples),
        "has_constraints": float(has_constraints),
        "n_explicit_steps": float(n_steps),
        "flesch_reading_ease": flesch,
        "word_count": float(len(text.split())),
    }
    return PromptQuality(heuristic=round(composite, 2), features=features)


_JUDGE_SYSTEM = (
    "You are an expert evaluator of LLM system prompts. Score the given system prompt "
    "from 0 to 100 on overall quality, considering clarity, specificity, alignment with "
    "the task of answering hard reasoning questions accurately, format guidance, and "
    "absence of biasing or leading content. Respond ONLY with JSON: "
    '{"score": <0-100>, "rationale": "<one sentence>"}.'
)


def llm_judge_quality(text: str, judge: LogprobClient) -> float | None:
    """Rubric score from a reference model. Returns None if the response is unparseable."""
    res = judge.chat(
        system=_JUDGE_SYSTEM,
        user=f"System prompt to evaluate:\n\n```\n{text}\n```",
        temperature=0.0,
        max_tokens=200,
        capture_logprobs=False,
    )
    m = re.search(r"\{.*\}", res.text, re.DOTALL)
    if not m:
        return None
    try:
        return float(json.loads(m.group(0))["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def score_prompt_quality(text: str, judge: LogprobClient | None = None) -> PromptQuality:
    pq = heuristic_quality(text)
    if judge is not None:
        pq.llm_judge = llm_judge_quality(text, judge)
    return pq
