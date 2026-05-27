"""Uniform question schema across all benchmarks.

Every loader normalizes its dataset into ``Question`` so the agents, formatting, and
grading code is benchmark-agnostic. ``answer_type`` distinguishes multiple-choice
(option-letter answer, enables clean option-logprob calibration) from free-form numeric
(MATH, graded by normalized value match).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnswerType(str, Enum):
    MCQ = "mcq"          # options present; answer_key is an option letter
    NUMERIC = "numeric"  # free-form; answer_key is a canonical value string


@dataclass(frozen=True)
class Question:
    qid: str
    benchmark: str
    prompt_stem: str               # the question text (no options appended)
    answer_key: str                # option letter (MCQ) or canonical value (numeric)
    answer_type: AnswerType
    options: list[str] = field(default_factory=list)  # option texts, in order (MCQ only)

    @property
    def option_letters(self) -> list[str]:
        """A, B, C, ... for the options present."""
        return [chr(ord("A") + i) for i in range(len(self.options))]
