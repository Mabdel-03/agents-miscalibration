"""Render a ``Question`` into the prompt text shown to an agent.

Two rendering helpers:

* ``render_question`` — the user-message body (stem + lettered options if MCQ).
* ``score_prompt`` — a *completion-style* prompt ending in ``"Answer: "`` used by
  ``LogprobClient.score_options`` to read the option-letter distribution directly.

Keeping option lettering here (single place) guarantees the agent prompt, the scoring
prompt, and grading all agree on which letter maps to which option.
"""

from __future__ import annotations

from agents_scaling.benchmarks.schema import AnswerType, Question

_VERBAL_CONF_INSTRUCTION = (
    "\n\nAfter your answer, on a new line write 'Confidence: X%' where X is your "
    "confidence (0-100) that your answer is correct."
)


def render_options(q: Question) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in zip(q.option_letters, q.options))


def render_question(q: Question, ask_verbalized_confidence: bool = True) -> str:
    """The user-message body shown to an agent."""
    if q.answer_type == AnswerType.MCQ:
        body = f"{q.prompt_stem}\n\n{render_options(q)}\n\nRespond with the letter of the best option."
    else:
        body = f"{q.prompt_stem}\n\nGive your final answer inside \\boxed{{}}."
    if ask_verbalized_confidence:
        body += _VERBAL_CONF_INSTRUCTION
    return body


def score_prompt(q: Question) -> str:
    """Completion-style prompt for option-mass scoring; ends in 'Answer: '."""
    assert q.answer_type == AnswerType.MCQ, "score_prompt is MCQ-only"
    return f"{q.prompt_stem}\n\n{render_options(q)}\n\nAnswer: "
