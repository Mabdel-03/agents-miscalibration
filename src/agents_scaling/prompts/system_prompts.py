"""Axis 3: the system-prompt complexity ladder (L0..L3).

The config knob is the *ladder index*. Token count and the prompt-quality score are
*measured attributes* recorded per cell so we can decouple 'more tokens' from 'higher
quality'. Prompts live in ``configs/prompts/level{0..3}.txt`` so they are easy to edit
without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# configs/prompts/ relative to repo root (.../agents_scaling)
_PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
N_LEVELS = 4


@lru_cache(maxsize=None)
def get_prompt(level: int) -> str:
    if not 0 <= level < N_LEVELS:
        raise ValueError(f"prompt level must be in [0, {N_LEVELS - 1}], got {level}")
    path = _PROMPT_DIR / f"level{level}.txt"
    return path.read_text().strip()


def token_count(level: int, tokenizer=None) -> int:
    """Token count of the level's prompt (the literal Axis-3 metric).

    If a HF tokenizer is provided, use it (exact for the served model). Otherwise fall
    back to a whitespace word count (good enough for monotonicity assertions / configs).
    """
    text = get_prompt(level)
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())
