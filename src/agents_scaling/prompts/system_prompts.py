"""Axis 3: the system-prompt complexity ladder (L0..L3).

The config knob is the *ladder index*. Token count and the prompt-quality score are
*measured attributes* recorded per cell so we can decouple 'more tokens' from 'higher
quality'. Prompts live in ``configs/prompts/level{0..3}.txt`` so they are easy to edit
without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Source-checkout default used by analysis, tests, and explicitly legacy runs.  Production
# workers execute a non-editable installation under ``python -I`` and must pass the prompt
# root from their immutable release worktree instead of relying on this layout.
_PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
N_LEVELS = 4


@lru_cache(maxsize=None)
def _get_prompt(level: int, prompt_root: str) -> str:
    if not 0 <= level < N_LEVELS:
        raise ValueError(f"prompt level must be in [0, {N_LEVELS - 1}], got {level}")
    path = Path(prompt_root) / f"level{level}.txt"
    return path.read_text().strip()


def get_prompt(level: int, *, prompt_root: str | Path | None = None) -> str:
    root = _PROMPT_DIR if prompt_root is None else Path(prompt_root)
    return _get_prompt(level, str(root))


def token_count(
    level: int, tokenizer=None, *, prompt_root: str | Path | None = None
) -> int:
    """Token count of the level's prompt (the literal Axis-3 metric).

    If a HF tokenizer is provided, use it (exact for the served model). Otherwise fall
    back to a whitespace word count (good enough for monotonicity assertions / configs).
    """
    text = get_prompt(level, prompt_root=prompt_root)
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())
