"""Pinned tokenizers, exact chat-template token accounting and the frozen envelopes (WP1).

Spec: §4.3 ("Preflight checks actual token IDs for every checkpoint"; 32,768 rendered input
tokens; 4,096 source-task envelope), §6.4 (caps), §3.6 (tokenizer_revision is part of the
request identity, so tokenizers are loaded from the pinned snapshot only).  Architecture
§1.5 (``inference/tokens.py``); corrections §4 item 1 (preflight) and item 7
(``ContextFailure`` is a used opportunity, never an infrastructure failure).

Every helper takes an explicit tokenizer (the real pinned ``Qwen2Tokenizer`` or the test
stub of ``tests/study/conftest.py``); ``load_tokenizer`` is the only place that touches
the HF cache, always with ``local_files_only=True`` and cached once per process.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from agents_scaling.serving.context import rendered_chat_token_ids
from agents_scaling.study.types import PROMPT_TOKENS_CAP, TASK_TOKENS_CAP, Checkpoint, ContextFailure

#: HF cache on the data filesystem (never ``~/.cache``); the env may override.
DEFAULT_HF_HOME = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"

_CACHE: dict[tuple[str, str], Any] = {}
_LOCK = threading.RLock()


class TokenizerUnavailable(RuntimeError):
    """The pinned tokenizer snapshot is not in the local cache (a configuration failure)."""


def _resolve_checkpoint(checkpoint: Checkpoint | str) -> Checkpoint:
    if isinstance(checkpoint, Checkpoint):
        return checkpoint
    if isinstance(checkpoint, str):
        from agents_scaling.study.config import load_config

        cfg = load_config()
        if checkpoint not in cfg.checkpoints:
            raise KeyError(f"unknown checkpoint size {checkpoint!r}; declared: {sorted(cfg.checkpoints)}")
        return cfg.checkpoints[checkpoint]
    raise TypeError(f"checkpoint must be a Checkpoint or a size key, got {type(checkpoint).__name__}")


def load_tokenizer(checkpoint: Checkpoint | str):
    """Return the pinned tokenizer of ``checkpoint`` (size key or ``Checkpoint``), cached per process.

    Loads ``hf_id @ tokenizer_revision`` with ``local_files_only=True`` (§6.4 pins; no network
    ever).  Raises :class:`TokenizerUnavailable` when the snapshot is not cached instead of
    falling back to another revision.
    """
    ckpt = _resolve_checkpoint(checkpoint)
    key = (ckpt.hf_id, ckpt.tokenizer_revision)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                ckpt.hf_id, revision=ckpt.tokenizer_revision, local_files_only=True
            )
        except Exception as exc:  # OSError / EnvironmentError / import failure
            raise TokenizerUnavailable(
                f"cannot load {ckpt.hf_id}@{ckpt.tokenizer_revision} from the local HF cache: {exc!r}"
            ) from exc
        if not getattr(tokenizer, "chat_template", None):
            raise TokenizerUnavailable(f"{ckpt.hf_id}@{ckpt.tokenizer_revision} has no chat template")
        _CACHE[key] = tokenizer
        return tokenizer


def flagship_tokenizer():
    """The flagship (32B) tokenizer that defines the §4.3 task envelope."""
    from agents_scaling.study.config import load_config

    return load_tokenizer(load_config().flagship_checkpoint)


def count_tokens(text: str, tokenizer=None) -> int:
    """Number of tokens of raw ``text`` without special tokens (the source-task measure, §4.3).

    ``tokenizer`` defaults to the flagship tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError("count_tokens expects str")
    tok = tokenizer if tokenizer is not None else flagship_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return len(ids)


def render_chat_token_ids(
    tokenizer, messages: Sequence[Mapping[str, Any]], enable_thinking: bool
) -> tuple[int, ...]:
    """Exact rendered prompt token ids (``add_generation_prompt=True`` + Qwen ``enable_thinking``).

    Thin wrapper over :func:`agents_scaling.serving.context.rendered_chat_token_ids`, which
    forces ``return_dict=False`` (transformers 5 otherwise returns a mapping).  These ids are
    what the server must echo back as ``prompt_token_ids`` (§4.3, §10.3).
    """
    if not isinstance(enable_thinking, bool):
        raise TypeError("enable_thinking must be a bool")
    for message in messages:
        if set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("messages must be {'role','content'} dicts with str content")
    return rendered_chat_token_ids(tokenizer, list(messages), enable_thinking=enable_thinking)


def render_chat_text(tokenizer, messages: Sequence[Mapping[str, Any]], enable_thinking: bool) -> str:
    """Rendered chat-template text (``tokenize=False``) for byte inspection and anchor mapping."""
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def chat_template_hash(tokenizer) -> str:
    """SHA-256 of the tokenizer's Jinja chat template (stored in ``RequestRecord.chat_template_hash``)."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer has no string chat_template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def envelope_check(
    messages: Sequence[Mapping[str, Any]],
    tokenizer,
    cap: int = PROMPT_TOKENS_CAP,
    *,
    enable_thinking: bool = True,
) -> int:
    """Return the rendered prompt token count or raise :class:`ContextFailure` if it exceeds ``cap``.

    §4.3: a wrapper that cannot fit is an explicit context-failure record for the affected
    opportunity (used, Y=0), never a reason to drop a peer or shorten the task.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    n = len(render_chat_token_ids(tokenizer, messages, enable_thinking))
    if n > cap:
        raise ContextFailure(f"rendered prompt has {n} tokens > cap {cap}")
    return n


def task_envelope_check(task_text: str, tokenizer=None, cap: int = TASK_TOKENS_CAP) -> int:
    """Return the raw task token count or raise :class:`ContextFailure` when it exceeds ``cap`` (§4.3).

    Used as the source-only eligibility predicate (before any outcome access) and by the
    preflight under every compared tokenizer.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    n = count_tokens(task_text, tokenizer)
    if n > cap:
        raise ContextFailure(f"task text has {n} tokens > cap {cap}")
    return n


__all__ = [
    "DEFAULT_HF_HOME",
    "TokenizerUnavailable",
    "chat_template_hash",
    "count_tokens",
    "envelope_check",
    "flagship_tokenizer",
    "load_tokenizer",
    "render_chat_text",
    "render_chat_token_ids",
    "task_envelope_check",
]
