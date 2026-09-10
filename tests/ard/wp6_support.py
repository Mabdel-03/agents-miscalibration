"""Shared helpers for the WP6 (neural capture and readout) tests.

CPU only.  The model is ``tests/study_neural/support.tiny_qwen3`` — a two-layer random
Qwen3 with the **real** Qwen3 vocabulary and the pinned offline tokenizer — so anchors are
resolved against real byte spans while the forward pass costs milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents_scaling.ard import contract as K
from tests.study_neural import support

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "study" / "fixtures" / "request_record.example.json"
REVISION_8B = K.REVISIONS[K.CHECKPOINT_SMALL]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def report_text(
    answer: str | None = "42",
    *,
    evidence: str | None = "checked twice",
    confidence: str | None = "0.87",
    prefix: str = "\n\n",
) -> str:
    """A post-``</think>`` report span in the frozen three-tag schema (spec §6.3)."""
    out = prefix
    if answer is not None:
        out += f"{K.FINAL_OPEN}{answer}{K.FINAL_CLOSE}"
    if evidence is not None:
        out += f"{K.EVIDENCE_OPEN}{evidence}{K.EVIDENCE_CLOSE}"
    if confidence is not None:
        out += f"{K.CONFIDENCE_OPEN}{confidence}{K.CONFIDENCE_CLOSE}"
    return out


def completion_ids(tokenizer, content: str, *, think: str | None = "reasoning 一二三", im_end: bool = True) -> list[int]:
    """``<think>…</think>`` + the report span, as vLLM stores the completion ids."""
    return support.completion_from_text(tokenizer, think, content, im_end=im_end)


def prompt_ids(tokenizer, text: str) -> tuple[list[dict[str, str]], list[int]]:
    messages = [{"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=True, return_dict=False)
    return messages, [int(t) for t in ids]


def request_record(
    tokenizer,
    request_id: str,
    *,
    task: str = "What is 6 times 7?",
    content: str | None = None,
    think: str | None = "reasoning 一二三",
    finish_reason: str = "stop",
    revision: str = REVISION_8B,
    im_end: bool = True,
):
    """A committed ``study.types.RequestRecord`` the capture runner accepts as-is."""
    from agents_scaling.study import types as T

    base = json.loads(FIXTURE.read_text())
    messages, ids = prompt_ids(tokenizer, task)
    body = report_text() if content is None else content
    completion = completion_ids(tokenizer, body, think=think, im_end=im_end)
    base.update({
        "request_id": request_id,
        "messages": messages,
        "prompt_token_ids": list(ids),
        "prompt_tokens": len(ids),
        "chat_template_kwargs": {"enable_thinking": True},
        "response": {**base["response"], "token_ids": completion, "completion_tokens": len(completion),
                     "content": body, "reasoning": think, "finish_reason": finish_reason},
        "identity": {**base["identity"], "model_revision": revision},
    })
    base["model"] = {**base["model"], "model_revision": revision}
    record = T.RequestRecord.from_dict({k: v for k, v in base.items() if k != "content_sha256"})
    return record.with_content_sha256()


def tiny_snapshot(tmp_path: Path, tokenizer) -> Path:
    """A two-layer random Qwen3 over the real vocabulary, beside the real tokenizer files."""
    import shutil

    path = Path(tmp_path)
    model = support.tiny_qwen3(seed=0, vocab_size=int(tokenizer.vocab_size) + len(tokenizer.added_tokens_decoder) + 64)
    model.save_pretrained(path, safe_serialization=True)
    src = support.snapshot_path(K.CHECKPOINT_SMALL)
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, path / name)
    return path


def tiny_engine(tokenizer, *, blocks: tuple[int, ...] = (0, 1), max_batch_tokens: int = 4096) -> Any:
    """A ``CaptureEngine`` around the tiny model on CPU (fp32, causal mask policy)."""
    import torch

    from agents_scaling.study.neural.engine import CaptureEngine

    model = support.tiny_qwen3(seed=0, vocab_size=int(tokenizer.vocab_size) + len(tokenizer.added_tokens_decoder) + 64)
    return CaptureEngine(None, blocks, device_map="cpu", dtype=torch.float32, model=model,
                         max_batch_tokens=max_batch_tokens)
