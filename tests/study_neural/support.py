"""Shared helpers for the N1 tests and the GPU smoke job.

* ``tiny_qwen3()`` — a two-layer, hidden-32 random Qwen3 causal LM on CPU (transformers
  ``Qwen3Config``), deterministic under a seed.
* ``synthetic_record(...)`` — a ``RequestRecord``-shaped dict from explicit token ids
  (the capture code accepts mappings with the on-disk field names or ``RequestRecord``).
* ``qwen_tokenizer(size)`` — the pinned Qwen3 tokenizer from the offline HF cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HF_HOME = Path(os.environ["HF_HOME"])
SNAPSHOTS = {
    "8B": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "14B": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    "32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
}

TINY_VOCAB = 256
TINY_HIDDEN = 32
TINY_LAYERS = 2


def snapshot_path(size: str) -> Path:
    hf_id, revision = SNAPSHOTS[size]
    return HF_HOME / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots" / revision


def qwen_tokenizer(size: str = "8B"):
    from transformers import AutoTokenizer

    path = snapshot_path(size)
    if not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(path)
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def tiny_qwen3(seed: int = 0, vocab_size: int = TINY_VOCAB, hidden: int = TINY_HIDDEN, layers: int = TINY_LAYERS):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).eval()
    return model


def synthetic_record(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    *,
    messages: list[dict[str, Any]] | None = None,
    content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str = "stop",
    enable_thinking: bool = True,
    request_id: str = "0" * 64,
) -> dict[str, Any]:
    """A record with the on-disk ``RequestRecord`` field names the neural code reads."""
    return {
        "schema_version": 1,
        "request_id": request_id,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_tokens": len(prompt_token_ids),
        "messages": messages if messages is not None else [{"role": "user", "content": "synthetic"}],
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "response": {
            "token_ids": list(completion_token_ids),
            "completion_tokens": len(completion_token_ids),
            "content": content,
            "reasoning": reasoning,
            "finish_reason": finish_reason,
        },
    }


def completion_from_text(tokenizer, think_text: str | None, content: str, *, im_end: bool = True) -> list[int]:
    """Completion ids ``<think>\\n...\\n</think><content>[<|im_end|>]`` as vLLM stores them: the
    reasoning parser splits at ``</think>`` and ``content`` keeps its leading ``\\n\\n``."""
    text = (f"<think>\n{think_text}\n</think>" if think_text is not None else "") + content
    ids = tokenizer.encode(text, add_special_tokens=False)
    if im_end:
        ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return ids


CANDIDATE_JSON = (
    '{"approach":"add the two integers","evidence":[{"claim":"2+2=4","support":"arithmetic","uncertainty":"low"}],'
    '"alternatives_considered":[],"failure_checks":["checked twice"],"final_answer":"4","confidence":0.9}'
)
CANDIDATE_JSON_CJK = (
    '{"approach":"两个整数相加","evidence":[{"claim":"二加二等于四","support":"算术","uncertainty":"low"}],'
    '"alternatives_considered":["五"],"failure_checks":[],"final_answer":"四","confidence":0.75}'
)
