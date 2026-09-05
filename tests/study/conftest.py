"""Shared fixtures for the study package tests (WP0).

* ``tmp_run_root`` — a run root with the architecture §2 directory skeleton.
* ``study_config`` — the committed ``configs/study_v4.yaml``.
* ``stub_tokenizer`` — a tiny whitespace tokenizer exposing the two methods the packet
  compiler and envelope code call, so those tests run in milliseconds.
* ``real_tokenizer_32b`` — the pinned Qwen3-32B tokenizer from the HF cache
  (``local_files_only``); skips when the cache is absent (fidelity T13/T15 need it).
* ``register_fake_server`` — writes a registry entry that
  ``agents_scaling.serving.registry.entry_matches_profile`` accepts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest

# The HF cache lives on the data filesystem; never let transformers touch ~/.cache.
os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agents_scaling.study import config as study_config_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
HANDOFF_DIR = REPO_ROOT / "docs" / "study_v4" / "handoff"
RUN_ROOT_SUBDIRS = ("servers", "cells", "requests", "data", "eval", "seals", "freeze")


@pytest.fixture
def tmp_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run_root"
    for name in RUN_ROOT_SUBDIRS:
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture(scope="session")
def study_config():
    return study_config_module.load_config()


class StubTokenizer:
    """Whitespace tokenizer with a stable, growing vocabulary.

    ``encode`` returns one id per whitespace-separated token; ``apply_chat_template``
    renders a Qwen-style ``<|im_start|>role\\ncontent<|im_end|>`` transcript (plus the
    generation prompt and the ``enable_thinking=False`` empty think block) and tokenizes it.
    Like transformers 5.x, ``tokenize=True`` returns ``{"input_ids", "attention_mask"}`` unless
    ``return_dict=False`` is passed, so envelope code must be written for both shapes.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = 1000 + len(self._vocab)
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(token) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        reverse = {value: key for key, value in self._vocab.items()}
        return " ".join(reverse.get(i, "<unk>") for i in ids)

    def render_chat(self, messages, add_generation_prompt: bool = True, enable_thinking: bool | None = None) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        if add_generation_prompt:
            tail = "<|im_start|>assistant\n"
            if enable_thinking is False:
                tail += "<think>\n\n</think>\n\n"
            parts.append(tail)
        return "\n".join(parts)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        return_dict: bool = True,
        **_: Any,
    ):
        text = self.render_chat(messages, add_generation_prompt, enable_thinking)
        if not tokenize:
            return text
        ids = self.encode(text)
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


@pytest.fixture
def stub_tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture(scope="session")
def real_tokenizer_32b(study_config):
    ckpt = study_config.checkpoints["32B"]
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            ckpt.hf_id, revision=ckpt.tokenizer_revision, local_files_only=True
        )
    except Exception as exc:  # cache absent, offline import failure, ...
        pytest.skip(f"Qwen3-32B tokenizer unavailable from the HF cache: {exc!r}")


def write_fake_server_entry(
    run_root: Path,
    profile: str,
    host: str = "node0000",
    port: int = 8001,
    **overrides: Any,
) -> Path:
    """Write ``<run_root>/servers/<profile>/<host>_<port>.json`` as a ``ServerEntry``.

    Fields come from the real serving profile so ``entry_matches_profile`` is satisfied;
    ``slurm_job_id`` is ``None`` so ``list_live_servers`` would probe ``/health`` instead of
    consulting Slurm (architecture §1 "registry layout").
    """
    from agents_scaling.serving.profiles import get_serving_profile

    spec = get_serving_profile(profile)
    entry: dict[str, Any] = {
        "model_size": spec.model_size,
        "hf_id": spec.hf_id,
        "host": host,
        "port": int(port),
        "slurm_job_id": None,
        "started_at": time.time(),
        "serving_profile": spec.name,
        "served_model_name": spec.served_model_name,
        "max_model_len": spec.max_model_len,
        "tp_size": spec.tp_size,
    }
    entry.update(overrides)
    path = Path(run_root) / "servers" / profile / f"{host}_{port}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def register_fake_server() -> Callable[..., Path]:
    return write_fake_server_entry


@pytest.fixture
def request_record_fixture() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "request_record.example.json").read_text(encoding="utf-8"))


@pytest.fixture
def episode_result_fixture() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "episode_result.example.json").read_text(encoding="utf-8"))
