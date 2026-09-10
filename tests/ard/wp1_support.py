"""Shared helpers for the ARD36 WP1 (inference) tests.

Builds the pieces ``ard/client.generate_decision`` needs — the committed
``configs/ard36.yaml``, a :class:`~agents_scaling.ard.client.DecisionContext` over the
fake tokenizer, a registered fake vLLM endpoint on an ARD36 serving profile, and a
:class:`~agents_scaling.study.inference.store.RequestStore` — so each test states only
what it is actually about.

Deliberately not a ``conftest.py``: ``tests/ard/conftest.py`` is shared ground for every
ARD work package and this file belongs to WP1 alone.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# The HF cache lives on the data filesystem; never let transformers touch ~/.cache.
# ``tests/study/conftest.py`` does the same, but it is not loaded for this directory.
os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agents_scaling.ard import contract as K  # noqa: E402
from agents_scaling.ard.client import DecisionContext, DecisionRequest  # noqa: E402
from agents_scaling.study.config import StudyConfig, load_config  # noqa: E402
from agents_scaling.study.inference.client import EndpointPool  # noqa: E402
from agents_scaling.study.inference.store import RequestStore  # noqa: E402
from agents_scaling.study.types import SeedKey  # noqa: E402
from tests.study.fake_vllm import FakeVllmServer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ARD_CONFIG_PATH = REPO_ROOT / "configs" / "ard36.yaml"
RUN_ROOT_SUBDIRS = ("servers", "cells", "requests", "data", "eval", "freeze")

#: The frozen report contract, as a prompt would state it (spec §6.3).  The literal
#: ``<final>`` is what routes the fake server to its ARD report generator.
REPORT_CONTRACT = (
    "Output contract: <final>answer</final><evidence>at most 120 tokens</evidence>"
    "<confidence>p in [0,1]</confidence>"
)


def ard_config() -> StudyConfig:
    """The committed ARD36 manifest (also proves the study loader accepts it)."""
    return load_config(ARD_CONFIG_PATH)


def make_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run_root"
    for name in RUN_ROOT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def decision_messages(task: str = "What is 2 + 2? Answer with the integer.") -> tuple[dict[str, str], ...]:
    """A neutral root prompt carrying the ARD36 report contract."""
    return (
        {
            "role": "user",
            "content": (
                "Solve the task independently and report in the required format.\n\n"
                f"Task: {task}\n{REPORT_CONTRACT}"
            ),
        },
    )


def make_context(cfg: StudyConfig, tokenizer: Any, *, run_id: str = "ard36-test") -> DecisionContext:
    """A context over one tokenizer shared by both checkpoints (the fake has one vocab)."""
    return DecisionContext(
        checkpoints=dict(cfg.checkpoints),
        tokenizers={size: tokenizer for size in cfg.checkpoints},
        study_seed_hex=cfg.study_seed_hex,
        study_id=cfg.study_id,
        run_id=run_id,
    )


def seed_key(
    cfg: StudyConfig,
    *,
    size: str = K.CHECKPOINT_SMALL,
    source_id: str = "math500:test-0001",
    split: str = "test",
    episode_rep: int = 0,
    actor_slot: int = 0,
    purpose: str = K.PURPOSE_ROOT,
    step_slot: int = 0,
    namespace: str = K.NS_STATELESS_BANK,
) -> SeedKey:
    return SeedKey(
        source_id=source_id,
        split=split,
        model_cell=cfg.checkpoints[size].model_cell,
        episode_rep=episode_rep,
        actor_slot=actor_slot,
        purpose=purpose,
        step_slot=step_slot,
        namespace=namespace,
    )


def make_request(
    cfg: StudyConfig,
    *,
    size: str = K.CHECKPOINT_SMALL,
    thinking_tokens: int = K.THINKING_TOKENS_LOW,
    messages: tuple[dict[str, Any], ...] | None = None,
    purpose: str = K.PURPOSE_ROOT,
    **key_kwargs: Any,
) -> DecisionRequest:
    return DecisionRequest(
        messages=messages if messages is not None else decision_messages(),
        checkpoint=size,
        thinking_tokens=thinking_tokens,
        seed_key=seed_key(cfg, size=size, purpose=purpose, **key_kwargs),
        purpose=purpose,
    )


@contextmanager
def fake_server(
    run_root: Path, profile: str = K.PROFILE_SMALL, **kwargs: Any
) -> Iterator[FakeVllmServer]:
    """A started, registered fake endpoint on an ARD36 serving profile.

    Pass ``tokenizer=`` the very instance the client renders with: ``FakeTokenizer.decode``
    only knows the tokens its own instance has seen, so a report decoded through a
    different instance would come back as ``<unk>``.
    """
    from agents_scaling.serving.profiles import get_serving_profile

    spec = get_serving_profile(profile)
    kwargs.setdefault("served_model_name", spec.served_model_name)
    kwargs.setdefault("max_model_len", spec.max_model_len)
    server = FakeVllmServer(**kwargs).start()
    server.register(run_root, profile)
    try:
        yield server
    finally:
        server.stop()


def make_pool(run_root: Path, profile: str = K.PROFILE_SMALL, **kwargs: Any) -> EndpointPool:
    kwargs.setdefault("refresh_min_interval_s", 0.0)
    kwargs.setdefault("probe_timeout", 2.0)
    return EndpointPool(run_root, profile, shard=0, **kwargs)


def make_store(run_root: Path) -> RequestStore:
    return RequestStore(run_root / "requests")
