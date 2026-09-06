"""Shared helpers for the WP2 tests (also usable by other packages' tests).

``make_spec`` builds a :class:`RequestSpec` from the committed manifest with a stateless
bank seed key (§3.6, architecture §2.3); ``fake_server`` starts a :class:`FakeVllmServer`
and registers it under ``<run_root>/servers/<profile>/``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agents_scaling.study import types as T
from agents_scaling.study.config import StudyConfig
from tests.study.fake_vllm import FakeVllmServer

SCHEMA_LINE = (
    '{"approach":string,"evidence":[{"claim":string,"support":string,"uncertainty":"low"|"medium"|"high"|"unknown"}],'
    '"alternatives_considered":[string],"failure_checks":[string],"final_answer":string,"confidence":number in [0,1]}'
)


def solver_messages(task: str = "What is 2 + 2? Answer with the integer.") -> tuple[dict[str, str], ...]:
    return (
        {
            "role": "user",
            "content": (
                "Solve the supplied task independently and return one complete answer.\n\n"
                f"Task: {task}\nOutput contract: {SCHEMA_LINE}"
            ),
        },
    )


def make_spec(
    cfg: StudyConfig,
    *,
    messages: tuple[dict[str, Any], ...] | None = None,
    decoding: T.Decoding = T.SOLVER_DECODING,
    size: str = "32B",
    source_id: str = "hle:test-item-1",
    split: str = "dev",
    step_slot: int = 0,
    actor_slot: int = 0,
    purpose: str = T.PURPOSE_ROOT,
    namespace: str = T.NS_STATELESS_BANK,
    episode_rep: int = 0,
    role: str = "root",
) -> T.RequestSpec:
    ckpt = cfg.checkpoints[size]
    key = T.SeedKey(
        source_id=source_id,
        split=split,
        model_cell=ckpt.model_cell,
        episode_rep=episode_rep,
        actor_slot=actor_slot,
        purpose=purpose,
        step_slot=step_slot,
        namespace=namespace,
    )
    return T.RequestSpec(
        messages=messages if messages is not None else solver_messages(),
        decoding=decoding,
        checkpoint=ckpt,
        seed_key=key,
        role=role,
        study_id=cfg.study_id,
        study_seed_hex=cfg.study_seed_hex,
    )


@contextmanager
def fake_server(run_root: Path, profile: str = "32B-long", **kwargs: Any) -> Iterator[FakeVllmServer]:
    """A started, registered fake endpoint for ``profile`` (stopped on exit)."""
    from agents_scaling.serving.profiles import get_serving_profile

    kwargs.setdefault("served_model_name", get_serving_profile(profile).served_model_name)
    server = FakeVllmServer(**kwargs).start()
    server.register(run_root, profile)
    try:
        yield server
    finally:
        server.stop()
