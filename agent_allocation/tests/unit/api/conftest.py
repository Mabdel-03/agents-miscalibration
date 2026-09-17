"""Shared fixtures for agent 01's tests. Every test here runs on the fake transport; real sockets are forbidden."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from agent_allocation.api.fake import fake_endpoint_lock
from agent_allocation.contracts import EndpointLock, GenerationKind, GenerationSpec

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "openrouter"
FLASH_LITE = "google/gemini-2.5-flash-lite"
FLASH = "google/gemini-2.5-flash"
DUMMY_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef"
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "api_seed_enabled": False, "response_cache": False,
            "n_per_request": 1}
MESSAGES = [{"role": "system", "content": "Solve the supplied task."},
            {"role": "user", "content": "TASK\nWhat is 6 * 7?\n"}]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("agent 01 unit tests must never open a real connection")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


def load_snapshot(model: str) -> dict[str, Any]:
    name = model.split("/", 1)[1]
    return json.loads((FIXTURES / f"endpoints_{name}_2026-09-17.json").read_text())


@pytest.fixture
def lock() -> EndpointLock:
    return fake_endpoint_lock()


@pytest.fixture
def make_spec(lock: EndpointLock) -> Callable[..., GenerationSpec]:
    def build(**overrides: Any) -> GenerationSpec:
        fields: dict[str, Any] = dict(
            run_id="unit", partition_id="pilot", item_id="math500/test/algebra/1", benchmark="math500",
            kind=GenerationKind.ROOT, model=lock.model, endpoint_lock_hash=lock.lock_hash,
            input_manifest_hash="a" * 64, decoding_config_hash="b" * 64, prompt_version="prompts-v1-000000000000",
            reasoning_budget=1024, completion_headroom=1024, root_slot=0, team_size=0, update_round=0, draw_id=0)
        fields.update(overrides)
        return GenerationSpec(**fields)

    return build


def attempt(spec: GenerationSpec, number: int = 1) -> str:
    return f"{spec.node_id}:{number}"
