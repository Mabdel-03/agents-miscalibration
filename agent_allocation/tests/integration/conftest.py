"""Shared fixtures for agent 08's integration/acceptance tests.

Rules these fixtures enforce for every test in this directory:
  * no network: ``httpx`` may only be used with a MockTransport; a guard fails any real socket connect
  * no key: ``OPENROUTER_API_KEY`` is removed; tests that need a key-shaped value use the fake canary key
  * every run directory is a fresh tmp dir with explicit ``locks_dir`` and ``local_state_dir`` (see
    ``docs/requests/08.md``: ``RunPaths.for_run`` otherwise shares SQLite state/locks across run dirs)
"""
from __future__ import annotations

import importlib
import importlib.util
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.config import grids_from_config, load_config
from agent_allocation.contracts import STUDY_ID, EndpointLock
from agent_allocation.paths import RunPaths

from tests.integration import adversarial

PROJECT = Path(__file__).resolve().parents[2]


def load_reference_dag_module():
    path = PROJECT / "tools" / "reference_dag.py"
    spec = importlib.util.spec_from_file_location("reference_dag_for_integration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def optional_module(name: str):
    """``pytest.importorskip`` that re-raises real import errors inside a module that *does* exist.

    A module that exists but fails to import (syntax error, bad dependency) is a defect, not "not merged yet".
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if name == missing or name.startswith(missing + "."):
            pytest.skip(f"waiting on module {name} (not merged: {missing})")
        raise


def contract_lock(model: str = adversarial.MODEL) -> EndpointLock:
    """A contract-valid lock for tests that must not depend on ``api.fake`` yet."""
    return EndpointLock(model=model, provider_tag="google-ai-studio", provider_name="Google AI Studio",
                        service_tier="standard", input_price_per_token="0.0000001",
                        output_price_per_token="0.0000004", internal_reasoning_price_per_token="0.0000004",
                        context_length=1048576, max_completion_tokens=65535,
                        supported_parameters=("max_tokens", "reasoning", "temperature", "top_p"))


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    return load_config(PROJECT / "configs" / "balanced150.json")


@pytest.fixture(scope="session")
def lean_cfg() -> dict[str, Any]:
    return load_config(PROJECT / "configs" / "lean100.json")


@pytest.fixture(scope="session")
def grids(cfg):
    return grids_from_config(cfg)


@pytest.fixture(scope="session")
def reference_dag():
    return load_reference_dag_module()


@pytest.fixture(scope="session")
def adv_items():
    return adversarial.load_items()


@pytest.fixture(scope="session")
def canaries() -> dict[str, str]:
    return adversarial.all_canaries()


@pytest.fixture()
def run_paths(tmp_path) -> RunPaths:
    paths = RunPaths.for_run(tmp_path / "artifacts" / STUDY_ID, local_state_dir=tmp_path / "local_state")
    paths.ensure()
    return paths


@pytest.fixture(autouse=True)
def _no_network_no_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_FILE", raising=False)
    real_connect = socket.socket.connect

    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"integration tests must not open network sockets (tried {address!r})")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield
