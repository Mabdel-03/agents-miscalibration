"""Shared fixtures for the WP-A unit tests: the frozen config, the fake upstream, a ledger in ``tmp_path`` and a
``ProxyCore`` wired over them. No socket is ever opened; no key is ever read."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from agent_allocation.api import fake
from agent_allocation.api.fake import FakeScript, fake_forced_endpoint_lock
from agent_allocation.api.forced import ForcedClient, ForcedSettings, template_sha256
from agent_allocation.dynorch.config import PROXY_VERSION, DynorchSettings
from agent_allocation.dynorch.core import ProxyCore
from agent_allocation.dynorch.journal import CallJournal, RunStore
from agent_allocation.dynorch.records import ChatRequest, Message, RunSpec, ToolSpec
from agent_allocation.dynorch.upstream import Upstream
from agent_allocation.runtime.ledger import CollectionLedger
from agent_allocation.storage.journal import LedgerJournal

PROJECT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT / "configs" / "dynorch_v1.json"
PROMPTS = PROJECT / "prompts"
ROOT_SYSTEM = (PROMPTS / "system_dynorch.txt").read_text()
SUBAGENT_SYSTEM = (PROMPTS / "subagent_dynorch.txt").read_text()
H0_SYSTEM = (PROMPTS / "system.txt").read_text()
AGENT_TOOL = ToolSpec(name="Agent", description="Launch a sub-agent and wait for its result.",
                      parameters={"type": "object", "properties": {"prompt": {"type": "string"},
                                                                   "subagent_type": {"type": "string"},
                                                                   "description": {"type": "string"}},
                                  "required": ["prompt", "subagent_type"]})
SPAWN_TOOL = ToolSpec(name="spawn_agent", description="Spawn a Codex sub-agent.",
                      parameters={"type": "object", "properties": {"message": {"type": "string"},
                                                                   "fork_context": {"type": "boolean"}}})


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No unit test may leave the host: only loopback (the proxy's own h11 server in WP-B's httpd tests) is
    reachable; every other host raises before a socket is opened."""
    real_async = httpx.AsyncHTTPTransport.handle_async_request
    real_sync = httpx.HTTPTransport.handle_request

    def _loopback(request: httpx.Request) -> bool:
        return request.url.host in ("127.0.0.1", "localhost", "::1")

    async def guarded_async(self: Any, request: httpx.Request) -> Any:
        if not _loopback(request):
            raise AssertionError(f"dynorch unit tests must never open a real connection (to {request.url.host})")
        return await real_async(self, request)

    def guarded_sync(self: Any, request: httpx.Request) -> Any:
        if not _loopback(request):
            raise AssertionError(f"dynorch unit tests must never open a real connection (to {request.url.host})")
        return real_sync(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded_sync)


def load_cfg() -> dict[str, Any]:
    return json.loads(CONFIG.read_text())


@pytest.fixture
def cfg() -> dict[str, Any]:
    return load_cfg()


@pytest.fixture
def settings(cfg: dict[str, Any]) -> DynorchSettings:
    return DynorchSettings.from_config(cfg)


def make_spec(*, rung: int = 4, arm: str = "H1", item_id: str = "aime/2025/1", benchmark: str = "aime",
              shard_id: str = "parasail", attempt: int = 1, generation: int = 0, repeat: int = 1,
              allowed_tools: tuple[str, ...] = ("Agent",), spawn_tools: tuple[str, ...] = ("Agent",),
              root_call_ceiling: int | None = None, run_key: str | None = None, harness: str = "openclaude",
              partition: str = "main") -> RunSpec:
    return RunSpec(
        run_key=run_key or f"rk{arm}{rung}r{repeat}a{attempt}{item_id.replace('/', '')}",
        arm=arm, harness=harness, item_id=item_id, benchmark=benchmark, stratum="main", shard_id=shard_id,
        endpoint_tag=f"{shard_id}/fp8", rung=rung, cap_tokens=1024 * rung, repeat=repeat, attempt=attempt,
        generation=generation, partition=partition, allowed_tools=allowed_tools, spawn_tools=spawn_tools,
        root_call_ceiling=root_call_ceiling, sampling_hash="s" * 16, template_sha256=template_sha256(),
        prompt_version="dynorch-v1", proxy_version=PROXY_VERSION, harness_version="test")


def make_request(messages: list[Message], *, tools: tuple[ToolSpec, ...] = (AGENT_TOOL,),
                 requested_max_tokens: int | None = 32000, effort: str | None = None, stream: bool = False,
                 wire: str = "chat", ignored: tuple[str, ...] = (), subagent: bool = False) -> ChatRequest:
    return ChatRequest(messages=tuple(messages), tools=tools, requested_max_tokens=requested_max_tokens,
                       requested_reasoning_effort=effort, stream=stream, ignored_keys=ignored, wire=wire,
                       subagent=subagent)


def root_messages(task: str = "TASK\nWhat is 6 * 7?\n", system: str = ROOT_SYSTEM) -> list[Message]:
    return [Message(role="system", content=system), Message(role="user", content=task)]


class Env:
    """One wired core: fake script, forced client, ledger, journal, store."""

    def __init__(self, tmp: Path, cfg: dict[str, Any], script: FakeScript | None = None,
                 ceiling_usd: str = "95", **upstream_kwargs: Any) -> None:
        self.tmp = tmp
        self.cfg = cfg
        self.settings = DynorchSettings.from_config(cfg)
        self.script = script if script is not None else FakeScript()
        forced_cfg = dict(cfg)
        forced_cfg["provider"] = {**cfg["provider"], "endpoint_tag": "parasail/fp8"}
        self.client = ForcedClient(settings=ForcedSettings.from_config(forced_cfg), api_key=None,
                                   transport=fake.fake_openrouter_transport(self.script),
                                   answer_backoff_base_seconds=0.0)
        self.run_root = tmp / "run_root"
        self.ledger_journal = LedgerJournal(self.run_root / "state" / "ledger_journal.jsonl", group_commit_ms=0)
        self.ledger = CollectionLedger(self.ledger_journal, ceiling_usd=Decimal(ceiling_usd),
                                       max_unresolved_usd=self.settings.max_unresolved_spend_usd)
        self.locks = {"parasail": fake_forced_endpoint_lock(tag="parasail/fp8"),
                      "mancer": fake_forced_endpoint_lock(tag="mancer/fp8")}
        self.sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        kwargs = {"sleep": fake_sleep, "reservation_multiplier": self.settings.reservation_multiplier}
        kwargs.update(upstream_kwargs)
        self.upstream = Upstream(self.client, self.locks, cfg["sampling"], self.settings.provider_in_flight,
                                 self.settings.provider_rps, self.ledger, **kwargs)
        self.journal = CallJournal(self.run_root)
        self.store = RunStore(self.run_root)
        self.core = ProxyCore(self.settings, self.upstream, self.journal, self.store)

    def open(self, spec: RunSpec, *, root_system: str = ROOT_SYSTEM, subagent_system: str | None = SUBAGENT_SYSTEM
             ) -> RunSpec:
        self.core.open_run(spec, root_system_text=root_system, subagent_system_text=subagent_system)
        return spec

    def think_calls(self, phase: str | None = None) -> list[fake.FakeCall]:
        return self.script.text_completion_calls(phase=phase)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.ledger_journal.flush()
        self.ledger_journal.release_writer()


@pytest.fixture
def make_env(tmp_path: Path, cfg: dict[str, Any]) -> Callable[..., Env]:
    envs: list[Env] = []

    def build(script: FakeScript | None = None, **kwargs: Any) -> Env:
        env = Env(tmp_path / f"env{len(envs)}", cfg, script, **kwargs)
        envs.append(env)
        return env

    yield build
    import asyncio

    for env in envs:
        try:
            asyncio.run(env.aclose())
        except RuntimeError:
            pass


@pytest.fixture(autouse=True)
def _no_real_key_settle_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract §10.36: the /key settle loop takes a confirming read ``KEY_SETTLE_INTERVAL_S`` after a match. No unit
    test waits for it on the wall clock; the tests that pin the timing pass their own ``sleep`` explicitly."""
    from agent_allocation.dynorch import reconcile

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(reconcile, "_sleep", instant)
