"""WP-B: the pre-registered fallback H3 -- OpenClaw 2026.3.8 with an ephemeral gateway per run (plan §6 OpenClaw).

OpenClaw's ``sessions_spawn`` is non-blocking and ``--local`` cannot spawn (spawn and announce both call the
gateway), so one run = one ephemeral gateway process group: ``openclaw gateway run --port <lane> --bind loopback
--auth token --allow-unconfigured`` with a generated config (``OPENCLAW_CONFIG_PATH``; ``models.providers.meter``
at the run's proxy URL with ``api: "openai-completions"`` and a ``compat`` block; ``tools.profile: minimal`` with
``allow: [sessions_spawn, subagents]``; ``contextPruning: off``; ``compaction: default`` with ``memoryFlush``
disabled; plugins off; ``skills.allowBundled: []``), ``HOME`` / ``OPENCLAW_STATE_DIR`` on scratch; the turn is a
``POST /v1/chat/completions`` on the gateway with ``x-openclaw-session-key``; the final answer is read from the
requester-session transcript, closing on the last ``<final>`` after ``idle_seconds`` of proxy silence with no
active child, or on exhaustion, or on the wall clock. The study system prompt has no ``--system-prompt``
equivalent: it is delivered as the agent workspace's ``AGENTS.md`` (pre-registered deviation).

This adapter is the fallback only: ``openclaw config validate`` and the announce-landing check belong to Gate 0;
unit tests here cover the generated files and the launch plan and skip when the CLI is absent.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from agent_allocation.dynorch.harness.base import (
    HarnessResult, LaunchPlan, WorkDir, default_base_env, harness_env, probe_version, write_text,
)
from agent_allocation.dynorch.records import RunSpec

__all__ = ["OpenclawAdapter", "openclaw_config"]

_FINAL_RE = re.compile(r"<final>.*?</final>", re.DOTALL)


def openclaw_config(*, model: str, proxy_base_url: str, run_key: str, workspace: Path, port: int, token: str,
                    max_spawn_depth: int = 1) -> dict[str, Any]:
    """The generated OpenClaw config (module docstring); plain JSON, which OpenClaw's JSON5 loader accepts."""
    model_id = model.split("/", 1)[1] if "/" in model else model
    return {
        "gateway": {"mode": "local", "port": port, "bind": "loopback", "auth": {"mode": "token", "token": token},
                    "http": {"endpoints": {"chatCompletions": {"enabled": True}}}, "controlUi": {"enabled": False}},
        "discovery": {"mdns": {"mode": "off"}},
        "models": {"mode": "replace", "providers": {"meter": {
            "baseUrl": proxy_base_url, "apiKey": run_key, "api": "openai-completions",
            "models": [{"id": model_id, "name": "metered model", "reasoning": True, "input": ["text"],
                        "cost": {"input": 0.24, "output": 2.5, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 131072, "maxTokens": 32000,
                        "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                                   "supportsReasoningEffort": False, "supportsStrictMode": False,
                                   "supportsUsageInStreaming": True, "maxTokensField": "max_tokens"}}]}}},
        "agents": {"defaults": {
            "workspace": str(workspace), "skipBootstrap": True, "userTimezone": "UTC",
            "model": {"primary": f"meter/{model_id}", "fallbacks": []},
            "thinkingDefault": "off", "verboseDefault": "off", "elevatedDefault": "off",
            "timeoutSeconds": 3600, "maxConcurrent": 4, "heartbeat": {"every": "0m"},
            "compaction": {"mode": "default", "memoryFlush": {"enabled": False}},
            "contextPruning": {"mode": "off"},
            "subagents": {"maxSpawnDepth": max_spawn_depth, "maxChildrenPerAgent": 4, "maxConcurrent": 4,
                          "runTimeoutSeconds": 3000, "archiveAfterMinutes": 60}},
            "list": [{"id": "main", "default": True, "workspace": str(workspace),
                      "subagents": {"allowAgents": ["main"]}}]},
        "tools": {"profile": "minimal", "allow": ["sessions_spawn", "subagents"],
                  "deny": ["group:runtime", "group:fs", "group:web", "group:ui", "group:automation",
                           "group:messaging", "group:nodes", "group:memory", "edit", "apply_patch", "image",
                           "read", "write", "sessions_send", "sessions_list", "sessions_history", "agents_list",
                           "session_status"],
                  "subagents": {"tools": {"allow": []}}},
        "skills": {"allowBundled": []},
        "plugins": {"enabled": False, "slots": {"memory": "none"}},
        "session": {"reset": {"mode": "idle", "idleMinutes": 100000}},
        "channels": {},
        "logging": {"level": "info"},
    }


class OpenclawAdapter:
    name = "openclaw"
    in_process = True
    _version_cache: dict[str, str] = {}

    def __init__(self, *, model: str, binary: str = "openclaw", port_base: int = 19000, idle_seconds: float = 90.0,
                 boot_seconds: float = 30.0) -> None:
        self.model = model
        self.binary = binary
        self.port_base = port_base
        self.idle_seconds = idle_seconds
        self.boot_seconds = boot_seconds
        self._lane = 0

    # -- files -------------------------------------------------------------------------------------------
    @staticmethod
    def paths(work: Path) -> dict[str, Path]:
        wd = WorkDir(Path(work))
        return {"state": wd.state / "openclaw", "config": wd.root / "openclaw.json", "workspace": wd.root / "workspace",
                "meta": wd.root / "openclaw_meta.json", "gateway_log": wd.root / "gateway.log"}

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None:
        wd = WorkDir(Path(work)).create()
        write_text(wd.system_md, system_text)
        write_text(wd.solver_md, subagent_text)
        write_text(wd.task_md, task_text)
        p = self.paths(wd.root)
        p["state"].mkdir(parents=True, exist_ok=True)
        p["workspace"].mkdir(parents=True, exist_ok=True)
        write_text(p["workspace"] / "AGENTS.md", system_text + "\n\n## Sub-agent contract\n\n" + subagent_text)
        self._lane += 1
        port = self.port_base + (self._lane % 900)
        token = secrets.token_hex(16)
        max_depth = 1 if spec.spawn_tools else 0
        write_text(p["config"], json.dumps(openclaw_config(model=self.model, proxy_base_url=proxy_base_url,
                                                            run_key=spec.run_key, workspace=p["workspace"], port=port,
                                                            token=token, max_spawn_depth=max_depth), indent=1))
        write_text(p["meta"], json.dumps({"port": port, "token": token}))

    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan:
        wd = WorkDir(Path(work))
        p = self.paths(wd.root)
        meta = json.loads(p["meta"].read_text("utf-8"))
        env = harness_env(default_base_env(), work=wd.root, proxy_base_url=proxy_base_url, run_key=spec.run_key,
                          extra={"OPENCLAW_STATE_DIR": str(p["state"]), "OPENCLAW_CONFIG_PATH": str(p["config"]),
                                 "OPENCLAW_GATEWAY_TOKEN": meta["token"], "OPENCLAW_GATEWAY_PORT": str(meta["port"])})
        argv = (self.binary, "gateway", "run", "--port", str(meta["port"]), "--bind", "loopback", "--auth", "token",
                "--token", meta["token"], "--allow-unconfigured")
        return LaunchPlan(argv=argv, env=env, cwd=wd.ws, stdin=None)

    # -- run ---------------------------------------------------------------------------------------------
    async def run_in_process(self, work: Path, spec: RunSpec, proxy_base_url: str, *, wall_seconds: float,
                             ) -> HarnessResult:
        wd = WorkDir(Path(work))
        p = self.paths(wd.root)
        plan = self.build_command(wd.root, spec, proxy_base_url)
        meta = json.loads(p["meta"].read_text("utf-8"))
        started = time.monotonic()
        log = open(p["gateway_log"], "ab")
        proc = await asyncio.create_subprocess_exec(*plan.argv, cwd=str(plan.cwd), env=plan.env, stdout=log,
                                                    stderr=subprocess.STDOUT, start_new_session=True)
        reported: dict[str, Any] = {"port": meta["port"], "closed_by": None}
        try:
            gateway = f"http://127.0.0.1:{meta['port']}"
            headers = {"Authorization": f"Bearer {meta['token']}", "x-openclaw-agent-id": "main",
                       "x-openclaw-session-key": f"agent:main:dyn-{spec.run_key}"}
            async with httpx.AsyncClient(timeout=httpx.Timeout(wall_seconds, connect=5.0)) as client:
                if not await self._wait_ready(client, gateway, proc):
                    reported["closed_by"] = "boot_failed"
                    return HarnessResult(exit_code=1, timed_out=False, cli_result=None, reported=reported)
                turn = {"model": "openclaw:main", "stream": False,
                        "messages": [{"role": "user", "content": wd.task_md.read_text("utf-8")}]}
                try:
                    response = await client.post(f"{gateway}/v1/chat/completions", json=turn, headers=headers)
                    reported["turn_status"] = response.status_code
                    write_text(wd.stdout, response.text)
                except httpx.HTTPError as exc:
                    reported["turn_error"] = f"{type(exc).__name__}: {exc}"
                final = await self._watch_transcript(p["state"], deadline=started + wall_seconds, reported=reported)
            return HarnessResult(exit_code=0 if final is not None else 1, timed_out=reported["closed_by"] == "wall_clock",
                                 cli_result=final, reported=reported)
        finally:
            self._kill_group(proc)
            log.close()

    async def _wait_ready(self, client: httpx.AsyncClient, gateway: str, proc: asyncio.subprocess.Process) -> bool:
        deadline = time.monotonic() + self.boot_seconds
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                return False
            try:
                r = await client.get(f"{gateway}/health", timeout=2.0)
                if r.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        return False

    async def _watch_transcript(self, state: Path, *, deadline: float, reported: dict[str, Any]) -> str | None:
        sessions_dir = state / "agents" / "main" / "sessions"
        last_final: str | None = None
        last_change = time.monotonic()
        seen_size = 0
        while True:
            now = time.monotonic()
            if now >= deadline:
                reported["closed_by"] = "wall_clock"
                return last_final
            size = 0
            texts: list[str] = []
            if sessions_dir.is_dir():
                for path in sorted(sessions_dir.glob("*.jsonl")):
                    if "subagent" in path.name:
                        continue
                    data = path.read_text("utf-8", "replace")
                    size += len(data)
                    texts.append(data)
            if size != seen_size:
                seen_size = size
                last_change = now
                finals = _FINAL_RE.findall("\n".join(texts))
                if finals:
                    last_final = "\n".join(texts)[-20000:]
            if last_final is not None and now - last_change >= self.idle_seconds:
                reported["closed_by"] = "final"
                return last_final
            await asyncio.sleep(2.0)

    @staticmethod
    def _kill_group(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult:
        raise NotImplementedError("OpenclawAdapter runs in-process; the runner uses run_in_process")

    def version(self) -> str:
        if self.binary in self._version_cache:
            return self._version_cache[self.binary]
        path = shutil.which(self.binary)
        version = "absent" if path is None else probe_version([path, "--version"])
        self._version_cache[self.binary] = version
        return version
