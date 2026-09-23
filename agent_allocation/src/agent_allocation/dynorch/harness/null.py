"""WP-B: H0, the harness-free control (contract §6 ``null.py``; plan §1 H0).

No subprocess: one non-streaming chat request ``[system.txt, user text]`` with no tools and ``max_tokens = C``
(recorded and ignored by the proxy, which grants ``C - spent`` anyway) through the same ``/r/<run_key>/v1``
route every harness uses, so the call is metered, journaled and classified exactly like a harness's root call.
The response JSON is written to ``stdout`` in the work dir; ``cli_result`` is the message content (a cross-check
only -- the delivered answer still comes from the proxy's reply log). Any HTTP error is reported, not raised:
the runner's rerun rule decides.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from agent_allocation.dynorch.harness.base import (
    HarnessResult, LaunchPlan, WorkDir, default_base_env, harness_env, write_text,
)
from agent_allocation.dynorch.records import RunSpec

__all__ = ["NullAdapter"]


class NullAdapter:
    name = "null"
    in_process = True

    def __init__(self, *, model: str, timeout_seconds: float | None = None) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    def version(self) -> str:
        return "null/1"

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None:
        wd = WorkDir(Path(work)).create()
        write_text(wd.system_md, system_text)
        write_text(wd.task_md, task_text)
        write_text(wd.solver_md, subagent_text)

    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan:
        env = harness_env(default_base_env(), work=Path(work), proxy_base_url=proxy_base_url, run_key=spec.run_key)
        return LaunchPlan(argv=(), env=env, cwd=WorkDir(Path(work)).ws, stdin=None)

    def request_body(self, work: Path, spec: RunSpec) -> dict[str, Any]:
        wd = WorkDir(Path(work))
        return {"model": self.model, "stream": False, "max_tokens": spec.cap_tokens,
                "messages": [{"role": "system", "content": wd.system_md.read_text("utf-8")},
                             {"role": "user", "content": wd.task_md.read_text("utf-8")}]}

    async def run_in_process(self, work: Path, spec: RunSpec, proxy_base_url: str, *, wall_seconds: float,
                             ) -> HarnessResult:
        wd = WorkDir(Path(work))
        timeout = self.timeout_seconds if self.timeout_seconds is not None else wall_seconds
        body = self.request_body(work, spec)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
                response = await client.post(f"{proxy_base_url}/chat/completions", json=body,
                                             headers={"Authorization": f"Bearer {spec.run_key}"})
        except httpx.HTTPError as exc:
            write_text(wd.stdout, json.dumps({"type": "result", "subtype": "transport_error", "error": str(exc)}))
            return HarnessResult(exit_code=1, timed_out=isinstance(exc, httpx.TimeoutException), cli_result=None,
                                 reported={"error": f"{type(exc).__name__}: {exc}"})
        write_text(wd.stdout, response.text)
        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {"error": {"code": "non_json", "message": response.text[:500]}}
            return HarnessResult(exit_code=1, timed_out=False, cli_result=None,
                                 reported={"status": response.status_code, "error": payload.get("error")})
        payload = response.json()
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        return HarnessResult(exit_code=0, timed_out=False, cli_result=content if content.strip() else None,
                             reported={"status": 200, "finish_reason": choice.get("finish_reason"),
                                       "usage": payload.get("usage"), "dynorch": payload.get("dynorch"),
                                       "num_turns": 1})

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult:
        raise NotImplementedError("NullAdapter runs in-process; the runner uses run_in_process")
