"""WP-B: the fake harness adapter (contract §6 ``fake.py``) driving ``tests/dynorch/fake_harness_cli.py``.

The CLI behaves like OpenClaude against the proxy: it streams SSE chat, offers one ``spawn_subagent`` tool, runs
each tool call as a second conversation under the solver system text, loops until a reply has no tool calls and
prints an OpenClaude-shaped ``{"type":"result","subtype":"success","result":...,"num_turns":...}`` line. It is
launched exactly like a real harness (subprocess, scrubbed env, stdin = TASK.md), which is what the e2e exercises.
"""
from __future__ import annotations

import sys
from pathlib import Path

from agent_allocation.dynorch.harness.base import (
    HarnessResult, LaunchPlan, WorkDir, default_base_env, harness_env, openclaude_result, write_text,
)
from agent_allocation.dynorch.records import RunSpec
from agent_allocation.paths import PROJECT_ROOT

__all__ = ["FAKE_CLI_PATH", "FakeAdapter"]

FAKE_CLI_PATH: Path = Path(PROJECT_ROOT) / "tests" / "dynorch" / "fake_harness_cli.py"


class FakeAdapter:
    name = "fake"
    in_process = False

    def __init__(self, *, model: str, cli_path: Path = FAKE_CLI_PATH, python: str | None = None,
                 max_turns: int = 8, extra_args: tuple[str, ...] = ()) -> None:
        self.model = model
        self.cli_path = Path(cli_path)
        self.python = python or sys.executable
        self.max_turns = max_turns
        self.extra_args = tuple(extra_args)

    def version(self) -> str:
        return "fake-harness/1"

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None:
        wd = WorkDir(Path(work)).create()
        write_text(wd.system_md, system_text)
        write_text(wd.solver_md, subagent_text)
        write_text(wd.task_md, task_text)

    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan:
        wd = WorkDir(Path(work))
        if not self.cli_path.is_file():
            raise FileNotFoundError(f"fake harness CLI missing: {self.cli_path}")
        env = harness_env(default_base_env(), work=wd.root, proxy_base_url=proxy_base_url, run_key=spec.run_key,
                          extra={"PYTHONDONTWRITEBYTECODE": "1"})
        argv = (self.python, str(self.cli_path), "--model", self.model, "--system-file", str(wd.system_md),
                "--solver-file", str(wd.solver_md), "--max-turns", str(self.max_turns),
                "--tools", ",".join(spec.allowed_tools), *self.extra_args)
        return LaunchPlan(argv=argv, env=env, cwd=wd.ws, stdin=wd.task_md.read_bytes())

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult:
        return openclaude_result(stdout, None, where="fake_harness_cli")
