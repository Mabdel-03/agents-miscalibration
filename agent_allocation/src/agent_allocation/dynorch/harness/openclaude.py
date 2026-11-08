"""WP-B: H1 -- OpenClaude 0.13.0 adapter (contract §6; plan §6 "OpenClaude (H1)", flags and env verbatim).

``prepare`` writes ``SYSTEM.md`` (the study system prompt, passed whole through ``--system-prompt``, which replaces
OpenClaude's default prompt entirely), ``TASK.md`` (stdin), ``agents.json`` (the custom ``solver`` sub-agent: fresh
context, prompt = ``prompts/subagent_dynorch.txt``, ``tools: []``) and ``$CLAUDE_CONFIG_DIR/.openclaude.json``
(``CLAUDE_CONFIG_DIR = $WORK/home/cfg``; onboarding + bypass flags accepted, ``agentModels`` / ``agentRouting``
pinning every role to the model at the run's proxy URL). ``build_command`` is the plan's command line:

    openclaude -p --output-format json --no-session-persistence --provider openai --model qwen/qwen3.8-27b
      --system-prompt <SYSTEM.md> --tools "Agent" --allowedTools "Agent" --disallowedTools <the plan's list>
      --permission-mode bypassPermissions --agents <agents.json> --setting-sources "" --strict-mcp-config
      --disable-slash-commands  < TASK.md

NEVER ``--bare`` (it removes the Agent tool) and NEVER ``--max-budget-usd`` (it prices unknown models at
$5/$25 per M) -- ``build_command`` raises if either would appear. Env (on top of the scrub): ``CLAUDE_CONFIG_DIR``,
``CLAUDE_CODE_USE_OPENAI=1``, ``OPENAI_MODEL``, ``CLAUDE_CODE_SUBAGENT_MODEL``, ``CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1``,
``CLAUDE_CODE_AGENT_LIST_IN_MESSAGES=0``, ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1``, ``DISABLE_TELEMETRY=1``,
``DISABLE_AUTOUPDATER=1``, ``DISABLE_ERROR_REPORTING=1``, ``DISABLE_COMPACT=1``, ``CLAUDE_CODE_MAX_RETRIES=1``,
``API_TIMEOUT_MS=3600000``, ``CLAUDE_CODE_MAX_OUTPUT_TOKENS=32000``, ``CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1``
(contract §10.6). A spawning-disabled arm (E2, ``allowed_tools`` empty) gets ``--tools ""``. The result line's
``result`` is a cross-check only (plan §2 B7); success is ``is_error == false`` (``subtype`` stays ``"success"`` even
on a 402, contract §10.5) and ``total_cost_usd`` comes from an unknown-model price table and is ignored.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent_allocation.dynorch.harness.base import (
    HarnessResult, LaunchPlan, WorkDir, default_base_env, harness_env, openclaude_result, probe_version, write_text,
)
from agent_allocation.dynorch.records import RunSpec

__all__ = ["DISALLOWED_TOOLS", "FORBIDDEN_FLAGS", "OpenclaudeAdapter", "SOLVER_DESCRIPTION"]

DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "Edit", "MultiEdit", "Write", "Read", "NotebookEdit", "Glob", "Grep", "WebSearch", "WebFetch",
    "TodoWrite", "TaskCreate", "TaskUpdate", "KillShell", "BashOutput", "SendMessage", "TaskStop", "ToolSearchTool",
    "Skill",
)
FORBIDDEN_FLAGS: tuple[str, ...] = ("--bare", "--max-budget-usd")
SOLVER_DESCRIPTION = ("Independent solver. Give it the FULL problem statement and a precise sub-task; it starts with "
                      "no context and returns its result inside <final>...</final>.")
FIXED_ENV: dict[str, str] = {
    "CLAUDE_CODE_USE_OPENAI": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_AGENT_LIST_IN_MESSAGES": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_COMPACT": "1",
    "CLAUDE_CODE_MAX_RETRIES": "1",
    "API_TIMEOUT_MS": "3600000",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
    "CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED": "1",      # contract §10.6: silences the spurious Codex-profile warning
}


class OpenclaudeAdapter:
    name = "openclaude"
    in_process = False
    _version_cache: dict[str, str] = {}

    def __init__(self, *, model: str, binary: str = "openclaude") -> None:
        self.model = model
        self.binary = binary

    # -- files -------------------------------------------------------------------------------------------
    @staticmethod
    def config_dir(work: Path) -> Path:
        return WorkDir(Path(work)).home / "cfg"

    def openclaude_json(self, spec: RunSpec, proxy_base_url: str) -> dict:
        return {
            "hasCompletedOnboarding": True,
            "bypassPermissionsModeAccepted": True,
            "agentModels": {self.model: {"base_url": proxy_base_url, "api_key": spec.run_key}},
            "agentRouting": {role: self.model for role in ("default", "solver", "general-purpose", "Explore", "Plan")},
        }

    @staticmethod
    def agents_json(subagent_text: str) -> dict:
        return {"solver": {"description": SOLVER_DESCRIPTION, "prompt": subagent_text, "tools": []}}

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None:
        wd = WorkDir(Path(work)).create()
        write_text(wd.system_md, system_text)
        write_text(wd.solver_md, subagent_text)
        write_text(wd.task_md, task_text)
        write_text(wd.root / "agents.json", json.dumps(self.agents_json(subagent_text), ensure_ascii=False, indent=1))
        write_text(self.config_dir(wd.root) / ".openclaude.json",
                   json.dumps(self.openclaude_json(spec, proxy_base_url), indent=1))

    # -- command -----------------------------------------------------------------------------------------
    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan:
        wd = WorkDir(Path(work))
        # plan §6 passes "$(cat SYSTEM.md)": command substitution drops the trailing newline, and Gate 0's check is
        # messages[0] == OPENCLAUDE_PREFIX + system_text.rstrip("\n") (contract §10.2, captures/openclaude/a)
        system_text = wd.system_md.read_text("utf-8").rstrip("\n")
        agents = (wd.root / "agents.json").read_text("utf-8")
        tools = ",".join(spec.allowed_tools)
        argv: list[str] = [
            self.binary, "-p", "--output-format", "json", "--no-session-persistence", "--provider", "openai",
            "--model", self.model, "--system-prompt", system_text, "--tools", tools, "--allowedTools", tools,
            "--disallowedTools", *DISALLOWED_TOOLS, "--permission-mode", "bypassPermissions", "--agents", agents,
            "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
        ]
        for flag in FORBIDDEN_FLAGS:
            if any(a == flag or a.startswith(flag + "=") for a in argv):
                raise ValueError(f"{flag} must never be passed to openclaude (plan §6)")
        env = harness_env(default_base_env(), work=wd.root, proxy_base_url=proxy_base_url, run_key=spec.run_key,
                          extra={**FIXED_ENV, "CLAUDE_CONFIG_DIR": str(self.config_dir(wd.root)),
                                 "OPENAI_MODEL": self.model, "CLAUDE_CODE_SUBAGENT_MODEL": self.model})
        return LaunchPlan(argv=tuple(argv), env=env, cwd=wd.ws, stdin=wd.task_md.read_bytes())

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult:
        return openclaude_result(stdout, None, where="openclaude")

    def version(self) -> str:
        if self.binary in self._version_cache:
            return self._version_cache[self.binary]
        path = shutil.which(self.binary)
        version = "absent" if path is None else probe_version([path, "--version"])
        self._version_cache[self.binary] = version
        return version
