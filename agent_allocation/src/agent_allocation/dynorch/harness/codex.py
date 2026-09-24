"""WP-B: H3 -- OpenAI Codex CLI 0.156.1 adapter (contract §6; plan §6 "Codex (H3)"; Codex deep dive §4, §6, §7).

Codex is installed once into ``<scratch>/npm-prefix`` (``bin/codex``). Per run ``CODEX_HOME = $WORK/state/codex``
is pre-created (Codex refuses a missing dir) and holds the plan's ``config.toml`` verbatim: the ``meter`` provider at
the run's proxy URL with ``wire_api = "responses"`` (the only wire Codex speaks), ``model_instructions_file`` = the
study system prompt (replacing Codex's 5k-token coding prompt), every execution feature off (``shell_tool``,
``unified_exec``, ``view_image``, ``code_mode``, ``apps``, ``plugins``, ``hooks``, ``image_generation``,
``skill_search``, and ``goals``), ``web_search = "disabled"``, ``multi_agent = true`` / ``multi_agent_v2 = false``,
``[tools.experimental_request_user_input] enabled = false`` and ``[tools.update_plan] enabled = true`` (contract
§10.3: with these the offered tools are exactly ``update_plan`` + the ``multi_agent_v1`` namespace; bool-valued
``tools.*`` keys are rejected by ``--strict-config``), ``[agents] enabled = true,
max_concurrent_threads_per_session = 4, max_depth = 1`` (a spawning-disabled arm sets ``enabled = false``),
``[shell_environment_policy] inherit = "none"``. Command:

    timeout --kill-after=30s ${WALL}s codex exec --json --ephemeral --strict-config --skip-git-repo-check
      -C $WORK/ws -s read-only -o $WORK/final.txt -   < TASK.md   > stdout (events.jsonl)

``WALL = base + per_rung * rung`` (the same formula as the runner's wall timeout). ``parse_result`` reads the
``--json`` event stream: the final message is the last ``item.completed`` whose item is an ``agent_message`` (also
written to ``final.txt``), usage the last ``turn.completed.usage``, sub-agent activity the ``collab_tool_call``
items; ``turn.failed`` / ``error`` events are reported. All of it is a cross-check: the delivered answer comes from
the proxy (plan §2 B7).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from agent_allocation.dynorch.harness.base import (
    HarnessResult, LaunchPlan, WorkDir, default_base_env, harness_env, probe_version, write_text,
)
from agent_allocation.dynorch.records import RunSpec

__all__ = ["CodexAdapter", "config_toml"]

_FEATURES_OFF = ("shell_tool", "unified_exec", "view_image", "code_mode", "apps", "plugins", "hooks",
                 "image_generation", "skill_search")


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def config_toml(*, model: str, proxy_base_url: str, instructions_file: Path, agents_enabled: bool = True) -> str:
    """The plan §6 ``config.toml`` for one run."""
    lines = [
        f"model = {_toml_str(model)}",
        'model_provider = "meter"',
        "model_context_window = 131072",
        'model_reasoning_summary = "none"',
        f"model_instructions_file = {_toml_str(str(instructions_file))}",
        "show_raw_agent_reasoning = false",
        'approval_policy = "never"',
        'sandbox_mode = "read-only"',
        'web_search = "disabled"',
        "check_for_update_on_startup = false",
        "",
        "[model_providers.meter]",
        'name = "Local metering proxy"',
        f"base_url = {_toml_str(proxy_base_url)}",
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
        "request_max_retries = 2",
        "stream_max_retries = 2",
        "stream_idle_timeout_ms = 1800000",
        "",
        "[features]",
        *[f"{name} = false" for name in _FEATURES_OFF],
        "goals = false",
        "multi_agent = true",
        "multi_agent_v2 = false",
        "",
        "[tools.experimental_request_user_input]",
        "enabled = false",
        "",
        "[tools.update_plan]",
        "enabled = true",
        "",
        "[agents]",
        f"enabled = {'true' if agents_enabled else 'false'}",
        "max_concurrent_threads_per_session = 4",
        "max_depth = 1",
        "",
        "[shell_environment_policy]",
        'inherit = "none"',
        "",
    ]
    return "\n".join(lines)


class CodexAdapter:
    name = "codex"
    in_process = False
    _version_cache: dict[str, str] = {}

    def __init__(self, *, model: str, prefix: str, wall_base: int = 600, wall_per_rung: int = 120,
                 binary: str | None = None, sigkill_after: int = 30) -> None:
        self.model = model
        self.prefix = Path(prefix)
        self.binary = binary or str(self.prefix / "bin" / "codex")
        self.wall_base = int(wall_base)
        self.wall_per_rung = int(wall_per_rung)
        self.sigkill_after = int(sigkill_after)

    def wall_seconds(self, spec: RunSpec) -> int:
        return self.wall_base + self.wall_per_rung * spec.rung

    @staticmethod
    def codex_home(work: Path) -> Path:
        return WorkDir(Path(work)).state / "codex"

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None:
        wd = WorkDir(Path(work)).create()
        write_text(wd.system_md, system_text)
        write_text(wd.solver_md, subagent_text)
        write_text(wd.task_md, task_text)
        home = self.codex_home(wd.root)
        home.mkdir(parents=True, exist_ok=True)
        agents_enabled = bool(spec.spawn_tools)
        write_text(home / "config.toml", config_toml(model=self.model, proxy_base_url=proxy_base_url,
                                                     instructions_file=wd.system_md, agents_enabled=agents_enabled))

    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan:
        wd = WorkDir(Path(work))
        home = self.codex_home(wd.root)
        if not home.is_dir():
            raise FileNotFoundError(f"CODEX_HOME {home} must be pre-created by prepare()")
        wall = self.wall_seconds(spec)
        argv = ("timeout", f"--kill-after={self.sigkill_after}s", f"{wall}s", self.binary, "exec", "--json",
                "--ephemeral", "--strict-config", "--skip-git-repo-check", "-C", str(wd.ws), "-s", "read-only",
                "-o", str(wd.root / "final.txt"), "-")
        env = harness_env(default_base_env(), work=wd.root, proxy_base_url=proxy_base_url, run_key=spec.run_key,
                          extra={"CODEX_HOME": str(home)})
        return LaunchPlan(argv=argv, env=env, cwd=wd.ws, stdin=wd.task_md.read_bytes())

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult:
        wd = WorkDir(Path(work))
        events: list[dict[str, Any]] = []
        malformed = 0
        for raw in stdout.decode("utf-8", "replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                events.append(obj)
        last_message: str | None = None
        usage: dict[str, Any] | None = None
        collab: list[dict[str, Any]] = []
        failures: list[str] = []
        exec_items = 0
        for event in events:
            kind = event.get("type")
            item = event.get("item") if isinstance(event.get("item"), dict) else None
            if kind == "item.completed" and item is not None:
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    last_message = item["text"]
                elif item.get("type") == "collab_tool_call":
                    collab.append({k: item.get(k) for k in ("tool", "status", "receiver_thread_ids")})
                elif item.get("type") in ("command_execution", "file_change"):
                    exec_items += 1
            elif kind == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            elif kind in ("turn.failed", "error"):
                failures.append(json.dumps(event.get("error") or event.get("message") or event)[:500])
        final_file = wd.root / "final.txt"
        file_text = final_file.read_text("utf-8") if final_file.is_file() else None
        reported: dict[str, Any] = {
            "events": len(events), "malformed_lines": malformed, "usage": usage, "collab_tool_calls": collab,
            "spawns": sum(1 for c in collab if c.get("tool") == "spawn_agent"), "failures": failures,
            "execution_items": exec_items, "final_file_present": file_text is not None,
            "final_file_matches_last_message": (file_text.strip() == last_message.strip())
            if (file_text is not None and last_message is not None) else None,
            "num_turns": sum(1 for e in events if e.get("type") == "turn.completed"),
        }
        text = last_message if last_message is not None else file_text
        ok = text is not None and text.strip() != "" and not failures
        return HarnessResult(exit_code=None, timed_out=False, cli_result=text if ok else None, reported=reported)

    def version(self) -> str:
        if self.binary in self._version_cache:
            return self._version_cache[self.binary]
        path = self.binary if Path(self.binary).is_file() else shutil.which(self.binary)
        version = "absent" if path is None else probe_version([path, "--version"])
        self._version_cache[self.binary] = version
        return version
