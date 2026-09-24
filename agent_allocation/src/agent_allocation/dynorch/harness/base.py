"""WP-B: the ``HarnessAdapter`` protocol, launch records, work-dir layout and the environment scrub (contract §6).

Every harness process runs from a fresh per-run directory
``<scratch>/work/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/`` (``WorkDir``) with ``home/`` (redirected
``HOME``), ``state/`` (harness state, e.g. ``CODEX_HOME``), ``ws/`` (an empty cwd, no CLAUDE.md), ``tmp/``
(``TMPDIR``), ``SYSTEM.md`` / ``SOLVER.md`` / ``TASK.md``, ``stdout``, ``stderr`` and ``launch.json`` (plan §6).

``harness_env`` builds the process environment from an allowlist of inherited keys only (``PATH``, locale, ``TZ``,
``USER``, ``LOGNAME``, ``TERM``): nothing else from the job environment reaches a harness, so every
``OPENROUTER_*`` / ``ANTHROPIC_*`` / other ``OPENAI_*`` variable is gone by construction, ``HOME`` /
``TMPDIR`` / ``XDG_*`` resolve under the work dir, and the only credential is ``OPENAI_API_KEY = <run_key>``
with ``OPENAI_BASE_URL = <proxy>/r/<run_key>/v1``. ``check_env`` re-validates a finished environment (used by
the runner before every launch and by the unit tests) and raises on any violation -- the OpenRouter key can never
enter a harness environment silently.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from agent_allocation.dynorch.records import RunSpec

__all__ = [
    "ALLOWED_OPENAI_KEYS", "FORBIDDEN_PREFIXES", "HarnessAdapter", "HarnessResult", "INHERITED_KEYS", "InProcessAdapter",
    "LaunchPlan", "WorkDir", "check_env", "harness_env", "last_json_line", "openclaude_result", "probe_version",
    "write_text",
]

INHERITED_KEYS: tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TERM")
FORBIDDEN_PREFIXES: tuple[str, ...] = ("OPENROUTER_", "ANTHROPIC_")
ALLOWED_OPENAI_KEYS: frozenset[str] = frozenset({"OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"})
#: Keys that must resolve under the run's work directory (never /home).
SCRATCH_KEYS: tuple[str, ...] = ("HOME", "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                                 "XDG_DATA_HOME", "XDG_STATE_HOME", "CLAUDE_CONFIG_DIR", "CODEX_HOME",
                                 "OPENCLAW_STATE_DIR", "OPENCLAW_CONFIG_PATH")
_KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}")


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: Path
    stdin: bytes | None


@dataclass
class HarnessResult:
    exit_code: int | None
    timed_out: bool
    cli_result: str | None
    reported: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HarnessAdapter(Protocol):
    name: str
    in_process: bool

    def version(self) -> str: ...

    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str,
                task_text: str) -> None: ...

    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan: ...

    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult: ...


@runtime_checkable
class InProcessAdapter(Protocol):
    """An adapter with no subprocess of its own (H0, the OpenClaw gateway driver): the runner awaits it."""
    in_process: bool

    async def run_in_process(self, work: Path, spec: RunSpec, proxy_base_url: str, *, wall_seconds: float,
                             ) -> HarnessResult: ...


@dataclass(frozen=True)
class WorkDir:
    """The per-run directory layout (module docstring)."""
    root: Path

    @property
    def home(self) -> Path: return self.root / "home"

    @property
    def state(self) -> Path: return self.root / "state"

    @property
    def ws(self) -> Path: return self.root / "ws"

    @property
    def tmp(self) -> Path: return self.root / "tmp"

    @property
    def system_md(self) -> Path: return self.root / "SYSTEM.md"

    @property
    def solver_md(self) -> Path: return self.root / "SOLVER.md"

    @property
    def task_md(self) -> Path: return self.root / "TASK.md"

    @property
    def stdout(self) -> Path: return self.root / "stdout"

    @property
    def stderr(self) -> Path: return self.root / "stderr"

    @property
    def launch_json(self) -> Path: return self.root / "launch.json"

    def create(self) -> "WorkDir":
        for path in (self.home, self.state, self.ws, self.tmp):
            path.mkdir(parents=True, exist_ok=True)
        return self


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def harness_env(base: Mapping[str, str], *, work: Path, proxy_base_url: str, run_key: str,
                extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The scrubbed process environment (module docstring); ``extra`` is the adapter's own variables."""
    work = Path(work)
    if run_key not in proxy_base_url:
        raise ValueError("proxy_base_url must be the run's own /r/<run_key>/v1 base")
    env: dict[str, str] = {k: base[k] for k in INHERITED_KEYS if k in base}
    env.update({
        "HOME": str(work / "home"), "TMPDIR": str(work / "tmp"), "TMP": str(work / "tmp"), "TEMP": str(work / "tmp"),
        "XDG_CONFIG_HOME": str(work / "home" / ".config"), "XDG_CACHE_HOME": str(work / "home" / ".cache"),
        "XDG_DATA_HOME": str(work / "home" / ".local" / "share"), "XDG_STATE_HOME": str(work / "home" / ".local" / "state"),
        "OPENAI_BASE_URL": proxy_base_url, "OPENAI_API_KEY": run_key,
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
        "NODE_OPTIONS": "--max-old-space-size=1024",
    })
    for key, value in (extra or {}).items():
        env[str(key)] = str(value)
    check_env(env, work=work, run_key=run_key)
    return env


def check_env(env: Mapping[str, str], *, work: Path, run_key: str) -> None:
    """Raise unless ``env`` satisfies the scrub rules (no key material, no forbidden vars, state under ``work``)."""
    work = Path(work).resolve()
    problems: list[str] = []
    for key, value in env.items():
        if any(key.startswith(p) for p in FORBIDDEN_PREFIXES):
            problems.append(f"forbidden variable {key}")
        if key.startswith("OPENAI_") and key not in ALLOWED_OPENAI_KEYS:
            problems.append(f"unexpected OPENAI_* variable {key}")
        if _KEY_PATTERN.search(value):
            problems.append(f"{key} carries an OpenRouter-shaped key")
        if key in SCRATCH_KEYS:
            try:
                Path(value).resolve().relative_to(work)
            except ValueError:
                problems.append(f"{key}={value} is not under the work dir {work}")
    if env.get("OPENAI_API_KEY") != run_key:
        problems.append("OPENAI_API_KEY must equal the run_key")
    if run_key not in env.get("OPENAI_BASE_URL", ""):
        problems.append("OPENAI_BASE_URL must contain the run_key")
    url = env.get("OPENAI_BASE_URL", "")
    if "localhost" in url or "ollama" in url or ":11434" in url:
        problems.append("proxy URL must not contain localhost/ollama/11434 (OpenClaude rewrites those)")
    if problems:
        raise ValueError("harness environment violates the scrub rules: " + "; ".join(problems))


def last_json_line(stdout: bytes, *, where: str) -> dict[str, Any] | None:
    """The last JSON object on stdout (OpenClaude's ``--output-format json`` result line)."""
    last: dict[str, Any] | None = None
    for raw in stdout.decode("utf-8", "replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict):
            last = obj
    return last


def openclaude_result(stdout: bytes, exit_code: int | None, *, where: str) -> HarnessResult:
    """Parse an OpenClaude-shaped ``{"type":"result",...}`` line: ``cli_result`` only when ``is_error`` is false
    (contract §10.5: ``subtype`` stays ``"success"`` on a 402) and the result text is non-empty."""
    result: dict[str, Any] | None = None
    for raw in stdout.decode("utf-8", "replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            result = obj
    if result is None:
        return HarnessResult(exit_code=exit_code, timed_out=False, cli_result=None,
                             reported={"result_line": False, "source": where})
    text = result.get("result")
    is_error = result.get("is_error")
    ok = is_error is False and result.get("subtype") == "success" and isinstance(text, str) and text.strip() != ""
    reported = {k: result.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd", "usage", "modelUsage",
                                            "stop_reason", "session_id", "duration_ms", "duration_api_ms")}
    reported["result_line"] = True
    reported["source"] = where
    if not ok and isinstance(text, str):
        reported["error_text"] = text[:2000]
    return HarnessResult(exit_code=exit_code, timed_out=False, cli_result=text if ok else None, reported=reported)


def redacted_env(env: Mapping[str, str]) -> dict[str, str]:
    """Environment as journaled in ``launch.json``: any key-shaped value is masked (belt and braces)."""
    return {k: _KEY_PATTERN.sub("[REDACTED_KEY]", v) for k, v in env.items()}


def default_base_env() -> Mapping[str, str]:
    return os.environ


def probe_version(argv: list[str], *, extra_env: Mapping[str, str] | None = None, timeout: float = 60.0) -> str:
    """Run ``<binary> --version`` with HOME and every harness state dir pointed at a throwaway scratch directory
    (``TMPDIR``, never ``/home``: Codex and OpenClaude both write caches into their home on start-up)."""
    import shutil as _shutil
    import subprocess
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="dynorch-version-"))
    try:
        home = scratch / "home"
        home.mkdir()
        env = {"PATH": default_base_env().get("PATH", ""), "HOME": str(home), "TMPDIR": str(scratch),
               "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache"),
               "CLAUDE_CONFIG_DIR": str(home / "cfg"), "CODEX_HOME": str(home / "codex"),
               "OPENCLAW_STATE_DIR": str(home / "openclaw"), "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
               "DISABLE_ERROR_REPORTING": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        (home / "codex").mkdir()
        env.update(extra_env or {})
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"error: {type(exc).__name__}"
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else "unknown"
    finally:
        _shutil.rmtree(scratch, ignore_errors=True)
