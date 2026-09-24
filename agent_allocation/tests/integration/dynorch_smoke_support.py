"""WP-E: the shared stand for the Gate 0 real-CLI smoke tests (plan §9 Gate 0; contract §3, §5, §6, §10).

One ``SmokeStack`` is exactly what ``dynorch-run`` assembles (``runner.build_stack``: settings -> ledger ->
journal / store -> ``Upstream`` -> the REAL ``ProxyCore``) bound through the real ``httpd.ProxyServer`` on
``127.0.0.1:0``, with two substitutions and nothing else: the upstream is ``api/fake.py``'s transport (scripted
replies, no network, no key, $0) and the harness adapters are the real ``OpenclaudeAdapter`` / ``CodexAdapter``
launched by the real ``Runner`` from a work dir under ``tmp_path`` (never the study scratch root, never ``/home``).

``RunScript`` scripts the fake upstream per conversation class of one run: the core tags every ``/completions``
call with ``coordinates = {node_id: run_key, conversation_class, phase, call_no}`` (``core._coordinates``), so a
hook can answer the n-th ROOT think call, the n-th SOLVER think call and the ANSWER / TERMINAL continuations
independently of the arrival order the harness produces (a Codex child and its parent race for the run's FIFO
lock). ``TimedReplyLog`` stamps every served reply / error with a monotonic time so a test can bound "the run ended
within N seconds of the 402".

``record_case`` merges one case's checks into ``$DYNORCH_GATE0_SMOKE_OUT`` (``{passed, failures, cases}``, the file
``dynorch-gates --gate 0`` reads): several test modules write the same report, so the writer re-reads the file and
re-derives ``passed`` / ``failures`` from every case recorded so far.
"""
from __future__ import annotations

import asyncio
import fcntl
import gzip
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agent_allocation.api import fake as fakeapi
from agent_allocation.api.fake import FakeCall, FakeReply, FakeScript
from agent_allocation.api.forced import ForcedClient, ForcedSettings
from agent_allocation.config import load_config
from agent_allocation.dynorch import cli as dcli
from agent_allocation.dynorch.harness.base import HarnessAdapter
from agent_allocation.dynorch.httpd import ProxyServer, ReplyLog
from agent_allocation.dynorch.plan import BaseInputs, plan_runs
from agent_allocation.dynorch.records import ChatReply, ProxyError, RunRecord, RunSpec
from agent_allocation.dynorch.runner import Runner, RunnerSettings, build_stack

PROJECT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT / "configs" / "dynorch_v1.json"
CFG: dict[str, Any] = load_config(CFG_PATH)
SYSTEM_TEXT = (PROJECT / CFG["dynorch"]["prompt_files"]["system"]).read_text("utf-8")
SUBAGENT_TEXT = (PROJECT / CFG["dynorch"]["prompt_files"]["subagent"]).read_text("utf-8")
OPENCLAUDE_PREFIX = str(CFG["dynorch"]["openclaude_system_prefix"])
KEEPALIVE_SECONDS = float(CFG["dynorch"]["proxy"]["keepalive_seconds"])
HOME_EXCLUDES = (".claude", ".cursor-server", ".vscode-server", ".cache/uv")
REPORT_ENV = "DYNORCH_GATE0_SMOKE_OUT"
_AGENT_ID = re.compile(r'\\?"agent_id\\?":\\?"([0-9a-f\-]{16,})\\?"')

ReplyMaker = FakeReply | Callable[[FakeCall], FakeReply]


# ---------------------------------------------------------------------------------------------------------------------
# Scripting the fake upstream per conversation class
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class RunScript:
    """Replies for one run: ``root[i]`` answers the i-th ROOT think call, ``solver[i]`` the i-th SOLVER think call
    (the last entry repeats; an empty list falls through to the fake's default), ``answer`` answers every ANSWER /
    TERMINAL continuation (the forced close and the B4 terminal call; None -> the fake's default content). An entry
    may be a callable taking the ``FakeCall`` (its ``body["prompt"]`` is the rendered prompt) for replies that
    depend on what the harness sent, e.g. a ``wait_agent`` on the agent id Codex minted."""
    root: list[ReplyMaker] = field(default_factory=list)
    solver: list[ReplyMaker] = field(default_factory=list)
    other: list[ReplyMaker] = field(default_factory=list)
    answer: ReplyMaker | None = None
    seen: dict[str, int] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def pick(self, call: FakeCall) -> FakeReply | None:
        coordinates = call.coordinates
        phase = str(coordinates.get("phase"))
        cls = str(coordinates.get("conversation_class"))
        prompt = str((call.body or {}).get("prompt", ""))
        self.calls.append({"phase": phase, "class": cls, "call_no": coordinates.get("call_no"),
                           "max_tokens": (call.body or {}).get("max_tokens"),
                           # the pinned template renders tool schemas under a ``<tools>`` block (plan B3: the grace
                           # continuation must carry none, contract §10.28)
                           "prompt_has_tools": "<tools>" in prompt})
        if phase in ("answer", "terminal"):
            return _make(self.answer, call) if self.answer is not None else None
        if phase != "think":
            raise AssertionError(f"unexpected upstream phase {phase!r} (coordinates {dict(coordinates)})")
        entries = {"root": self.root, "solver": self.solver, "other": self.other}.get(cls)
        if entries is None:
            raise AssertionError(f"unexpected conversation class {cls!r}")
        n = self.seen.get(cls, 0)
        self.seen[cls] = n + 1
        if not entries:
            return None
        return _make(entries[min(n, len(entries) - 1)], call)

    def think_calls(self, cls: str | None = None) -> int:
        return sum(1 for c in self.calls if c["phase"] == "think" and (cls is None or c["class"] == cls))

    def phase_calls(self, phase: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["phase"] == phase]


def _make(entry: ReplyMaker, call: FakeCall) -> FakeReply:
    return entry(call) if callable(entry) else entry


def script_hook(scripts: dict[str, RunScript]) -> Callable[[FakeCall], FakeReply | None]:
    """The ``FakeScript.hook`` dispatching on ``coordinates["node_id"]`` (= the run_key the core tags)."""
    def hook(call: FakeCall) -> FakeReply | None:
        node = call.coordinates.get("node_id")
        script = scripts.get(str(node)) if node is not None else None
        if script is None:
            return None
        return script.pick(call)
    return hook


def agent_id_in_prompt(call: FakeCall) -> str:
    """The agent id Codex returned for a ``spawn_agent`` (``{"agent_id":"...","nickname":"..."}`` in the rendered
    tool response); raises when the prompt carries none (the test then fails loudly, never waits on nothing)."""
    prompt = str((call.body or {}).get("prompt", ""))
    match = _AGENT_ID.search(prompt)
    if match is None:
        raise AssertionError("no agent_id in the rendered prompt: the spawn_agent round trip did not happen")
    return match.group(1)


# ---------------------------------------------------------------------------------------------------------------------
# The stand
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class TimedEvent:
    at: float
    kind: str            # reply | error
    status: int
    code: str | None
    conversation_class: str | None


class TimedReplyLog(ReplyLog):
    """``ReplyLog`` plus a monotonic timestamp per served reply / error (for "ends within N s of the 402")."""

    def __init__(self) -> None:
        super().__init__()
        self.timed: dict[str, list[TimedEvent]] = {}

    def record(self, run_key: str, reply: ChatReply, *, delivered: bool = True) -> None:
        super().record(run_key, reply, delivered=delivered)
        self.timed.setdefault(run_key, []).append(TimedEvent(time.monotonic(), "reply", 200, None,
                                                             reply.call.conversation_class))

    def record_error(self, run_key: str, err: ProxyError, *, wire: str) -> None:
        super().record_error(run_key, err, wire=wire)
        self.timed.setdefault(run_key, []).append(TimedEvent(time.monotonic(), "error", err.status, err.code, None))

    def first_error(self, run_key: str, status: int) -> TimedEvent | None:
        for event in self.timed.get(run_key, []):
            if event.kind == "error" and event.status == status:
                return event
        return None

    def events_after(self, run_key: str, at: float) -> list[TimedEvent]:
        return [e for e in self.timed.get(run_key, []) if e.at > at]


@dataclass
class Probe:
    """One request the stand sent to the proxy for a run AFTER its harness exited and before ``close_run``."""
    sent_at: float       # monotonic, before the request left (events at or after it are the probe's own)
    at: float
    status: int
    code: str | None
    upstream_calls_before: int
    upstream_calls_after: int


class ProbingRunner(Runner):
    """The real ``Runner`` plus one extra step between the harness's exit and ``close_run``: a plain chat request on
    the run's ROOT system text is sent through the real ``ProxyServer`` (``probe_after_exit``), so a case can pin
    "any further call is refused 402" over the wire instead of relying on the harness to try one (plan B4: after the
    grace every call is refused; contract §10.28: a forced close answers with text, so the harness never asks)."""

    stack: "SmokeStack"

    def _harness_gone(self, spec: RunSpec) -> None:
        # deferred to _await_settled (which the runner calls right after): the probe must go out after the harness
        # exited but BEFORE the core is told so (contract §10.33: from then on every call is ``run_closed``), so it
        # pins the BUDGET's verdict on one more call (B4: after the grace / the TERMINAL every call is 402)
        if not self.stack.probe_after_exit:
            super()._harness_gone(spec)

    async def _await_settled(self, spec: RunSpec) -> None:
        if self.stack.probe_after_exit:
            await self.stack.probe(spec)
            super()._harness_gone(spec)
        await super()._await_settled(spec)


def load_base() -> BaseInputs | None:
    root = Path(CFG["base_run_root"])
    if not (root / "manifests" / "panels.json").is_file():
        return None
    return BaseInputs.load(root, model=CFG["primary_model"])


def dev_item(base: BaseInputs, benchmark: str, index: int = 0) -> str:
    return base.panels[benchmark].dev_item_ids[index]


class SmokeStack:
    """The real stack (module docstring) around ``tmp``: ``run_root`` = ``tmp/run_root``, scratch = ``tmp/scratch``.
    ``adapters`` maps harness names to the REAL adapters under test; ``server_factory`` lets a test substitute a
    fault-injecting ``ProxyServer`` subclass (the Codex missing-``response.completed`` case)."""

    def __init__(self, tmp: Path, *, adapters: dict[str, HarnessAdapter], scripts: dict[str, RunScript] | None = None,
                 concurrency: int = 2, keepalive_seconds: float = KEEPALIVE_SECONDS,
                 sigterm_after_exhaustion: float = 60.0, sigkill_after: float = 30.0, poll_seconds: float = 0.5,
                 wall_base: int | None = None, wall_per_rung: int | None = None,
                 server_factory: Callable[..., ProxyServer] = ProxyServer, read_timeout: float | None = None,
                 probe_after_exit: bool = False) -> None:
        self.tmp = Path(tmp)
        self.run_root = self.tmp / "run_root"
        self.scratch = self.tmp / "scratch"
        self.scripts: dict[str, RunScript] = scripts if scripts is not None else {}
        self.script = FakeScript(hook=script_hook(self.scripts))
        self.reply_log = TimedReplyLog()
        forced_cfg = dict(CFG)
        forced_cfg["provider"] = {**CFG["provider"], "endpoint_tag": "parasail/fp8"}
        self.client = ForcedClient(settings=ForcedSettings.from_config(forced_cfg), api_key=None,
                                   transport=fakeapi.fake_openrouter_transport(self.script),
                                   answer_backoff_base_seconds=0.0,
                                   read_timeout=float(read_timeout if read_timeout is not None
                                                      else CFG["scheduler"]["read_timeout_seconds"]))
        self.locks = {"parasail": fakeapi.fake_forced_endpoint_lock(tag="parasail/fp8"),
                      "mancer": fakeapi.fake_forced_endpoint_lock(tag="mancer/fp8")}
        settings = RunnerSettings.from_config(CFG, run_root=self.run_root, project_root=PROJECT,
                                              scratch_root=self.scratch, concurrency=concurrency)
        overrides: dict[str, Any] = {"poll_seconds": poll_seconds, "sigkill_after": sigkill_after,
                                     "sigterm_after_exhaustion": sigterm_after_exhaustion}
        if wall_base is not None:
            overrides["wall_base"] = int(wall_base)
        if wall_per_rung is not None:
            overrides["wall_per_rung"] = int(wall_per_rung)
        self.settings = RunnerSettings(**{**settings.__dict__, **overrides})
        self.adapters = dict(adapters)
        self.keepalive_seconds = keepalive_seconds
        self.server_factory = server_factory
        self.probe_after_exit = probe_after_exit
        self.probes: dict[str, Probe] = {}
        self.stack: Any = None
        self.server: ProxyServer | None = None
        self.runner: Runner | None = None

    async def __aenter__(self) -> "SmokeStack":
        self.stack = await build_stack(CFG, self.run_root, client=self.client, locks=self.locks, ceiling_usd="95")
        self.core, self.store, self.journal = self.stack.core, self.stack.store, self.stack.journal
        self.server = self.server_factory(self.core, keepalive_seconds=self.keepalive_seconds, reply_log=self.reply_log)
        await self.server.start()
        self.runner = ProbingRunner(self.core, self.adapters, self.settings, proxy_url=self.server.base_url,
                                    store=self.store, inputs=None, reply_log=self.reply_log, cfg=CFG,
                                    journal=self.journal, wait_for_run=self.server.wait_for_run)
        self.runner.stack = self
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.settle_upstream()
        assert self.server is not None
        await self.server.close()
        await self.client.aclose()
        await self.stack.aclose()

    async def settle_upstream(self, timeout: float = 120.0) -> None:
        """Wait for every fake upstream call still "generating" (a harness killed mid-call leaves the shielded core
        call running; B2: it is finished and settled, never abandoned) before the ledger journal is released."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.script.in_flight == 0 and self.stack.ledger.snapshot().in_flight == 0:
                return
            await asyncio.sleep(0.2)
        raise AssertionError(f"upstream calls still in flight after {timeout:.0f} s")

    async def probe(self, spec: RunSpec) -> Probe:
        """POST one non-streaming chat request on the run's root system text (as the harness sent it: the first
        journaled request's ``messages[0]``) to ``/r/<run_key>/v1/chat/completions`` and record the answer. The
        request is classified ``root`` (same system text); what it gets is the proxy's verdict on "one more call"."""
        import httpx
        docs = self.journaled(spec)
        system = docs[0]["harness_request"]["messages"][0]["content"] if docs else SYSTEM_TEXT
        body = {"model": CFG["primary_model"], "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": "One more question after the answer: are you sure?"}]}
        before = len(self.script.calls)            # every upstream request the fake received (all routes)
        assert self.server is not None
        sent_at = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.post(f"{self.server.run_base_url(spec.run_key)}/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {spec.run_key}"})
        try:
            code = (response.json().get("error") or {}).get("code")
        except ValueError:
            code = None
        probe = Probe(sent_at=sent_at, at=time.monotonic(), status=response.status_code, code=code, upstream_calls_before=before,
                      upstream_calls_after=len(self.script.calls))
        self.probes[spec.run_key] = probe
        return probe

    def specs(self, base: BaseInputs, *, arm: str, rung: int, items: Sequence[str], stage: str = "main") -> list[RunSpec]:
        harness = str(CFG["dynorch"]["arms"][arm]["harness"])
        versions = {harness: self.adapters[harness].version()}
        return plan_runs(CFG, stage=stage, arms=[arm], rungs=[rung], repeats=[1], items=list(items), base=base,
                         proxy_version=self.core.proxy_version, harness_versions=versions)

    async def run(self, base: BaseInputs, specs: Sequence[RunSpec], *, stage: str = "main") -> dict[str, RunRecord]:
        assert self.runner is not None
        self.runner.inputs = base
        summary = await self.runner.run_specs(stage, list(specs))
        return dict(summary.records)

    # -- what the proxy saw ------------------------------------------------------------------------------------
    def journaled(self, spec: RunSpec) -> list[dict[str, Any]]:
        """The run's raw call journal in call order (``harness_request`` = the request as the proxy parsed it)."""
        return sorted(self.journal.scan(spec), key=lambda d: int(d["call_no"]))

    def work_dir(self, spec: RunSpec) -> Path:
        return self.settings.work_dir(spec)

    def stdout_json_lines(self, spec: RunSpec) -> list[dict[str, Any]]:
        path = self.work_dir(spec) / "stdout"
        out: list[dict[str, Any]] = []
        if not path.is_file():
            return out
        for raw in path.read_text("utf-8", "replace").splitlines():
            raw = raw.strip()
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out

    def result_line(self, spec: RunSpec) -> dict[str, Any] | None:
        lines = [o for o in self.stdout_json_lines(spec) if o.get("type") == "result"]
        return lines[-1] if lines else None


# ---------------------------------------------------------------------------------------------------------------------
# Checks shared by both harnesses
# ---------------------------------------------------------------------------------------------------------------------

def budget_usd_hits(stack: SmokeStack, spec: RunSpec) -> list[str]:
    """Where the string ``budget_usd`` appears: journaled requests, the launch record, stdout, stderr (must be [])."""
    hits: list[str] = []
    for doc in stack.journaled(spec):
        if "budget_usd" in json.dumps(doc["harness_request"]):
            hits.append(f"call_{doc['call_no']}.harness_request")
    work = stack.work_dir(spec)
    for name in ("launch.json", "stdout", "stderr"):
        path = work / name
        if path.is_file() and "budget_usd" in path.read_text("utf-8", "replace"):
            hits.append(name)
    return hits


def key_material_hits(stack: SmokeStack, *, max_bytes: int = 4_000_000) -> list[str]:
    """Key-shaped strings in the run root (``cli.key_hits_in_run_tree``, gzip journals included) and in the text
    files of the scratch tree. Like ``doctor.key_shaped_strings`` the scratch scan skips symlinks, files above
    ``max_bytes`` and binaries (a NUL byte in the first 8 KB): Codex leaves ``state/codex/tmp/arg0/*`` symlinks to
    its own 100 MB binary behind on a failed exit, and that binary carries a ``sk-or-OPENAI...`` substring the key
    regex matches (WP-E Gate 0 smoke, 2026-09-23) -- not key material, and never a file a harness could exfiltrate
    a key into."""
    hits = dcli.key_hits_in_run_tree(stack.run_root)
    pattern = re.compile(r"sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{8,}")
    for path in sorted(stack.scratch.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.suffix not in ("", ".json", ".jsonl", ".md", ".txt", ".toml"):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        if pattern.search(data.decode("utf-8", "replace")):
            hits.append(str(path.relative_to(stack.scratch)))
    return hits


_HARNESS_CMD = re.compile(r"(^|/)(codex|openclaude|claude)(\s|$)")


def _descendants_of(pid: int, parents: dict[int, int]) -> set[int]:
    out: set[int] = set()
    frontier = [pid]
    while frontier:
        current = frontier.pop()
        for child, parent in parents.items():
            if parent == current and child not in out:
                out.add(child)
                frontier.append(child)
    return out


def foreign_harness_state_dirs() -> dict[str, str]:
    """``{state dir under $HOME: "pid cmd"}`` of harness processes (codex / openclaude / claude) alive on this node
    that are NOT descendants of this test process -- e.g. the Codex app-server a cursor-server keeps running with
    ``CODEX_HOME=$HOME/.codex`` (observed on 2026-09-23: it writes ``logs_2.sqlite*`` / ``models_cache.json`` there
    continuously) and the Claude Code sessions driving this worktree (``~/.claude.json``). Their state paths are
    read from ``/proc/<pid>/environ`` (``CODEX_HOME`` / ``CLAUDE_CONFIG_DIR``, else the tool's default under that
    process's ``HOME``). A write there during a run is attributed to them, not to the harness under test; the
    attribution is reported (pid + command), never silent. The harness processes under test are descendants of
    this process with HOME / CODEX_HOME / CLAUDE_CONFIG_DIR under the work dir (``harness.base.check_env``)."""
    home = Path.home().resolve()
    # ``ps`` truncates its lines to $COLUMNS (pytest exports it): -ww and no COLUMNS keep the full command line
    env = {k: v for k, v in os.environ.items() if k != "COLUMNS"}
    try:
        out = subprocess.run(["ps", "-ww", "-eo", "pid=,ppid=,args=", "--user", str(os.getuid())],
                             capture_output=True, text=True, timeout=30, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    parents: dict[int, int] = {}
    cmds: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        parents[pid], cmds[pid] = ppid, parts[2]
    mine = _descendants_of(os.getpid(), parents) | {os.getpid()}
    dirs: dict[str, str] = {}
    for pid, cmd in cmds.items():
        if pid in mine or not _HARNESS_CMD.search(cmd.split()[0] if cmd else ""):
            continue
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            continue
        env = dict(item.split("=", 1) for item in raw.decode("utf-8", "replace").split("\0") if "=" in item)
        proc_home = Path(env.get("HOME") or home)
        candidates = []
        if "codex" in cmd:
            candidates.append(Path(env.get("CODEX_HOME") or proc_home / ".codex"))
        if "claude" in cmd:
            # Claude Code / OpenClaude keep the config dir (CLAUDE_CONFIG_DIR, default ~/.claude) and, next to it,
            # the main config file (~/.claude.json; the sessions driving this worktree write it periodically)
            candidates.append(Path(env.get("CLAUDE_CONFIG_DIR") or proc_home / ".claude"))
            candidates.append(proc_home / ".claude.json")
        for candidate in candidates:
            try:
                rel = candidate.resolve().relative_to(home)
            except (ValueError, OSError):
                continue
            dirs[str(rel)] = f"{pid} {cmd[:120]}"
    return dirs


#: Directories under $HOME a harness would create or touch if it ignored the redirected HOME / CODEX_HOME /
#: CLAUDE_CONFIG_DIR: a pre-existing file touched there during a run fails unless a live foreign instance owns it.
HARNESS_HOME_DIRS: tuple[str, ...] = (
    ".codex", ".claude", ".claude.json", ".openclaude", ".openclaude.json", ".config/openclaude", ".config/codex",
    ".config/claude", ".local/share/openclaude", ".local/share/codex", ".local/state/openclaude", ".local/state/codex",
    ".cache/openclaude", ".cache/codex", ".cache/claude", ".npm", ".openclaw",
)


def _under(rel: str, dirs: Iterable[str]) -> str | None:
    return next((d for d in dirs if rel == d or rel.startswith(d + "/")), None)


def _walk_home(max_depth: int, budget_seconds: float) -> tuple[dict[str, float], bool]:
    """``{relative path: mtime}`` of the files under $HOME up to ``max_depth`` (this session's own interactive-tool
    state pruned); the flag is False when the walk ran out of its time budget."""
    home = Path.home()
    started = time.monotonic()
    seen: dict[str, float] = {}
    for dirpath, dirs, files in os.walk(home, onerror=lambda _e: None):
        rel_dir = os.path.relpath(dirpath, home)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        kept = []
        for d in dirs:
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if any(rel == e for e in HOME_EXCLUDES):
                continue
            kept.append(d)
        dirs[:] = [] if depth >= max_depth else kept
        for name in files:
            path = Path(dirpath) / name
            try:
                seen[str(path.relative_to(home))] = path.lstat().st_mtime
            except OSError:
                continue
        if time.monotonic() - started > budget_seconds:
            return seen, False
    return seen, True


@dataclass
class HomeMarker:
    """Taken BEFORE a run: the wall-clock marker and the set of files that already existed under $HOME."""
    since: float
    existing: frozenset[str]
    complete: bool


def home_marker(*, max_depth: int = 3, budget_seconds: float = 120.0) -> HomeMarker:
    seen, complete = _walk_home(max_depth, budget_seconds)
    return HomeMarker(since=time.time() - 1.0, existing=frozenset(seen), complete=complete)


@dataclass
class HomeScan:
    """The verdict on $HOME after a run. ``new_files`` = files created during the run that no live foreign harness
    instance accounts for (a harness that ignored its redirected HOME creates its state dir: always a failure);
    ``harness_dir_touches`` = pre-existing files under ``HARNESS_HOME_DIRS`` modified during the run and
    unattributed (a failure); ``attributed`` = writes a live foreign instance's state dir explains (reported);
    ``churn`` = other pre-existing files touched during the run (unrelated periodic writers such as a root-owned
    quota file or another tool's lock file; reported, never a harness verdict); ``scanned`` False when either walk
    ran out of its budget (never reported as clean)."""
    new_files: list[str]
    harness_dir_touches: list[str]
    attributed: dict[str, str]
    churn: list[str]
    scanned: bool

    @property
    def clean(self) -> bool:
        return self.scanned and not self.new_files and not self.harness_dir_touches

    def as_detail(self) -> dict[str, Any]:
        return {"scanned": self.scanned, "new_files": self.new_files[:20],
                "harness_dir_touches": self.harness_dir_touches[:20], "attributed": self.attributed,
                "churn": self.churn[:20], "excluded": list(HOME_EXCLUDES)}


def new_under_home(marker: HomeMarker, *, max_depth: int = 3, budget_seconds: float = 120.0) -> HomeScan:
    """Files under ``$HOME`` (depth <= ``max_depth``) modified after ``marker.since``, classified as the ``HomeScan``
    docstring says; writes under a live foreign harness instance's state dir (``foreign_harness_state_dirs``) are
    attributed to it. The full ``/home`` tree holds ~860k files, so the walk is bounded like
    ``cli.new_files_under_home``."""
    foreign = foreign_harness_state_dirs()
    seen, complete = _walk_home(max_depth, budget_seconds)
    new_files: list[str] = []
    touches: list[str] = []
    attributed: dict[str, str] = {}
    churn: list[str] = []
    for rel, mtime in seen.items():
        if mtime <= marker.since:
            continue
        owner = _under(rel, foreign)
        if owner is not None:
            attributed[rel] = foreign[owner]
        elif rel not in marker.existing:
            new_files.append(rel)
        elif _under(rel, HARNESS_HOME_DIRS) is not None:
            touches.append(rel)
        else:
            churn.append(rel)
    return HomeScan(new_files=sorted(new_files), harness_dir_touches=sorted(touches), attributed=attributed,
                    churn=sorted(churn), scanned=complete and marker.complete)


def common_checks(stack: SmokeStack, spec: RunSpec, record: RunRecord, marker: HomeMarker,
                  ) -> tuple[dict[str, bool], dict[str, Any]]:
    """The Gate 0 items every case pins: no ``budget_usd`` text, no key material, nothing written under ``$HOME``
    (unattributed), the harness version stamped on the spec. Returns ``(checks, detail)``."""
    budget_hits = budget_usd_hits(stack, spec)
    key_hits = key_material_hits(stack)
    home = new_under_home(marker)
    checks = {
        "no_budget_usd_text": budget_hits == [],
        "no_key_material": key_hits == [],
        "nothing_written_under_home": home.clean,
        "harness_version_stamped": record.spec.harness_version not in ("absent", "unrecorded", "unknown"),
    }
    detail = {"budget_usd_hits": budget_hits, "key_material_hits": key_hits, "home": home.as_detail(),
              "harness_version": record.spec.harness_version}
    return checks, detail


def reinjection_hit_rate(record: RunRecord) -> float | None:
    hits, misses = int(record.reinjection_hits), int(record.reinjection_misses)
    return None if hits + misses == 0 else hits / (hits + misses)


def process_group_alive(pgid: int) -> bool:
    """Whether any process of the group still exists (``killpg(pgid, 0)``)."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def raw_journal_docs(run_root: Path) -> list[dict[str, Any]]:
    out = []
    for raw in sorted((Path(run_root) / "raw").rglob("call_*.json.gz")):
        out.append(json.loads(gzip.open(raw).read()))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The Gate 0 smoke report
# ---------------------------------------------------------------------------------------------------------------------

def record_case(case: str, checks: dict[str, bool], *, detail: dict[str, Any] | None = None) -> list[str]:
    """Merge one case into ``$DYNORCH_GATE0_SMOKE_OUT`` (when set) and return the names of its failing checks."""
    failed = [name for name, ok in checks.items() if not ok]
    out = os.environ.get(REPORT_ENV)
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target.with_suffix(target.suffix + ".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)          # several pytest processes merge into one report
            try:
                _merge_report(target, case, checks, detail)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return failed


def _merge_report(target: Path, case: str, checks: dict[str, bool], detail: dict[str, Any] | None) -> None:
    """Read-modify-write of the report under the caller's lock: cases keyed by name, ``passed`` / ``failures``
    re-derived over every recorded case."""
    report: dict[str, Any] = {"passed": True, "failures": [], "cases": {}}
    if target.is_file():
        try:
            loaded = json.loads(target.read_text("utf-8"))
        except ValueError:
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("cases"), dict):
            report["cases"] = dict(loaded["cases"])
    report["cases"][case] = {"checks": {k: bool(v) for k, v in checks.items()}, "detail": detail or {},
                             "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    failures = []
    for name, body in report["cases"].items():
        case_checks = body.get("checks", body) if isinstance(body, dict) else {}
        for check, ok in case_checks.items():
            if isinstance(ok, bool) and not ok:
                failures.append(f"{name}: {check}")
    report["failures"] = failures
    report["passed"] = not failures
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(report, indent=1, sort_keys=True)
    if "sk-or-" in text:
        raise RuntimeError("the smoke report would contain key material; refusing to write it")
    target.write_text(text)
