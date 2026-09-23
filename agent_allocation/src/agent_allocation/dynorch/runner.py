"""WP-B: the harness runner (contract §6 ``runner.py``; plan §6; frozen table: wall timeout, rerun rule).

``Runner(core, adapters, settings, ...).run_stage(stage, resume=...)`` plans the stage (``plan.plan_stage``),
pre-registers it (``plan.write_plan``), and drives an asyncio subprocess pool of ``settings.concurrency`` harness
processes. Per run (one attempt):

1. a fresh work dir ``<scratch>/work/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/`` (``harness.base.WorkDir``);
   ``SYSTEM.md`` = the study system prompt (H0: the ladder's ``system.txt``), ``TASK.md`` = ``root.txt`` rendered
   with the task and its output requirement + the B8 budget sentence with ``C = cap_tokens``;
2. ``core.open_run(spec, root_system_text=..., subagent_system_text=...)``; ``adapter.prepare``; ``build_command``;
   the environment re-checked by ``harness.base.check_env`` (no ``OPENROUTER_*`` / ``ANTHROPIC_*`` / other
   ``OPENAI_*`` variable, every state path under scratch); ``launch.json`` journaled with key-shaped values masked;
3. ``asyncio.create_subprocess_exec(..., start_new_session=True)``, stdout/stderr to files, stdin = TASK.md; wall
   timeout ``base + per_rung * rung`` (600 + 120 k) -> SIGTERM to the process group, SIGKILL ``sigkill_after`` s
   later; the exhaustion watchdog polls ``core.budget_view`` and sends SIGTERM ``sigterm_after_exhaustion`` s after
   the last served reply once ``remaining < terminal_threshold``, SIGKILL ``sigkill_after`` s later (plan §2 B4);
   in-process adapters (H0, OpenClaw) are awaited instead;
4. the status and the outcome-blind rerun rule (frozen table): ``killed_by_drain`` -> rerun; zero proxy calls served
   -> rerun; the last proxy event a transport/ledger error (``upstream_failed`` / ``ledger_refused`` / a 5xx) ->
   ``transport_failed``, rerun; NEVER after a 402, a forced close, a parse failure or a wall timeout with calls
   served; at most ``max_reruns`` reruns (attempt numbering ``a1, a2, a3``; a rerun gets a new ``run_key`` and
   ``rerun_reason`` on the superseded attempt); every attempt is journaled (``<run_root>/state/dynorch_attempts.jsonl``)
   and paid; all-infra -> the final record stays an infra failure and ``grade.finalize_run`` leaves y NaN;
5. ``core.close_run``; the delivered answer from the proxy's reply log (plan §2 B7); ``cli_result`` cross-check;
   ``grade.finalize_run``; ``store.save`` (``replace_atomic``).

``--resume`` (``run_stage(resume=True)``) skips terminal records without a pending rerun, re-executes interrupted
(``running``) and rerun-eligible ones as the next attempt with ``generation + 1`` (the core's ledger ids embed the
generation, so a resumed run never collides), moving the interrupted attempt's raw journal to
``<run_root>/superseded/<timestamp>/``. The Slurm drain signal (``--signal=B:TERM@900``; ``install_drain_handler``)
stops admissions, SIGTERMs in-flight harness groups (their attempts are ``killed_by_drain`` -> rerun on the next
``--resume``), waits at most ``drain_wait_seconds`` (800) for settles and returns so the CLI can exit 0.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.apikey import redact
from agent_allocation.contracts import PublicTask
from agent_allocation.dynorch import grade as grademod
from agent_allocation.dynorch import plan as planmod
from agent_allocation.dynorch.harness.base import (
    HarnessAdapter, HarnessResult, InProcessAdapter, LaunchPlan, WorkDir, check_env, harness_env, redacted_env,
)
from agent_allocation.dynorch.httpd import ReplyLog
from agent_allocation.dynorch.plan import BaseInputs, item_slug
from agent_allocation.dynorch.records import RunRecord, RunSpec
from agent_allocation.paths import PROJECT_ROOT
from agent_allocation.prompts.render import output_requirement
from agent_allocation.storage._atomic import replace_atomic

__all__ = [
    "RERUN_REASONS", "Runner", "RunnerSettings", "StageSummary", "TERMINAL_STATUSES", "install_drain_handler",
    "render_task_text", "scrubbed_env", "wall_timeout_seconds",
]

log = logging.getLogger("agent_allocation.dynorch.runner")

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "timeout", "transport_failed", "refused_before_start"})
RERUN_REASONS: tuple[str, ...] = ("zero_proxy_calls_served", "last_reply_transport_or_ledger_error", "killed_by_drain",
                                  "interrupted")
TRANSPORT_ERROR_CODES: frozenset[str] = frozenset({"upstream_failed", "ledger_refused", "internal_error"})
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
scrubbed_env = harness_env


def wall_timeout_seconds(base: int, per_rung: int, rung: int) -> int:
    """Frozen table: ``600 + 120 * k`` (k = 25 -> 3,600 s)."""
    return int(base) + int(per_rung) * int(rung)


def render_task_text(task: PublicTask, cap_tokens: int, *, user_template: str, budget_sentence: str) -> str:
    """``root.txt`` with the task and its output requirement (one pass, never rescanned), then the B8 sentence."""
    values = {"task_prompt": task.prompt, "output_requirement": output_requirement(task)}
    declared = set(_PLACEHOLDER.findall(user_template))
    if declared != set(values):
        raise ValueError(f"user template placeholders {sorted(declared)} != {sorted(values)}")
    body = _PLACEHOLDER.sub(lambda m: values[m.group(1)], user_template)
    if "{C}" not in budget_sentence:
        raise ValueError("budget sentence must contain {C}")
    return body.rstrip("\n") + "\n\n" + budget_sentence.replace("{C}", str(int(cap_tokens))) + "\n"


@dataclass(frozen=True)
class RunnerSettings:
    run_root: Path
    scratch_root: Path
    model: str
    concurrency: int
    wall_base: int
    wall_per_rung: int
    sigterm_after_exhaustion: float
    sigkill_after: float
    terminal_threshold: int
    max_reruns: int
    system_text: str
    subagent_text: str
    h0_system_text: str
    user_template: str
    budget_sentence: str
    drain_wait_seconds: float = 800.0
    poll_seconds: float = 2.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, run_root: str | Path, project_root: str | Path = PROJECT_ROOT,
                    scratch_root: str | Path | None = None, concurrency: int | None = None) -> "RunnerSettings":
        block = cfg["dynorch"]
        root = Path(project_root)
        files = block["prompt_files"]

        def read(name: str) -> str:
            path = root / files[name]
            if not path.is_file():
                raise FileNotFoundError(f"prompt file {path} missing")
            return path.read_text("utf-8")

        wall = block["wall_timeout_seconds"]
        runner = block["runner"]
        return cls(
            run_root=Path(run_root), scratch_root=Path(scratch_root or block["scratch_root"]),
            model=str(cfg["primary_model"]),
            concurrency=int(concurrency if concurrency is not None else runner["concurrency"]),
            wall_base=int(wall["base"]), wall_per_rung=int(wall["per_rung"]),
            sigterm_after_exhaustion=float(runner["sigterm_after_exhaustion_seconds"]),
            sigkill_after=float(runner["sigkill_after_seconds"]),
            terminal_threshold=int(block["terminal_threshold_tokens"]), max_reruns=int(block["rerun_rule"]["max_reruns"]),
            system_text=read("system"), subagent_text=read("subagent"), h0_system_text=read("h0_system"),
            user_template=read("user"), budget_sentence=str(block["budget_sentence"]))

    def wall_seconds(self, spec: RunSpec) -> int:
        return wall_timeout_seconds(self.wall_base, self.wall_per_rung, spec.rung)

    def work_dir(self, spec: RunSpec) -> Path:
        return (self.scratch_root / "work" / spec.arm / str(spec.rung) / str(spec.repeat) / item_slug(spec.item_id)
                / f"a{spec.attempt}")


@dataclass
class StageSummary:
    stage: str
    planned: int
    executed: int = 0
    skipped: int = 0
    reruns: int = 0
    drained: bool = False
    statuses: dict[str, int] = field(default_factory=dict)
    records: dict[str, RunRecord] = field(default_factory=dict)


class Runner:
    """See the module docstring. ``store`` implements ``save``/``scan``/``is_terminal`` (contract §5 RunStore)."""

    def __init__(self, core: Any, adapters: Mapping[str, HarnessAdapter], settings: RunnerSettings, *,
                 proxy_url: str, store: Any, inputs: BaseInputs, reply_log: ReplyLog, cfg: Mapping[str, Any],
                 base_env: Mapping[str, str] | None = None, journal: Any = None) -> None:
        self.core = core
        self.journal = journal
        self.adapters = dict(adapters)
        self.settings = settings
        self.proxy_url = proxy_url.rstrip("/")
        self.store = store
        self.inputs = inputs
        self.reply_log = reply_log
        self.cfg = cfg
        self.base_env = dict(base_env) if base_env is not None else dict(os.environ)
        self._draining = asyncio.Event()
        self._in_flight: dict[str, asyncio.subprocess.Process] = {}
        self._drain_killed: set[str] = set()
        self.attempts_path = settings.run_root / "state" / "dynorch_attempts.jsonl"

    # -- drain ---------------------------------------------------------------------------------------------
    def request_drain(self) -> None:
        """Stop admitting runs and terminate in-flight harnesses (their attempts become ``killed_by_drain``)."""
        if not self._draining.is_set():
            log.warning("dynorch runner: drain requested; %d in flight", len(self._in_flight))
            self._draining.set()
            for run_key, proc in list(self._in_flight.items()):
                self._drain_killed.add(run_key)
                _signal_group(proc, signal.SIGTERM)

    @property
    def draining(self) -> bool:
        return self._draining.is_set()

    # -- stage ---------------------------------------------------------------------------------------------
    async def run_stage(self, stage: str, *, resume: bool = False, arms: Sequence[str] | None = None,
                        rungs: Sequence[int] | None = None) -> StageSummary:
        harness_versions = {name: adapter.version() for name, adapter in self.adapters.items()}
        proxy_version = str(getattr(self.core, "proxy_version", "unrecorded"))
        specs = planmod.plan_stage(self.cfg, stage, self.inputs, arms=arms, rungs=rungs, proxy_version=proxy_version,
                                   harness_versions=harness_versions)
        planmod.write_plan(specs, self.settings.run_root, stage)
        summary = StageSummary(stage=stage, planned=len(specs))
        existing = self._existing_by_cell() if resume else {}
        todo: list[RunSpec] = []
        for spec in specs:
            previous = existing.get(_cell_key(spec))
            if previous is None:
                todo.append(spec)
                continue
            if self.store.is_terminal(previous) and not self._rerun_pending(previous):
                summary.skipped += 1
                summary.records[previous.spec.run_key] = previous
                continue
            todo.append(self._next_attempt(previous, resume=True))
        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def one(spec: RunSpec) -> None:
            async with semaphore:
                if self._draining.is_set():
                    return
                record = await self.run_spec(spec)
                summary.executed += 1
                summary.records[record.spec.run_key] = record
                summary.statuses[record.status] = summary.statuses.get(record.status, 0) + 1

        tasks = [asyncio.create_task(one(spec)) for spec in todo]
        if tasks:
            waiter = asyncio.gather(*tasks)
            drain = asyncio.create_task(self._draining.wait())
            done, _ = await asyncio.wait({waiter, drain}, return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                drain.cancel()
                await waiter
            else:
                summary.drained = True
                try:
                    await asyncio.wait_for(waiter, timeout=self.settings.drain_wait_seconds)
                except asyncio.TimeoutError:
                    log.error("dynorch runner: drain wait of %.0f s exceeded; abandoning in-flight runs",
                              self.settings.drain_wait_seconds)
                    for task in tasks:
                        task.cancel()
        summary.reruns = sum(1 for r in summary.records.values() if r.spec.attempt > 1)
        return summary

    def _existing_by_cell(self) -> dict[tuple[str, str, int, int], RunRecord]:
        return {_cell_key(record.spec): record for record in self.store.scan().values()}

    def _rerun_pending(self, record: RunRecord) -> bool:
        return (record.rerun_reason in RERUN_REASONS and record.spec.attempt <= self.settings.max_reruns
                and record.status != "done")

    def _next_attempt(self, previous: RunRecord, *, resume: bool) -> RunSpec:
        spec = previous.spec
        if not self.store.is_terminal(previous) and resume:
            self._supersede(previous)
        next_no = spec.attempt + 1
        return dataclasses.replace(spec, attempt=next_no, generation=spec.generation + (1 if resume else 0),
                                   run_key=planmod.run_key_for(str(self.cfg["study_id"]), spec.arm, spec.item_id,
                                                               spec.rung, spec.repeat, next_no))

    def _supersede(self, previous: RunRecord) -> None:
        """Move an interrupted attempt's raw journal aside so its replacement starts from a clean directory.

        Uses the core journal's ``supersede`` / the store's ``archive_previous`` when they exist (WP-A R-A7; the CLI
        must have run ``resolve_inherited`` first), else moves ``raw/.../a<attempt>`` to ``superseded/<ts>/``.
        """
        spec = previous.spec
        supersede = getattr(self.journal, "supersede", None) if self.journal is not None else None
        if supersede is not None:
            supersede(spec)
        else:
            raw = (self.settings.run_root / "raw" / spec.arm / str(spec.rung) / str(spec.repeat)
                   / item_slug(spec.item_id) / f"a{spec.attempt}")
            if raw.exists():
                target = (self.settings.run_root / "superseded" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                          / raw.relative_to(self.settings.run_root))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(raw), str(target))
        archive = getattr(self.store, "archive_previous", None)
        if archive is not None:
            archive(spec)
        previous.status = "failed"
        previous.rerun_reason = "interrupted"
        previous.error = previous.error or "interrupted before close (resume)"
        self._journal_attempt(previous)

    # -- one cell ------------------------------------------------------------------------------------------
    async def run_spec(self, spec: RunSpec) -> RunRecord:
        """All attempts of one cell under the rerun rule; returns the final record (finalised and stored)."""
        current = spec
        while True:
            record = await self._attempt(current)
            self._journal_attempt(record)
            if record.rerun_reason in RERUN_REASONS and current.attempt <= self.settings.max_reruns \
                    and not self._draining.is_set():
                log.info("dynorch runner: rerun %s attempt %d (%s)", current.item_id, current.attempt + 1,
                         record.rerun_reason)
                current = self._next_attempt(record, resume=False)
                continue
            return record

    async def _attempt(self, spec: RunSpec) -> RunRecord:
        settings = self.settings
        adapter = self.adapters.get(spec.harness)
        if adapter is None:
            raise KeyError(f"no adapter for harness {spec.harness!r} (arm {spec.arm})")
        task = self.inputs.tasks[spec.item_id]
        private = self.inputs.privates[spec.item_id]
        work = WorkDir(settings.work_dir(spec)).create()
        system_text = settings.h0_system_text if spec.harness == "null" else settings.system_text
        task_text = render_task_text(task, spec.cap_tokens, user_template=settings.user_template,
                                     budget_sentence=settings.budget_sentence)
        proxy_base = f"{self.proxy_url}/r/{spec.run_key}/v1"
        wall = settings.wall_seconds(spec)
        started = time.monotonic()
        self.core.open_run(spec, root_system_text=system_text, subagent_system_text=settings.subagent_text)
        self.reply_log.forget(spec.run_key)
        adapter.prepare(work.root, spec, proxy_base, system_text, settings.subagent_text, task_text)
        plan = adapter.build_command(work.root, spec, proxy_base)
        check_env(plan.env, work=work.root, run_key=spec.run_key)
        self._write_launch_json(work, spec, plan, wall)
        killed_by_drain = False
        exhausted_kill = False
        if getattr(adapter, "in_process", False):
            result = await self._run_in_process(adapter, work, spec, proxy_base, wall)
            timed_out = result.timed_out
            exit_code = result.exit_code
        else:
            exit_code, timed_out, killed_by_drain, exhausted_kill = await self._run_subprocess(spec, plan, work, wall)
            result = adapter.parse_result(work.stdout.read_bytes() if work.stdout.exists() else b"",
                                          work.stderr.read_bytes() if work.stderr.exists() else b"", work.root)
        wall_seconds = time.monotonic() - started
        status, rerun_reason, error = self._status_of(spec, timed_out=timed_out, killed_by_drain=killed_by_drain)
        record: RunRecord = self.core.close_run(spec.run_key, status=status)
        if not isinstance(record, RunRecord):
            raise TypeError(f"core.close_run returned {type(record).__name__}, expected RunRecord")
        record.status = status
        record.rerun_reason = rerun_reason
        if error:
            record.error = error
        record.wall_seconds = round(wall_seconds, 3)
        record.harness_exit_code = exit_code
        reported = dict(result.reported)
        reported.update({"timed_out": timed_out, "killed_by_drain": killed_by_drain,
                         "sigterm_after_exhaustion": exhausted_kill, "wall_timeout_seconds": wall,
                         "work_dir": str(work.root)})
        record.harness_reported = reported
        served_message, served_source = self.reply_log.delivered(spec.run_key)
        if record.final_message is None:            # the core may fill it at close_run (WP-A R-A6); the log is the check
            record.final_message, record.final_source = served_message, served_source
        elif served_message is not None and served_message.strip() != record.final_message.strip():
            log.warning("dynorch runner: %s final_message from the core differs from the served reply log", spec.run_key)
            reported["final_message_disagreement"] = True
        record.cli_result = result.cli_result
        if record.final_message is not None and result.cli_result is not None:
            record.cli_result_mismatch = record.final_message.strip() != result.cli_result.strip()
        elif record.final_message is not None or result.cli_result is not None:
            record.cli_result_mismatch = True
        else:
            record.cli_result_mismatch = None
        grademod.finalize_run(record, task, private)
        self.store.save(record)
        return record

    def _status_of(self, spec: RunSpec, *, timed_out: bool, killed_by_drain: bool) -> tuple[str, str | None, str | None]:
        """The frozen rerun rule, decided from what the proxy served (outcome-blind)."""
        view = self.core.budget_view(spec.run_key)
        calls = int(view["calls"])
        last = self.reply_log.last_event(spec.run_key)
        if killed_by_drain:
            return "failed", "killed_by_drain", "killed by the drain signal"
        if calls == 0:
            status = "timeout" if timed_out else "failed"
            return status, "zero_proxy_calls_served", "harness served no proxy call"
        if last is not None and last.kind == "error" and (last.code in TRANSPORT_ERROR_CODES or last.status >= 500):
            return "transport_failed", "last_reply_transport_or_ledger_error", f"last proxy reply: {last.status} {last.code}"
        core_error = getattr(self.core, "record_of", None)
        if core_error is not None:
            error = str(core_error(spec.run_key).error or "")
            if error.startswith("upstream_failed:"):
                return "transport_failed", "last_reply_transport_or_ledger_error", error
        if timed_out:
            return "timeout", None, f"wall timeout after {calls} served call(s)"
        return "done", None, None

    async def _run_in_process(self, adapter: Any, work: WorkDir, spec: RunSpec, proxy_base: str, wall: int,
                              ) -> HarnessResult:
        if not isinstance(adapter, InProcessAdapter):
            raise TypeError(f"adapter {adapter.name} claims in_process but has no run_in_process")
        try:
            return await asyncio.wait_for(adapter.run_in_process(work.root, spec, proxy_base, wall_seconds=wall),
                                          timeout=wall + 5)
        except asyncio.TimeoutError:
            return HarnessResult(exit_code=None, timed_out=True, cli_result=None, reported={"error": "wall timeout"})

    async def _run_subprocess(self, spec: RunSpec, plan: LaunchPlan, work: WorkDir, wall: int,
                              ) -> tuple[int | None, bool, bool, bool]:
        """``(exit_code, timed_out, killed_by_drain, sigterm_after_exhaustion)``."""
        settings = self.settings
        with open(work.stdout, "ab") as out, open(work.stderr, "ab") as err:
            proc = await asyncio.create_subprocess_exec(
                *plan.argv, cwd=str(plan.cwd), env=plan.env, stdin=subprocess.PIPE if plan.stdin is not None else subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True)
        self._in_flight[spec.run_key] = proc
        timed_out = killed_by_drain = exhausted_kill = False
        watchdog = asyncio.create_task(self._exhaustion_watchdog(spec, proc))
        try:
            if plan.stdin is not None and proc.stdin is not None:
                try:
                    proc.stdin.write(plan.stdin)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=wall)
            except asyncio.TimeoutError:
                timed_out = True
                await _terminate(proc, settings.sigkill_after)
        finally:
            if not watchdog.done():
                watchdog.cancel()
            try:
                exhausted_kill = bool(await watchdog)
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - a watchdog error must not lose the run
                exhausted_kill = False
            self._in_flight.pop(spec.run_key, None)
            if proc.returncode is None:
                await _terminate(proc, settings.sigkill_after)
        killed_by_drain = spec.run_key in self._drain_killed and not timed_out and not exhausted_kill
        return proc.returncode, timed_out, killed_by_drain, exhausted_kill

    async def _exhaustion_watchdog(self, spec: RunSpec, proc: asyncio.subprocess.Process) -> bool:
        """SIGTERM the group ``sigterm_after_exhaustion`` s after the last served reply once exhausted; SIGKILL later."""
        settings = self.settings
        exhausted_since: float | None = None
        seen_calls = -1
        while proc.returncode is None:
            await asyncio.sleep(settings.poll_seconds)
            try:
                view = self.core.budget_view(spec.run_key)
            except Exception:  # noqa: BLE001 - the run may already be closing
                return False
            calls, remaining = int(view["calls"]), int(view["remaining"])
            exhausted = remaining < settings.terminal_threshold or bool(view.get("exhausted", False))
            if not exhausted:
                continue
            if calls != seen_calls or exhausted_since is None:
                seen_calls, exhausted_since = calls, time.monotonic()
                continue
            if time.monotonic() - exhausted_since >= settings.sigterm_after_exhaustion:
                log.info("dynorch runner: %s exhausted; SIGTERM %.0f s after the last served reply", spec.run_key,
                         settings.sigterm_after_exhaustion)
                await _terminate(proc, settings.sigkill_after)
                return True
        return False

    def _write_launch_json(self, work: WorkDir, spec: RunSpec, plan: LaunchPlan, wall: int) -> None:
        payload = {"run_key": spec.run_key, "arm": spec.arm, "harness": spec.harness, "item_id": spec.item_id,
                   "rung": spec.rung, "repeat": spec.repeat, "attempt": spec.attempt, "generation": spec.generation,
                   "argv": [redact(a) for a in plan.argv], "env": redacted_env(plan.env), "cwd": str(plan.cwd),
                   "stdin_bytes": len(plan.stdin) if plan.stdin is not None else 0, "wall_timeout_seconds": wall,
                   "proxy_url": self.proxy_url, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        if "sk-or-" in text:
            raise RuntimeError("launch.json would contain key material; refusing to write it")
        replace_atomic(work.launch_json, text.encode("utf-8"))

    def _journal_attempt(self, record: RunRecord) -> None:
        line = json.dumps(dataclasses.asdict(record), ensure_ascii=False, sort_keys=True)
        line = redact(line) or line
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.attempts_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _cell_key(spec: RunSpec) -> tuple[str, str, int, int]:
    return spec.arm, spec.item_id, spec.rung, spec.repeat


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


async def _terminate(proc: asyncio.subprocess.Process, sigkill_after: float) -> None:
    """SIGTERM the process group, SIGKILL after ``sigkill_after`` seconds, wait for exit."""
    _signal_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=sigkill_after)
        return
    except asyncio.TimeoutError:
        pass
    _signal_group(proc, signal.SIGKILL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        log.error("dynorch runner: process group %d survived SIGKILL", proc.pid)


def install_drain_handler(runner: Runner, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Trap the Slurm drain signal (``--signal=B:TERM@900``): SIGTERM -> ``runner.request_drain()``."""
    loop = loop or asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, runner.request_drain)
