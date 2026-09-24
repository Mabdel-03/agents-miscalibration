"""WP-B: the harness runner (contract §6 ``runner.py``; plan §6; frozen table: wall timeout, rerun rule).

``Runner(core, adapters, settings, ...).run_stage(stage, resume=...)`` plans the stage (``plan.plan_stage``),
pre-registers it (``plan.write_plan``), and drives an asyncio subprocess pool of ``settings.concurrency`` harness
processes. Per run (one attempt):

1. a fresh work dir ``<scratch>/work/<partition>/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/``
   (``harness.base.WorkDir``);
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
``--resume``), waits at most ``drain_wait_seconds`` (540) for them to finish, terminates the survivors and returns by
``drain_total_seconds`` (660) so the server close, tables and reconciliation fit in Slurm's 900 s (contract §10.33).
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
from decimal import Decimal
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
from agent_allocation.live import RunRefused
from agent_allocation.paths import PROJECT_ROOT
from agent_allocation.prompts.render import output_requirement
from agent_allocation.storage._atomic import replace_atomic

__all__ = [
    "RERUN_REASONS", "Runner", "RunnerSettings", "Stack", "StageSummary", "TERMINAL_STATUSES", "apply_provider_map",
    "build_stack", "install_drain_handler", "launch_stage", "render_task_text", "scrubbed_env", "specs_for_cells",
    "summary_dict", "wall_timeout_seconds",
]

log = logging.getLogger("agent_allocation.dynorch.runner")

TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "failed", "timeout", "transport_failed", "refused_before_start"})
RERUN_REASONS: tuple[str, ...] = ("zero_proxy_calls_served", "last_reply_transport_or_ledger_error", "killed_by_drain",
                                  "interrupted", "infrastructure_stop", "ledger_stop")
#: An upstream 402/401 (key limit, account) is an infrastructure stop (plan B6 / Gate 3): the account refused, no
#: model ran, nothing was paid. It drains the stage and NEVER counts against the frozen ``<= 2 reruns`` (which
#: bounds reruns of runs the model actually served); the cell is re-executed by the next ``--resume``.
INFRASTRUCTURE_STOP = "infrastructure_stop"
#: The $ ledger refused the run's last call (the active ceiling, or unresolved exposure > max; plan §5: this PAUSES
#: admissions). Like an infrastructure stop: nothing left the process, the stage drains, no in-job rerun, and the
#: cell stays rerun-eligible on --resume without counting against the frozen ``<= 2 reruns`` (re-review finding 4).
LEDGER_STOP = "ledger_stop"
STOP_REASONS: frozenset[str] = frozenset({INFRASTRUCTURE_STOP, LEDGER_STOP})
TRANSPORT_ERROR_CODES: frozenset[str] = frozenset({"upstream_failed", "ledger_refused", "internal_error"})
#: GNU ``timeout``'s own exit codes: 124 = the deadline fired, 137 = the child was SIGKILLed by ``--kill-after``.
TIMEOUT_EXIT_CODES: frozenset[int] = frozenset({124, 137})
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
    #: Slurm sends the drain signal 900 s before the kill (``--signal=B:TERM@900``). The runner's whole drain must
    #: fit with the server close (<= 60 s), the tables and the reconciliation (<= ~90 s when drained) after it
    #: (re-review finding 5, contract §10.33): in-flight harnesses get ``drain_wait_seconds`` to finish on their own,
    #: the survivors are terminated, and everything -- the settle waits included -- ends by ``drain_total_seconds``.
    drain_wait_seconds: float = 540.0
    drain_total_seconds: float = 660.0
    poll_seconds: float = 2.0
    #: how long ``_attempt`` waits, after its harness is gone, for a call the proxy is still serving for it
    settle_wait_seconds: float = 300.0

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
        return (self.scratch_root / "work" / spec.partition / spec.arm / str(spec.rung) / str(spec.repeat)
                / item_slug(spec.item_id) / f"a{spec.attempt}")


@dataclass
class StageSummary:
    stage: str
    planned: int
    executed: int = 0
    skipped: int = 0
    reruns: int = 0
    errors: int = 0            # cells whose attempt raised outside the record (the stage still finishes)
    drained: bool = False
    statuses: dict[str, int] = field(default_factory=dict)
    records: dict[str, RunRecord] = field(default_factory=dict)


class Runner:
    """See the module docstring. ``store`` implements ``save``/``scan``/``is_terminal`` (contract §5 RunStore)."""

    def __init__(self, core: Any, adapters: Mapping[str, HarnessAdapter], settings: RunnerSettings, *,
                 proxy_url: str, store: Any, inputs: BaseInputs, reply_log: ReplyLog, cfg: Mapping[str, Any],
                 base_env: Mapping[str, str] | None = None, journal: Any = None,
                 wait_for_run: Any = None, admission_check: Any = None) -> None:
        self.core = core
        #: ``() -> str | None``: why no new cell may be admitted now (the $ ledger has no headroom or its unresolved
        #: exposure is over the limit; plan §5 "pauses admissions"), else None. The stage drains on a reason.
        self.admission_check = admission_check
        self.journal = journal
        self.adapters = dict(adapters)
        self.settings = settings
        self.proxy_url = proxy_url.rstrip("/")
        self.store = store
        self.inputs = inputs
        self.reply_log = reply_log
        self.cfg = cfg
        self.base_env = dict(base_env) if base_env is not None else dict(os.environ)
        #: ``ProxyServer.wait_for_run``: the calls the proxy is still finishing for a harness that went away are
        #: awaited before ``close_run`` so the record is finalised once, with every token counted (major 14).
        self.wait_for_run = wait_for_run
        self._draining = asyncio.Event()
        self._drain_deadline: float | None = None             # monotonic; set by request_drain
        self._in_flight: dict[str, asyncio.subprocess.Process] = {}
        self._drain_killed: set[str] = set()
        self.attempts_path = settings.run_root / "state" / "dynorch_attempts.jsonl"

    # -- drain ---------------------------------------------------------------------------------------------
    def request_drain(self) -> None:
        """Plan §6: stop admitting runs and LET THE IN-FLIGHT HARNESSES FINISH (``drain_wait_seconds``, 540 s; the
        plan's "wait <= 800 s" is an upper bound, contract §10.33 fits the whole drain into Slurm's 900 s).

        Killing them at once threw away up to ``concurrency`` paid runs at every 12 h job boundary and burned one
        of their two reruns each; only the survivors of the wait are terminated (``killed_by_drain``).
        """
        if not self._draining.is_set():
            log.warning("dynorch runner: drain requested; %d in flight, waiting up to %.0f s for them to settle "
                        "(everything done within %.0f s)", len(self._in_flight), self.settings.drain_wait_seconds,
                        self.settings.drain_total_seconds)
            self._drain_deadline = time.monotonic() + self.settings.drain_total_seconds
            self._draining.set()

    def _drain_remaining(self) -> float | None:
        """Seconds left before the drain deadline (``None`` when not draining)."""
        if self._drain_deadline is None:
            return None
        return max(0.0, self._drain_deadline - time.monotonic())

    async def _terminate_in_flight(self) -> None:
        """The drain wait expired: SIGTERM (then SIGKILL) every harness still running; those attempts rerun."""
        procs = list(self._in_flight.items())
        for run_key, _ in procs:
            self._drain_killed.add(run_key)
        if procs:
            log.warning("dynorch runner: drain wait over; terminating %d harness group(s)", len(procs))
            await asyncio.gather(*[_terminate(proc, self.settings.sigkill_after) for _, proc in procs],
                                 return_exceptions=True)

    @property
    def draining(self) -> bool:
        return self._draining.is_set()

    # -- stage ---------------------------------------------------------------------------------------------
    def harness_versions(self) -> dict[str, str]:
        return {name: adapter.version() for name, adapter in self.adapters.items()}

    async def run_stage(self, stage: str, *, resume: bool = False, arms: Sequence[str] | None = None,
                        rungs: Sequence[int] | None = None) -> StageSummary:
        """Plan the stage's frozen cell set, pre-register it, and execute it (module docstring)."""
        proxy_version = str(getattr(self.core, "proxy_version", "unrecorded"))
        versions = self.harness_versions()
        specs = planmod.plan_stage(self.cfg, stage, self.inputs, arms=arms, rungs=rungs, proxy_version=proxy_version,
                                   harness_versions=versions)
        full = planmod.plan_stage(self.cfg, stage, self.inputs, proxy_version=proxy_version, harness_versions=versions)
        from agent_allocation.dynorch.supersede import apply_generation, execution_generation
        specs = apply_generation(specs, execution_generation(self.settings.run_root, stage))
        return await self.run_specs(stage, specs, resume=resume, registered=full)

    async def run_specs(self, stage: str, specs: Sequence[RunSpec], *, resume: bool = False,
                        registered: Sequence[RunSpec] | None = None) -> StageSummary:
        """Execute an explicit cell list under ``stage`` (what ``run_stage`` does after planning; the e2e's entry).
        ``registered``: the stage's full planned cell set, pre-registered on the first launch whatever subset runs
        (re-review minor 11); ``None`` registers ``specs`` themselves (explicit test cell lists)."""
        planmod.write_plan(specs, self.settings.run_root, stage, registered=registered)
        summary = StageSummary(stage=stage, planned=len(specs))
        # The store is ALWAYS consulted (review major 4/15): re-running a stage without --resume must never
        # overwrite a paid record in place, collide with the ledger attempt ids it already reserved or conflict in
        # the write-once call journal after the upstream call was paid. A finished cell is skipped; an unfinished
        # or rerun-eligible one needs --resume (which resolves the inherited ledger attempts and bumps the
        # execution generation first).
        existing = self._existing_by_cell()
        todo: list[RunSpec] = []
        blocked: list[str] = []
        for spec in specs:
            previous = existing.get(_cell_key(spec))
            if previous is None:
                todo.append(spec)
                continue
            if self.store.is_terminal(previous) and not self._rerun_pending(previous):
                summary.skipped += 1
                summary.records[previous.spec.run_key] = previous
                continue
            if not resume:
                blocked.append(f"{previous.spec.arm}/{previous.spec.rung}/{previous.spec.item_id} "
                               f"({previous.status}, attempt {previous.spec.attempt})")
                continue
            if not self._attempts_left(previous):
                summary.skipped += 1
                summary.records[previous.spec.run_key] = self._exhaust_attempts(previous)
                continue
            todo.append(self._next_attempt(previous, resume=True))
        if blocked:
            raise RunRefused(f"{len(blocked)} cell(s) of stage {stage} already have an unfinished or rerun-eligible "
                             f"record; pass --resume to continue them (e.g. {blocked[:3]})")
        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def one(spec: RunSpec) -> None:
            async with semaphore:
                if self._draining.is_set():
                    return
                blocked_by = self.admission_check() if self.admission_check is not None else None
                if blocked_by:
                    log.error("dynorch runner: admissions paused (%s); draining the stage", blocked_by)
                    self.request_drain()
                    return
                try:
                    record = await self.run_spec(spec)
                except Exception as exc:  # noqa: BLE001 - one bad cell must never take the stage down (major 8)
                    log.exception("dynorch runner: cell %s/%s/%s failed outside the attempt: %s", spec.arm,
                                  spec.rung, spec.item_id, exc)
                    summary.errors += 1
                    return
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
                natural = min(self.settings.drain_wait_seconds, self._drain_remaining() or 0.0)
                try:
                    # shielded: a timeout here must not cancel the in-flight runs, it must terminate them cleanly
                    await asyncio.wait_for(asyncio.shield(waiter), timeout=natural)
                except asyncio.TimeoutError:
                    log.error("dynorch runner: drain wait of %.0f s exceeded; terminating the survivors", natural)
                    await self._terminate_in_flight()
                    try:
                        # the survivors' attempts finish (settle wait bounded by the drain deadline) and close
                        await asyncio.wait_for(asyncio.shield(waiter), timeout=max(1.0, self._drain_remaining() or 0.0))
                    except asyncio.TimeoutError:
                        log.error("dynorch runner: runs still unfinished at the drain deadline; cancelling them "
                                  "(each is closed killed_by_drain)")
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        summary.reruns = sum(1 for r in summary.records.values() if r.spec.attempt > 1)
        summary.drained = summary.drained or self._draining.is_set()
        return summary

    def _existing_by_cell(self) -> dict[tuple[str, str, str, int, int], RunRecord]:
        return {_cell_key(record.spec): record for record in self.store.scan().values()}

    def _rerun_pending(self, record: RunRecord) -> bool:
        if record.status == "done":
            return False
        if record.rerun_reason in STOP_REASONS:
            return True                         # the account / the ledger refused: no model ran, no rerun consumed
        return record.rerun_reason in RERUN_REASONS and record.spec.attempt <= self.settings.max_reruns

    def _attempts_left(self, record: RunRecord) -> bool:
        """The frozen table's ``<= 2 reruns`` also bounds INTERRUPTED records (review minor 17): a cell killed by
        crash / scancel / drain-timeout over and over must not accumulate paid attempts a4, a5, ..."""
        if record.rerun_reason in STOP_REASONS:
            return True
        return record.spec.attempt <= self.settings.max_reruns

    def _exhaust_attempts(self, record: RunRecord) -> RunRecord:
        """Close a record that has used its reruns: terminal ``failed``, no further attempt, y stays NaN."""
        record.status = "failed"
        record.rerun_reason = None
        record.error = record.error or f"interrupted; attempt cap ({self.settings.max_reruns} reruns) reached"
        record.finished_at = record.finished_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.store.save(record)
        self._journal_attempt(record)
        log.warning("dynorch runner: %s/%s/%s has used its reruns (attempt %d); left failed", record.spec.arm,
                    record.spec.rung, record.spec.item_id, record.spec.attempt)
        return record

    def _next_attempt(self, previous: RunRecord, *, resume: bool) -> RunSpec:
        spec = previous.spec
        if not self.store.is_terminal(previous) and resume:
            self._supersede(previous)
        next_no = spec.attempt + 1
        return dataclasses.replace(spec, attempt=next_no, generation=spec.generation + (1 if resume else 0),
                                   run_key=planmod.run_key_for(str(self.cfg["study_id"]), spec.partition, spec.arm,
                                                               spec.item_id, spec.rung, spec.repeat, next_no))

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
            raw = (self.settings.run_root / "raw" / spec.partition / spec.arm / str(spec.rung) / str(spec.repeat)
                   / item_slug(spec.item_id) / f"a{spec.attempt}")
            if raw.exists():
                target = (self.settings.run_root / "superseded" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                          / raw.relative_to(self.settings.run_root))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(raw), str(target))
        previous.status = "failed"
        previous.rerun_reason = "interrupted"
        previous.error = previous.error or "interrupted before close (resume)"
        # save BEFORE archiving (review minor 20): archive_previous moves the file on disk, so an unsaved mutation
        # left runs_superseded/ claiming the attempt was still `running` while the attempts journal said `failed`
        self.store.save(previous)
        archive = getattr(self.store, "archive_previous", None)
        if archive is not None:
            archive(spec)
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
        """One attempt (module docstring). If the drain's last resort cancels it after ``open_run``, the attempt is
        closed ``failed`` / ``killed_by_drain`` before the cancellation propagates, so no record is left
        ``running`` -- which ``--resume`` would count as an interrupted attempt, burning a rerun (finding 5)."""
        started = time.monotonic()
        try:
            return await self._attempt_body(spec)
        except asyncio.CancelledError:
            try:
                opened = self.core.record_of(spec.run_key).status == "running"
            except Exception:  # noqa: BLE001 - never opened
                opened = False
            if opened:
                self._harness_gone(spec)
                await asyncio.shield(self._close_drained(spec, started))
            raise

    async def _attempt_body(self, spec: RunSpec) -> RunRecord:
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
        try:
            adapter.prepare(work.root, spec, proxy_base, system_text, settings.subagent_text, task_text)
            plan = adapter.build_command(work.root, spec, proxy_base)
            check_env(plan.env, work=work.root, run_key=spec.run_key)
            self._write_launch_json(work, spec, plan, wall)
        except Exception as exc:
            log.error("dynorch runner: %s refused before start: %s", spec.run_key, exc)
            self._harness_gone(spec)
            return await self._close_failed(spec, "refused_before_start", exc, started)
        killed_by_drain = False
        exhausted_kill = False
        try:
            if getattr(adapter, "in_process", False):
                result = await self._run_in_process(adapter, work, spec, proxy_base, wall)
                timed_out = result.timed_out
                exit_code = result.exit_code
            else:
                exit_code, timed_out, killed_by_drain, exhausted_kill = await self._run_subprocess(spec, plan, work,
                                                                                                   wall)
                result = adapter.parse_result(work.stdout.read_bytes() if work.stdout.exists() else b"",
                                              work.stderr.read_bytes() if work.stderr.exists() else b"", work.root)
        except Exception as exc:  # noqa: BLE001 - ENOENT on the binary, an adapter bug, a full disk: one cell,
            log.exception("dynorch runner: %s failed to run: %s", spec.run_key, exc)   # not the whole 12 h job (8)
            self._harness_gone(spec)
            return await self._close_failed(spec, "failed", exc, started)
        self._harness_gone(spec)
        wall_seconds = time.monotonic() - started
        await self._await_settled(spec)
        status, rerun_reason, error = self._status_of(spec, timed_out=timed_out, killed_by_drain=killed_by_drain)
        # re-review finding 3: the account refusal decides the run only if it is still the run's LAST upstream
        # outcome (the core clears the flag on any later served reply); a refusal the harness recovered from --
        # the root delivered an answer afterwards -- still drains the stage, but never discards that answer
        try:
            still_stopped = bool(int(self.core.budget_view(spec.run_key).get(INFRASTRUCTURE_STOP, 0)))
        except Exception:  # noqa: BLE001 - an older core without the flag: fall back to the record's error below
            still_stopped = None
        record: RunRecord = self.core.close_run(spec.run_key, status=status)
        if not isinstance(record, RunRecord):
            raise TypeError(f"core.close_run returned {type(record).__name__}, expected RunRecord")
        core_error = str(record.error or "")
        infrastructure_seen = core_error.startswith(f"{INFRASTRUCTURE_STOP}:")
        infrastructure_stop = infrastructure_seen if still_stopped is None else (still_stopped and infrastructure_seen)
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
        if infrastructure_stop:
            # plan B6 / Gate 3: an upstream 402/401 is an infrastructure stop, not a harness outcome. Draining here
            # keeps the stage from burning every remaining cell's attempts against a refusing account (major 11).
            record.error = core_error
            record.rerun_reason = INFRASTRUCTURE_STOP
            record.status = "transport_failed"
            log.error("dynorch runner: %s hit an infrastructure stop (%s); draining the stage", spec.run_key,
                      core_error)
            self.request_drain()
        elif rerun_reason == LEDGER_STOP:
            log.error("dynorch runner: %s was refused by the $ ledger (%s); pausing admissions (draining the stage)",
                      spec.run_key, core_error or error)
            self.request_drain()
        elif infrastructure_seen:
            record.error = core_error                 # recorded (reconciliation reports the 402) ...
            log.error("dynorch runner: %s recovered from an account refusal (%s); its outcome stands, the stage "
                      "drains", spec.run_key, core_error)
            self.request_drain()                      # ... and the stage stops admitting (Gate 3 stop rule)
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

    async def _await_settled(self, spec: RunSpec) -> None:
        """A call the harness abandoned (wall kill, watchdog, crash) is still running upstream and is already paid
        for: wait for the core to finish serving it -- and for the wire layer to park and record it -- before
        ``close_run``, so the record is finalised ONCE with its tokens counted and its delivery verdict known
        (review major 14; the proxy never cancels such a call, review major 5)."""
        deadline = time.monotonic() + self.settings.settle_wait_seconds
        remaining = self._drain_remaining()
        if remaining is not None:                      # a drain bounds every settle wait (finding 5)
            deadline = min(deadline, time.monotonic() + remaining)
        while time.monotonic() < deadline:
            try:
                view = self.core.budget_view(spec.run_key)
            except Exception:  # noqa: BLE001 - the run may already be gone
                return
            if int(view.get("in_flight", 0)) <= 0:
                break
            await asyncio.sleep(min(0.25, max(0.01, self.settings.poll_seconds)))
        else:
            log.warning("dynorch runner: %s still has a call in flight after %.0f s; closing anyway", spec.run_key,
                        self.settings.settle_wait_seconds)
        if self.wait_for_run is not None:
            try:
                await self.wait_for_run(spec.run_key, max(1.0, deadline - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 - never lose a run over the wait
                log.warning("dynorch runner: waiting for %s's parked calls failed: %s", spec.run_key, exc)

    def _harness_gone(self, spec: RunSpec) -> None:
        """Tell the core the harness process is gone (re-review finding 6): calls still queued for the run are
        refused instead of being granted and sent upstream for nobody."""
        notify = getattr(self.core, "harness_exited", None)
        if notify is not None:
            try:
                notify(spec.run_key)
            except Exception as exc:  # noqa: BLE001 - bookkeeping must never lose the run
                log.warning("dynorch runner: harness_exited(%s) failed: %s", spec.run_key, exc)

    async def _close_drained(self, spec: RunSpec, started: float) -> None:
        """Close an attempt the drain cancelled: ``failed`` / ``killed_by_drain`` (the frozen rule reruns it)."""
        try:
            record: RunRecord = self.core.close_run(spec.run_key, status="failed")
        except Exception:  # noqa: BLE001 - the core may already have closed it
            record = self.core.record_of(spec.run_key)
            record.status = "failed"
        record.rerun_reason = "killed_by_drain"
        record.error = record.error or "cancelled by the drain deadline"
        record.wall_seconds = round(time.monotonic() - started, 3)
        self.store.save(record)
        self._journal_attempt(record)

    async def _close_failed(self, spec: RunSpec, status: str, exc: BaseException, started: float) -> RunRecord:
        """Close an attempt that never reached its harness (or died outside the proxy) as a stored record instead
        of letting the exception escape into the stage's gather (review major 8). A call the proxy may still be
        serving for it is awaited first (re-review minor 13), exactly as on the normal path."""
        await self._await_settled(spec)
        try:
            record: RunRecord = self.core.close_run(spec.run_key, status=status)
        except Exception:  # noqa: BLE001 - the core may already have closed it
            record = self.core.record_of(spec.run_key)
            record.status = status
        record.error = f"{type(exc).__name__}: {exc}"
        record.wall_seconds = round(time.monotonic() - started, 3)
        calls = int(self.core.budget_view(spec.run_key).get("calls", 0))
        record.rerun_reason = "zero_proxy_calls_served" if calls == 0 and status != "refused_before_start" else None
        self.store.save(record)
        return record

    def _status_of(self, spec: RunSpec, *, timed_out: bool, killed_by_drain: bool) -> tuple[str, str | None, str | None]:
        """The frozen rerun rule, decided from what the proxy served (outcome-blind)."""
        view = self.core.budget_view(spec.run_key)
        calls = int(view["calls"])
        last = self.reply_log.last_event(spec.run_key)
        if killed_by_drain:
            return "failed", "killed_by_drain", "killed by the drain signal"
        if last is not None and last.kind == "error" and last.code == "ledger_refused":
            # the frozen rule makes a ledger error rerun-eligible; plan §5 makes ledger exposure PAUSE admissions:
            # the stage drains and the rerun happens on --resume, never burning the in-job reruns (finding 4)
            return "transport_failed", LEDGER_STOP, f"last proxy reply: {last.status} {last.code}"
        if last is not None and last.kind == "error" and (last.code in TRANSPORT_ERROR_CODES or last.status >= 500):
            return "transport_failed", "last_reply_transport_or_ledger_error", f"last proxy reply: {last.status} {last.code}"
        # NOTHING else makes a run transport_failed (review major 10): the frozen rule is "the LAST proxy reply was
        # a transport/ledger error". An earlier sub-agent failure the harness recovered from -- it delegated again,
        # the root answered -- must not discard a delivered answer, and a 402 after it never reruns.
        if calls == 0:
            status = "timeout" if timed_out else "failed"
            return status, "zero_proxy_calls_served", "harness served no proxy call"
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
        if proc.returncode in TIMEOUT_EXIT_CODES and not timed_out and not exhausted_kill and not killed_by_drain:
            # a `timeout`-wrapped argv (Codex) reports 124 for its own deadline and 137 for its kill-after SIGKILL:
            # the wall was reached, so the run is a timeout, not a `done` counted in valid_final_share (major 6)
            log.warning("dynorch runner: %s exited %d: the harness's own timeout wrapper fired", spec.run_key,
                        proc.returncode)
            timed_out = True
        return proc.returncode, timed_out, killed_by_drain, exhausted_kill

    async def _exhaustion_watchdog(self, spec: RunSpec, proc: asyncio.subprocess.Process) -> bool:
        """Plan §2 B4: SIGTERM the group ``sigterm_after_exhaustion`` s after the run's last SERVED reply once it is
        exhausted, SIGKILL ``sigkill_after`` s later -- and NEVER while the proxy has a call of this run in flight.

        The clock is the core's (``budget_view["idle_seconds"]``, restarted by every reply the proxy hands back,
        duplicates included), not the runner's view of the call counter: the TERMINAL call (B4) is admitted while
        ``remaining < 64`` already holds and takes as long as a 1,024-token answer needs, so a timer started at the
        grant would SIGTERM the harness during the one call whose purpose is the delivered answer.
        """
        settings = self.settings
        while proc.returncode is None:
            await asyncio.sleep(settings.poll_seconds)
            try:
                view = self.core.budget_view(spec.run_key)
            except Exception:  # noqa: BLE001 - the run may already be closing
                return False
            remaining = int(view["remaining"])
            exhausted = remaining < settings.terminal_threshold or bool(view.get("exhausted", False))
            if not exhausted:
                continue
            if int(view.get("in_flight", 0)) > 0:
                continue                      # a call is being served: never kill the harness under it
            idle = float(view.get("idle_seconds", 0.0))
            if idle >= settings.sigterm_after_exhaustion:
                log.info("dynorch runner: %s exhausted and idle %.1f s; SIGTERM (%.0f s after the last served reply)",
                         spec.run_key, idle, settings.sigterm_after_exhaustion)
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


def _cell_key(spec: RunSpec) -> tuple[str, str, str, int, int]:
    """The cell identity INCLUDING the partition (contract §10.22): the canary's k=16 ceiling probe and the pilot
    plan the same (arm, item, rung, repeat) and must not be confused for one another."""
    return spec.partition, spec.arm, spec.item_id, spec.rung, spec.repeat


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


# ======================================================================================================================
# Stage launch (integration: docs/dynorch/requests/wpD.md item 1) -- the one place the proxy stack is assembled
# ======================================================================================================================

CEILING_REFRESH_SECONDS = 300.0
SENTINEL_POLL_SECONDS = 5.0


@dataclass
class Stack:
    """The assembled proxy stack of one process: settings, ledger (+ its journal), call journal, run store,
    upstream and core. ``aclose`` flushes and releases the ledger journal (the client is the caller's)."""
    settings: Any
    ledger_journal: Any
    ledger: Any
    journal: Any
    store: Any
    upstream: Any
    core: Any
    inherited: dict[str, int] = field(default_factory=dict)

    async def aclose(self) -> None:
        await self.ledger_journal.flush()
        self.ledger_journal.release_writer()


async def build_stack(cfg: Mapping[str, Any], run_root: str | Path, *, client: Any, locks: Mapping[str, Any],
                      ceiling_usd: Decimal | str, active_ceiling_usd: Decimal | str | None = None,
                      group_commit_ms: int = 0, resolve: bool = True) -> Stack:
    """Assemble ``DynorchSettings`` -> ``LedgerJournal`` / ``CollectionLedger`` -> ``CallJournal`` / ``RunStore``
    -> ``Upstream`` -> ``ProxyCore`` (contract §3, §5) over a ``ForcedClient`` and the shard ``EndpointLock``s.
    With ``resolve`` the inherited SENT attempts of an earlier process are settled from the call journal FIRST
    (``journal.resolve_inherited``; WP-A R-A7: before any supersede / archive), so a resumed process never starts
    with open reservations. ``active_ceiling_usd`` (``live.active_ceiling``) lowers the ledger's ceiling."""
    from agent_allocation.api.forced import ForcedClient
    from agent_allocation.dynorch.config import DynorchSettings
    from agent_allocation.dynorch.core import ProxyCore
    from agent_allocation.dynorch.journal import CallJournal, RunStore, resolve_inherited
    from agent_allocation.dynorch.upstream import Upstream
    from agent_allocation.runtime.ledger import CollectionLedger
    from agent_allocation.storage.journal import LedgerJournal

    if not isinstance(client, ForcedClient):
        raise TypeError(f"build_stack needs a ForcedClient (raw /completions), got {type(client).__name__}")
    if not locks:
        raise ValueError("build_stack needs at least one shard EndpointLock (locks/openrouter_endpoints.json)")
    root = Path(run_root)
    settings = DynorchSettings.from_config(cfg)
    (root / "state").mkdir(parents=True, exist_ok=True)
    ledger_journal = LedgerJournal(root / "state" / "ledger_journal.jsonl", group_commit_ms=group_commit_ms)
    ledger = CollectionLedger(ledger_journal, ceiling_usd=Decimal(str(ceiling_usd)),
                              max_unresolved_usd=settings.max_unresolved_spend_usd)
    if active_ceiling_usd is not None:
        ledger.set_active_ceiling(Decimal(str(active_ceiling_usd)))
    journal = CallJournal(root)
    store = RunStore(root)
    inherited = await resolve_inherited(ledger, store, journal) if resolve else {}
    upstream = Upstream(client, dict(locks), cfg["sampling"], settings.provider_in_flight, settings.provider_rps,
                        ledger, reservation_multiplier=settings.reservation_multiplier)
    core = ProxyCore(settings, upstream, journal, store)
    return Stack(settings=settings, ledger_journal=ledger_journal, ledger=ledger, journal=journal, store=store,
                 upstream=upstream, core=core, inherited=dict(inherited))


def ledger_block_reason(ledger: Any) -> str | None:
    """Plan §5: "unresolved exposure > $2 pauses admissions"; a ledger with no headroom left under its active
    ceiling cannot admit a call either. ``None`` when a new cell may start (re-review finding 4)."""
    if getattr(ledger, "unresolved_exceeded", False):
        return "unresolved_exposure"
    fits = getattr(ledger, "fits", None)
    if fits is not None and not fits(Decimal("0")):
        return "ceiling"
    return None


def apply_provider_map(inputs: BaseInputs, run_root: str | Path) -> BaseInputs:
    """Re-point every item's shard at the frozen provider decision (``locks/provider_map.json``, plan §4: after a
    switch every item runs on the passing shard's endpoint; ``forced_v2_shard`` stays the covariate). A missing or
    foreign-layout map raises: the endpoint rule must have been frozen before a stage starts."""
    path = Path(run_root) / "locks" / "provider_map.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: freeze the endpoint rule (dynorch-run resolves it at the canary)")
    doc = json.loads(path.read_text("utf-8"))
    if doc.get("layout") != "dynorch/v1" or not isinstance(doc.get("items"), Mapping):
        raise ValueError(f"{path} is not a dynorch/v1 provider map")
    assignment: dict[str, str] = {}
    tags: dict[str, str] = dict(inputs.endpoint_tags)
    for item_id, row in doc["items"].items():
        shard_id, tag = row["shard_id"], row["endpoint_tag"]
        if not isinstance(shard_id, str) or not isinstance(tag, str):
            raise ValueError(f"{path}: item {item_id} has a malformed row {row!r}")
        assignment[item_id] = shard_id
        tags[shard_id] = tag
    missing = sorted(set(inputs.assignment) - set(assignment))
    if missing:
        raise ValueError(f"{path}: {len(missing)} base items have no provider row (e.g. {missing[:3]})")
    return dataclasses.replace(inputs, assignment=assignment, endpoint_tags=tags)


def specs_for_cells(cfg: Mapping[str, Any], stage: str, inputs: BaseInputs, cells: Sequence[Mapping[str, Any]], *,
                    proxy_version: str, harness_versions: Mapping[str, str] | None = None) -> list[RunSpec]:
    """The stage's planned ``RunSpec``s (``plan.plan_stage``, in plan order) restricted to ``cells``
    (``{arm, rung, item_id, repeat, ...}`` from ``cli.enumerate_cells``); a requested cell the plan does not
    contain raises -- the two enumerations must agree cell for cell (Gate 4 checks planned cells against rows)."""
    wanted = {(str(c["arm"]), int(c["rung"]), str(c["item_id"]), int(c.get("repeat", 1))) for c in cells}
    planned = planmod.plan_stage(cfg, stage, inputs, proxy_version=proxy_version, harness_versions=harness_versions)
    keyed = {(s.arm, s.rung, s.item_id, s.repeat): s for s in planned}
    unknown = sorted(wanted - set(keyed))
    if unknown:
        raise ValueError(f"{len(unknown)} requested cell(s) are not in the {stage} plan, e.g. {unknown[:3]}")
    return [s for s in planned if (s.arm, s.rung, s.item_id, s.repeat) in wanted]


def summary_dict(summary: StageSummary, ledger: Any, *, inherited: Mapping[str, int] | None = None,
                 upstream_402: bool = False) -> dict[str, Any]:
    """The JSON-able stage summary ``dynorch-run`` records in ``status.json`` (requests/wpD.md item 1)."""
    records = list(summary.records.values())
    infra_402 = any(str(r.error or "").startswith("infrastructure_stop:") for r in records)
    snap = ledger.snapshot()
    counts = {status: sum(1 for r in records if r.status == status) for status in sorted(TERMINAL_STATUSES)}
    return {"stage": summary.stage, "runs": summary.planned, "executed": summary.executed, "skipped": summary.skipped,
            "reruns": summary.reruns, "errors": summary.errors, "drained": summary.drained, **counts,
            "upstream_402": bool(upstream_402 or infra_402), "settled_usd": str(snap.settled),
            "in_flight_usd": str(snap.in_flight), "unresolved_usd": str(snap.unresolved),
            "inherited": dict(inherited or {})}


async def launch_stage(cfg: Mapping[str, Any], paths: Any, *, stage: str, cells: Sequence[Mapping[str, Any]],
                       max_usd: Decimal, active_ceiling_usd: Decimal, resume: bool, concurrency: int | None,
                       client: Any, account_auto_reload: bool, proxy_host: str, proxy_port: int,
                       on_bound: Any) -> dict[str, Any]:
    """Bind the proxy and run one stage end to end (the entry point ``dynorch-run`` calls; requests/wpD.md item 1).

    Order: locks (``endpoint_rule.read_locks``) and the provider map -> ``build_stack`` (ledger ceiling ``max_usd``
    lowered to ``active_ceiling_usd``; inherited attempts resolved) -> ``httpd.ProxyServer`` bound on
    ``proxy_host:proxy_port`` and ``on_bound(host, port)`` BEFORE the first harness -> ``Runner.run_specs`` over the
    stage's cells (``--resume`` skips terminal records) with the Slurm drain trap, a ``control/STOP`` /
    ``control/DRAIN`` sentinel watch, and a ``/key`` refresh of the active ceiling every 5 minutes (plan §5; a 402
    from the account endpoints drains the stage and is reported as ``upstream_402``). Returns ``summary_dict``.
    The client is the caller's (closed by ``dynorch-run``); the ledger journal is flushed and released here.
    """
    from agent_allocation.dynorch import endpoint_rule, reconcile
    from agent_allocation.dynorch.harness import adapters_from_config
    from agent_allocation.dynorch.httpd import ProxyServer
    from agent_allocation.live import active_ceiling

    run_root = Path(paths.run_dir)
    for sentinel in (Path(paths.stop_sentinel), Path(paths.drain_sentinel)):
        if sentinel.exists():
            # review minor 22: an unconsumed sentinel drained every later job at its first poll, executing nothing
            raise RunRefused(f"control sentinel {sentinel} is present: remove it before starting a stage "
                             f"(`rm {sentinel}`) -- it would drain this job immediately")
    locks = endpoint_rule.read_locks(run_root)
    inputs = apply_provider_map(BaseInputs.load(cfg["base_run_root"], model=cfg["primary_model"]), run_root)
    stack = await build_stack(cfg, run_root, client=client, locks=locks, ceiling_usd=Decimal(str(max_usd)),
                              active_ceiling_usd=Decimal(str(active_ceiling_usd)))
    server = ProxyServer(stack.core, host=proxy_host, port=int(proxy_port),
                         keepalive_seconds=float(stack.settings.keepalive_seconds))
    flags = {"upstream_402": False}
    try:
        await server.start()
        on_bound(server.host, server.port)
        adapters = adapters_from_config(cfg)
        settings = RunnerSettings.from_config(cfg, run_root=run_root, concurrency=concurrency)
        runner = Runner(stack.core, adapters, settings, proxy_url=server.base_url, store=stack.store, inputs=inputs,
                        reply_log=server.reply_log, cfg=cfg, journal=stack.journal,
                        wait_for_run=server.wait_for_run, admission_check=lambda: ledger_block_reason(stack.ledger))
        try:
            install_drain_handler(runner)
        except (ValueError, RuntimeError, NotImplementedError) as exc:     # not the main thread / no signals
            log.warning("dynorch runner: drain handler not installed (%s)", exc)
        specs = specs_for_cells(cfg, stage, inputs, cells, proxy_version=stack.core.proxy_version,
                                harness_versions=runner.harness_versions())
        # the pre-registration is ALWAYS the stage's full plan, whatever --arms/--rungs filter this launch uses
        full_plan = planmod.plan_stage(cfg, stage, inputs, proxy_version=stack.core.proxy_version,
                                       harness_versions=runner.harness_versions())
        # a superseded stage re-executes under a higher execution generation (contract §10.35): same run_keys and
        # pre-registration, new ledger attempt ids
        from agent_allocation.dynorch.supersede import apply_generation, execution_generation
        specs = apply_generation(specs, execution_generation(run_root, stage))

        async def refresh_ceiling() -> None:
            while True:
                await asyncio.sleep(CEILING_REFRESH_SECONDS)
                try:
                    snap = await reconcile.snapshot_account(client)
                except Exception as exc:  # noqa: BLE001 - a failed refresh keeps the last ceiling
                    log.warning("dynorch runner: /key refresh failed (%s); ceiling unchanged", exc)
                    continue
                if snap.payment_required:
                    flags["upstream_402"] = True
                    log.error("dynorch runner: account endpoints answered 402; draining the stage (B6 stop_notify)")
                    runner.request_drain()
                    return
                balance = (snap.total_credits_usd or Decimal(0)) - (snap.total_usage_usd or Decimal(0))
                ceiling = active_ceiling(Decimal(str(max_usd)), stack.ledger.snapshot().settled,
                                         {"key": {"limit_remaining": snap.key_limit_remaining_usd},
                                          "account_balance_usd": str(balance)},
                                         account_auto_reload=bool(account_auto_reload))
                stack.ledger.set_active_ceiling(ceiling)

        async def watch_sentinels() -> None:
            stop, drain = Path(paths.stop_sentinel), Path(paths.drain_sentinel)
            while True:
                await asyncio.sleep(SENTINEL_POLL_SECONDS)
                if stop.exists() or drain.exists():
                    log.warning("dynorch runner: control sentinel present (%s); draining", stop if stop.exists() else drain)
                    runner.request_drain()
                    return

        background = [asyncio.create_task(refresh_ceiling()), asyncio.create_task(watch_sentinels())]
        try:
            summary = await runner.run_specs(stage, specs, resume=resume, registered=full_plan)
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
    finally:
        await server.close()
        await stack.aclose()
    return summary_dict(summary, stack.ledger, inherited=stack.inherited, upstream_402=flags["upstream_402"])
