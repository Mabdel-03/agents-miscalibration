"""WP-B runner rules that the review pass pinned: the exhaustion watchdog's clock (plan §2 B4), the outcome-blind
rerun decision (frozen table), the attempt cap on interrupted records, the drain's grace period and the
infrastructure-stop drain. No subprocess is spawned: the watchdog and the drain are driven against a process
double, everything else against an in-memory store."""
from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.dynorch import runner as runnermod
from agent_allocation.dynorch.httpd import ReplyLog, ServedError, ServedReply
from agent_allocation.dynorch.records import RunRecord, RunSpec, new_run_record

from .conftest import make_spec

PROJECT = Path(__file__).resolve().parents[3]


class FakeProc:
    """Enough of ``asyncio.subprocess.Process`` for the watchdog / drain paths."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = -424242
        self.signals: list[int] = []

    async def wait(self) -> int | None:
        while self.returncode is None:
            await asyncio.sleep(0.005)
        return self.returncode


class FakeCore:
    proxy_version = "dynorch-proxy-1.0.0"

    def __init__(self, view: dict[str, Any]) -> None:
        self.view = view

    def budget_view(self, run_key: str) -> dict[str, Any]:
        return dict(self.view)


def make_settings(**over: Any) -> runnermod.RunnerSettings:
    base = dict(run_root=Path("/nonexistent/run"), scratch_root=Path("/nonexistent/scratch"), model="m",
                concurrency=1, wall_base=600, wall_per_rung=120, sigterm_after_exhaustion=0.3, sigkill_after=0.2,
                terminal_threshold=64, max_reruns=2, system_text="s", subagent_text="s", h0_system_text="s",
                user_template="{task_prompt}{output_requirement}", budget_sentence="{C}", poll_seconds=0.02,
                drain_wait_seconds=1.0)
    base.update(over)
    return runnermod.RunnerSettings(**base)


def make_runner(core: Any, settings: runnermod.RunnerSettings, *, store: Any = None,
                reply_log: ReplyLog | None = None) -> runnermod.Runner:
    return runnermod.Runner(core, {}, settings, proxy_url="http://127.0.0.1:1", store=store,
                            inputs=None, reply_log=reply_log or ReplyLog(), cfg={"study_id": "openrouter_dynorch_v1"})


# ---- B4: the exhaustion watchdog -------------------------------------------------------------------------------

async def test_watchdog_never_fires_while_a_call_is_in_flight_and_counts_from_the_last_served_reply(monkeypatch):
    """Plan §2 B4: 'SIGTERM ... 60 s after the run's last served reply'. The TERMINAL call is admitted while
    ``remaining < 64`` already holds and may take a minute to produce the answer being graded, so the watchdog
    must not fire while the proxy has a call of the run in flight, and its clock must be the last served reply."""
    view = {"cap": 1024, "spent": 1000, "remaining": 24, "calls": 3, "grace_tokens": 0, "prompt_total": 10,
            "exhausted": 0, "in_flight": 1, "idle_seconds": 99.0, "replies_served": 2}
    proc = FakeProc()
    killed: list[int] = []

    def fake_signal(target: Any, sig: int) -> None:
        killed.append(sig)
        target.returncode = -sig

    monkeypatch.setattr(runnermod, "_signal_group", fake_signal)
    runner = make_runner(FakeCore(view), make_settings())
    task = asyncio.create_task(runner._exhaustion_watchdog(make_spec(rung=1), proc))
    await asyncio.sleep(0.3)                      # 10x sigterm_after_exhaustion worth of polls, call in flight
    assert killed == [] and not task.done(), "the watchdog killed the harness during an in-flight call"
    view["in_flight"] = 0                         # the reply landed: the clock restarts
    view["idle_seconds"] = 0.0
    await asyncio.sleep(0.15)
    assert killed == [], "the watchdog fired before sigterm_after_exhaustion elapsed since the reply"
    view["idle_seconds"] = 5.0
    assert await asyncio.wait_for(task, timeout=2.0) is True
    assert killed and killed[0] == 15


async def test_watchdog_does_not_fire_while_the_run_still_has_budget(monkeypatch):
    view = {"cap": 4096, "spent": 100, "remaining": 3996, "calls": 1, "grace_tokens": 0, "prompt_total": 10,
            "exhausted": 0, "in_flight": 0, "idle_seconds": 500.0, "replies_served": 1}
    proc = FakeProc()
    monkeypatch.setattr(runnermod, "_signal_group", lambda *a: pytest.fail("watchdog fired with budget left"))
    runner = make_runner(FakeCore(view), make_settings())
    task = asyncio.create_task(runner._exhaustion_watchdog(make_spec(rung=4), proc))
    await asyncio.sleep(0.15)
    proc.returncode = 0
    assert await asyncio.wait_for(task, timeout=2.0) is False


# ---- the store is always consulted; attempts are bounded (review majors 4/15, minor 17) ------------------------

def _store_with(tmp_path: Path, record: RunRecord):
    from agent_allocation.dynorch.journal import RunStore

    store = RunStore(tmp_path)
    store.save(record)
    return store


def _record(spec: RunSpec, *, status: str, rerun_reason: str | None = None, error: str | None = None) -> RunRecord:
    record = new_run_record(spec, status=status, started_at="2026-09-23T00:00:00Z")
    record.status = status
    record.rerun_reason = rerun_reason
    record.error = error
    if status != "running":
        record.finished_at = "2026-09-23T00:05:00Z"
    return record


async def test_a_finished_cell_is_skipped_even_without_resume(tmp_path: Path):
    """Review major 4: re-running a stage without --resume must not re-execute (and overwrite in place, and
    conflict in the write-once journal after paying for) a cell that already has a paid, terminal record."""
    spec = make_spec(rung=4, partition="canary")
    store = _store_with(tmp_path, _record(spec, status="done"))
    runner = make_runner(FakeCore({}), make_settings(run_root=tmp_path), store=store)
    summary = await runner.run_specs("canary", [spec], resume=False)
    assert summary.skipped == 1 and summary.executed == 0
    assert store.load(spec).status == "done"


async def test_an_unfinished_cell_without_resume_is_refused_loudly(tmp_path: Path):
    """Review major 15: continuing an interrupted or rerun-eligible cell needs --resume (which resolves the
    inherited ledger attempts and bumps the execution generation); silently re-running it as attempt 1 collides
    with the ledger attempt ids and the call journal AFTER the upstream call was paid."""
    from agent_allocation.live import RunRefused

    spec = make_spec(rung=4, partition="canary")
    store = _store_with(tmp_path, _record(spec, status="running"))
    runner = make_runner(FakeCore({}), make_settings(run_root=tmp_path), store=store)
    with pytest.raises(RunRefused, match="--resume"):
        await runner.run_specs("canary", [spec], resume=False)
    killed = make_spec(rung=8, partition="canary", item_id="aime/2025/2")
    store2 = _store_with(tmp_path / "b", _record(killed, status="failed", rerun_reason="killed_by_drain"))
    runner2 = make_runner(FakeCore({}), make_settings(run_root=tmp_path / "b"), store=store2)
    with pytest.raises(RunRefused):
        await runner2.run_specs("canary", [killed], resume=False)


async def test_an_interrupted_record_beyond_the_rerun_cap_is_not_executed_again(tmp_path: Path):
    """Review minor 17: only terminal records went through the ``<= 2 reruns`` cap, so a cell interrupted over and
    over (crash, scancel, drain timeout) accumulated paid attempts a4, a5, ... The cap now applies to both."""
    spec = dataclasses.replace(make_spec(rung=4, partition="main"), attempt=3, generation=3)
    store = _store_with(tmp_path, _record(spec, status="running"))
    runner = make_runner(FakeCore({}), make_settings(run_root=tmp_path, max_reruns=2), store=store)
    summary = await runner.run_specs("main", [spec], resume=True)
    assert summary.executed == 0 and summary.skipped == 1
    final = store.load(spec)
    assert final.status == "failed" and final.rerun_reason is None and "attempt cap" in (final.error or "")
    assert final.correct is None                      # y stays NaN


async def test_an_infrastructure_stop_is_rerun_without_consuming_a_rerun(tmp_path: Path):
    """Plan B6 / Gate 3 (review major 11): an upstream 402/401 is an infrastructure stop -- no model ran, nothing
    was paid -- so the cell stays rerun-eligible on the next --resume however many times the account refused."""
    spec = dataclasses.replace(make_spec(rung=4, partition="main"), attempt=4, generation=4)
    record = _record(spec, status="transport_failed", rerun_reason="infrastructure_stop",
                     error="infrastructure_stop: payment_required")
    runner = make_runner(FakeCore({}), make_settings(run_root=tmp_path, max_reruns=2),
                         store=_store_with(tmp_path, record))
    assert runner._rerun_pending(record) is True
    assert runner._attempts_left(record) is True
    spent = _record(dataclasses.replace(spec, attempt=4), status="transport_failed",
                    rerun_reason="last_reply_transport_or_ledger_error", error="upstream_failed: ambiguous_timeout")
    assert runner._rerun_pending(spent) is False      # a paid transport failure still obeys the cap


async def test_a_timeout_wrapped_argv_that_fires_is_recorded_as_a_timeout(tmp_path: Path):
    """Review major 6: Codex runs under ``timeout --kill-after=Ns Ws``. When that wrapper fires (exit 124, or 137
    after its own SIGKILL) the wall WAS reached, so the run is a timeout -- it used to be recorded ``done`` and
    counted in valid_final_share, leaving Gate 2's timeout_share and the Gate 3 stop rule blind for H3."""
    from agent_allocation.dynorch.harness.base import LaunchPlan, WorkDir

    work = WorkDir(tmp_path / "w").create()
    view = {"cap": 4096, "spent": 0, "remaining": 4096, "calls": 0, "grace_tokens": 0, "prompt_total": 0,
            "exhausted": 0, "in_flight": 0, "idle_seconds": 0.0}
    runner = make_runner(FakeCore(view), make_settings(run_root=tmp_path))
    plan = LaunchPlan(argv=("timeout", "--kill-after=2s", "1s", "sleep", "60"), env={"PATH": os.environ["PATH"]},
                      cwd=work.ws, stdin=None)
    exit_code, timed_out, killed_by_drain, exhausted = await runner._run_subprocess(make_spec(rung=4), plan, work, 60)
    assert exit_code == 124 and timed_out is True and not killed_by_drain and not exhausted


async def test_admissions_pause_when_the_ledger_has_no_headroom(tmp_path: Path):
    """Re-review finding 4 / plan §5 ("unresolved exposure > $2 pauses admissions"): no cell is started while the
    $ ledger cannot admit a call; the stage drains instead of opening runs that are refused at their first call."""
    from decimal import Decimal

    class Ledger:
        unresolved_exceeded = False

        def __init__(self, headroom: bool) -> None:
            self.headroom = headroom

        def fits(self, amount: Decimal) -> bool:
            return self.headroom

    assert runnermod.ledger_block_reason(Ledger(True)) is None
    assert runnermod.ledger_block_reason(Ledger(False)) == "ceiling"
    exposed = Ledger(True)
    exposed.unresolved_exceeded = True
    assert runnermod.ledger_block_reason(exposed) == "unresolved_exposure"
    from agent_allocation.dynorch.journal import RunStore

    runner = runnermod.Runner(FakeCore({}), {}, make_settings(run_root=tmp_path), proxy_url="http://127.0.0.1:1",
                              store=RunStore(tmp_path), inputs=None, reply_log=ReplyLog(),
                              cfg={"study_id": "openrouter_dynorch_v1"},
                              admission_check=lambda: runnermod.ledger_block_reason(Ledger(False)))
    summary = await runner.run_specs("main", [make_spec(rung=4, partition="main")])
    assert summary.executed == 0 and summary.errors == 0 and summary.drained is True


def test_the_whole_drain_fits_inside_slurms_900_second_warning():
    """Re-review finding 5: 800 s of drain wait + ~60 s terminating + a 90 s shielded wait + 60 s wait_idle + the
    reconciliation overran ``--signal=B:TERM@900``. The drain now ends by ``drain_total_seconds`` and the CLI cuts
    the /key settle wait of a drained stage, so the worst case fits with margin."""
    from agent_allocation.dynorch import cli, reconcile

    settings = make_settings(drain_wait_seconds=runnermod.RunnerSettings.drain_wait_seconds,
                             drain_total_seconds=runnermod.RunnerSettings.drain_total_seconds)
    server_close = 60.0                                   # ProxyServer.close(drain_seconds=60)
    generation_fetch = 6 * 5.0                            # reconcile.fetch_generations worst case
    worst = settings.drain_total_seconds + server_close + cli.DRAINED_KEY_SETTLE_MAX_S + generation_fetch
    assert settings.drain_wait_seconds + settings.sigkill_after <= settings.drain_total_seconds
    assert worst <= 900 - 60, worst
    assert reconcile.KEY_SETTLE_MAX_S > cli.DRAINED_KEY_SETTLE_MAX_S


async def test_a_drain_cancelled_attempt_is_closed_killed_by_drain(tmp_path: Path):
    """Re-review finding 5: the drain's last resort cancelled attempts that were still running; their records stayed
    ``running``, became ``interrupted`` on --resume and burned an attempt. A cancelled attempt is now closed
    ``failed`` / ``killed_by_drain`` (the frozen rule reruns it) before the cancellation propagates."""
    from agent_allocation.dynorch.journal import RunStore

    spec = make_spec(rung=4, partition="main")
    record = _record(spec, status="running")

    class OpenCore(FakeCore):
        closed: list[str] = []
        exited: list[str] = []

        def record_of(self, run_key: str) -> RunRecord:
            return record

        def close_run(self, run_key: str, *, status: str) -> RunRecord:
            record.status = status
            self.closed.append(status)
            return record

        def harness_exited(self, run_key: str) -> None:
            self.exited.append(run_key)

    class Stuck(runnermod.Runner):
        async def _attempt_body(self, spec: RunSpec) -> RunRecord:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    store = RunStore(tmp_path)
    core = OpenCore({})
    runner = Stuck(core, {}, make_settings(run_root=tmp_path), proxy_url="http://127.0.0.1:1", store=store,
                   inputs=None, reply_log=ReplyLog(), cfg={"study_id": "openrouter_dynorch_v1"})
    task = asyncio.create_task(runner._attempt(spec))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = store.load(spec)
    assert core.closed == ["failed"] and core.exited == [spec.run_key]
    assert saved.status == "failed" and saved.rerun_reason == "killed_by_drain"
    assert runner._rerun_pending(saved)


def test_run_closed_refusals_never_decide_the_rerun_rule():
    """Contract §10.33: a ``run_closed`` refusal answers a call made after the harness exited; nobody is listening,
    so the rerun rule reads the last event the harness actually got."""
    log = ReplyLog()
    log.events["rk"] = [ServedError(status=502, code="upstream_failed", wire="chat"),
                        ServedError(status=409, code="run_closed", wire="chat")]
    assert log.last_event("rk").code == "upstream_failed"


async def test_close_failed_waits_for_the_call_in_flight_first(tmp_path: Path):
    """Re-review minor 13: ``_close_failed`` (an adapter that raised after the harness was launched) closed the
    record while a call of the run could still be in flight, so its tokens and delivery verdict were lost. It now
    awaits the settle exactly like the normal path."""
    from agent_allocation.dynorch.journal import RunStore

    spec = make_spec(rung=4, partition="main")
    record = _record(spec, status="running")
    view = {"calls": 1, "in_flight": 1}
    order: list[str] = []

    class SettlingCore(FakeCore):
        def close_run(self, run_key: str, *, status: str) -> RunRecord:
            order.append(f"close(in_flight={view['in_flight']})")
            record.status = status
            return record

        def record_of(self, run_key: str) -> RunRecord:
            return record

    runner = make_runner(SettlingCore(view), make_settings(run_root=tmp_path), store=RunStore(tmp_path))

    async def land() -> None:
        await asyncio.sleep(0.2)
        view["in_flight"] = 0
        order.append("landed")

    lander = asyncio.create_task(land())
    closed = await runner._close_failed(spec, "failed", RuntimeError("adapter bug"), 0.0)
    await lander
    assert order == ["landed", "close(in_flight=0)"] and closed.status == "failed"
