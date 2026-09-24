"""WP-B runner rules that the review pass pinned: the exhaustion watchdog's clock (plan §2 B4), the outcome-blind
rerun decision (frozen table), the attempt cap on interrupted records, the drain's grace period and the
infrastructure-stop drain. No subprocess is spawned: the watchdog and the drain are driven against a process
double, everything else against an in-memory store."""
from __future__ import annotations

import asyncio
import json
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
