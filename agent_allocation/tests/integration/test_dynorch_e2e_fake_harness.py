"""End-to-end: httpd + the REAL ``ProxyCore`` (ledger, journal, store, Jinja2 render, budget) over ``api/fake.py``'s
upstream + the runner + the fake harness subprocess, $0 (contract §3, §5, §6; plan Verification).

The stack is assembled exactly as ``dynorch-run`` assembles it (``runner.build_stack``); the only substitutions are
the fake OpenRouter transport (scripted replies, delays and faults, per node or per item) and the fake harness
CLI (an OpenClaude-shaped SSE client with one ``spawn_subagent`` tool). Cases:

* three items (one per math benchmark + one GPQA) × rungs {1, 4} × arms {HF fake harness, H0 null}: every planned
  cell ends ``done`` with a graded record, the delivered answer comes from the proxy (``final_source ==
  "proxy_root"``), the CLI result cross-checks, ``launch.json`` / work-dir layout / attempts journal / raw call
  journal exist and carry no key material, the pre-registration file has one line per cell, and one scripted
  delegation runs through the real proxy (root -> solver -> root, reasoning re-injected by the cache);
* kill / resume: a drain request mid-stage SIGTERMs the in-flight harness (``killed_by_drain``), the next
  ``run_specs(resume=True)`` (a fresh stack over the same run root: inherited ledger attempts resolved first)
  re-executes only that cell as attempt 2 / generation 2 and skips the finished ones;
* the rerun rule: an ambiguous upstream outcome (read timeout) on the first attempt reruns once (attempt 2
  succeeds); zero proxy calls rerun up to ``max_reruns`` then leave an infra failure with y = NaN; a 402 served
  to a sub-agent after the root's call never reruns;
* 330 s and 660 s delayed upstream replies (``@pytest.mark.slow``, ``@pytest.mark.timeout(1500)``): headers +
  keepalive comments keep the fake harness on one request and ``subtype: success``;
* the exhaustion watchdog terminates a harness that lingers after its budget is spent.
"""
from __future__ import annotations

import asyncio
import json
import math
import socket
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.api import fake as fakeapi
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_tool_call_text
from agent_allocation.api.forced import ForcedClient, ForcedSettings
from agent_allocation.dynorch.harness.fake import FakeAdapter
from agent_allocation.dynorch.harness.null import NullAdapter
from agent_allocation.dynorch.httpd import ProxyServer, ReplyLog
from agent_allocation.dynorch.plan import BaseInputs, plan_runs, plan_path, read_plan, run_key_for
from agent_allocation.dynorch.runner import Runner, RunnerSettings, build_stack, summary_dict

PROJECT = Path(__file__).resolve().parents[2]
CFG = json.loads((PROJECT / "configs" / "dynorch_v1.json").read_text("utf-8"))
_REAL_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """The integration conftest forbids every AF_INET connect; this stack IS a loopback server, so allow 127.0.0.1
    and keep refusing anything else (no network, no key -- the upstream is the in-process fake transport)."""
    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"dynorch e2e must not leave loopback (tried {address!r})")
        return _REAL_CONNECT(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def _cfg_with_fake_arm() -> dict[str, Any]:
    cfg = json.loads(json.dumps(CFG))
    cfg["dynorch"]["arms"]["HF"] = {"harness": "fake", "spawn_tools": ["spawn_subagent"],
                                    "allowed_tools": ["spawn_subagent"], "root_call_ceiling": None}
    return cfg


@pytest.fixture(scope="module")
def base() -> BaseInputs:
    root = Path(CFG["base_run_root"])
    if not (root / "manifests" / "panels.json").is_file():
        pytest.skip("base run root not readable here")
    return BaseInputs.load(root, model=CFG["primary_model"])


def _items(base: BaseInputs) -> list[str]:
    return [base.panels["aime"].main_item_ids[0], base.panels["hmmt"].main_item_ids[0],
            base.panels["gpqa_diamond"].main_item_ids[0]]


SPAWN_REPLY = FakeReply(content=fake_tool_call_text("spawn_subagent", {"task": "solve the whole problem"},
                                                    prefix="Let me delegate."), think_tokens=200)


class Harness:
    """One wired stack: the real core over the fake upstream, httpd, store, runner settings, adapters, work under
    tmp. ``script`` is the fake upstream's script (replies per item via ``set_reply`` or per run via ``by_node``)."""

    def __init__(self, tmp: Path, cfg: dict[str, Any], *, script: FakeScript | None = None, concurrency: int = 2,
                 max_reruns: int | None = None, drain_wait_seconds: float | None = None,
                 ceiling_usd: str = "95") -> None:
        self.tmp = tmp
        self.ceiling_usd = ceiling_usd
        self.cfg = cfg
        self.run_root = tmp / "run_root"
        self.scratch = tmp / "scratch"
        self.reply_log = ReplyLog()
        self.script = script if script is not None else FakeScript()
        forced_cfg = dict(cfg)
        forced_cfg["provider"] = {**cfg["provider"], "endpoint_tag": "parasail/fp8"}
        self.client = ForcedClient(settings=ForcedSettings.from_config(forced_cfg), api_key=None,
                                   transport=fakeapi.fake_openrouter_transport(self.script),
                                   answer_backoff_base_seconds=0.0,
                                   read_timeout=float(cfg["scheduler"]["read_timeout_seconds"]))
        self.locks = {"parasail": fakeapi.fake_forced_endpoint_lock(tag="parasail/fp8"),
                      "mancer": fakeapi.fake_forced_endpoint_lock(tag="mancer/fp8")}
        self.settings = RunnerSettings.from_config(cfg, run_root=self.run_root, project_root=PROJECT,
                                                   scratch_root=self.scratch, concurrency=concurrency)
        if max_reruns is not None:
            self.settings = RunnerSettings(**{**self.settings.__dict__, "max_reruns": max_reruns})
        self.settings = RunnerSettings(**{**self.settings.__dict__, "poll_seconds": 0.2, "sigkill_after": 2.0,
                                          "sigterm_after_exhaustion": 1.0})
        if drain_wait_seconds is not None:
            self.settings = RunnerSettings(**{**self.settings.__dict__, "drain_wait_seconds": drain_wait_seconds})
        self.adapters = {"fake": FakeAdapter(model=cfg["primary_model"], python=sys.executable),
                         "null": NullAdapter(model=cfg["primary_model"])}
        self.stack: Any = None

    async def __aenter__(self) -> "Harness":
        self.stack = await build_stack(self.cfg, self.run_root, client=self.client, locks=self.locks,
                                       ceiling_usd=self.ceiling_usd)
        self.core, self.store, self.journal = self.stack.core, self.stack.store, self.stack.journal
        self.server = ProxyServer(self.core, keepalive_seconds=1.0, reply_log=self.reply_log)
        await self.server.start()
        self.runner = Runner(self.core, self.adapters, self.settings, proxy_url=self.server.base_url, store=self.store,
                             inputs=None, reply_log=self.reply_log, cfg=self.cfg, journal=self.journal,
                             wait_for_run=self.server.wait_for_run)
        return self

    def with_inputs(self, base: BaseInputs) -> "Harness":
        self.runner.inputs = base
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.server.close()
        await self.client.aclose()
        await self.stack.aclose()

    def specs(self, base: BaseInputs, *, arms=("HF", "H0"), rungs=(1, 4), items=None):
        return plan_runs(self.cfg, stage="main", arms=list(arms), rungs=list(rungs), repeats=[1],
                         items=items or _items(base), base=base, proxy_version=self.core.proxy_version)

    def attempt_key(self, spec, attempt: int) -> str:
        return run_key_for(str(self.cfg["study_id"]), spec.partition, spec.arm, spec.item_id, spec.rung,
                           spec.repeat, attempt)


def _classes(record) -> dict[str, int]:
    """``calls_by_class`` without the zero entries (the core initialises every class)."""
    return {k: v for k, v in record.calls_by_class.items() if v}


def _assert_no_key_material(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in ("", ".json", ".jsonl", ".md", ".txt", ".toml"):
            assert b"sk-or-" not in path.read_bytes(), path


@pytest.mark.timeout(600)
async def test_three_items_two_rungs_fake_and_null_arms(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script, concurrency=3) as h:
        h.with_inputs(base)
        specs = h.specs(base)
        assert len(specs) == 3 * 2 * 2
        # one delegation through the real proxy: the HF hmmt run at rung 4 spawns once (root -> solver -> root)
        delegating = next(s for s in specs if s.arm == "HF" and s.rung == 4 and s.item_id == _items(base)[1])
        script.by_node[delegating.run_key] = [SPAWN_REPLY, FakeReply(think_tokens=150), FakeReply(think_tokens=150)]
        summary = await h.runner.run_specs("main", specs)
    assert summary.planned == 12 and summary.executed == 12 and summary.skipped == 0 and not summary.drained
    assert summary.statuses == {"done": 12}
    plan_file = plan_path(h.run_root, "main")
    assert plan_file.is_file() and [s.run_key for s in read_plan(plan_file)] == [s.run_key for s in specs]
    records = h.store.scan()
    assert set(records) == {s.run_key for s in specs}
    for spec in specs:
        record = records[spec.run_key]
        assert record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1
        assert record.calls >= 1 and record.completion_total <= spec.cap_tokens
        assert record.final_source == "proxy_root" and record.final_message and "<final>" in record.final_message
        assert record.parse_status == "valid" and record.grade_status == "graded" and record.correct in (True, False)
        assert record.cli_result_mismatch is False, (spec.arm, record.cli_result, record.final_message)
        assert record.harness_exit_code == 0 and record.wall_seconds > 0
        assert record.spec.proxy_version == h.core.proxy_version and record.calls_detail
        assert all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail)
        # B2: every grant is exactly C minus what was counted before it
        spent = 0
        for call in record.calls_detail:
            if not call.duplicate_served and call.forced_close_kind != "terminal":
                assert call.granted_max_tokens == spec.cap_tokens - spent
                spent += call.completion_tokens
        work = h.settings.work_dir(spec)
        for sub in ("home", "state", "ws", "tmp"):
            assert (work / sub).is_dir()
        launch = json.loads((work / "launch.json").read_text())
        assert launch["run_key"] == spec.run_key and launch["env"]["OPENAI_API_KEY"] == spec.run_key
        assert spec.run_key in launch["env"]["OPENAI_BASE_URL"] and launch["wall_timeout_seconds"] == 600 + 120 * spec.rung
        assert (work / "stdout").is_file() and (work / "TASK.md").read_text().rstrip().endswith("Input tokens are not counted.")
        assert f"total budget of {spec.cap_tokens} completion tokens" in (work / "TASK.md").read_text()
        journaled = h.journal.scan(spec)
        assert len(journaled) == len([c for c in record.calls_detail if not c.duplicate_served])
        assert all(j["layout"] == "dynorch/v1" and j["run_key"] == spec.run_key for j in journaled)
        if spec.arm == "HF":
            assert launch["argv"][1].endswith("fake_harness_cli.py") and launch["stdin_bytes"] > 0
            stdout_line = json.loads((work / "stdout").read_text().strip().splitlines()[-1])
            assert stdout_line["type"] == "result" and stdout_line["subtype"] == "success"
            if spec.run_key == delegating.run_key:
                assert record.spawns == 1 and record.calls == 3 and _classes(record) == {"root": 2, "solver": 1}
                assert record.conversations == 2 and stdout_line["spawns"] == 1
                assert [c.conversation_class for c in record.calls_detail] == ["root", "solver", "root"]
                assert record.calls_detail[2].reinjection == "hit" and record.reinjection_hits == 1
                assert record.calls_detail[0].tool_calls == 1 and record.calls_detail[0].tool_parse == "ok"
            else:
                assert record.spawns == 0 and record.calls == 1 and _classes(record) == {"root": 1}
        else:
            assert launch["argv"] == [] and record.calls == 1 and record.spawns == 0
            assert (work / "SYSTEM.md").read_text() == h.settings.h0_system_text
            assert record.calls_detail[0].stream is False and record.calls_detail[0].requested_max_tokens == spec.cap_tokens
    attempts = [json.loads(l) for l in (h.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 12 and all(a["status"] == "done" for a in attempts)
    snap = h.stack.ledger.snapshot()
    assert snap.in_flight == 0 and snap.unresolved == 0 and snap.settled > 0
    assert (h.run_root / "state" / "ledger_journal.jsonl").is_file()
    _assert_no_key_material(h.run_root)
    _assert_no_key_material(h.scratch)
    assert not any(p.is_relative_to(Path.home()) for p in [h.run_root, h.scratch])


@pytest.mark.timeout(600)
async def test_drain_lets_the_in_flight_run_finish(tmp_path: Path, base: BaseInputs):
    """Plan §6 / review major 7: the drain stops ADMISSIONS and waits ``drain_wait_seconds`` for the runs already
    in flight. Killing them at once threw away up to 12 paid runs at every job boundary and burned one of their
    two reruns each."""
    cfg = _cfg_with_fake_arm()
    slow_item = base.panels["hmmt"].main_item_ids[0]
    script = FakeScript()
    script.set_reply(slow_item, reply=FakeReply(delay_seconds=1.0))
    async with Harness(tmp_path, cfg, script=script, concurrency=1) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,))
        specs = sorted(specs, key=lambda s: s.item_id == slow_item)
        specs = [specs[0], specs[-1], specs[1]] if specs[-1].item_id == slow_item else specs
        slow_key = next(s.run_key for s in specs if s.item_id == slow_item)
        task = asyncio.create_task(h.runner.run_specs("main", specs))
        while slow_key not in h.runner._in_flight:
            await asyncio.sleep(0.05)
        h.runner.request_drain()
        summary = await task
    assert summary.drained is True and summary.executed == 2
    records = h.store.scan()
    assert len(records) == 2 and {r.status for r in records.values()} == {"done"}
    slow = records[slow_key]
    assert slow.rerun_reason is None and slow.parse_status == "valid" and slow.final_message
    assert slow.harness_reported["killed_by_drain"] is False
    # the third cell was never admitted: --resume runs exactly it
    async with Harness(tmp_path, cfg, script=script, concurrency=1) as h2:
        h2.with_inputs(base)
        summary2 = await h2.runner.run_specs("main", specs, resume=True)
    assert summary2.skipped == 2 and summary2.executed == 1 and summary2.reruns == 0
    assert len(h2.store.scan()) == 3 and all(r.status == "done" for r in h2.store.scan().values())


@pytest.mark.timeout(600)
async def test_kill_and_resume(tmp_path: Path, base: BaseInputs):
    """A harness that outlives the drain wait is SIGTERMed (``killed_by_drain``) and rerun by the next
    ``--resume`` as attempt 2 / generation 2, with the interrupted attempt superseded, never released."""
    cfg = _cfg_with_fake_arm()
    slow_item = base.panels["hmmt"].main_item_ids[0]

    def script() -> FakeScript:
        s = FakeScript()
        s.set_reply(slow_item, reply=FakeReply(delay_seconds=1.5))
        return s

    async with Harness(tmp_path, cfg, script=script(), concurrency=1, drain_wait_seconds=0.2) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,))
        # order the slow item second so the first cell completes before the drain and the third never starts
        specs = sorted(specs, key=lambda s: s.item_id == slow_item)
        specs = [specs[0], specs[-1], specs[1]] if specs[-1].item_id == slow_item else specs
        slow_key = next(s.run_key for s in specs if s.item_id == slow_item)
        task = asyncio.create_task(h.runner.run_specs("main", specs))
        while slow_key not in h.runner._in_flight:        # wait until the slow harness is in flight
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        assert sum(1 for r in h.store.scan().values() if r.status == "done") == 1
        h.runner.request_drain()
        summary = await task
        assert summary.drained is True
        records = h.store.scan()
        done = [r for r in records.values() if r.status == "done"]
        killed = [r for r in records.values() if r.rerun_reason == "killed_by_drain"]
        assert len(done) == 1 and len(killed) == 1 and killed[0].spec.item_id == slow_item
        assert killed[0].status == "failed" and killed[0].spec.attempt == 1
        assert killed[0].harness_reported["killed_by_drain"] is True
        assert killed[0].correct is None and killed[0].parse_status is None
        first_done = done[0]
    # resume: a fresh stack over the same run root (inherited ledger attempts resolved by build_stack);
    # only the killed cell and the never-started one execute
    async with Harness(tmp_path, cfg, script=script(), concurrency=1) as h2:
        h2.with_inputs(base)
        assert h2.stack.ledger.snapshot().unresolved == 0
        summary2 = await h2.runner.run_specs("main", specs, resume=True)
    assert summary2.skipped == 1 and summary2.executed == 2 and not summary2.drained
    records = h2.store.scan()
    assert all(r.status == "done" for r in records.values()) and len(records) == 3
    resumed = next(r for r in records.values() if r.spec.item_id == slow_item)
    assert resumed.spec.attempt == 2 and resumed.spec.generation == 2 and resumed.spec.run_key != killed[0].spec.run_key
    assert resumed.parse_status == "valid"
    superseded = h2.store.scan(include_superseded=True)
    assert killed[0].spec.run_key in superseded and killed[0].spec.run_key not in records
    # review minor 20: the superseded copy on disk carries the mutated verdict, not a stale `running`
    assert superseded[killed[0].spec.run_key].status == "failed"
    assert superseded[killed[0].spec.run_key].rerun_reason == "killed_by_drain"
    untouched = next(r for r in records.values() if r.spec.item_id == first_done.spec.item_id)
    assert untouched.started_at == first_done.started_at and untouched.spec.run_key == first_done.spec.run_key
    attempts = [json.loads(l) for l in (h2.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert [a["rerun_reason"] for a in attempts].count("killed_by_drain") == 1 and len(attempts) == 4
    assert summary2.reruns == 1
    assert h2.stack.ledger.snapshot().in_flight == 0 and h2.stack.ledger.snapshot().unresolved == 0
    # review major 12: the tables read the real RunStore layout, so the paid first attempt is a row of its own and
    # its cost lands in retry_cost_usd instead of vanishing from the $-frontier
    from agent_allocation.dynorch import tables as T

    built = T.build_tables(h2.run_root, cfg, stage="main")
    runs_table = built["dynorch_runs"]
    assert len(runs_table) == 4 and sorted(runs_table["attempt"]) == [1, 1, 1, 2]
    assert killed[0].spec.run_key in set(runs_table["run_key"])
    costs = built["policy_costs_dynorch"]
    row = costs[costs["item_id"] == slow_item].iloc[0]
    assert Decimal(row["retry_cost_usd"]) == Decimal(killed[0].cost_usd) > 0
    assert Decimal(row["total_charged_usd"]) == Decimal(row["success_cost_usd"]) + Decimal(row["retry_cost_usd"])


@pytest.mark.timeout(600)
async def test_a_call_in_flight_when_the_harness_dies_is_settled_and_journaled(tmp_path: Path, base: BaseInputs):
    """Review major 5: the harness of the stage's last run is killed while its call is upstream. The parked call
    must be awaited (never cancelled), settled at its real cost and journaled -- a cancelled one left a SENT
    reservation with nothing journaled, so ``open_reservations_usd != 0`` failed the stage's reconciliation and
    the next --resume charged the call at its maximum."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script, concurrency=1, drain_wait_seconds=0.2) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,), items=_items(base)[:1])
        script.by_node[specs[0].run_key] = FakeReply(delay_seconds=3.0, content="<final>21</final>")
        task = asyncio.create_task(h.runner.run_specs("main", specs))
        while specs[0].run_key not in h.runner._in_flight:
            await asyncio.sleep(0.05)
        h.runner.request_drain()
        await task
        snapshot = h.stack.ledger.snapshot()
        record = h.store.scan()[specs[0].run_key]
        journaled = h.journal.scan(record.spec)
    assert snapshot.in_flight == 0 and snapshot.unresolved == 0 and snapshot.settled > 0
    assert record.rerun_reason == "killed_by_drain" and record.calls == 1
    assert len(journaled) == 1 and journaled[0]["phases"][0]["charged_max"] is False
    assert Decimal(record.cost_usd) == Decimal(journaled[0]["phases"][0]["cost_usd"]) > 0
    assert record.completion_total > 0                       # the tokens were paid for, so they are counted
    assert record.final_message is None                      # ... but the harness never received the answer


@pytest.mark.timeout(600)
async def test_a_failed_sub_agent_call_does_not_discard_the_delivered_answer(tmp_path: Path, base: BaseInputs):
    """Review major 10: the frozen rule reruns only when the LAST proxy reply was a transport/ledger error. A
    sub-agent call that failed mid-run (charged whole, the harness carried on, the root answered) made the whole
    run transport_failed and threw the delivered answer away."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        h.runner.adapters["fake"] = FakeAdapter(model=cfg["primary_model"], python=sys.executable,
                                                extra_args=("--tolerate-tool-errors",))
        specs = h.specs(base, arms=("HF",), rungs=(16,), items=_items(base)[:1])
        script.by_node[specs[0].run_key] = [SPAWN_REPLY,                                   # root: delegate
                                            FakeReply(fault=Fault.READ_TIMEOUT),           # solver: ambiguous
                                            FakeReply(content="<final>43</final>", think_tokens=150)]  # root answers
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert summary.statuses == {"done": 1} and record.spec.attempt == 1
    assert record.status == "done" and record.rerun_reason is None
    assert record.final_message == "<final>43</final>" and record.parse_status == "valid"
    assert record.final_answer == "43" and record.correct in (True, False)
    assert record.error and record.error.startswith("upstream_failed:")     # recorded, but not a rerun
    assert [c.conversation_class for c in record.calls_detail] == ["root", "solver", "root"]


@pytest.mark.timeout(600)
async def test_a_recovered_upstream_402_keeps_the_delivered_answer(tmp_path: Path, base: BaseInputs):
    """Re-review finding 3: a sub-agent call refused by the account (upstream 402) set the run's error to
    ``infrastructure_stop: ...`` and nothing cleared it, so a run whose root then DELIVERED an answer was rewritten
    to transport_failed and re-executed -- the review's own rule (only the last event decides) was violated. The
    stage still drains (the account refused once: Gate 3's stop rule), but a run that recovered keeps its answer."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        h.runner.adapters["fake"] = FakeAdapter(model=cfg["primary_model"], python=sys.executable,
                                                extra_args=("--tolerate-tool-errors",))
        specs = h.specs(base, arms=("HF",), rungs=(16,), items=_items(base)[:1])
        script.by_node[specs[0].run_key] = [SPAWN_REPLY,                                   # root: delegate
                                            FakeReply(fault=Fault.PAYMENT_REQUIRED),       # solver: account 402
                                            FakeReply(content="<final>43</final>", think_tokens=150)]  # root answers
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert [c.conversation_class for c in record.calls_detail] == ["root", "solver", "root"]
    assert record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1
    assert record.final_message == "<final>43</final>" and record.parse_status == "valid"
    assert str(record.error).startswith("infrastructure_stop:")        # recorded: the account did refuse
    assert summary.drained is True                                      # ... so the stage stops admitting
    assert summary_dict(summary, h.stack.ledger)["upstream_402"] is True


@pytest.mark.timeout(600)
async def test_the_ledger_ceiling_pauses_admissions_instead_of_burning_reruns(tmp_path: Path, base: BaseInputs):
    """Re-review finding 4: at the $ ceiling (or the unresolved-exposure limit) every call is refused 503
    ``ledger_refused``; that made each run transport_failed, reran it as a2 and a3 in the same job, and left every
    remaining cell at attempt 3 -- beyond the cap for good. Plan §5: ledger exposure PAUSES admissions. The refused
    run stays rerun-eligible without consuming a rerun, the stage drains, and no further cell is admitted."""
    cfg = _cfg_with_fake_arm()
    async with Harness(tmp_path, cfg, script=FakeScript(), concurrency=1, ceiling_usd="0.000001") as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,))
        summary = await h.runner.run_specs("main", specs)
        records = h.store.scan(include_superseded=True)
    assert summary.drained is True
    assert len(records) == 1, "cells kept being admitted (or rerun) against a refusing ledger"
    stopped = next(iter(records.values()))
    assert stopped.spec.attempt == 1 and stopped.status == "transport_failed"
    assert stopped.rerun_reason == "ledger_stop" and stopped.calls == 0
    assert h.runner._rerun_pending(stopped) and h.runner._attempts_left(stopped)
    # the next job (ceiling raised) re-executes it and the unadmitted cells
    async with Harness(tmp_path, cfg, script=FakeScript(), concurrency=1) as h2:
        h2.with_inputs(base)
        summary2 = await h2.runner.run_specs("main", specs, resume=True)
    assert not summary2.drained and summary2.executed == len(specs)
    assert all(r.status == "done" for r in h2.store.scan().values())


@pytest.mark.timeout(600)
async def test_an_upstream_402_drains_the_stage_and_resumes_without_consuming_reruns(tmp_path: Path,
                                                                                     base: BaseInputs):
    """Plan B6 / Gate 3 (review major 11): an upstream 402 is an infrastructure stop. The stage must drain instead
    of burning every remaining cell against a refusing account, and --resume must be able to re-execute them."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script, concurrency=1) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,))
        script.by_node[specs[1].run_key] = FakeReply(fault=Fault.PAYMENT_REQUIRED)
        summary = await h.runner.run_specs("main", specs)
    assert summary.drained is True
    records = h.store.scan()
    assert len(records) == 2, "the stage kept admitting cells after the account refused"
    stopped = records[specs[1].run_key]
    assert stopped.rerun_reason == "infrastructure_stop" and stopped.spec.attempt == 1
    assert str(stopped.error).startswith("infrastructure_stop:") and stopped.status == "transport_failed"
    assert summary_dict(summary, h.stack.ledger)["upstream_402"] is True
    async with Harness(tmp_path, cfg, script=FakeScript(), concurrency=1) as h2:
        h2.with_inputs(base)
        summary2 = await h2.runner.run_specs("main", specs, resume=True)
    assert summary2.executed == 2 and summary2.skipped == 1 and not summary2.drained
    assert all(r.status == "done" for r in h2.store.scan().values())


@pytest.mark.timeout(600)
async def test_one_unlaunchable_cell_does_not_take_the_stage_down(tmp_path: Path, base: BaseInputs):
    """Review major 8: an OSError from create_subprocess_exec (``openclaude`` not on the job's PATH) propagated
    through the gather and tore down the proxy under the other 11 runs, orphaning harnesses and skipping the
    stage's tables and reconciliation."""
    cfg = _cfg_with_fake_arm()
    async with Harness(tmp_path, cfg, script=FakeScript(), concurrency=2, max_reruns=1) as h:
        h.with_inputs(base)
        h.runner.adapters["fake"] = FakeAdapter(model=cfg["primary_model"], python="/nonexistent/python3")
        specs = h.specs(base, arms=("HF", "H0"), rungs=(4,), items=_items(base)[:1])
        summary = await h.runner.run_specs("main", specs)
    assert summary.errors == 0 and summary.executed == 2
    assert summary.statuses == {"failed": 1, "done": 1}
    records = h.store.scan()
    broken = next(r for r in records.values() if r.spec.arm == "HF")
    assert broken.status == "failed" and "FileNotFoundError" in str(broken.error)
    assert broken.spec.attempt == 2 and broken.calls == 0 and broken.correct is None
    good = next(r for r in records.values() if r.spec.arm == "H0")
    assert good.status == "done" and good.parse_status == "valid"


@pytest.mark.timeout(600)
async def test_rerun_rule_transport_error_then_success(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        # attempt 1's only call: an ambiguous upstream outcome (read timeout after the request was sent)
        script.by_node[specs[0].run_key] = FakeReply(fault=Fault.READ_TIMEOUT)
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert record.status == "done" and record.spec.attempt == 2 and record.rerun_reason is None
    assert record.parse_status == "valid" and summary.reruns == 1
    attempts = [json.loads(l) for l in (h.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert [a["status"] for a in attempts] == ["transport_failed", "done"]
    assert attempts[0]["rerun_reason"] == "last_reply_transport_or_ledger_error" and attempts[0]["spec"]["attempt"] == 1
    assert attempts[0]["parse_status"] is None and "upstream_failed" in attempts[0]["error"]
    assert attempts[0]["calls_detail"][0]["transport_status"] != "ok" and attempts[0]["calls"] == 1
    assert attempts[1]["spec"]["run_key"] != attempts[0]["spec"]["run_key"]
    # the ambiguous call was charged at its maximum and the first attempt was paid, never released (B2)
    snap = h.stack.ledger.snapshot()
    assert snap.unresolved == 0 and snap.in_flight == 0
    assert Decimal(attempts[0]["cost_usd"]) > 0


@pytest.mark.timeout(600)
async def test_rerun_rule_zero_calls_exhausts_reruns_and_leaves_nan(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    broken = tmp_path / "broken_cli.py"
    broken.write_text("import sys\nsys.stdin.read()\nprint('boot failure')\nsys.exit(3)\n")
    async with Harness(tmp_path, cfg) as h:
        h.with_inputs(base)
        h.runner.adapters["fake"] = FakeAdapter(model=cfg["primary_model"], python=sys.executable, cli_path=broken)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert record.status == "failed" and record.rerun_reason == "zero_proxy_calls_served"
    assert record.spec.attempt == 1 + cfg["dynorch"]["rerun_rule"]["max_reruns"] == 3
    assert record.harness_exit_code == 3 and record.calls == 0
    assert record.parse_status is None and record.correct is None
    from agent_allocation.dynorch.grade import delivered_state, outcome_y
    assert delivered_state(record) == "unknown" and math.isnan(outcome_y(record))
    attempts = (h.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 3 and summary.statuses == {"failed": 1}
    assert h.stack.ledger.snapshot().settled == 0


@pytest.mark.timeout(600)
async def test_402_after_a_served_call_never_reruns(tmp_path: Path, base: BaseInputs):
    """Rung 2 (cap 2048): the root spawns two solvers in one turn; the first solver is truncated at the remaining
    budget (``truncated_subagent``), the second finds ``remaining < 64`` on a non-root conversation -> HTTP 402
    (B4). The harness exits 1 without retrying; the run is ``done`` on what was served and is never rerun."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    two_spawns = (fake_tool_call_text("spawn_subagent", {"task": "part one"}) + "\n"
                  + fake_tool_call_text("spawn_subagent", {"task": "part two"}))
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(2,), items=_items(base)[:1])
        script.by_node[specs[0].run_key] = [FakeReply(content=two_spawns, think_tokens=900),
                                            FakeReply(think_tokens=5000)]
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1
    assert record.calls == 2 and record.refused_calls == 1 and record.exhausted is True
    assert [c.conversation_class for c in record.calls_detail] == ["root", "solver"]
    assert record.calls_detail[0].tool_calls == 2 and record.spawns == 2
    assert record.calls_detail[1].forced_close_kind == "truncated_subagent" and record.grace_tokens == 0
    assert record.completion_total == record.spec.cap_tokens
    assert record.harness_exit_code == 1 and record.final_message is None and record.final_source == "none"
    assert record.parse_status == "empty_content" and record.correct is False
    assert record.cli_result is None and record.cli_result_mismatch is None
    assert summary.reruns == 0
    stdout_line = json.loads((h.settings.work_dir(record.spec) / "stdout").read_text().strip().splitlines()[-1])
    assert stdout_line["is_error"] is True and "402" in stdout_line["result"]


def _delay_case(delay: float):
    async def run(tmp_path: Path, base: BaseInputs):
        cfg = _cfg_with_fake_arm()
        script = FakeScript()
        script.set_reply(_items(base)[0], reply=[FakeReply(delay_seconds=delay), FakeReply()])
        async with Harness(tmp_path, cfg, script=script) as h:
            h.with_inputs(base)
            specs = h.specs(base, arms=("HF",), rungs=(4,), items=_items(base)[:1])
            summary = await h.runner.run_specs("main", specs)
        record = next(iter(h.store.scan().values()))
        assert summary.statuses == {"done": 1}
        assert record.calls == 1 and record.status == "done" and record.parse_status == "valid"
        assert record.wall_seconds >= delay
        assert record.calls_detail[0].latency_seconds >= delay
        stdout_line = json.loads((h.settings.work_dir(record.spec) / "stdout").read_text().strip().splitlines()[-1])
        assert stdout_line["subtype"] == "success" and stdout_line["requests"] == 1
        assert stdout_line["keepalives"] >= delay - 2
    return run


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_delayed_reply_330s(tmp_path: Path, base: BaseInputs):
    await _delay_case(330.0)(tmp_path, base)


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_delayed_reply_660s(tmp_path: Path, base: BaseInputs):
    await _delay_case(660.0)(tmp_path, base)


@pytest.mark.timeout(600)
async def test_watchdog_lets_a_slow_terminal_call_deliver_its_answer(tmp_path: Path, base: BaseInputs):
    """Plan §2 B4 / review blocker 2: the TERMINAL call is admitted when ``remaining < 64`` already holds and is
    the one call whose output is graded. A watchdog whose 60 s clock starts at the grant (or at the call counter,
    which ``_terminal`` bumps before the upstream call) SIGTERMs the harness mid-answer and the delivered answer is
    lost. Here the terminal reply takes 4x ``sigterm_after_exhaustion``: the harness must survive it and the
    delivered answer must be the terminal answer."""
    cfg = _cfg_with_fake_arm()
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        spec = specs[0]
        script.by_node[spec.run_key] = [
            FakeReply(content=fake_tool_call_text("spawn_subagent", {"task": "solve it"}, prefix="Delegating."),
                      think_tokens=200),                       # root call 1: spawn (spent 200 of 1024)
            FakeReply(content="the sub-answer", think_tokens=790),   # solver: spent 990, remaining 34 < 64
            FakeReply(content="<final>77</final>", think_tokens=120, delay_seconds=4.0),   # TERMINAL, slow
        ]
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert summary.statuses == {"done": 1}, record.error
    assert record.harness_reported["sigterm_after_exhaustion"] is False
    assert record.calls == 3 and [c.forced_close_kind for c in record.calls_detail][-1] == "terminal"
    assert record.forced_close_kind == "terminal" and 0 < record.grace_tokens <= cfg["dynorch"]["grace_tokens"]
    assert record.final_message == "<final>77</final>" and record.final_source == "proxy_root"
    assert record.parse_status == "valid" and record.final_answer == "77"
    assert record.harness_exit_code == 0


@pytest.mark.timeout(600)
async def test_exhaustion_watchdog_terminates_a_lingering_harness(tmp_path: Path, base: BaseInputs):
    """Plan §2 B4: once exhausted, SIGTERM the harness group ``sigterm_after_exhaustion`` s after the last served
    reply (1 s here), SIGKILL ``sigkill_after`` s later; the run is ``done`` on what was served, never rerun."""
    cfg = _cfg_with_fake_arm()
    lingering = tmp_path / "lingering_cli.py"
    lingering.write_text(
        "import json, os, sys, time, http.client\n"
        "from urllib.parse import urlsplit\n"
        "sys.stdin.read()\n"
        "u = urlsplit(os.environ['OPENAI_BASE_URL'])\n"
        "c = http.client.HTTPConnection(u.hostname, u.port, timeout=60)\n"
        "body = json.dumps({'model': 'm', 'stream': False, 'messages': [{'role': 'system', 'content': open(sys.argv[sys.argv.index('--system-file') + 1]).read()}, {'role': 'user', 'content': 'TASK'}]})\n"
        "c.request('POST', u.path.rstrip('/') + '/chat/completions', body=body, headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': 'application/json'})\n"
        "r = c.getresponse(); r.read()\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'lingering'}), flush=True)\n"
        "time.sleep(120)\n")
    script = FakeScript()
    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        h.runner.adapters["fake"] = FakeAdapter(model=cfg["primary_model"], python=sys.executable, cli_path=lingering)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        # thinking runs to the cap: the proxy closes the call with the grace (mid_call) and the run is exhausted
        script.by_node[specs[0].run_key] = FakeReply(content="<final>3</final>", think_tokens=specs[0].cap_tokens)
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert summary.statuses == {"done": 1}
    assert record.harness_reported["sigterm_after_exhaustion"] is True
    assert record.harness_reported["timed_out"] is False and record.rerun_reason is None
    assert record.harness_exit_code not in (0, None) and record.wall_seconds < 30
    assert record.calls == 1 and record.completion_total == record.spec.cap_tokens
    assert record.forced_close_kind == "mid_call" and 0 < record.grace_tokens <= cfg["dynorch"]["grace_tokens"]
    assert record.final_message == "<final>3</final>" and record.parse_status == "valid"
    assert record.cli_result_mismatch is True          # the CLI's own result is only a cross-check
