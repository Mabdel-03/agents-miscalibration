"""WP-B end-to-end: proxy httpd + core + runner + the fake harness subprocess, $0 (contract §6; plan Verification).

The core is WP-A's real ``ProxyCore`` over ``api/fake.py``'s transport when ``agent_allocation.dynorch.core`` is
importable (after the merge), else the scripted ``StubProxyCore`` (before it). Every assertion below holds for both;
the ones about scripted spawns are conditioned on the stub. Cases:

* three items (one per math benchmark + one GPQA) × rungs {1, 4} × arms {HF fake harness, H0 null}: every planned
  cell ends ``done`` with a graded record, the delivered answer comes from the proxy (``final_source ==
  "proxy_root"``), the CLI result cross-checks, ``launch.json`` / work-dir layout / attempts journal exist and carry
  no key material, and the pre-registration file has one line per cell;
* kill / resume: a drain request mid-stage SIGTERMs the in-flight harness (``killed_by_drain``), the next
  ``run_specs(resume=True)`` re-executes only that cell as attempt 2 / generation 2 and skips the finished ones;
* the rerun rule: a transport error on the first attempt reruns once (attempt 2 succeeds); zero proxy calls rerun
  up to ``max_reruns`` then leave an infra failure with y = NaN; a 402 after a served call never reruns;
* 330 s and 660 s delayed replies (``@pytest.mark.slow``, ``@pytest.mark.timeout(1500)``; opt in with
  ``DYNORCH_SLOW=1``): headers + keepalive comments keep the fake harness on one request and ``subtype: success``.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.dynorch import stubs
from tests.unit.dynorch.stubs import Scripted, StubProxyCore, StubRunStore, real_core_available
from agent_allocation.dynorch.harness.fake import FakeAdapter
from agent_allocation.dynorch.harness.null import NullAdapter
from agent_allocation.dynorch.httpd import ProxyServer, ReplyLog
from agent_allocation.dynorch.plan import BaseInputs, plan_runs, plan_path, read_plan
from agent_allocation.dynorch.records import ProxyError
from agent_allocation.dynorch.runner import Runner, RunnerSettings

PROJECT = Path(__file__).resolve().parents[2]
CFG = json.loads((PROJECT / "configs" / "dynorch_v1.json").read_text("utf-8"))
SLOW = pytest.mark.skipif(not os.environ.get("DYNORCH_SLOW"), reason="set DYNORCH_SLOW=1 for the long-delay cases")
_REAL_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """The integration conftest forbids every AF_INET connect; this stack IS a loopback server, so allow 127.0.0.1
    and keep refusing anything else (no network, no key -- the fake harness runs in a subprocess anyway)."""
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


class Harness:
    """One wired stack: core (real or stub), httpd, store, runner settings, adapters, work under tmp."""

    def __init__(self, tmp: Path, cfg: dict[str, Any], *, script=None, concurrency: int = 2,
                 max_reruns: int | None = None) -> None:
        self.tmp = tmp
        self.cfg = cfg
        self.run_root = tmp / "run_root"
        self.scratch = tmp / "scratch"
        self.reply_log = ReplyLog()
        self.env: Any = None
        if real_core_available() and script is None:
            self.core, self.store, self.journal = self._real_core()
        else:
            self.core = StubProxyCore(script)
            self.store = StubRunStore(self.run_root)
            self.journal = None
        self.settings = RunnerSettings.from_config(cfg, run_root=self.run_root, project_root=PROJECT,
                                                   scratch_root=self.scratch, concurrency=concurrency)
        if max_reruns is not None:
            self.settings = RunnerSettings(**{**self.settings.__dict__, "max_reruns": max_reruns})
        self.settings = RunnerSettings(**{**self.settings.__dict__, "poll_seconds": 0.2, "sigkill_after": 2.0,
                                          "sigterm_after_exhaustion": 1.0})
        self.adapters = {"fake": FakeAdapter(model=cfg["primary_model"], python=sys.executable),
                         "null": NullAdapter(model=cfg["primary_model"])}
        self.server = ProxyServer(self.core, keepalive_seconds=1.0, reply_log=self.reply_log)

    def _real_core(self):
        from decimal import Decimal
        from agent_allocation.api import fake as fakeapi
        from agent_allocation.api.forced import ForcedClient, ForcedSettings
        from agent_allocation.dynorch.config import DynorchSettings
        from agent_allocation.dynorch.core import ProxyCore
        from agent_allocation.dynorch.journal import CallJournal, RunStore
        from agent_allocation.dynorch.upstream import Upstream
        from agent_allocation.runtime.ledger import CollectionLedger
        from agent_allocation.storage.journal import LedgerJournal

        settings = DynorchSettings.from_config(self.cfg)
        script = fakeapi.FakeScript()
        forced_cfg = dict(self.cfg)
        forced_cfg["provider"] = {**self.cfg["provider"], "endpoint_tag": "parasail/fp8"}
        client = ForcedClient(settings=ForcedSettings.from_config(forced_cfg), api_key=None,
                              transport=fakeapi.fake_openrouter_transport(script), answer_backoff_base_seconds=0.0)
        ledger_journal = LedgerJournal(self.run_root / "state" / "ledger_journal.jsonl", group_commit_ms=0)
        ledger = CollectionLedger(ledger_journal, ceiling_usd=Decimal("95"),
                                  max_unresolved_usd=settings.max_unresolved_spend_usd)
        locks = {"parasail": fakeapi.fake_forced_endpoint_lock(tag="parasail/fp8"),
                 "mancer": fakeapi.fake_forced_endpoint_lock(tag="mancer/fp8")}
        upstream = Upstream(client, locks, self.cfg["sampling"], settings.provider_in_flight, settings.provider_rps,
                            ledger, reservation_multiplier=settings.reservation_multiplier)
        journal = CallJournal(self.run_root)
        store = RunStore(self.run_root)
        self.env = {"client": client, "ledger_journal": ledger_journal, "script": script}
        return ProxyCore(settings, upstream, journal, store), store, journal

    async def __aenter__(self) -> "Harness":
        await self.server.start()
        self.runner = Runner(self.core, self.adapters, self.settings, proxy_url=self.server.base_url, store=self.store,
                             inputs=None, reply_log=self.reply_log, cfg=self.cfg, journal=self.journal)
        return self

    def with_inputs(self, base: BaseInputs) -> "Harness":
        self.runner.inputs = base
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.server.close()
        if self.env is not None:
            await self.env["client"].aclose()
            await self.env["ledger_journal"].flush()
            self.env["ledger_journal"].release_writer()

    def specs(self, base: BaseInputs, *, arms=("HF", "H0"), rungs=(1, 4), items=None):
        return plan_runs(self.cfg, stage="main", arms=list(arms), rungs=list(rungs), repeats=[1],
                         items=items or _items(base), base=base, proxy_version="e2e")


def _assert_no_key_material(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in ("", ".json", ".jsonl", ".md", ".txt", ".toml"):
            assert b"sk-or-" not in path.read_bytes(), path


@pytest.mark.timeout(600)
async def test_three_items_two_rungs_fake_and_null_arms(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    async with Harness(tmp_path, cfg, concurrency=3) as h:
        h.with_inputs(base)
        specs = h.specs(base)
        assert len(specs) == 3 * 2 * 2
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
        work = h.settings.work_dir(spec)
        for sub in ("home", "state", "ws", "tmp"):
            assert (work / sub).is_dir()
        launch = json.loads((work / "launch.json").read_text())
        assert launch["run_key"] == spec.run_key and launch["env"]["OPENAI_API_KEY"] == spec.run_key
        assert spec.run_key in launch["env"]["OPENAI_BASE_URL"] and launch["wall_timeout_seconds"] == 600 + 120 * spec.rung
        assert (work / "stdout").is_file() and (work / "TASK.md").read_text().rstrip().endswith("Input tokens are not counted.")
        assert f"total budget of {spec.cap_tokens} completion tokens" in (work / "TASK.md").read_text()
        if spec.arm == "HF":
            assert launch["argv"][1].endswith("fake_harness_cli.py") and launch["stdin_bytes"] > 0
            stdout_line = json.loads((work / "stdout").read_text().strip().splitlines()[-1])
            assert stdout_line["type"] == "result" and stdout_line["subtype"] == "success"
            if isinstance(h.core, StubProxyCore):
                if spec.rung >= 4:
                    assert record.spawns == 1 and record.calls == 3 and record.calls_by_class == {"root": 2, "solver": 1}
                    assert record.conversations == 2 and stdout_line["spawns"] == 1
                else:
                    assert record.spawns == 0 and record.calls == 1 and record.calls_by_class == {"root": 1}
        else:
            assert launch["argv"] == [] and record.calls == 1 and record.spawns == 0
            assert (work / "SYSTEM.md").read_text() == h.settings.h0_system_text
    attempts = [json.loads(l) for l in (h.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 12 and all(a["status"] == "done" for a in attempts)
    _assert_no_key_material(h.run_root)
    _assert_no_key_material(h.scratch)
    assert not any(p.is_relative_to(Path.home()) for p in [h.run_root, h.scratch])


@pytest.mark.timeout(600)
async def test_kill_and_resume(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    slow_item = base.panels["hmmt"].main_item_ids[0]

    def script(run, request, call_no):
        reply = stubs.default_script(run, request, call_no)
        if run.spec.item_id == slow_item:
            reply.delay_seconds = 1.5
        return reply

    async with Harness(tmp_path, cfg, script=script, concurrency=1) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(4,))
        # order the slow item second so the first cell completes before the drain and the third never starts
        specs = sorted(specs, key=lambda s: s.item_id == slow_item)
        specs = [specs[0], specs[-1], specs[1]] if specs[-1].item_id == slow_item else specs
        slow_key = next(s.run_key for s in specs if s.item_id == slow_item)
        task = asyncio.create_task(h.runner.run_specs("main", specs))
        while slow_key not in h.runner._in_flight:        # wait until the slow harness is in flight
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        assert len(h.store.scan()) == 1
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
    # resume: a fresh runner, same run_root/store; only the killed cell and the never-started one execute
    async with Harness(tmp_path, cfg, script=script, concurrency=1) as h2:
        h2.with_inputs(base)
        summary2 = await h2.runner.run_specs("main", specs, resume=True)
    assert summary2.skipped == 1 and summary2.executed == 2 and not summary2.drained
    records = h2.store.scan()
    assert all(r.status == "done" for r in records.values()) and len(records) == 3
    resumed = next(r for r in records.values() if r.spec.item_id == slow_item)
    assert resumed.spec.attempt == 2 and resumed.spec.generation == 2 and resumed.spec.run_key != killed[0].spec.run_key
    assert resumed.parse_status == "valid"
    untouched = next(r for r in records.values() if r.spec.item_id == first_done.spec.item_id)
    assert untouched.started_at == first_done.started_at and untouched.spec.run_key == first_done.spec.run_key
    attempts = [json.loads(l) for l in (h2.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert [a["rerun_reason"] for a in attempts].count("killed_by_drain") == 1 and len(attempts) == 4
    assert summary2.reruns == 1


@pytest.mark.timeout(600)
async def test_rerun_rule_transport_error_then_success(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()
    failed_once: set[str] = set()

    def script(run, request, call_no):
        cell = (run.spec.item_id, run.spec.rung)
        if run.spec.attempt == 1 and cell not in failed_once:
            failed_once.add(cell)
            return Scripted(error=ProxyError(502, "upstream_failed", "ambiguous upstream outcome", False))
        return stubs.default_script(run, request, call_no)

    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert record.status == "done" and record.spec.attempt == 2 and record.rerun_reason is None
    assert record.parse_status == "valid" and summary.reruns == 1
    attempts = [json.loads(l) for l in (h.run_root / "state" / "dynorch_attempts.jsonl").read_text().splitlines()]
    assert [a["status"] for a in attempts] == ["transport_failed", "done"]
    assert attempts[0]["rerun_reason"] == "last_reply_transport_or_ledger_error" and attempts[0]["spec"]["attempt"] == 1
    assert attempts[0]["parse_status"] is None
    assert attempts[1]["spec"]["run_key"] != attempts[0]["spec"]["run_key"]


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


@pytest.mark.timeout(600)
async def test_402_after_a_served_call_never_reruns(tmp_path: Path, base: BaseInputs):
    cfg = _cfg_with_fake_arm()

    def script(run, request, call_no):
        # rung 1 (cap 1024): the root spends 1000 on a spawn; the solver then finds remaining < 64 -> 402
        klass = run.conversations.get(stubs.conversation_key(request), "root")
        if klass == "root":
            return Scripted(reasoning="delegate", content="",
                            tool_calls=[("spawn_subagent", {"task": "solve"})], completion_tokens=1000)
        return Scripted(content="<final>1</final>", completion_tokens=100)

    async with Harness(tmp_path, cfg, script=script) as h:
        h.with_inputs(base)
        specs = h.specs(base, arms=("HF",), rungs=(1,), items=_items(base)[:1])
        summary = await h.runner.run_specs("main", specs)
    record = next(iter(h.store.scan().values()))
    assert record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1
    assert record.calls == 1 and record.refused_calls == 1 and record.exhausted is True
    assert record.harness_exit_code == 1 and record.final_message is None and record.final_source == "none"
    assert record.parse_status == "empty_content" and record.correct is False
    assert record.cli_result is None and record.cli_result_mismatch is None
    assert summary.reruns == 0


def _delay_case(delay: float):
    async def run(tmp_path: Path, base: BaseInputs):
        cfg = _cfg_with_fake_arm()
        seen = {"delayed": False}

        def script(run, request, call_no):
            reply = stubs.default_script(run, request, call_no)
            reply.tool_calls = []
            reply.content = "<rationale>slow</rationale>\n<final>7</final>"
            if not seen["delayed"]:
                seen["delayed"] = True
                reply.delay_seconds = delay
            return reply

        async with Harness(tmp_path, cfg, script=script) as h:
            h.with_inputs(base)
            specs = h.specs(base, arms=("HF",), rungs=(4,), items=_items(base)[:1])
            summary = await h.runner.run_specs("main", specs)
        record = next(iter(h.store.scan().values()))
        assert summary.statuses == {"done": 1}
        assert record.calls == 1 and record.status == "done" and record.parse_status == "valid"
        assert record.wall_seconds >= delay
        stdout_line = json.loads((h.settings.work_dir(record.spec) / "stdout").read_text().strip().splitlines()[-1])
        assert stdout_line["subtype"] == "success" and stdout_line["requests"] == 1
        assert stdout_line["keepalives"] >= delay - 2
    return run


@SLOW
@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_delayed_reply_330s(tmp_path: Path, base: BaseInputs):
    await _delay_case(330.0)(tmp_path, base)


@SLOW
@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_delayed_reply_660s(tmp_path: Path, base: BaseInputs):
    await _delay_case(660.0)(tmp_path, base)
