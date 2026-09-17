"""status.json, ``allocation status`` / ``allocation stop``, the single-dispatcher claim, and the recovery guard."""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import pytest

import agent_allocation.runtime.scheduler as scheduler_mod
from agent_allocation.cli import main as cli_main
from agent_allocation.runtime.scheduler import DispatchRefused
from agent_allocation.storage.state import StateError, StateStore
from tests.unit.runtime.fakes import Env, FakeClient, make_graph, plan_of


async def test_status_json_is_written_live_and_at_the_end(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler_mod, "STATUS_INTERVAL_SECONDS", 0.05)
    env = Env.create(tmp_path, concurrency_start=4)
    graphs = [make_graph(f"s{i}", bank=3, chain=2, team_slots=1, peer_sizes=(), deep=()) for i in range(2)]
    run = asyncio.create_task(env.dispatcher(FakeClient(delay=0.1)).run(plan_of(graphs), allow_paid=False,
                                                                       max_usd=Decimal("5")))
    await asyncio.sleep(0.15)
    live = json.loads(env.paths.status_json.read_text())
    assert live["state"] == "running" and live["concurrency"]["configured_now"] == 4
    assert live["concurrency"]["in_flight_now"] > 0 and Decimal(live["spend"]["in_flight_usd"]) > 0
    summary = await run
    final = json.loads(env.paths.status_json.read_text())
    assert final["state"] == "finished" and final["stop_reason"] == "completed"
    assert final["counts"] == {"main": {"terminal_succeeded": 8}}
    assert final["concurrency"]["peak_in_flight"] == summary.peak_in_flight
    assert final["rate_limited_429"] == 0 and final["circuit"] == {"open": False, "reason": None}
    assert Decimal(final["spend"]["settled_usd"]) == summary.settled_usd


async def test_cli_status_and_stop(tmp_path, capsys):
    env = Env.create(tmp_path)
    graph = make_graph("cli", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())
    summary = await env.dispatcher(FakeClient()).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    capsys.readouterr()
    run_dir, local = str(env.paths.run_dir), str(env.paths.local_state_dir)
    assert cli_main(["status", "--run-dir", run_dir, "--local-state-dir", local]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["counts"] == {"main": {"terminal_succeeded": 2}} and printed["counts_source"] == "local_sqlite"
    assert Decimal(printed["journal_spend"]["settled_usd"]) == summary.settled_usd
    assert printed["journal_spend"]["open_attempts"] == 0 and printed["dispatcher_live"] is False
    assert cli_main(["stop", "--run-dir", run_dir, "--drain"]) == 0 and env.paths.drain_sentinel.exists()
    assert cli_main(["stop", "--run-dir", run_dir]) == 0 and env.paths.stop_sentinel.exists()
    assert json.loads(env.paths.stop_sentinel.read_text())["mode"] == "stop"
    capsys.readouterr()
    assert cli_main(["status", "--run-dir", run_dir, "--local-state-dir", local, "--no-write"]) == 0
    assert json.loads(capsys.readouterr().out)["control"] == {"stop": True, "drain": True}
    assert cli_main(["stop", "--run-dir", run_dir, "--clear"]) == 0
    assert not env.paths.stop_sentinel.exists() and not env.paths.drain_sentinel.exists()


async def test_second_live_dispatcher_is_refused_and_recovery_waits_for_it(tmp_path):
    env = Env.create(tmp_path)
    graph = make_graph("claim", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())
    first = env.dispatcher(FakeClient(delay=0.5))
    run = asyncio.create_task(first.run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5")))
    await asyncio.sleep(0.1)
    with pytest.raises(DispatchRefused):
        await env.dispatcher(FakeClient()).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    with pytest.raises(StateError):
        StateStore.recover(env.paths)                     # would charge the live dispatcher's in-flight money
    summary = await run
    assert summary.stop_reason == "completed"
    StateStore.recover(env.paths).close()                  # after a clean exit the claim is released
