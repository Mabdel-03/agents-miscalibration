"""AIMD arithmetic, wave admission, critical-path priority, per-item caps, optional blocks, budget ceilings."""
from __future__ import annotations

from decimal import Decimal

from agent_allocation.contracts import GenerationKind
from agent_allocation.runtime.scheduler import Aimd, SchedulerSettings
from tests.unit.runtime.fakes import Env, FakeClient, make_config, make_graph, plan_of


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_aimd_follows_the_scheduler_block_exactly():
    cfg = make_config(concurrency_start=16, concurrency_max=128, requests_per_second_start=2,
                      requests_per_second_max=16, aimd_window_seconds=60)
    clock = Clock()
    aimd = Aimd(SchedulerSettings.from_config(cfg), clock)
    assert (aimd.concurrency_limit, aimd.rps) == (16, 2)
    clock.t += 59
    assert not aimd.tick(clock.t)
    clock.t += 1
    assert not aimd.tick(clock.t)                     # stable but never binding: no evidence, no increase
    aimd.bound_in_window = True
    clock.t += 60
    assert aimd.tick(clock.t) and (aimd.concurrency_limit, aimd.rps) == (24, 3)
    for _ in range(40):
        aimd.bound_in_window = True
        clock.t += 60
        aimd.tick(clock.t)
    assert (aimd.concurrency_limit, aimd.rps) == (128, 16)
    sent_before = clock.t - 1
    clock.t += 1
    assert aimd.on_rate_limited(clock.t, sent_before, retry_after=7.0)
    assert (aimd.concurrency_limit, aimd.rps) == (64, 8) and aimd.pause_until == clock.t + 7.0
    # the rest of the burst was admitted under the old allowance: one decrease per congestion event
    assert not aimd.on_rate_limited(clock.t + 0.1, sent_before, retry_after=1.0)
    assert (aimd.concurrency_limit, aimd.rps) == (64, 8)
    assert aimd.on_rate_limited(clock.t + 1, clock.t + 0.5, retry_after=None)
    assert (aimd.concurrency_limit, aimd.rps) == (32, 4)
    aimd.bound_in_window = True
    clock.t += 30
    assert not aimd.tick(clock.t)                     # the window restarted at the decrease
    # token bucket at 4/s
    clock.t += 100
    assert aimd.token_available(clock.t)
    for _ in range(4):
        aimd.take_token()
    assert not aimd.token_available(clock.t) and 0 < aimd.seconds_until_token(clock.t) <= 0.25


async def test_waves_open_in_order_and_only_to_keep_the_dispatcher_busy(tmp_path):
    env = Env.create(tmp_path, max_active_waves=1, wave_items=2, concurrency_start=4)
    waves = [[make_graph(f"w{w}i{i}", bank=3, chain=2, team_slots=1, peer_sizes=(), deep=()) for i in range(2)]
             for w in range(3)]
    client = FakeClient(delay=0.005)
    summary = await env.dispatcher(client).run(plan_of(*waves), allow_paid=False, max_usd=Decimal("5"))
    assert summary.stop_reason == "completed" and summary.waves_completed == 3
    wave_of = {s.node_id: w for w, graphs in enumerate(waves) for g in graphs for s in g.nodes}
    sequence = [wave_of[n] for n in client.sent_nodes()]
    assert sequence == sorted(sequence)               # one active wave: strictly wave by wave


async def test_older_waves_first_and_optional_block_waits_for_core(tmp_path):
    env = Env.create(tmp_path, max_active_waves=4, wave_items=2, concurrency_start=2)
    core = [[make_graph(f"c{w}{i}", bank=3, chain=3, team_slots=1, peer_sizes=(), deep=()) for i in range(2)]
            for w in range(2)]
    optional = [make_graph(f"r{i}", partition="replication", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())
                for i in range(2)]
    client = FakeClient(delay=0.003)
    summary = await env.dispatcher(client).run(plan_of(*core, optional), allow_paid=False, max_usd=Decimal("5"))
    assert summary.stop_reason == "completed"
    sent = client.sent_nodes()
    core_ids = {s.node_id for graphs in core for g in graphs for s in g.nodes}
    optional_ids = {s.node_id for g in optional for s in g.nodes}
    last_core = max(i for i, n in enumerate(sent) if n in core_ids)
    first_optional = min(i for i, n in enumerate(sent) if n in optional_ids)
    assert first_optional > last_core


async def test_critical_path_first_then_fair_buckets(tmp_path):
    env = Env.create(tmp_path, concurrency_start=1, max_calls_per_item_in_flight=1)
    graph = make_graph("solo", bank=7, chain=4, team_slots=3, team_rounds=1, peer_sizes=(2,), peer_rounds=1, deep=(2,))
    client = FakeClient()
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    assert summary.stop_reason == "completed"
    by_id = {s.node_id: s for s in graph.nodes}
    sent = [by_id[n] for n in client.sent_nodes()]
    # root[0] and its whole private chain go first, then roots 1-4, then everything else
    assert [(s.kind, s.root_slot, s.update_round) for s in sent[:4]] == \
        [(GenerationKind.ROOT, 0, 0), (GenerationKind.PRIVATE, 0, 1), (GenerationKind.PRIVATE, 0, 2),
         (GenerationKind.PRIVATE, 0, 3)]
    assert [(s.kind, s.root_slot) for s in sent[4:8]] == [(GenerationKind.ROOT, j) for j in range(1, 5)]
    # the rest interleaves kinds instead of draining one bucket
    rest_kinds = [s.kind for s in sent[8:]]
    assert rest_kinds[:3] != [rest_kinds[0]] * 3


async def test_per_item_in_flight_cap(tmp_path):
    env = Env.create(tmp_path, concurrency_start=16, max_calls_per_item_in_flight=2)
    graph = make_graph("wide", bank=12, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(delay=0.03)
    summary = await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    assert summary.stop_reason == "completed" and client.peak == 2 and summary.peak_in_flight == 2


def _big_render(inner, chars: int):
    def render(spec, parents):
        messages = inner(spec, parents)
        messages[-1]["content"] += "x" * chars
        return messages
    return render


async def test_tiny_budget_never_exceeded_and_pilot_allowance_caps_pilot_plans(tmp_path):
    env = Env.create(tmp_path, concurrency_start=32)
    from tests.unit.runtime.fakes import Behavior

    graphs = [make_graph(f"b{i}", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(4)]
    client = FakeClient(lambda spec, aid, i: Behavior(delay=0.01, cost=Decimal("0.0019")))
    dispatcher = env.dispatcher(client)
    # 20k-char prompts reserve ~0.0020 USD each while the wave forecast (default prompt size) is ~0.0297: the wave
    # opens under a 0.03 ceiling and the ceiling then binds concurrent reservations
    dispatcher.render = _big_render(env.render, 20000)
    summary = await dispatcher.run(plan_of(graphs), allow_paid=False, max_usd=Decimal("0.03"))
    assert summary.stop_reason == "budget_exhausted"
    assert env.ledger.peak_exposure <= Decimal("0.03")
    assert summary.settled_usd + summary.in_flight_usd + summary.unresolved_usd <= Decimal("0.03")
    assert 0 < summary.requests_sent < 24

    pilot_env = Env.create(tmp_path / "pilot")
    pilot = [make_graph("p0", partition="pilot", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())]
    pilot_summary = await pilot_env.dispatcher(FakeClient()).run(plan_of(pilot), allow_paid=False,
                                                                 max_usd=Decimal("135"))
    assert pilot_summary.ceiling_usd == Decimal("5") and pilot_summary.soft_ceiling_usd == Decimal("5")


async def test_future_waves_are_not_opened_when_their_forecast_does_not_fit(tmp_path):
    env = Env.create(tmp_path, max_active_waves=1, wave_items=1)
    waves = [[make_graph(f"f{w}", bank=8, chain=1, team_slots=1, peer_sizes=(), deep=())] for w in range(3)]
    from tests.unit.runtime.fakes import Behavior

    client = FakeClient(lambda spec, aid, i: Behavior(cost=Decimal("0.0008")))
    # forecast per wave ~ 8 x 0.00124 = 0.0099; after two waves (~0.0128 settled) the third no longer fits 0.02
    summary = await env.dispatcher(client).run(plan_of(*waves), allow_paid=False, max_usd=Decimal("0.02"))
    assert summary.stop_reason == "budget_exhausted"
    assert summary.stop_detail.startswith("wave_forecast_exceeds_soft_ceiling")
    assert summary.waves_opened == 2 and summary.waves_completed == 2 and summary.nodes_pending == 8
    assert {n for n, _ in client.calls}.isdisjoint({s.node_id for s in waves[2][0].nodes})   # no half-started wave


async def test_retries_may_use_the_reserve_above_the_soft_ceiling(tmp_path, monkeypatch):
    import agent_allocation.runtime.scheduler as scheduler_mod
    from agent_allocation.contracts import TransportStatus
    from tests.unit.runtime.fakes import Behavior

    monkeypatch.setattr(scheduler_mod, "BACKOFF_BASE_SECONDS", 0.05)
    env = Env.create(tmp_path, concurrency_start=16)
    env.config["budget"]["soft_admission_ceiling_usd"] = 0.0125
    graph = make_graph("r", bank=10, chain=1, team_slots=1, peer_sizes=(), deep=())
    first_root = graph.nodes[0].node_id

    def script(spec, attempt_id, index):
        if attempt_id == f"{first_root}:1":
            return Behavior(status=TransportStatus.UPSTREAM_ERROR, cost=None, delay=0.05)
        return Behavior(cost=Decimal("0.0013"))

    dispatcher = env.dispatcher(FakeClient(script))
    summary = await dispatcher.run(plan_of([graph]), allow_paid=False, max_usd=Decimal("0.02"))
    assert summary.soft_ceiling_usd == Decimal("0.0125") and summary.ceiling_usd == Decimal("0.02")
    assert summary.stop_reason == "completed" and summary.nodes_succeeded == 10
    assert summary.extra["reserve_admissions"] == 1                 # the retry used the contingency, logged
    assert summary.settled_usd > summary.soft_ceiling_usd


async def test_aimd_allowances_carry_over_between_runs_on_one_instance(tmp_path):
    from agent_allocation.contracts import TransportStatus
    from tests.unit.runtime.fakes import Behavior

    env = Env.create(tmp_path, concurrency_start=8, requests_per_second_start=1000, aimd_window_seconds=3600)
    first_graph = make_graph("first", bank=3, chain=1, team_slots=1, peer_sizes=(), deep=())
    second_graph = make_graph("second", bank=3, chain=1, team_slots=1, peer_sizes=(), deep=())

    def script(spec, attempt_id, index):
        if index == 0:
            return Behavior(status=TransportStatus.RATE_LIMITED, cost=None, retry_after=0.05)
        return Behavior(delay=0.01)

    client = FakeClient(script)
    dispatcher = env.dispatcher(client)
    first = await dispatcher.run(plan_of([first_graph]), allow_paid=False, max_usd=Decimal("5"))
    assert first.stop_reason == "completed" and first.extra["aimd_decreases"] == 1
    assert (first.concurrency_start, first.concurrency_final, first.rps_final) == (8, 4, 500)
    second = await dispatcher.run(plan_of([second_graph]), allow_paid=False, max_usd=Decimal("5"))
    assert second.stop_reason == "completed"
    assert (second.concurrency_start, second.extra["rps_start"]) == (4, 500)    # no re-ramp from the start
    assert second.extra["aimd_decreases"] == 0 and client.peak <= 4
    fresh = env.dispatcher(FakeClient())
    assert (fresh.aimd.concurrency_limit, fresh.aimd.rps) == (8, 1000)          # a new instance starts over


async def test_run_caps_do_not_stick_to_the_ledger_between_runs(tmp_path):
    env = Env.create(tmp_path)
    env.ledger.set_active_ceiling(Decimal("100"))                   # the lead's provider-derived ceiling
    dispatcher = env.dispatcher(FakeClient())
    pilot = [make_graph("p", partition="pilot", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())]
    main = [make_graph("m", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())]
    first = await dispatcher.run(plan_of(pilot), allow_paid=False, max_usd=Decimal("135"))
    assert first.ceiling_usd == Decimal("5")                         # pilot allowance for a pilot-only plan
    assert env.ledger.snapshot().ceiling == Decimal("100")           # cleared after the run
    second = await dispatcher.run(plan_of(main), allow_paid=False, max_usd=Decimal("60"))
    assert second.ceiling_usd == Decimal("60") and second.soft_ceiling_usd == Decimal("60")
    third = await dispatcher.run(plan_of(main), allow_paid=False, max_usd=Decimal("135"))
    assert third.ceiling_usd == Decimal("100") and third.soft_ceiling_usd == Decimal("100")
    assert env.ledger.active_ceiling == Decimal("100")


async def test_forecast_ratio_is_seeded_from_stored_history(tmp_path):
    from tests.unit.runtime.fakes import Behavior

    env = Env.create(tmp_path)
    graph = make_graph("hist", bank=25, chain=1, team_slots=1, peer_sizes=(), deep=())
    client = FakeClient(lambda spec, aid, i: Behavior(cost=Decimal("0.0003")))
    await env.dispatcher(client).run(plan_of([graph]), allow_paid=False, max_usd=Decimal("5"))
    fresh = env.dispatcher(FakeClient())                               # a new process would start here
    assert fresh.ok_cost_samples == 25
    assert Decimal("0.3") < fresh._cost_ratio() < Decimal("0.4")       # 0.0003 / ~0.00091 reserved
