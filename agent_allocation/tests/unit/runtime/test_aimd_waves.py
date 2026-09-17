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


async def test_tiny_budget_never_exceeded_and_pilot_allowance_caps_pilot_plans(tmp_path):
    env = Env.create(tmp_path, concurrency_start=16)
    graphs = [make_graph(f"b{i}", bank=6, chain=1, team_slots=1, peer_sizes=(), deep=()) for i in range(4)]
    client = FakeClient(delay=0.01)
    # each root reserves ~0.00092 USD: a 0.004 ceiling admits at most four concurrently
    summary = await env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=Decimal("0.004"))
    assert summary.stop_reason == "budget_exhausted"
    assert env.ledger.peak_exposure <= Decimal("0.004")
    assert summary.settled_usd + summary.in_flight_usd + summary.unresolved_usd <= Decimal("0.004")
    assert 0 < summary.requests_sent < 24

    pilot_env = Env.create(tmp_path / "pilot")
    pilot = [make_graph("p0", partition="pilot", bank=2, chain=1, team_slots=1, peer_sizes=(), deep=())]
    pilot_summary = await pilot_env.dispatcher(FakeClient()).run(plan_of(pilot), allow_paid=False,
                                                                 max_usd=Decimal("135"))
    assert pilot_summary.ceiling_usd == Decimal("5") and pilot_summary.soft_ceiling_usd == Decimal("5")
