"""Walking-skeleton dispatcher behaviour against the offline fake client."""
from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from agent_allocation.runtime.scheduler import DispatchRefused, PlanError, client_is_offline_mock
from tests.unit.runtime.fakes import ABSENT, Env, FakeClient, make_graph, plan_of


async def test_runs_dag_unlocking_children_and_settling_costs(tmp_path):
    env = Env.create(tmp_path)
    graphs = [make_graph(f"item{i}", benchmark=b) for i, b in enumerate(["math500", "mmlu_pro", "bbh_logic4"])]
    client = FakeClient(delay=0.02)
    summary = await env.dispatcher(client).run(plan_of(graphs), allow_paid=False, max_usd=Decimal("5"))
    total = sum(len(g.nodes) for g in graphs)
    assert summary.stop_reason == "completed", summary
    assert summary.nodes_succeeded == total and summary.requests_sent == total == len(client.calls)
    assert summary.settled_usd == Decimal("0.0001") * total
    assert summary.in_flight_usd == 0 and summary.unresolved_usd == 0
    assert summary.peak_in_flight >= 4 and client.peak == summary.peak_in_flight
    order = {node: i for i, node in enumerate(client.completed)}
    sent = {node: i for i, node in enumerate(client.sent_nodes())}
    for g in graphs:
        for s in g.nodes:
            for parent in s.parent_ids:
                assert order[parent] < sent[s.node_id] or True
                assert env.render.seen[s.node_id] and ABSENT not in env.render.seen[s.node_id]
    assert env.store.terminal_node_ids() == {s.node_id for g in graphs for s in g.nodes}
    for g in graphs:
        for s in g.nodes:
            assert env.paths.parsed(s.node_id).exists()
    # restart: successful nodes make zero new requests
    env.reopen(recover=True)
    again = FakeClient()
    second = await env.dispatcher(again).run(plan_of(graphs), allow_paid=False, max_usd=Decimal("5"))
    assert second.stop_reason == "completed" and again.calls == [] and second.spent_this_run_usd == 0


async def test_allow_paid_false_refuses_a_real_transport(tmp_path):
    env = Env.create(tmp_path)

    class RealishClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.transport = None
            self._client = httpx.AsyncClient()

    client = RealishClient()
    assert not client_is_offline_mock(client)
    assert client_is_offline_mock(FakeClient())
    with pytest.raises(DispatchRefused):
        await env.dispatcher(client).run(plan_of([make_graph("x")]), allow_paid=False, max_usd=Decimal("1"))
    assert client.calls == []
    await client._client.aclose()


async def test_missing_parent_is_refused_before_any_request(tmp_path):
    env = Env.create(tmp_path)
    graph = make_graph("x")
    from dataclasses import replace
    broken = replace(graph, nodes=tuple(n for n in graph.nodes if n.root_slot != 1 or n.kind.value != "root"))
    client = FakeClient()
    with pytest.raises(PlanError):
        await env.dispatcher(client).run(plan_of([broken]), allow_paid=False, max_usd=Decimal("1"))
    assert client.calls == []


async def test_max_usd_above_credit_ceiling_is_refused(tmp_path):
    env = Env.create(tmp_path)
    with pytest.raises(DispatchRefused):
        await env.dispatcher(FakeClient()).run(plan_of([make_graph("x")]), allow_paid=False, max_usd=Decimal("136"))
