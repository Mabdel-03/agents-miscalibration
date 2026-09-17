"""Policy ledger: ancestry recompute and per-checkpoint cost rows with unknowns kept unknown (agent 03)."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from agent_allocation.contracts import (
    CheckpointView, CompletionStatus, GenerationResult, TransportStatus, Usage,
)
from agent_allocation.protocols.grid import checkpoint_by_name
from agent_allocation.protocols.policy_cost import PolicyCostError, policy_ancestors, policy_cost_rows

from .helpers import graph


def _result(node_id: str, *, cost: str | None = "0.0001", reasoning: int | None = 100, latency: float | None = 2.0):
    usage = Usage(prompt_tokens=1000, completion_tokens=300, reasoning_tokens=reasoning, cached_prompt_tokens=0,
                  actual_cost_usd=Decimal(cost) if cost is not None else None)
    return GenerationResult(node_id=node_id, attempt_id=f"{node_id}:1", transport_status=TransportStatus.OK,
                            completion_status=CompletionStatus.COMPLETED, generation_id="g", observed_model="m",
                            observed_provider="Google AI Studio", finish_reason="stop", native_finish_reason="STOP",
                            visible_content="<final>1</final>", reasoning_returned=False, usage=usage,
                            cache_status="MISS", raw_response_path=None, raw_response_sha256=None,
                            latency_seconds=latency, http_status=200)


def test_policy_ancestors_recompute_required_node_ids_for_every_checkpoint():
    for name in ("main_extension", "high", "replication", "pilot_r2"):
        g = graph(name)
        for cp in g.checkpoints:
            assert policy_ancestors(g, cp) == cp.required_node_ids


def test_policy_ancestors_refuse_foreign_nodes():
    g = graph("main")
    cp = replace(checkpoint_by_name(g, "sample/2"), candidate_ids=("not-a-node",))
    with pytest.raises(PolicyCostError):
        policy_ancestors(g, cp)


def test_cost_rows_charge_each_ancestor_once_with_exact_decimals():
    g = graph("main_extension")
    results = {n.node_id: _result(n.node_id) for n in g.nodes}
    rows = {(r["name"], r["view"]): r for r in policy_cost_rows(g, results)}
    assert len(rows) == len(g.checkpoints)
    peer = rows["peer/5/4", "current"]
    assert peer["nominal_generations"] == 25 and peer["pool_size"] == 5
    assert peer["actual_cost_usd"] == Decimal("0.0025") and peer["prompt_tokens"] == 25_000
    assert peer["requested_reasoning_tokens"] == 25 * 1024 and peer["requested_max_completion_tokens"] == 25 * 2048
    assert peer["critical_path_generations"] == 5 and peer["critical_path_latency_seconds"] == 10.0
    assert peer["latency_seconds_total"] == 50.0
    assert rows["peer/5/4", "archive"]["pool_size"] == 25
    assert rows["cp/3", "current"]["nominal_generations"] == 20 and rows["pooled/4", "current"]["nominal_generations"] == 5
    assert rows["pooled/4", "current"]["critical_path_generations"] == 2
    deep = rows["deep/16", "current"]
    assert deep["requested_reasoning_tokens"] == 16384 and deep["nominal_generations"] == 1
    # sharing root[0] between SAMPLE and PRIVATE_SINGLE does not remove it from either policy
    assert rows["sample/1", "current"]["actual_cost_usd"] == rows["private_single/1", "current"]["actual_cost_usd"]


def test_unknown_usage_stays_unknown():
    g = graph("main")
    cp = checkpoint_by_name(g, "private_single/3")
    results = {n.node_id: _result(n.node_id) for n in g.nodes}
    missing = cp.required_node_ids[1]
    results[cp.required_node_ids[0]] = _result(cp.required_node_ids[0], cost=None, reasoning=None, latency=None)
    del results[missing]
    row = next(r for r in policy_cost_rows(g, results) if r["policy_id"] == cp.policy_id)
    assert row["results_missing"] == 1 and row["results_present"] == 2
    assert row["actual_cost_usd"] is None and row["actual_cost_known_nodes"] == 1
    assert row["reasoning_tokens"] is None and row["prompt_tokens"] is None and row["prompt_tokens_known_nodes"] == 2
    assert row["critical_path_latency_seconds"] is None and row["critical_path_generations"] == 3
    assert policy_cost_rows(g, {})[0]["actual_cost_usd"] is None


def test_declared_ancestry_mismatch_is_refused():
    g = graph("main")
    cp = checkpoint_by_name(g, "private_team/2/1", CheckpointView.ARCHIVE)
    broken = replace(g, checkpoints=(replace(cp, required_node_ids=cp.required_node_ids[:-1]),))
    with pytest.raises(PolicyCostError):
        policy_cost_rows(broken, {})
