"""Policy ledger: ancestry-based deployment cost of every checkpoint (agent 03; MASTER_SPEC §5.8, §7.2).

A policy is charged every generation node it would need to run from scratch -- its ancestors, each once -- no
matter how many other policies share those nodes in the collection. This is separate from the collection ledger
(agent 02), which charges each API attempt once.

:func:`policy_ancestors` recomputes ancestry from ``parent_ids`` (it must equal ``required_node_ids``);
:func:`policy_cost_rows` gives one row per checkpoint with nominal quantities from the graph and realized
quantities summed over the ancestors' accepted results. Unknown stays unknown (R9): a sum is ``None`` as soon as
one ancestor has no result or lacks that usage field, and the ``*_known_nodes`` columns say how many were known.
Retry charges live in the collection ledger (attempts), not here: these rows carry successful-generation results.
Monetary sums are exact ``Decimal``.

Config keys consumed (R9): none.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from agent_allocation.contracts import GenerationResult, ItemGraph, PolicyCheckpoint, TransportStatus

_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_prompt_tokens")


class PolicyCostError(ValueError):
    """A checkpoint that references nodes its graph does not contain, or whose declared ancestry is wrong."""


def policy_ancestors(graph: ItemGraph, checkpoint: PolicyCheckpoint) -> tuple[str, ...]:
    """Every node needed to deploy ``checkpoint`` from scratch: its candidates and their ancestors, deduplicated,
    in the graph's topological order."""
    position = {node.node_id: i for i, node in enumerate(graph.nodes)}
    parents = {node.node_id: node.parent_ids for node in graph.nodes}
    found: set[str] = set()
    todo = list(checkpoint.candidate_ids)
    while todo:
        key = todo.pop()
        if key in found:
            continue
        if key not in parents:
            raise PolicyCostError(f"{checkpoint.policy_id}: node {key} is not in graph {graph.item_id}")
        found.add(key)
        todo.extend(parents[key])
    return tuple(sorted(found, key=position.__getitem__))


def _sum_optional(values: list[Any]) -> Any:
    known = [v for v in values if v is not None]
    return sum(known[1:], known[0]) if known and len(known) == len(values) else None


def _critical_path(graph: ItemGraph, required: tuple[str, ...],
                   results: Mapping[str, GenerationResult]) -> tuple[int, float | None]:
    """Longest parent chain inside the ancestor set (nodes) and its summed latency (None if any latency unknown)."""
    parents = {node.node_id: node.parent_ids for node in graph.nodes}
    depth: dict[str, int] = {}
    latency: dict[str, float | None] = {}
    for node_id in required:  # topological order
        result = results.get(node_id)
        own = result.latency_seconds if result is not None else None
        ups = [p for p in parents[node_id] if p in depth]
        depth[node_id] = 1 + max((depth[p] for p in ups), default=0)
        if own is None or any(latency[p] is None for p in ups):
            latency[node_id] = None
        else:
            latency[node_id] = own + max((latency[p] for p in ups), default=0.0)  # type: ignore[type-var]
    longest = max(depth.values(), default=0)
    if not required or any(latency[n] is None for n in required):
        return longest, None
    return longest, max(latency[n] for n in required)  # type: ignore[type-var]


def policy_cost_rows(graph: ItemGraph, results: Mapping[str, GenerationResult]) -> list[dict[str, Any]]:
    """One row per checkpoint (both views): nominal generations/allowances and realized usage/cost of ancestors."""
    nodes = {node.node_id: node for node in graph.nodes}
    rows: list[dict[str, Any]] = []
    for checkpoint in graph.checkpoints:
        required = policy_ancestors(graph, checkpoint)
        if set(required) != set(checkpoint.required_node_ids) or len(required) != len(checkpoint.required_node_ids):
            raise PolicyCostError(f"{checkpoint.policy_id}: declared required_node_ids differ from recomputed ancestry")
        present = [results[n] for n in required if n in results]
        for result in present:
            if result.node_id not in nodes:
                raise PolicyCostError(f"result for unknown node {result.node_id}")
        row: dict[str, Any] = {
            "policy_id": checkpoint.policy_id,
            "partition_id": checkpoint.partition_id,
            "item_id": checkpoint.item_id,
            "name": checkpoint.name,
            "view": checkpoint.view.value,
            "family": checkpoint.family,
            "team_size": checkpoint.team_size,
            "total_cycles": checkpoint.total_cycles,
            "peer_rounds": checkpoint.peer_rounds,
            "reasoning_budget": checkpoint.reasoning_budget,
            "pool_size": len(checkpoint.candidate_ids),
            "nominal_generations": len(required),
            "requested_reasoning_tokens": sum(nodes[n].reasoning_budget for n in required),
            "requested_max_completion_tokens": sum(nodes[n].max_completion_tokens for n in required),
            "results_present": len(present),
            "results_missing": len(required) - len(present),
            "transport_ok_nodes": sum(1 for r in present if r.transport_status is TransportStatus.OK),
        }
        usages = [results[n].usage if n in results else None for n in required]
        for name in _USAGE_FIELDS:
            values = [getattr(u, name) if u is not None else None for u in usages]
            row[name] = _sum_optional(values)
            row[f"{name}_known_nodes"] = sum(1 for v in values if v is not None)
        costs: list[Decimal | None] = [u.actual_cost_usd if u is not None else None for u in usages]
        row["actual_cost_usd"] = _sum_optional(costs)
        row["actual_cost_known_nodes"] = sum(1 for c in costs if c is not None)
        upstream = [u.upstream_inference_cost_usd if u is not None else None for u in usages]
        row["upstream_inference_cost_usd"] = _sum_optional(upstream)
        latencies = [results[n].latency_seconds if n in results else None for n in required]
        row["latency_seconds_total"] = _sum_optional(latencies)
        row["critical_path_generations"], row["critical_path_latency_seconds"] = _critical_path(graph, required, results)
        rows.append(row)
    return rows
