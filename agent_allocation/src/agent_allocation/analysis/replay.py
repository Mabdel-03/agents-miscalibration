"""Budgeted policy replay (MASTER_SPEC §7.3) over stored policy prefixes.

For one item, one trajectory (``sample``, ``private_single``, ``private_team/n``, ``peer/n``, ``cp``, ``pooled``,
single-step ``deep/m``) and one per-item dollar budget: start with nothing; admit the next call or synchronous
round only if its *ex ante* reservation fits the remaining budget; after admission deduct the round's observed
charges (accepted + retried attempts). Deliver the last fully admitted checkpoint's sealed selection. If
initialization does not fit, the item is kept and scored as an abstention (0), never dropped.

Prefix-observable information only: the reservation of a step uses its rendered prompt size (from the stored
request body) and its declared completion caps — never the step's realized length, correctness or candidates.
Reservation per call (contract, agent 02 ``reservation_usd``): ``(ceil(chars/2) + 64) * input_price +
max_completion_tokens * output_price``, times ``budget.reservation_multiplier``. When a call's request body is
not stored, its realized ``prompt_tokens`` stand in for the rendered size (``basis=prompt_tokens``, an exact
token count available before sending); when neither exists the trajectory is reported ``descriptive`` (replay
unavailable). Prompts are horizon-independent, which is what makes prefix replay valid.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from decimal import ROUND_CEILING, Decimal
from typing import Any, Mapping

import numpy as np
import pandas as pd

from agent_allocation.analysis.costs import MILLION, attempt_charges, load_endpoint_locks, planning_prices
from agent_allocation.analysis.load import RunData
from agent_allocation.analysis.names import CheckpointNameError, parse_name
from agent_allocation.analysis.summary import benchmark_of_stratum, strata_order, stratum_weights
from agent_allocation.contracts import BENCHMARKS, GenerationSpec

GRID_POINTS = 12


def _rendered_chars(run: RunData, node_id: str) -> int | None:
    parsed = run.parsed.get(node_id)
    if not parsed or not parsed[0].raw_response_path:
        return None
    path = run.paths.run_dir / parsed[0].raw_response_path
    if not path.is_file():
        path = run.paths.raw_response(parsed[0].attempt_id)
        if not path.is_file():
            return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            body = json.load(handle).get("request_body", {})
    except (OSError, ValueError):
        return None
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return None
    return sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))


def node_reservations(run: RunData) -> tuple[dict[str, Decimal | None], dict[str, str]]:
    locks = load_endpoint_locks(run)
    prices = planning_prices(run.config)
    multiplier = Decimal(str(run.config.get("budget", {}).get("reservation_multiplier", "1")))
    reservations: dict[str, Decimal | None] = {}
    basis: dict[str, str] = {}
    for part in run.jobs.values():
        for nid, spec in part.items():
            in_price, out_price = _per_token_prices(spec, locks, prices)
            chars = _rendered_chars(run, nid)
            if chars is not None:
                in_tokens = (Decimal(chars) / 2).to_integral_value(rounding=ROUND_CEILING) + 64
                basis[nid] = "rendered_chars"
            else:
                usage = run.parsed[nid][0].usage if nid in run.parsed else None
                if usage is None or usage.prompt_tokens is None:
                    reservations[nid] = None
                    basis[nid] = "unavailable"
                    continue
                in_tokens = Decimal(usage.prompt_tokens)
                basis[nid] = "prompt_tokens"
            if in_price is None:
                reservations[nid] = None
                basis[nid] = "unavailable"
                continue
            reservations[nid] = (in_tokens * in_price + Decimal(spec.max_completion_tokens) * out_price) * multiplier
    return reservations, basis


def _per_token_prices(spec: GenerationSpec, locks, prices) -> tuple[Decimal | None, Decimal | None]:
    lock = locks.get(spec.model)
    if lock is not None:
        return Decimal(str(lock.input_price_per_token)), Decimal(str(lock.output_price_per_token))
    if spec.model in prices:
        return prices[spec.model]["input"] / MILLION, prices[spec.model]["output"] / MILLION
    return None, None


def trajectories(run: RunData) -> dict[tuple[str, str, str], list[tuple[str, tuple[tuple[str, ...], ...]]]]:
    """``{(partition, item, trajectory): [(checkpoint name, synchronous rounds of new node ids), ...]}``.

    A step's new nodes are split into dependency layers (a layer's parents are all admitted earlier), so a
    checkpoint that needs roots and then a peer round (e.g. ``cp/0``) is admitted round by round.
    """
    specs = {nid: spec for part in run.jobs.values() for nid, spec in part.items()}
    grouped: dict[tuple[str, str, str], dict[int, Any]] = defaultdict(dict)
    for partition in run.partitions:
        for cp in run.policies.get(partition, []):
            if cp.view.value != "current":
                continue
            try:
                name = parse_name(cp.name)
            except CheckpointNameError:
                continue
            if name.trajectory is not None:
                key, step = name.trajectory
                label = "/".join(map(str, key))
            elif name.family in ("deep", "pooled"):
                label, step = cp.name, 1
            else:
                continue
            grouped[(partition, cp.item_id, label)][step] = cp
    out: dict[tuple[str, str, str], list[tuple[str, tuple[tuple[str, ...], ...]]]] = {}
    for key, steps in grouped.items():
        seen: set[str] = set()
        ordered = []
        for step in sorted(steps):
            cp = steps[step]
            new = [n for n in cp.required_node_ids if n not in seen]
            ordered.append((cp.name, _layers(new, seen, specs)))
            seen.update(new)
        out[key] = ordered
    return out


def _layers(new: list[str], seen: set[str], specs: Mapping[str, GenerationSpec]) -> tuple[tuple[str, ...], ...]:
    depth: dict[str, int] = {}
    pending = set(new)

    def d(nid: str) -> int:
        if nid in depth:
            return depth[nid]
        spec = specs.get(nid)
        parents = [p for p in (spec.parent_ids if spec else ()) if p in pending]
        depth[nid] = 0 if not parents else 1 + max(d(p) for p in parents)
        return depth[nid]
    for nid in new:
        d(nid)
    layers: dict[int, list[str]] = defaultdict(list)
    for nid in new:
        layers[depth[nid]].append(nid)
    return tuple(tuple(layers[k]) for k in sorted(layers))


def replay(run: RunData, outcomes: pd.DataFrame, *, budgets: Mapping[str, list[float]] | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Item-level replay rows and the aggregated budget curves (per partition, trajectory, budget, level)."""
    notes: list[str] = []
    reservations, basis = node_reservations(run)
    charges, attempts_by_node, _ = attempt_charges(run)
    charged: dict[str, Decimal | None] = {}
    for part in run.jobs.values():
        for nid in part:
            attempts = attempts_by_node.get(nid, [])
            if nid in run.parsed and run.parsed[nid][0].usage and run.parsed[nid][0].usage.actual_cost_usd is not None:
                accepted = run.parsed[nid][0].attempt_id
                retry = sum((charges.get(a, Decimal(0)) for a in attempts if a != accepted), Decimal(0))
                charged[nid] = run.parsed[nid][0].usage.actual_cost_usd + retry
            else:
                charged[nid] = None
    y = outcomes[outcomes["view"] == "current"].set_index("policy_id")
    trajs = trajectories(run)
    if budgets is None:
        budgets = _budget_grid(trajs, reservations)
    item_rows: list[dict[str, Any]] = []
    for (partition, item, label), steps in sorted(trajs.items()):
        grid = budgets.get(partition, [])
        layer_res = [[_sum_or_none(reservations.get(n) for n in layer) for layer in layers] for _, layers in steps]
        layer_cost = [[_sum_or_none(charged.get(n) for n in layer) for layer in layers] for _, layers in steps]
        bases = sorted({basis.get(n, "unavailable") for _, layers in steps for layer in layers for n in layer})
        available = all(r is not None for rs in layer_res for r in rs)
        first_pid = f"{partition}/{item}/{steps[0][0]}/current"
        stratum = y.loc[first_pid, "stratum"] if first_pid in y.index else None
        for budget in grid:
            row = {"partition_id": partition, "item_id": item, "trajectory": label, "budget_usd": budget,
                   "stratum": stratum, "benchmark": benchmark_of_stratum(stratum) if stratum else None,
                   "reservation_basis": "+".join(bases), "replay_available": available}
            if not available:
                row.update({"status": "descriptive_replay_unavailable", "y": math.nan})
                item_rows.append(row)
                continue
            remaining = Decimal(str(budget))
            last_name = None
            binding = False
            unknown_cost = False
            admitted = 0
            for (name, _layers_), reservations_, costs_ in zip(steps, layer_res, layer_cost):
                complete_step = True
                for res, cost in zip(reservations_, costs_):
                    if res > remaining:
                        binding = True
                        complete_step = False
                        break
                    if cost is None:
                        unknown_cost = True
                        complete_step = False
                        break
                    remaining -= cost
                if not complete_step:
                    break
                admitted += 1
                last_name = name
            if unknown_cost:
                row.update({"status": "unknown_cost", "y": math.nan})
            elif last_name is None:
                row.update({"status": "abstain_initialization_does_not_fit", "y": 0.0})
            else:
                pid = f"{partition}/{item}/{last_name}/current"
                value = y.loc[pid, "y"] if pid in y.index else math.nan
                row.update({"status": "admitted", "y": float(value) if not pd.isna(value) else math.nan,
                            "delivered_checkpoint": last_name})
            row.update({"steps_admitted": admitted, "steps_total": len(steps), "binding_cap": binding,
                        "spend_usd": float(Decimal(str(budget)) - remaining),
                        "slack_usd": float(remaining)})
            item_rows.append(row)
    items = pd.DataFrame(item_rows)
    curves = _aggregate(items)
    if not items.empty and (items["status"] == "descriptive_replay_unavailable").any():
        missing = sorted(set(items.loc[items["status"] == "descriptive_replay_unavailable", "trajectory"]))
        notes.append("Budget replay unavailable (no rendered size or prompt tokens) for: " + ", ".join(missing)
                     + "; those curves are descriptive only.")
    return items, curves, notes


def _sum_or_none(values) -> Decimal | None:
    total = Decimal(0)
    for v in values:
        if v is None:
            return None
        total += v
    return total


def _budget_grid(trajs, reservations) -> dict[str, list[float]]:
    lows: dict[str, list[Decimal]] = defaultdict(list)
    highs: dict[str, list[Decimal]] = defaultdict(list)
    for (partition, _item, _label), steps in trajs.items():
        first = _sum_or_none(reservations.get(n) for n in steps[0][1][0])
        full = _sum_or_none(reservations.get(n) for _, layers in steps for layer in layers for n in layer)
        if first is not None and first > 0:
            lows[partition].append(first)
        if full is not None:
            highs[partition].append(full)
    grid = {}
    for partition in sorted(set(lows) | set(highs)):
        if not lows[partition] or not highs[partition]:
            continue
        lo = float(min(lows[partition])) * 0.5
        hi = float(max(highs[partition])) * 1.05
        grid[partition] = [float(f"{v:.6g}") for v in np.geomspace(lo, hi, GRID_POINTS)]
    return grid


def _aggregate(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (partition, label, budget), group in items.groupby(["partition_id", "trajectory", "budget_usd"], sort=True):
        strata = strata_order(set(group["stratum"].dropna()))
        levels = [("macro", stratum_weights(strata))] + [
            (b, stratum_weights(strata, within=b)) for b in sorted({benchmark_of_stratum(s) for s in strata},
                                                                   key=BENCHMARKS.index)]
        available = bool(group["replay_available"].all())
        for level, weights in levels:
            parts = {s: group[group["stratum"] == s] for s in weights}
            known = all(parts[s]["y"].notna().all() for s in weights)
            acc = math.fsum(w * float(parts[s]["y"].mean()) for s, w in weights.items()) if known else None
            rows.append({
                "partition_id": partition, "trajectory": label, "budget_usd": budget,
                "level": "macro" if level == "macro" else "benchmark", "benchmark": level,
                "n_items": int(sum(len(parts[s]) for s in weights)),
                "label": "budgeted policy replay" if available else "descriptive (replay unavailable)",
                "accuracy": acc,
                "n_unknown": int(sum(parts[s]["y"].isna().sum() for s in weights)),
                "abstention_rate": _wmean(parts, weights, lambda g: (g["status"] == "abstain_initialization_does_not_fit").mean()),
                "binding_cap_rate": _wmean(parts, weights, lambda g: g["binding_cap"].astype(float).mean()
                                           if "binding_cap" in g and g["binding_cap"].notna().all() else math.nan),
                "mean_spend_usd": _wmean(parts, weights, lambda g: g["spend_usd"].mean() if "spend_usd" in g else math.nan),
                "mean_slack_usd": _wmean(parts, weights, lambda g: g["slack_usd"].mean() if "slack_usd" in g else math.nan),
            })
    return pd.DataFrame(rows)


def _wmean(parts, weights, fn) -> float | None:
    vals = []
    for s, w in weights.items():
        v = fn(parts[s])
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        vals.append(w * float(v))
    return math.fsum(vals)
