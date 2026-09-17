"""Policy-ledger costs (MASTER_SPEC §7.2): every checkpoint is charged its own from-scratch ancestor set.

Sources, in order of authority:

* Agent 03's ``protocols.policy_cost.policy_cost_rows(graph, results)`` when importable: its rows are used for
  every column in :data:`EXTERNAL_COLUMN_MAP` (agent 03 name -> local name); the local recomputation fills the rest
  (retry charges, standardized price, CPU) and every overlapping column is cross-checked (a disagreement is a
  reported problem, never averaged).
* The local recomputation from ``PolicyCheckpoint.required_node_ids`` + accepted ``GenerationResult.usage`` +
  job specs + the ledger journal.

Never total collection spend: a policy's dollars are the sum over *its* ancestors only. Retry charges (ledger
charges on non-accepted attempts of those ancestors) are reported separately from successful-generation charges.
The standardized view prices all prompt tokens uncached at the planning snapshot (config ``planning_prices``,
USD per million tokens); ``completion_tokens`` already include reasoning tokens (no double counting).
"""
from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from decimal import Decimal
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation.analysis.load import RunData
from agent_allocation.analysis.names import CheckpointNameError, name_sort_key, parse_name
from agent_allocation.analysis.summary import benchmark_of_stratum, strata_order, stratum_weights
from agent_allocation.contracts import (
    BENCHMARKS, EndpointLock, GenerationSpec, ItemGraph, LedgerEventKind, PolicyCheckpoint, load_record,
)

MILLION = Decimal(1_000_000)
EXTERNAL_COLUMN_MAP = {                     # agent 03 policy_cost_rows column -> local column
    "nominal_generations": "nominal_generations",
    "requested_reasoning_tokens": "requested_reasoning_tokens",
    "requested_max_completion_tokens": "requested_completion_cap_tokens",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "reasoning_tokens": "reasoning_tokens",
    "cached_prompt_tokens": "cached_prompt_tokens",
    "actual_cost_usd": "success_cost_usd",
    "latency_seconds_total": "total_latency_seconds",
    "critical_path_latency_seconds": "critical_path_latency_seconds",
}
EXTERNAL_EXTRA_COLUMNS = ("upstream_inference_cost_usd", "critical_path_generations", "results_missing",
                          "transport_ok_nodes", "pool_size")
MONEY_COLUMNS = frozenset({"success_cost_usd"})
COST_MEASURES = {
    "total_charged_usd": "Actual API dollars per item (successful + retry charges)",
    "success_cost_usd": "Successful-generation API dollars per item",
    "standardized_uncached_cost_usd": "Standardized uncached planning-price dollars per item",
    "requested_reasoning_tokens": "Requested reasoning allowance per item (tokens)",
    "realized_tokens": "Realized prompt + completion tokens per item",
    "reasoning_tokens": "Realized reasoning tokens per item",
    "nominal_generations": "Nominal generations per item",
}
PRICE_SANITY_PER_MILLION = (Decimal("0.001"), Decimal("1000"))


class PriceUnitError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------------------------------

def planning_prices(cfg: Mapping[str, Any]) -> dict[str, dict[str, Decimal]]:
    """``{model: {"input": USD/M, "output": USD/M}}`` from config; refuses per-token values given as per-M."""
    prices = cfg.get("planning_prices", {})
    out: dict[str, dict[str, Decimal]] = {}
    for role, model_key in (("primary", "primary_model"), ("replication", "replication_model")):
        if role not in prices or model_key not in cfg:
            continue
        entry = {"input": Decimal(str(prices[role]["input_price"])),
                 "output": Decimal(str(prices[role]["output_price"]))}
        for side, value in entry.items():
            lo, hi = PRICE_SANITY_PER_MILLION
            if not lo <= value <= hi:
                raise PriceUnitError(f"planning_prices.{role}.{side}_price={value} is not a USD-per-million price")
        out[str(cfg[model_key])] = entry
    return out


def standardized_cost(prompt_tokens: int | None, completion_tokens: int | None,
                      price: Mapping[str, Decimal] | None) -> Decimal | None:
    if prompt_tokens is None or completion_tokens is None or price is None:
        return None
    return (Decimal(prompt_tokens) * price["input"] + Decimal(completion_tokens) * price["output"]) / MILLION


def load_endpoint_locks(run: RunData) -> dict[str, EndpointLock]:
    path = run.paths.endpoint_lock
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    records: list[Mapping[str, Any]] = []
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and raw.get("__record__") == "EndpointLock":
        records = [raw]
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, dict) and value.get("__record__") == "EndpointLock":
                records.append(value)
            elif isinstance(value, dict) and isinstance(value.get("record"), dict):
                records.append(value["record"])
    out = {}
    for rec in records:
        try:
            lock = load_record(EndpointLock, rec)
        except (TypeError, ValueError):
            continue
        out[lock.model] = lock
    return out


def price_unit_checks(cfg: Mapping[str, Any], locks: Mapping[str, EndpointLock], candidates: pd.DataFrame
                      ) -> list[dict[str, Any]]:
    """Unit checks: config per-million prices, lock per-token prices, and realized cost / standardized ratio."""
    rows: list[dict[str, Any]] = []
    try:
        prices = planning_prices(cfg)
        rows.append({"check": "planning_prices_per_million", "passed": True,
                     "detail": "; ".join(f"{m}: in {p['input']}/M out {p['output']}/M" for m, p in prices.items())})
    except PriceUnitError as exc:
        return [{"check": "planning_prices_per_million", "passed": False, "detail": str(exc)}]
    for model, lock in sorted(locks.items()):
        if model not in prices:
            continue
        for side, attr in (("input", "input_price_per_token"), ("output", "output_price_per_token")):
            per_million = Decimal(str(getattr(lock, attr))) * MILLION
            planned = prices[model][side]
            ok = planned > 0 and abs(per_million - planned) / planned <= Decimal("0.01")
            rows.append({"check": f"endpoint_lock_{side}_price_matches_planning", "passed": bool(ok),
                         "detail": f"{model}: lock {getattr(lock, attr)}/token = {per_million}/M vs planning {planned}/M"})
    if not candidates.empty and "actual_cost_usd" in candidates:
        known = candidates[candidates["actual_cost_usd"].notna() & candidates["prompt_tokens"].notna()
                           & candidates["completion_tokens"].notna()]
        ratios = []
        for _, row in known.iterrows():
            std = standardized_cost(int(row["prompt_tokens"]), int(row["completion_tokens"]), prices.get(row["model"]))
            actual = Decimal(str(row["actual_cost_usd"]))
            if std and std > 0:
                ratios.append(float(actual / std))
        if ratios:
            median = float(np.median(ratios))
            rows.append({"check": "realized_cost_over_standardized_ratio", "passed": 0.2 <= median <= 5.0,
                         "detail": f"median actual/standardized = {median:.4f} over {len(ratios)} generations "
                                   "(a unit error shows as ~1e6 or ~1e-6; cache discounts and price drift move it "
                                   "moderately)"})
    return rows


# ----------------------------------------------------------------------------------------------------
# Ledger
# ----------------------------------------------------------------------------------------------------

def attempt_charges(run: RunData) -> tuple[dict[str, Decimal], dict[str, list[str]], list[str]]:
    """Charged USD per attempt (SETTLE + CHARGE_MAX), attempts per node, and ledger anomalies."""
    charges: dict[str, Decimal] = defaultdict(Decimal)
    kinds: dict[str, set[str]] = defaultdict(set)
    by_node: dict[str, list[str]] = defaultdict(list)
    for event in run.ledger:
        if event.attempt_id is None:
            continue
        if event.node_id and event.attempt_id not in by_node[event.node_id]:
            by_node[event.node_id].append(event.attempt_id)
        if event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX):
            charges[event.attempt_id] += event.amount_usd
            kinds[event.attempt_id].add(event.kind.value)
    anomalies = [f"attempt {a} both settled and charged max" for a, k in sorted(kinds.items()) if len(k) > 1]
    return dict(charges), dict(by_node), anomalies


# ----------------------------------------------------------------------------------------------------
# Policy cost rows
# ----------------------------------------------------------------------------------------------------

def _external_policy_cost_rows():
    try:
        from agent_allocation.protocols.policy_cost import policy_cost_rows  # type: ignore
    except ImportError:
        return None
    return policy_cost_rows


def item_graphs(run: RunData) -> list[ItemGraph]:
    """Reconstruct per-item graphs from the manifests (for agent 03's ``policy_cost_rows``)."""
    graphs = []
    for partition in run.partitions:
        specs = run.jobs.get(partition, {})
        nodes_by_item: dict[str, list[GenerationSpec]] = defaultdict(list)
        for spec in specs.values():
            nodes_by_item[spec.item_id].append(spec)
        cps_by_item: dict[str, list[PolicyCheckpoint]] = defaultdict(list)
        for cp in run.policies.get(partition, []):
            cps_by_item[cp.item_id].append(cp)
        for item in sorted(set(nodes_by_item) | set(cps_by_item)):
            ordered = _topological(nodes_by_item.get(item, []))
            graphs.append(ItemGraph(item, partition, _grid_placeholder(), tuple(ordered), tuple(cps_by_item.get(item, []))))
    return graphs


def _grid_placeholder():
    from agent_allocation.contracts import GridSpec
    return GridSpec(name="reconstructed", bank_size=0, private_single_cycles=0, private_team_rounds=0, team_sizes=(),
                    peer_rounds=0, deep_multipliers=())


def _topological(specs: Sequence[GenerationSpec]) -> list[GenerationSpec]:
    by_id = {s.node_id: s for s in specs}
    done: dict[str, None] = {}

    def visit(nid: str) -> None:
        if nid in done or nid not in by_id:
            return
        for parent in by_id[nid].parent_ids:
            visit(parent)
        done[nid] = None
    for s in sorted(specs, key=lambda s: (s.update_round, s.root_slot, s.node_id)):
        visit(s.node_id)
    return [by_id[n] for n in done]


def policy_cost_table(run: RunData, candidates: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    prices = planning_prices(run.config)
    charges, attempts_by_node, anomalies = attempt_charges(run)
    problems.extend(anomalies)
    specs: dict[str, GenerationSpec] = {nid: s for part in run.jobs.values() for nid, s in part.items()}
    latency_cp = _critical_path_latency(run, specs)
    rows: list[dict[str, Any]] = []
    selections = {pid: sel for part in run.selections.values() for pid, sel in part.items()}
    for partition in run.partitions:
        for cp in run.policies.get(partition, []):
            rows.append(_local_row(run, cp, specs, charges, attempts_by_node, prices, latency_cp, selections))
    frame = pd.DataFrame(rows)
    external = _external_policy_cost_rows()
    source = "local recomputation (agent 03 policy_cost_rows not importable)"
    if external is not None and not frame.empty:
        results = {nid: pr[0] for nid, pr in run.parsed.items()}
        ext_rows: list[dict[str, Any]] = []
        try:
            for graph in item_graphs(run):
                ext_rows.extend(external(graph, results))
        except Exception as exc:  # independent module: record and fall back
            problems.append(f"agent 03 policy_cost_rows raised {type(exc).__name__}: {exc}; local costs used")
            ext_rows = []
        if ext_rows:
            frame, merge_problems = _merge_external(frame, pd.DataFrame(ext_rows))
            problems.extend(merge_problems)
            source = "agent 03 policy_cost_rows (cross-checked against local recomputation)"
    if not frame.empty:
        frame = frame.merge(outcomes[["policy_id", "benchmark", "subtask", "stratum", "y", "delivered_state"]],
                            on="policy_id", how="left")
        frame["cost_source"] = source
    return frame, problems


def _merge_external(local: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    if "policy_id" not in external:
        return local, ["agent 03 policy_cost_rows lack policy_id; local costs used"]
    ext = external.drop_duplicates("policy_id").set_index("policy_id")
    merged = local.set_index("policy_id")
    missing_ids = merged.index.difference(ext.index)
    if len(missing_ids):
        problems.append(f"agent 03 policy_cost_rows omit {len(missing_ids)} planned checkpoints (e.g. {missing_ids[0]})")
    for ext_col, col in EXTERNAL_COLUMN_MAP.items():
        if ext_col not in ext:
            problems.append(f"agent 03 policy_cost_rows lack column {ext_col}; local {col} used")
            continue
        theirs = ext[ext_col].reindex(merged.index)
        ours = merged[col] if col in merged else pd.Series(None, index=merged.index, dtype=object)
        a, b = _normalized(ours, col in MONEY_COLUMNS), _normalized(theirs, col in MONEY_COLUMNS)
        differ = (a.isna() != b.isna()) | (a.notna() & b.notna() & (a != b))
        if col.endswith("latency_seconds"):
            fa, fb = pd.to_numeric(ours, errors="coerce"), pd.to_numeric(theirs, errors="coerce")
            differ = (fa.isna() != fb.isna()) | ((fa - fb).abs() > 1e-6)
        differ &= theirs.index.isin(ext.index)
        if bool(differ.any()):
            first = differ[differ].index[0]
            problems.append(f"policy cost mismatch in {int(differ.sum())} checkpoint(s) for {col} (agent 03 "
                            f"{ext_col}); first {first}: local={ours.loc[first]} agent03={theirs.loc[first]}")
        merged[col] = [(str(v) if col in MONEY_COLUMNS else v) if not _missing(v) else None for v in theirs] \
            if col in MONEY_COLUMNS else theirs.where(theirs.notna(), ours)
    for extra in EXTERNAL_EXTRA_COLUMNS:
        if extra in ext:
            merged[extra] = [None if _missing(v) else (str(v) if isinstance(v, Decimal) else v)
                             for v in ext[extra].reindex(merged.index)]
    out = merged.reset_index()
    success = out["success_cost_usd"]
    retry = out["retry_cost_usd"]
    out["total_charged_usd"] = [None if _missing(a) else str(Decimal(str(a)) + Decimal(str(r)))
                                for a, r in zip(success, retry)]
    return out, problems


def _normalized(series: pd.Series, money: bool) -> pd.Series:
    if money:
        return series.map(lambda v: None if _missing(v) else str(Decimal(str(v)).normalize()))
    return pd.to_numeric(series, errors="coerce")


def _missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _critical_path_latency(run: RunData, specs: Mapping[str, GenerationSpec]) -> dict[str, float | None]:
    memo: dict[str, float | None] = {}

    def value(nid: str) -> float | None:
        if nid in memo:
            return memo[nid]
        parsed = run.parsed.get(nid)
        own = parsed[0].latency_seconds if parsed else None
        spec = specs.get(nid)
        parents = [value(p) for p in spec.parent_ids] if spec else []
        if own is None or any(p is None for p in parents):
            memo[nid] = None
        else:
            memo[nid] = own + (max(parents) if parents else 0.0)
        return memo[nid]
    for nid in specs:
        value(nid)
    return memo


def _local_row(run, cp: PolicyCheckpoint, specs, charges, attempts_by_node, prices, latency_cp, selections) -> dict[str, Any]:
    required = cp.required_node_ids
    token_sums: dict[str, int | None] = {k: 0 for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                                         "cached_prompt_tokens")}
    success: Decimal | None = Decimal(0)
    standardized: Decimal | None = Decimal(0)
    retry = Decimal(0)
    n_terminal = 0
    for nid in required:
        spec = specs.get(nid)
        parsed = run.parsed.get(nid)
        accepted = parsed[0].attempt_id if parsed else None
        for attempt in attempts_by_node.get(nid, []):
            if attempt != accepted:
                retry += charges.get(attempt, Decimal(0))
        usage = parsed[0].usage if parsed else None
        if parsed:
            n_terminal += 1
        for key in token_sums:
            value = getattr(usage, key) if usage else None
            token_sums[key] = None if (value is None or token_sums[key] is None) else token_sums[key] + value
        cost = usage.actual_cost_usd if usage else None
        success = None if (cost is None or success is None) else success + cost
        std = standardized_cost(usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None,
                                prices.get(spec.model) if spec else None)
        standardized = None if (std is None or standardized is None) else standardized + std
    try:
        expected = parse_name(cp.name).nominal_generations
    except CheckpointNameError:
        expected = None
    pool_checks = [run.public_checks[n].cpu_seconds for n in cp.candidate_ids if n in run.public_checks]
    grade_cpu = [run.grades[n].cpu_seconds for n in cp.candidate_ids if n in run.grades
                 and run.grades[n].cpu_seconds is not None]
    selection = selections.get(cp.policy_id)
    latencies = [run.parsed[n][0].latency_seconds for n in required if n in run.parsed]
    critical = [latency_cp.get(n) for n in required]
    realized = (None if token_sums["prompt_tokens"] is None or token_sums["completion_tokens"] is None
                else token_sums["prompt_tokens"] + token_sums["completion_tokens"])
    return {
        "partition_id": cp.partition_id, "policy_id": cp.policy_id, "item_id": cp.item_id, "name": cp.name,
        "view": cp.view.value, "family": cp.family,
        "nominal_generations": len(required), "nominal_generations_expected": expected,
        "n_required_terminal": n_terminal,
        "requested_reasoning_tokens": sum(specs[n].reasoning_budget for n in required if n in specs),
        "requested_completion_cap_tokens": sum(specs[n].max_completion_tokens for n in required if n in specs),
        **token_sums,
        "realized_tokens": realized,
        "success_cost_usd": None if success is None else str(success),
        "retry_cost_usd": str(retry),
        "total_charged_usd": None if success is None else str(success + retry),
        "standardized_uncached_cost_usd": None if standardized is None else str(standardized),
        "public_check_cpu_seconds": float(sum(pool_checks)) if pool_checks else None,
        "selection_cpu_seconds": selection.cpu_seconds if selection else None,
        "private_grading_cpu_seconds": float(sum(grade_cpu)) if grade_cpu else None,
        "total_latency_seconds": float(sum(x for x in latencies if x is not None)) if len(latencies) == len(required)
        and all(x is not None for x in latencies) else None,
        "critical_path_latency_seconds": (max(critical) if critical and all(c is not None for c in critical) else None),
    }


# ----------------------------------------------------------------------------------------------------
# Aggregation and descriptive frontier
# ----------------------------------------------------------------------------------------------------

def cost_summary(costs: pd.DataFrame, accuracy: pd.DataFrame) -> pd.DataFrame:
    """Mean per-item cost measures per (partition, checkpoint) at benchmark and macro level (equal weights),
    joined with delivered accuracy. Cost means use items whose cost is known; ``n_<measure>_unknown`` counts the rest."""
    from agent_allocation.analysis.summary import weighted_means

    if costs.empty:
        return pd.DataFrame()
    current = costs[costs["view"] == "current"]
    summary = weighted_means(current, ["partition_id", "name"], list(COST_MEASURES))
    summary = summary.rename(columns={m: f"mean_{m}" for m in COST_MEASURES})
    summary = summary.drop(columns=[c for c in summary.columns if c.endswith("_complete")])
    acc = accuracy[(accuracy["view"] == "current") & (accuracy["level"] != "stratum")][
        ["partition_id", "name", "benchmark", "acc", "acc_lower", "acc_upper"]]
    summary = summary.merge(acc, on=["partition_id", "name", "benchmark"], how="left")
    summary["family"] = summary["name"].map(lambda n: str(n).split("/", 1)[0])
    level_rank = {b: i for i, b in enumerate(("macro",) + BENCHMARKS)}
    order = sorted(range(len(summary)), key=lambda i: (str(summary.at[i, "partition_id"]),
                                                       name_sort_key(summary.at[i, "name"]),
                                                       level_rank.get(summary.at[i, "benchmark"], 99)))
    return summary.iloc[order].reset_index(drop=True)


def descriptive_frontier(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Upper-left Pareto set of measured configurations (accuracy up, mean cost down). Descriptive only: the
    best among evaluated configurations on this sample, never a validated optimal policy."""
    col = f"mean_{measure}"
    if summary.empty or col not in summary:
        return pd.DataFrame()
    rows = []
    for (partition, benchmark), group in summary.groupby(["partition_id", "benchmark"], sort=True):
        known = group[group[col].notna() & group["acc"].notna()].copy()
        known[col] = known[col].astype(float)
        known["acc"] = known["acc"].astype(float)
        known = known.sort_values([col, "acc"], ascending=[True, False], kind="mergesort")
        best = -math.inf
        for _, row in known.iterrows():
            if row["acc"] > best:
                best = row["acc"]
                rows.append({"partition_id": partition, "benchmark": benchmark, "measure": measure,
                             "name": row["name"], "mean_cost": row[col], "acc": row["acc"],
                             "label": "descriptive frontier among evaluated configurations (not validated)"})
    return pd.DataFrame(rows)
