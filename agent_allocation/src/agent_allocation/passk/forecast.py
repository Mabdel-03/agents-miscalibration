"""passk_cont_v1 spend forecast over the frozen node lists (WP-G).

Per-call unit costs are the measured forced_v2 main means per shard and allowance (plan, "Cost and throughput"). The
forecast of a shard is ``sum(nodes x unit cost) x RETRY_FACTOR + PROBE_USD_PER_RUN x runs`` where ``runs`` counts the
passk-run invocations that each fire the forced Gate-0 probes (canary + smoke + main by default). With the frozen
assignment (L96 legacy 58:38, T200 1:1) this gives about Mancer $56.4 / Parasail $40.8 (+ probes), total ~$99.
"""
from __future__ import annotations

from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable, Mapping, Sequence

MAIN_A = 1024
RETRY_FACTOR = Decimal("1.02")
PROBE_USD_PER_RUN = Decimal("0.05")          # banana + </think> + forced-2,048 + ANSWER probes, rounded up
DEFAULT_RUNS = 3                              # canary, smoke, main
UNIT_COST_USD: dict[str, dict[int, Decimal]] = {
    "mancer": {1: Decimal("0.00381"), 2: Decimal("0.00659"), 4: Decimal("0.01203"), 8: Decimal("0.02307"),
               16: Decimal("0.04519")},
    "parasail": {1: Decimal("0.00335"), 2: Decimal("0.00572"), 4: Decimal("0.01056"), 8: Decimal("0.02033"),
                 16: Decimal("0.03989")},
}


def _cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def budget_counts(nodes: Iterable[Any], main_a: int = MAIN_A) -> Counter:
    """``{multiplier: n_nodes}`` from GenerationSpecs (or anything with ``reasoning_budget``)."""
    out: Counter = Counter()
    for node in nodes:
        budget = int(node.reasoning_budget)
        if budget % main_a:
            raise ValueError(f"budget {budget} is not a multiple of A={main_a}")
        out[budget // main_a] += 1
    return out


def forecast_counts(shard: str, counts: Mapping[int, int], *, runs: int = DEFAULT_RUNS) -> dict[str, Any]:
    units = UNIT_COST_USD[shard]
    missing = sorted(set(counts) - set(units))
    if missing:
        raise ValueError(f"no unit cost for multipliers {missing} on shard {shard}")
    calls = sum((units[m] * n for m, n in counts.items()), Decimal("0"))
    total = calls * RETRY_FACTOR + PROBE_USD_PER_RUN * runs
    return {"shard": shard, "calls": sum(counts.values()), "by_multiplier": {str(m): n for m, n in sorted(counts.items())},
            "calls_usd": str(_cents(calls)), "with_retries_usd": str(_cents(calls * RETRY_FACTOR)),
            "probes_usd": str(_cents(PROBE_USD_PER_RUN * runs)), "forecast_usd": str(_cents(total))}


def forecast_graphs(shard: str, graphs: Mapping[str, Sequence[Any]], *, runs: int = DEFAULT_RUNS,
                    main_a: int = MAIN_A) -> dict[str, Any]:
    """Forecast one shard's plan (``{partition: [ItemGraph]}``), with a per-partition breakdown."""
    per_partition = {}
    total_counts: Counter = Counter()
    for partition, gs in graphs.items():
        counts = budget_counts((n for g in gs for n in g.nodes), main_a)
        total_counts.update(counts)
        per_partition[partition] = forecast_counts(shard, counts, runs=0)
    out = forecast_counts(shard, total_counts, runs=runs)
    out["partitions"] = per_partition
    return out


def forecast_study(shard_plans: Mapping[str, Mapping[str, Sequence[Any]]], *, runs: int = DEFAULT_RUNS,
                   max_usd: Decimal | str | None = None, slices: Mapping[str, Decimal | str] | None = None
                   ) -> dict[str, Any]:
    """``{shard: forecast}`` plus the total and the Gate-0 checks (total <= ``max_usd``; shard <= its soft slice)."""
    shards = {s: forecast_graphs(s, plan, runs=runs) for s, plan in shard_plans.items()}
    total = sum((Decimal(v["forecast_usd"]) for v in shards.values()), Decimal("0"))
    checks = {}
    if max_usd is not None:
        checks["total_within_max"] = total <= Decimal(str(max_usd))
    for shard, value in (slices or {}).items():
        if shard in shards:
            checks[f"{shard}_within_slice"] = Decimal(shards[shard]["forecast_usd"]) <= Decimal(str(value))
    return {"shards": shards, "total_usd": str(_cents(total)), "checks": checks,
            "passed": all(checks.values()) if checks else None,
            "unit_cost_usd": {s: {str(m): str(c) for m, c in u.items()} for s, u in UNIT_COST_USD.items()},
            "retry_factor": str(RETRY_FACTOR), "probe_usd_per_run": str(PROBE_USD_PER_RUN), "runs": runs}
