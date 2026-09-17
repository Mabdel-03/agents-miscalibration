"""Gate 1 capability canary summary (MASTER_SPEC section 13): facts per requested reasoning budget, no verdicts.

``canary_report(rows)`` takes ``(spec, result, generation_metadata_or_None)`` triples and returns a JSON-serializable
dict (schema ``CANARY_REPORT_SCHEMA``):

* ``budgets``: one summary per ``spec.reasoning_budget`` (ascending); ``overall``: the same summary for all rows.
  A summary holds counts of transport statuses, completion statuses, finish and native finish reasons, cache
  statuses, observed providers/models; shares of completed/length/empty-content/reasoning-returned among
  transport-OK rows; distributions (``n``, ``n_none``, ``share_none``, ``min``, ``median``, ``max``) of reasoning,
  completion, visible (completion - reasoning), prompt and cached tokens, reasoning/budget ratio, headroom
  utilization (completion tokens / ``spec.max_completion_tokens``) and latency; counts of rows over the requested
  budget, at the completion cap, and with reasoning > completion (an accounting inconsistency); generation-metadata
  identity (provider_name, service_tier, response-cache source ids, native reasoning tokens, disagreements with the
  response usage); and money as decimal strings (``total_known``, ``unknown_count``).
* ``cross_budget``: median reasoning tokens per budget, whether they are non-decreasing, and how many are distinct.
* ``flags``: sorted fact strings the lead's Gate 1 evaluator can match on (e.g. ``A=1024:reasoning_tokens_unaudited``,
  ``overall:cache_hit_observed``, ``cross_budget:reasoning_budget_possibly_ignored``). Thresholds and pass/fail are the
  evaluator's decision, not this module's.
* ``rows``: one compact per-attempt row for the evidence file.

Token distributions use transport-OK rows only; cost totals use every row that reports a cost.
"""
from __future__ import annotations

import statistics
from collections import Counter
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.api.normalize import parse_money
from agent_allocation.contracts import CompletionStatus, GenerationResult, GenerationSpec, TransportStatus

CANARY_REPORT_SCHEMA = "agent_allocation.canary_report/v1"
Row = tuple[GenerationSpec, GenerationResult, Mapping[str, Any] | None]


def _key(value: Any) -> str:
    if value is None:
        return "null"
    return str(getattr(value, "value", value))


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(_key(v) for v in values).items()))


def _number(value: Any) -> float | int:
    return int(value) if isinstance(value, int) or (isinstance(value, float) and value.is_integer()) else float(value)


def _distribution(values: Sequence[float | int | None]) -> dict[str, Any]:
    known = [v for v in values if v is not None]
    total = len(values)
    return {
        "n": len(known),
        "n_none": total - len(known),
        "share_none": None if total == 0 else (total - len(known)) / total,
        "min": _number(min(known)) if known else None,
        "median": _number(statistics.median(known)) if known else None,
        "max": _number(max(known)) if known else None,
    }


def _share(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_empty(content: str | None) -> bool:
    return content is None or not content.strip()


def _summary(budget: int | None, rows: Sequence[Row]) -> dict[str, Any]:
    ok = [(spec, result, meta) for spec, result, meta in rows if result.transport_status is TransportStatus.OK]
    usage_of = [(spec, result.usage, meta) for spec, result, meta in ok]
    reasoning = [u.reasoning_tokens if u else None for _, u, _ in usage_of]
    completion = [u.completion_tokens if u else None for _, u, _ in usage_of]
    visible = [(c - r) if (c is not None and r is not None) else None for c, r in zip(completion, reasoning)]
    ratios = [None if r is None else r / spec.reasoning_budget for (spec, _, _), r in zip(usage_of, reasoning)]
    headroom = [None if c is None else c / spec.max_completion_tokens for (spec, _, _), c in zip(usage_of, completion)]

    metadata = [(result, meta) for _, result, meta in rows if meta is not None]
    native_reasoning = [_int_or_none(meta.get("native_tokens_reasoning")) for _, meta in metadata]
    reasoning_disagreements = sum(
        1 for (result, meta), native in zip(metadata, native_reasoning)
        if native is not None and result.usage is not None and result.usage.reasoning_tokens is not None
        and native != result.usage.reasoning_tokens)
    cost_disagreements = 0
    for result, meta in metadata:
        meta_cost = parse_money(meta.get("total_cost"))
        usage_cost = result.usage.actual_cost_usd if result.usage else None
        if meta_cost is not None and usage_cost is not None and meta_cost != usage_cost:
            cost_disagreements += 1

    costs = [r.usage.actual_cost_usd if r.usage else None for _, r, _ in rows]
    known_costs = [c for c in costs if c is not None]
    total_cost = sum(known_costs, Decimal("0"))
    return {
        "requested_reasoning_budget": budget,
        "count": len(rows),
        "transport_ok_count": len(ok),
        "max_completion_tokens": sorted({spec.max_completion_tokens for spec, _, _ in rows}),
        "transport_statuses": _counts(r.transport_status for _, r, _ in rows),
        "completion_statuses": _counts(r.completion_status for _, r, _ in ok),
        "finish_reasons": _counts(r.finish_reason for _, r, _ in ok),
        "native_finish_reasons": _counts(r.native_finish_reason for _, r, _ in ok),
        "completed_share": _share(sum(r.completion_status is CompletionStatus.COMPLETED for _, r, _ in ok), len(ok)),
        "length_share": _share(sum(r.completion_status is CompletionStatus.LENGTH for _, r, _ in ok), len(ok)),
        "content_empty_count": sum(_is_empty(r.visible_content) for _, r, _ in ok),
        "content_empty_share": _share(sum(_is_empty(r.visible_content) for _, r, _ in ok), len(ok)),
        "reasoning_returned_share": _share(sum(r.reasoning_returned for _, r, _ in ok), len(ok)),
        "reasoning_tokens": _distribution(reasoning),
        "reasoning_budget_ratio": _distribution(ratios),
        "reasoning_over_budget_count": sum(1 for (spec, _, _), r in zip(usage_of, reasoning)
                                           if r is not None and r > spec.reasoning_budget),
        "reasoning_exceeds_completion_count": sum(1 for c, r in zip(completion, reasoning)
                                                  if c is not None and r is not None and r > c),
        "completion_tokens": _distribution(completion),
        "visible_tokens": _distribution(visible),
        "prompt_tokens": _distribution([u.prompt_tokens if u else None for _, u, _ in usage_of]),
        "cached_prompt_tokens": _distribution([u.cached_prompt_tokens if u else None for _, u, _ in usage_of]),
        "headroom_utilization": _distribution(headroom),
        "at_completion_cap_count": sum(1 for (spec, _, _), c in zip(usage_of, completion)
                                       if c is not None and c >= spec.max_completion_tokens),
        "latency_seconds": _distribution([r.latency_seconds for _, r, _ in rows]),
        "observed_providers": _counts(r.observed_provider for _, r, _ in rows),
        "observed_models": _counts(r.observed_model for _, r, _ in rows),
        "cache_statuses": _counts(r.cache_status for _, r, _ in rows),
        "metadata": {
            "available": len(metadata),
            "missing": len(rows) - len(metadata),
            "provider_names": _counts(meta.get("provider_name") for _, meta in metadata),
            "service_tiers": _counts(meta.get("service_tier") for _, meta in metadata),
            "models": _counts(meta.get("model") for _, meta in metadata),
            "native_tokens_reasoning": _distribution(native_reasoning),
            "response_cache_source_ids": sum(1 for _, meta in metadata if meta.get("response_cache_source_id")),
            "reasoning_usage_disagreements": reasoning_disagreements,
            "cost_disagreements": cost_disagreements,
        },
        "cost_usd": {
            "total_known": str(total_cost),
            "known_count": len(known_costs),
            "unknown_count": len(costs) - len(known_costs),
            "mean_known": str(total_cost / len(known_costs)) if known_costs else None,
        },
    }


def _flags(scope: str, summary: Mapping[str, Any]) -> list[str]:
    flags = []
    for status, count in summary["transport_statuses"].items():
        if status != TransportStatus.OK.value:
            flags.append(f"{scope}:transport_{status}={count}")
    if summary["transport_statuses"].get(TransportStatus.CACHE_HIT.value) or \
            summary["cache_statuses"].get("HIT") or summary["metadata"]["response_cache_source_ids"]:
        flags.append(f"{scope}:cache_hit_observed")
    if summary["transport_statuses"].get(TransportStatus.PROVIDER_MISMATCH.value):
        flags.append(f"{scope}:provider_mismatch_observed")
    if len([p for p in summary["observed_providers"] if p != "null"]) > 1:
        flags.append(f"{scope}:multiple_providers_observed")
    reasoning = summary["reasoning_tokens"]
    if summary["transport_ok_count"] and reasoning["n"] == 0:
        flags.append(f"{scope}:reasoning_tokens_unaudited")
    elif reasoning["n_none"]:
        flags.append(f"{scope}:reasoning_tokens_partially_missing={reasoning['n_none']}")
    if summary["reasoning_exceeds_completion_count"]:
        flags.append(f"{scope}:reasoning_exceeds_completion={summary['reasoning_exceeds_completion_count']}")
    if summary["reasoning_over_budget_count"]:
        flags.append(f"{scope}:reasoning_over_requested_budget={summary['reasoning_over_budget_count']}")
    if summary["content_empty_count"]:
        flags.append(f"{scope}:empty_content={summary['content_empty_count']}")
    if summary["completion_statuses"].get(CompletionStatus.LENGTH.value):
        flags.append(f"{scope}:length_finishes={summary['completion_statuses'][CompletionStatus.LENGTH.value]}")
    metadata = summary["metadata"]
    if metadata["missing"]:
        flags.append(f"{scope}:generation_metadata_missing={metadata['missing']}")
    for tier in metadata["service_tiers"]:
        if tier == "null":
            flags.append(f"{scope}:service_tier_unknown")
        elif tier != "standard":
            flags.append(f"{scope}:service_tier_not_standard:{tier}")
    if len([p for p in metadata["provider_names"] if p != "null"]) > 1:
        flags.append(f"{scope}:multiple_metadata_providers")
    if metadata["reasoning_usage_disagreements"]:
        flags.append(f"{scope}:reasoning_usage_disagrees_with_metadata={metadata['reasoning_usage_disagreements']}")
    if metadata["cost_disagreements"]:
        flags.append(f"{scope}:cost_disagrees_with_metadata={metadata['cost_disagreements']}")
    if summary["cost_usd"]["unknown_count"]:
        flags.append(f"{scope}:cost_unknown={summary['cost_usd']['unknown_count']}")
    return flags


def _row(spec: GenerationSpec, result: GenerationResult, meta: Mapping[str, Any] | None) -> dict[str, Any]:
    usage = result.usage
    return {
        "node_id": spec.node_id,
        "attempt_id": result.attempt_id,
        "item_id": spec.item_id,
        "benchmark": spec.benchmark,
        "requested_reasoning_budget": spec.reasoning_budget,
        "max_completion_tokens": spec.max_completion_tokens,
        "transport_status": result.transport_status.value,
        "completion_status": _key(result.completion_status) if result.completion_status else None,
        "finish_reason": result.finish_reason,
        "native_finish_reason": result.native_finish_reason,
        "reasoning_tokens": usage.reasoning_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "cost_usd": None if usage is None or usage.actual_cost_usd is None else str(usage.actual_cost_usd),
        "content_empty": _is_empty(result.visible_content),
        "reasoning_returned": result.reasoning_returned,
        "observed_provider": result.observed_provider,
        "observed_model": result.observed_model,
        "cache_status": result.cache_status,
        "generation_id": result.generation_id,
        "metadata_provider_name": None if meta is None else meta.get("provider_name"),
        "metadata_service_tier": None if meta is None else meta.get("service_tier"),
        "metadata_native_tokens_reasoning": None if meta is None else _int_or_none(meta.get("native_tokens_reasoning")),
    }


def canary_report(rows: Sequence[tuple[GenerationSpec, GenerationResult, Mapping[str, Any] | None]]) -> dict:
    rows = list(rows)
    for spec, result, _ in rows:
        if result.node_id != spec.node_id:
            raise ValueError(f"result {result.attempt_id} does not belong to spec {spec.node_id}")
    by_budget: dict[int, list[Row]] = {}
    for row in rows:
        by_budget.setdefault(row[0].reasoning_budget, []).append(row)
    budgets = [_summary(budget, by_budget[budget]) for budget in sorted(by_budget)]
    overall = _summary(None, rows)
    flags = [flag for summary in budgets for flag in _flags(f"A={summary['requested_reasoning_budget']}", summary)]
    flags += _flags("overall", overall)
    medians = {str(s["requested_reasoning_budget"]): s["reasoning_tokens"]["median"] for s in budgets}
    known_medians = [m for m in medians.values() if m is not None]
    nondecreasing = None if len(known_medians) < 2 else all(a <= b for a, b in zip(known_medians,
                                                                                    known_medians[1:]))
    if nondecreasing is False:
        flags.append("cross_budget:median_reasoning_not_nondecreasing")
    if len(known_medians) >= 2 and len(set(known_medians)) == 1:
        flags.append("cross_budget:reasoning_budget_possibly_ignored")
    return {
        "schema": CANARY_REPORT_SCHEMA,
        "row_count": len(rows),
        "budgets": budgets,
        "overall": overall,
        "cross_budget": {
            "median_reasoning_tokens_by_budget": medians,
            "median_nondecreasing": nondecreasing,
            "distinct_medians": len(set(known_medians)),
        },
        "flags": sorted(flags),
        "rows": [_row(spec, result, meta) for spec, result, meta in rows],
    }
