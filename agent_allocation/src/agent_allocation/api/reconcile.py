"""Generation-metadata reconciliation (MASTER_SPEC 7.1, 10.5). Identity and cost only: a lost completion body is
never "recovered" from metadata, and an attempt without a generation id cannot be reconciled by id at all (settle
it conservatively or through key-level usage).

``metadata_identity_problems`` is also the only way to detect a *tier* drift: the response's ``provider`` display name
is identical for ``google-ai-studio``, ``google-ai-studio/flex`` and ``google-ai-studio/priority`` (all
"Google AI Studio"), so ``generate`` cannot see it; ``/generation`` reports ``service_tier``.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from agent_allocation.api.normalize import parse_money
from agent_allocation.contracts import EndpointLock, GenerationResult

PENDING = "pending"                # metadata not available yet (404 / None): retain the reservation
CONSISTENT = "consistent"          # identity matches the lock and cost/usage agree
DISCREPANT = "discrepant"          # identity or accounting disagrees: open the circuit / audit
UNRECONCILABLE = "unreconcilable"  # no generation id (e.g. ambiguous timeout): settle conservatively


@dataclass(frozen=True)
class Reconciliation:
    attempt_id: str
    generation_id: str | None
    state: str
    metadata_cost_usd: Decimal | None      # what /generation says was charged
    result_cost_usd: Decimal | None        # what the response usage said
    native_reasoning_tokens: int | None
    problems: tuple[str, ...] = ()


# Observed live on 2026-09-17 (Gate 1 canary, 22/22 calls): /generation reports service_tier "default" for the
# standard google-ai-studio endpoint; flex and priority report their own names. A standard-tier lock accepts both.
SERVICE_TIER_ALIASES = {"standard": {"standard", "default"}}

def metadata_identity_problems(metadata: Mapping[str, Any], lock: EndpointLock) -> list[str]:
    problems = []
    provider = metadata.get("provider_name")
    if provider is None:
        problems.append("metadata provider_name unknown")
    elif provider != lock.provider_name:
        problems.append(f"metadata provider_name {provider!r} != locked {lock.provider_name!r}")
    model = metadata.get("model")
    if model is not None and model != lock.model:
        problems.append(f"metadata model {model!r} != locked {lock.model!r}")
    tier = metadata.get("service_tier")
    if tier is None:
        problems.append("metadata service_tier unknown")
    elif tier not in SERVICE_TIER_ALIASES.get(lock.service_tier, {lock.service_tier}):
        problems.append(f"metadata service_tier {tier!r} != locked {lock.service_tier!r}")
    if metadata.get("response_cache_source_id"):
        problems.append(f"served from the response cache (source {metadata.get('response_cache_source_id')!r})")
    return problems


def reconcile_generation(result: GenerationResult, metadata: Mapping[str, Any] | None,
                         lock: EndpointLock) -> Reconciliation:
    result_cost = result.usage.actual_cost_usd if result.usage else None
    if result.generation_id is None:
        return Reconciliation(result.attempt_id, None, UNRECONCILABLE, None, result_cost, None,
                              ("no generation id: cost cannot be confirmed by id",))
    if metadata is None:
        return Reconciliation(result.attempt_id, result.generation_id, PENDING, None, result_cost, None)
    if metadata.get("id") not in (None, result.generation_id):
        raise ValueError(f"metadata {metadata.get('id')!r} is not for generation {result.generation_id!r}")
    problems = metadata_identity_problems(metadata, lock)
    metadata_cost = parse_money(metadata.get("total_cost"))
    if metadata_cost is None:
        problems.append("metadata total_cost unknown")
    elif result_cost is not None and metadata_cost != result_cost:
        problems.append(f"charged {metadata_cost} != response usage cost {result_cost}")
    native = metadata.get("native_tokens_reasoning")
    native_reasoning = native if isinstance(native, int) and not isinstance(native, bool) else None
    usage_reasoning = result.usage.reasoning_tokens if result.usage else None
    if native_reasoning is not None and usage_reasoning is not None and native_reasoning != usage_reasoning:
        problems.append(f"native reasoning tokens {native_reasoning} != response usage {usage_reasoning}")
    return Reconciliation(result.attempt_id, result.generation_id, DISCREPANT if problems else CONSISTENT,
                          metadata_cost, result_cost, native_reasoning, tuple(problems))


def merge_forced_metadata(think: Mapping[str, Any], answer: Mapping[str, Any]) -> dict[str, Any]:
    """One ``/generation``-shaped record for a forced node (THINK + ANSWER): the ANSWER's id, identity fields that
    must agree across both calls (a disagreement surfaces as a value no lock can match), costs and token counts
    summed. ``native_tokens_reasoning`` is left ``None`` (a raw completion reports no reasoning split; the node's
    reasoning is THINK's completion tokens, kept under ``forced.think``)."""
    def same(key: str) -> Any:
        a, t = answer.get(key), think.get(key)
        if a == t or t is None:
            return a
        if a is None:
            return t
        return f"{t} | {a}"                                 # THINK and ANSWER disagree: never equals a locked value

    def total(key: str) -> Any:
        values = [think.get(key), answer.get(key)]
        if any(v is None or isinstance(v, bool) for v in values):
            return None
        money = [parse_money(v) for v in values]
        if any(m is None for m in money):
            return None
        return money[0] + money[1]

    def count(key: str) -> int | None:
        values = [think.get(key), answer.get(key)]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return values[0] + values[1]
        return None

    return {
        "id": answer.get("id"),
        "provider_name": same("provider_name"),
        "model": same("model"),
        "service_tier": same("service_tier"),
        "response_cache_source_id": think.get("response_cache_source_id") or answer.get("response_cache_source_id"),
        "total_cost": total("total_cost"),
        "tokens_prompt": count("tokens_prompt"),
        "tokens_completion": count("tokens_completion"),
        "native_tokens_prompt": count("native_tokens_prompt"),
        "native_tokens_completion": count("native_tokens_completion"),
        "native_tokens_reasoning": None,
        "finish_reason": answer.get("finish_reason"),
        "forced": {"think": dict(think), "answer": dict(answer)},
    }


def reconcile_forced_generation(result: GenerationResult, think_metadata: Mapping[str, Any] | None,
                                answer_metadata: Mapping[str, Any] | None, lock: EndpointLock) -> Reconciliation:
    """``reconcile_generation`` for a forced node: both generation ids must be known to /generation; identity is
    checked on both and the charged cost is THINK + ANSWER."""
    if result.generation_id is None:
        return reconcile_generation(result, None, lock)
    if think_metadata is None or answer_metadata is None:
        result_cost = result.usage.actual_cost_usd if result.usage else None
        return Reconciliation(result.attempt_id, result.generation_id, PENDING, None, result_cost, None)
    merged = merge_forced_metadata(think_metadata, answer_metadata)
    reconciliation = reconcile_generation(result, merged, lock)
    extra = [f"think: {p}" for p in metadata_identity_problems(think_metadata, lock)]
    if extra and reconciliation.state != DISCREPANT:
        return Reconciliation(reconciliation.attempt_id, reconciliation.generation_id, DISCREPANT,
                              reconciliation.metadata_cost_usd, reconciliation.result_cost_usd,
                              reconciliation.native_reasoning_tokens, reconciliation.problems + tuple(extra))
    return reconciliation
