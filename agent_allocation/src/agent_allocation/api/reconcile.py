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
    elif tier != lock.service_tier:
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
