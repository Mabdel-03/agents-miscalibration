"""Frozen auto-advance gate criteria (lead-owned; approved plan "Frozen auto-advance criteria").

Committed before the canary. Every criterion is computed from written artifacts. Decisions:
``advance`` (all criteria pass), ``amend_headroom_once`` (only the pre-authorized truncation amendment applies and has
not been used for that benchmark family), ``stop_notify`` (anything else). Floor/ceiling effects are reported but
never block.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.contracts import (
    BENCHMARKS, CompletionStatus, EndpointLock, GateCriterion, GateResult, GenerationResult, GenerationSpec,
    ParseStatus, Candidate, TransportStatus, dump_record,
)

# ---- Gate 1 thresholds -------------------------------------------------------------------------------------------
G1_REASONING_REPORTED_MIN_SHARE = 0.90
G1_SMALL_BUDGET = 512
G1_SMALL_BUDGET_MAX_REASONING = 4 * 512          # egregious overshoot means the numeric budget is not applied
G1_LARGE_BUDGET_MIN = 4096
G1_STANDARD_PRICE_RATIO = (Decimal("0.90"), Decimal("1.10"))   # flex is ~0.5x, priority ~1.8x
G1_RECONCILE_TOLERANCE_USD = Decimal("0.01")
G1_RECONCILE_TOLERANCE_SHARE = Decimal("0.05")

# ---- Gate 2 thresholds -------------------------------------------------------------------------------------------
G2_VALID_FINAL_MIN_SHARE = 0.95
G2_TRUNCATED_NO_FINAL_MAX_SHARE = 0.03
G2_EVALUATOR_ERROR_MAX_SHARE = 0.01
G2_CORE_FORECAST_MAX_USD = Decimal("125")
HEADROOM_AMENDMENT = {"text": (1024, 2048), "code": (2048, 4096)}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CallRow:
    spec: GenerationSpec
    result: GenerationResult
    candidate: Candidate | None
    metadata: Mapping[str, Any] | None = None      # /generation data when fetched
    raw_payload: Mapping[str, Any] | None = None   # forced mode: the stored raw artifact (THINK finish reason)


def is_forced_row(row: CallRow) -> bool:
    usage = row.result.usage
    raw = usage.raw_usage if usage is not None else None
    return isinstance(raw, Mapping) and "think" in raw and "answer" in raw


def forced_budget_problems(rows: Sequence[CallRow]) -> list[dict[str, Any]]:
    """Forced Gate 1: every transport-OK forced row realized exactly its reasoning budget with THINK finish
    ``length`` (read from the raw payload when attached; ``ForcedClient`` never returns OK otherwise)."""
    from agent_allocation.api.forced import forced_usage_from_raw
    bad = []
    for r in rows:
        if r.result.transport_status != TransportStatus.OK:
            continue
        realized = r.result.usage.reasoning_tokens if r.result.usage is not None else None
        finish = forced_usage_from_raw(r.raw_payload).get("think_finish_reason") if r.raw_payload else "length"
        if realized != r.spec.reasoning_budget or finish != "length":
            bad.append({"node": r.spec.node_id[:12], "budget": r.spec.reasoning_budget, "reasoning": realized,
                        "think_finish_reason": finish})
    return bad


def _crit(name: str, observed: Any, threshold: str, passed: bool, evidence: str) -> GateCriterion:
    return GateCriterion(name=name, observed=json.dumps(observed, default=str, sort_keys=True), threshold=threshold,
                         passed=bool(passed), evidence=evidence)


def standard_price_ratio(row: CallRow, lock: EndpointLock) -> Decimal | None:
    usage = row.result.usage
    if usage is None or usage.actual_cost_usd is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        return None
    expected = (Decimal(usage.prompt_tokens) * Decimal(lock.input_price_per_token)
                + Decimal(usage.completion_tokens) * Decimal(lock.output_price_per_token))
    if expected == 0:
        return None
    return usage.actual_cost_usd / expected


def evaluate_gate1(rows: Sequence[CallRow], locks: Mapping[str, EndpointLock], *, key_usage_delta_usd: Decimal | None,
                   ledger_settled_usd: Decimal, evidence: str = "manifests/jobs/canary.jsonl + parsed/ + state/generation_metadata.jsonl",
                   forced: bool | None = None) -> GateResult:
    """``forced`` (default: detected from the rows) replaces ``numeric_budget_applied`` by ``forced_exact_budget``:
    every canary node's reasoning tokens equal its budget, THINK finish ``length``, at every canary budget."""
    if forced is None:
        forced = bool(rows) and any(is_forced_row(r) for r in rows)
    criteria: list[GateCriterion] = []
    transport = [r.result.transport_status.value for r in rows]
    criteria.append(_crit("all_transport_ok", {"n": len(rows), "not_ok": [t for t in transport if t != "ok"]},
                          "every canary call TransportStatus.ok", bool(rows) and all(t == "ok" for t in transport),
                          evidence))
    cache_hits = [r.spec.node_id for r in rows if (r.result.cache_status or "").upper() == "HIT"]
    criteria.append(_crit("zero_cache_hits", len(cache_hits), "0", not cache_hits, evidence))

    identity_bad = []
    tier_bad = []
    for r in rows:
        lock = locks[r.spec.model]
        served = (r.metadata or {}).get("provider_name") or r.result.observed_provider
        if served != lock.provider_name:
            identity_bad.append({"node": r.spec.node_id[:12], "served": served})
        tier = (r.metadata or {}).get("service_tier")
        ratio = standard_price_ratio(r, lock)
        tier_ok = (tier in (None, "standard", "default")) and ratio is not None and \
            G1_STANDARD_PRICE_RATIO[0] <= ratio <= G1_STANDARD_PRICE_RATIO[1]
        if not tier_ok:
            tier_bad.append({"node": r.spec.node_id[:12], "service_tier": tier, "price_ratio": str(ratio)})
    criteria.append(_crit("provider_identity", identity_bad, "100% served by the locked provider", not identity_bad,
                          evidence))
    criteria.append(_crit("standard_tier_by_metadata_and_price", tier_bad,
                          f"service_tier standard/absent and billed/standard price in {G1_STANDARD_PRICE_RATIO}",
                          not tier_bad, evidence))

    ok_rows = [r for r in rows if r.result.transport_status == TransportStatus.OK]
    reported = [r for r in ok_rows if r.result.usage is not None and r.result.usage.reasoning_tokens is not None]
    share = len(reported) / len(ok_rows) if ok_rows else 0.0
    criteria.append(_crit("reasoning_tokens_reported", round(share, 3), f">= {G1_REASONING_REPORTED_MIN_SHARE}",
                          share >= G1_REASONING_REPORTED_MIN_SHARE, evidence))

    primary = [r for r in reported if r.spec.model == min(locks, key=lambda m: Decimal(locks[m].output_price_per_token))]
    small = [r.result.usage.reasoning_tokens for r in primary if r.spec.reasoning_budget == G1_SMALL_BUDGET]
    large = [r.result.usage.reasoning_tokens for r in primary if r.spec.reasoning_budget >= G1_LARGE_BUDGET_MIN]
    by_budget = {}
    for r in primary:
        by_budget.setdefault(r.spec.reasoning_budget, []).append(r.result.usage.reasoning_tokens)
    if forced:
        bad = forced_budget_problems(rows)
        not_forced = [r.spec.node_id[:12] for r in ok_rows if not is_forced_row(r)]
        criteria.append(_crit("forced_exact_budget",
                              {"budgets": {str(k): {"n": len(v), "min": min(v), "max": max(v)}
                                           for k, v in sorted(by_budget.items())},
                               "not_exact": bad, "not_forced": not_forced},
                              "every canary node: reasoning_tokens == reasoning_budget and THINK finish_reason length",
                              bool(ok_rows) and not bad and not not_forced, evidence))
    else:
        responsive = bool(small) and bool(large) and max(large) > max(small) and \
            max(small) <= G1_SMALL_BUDGET_MAX_REASONING
        criteria.append(_crit("numeric_budget_applied",
                              {str(k): {"n": len(v), "median": median(v), "max": max(v)}
                               for k, v in sorted(by_budget.items())},
                              f"max reasoning at A>={G1_LARGE_BUDGET_MIN} exceeds max at A={G1_SMALL_BUDGET}, and A=512 "
                              f"never exceeds {G1_SMALL_BUDGET_MAX_REASONING}", responsive, evidence))

    bad_request = [r.spec.reasoning_budget for r in rows if r.result.transport_status == TransportStatus.BAD_REQUEST]
    criteria.append(_crit("no_rejected_budget", bad_request, "no 4xx for any configured A", not bad_request, evidence))

    if key_usage_delta_usd is None:
        criteria.append(_crit("ledger_reconciles_with_key_usage", None, "provider usage delta available", False,
                              "state/provider_usage.jsonl"))
    else:
        tolerance = max(G1_RECONCILE_TOLERANCE_USD, ledger_settled_usd * G1_RECONCILE_TOLERANCE_SHARE)
        diff = key_usage_delta_usd - ledger_settled_usd
        criteria.append(_crit("ledger_reconciles_with_key_usage",
                              {"key_delta": str(key_usage_delta_usd), "ledger_settled": str(ledger_settled_usd)},
                              f"ledger settled >= key delta - {tolerance}", diff <= tolerance,
                              "state/provider_usage.jsonl"))
    passed = all(c.passed for c in criteria)
    return GateResult(gate="gate1", stage="canary", passed=passed, decision="advance" if passed else "stop_notify",
                      criteria=tuple(criteria), evaluated_at=utcnow())


def family_of(benchmark: str) -> str:
    from agent_allocation.config import CODE_BENCHMARKS
    return "code" if benchmark in CODE_BENCHMARKS else "text"


def _study_has_code() -> bool:
    from agent_allocation.config import CODE_BENCHMARKS
    return any(benchmark in CODE_BENCHMARKS for benchmark in BENCHMARKS)


def evaluate_gate2(rows: Sequence[CallRow], *, duplicate_terminal_nodes: int, budget_stop_max_overshoot_usd: Decimal,
                   selector_rows_missing: int, code_candidates: int, code_public_rows: int, code_private_grades: int,
                   evaluator_errors: int, graded_total: int, packet_envelope_violations: int,
                   core_forecast_usd: Decimal, amendments_used: Iterable[str] = (),
                   evidence: str = "reports/PILOT_REPORT.md",
                   core_forecast_max_usd: Decimal = G2_CORE_FORECAST_MAX_USD) -> GateResult:
    """``core_forecast_max_usd``: the run's (a shard's) ``budget.soft_admission_ceiling_usd``; v1's 125 by default."""
    criteria: list[GateCriterion] = []
    truncation_failures: list[str] = []
    for benchmark in BENCHMARKS:
        base = [r for r in rows if r.spec.benchmark == benchmark and r.spec.reasoning_budget == 1024
                and r.result.transport_status == TransportStatus.OK]
        valid = [r for r in base if r.candidate is not None and r.candidate.parse_status == ParseStatus.VALID]
        truncated = [r for r in base if r.result.completion_status == CompletionStatus.LENGTH
                     and (r.candidate is None or r.candidate.parse_status != ParseStatus.VALID)]
        valid_share = len(valid) / len(base) if base else 0.0
        trunc_share = len(truncated) / len(base) if base else 1.0
        criteria.append(_crit(f"{benchmark}_valid_final", {"n": len(base), "share": round(valid_share, 3)},
                              f">= {G2_VALID_FINAL_MIN_SHARE}", bool(base) and valid_share >= G2_VALID_FINAL_MIN_SHARE,
                              evidence))
        trunc_ok = bool(base) and trunc_share <= G2_TRUNCATED_NO_FINAL_MAX_SHARE
        criteria.append(_crit(f"{benchmark}_truncated_without_final", {"n": len(base), "share": round(trunc_share, 3)},
                              f"<= {G2_TRUNCATED_NO_FINAL_MAX_SHARE}", trunc_ok, evidence))
        if not trunc_ok and base:
            truncation_failures.append(benchmark)
    hits = sum(1 for r in rows if (r.result.cache_status or "").upper() == "HIT")
    criteria.append(_crit("zero_cache_hits", hits, "0", hits == 0, evidence))
    drift = sum(1 for r in rows if r.result.transport_status == TransportStatus.PROVIDER_MISMATCH)
    criteria.append(_crit("zero_provider_drift", drift, "0", drift == 0, evidence))
    criteria.append(_crit("resume_no_duplicate_terminal_nodes", duplicate_terminal_nodes, "0",
                          duplicate_terminal_nodes == 0, "state/ledger_journal.jsonl"))
    criteria.append(_crit("budget_stop_never_over_reserved", str(budget_stop_max_overshoot_usd), "<= 0",
                          budget_stop_max_overshoot_usd <= 0, "state/ledger_journal.jsonl"))
    criteria.append(_crit("selector_rows_populated", selector_rows_missing, "0 missing", selector_rows_missing == 0,
                          "selections/pilot.jsonl"))
    criteria.append(_crit("code_public_and_private_rows",
                          {"candidates": code_candidates, "public": code_public_rows, "private": code_private_grades},
                          "every code candidate has both (vacuous when the study has no code benchmark)",
                          (code_candidates > 0 or not _study_has_code()) and code_public_rows == code_candidates
                          and code_private_grades == code_candidates, "grades/"))
    err_share = evaluator_errors / graded_total if graded_total else 1.0
    criteria.append(_crit("evaluator_error_share", round(err_share, 4), f"<= {G2_EVALUATOR_ERROR_MAX_SHARE}",
                          graded_total > 0 and err_share <= G2_EVALUATOR_ERROR_MAX_SHARE, "grades/"))
    criteria.append(_crit("packet_envelope_respected", packet_envelope_violations, "0",
                          packet_envelope_violations == 0, evidence))
    criteria.append(_crit("core_forecast_within_soft_ceiling", str(core_forecast_usd), f"<= {core_forecast_max_usd}",
                          core_forecast_usd <= core_forecast_max_usd, evidence))
    passed = all(c.passed for c in criteria)
    if passed:
        decision = "advance"
    else:
        failing = {c.name for c in criteria if not c.passed}
        amendable = {f"{b}_truncated_without_final" for b in truncation_failures}
        # A benchmark failing validity only because of truncation is covered by the same single amendment.
        amendable |= {f"{b}_valid_final" for b in truncation_failures}
        families = {family_of(b) for b in truncation_failures}
        unused = all(f"headroom_{family}" not in set(amendments_used) for family in families)
        decision = "amend_headroom_once" if failing <= amendable and truncation_failures and unused else "stop_notify"
    return GateResult(gate="gate2", stage="pilot", passed=passed, decision=decision, criteria=tuple(criteria),
                      evaluated_at=utcnow())


def write_gate(paths, result: GateResult) -> None:
    target = paths.gate_result(result.gate, result.stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dump_record(result), indent=1, sort_keys=True))


EXIT_ADVANCE, EXIT_STOP, EXIT_AMEND = 0, 3, 4


def cmd_gates(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation gates")
    parser.add_argument("--stage", choices=["canary", "pilot"], required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    from agent_allocation.paths import RunPaths
    from agent_allocation.contracts import load_record
    paths = RunPaths.for_run(args.run_dir)
    gate = "gate1" if args.stage == "canary" else "gate2"
    target = paths.gate_result(gate, args.stage)
    if not target.exists():
        print(f"no evaluated {gate} for stage {args.stage}; run `allocation {args.stage}` first")
        return EXIT_STOP
    result = load_record(GateResult, json.loads(target.read_text()))
    print(json.dumps(dump_record(result), indent=1, sort_keys=True))
    return {"advance": EXIT_ADVANCE, "amend_headroom_once": EXIT_AMEND}.get(result.decision, EXIT_STOP)
