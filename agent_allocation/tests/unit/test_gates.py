"""Lead-owned: the frozen gate criteria make the decisions the approved plan specifies."""
from __future__ import annotations

from decimal import Decimal

from agent_allocation.contracts import (
    STUDY_ID, Candidate, CompletionStatus, EndpointLock, GenerationKind, GenerationResult, GenerationSpec, OutputKind,
    ParseStatus, TransportStatus, Usage,
)
from agent_allocation.gates import CallRow, evaluate_gate1, evaluate_gate2

LITE = EndpointLock("google/gemini-2.5-flash-lite", "google-ai-studio", "Google AI Studio", "standard",
                    "0.0000001", "0.0000004", "0.0000004", 1048576, 65535, ("reasoning",))
FLASH = EndpointLock("google/gemini-2.5-flash", "google-ai-studio", "Google AI Studio", "standard",
                     "0.0000003", "0.0000025", "0.0000025", 1048576, 65535, ("reasoning",))
LOCKS = {LITE.model: LITE, FLASH.model: FLASH}


def row(*, benchmark="math500", budget=1024, reasoning=300, model=LITE.model, cache="MISS", provider="Google AI Studio",
        transport=TransportStatus.OK, completion=CompletionStatus.COMPLETED, parse=ParseStatus.VALID, slot=0,
        service_tier="standard", price_scale=Decimal("1")):
    lock = LOCKS[model]
    spec = GenerationSpec(run_id=STUDY_ID, partition_id="canary", item_id=f"{benchmark}/x{slot}", benchmark=benchmark,
                          kind=GenerationKind.ROOT, model=model, endpoint_lock_hash=lock.lock_hash,
                          input_manifest_hash="i", decoding_config_hash="d", prompt_version="p",
                          reasoning_budget=budget, completion_headroom=1024, root_slot=slot, team_size=0,
                          update_round=0, draw_id=0)
    prompt, completion_tokens = 500, reasoning + 100
    cost = (Decimal(prompt) * Decimal(lock.input_price_per_token)
            + Decimal(completion_tokens) * Decimal(lock.output_price_per_token)) * price_scale
    usage = Usage(prompt, completion_tokens, reasoning, 0, cost)
    result = GenerationResult(spec.node_id, spec.node_id + ":1", transport, completion, "gen", model, provider,
                              "stop", "STOP", "<final>1</final>", False, usage, cache, None, None, 1.0, 200)
    cand = Candidate(spec.node_id, spec.node_id, spec.item_id, "<final>1</final>", "1", None, parse, "stop",
                     OutputKind.MATH)
    return CallRow(spec, result, cand, {"provider_name": provider, "service_tier": service_tier})


def canary_rows(**overrides):
    rows = []
    for slot, bench in enumerate(("math500", "mmlu_pro", "bbh_logic4")):
        for budget, reasoning in ((512, 480), (1024, 900), (2048, 1500), (4096, 2600), (8192, 2900), (16384, 3100)):
            rows.append(row(benchmark=bench, budget=budget, reasoning=reasoning, slot=slot, **overrides))
    rows.append(row(model=FLASH.model, reasoning=700, slot=9, **overrides))
    return rows


def ledger_total(rows):
    return sum(r.result.usage.actual_cost_usd for r in rows)


def test_gate1_advances_on_a_clean_canary():
    rows = canary_rows()
    result = evaluate_gate1(rows, LOCKS, key_usage_delta_usd=ledger_total(rows), ledger_settled_usd=ledger_total(rows))
    assert result.passed and result.decision == "advance", [c for c in result.criteria if not c.passed]


def test_gate1_stops_on_cache_hit_flex_pricing_or_ignored_budget():
    rows = canary_rows()
    total = ledger_total(rows)
    hit = [*rows[:-1], row(model=FLASH.model, reasoning=700, slot=9, cache="HIT")]
    assert evaluate_gate1(hit, LOCKS, key_usage_delta_usd=total, ledger_settled_usd=total).decision == "stop_notify"
    flex = canary_rows(price_scale=Decimal("0.5"))
    assert evaluate_gate1(flex, LOCKS, key_usage_delta_usd=ledger_total(flex),
                          ledger_settled_usd=ledger_total(flex)).decision == "stop_notify"
    flat = [row(benchmark=r.spec.benchmark, budget=r.spec.reasoning_budget, reasoning=900, slot=r.spec.root_slot,
                model=r.spec.model) for r in rows]
    assert evaluate_gate1(flat, LOCKS, key_usage_delta_usd=ledger_total(flat),
                          ledger_settled_usd=ledger_total(flat)).decision == "stop_notify"


def test_gate1_stops_when_provider_bills_more_than_the_ledger_settled():
    rows = canary_rows()
    total = ledger_total(rows)
    result = evaluate_gate1(rows, LOCKS, key_usage_delta_usd=total + Decimal("0.05"), ledger_settled_usd=total)
    assert result.decision == "stop_notify"


def pilot_rows(truncate_benchmark=None):
    rows = []
    for bench in ("math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"):
        for slot in range(40):
            if bench == truncate_benchmark and slot < 4:
                rows.append(row(benchmark=bench, slot=slot, completion=CompletionStatus.LENGTH,
                                parse=ParseStatus.MISSING))
            else:
                rows.append(row(benchmark=bench, slot=slot))
    return rows


GOOD = dict(duplicate_terminal_nodes=0, budget_stop_max_overshoot_usd=Decimal("0"), selector_rows_missing=0,
            code_candidates=40, code_public_rows=40, code_private_grades=40, evaluator_errors=0, graded_total=160,
            packet_envelope_violations=0, core_forecast_usd=Decimal("80"))


def test_gate2_advances_on_a_clean_pilot():
    assert evaluate_gate2(pilot_rows(), **GOOD).decision == "advance"


def test_gate2_truncation_gets_exactly_one_headroom_amendment_per_family():
    rows = pilot_rows(truncate_benchmark="mbpp_plus")
    assert evaluate_gate2(rows, **GOOD).decision == "amend_headroom_once"
    assert evaluate_gate2(rows, **GOOD, amendments_used=["headroom_code"]).decision == "stop_notify"


def test_gate2_any_other_failure_stops():
    assert evaluate_gate2(pilot_rows(), **{**GOOD, "duplicate_terminal_nodes": 1}).decision == "stop_notify"
    assert evaluate_gate2(pilot_rows(), **{**GOOD, "core_forecast_usd": Decimal("126")}).decision == "stop_notify"
    mixed = {**GOOD, "evaluator_errors": 5}
    assert evaluate_gate2(pilot_rows(truncate_benchmark="math500"), **mixed).decision == "stop_notify"
