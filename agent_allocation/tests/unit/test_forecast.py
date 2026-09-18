"""Lead-owned: the forecast reproduces the pack's saturated-cap estimate and honors the cut order."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from agent_allocation.config import grids_from_config, load_config
from agent_allocation.contracts import BENCHMARKS, EndpointLock, GenerationKind
from agent_allocation.forecast import (
    BenchmarkModel, SAFETY_MULTIPLIER, call_mix, core_forecast, fit_models, Observation, per_item_costs, wave_admission,
)

CFG = load_config(Path(__file__).resolve().parents[2] / "configs" / "balanced150.json")
GRIDS = grids_from_config(CFG)
LITE = EndpointLock("google/gemini-2.5-flash-lite", "google-ai-studio", "Google AI Studio", "standard",
                    "0.0000001", "0.0000004", "0.0000004", 1048576, 65535, ())


def test_call_mix_counts_match_the_grid_counts():
    assert len(call_mix(GRIDS["main"], 1024)) == 125
    assert len(call_mix(GRIDS["main_extension"], 1024)) == 138
    assert len(call_mix(GRIDS["high"], 4096)) == 123
    assert len(call_mix(GRIDS["replication"], 1024)) == 57


def test_saturated_caps_reproduce_the_pack_estimate():
    # Pack assumption: 2,500 input tokens on every call and completion caps saturated.
    saturated = BenchmarkModel(root_prompt=2500, private_prompt=2500, packet_tokens=0, completion={})
    models = {b: saturated for b in BENCHMARKS}
    headroom = {b: 1024 for b in BENCHMARKS}             # four text benchmarks: no 2,048-token code headroom
    costs = per_item_costs(models, GRIDS, lock=LITE, headroom_by_benchmark=headroom, main_a=1024, high_a=4096)
    per = {"main_items": 100, "extension_items": 48, "high_items": 24}
    unsafe = core_forecast({b: {k: v / SAFETY_MULTIPLIER for k, v in c.items()} for b, c in costs.items()}, per)
    # The pack estimate (estimate_budget.py) was 94.73 with one code benchmark at 2,048 headroom; with the MBPP+ family
    # replaced by a text benchmark at 1,024 headroom the same saturated-cap arithmetic gives 88.16.
    assert Decimal("88.13") <= unsafe <= Decimal("88.19")


def test_measured_usage_lowers_the_forecast_and_peer_prompts_grow_with_team_size():
    obs = []
    for b in BENCHMARKS:
        obs += [Observation(b, GenerationKind.ROOT, 0, 1024, 1024, 400, 900)] * 5
        obs += [Observation(b, GenerationKind.PRIVATE, 0, 1024, 1024, 900, 800)] * 5
        obs += [Observation(b, GenerationKind.PEER, 3, 1024, 1024, 1900, 850)] * 5
    models = fit_models(obs)
    assert models[BENCHMARKS[0]].prompt_tokens(GenerationKind.PEER, 5) > models[BENCHMARKS[0]].prompt_tokens(GenerationKind.PEER, 3)
    assert models[BENCHMARKS[0]].packet_tokens == 500
    # unmeasured DEEP at 16,384 stays saturated (conservative)
    assert models[BENCHMARKS[0]].completion_tokens(GenerationKind.DEEP, 16384, 1024) == 16384 + 1024


def test_wave_admission_drops_high_tail_before_main_tail():
    costs = {b: {"main": Decimal("0.10"), "extension": Decimal("0.01"), "high": Decimal("0.20")} for b in BENCHMARKS}
    # per wave: main 1.60, extension 0.16, high 3.20
    everything = 25 * Decimal("1.60") + 12 * Decimal("0.16") + 6 * Decimal("3.20")
    assert wave_admission(costs, n_waves=25, extension_waves=12, high_waves=6, available_usd=everything) == \
        {"main_waves": 25, "high_waves": 6}
    assert wave_admission(costs, n_waves=25, extension_waves=12, high_waves=6,
                          available_usd=everything - Decimal("3.20")) == {"main_waves": 25, "high_waves": 5}
    assert wave_admission(costs, n_waves=25, extension_waves=12, high_waves=6, available_usd=Decimal("40")) == \
        {"main_waves": 23, "high_waves": 0}
