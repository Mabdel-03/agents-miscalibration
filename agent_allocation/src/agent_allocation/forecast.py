"""Measured cost forecast (lead-owned): pilot usage -> per-item cost per partition -> admission plan under a ceiling.

Model, per benchmark (conservative wherever a quantity was not measured):
* prompt tokens: ROOT/DEEP = measured root prompt; PRIVATE/CP = measured private-update prompt; PEER n = private
  prompt + (n - 1) x per-packet tokens, where per-packet tokens come from the measured n=3 PEER prompts; POOLED = root
  prompt + 4 packets. Latest-visible-solution packets do not grow with rounds.
* completion tokens: measured means for (single-call, update) x requested budget. An unmeasured budget uses the
  saturated cap A + headroom, except that a larger measured budget's mean may lower it (a smaller budget cannot
  use more tokens than the larger one's cap).
* a safety multiplier (default 1.15) on every forecast figure.

Forced mode (THINK + ANSWER per node; detected from the results' ``usage.raw_usage = {"think", "answer"}`` or passed
explicitly): an observation's prompt is the THINK prompt P (not the summed THINK + ANSWER input), and a call costs
``input * (2P + budget) + output * max(budget, completion)`` -- the ANSWER re-reads the prompt and the reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from statistics import mean
from typing import Iterable, Mapping, Sequence

from agent_allocation.contracts import BENCHMARKS, EndpointLock, GenerationKind, GridSpec, TransportStatus

SAFETY_MULTIPLIER = Decimal("1.15")
SINGLE_KINDS = {GenerationKind.ROOT, GenerationKind.DEEP}


@dataclass(frozen=True)
class Observation:
    benchmark: str
    kind: GenerationKind
    team_size: int
    reasoning_budget: int
    headroom: int
    prompt_tokens: int
    completion_tokens: int
    forced: bool = False                   # prompt_tokens is the THINK prompt of a two-call forced node


@dataclass
class BenchmarkModel:
    root_prompt: float
    private_prompt: float
    packet_tokens: float
    completion: dict[tuple[str, int], float] = field(default_factory=dict)   # ("single"|"update", budget) -> mean
    forced: bool = False

    def prompt_tokens(self, kind: GenerationKind, team_size: int) -> float:
        if kind in SINGLE_KINDS:
            return self.root_prompt
        if kind in (GenerationKind.PRIVATE, GenerationKind.CP):
            return self.private_prompt
        if kind == GenerationKind.PEER:
            return self.private_prompt + max(team_size - 1, 0) * self.packet_tokens
        return self.root_prompt + 4 * self.packet_tokens            # POOLED

    def completion_tokens(self, kind: GenerationKind, budget: int, headroom: int) -> float:
        cls = "single" if kind in SINGLE_KINDS else "update"
        cap = budget + headroom
        measured = self.completion.get((cls, budget))
        if measured is not None:
            return min(measured, cap)
        other = self.completion.get(("single" if cls == "update" else "update", budget))
        if other is not None:
            return min(cap, other)
        larger = [v for (c, b), v in self.completion.items() if c == cls and b > budget]
        return min(cap, max(larger)) if larger else float(cap)


def fit_models(observations: Iterable[Observation]) -> dict[str, BenchmarkModel]:
    by_bench: dict[str, list[Observation]] = {}
    for obs in observations:
        by_bench.setdefault(obs.benchmark, []).append(obs)
    models = {}
    for benchmark, rows in by_bench.items():
        roots = [r.prompt_tokens for r in rows if r.kind in SINGLE_KINDS]
        privates = [r.prompt_tokens for r in rows if r.kind == GenerationKind.PRIVATE]
        peers = [(r.prompt_tokens, r.team_size) for r in rows if r.kind == GenerationKind.PEER and r.team_size > 1]
        root_prompt = mean(roots) if roots else 2500.0
        private_prompt = mean(privates) if privates else root_prompt * 2
        if peers:
            per_packet = [(p - private_prompt) / (n - 1) for p, n in peers]
            packet = max(mean(per_packet), 0.0)
        else:
            packet = private_prompt - root_prompt
        completion: dict[tuple[str, int], list[int]] = {}
        for r in rows:
            cls = "single" if r.kind in SINGLE_KINDS else "update"
            completion.setdefault((cls, r.reasoning_budget), []).append(r.completion_tokens)
        models[benchmark] = BenchmarkModel(root_prompt, private_prompt, packet,
                                           {k: float(mean(v)) for k, v in completion.items()},
                                           forced=any(r.forced for r in rows))
    return models


def call_mix(grid: GridSpec, base_budget: int) -> list[tuple[GenerationKind, int, int]]:
    """(kind, team_size, reasoning_budget) for every generation node of one item under ``grid``."""
    mix: list[tuple[GenerationKind, int, int]] = []
    mix += [(GenerationKind.ROOT, 0, base_budget)] * grid.bank_size
    mix += [(GenerationKind.PRIVATE, 0, base_budget)] * (grid.private_single_cycles - 1)
    mix += [(GenerationKind.PRIVATE, 0, base_budget)] * ((max(grid.team_sizes) - 1) * grid.private_team_rounds
                                                         if grid.team_sizes else 0)
    for n in grid.team_sizes:
        mix += [(GenerationKind.PEER, n, base_budget)] * (n * grid.peer_rounds)
    mix += [(GenerationKind.DEEP, 0, base_budget * m) for m in grid.deep_multipliers]
    if grid.extension:
        mix += [(GenerationKind.CP, 4, base_budget)] * 12 + [(GenerationKind.POOLED, 4, base_budget)]
    return mix


def item_cost(model: BenchmarkModel, grid: GridSpec, base_budget: int, headroom: int, lock: EndpointLock, *,
              forced: bool | None = None) -> Decimal:
    forced = model.forced if forced is None else forced
    total = Decimal("0")
    price_in, price_out = Decimal(lock.input_price_per_token), Decimal(lock.output_price_per_token)
    for kind, n, budget in call_mix(grid, base_budget):
        prompt = Decimal(str(model.prompt_tokens(kind, n)))
        completion = Decimal(str(model.completion_tokens(kind, budget, headroom)))
        if forced:
            total += (2 * prompt + budget) * price_in + max(Decimal(budget), completion) * price_out
        else:
            total += prompt * price_in + completion * price_out
    return total


def per_item_costs(models: Mapping[str, BenchmarkModel], grids: Mapping[str, GridSpec], *, lock: EndpointLock,
                   headroom_by_benchmark: Mapping[str, int], main_a: int, high_a: int | None,
                   forced: bool | None = None) -> dict[str, dict[str, Decimal]]:
    """Per-item forecast per partition. With no high partition (``high_a`` None or no ``high`` grid) its cost is 0."""
    out: dict[str, dict[str, Decimal]] = {}
    for benchmark in BENCHMARKS:
        model = models[benchmark]
        h = headroom_by_benchmark[benchmark]
        main = item_cost(model, grids["main"], main_a, h, lock, forced=forced)
        extension = item_cost(model, grids["main_extension"], main_a, h, lock, forced=forced) - main
        high = (item_cost(model, grids["high"], high_a, h, lock, forced=forced) * SAFETY_MULTIPLIER
                if high_a is not None and "high" in grids else Decimal("0"))
        out[benchmark] = {"main": main * SAFETY_MULTIPLIER, "extension": extension * SAFETY_MULTIPLIER,
                          "high": high}
    return out


def core_forecast(costs: Mapping[str, Mapping[str, Decimal]], panels_per_benchmark: Mapping[str, int]) -> Decimal:
    total = Decimal("0")
    for benchmark in BENCHMARKS:
        total += costs[benchmark]["main"] * panels_per_benchmark["main_items"]
        total += costs[benchmark]["extension"] * panels_per_benchmark["extension_items"]
        total += costs[benchmark]["high"] * panels_per_benchmark.get("high_items", 0)
    return total.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def wave_admission(costs: Mapping[str, Mapping[str, Decimal]], *, n_waves: int, extension_waves: int, high_waves: int,
                   available_usd: Decimal) -> dict[str, int]:
    """Largest (main waves, high waves) that fit, honoring the pre-declared cut order: high-A tail waves are dropped
    before main tail waves. Extension CP/POOLED stay attached to their main waves. Four items per benchmark per wave."""
    per_wave_main = sum(costs[b]["main"] for b in BENCHMARKS) * 4
    per_wave_ext = sum(costs[b]["extension"] for b in BENCHMARKS) * 4
    per_wave_high = sum(costs[b]["high"] for b in BENCHMARKS) * 4

    def cost(main_waves: int, high: int) -> Decimal:
        return per_wave_main * main_waves + per_wave_ext * min(main_waves, extension_waves) + per_wave_high * high

    for high in range(high_waves, -1, -1):
        if cost(n_waves, high) <= available_usd:
            return {"main_waves": n_waves, "high_waves": high}
    for main_waves in range(n_waves, -1, -1):
        if cost(main_waves, 0) <= available_usd:
            return {"main_waves": main_waves, "high_waves": 0}
    return {"main_waves": 0, "high_waves": 0}


def observations_from_rows(rows: Sequence) -> list[Observation]:
    out = []
    for r in rows:
        usage = r.result.usage
        if r.result.transport_status != TransportStatus.OK or usage is None:
            continue
        if usage.prompt_tokens is None or usage.completion_tokens is None:
            continue
        raw = usage.raw_usage if isinstance(usage.raw_usage, Mapping) else {}
        forced = "think" in raw and "answer" in raw
        prompt = usage.prompt_tokens
        if forced:
            think_prompt = (raw.get("think") or {}).get("prompt_tokens")
            prompt = think_prompt if isinstance(think_prompt, int) and not isinstance(think_prompt, bool) \
                else max(0, (usage.prompt_tokens - r.spec.reasoning_budget) // 2)
        out.append(Observation(r.spec.benchmark, r.spec.kind, r.spec.team_size, r.spec.reasoning_budget,
                               r.spec.completion_headroom, prompt, usage.completion_tokens, forced=forced))
    return out
