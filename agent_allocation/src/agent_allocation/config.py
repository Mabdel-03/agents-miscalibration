"""Configuration compilation (lead-owned). The pack's JSON configs are used verbatim; this module derives the
typed pieces every other module needs so no one re-reads raw config keys on their own."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.contracts import BENCHMARKS, STUDY_ID, GridSpec, canonical_hash

PRESET_CEILINGS_USD = {"balanced150": Decimal("135"), "lean100": Decimal("90")}
CODE_BENCHMARKS: frozenset[str] = frozenset()   # no code benchmark in the study; MBPP+ grading stays dormant


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def config_hash(cfg: Mapping[str, Any]) -> str:
    return canonical_hash(cfg)


def validate_config(cfg: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    def need(cond: bool, message: str) -> None:
        if not cond:
            problems.append(message)
    need(cfg.get("study_id") == STUDY_ID, "study_id mismatch")
    preset = cfg.get("preset")
    need(preset in PRESET_CEILINGS_USD, f"unknown preset {preset!r}")
    budget = cfg.get("budget", {})
    if preset in PRESET_CEILINGS_USD:
        need(Decimal(str(budget.get("credit_ceiling_usd"))) == PRESET_CEILINGS_USD[preset],
             "credit ceiling differs from the preset; a new approved configuration is required")
    need(Decimal(str(budget.get("soft_admission_ceiling_usd", 0))) <= Decimal(str(budget.get("credit_ceiling_usd", 0))),
         "soft admission ceiling exceeds the credit ceiling")
    need([b["id"] for b in cfg.get("benchmarks", [])] == list(BENCHMARKS), "benchmark ids/order differ from contracts")
    for name, count in cfg.get("panels", {}).items():
        need(count % 16 == 0, f"panel {name} is not a multiple of 16")
    provider = cfg.get("provider", {})
    need(provider.get("allow_fallbacks") is False, "fallbacks must be disabled")
    need(provider.get("require_parameters") is True, "require_parameters must be true")
    need(provider.get("auto_provider_switch") is False, "automatic provider switching must be off")
    sampling = cfg.get("sampling", {})
    need(sampling.get("response_cache") is False, "response cache must be disabled")
    need(sampling.get("api_seed_enabled") is False, "API seed must stay disabled")
    execution = cfg.get("execution", {})
    need(execution.get("default_mode") == "dry_run", "default mode must be dry_run")
    need(execution.get("require_allow_paid_flag") is True, "--allow-paid must be required")
    scheduler = cfg.get("scheduler", {})
    need(scheduler.get("single_paid_dispatcher") is True, "single paid dispatcher required")
    need(scheduler.get("api_sdk_automatic_retries") is False, "SDK automatic retries must be off")
    # v2 (D7b, D7c, D6): budgets are validated, never clipped; safety knobs cannot be loosened; lock names match RunPaths
    reasoning = cfg.get("reasoning", {})
    protocols = cfg.get("protocols", {})
    ceiling_tokens = reasoning.get("requested_reasoning_max", 0)
    for a_key, mult_key in (("main_a", "deep_main_multipliers"), ("high_a", "deep_high_multipliers")):
        top = reasoning.get(a_key, 0) * max(protocols.get(mult_key, [1]) or [1])
        need(top <= ceiling_tokens, f"{a_key} x max({mult_key}) = {top} exceeds requested_reasoning_max {ceiling_tokens}")
    rep_grid = cfg.get("replication_grid", {})
    top = reasoning.get("main_a", 0) * max(rep_grid.get("deep_multipliers", [1]) or [1])
    need(top <= ceiling_tokens, f"replication deep budget {top} exceeds requested_reasoning_max")
    need(Decimal(str(budget.get("reservation_multiplier", 0))) >= 1, "reservation_multiplier must be >= 1")
    need(Decimal(str(budget.get("max_unresolved_spend_usd", 99))) <= 2, "max_unresolved_spend_usd must be <= 2 (spec 7.1)")
    need(Path(str(cfg.get("endpoint_lock_file", ""))).name == "openrouter_endpoints.json",
         "endpoint_lock_file basename must be openrouter_endpoints.json (locks are run-scoped in RunPaths)")
    need(Path(str(cfg.get("dataset_lock_file", ""))).name == "datasets.json",
         "dataset_lock_file basename must be datasets.json (locks are run-scoped in RunPaths)")
    # v2 (D7a, R03-3): CP/POOLED knobs must equal the builder's constants
    try:
        from agent_allocation.protocols.grid import check_config_constants
    except ImportError:
        check_config_constants = None
    if check_config_constants is not None:
        try:
            check_config_constants(cfg)
        except Exception as exc:  # noqa: BLE001 -- report every constant mismatch as a validation problem
            problems.append(f"protocol constants: {exc}")
    return problems


def credit_ceiling(cfg: Mapping[str, Any]) -> Decimal:
    return Decimal(str(cfg["budget"]["credit_ceiling_usd"]))


def headroom_for(cfg: Mapping[str, Any], benchmark: str, amendments: Mapping[str, int] | None = None) -> int:
    """Completion headroom for a benchmark family, after any recorded pre-authorized headroom amendment."""
    if amendments and benchmark in amendments:
        return int(amendments[benchmark])
    key = "code_completion_headroom" if benchmark in CODE_BENCHMARKS else "text_completion_headroom"
    return int(cfg["reasoning"][key])


def decoding_config_hash(cfg: Mapping[str, Any]) -> str:
    """Everything about a request's decoding except per-node budget/headroom (those are node fields)."""
    sampling = cfg["sampling"]
    provider = cfg["provider"]
    return canonical_hash({
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "n": sampling["n_per_request"],
        "response_cache": sampling["response_cache"],
        "api_seed_enabled": sampling["api_seed_enabled"],
        "allow_fallbacks": provider["allow_fallbacks"],
        "require_parameters": provider["require_parameters"],
        "service_tier": provider["service_tier"],
        "stream": False,
        "reasoning_exclude": False,
        "reasoning_field": "reasoning.max_tokens",
        "budget_semantics": cfg["reasoning"]["budget_semantics"],
    })


def grids_from_config(cfg: Mapping[str, Any]) -> dict[str, GridSpec]:
    p = cfg["protocols"]
    r = cfg["replication_grid"]
    main = GridSpec(
        name="main",
        bank_size=p["root_bank_size"],
        private_single_cycles=p["private_total_cycles"],
        private_team_rounds=max(p["private_team_update_rounds"]),
        team_sizes=tuple(p["team_sizes"]),
        peer_rounds=max(p["peer_update_rounds"]),
        deep_multipliers=tuple(p["deep_main_multipliers"]),
    )
    grids = {
        "main": main,
        "main_extension": GridSpec(**{**main.__dict__, "name": "main_extension", "extension": True}),
        "high": GridSpec(**{**main.__dict__, "name": "high", "deep_multipliers": tuple(p["deep_high_multipliers"])}),
        "replication": GridSpec(
            name="replication",
            bank_size=r["root_bank_size"],
            private_single_cycles=r["private_total_cycles"],
            private_team_rounds=max(r["private_team_update_rounds"]),
            team_sizes=tuple(r["team_sizes"]),
            peer_rounds=max(r["peer_update_rounds"]),
            deep_multipliers=tuple(r["deep_multipliers"]),
        ),
        # Pilot grids (lead-defined, plan section D): n=3 PEER, two private-team rounds, no deep calls here;
        # the pilot's long and high-A calls are added per item by the lead's pilot plan.
        "pilot_r1": GridSpec(name="pilot_r1", bank_size=3, private_single_cycles=3, private_team_rounds=2,
                             team_sizes=(3,), peer_rounds=1, deep_multipliers=()),
        "pilot_r2": GridSpec(name="pilot_r2", bank_size=3, private_single_cycles=3, private_team_rounds=2,
                             team_sizes=(3,), peer_rounds=2, deep_multipliers=()),
    }
    return grids
