"""Configuration compilation (lead-owned). The pack's JSON configs are used verbatim; this module derives the
typed pieces every other module needs so no one re-reads raw config keys on their own."""
from __future__ import annotations

import json
import re
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.contracts import BENCHMARKS, STUDY_ID, GridSpec, canonical_hash

PRESET_CEILINGS_USD = {"balanced150": Decimal("135"), "lean100": Decimal("90"), "forced27b": Decimal("95"),
                       "passk_cont_v1": Decimal("150"), "dynorch_v1": Decimal("95")}
# Study id per preset (default: contracts.STUDY_ID). forced27b is the forced-budget rerun (openrouter_forced_budget_v2);
# passk_cont_v1 is its fast-path continuation (E0+E1+E2+E5, docs/passk_cont/00_contract.md).
PRESET_STUDY_IDS = {"forced27b": "openrouter_forced_budget_v2", "passk_cont_v1": "openrouter_passk_cont_v1",
                    "dynorch_v1": "openrouter_dynorch_v1"}
CODE_BENCHMARKS: frozenset[str] = frozenset()   # no code benchmark in the study; MBPP+ grading stays dormant
BUDGET_SEMANTICS = frozenset({"soft_requested_allowance", "forced_exact"})
FORCED_ENDPOINTS = frozenset({"completions"})
SHARD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
# Budget keys a shard receives share/sum(share) of (never max_unresolved_spend_usd: that is a per-dispatcher safety knob).
SHARD_SCALED_BUDGET_KEYS = ("cash_target_usd", "credit_ceiling_usd", "pilot_allowance_usd", "contingency_usd",
                            "soft_admission_ceiling_usd")
CENT = Decimal("0.01")


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def config_hash(cfg: Mapping[str, Any]) -> str:
    return canonical_hash(cfg)


def high_enabled(cfg: Mapping[str, Any]) -> bool:
    """A high-A partition exists only when ``reasoning.high_a`` is set and the high panel is non-empty."""
    return cfg.get("reasoning", {}).get("high_a") is not None and int(cfg.get("panels", {}).get("high_items", 0)) > 0


def replication_enabled(cfg: Mapping[str, Any]) -> bool:
    """A second-model replication partition exists only with a replication model, a non-empty replication panel and
    ``replication_grid.enabled`` not false (older presets carry no ``enabled`` key: enabled)."""
    return (cfg.get("replication_model") is not None
            and int(cfg.get("panels", {}).get("replication_items", 0)) > 0
            and cfg.get("replication_grid", {}).get("enabled", True) is not False)


def forced_mode(cfg: Mapping[str, Any]) -> bool:
    """Every generation's reasoning is forced to exactly its budget (forced_budget_v2)."""
    return (cfg.get("reasoning", {}).get("budget_semantics") == "forced_exact"
            or bool(cfg.get("forced", {}).get("enabled")))


def study_id_for(cfg: Mapping[str, Any]) -> str:
    return PRESET_STUDY_IDS.get(str(cfg.get("preset")), STUDY_ID)


def shard_fraction(share: Any, total: Any) -> Decimal:
    return Decimal(str(share)) / Decimal(str(total))


def scale_usd(value: Any, fraction: Decimal) -> Decimal:
    """A shard's slice of a budget figure, rounded down to the cent (slices never sum above the whole)."""
    return (Decimal(str(value)) * fraction).quantize(CENT, rounding=ROUND_DOWN)


def validate_config(cfg: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    def need(cond: bool, message: str) -> None:
        if not cond:
            problems.append(message)
    preset = cfg.get("preset")
    need(cfg.get("study_id") == study_id_for(cfg), "study_id mismatch")
    need(preset in PRESET_CEILINGS_USD, f"unknown preset {preset!r}")
    budget = cfg.get("budget", {})
    shard = cfg.get("shard")
    if preset in PRESET_CEILINGS_USD:
        expected_ceiling = PRESET_CEILINGS_USD[preset]
        if isinstance(shard, Mapping):
            try:
                expected_ceiling = scale_usd(expected_ceiling, shard_fraction(shard["share"], shard["total_share"]))
            except Exception:  # noqa: BLE001 -- reported below as a malformed shard block
                pass
            base = shard.get("base_config") or {}
            override = ((base.get("budget_by_shard") or {}).get(shard.get("id")) or {}).get("credit_ceiling_usd")
            if override is not None:                  # owner-set slice; the slices' sum is checked on the base
                need(Decimal(str(base.get("budget", {}).get("credit_ceiling_usd"))) == PRESET_CEILINGS_USD[preset],
                     "base credit ceiling differs from the preset")
                expected_ceiling = Decimal(str(override))
        need(Decimal(str(budget.get("credit_ceiling_usd"))) == expected_ceiling,
             "credit ceiling differs from the preset; a new approved configuration is required")
    need(Decimal(str(budget.get("soft_admission_ceiling_usd", 0))) <= Decimal(str(budget.get("credit_ceiling_usd", 0))),
         "soft admission ceiling exceeds the credit ceiling")
    need([b["id"] for b in cfg.get("benchmarks", [])] == list(BENCHMARKS), "benchmark ids/order differ from contracts")
    for name, count in cfg.get("panels", {}).items():
        need(count % 16 == 0, f"panel {name} is not a multiple of 16")
    panels = cfg.get("panels", {})
    need(int(panels.get("extension_items", 0)) <= int(panels.get("main_items", 0)),
         "extension panel is larger than the main panel (extension waves are a prefix of main)")
    provider = cfg.get("provider", {})
    need(provider.get("allow_fallbacks") is False, "fallbacks must be disabled")
    need(provider.get("require_parameters") is True, "require_parameters must be true")
    need(provider.get("auto_provider_switch") is False, "automatic provider switching must be off")
    sampling = cfg.get("sampling", {})
    need(sampling.get("response_cache") is False, "response cache must be disabled")
    need(sampling.get("api_seed_enabled") is False, "API seed must stay disabled")
    if "top_k" in sampling:
        top_k = sampling["top_k"]
        need(top_k is None or (isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 1),
             "sampling.top_k must be null or a positive integer")
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
    need(reasoning.get("budget_semantics") in BUDGET_SEMANTICS,
         f"reasoning.budget_semantics must be one of {sorted(BUDGET_SEMANTICS)}")
    for a_key, mult_key in (("main_a", "deep_main_multipliers"), ("high_a", "deep_high_multipliers")):
        if reasoning.get(a_key) is None:
            continue
        top = reasoning.get(a_key, 0) * max(protocols.get(mult_key, [1]) or [1])
        need(top <= ceiling_tokens, f"{a_key} x max({mult_key}) = {top} exceeds requested_reasoning_max {ceiling_tokens}")
    if reasoning.get("high_a") is None:
        need(int(panels.get("high_items", 0)) == 0, "reasoning.high_a is null but panels.high_items is not 0")
        need(not protocols.get("deep_high_multipliers"), "reasoning.high_a is null but deep_high_multipliers is set")
    rep_grid = cfg.get("replication_grid", {})
    if cfg.get("replication_model") is None or rep_grid.get("enabled", True) is False:
        need(int(panels.get("replication_items", 0)) == 0,
             "replication is disabled (replication_model null or replication_grid.enabled false) but "
             "panels.replication_items is not 0")
        need(cfg.get("planning_prices", {}).get("replication") is None or cfg.get("replication_model") is not None,
             "planning_prices.replication is set without a replication_model")
    else:
        top = reasoning.get("main_a", 0) * max(rep_grid.get("deep_multipliers", [1]) or [1])
        need(top <= ceiling_tokens, f"replication deep budget {top} exceeds requested_reasoning_max")
    need(Decimal(str(budget.get("reservation_multiplier", 0))) >= 1, "reservation_multiplier must be >= 1")
    need(Decimal(str(budget.get("max_unresolved_spend_usd", 99))) <= 2, "max_unresolved_spend_usd must be <= 2 (spec 7.1)")
    need(Path(str(cfg.get("endpoint_lock_file", ""))).name == "openrouter_endpoints.json",
         "endpoint_lock_file basename must be openrouter_endpoints.json (locks are run-scoped in RunPaths)")
    need(Path(str(cfg.get("dataset_lock_file", ""))).name == "datasets.json",
         "dataset_lock_file basename must be datasets.json (locks are run-scoped in RunPaths)")
    problems.extend(_forced_problems(cfg))
    problems.extend(_shard_problems(cfg))
    problems.extend(_calibration_problems(cfg))
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


def _forced_problems(cfg: Mapping[str, Any]) -> list[str]:
    """``forced.*``: the two-call forced-budget mechanism (THINK to exactly the budget, then ANSWER)."""
    problems: list[str] = []
    forced = cfg.get("forced")
    semantics = cfg.get("reasoning", {}).get("budget_semantics")
    if forced is None:
        if semantics == "forced_exact":
            problems.append("budget_semantics forced_exact requires a forced block")
        return problems
    if not isinstance(forced, Mapping):
        return ["forced must be an object"]
    enabled = forced.get("enabled")
    if not isinstance(enabled, bool):
        problems.append("forced.enabled must be a boolean")
    if enabled is True and semantics != "forced_exact":
        problems.append("forced.enabled requires reasoning.budget_semantics = forced_exact")
    if enabled is False and semantics == "forced_exact":
        problems.append("budget_semantics forced_exact requires forced.enabled = true")
    if enabled:
        if forced.get("endpoint") not in FORCED_ENDPOINTS:
            problems.append(f"forced.endpoint must be one of {sorted(FORCED_ENDPOINTS)}")
        for key in ("template", "template_file", "think_close_text"):
            if not isinstance(forced.get(key), str) or not forced.get(key):
                problems.append(f"forced.{key} must be a non-empty string")
        if "</think>" not in str(forced.get("think_close_text", "")):
            problems.append("forced.think_close_text must close the reasoning block (</think>)")
        ids = forced.get("ban_token_ids")
        if not (isinstance(ids, list) and ids and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0
                                                      for i in ids) and len(set(ids)) == len(ids)):
            problems.append("forced.ban_token_ids must be a non-empty list of distinct token ids")
        bias = forced.get("ban_bias")
        if not (isinstance(bias, (int, float)) and not isinstance(bias, bool) and bias <= -100):
            problems.append("forced.ban_bias must be <= -100 (a hard ban)")
        probe = forced.get("bias_probe_token_id")
        if not (isinstance(probe, int) and not isinstance(probe, bool) and probe >= 0):
            problems.append("forced.bias_probe_token_id must be a token id")
        attempts = forced.get("answer_attempts")
        if not (isinstance(attempts, int) and not isinstance(attempts, bool) and 1 <= attempts <= 10):
            problems.append("forced.answer_attempts must be an integer in [1, 10]")
    return problems


def _shard_problems(cfg: Mapping[str, Any]) -> list[str]:
    """``shards[]`` (one dispatcher per provider endpoint on disjoint items) and a derived config's ``shard`` block."""
    problems: list[str] = []
    shards = cfg.get("shards")
    provider_tag = cfg.get("provider", {}).get("endpoint_tag")
    if "endpoint_tag" in cfg.get("provider", {}) and provider_tag is not None and not isinstance(provider_tag, str):
        problems.append("provider.endpoint_tag must be null or a string")
    if shards is not None:
        if not isinstance(shards, list) or not shards:
            return problems + ["shards must be a non-empty list"]
        ids, tags = [], []
        for index, entry in enumerate(shards):
            if not isinstance(entry, Mapping):
                problems.append(f"shards[{index}] must be an object")
                continue
            sid, tag, share = entry.get("id"), entry.get("endpoint_tag"), entry.get("share")
            if not (isinstance(sid, str) and SHARD_ID_RE.match(sid)):
                problems.append(f"shards[{index}].id must match {SHARD_ID_RE.pattern}")
            if not (isinstance(tag, str) and tag.strip()):
                problems.append(f"shards[{index}].endpoint_tag must be a non-empty string")
            try:
                ok = not isinstance(share, bool) and Decimal(str(share)) > 0
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                problems.append(f"shards[{index}].share must be a number > 0")
            ids.append(sid)
            tags.append(tag)
        if len(set(ids)) != len(ids):
            problems.append("shard ids must be unique")
        if len(set(tags)) != len(tags):
            problems.append("shard endpoint tags must be unique (one dispatcher per endpoint)")
        if cfg.get("shard") is None and provider_tag is not None:
            problems.append("a sharded base config must leave provider.endpoint_tag null (each shard sets its own)")
    if "shards_note" in cfg and not isinstance(cfg["shards_note"], str):
        problems.append("shards_note must be a string")
    shard = cfg.get("shard")
    if shard is not None:
        if not isinstance(shard, Mapping):
            return problems + ["shard must be an object"]
        entry = next((s for s in shards or [] if isinstance(s, Mapping) and s.get("id") == shard.get("id")), None)
        if entry is None:
            problems.append(f"shard {shard.get('id')!r} is not listed in shards")
        else:
            if provider_tag != entry.get("endpoint_tag") or shard.get("endpoint_tag") != entry.get("endpoint_tag"):
                problems.append("a shard config's provider.endpoint_tag must equal its shard's endpoint_tag")
            total = sum(Decimal(str(s["share"])) for s in shards)
            if Decimal(str(shard.get("share", 0))) != Decimal(str(entry["share"])) \
                    or Decimal(str(shard.get("total_share", 0))) != total:
                problems.append("shard share/total_share differ from the shards list")
    return problems


def _calibration_problems(cfg: Mapping[str, Any]) -> list[str]:
    calibration = cfg.get("calibration")
    if calibration is None:
        return []
    if not isinstance(calibration, Mapping):
        return ["calibration must be an object"]
    problems: list[str] = []
    per = calibration.get("dev_items_per_benchmark")
    if not (isinstance(per, int) and not isinstance(per, bool) and per >= 1):
        problems.append("calibration.dev_items_per_benchmark must be a positive integer")
    lo, hi = calibration.get("forced_pass_at_1_min"), calibration.get("forced_pass_at_1_max")
    try:
        if not (0 <= float(lo) < float(hi) <= 1):
            problems.append("calibration gate must satisfy 0 <= forced_pass_at_1_min < forced_pass_at_1_max <= 1")
    except (TypeError, ValueError):
        problems.append("calibration.forced_pass_at_1_min/max must be numbers")
    for key in ("fallback_easier", "fallback_harder"):
        value = calibration.get(key)
        if value is not None and not (isinstance(value, Mapping)
                                      and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())):
            problems.append(f"calibration.{key} must map benchmark ids to benchmark ids")
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
    forced = cfg.get("forced")
    if isinstance(forced, Mapping) and forced.get("enabled") is True:
        # Forced mode (two raw /completions calls, budget enforced by logit_bias + max_tokens): a different decoding
        # contract, so every node id differs from its chat-mode twin. Chat mode's hash below is unchanged.
        from agent_allocation.api.forced import template_sha256
        return canonical_hash({
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "top_k": sampling.get("top_k"),
            "n": sampling["n_per_request"],
            "response_cache": sampling["response_cache"],
            "api_seed_enabled": sampling["api_seed_enabled"],
            "allow_fallbacks": provider["allow_fallbacks"],
            "require_parameters": provider["require_parameters"],
            "service_tier": provider["service_tier"],
            "stream": False,
            "endpoint_kind": forced.get("endpoint"),
            "reasoning_field": "forced_logit_bias",
            "budget_semantics": "forced_exact",
            "ban_token_ids": list(forced["ban_token_ids"]),
            "ban_bias": forced["ban_bias"],
            "think_close_text": forced["think_close_text"],
            "template": forced["template"],
            "template_sha256": template_sha256(forced["template"]),
        })
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
        # Pilot grids (lead-defined, plan section D): n=3 PEER, two private-team rounds, no deep calls here;
        # the pilot's long and high-A calls are added per item by the lead's pilot plan.
        "pilot_r1": GridSpec(name="pilot_r1", bank_size=3, private_single_cycles=3, private_team_rounds=2,
                             team_sizes=(3,), peer_rounds=1, deep_multipliers=()),
        "pilot_r2": GridSpec(name="pilot_r2", bank_size=3, private_single_cycles=3, private_team_rounds=2,
                             team_sizes=(3,), peer_rounds=2, deep_multipliers=()),
    }
    # Disabled partitions (forced27b: high_a null, replication_model null) have no grid, so nothing can build them.
    if cfg["reasoning"].get("high_a") is not None:
        grids["high"] = GridSpec(**{**main.__dict__, "name": "high",
                                    "deep_multipliers": tuple(p.get("deep_high_multipliers") or ())})
    if cfg.get("replication_model") is not None and r.get("enabled", True) is not False:
        grids["replication"] = GridSpec(
            name="replication",
            bank_size=r["root_bank_size"],
            private_single_cycles=r["private_total_cycles"],
            private_team_rounds=max(r["private_team_update_rounds"]),
            team_sizes=tuple(r["team_sizes"]),
            peer_rounds=max(r["peer_update_rounds"]),
            deep_multipliers=tuple(r["deep_multipliers"]),
        )
    return grids
