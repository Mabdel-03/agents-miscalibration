"""WP-A, contract §5 / plan "Frozen parameter table": ``DynorchSettings.from_config(cfg)`` validates the ``dynorch``
and ``budget`` blocks of ``configs/dynorch_v1.json`` (plus the ``sampling``/``forced`` keys the proxy reuses) and
exposes them as one frozen dataclass. Anything the proxy relies on semantically is pinned to the frozen value
(``finish_reasons_returned`` exactly ``{stop, tool_calls}``, one root-only grace, per-run serialisation, harness
``max_tokens`` ignored); every other key is type- and range-checked. Unknown shapes raise ``ConfigError`` --
there is no fallback default for a frozen parameter.

``PROXY_VERSION`` is stamped on every ``RunSpec``/``RunRecord``; bump it for any change that affects scoring or the
cap (plan §9 Gate 3: such a change forces re-execution of affected cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

PROXY_VERSION = "dynorch-proxy-1.0.0"

FROZEN_RUNGS: tuple[int, ...] = (1, 2, 4, 8, 16, 25)
ROOT_CEILING_RULES: Mapping[str, str] = {"max(A, C/4)": "max_a_c_over_4"}


class ConfigError(ValueError):
    """The config is outside the frozen parameter table (raised before anything runs)."""


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def _int(block: Mapping[str, Any], key: str, *, low: int = 0, where: str) -> int:
    _need(key in block, f"{where}.{key} is required")
    value = block[key]
    _need(isinstance(value, int) and not isinstance(value, bool) and value >= low,
          f"{where}.{key} must be an integer >= {low}, got {value!r}")
    return value


def _bool(block: Mapping[str, Any], key: str, *, where: str) -> bool:
    _need(key in block, f"{where}.{key} is required")
    value = block[key]
    _need(isinstance(value, bool), f"{where}.{key} must be a boolean, got {value!r}")
    return value


def _str(block: Mapping[str, Any], key: str, *, where: str) -> str:
    _need(key in block, f"{where}.{key} is required")
    value = block[key]
    _need(isinstance(value, str) and value.strip() != "", f"{where}.{key} must be a non-empty string")
    return value


def _decimal(block: Mapping[str, Any], key: str, *, where: str, low: str = "0") -> Decimal:
    _need(key in block, f"{where}.{key} is required")
    value = block[key]
    _need(not isinstance(value, bool) and isinstance(value, (int, float, str)), f"{where}.{key} must be a number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"{where}.{key} is not a decimal: {value!r}") from exc
    _need(amount.is_finite() and amount >= Decimal(low), f"{where}.{key} must be >= {low}, got {value!r}")
    return amount


def _str_tuple(block: Mapping[str, Any], key: str, *, where: str) -> tuple[str, ...]:
    _need(key in block, f"{where}.{key} is required")
    value = block[key]
    _need(isinstance(value, list) and all(isinstance(v, str) for v in value),
          f"{where}.{key} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    harness: str
    spawn_tools: tuple[str, ...]
    root_call_ceiling_rule: str | None       # None (pure dynamic) or "max(A, C/4)" (E0 capped variant)
    spawning_disabled: bool = False
    fallback_for: str | None = None

    def root_call_ceiling(self, *, cap_tokens: int, a_tokens: int) -> int | None:
        """The per-call root ceiling for one run (contract §1 ``grant``): ``None`` in the core arms."""
        if self.root_call_ceiling_rule is None:
            return None
        if ROOT_CEILING_RULES.get(self.root_call_ceiling_rule) == "max_a_c_over_4":
            return max(a_tokens, cap_tokens // 4)
        raise ConfigError(f"unknown root_call_ceiling rule {self.root_call_ceiling_rule!r}")


@dataclass(frozen=True)
class DynorchSettings:
    study_id: str
    primary_model: str
    base_run_root: str
    template: str
    think_close_text: str
    sampling: Mapping[str, Any]
    arms: Mapping[str, ArmSpec]
    core_arms: tuple[str, ...]
    rungs: tuple[int, ...]
    a_tokens: int
    repeats: int
    cell_rung_order: tuple[int, ...]
    grace_tokens: int
    grace_root_only: bool
    max_grace_events_per_run: int
    terminal_threshold_tokens: int
    per_run_serialization: bool
    harness_max_tokens_ignored: bool
    finish_reasons_returned: tuple[str, ...]
    reasoning_effort: str
    subagent_depth: int
    tool_surface: str
    max_calls_per_run: int
    max_prompt_tokens_per_run: int
    wall_timeout_base_seconds: int
    wall_timeout_per_rung_seconds: int
    max_reruns: int
    proxy_host: str
    proxy_port: int
    keepalive_seconds: int
    path_prefix: str
    runner_concurrency: int
    sigterm_after_exhaustion_seconds: int
    sigkill_after_seconds: int
    provider_in_flight: Mapping[str, int]
    provider_rps: Mapping[str, int]
    scratch_root: str
    prompt_files: Mapping[str, str]
    budget_sentence: str
    terminal_sentence: str
    # budget block
    credit_ceiling_usd: Decimal
    soft_admission_ceiling_usd: Decimal
    pilot_allowance_usd: Decimal
    canary_allowance_usd: Decimal
    reservation_multiplier: Decimal
    max_unresolved_spend_usd: Decimal
    account_auto_reload: bool
    forecast_contingency_share: Decimal
    proxy_version: str = PROXY_VERSION

    def wall_timeout_seconds(self, rung: int) -> int:
        """Frozen table: ``600 + 120·k`` seconds."""
        return self.wall_timeout_base_seconds + self.wall_timeout_per_rung_seconds * int(rung)

    def cap_tokens(self, rung: int) -> int:
        _need(rung in self.rungs, f"rung {rung} is not in the frozen rungs {self.rungs}")
        return self.a_tokens * int(rung)

    def budget_sentence_for(self, cap_tokens: int) -> str:
        """B8: the one sentence that tells the harness its budget."""
        return self.budget_sentence.replace("{C}", str(int(cap_tokens)))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DynorchSettings":
        _need(isinstance(cfg, Mapping), "config must be a mapping")
        d = cfg.get("dynorch")
        _need(isinstance(d, Mapping), "config has no 'dynorch' block")
        b = cfg.get("budget")
        _need(isinstance(b, Mapping), "config has no 'budget' block")
        forced = cfg.get("forced")
        _need(isinstance(forced, Mapping) and forced.get("enabled") is True, "forced.enabled must be true")
        sampling = cfg.get("sampling")
        _need(isinstance(sampling, Mapping), "config has no 'sampling' block")
        for key in ("temperature", "top_p", "top_k"):
            _need(key in sampling, f"sampling.{key} is required (frozen table: T 1.0 / top_p 0.95 / top_k 20)")

        arms: dict[str, ArmSpec] = {}
        for block_name in ("arms", "extension_arms"):
            block = d.get(block_name)
            _need(isinstance(block, Mapping) and block, f"dynorch.{block_name} must be a non-empty mapping")
            for name, arm in block.items():
                _need(isinstance(arm, Mapping), f"dynorch.{block_name}.{name} must be a mapping")
                where = f"dynorch.{block_name}.{name}"
                rule = arm.get("root_call_ceiling")
                _need(rule is None or rule in ROOT_CEILING_RULES,
                      f"{where}.root_call_ceiling must be null or one of {sorted(ROOT_CEILING_RULES)}")
                disabled = arm.get("spawning_disabled", False)
                _need(isinstance(disabled, bool), f"{where}.spawning_disabled must be a boolean")
                fallback = arm.get("fallback_for")
                _need(fallback is None or isinstance(fallback, str), f"{where}.fallback_for must be a string")
                spawn = _str_tuple(arm, "spawn_tools", where=where)
                _need(not (disabled and spawn), f"{where}: spawning_disabled arms must list no spawn tools")
                _need(name not in arms, f"arm {name!r} defined twice")
                arms[name] = ArmSpec(name=name, harness=_str(arm, "harness", where=where), spawn_tools=spawn,
                                     root_call_ceiling_rule=rule, spawning_disabled=disabled, fallback_for=fallback)
        core_arms = _str_tuple(d, "core_arms", where="dynorch")
        _need(all(a in d["arms"] for a in core_arms), "dynorch.core_arms must all be defined in dynorch.arms")
        _need(all(arms[a].root_call_ceiling_rule is None for a in core_arms),
              "frozen table: no per-call ceiling in the core arms")

        rungs_raw = d.get("rungs")
        _need(isinstance(rungs_raw, list) and all(isinstance(r, int) and not isinstance(r, bool) and r > 0
                                                   for r in rungs_raw), "dynorch.rungs must be positive integers")
        rungs = tuple(rungs_raw)
        _need(rungs == FROZEN_RUNGS, f"frozen table: rungs must be {list(FROZEN_RUNGS)}, got {list(rungs)}")
        order = d.get("cell_rung_order")
        _need(isinstance(order, list) and sorted(order) == sorted(rungs),
              "dynorch.cell_rung_order must be a permutation of dynorch.rungs")

        finish = _str_tuple(d, "finish_reasons_returned", where="dynorch")
        _need(set(finish) == {"stop", "tool_calls"} and len(finish) == 2,
              "frozen table: finish_reasons_returned must be exactly [stop, tool_calls]")
        grace = _int(d, "grace_tokens", low=1, where="dynorch")
        _need(_bool(d, "grace_root_only", where="dynorch") is True, "frozen table: grace is root-only")
        _need(_int(d, "max_grace_events_per_run", low=1, where="dynorch") == 1,
              "frozen table: at most one grace event per run")
        _need(_bool(d, "per_run_serialization", where="dynorch") is True,
              "frozen table: one upstream call in flight per run")
        _need(_bool(d, "harness_max_tokens_ignored", where="dynorch") is True,
              "frozen table: the harness max_tokens is recorded and ignored")
        effort = _str(d, "reasoning_effort", where="dynorch")
        _need(effort in ("xhigh", "medium", "low"), "dynorch.reasoning_effort must be xhigh|medium|low")
        threshold = _int(d, "terminal_threshold_tokens", low=1, where="dynorch")
        _need(threshold < grace, "terminal_threshold_tokens must be below grace_tokens")

        wall = d.get("wall_timeout_seconds")
        _need(isinstance(wall, Mapping), "dynorch.wall_timeout_seconds must be {base, per_rung}")
        rerun = d.get("rerun_rule")
        _need(isinstance(rerun, Mapping), "dynorch.rerun_rule must be a mapping")
        proxy = d.get("proxy")
        _need(isinstance(proxy, Mapping), "dynorch.proxy must be a mapping")
        runner = d.get("runner")
        _need(isinstance(runner, Mapping), "dynorch.runner must be a mapping")
        for key in ("provider_in_flight", "provider_rps"):
            block = d.get(key)
            _need(isinstance(block, Mapping) and block and all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in block.values()),
                f"dynorch.{key} must map shard ids to positive integers")
        _need(set(d["provider_in_flight"]) == set(d["provider_rps"]),
              "dynorch.provider_in_flight and provider_rps must name the same shards")
        prompt_files = d.get("prompt_files")
        _need(isinstance(prompt_files, Mapping) and {"system", "subagent", "h0_system", "user"} <= set(prompt_files)
              and all(isinstance(v, str) for v in prompt_files.values()),
              "dynorch.prompt_files must map system/subagent/h0_system/user to paths")
        sentence = _str(d, "budget_sentence", where="dynorch")
        _need("{C}" in sentence, "dynorch.budget_sentence must contain the {C} placeholder")
        close = forced.get("think_close_text")
        _need(isinstance(close, str) and "</think>" in close, "forced.think_close_text must contain </think>")
        template = forced.get("template")
        _need(isinstance(template, str) and template, "forced.template is required")
        multiplier = _decimal(b, "reservation_multiplier", where="budget", low="1")
        ceiling = _decimal(b, "credit_ceiling_usd", where="budget")
        soft = _decimal(b, "soft_admission_ceiling_usd", where="budget")
        _need(soft <= ceiling, "budget.soft_admission_ceiling_usd must not exceed credit_ceiling_usd")
        contingency = _decimal(b, "forecast_contingency_share", where="budget")
        _need(contingency < 1, "budget.forecast_contingency_share must be a share below 1")

        return cls(
            study_id=_str(cfg, "study_id", where="config"),
            primary_model=_str(cfg, "primary_model", where="config"),
            base_run_root=_str(cfg, "base_run_root", where="config"),
            template=template,
            think_close_text=close,
            sampling=dict(sampling),
            arms=arms,
            core_arms=core_arms,
            rungs=rungs,
            a_tokens=_int(d, "a_tokens", low=1, where="dynorch"),
            repeats=_int(d, "repeats", low=1, where="dynorch"),
            cell_rung_order=tuple(order),
            grace_tokens=grace,
            grace_root_only=True,
            max_grace_events_per_run=1,
            terminal_threshold_tokens=threshold,
            per_run_serialization=True,
            harness_max_tokens_ignored=True,
            finish_reasons_returned=("stop", "tool_calls"),
            reasoning_effort=effort,
            subagent_depth=_int(d, "subagent_depth", low=0, where="dynorch"),
            tool_surface=_str(d, "tool_surface", where="dynorch"),
            max_calls_per_run=_int(d, "max_calls_per_run", low=1, where="dynorch"),
            max_prompt_tokens_per_run=_int(d, "max_prompt_tokens_per_run", low=1, where="dynorch"),
            wall_timeout_base_seconds=_int(wall, "base", low=1, where="dynorch.wall_timeout_seconds"),
            wall_timeout_per_rung_seconds=_int(wall, "per_rung", low=0, where="dynorch.wall_timeout_seconds"),
            max_reruns=_int(rerun, "max_reruns", low=0, where="dynorch.rerun_rule"),
            proxy_host=_str(proxy, "host", where="dynorch.proxy"),
            proxy_port=_int(proxy, "port", low=0, where="dynorch.proxy"),
            keepalive_seconds=_int(proxy, "keepalive_seconds", low=1, where="dynorch.proxy"),
            path_prefix=_str(proxy, "path_prefix", where="dynorch.proxy"),
            runner_concurrency=_int(runner, "concurrency", low=1, where="dynorch.runner"),
            sigterm_after_exhaustion_seconds=_int(runner, "sigterm_after_exhaustion_seconds", low=0,
                                                  where="dynorch.runner"),
            sigkill_after_seconds=_int(runner, "sigkill_after_seconds", low=0, where="dynorch.runner"),
            provider_in_flight=dict(d["provider_in_flight"]),
            provider_rps=dict(d["provider_rps"]),
            scratch_root=_str(d, "scratch_root", where="dynorch"),
            prompt_files=dict(prompt_files),
            budget_sentence=sentence,
            terminal_sentence=_str(d, "terminal_sentence", where="dynorch"),
            credit_ceiling_usd=ceiling,
            soft_admission_ceiling_usd=soft,
            pilot_allowance_usd=_decimal(b, "pilot_allowance_usd", where="budget"),
            canary_allowance_usd=_decimal(b, "canary_allowance_usd", where="budget"),
            reservation_multiplier=multiplier,
            max_unresolved_spend_usd=_decimal(b, "max_unresolved_spend_usd", where="budget"),
            account_auto_reload=_bool(b, "account_auto_reload", where="budget"),
            forecast_contingency_share=contingency,
        )
