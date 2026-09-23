"""dynorch_v1, WP-C: the cost forecast of plan section 8 and the Gate 2 re-fit (contract section 7).

The plan's table is reproduced from ``configs/dynorch_v1.json`` and the named assumptions below (every one of them
is a constant here, none is read from a prose document):

* Parasail prices, ``$0.24/M`` input and ``$2.20/M`` output (plan "Verified facts", 2026-09-23; the config's
  ``planning_prices`` output price of 2.50 is the standardized-cost price of the report, not this forecast's);
* output per run at the ceiling is ``C + G`` (cap plus one grace);
* calls per run ``n_k = {1: 2, 2: 3, 4: 4, 8: 6, 16: 8, 25: 10}`` for a harness arm, 1 for H0;
* per-call input at the ceiling is ``5,500 + the run's output`` (re-injected reasoning; the ceiling charges the
  whole run output on every call), and ``5,500`` flat for the expected (lower) figure; H0's single call carries
  ``5,500`` (there is nothing to re-inject);
* every block carries a 25 % contingency line (``budget.forecast_contingency_share``) for reruns, duplicates and
  announce turns, reported beside the ceiling, not inside it.

``plan_table(cfg)`` gives the block table (canary + Gate 1, pilot, core H1+H3, core H0, core total, E0, E1, E2, E3);
``refit(runs, cfg)`` recomputes the per-(arm, rung) cost from measured calls, prompt tokens, completion tokens and
realized share in ``dynorch_runs.parquet`` with the same contingency; both return DataFrames, and ``main`` prints
them (``python -m agent_allocation.dynorch.forecast --config configs/dynorch_v1.json [--runs tables/dynorch_runs.parquet]``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

MILLION = Decimal(1_000_000)
PARASAIL_INPUT_USD_PER_M = Decimal("0.24")
PARASAIL_OUTPUT_USD_PER_M = Decimal("2.20")
BASE_PROMPT_TOKENS = 5_500
CALLS_PER_RUN: dict[int, int] = {1: 2, 2: 3, 4: 4, 8: 6, 16: 8, 25: 10}
H0_CALLS_PER_RUN = 1
HARNESS_KIND = "harness"
NULL_KIND = "null"
CORE_HARNESS_ARMS = ("H1", "H3")
CORE_NULL_ARM = "H0"

#: Plan section 8, frozen: block -> (runs, ceiling USD, expected low USD, expected high USD).
PLAN_SECTION_8: dict[str, tuple[int, float, float, float]] = {
    "canary_gate1": (10, 1.5, 0.7, 0.7),
    "pilot": (144, 7.0, 4.0, 4.0),
    "core_h1_h3": (1152, 58.0, 35.0, 45.0),
    "core_h0": (576, 14.0, 10.0, 10.0),
    "core_total": (1882, 80.0, 50.0, 60.0),
    "e0_capped": (384, 12.0, 8.0, 8.0),
    "e1_repeat": (1152, 58.0, 35.0, 45.0),
    "e2_spawn_disabled": (384, 12.0, 8.0, 8.0),
}


class ForecastError(ValueError):
    """The config lacks a block the forecast needs, or a runs table lacks a column the re-fit needs."""


@dataclass(frozen=True)
class RunForecast:
    """Tokens and dollars of one run at rung ``rung`` under the section 8 assumptions."""
    kind: str
    rung: int
    calls: int
    output_tokens: int
    input_tokens_ceiling: int
    input_tokens_expected: int

    @property
    def usd_output(self) -> Decimal:
        return Decimal(self.output_tokens) * PARASAIL_OUTPUT_USD_PER_M / MILLION

    @property
    def usd_ceiling(self) -> Decimal:
        return self.usd_output + Decimal(self.input_tokens_ceiling) * PARASAIL_INPUT_USD_PER_M / MILLION

    @property
    def usd_expected(self) -> Decimal:
        return self.usd_output + Decimal(self.input_tokens_expected) * PARASAIL_INPUT_USD_PER_M / MILLION


def _dyn(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "dynorch" not in cfg:
        raise ForecastError("config has no 'dynorch' block")
    return cfg["dynorch"]


def contingency_share(cfg: Mapping[str, Any]) -> Decimal:
    """``budget.forecast_contingency_share`` (0.25 in the frozen config)."""
    try:
        return Decimal(str(cfg["budget"]["forecast_contingency_share"]))
    except KeyError:
        raise ForecastError("config lacks budget.forecast_contingency_share") from None


def run_forecast(kind: str, rung: int, cfg: Mapping[str, Any]) -> RunForecast:
    """Section 8 per-run forecast: ``kind`` is ``"harness"`` (multi-call arms H1/H3 and their variants) or
    ``"null"`` (H0, one call)."""
    dyn = _dyn(cfg)
    a_tokens, grace = int(dyn["a_tokens"]), int(dyn["grace_tokens"])
    if kind == HARNESS_KIND:
        if rung not in CALLS_PER_RUN:
            raise ForecastError(f"no calls-per-run assumption for rung {rung}")
        calls = CALLS_PER_RUN[rung]
    elif kind == NULL_KIND:
        calls = H0_CALLS_PER_RUN
    else:
        raise ForecastError(f"unknown run kind {kind!r}")
    output = rung * a_tokens + grace
    reinjected = output if kind == HARNESS_KIND else 0
    return RunForecast(kind, rung, calls, output, calls * (BASE_PROMPT_TOKENS + reinjected),
                       calls * BASE_PROMPT_TOKENS)


def arm_kind(arm: str, cfg: Mapping[str, Any]) -> str:
    """``"null"`` for the harness-free control, ``"harness"`` for every other arm in the config."""
    dyn = _dyn(cfg)
    arms = {**dyn["arms"], **dyn.get("extension_arms", {})}
    if arm not in arms:
        raise ForecastError(f"arm {arm!r} is not in the config")
    return NULL_KIND if arms[arm]["harness"] == "null" else HARNESS_KIND


@dataclass(frozen=True)
class Block:
    name: str
    cells: tuple[tuple[str, int, int], ...]      # (arm, rung, n_runs)


def blocks(cfg: Mapping[str, Any]) -> list[Block]:
    """The section 8 blocks, enumerated from the config (rungs, panels, pilot, canary, extension arms)."""
    dyn = _dyn(cfg)
    rungs = [int(r) for r in dyn["rungs"]]
    n_main = int(cfg["panels"]["main_items"])
    n_bench = len(cfg["benchmarks"])
    pilot, canary = dyn["pilot"], dyn["canary"]
    n_pilot_items = int(pilot["dev_items_per_benchmark"]) * n_bench
    n_pilot_h0 = int(pilot["h0_items_per_benchmark"]) * n_bench
    n_canary_items = len(canary["items"])
    out: list[Block] = []
    out.append(Block("canary_gate1", tuple(
        [(arm, int(canary["rung"]), n_canary_items) for arm in (*CORE_HARNESS_ARMS, CORE_NULL_ARM)]
        + [(arm, int(canary["ceiling_probe_rung"]), int(canary["ceiling_probe_items"])) for arm in CORE_HARNESS_ARMS])))
    out.append(Block("pilot", tuple(
        [(arm, int(r), n_pilot_items) for arm in CORE_HARNESS_ARMS for r in pilot["rungs"]]
        + [(CORE_NULL_ARM, int(pilot["h0_rung"]), n_pilot_h0)])))
    out.append(Block("core_h1_h3", tuple((arm, r, n_main) for arm in CORE_HARNESS_ARMS for r in rungs)))
    out.append(Block("core_h0", tuple((CORE_NULL_ARM, r, n_main) for r in rungs)))
    out.append(Block("e0_capped", tuple((arm, r, n_main) for arm in ("H1c", "H3c") for r in (4, 16))))
    out.append(Block("e1_repeat", tuple((arm, r, n_main) for arm in CORE_HARNESS_ARMS for r in rungs)))
    out.append(Block("e2_spawn_disabled", tuple((arm, r, n_main) for arm in ("H1n", "H3n") for r in (4, 16))))
    out.append(Block("e3_openclaw", tuple(("H2", r, n_main) for r in (4, 16))))
    return out


def block_forecast(block: Block, cfg: Mapping[str, Any]) -> dict[str, Any]:
    runs = 0
    ceiling = expected = Decimal(0)
    for arm, rung, n in block.cells:
        rf = run_forecast(arm_kind(arm, cfg), rung, cfg)
        runs += n
        ceiling += n * rf.usd_ceiling
        expected += n * rf.usd_expected
    share = contingency_share(cfg)
    return {"block": block.name, "runs": runs, "expected_usd": expected, "ceiling_usd": ceiling,
            "contingency_usd": ceiling * share, "ceiling_with_contingency_usd": ceiling * (1 + share)}


def plan_table(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The plan section 8 table: one row per block plus ``core_total`` (canary + pilot + core H1/H3 + core H0),
    with ``expected_usd`` (flat-input model), ``ceiling_usd`` (re-injection ceiling) and the contingency line;
    ``plan_*`` columns carry the frozen section 8 numbers where the plan states them."""
    rows = [block_forecast(b, cfg) for b in blocks(cfg)]
    core = [r for r in rows if r["block"] in ("canary_gate1", "pilot", "core_h1_h3", "core_h0")]
    share = contingency_share(cfg)
    total_ceiling = sum((r["ceiling_usd"] for r in core), Decimal(0))
    rows.insert(4, {"block": "core_total", "runs": sum(r["runs"] for r in core),
                    "expected_usd": sum((r["expected_usd"] for r in core), Decimal(0)),
                    "ceiling_usd": total_ceiling, "contingency_usd": total_ceiling * share,
                    "ceiling_with_contingency_usd": total_ceiling * (1 + share)})
    frame = pd.DataFrame(rows)
    for col in ("expected_usd", "ceiling_usd", "contingency_usd", "ceiling_with_contingency_usd"):
        frame[col] = frame[col].map(lambda d: round(float(d), 4))
    frame["plan_runs"] = frame["block"].map(lambda b: PLAN_SECTION_8.get(b, (None,) * 4)[0])
    frame["plan_ceiling_usd"] = frame["block"].map(lambda b: PLAN_SECTION_8.get(b, (None,) * 4)[1])
    frame["plan_expected_lo_usd"] = frame["block"].map(lambda b: PLAN_SECTION_8.get(b, (None,) * 4)[2])
    frame["plan_expected_hi_usd"] = frame["block"].map(lambda b: PLAN_SECTION_8.get(b, (None,) * 4)[3])
    return frame


def per_item_table(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per item over the core rungs: output USD, input USD (ceiling), total, for a harness arm and for H0."""
    dyn = _dyn(cfg)
    rows = []
    for kind in (HARNESS_KIND, NULL_KIND):
        rfs = [run_forecast(kind, int(r), cfg) for r in dyn["rungs"]]
        rows.append({"kind": kind, "output_usd": float(sum(r.usd_output for r in rfs)),
                     "input_usd_ceiling": float(sum(r.usd_ceiling - r.usd_output for r in rfs)),
                     "total_usd_ceiling": float(sum(r.usd_ceiling for r in rfs)),
                     "total_usd_expected": float(sum(r.usd_expected for r in rfs))})
    return pd.DataFrame(rows)


REFIT_COLUMNS = ("arm", "rung", "cap_tokens", "calls", "prompt_total", "completion_total", "grace_tokens", "cost_usd")


def refit(runs: pd.DataFrame | str | Path, cfg: Mapping[str, Any], *,
          planned_runs: Mapping[tuple[str, int], int] | None = None) -> pd.DataFrame:
    """Gate 2 re-fit from ``dynorch_runs.parquet``: per (arm, rung) the measured mean calls, prompt tokens,
    completion tokens (with grace), realized share and ledger cost, the re-fitted cost per run at Parasail prices
    (``prompt * in + completion * out``), and, when ``planned_runs`` gives the planned count per (arm, rung), the
    forecast for the remaining runs with the 25 % contingency."""
    if not isinstance(runs, pd.DataFrame):
        runs = pd.read_parquet(runs)
    missing = [c for c in REFIT_COLUMNS if c not in runs.columns]
    if missing:
        raise ForecastError(f"runs table lacks {missing}")
    if runs.empty:
        raise ForecastError("runs table is empty; nothing to re-fit")
    share = contingency_share(cfg)
    rows = []
    for (arm, rung), g in runs.groupby(["arm", "rung"], sort=True):
        prompt = float(g["prompt_total"].mean())
        completion = float((g["completion_total"] + g["grace_tokens"]).mean())
        cost_refit = (Decimal(str(prompt)) * PARASAIL_INPUT_USD_PER_M
                      + Decimal(str(completion)) * PARASAIL_OUTPUT_USD_PER_M) / MILLION
        measured = g["cost_usd"].map(lambda v: Decimal(str(v)))
        planned = None if planned_runs is None else int(planned_runs.get((str(arm), int(rung)), 0))
        remaining = None if planned is None else max(0, planned - len(g))
        row = {"arm": arm, "rung": int(rung), "n_runs": int(len(g)), "mean_calls": float(g["calls"].mean()),
               "mean_prompt_tokens": prompt, "mean_completion_tokens": completion,
               "mean_realized_share": float((g["completion_total"] / g["cap_tokens"]).mean()),
               "mean_cost_usd_measured": float(sum(measured, Decimal(0)) / len(g)),
               "cost_usd_per_run_refit": float(cost_refit),
               "cost_usd_per_run_refit_with_contingency": float(cost_refit * (1 + share)),
               "planned_runs": planned, "remaining_runs": remaining,
               "forecast_remaining_usd": None if remaining is None else float(remaining * cost_refit * (1 + share))}
        rows.append(row)
    return pd.DataFrame(rows)


def planned_runs_core(cfg: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    """Planned runs per (arm, rung) of the core (every core arm at every rung on the main panel)."""
    dyn = _dyn(cfg)
    n_main = int(cfg["panels"]["main_items"])
    return {(arm, int(r)): n_main for arm in dyn["core_arms"] for r in dyn["rungs"]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", default=None, help="dynorch_runs.parquet for the Gate 2 re-fit")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(plan_table(cfg).to_string(index=False))
        print()
        print(per_item_table(cfg).to_string(index=False))
        if a.runs:
            print()
            print(refit(a.runs, cfg, planned_runs=planned_runs_core(cfg)).to_string(index=False))
    return 0


__all__ = ["BASE_PROMPT_TOKENS", "CALLS_PER_RUN", "PARASAIL_INPUT_USD_PER_M", "PARASAIL_OUTPUT_USD_PER_M",
           "PLAN_SECTION_8", "Block", "ForecastError", "RunForecast", "arm_kind", "block_forecast", "blocks",
           "contingency_share", "per_item_table", "plan_table", "planned_runs_core", "refit", "run_forecast"]


if __name__ == "__main__":
    raise SystemExit(main())
