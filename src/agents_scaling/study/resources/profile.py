"""Outcome-blind B0 profiler → ``<run_root>/FROZEN.yaml`` (WP4).

    python -m agents_scaling.study.resources.profile --run-id study_v4 --config configs/study_v4.yaml \\
        [--wrapper-tokens-json <run_root>/data/preflight_report.json] [--from-pilot <cells json>] [--dry-run]

Spec §6.5 ("A deterministic outcome-blind development profiler resolves the numerical B0
using the pinned reference models, declared maximum input/output bounds and required
architecture skeletons. B0 is at least the largest full reservation needed for each
mandatory base-N=5 method to produce a valid final-answer attempt, including coordinator
work"; "Validate every mandatory cell at its lowest assigned budget, including N=9 at B1
in the focused checkpoint panel"; B1=B0, B2, B4 (primary), B8), §10.2 (publish the count
convention, configuration hashes, reference dimensions, prefill/decode tables and
worst-case reservation envelopes; "Check ... especially the largest checkpoint at
N=9/B1"), §6.8 ("The expected schedule uses outcome-blind development role/cost
telemetry; the hard ... ceilings use the full reservation envelope").  Architecture
§1.10, §4; corrections P1-1 (B0 analytic: ``L_root_max`` = task envelope + the measured
wrapper of the longest root cell; Table E frozen), P1-2 (N=9 at B1 expected infeasible →
recorded, never patched), §3 (realized admitted-call schedule from the pilot is for
scheduling only).  Amendments B1/B2 (audit §1).

B0 never depends on pilot outcomes: ``--from-pilot`` only appends a realized-call schedule
to the manifest's ``profile.pilot_schedule`` block.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agents_scaling.study.config import StudyConfig, freeze, load_config
from agents_scaling.study.resources.envelopes import (
    B0_METHODS,
    CEN_MINIMAL_PATHS,
    ENVELOPE_NS,
    FALLBACK_WRAPPER_TOKENS,
    TableE,
    b0_candidates,
    fallback_wrappers,
    measure_wrappers,
    minimal_path,
)
from agents_scaling.study.resources.oracle import FlopOracle, oracles_from_checkpoints
from agents_scaling.study.types import SOLVER_OUT_CAP, ProtocolError

DEFAULT_RESULTS_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results")
TABLE_L: tuple[int, ...] = (1024, 6144, 16384, 32768)
TABLE_T: tuple[int, ...] = (1024, 8192)
FEASIBILITY_CHECKPOINTS: tuple[str, ...] = ("8B", "32B")
FEASIBILITY_NS: tuple[int, ...] = (5, 9)
FEASIBILITY_MULTIPLIERS: tuple[int, ...] = (1, 4)
PROFILE_VERSION = "profile-v1"


# --------------------------------------------------------------------------- wrappers


def root_wrapper_from_report(report: Mapping[str, Any], task_cap: int) -> int:
    """``max(root_prompt_max_tokens) − task_cap`` over every checkpoint × root cell of WP1's
    preflight report (``data/export.py --preflight``)."""
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ProtocolError("preflight report has no 'checkpoints' block")
    maxima: list[int] = []
    for size, block in checkpoints.items():
        cells = block.get("cells") if isinstance(block, Mapping) else None
        if not isinstance(cells, Mapping) or not cells:
            raise ProtocolError(f"preflight report: checkpoint {size} has no root cells")
        for cell, stats in cells.items():
            value = stats.get("wrapper_tokens_max") if isinstance(stats, Mapping) else None
            if not isinstance(value, int) or value <= 0:
                raise ProtocolError(f"preflight report: {size}/{cell} lacks a positive wrapper_tokens_max")
            maxima.append(value)
    top = report.get("root_prompt_max_tokens")
    if isinstance(top, int) and top - task_cap != max(maxima):
        raise ProtocolError(
            f"preflight report is inconsistent: root_prompt_max_tokens={top} but max wrapper is {max(maxima)} (+{task_cap})"
        )
    return max(maxima)


def resolve_wrappers(
    cfg: StudyConfig,
    *,
    report: Mapping[str, Any] | None,
    tokenizer: Any | None,
) -> tuple[dict[str, int], dict[str, str]]:
    """Wrapper table and its provenance per key (``preflight`` / ``measured`` / ``fallback``)."""
    provenance: dict[str, str] = {}
    if tokenizer is not None:
        wrappers = measure_wrappers(tokenizer)
        provenance.update({k: "measured" for k in wrappers})
    else:
        wrappers = fallback_wrappers()
        provenance.update({k: "fallback" for k in wrappers})
    if report is not None:
        wrappers["root"] = root_wrapper_from_report(report, int(cfg.caps.task_tokens))
        provenance["root"] = "preflight"
    elif tokenizer is None:
        wrappers["root"] = FALLBACK_WRAPPER_TOKENS
        provenance["root"] = "fallback"
    return wrappers, provenance


def load_flagship_tokenizer(cfg: StudyConfig) -> Any | None:
    """The pinned flagship tokenizer, or ``None`` when the cache lacks it (recorded, not silent)."""
    from agents_scaling.study.inference.tokens import TokenizerUnavailable, load_tokenizer

    try:
        return load_tokenizer(cfg.flagship_checkpoint)
    except TokenizerUnavailable:
        return None


# --------------------------------------------------------------------------- profile


def oracle_tables(oracles: Mapping[str, FlopOracle]) -> dict[str, Any]:
    """Per-checkpoint prefill/decode/call work at the frozen (L, T) grid (§10.2)."""
    out: dict[str, Any] = {}
    for size, oracle in oracles.items():
        out[size] = {
            "c_lin": oracle.c_lin,
            "c_attn": oracle.c_attn,
            "prefill": {str(L): oracle.prefill(L) for L in TABLE_L},
            "decode_at": {str(L): oracle.decode(L) for L in TABLE_L},
            "call": {f"L{L}_T{T}": oracle.call(L, T) for L in TABLE_L for T in TABLE_T},
            "convention": oracle.convention(),
            "oracle_hash": oracle.oracle_hash,
        }
    return out


def feasibility(
    oracles: Mapping[str, FlopOracle],
    table: TableE,
    B0: int,
    *,
    checkpoints: Sequence[str] = FEASIBILITY_CHECKPOINTS,
    Ns: Sequence[int] = FEASIBILITY_NS,
    multipliers: Sequence[int] = FEASIBILITY_MULTIPLIERS,
    cen_path: str = "cycle",
) -> dict[str, Any]:
    """Minimal-path feasibility of every mandatory method at (checkpoint, N, B) (§6.5, P1-2).

    ``feasible`` compares the minimal full-cap path against ``m·B0``; ``first_round_fits``
    additionally reports whether one DEC revision round fits on top of the roots.
    """
    out: dict[str, Any] = {}
    for size in checkpoints:
        oracle = oracles[size]
        for N in Ns:
            for m in multipliers:
                B = m * B0
                row: dict[str, Any] = {}
                for method in B0_METHODS:
                    path = minimal_path(method, oracle, table, N, cen_path=cen_path)
                    entry = {"minimal_path": path["total"], "feasible": path["total"] <= B}
                    if "first_round" in path:
                        entry["first_round"] = path["first_round"]
                        entry["first_round_fits"] = path["total"] + path["first_round"] <= B
                    row[method.value] = entry
                out[f"{size}/N{N}/B{m}"] = {"B": B, "all_feasible": all(v["feasible"] for v in row.values()), "methods": row}
    return out


ALIAS_TABLE: tuple[dict[str, str], ...] = (
    {"producer": "S_FRESH-00 draw k", "aliases": "F00 draw k (k<=9)", "key": "(0,root,k,stateless_bank)"},
    {"producer": "S_HISTORY draw 0", "aliases": "F00 draw 0", "key": "(0,root,0,stateless_bank)"},
    {"producer": "IND_VOTE-11 draw k (module A)", "aliases": "F11 draw k (k<=9)", "key": "(0,root,k,stateless_bank)"},
    {"producer": "IND_VOTE-00 draw k (N panel)", "aliases": "F00 draw k, then the S_FRESH-00 sequence", "key": "(0,root,k,stateless_bank)"},
    {"producer": "DEC NATIVE root s (module A/E/M/B)", "aliases": "none (truthful clause changes the bytes)", "key": "(0,root,s,stateless_bank)"},
    {"producer": "DEC-00 / DEC_ONE_ROUND / IND_PRIVATE_REVISION root s (N panel)", "aliases": "F00 draw s", "key": "(0,root,s,stateless_bank)"},
    {"producer": "CEN_FLAT hub/worker", "aliases": "none", "key": "(0,hub,c,CEN_FLAT) / (slot,worker,c,CEN_FLAT)"},
    {"producer": "DEGREE root i", "aliases": "F00 draw i (i<=8)", "key": "(0,root,i,stateless_bank)"},
)


def build_profile(
    cfg: StudyConfig,
    oracles: Mapping[str, FlopOracle],
    wrappers: Mapping[str, int],
    provenance: Mapping[str, str],
    *,
    pilot_schedule: Mapping[str, Any] | None = None,
    cen_path: str = "cycle",
) -> dict[str, Any]:
    """The manifest ``profile`` block plus ``B0_flops`` (all YAML-native values).

    ``cen_path`` selects how CEN_FLAT's minimal path enters B0 (recorded as
    ``cen_minimal_path``; see :data:`~agents_scaling.study.resources.envelopes.CEN_MINIMAL_PATHS`).
    """
    flagship = cfg.flagship
    if flagship not in oracles:
        raise ProtocolError(f"no oracle for the flagship {flagship}")
    if cen_path not in CEN_MINIMAL_PATHS:
        raise ValueError(f"cen_path must be one of {CEN_MINIMAL_PATHS}")
    table = TableE(cfg.caps, dict(wrappers))
    candidates = b0_candidates(oracles[flagship], table, N=5, cen_path=cen_path)
    B0 = max(c["total"] for c in candidates.values())
    binding = max(candidates, key=lambda k: candidates[k]["total"])
    budgets = {f"B{m}": m * B0 for m in cfg.budget.budget_multipliers}
    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "flagship": flagship,
        "B0_flops": B0,
        "B0_binding_method": binding,
        "B0_rule": "max over {IND_VOTE, DEC, CEN_FLAT, S_FRESH, S_HISTORY} at N=5 of the minimal full-cap path (amendment B1)",
        "cen_minimal_path": cen_path,
        "B0_candidates": {k: {"total": v["total"], "calls": v["calls"], **{kk: vv for kk, vv in v.items() if kk in ("first_round", "next_revision")}} for k, v in candidates.items()},
        "budgets": budgets,
        "primary_multiplier": cfg.budget.primary,
        "L_root_max": table.root_prompt_max(),
        "solver_out_cap": SOLVER_OUT_CAP,
        "root_reservation_flagship": oracles[flagship].reservation(table.root_prompt_max(), SOLVER_OUT_CAP),
        "wrapper_source": dict(provenance),
        "table_e": table.to_dict(),
        "oracle_tables": oracle_tables(oracles),
        "feasibility": feasibility(oracles, table, B0, cen_path=cen_path),
        "minimal_paths_by_N": {
            size: {
                str(N): {m.value: minimal_path(m, oracles[size], table, N, cen_path=cen_path)["total"] for m in B0_METHODS}
                for N in ENVELOPE_NS
            }
            for size in oracles
        },
        "alias_table": [dict(row) for row in ALIAS_TABLE],
        "pilot_schedule": dict(pilot_schedule) if pilot_schedule else None,
        "notes": [
            "B0 is analytic (frozen envelopes × full decode reservations); pilot telemetry never enters it (P1-1).",
            "Dense-panel checkpoints share the same numerical allowance (§6.5); their feasibility rows are informative.",
        ],
    }
    n9 = profile["feasibility"].get(f"{flagship}/N9/B1")
    if n9 is not None:
        profile["n9_b1_flagship_feasible"] = n9["all_feasible"]
    return profile


# --------------------------------------------------------------------------- pilot schedule


def pilot_schedule(run_root: Path, cells_file: Path) -> dict[str, Any]:
    """Realized admitted calls / spent / stop reasons per method from completed pilot item
    files (scheduling telemetry only, §6.8).  ``cells_file`` lists cells (a JSON list of
    objects with ``cell_id``, or a ``{"cells": [...]}`` object)."""
    data = json.loads(cells_file.read_text(encoding="utf-8"))
    cells = data.get("cells") if isinstance(data, Mapping) else data
    if not isinstance(cells, list):
        raise ProtocolError(f"{cells_file}: expected a list of cells")
    per_method: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id") if isinstance(cell, Mapping) else None
        if not isinstance(cell_id, str):
            raise ProtocolError(f"{cells_file}: a cell entry lacks cell_id")
        items_dir = run_root / "cells" / cell_id / "items"
        if not items_dir.is_dir():
            continue
        for path in sorted(items_dir.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            episode = item.get("episode")
            if not isinstance(episode, Mapping):
                continue
            method = str(item.get("cell", {}).get("method") or episode.get("method"))
            ledger = episode.get("ledger") or {}
            bucket = per_method.setdefault(method, {"episodes": 0, "calls_admitted": [], "spent": [], "stop_reasons": {}})
            bucket["episodes"] += 1
            bucket["calls_admitted"].append(int(ledger.get("calls_admitted") or 0))
            bucket["spent"].append(int(ledger.get("spent") or 0))
            reason = str(ledger.get("stop_reason"))
            bucket["stop_reasons"][reason] = bucket["stop_reasons"].get(reason, 0) + 1
    summary: dict[str, Any] = {}
    for method, bucket in per_method.items():
        calls = bucket["calls_admitted"]
        spent = bucket["spent"]
        summary[method] = {
            "episodes": bucket["episodes"],
            "calls_admitted": {"min": min(calls), "max": max(calls), "mean": sum(calls) / len(calls)},
            "spent": {"min": min(spent), "max": max(spent), "mean": sum(spent) // len(spent)},
            "stop_reasons": bucket["stop_reasons"],
        }
    return {"cells_file": str(cells_file), "methods": summary}


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agents_scaling.study.resources.profile", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=None, help="study yaml (default configs/study_v4.yaml)")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--wrapper-tokens-json", default=None, help="WP1 preflight_report.json (root wrapper source)")
    parser.add_argument("--from-pilot", default=None, help="cells json of the pilot: appends a realized-call schedule (scheduling only)")
    parser.add_argument("--hf-home", default=None, help="override HF_HOME for config.json lookups")
    parser.add_argument("--no-tokenizer", action="store_true", help="skip the flagship tokenizer (fallback wrappers, recorded)")
    parser.add_argument(
        "--cen-minimal-path",
        default="cycle",
        choices=list(CEN_MINIMAL_PATHS),
        help="how CEN_FLAT's minimal path enters B0: 'cycle' = hub + (N-1) workers + final hub (default), 'hub_only' = hub + final hub",
    )
    parser.add_argument("--code-version", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the profile; do not write FROZEN.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_root = Path(args.results_root) / args.run_id
    oracles = oracles_from_checkpoints(cfg.checkpoints, args.hf_home)
    report = None
    if args.wrapper_tokens_json:
        report = json.loads(Path(args.wrapper_tokens_json).read_text(encoding="utf-8"))
    tokenizer = None if args.no_tokenizer else load_flagship_tokenizer(cfg)
    wrappers, provenance = resolve_wrappers(cfg, report=report, tokenizer=tokenizer)
    schedule = pilot_schedule(run_root, Path(args.from_pilot)) if args.from_pilot else None
    profile = build_profile(cfg, oracles, wrappers, provenance, pilot_schedule=schedule, cen_path=args.cen_minimal_path)
    extra = {"B0_flops": str(profile["B0_flops"]), "profile": profile}
    brief = {
        "B0_flops": profile["B0_flops"],
        "B0_binding_method": profile["B0_binding_method"],
        "cen_minimal_path": profile["cen_minimal_path"],
        "budgets": profile["budgets"],
        "L_root_max": profile["L_root_max"],
        "wrapper_source": sorted(set(provenance.values())),
        "n9_b1_flagship_feasible": profile.get("n9_b1_flagship_feasible"),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_root": str(run_root), **brief}, indent=2))
        return 0
    target = freeze(run_root, extra, config=cfg, code_version=args.code_version)
    print(json.dumps({"frozen": str(target), **brief}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "ALIAS_TABLE",
    "DEFAULT_RESULTS_ROOT",
    "FEASIBILITY_CHECKPOINTS",
    "FEASIBILITY_MULTIPLIERS",
    "FEASIBILITY_NS",
    "PROFILE_VERSION",
    "TABLE_L",
    "TABLE_T",
    "build_parser",
    "build_profile",
    "feasibility",
    "load_flagship_tokenizer",
    "main",
    "oracle_tables",
    "pilot_schedule",
    "resolve_wrappers",
    "root_wrapper_from_report",
]
