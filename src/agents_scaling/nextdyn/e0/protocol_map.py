"""What each legacy run's protocol actually was (spec 7.2; plan §2.6 output 2).

A legacy claim can only be compared with a new one if the two ran the same protocol, and the
three legacy runs describe their protocols in three different vocabularies -- ARD36 in a frozen
``contract.py`` plus per-episode columns, the study_v4 atlas in a ``FROZEN.yaml`` with an
amendments ledger, Run0 in a ``cells.json`` axis dictionary.  This module reads each one in its
own vocabulary and writes ``legacy_protocol_map.json`` in one.

The honest part is the ``serving_regime`` field.  Run0's serving conditions changed during the
run (the ``era_post_trim`` and ``ctx32b_capped`` flags in its analysis view are the surviving
traces), and the full split is only recoverable from the server logs, which this module does
not parse.  So the Run0 entry is marked ``partial`` rather than being reported as a clean axis:
a mixed-protocol run labelled as a single protocol is exactly how a legacy number becomes a
comparison it cannot support.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Final, Mapping

from agents_scaling.nextdyn.e0 import registry as R

REGIME_COMPLETE: Final[str] = "complete"
REGIME_PARTIAL: Final[str] = "partial"
BLOCKED: Final[str] = R.BLOCKED_MISSING_SOURCE

#: The ARD36 episode columns that carry its protocol, and the decision columns that carry its
#: truncation behaviour.  Named here because the legacy tables have no schema module to import.
_ARD36_EPISODE_AXES: Final[tuple[str, ...]] = (
    "protocol", "memory_policy", "packet_policy", "horizon", "reasoning_allowance",
    "final_allowance", "N", "checkpoint", "cohort",
)
_ARD36_DECISION_AXES: Final[tuple[str, ...]] = (
    "budget_tokens", "boundary_injected", "cap_hit", "final_overrun",
)


def _read_parquet(path: Path, columns) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema_arrow.names)
        keep = [c for c in columns if c in available]
        return pq.read_table(path, columns=keep).to_pylist()
    except Exception:
        return []


def _histogram(rows, column: str, limit: int = 12) -> dict[str, int]:
    counts = Counter(str(r.get(column)) for r in rows if column in r)
    return {k: int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]}


def ard36_entry(sources: R.LegacySources) -> dict[str, Any]:
    """ARD36: protocol axes off ``episodes`` and the truncation behaviour off ``decisions``."""
    episodes = R.resolve(sources, "ard36.tables", hash_files=False)
    if not episodes.exists:
        return {"status": BLOCKED, "reason": f"{episodes.path} is absent"}
    ep_rows = _read_parquet(episodes.path / "episodes.parquet", _ARD36_EPISODE_AXES)
    de_rows = _read_parquet(episodes.path / "decisions.parquet", _ARD36_DECISION_AXES)
    if not ep_rows:
        return {"status": BLOCKED, "reason": f"{episodes.path}/episodes.parquet is absent or unreadable"}
    return {
        "status": REGIME_COMPLETE,
        "tables_read": str(episodes.path),
        "root_convention": "five independent draws of the same item seed one episode per arm",
        "memory": _histogram(ep_rows, "memory_policy"),
        "packet_policy": _histogram(ep_rows, "packet_policy"),
        "horizon": _histogram(ep_rows, "horizon"),
        "reasoning_allowance": _histogram(ep_rows, "reasoning_allowance"),
        "final_allowance": _histogram(ep_rows, "final_allowance"),
        "protocols": _histogram(ep_rows, "protocol"),
        "checkpoints": _histogram(ep_rows, "checkpoint"),
        "team_sizes": _histogram(ep_rows, "N"),
        "sampling": "one sample per (episode, round, agent); no repair calls",
        "truncation": {
            "budget_tokens": _histogram(de_rows, "budget_tokens"),
            "boundary_injected_rate": _rate(de_rows, "boundary_injected"),
            "cap_hit_rate": _rate(de_rows, "cap_hit"),
            "final_overrun_rate": _rate(de_rows, "final_overrun"),
        },
        "output_schema": "tagged text report (three tags), not JSON",
        "selection": "current-round plurality with an archive variant; audit output never joined",
        "grading": "binary is_correct beside a four-valued evaluator_status",
        "serving_regime": REGIME_COMPLETE,
        "comparability_note": (
            "ARD36 is the only legacy run with both arms of a PEER/PRIVATE contrast under one "
            "contract, so it is the only one whose contrast rows E0 labels as such."),
    }


def _rate(rows, column: str) -> float | None:
    values = [bool(r.get(column)) for r in rows if column in r and r.get(column) is not None]
    return (sum(values) / len(values)) if values else None


def atlas_entry(sources: R.LegacySources) -> dict[str, Any]:
    """study_v4: the frozen manifest and amendments, plus the per-episode design axes."""
    root = R.resolve(sources, "study_v4.root", hash_files=False)
    tables_dir = R.resolve(sources, "study_v4.tables", hash_files=False)
    if not root.exists:
        return {"status": BLOCKED, "reason": f"{root.path} is absent"}
    frozen = root.path / "FROZEN.yaml"
    amendments = root.path / "freeze" / "amendments.json"
    rows = _read_parquet(tables_dir.path / "episodes.parquet",
                         ("module", "method", "N", "B", "degree", "framing", "stop_reason"))
    return {
        "status": REGIME_COMPLETE if rows else REGIME_PARTIAL,
        "tables_read": str(tables_dir.path),
        "frozen_manifest_present": frozen.exists(),
        "amendments_present": amendments.exists(),
        "root_convention": "per-module candidate generation; no five-agent team",
        "modules": _histogram(rows, "module"),
        "methods": _histogram(rows, "method"),
        "team_sizes": _histogram(rows, "N"),
        "budgets": _histogram(rows, "B"),
        "degrees": _histogram(rows, "degree"),
        "framings": _histogram(rows, "framing"),
        "stop_reasons": _histogram(rows, "stop_reason"),
        "sampling": "one candidate pool per cell; selection over the pool",
        "truncation": "per-module budget B; no thinking-boundary instrumentation",
        "output_schema": "per-module free text with a code/answer extractor",
        "selection": "AST vote on code, judge on HLE",
        "grading": "hidden tests for BCB; an LLM judge for HLE (judge agreement, not accuracy)",
        "serving_regime": REGIME_COMPLETE,
        "comparability_note": (
            "No PEER/PRIVATE axis exists here.  Atlas rows are never reported as a "
            "communication contrast; they carry design and selection facts only."),
    }


def run0_entry(sources: R.LegacySources) -> dict[str, Any]:
    """Run0: axes from ``cells.json`` and the surviving serving flags from the analysis view."""
    cells = R.resolve(sources, "full_sweep_v1.cells_json", hash_files=False)
    view = R.resolve(sources, "full_sweep_v1.cache.analysis_view", hash_files=False)
    if not cells.exists:
        return {"status": BLOCKED, "reason": f"{cells.path} is absent"}
    try:
        data = json.loads(cells.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": BLOCKED, "reason": f"{cells.path} is unreadable ({exc})"}
    entries = data if isinstance(data, list) else data.get("cells", [])
    axes: dict[str, dict[str, int]] = {}
    for key in ("benchmark", "protocol", "n_agents", "rounds", "model", "seed"):
        axes[key] = _histogram(entries, key)
    flags = _read_parquet(view.path, ("era_post_trim", "ctx32b_capped", "design_cell", "is_core"))
    return {
        "status": REGIME_PARTIAL,
        "cells_read": str(cells.path),
        "n_cells": len(entries),
        "axes": axes,
        "root_convention": "per-benchmark item sampling with a permutation seed per (benchmark, seed)",
        "sampling": "one draw per (cell, item)",
        "truncation": "context cap changed mid-run; the surviving trace is ctx32b_capped",
        "output_schema": "per-benchmark extractor",
        "selection": "majority vote where a vote existed",
        "grading": "per-benchmark exact match / hidden tests",
        "serving_regime": REGIME_PARTIAL,
        "serving_flags": {
            "era_post_trim": _histogram(flags, "era_post_trim"),
            "ctx32b_capped": _histogram(flags, "ctx32b_capped"),
            "is_core": _histogram(flags, "is_core"),
        },
        "comparability_note": (
            "The serving regime changed during the run and the full split is only in the server "
            "logs, which this module does not parse: every Run0 row is labelled supplementary "
            "mixed-protocol evidence and is never reported as a PEER/PRIVATE contrast."),
    }


def protocol_map(sources: R.LegacySources) -> dict[str, Any]:
    return {"ard36": ard36_entry(sources), "study_v4": atlas_entry(sources),
            "full_sweep_v1": run0_entry(sources)}


def write(out_dir: str | os.PathLike, payload: Mapping[str, Any]) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "legacy_protocol_map.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


__all__ = ["BLOCKED", "REGIME_COMPLETE", "REGIME_PARTIAL", "ard36_entry", "atlas_entry",
           "protocol_map", "run0_entry", "write"]
