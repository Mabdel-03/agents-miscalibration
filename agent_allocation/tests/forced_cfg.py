"""The forced27b study config, made valid on a branch whose ``contracts.BENCHMARKS`` predates WP-2.

``configs/forced27b.json`` names the forced_budget_v2 benchmarks (aime, hmmt, omni_math_hard, gpqa_diamond). Until
WP-2's benchmark swap is merged, ``validate_config`` (benchmark ids must equal ``contracts.BENCHMARKS``) rejects that
block, so tests that need a *valid* forced config swap in the current benchmarks block.

AFTER THE WP-2 MERGE: ``needs_benchmark_swap()`` turns False (contracts.BENCHMARKS == the config's ids) and the
config is used verbatim; no test changes are needed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent_allocation.contracts import BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]


def raw_forced_config() -> dict[str, Any]:
    return json.loads((PROJECT / "configs" / "forced27b.json").read_text())


def needs_benchmark_swap(cfg: dict[str, Any] | None = None) -> bool:
    cfg = cfg or raw_forced_config()
    return [b["id"] for b in cfg["benchmarks"]] != list(BENCHMARKS)


def forced_test_config() -> dict[str, Any]:
    cfg = raw_forced_config()
    if needs_benchmark_swap(cfg):
        current = json.loads((PROJECT / "configs" / "lean100.json").read_text())["benchmarks"]
        cfg = copy.deepcopy(cfg)
        cfg["benchmarks"] = current
    return cfg
