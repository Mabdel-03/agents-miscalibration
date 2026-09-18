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


#: The provisional three-equal-shard layout the shard-logic tests were written against. The live config's shards
#: (Mancer 3 : Parasail 2, from the 2026-09-18 screen) are checked separately in test_shards.py.
THREE_EQUAL_SHARDS = [{"id": "parasail", "endpoint_tag": "parasail/fp8", "share": 1},
                      {"id": "mancer", "endpoint_tag": "mancer/fp8", "share": 1},
                      {"id": "dekallm", "endpoint_tag": "dekallm", "share": 1}]


def three_shard_config() -> dict[str, Any]:
    cfg = copy.deepcopy(forced_test_config())
    cfg["shards"] = copy.deepcopy(THREE_EQUAL_SHARDS)
    cfg.pop("scheduler_by_shard", None)
    return cfg


def request_ready(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop sampling keys the request builder on this branch does not know yet (``top_k``: WP-1 adds it to the forced
    request contract). AFTER THE WP-1 MERGE this is a no-op."""
    from agent_allocation.api import request
    known = getattr(request, "_KNOWN_SAMPLING_KEYS", None)
    if known is None:
        return cfg
    cfg = copy.deepcopy(cfg)
    for key in [k for k in cfg["sampling"] if k not in known]:
        cfg["sampling"].pop(key)
    return cfg
