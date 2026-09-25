"""Evaluated pools per (item, panel, configuration) (contract §5.4) -> ``pools.parquet``.

A configuration ``n x a`` is ``n`` independent calls at allowance ``a·A``; its members are the prefix replicates
r0 … r(n−1) at that allowance (A-level L96/C48 pools use legacy slots 0 … n−1). ``diag5`` is the 5-draw
diagnostic scope: L96 slots 20–24 (legacy) or ``t200_diag`` r0–4. Nothing here reads labels.

Panel draw sources (rows of ``analysis_inputs/draws.parquet``):

* ``L96``: legacy main roots/deeps + ``e1_fill`` (+ ``c48_extra`` for C48 items);
* ``C48``: the same rows restricted to C48 items (B = 32A configurations only here);
* ``V48``: ``v48_bank``; ``T200``: ``t200_bank`` (``diag5`` from ``t200_diag``).

A pool is formed only when every prefix replicate is planned (present as a row); members that ended without a
candidate stay in ``member_ids`` and are handled by the readout as absent (07-R12).
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import pandas as pd

__all__ = ["B16_MENU", "B8_MENU", "CONFIGS", "DEV_TRAINING", "LEGACY_CONFIGS", "POOL_COLUMNS", "WITHHELD",
           "build_pools", "c48_fit_scopes", "config",
           "pool_id"]

PASSK_SEED = 20260925
#: config_id -> (n, a/A)
CONFIGS: dict[str, tuple[int, int]] = {
    "16x1": (16, 1), "8x2": (8, 2), "4x4": (4, 4), "2x8": (2, 8), "1x16": (1, 16),
    "8x1": (8, 1), "4x2": (4, 2), "2x4": (2, 4), "1x8": (1, 8),
    "1x1": (1, 1), "2x1": (2, 1), "1x2": (1, 2), "4x1": (4, 1), "2x2": (2, 2), "1x4": (1, 4),
    "2x16": (2, 16), "4x8": (4, 8), "8x4": (8, 4),
    "diag5": (5, 1),
}
#: legacy-only readouts (E0 replay of the forced_v2 headlines): vote@25 and deep/25
LEGACY_CONFIGS: dict[str, tuple[int, int]] = {"25x1": (25, 1), "1x25": (1, 25)}
B16_MENU = ("16x1", "8x2", "4x4", "2x8", "1x16")
B8_MENU = ("8x1", "4x2", "2x4", "1x8")
DEV_TRAINING = ("1x1", "2x1", "1x2", "4x1", "2x2", "1x4", "16x1", "8x2", "2x8", "1x16", "1x8", "2x16", "4x8", "8x4")
C48_ONLY = ("2x16", "4x8", "8x4")
WITHHELD = frozenset({(4, 4), (8, 1), (4, 2), (2, 4)})
POOL_COLUMNS = ["pool_id", "item_id", "benchmark", "panel", "config_id", "n", "a", "R", "B", "member_ids", "view",
                "candidate_order_seed", "source_partitions"]

_PANEL_PARTITIONS = {
    "L96": ("legacy_main", "e1_fill", "c48_extra"),
    "C48": ("legacy_main", "e1_fill", "c48_extra"),
    "V48": ("v48_bank",),
    "T200": ("t200_bank",),
}


def config(config_id: str) -> tuple[int, int]:
    return CONFIGS.get(config_id) or LEGACY_CONFIGS[config_id]


def pool_id(item_id: str, config_id: str, view: str, member_ids: Iterable[str]) -> str:
    blob = json.dumps(["passk_cont_v1/pool", item_id, config_id, view, list(member_ids)], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _replicate_index(rows: pd.DataFrame) -> dict[tuple[int, int], str]:
    """(allowance, replicate) -> candidate id; a duplicate coordinate is a manifest error."""
    index: dict[tuple[int, int], str] = {}
    for cid, a, r in zip(rows["candidate_id"], rows["allowance_a"], rows["replicate"]):
        key = (int(a), int(r))
        if key in index:
            raise ValueError(f"duplicate replicate {key} for one item ({index[key]}, {cid})")
        index[key] = cid
    return index


def _panel_items(draws: pd.DataFrame, panel: str) -> list[str]:
    has = draws["panel_roles"].map(lambda roles: roles is not None and panel in list(roles))
    return sorted(draws.loc[has, "item_id"].unique())


def build_pools(draws: pd.DataFrame, *, panels: Iterable[str] = ("L96", "C48", "V48", "T200"),
                configs: Iterable[str] | None = None, view: str = "current") -> pd.DataFrame:
    """Pools for every panel item and configuration whose prefix replicates are all planned."""
    wanted = tuple(configs) if configs is not None else tuple(CONFIGS)
    rows = []
    for panel in panels:
        items = _panel_items(draws, panel)
        if panel in ("V48", "T200"):
            extra = _PANEL_PARTITIONS[panel] + (("t200_diag",) if panel == "T200" else ())
            items = sorted(set(items) | set(draws.loc[draws["partition"].isin(extra), "item_id"]))
        sources = _PANEL_PARTITIONS[panel]
        for item_id in items:
            mine = draws[(draws["item_id"] == item_id)]
            bench = mine["benchmark"].iloc[0] if len(mine) else None
            main = mine[mine["partition"].isin(sources)]
            index = _replicate_index(main)
            for config_id in wanted:
                n, a = CONFIGS.get(config_id) or LEGACY_CONFIGS[config_id]
                if config_id in C48_ONLY and panel not in ("C48",):
                    continue
                if config_id == "diag5":
                    if panel == "L96":
                        diag = mine[(mine["partition"] == "legacy_main") & (mine["kind"] == "root")
                                    & mine["replicate"].between(20, 24)].sort_values("replicate")
                    elif panel == "T200":
                        diag = mine[mine["partition"] == "t200_diag"].sort_values("replicate")
                    else:
                        continue
                    if len(diag) != 5:
                        continue
                    members = list(diag["candidate_id"])
                    parts = sorted(set(diag["partition"]))
                else:
                    members = [index.get((a, r)) for r in range(n)]
                    if any(m is None for m in members):
                        continue
                    parts = sorted(set(main.loc[main["candidate_id"].isin(members), "partition"]))
                rows.append({"pool_id": pool_id(item_id, config_id, view, members), "item_id": item_id,
                             "benchmark": bench, "panel": panel, "config_id": config_id, "n": n, "a": a, "R": 1,
                             "B": n * a, "member_ids": members, "view": view, "candidate_order_seed": PASSK_SEED,
                             "source_partitions": parts})
    return pd.DataFrame(rows, columns=POOL_COLUMNS)


def c48_fit_scopes(draws: pd.DataFrame, *, a_slots: int = 16, view: str = "current") -> pd.DataFrame:
    """One class-building scope per C48 item over its fitting draws only (A slots 0-15 + every deep replicate
    from legacy/e1_fill/c48_extra), in frozen (allowance, replicate) order. ``config_id = "c48fit"``."""
    rows = []
    for item_id in _panel_items(draws, "C48"):
        mine = draws[(draws["item_id"] == item_id) & draws["partition"].isin(_PANEL_PARTITIONS["C48"])]
        mine = mine[(mine["allowance_a"] > 1) | (mine["replicate"] < a_slots)]
        mine = mine[mine["allowance_a"] != 25].sort_values(["allowance_a", "replicate"])
        members = list(mine["candidate_id"])
        if not members:
            continue
        rows.append({"pool_id": pool_id(item_id, "c48fit", view, members), "item_id": item_id,
                     "benchmark": mine["benchmark"].iloc[0], "panel": "C48", "config_id": "c48fit",
                     "n": len(members), "a": 0, "R": 1, "B": int(mine["allowance_a"].sum()), "member_ids": members,
                     "view": view, "candidate_order_seed": PASSK_SEED,
                     "source_partitions": sorted(set(mine["partition"]))})
    return pd.DataFrame(rows, columns=POOL_COLUMNS)
