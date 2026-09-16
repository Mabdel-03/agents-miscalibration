"""The spec-7.3 contrast table and its standardised delta (plan §2.6 output 5).

One row per ``(study, task, model, budget, protocol)`` stratum, carrying the root distribution,
the round outcomes as lo/hi pairs, the transitions, validity, selected accuracy and realised
resources -- and, for **ARD36 8B only**, the PEER vs PRIVATE standardised difference.

Only ARD36 gets a contrast row, and that restriction is the finding: it is the only legacy run
with both arms of a communication contrast under one contract.  The atlas has no such axis and
Run0's serving regime changed mid-run, so their rows carry descriptives and no delta.  A
standardised difference computed across runs that did not share a protocol would be the exact
mistake spec 7.3 exists to prevent.

The standardisation weights come from the **pre-outcome** recruitment frame -- the round-0 root
distribution -- so they cannot have been chosen by the outcome, and ``overlap_retained`` and the
effective sample size are reported with the delta so a reader can see how much of the frame the
supported strata cover.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from agents_scaling.nextdyn import outcomes, tables
from agents_scaling.nextdyn.e0 import ard36, registry as R

#: Minimum items per arm for a stratum to be *supported* (spec 7.3).
MIN_PER_ARM: Final[int] = 5

#: The strata axes of spec 7.3, as far as the legacy tables can express them.
STRATUM_AXES: Final[tuple[str, ...]] = ("c0_bin", "largest_wrong_class", "family")


def ard36_contrast_rows(sources: R.LegacySources) -> list[dict[str, Any]]:
    """Per-stratum PEER/PRIVATE descriptives and the standardised delta on ARD36 8B."""
    per_team, episodes = ard36._team_states(sources)
    if not per_team:
        return []
    frame: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    cells: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for (_cell_id, episode_id), by_round in per_team.items():
        meta = episodes.get(episode_id, {})
        if (str(meta.get("cohort")) != "natural" or str(meta.get("split")) != "test"
                or str(meta.get("checkpoint")) != "8B" or meta.get("fallback")):
            continue
        arm = str(meta.get("protocol"))
        if arm not in ard36.A1_ARMS or 0 not in by_round:
            continue
        root = by_round[0]
        stratum = (int(root["c_count"]),
                   _largest_wrong(root),
                   str(meta.get("family") or ""))
        frame[stratum].add(str(meta.get("item_id")))
        terminal = by_round.get(ard36.ARD36_ROUNDS)
        if terminal is None:
            continue
        cells[stratum][arm].append({"item_id": str(meta.get("item_id")), "root": root,
                                    "terminal": terminal})
    total = sum(len(items) for items in frame.values()) or 1
    rows: list[dict[str, Any]] = []
    delta = 0.0
    supported_weight = 0.0
    weights_used: list[float] = []
    for stratum in sorted(cells, key=lambda s: [str(x) for x in s]):
        arms = cells[stratum]
        weight = len(frame.get(stratum, ())) / total
        n_t, n_c = len(arms.get(ard36.A1_ARMS[0], [])), len(arms.get(ard36.A1_ARMS[1], []))
        supported = n_t >= MIN_PER_ARM and n_c >= MIN_PER_ARM
        contribution = None
        if supported:
            contribution = float(
                np.mean([m["terminal"]["q_lo"] for m in arms[ard36.A1_ARMS[0]]])
                - np.mean([m["terminal"]["q_lo"] for m in arms[ard36.A1_ARMS[1]]]))
            delta += weight * contribution
            supported_weight += weight
            weights_used.append(weight)
        row: dict[str, Any] = {
            "study": "ard36", "model": "8B", "budget": "1024/256",
            "stratum_json": json.dumps(dict(zip(STRATUM_AXES, stratum)), default=str),
            "family": stratum[2], "weight": weight, "supported": supported,
            "n_treat": n_t, "n_control": n_c, "delta_contribution": contribution,
        }
        for arm in ard36.A1_ARMS:
            members = arms.get(arm, [])
            prefix = arm.lower()
            row[f"{prefix}_n"] = len(members)
            for name, key in (("root_c", "c_count"), ("root_u", "u_count")):
                row[f"{prefix}_{name}"] = (float(np.mean([m["root"][key] for m in members]))
                                           if members else None)
            for bound in ("lo", "hi"):
                row[f"{prefix}_q_{bound}"] = (
                    float(np.mean([m["terminal"][f"q_{bound}"] for m in members])) if members else None)
                row[f"{prefix}_m_{bound}"] = (
                    float(np.mean([m["terminal"][f"m_{bound}"] for m in members])) if members else None)
            row[f"{prefix}_wrong_unanimity"] = (
                float(np.mean([bool(m["terminal"]["wrong_unanimity"]) for m in members]))
                if members else None)
            row[f"{prefix}_validity"] = (
                float(np.mean([m["terminal"]["n_valid"] / m["terminal"]["n_agents"] for m in members]))
                if members else None)
        rows.append(row)
    ess = (sum(weights_used) ** 2 / sum(w * w for w in weights_used)) if weights_used else 0.0
    rows.append({
        "study": "ard36", "model": "8B", "budget": "1024/256",
        "stratum_json": json.dumps({"__summary__": True}),
        "family": "__all__", "weight": supported_weight, "supported": bool(weights_used),
        "n_treat": sum(r.get("n_treat") or 0 for r in rows),
        "n_control": sum(r.get("n_control") or 0 for r in rows),
        "delta_contribution": (delta / supported_weight) if supported_weight else None,
        "delta_std": (delta / supported_weight) if supported_weight else None,
        "overlap_retained": supported_weight, "ess": ess,
        "n_strata": len(rows), "n_strata_supported": len(weights_used),
    })
    return rows


def _largest_wrong(root: Mapping[str, Any]) -> int:
    """``min(3, size of the largest wrong class)`` at the root (spec 7.3 bins 0/1/2/>=3)."""
    share = root.get("wrong_class_max_share")
    if share is None:
        return 0
    return int(min(3, round(float(share) * int(root.get("w_count") or 0))))


def contrast_table(sources: R.LegacySources) -> list[dict[str, Any]]:
    """Every legacy study's strata rows; only ARD36 carries a treatment delta."""
    rows = ard36_contrast_rows(sources)
    for study, note in (("study_v4", "no communication axis exists in this run"),
                        ("full_sweep_v1", "mixed serving regime; supplementary evidence only")):
        rows.append({"study": study, "model": "-", "budget": "-",
                     "stratum_json": json.dumps({"__not_applicable__": note}),
                     "family": "-", "weight": None, "supported": False,
                     "n_treat": 0, "n_control": 0, "delta_contribution": None})
    return rows


def write(out_dir: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "contrast_table.parquet"
    columns = sorted({key for row in rows for key in row})
    tables.write_parquet_rows(path, [dict(r) for r in rows], columns)
    return path


def overlap_inventory(sources: R.LegacySources, *, new_items: Mapping[str, str] | None = None,
                      ) -> list[dict[str, Any]]:
    """``(source_item_id, legacy_study)`` overlap rows with an explicit ``match_method``.

    ``new_items`` maps this study's ``source_item_id`` to its normalized text hash; it is empty
    until ``inventory`` has run, and the rows then say ``match_method='none'`` rather than
    claiming that nothing overlaps.  The atlas overlaps the new frames only on BigCodeBench
    (identity by ``task_id``), which is recorded as a note rather than inferred from text.
    """
    items = R.resolve(sources, "ard36.items", hash_files=False)
    rows: list[dict[str, Any]] = []
    if not items.exists:
        return [{"source_item_id": None, "legacy_study": "ard36", "family": None,
                 "legacy_id": None, "overlap_status": R.BLOCKED_MISSING_SOURCE,
                 "match_method": "none", "note": f"{items.path} is absent"}]
    try:
        import pyarrow.parquet as pq

        legacy = pq.read_table(items.path).to_pylist()
    except Exception as exc:
        return [{"source_item_id": None, "legacy_study": "ard36", "family": None,
                 "legacy_id": None, "overlap_status": R.BLOCKED_MISSING_SOURCE,
                 "match_method": "none", "note": f"{items.path} unreadable ({exc})"}]
    lookup = dict(new_items or {})
    for row in legacy:
        legacy_id = str(row.get("item_id") or row.get("source_id") or "")
        matched = legacy_id in lookup
        rows.append({
            "source_item_id": lookup.get(legacy_id),
            "legacy_study": "ard36", "family": str(row.get("family") or ""),
            "legacy_id": legacy_id,
            "overlap_status": "overlapping" if matched else "not_matched",
            "match_method": "source_id" if matched else "none",
            "note": ("" if lookup else
                     "no new-item inventory was supplied; overlap is unevaluated, not absent"),
        })
    return rows


def write_overlap(out_dir: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "overlap_inventory.parquet"
    columns = sorted({key for row in rows for key in row})
    tables.write_parquet_rows(path, [dict(r) for r in rows], columns)
    return path


__all__ = ["MIN_PER_ARM", "STRATUM_AXES", "ard36_contrast_rows", "contrast_table",
           "overlap_inventory", "write", "write_overlap"]
