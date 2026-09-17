"""Completeness matrices: planned / started / generated / selected / graded / unknown for every
partition x benchmark x condition, plus node-level generation and grading coverage.

Unrun administrative jobs stay in their own columns (``not_started``, ``in_progress``); they are never counted
as wrong answers, and operational failures stay distinguishable from unknown grades.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from agent_allocation.analysis.outcomes import KNOWN_STATES, UNKNOWN_STATES

CONDITION_KEYS = ("partition_id", "benchmark", "name", "view")


def completeness_matrix(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(columns=[*CONDITION_KEYS, "planned"])
    rows: list[dict[str, Any]] = []
    for keys, group in outcomes.groupby(list(CONDITION_KEYS), sort=True, dropna=False):
        states = group["delivered_state"].value_counts().to_dict()
        started = int((group["n_required_terminal"] > 0).sum())
        generated = int((group["n_required_terminal"] == group["n_required"]).sum())
        row = dict(zip(CONDITION_KEYS, keys))
        row.update({
            "planned": int(len(group)),
            "started": started,
            "generated": generated,
            "selected_sealed": int((group["selection_status"].notna() & group["sealed"]).sum()),
            "graded_known": int(sum(states.get(s, 0) for s in KNOWN_STATES)),
            "correct": int(states.get("correct", 0)),
            "wrong": int(states.get("wrong", 0)),
            "op_failure": int(sum(states.get(s, 0) for s in KNOWN_STATES if s not in ("correct", "wrong"))),
        })
        for state in UNKNOWN_STATES:
            row[state] = int(states.get(state, 0))
        row["complete"] = row["graded_known"] == row["planned"]
        rows.append(row)
    return pd.DataFrame(rows)


def node_completeness(candidates: pd.DataFrame) -> pd.DataFrame:
    keys = ["partition_id", "benchmark", "kind"]
    if candidates.empty:
        return pd.DataFrame(columns=[*keys, "planned"])
    rows: list[dict[str, Any]] = []
    for group_keys, group in candidates.groupby(keys, sort=True, dropna=False):
        gstates = group["grade_state"].value_counts().to_dict()
        rows.append({
            **dict(zip(keys, group_keys)),
            "planned": int(len(group)),
            "terminal": int(group["terminal"].sum()),
            "transport_not_ok": int((group["terminal"] & (group["transport_status"] != "ok")).sum()),
            "graded_correct": int(gstates.get("correct", 0)),
            "graded_wrong": int(gstates.get("wrong", 0)),
            "invalid_output": int(gstates.get("op_invalid_output", 0)),
            "evaluator_error": int(gstates.get("unknown_evaluator_error", 0) + gstates.get("graded_unknown", 0)),
            "pending_grade": int(gstates.get("pending_grade", 0)),
            "not_generated": int(gstates.get("not_generated", 0)),
        })
    return pd.DataFrame(rows)


def collection_status(outcomes: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    """Whole-run flags used to label reports interim vs final."""
    nodes_planned = int(len(candidates))
    nodes_terminal = int(candidates["terminal"].sum()) if nodes_planned else 0
    rows = int(len(outcomes))
    known = int(outcomes["y"].notna().sum()) if rows else 0
    unfinished = []
    if rows:
        for (partition, benchmark), group in outcomes.groupby(["partition_id", "benchmark"], sort=True,
                                                               dropna=False):
            missing = int(group["y"].isna().sum())
            if missing:
                unfinished.append({"partition_id": partition, "benchmark": benchmark, "rows_unknown": missing,
                                   "rows_planned": int(len(group))})
    return {
        "nodes_planned": nodes_planned,
        "nodes_terminal": nodes_terminal,
        "checkpoint_rows": rows,
        "checkpoint_rows_known": known,
        "complete": nodes_planned > 0 and nodes_terminal == nodes_planned and rows > 0 and known == rows,
        "unfinished": unfinished,
    }
