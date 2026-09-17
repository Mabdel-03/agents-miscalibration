"""Cross-check against agent 06's per-item metric tables (``paths.metrics_table("<partition>.<table>")``).

Analysis computes its outcome, bank and transition tables itself (``outcomes``, ``itemmetrics``). When agent 06's
tables exist on disk they are read and compared field by field; this is the narrow join point to agent 06:

* ``<partition>.checkpoints`` - ``policy_id, delivered_lo, delivered_hi, coverage_lo, coverage_hi``
* ``<partition>.bank`` - ``item_id, k, bank_size, known, pass_at_k_lo, pass_at_k_hi``
* ``<partition>.team_transitions`` - ``item_id, family, n, from_step, to_step, fully_known, c_prev, c_next, n_WC,
  n_CW, n_IC, n_CI``

A *known-versus-known* disagreement fails the audit. A value this module knows while agent 06 reports bounds (or the
reverse) is listed, not failed: analysis also reads the coordinator's state backup, so a transport-exhausted node is a
known no-output failure here and a missing candidate there.
"""
from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from agent_allocation.analysis.load import RunData

TABLES = ("checkpoints", "bank", "team_transitions")
TOL = 1e-9


def load_metrics_tables(run: RunData) -> dict[str, dict[str, pd.DataFrame]]:
    """``{partition: {table: frame}}`` for every table agent 06 wrote; empty when none exist."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for partition in run.partitions:
        for table in TABLES:
            path = run.paths.metrics_table(f"{partition}.{table}")
            if not path.is_file():
                continue
            run.input_files.append(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            out.setdefault(partition, {})[table] = pd.DataFrame(rows)
    return out


def _bounds(value: float) -> tuple[int, int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0, 1
    return int(value), int(value)


def compare(run: RunData, outcomes: pd.DataFrame, bank: pd.DataFrame, transitions: pd.DataFrame,
            tables: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    """Known-vs-known violations and bound-level differences per table."""
    violations: list[str] = []
    differences: list[str] = []
    compared = 0
    by_policy = outcomes.set_index("policy_id") if not outcomes.empty else pd.DataFrame()
    for partition, frames in sorted(tables.items()):
        cps = frames.get("checkpoints")
        if cps is not None and not cps.empty:
            for rec in cps.to_dict("records"):
                pid = rec.get("policy_id")
                if pid not in by_policy.index:
                    violations.append(f"{partition}.checkpoints: {pid} is not a planned checkpoint here")
                    continue
                mine = by_policy.loc[pid]
                for label, lo_key, hi_key, value in (("delivered", "delivered_lo", "delivered_hi", mine["y"]),
                                                     ("coverage", "coverage_lo", "coverage_hi", mine["coverage"])):
                    if lo_key not in rec or hi_key not in rec:
                        continue
                    compared += 1
                    theirs = (int(rec[lo_key]), int(rec[hi_key]))
                    ours = _bounds(value)
                    if ours == theirs:
                        continue
                    if ours[0] == ours[1] and theirs[0] == theirs[1]:
                        violations.append(f"{pid}: {label} analysis={ours[0]} agent06={theirs[0]}")
                    else:
                        differences.append(f"{pid}: {label} analysis={ours} agent06={theirs}")
        their_bank = frames.get("bank")
        if their_bank is not None and not their_bank.empty and not bank.empty:
            mine = bank[bank["partition_id"] == partition].set_index(["item_id", "k"])
            for rec in their_bank.to_dict("records"):
                key = (rec.get("item_id"), rec.get("k"))
                if key not in mine.index:
                    continue
                ours = mine.loc[key, "oracle_pass_at_k"]
                if not bool(rec.get("known")) or ours is None or pd.isna(ours):
                    if bool(rec.get("known")) != (ours is not None and not pd.isna(ours)):
                        differences.append(f"{partition}/{key[0]} pass@{key[1]}: known differs")
                    continue
                compared += 1
                if abs(float(ours) - float(rec.get("pass_at_k_lo"))) > TOL:
                    violations.append(f"{partition}/{key[0]} pass@{key[1]}: analysis={ours} "
                                      f"agent06={rec.get('pass_at_k_lo')}")
        their_trans = frames.get("team_transitions")
        if their_trans is not None and not their_trans.empty and not transitions.empty:
            mine = transitions[transitions["partition_id"] == partition].set_index(
                ["item_id", "trajectory", "from_step", "to_step"])
            for rec in their_trans.to_dict("records"):
                family, n = rec.get("family"), rec.get("n")
                label = f"{family}/{n}" if family in ("private_team", "peer") and n is not None else str(family)
                key = (rec.get("item_id"), label, rec.get("from_step"), rec.get("to_step"))
                if key not in mine.index:
                    continue
                row = mine.loc[key]
                if not bool(rec.get("fully_known")) or not bool(row["known"]):
                    continue
                compared += 1
                pairs = (("correct_prev", "c_prev"), ("correct_next", "c_next"), ("W_to_C", "n_WC"),
                         ("C_to_W", "n_CW"), ("I_to_C", "n_IC"), ("C_to_I", "n_CI"))
                for ours_key, their_key in pairs:
                    if their_key in rec and int(row[ours_key]) != int(rec[their_key]):
                        violations.append(f"{partition}/{key[0]}/{label} {key[2]}->{key[3]}: {ours_key} "
                                          f"analysis={row[ours_key]} agent06={rec[their_key]}")
    return {"compared": compared, "violations": violations, "differences": differences,
            "partitions": sorted(tables)}
