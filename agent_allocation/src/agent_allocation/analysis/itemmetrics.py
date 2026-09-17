"""Per-item metrics that analysis needs from candidate grades: oracle pass@k on the independent bank, prefix
coverage and deployable readouts, and round-to-round transitions.

Join point: :func:`load_item_metrics` is the single function to repoint at agent 06's per-item metric tables
(``agent_allocation.metrics``) once they exist; until then it computes the same quantities locally from the
stored grades. Semantics (MASTER_SPEC §5.1, §12, §15.14):

* oracle ``pass@k`` uses the finite-bank estimator ``1 - C(n-c, k) / C(n, k)`` on the full independent bank of the
  ``sample/K`` checkpoint (exact integer arithmetic) and only for items whose whole bank is known;
* ``prefix_coverage@k`` and the deployable readout (``vote@k`` for text, ``public_select@k`` for MBPP+) use the
  same fixed prefix ``root[0:k]``; the same-pool selection gap is their difference;
* transitions keep invalid outputs distinct from valid wrong answers.
"""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Any

import pandas as pd

from agent_allocation.analysis.load import RunData
from agent_allocation.analysis.names import CheckpointNameError, parse_name
from agent_allocation.analysis.outcomes import grade_value

CODE_BENCHMARKS = frozenset({"mbpp_plus"})


def pass_at_k(n: int, c: int, k: int) -> float:
    """Exact finite-bank estimator; defined edges: k > n raises, c = 0 -> 0, n - c < k -> 1."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"pass_at_k undefined for n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return float(1 - Fraction(math.comb(n - c, k), math.comb(n, k)))


def readout_label(benchmark: str) -> str:
    return "public_select@k" if benchmark in CODE_BENCHMARKS else "vote@k"


def _candidate_state(run: RunData, node_id: str) -> str:
    if run.terminal_without_output(node_id) is not None:
        return "invalid"                      # no output: an operational failure, never a semantic answer
    if node_id not in run.parsed:
        return "unknown"
    value, state = grade_value(run.grades.get(node_id))
    if state == "correct":
        return "correct"
    if state == "wrong":
        return "wrong"
    if state == "op_invalid_output":
        return "invalid"
    return "unknown"


def sample_bank_metrics(run: RunData, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per (partition, item, k): oracle pass@k (bank), prefix coverage, deployable readout, same-pool gap."""
    rows: list[dict[str, Any]] = []
    current = outcomes[(outcomes["view"] == "current") & (outcomes["family"] == "sample")]
    if current.empty:
        return pd.DataFrame(rows)
    policies = {cp.policy_id: cp for part in run.policies.values() for cp in part}
    for (partition, item), group in current.groupby(["partition_id", "item_id"], sort=True):
        ks = {parse_name(n).params[0]: r for n, r in zip(group["name"], group.to_dict("records"))}
        K = max(ks)
        bank = policies[ks[K]["policy_id"]].candidate_ids
        states = [_candidate_state(run, nid) for nid in bank]
        bank_known = all(s != "unknown" for s in states)
        c = sum(1 for s in states if s == "correct")
        for k in sorted(ks):
            rec = ks[k]
            rows.append({
                "partition_id": partition, "item_id": item, "benchmark": rec["benchmark"], "stratum": rec["stratum"],
                "k": k, "bank_size": len(bank), "bank_correct": c if bank_known else None,
                "oracle_pass_at_k": pass_at_k(len(bank), c, k) if bank_known and k <= len(bank) else None,
                "prefix_coverage": rec["coverage"], "readout": rec["y"],
                "readout_label": readout_label(rec["benchmark"]),
                "selection_gap_same_pool": (rec["coverage"] - rec["y"]
                                            if not (pd.isna(rec["coverage"]) or pd.isna(rec["y"])) else None),
                "mean_draw_correct": (c / len(bank)) if bank_known else None,
            })
    return pd.DataFrame(rows)


def transitions(run: RunData, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per (partition, item, trajectory, step): slot-aligned state transitions between consecutive checkpoints."""
    policies = {cp.policy_id: cp for part in run.policies.values() for cp in part}
    current = outcomes[outcomes["view"] == "current"]
    rows: list[dict[str, Any]] = []
    by_traj: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for rec in current.to_dict("records"):
        try:
            parsed = parse_name(rec["name"])
        except CheckpointNameError:
            continue
        if parsed.family not in ("private_single", "private_team", "peer", "cp") or parsed.trajectory is None:
            continue
        key, step = parsed.trajectory
        by_traj[(rec["partition_id"], rec["item_id"], key)][step] = rec
    for (partition, item, key), steps in sorted(by_traj.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ordered = sorted(steps)
        for prev_step, step in zip(ordered, ordered[1:]):
            prev_rec, rec = steps[prev_step], steps[step]
            prev_pool = policies[prev_rec["policy_id"]].candidate_ids
            pool = policies[rec["policy_id"]].candidate_ids
            row = {"partition_id": partition, "item_id": item, "benchmark": rec["benchmark"],
                   "stratum": rec["stratum"], "trajectory": "/".join(map(str, key)), "from_step": prev_step,
                   "to_step": step, "slots_prev": len(prev_pool), "slots": len(pool)}
            counts = defaultdict(int)
            known = len(prev_pool) == len(pool)
            if known:
                for a, b in zip(prev_pool, pool):
                    sa, sb = _candidate_state(run, a), _candidate_state(run, b)
                    if "unknown" in (sa, sb):
                        known = False
                    counts[f"{sa[0].upper()}_to_{sb[0].upper()}"] += 1
            prev_correct = sum(1 for n in prev_pool if _candidate_state(run, n) == "correct")
            cur_correct = sum(1 for n in pool if _candidate_state(run, n) == "correct")
            row.update({
                "known": known, "correct_prev": prev_correct if known else None,
                "correct_next": cur_correct if known else None,
                "W_to_C": counts["W_to_C"] if known else None, "C_to_W": counts["C_to_W"] if known else None,
                "I_to_C": counts["I_to_C"] if known else None, "C_to_I": counts["C_to_I"] if known else None,
                "W_to_I": counts["W_to_I"] if known else None, "I_to_W": counts["I_to_W"] if known else None,
                "delivered_prev": prev_rec["y"], "delivered_next": rec["y"],
                "last_correct_lost": (prev_correct > 0 and cur_correct == 0) if known else None,
                "recovered": (prev_correct == 0 and cur_correct > 0) if known else None,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def load_item_metrics(run: RunData, outcomes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The one join point for per-item metrics (repoint at agent 06's tables when available)."""
    return {"sample_bank": sample_bank_metrics(run, outcomes), "transitions": transitions(run, outcomes)}
