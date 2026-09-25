"""Plurality readout over a pool's gold-blind answer classes (contract §5.5).

Classes: ``selection.plurality.build_classes`` over the pool's voting members in frozen member order (replicate
index), MATH equivalence served by the pair cache (``PairCache.equivalence``; a missing pair raises
``MissingPair`` -- the cache is filled first, never inline), GPQA by option identity.

Winner: the modal class. Ties:

* ``mode="passk"`` (new-run policy): the tied class with the smallest
  ``sha256(json(["passk_cont_v1/tie", 20260925, pool_id, class_id]))`` -- a seeded uniform draw per pool;
* ``mode="legacy"``: ``selection.plurality.class_tie_key(20260917, item_id, class_id)`` (forced_v2 replay only).

The delivered answer is the first member of the winning class. A single-member pool delivers its member whatever
its parse (legacy ``SINGLE_CANDIDATE``); absent members (no candidate) never vote (07-R12); a pool with no vote
abstains (delivered None, counted not-correct operationally). The analytic tie-averaged accuracy averages the
correctness of each tied class's first member. Nothing here reads labels: correctness is passed in by the caller.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import pandas as pd

from agent_allocation.contracts import OutputKind
from agent_allocation.selection.keys import VoteKey
from agent_allocation.selection.plurality import AnswerClass, build_classes, class_tie_key

__all__ = ["LEGACY_SEED", "PASSK_SEED", "Readout", "kind_of", "passk_tie_key", "plurality", "pool_classes",
           "readout_pools", "tie_averaged"]

PASSK_SEED = 20260925
LEGACY_SEED = 20260917


def kind_of(benchmark: str) -> OutputKind:
    return OutputKind.OPTION if benchmark == "gpqa_diamond" else OutputKind.MATH


def passk_tie_key(pool_id: str, class_id: str) -> str:
    blob = json.dumps(["passk_cont_v1/tie", PASSK_SEED, pool_id, class_id], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Readout:
    pool_id: str
    item_id: str
    status: str                              # single | selected | abstained
    delivered_id: str | None
    winner_class: str | None
    support: int
    n_votes: int
    classes: tuple[AnswerClass, ...]
    tied_class_ids: tuple[str, ...]          # tie order, winner first
    tied_first_ids: tuple[str, ...]          # first member of each tied class, same order
    equivalence_unknown: int


def pool_classes(votes: Sequence[tuple[str, VoteKey]], kind: OutputKind,
                 equivalence: Callable[[str, str], bool | None] | None) -> tuple[tuple[AnswerClass, ...], int]:
    classes, _calls, unknown = build_classes(votes, kind=kind,
                                             math_equivalent=equivalence if kind is OutputKind.MATH else None)
    return classes, unknown


def plurality(pool_id: str, item_id: str, member_ids: Sequence[str], votes: Sequence[tuple[str, VoteKey]],
              kind: OutputKind, *, present: set[str] | frozenset[str],
              equivalence: Callable[[str, str], bool | None] | None = None, mode: str = "passk") -> Readout:
    if mode not in ("passk", "legacy"):
        raise ValueError(f"unknown tie mode {mode!r}")
    if len(set(member_ids)) != len(member_ids):
        raise ValueError(f"{pool_id}: duplicate members")
    order = {cid: i for i, cid in enumerate(member_ids)}
    votes = sorted(votes, key=lambda v: order[v[0]])               # frozen member order, never completion order
    if len(member_ids) == 1:
        only = member_ids[0]
        if only not in present:
            return Readout(pool_id, item_id, "abstained", None, None, 0, 0, (), (), (), 0)
        classes, unknown = pool_classes(votes, kind, equivalence) if votes else ((), 0)
        winner = classes[0].class_id if classes else None
        return Readout(pool_id, item_id, "single", only, winner, len(votes), len(votes), classes,
                       (winner,) if winner else (), (only,) if winner else (), unknown)
    classes, unknown = pool_classes(votes, kind, equivalence) if votes else ((), 0)
    if not classes:
        return Readout(pool_id, item_id, "abstained", None, None, 0, 0, (), (), (), unknown)
    top = max(c.support for c in classes)
    if mode == "legacy":
        tied = sorted((c for c in classes if c.support == top),
                      key=lambda c: (class_tie_key(LEGACY_SEED, item_id, c.class_id), c.class_id))
    else:
        tied = sorted((c for c in classes if c.support == top), key=lambda c: (passk_tie_key(pool_id, c.class_id),
                                                                               c.class_id))
    winner = tied[0]
    return Readout(pool_id, item_id, "selected", winner.member_ids[0], winner.class_id, winner.support, len(votes),
                   classes, tuple(c.class_id for c in tied), tuple(c.member_ids[0] for c in tied), unknown)


def tie_averaged(readout: Readout, correct: Mapping[str, bool | None]) -> tuple[float, float]:
    """(lo, hi) analytic tie-averaged delivered correctness; unknown member grades give the bounds."""
    if readout.delivered_id is None:
        return 0.0, 0.0
    if readout.status == "single" or not readout.tied_first_ids:
        value = correct.get(readout.delivered_id)
        return (float(value is True), float(value is not False))
    lo = sum(correct.get(c) is True for c in readout.tied_first_ids) / len(readout.tied_first_ids)
    hi = sum(correct.get(c) is not False for c in readout.tied_first_ids) / len(readout.tied_first_ids)
    return lo, hi


def _tri(value) -> bool | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return bool(value)


def readout_pools(pools: pd.DataFrame, draws: pd.DataFrame, *, cache=None, mode: str = "passk",
                  with_correct: bool = True) -> pd.DataFrame:
    """One row per pool: delivered id, classes, and (when labelled rows are present) delivered correctness."""
    by_id = draws.set_index("candidate_id")
    present = set(draws.loc[~draws["absent"].astype(bool), "candidate_id"])
    correct = ({cid: _tri(v) for cid, v in zip(draws["candidate_id"], draws["correct"])}
               if with_correct and "correct" in draws else {})
    rows = []
    for pool in pools.itertuples(index=False):
        members = list(pool.member_ids)
        sub = by_id.loc[members]
        votes = [(cid, VoteKey(exact=e, text=t)) for cid, e, t in zip(members, sub["vote_exact_key"],
                                                                       sub["vote_text"])
                 if cid in present and isinstance(e, str) and isinstance(t, str)]
        kind = kind_of(pool.benchmark)
        equivalence = None if cache is None or kind is not OutputKind.MATH else cache.equivalence(pool.item_id)
        r = plurality(pool.pool_id, pool.item_id, members, votes, kind, present=present, equivalence=equivalence,
                      mode=mode)
        lo, hi = tie_averaged(r, correct) if correct else (float("nan"), float("nan"))
        delivered = correct.get(r.delivered_id) if (correct and r.delivered_id) else None
        if correct and r.delivered_id is None:
            delivered = False
        member_correct = [correct.get(c) if c in present else False for c in members] if correct else []
        rows.append({
            "pool_id": r.pool_id, "item_id": r.item_id, "benchmark": pool.benchmark, "panel": pool.panel,
            "config_id": pool.config_id, "n": pool.n, "a": pool.a, "mode": mode, "status": r.status,
            "delivered_id": r.delivered_id, "winner_class": r.winner_class, "support": r.support,
            "n_votes": r.n_votes, "n_classes": len(r.classes), "n_tied": len(r.tied_class_ids),
            "class_sizes": [c.support for c in r.classes],
            "class_first_ids": [c.member_ids[0] for c in r.classes],
            "equivalence_unknown": r.equivalence_unknown,
            "delivered_correct": delivered,
            "tie_avg_lo": lo, "tie_avg_hi": hi,
            "n_member_correct": sum(v is True for v in member_correct),
            "n_member_unknown": sum(v is None for v in member_correct),
        })
    return pd.DataFrame(rows)
