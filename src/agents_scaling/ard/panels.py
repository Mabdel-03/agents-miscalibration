"""The canonical analysis panels: named row filters whose sizes are *measured*, not assumed.

Half the 36-hour study's own sweep claims were retracted because a statistic was computed on
one row set and read as though it had been computed on another.  Three failure modes did all
the damage, and each is a panel this module names:

* **The team filter.**  ``minority.build_frame`` keeps an ``(episode, round)`` group only when
  it holds exactly five rows.  But a ROOTS bank and the PRIVATE episode of the same item share
  an ``episode_id``, so a PRIVATE round-0 group holds 16 bank draws *plus* its five team
  members and is silently dropped.  That is 480 teams / 2,400 rows -- every PRIVATE round-0
  team in the run.  :func:`panel_p0` separates the lanes by ``protocol``/``phase`` instead of
  by group size and keeps them.
* **Round-0 replication.**  A round-0 decision is generated once and aliased into every arm
  that starts from it.  On the team rows it appears 3, 5, 7 or 9 times (never a constant), so
  round-0 rows carry a mean multiplicity of **5.867**; ``minority_frame`` -- missing PRIVATE --
  shows **4.867**.  Any per-decision rate that pools rounds without deduplicating is weighted
  by how many protocol arms happened to branch off each root.  :func:`panel_p0d` is the
  deduplicated view.
* **Conditioning on a quantity that is only sometimes defined.**  ``n_correct`` is a property
  of a team, and 2,023 of 11,552 teams have at least one ungradeable member.  Every
  ``n_correct``-conditioned statistic therefore belongs on :func:`panel_p2`, not on P0 or P1.

Each panel exposes ``keys`` (its membership set, in its own unit), a ``row_filter`` over the
decision rows :func:`load_rows` builds, and a ``census`` carrying ``size``, ``universe``,
``coverage_rate`` and the panel-specific counts.  ``expected``/``matches_expected`` in the
census compare against :data:`V1_PANEL_SIZES`, which is a *transcript of the frozen v1 tables*,
not a specification: if the data moves, the panel reports the mismatch rather than hiding it.

**Census measured on the frozen v1 tables** (``ard36/tables/``, run 2026-09-12;
``python -m agents_scaling.ard.panels --tables <run>/tables --check`` exits 0 on them)::

    P0  team rows            57,760 / 67,104 decision rows      coverage 0.860753
        11,552 teams, every one exactly 5 distinct agents, every row labelled.
        Excluded: 8,064 ROOTS bank draws + 1,280 receiver probes.
        480 teams rescued that a size-5 filter applied before the lane split loses.
        rows by round 14,080 / 14,080 / 14,800 / 14,800
    P0d distinct decisions   46,080 / 57,760 P0 rows             coverage 0.797784
        11,680 rows are aliased copies.  round 0: 14,080 rows -> 2,400 decisions (x5.866667);
        rounds 1-3: x1.0 exactly.  multiplicity {1: 43,680, 3: 320, 5: 880, 7: 1,040, 9: 160}
    P1  graded rows          52,151 / 57,760 P0 rows             coverage 0.902891
        correct 41,563 / incorrect 10,588 / ambiguous 5,606 / timeout 3; 41,583 distinct
    P2  class-complete teams  9,529 / 11,552 teams               coverage 0.824879
        47,645 rows; by round 2,275 / 2,320 / 2,457 / 2,477 of 2,816 / 2,816 / 2,960 / 2,960
    P3  common-item 8B           240 / 240 test items            coverage 1.000000
        7 arms x 240 items x 4 rounds = 6,720 teams, all present (cell completeness 1.0);
        33,600 rows, 30,400 graded (0.904762)
    P4  shared-32B                96 / 96 items                  coverage 1.000000
        60 test + 24 train + 12 validation; 3 x 96 x 4 = 1,152 teams, all present;
        5,760 rows, 5,219 graded (0.906076)
    P5  budget-matched            32 / 32 items                  coverage 1.000000
        2 protocols x 2 checkpoints x 2 allowances x 3 rounds x 32 = 768 teams, all present;
        3,840 rows (480 per allowance cell), 3,510 graded (0.914062).
        round 0: all 320 high-budget decisions are T1024 and shared with the T1024 arms.
    P6  fork pairs            1,440 / 1,440 fork rows            coverage 1.000000
        0 missing children, 0 missing peers, 0 ambiguous joins; 2,880 decision rows;
        1,267 pairs graded on both sides (0.879861); 144 items, rounds 2 and 3, 720 each
    P7  receiver non-degen.      33 / 64 probes                  coverage 0.515625
        20 draws each; distinct classes {1: 31, 2: 26, 3: 6, 4: 1}; 660 rows, 0 graded
    P8  scale-matched            96 / 384 natural items          coverage 0.250000
        3 protocols x 96 x 2 checkpoints x 4 rounds = 2,304 teams, all present;
        11,520 rows (5,760 per checkpoint), 10,276 graded (0.892014);
        60 of the 96 are also in P3

    dynamic_range -- 240 8B test items, 16 draws each:
        all-correct 123, all-wrong 34, non-degenerate 83 (0.345833)

**Where the data contradicted the brief.**

* **P8 is 96 items, not 240 and not 60 -- and the number depends on a choice the brief did not
  make.**  32B ran only three protocols (BALANCED, PEER, PRIVATE) on 96 items.  Those 96 are a
  subset of the 8B coverage for the *same three* protocols, so the panel a scale contrast can
  actually use is **96 items x 3 protocols x 2 checkpoints**.  If the panel is instead required
  to be P3 ∩ P4 -- items that additionally carry all seven 8B arms -- it collapses to **60**,
  because the 8B-only arms (GATE, RANDOM_PROTECT) ran on the test split alone while 36 of the
  32B items are train/validation.  This module takes the 96 and reports the 60 beside it as
  ``items_also_in_p3``.  A scale analysis is therefore possible, but only on three protocols,
  and 37.5% of its items are not in the 8B common-item panel.
* **P5's "rounds 1-3 only" is forced, not a convention.**  The high-budget arms' round-0
  decisions are byte-identical to the T1024 arms': the same 320 ``decision_id``s, carrying
  ``budget_tokens = 1024`` and ``aliased = True``.  Round 0 cannot separate the allowances
  because it is the same generation.
* **"All five graded" and "all five hold a canonical class" are different teams.**  9,529 teams
  are fully graded; 9,894 have five non-empty ``answer_class`` values.  503 P0 rows are graded
  with an empty class and 2,101 carry a class but no usable grade.  P2 follows the brief and
  means *graded*; ``teams_all_classed`` is reported next to it so the two are never conflated.
* **8 bank rows carry ``n_draws = 32``** (the D-11 aliased-root duplication).  All 8 are 8B
  *train* items, so :func:`dynamic_range` -- which reads the 240 8B test banks, every one of
  them at ``n_draws = 16`` -- is unaffected.  The brief's 123 / 34 / 83 reproduce exactly.

Everything else the brief stated reproduced: 240 common 8B items, 96 x 3 at 32B, 32
budget-matched items, 1,440 fork pairs, 31 degenerate receiver probes, and 123 / 34 / 83.

**Not verified here.**  Whether the 8B and 32B item sets were *drawn* to be nested is a claim
about the sampler, not about these tables; this module only reports that the 96 32B items are
contained in the 8B coverage of the three shared protocols.  Nothing here checks that a fork
pair's two branches share a byte-exact prefix -- it trusts ``forks.shared_prefix_hash``, whose
byte-exactness is asserted by ``tests/ard/test_wp2_forks.py`` rather than by the compacted
tables, and ``forks.parent_decision_ids`` is an all-null column in v1, so the parent side of a
pair is recovered through ``decisions.parent_state_id`` and never through the fork row itself.

    PYTHONPATH=src python -m agents_scaling.ard.panels --tables <run>/tables [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.ard.contract import N_AGENTS, ROOT_DRAWS, ROUNDS

DECISIONS_TABLE: Final[str] = "decisions.parquet"
EPISODES_TABLE: Final[str] = "episodes.parquet"
LABELS_TABLE: Final[str] = "labels.parquet"
FORKS_TABLE: Final[str] = "forks.parquet"
BANKS_TABLE: Final[str] = "bank_summary.parquet"

#: The two lanes that are not teams.  A ROOTS bank draw carries ``protocol = "NONE"`` and a
#: RECEIVER probe carries ``phase = "receiver"``; both are excluded by name rather than by
#: group size, because a bank and a team can share an ``episode_id`` (see the module docstring).
BANK_PROTOCOL: Final[str] = "NONE"
RECEIVER_PHASE: Final[str] = "receiver"

#: The fork lane's cohort.  A fork child is a counterfactual, never a member of a natural team,
#: so it is excluded when looking for a fork's peer sibling.
FORK_COHORT: Final[str] = "fork"

#: ``evaluator_status`` values that yield an observed outcome.  ``ambiguous``/``timeout`` are
#: missing data, not zeros -- the same rule ``minority.py`` and ``frames.py`` already apply.
GRADED_STATUSES: Final[frozenset[str]] = frozenset({"correct", "incorrect"})

#: The arms each common-item panel intersects over.  These are *asserted* against the arms
#: found in the data, not used to select it: a panel that silently lost an arm would otherwise
#: keep reporting a full-looking intersection.
ARMS_8B: Final[tuple[str, ...]] = (
    "BALANCED", "FRESH", "GATE", "HUB5", "PEER", "PRIVATE", "RANDOM_PROTECT")
ARMS_32B: Final[tuple[str, ...]] = ("BALANCED", "PEER", "PRIVATE")
ARMS_BUDGET: Final[tuple[str, ...]] = ("PEER", "PRIVATE")

CHECKPOINT_SMALL: Final[str] = "8B"
CHECKPOINT_LARGE: Final[str] = "32B"
COHORT_NATURAL: Final[str] = "natural"
COHORT_HIGH_BUDGET: Final[str] = "high_budget"
SPLIT_TEST: Final[str] = "test"

#: Round 0 is the same generation in both allowances (identical ``decision_id``s at
#: ``budget_tokens = 1024``), so it carries no budget contrast and P5 excludes it.
BUDGET_ROUNDS: Final[tuple[int, ...]] = (1, 2, 3)

#: Sizes read off the frozen v1 tables on 2026-09-12.  A transcript, not a target: a mismatch
#: is reported by :func:`census` and made fatal only by ``main --check``.
V1_PANEL_SIZES: Final[Mapping[str, int]] = {
    "P0": 57760, "P0d": 46080, "P1": 52151, "P2": 9529,
    "P3": 240, "P4": 96, "P5": 32, "P6": 1440, "P7": 33, "P8": 96,
}
V1_DYNAMIC_RANGE: Final[Mapping[str, int]] = {
    "items": 240, "all_correct": 123, "all_wrong": 34, "non_degenerate": 83,
}


@dataclass(frozen=True)
class Panel:
    """One analysis panel: what it selects, how big it is, and out of what.

    ``keys`` is the membership set in the panel's own ``unit`` -- decision rows for P0/P1,
    teams for P2, items for P3-P5 and P8, fork ids for P6, probe ids for P7 -- so a caller can
    join on it directly.  ``row_filter`` is the same selection expressed over the decision rows
    of :func:`load_rows`, so *every* panel can be applied to the same frame; for P6 and P7 it
    selects the decision rows the panel's units are built from.
    """

    name: str
    unit: str
    question: str
    keys: frozenset
    universe: int
    census: dict[str, Any] = field(repr=False)
    row_filter: Callable[[Mapping[str, Any]], bool] = field(repr=False)

    @property
    def size(self) -> int:
        return len(self.keys)

    @property
    def coverage_rate(self) -> float:
        """Share of the panel's stated ``universe`` that survives its filter."""
        return (self.size / self.universe) if self.universe else 0.0

    def select(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if self.row_filter(row)]


def _read(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        raise SystemExit(f"missing {path}; compact the run first")
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def load_rows(tables_dir: str | Path) -> list[dict[str, Any]]:
    """One row per decision, joined to its item, split and grade.

    ``episode_id`` is not a key of ``episodes.parquet`` -- 624 ids carry more than one row,
    because a ROOTS bank and the episode it seeds share an id (and 144 fork episodes appear
    once per fork round).  The join is therefore made only after checking that the duplicate
    rows agree on ``item_id``, ``split`` and ``family``; on v1 they all do.  Protocol,
    checkpoint and cohort are taken from the decision itself, where they are unambiguous.
    """
    tables = Path(tables_dir)
    decisions = _read(tables / DECISIONS_TABLE, [
        "episode_id", "round", "agent_id", "decision_id", "protocol", "checkpoint",
        "cohort", "phase", "aliased", "budget_tokens", "answer_class", "parent_state_id"])
    episodes: dict[str, dict[str, Any]] = {}
    for row in _read(tables / EPISODES_TABLE, ["episode_id", "item_id", "split", "family"]):
        seen = episodes.setdefault(str(row["episode_id"]), row)
        disagree = [k for k in ("item_id", "split", "family") if seen[k] != row[k]]
        if disagree:
            raise ValueError(
                f"episodes.parquet rows for {row['episode_id']} disagree on {disagree}; "
                "the decision -> item join is not well defined")
    labels = {str(r["decision_id"]): r
              for r in _read(tables / LABELS_TABLE,
                             ["decision_id", "evaluator_status", "is_correct"])}

    rows: list[dict[str, Any]] = []
    for row in decisions:
        episode = episodes.get(str(row["episode_id"]), {})
        label = labels.get(str(row["decision_id"]))
        status = str(label["evaluator_status"]) if label else ""
        rows.append({
            **row,
            "item_id": str(episode.get("item_id") or ""),
            "split": str(episode.get("split") or ""),
            "family": str(episode.get("family") or ""),
            "evaluator_status": status,
            # Ungradeable stays missing.  Scoring it 0 would move 5,609 P0 rows -- 9.7% -- from
            # "unobserved" to "wrong", and they are concentrated on unparsed answers.
            "correct": int(bool(label["is_correct"])) if status in GRADED_STATUSES else None,
        })
    return rows


def is_graded(row: Mapping[str, Any]) -> bool:
    return str(row.get("evaluator_status") or "") in GRADED_STATUSES


def _team_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["episode_id"]), int(row["round"] or 0)


def _in_team_lane(row: Mapping[str, Any]) -> bool:
    return (str(row.get("protocol") or "") != BANK_PROTOCOL
            and str(row.get("phase") or "") != RECEIVER_PHASE)


# --------------------------------------------------------------------------- #
# P0 / P0d / P1 / P2 -- the row-level panels
# --------------------------------------------------------------------------- #

def panel_p0(rows: Sequence[Mapping[str, Any]]) -> Panel:
    """Team rows: exactly five members per ``(episode, round)``, bank and receiver excluded.

    The five-member property is *checked*, not used to select: the lanes are separated by
    ``protocol``/``phase`` first, and a group that then fails to hold exactly five distinct
    agents is a defect worth raising rather than a row to drop quietly.  On v1 all 11,552
    groups hold five, so nothing is lost -- which is the point: the size-5 test that
    ``minority.py`` applies *before* separating the lanes drops 480 real teams.
    """
    lane = [row for row in rows if _in_team_lane(row)]
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in lane:
        groups[_team_key(row)].append(row)
    malformed = {key: len(members) for key, members in groups.items()
                 if len(members) != N_AGENTS
                 or len({int(m["agent_id"] or 0) for m in members}) != N_AGENTS}
    if malformed:
        raise ValueError(
            f"{len(malformed)} team groups are not {N_AGENTS} distinct agents, e.g. "
            f"{sorted(malformed.items())[:3]}; the team lane filter is wrong")

    keys = frozenset(
        (str(r["episode_id"]), int(r["round"] or 0), int(r["agent_id"] or 0)) for r in lane)
    unlabelled = sum(1 for row in lane if not str(row.get("evaluator_status") or ""))
    raw_sizes = Counter(_team_key(r) for r in rows)
    census = {
        "unit": "decision_row",
        "size": len(lane),
        "universe": len(rows),
        "universe_name": "all decision rows",
        "coverage_rate": round(len(lane) / len(rows), 6) if rows else 0.0,
        "teams": len(groups),
        "rows_excluded_bank": sum(
            1 for r in rows if str(r.get("protocol") or "") == BANK_PROTOCOL
            and str(r.get("phase") or "") != RECEIVER_PHASE),
        "rows_excluded_receiver": sum(
            1 for r in rows if str(r.get("phase") or "") == RECEIVER_PHASE),
        "rows_by_round": {str(k): v for k, v in
                          sorted(Counter(int(r["round"] or 0) for r in lane).items())},
        "rows_without_any_label": unlabelled,
        # Teams whose raw (episode, round) group is not five rows wide because a bank shares
        # its episode_id: exactly the teams a size-5 filter applied before the lane split loses.
        "teams_rescued_from_size_filter": sum(
            1 for key in groups if raw_sizes[key] != N_AGENTS),
    }
    return Panel(
        name="P0", unit="decision_row",
        question="which rows are a five-agent team's own decisions?",
        keys=keys, universe=len(rows), census=census,
        row_filter=_in_team_lane)


def panel_p0d(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """P0 deduplicated on ``decision_id``: one row per generation actually performed.

    Round-0 decisions are generated once and aliased into every arm that branches off them, so
    the same ``decision_id`` appears 3, 5, 7 or 9 times.  The multiplicity is *not* constant --
    it is the number of arms that happened to start from that root, which differs by split
    (GATE and RANDOM_PROTECT ran test only) -- so a pooled per-decision rate computed on P0 is
    silently reweighted toward the test split.  The filter keeps the first occurrence in the
    frame's own order, which is stable for a given table.
    """
    lane = p0.select(rows)
    first_seen: dict[str, tuple[str, int, int]] = {}
    for row in lane:
        first_seen.setdefault(
            str(row["decision_id"]),
            (str(row["episode_id"]), int(row["round"] or 0), int(row["agent_id"] or 0)))
    kept = frozenset(first_seen.values())
    multiplicity = Counter(str(r["decision_id"]) for r in lane)
    round0 = [r for r in lane if int(r["round"] or 0) == 0]
    distinct_round0 = len({str(r["decision_id"]) for r in round0})
    census = {
        "unit": "decision_row",
        "size": len(kept),
        "universe": len(lane),
        "universe_name": "P0 rows",
        "coverage_rate": round(len(kept) / len(lane), 6) if lane else 0.0,
        "duplicate_rows": len(lane) - len(kept),
        "multiplicity_histogram": {str(k): v for k, v in
                                   sorted(Counter(multiplicity.values()).items())},
        "round0_rows": len(round0),
        "round0_distinct_decisions": distinct_round0,
        "round0_replication": round(len(round0) / distinct_round0, 6) if distinct_round0 else 0.0,
        "rounds_1_3_replication": 1.0,
    }
    later = [r for r in lane if int(r["round"] or 0) > 0]
    distinct_later = len({str(r["decision_id"]) for r in later})
    census["rounds_1_3_replication"] = (
        round(len(later) / distinct_later, 6) if distinct_later else 0.0)
    return Panel(
        name="P0d", unit="decision_row",
        question="which rows are distinct generations rather than aliased copies?",
        keys=kept, universe=len(lane), census=census,
        row_filter=lambda row: (
            _in_team_lane(row)
            and (str(row["episode_id"]), int(row["round"] or 0), int(row["agent_id"] or 0))
            in kept))


def panel_p1(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """Graded team rows: ``evaluator_status in {correct, incorrect}``.

    The complement is not "wrong"; it is 5,606 ``ambiguous`` and 3 ``timeout`` rows the
    symbolic grader could not resolve.  Every P0 row carries *some* label on v1 (the 1,408
    unlabelled decisions are all bank draws and receiver probes), so P1's denominator loss is
    entirely gradeability and not missing evaluation.
    """
    lane = p0.select(rows)
    graded = [row for row in lane if is_graded(row)]
    keys = frozenset(
        (str(r["episode_id"]), int(r["round"] or 0), int(r["agent_id"] or 0)) for r in graded)
    census = {
        "unit": "decision_row",
        "size": len(graded),
        "universe": len(lane),
        "universe_name": "P0 rows",
        "coverage_rate": round(len(graded) / len(lane), 6) if lane else 0.0,
        "status_histogram": dict(sorted(Counter(
            str(r.get("evaluator_status") or "<none>") for r in lane).items())),
        "distinct_decisions": len({str(r["decision_id"]) for r in graded}),
        "graded_with_empty_answer_class": sum(
            1 for r in graded if not str(r.get("answer_class") or "")),
        "ungraded_with_answer_class": sum(
            1 for r in lane if not is_graded(r) and str(r.get("answer_class") or "")),
    }
    return Panel(
        name="P1", unit="decision_row",
        question="which team rows carry an observed correctness outcome?",
        keys=keys, universe=len(lane), census=census,
        row_filter=lambda row: _in_team_lane(row) and is_graded(row))


def panel_p2(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """Class-complete teams: all five members graded.  Required for any ``n_correct`` claim.

    ``n_correct`` counts the correct members *of five*.  When a member is ungradeable the
    quantity is censored, not smaller: a team with four graded members can report at most 4,
    and pooling it with complete teams biases every rate conditioned on it.  17.5% of v1 teams
    are incomplete, and the loss is not uniform across rounds, so the per-round counts travel
    with the panel.
    """
    lane = p0.select(rows)
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in lane:
        groups[_team_key(row)].append(row)
    complete = frozenset(key for key, members in groups.items()
                         if all(is_graded(m) for m in members))
    classed = sum(1 for members in groups.values()
                  if all(str(m.get("answer_class") or "") for m in members))
    census = {
        "unit": "team",
        "size": len(complete),
        "universe": len(groups),
        "universe_name": "P0 teams",
        "coverage_rate": round(len(complete) / len(groups), 6) if groups else 0.0,
        "rows": len(complete) * N_AGENTS,
        "teams_by_round": {str(k): v for k, v in
                           sorted(Counter(r for _e, r in complete).items())},
        "universe_by_round": {str(k): v for k, v in
                              sorted(Counter(r for _e, r in groups).items())},
        # Not the same set: grading and canonical-class extraction fail on different rows.
        "teams_all_classed": classed,
    }
    return Panel(
        name="P2", unit="team",
        question="which teams can support an n_correct-conditioned statistic?",
        keys=complete, universe=len(groups), census=census,
        row_filter=lambda row: _in_team_lane(row) and _team_key(row) in complete)


# --------------------------------------------------------------------------- #
# P3 / P4 / P5 / P8 -- the common-item panels
# --------------------------------------------------------------------------- #

def _arm_items(rows: Sequence[Mapping[str, Any]],
               select: Callable[[Mapping[str, Any]], bool],
               arm: Callable[[Mapping[str, Any]], Any]) -> dict[Any, set[str]]:
    out: dict[Any, set[str]] = defaultdict(set)
    for row in rows:
        if select(row):
            out[arm(row)].add(str(row["item_id"]))
    return dict(out)


def _common_item_census(rows: Sequence[Mapping[str, Any]],
                        *, arms: Mapping[Any, set[str]], common: frozenset[str],
                        declared: Sequence[Any], member: Callable[[Mapping[str, Any]], bool],
                        expected_cells: int) -> dict[str, Any]:
    """Shared census for an item-intersection panel, including its cell completeness.

    ``coverage_rate`` is the intersection over the union -- what fraction of the items any arm
    touched are usable by all of them.  ``cell_completeness`` is the orthogonal question: given
    the items, is every (item x arm x round) team actually present?  A panel can score 1.0 on
    one and badly on the other, so both are reported.
    """
    union: set[str] = set().union(*arms.values()) if arms else set()
    panel_rows = [row for row in rows if member(row)]
    observed_cells = len({_team_key(row) for row in panel_rows})
    graded = [row for row in panel_rows if is_graded(row)]
    return {
        "unit": "item",
        "size": len(common),
        "universe": len(union),
        "universe_name": "items any arm ran",
        "coverage_rate": round(len(common) / len(union), 6) if union else 0.0,
        "arms_found": sorted(str(a) for a in arms),
        "arms_declared": sorted(str(a) for a in declared),
        "arms_match_declared": sorted(str(a) for a in arms) == sorted(str(a) for a in declared),
        "items_per_arm": {str(a): len(v) for a, v in
                          sorted(arms.items(), key=lambda kv: str(kv[0]))},
        "decision_rows": len(panel_rows),
        "graded_rows": len(graded),
        "graded_rate": round(len(graded) / len(panel_rows), 6) if panel_rows else 0.0,
        "cells_observed": observed_cells,
        "cells_expected": expected_cells,
        "cell_completeness": round(observed_cells / expected_cells, 6) if expected_cells else 0.0,
    }


def panel_p3(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """Common-item 8B panel: test items every one of the seven 8B protocol arms ran.

    Seven arms is the whole 8B protocol grid; GATE and RANDOM_PROTECT exist on the test split
    only, which is why the common-item panel is a *test* panel and cannot be extended to
    train/validation without dropping two arms.
    """
    lane = p0.select(rows)

    def eligible(row: Mapping[str, Any]) -> bool:
        return (str(row["checkpoint"]) == CHECKPOINT_SMALL
                and str(row["cohort"]) == COHORT_NATURAL
                and str(row["split"]) == SPLIT_TEST)

    arms = _arm_items(lane, eligible, lambda r: str(r["protocol"]))
    common = frozenset(set.intersection(*arms.values())) if arms else frozenset()

    def member(row: Mapping[str, Any]) -> bool:
        return _in_team_lane(row) and eligible(row) and str(row["item_id"]) in common

    census = _common_item_census(
        lane, arms=arms, common=common, declared=ARMS_8B, member=member,
        expected_cells=len(common) * len(ARMS_8B) * len(ROUNDS))
    return Panel(
        name="P3", unit="item",
        question="which test items are comparable across every 8B protocol?",
        keys=common, universe=census["universe"], census=census, row_filter=member)


def panel_p4(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """Shared-32B panel: items run by all three 32B arms, across all three splits.

    32B is a frozen hash-chosen subset within each split (96 = 60 test + 24 train + 12
    validation), so unlike P3 this panel is not test-only -- and that is exactly why it does
    not nest inside P3.
    """
    lane = p0.select(rows)

    def eligible(row: Mapping[str, Any]) -> bool:
        return (str(row["checkpoint"]) == CHECKPOINT_LARGE
                and str(row["cohort"]) == COHORT_NATURAL)

    arms = _arm_items(lane, eligible, lambda r: str(r["protocol"]))
    common = frozenset(set.intersection(*arms.values())) if arms else frozenset()

    def member(row: Mapping[str, Any]) -> bool:
        return _in_team_lane(row) and eligible(row) and str(row["item_id"]) in common

    census = _common_item_census(
        lane, arms=arms, common=common, declared=ARMS_32B, member=member,
        expected_cells=len(common) * len(ARMS_32B) * len(ROUNDS))
    census["items_by_split"] = dict(sorted(Counter(
        str(r["split"]) for r in lane if eligible(r) and str(r["item_id"]) in common
        and int(r["round"] or 0) == 0 and int(r["agent_id"] or 0) == 0
        and str(r["protocol"]) == ARMS_32B[0]).items()))
    return Panel(
        name="P4", unit="item",
        question="which items are comparable across every 32B protocol?",
        keys=common, universe=census["universe"], census=census, row_filter=member)


def panel_p5(rows: Sequence[Mapping[str, Any]], p0: Panel) -> Panel:
    """Budget-matched panel: items run at both reasoning allowances, PEER/PRIVATE, rounds 1-3.

    Round 0 is excluded because it carries no contrast: the high-budget arms *reuse* the
    T1024 root draws, so their round-0 rows are the same ``decision_id``s at
    ``budget_tokens = 1024``.  The panel is checked for that rather than assuming it.
    """
    lane = p0.select(rows)

    def high_budget(row: Mapping[str, Any]) -> bool:
        return str(row["cohort"]) == COHORT_HIGH_BUDGET

    arms = _arm_items(lane, high_budget,
                      lambda r: (str(r["checkpoint"]), str(r["protocol"])))
    common = frozenset(set.intersection(*arms.values())) if arms else frozenset()

    def member(row: Mapping[str, Any]) -> bool:
        return (_in_team_lane(row)
                and str(row["item_id"]) in common
                and str(row["protocol"]) in ARMS_BUDGET
                and int(row["round"] or 0) in BUDGET_ROUNDS
                and str(row["cohort"]) in (COHORT_NATURAL, COHORT_HIGH_BUDGET))

    declared = [(c, p) for c in (CHECKPOINT_SMALL, CHECKPOINT_LARGE) for p in ARMS_BUDGET]
    census = _common_item_census(
        lane, arms=arms, common=common, declared=declared, member=member,
        expected_cells=(len(common) * len(ARMS_BUDGET) * 2 * 2 * len(BUDGET_ROUNDS)))
    panel_rows = [row for row in lane if member(row)]
    census["allowance_cells"] = dict(sorted(Counter(
        f"{r['checkpoint']}|{r['protocol']}|T{r['budget_tokens']}" for r in panel_rows).items()))
    # The claim that justifies dropping round 0, measured rather than asserted.
    round0 = [r for r in lane if str(r["item_id"]) in common and int(r["round"] or 0) == 0]
    hb0 = {str(r["decision_id"]) for r in round0 if high_budget(r)}
    nat0 = {str(r["decision_id"]) for r in round0 if str(r["cohort"]) == COHORT_NATURAL}
    census["round0_high_budget_decisions"] = len(hb0)
    census["round0_shared_with_t1024"] = len(hb0 & nat0)
    census["round0_high_budget_allowances"] = sorted(
        {int(r["budget_tokens"] or 0) for r in round0 if high_budget(r)})
    census["round0_high_budget_aliased"] = sum(
        1 for r in round0 if high_budget(r) and bool(r.get("aliased")))
    return Panel(
        name="P5", unit="item",
        question="which items carry both reasoning allowances under the same protocol?",
        keys=common, universe=census["universe"], census=census, row_filter=member)


def panel_p8(rows: Sequence[Mapping[str, Any]], p0: Panel, p3: Panel, p4: Panel) -> Panel:
    """Scale-matched panel: items both checkpoints ran under the protocols they share.

    The plan said "declare it or delete the scale category", so the size is the finding.  Only
    BALANCED, PEER and PRIVATE exist at both scales; on those three the 32B subset is wholly
    contained in the 8B coverage, giving **96 items x 3 protocols x 2 checkpoints**.  The
    stricter reading -- items that also carry all seven 8B arms, i.e. P3 ∩ P4 -- yields 60,
    because the 8B-only arms are test-only while 36 of the 32B items are train/validation.
    Both numbers are reported; the panel takes the 96, since a scale contrast can only be drawn
    on protocols that exist at both scales anyway.
    """
    lane = p0.select(rows)

    def eligible(row: Mapping[str, Any]) -> bool:
        return (str(row["cohort"]) == COHORT_NATURAL
                and str(row["protocol"]) in ARMS_32B)

    arms = _arm_items(lane, eligible,
                      lambda r: (str(r["checkpoint"]), str(r["protocol"])))
    common = frozenset(set.intersection(*arms.values())) if arms else frozenset()
    natural_items = {str(r["item_id"]) for r in lane
                     if str(r["cohort"]) == COHORT_NATURAL and str(r["item_id"])}

    def member(row: Mapping[str, Any]) -> bool:
        return _in_team_lane(row) and eligible(row) and str(row["item_id"]) in common

    declared = [(c, p) for c in (CHECKPOINT_SMALL, CHECKPOINT_LARGE) for p in ARMS_32B]
    census = _common_item_census(
        lane, arms=arms, common=common, declared=declared, member=member,
        expected_cells=len(common) * len(ARMS_32B) * 2 * len(ROUNDS))
    # The intersection-over-union above answers "of the items these ten arms touched"; the
    # decision-relevant denominator for a scale claim is the study's whole item pool.
    census["universe"] = len(natural_items)
    census["universe_name"] = "natural items in the run"
    census["coverage_rate"] = (round(len(common) / len(natural_items), 6)
                               if natural_items else 0.0)
    census["items_also_in_p3"] = len(common & p3.keys)
    census["items_in_p3_and_p4"] = len(p3.keys & p4.keys)
    census["p4_items_contained"] = p4.keys <= common
    census["rows_by_checkpoint"] = dict(sorted(Counter(
        str(r["checkpoint"]) for r in lane if member(r)).items()))
    return Panel(
        name="P8", unit="item",
        question="which items support an 8B-vs-32B contrast at all?",
        keys=common, universe=census["universe"], census=census, row_filter=member)


# --------------------------------------------------------------------------- #
# P6 / P7 -- the two panels that are not teams
# --------------------------------------------------------------------------- #

def panel_p6(tables_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> Panel:
    """Fork pairs: a private counterfactual child and the peer child of the same parent state.

    ``forks.parent_decision_ids`` is an all-null column in v1, so the pair cannot be closed
    from the fork row alone.  It is closed instead through ``decisions.parent_state_id``, a
    content address: the peer sibling is the *other* child of the same sealed state by the same
    focal agent.  A join that is not a genuine one-to-one match is counted, never guessed --
    ``pairs_ambiguous`` and ``pairs_missing_peer`` are both 0 on v1.
    """
    forks = _read(Path(tables_dir) / FORKS_TABLE, [
        "fork_id", "episode_id", "item_id", "round", "focal_agent",
        "child_decision_ids", "shared_prefix_hash"])
    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        parent = row.get("parent_state_id")
        if parent:
            by_parent[str(parent)].append(row)
    by_id = {str(row["decision_id"]): row for row in rows}

    paired: set[str] = set()
    decision_ids: set[str] = set()
    missing_child = missing_peer = ambiguous = 0
    graded_pairs = 0
    for fork in forks:
        children = [str(c) for c in (fork.get("child_decision_ids") or []) if c]
        child = by_id.get(children[0]) if children else None
        if child is None:
            missing_child += 1
            continue
        siblings = [row for row in by_parent[str(fork["shared_prefix_hash"])]
                    if int(row["agent_id"] or 0) == int(fork["focal_agent"] or 0)
                    and str(row["decision_id"]) != str(child["decision_id"])
                    and str(row["cohort"]) != FORK_COHORT]
        if not siblings:
            missing_peer += 1
            continue
        if len(siblings) > 1:
            ambiguous += 1
            continue
        paired.add(str(fork["fork_id"]))
        decision_ids.update({str(child["decision_id"]), str(siblings[0]["decision_id"])})
        if is_graded(child) and is_graded(siblings[0]):
            graded_pairs += 1

    census = {
        "unit": "fork_pair",
        "size": len(paired),
        "universe": len(forks),
        "universe_name": "forks.parquet rows",
        "coverage_rate": round(len(paired) / len(forks), 6) if forks else 0.0,
        "pairs_missing_child": missing_child,
        "pairs_missing_peer": missing_peer,
        "pairs_ambiguous": ambiguous,
        "pairs_graded_both_sides": graded_pairs,
        "graded_rate": round(graded_pairs / len(paired), 6) if paired else 0.0,
        "decision_rows": len(decision_ids),
        "items": len({str(f["item_id"]) for f in forks}),
        "rounds": {str(k): v for k, v in
                   sorted(Counter(int(f["round"] or 0) for f in forks).items())},
    }
    return Panel(
        name="P6", unit="fork_pair",
        question="which one-step private-vs-peer counterfactual pairs actually close?",
        keys=frozenset(paired), universe=len(forks), census=census,
        row_filter=lambda row: str(row.get("decision_id") or "") in decision_ids)


def panel_p7(rows: Sequence[Mapping[str, Any]]) -> Panel:
    """Receiver probes whose 20 draws are not all one answer class.

    A probe that returns one class on every draw carries no variance, so it cannot contribute
    to any sensitivity, agreement or susceptibility estimate -- including it only inflates the
    denominator.  31 of the 64 probes are degenerate in that sense, which caps every
    receiver-side statistic at 33 probes.
    """
    draws: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase") or "") == RECEIVER_PHASE:
            draws[str(row["item_id"])].append(str(row.get("answer_class") or ""))
    non_degenerate = frozenset(p for p, classes in draws.items() if len(set(classes)) > 1)
    census = {
        "unit": "probe",
        "size": len(non_degenerate),
        "universe": len(draws),
        "universe_name": "receiver probes",
        "coverage_rate": round(len(non_degenerate) / len(draws), 6) if draws else 0.0,
        "degenerate_probes": len(draws) - len(non_degenerate),
        "draws_per_probe": {str(k): v for k, v in
                            sorted(Counter(len(v) for v in draws.values()).items())},
        "distinct_classes_histogram": {str(k): v for k, v in sorted(Counter(
            len(set(v)) for v in draws.values()).items())},
        "decision_rows": sum(len(v) for p, v in draws.items() if p in non_degenerate),
        # Receiver probes are never graded in v1; the panel is a variance filter, not a
        # correctness filter, and a caller expecting labels here will find none.
        "graded_rows": sum(1 for row in rows
                           if str(row.get("phase") or "") == RECEIVER_PHASE and is_graded(row)),
    }
    return Panel(
        name="P7", unit="probe",
        question="which receiver probes can discriminate anything at all?",
        keys=non_degenerate, universe=len(draws), census=census,
        row_filter=lambda row: (str(row.get("phase") or "") == RECEIVER_PHASE
                                and str(row.get("item_id") or "") in non_degenerate))


# --------------------------------------------------------------------------- #
# The bound on every disagreement statistic
# --------------------------------------------------------------------------- #

def dynamic_range(tables_dir: str | Path) -> dict[str, Any]:
    """How much room the 8B test bank leaves for any disagreement statistic.

    An item whose 16 root draws are unanimously correct, or unanimously wrong, cannot exhibit
    disagreement, minority correctness, or a selection gap under any protocol: the team is
    drawn from that bank.  So the 83 non-degenerate items are the ceiling on every such
    statistic, and a claim quoted over "240 items" is quoting a denominator 2.9x too large.
    """
    banks = _read(Path(tables_dir) / BANKS_TABLE,
                  ["item_id", "checkpoint", "split", "n_draws", "c_correct"])
    panel = [b for b in banks
             if str(b["checkpoint"]) == CHECKPOINT_SMALL and str(b["split"]) == SPLIT_TEST]
    all_correct = [b for b in panel if int(b["c_correct"] or 0) == int(b["n_draws"] or 0)]
    all_wrong = [b for b in panel if int(b["c_correct"] or 0) == 0]
    out = {
        "unit": "item",
        "items": len(panel),
        "draws_per_item": {str(k): v for k, v in
                           sorted(Counter(int(b["n_draws"] or 0) for b in panel).items())},
        "all_correct": len(all_correct),
        "all_wrong": len(all_wrong),
        "non_degenerate": len(panel) - len(all_correct) - len(all_wrong),
        "coverage_rate": round(
            (len(panel) - len(all_correct) - len(all_wrong)) / len(panel), 6) if panel else 0.0,
        "c_correct_histogram": {str(k): v for k, v in
                                sorted(Counter(int(b["c_correct"] or 0) for b in panel).items())},
        # D-11: 8 bank rows were built over a doubled draw pool.  None of them is in this
        # panel, so the counts above are not contaminated -- but a caller widening the panel
        # to train/validation would be pooling 16- and 32-draw banks.
        "banks_with_unexpected_draw_count": sorted(
            str(b["item_id"]) for b in banks if int(b["n_draws"] or 0) != ROOT_DRAWS),
        "expected": dict(V1_DYNAMIC_RANGE),
    }
    out["matches_expected"] = all(
        out[key] == value for key, value in V1_DYNAMIC_RANGE.items())
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_panels(tables_dir: str | Path,
                 rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Panel]:
    """Every panel, built from one pass over the tables.

    ``rows`` is accepted so a caller that already holds the joined frame (or a test that
    synthesises one) does not pay for ``decisions.parquet`` twice.
    """
    frame = list(rows) if rows is not None else load_rows(tables_dir)
    p0 = panel_p0(frame)
    p3 = panel_p3(frame, p0)
    p4 = panel_p4(frame, p0)
    panels = {
        "P0": p0,
        "P0d": panel_p0d(frame, p0),
        "P1": panel_p1(frame, p0),
        "P2": panel_p2(frame, p0),
        "P3": p3,
        "P4": p4,
        "P5": panel_p5(frame, p0),
        "P6": panel_p6(tables_dir, frame),
        "P7": panel_p7(frame),
        "P8": panel_p8(frame, p0, p3, p4),
    }
    for name, panel in panels.items():
        expected = V1_PANEL_SIZES.get(name)
        panel.census["name"] = name
        panel.census["question"] = panel.question
        panel.census["expected"] = expected
        panel.census["matches_expected"] = (expected is None or panel.size == expected)
    return panels


def census(tables_dir: str | Path,
           rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The full census: every panel's size, universe and coverage, plus the dynamic range."""
    panels = build_panels(tables_dir, rows)
    mismatches = sorted(name for name, panel in panels.items()
                        if not panel.census["matches_expected"])
    dynamic = dynamic_range(tables_dir)
    if not dynamic["matches_expected"]:
        mismatches.append("dynamic_range")
    return {
        "tables": str(tables_dir),
        "panels": {name: panel.census for name, panel in panels.items()},
        "dynamic_range": dynamic,
        "mismatches": mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.panels",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True, help="a run's tables/ directory")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any panel size differs from V1_PANEL_SIZES")
    args = ap.parse_args(argv)

    report = census(args.tables)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.check and report["mismatches"]:
        print(f"panel sizes moved: {', '.join(report['mismatches'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
