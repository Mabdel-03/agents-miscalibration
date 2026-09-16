"""The minority-correct analysis frame: one row per (episode, round, agent).

The study's empirical question is not "does a controller work" but **what characterises an
agent that is correct while its team is not**, across tasks, configurations and measurables.
This module builds the single table that question is asked of, joining:

* the agent's own decision (confidence, answer logprob, thinking length, cap hit, packets seen,
  whether it was protected, realised FLOPs),
* the team's state at that round (how many agents hold the correct class, class entropy,
  agreement, whether the correct class is the plurality),
* the task (family, subject, declared difficulty, split),
* the configuration (protocol, checkpoint, reasoning allowance, cohort),
* and what happened next (stayed correct, changed class, whether the team gained or lost).

**Definitions, stated once and used everywhere.** ``n_correct`` is the number of the five
agents whose canonical answer class is correct at that round. An agent is:

* ``correct`` — its own answer is correct;
* ``minority_correct`` — correct while ``n_correct <= 2`` (correct is held by a minority of
  five). This is the spec's motivating case;
* ``lone_correct`` — correct while ``n_correct == 1`` (it is the only one);
* ``against_plurality`` — correct while the *plurality* answer class is wrong, which is the
  decision-relevant sense: it is the case where the team's own aggregation rule would discard
  the right answer.

These are reported separately rather than collapsed, because they are not the same event and
a reader should be able to see which one a claim is about.

Rows where the agent's answer is ungradeable (``ambiguous``/``timeout``) carry
``correct = None`` and are excluded from correctness-conditioned statistics rather than scored
as wrong — the missingness concentrates on the answers the symbolic grader could not parse.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents_scaling.ard.contract import N_AGENTS

# ROOTS is the stateless 16-draw bank; RECEIVER is a single-agent probe. Neither has a
# "how many of the five hold the correct class", so neither may enter the team frame.
_NON_TEAM_KINDS: frozenset[str] = frozenset({"ROOTS", "RECEIVER"})

TABLE = "minority_frame.parquet"


def _read(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        raise SystemExit(f"missing {path}; compact the run first")
    return pq.read_table(path).to_pylist()


def build_frame(tables: Path) -> list[dict[str, Any]]:
    decisions = _read(tables / "decisions.parquet")
    labels = {str(r["decision_id"]): r for r in _read(tables / "labels.parquet")}
    rounds = {(str(r["episode_id"]), int(r["round"])): r
              for r in _read(tables / "round_summary.parquet")}
    episodes = {str(r["episode_id"]): r for r in _read(tables / "episodes.parquet")}
    items = {str(r["item_id"]): r for r in _read(tables / "items.parquet")}

    def correctness(decision_id: str) -> int | None:
        row = labels.get(str(decision_id))
        if row is None:
            return None
        status = str(row.get("evaluator_status") or "")
        if status not in ("correct", "incorrect"):
            return None                    # ungradeable: unobserved, never scored wrong
        return int(bool(row.get("is_correct")))

    # Group the decisions of each (cell, episode, round) so team state can be computed from the
    # same rows the agent statistics come from, rather than trusting two sources to agree.
    #
    # The cell is part of the key, not decoration. A ROOTS cell defaults its protocol to
    # PRIVATE, so its 16-draw bank and the five-agent PRIVATE episode of the same item share an
    # ``episode_id`` (M2/D-11) and collide into one 21-member group. Selecting teams by
    # ``len(members) == N_AGENTS`` -- what this module used to do -- therefore did not exclude
    # the bank: it excluded the *collision*, silently dropping all 2,400 natural PRIVATE
    # round-0 rows, the arm PRIVATE is measured against. Keying on ``cell_id`` separates them,
    # so those rows come back and the bank never enters as a team.
    by_round: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        kind = str(row.get("cell_kind") or "")
        if kind in _NON_TEAM_KINDS:
            continue                       # a bank of draws and a single-agent probe are not teams
        by_round[(str(row.get("cell_id") or ""), str(row.get("episode_id")),
                  int(row.get("round") or 0))].append(row)

    excluded: Counter = Counter()
    teams: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for key, members in by_round.items():
        if len(members) == N_AGENTS:
            teams[key] = members
        else:
            excluded[len(members)] += 1
    build_frame.excluded_groups = dict(sorted(excluded.items()))

    frame: list[dict[str, Any]] = []
    for (cell_id, episode_id, round_), members in sorted(teams.items(),
                                                         key=lambda kv: str(kv[0])):
        episode = episodes.get(episode_id) or {}
        item_id = str(episode.get("item_id") or "")
        item = items.get(item_id) or {}
        # Team state, computed here from the members themselves.
        classes = [m.get("answer_class") for m in members]
        valid_classes = [c for c in classes if c]
        counts = Counter(valid_classes)
        plurality = max(counts, key=lambda c: (counts[c], c)) if counts else None
        marks = {str(m.get("decision_id")): correctness(m.get("decision_id")) for m in members}
        correct_classes = {m.get("answer_class") for m in members
                           if marks.get(str(m.get("decision_id"))) == 1 and m.get("answer_class")}
        n_correct = sum(1 for v in marks.values() if v == 1)
        n_graded = sum(1 for v in marks.values() if v is not None)
        plurality_correct = (plurality in correct_classes) if plurality else None

        for row in members:
            decision_id = str(row.get("decision_id"))
            correct = marks.get(decision_id)
            own_class = row.get("answer_class")
            share = counts.get(own_class, 0) if own_class else 0
            answer = row.get("answer_raw")
            summary = rounds.get((episode_id, round_)) or {}
            frame.append({
                # identity
                "episode_id": episode_id, "cell_id": cell_id, "item_id": item_id,
                "agent_id": int(row.get("agent_id") or 0), "round": round_,
                "decision_id": decision_id,
                # configuration
                "protocol": str(row.get("protocol") or episode.get("protocol") or ""),
                "checkpoint": str(row.get("checkpoint") or ""),
                "cohort": str(row.get("cohort") or ""),
                "reasoning_allowance": int(row.get("budget_tokens") or 0),
                "split": str(episode.get("split") or item.get("split") or ""),
                # task
                "family": str(episode.get("family") or item.get("family") or ""),
                "subject": str(item.get("subject") or ""),
                "difficulty": str(item.get("declared_difficulty_metadata") or ""),
                # the agent's own measurables
                "correct": correct,
                "valid": bool(own_class),
                "answer_class": own_class,
                "confidence": row.get("confidence"),
                "answer_logprob": row.get("answer_logprob"),
                "thinking_tokens": row.get("thinking_tokens"),
                "final_tokens": row.get("final_tokens"),
                "cap_hit": bool(row.get("boundary_injected")),
                "answer_length": len(str(answer)) if answer is not None else None,
                "n_packets": row.get("n_packets"),
                "packet_tokens": row.get("packet_tokens"),
                "protected": bool(row.get("protected")),
                "flops_total": row.get("flops_total"),
                "latency": row.get("latency"),
                "parse_status": str(row.get("parse_status") or ""),
                # team state at this round
                "n_correct": n_correct,
                "n_graded": n_graded,
                "n_valid": len(valid_classes),
                "n_unique_classes": len(counts),
                "own_class_share": share,
                "plurality_class": plurality,
                "plurality_correct": plurality_correct,
                "agreement_fraction": summary.get("agreement_fraction"),
                "class_entropy": summary.get("class_entropy"),
                "coverage_current": summary.get("coverage_current"),
                "coverage_archive": summary.get("coverage_archive"),
                # the four minority senses, kept separate on purpose
                "minority_correct": (None if correct is None
                                     else bool(correct == 1 and n_correct <= 2)),
                "lone_correct": (None if correct is None
                                 else bool(correct == 1 and n_correct == 1)),
                "against_plurality": (None if correct is None or plurality_correct is None
                                      else bool(correct == 1 and not plurality_correct)),
                "majority_correct": (None if correct is None
                                     else bool(correct == 1 and n_correct >= 3)),
            })

    # What happened next, joined on (episode, agent, round -> round+1).
    nxt = {(r["cell_id"], r["episode_id"], r["agent_id"], r["round"]): r for r in frame}
    for row in frame:
        after = nxt.get((row["cell_id"], row["episode_id"], row["agent_id"], row["round"] + 1))
        row["next_correct"] = None if after is None else after["correct"]
        row["next_class"] = None if after is None else after["answer_class"]
        row["changed_class"] = (None if after is None
                                else bool(after["answer_class"] != row["answer_class"]))
        row["next_n_correct"] = None if after is None else after["n_correct"]
        row["team_delta"] = (None if after is None
                             else after["n_correct"] - row["n_correct"])
        # The event the study is named for: a correct agent that gives up its correct answer.
        row["capitulated"] = (None if after is None or row["correct"] != 1
                              else bool(after["correct"] == 0))
        row["converted_others"] = (None if after is None
                                   else bool(row["correct"] == 1 and
                                             after["n_correct"] > row["n_correct"]))
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.minority",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True)
    args = ap.parse_args(argv)
    tables = Path(args.tables)
    frame = build_frame(tables)

    from agents_scaling.study.aggregate import _write_parquet

    _write_parquet(tables / TABLE, frame)
    graded = [r for r in frame if r["correct"] is not None]
    summary = {
        "rows": len(frame),
        "graded_rows": len(graded),
        "ungradeable_rows": len(frame) - len(graded),
        "protocols": sorted({r["protocol"] for r in frame}),
        "checkpoints": sorted({r["checkpoint"] for r in frame}),
        "families": sorted({r["family"] for r in frame if r["family"]}),
        "cohorts": sorted({r["cohort"] for r in frame}),
        "minority_correct_rows": sum(1 for r in graded if r["minority_correct"]),
        "lone_correct_rows": sum(1 for r in graded if r["lone_correct"]),
        "against_plurality_rows": sum(1 for r in graded if r["against_plurality"]),
        "majority_correct_rows": sum(1 for r in graded if r["majority_correct"]),
        "excluded_non_team_groups": getattr(build_frame, "excluded_groups", {}),
        "out": str(tables / TABLE),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
