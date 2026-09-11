"""Run artifacts -> controller frames (spec §12.1, §12.2; plan §5.4).

WP5 built both ends of the controller and nothing in between.  ``features.features_from_state``
turns a WP2 ``RoundState`` into the frozen v1 vector; ``controller.fit_controller`` consumes
``PairedFrame``s.  Neither reads the run, ``compact.py`` emits ``"features": []``, and nothing
assembled a frame from the episodes and forks on disk — so the primary estimand of the study had
no path from generated data to a fitted controller (defect D-07).

This module is that path.  It reconstructs the round states an episode already records, builds
the frozen feature vector on each, and pairs the two branches spec §12.1 asks for:

* **t = 0 pairs, free of charge.** PRIVATE's round-1 child and PEER's round-1 child grow from
  the *same* root state, so every train/val item yields five paired rows without a fork.
* **t = 1, 2 pairs from the forks.** The private branch is the FORK cell's child; the peer
  branch is the main PEER trajectory's child off the same parent state.

Both branches are joined on ``parent_state_id``, which is a content address: a pair is only a
pair when the two children genuinely share a prefix, so a mismatched join cannot silently
become training data.  Labels come from the evaluator lane keyed by ``decision_id``; a decision
with no label yields ``None`` (an unobserved outcome the fitter skips), never a zero.

    PYTHONPATH=src python -m agents_scaling.ard.frames --run-root <root> \
        --peer-cells cells_P1_8B_trainval_v2.json --fork-cells cells_E4_forks_8B.json \
        --out frames
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.ard import graders
from agents_scaling.ard.cells import load_ard_cells_file
from agents_scaling.ard.controller import PairedFrame, PairedRow
from agents_scaling.ard.report import Report
from agents_scaling.ard.states import AgentState, ReportRef, RoundState
from agents_scaling.study.types import ProtocolError

#: Rounds whose *post-state* is a controller decision point: GATE acts before rounds 1, 2, 3,
#: so it needs features of the states at the end of rounds 0, 1 and 2.
DECISION_ROUNDS: tuple[int, ...] = (0,) + tuple(K.FORK_PARENT_ROUNDS)


def _report_of(decision: Mapping[str, Any]) -> Report:
    """The Report a decision row carries.

    Only the four fields the frozen v1 features read are reconstructed with real values
    (`final_answer`, `confidence`, `parse_status` -> `valid`, `content_sha256`); the rest keep
    their defaults and are never consulted here.  Reconstructing more would invent provenance
    the row does not carry.
    """
    return Report(
        final_answer=decision.get("answer_raw"),
        evidence=decision.get("evidence_text"),
        confidence=decision.get("confidence"),
        parse_status=K.ParseStatus(str(decision.get("parse_status") or K.ParseStatus.OK.value)),
        content_sha256=str(decision.get("content_sha256") or ""),
    )


def _classifier(family: str) -> Callable[[str | None], str | None]:
    """The gold-independent canonical classifier for this item's family."""
    return lambda answer: graders.canonical_class(family, answer)


def reconstruct_states(payload: Mapping[str, Any]) -> dict[tuple[str, int], RoundState]:
    """``{(episode_id, round): RoundState}`` for one episode item file.

    The episode already records, per round, the decision ids of each slot; the decisions
    carry the reports.  ``pseudonyms`` and ``study_seed`` are placeholders: the frozen v1
    feature builder reads only ``agents``, ``latest()``, ``reports_of_round()`` and
    ``canonical_class``, and never the packet-facing fields.
    """
    family = str(payload.get("family") or "")
    classify = _classifier(family)
    out: dict[tuple[str, int], RoundState] = {}
    for episode in payload.get("episodes") or []:
        by_id = {str(d.get("decision_id") or ""): d for d in episode.get("decisions") or []}
        by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for decision in episode.get("decisions") or []:
            by_round[int(decision.get("round") or 0)].append(decision)
        archive: dict[int, tuple[ReportRef, ...]] = {}
        for round_, rows in by_round.items():
            refs = tuple(
                ReportRef(decision_id=str(r.get("decision_id") or ""), report=_report_of(r),
                          round_=round_, agent_slot=int(r.get("agent_id") or 0))
                for r in sorted(rows, key=lambda r: int(r.get("agent_id") or 0))
            )
            archive[round_] = refs
        if 0 not in archive:
            continue                      # a cell that records no root cannot be reconstructed
        roots = archive[0]
        episode_id = str(episode.get("episode_id") or "")
        for state in episode.get("states") or []:
            round_ = int(state.get("round", -1))
            ids = [str(x) for x in (state.get("decision_ids") or [])]
            if round_ not in K.ROUNDS or len(ids) != K.N_AGENTS:
                continue
            agents = []
            ok = True
            for slot, decision_id in enumerate(ids):
                row = by_id.get(decision_id)
                if row is None or slot >= len(roots):
                    ok = False
                    break
                agents.append(AgentState(
                    item_id=str(episode.get("item_id") or ""),
                    checkpoint=str(episode.get("checkpoint") or ""),
                    agent_slot=slot, round_=round_,
                    root=roots[slot],
                    latest=ReportRef(decision_id=decision_id, report=_report_of(row),
                                     round_=round_, agent_slot=slot),
                    episode_id=episode_id,
                    protocol=str(episode.get("protocol") or ""),
                ))
            if not ok:
                continue
            out[(episode_id, round_)] = RoundState(
                episode_id=episode_id,
                pairing_id=str(episode.get("item_id") or ""),
                item_id=str(episode.get("item_id") or ""),
                protocol=K.Protocol(str(episode.get("protocol") or "PEER")),
                checkpoint=str(episode.get("checkpoint") or ""),
                round_=round_,
                agents=tuple(agents),
                archive={r: refs for r, refs in archive.items() if r <= round_},
                pseudonyms=tuple(f"agent-{i}" for i in range(K.N_AGENTS)),
                study_seed=b"\x00" * 32,
                classify=classify,
                family=family,
            )
    return out


def load_labels(run_root: str | os.PathLike) -> dict[str, int]:
    """``{decision_id: is_correct}`` from the evaluator lane.

    Grades are written per grade cell per item; a decision graded by several cells is graded
    identically (the grader is a pure function of answer and gold), so the last write wins.
    """
    labels: dict[str, int] = {}
    grades = Path(run_root) / "eval" / "grades"
    if not grades.exists():
        return labels
    # The evaluator's own vocabulary, imported rather than spelled out.  ``is_correct`` alone
    # is NOT the label: the grader sets it False for ``ambiguous`` and ``timeout`` as well as
    # ``incorrect``, so reading it directly would score every ungradeable answer as wrong --
    # and on this run that is 10% of decisions, concentrated on the answers the symbolic
    # comparison could not parse rather than on the ones the model got wrong.
    observed = {graders.STATUS_CORRECT, graders.STATUS_INCORRECT}
    for path in grades.glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows") or []:
            decision_id = str(row.get("decision_id") or "")
            status = str(row.get("evaluator_status") or "")
            if not decision_id or status not in observed:
                continue                  # ambiguous / timeout stay unobserved, never 0
            labels[decision_id] = int(status == graders.STATUS_CORRECT)
    return labels


def _episode_files(run_root: Path, cell_ids: Sequence[str]) -> Iterable[tuple[str, Path]]:
    for cell_id in cell_ids:
        cell = run_root / "cells" / cell_id
        if not (cell / "meta.json").exists():
            continue
        for path in sorted((cell / "items").glob("*.json")):
            yield cell_id, path


def _children_by_parent(payload: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    """``{(parent_state_id, agent_slot): decision}`` for every update in this item file."""
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for episode in payload.get("episodes") or []:
        for decision in episode.get("decisions") or []:
            parent = str(decision.get("parent_state_id") or "")
            if not parent or int(decision.get("round") or 0) == 0:
                continue
            out[(parent, int(decision.get("agent_id") or 0))] = decision
    return out


def build_paired_rows(
    run_root: str | os.PathLike,
    *,
    peer_cells: Sequence[str],
    private_cells: Sequence[str] = (),
    fork_cells: Sequence[str] = (),
    labels: Mapping[str, int] | None = None,
) -> list[PairedRow]:
    """The fork frame: one row per (shared parent state, focal agent) with both outcomes."""
    root = Path(run_root)
    labels = dict(labels if labels is not None else load_labels(root))

    # Peer branch: the main trajectory's child off each parent state, plus the states and
    # features those parents carry.
    peer_children: dict[tuple[str, int], Mapping[str, Any]] = {}
    states: dict[str, RoundState] = {}
    meta: dict[str, dict[str, Any]] = {}
    for _cell_id, path in _episode_files(root, peer_cells):
        payload = json.loads(path.read_text(encoding="utf-8"))
        peer_children.update(_children_by_parent(payload))
        for (episode_id, round_), state in reconstruct_states(payload).items():
            for slot, agent in enumerate(state.agents):
                states[agent.state_id] = state
                meta[agent.state_id] = {
                    "slot": slot, "round": round_, "episode_id": episode_id,
                    "item_id": state.item_id, "family": str(payload.get("family") or ""),
                    "split": str(payload.get("split") or ""),
                }

    # Private branch: the t=0 pair comes from PRIVATE's own round-1 child off the shared root
    # state; t=1,2 come from the fork cells.
    private_children: dict[tuple[str, int], Mapping[str, Any]] = {}
    for cells in (private_cells, fork_cells):
        for _cell_id, path in _episode_files(root, cells):
            payload = json.loads(path.read_text(encoding="utf-8"))
            private_children.update(_children_by_parent(payload))

    rows: list[PairedRow] = []
    for state_id, info in sorted(meta.items()):
        if int(info["round"]) not in DECISION_ROUNDS:
            continue
        slot = int(info["slot"])
        peer = peer_children.get((state_id, slot))
        private = private_children.get((state_id, slot))
        if peer is None and private is None:
            continue
        state = states[state_id]
        features = FE.features_from_state(state, slot, round_=int(info["round"]))
        rows.append(PairedRow(
            row_id=f"{state_id}|{slot}",
            item_id=str(info["item_id"]),
            family=str(info["family"]),
            split=str(info["split"]),
            round_=int(info["round"]),
            agent_slot=slot,
            features=features,
            y_private=labels.get(str(private.get("decision_id"))) if private else None,
            y_peer=labels.get(str(peer.get("decision_id"))) if peer else None,
            episode_id=str(info["episode_id"]),
            parent_state_id=state_id,
        ))
    return rows


def split_frames(rows: Sequence[PairedRow]) -> tuple[PairedFrame, PairedFrame]:
    """``(train, validation)`` frames; test rows are refused, not filtered silently."""
    names = tuple(FE.FEATURES_V1)
    leaked = sorted({r.row_id for r in rows if r.split == "test"})
    if leaked:
        raise ProtocolError(
            f"{len(leaked)} test rows reached the controller frame (e.g. {leaked[:3]}); "
            "spec §12.1 forbids any test item in training or calibration"
        )
    train = PairedFrame(tuple(r for r in rows if r.split == "train"), names)
    val = PairedFrame(tuple(r for r in rows if r.split == "validation"), names)
    return train, val


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.frames",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--peer-cells", required=True, help="manifest of the PEER trajectories")
    ap.add_argument("--private-cells", default=None, help="manifest of the PRIVATE trajectories")
    ap.add_argument("--fork-cells", default=None, help="manifest of the FORK cells")
    ap.add_argument("--out", default=None, help="directory for frames.json (optional)")
    args = ap.parse_args(argv)

    root = Path(args.run_root)

    def ids(name: str | None) -> list[str]:
        if not name:
            return []
        path = Path(name) if Path(name).is_absolute() else root / name
        return [cell.cell_id for cell in load_ard_cells_file(path)]

    rows = build_paired_rows(root, peer_cells=ids(args.peer_cells),
                             private_cells=ids(args.private_cells),
                             fork_cells=ids(args.fork_cells))
    train, val = split_frames(rows)
    summary = {
        "rows": len(rows),
        "train_rows": len(train.rows), "validation_rows": len(val.rows),
        "by_round": {str(r): sum(1 for x in rows if x.round_ == r) for r in DECISION_ROUNDS},
        "with_both_outcomes": sum(1 for r in rows if r.y_private is not None and r.y_peer is not None),
        "y_private_observed": sum(1 for r in rows if r.y_private is not None),
        "y_peer_observed": sum(1 for r in rows if r.y_peer is not None),
        "features": len(FE.FEATURES_V1),
    }
    if args.out:
        out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frames_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
