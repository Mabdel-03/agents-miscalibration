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
from agents_scaling.ard.states import RoundState, rebuild_round_states
from agents_scaling.study.types import ProtocolError

#: Rounds whose *post-state* is a controller decision point: GATE acts before rounds 1, 2, 3,
#: so it needs features of the states at the end of rounds 0, 1 and 2.
DECISION_ROUNDS: tuple[int, ...] = (0,) + tuple(K.FORK_PARENT_ROUNDS)


def reconstruct_states(payload: Mapping[str, Any]) -> dict[tuple[str, int], RoundState]:
    """``{(episode_id, round): RoundState}`` for one episode item file.

    Thin wrapper over WP2's :func:`states.rebuild_round_states` with the evaluator-side
    canonical classifier bound in.  The rebuild itself lives in ``states.py`` because the
    FORK executor needs it too and runs on the generation path, where importing the grader
    would breach the no-gold firewall (contract §0.2).
    """
    family = str(payload.get("family") or "")
    return rebuild_round_states(
        payload, lambda answer: graders.canonical_class(family, answer))


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


def _children_by_parent(
    payload: Mapping[str, Any], *, protocols: Sequence[str] | None = None
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """``{(parent_state_id, agent_slot): decision}`` for every update in this item file.

    ``protocols`` filters by the *episode's* protocol rather than trusting the caller to have
    passed a protocol-pure manifest.  A manifest is a dispatch unit, not a branch: the P1
    manifest holds PEER and PRIVATE cells together, so taking it as "the peer branch" pairs
    each PRIVATE state with its own PRIVATE child and calls the result a counterfactual.
    """
    wanted = {str(p) for p in protocols} if protocols else None
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for episode in payload.get("episodes") or []:
        if wanted is not None and str(episode.get("protocol")) not in wanted:
            continue
        for decision in episode.get("decisions") or []:
            parent = str(decision.get("parent_state_id") or "")
            if not parent or int(decision.get("round") or 0) == 0:
                continue
            out[(parent, int(decision.get("agent_id") or 0))] = decision
    return out


#: The protocol whose trajectory is the *peer* branch, and the one whose round-1 child is the
#: private branch of the t=0 pair.  Both are filtered by episode, never by manifest.
PEER_PROTOCOL = K.Protocol.PEER.value
PRIVATE_PROTOCOL = K.Protocol.PRIVATE.value


def build_paired_rows(
    run_root: str | os.PathLike,
    *,
    peer_cells: Sequence[str],
    private_cells: Sequence[str] = (),
    fork_cells: Sequence[str] = (),
    labels: Mapping[str, int] | None = None,
) -> list[PairedRow]:
    """The fork frame: one row per (shared parent state, focal agent) with both outcomes.

    The branches are selected by *episode protocol*, so the same manifest may be passed for
    both ``peer_cells`` and ``private_cells`` -- which is how P1 is actually shaped.
    """
    root = Path(run_root)
    labels = dict(labels if labels is not None else load_labels(root))

    # Peer branch: the PEER trajectory's child off each parent state, plus the states and
    # features those PEER parents carry.  Only PEER states are decision points here: a
    # PRIVATE state has no peer-branch counterfactual to compare against.
    peer_children: dict[tuple[str, int], Mapping[str, Any]] = {}
    states: dict[str, RoundState] = {}
    meta: dict[str, dict[str, Any]] = {}
    for _cell_id, path in _episode_files(root, peer_cells):
        payload = json.loads(path.read_text(encoding="utf-8"))
        peer_children.update(_children_by_parent(payload, protocols=(PEER_PROTOCOL,)))
        for (episode_id, round_), state in reconstruct_states(payload).items():
            if state.protocol.value != PEER_PROTOCOL:
                continue
            for slot, agent in enumerate(state.agents):
                states[agent.state_id] = state
                meta[agent.state_id] = {
                    "slot": slot, "round": round_, "episode_id": episode_id,
                    "item_id": state.item_id, "family": str(payload.get("family") or ""),
                    "split": str(payload.get("split") or ""),
                }

    # Private branch: at t=0 it is PRIVATE's own round-1 child, which hangs off the *shared*
    # root state; at t=1,2 it is the FORK cell's branch off the sealed PEER state.  A PRIVATE
    # trajectory's round-1/2 children hang off PRIVATE's own parents and must never be read as
    # a counterfactual for a PEER state -- filtering by protocol is what prevents that.
    private_children: dict[tuple[str, int], Mapping[str, Any]] = {}
    for _cell_id, path in _episode_files(root, private_cells):
        payload = json.loads(path.read_text(encoding="utf-8"))
        private_children.update(
            _children_by_parent(payload, protocols=(PRIVATE_PROTOCOL,)))
    for _cell_id, path in _episode_files(root, fork_cells):
        payload = json.loads(path.read_text(encoding="utf-8"))
        private_children.update(_children_by_parent(payload))   # a fork cell is all forks

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
