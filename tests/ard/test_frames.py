"""Run artifacts -> controller frames (spec §12.1; plan §5.4).

WP5 built feature extraction and the fitter; nothing joined them to the run, and
``compact.py`` emits ``"features": []``.  These tests pin the join, and especially the two
things that decide whether the frame is training data or garbage: that a pair is only a pair
when both children genuinely share a parent state, and that an ungraded decision is an
*unobserved* outcome rather than a zero.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.ard import frames
from agents_scaling.study.types import ProtocolError


def _episode(item, protocol, *, split="train", root_ids=None, child_suffix=""):
    """One episode: five roots, then one update round whose parents are the root states."""
    roots = root_ids or [f"{item}-root-{k}" for k in range(K.N_AGENTS)]
    decisions = [
        {"decision_id": roots[k], "agent_id": k, "round": 0, "phase": "root",
         "answer_raw": "42", "answer_class": "expr:42", "confidence": 0.8,
         "content_sha256": f"c-{roots[k]}", "parse_status": "ok", "parent_state_id": None}
        for k in range(K.N_AGENTS)
    ]
    # The round-0 AgentState ids are computed by WP2 from the root refs, so build the states
    # the way the episode writer does: reconstruct, then read the ids back.
    payload = {"family": "math500", "split": split, "episodes": [{
        "episode_id": f"{item}|8B|natural|e0|{protocol}", "item_id": item,
        "protocol": protocol, "checkpoint": "8B", "decisions": decisions,
        "states": [{"state_id": f"{item}-r0", "round": 0, "decision_ids": roots,
                    "answer_classes": ["expr:42"] * 5, "valid": [True] * 5,
                    "protected_slot": None, "leader_id": None}],
    }]}
    states = frames.reconstruct_states(payload)
    state = states[(f"{item}|8B|natural|e0|{protocol}", 0)]
    for k in range(K.N_AGENTS):
        decisions.append({
            "decision_id": f"{item}-{protocol}{child_suffix}-r1-{k}", "agent_id": k, "round": 1,
            "phase": "update", "answer_raw": "43", "answer_class": "expr:43",
            "confidence": 0.6, "content_sha256": f"c-{item}-{protocol}-r1-{k}",
            "parse_status": "ok", "parent_state_id": state.agents[k].state_id,
        })
    return payload, state


def _write(run_root, cell_id, payload, item):
    cell = run_root / "cells" / cell_id
    (cell / "items").mkdir(parents=True, exist_ok=True)
    (cell / "items" / f"{item}.json").write_text(json.dumps(payload))
    (cell / "meta.json").write_text("{}")
    return cell_id


def test_states_reconstruct_with_the_full_frozen_feature_vector(tmp_path):
    payload, state = _episode("math500-1", "PEER")
    assert len(state.agents) == K.N_AGENTS
    assert sorted(state.archive) == [0]
    row = FE.features_from_state(state, 0, round_=0)
    assert set(row) == set(FE.FEATURES_V1)
    assert all(isinstance(v, float) for v in row.values())


def test_the_t0_pair_is_free_because_both_protocols_share_the_root_state(tmp_path):
    """Spec §12.1: PRIVATE's and PEER's round-1 children grow from the *same* root state.

    The state id is a content address over the root reports, so the two protocols produce the
    same id without coordinating -- which is what makes the t=0 pair free.  If that ever stops
    holding, the controller silently loses a third of its training rounds.
    """
    roots = [f"math500-1-root-{k}" for k in range(K.N_AGENTS)]
    peer, peer_state = _episode("math500-1", "PEER", root_ids=roots)
    private, private_state = _episode("math500-1", "PRIVATE", root_ids=roots)
    assert [a.state_id for a in peer_state.agents] == [a.state_id for a in private_state.agents]

    _write(tmp_path, "EPISODE.natural.PEER.8B.T1024.train.s000", peer, "math500-1")
    _write(tmp_path, "EPISODE.natural.PRIVATE.8B.T1024.train.s000", private, "math500-1")
    rows = frames.build_paired_rows(
        tmp_path,
        peer_cells=["EPISODE.natural.PEER.8B.T1024.train.s000"],
        private_cells=["EPISODE.natural.PRIVATE.8B.T1024.train.s000"],
        labels={f"math500-1-PEER-r1-{k}": 1 for k in range(K.N_AGENTS)} |
               {f"math500-1-PRIVATE-r1-{k}": 0 for k in range(K.N_AGENTS)},
    )
    assert len(rows) == K.N_AGENTS
    assert all(r.round_ == 0 for r in rows)
    assert all(r.y_peer == 1 and r.y_private == 0 for r in rows), "both branches must join"


def test_an_ungraded_decision_is_unobserved_not_zero(tmp_path):
    """A missing label is a missing outcome.  Scoring it 0 would teach the controller that
    every ungraded branch failed -- and ungraded correlates with hard, not with wrong."""
    peer, _ = _episode("math500-2", "PEER")
    _write(tmp_path, "EPISODE.natural.PEER.8B.T1024.train.s000", peer, "math500-2")
    rows = frames.build_paired_rows(
        tmp_path, peer_cells=["EPISODE.natural.PEER.8B.T1024.train.s000"], labels={})
    assert rows and all(r.y_peer is None and r.y_private is None for r in rows)

    # The evaluator's ambiguous/timeout statuses never become labels either.
    grades = tmp_path / "eval" / "grades" / "GRADE.s000"
    grades.mkdir(parents=True)
    (grades / "math500-2.json").write_text(json.dumps({"rows": [
        {"decision_id": "math500-2-PEER-r1-0", "status": "ambiguous"},
        {"decision_id": "math500-2-PEER-r1-1", "status": "timeout"},
        {"decision_id": "math500-2-PEER-r1-2", "status": "correct"},
    ]}))
    labels = frames.load_labels(tmp_path)
    assert labels == {"math500-2-PEER-r1-2": 1}


def test_a_test_row_never_reaches_the_controller_frame(tmp_path):
    """Spec §12.1 forbids test items in training or calibration; refuse rather than filter."""
    peer, _ = _episode("math500-3", "PEER", split="test")
    _write(tmp_path, "EPISODE.natural.PEER.8B.T1024.test.s000", peer, "math500-3")
    rows = frames.build_paired_rows(
        tmp_path, peer_cells=["EPISODE.natural.PEER.8B.T1024.test.s000"], labels={})
    with pytest.raises(ProtocolError, match="test rows reached"):
        frames.split_frames(rows)


def test_only_finished_cells_contribute(tmp_path):
    peer, _ = _episode("math500-4", "PEER")
    cell = tmp_path / "cells" / "EPISODE.natural.PEER.8B.T1024.train.s000"
    (cell / "items").mkdir(parents=True)
    (cell / "items" / "math500-4.json").write_text(json.dumps(peer))   # no meta.json
    assert frames.build_paired_rows(
        tmp_path, peer_cells=["EPISODE.natural.PEER.8B.T1024.train.s000"], labels={}) == []
