"""A FORK cell branches off a sealed PEER state (spec §12.1, §19; defect D-10).

``forks.fork_private`` was fully built and the runner never called it: a FORK cell fell
through to ``run_episode`` and produced an ordinary three-round PRIVATE trajectory from its
own roots.  That pairs with nothing -- the whole point of a fork is one extra private revision
of one agent from a prefix the peer branch also continues from.

These tests pin the three properties that make a fork a counterfactual: exactly one new
decision per focal agent, every one of them parented on the sealed PEER state, and a refusal
when the parent is missing or has moved since it was sealed.
"""

from __future__ import annotations

import json
import threading

import pytest

from agents_scaling.ard import cells as ard_cells
from agents_scaling.ard import contract as K
from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard.types import ArdCellKind, EXIT_DONE
from agents_scaling.study.types import ProtocolError
from tests.ard.wp4_support import make_run_root, runner_kwargs, synthetic_tasks, write_items
from tests.ard.test_wp4_runner import _episode_cell, _run, _tasks, fake_server


@pytest.fixture
def run_root(tmp_path):
    root = make_run_root(tmp_path)
    # Fork frames are training data, so the parent items must be trainable splits.
    write_items(root, synthetic_tasks(1, splits=("train",)))
    return root


def _peer_cell_with_states(run_root):
    """Run a real PEER episode so its round-1/2 states exist to branch from."""
    peer = _episode_cell(run_root, protocol=K.Protocol.PEER)
    with fake_server(run_root):
        assert _run(peer, run_root) == EXIT_DONE
    return peer


def _seal(run_root, peer_cell, rounds=(1, 2)):
    from agents_scaling.ard import seal
    sha, census = seal.seal_states(run_root, [peer_cell.cell_id], rounds=rounds,
                                   protocols=(K.Protocol.PEER.value,))
    assert census["states_sealed"] > 0
    return sha


def _fork_cell(run_root, peer_cell, sha, *, round_=1):
    builder = ard_cells.ArdBuilder()
    cells = ard_cells.build_e4_forks(builder, _tasks(run_root),
                                     checkpoint=K.CHECKPOINT_SMALL, state_sha=sha,
                                     parent_rounds=(round_,),
                                     parent_cells=[peer_cell.cell_id])
    return cells[0]


def test_a_fork_is_one_private_revision_per_agent_off_the_sealed_parent(run_root):
    peer = _peer_cell_with_states(run_root)
    sha = _seal(run_root, peer)
    fork = _fork_cell(run_root, peer, sha, round_=1)

    with fake_server(run_root):
        assert _run(fork, run_root) == EXIT_DONE

    payload = json.loads(
        next((run_root / "cells" / fork.cell_id / "items").glob("*.json")).read_text())
    episode = payload["episodes"][0]
    decisions = episode["decisions"]

    # Exactly one new decision per focal agent -- not a fresh 20-decision episode.
    assert len(decisions) == K.N_AGENTS, f"a fork cell produced {len(decisions)} decisions"
    assert sorted(d["agent_id"] for d in decisions) == list(range(K.N_AGENTS))
    assert {d["round"] for d in decisions} == {2}          # child of the round-1 parent

    # Every child is parented on the sealed PEER state, which is what makes it pair.
    peer_payload = json.loads(
        next((run_root / "cells" / peer.cell_id / "items").glob("*.json")).read_text())
    from agents_scaling.ard.states import rebuild_round_states
    parents = rebuild_round_states(peer_payload, lambda a: a,
                                   protocols=(K.Protocol.PEER.value,))
    parent = next(s for (_e, r), s in parents.items() if r == 1)
    expected = {a.state_id for a in parent.agents}
    assert {d["parent_state_id"] for d in decisions} == expected


def test_a_fork_refuses_a_parent_that_is_not_there(run_root):
    peer = _peer_cell_with_states(run_root)
    sha = _seal(run_root, peer)
    fork = _fork_cell(run_root, peer, sha, round_=1)
    # Point the cell at a parent cell that holds no episode for this item.
    broken = ard_cells.ArdCellSpec.from_dict(
        {**fork.to_dict(), "options": {**fork.options, "parent_cells": ["EPISODE.absent.s000"]}})
    with fake_server(run_root):
        assert _run(broken, run_root) != EXIT_DONE       # suspended, not silently empty
    assert (run_root / "cells" / broken.cell_id / "SUSPENDED.json").exists()


def test_a_parent_that_moved_since_sealing_is_refused(run_root):
    """The seal is the promise that the parent cohort is frozen; check it, or it is decor."""
    peer = _peer_cell_with_states(run_root)
    sha = _seal(run_root, peer)
    fork = _fork_cell(run_root, peer, sha, round_=1)

    register = run_root / "states" / sha / "REGISTER.json"
    payload = json.loads(register.read_text())
    for row in payload["states"]:
        if row["round"] == 1:
            row["decision_ids"] = ["moved"] * K.N_AGENTS
    # Re-stamp so the dependency gate still passes and the *content* check is what fires.
    sha2 = ard_runner.freeze_register(run_root, "state",
                                      {k: v for k, v in payload.items()
                                       if k != "state_sha256"})
    moved = ard_cells.ArdCellSpec.from_dict({
        **fork.to_dict(), "depends_on": [f"state:{sha2}"], "parent_ref": f"state:{sha2}"})
    with fake_server(run_root):
        assert _run(moved, run_root) != EXIT_DONE
    error = json.loads((run_root / "cells" / moved.cell_id / "SUSPENDED.json").read_text())
    assert "changed since it was sealed" in error["error"]
