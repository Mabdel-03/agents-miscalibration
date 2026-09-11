"""Sealing the round-t states a FORK cell depends on (plan §5.3, spec §12.1).

``runner.resolve_hard_dependencies`` refuses a FORK cell until ``state:<sha>`` resolves, and
``runner.freeze_register`` is the writer that matches that reader — but nothing called it, so
the fork stage had a dependency mechanism at both ends and no step in the middle.  These tests
pin the step, and in particular the two rules that make a seal trustworthy: it covers only
finished cells, and its sha is a content address over what it actually contains.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard import seal
from agents_scaling.ard.types import ArdCellKind, ArdCellSpec
from agents_scaling.ard import contract as K
from agents_scaling.study.types import ProtocolError


def _write_cell(run_root, cell_id, *, items, complete=True, rounds=(0, 1, 2, 3), protocol="PEER"):
    cell = run_root / "cells" / cell_id
    (cell / "items").mkdir(parents=True, exist_ok=True)
    for item in items:
        episode = {
            "episode_id": f"{item}|8B|natural|e0|{protocol}",
            "item_id": item, "protocol": protocol, "checkpoint": "8B",
            "states": [
                {"state_id": f"{item}-r{r}", "episode_id": f"{item}|8B|natural|e0|{protocol}",
                 "round": r, "decision_ids": [f"{item}-d{r}-{k}" for k in range(5)],
                 "answer_classes": ["A"] * 5, "valid": [True] * 5,
                 "protected_slot": None, "leader_id": None}
                for r in rounds
            ],
            "decisions": [],
        }
        (cell / "items" / f"{item}.json").write_text(json.dumps({"episodes": [episode]}))
    if complete:
        (cell / "meta.json").write_text(json.dumps({"cell_id": cell_id}))
    return cell_id


def test_only_finished_cells_are_sealed(tmp_path):
    """A cell still generating would freeze a cohort that is about to grow."""
    done = _write_cell(tmp_path, "EPISODE.a.s000", items=["i1", "i2"])
    running = _write_cell(tmp_path, "EPISODE.a.s001", items=["i3"], complete=False)

    states, census = seal.collect_states(tmp_path, [done, running], rounds=(1, 2))
    assert census["cells_read"] == 1 and census["cells_skipped_incomplete"] == 1
    assert {s["item_id"] for s in states} == {"i1", "i2"}        # i3 is not sealed
    assert sorted({s["round"] for s in states}) == [1, 2]
    assert len(states) == 4                                      # 2 items x 2 rounds


def test_the_sha_is_a_content_address_over_what_was_sealed(tmp_path):
    a = _write_cell(tmp_path, "EPISODE.a.s000", items=["i1"])
    sha1, _ = seal.seal_states(tmp_path, [a], rounds=(1, 2))
    sha2, _ = seal.seal_states(tmp_path, [a], rounds=(1, 2))
    assert sha1 == sha2, "re-sealing the same states must give the same sha"

    # A different round selection, and a larger cohort, are different seals.
    sha_r1, _ = seal.seal_states(tmp_path, [a], rounds=(1,))
    assert sha_r1 != sha1
    b = _write_cell(tmp_path, "EPISODE.a.s001", items=["i2"])
    sha_more, _ = seal.seal_states(tmp_path, [a, b], rounds=(1, 2))
    assert sha_more != sha1, "adding a cell must produce a new register, not mutate the old one"

    # The old register is still there and still resolves: seals are immutable.
    for sha in (sha1, sha_more):
        assert (tmp_path / "states" / sha / "REGISTER.json").exists()


def test_a_sealed_register_satisfies_the_fork_cells_hard_dependency(tmp_path):
    """The whole point: the file the sealer writes is the file the runner's gate accepts."""
    cell_id = _write_cell(tmp_path, "EPISODE.a.s000", items=["i1"])
    sha, census = seal.seal_states(tmp_path, [cell_id], rounds=(1, 2))
    assert census["states_sealed"] == 2

    fork = ArdCellSpec(
        cell_id="FORK.natural.PRIVATE.8B.T1024.train.s000", kind=ArdCellKind.FORK,
        protocol=K.Protocol.PRIVATE, round=None, cohort=K.Cohort.NATURAL,
        checkpoint=K.CHECKPOINT_SMALL, reasoning_allowance=K.THINKING_TOKENS_LOW,
        final_allowance=K.FINAL_TOKENS, items=("i1",), team_id=None, parent_ref=None,
        leader_seed=None, depends_on=(f"state:{sha}",), max_inflight=5, parallel_items=1,
        lane="8B", split="train",
    )
    assert ard_runner.unmet_hard_dependencies(fork, tmp_path) == []

    # Tampering with the register is caught, not silently accepted.
    register = tmp_path / "states" / sha / "REGISTER.json"
    payload = json.loads(register.read_text())
    payload["states"] = []
    register.write_text(json.dumps(payload))
    unmet = ard_runner.unmet_hard_dependencies(fork, tmp_path)
    assert len(unmet) == 1 and unmet[0].status == "tampered"


def test_sealing_nothing_is_refused(tmp_path):
    """A dependency that resolves but contains no parents is worse than an unmet one."""
    cell_id = _write_cell(tmp_path, "EPISODE.a.s000", items=["i1"], rounds=(0,))
    with pytest.raises(ProtocolError, match="empty state register"):
        seal.seal_states(tmp_path, [cell_id], rounds=(1, 2))


def test_a_state_id_that_disagrees_with_itself_is_a_defect(tmp_path):
    """state_id is a content address; two different contents under one id is a harness bug."""
    a = _write_cell(tmp_path, "EPISODE.a.s000", items=["i1"])
    b = tmp_path / "cells" / "EPISODE.a.s001"
    (b / "items").mkdir(parents=True)
    episode = json.loads((tmp_path / "cells" / a / "items" / "i1.json").read_text())
    episode["episodes"][0]["states"][1]["answer_classes"] = ["B"] * 5   # same id, new content
    (b / "items" / "i1.json").write_text(json.dumps(episode))
    (b / "meta.json").write_text("{}")
    with pytest.raises(ProtocolError, match="disagree"):
        seal.collect_states(tmp_path, [a, "EPISODE.a.s001"], rounds=(1, 2))
