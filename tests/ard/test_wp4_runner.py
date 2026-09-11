"""WP4 dispatch: the hard dependency gate, resume, the injected task loader, FLOP wiring.

Spec §10.4 (resumable idempotent workers; ``meta.json`` last), §12.3 (a GATE cell refuses to
run without the frozen controller), §18 (cache identity).  Plan §10: *"fork/suffix state
reads and ``state:``/``controller:`` gating (a GATE cell run with an absent or tampered
``controller/<sha>/MODEL.json`` exits ``EXIT_DEPS_UNMET`` with no ``meta.json`` and zero
requests generated)"*, and the resume assertion *"delete items + meta.json, rerun, HTTP call
count unchanged"*.
"""

from __future__ import annotations

import ast
import json
import shutil
import threading
from pathlib import Path

import pytest

from agents_scaling.ard import cells as ard_cells
from agents_scaling.ard import contract as K
from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard.types import (
    EXIT_DEPS_UNMET,
    EXIT_DONE,
    EXIT_INCOMPLETE,
    EXIT_NO_SERVER,
    EXIT_SUSPENDED,
    ArdCellKind,
)
from tests.ard.wp4_support import (
    freeze_minimal_controller,
    fake_server,
    make_run_root,
    runner_kwargs,
    synthetic_tasks,
    write_items,
)


@pytest.fixture
def run_root(tmp_path):
    root = make_run_root(tmp_path)
    write_items(root, synthetic_tasks(1, splits=("test",)))
    return root


def _tasks(run_root):
    from agents_scaling.ard.items import load_ard_tasks

    return list(load_ard_tasks(run_root).values())


def _episode_cell(run_root, *, protocol=K.Protocol.PRIVATE, depends_on=(), parent_ref=None):
    builder = ard_cells.ArdBuilder()
    cells = builder.emit(ArdCellKind.EPISODE, _tasks(run_root), cohort=K.Cohort.NATURAL,
                         checkpoint=K.CHECKPOINT_SMALL, protocol=protocol,
                         depends_on=depends_on, parent_ref=parent_ref)
    return cells[0]


def _run(cell, run_root, **overrides):
    return ard_runner.run_ard_cell(cell, run_root, run_root, 0, threading.Event(),
                                   **runner_kwargs(**overrides))


# --------------------------------------------------------------------------- exit codes


def test_exit_deps_unmet_is_a_new_distinct_code():
    assert EXIT_DEPS_UNMET == 5
    assert len({EXIT_DONE, EXIT_NO_SERVER, EXIT_INCOMPLETE, EXIT_SUSPENDED, EXIT_DEPS_UNMET}) == 5


# --------------------------------------------------------------------------- the gate


def test_a_gate_cell_without_its_controller_exits_deps_unmet_with_no_meta_and_no_requests(run_root):
    cell = _episode_cell(run_root, protocol=K.Protocol.GATE,
                         depends_on=(f"controller:{'ab' * 32}",), parent_ref=f"controller:{'ab' * 32}")
    with fake_server(run_root) as server:
        before = server.chat_count
        code = _run(cell, run_root)
        assert code == EXIT_DEPS_UNMET
        assert server.chat_count == before          # zero requests generated
    cell_dir = run_root / "cells" / cell.cell_id
    assert not (cell_dir / "meta.json").exists()
    assert not list((cell_dir / "items").glob("*.json"))
    unmet = json.loads((cell_dir / ard_runner.DEPS_UNMET_FILE).read_text())
    assert unmet["exit_code"] == EXIT_DEPS_UNMET
    assert unmet["unmet"][0]["status"] == "missing"


def test_a_tampered_controller_file_is_refused_exactly_like_an_absent_one(run_root):
    sha = freeze_minimal_controller(run_root)
    path = ard_runner.register_path(run_root, "controller", sha)
    payload = json.loads(path.read_text())
    payload["w"] = [1, 3]                       # same declared hash, different content
    path.write_text(json.dumps(payload))
    cell = _episode_cell(run_root, protocol=K.Protocol.GATE, depends_on=(f"controller:{sha}",))
    with fake_server(run_root) as server:
        before = server.chat_count
        assert _run(cell, run_root) == EXIT_DEPS_UNMET
        assert server.chat_count == before
    cell_dir = run_root / "cells" / cell.cell_id
    assert not (cell_dir / "meta.json").exists()
    unmet = json.loads((cell_dir / ard_runner.DEPS_UNMET_FILE).read_text())
    assert unmet["unmet"][0]["status"] == "tampered"


def test_a_state_dependency_is_gated_the_same_way(run_root):
    cell = _episode_cell(run_root, depends_on=(f"state:{'cd' * 32}",))
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DEPS_UNMET
    statuses = ard_runner.unmet_hard_dependencies(cell, run_root)
    assert [status.kind for status in statuses] == ["state"]


def test_a_met_dependency_admits_the_cell_and_clears_the_waiting_marker(run_root):
    sha = ard_runner.freeze_register(run_root, "state", {"schema": "v1", "states": []})
    cell = _episode_cell(run_root, protocol=K.Protocol.PEER, depends_on=(f"state:{sha}",))
    (run_root / "cells" / cell.cell_id).mkdir(parents=True, exist_ok=True)
    (run_root / "cells" / cell.cell_id / ard_runner.DEPS_UNMET_FILE).write_text("{}")
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE
    assert (run_root / "cells" / cell.cell_id / "meta.json").exists()
    assert not (run_root / "cells" / cell.cell_id / ard_runner.DEPS_UNMET_FILE).exists()


def test_gate_passes_the_gate_but_refuses_to_invent_a_protection_choice(run_root):
    """The controller file being present is not the same as WP5's chooser being available.

    Spec §12.3: GATE's protected agent is the frozen controller's argmax.  With the register
    in place the cell is admitted (no ``DEPS_UNMET``), and then refuses loudly rather than
    protecting an arbitrary agent — unless ``ard/controller.py`` (WP5) is importable, in
    which case it runs.
    """
    sha = freeze_minimal_controller(run_root)
    cell = _episode_cell(run_root, protocol=K.Protocol.GATE, depends_on=(f"controller:{sha}",))
    chooser_available = ard_runner.resolve_controller_chooser(run_root, cell) is not None
    with fake_server(run_root):
        code = _run(cell, run_root)
    cell_dir = run_root / "cells" / cell.cell_id
    assert not (cell_dir / ard_runner.DEPS_UNMET_FILE).exists()   # the gate itself passed
    if chooser_available:
        assert code == EXIT_DONE
    else:
        assert code == EXIT_SUSPENDED
        assert "controller" in json.loads((cell_dir / "SUSPENDED.json").read_text())["error"]


def test_advisory_cell_dependencies_do_not_gate(run_root):
    cell = _episode_cell(run_root, depends_on=("cell:some.other.cell.s000",))
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE


# --------------------------------------------------------------------------- run and resume


def test_an_episode_cell_runs_publishes_and_resumes_without_new_http_calls(run_root):
    cell = _episode_cell(run_root, protocol=K.Protocol.PEER)
    with fake_server(run_root) as server:
        assert _run(cell, run_root) == EXIT_DONE
        after_first = server.chat_count
        assert after_first > 0
        cell_dir = run_root / "cells" / cell.cell_id
        assert (cell_dir / "meta.json").exists()
        assert len(list((cell_dir / "items").glob("*.json"))) == len(cell.items)

        shutil.rmtree(cell_dir / "items")
        (cell_dir / "meta.json").unlink()
        assert _run(cell, run_root) == EXIT_DONE
        assert server.chat_count == after_first     # every request aliased in the store
    assert (run_root / "cells" / cell.cell_id / "meta.json").exists()


def test_the_task_loader_is_injected_so_ard_splits_are_accepted(run_root):
    """``study/runner.py`` calls ``load_public_tasks``, which rejects ARD splits (contract §3)."""
    from agents_scaling.study.data.public import load_public_tasks

    with pytest.raises(Exception):
        load_public_tasks(run_root)
    cell = _episode_cell(run_root)
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE


def test_a_missing_item_is_still_refused(run_root):
    cell = _episode_cell(run_root)
    broken = ard_cells.ArdCellSpec.from_dict({**cell.to_dict(), "items": ["not-an-item"]})
    with fake_server(run_root):
        assert _run(broken, run_root) == EXIT_SUSPENDED
    assert (run_root / "cells" / broken.cell_id / "SUSPENDED.json").exists()


# --------------------------------------------------------------------------- episode shape


def _episodes(run_root, cell):
    payloads = [json.loads(path.read_text())
                for path in sorted((run_root / "cells" / cell.cell_id / "items").glob("*.json"))]
    return [episode for payload in payloads for episode in payload["episodes"]]


def test_every_episode_runs_five_agents_in_all_three_rounds_with_no_early_stop(run_root):
    cell = _episode_cell(run_root, protocol=K.Protocol.PEER)
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE
    for episode in _episodes(run_root, cell):
        per_round = {}
        for decision in episode["decisions"]:
            per_round.setdefault(decision["round"], []).append(decision)
        assert sorted(per_round) == list(K.ROUNDS)
        for round_ in range(1, K.UPDATE_ROUNDS + 1):
            assert len(per_round[round_]) == K.N_AGENTS
        assert episode["horizon"] == K.UPDATE_ROUNDS and episode["N"] == K.N_AGENTS


def test_a_private_round_carries_an_empty_packet_and_a_protection_arm_protects_one_slot(run_root):
    """RANDOM_PROTECT derives its own recorded schedule, so it needs no controller (spec §12.3)."""
    private = _episode_cell(run_root, protocol=K.Protocol.PRIVATE)
    protect = _episode_cell(run_root, protocol=K.Protocol.RANDOM_PROTECT)
    with fake_server(run_root):
        assert _run(private, run_root) == EXIT_DONE
        assert _run(protect, run_root) == EXIT_DONE
    for episode in _episodes(run_root, private):
        assert all(decision["n_packets"] == 0 for decision in episode["decisions"])
    for episode in _episodes(run_root, protect):
        for round_ in range(1, K.UPDATE_ROUNDS + 1):
            refs = [d for d in episode["decisions"] if d["round"] == round_]
            protected = [d for d in refs if d["protected"]]
            assert len(protected) == K.AGENTS_PROTECTED_PER_ROUND
            assert protected[0]["n_packets"] == 0
            assert all(d["n_packets"] == K.N_AGENTS - 1 for d in refs if not d["protected"])


def test_every_decision_carries_a_populated_flop_block(run_root):
    """Contract §2: without the oracle wiring ``flops`` is the ``None`` placeholder."""
    from agents_scaling.ard.resources import assert_flops_populated

    cell = _episode_cell(run_root)
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE
    for episode in _episodes(run_root, cell):
        for decision in episode["decisions"]:
            assert_flops_populated(decision["flops"], where=decision["decision_id"])
            assert decision["flops"]["total"] > 0


def test_roots_alias_the_frozen_low_budget_bank_even_in_a_high_budget_cell(run_root):
    """Spec §10 E2d: no root request is ever issued at 4,096 thinking tokens."""
    builder = ard_cells.ArdBuilder()
    high = builder.emit(ArdCellKind.EPISODE, _tasks(run_root), cohort=K.Cohort.HIGH_BUDGET,
                        checkpoint=K.CHECKPOINT_SMALL, protocol=K.Protocol.PRIVATE,
                        reasoning_allowance=K.THINKING_TOKENS_HIGH)[0]
    with fake_server(run_root):
        assert _run(high, run_root) == EXIT_DONE
    for episode in _episodes(run_root, high):
        roots = [d for d in episode["decisions"] if d["round"] == 0]
        assert roots and all(d["budget_tokens"] == K.ROOT_THINKING_TOKENS for d in roots)
        updates = [d for d in episode["decisions"] if d["round"] > 0]
        assert updates and all(d["budget_tokens"] == K.THINKING_TOKENS_HIGH for d in updates)


def test_the_baseline_executor_stamps_every_row_it_writes(run_root):
    """Until WP2's ``ard/episode.py`` lands the loop is WP4's dispatch shape, and says so."""
    executor, is_wp2 = ard_runner.resolve_episode_executor()
    cell = _episode_cell(run_root)
    with fake_server(run_root):
        assert _run(cell, run_root) == EXIT_DONE
    for episode in _episodes(run_root, cell):
        if is_wp2:
            assert not episode["packet_policy"].startswith("fallback:")
        else:
            assert executor is ard_runner.baseline_episode_executor
            assert episode["packet_policy"] == f"fallback:{cell.protocol.value}"
            assert all(decision["fallback"] for decision in episode["decisions"])


# --------------------------------------------------------------------------- workers / firewall


def test_every_kind_resolves_to_a_worker_or_names_its_missing_owner():
    from agents_scaling.ard.kinds import KIND_TABLE

    for kind, spec in KIND_TABLE.items():
        try:
            worker = ard_runner.resolve_worker(kind)
        except Exception as exc:  # noqa: BLE001 - the message is the contract
            assert spec.owner != "WP4", f"{kind} is WP4's and must resolve: {exc}"
            assert spec.worker.split(":")[0] in str(exc)
        else:
            assert callable(worker)


def test_the_runner_module_level_import_closure_never_reaches_gold():
    """The evaluator's gold reader is imported inside ``grade_worker``, never at module level."""
    forbidden = {"agents_scaling.ard.graders", "agents_scaling.ard.items_build", "datasets",
                 "agents_scaling.benchmarks.loaders"}
    tree = ast.parse(Path(ard_runner.__file__).read_text())
    for node in tree.body:                      # module level only
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module, f"{node.module}.{node.names[0].name}"]
        for target in targets:
            assert target not in forbidden, f"ard/runner.py imports {target} at module level"
