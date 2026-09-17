"""Shared fixtures for agent 05 tests: a tiny run dir with real MBPP+ items installed, and candidate builders."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_allocation.benchmarks import mbpp_plus as mp
from agent_allocation.contracts import Candidate, OutputKind, ParseStatus
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.paths import RunPaths

SMALL_TASKS = ("Mbpp/2", "Mbpp/3", "Mbpp/6")

HAVE_SANDBOX = all(shutil.which(tool) for tool in ("bwrap", "prlimit")) and \
    Path(profile_mod.CERTIFIED_PYTHON_PREFIX, "bin", "python").exists()
HAVE_SOURCES = (mp.source_root() / mp.FROZEN_DIR / "manifest.json").exists()


def pytest_collection_modifyitems(config, items):
    skip_sandbox = pytest.mark.skip(reason="bwrap/prlimit/certified interpreter unavailable")
    skip_sources = pytest.mark.skip(reason="frozen MBPP+ sources unavailable")
    for item in items:
        if "sandbox" in item.keywords and not HAVE_SANDBOX:
            item.add_marker(skip_sandbox)
        if "needs_sources" in item.keywords and not HAVE_SOURCES:
            item.add_marker(skip_sources)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_sources: requires the frozen MBPP+ sources of agent_methods_bench")


@pytest.fixture(autouse=True)
def _default_profile():
    profile_mod.configure(profile_mod.profile_from_config(json.loads(profile_mod.DEFAULT_CONFIG.read_text())))
    yield


@pytest.fixture
def run_paths(tmp_path) -> RunPaths:
    return RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")


@pytest.fixture
def small_inventory(run_paths):
    """(items_by_task_id, plan) for SMALL_TASKS with their private records installed in the run dir."""
    items, plan = mp.build_inventory(run_paths)
    chosen = {}
    for item in items:
        task_id = plan[item.public.item_id]["task_id"]
        if task_id in SMALL_TASKS:
            mp.install_private_record(plan[item.public.item_id])
            chosen[task_id] = item
    assert set(chosen) == set(SMALL_TASKS)
    return chosen, plan


def make_candidate(item_id: str, program: str | None, *, node: str = "ab" * 32,
                   status: ParseStatus = ParseStatus.VALID) -> Candidate:
    return Candidate(candidate_id=node, node_id=node, item_id=item_id, visible_content=program or "",
                     final_answer=program, rationale=None, parse_status=status, finish_reason="stop",
                     output_kind=OutputKind.PYTHON)


def reference_program(plan: dict, item_id: str) -> str:
    from agent_allocation._vendor.evalplus_external import decode_evalplus_record
    native = decode_evalplus_record(plan[item_id]["restricted"])
    return native["prompt"] + native["canonical_solution"]
