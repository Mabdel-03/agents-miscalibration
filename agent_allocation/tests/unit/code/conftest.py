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


GOOD_PROGRAM = "def f(x):\n    return x + 1\n"
SYNTHETIC_PROMPT = '"""\nWrite a function f that returns its argument plus one.\nassert f(1) == 2\n"""\n'
EXPECTED_CANARY = "EXPECTED-OUTPUT-CANARY-7f3a91"


def install_synthetic_task(run_paths: RunPaths, *, task_id: str = "Mbpp/900", entry: str = "f",
                           base: tuple[list, list] = ([[1]], [2]), plus: tuple[list, list] = ([[41]], [42]),
                           prompt: str = SYNTHETIC_PROMPT, examples: tuple[str, ...] | None = None):
    """Install a tiny private record (one input per split by default) and return (PublicTask, PrivateEvalSpec).

    Per-input limits are EvalPlus's 4 s floor, so a timing-out candidate costs ~4 s per input, not a minute.
    """
    from agent_allocation._vendor.learnagents_io import content_hash, file_hash, write_json
    from agent_allocation.contracts import EvalKind, PrivateEvalSpec, PublicTask, canonical_hash
    from agent_allocation.sandbox.evalplus_scorer import REVISION, pack_native

    row = {"task_id": task_id, "prompt": prompt, "canonical_solution": "", "entry_point": entry,
           "base_input": base[0], "plus_input": plus[0], "atol": 0, "assertion": "assert private_only(1) == 1"}
    restricted = {"problem_id": "synthetic_" + task_id.replace("/", "_"), "native_scorer": "evalplus",
                  "native_record": row}
    oracle = {"restricted_record_sha256": content_hash(restricted), "native_revision": REVISION,
              "task_id": task_id, "batches": {
                  split: {"inputs_sha256": content_hash(pack_native(inputs)),
                          "expected": [pack_native(x) for x in expected], "reference_times": [0.001] * len(expected)}
                  for split, (inputs, expected) in (("base", base), ("plus", plus))}}
    directory = run_paths.code_oracles_dir / task_id.replace("/", "_")
    write_json(directory / "restricted.json", restricted)
    write_json(directory / "oracle.json", oracle)
    item_id = mp.item_id_for(task_id)
    private_hash = canonical_hash({"task_id": task_id, "native_revision": REVISION,
                                   "restricted_record_sha256": content_hash(restricted),
                                   "oracle_sha256": file_hash(directory / "oracle.json")})
    public = PublicTask(item_id=item_id, benchmark="mbpp_plus", source_revision="synthetic", prompt=prompt,
                        output_kind=OutputKind.PYTHON,
                        public_examples=examples if examples is not None else (mp.parse_public_assert(prompt),))
    private = PrivateEvalSpec(item_id=item_id, benchmark="mbpp_plus", source_revision="synthetic",
                              eval_kind=EvalKind.EVALPLUS_FULL, gold_answer=None,
                              private_record_ref=f"private/evalplus_oracles/{task_id.replace('/', '_')}",
                              private_hash=private_hash)
    return public, private
