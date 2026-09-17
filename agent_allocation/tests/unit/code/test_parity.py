"""EvalPlus verdict parity: official upstream vs the vendored wrapper, byte-identity of the sandbox transport with
the certified wrapper, and the reference inventory's imports resolving inside the sandbox (nothing dropped)."""
from __future__ import annotations

import ast
import json
import subprocess
import warnings

import pytest

from agent_allocation.benchmarks import mbpp_plus as mp
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.sandbox import evalplus_scorer
from agent_allocation.sandbox.evalplus_validation import EVALPLUS_EXTRAS, fixtures, validate
from agent_allocation.sandbox.sandbox import Sandbox, SandboxLimits

pytestmark = [pytest.mark.sandbox, pytest.mark.needs_sources]

ORIGIN_COMMIT = "2af2879"
UNCHANGED = ("native_functions", "StringTupleRows", "pack_native", "unpack_native", "_invoke_native", "_load_native",
             "native_record")


def _definitions(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {node.name: ast.get_source_segment(source, node) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in UNCHANGED}


def test_sandbox_transport_and_comparison_code_match_the_certified_wrapper():
    origin = subprocess.run(["git", "-C", str(mp.source_root()), "show",
                             f"{ORIGIN_COMMIT}:learnagents/data/evalplus_scorer.py"],
                            capture_output=True, text=True, check=True).stdout
    ours = open(evalplus_scorer.__file__).read()
    assert _definitions(ours) == _definitions(origin)
    assert set(_definitions(ours)) == set(UNCHANGED)


@pytest.mark.skipif(not EVALPLUS_EXTRAS.exists(), reason="pinned evalplus venv (psutil) unavailable")
def test_twenty_fixture_comparison_parity_with_official_untrusted_check():
    profile = profile_mod.active_profile()
    report = validate(python_prefix=profile.python_prefix, memory_bytes=profile.memory_bytes,
                      output_bytes=profile.stdout_bytes, stack_bytes=profile.stack_bytes,
                      suite_wall_seconds=profile.suite_wall_seconds)
    assert report["count"] == len(fixtures()) == 20
    mismatches = [row for row in report["results"] if not row["match"]]
    assert report["status"] == "passed", mismatches


def test_every_reference_import_resolves_inside_the_sandbox(run_paths):
    items, plan = mp.build_inventory(run_paths)
    modules: set[str] = set()
    for item in items:
        from agent_allocation._vendor.evalplus_external import decode_evalplus_record
        native = decode_evalplus_record(plan[item.public.item_id]["restricted"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
            tree = ast.parse(native["prompt"] + native["canonical_solution"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
    assert modules, "reference inventory unexpectedly imports nothing"
    probe = ("import importlib, json\nout = {}\nfor name in " + json.dumps(sorted(modules)) + ":\n"
             "    try:\n        importlib.import_module(name); out[name] = 'ok'\n"
             "    except Exception as exc:\n        out[name] = type(exc).__name__\nprint(json.dumps(out))\n")
    with Sandbox({}, limits=SandboxLimits(cpu_seconds=30, wall_seconds=30, memory_bytes=512 * 1024 ** 2,
                                          stack_bytes=512 * 1024 ** 2)) as sandbox:
        result = sandbox.execute(probe)
    observed = json.loads(result["stdout"].strip().splitlines()[-1])
    assert {name: status for name, status in observed.items() if status != "ok"} == {}
