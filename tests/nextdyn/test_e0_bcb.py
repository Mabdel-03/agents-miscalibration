"""The BigCodeBench selection repair may never see the official tests (spec 18.1, 7.4).

The atlas's own appendix prescribes this repair: 77% of its code-selection failures were
structural, equivalent programs split across syntactic classes by AST grouping, and grouping
by *behaviour* on generated inputs would make the vote meaningful.  The hazard in doing it is
obvious -- the easiest way to group programs by behaviour is to run the official tests -- and
that would make the repair a measurement of the answer rather than of the selector.

So the rule is a firewall with a seal, not a convention: the input generator sees only the
public ``instruct_prompt``, the sandbox job carries only a candidate program and generated
inputs, and the legacy hidden-test statuses cannot be read at all until the selection has
been sealed to disk.  These tests hold each half.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agents_scaling.nextdyn.e0 import bcb_inputgen, bcb_repair
from agents_scaling.nextdyn.errors import LeakageError

PROMPT = (
    "Write a function that scales a list of numbers.\n"
    "def task_func(values: list, factor: int) -> list:\n"
    '    """Multiply each element of values by factor.\n\n'
    "    Args:\n        values (list): the numbers.\n        factor (int): the multiplier.\n"
    '    Returns:\n        list: the scaled numbers.\n    """\n'
)
PROGRAM = "def task_func(values, factor):\n    return [v * factor for v in values]\n"
#: Every name by which the official tests could reach this code path.  These are checked
#: against code -- imports and non-docstring string literals -- never against prose: the
#: generator's own docstring says it never reads a hidden test, and a text scan would flag it.
FORBIDDEN = ("bcb_tests", "canonical_solution", "study.data.protected",
             "agents_scaling.study.data.protected", "eval/bcb")


def _module_source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def _imported_names(module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(_module_source(module))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _code_strings(module) -> set[str]:
    """Every string literal in the module that is not a docstring."""
    tree = ast.parse(_module_source(module))
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docstrings}


def test_hidden_code_tests_absent_from_generator_and_selection_sandbox():
    """Spec 18.1: no hidden test reaches the input generator, the job payload or the grouping."""
    # 1. The generator's import closure and source never name the protected namespace.
    for module in (bcb_inputgen, bcb_repair):
        imported = _imported_names(module)
        assert not any("protected" in name for name in imported), (module.__name__, imported)
        assert "agents_scaling.benchmarks" not in imported

    for needle in FORBIDDEN:
        for module in (bcb_inputgen, bcb_repair):
            offenders = [text for text in _code_strings(module) if needle in text]
            if module is bcb_repair and needle == "eval/bcb":
                continue   # the audit stage resolves it through the registry, after the seal
            assert not offenders, f"{needle} is reachable from {module.__name__}: {offenders}"

    # 2. The generated inputs are a function of the PUBLIC prompt and the study seed only.
    inputs = bcb_inputgen.generate("BigCodeBench/1", PROMPT, b"\x01" * 32, max_inputs=5)
    assert inputs.eligible and inputs.n_inputs > 0
    assert inputs == bcb_inputgen.generate("BigCodeBench/1", PROMPT, b"\x01" * 32, max_inputs=5)
    assert inputs != bcb_inputgen.generate("BigCodeBench/1", PROMPT, b"\x02" * 32, max_inputs=5)

    # 3. The sandbox job carries the candidate program and those inputs, and nothing else.
    job = bcb_repair.build_job("BigCodeBench/1", "cand-a", PROGRAM, inputs, seed=7)
    payload = job.payload()
    for needle in FORBIDDEN:
        assert needle not in payload, f"{needle} reached the sandbox payload"
    assert job.source == "public_instruct_prompt_only"
    # The payload carries the candidate program as a quoted literal, so assert it round-trips
    # rather than that its raw text appears: what must hold is that the sandbox runs THIS
    # program and that the generated inputs travelled with it.
    embedded = next(line for line in payload.splitlines() if line.startswith("PROGRAM = "))
    assert ast.literal_eval(embedded[len("PROGRAM = "):]) == PROGRAM
    assert all(str(value) in payload for row in inputs.inputs for value in row.values())

    # 4. An ineligible task is refused outright rather than run on guessed inputs.
    ineligible = bcb_inputgen.TaskInputs(task_id="BigCodeBench/2", eligible=False,
                                         reason="signature_unparseable", arguments=(), inputs=())
    with pytest.raises(LeakageError):
        bcb_repair.build_job("BigCodeBench/2", "cand-a", PROGRAM, ineligible, seed=7)


def test_hidden_test_statuses_are_unreadable_until_the_selection_is_sealed(tmp_path):
    """Spec 7.4: official outcomes are an audit of a frozen selection, never an input to it."""
    sources = bcb_repair.R.load()
    with pytest.raises(LeakageError, match="selection is sealed"):
        bcb_repair.read_hidden_status(tmp_path, sources)

    seal = bcb_repair.write_selection_seal(tmp_path, {"BigCodeBench/1": "cand-a"})
    assert seal.exists() and json.loads(seal.read_text(encoding="utf-8"))
    # Now the read is permitted; whether the legacy directory is present is a separate fact and
    # is reported as a typed blocked status rather than as an empty result.
    status = bcb_repair.read_hidden_status(tmp_path, sources)
    assert status["status"] in (bcb_repair.OUTCOME_OK, bcb_repair.R.BLOCKED_MISSING_SOURCE)


def test_a_crash_or_timeout_signature_never_merges_with_a_success():
    """Spec 7.4: programs are not equivalent because they both fell over."""
    ok = [{"index": 0, "outcome": bcb_repair.OUTCOME_OK, "repr": "[2, 4]"}]
    crash_a = [{"index": 0, "outcome": "error", "exception": "ValueError"}]
    crash_b = [{"index": 0, "outcome": "error", "exception": "TypeError"}]
    timeout = [{"index": 0, "outcome": bcb_repair.OUTCOME_TIMEOUT}]
    signatures = {bcb_repair.signature(r) for r in (ok, crash_a, crash_b, timeout)}
    assert len(signatures) == 4, "distinct failure modes collapsed into one behavioural class"
    # A candidate that never produced an output is marked as a crash class, not as a behaviour:
    # "they both fell over" is not evidence that two programs compute the same thing.
    assert bcb_repair.signature(ok).startswith("behaviour:")
    assert all(bcb_repair.signature(r).startswith("crash:") for r in (crash_a, crash_b, timeout))
    groups = bcb_repair.group_candidates({"a": ok, "b": list(ok), "c": crash_a, "d": timeout})
    by_size = sorted(len(members) for members in groups.values())
    assert by_size == [1, 1, 2], groups
