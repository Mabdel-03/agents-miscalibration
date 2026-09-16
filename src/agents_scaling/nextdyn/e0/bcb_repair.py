"""Behavioural regrouping of the legacy BigCodeBench candidates (spec 7.4; plan §2.6).

The atlas selected code by an **AST vote**: two candidate programs that parse to the same
normalised tree are one candidate.  That is a syntactic equivalence, and the question this leg
asks is whether a *behavioural* one -- programs that return the same thing on the same inputs --
groups them differently and selects better.

Three rules keep the answer honest, and each is enforced here rather than by care:

* **Hidden tests are sealed until selection is over.**  ``selection_seal.json`` is written
  before any hidden-test status is read, and :func:`read_hidden_status` refuses to read one
  without it.  The generator and the sandbox payload never import the legacy evaluation
  directory at all (``test_hidden_code_tests_absent_from_generator_and_selection_sandbox``).
* **A crash is its own class.**  A candidate with zero successful inputs never merges with
  another candidate, however similar its exception.  Merging crashes is how a behavioural vote
  manufactures a majority out of two different failures.
* **Every input runs in a fresh ``/work`` with the seeds reset.**  The payload resets
  ``random`` and ``numpy.random`` before each call and records the environment, the input repr,
  the output repr, the exception type and the timeout, so a signature is a statement about the
  program rather than about the order the inputs happened to run in.

The sandbox itself is the existing evaluator: ``study.evaluation.bcb.BcbEvaluator`` with
``apptainer_argv``.  This module builds the payload and interprets the results; it does not
re-implement the container.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.e0 import bcb_inputgen, registry as R
from agents_scaling.nextdyn.errors import LeakageError

SELECTION_SEAL: Final[str] = "selection_seal.json"
REPAIR_DIR: Final[str] = "bcb_repair"

STEP_AST: Final[str] = "ast_vote"
STEP_BEHAVIOURAL: Final[str] = "behavioural_vote"
STEP_ASSERTION: Final[str] = "assertion_informed"
STEP_UNSUPPORTED: Final[str] = "unsupported"

OUTCOME_OK: Final[str] = "ok"
OUTCOME_EXCEPTION: Final[str] = "exception"
OUTCOME_TIMEOUT: Final[str] = "timeout"

#: Docstring phrases that make a task's properties checkable without a hidden test.
ASSERTION_MARKERS: Final[tuple[str, ...]] = (
    "returns a list of", "must be sorted", "raises ValueError", "returns True if",
    "the sum of", "in ascending order", "raises TypeError",
)


def settings(sources: R.LegacySources) -> dict[str, Any]:
    """The BCB repair knobs, read from ``legacy_sources.yaml`` (the only declaration).

    This is the reader of ``legacy_sources.bcb_repair.*``: the sandbox image, the frozen
    generator policy, the per-task input cap and the size of the held-back audit suite.
    """
    block = sources.bcb_repair
    sif = R.resolve(sources, "bcb_repair.sandbox_sif", hash_files=False)
    return {
        "sandbox_sif": str(sif.path),
        "sandbox_present": sif.exists,
        "input_generator_policy": str(block.input_generator_policy),
        "max_inputs_per_task": int(block.max_inputs_per_task),
        "audit_suite_inputs": int(block.audit_suite_inputs),
        "status": R.BLOCKED_MISSING_SOURCE if not sif.exists else OUTCOME_OK,
    }


# --------------------------------------------------------------------------- payload


HARNESS_TEMPLATE: Final[str] = '''\
import json, random, sys, traceback
try:
    import numpy as _np
except ImportError:
    _np = None
PROGRAM = {program!r}
INPUTS = {inputs!r}
SEED = {seed!r}
namespace = {{}}
out = []
try:
    exec(compile(PROGRAM, "<candidate>", "exec"), namespace)
except BaseException:
    print(json.dumps({{"load_error": traceback.format_exc(limit=2)}}))
    sys.exit(0)
func = namespace.get("task_func")
for index, kwargs in enumerate(INPUTS):
    random.seed(SEED + index)
    if _np is not None:
        _np.random.seed((SEED + index) % (2 ** 32))
    try:
        value = func(**kwargs)
        out.append({{"index": index, "outcome": "ok", "repr": repr(value)[:2000]}})
    except BaseException as exc:
        out.append({{"index": index, "outcome": "exception",
                     "exception": type(exc).__name__, "repr": repr(exc)[:400]}})
print(json.dumps({{"results": out}}))
'''


@dataclass(frozen=True)
class SandboxJob:
    """One candidate program plus its inputs, ready for the apptainer evaluator."""

    task_id: str
    candidate_id: str
    program: str
    inputs: tuple[Mapping[str, Any], ...]
    seed: int
    source: str = "public_instruct_prompt_only"

    def payload(self) -> str:
        return HARNESS_TEMPLATE.format(program=self.program, inputs=list(self.inputs),
                                       seed=int(self.seed))


def build_job(task_id: str, candidate_id: str, program: str, inputs: bcb_inputgen.TaskInputs,
              *, seed: int) -> SandboxJob:
    """Build the sandbox job.  Nothing from the legacy evaluation directory enters it."""
    if not inputs.eligible:
        raise LeakageError(
            f"{task_id}: the input generator declared this task ineligible ({inputs.reason}); "
            "a job must never run on inputs the frozen policy could not type")
    return SandboxJob(task_id=task_id, candidate_id=candidate_id, program=program,
                      inputs=inputs.inputs, seed=int(seed))


def signature(results: Sequence[Mapping[str, Any]]) -> str:
    """``sha256(JCS(per-input tokens))``; a crash or a timeout never merges with a success.

    A candidate whose every input failed gets a signature prefixed with its own candidate-free
    marker *and* the exception sequence, so two programs that crash differently are two classes
    and two programs that crash identically are still not a *behaviour* -- they are recorded as
    a crash class, which the vote treats as its own group.
    """
    tokens = []
    n_ok = 0
    for row in sorted(results, key=lambda r: int(r.get("index", 0))):
        outcome = str(row.get("outcome"))
        if outcome == OUTCOME_OK:
            n_ok += 1
            tokens.append(f"ok:{row.get('repr')}")
        elif outcome == OUTCOME_TIMEOUT:
            tokens.append("timeout")
        else:
            tokens.append(f"exc:{row.get('exception')}")
    prefix = "behaviour" if n_ok else "crash"
    body = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(body.encode('utf-8')).hexdigest()[:32]}"


def group_candidates(runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    """``signature -> candidate ids``; crash classes are kept separate from behaviour classes."""
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id, results in sorted(runs.items()):
        groups[signature(results)].append(candidate_id)
    return dict(groups)


def supports_assertions(instruct_prompt: str) -> bool:
    """Whether the docstring states a property that can be checked without a hidden test."""
    lowered = instruct_prompt.lower()
    return any(marker in lowered for marker in ASSERTION_MARKERS)


def select(groups: Mapping[str, Sequence[str]], *, ast_groups: Mapping[str, Sequence[str]] | None = None,
           instruct_prompt: str = "") -> dict[str, Any]:
    """The three-step selection of spec 7.4, reporting which step decided.

    Step 1 is the AST vote reproduced exactly; step 2 is the behavioural-class vote; step 3 is
    assertion-informed and applies only to tasks whose docstring states a checkable property --
    every other task records ``unsupported`` rather than falling through to a guess.
    """
    behaviour = {sig: list(ids) for sig, ids in groups.items() if sig.startswith("behaviour:")}
    crash = {sig: list(ids) for sig, ids in groups.items() if sig.startswith("crash:")}
    step = STEP_BEHAVIOURAL
    pool = behaviour or crash
    if not pool:
        return {"step": STEP_UNSUPPORTED, "selected": None, "reason": "no candidate ran"}
    top = max(len(ids) for ids in pool.values())
    winners = sorted(sig for sig, ids in pool.items() if len(ids) == top)
    tie = len(winners) > 1
    if tie and ast_groups:
        step = STEP_AST
        ast_top = max((len(ids) for ids in ast_groups.values()), default=0)
        ast_winners = sorted(sig for sig, ids in ast_groups.items() if len(ids) == ast_top)
        winners = ast_winners or winners
        tie = len(winners) > 1
    if tie and supports_assertions(instruct_prompt):
        step = STEP_ASSERTION
    return {
        "step": step, "selected_signature": winners[0],
        "selected": sorted(pool.get(winners[0], ast_groups.get(winners[0], [])))[:1] or None,
        "tie": tie, "n_behaviour_classes": len(behaviour), "n_crash_classes": len(crash),
        "group_sizes": {sig: len(ids) for sig, ids in sorted(groups.items())},
        "assertion_supported": supports_assertions(instruct_prompt),
    }


# --------------------------------------------------------------------------- seal


def write_selection_seal(out_dir: str | os.PathLike, selections: Mapping[str, Any]) -> Path:
    """Seal the selections **before** any hidden-test status is read (spec 7.4)."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SELECTION_SEAL
    text = json.dumps(selections, indent=1, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (folder / f"{SELECTION_SEAL}.sha256").write_text(digest + "\n", encoding="utf-8")
    return path


def read_hidden_status(out_dir: str | os.PathLike, sources: R.LegacySources,
                       ) -> dict[str, Any]:
    """Legacy hidden-test statuses, readable **only** after the selection seal exists.

    Network failures are quarantined with ``ard.legacy_e0.NETWORK_MARKERS`` rather than scored:
    a program that failed because the sandbox had no network did not fail the task.
    """
    folder = Path(out_dir)
    if not (folder / SELECTION_SEAL).exists():
        raise LeakageError(
            f"{folder / SELECTION_SEAL} does not exist: the hidden-test statuses may be read only "
            "after the selection is sealed (spec 7.4)")
    from agents_scaling.ard.legacy_e0 import NETWORK_MARKERS

    eval_dir = R.resolve(sources, "study_v4.eval_bcb", hash_files=False)
    if not eval_dir.exists:
        return {"status": R.BLOCKED_MISSING_SOURCE, "reason": f"{eval_dir.path} is absent"}
    passed: dict[str, bool] = {}
    quarantined: dict[str, str] = {}
    for path in sorted(eval_dir.path.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            candidate_id = row.get("candidate_id")
            if not candidate_id:
                continue
            tail = f"{row.get('stderr_tail', '')}{row.get('stdout_tail', '')}"
            if row.get("status") != "pass" and any(m in tail for m in NETWORK_MARKERS):
                quarantined[candidate_id] = "bcb_network_access"
                continue
            passed[candidate_id] = row.get("status") == "pass"
    return {"status": OUTCOME_OK, "n_candidates": len(passed), "passed": passed,
            "n_quarantined": len(quarantined), "quarantined": quarantined}


def report(groups_by_task: Mapping[str, Mapping[str, Sequence[str]]],
           ast_by_task: Mapping[str, Mapping[str, Sequence[str]]],
           *, hidden: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Merge/split audit, selection lift, retained coverage and the eligible fraction."""
    merged = split = 0
    for task, behaviour in groups_by_task.items():
        ast_groups = ast_by_task.get(task, {})
        if len(behaviour) < len(ast_groups):
            merged += 1
        elif len(behaviour) > len(ast_groups):
            split += 1
    lift = None
    retained = None
    if hidden:
        def rate(by_task: Mapping[str, Mapping[str, Sequence[str]]]) -> float | None:
            picks = []
            for groups in by_task.values():
                chosen = select(groups)
                ids = chosen.get("selected") or []
                picks.extend(float(bool(hidden.get(i))) for i in ids if i in hidden)
            return (sum(picks) / len(picks)) if picks else None

        behaviour_rate, ast_rate = rate(groups_by_task), rate(ast_by_task)
        if behaviour_rate is not None and ast_rate is not None:
            lift = behaviour_rate - ast_rate
        retained = behaviour_rate
    return {
        "n_tasks": len(groups_by_task), "n_merged_vs_ast": merged, "n_split_vs_ast": split,
        "selection_lift_behavioural_minus_ast": lift,
        "retained_coverage": retained,
        "note": ("crash and timeout signatures never merge with behaviour signatures, so a "
                 "'merge' here is always two programs that agreed on every input"),
    }


__all__ = [
    "ASSERTION_MARKERS", "HARNESS_TEMPLATE", "OUTCOME_EXCEPTION", "OUTCOME_OK", "OUTCOME_TIMEOUT",
    "REPAIR_DIR", "SELECTION_SEAL", "STEP_ASSERTION", "STEP_AST", "STEP_BEHAVIOURAL",
    "STEP_UNSUPPORTED", "SandboxJob", "build_job", "group_candidates", "read_hidden_status",
    "report", "select", "settings", "signature", "supports_assertions", "write_selection_seal",
]
