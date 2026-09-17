"""Private full EvalPlus MBPP+ grading (agent 05). Offline only; results never reach generation or selection.

Semantics are the pinned EvalPlus ``26d6d00b`` ones, via the vendored comparison code: base and plus batches each
run in a fresh sandbox with fresh module globals; per-input limit ``max(4 s, 4 x reference time)``; batch ceiling
``min(grading.full_suite_wall_seconds, sum(limits)) + 3``; special oracles and float tolerance from upstream
source. Expected outputs, reference code and inputs of other cases never enter the candidate process: the
sandbox receives the program and one input at a time; comparison happens here, in the trusted process.

Status mapping:
* no parseable program -> ``INVALID_OUTPUT`` (semantic_correct None, operational_success False, "no_program"; the
  contract v2 tri-state rule D3);
* any real verdict (pass, wrong output, exception, per-input timeout, batch timeout, CPU limit, memory error,
  crash reproduced after a healthy canary) -> ``GRADED`` with semantic_correct True/False;
* evaluator fault (sandbox cannot start, trusted bootstrap fails, unexplained death with an unhealthy canary, or a
  fault that survives one fresh retry after a canary; private record missing or hash drift) -> ``EVALUATOR_ERROR``
  with semantic_correct None.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agent_allocation._vendor.learnagents_io import content_hash, file_hash
from agent_allocation.contracts import (Candidate, EvalKind, GradeResult, GradeStatus, ParseStatus, PrivateEvalSpec,
                                        canonical_hash)
from agent_allocation.grading.code.profile import GradingProfile, active_profile, environment_digest
from agent_allocation.paths import RunPaths
from agent_allocation.sandbox.evalplus_scorer import (REVISION, SCORER_ID, EvaluatorOutage, Grade, grade_evalplus,
                                                      native_record, pack_native)

GRADER_VERSION = f"{SCORER_ID}+alloc-private-v1"
_RECORD_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_RECORD_CACHE_MAX = 4


class PrivateRecordError(RuntimeError):
    pass


def load_private_record(private: PrivateEvalSpec, paths: RunPaths) -> dict[str, Any]:
    """Load and verify restricted.json + oracle.json for one item (trusted process only)."""
    if not private.private_record_ref:
        raise PrivateRecordError(f"{private.item_id}: no private_record_ref")
    directory = (paths.run_dir / private.private_record_ref).resolve()
    if not directory.is_relative_to(paths.code_oracles_dir.resolve()):
        raise PrivateRecordError(f"{private.item_id}: private record outside the oracle directory")
    key = str(directory)
    cached = _RECORD_CACHE.get(key)
    if cached is not None and cached["private_hash"] == private.private_hash:
        _RECORD_CACHE.move_to_end(key)
        return cached
    try:
        restricted = json.loads((directory / "restricted.json").read_text())
        oracle_path = directory / "oracle.json"
        oracle = json.loads(oracle_path.read_text())
    except (OSError, ValueError) as exc:
        raise PrivateRecordError(f"{private.item_id}: cannot read private record: {type(exc).__name__}") from exc
    row, dataset = native_record(restricted)
    restricted_sha = content_hash(restricted)
    oracle_sha = file_hash(oracle_path)
    expected = canonical_hash({"task_id": row["task_id"], "native_revision": REVISION,
                               "restricted_record_sha256": restricted_sha, "oracle_sha256": oracle_sha})
    if expected != private.private_hash:
        raise PrivateRecordError(f"{private.item_id}: private record hash drift")
    record = {"restricted": restricted, "oracle": oracle, "row": row, "dataset": dataset,
              "private_hash": private.private_hash}
    _RECORD_CACHE[key] = record
    while len(_RECORD_CACHE) > _RECORD_CACHE_MAX:
        _RECORD_CACHE.popitem(last=False)
    return record


def _grade(program: str, record: dict[str, Any], profile: GradingProfile) -> Grade:
    return grade_evalplus(program, record["restricted"], record["oracle"], python_prefix=profile.python_prefix,
                          memory_bytes=profile.memory_bytes, output_bytes=profile.stdout_bytes,
                          stack_bytes=profile.stack_bytes, suite_wall_seconds=profile.suite_wall_seconds,
                          row=record["row"], scratch_bytes=profile.scratch_bytes, file_bytes=profile.file_bytes)


def _canary_ok(profile: GradingProfile) -> bool:
    """Known-good program through the same EvalPlus transport and comparison path."""
    row = {"task_id": "Mbpp/900", "prompt": "", "canonical_solution": "", "entry_point": "canary_add_one",
           "base_input": [[1], [41]], "plus_input": [[-1]], "atol": 0}
    private = {"problem_id": "canary", "native_scorer": "evalplus", "native_record": row}
    oracle = {"restricted_record_sha256": content_hash(private), "native_revision": REVISION, "batches": {
        split: {"inputs_sha256": content_hash(pack_native(row[f"{split}_input"])),
                "expected": [pack_native(x) for x in expected], "reference_times": [0.001] * len(expected)}
        for split, expected in (("base", [2, 42]), ("plus", [0]))}}
    try:
        grade = grade_evalplus("def canary_add_one(x):\n    return x + 1\n", private, oracle,
                               python_prefix=profile.python_prefix, memory_bytes=profile.memory_bytes,
                               output_bytes=profile.stdout_bytes, stack_bytes=profile.stack_bytes,
                               suite_wall_seconds=profile.suite_wall_seconds, row=row,
                               scratch_bytes=profile.scratch_bytes, file_bytes=profile.file_bytes)
    except (EvaluatorOutage, ValueError):
        return False
    return grade.success and not grade.details.get("candidate_crash")


def _detail(grade: Grade) -> str:
    """``pass`` | ``{split}_{failure}``; a seccomp denial seen in candidate errors is appended as
    ``;sandbox_denied=thread|eperm`` (documented limitation: threads/processes/sockets are unavailable)."""
    detail = "pass" if grade.success else "fail"
    if not grade.success:
        for split in ("base", "plus"):
            batch = grade.details[split]
            if not batch["success"]:
                reason = batch["failure_class"] or ("input_timeout" if batch.get("input_timeouts") else "wrong_output")
                detail = f"{split}_{reason}"
                break
    denied = grade.details.get("sandbox_denied") or []
    return detail + (f";sandbox_denied={','.join(denied)}" if denied else "")


def grade_code(candidate: Candidate, private: PrivateEvalSpec, paths: RunPaths) -> GradeResult:
    if private.eval_kind is not EvalKind.EVALPLUS_FULL:
        raise ValueError(f"{private.item_id}: grade_code needs EVALPLUS_FULL, got {private.eval_kind}")
    if candidate.item_id != private.item_id:
        raise ValueError(f"candidate {candidate.candidate_id} belongs to {candidate.item_id}, not {private.item_id}")
    profile = active_profile()
    env = environment_digest(profile)

    def result(status: GradeStatus, correct: bool | None, operational: bool | None, detail: str,
               cpu: float | None) -> GradeResult:
        return GradeResult(candidate_id=candidate.candidate_id, status=status, semantic_correct=correct,
                           operational_success=operational, grader_version=GRADER_VERSION,
                           private_eval_hash=private.private_hash, detail_code=detail, environment_digest=env,
                           cpu_seconds=cpu)

    program = candidate.final_answer
    if candidate.parse_status is not ParseStatus.VALID or program is None or not program.strip():
        return result(GradeStatus.INVALID_OUTPUT, None, False, "no_program", 0.0)
    try:
        record = load_private_record(private, paths)
    except PrivateRecordError:
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "private_record_unavailable", None)
    cpu = 0.0
    try:
        grade = _grade(program, record, profile)
        cpu += grade.details["_resource_usage"]["sandbox_cpu_seconds"]
        if not grade.details.get("candidate_crash"):
            return result(GradeStatus.GRADED, bool(grade.success), True, _detail(grade), cpu)
    except EvaluatorOutage:
        pass
    if not _canary_ok(profile):
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "sandbox_outage_canary_failed", cpu)
    try:
        grade = _grade(program, record, profile)
    except EvaluatorOutage:
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "sandbox_outage_after_retry", cpu)
    cpu += grade.details["_resource_usage"]["sandbox_cpu_seconds"]
    return result(GradeStatus.GRADED, bool(grade.success), True, _detail(grade), cpu)
