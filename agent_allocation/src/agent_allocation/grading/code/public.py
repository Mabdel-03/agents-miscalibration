"""Public-example checks for MBPP+ terminal selection (agent 05).

Each candidate gets the same checks: the official prompt-visible ``assert`` statements of ``task.public_examples``
(never private base/plus tests), executed in a fresh OS sandbox under the frozen profile
(``grading.public_case_wall_seconds`` per case, ``memory_mib_per_sandbox``, ``stdout_limit_bytes``). The candidate
module is pre-populated with ``math`` and ``sys``. Outcomes per case, in official order: ``pass`` | ``fail``
(AssertionError) | ``timeout`` | ``error`` (any other exception, program load failure, or a reproduced crash).

Infrastructure faults (sandbox cannot start, trusted bootstrap breaks, or an unexplained sandbox death) are retried
once in a fresh sandbox after a known-good canary; if that fails the result is ``CheckStatus.INFRASTRUCTURE_ERROR``
and the selector pauses. A limit hit (wall timer, CPU rlimit) is a real ``timeout`` outcome, never an outage.
Pass/fail results never reach generation; they feed only the terminal selector.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from agent_allocation.contracts import (CheckStatus, OutputKind, ParseStatus, PublicCheckResult, PublicTask,
                                        Candidate, canonical_hash)
from agent_allocation.grading.code.profile import GradingProfile, active_profile, environment_digest
from agent_allocation.sandbox.evalplus_scorer import SIGXCPU_EXIT, EvaluatorOutage
from agent_allocation.sandbox.public_runner import PUBLIC_TRANSPORT
from agent_allocation.sandbox.sandbox import (PROCESS_EXITED, WALL_TIMEOUT, Sandbox, SandboxError,
                                              SandboxLimits)

CHECKER_VERSION = "mbpp_plus-public-assert-v1"
PREIMPORTS = ("math", "sys")
CANARY_PROGRAM = "def canary_add_one(x):\n    return x + 1\n"
CANARY_ASSERT = "assert canary_add_one(1) == 2"


def code_sha256(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8", "surrogatepass")).hexdigest()


def public_inputs_sha256(task: PublicTask) -> str:
    return canonical_hash({"checker_version": CHECKER_VERSION, "item_id": task.item_id,
                           "source_revision": task.source_revision, "public_examples": list(task.public_examples),
                           "preimports": list(PREIMPORTS)})


def _limits(profile: GradingProfile, n_cases: int) -> SandboxLimits:
    total = profile.public_case_wall_seconds * (n_cases + 1)
    return SandboxLimits(cpu_seconds=math.ceil(total) + 1, wall_seconds=profile.public_case_wall_seconds + 2.0,
                         memory_bytes=profile.memory_bytes, output_bytes=profile.stdout_bytes,
                         file_bytes=profile.file_bytes, stack_bytes=profile.stack_bytes,
                         max_protocol_bytes=1024 * 1024)


def _run_once(program: str, examples: Sequence[str], profile: GradingProfile) -> tuple[list[str], float, bool]:
    """One fresh sandbox. Returns (outcomes, cpu_seconds, crashed); raises EvaluatorOutage for evaluator faults."""
    outcomes: list[str] = []
    cpu = 0.0
    crashed = False
    wall = profile.public_case_wall_seconds
    try:
        with Sandbox({}, python_prefix=profile.python_prefix, limits=_limits(profile, len(examples))) as sandbox:
            boot = sandbox.execute(PUBLIC_TRANSPORT, phase="trusted")
            cpu += boot.get("_resource_usage", {}).get("sandbox_cpu_seconds", 0.0)
            if not boot.get("ok"):
                raise EvaluatorOutage("public checker bootstrap failed", kind="bootstrap_failed", phase="trusted")
            loaded = sandbox.invoke({"source": program, "timeout": wall}, "_load_public")
            cpu += loaded.get("_resource_usage", {}).get("sandbox_cpu_seconds", 0.0)
            load_outcome = _outcome(loaded, loading=True)
            if load_outcome != "loaded":
                return [load_outcome] * len(examples), cpu, False
            for example in examples:
                result = sandbox.invoke({"assert": example, "timeout": wall}, "_run_public_case")
                cpu += result.get("_resource_usage", {}).get("sandbox_cpu_seconds", 0.0)
                outcomes.append(_outcome(result, loading=False))
    except SandboxError as error:
        if error.phase != "candidate":
            raise EvaluatorOutage(f"public sandbox failed outside candidate code: {error}", kind=error.kind,
                                  phase=error.phase, returncode=error.returncode) from error
        if error.kind == WALL_TIMEOUT or (error.kind == PROCESS_EXITED and error.returncode == SIGXCPU_EXIT):
            fill = "timeout"
        else:
            fill, crashed = "error", True
        outcomes.extend([fill] * (len(examples) - len(outcomes)))
    return outcomes, cpu, crashed


def _outcome(message: dict, *, loading: bool) -> str:
    if message.get("ok"):
        value = message.get("value") or {}
        outcome = value.get("outcome") if isinstance(value, dict) else None
        allowed = {"loaded", "timeout", "error"} if loading else {"pass", "fail", "timeout", "error"}
        return outcome if outcome in allowed else "error"
    return "timeout" if message.get("error_type") == "_PublicTimeout" else "error"


def canary_ok(profile: GradingProfile) -> bool:
    try:
        outcomes, _, crashed = _run_once(CANARY_PROGRAM, (CANARY_ASSERT,), profile)
    except EvaluatorOutage:
        return False
    return outcomes == ["pass"] and not crashed


def run_public_checks(candidate: Candidate, task: PublicTask) -> PublicCheckResult:
    if task.output_kind is not OutputKind.PYTHON:
        raise ValueError(f"{task.item_id}: public checks apply to code tasks only")
    if candidate.item_id != task.item_id:
        raise ValueError(f"candidate {candidate.candidate_id} belongs to {candidate.item_id}, not {task.item_id}")
    examples = tuple(task.public_examples)
    if not examples:
        raise ValueError(f"{task.item_id}: no public examples; the code selector cannot run")
    profile = active_profile()
    env = environment_digest(profile)
    inputs_hash = public_inputs_sha256(task)
    n = len(examples)
    program = candidate.final_answer

    def result(status: CheckStatus, outcomes: Sequence[str], sha: str | None, cpu: float) -> PublicCheckResult:
        return PublicCheckResult(candidate_id=candidate.candidate_id, item_id=task.item_id, code_sha256=sha,
                                 status=status, cases_total=n, cases_passed=sum(o == "pass" for o in outcomes),
                                 case_outcomes=tuple(outcomes), checker_version=CHECKER_VERSION,
                                 public_inputs_sha256=inputs_hash, environment_digest=env, cpu_seconds=cpu)

    if candidate.parse_status is not ParseStatus.VALID or program is None or not program.strip():
        return result(CheckStatus.INVALID_CANDIDATE, ["fail"] * n, None, 0.0)
    sha = code_sha256(program)
    cpu = 0.0
    try:
        outcomes, cpu, crashed = _run_once(program, examples, profile)
        if not crashed:
            return result(CheckStatus.OK, outcomes, sha, cpu)
    except EvaluatorOutage:
        pass
    # Outage or unexplained death: confirm the sandbox with a known-good canary, then retry once, fresh.
    if not canary_ok(profile):
        return result(CheckStatus.INFRASTRUCTURE_ERROR, ["error"] * n, sha, cpu)
    try:
        outcomes, retry_cpu, _ = _run_once(program, examples, profile)
    except EvaluatorOutage:
        return result(CheckStatus.INFRASTRUCTURE_ERROR, ["error"] * n, sha, cpu)
    # A crash reproduced in a healthy sandbox is the candidate's behaviour: its cases stay "error".
    return result(CheckStatus.OK, outcomes, sha, cpu + retry_cpu)
