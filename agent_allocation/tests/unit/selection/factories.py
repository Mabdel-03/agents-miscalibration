"""Typed record factories shared by agent 06's selection and metrics tests (no network, no gold in selectors)."""
from __future__ import annotations

from decimal import Decimal

from agent_allocation.contracts import (
    STUDY_ID, Candidate, CheckpointView, CheckStatus, CompletionStatus, GenerationKind, GenerationResult,
    GenerationSpec, GradeResult, GradeStatus, OutputKind, ParseStatus, PolicyCheckpoint, PublicCheckResult,
    PublicTask, TransportStatus, Usage, public_task_hash,
)

BENCHMARK_OF = {OutputKind.MATH: "math500", OutputKind.OPTION: "mmlu_pro", OutputKind.BOOLEAN: "bbh_logic4",
                OutputKind.PYTHON: "mbpp_plus"}


def task(kind: OutputKind = OutputKind.MATH, item_id: str = "item-1", *, choices: tuple[str, ...] = (),
         subtask: str | None = None) -> PublicTask:
    if kind is OutputKind.OPTION and not choices:
        choices = tuple("ABCDEFGHIJ")
    return PublicTask(item_id=item_id, benchmark=BENCHMARK_OF[kind], source_revision="rev", prompt="Solve.",
                      output_kind=kind, allowed_choices=choices, subtask=subtask)


def spec(t: PublicTask, *, kind: GenerationKind = GenerationKind.ROOT, root_slot: int = 0, team_size: int = 0,
         update_round: int = 0, parents: tuple[str, ...] = (), partition: str = "main") -> GenerationSpec:
    return GenerationSpec(run_id=STUDY_ID, partition_id=partition, item_id=t.item_id, benchmark=t.benchmark,
                          kind=kind, model="google/gemini-2.5-flash-lite", endpoint_lock_hash="lock",
                          input_manifest_hash=public_task_hash(t), decoding_config_hash="decoding",
                          prompt_version="prompts-v1-test", reasoning_budget=1024, completion_headroom=1024,
                          root_slot=root_slot, team_size=team_size, update_round=update_round, draw_id=0,
                          parent_ids=parents)


def result(s: GenerationSpec, content: str | None, *, finish_reason: str | None = "stop",
           completion: CompletionStatus | None = CompletionStatus.COMPLETED,
           transport: TransportStatus = TransportStatus.OK) -> GenerationResult:
    usage = Usage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=None, cached_prompt_tokens=None,
                  actual_cost_usd=Decimal("0.00001"))
    return GenerationResult(node_id=s.node_id, attempt_id=f"{s.node_id}:1", transport_status=transport,
                            completion_status=completion, generation_id="gen", observed_model=s.model,
                            observed_provider="Google AI Studio", finish_reason=finish_reason,
                            native_finish_reason=None, visible_content=content, reasoning_returned=False,
                            usage=usage, cache_status="MISS", raw_response_path=None, raw_response_sha256=None,
                            latency_seconds=0.5, http_status=200)


def candidate(cid: str, final: str | None, *, t: PublicTask, status: ParseStatus | None = None) -> Candidate:
    status = status if status is not None else (ParseStatus.VALID if final is not None else ParseStatus.MISSING)
    content = f"<final>{final}</final>" if final is not None else "no final"
    return Candidate(candidate_id=cid, node_id=cid, item_id=t.item_id, visible_content=content, final_answer=final,
                     rationale=None, parse_status=status, finish_reason="stop", output_kind=t.output_kind)


def pool(t: PublicTask, finals: list[str | None], prefix: str = "c") -> dict[str, Candidate]:
    return {f"{prefix}{i}": candidate(f"{prefix}{i}", final, t=t) for i, final in enumerate(finals)}


def checkpoint(t: PublicTask, name: str, family: str, candidate_ids: tuple[str, ...], *,
               view: CheckpointView = CheckpointView.CURRENT, partition: str = "main",
               required: tuple[str, ...] | None = None, team_size: int = 0, total_cycles: int = 0,
               peer_rounds: int = 0) -> PolicyCheckpoint:
    return PolicyCheckpoint(policy_id=f"{partition}/{t.item_id}/{name}/{view.value}", item_id=t.item_id,
                            family=family, team_size=team_size, total_cycles=total_cycles, peer_rounds=peer_rounds,
                            candidate_ids=tuple(candidate_ids),
                            required_node_ids=tuple(required if required is not None else candidate_ids),
                            selector_name="", selector_version="", partition_id=partition, name=name, view=view,
                            reasoning_budget=1024)


def check(cid: str, t: PublicTask, passed: int, total: int = 2,
          status: CheckStatus = CheckStatus.OK) -> PublicCheckResult:
    outcomes = tuple(["pass"] * passed + ["fail"] * (total - passed))
    return PublicCheckResult(candidate_id=cid, item_id=t.item_id, code_sha256=f"sha-{cid}", status=status,
                             cases_total=total, cases_passed=passed, case_outcomes=outcomes, checker_version="pc-1",
                             public_inputs_sha256="inputs", environment_digest="env", cpu_seconds=0.01)


def grade(cid: str, correct: bool | None, status: GradeStatus | None = None) -> GradeResult:
    """A contract-valid grade: ``correct=None`` defaults to EVALUATOR_ERROR (unknown), INVALID_OUTPUT is never bool."""
    status = status or (GradeStatus.GRADED if correct is not None else GradeStatus.EVALUATOR_ERROR)
    semantic = correct if status is GradeStatus.GRADED else None
    operational = False if status is GradeStatus.INVALID_OUTPUT else True
    return GradeResult(candidate_id=cid, status=status, semantic_correct=semantic, operational_success=operational,
                       grader_version="g-1", private_eval_hash="private")
