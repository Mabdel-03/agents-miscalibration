"""Semantic cell completion, repair, locking, and failure-state tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents_scaling.agents.base_agent import (
    Agent,
    AgentOutput,
    SelfConsistencySample,
    requested_generation_tokens,
)
from agents_scaling.agents.aggregate import system_confidences
from agents_scaling.agents.message_builder import (
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_COT_CHAR_LIMIT,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
)
from agents_scaling.agents.topologies.base import TopologyResult
from agents_scaling.benchmarks.contracts import (
    BenchmarkContractKey,
    build_question_contract,
    canonical_question_sha256,
)
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import completion, io, runner
from agents_scaling.experiment.result_schema import (
    ARTIFACT_SCHEMA_VERSION,
    SELF_CONSISTENCY_PROTOCOL_HASH,
    SELF_CONSISTENCY_PROTOCOL_VERSION,
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)
from agents_scaling.experiment.completion import (
    CellLockUnavailable,
    CompletionState,
    CorruptArtifactError,
    FailureClass,
    canonicalize_results,
    cell_lock,
    get_completion_status,
    infer_failure_class,
    is_cell_active,
    read_failure,
    record_failure,
    summarize_serving_provenance,
    validate_completion_payload,
)
from agents_scaling.models import get_model
from agents_scaling.serving.registry import ServerEntry
from agents_scaling.serving.context import ContextCapacityError, ContextPreflight
from agents_scaling.serving.client import (
    ChatResult,
    GENERATION_CENSOR_PROTOCOL_HASH,
    GENERATION_CENSOR_PROTOCOL_VERSION,
    GenerationProtocolCensorError,
    GenerationTruncationError,
    OptionScores,
    ServerResponseProtocolError,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
    ThinkingBudgetProtocolError,
    QWEN_IM_END_TOKEN_ID,
    QWEN_THINK_START_TOKEN_ID,
    QWEN_THINK_END_TOKEN_ID,
    token_ids_sha256,
)


# Most fixtures below intentionally use synthetic q1/q2 identifiers and exercise the
# low-level JSON/metadata parser without loading a benchmark. Scientific-path tests pass
# explicit Questions (which take precedence) or call the module API directly.
_scientific_get_completion_status = get_completion_status
_scientific_canonicalize_results = canonicalize_results
_scientific_validate_completion_payload = validate_completion_payload


def get_completion_status(*args, **kwargs):
    kwargs.setdefault("_structural_only", True)
    return _scientific_get_completion_status(*args, **kwargs)


def canonicalize_results(*args, **kwargs):
    kwargs.setdefault("_structural_only", True)
    return _scientific_canonicalize_results(*args, **kwargs)


def validate_completion_payload(*args, **kwargs):
    kwargs.setdefault("_structural_only", True)
    return _scientific_validate_completion_payload(*args, **kwargs)


@pytest.fixture
def cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="single_agent",
        benchmark="gpqa",
        n_agents=1,
        n_samples=1,
        n_questions=2,
        seed=7,
    )


def _agent_coordinates(cell: ExperimentCell) -> list[tuple[int, int]]:
    n_agents = 1 if cell.topology.value == "single_agent" else cell.n_agents
    n_rounds = 1 if cell.topology.value in {"single_agent", "independent"} else cell.rounds
    return [(agent, round_idx) for round_idx in range(n_rounds) for agent in range(n_agents)]


def _agent_seed(cell: ExperimentCell, agent_id: int, round_idx: int) -> int:
    if cell.topology.value == "single_agent":
        return cell.seed
    if cell.topology.value == "independent":
        return cell.seed + agent_id
    if cell.topology.value == "decentralized":
        return cell.seed + round_idx * 100 + agent_id
    offset = 999 if agent_id == 0 else agent_id - 1
    return cell.seed + round_idx * 100 + offset


def _peer_count(cell: ExperimentCell, agent_id: int, round_idx: int) -> int:
    if cell.topology.value in {"single_agent", "independent"}:
        return 0
    if cell.topology.value == "decentralized":
        return 0 if round_idx == 0 else cell.n_agents - 1
    if agent_id == 0:
        return max(0, cell.n_agents - 1)
    return 0 if round_idx == 0 else 1


def _perfect_system_conf(cell: ExperimentCell) -> dict[str, float]:
    confidence = {
        "vote_fraction": 1.0,
        "mean_agreeing_logprob": 1.0,
        "mean_agreeing_verbal": 1.0,
        "mean_all_logprob": 1.0,
        "final_producer_logprob": 1.0,
        "final_producer_verbal": 1.0,
        "mean_producer_logprob": 1.0,
    }
    if cell.topology.value == "centralized":
        confidence.update(
            orchestrator_logprob=1.0,
            orchestrator_verbal=1.0,
        )
    return confidence


def _mock_verified_questions(
    monkeypatch, cell: ExperimentCell, questions: list[Question]
) -> None:
    retained = tuple(questions)
    contract = build_question_contract(
        BenchmarkContractKey.from_cell(cell), retained
    )
    monkeypatch.setattr(
        runner,
        "load_verified_questions",
        lambda *args, **kwargs: (retained, contract),
    )

    class _FrozenContracts:
        def __init__(self, root):
            self.path = root / "benchmark_contracts.v1.json"
            self.sidecar_sha256 = "f" * 64

        def contract_for_cell(self, observed_cell):
            assert observed_cell == cell
            return dict(contract)

        def verify_questions(self, observed_cell, observed_questions):
            assert observed_cell == cell
            assert tuple(observed_questions) == retained
            return dict(contract)

    monkeypatch.setattr(
        runner,
        "load_frozen_benchmark_contracts",
        lambda root, **_kwargs: _FrozenContracts(root),
    )
    monkeypatch.setattr(
        runner,
        "load_manifest",
        lambda root: SimpleNamespace(
            path=root / "cells.json",
            cells=(cell,),
        ),
    )


def _record(
    cell: ExperimentCell,
    qid: str,
    reasoning_tokens: int = 10,
    timestamp: float = 10.0,
    *,
    current: bool = False,
):
    coordinates = _agent_coordinates(cell)
    agents = []
    for index, (agent_id, round_idx) in enumerate(coordinates):
        exact_reasoning = reasoning_tokens if index == 0 else 0
        if current and not cell.reasoning_level.enable_thinking:
            exact_reasoning = 0
        agent = {
            "agent_id": f"agent{agent_id}",
            "round": round_idx,
            "answer": "A",
            "raw_text": "Answer: A",
            "option_logprobs": {"A": 1.0},
            "verbalized_conf": 1.0,
            "cot_text": "",
            "intermediate_results": "A",
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "reasoning_tokens": exact_reasoning,
            "reasoning_text": "",
        }
        if current:
            finite_budget = cell.reasoning_level.thinking_budget
            peer_count = _peer_count(cell, agent_id, round_idx)
            output_floor = requested_generation_tokens(cell.reasoning_level)
            requested_max = get_model(cell.model_size).max_model_len - 2 - 128
            if cell.reasoning_level.enable_thinking:
                # think-start + exact reasoning span + think-end + terminal im-end
                agent["completion_tokens"] = agent["reasoning_tokens"] + 3
            agent.update(
                finish_reason="stop",
                reasoning_token_source=(
                    "vllm_native_token_ids"
                    if cell.reasoning_level.enable_thinking
                    else "none"
                ),
                reasoning_word_count_legacy=0,
                thinking_budget_protocol_version=THINKING_BUDGET_PROTOCOL_VERSION,
                thinking_budget_protocol_hash=THINKING_BUDGET_PROTOCOL_HASH,
                thinking_budget_requested=finite_budget,
                thinking_budget_saturated=(
                    finite_budget is not None
                    and agent["reasoning_tokens"] == finite_budget
                ),
                thinking_budget_consumed_tokens=agent["reasoning_tokens"],
                generation_phase_count=1,
                generation_phase_finish_reasons=["stop"],
                generation_phase_seeds=[_agent_seed(cell, agent_id, round_idx)],
                generation_phase_prompt_tokens=[2],
                generation_phase_completion_tokens=[agent["completion_tokens"]],
                output_capacity_floor_tokens=output_floor,
                generation_phase_requested_max_tokens=[requested_max],
                generation_phase_prompt_token_id_hashes=["0" * 64],
                generation_phase_completion_token_id_hashes=["1" * 64],
                endpoint_generation="test-endpoint-generation",
                reasoning_start_token_index=(
                    2 if cell.reasoning_level.enable_thinking else None
                ),
                reasoning_end_token_index=(
                    3 + agent["reasoning_tokens"]
                    if cell.reasoning_level.enable_thinking
                    else None
                ),
                reasoning_start_token_count=(
                    1 if cell.reasoning_level.enable_thinking else 0
                ),
                reasoning_end_token_count=(
                    1 if cell.reasoning_level.enable_thinking else 0
                ),
                injected_transition_tokens=0,
                peer_context_tokens=peer_count * 10,
                peer_context_sha256=(
                    (
                        "e3b0c44298fc1c149afbf4c8996fb924"
                        "27ae41e4649b934ca495991b7852b855"
                    )
                    if peer_count == 0
                    else "2" * 64
                ),
                peer_context_block_token_counts=[10] * peer_count,
                peer_context_truncation_marker_count=0,
            )
        agents.append(agent)
    n_agents = 1 if cell.topology.value == "single_agent" else cell.n_agents
    n_rounds = 1 if cell.topology.value in {"single_agent", "independent"} else cell.rounds
    if cell.topology.value in {"single_agent", "independent"}:
        n_messages = 0
    elif cell.topology.value == "decentralized":
        n_messages = n_agents * (n_agents - 1) * (n_rounds - 1)
    else:
        n_messages = max(1, n_agents - 1) * (2 * n_rounds - 1)
    record = {
        "cell_id": cell.cell_id,
        "qid": qid,
        "benchmark": cell.benchmark,
        "model_size": cell.model_size,
        "topology": cell.topology.value,
        "context_share_level": cell.context_share_level.value,
        "prompt_complexity_level": cell.prompt_complexity_level,
        "reasoning_level": cell.reasoning_level.value,
        "final_answer": "A",
        "answer_key": "A",
        "correct": True,
        "per_agent": agents,
        "system_conf": {},
        "self_consistency": {},
        "efficiency_raw": {
            "n_turns": len(agents),
            "n_messages": n_messages,
            "n_rounds": n_rounds,
            "n_agents": n_agents,
            "total_prompt_tokens": sum(agent["prompt_tokens"] for agent in agents),
            "total_completion_tokens": sum(
                agent["completion_tokens"] for agent in agents
            ),
            "total_reasoning_tokens": sum(
                agent["reasoning_tokens"] for agent in agents
            ),
            "wall_ms": 5.0,
        },
        "timestamp": timestamp,
    }
    if current:
        questions = _truth_questions(cell)
        contract = build_question_contract(
            BenchmarkContractKey.from_cell(cell), questions
        )
        question_sha256s = {
            question.qid: canonical_question_sha256(question)
            for question in questions
        }
        record["system_conf"] = _perfect_system_conf(cell)
        record.update(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            question_sha256=question_sha256s[qid],
            benchmark_contract_sha256=contract["question_contract_sha256"],
            serving_profile=cell.model_size,
            effective_context_limit=get_model(cell.model_size).max_model_len,
            tensor_parallel_size=get_model(cell.model_size).tp_size,
            termination_status=TERMINATION_COMPLETED,
            censored_generation=None,
            observed_topology_coordinates=[],
        )
    return record


def _meta(cell: ExperimentCell, mean: float = 15.0):
    return {
        "cell_id": cell.cell_id,
        "config": cell.to_dict(),
        "config_hash": cell.config_hash(),
        "model_hf_id": get_model(cell.model_size).hf_id,
        "served_model_name": cell.model_size,
        "prompt_token_count": 8,
        "prompt_quality": {
            "heuristic": 50.0,
            "llm_judge": None,
            "features": {"word_count": 8.0},
        },
        "git_commit": "legacy-commit",
        "n_questions": 2,
        "mean_reasoning_tokens": mean,
        "started_at": 5.0,
        "finished_at": 20.0,
    }


def _current_meta(cell: ExperimentCell, mean: float | None = None):
    profile = get_model(cell.model_size)
    contract = build_question_contract(
        BenchmarkContractKey.from_cell(cell), _truth_questions(cell)
    )
    exact_mean = (
        15.0 if cell.reasoning_level.enable_thinking else 0.0
    ) if mean is None else mean
    return _meta(cell, mean) | {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "benchmark_contract_sha256": contract["question_contract_sha256"],
        "serving_profile": cell.model_size,
        "effective_context_limit": profile.max_model_len,
        "tensor_parallel_size": profile.tp_size,
        "serving_profile_counts": {cell.model_size: 2},
        "serving_profile_inferred_counts": {},
        "completed_question_count": 2,
        "length_censored_question_count": 0,
        "protocol_censored_question_count": 0,
        "artifact_schema_counts": {str(ARTIFACT_SCHEMA_VERSION): 2},
        "mean_reasoning_tokens": exact_mean,
        "mean_reasoning_tokens_exact": exact_mean,
        "exact_reasoning_question_count": 2,
        "nonexact_reasoning_question_count": 0,
        "mean_reasoning_word_count_legacy": 0.0,
        "peer_context_protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
        "peer_context_protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
        "peer_cot_char_limit": PEER_COT_CHAR_LIMIT,
        "peer_rendered_block_token_limit": PEER_RENDERED_BLOCK_TOKEN_LIMIT,
        "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
        "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
        "generation_censor_protocol_version": GENERATION_CENSOR_PROTOCOL_VERSION,
        "generation_censor_protocol_hash": GENERATION_CENSOR_PROTOCOL_HASH,
        "git_commit": "v2-code-version",
    }


def _full_envelope_censor(
    cell: ExperimentCell, question: Question
) -> tuple[dict, GenerationTruncationError]:
    """Build one exact, non-resampled standard-profile boundary outcome."""

    prompt_ids = [11, 12]
    context_limit = get_model(cell.model_size).max_model_len
    requested = context_limit - len(prompt_ids) - 128
    completion_ids = [42] * requested
    error = GenerationTruncationError(
        finish_reason="length",
        requested_output_tokens=requested,
        completion_tokens=requested,
        output_capacity_floor_tokens=requested_generation_tokens(
            cell.reasoning_level
        ),
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        decoded_completion="unfinished",
        server_content="unfinished",
        server_reasoning="",
        seed=cell.seed,
        serving_profile=cell.model_size,
        effective_context_limit=context_limit,
        endpoint_generation="test-endpoint-generation",
        created_at=100.0,
    ).enrich(
        qid=question.qid,
        agent_id="agent0",
        round_idx=0,
        generation_role="topology",
    )
    record = runner._length_censored_record(
        cell=cell,
        question=question,
        error=error,
        wall_ms=5.0,
        profile_meta={
            "serving_profile": cell.model_size,
            "effective_context_limit": context_limit,
            "tensor_parallel_size": get_model(cell.model_size).tp_size,
        },
        benchmark_contract_sha256=build_question_contract(
            BenchmarkContractKey.from_cell(cell), _truth_questions(cell)
        )["question_contract_sha256"],
        observed_topology_coordinates=[
            {
                "coordinate_key": "topology:agent0:0",
                "request": {
                    "generation_role": "topology",
                    "qid": question.qid,
                    "agent_id": "agent0",
                    "round": 0,
                    "seed": cell.seed,
                    "sample_index": None,
                    "peer_context": {
                        "sha256": (
                            "e3b0c44298fc1c149afbf4c8996fb924"
                            "27ae41e4649b934ca495991b7852b855"
                        ),
                        "utf8_bytes": 0,
                    },
                    "max_tokens": 4096,
                    "elicit_cot": True,
                },
                "outcome": {
                    "termination_status": TERMINATION_LENGTH_CENSORED,
                    "agent_output": None,
                    "censored_generation": error.to_censored_generation(),
                },
                "observed_at": 100.0,
                "producer_wall_ms": 5.0,
            }
        ],
    )
    return record, error


def _protocol_censor(
    cell: ExperimentCell, question: Question
) -> tuple[dict, GenerationProtocolCensorError]:
    prompt_ids = [11, 12]
    completion_ids = [QWEN_THINK_END_TOKEN_ID, QWEN_IM_END_TOKEN_ID]
    context_limit = get_model(cell.model_size).max_model_len
    requested = context_limit - len(prompt_ids) - 128
    error = GenerationProtocolCensorError(
        protocol_violation_codes=["unexpected_reasoning_delimiter"],
        finish_reason="stop",
        requested_output_tokens=requested,
        completion_tokens=len(completion_ids),
        output_capacity_floor_tokens=requested_generation_tokens(
            cell.reasoning_level
        ),
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        decoded_completion="</think>",
        server_content="</think>",
        server_reasoning="",
        seed=cell.seed,
        serving_profile=cell.model_size,
        effective_context_limit=context_limit,
        completion_think_end_positions=[0],
        endpoint_generation="test-endpoint-generation",
        created_at=101.0,
    ).enrich(
        qid=question.qid,
        agent_id="agent0",
        round_idx=0,
        generation_role="topology",
    )
    coordinate_censor = error.to_censored_generation()
    record = runner._length_censored_record(
        cell=cell,
        question=question,
        error=error,
        wall_ms=5.0,
        profile_meta={
            "serving_profile": cell.model_size,
            "effective_context_limit": context_limit,
            "tensor_parallel_size": get_model(cell.model_size).tp_size,
        },
        benchmark_contract_sha256=build_question_contract(
            BenchmarkContractKey.from_cell(cell), _truth_questions(cell)
        )["question_contract_sha256"],
        observed_topology_coordinates=[
            {
                "coordinate_key": "topology:agent0:0",
                "request": {
                    "generation_role": "topology",
                    "qid": question.qid,
                    "agent_id": "agent0",
                    "round": 0,
                    "seed": cell.seed,
                    "sample_index": None,
                    "peer_context": {
                        "sha256": (
                            "e3b0c44298fc1c149afbf4c8996fb924"
                            "27ae41e4649b934ca495991b7852b855"
                        ),
                        "utf8_bytes": 0,
                    },
                    "max_tokens": 4096,
                    "elicit_cot": True,
                },
                "outcome": {
                    "termination_status": TERMINATION_PROTOCOL_CENSORED,
                    "agent_output": None,
                    "censored_generation": coordinate_censor,
                },
                "observed_at": 101.0,
                "producer_wall_ms": 5.0,
            }
        ],
    )
    return record, error


def _self_consistency_cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="single_agent",
        benchmark="gpqa",
        n_agents=1,
        n_samples=3,
        n_questions=2,
        seed=7,
    )


def _self_consistency_payload_for_test(
    cell: ExperimentCell,
    qid: str,
    *,
    censored_index: int | None = None,
) -> dict:
    primary = _record(cell, qid, current=True)
    outcomes = []
    total_prompt = 0
    total_completion = 0
    total_reasoning = 0
    for sample_index in range(cell.n_samples):
        seed = cell.seed + 1000 + sample_index
        if sample_index == censored_index:
            question = Question(
                qid, "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
            )
            censor_row, _ = _full_envelope_censor(cell, question)
            censor = censor_row["censored_generation"]
            censor["generation_role"] = "self_consistency"
            censor["sample_index"] = sample_index
            censor["seed"] = seed
            outcomes.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "termination_status": TERMINATION_LENGTH_CENSORED,
                    "agent_output": None,
                    "censored_generation": censor,
                }
            )
            total_prompt += censor["prompt_tokens"]
            total_completion += censor["completion_tokens"]
        else:
            output = json.loads(json.dumps(primary["per_agent"][0]))
            output["generation_phase_seeds"] = [seed]
            outcomes.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "termination_status": TERMINATION_COMPLETED,
                    "agent_output": output,
                    "censored_generation": None,
                }
            )
            total_prompt += output["prompt_tokens"]
            total_completion += output["completion_tokens"]
            total_reasoning += output["reasoning_tokens"]
    censored_count = int(censored_index is not None)
    return {
        "protocol_version": SELF_CONSISTENCY_PROTOCOL_VERSION,
        "protocol_hash": SELF_CONSISTENCY_PROTOCOL_HASH,
        "sample_count": cell.n_samples,
        "completed_sample_count": cell.n_samples - censored_count,
        "length_censored_sample_count": censored_count,
        "protocol_censored_sample_count": 0,
        "samples": outcomes,
        "majority": None if censored_count else "A",
        "self_consistency_conf": None if censored_count else 1.0,
        "semantic_entropy_conf": None if censored_count else 1.0,
        "semantic_entropy": None if censored_count else 0.0,
        "auxiliary_efficiency_raw": {
            "n_samples": cell.n_samples,
            "completed_samples": cell.n_samples - censored_count,
            "length_censored_samples": censored_count,
            "protocol_censored_samples": 0,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_reasoning_tokens": total_reasoning,
            "wall_ms": 6.0,
        },
    }


def _current_agent_output(cell: ExperimentCell, seed: int) -> AgentOutput:
    return AgentOutput(
        agent_id="agent0",
        round=0,
        answer_choice="A",
        raw_text="Answer: A",
        cot_text="",
        intermediate_results="Answer: A",
        option_logprobs={"A": 0.8, "B": 0.2},
        prompt_tokens=2,
        completion_tokens=3,
        reasoning_tokens=0,
        finish_reason="stop",
        reasoning_token_source="none",
        thinking_budget_consumed_tokens=0,
        generation_phase_finish_reasons=["stop"],
        generation_phase_seeds=[seed],
        generation_phase_prompt_tokens=[2],
        generation_phase_completion_tokens=[3],
        output_capacity_floor_tokens=requested_generation_tokens(
            cell.reasoning_level
        ),
        generation_phase_requested_max_tokens=[
            get_model(cell.model_size).max_model_len - 2 - 128
        ],
        generation_phase_prompt_token_id_hashes=["0" * 64],
        generation_phase_completion_token_id_hashes=["1" * 64],
        endpoint_generation="test-endpoint-generation",
    )


class _NoopScoringClient:
    def score_options(self, _prompt, option_letters):
        return OptionScores(
            probs={letter: 1.0 / len(option_letters) for letter in option_letters},
            raw_logprobs={letter: -1.0 for letter in option_letters},
        )


def _long_32b_cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="32B",
        context_share_level="plus_cot",
        prompt_complexity_level=0,
        reasoning_level="unlimited",
        topology="decentralized",
        benchmark="gpqa",
        n_agents=3,
        n_samples=1,
        n_questions=2,
        seed=7,
    )


def _thinking_single_agent_cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="b2048",
        topology="single_agent",
        benchmark="gpqa",
        n_agents=1,
        n_samples=1,
        n_questions=2,
        seed=7,
    )


def test_semantic_status_requires_exact_qids_and_matching_meta(tmp_path, cell):
    qids = ("q1", "q2")
    assert get_completion_status(cell, tmp_path, expected_qids=qids).status is CompletionState.MISSING

    io.write_jsonl(tmp_path / "results.jsonl", [_record(cell, "q1")])
    partial = get_completion_status(cell, tmp_path, expected_qids=qids)
    assert partial.status is CompletionState.PARTIAL
    assert partial.missing_qids == ("q2",)

    io.write_jsonl(
        tmp_path / "results.jsonl",
        [_record(cell, "q1", 10), _record(cell, "q2", 20)],
    )
    io.write_json(tmp_path / "meta.json", _meta(cell))
    complete = get_completion_status(cell, tmp_path, expected_qids=qids)
    assert complete.status is CompletionState.COMPLETE
    assert complete.valid_count == complete.expected_count == 2


def _truth_questions(cell: ExperimentCell) -> tuple[Question, ...]:
    questions = (
        Question("q1", cell.benchmark, "one?", "A", AnswerType.MCQ, ["yes", "no"]),
        Question("q2", cell.benchmark, "two?", "A", AnswerType.MCQ, ["yes", "no"]),
    )
    return questions if cell.n_questions is None else questions[: cell.n_questions]


@pytest.mark.parametrize(
    ("field", "value", "error_fragment"),
    [
        ("answer_key", "B", "answer_key does not match the benchmark question"),
        ("correct", False, "correct does not match benchmark grading"),
        ("final_answer", "B", "correct does not match benchmark grading"),
        ("final_answer", None, "correct does not match benchmark grading"),
    ],
)
def test_question_truth_contract_rejects_forged_outcomes(
    cell, field, value, error_fragment
):
    questions = _truth_questions(cell)
    rows = [_record(cell, "q1", 10), _record(cell, "q2", 20)]
    rows[0][field] = value

    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        rows,
        _meta(cell),
        expected_questions=questions,
    )

    assert any(error_fragment in error for error in errors)


def test_current_question_digest_rejects_same_qid_and_gold_with_changed_stem(cell):
    questions = _truth_questions(cell)
    rows = [
        _record(cell, "q1", current=True),
        _record(cell, "q2", current=True),
    ]
    for row in rows:
        row["per_agent"][0]["option_logprobs"]["B"] = 0.0
    meta = _current_meta(cell)
    assert validate_completion_payload(
        cell,
        ("q1", "q2"),
        rows,
        meta,
        expected_questions=questions,
    ) == ()

    changed_questions = (
        Question(
            "q1",
            cell.benchmark,
            "one, but with scientifically different wording?",
            "A",
            AnswerType.MCQ,
            ["yes", "no"],
        ),
        questions[1],
    )
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        rows,
        meta,
        expected_questions=changed_questions,
    )

    assert any("question_sha256 does not match" in error for error in errors)
    assert any("benchmark_contract_sha256 does not match" in error for error in errors)


@pytest.mark.parametrize(
    ("target", "field", "error_fragment"),
    [
        ("row", "question_sha256", "question_sha256 does not match"),
        (
            "row",
            "benchmark_contract_sha256",
            "benchmark_contract_sha256 does not match",
        ),
        (
            "meta",
            "benchmark_contract_sha256",
            "meta benchmark_contract_sha256 does not match",
        ),
    ],
)
def test_current_artifacts_reject_wrong_benchmark_contract_hashes(
    cell, target, field, error_fragment
):
    questions = _truth_questions(cell)
    rows = [
        _record(cell, "q1", current=True),
        _record(cell, "q2", current=True),
    ]
    meta = _current_meta(cell)
    if target == "row":
        rows[0][field] = "f" * 64
    else:
        meta[field] = "f" * 64

    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        rows,
        meta,
        expected_questions=questions,
    )

    assert any(error_fragment in error for error in errors)


def test_real_run_status_loads_frozen_questions_when_not_explicit(
    tmp_path, cell, monkeypatch
):
    questions = _truth_questions(cell)
    contract = build_question_contract(
        BenchmarkContractKey.from_cell(cell), questions
    )
    run_root = tmp_path / "frozen-run"
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    observed: list[tuple[object, str]] = []

    def _load_verified(root, observed_cell):
        observed.append((root, observed_cell.cell_id))
        return questions, contract

    monkeypatch.setattr(completion, "load_verified_questions", _load_verified)
    monkeypatch.setattr(
        completion,
        "load_manifest",
        lambda root: SimpleNamespace(path=root / "cells.json", cells=(cell,)),
    )
    monkeypatch.setattr(
        completion,
        "expected_questions_for_cell",
        lambda _cell: pytest.fail("real run must use its frozen benchmark sidecar"),
    )

    status = completion.get_completion_status(cell, cdir)

    assert status.status is CompletionState.MISSING
    assert observed == [(run_root, cell.cell_id)]


def test_real_run_status_rejects_supplied_questions_that_bypass_frozen_sidecar(
    tmp_path, cell
):
    from agents_scaling.benchmarks.contracts import (
        BenchmarkContractError,
        freeze_benchmark_contracts,
    )
    from agents_scaling.experiment.manifest import freeze_manifest, load_manifest

    questions = _truth_questions(cell)
    run_root = tmp_path / "frozen-run"
    run_root.mkdir()
    (run_root / "cells.json").write_text(
        json.dumps([cell.to_dict()]) + "\n", encoding="utf-8"
    )
    freeze_manifest(run_root)
    snapshot = load_manifest(run_root)
    freeze_benchmark_contracts(
        run_root,
        snapshot=snapshot,
        benchmark_loader=lambda *_args, **_kwargs: list(questions),
    )
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    changed = (
        Question(
            "q1",
            cell.benchmark,
            "scientifically changed wording?",
            "A",
            AnswerType.MCQ,
            ["yes", "no"],
        ),
        questions[1],
    )

    with pytest.raises(BenchmarkContractError, match="normalized Question contract drift"):
        completion.get_completion_status(
            cell,
            cdir,
            expected_qids=("q1", "q2"),
            expected_questions=changed,
        )

    frozen = completion.load_frozen_benchmark_contracts(run_root)
    impostor = ExperimentCell.from_dict(cell.to_dict() | {"temperature": 0.123})
    assert impostor.cell_id == cell.cell_id
    assert impostor != cell
    with pytest.raises(ValueError, match="not the exact entry in the frozen run manifest"):
        completion.get_completion_status(
            impostor,
            cdir,
            expected_qids=("q1", "q2"),
            expected_questions=questions,
            verified_benchmark_contracts=frozen,
        )


def test_truth_aware_canonicalization_keeps_later_valid_duplicate(tmp_path, cell):
    questions = _truth_questions(cell)
    forged = _record(cell, "q1", 99)
    forged["answer_key"] = "B"
    valid_q1 = _record(cell, "q1", 11)
    valid_q2 = _record(cell, "q2", 22)
    io.write_jsonl(tmp_path / "results.jsonl", [forged, valid_q1, valid_q2])

    with cell_lock(tmp_path):
        parsed = canonicalize_results(
            cell,
            tmp_path,
            expected_qids=("q1", "q2"),
            expected_questions=questions,
        )

    assert parsed.invalid_rows == 1
    assert [record["qid"] for record in parsed.records] == ["q1", "q2"]
    assert parsed.records[0]["efficiency_raw"]["total_reasoning_tokens"] == 11
    repaired = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert [record["qid"] for record in repaired] == ["q1", "q2"]


def test_current_outcome_and_calibration_are_derived_from_retained_agent_data(cell):
    questions = _truth_questions(cell)
    rows = [_record(cell, "q1", current=True), _record(cell, "q2", current=True)]
    for row in rows:
        row["per_agent"][0]["option_logprobs"]["B"] = 0.0
    meta = _current_meta(cell)
    assert validate_completion_payload(
        cell,
        ("q1", "q2"),
        rows,
        meta,
        expected_questions=questions,
    ) == ()

    forged_final = json.loads(json.dumps(rows))
    forged_final[0]["final_answer"] = "B"
    forged_final[0]["correct"] = False
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        forged_final,
        meta,
        expected_questions=questions,
    )
    assert any("final_answer does not match retained topology outputs" in error for error in errors)

    forged_confidence = json.loads(json.dumps(rows))
    forged_confidence[0]["system_conf"]["vote_fraction"] = 0.5
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        forged_confidence,
        meta,
        expected_questions=questions,
    )
    assert any("system_conf.vote_fraction" in error for error in errors)

    deleted_confidence = json.loads(json.dumps(rows))
    deleted_confidence[0]["system_conf"].pop("mean_all_logprob")
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        deleted_confidence,
        meta,
        expected_questions=questions,
    )
    assert any("system_conf keys" in error for error in errors)

    forged_agent_answer = json.loads(json.dumps(rows))
    forged_agent_answer[0]["per_agent"][0]["answer"] = "B"
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        forged_agent_answer,
        meta,
        expected_questions=questions,
    )
    assert any("answer does not match raw_text" in error for error in errors)

    unnormalised_probe = json.loads(json.dumps(rows))
    unnormalised_probe[0]["per_agent"][0]["option_logprobs"] = {
        "A": 0.8,
        "B": 0.8,
    }
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        unnormalised_probe,
        meta,
        expected_questions=questions,
    )
    assert any("must sum to one" in error for error in errors)


@pytest.mark.parametrize(
    ("topology", "n_agents"),
    [("independent", 3), ("decentralized", 3), ("centralized", 3)],
)
def test_each_multiagent_topology_recomputes_its_final_producer(
    cell, topology, n_agents
):
    multi = ExperimentCell.from_dict(
        cell.to_dict()
        | {
            "topology": topology,
            "n_agents": n_agents,
            "rounds": 2,
        }
    )
    rows = [
        _record(multi, "q1", current=True),
        _record(multi, "q2", current=True),
    ]
    meta = _current_meta(multi)
    assert validate_completion_payload(
        multi, ("q1", "q2"), rows, meta
    ) == ()

    rows[0]["final_answer"] = "B"
    rows[0]["correct"] = False
    errors = validate_completion_payload(multi, ("q1", "q2"), rows, meta)
    assert any(
        "final_answer does not match retained topology outputs" in error
        for error in errors
    )


def test_truth_contract_is_derived_only_when_qids_are_not_explicit(
    tmp_path, cell, monkeypatch
):
    questions = _truth_questions(cell)
    forged = _record(cell, "q1")
    forged["correct"] = False
    io.write_jsonl(
        tmp_path / "results.jsonl", [forged, _record(cell, "q2", 20)]
    )
    io.write_json(tmp_path / "meta.json", _meta(cell))

    monkeypatch.setattr(
        completion, "expected_questions_for_cell", lambda _cell: questions
    )
    truth_aware = completion.get_completion_status(cell, tmp_path)
    assert truth_aware.status is CompletionState.CORRUPT
    assert truth_aware.invalid_rows == 1
    explicit_qids_remain_truth_aware = completion.get_completion_status(
        cell, tmp_path, expected_qids=("q1", "q2")
    )
    assert explicit_qids_remain_truth_aware.status is CompletionState.CORRUPT
    with pytest.raises(ValueError, match="order/QIDs"):
        completion.get_completion_status(
            cell, tmp_path, expected_qids=("q2", "q1")
        )

    monkeypatch.setattr(
        completion,
        "expected_questions_for_cell",
        lambda _cell: (_ for _ in ()).throw(AssertionError("unexpected benchmark load")),
    )
    structural_only = completion.get_completion_status(
        cell,
        tmp_path,
        expected_qids=("q1", "q2"),
        _structural_only=True,
    )
    assert structural_only.status is CompletionState.COMPLETE


def test_v3_accepts_later_end_delimiter_as_answer_content(tmp_path):
    cell = _thinking_single_agent_cell()
    row = _record(cell, "q1", current=True)
    agent = row["per_agent"][0]
    agent["reasoning_end_token_count"] = 2
    agent["completion_tokens"] += 1
    agent["generation_phase_completion_tokens"] = [agent["completion_tokens"]]
    row["efficiency_raw"]["total_completion_tokens"] += 1
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.PARTIAL
    assert status.valid_count == 1
    assert status.invalid_rows == 0


def test_v3_accepts_exact_saturation_with_later_answer_end(tmp_path):
    cell = _thinking_single_agent_cell()
    assert cell.reasoning_level.thinking_budget is not None
    row = _record(
        cell,
        "q1",
        reasoning_tokens=cell.reasoning_level.thinking_budget,
        current=True,
    )
    agent = row["per_agent"][0]
    assert agent["thinking_budget_saturated"] is True
    agent["reasoning_end_token_count"] = 2
    agent["completion_tokens"] += 1
    agent["generation_phase_completion_tokens"] = [agent["completion_tokens"]]
    row["efficiency_raw"]["total_completion_tokens"] += 1
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.PARTIAL
    assert status.valid_count == 1
    assert status.invalid_rows == 0


def test_v3_rejects_end_delimiter_count_that_cannot_fit_completion(tmp_path):
    cell = _thinking_single_agent_cell()
    row = _record(cell, "q1", current=True)
    agent = row["per_agent"][0]
    # Leave no room for this many ends once start, exact reasoning, and im_end are
    # accounted for.  This is deliberately below the weaker completion_tokens-2 bound.
    agent["reasoning_end_token_count"] = (
        agent["completion_tokens"] - agent["reasoning_tokens"] - 1
    )
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert any("end-delimiter count" in error for error in status.errors)


@pytest.mark.parametrize("corruption", ["truncated", "extra_meta", "wrong_hash", "incomplete"])
def test_corrupt_artifacts_are_never_complete(tmp_path, cell, corruption):
    qids = ("q1", "q2")
    rows = [_record(cell, "q1", 10), _record(cell, "q2", 20)]
    io.write_jsonl(tmp_path / "results.jsonl", rows)
    meta = _meta(cell)
    io.write_json(tmp_path / "meta.json", meta)

    if corruption == "truncated":
        with (tmp_path / "results.jsonl").open("a") as handle:
            handle.write('{"qid":')
    elif corruption == "extra_meta":
        (tmp_path / "meta.json").write_text(json.dumps(meta) + "\n{}")
    elif corruption == "wrong_hash":
        meta["config_hash"] = "wrong"
        io.write_json(tmp_path / "meta.json", meta)
    else:
        io.write_jsonl(tmp_path / "results.jsonl", rows[:1])

    status = get_completion_status(cell, tmp_path, expected_qids=qids)
    assert status.status is CompletionState.CORRUPT


def test_canonicalization_keeps_first_valid_row_and_atomically_repairs(tmp_path, cell):
    first = _record(cell, "q1", 11)
    duplicate = _record(cell, "q1", 99)
    second = _record(cell, "q2", 22)
    payload = "\n".join(
        [json.dumps(first), "not-json", json.dumps(duplicate), json.dumps(second)]
    ) + "\n"
    (tmp_path / "results.jsonl").write_text(payload)

    with cell_lock(tmp_path):
        parsed = canonicalize_results(cell, tmp_path, expected_qids=("q1", "q2"))

    assert parsed.malformed_lines == 1
    assert parsed.duplicate_qids == ("q1",)
    repaired = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [row["qid"] for row in repaired] == ["q1", "q2"]
    assert repaired[0]["efficiency_raw"]["total_reasoning_tokens"] == 11
    assert get_completion_status(
        cell, tmp_path, expected_qids=("q1", "q2")
    ).status is CompletionState.PARTIAL


def test_serving_provenance_infers_only_wholly_legacy_rows():
    cell = _long_32b_cell()
    legacy = _record(cell, "q1")
    explicit_long = _record(cell, "q2", current=True) | {
        "serving_profile": "32B-long",
        "effective_context_limit": 40960,
        "tensor_parallel_size": 2,
    }

    summary = summarize_serving_provenance(cell, [legacy, explicit_long])

    assert summary == {
        "serving_profile": "mixed",
        "effective_context_limit": None,
        "tensor_parallel_size": None,
        "serving_profile_counts": {"32B": 1, "32B-long": 1},
        "serving_profile_inferred_counts": {"32B": 1},
    }


def test_valid_legacy_meta_without_provenance_remains_complete(tmp_path):
    cell = _long_32b_cell()
    io.write_jsonl(
        tmp_path / "results.jsonl",
        [_record(cell, "q1", 10), _record(cell, "q2", 20)],
    )
    io.write_json(tmp_path / "meta.json", _meta(cell))

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.COMPLETE


def test_legacy_finite_budget_rows_below_old_cap_are_retained(tmp_path, cell):
    finite = ExperimentCell.from_dict(cell.to_dict() | {"reasoning_level": "b512"})
    io.write_jsonl(tmp_path / "results.jsonl", [_record(finite, "q1")])

    status = get_completion_status(finite, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.PARTIAL
    assert status.valid_count == 1
    assert status.missing_qids == ("q2",)


@pytest.mark.parametrize(
    ("reasoning_level", "old_cap"),
    [
        ("off", 1024),
        ("b512", 1536),
        ("b2048", 3072),
        ("b8192", 9216),
        ("unlimited", 9216),
    ],
)
def test_legacy_rows_at_every_old_cap_are_rejected(
    tmp_path, cell, reasoning_level, old_cap
):
    capped = ExperimentCell.from_dict(
        cell.to_dict() | {"reasoning_level": reasoning_level}
    )
    row = _record(capped, "q1")
    row["per_agent"][0]["completion_tokens"] = old_cap
    row["efficiency_raw"]["total_completion_tokens"] = old_cap
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(capped, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert any("old output cap" in error for error in status.errors)


def test_v3_rows_and_metadata_satisfy_strict_semantic_contract(tmp_path, cell):
    rows = [
        _record(cell, "q1", 10, current=True),
        _record(cell, "q2", 20, current=True),
    ]
    io.write_jsonl(tmp_path / "results.jsonl", rows)
    io.write_json(tmp_path / "meta.json", _current_meta(cell))

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.COMPLETE


def test_v3_finite_budget_row_requires_registered_protocol_provenance(tmp_path, cell):
    finite = ExperimentCell.from_dict(cell.to_dict() | {"reasoning_level": "b512"})
    rows = [
        _record(finite, "q1", 10, current=True),
        _record(finite, "q2", 20, current=True),
    ]
    io.write_jsonl(tmp_path / "results.jsonl", rows)
    io.write_json(tmp_path / "meta.json", _current_meta(finite))

    status = get_completion_status(finite, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.COMPLETE


@pytest.mark.parametrize("topology", ["single_agent", "independent", "decentralized", "centralized"])
def test_v3_peer_block_counts_follow_topology_role_and_round(tmp_path, cell, topology):
    configured = ExperimentCell.from_dict(
        cell.to_dict()
        | {
            "topology": topology,
            "n_agents": 1 if topology == "single_agent" else 3,
            "rounds": 2,
        }
    )
    row = _record(configured, "q1", current=True)
    for agent in row["per_agent"]:
        agent_index = int(agent["agent_id"].removeprefix("agent"))
        assert len(agent["peer_context_block_token_counts"]) == _peer_count(
            configured, agent_index, agent["round"]
        )
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(
        configured, tmp_path, expected_qids=("q1", "q2")
    )

    assert status.status is CompletionState.PARTIAL


def test_v3_rejects_missing_required_peer_blocks_and_wrong_empty_hash(tmp_path, cell):
    decentralized = ExperimentCell.from_dict(
        cell.to_dict()
        | {"topology": "decentralized", "n_agents": 3, "rounds": 2}
    )
    row = _record(decentralized, "q1", current=True)
    round_one = next(agent for agent in row["per_agent"] if agent["round"] == 1)
    round_one.update(
        peer_context_tokens=0,
        peer_context_sha256=(
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
        peer_context_block_token_counts=[],
    )
    io.write_jsonl(tmp_path / "results.jsonl", [row])
    status = get_completion_status(
        decentralized, tmp_path, expected_qids=("q1", "q2")
    )
    assert status.status is CompletionState.CORRUPT
    assert any("peer block count" in error for error in status.errors)

    io.write_jsonl(tmp_path / "results.jsonl", [_record(cell, "q1", current=True)])
    raw = json.loads((tmp_path / "results.jsonl").read_text())
    raw["per_agent"][0]["peer_context_sha256"] = "f" * 64
    io.write_jsonl(tmp_path / "results.jsonl", [raw])
    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))
    assert status.status is CompletionState.CORRUPT
    assert any("empty peer context" in error for error in status.errors)


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (
            lambda row: row["efficiency_raw"].pop("wall_ms"),
            "efficiency_raw.wall_ms",
        ),
        (
            lambda row: row["per_agent"][0].pop("finish_reason"),
            "finish_reason",
        ),
        (
            lambda row: row["per_agent"][0].update(finish_reason="length"),
            "finish_reason must be stop",
        ),
        (
            lambda row: row["per_agent"][0].update(
                reasoning_token_source="legacy_word_count"
            ),
            "reasoning_token_source",
        ),
        (
            lambda row: row["per_agent"][0].pop("raw_text"),
            "raw_text",
        ),
        (
            lambda row: row["per_agent"][0].update(invented_provenance=True),
            "wrong fields",
        ),
        (
            lambda row: row["per_agent"][0].update(
                reasoning_start_token_count=1
            ),
            "generated delimiter counts",
        ),
        (
            lambda row: row["per_agent"][0].update(
                generation_phase_prompt_token_id_hashes=["not-a-hash"]
            ),
            "prompt_token_id_hashes",
        ),
    ],
)
def test_v3_rejects_invalid_efficiency_and_agent_payloads(
    tmp_path, cell, mutation, expected_error
):
    row = _record(cell, "q1", current=True)
    mutation(row)
    io.write_jsonl(tmp_path / "results.jsonl", [row])

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert any(expected_error in error for error in status.errors)


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_hf_id", "wrong/model"),
        ("prompt_token_count", "8"),
        ("serving_profile_counts", {}),
        ("peer_cot_char_limit", 3999),
        ("peer_context_protocol_hash", "wrong"),
    ],
)
def test_v3_rejects_invalid_model_prompt_profile_and_protocol_meta(
    tmp_path, cell, field, value
):
    rows = [
        _record(cell, "q1", 10, current=True),
        _record(cell, "q2", 20, current=True),
    ]
    meta = _current_meta(cell)
    meta[field] = value
    io.write_jsonl(tmp_path / "results.jsonl", rows)
    io.write_json(tmp_path / "meta.json", meta)

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert any(field in error for error in status.errors)


def test_metadata_cannot_relabel_legacy_rows_as_current_long_route(tmp_path):
    cell = _long_32b_cell()
    io.write_jsonl(
        tmp_path / "results.jsonl",
        [_record(cell, "q1", 10), _record(cell, "q2", 20)],
    )
    meta = _meta(cell)
    meta.update(
        serving_profile="32B-long",
        effective_context_limit=40960,
        tensor_parallel_size=2,
        serving_profile_counts={"32B-long": 2},
        serving_profile_inferred_counts={},
    )
    io.write_json(tmp_path / "meta.json", meta)

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert any("serving_profile" in error for error in status.errors)


@pytest.mark.parametrize(
    "provenance",
    [
        {"serving_profile": "32B-long"},
        {
            "serving_profile": "32B-long",
            "effective_context_limit": 16384,
            "tensor_parallel_size": 2,
        },
        {
            "serving_profile": "14B",
            "effective_context_limit": 16384,
            "tensor_parallel_size": 1,
        },
    ],
)
def test_invalid_or_ambiguous_question_provenance_is_corrupt(
    tmp_path, provenance
):
    cell = _long_32b_cell()
    io.write_jsonl(tmp_path / "results.jsonl", [_record(cell, "q1") | provenance])

    status = get_completion_status(cell, tmp_path, expected_qids=("q1", "q2"))

    assert status.status is CompletionState.CORRUPT
    assert status.invalid_rows == 1


def test_cell_lock_is_nonblocking_and_reports_active(tmp_path):
    cell = ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="single_agent",
        benchmark="gpqa",
        n_agents=1,
        n_samples=1,
        n_questions=2,
        seed=7,
    )
    io.write_jsonl(tmp_path / "results.jsonl", [_record(cell, "q1")])
    with cell_lock(tmp_path):
        assert is_cell_active(tmp_path)
        status = get_completion_status(
            cell, tmp_path, expected_qids=("q1", "q2")
        )
        assert status.status is CompletionState.ACTIVE
        assert status.valid_count == status.completed_question_count == 1
        assert status.missing_qids == ("q2",)
        with pytest.raises(CellLockUnavailable):
            with cell_lock(tmp_path):
                pass
    assert not is_cell_active(tmp_path)


def test_failure_backoff_dormancy_and_context_unblock(tmp_path, cell):
    err = RuntimeError("endpoint unavailable")
    first = record_failure(
        tmp_path,
        cell,
        err,
        classification=FailureClass.CONNECTION,
        now=100.0,
        base_backoff_s=10.0,
        max_attempts=3,
        serving_profile="0.6B",
        code_version="a",
        server_pool_generation="fleet-1",
    )
    assert first.next_eligible_at == 110.0
    waiting = get_completion_status(
        cell, tmp_path, expected_qids=("q1", "q2"), now=105.0
    )
    assert waiting.status is CompletionState.RETRYABLE
    assert not waiting.eligible_for_retry

    for timestamp in (111.0, 112.0):
        record_failure(
            tmp_path,
            cell,
            err,
            classification=FailureClass.CONNECTION,
            now=timestamp,
            base_backoff_s=10.0,
            max_attempts=3,
            serving_profile="0.6B",
            code_version="a",
            server_pool_generation="fleet-1",
        )
    dormant = read_failure(tmp_path)
    assert dormant is not None and dormant.dormant
    status = get_completion_status(
        cell,
        tmp_path,
        expected_qids=("q1", "q2"),
        now=1000.0,
        server_pool_generation="fleet-1",
    )
    assert status.status is CompletionState.RETRYABLE
    assert not status.eligible_for_retry
    revived = get_completion_status(
        cell,
        tmp_path,
        expected_qids=("q1", "q2"),
        now=1000.0,
        server_pool_generation="fleet-2",
    )
    assert revived.status is CompletionState.RETRYABLE
    assert revived.eligible_for_retry

    context_dir = tmp_path / "context"
    record_failure(
        context_dir,
        cell,
        RuntimeError("too long"),
        classification=FailureClass.CONTEXT_CAPACITY,
        now=200.0,
        serving_profile="32B",
        code_version="a",
    )
    permanent = get_completion_status(
        cell,
        context_dir,
        expected_qids=("q1", "q2"),
        serving_profile="32B",
    )
    assert permanent.status is CompletionState.PERMANENT
    unblocked = get_completion_status(
        cell,
        context_dir,
        expected_qids=("q1", "q2"),
        serving_profile="32B-long",
    )
    assert unblocked.status is CompletionState.RETRYABLE
    assert unblocked.eligible_for_retry

    code_unblocked = get_completion_status(
        cell,
        context_dir,
        expected_qids=("q1", "q2"),
        serving_profile="32B",
        code_version="b",
    )
    assert code_unblocked.status is CompletionState.RETRYABLE
    assert code_unblocked.eligible_for_retry


def test_failure_ledger_rejects_semantically_impossible_state(tmp_path, cell):
    failure = record_failure(
        tmp_path,
        cell,
        RuntimeError("endpoint unavailable"),
        classification=FailureClass.CONNECTION,
        now=100.0,
        base_backoff_s=10.0,
        server_pool_generation="fleet-1",
    ).to_dict()
    failure["disposition"] = "permanent"
    failure["next_eligible_at"] = None
    io.write_json(tmp_path / "failure.json", failure)

    with pytest.raises(CorruptArtifactError, match="must have retryable disposition"):
        read_failure(tmp_path)
    status = get_completion_status(
        cell, tmp_path, expected_qids=("q1", "q2")
    )
    assert status.status is CompletionState.CORRUPT
    assert any("retryable disposition" in error for error in status.errors)


def test_runner_skips_dormant_failure_until_server_pool_generation_changes(
    tmp_path, monkeypatch, cell
):
    cell = ExperimentCell.from_dict(cell.to_dict() | {"n_questions": 1})
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    question = Question(
        "q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
    )
    _mock_verified_questions(monkeypatch, cell, [question])
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    cdir = tmp_path / "pool-revival" / "cells" / cell.cell_id
    record_failure(
        cdir,
        cell,
        RuntimeError("fleet unavailable"),
        classification=FailureClass.CONNECTION,
        now=100.0,
        max_attempts=1,
        serving_profile="0.6B",
        code_version="test-commit",
        server_pool_generation="stable-fleet",
    )
    selections = 0

    def select(*_args, **_kwargs):
        nonlocal selections
        selections += 1
        raise RuntimeError("revived worker reached endpoint selection")

    monkeypatch.setattr(runner, "_pick_endpoint", select)
    monkeypatch.setattr(
        runner,
        "_current_server_pool_generation",
        lambda *_args, **_kwargs: "stable-fleet",
    )
    runner.run_cell(cell, "pool-revival")
    assert selections == 0

    monkeypatch.setattr(
        runner,
        "_current_server_pool_generation",
        lambda *_args, **_kwargs: "replacement-fleet",
    )
    with pytest.raises(RuntimeError, match="reached endpoint selection"):
        runner.run_cell(cell, "pool-revival")
    assert selections == 1
    failure = read_failure(cdir)
    assert failure is not None
    assert failure.server_pool_generation == "replacement-fleet"


def test_runner_rejects_nonmanifest_cell_before_directory_or_endpoint_side_effects(
    tmp_path, monkeypatch, cell
):
    from agents_scaling.experiment.manifest import freeze_manifest

    manifested = ExperimentCell.from_dict(
        cell.to_dict() | {"n_questions": 1, "temperature": 0.0}
    )
    impostor = ExperimentCell.from_dict(
        manifested.to_dict() | {"temperature": 0.9}
    )
    assert impostor.cell_id == manifested.cell_id
    assert impostor != manifested

    run_root = tmp_path / "membership"
    run_root.mkdir()
    (run_root / "cells.json").write_text(
        json.dumps([manifested.to_dict()]) + "\n", encoding="utf-8"
    )
    freeze_manifest(run_root)
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(
        runner,
        "_current_server_pool_generation",
        lambda *_args, **_kwargs: pytest.fail(
            "manifest mismatch reached server-pool inspection"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "manifest mismatch reached endpoint selection"
        ),
    )

    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="not the exact entry in frozen manifest",
    ):
        runner.run_cell(impostor, "membership")

    assert not (run_root / "cells").exists()


def test_runner_rejects_dispatcher_sidecar_pin_before_cell_or_server_side_effects(
    tmp_path, monkeypatch, cell
):
    from agents_scaling.experiment.manifest import freeze_manifest

    manifested = ExperimentCell.from_dict(
        cell.to_dict() | {"n_questions": 1, "temperature": 0.0}
    )
    run_root = tmp_path / "sidecar-pin"
    run_root.mkdir()
    (run_root / "cells.json").write_text(
        json.dumps([manifested.to_dict()]) + "\n", encoding="utf-8"
    )
    freeze_manifest(run_root)
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(
        runner,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: SimpleNamespace(sidecar_sha256="f" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_current_server_pool_generation",
        lambda *_args, **_kwargs: pytest.fail(
            "sidecar mismatch reached server-pool inspection"
        ),
    )

    with pytest.raises(
        completion.ExperimentConfigurationError,
        match="dispatcher-pinned benchmark contract",
    ):
        runner.run_cell(
            manifested,
            "sidecar-pin",
            expected_benchmark_contracts_sha256="e" * 64,
        )

    assert not (run_root / "cells").exists()


def test_failure_ledger_reads_legacy_endpoint_generation_as_pool_generation(
    tmp_path, cell
):
    failure = record_failure(
        tmp_path,
        cell,
        RuntimeError("legacy fleet failure"),
        classification=FailureClass.CONNECTION,
        now=100.0,
        max_attempts=1,
        server_pool_generation="legacy-pool-hash",
    )
    payload = failure.to_dict()
    payload["schema_version"] = 1
    payload["endpoint_generation"] = payload.pop("server_pool_generation")
    io.write_json(tmp_path / "failure.json", payload)

    restored = read_failure(tmp_path)

    assert restored is not None
    assert restored.schema_version == 1
    assert restored.server_pool_generation == "legacy-pool-hash"

    untracked = tmp_path / "untracked"
    record_failure(
        untracked,
        cell,
        RuntimeError("pre-generation-ledger failure"),
        classification=FailureClass.CONNECTION,
        now=101.0,
        max_attempts=1,
    )
    revived = get_completion_status(
        cell,
        untracked,
        expected_qids=("q1", "q2"),
        now=102.0,
        server_pool_generation="first-observed-pool",
    )
    assert revived.eligible_for_retry


def test_serving_context_capacity_error_is_classified_as_permanent_kind():
    error = ContextCapacityError(
        ContextPreflight(
            profile_name="32B",
            prompt_tokens=16000,
            requested_output_tokens=1024,
            reserve_tokens=128,
            served_context=16384,
        )
    )
    assert infer_failure_class(error) is FailureClass.CONTEXT_CAPACITY


def test_local_protocol_and_http_bad_requests_are_configuration_failures():
    bad_request = type("BadRequestError", (RuntimeError,), {})("invalid request")
    assert infer_failure_class(bad_request) is FailureClass.CONFIGURATION
    assert infer_failure_class(
        ThinkingBudgetProtocolError("invalid phase contract")
    ) is FailureClass.CONFIGURATION


def test_untrusted_server_response_protocol_failure_is_configuration_blocked(tmp_path, cell):
    error = ServerResponseProtocolError(
        "delimiter mismatch: start_positions=[0, 7], end_positions=[9]"
    )
    assert isinstance(error, ThinkingBudgetProtocolError)
    assert infer_failure_class(error) is FailureClass.CONFIGURATION

    failure = record_failure(
        tmp_path,
        cell,
        error,
        now=100.0,
        base_backoff_s=10.0,
        max_backoff_s=60.0,
    )
    assert failure.classification == FailureClass.CONFIGURATION.value
    assert failure.disposition == "permanent"
    assert failure.next_eligible_at is None
    assert not failure.dormant
    assert failure.last_error == {
        "type": "ServerResponseProtocolError",
        "message": str(error),
    }


def test_full_envelope_censor_is_valid_but_smaller_or_unproven_caps_are_rejected(
    cell,
):
    q1 = Question("q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"])
    censored, _ = _full_envelope_censor(cell, q1)
    completed = _record(cell, "q2", current=True)
    meta = _current_meta(cell) | {
        "completed_question_count": 1,
        "length_censored_question_count": 1,
        "mean_reasoning_tokens": None,
        "mean_reasoning_tokens_exact": 0.0,
        "exact_reasoning_question_count": 1,
        "nonexact_reasoning_question_count": 1,
    }
    records = [censored, completed]
    assert validate_completion_payload(cell, ("q1", "q2"), records, meta) == ()

    wrong_seed = json.loads(json.dumps(censored))
    wrong_seed["censored_generation"]["seed"] = cell.seed + 1
    errors = validate_completion_payload(
        cell, ("q1", "q2"), [wrong_seed, completed], meta
    )
    assert any("seed does not match" in error for error in errors)

    smaller_cap = json.loads(json.dumps(censored))
    censor = smaller_cap["censored_generation"]
    censor["requested_output_tokens"] -= 1
    censor["completion_tokens"] -= 1
    censor["completion_token_ids"] = censor["completion_token_ids"][:-1]
    censor["completion_token_id_sha256"] = token_ids_sha256(
        censor["completion_token_ids"]
    )
    errors = validate_completion_payload(
        cell, ("q1", "q2"), [smaller_cap, completed], meta
    )
    assert any("full registered context envelope" in error for error in errors)

    below_floor = json.loads(json.dumps(censored))
    censor = below_floor["censored_generation"]
    requested = requested_generation_tokens(cell.reasoning_level) - 1
    prompt_tokens = get_model(cell.model_size).max_model_len - requested - 128
    censor["requested_output_tokens"] = requested
    censor["completion_tokens"] = requested
    censor["completion_token_ids"] = [42] * requested
    censor["completion_token_id_sha256"] = token_ids_sha256(
        censor["completion_token_ids"]
    )
    censor["prompt_tokens"] = prompt_tokens
    censor["prompt_token_ids"] = [11] * prompt_tokens
    censor["prompt_token_id_sha256"] = token_ids_sha256(censor["prompt_token_ids"])
    below_floor["efficiency_raw"]["total_prompt_tokens"] = prompt_tokens
    below_floor["efficiency_raw"]["total_completion_tokens"] = requested
    errors = validate_completion_payload(
        cell, ("q1", "q2"), [below_floor, completed], meta
    )
    assert any("below the registered output floor" in error for error in errors)

    missing_ids = json.loads(json.dumps(censored))
    missing_ids["censored_generation"].pop("completion_token_ids")
    errors = validate_completion_payload(
        cell, ("q1", "q2"), [missing_ids, completed], meta
    )
    assert any("completion_token_ids" in error for error in errors)

    invented_delimiter = json.loads(json.dumps(censored))
    invented_delimiter["censored_generation"][
        "completion_think_start_positions"
    ] = [0]
    errors = validate_completion_payload(
        cell, ("q1", "q2"), [invented_delimiter, completed], meta
    )
    assert any("exactly match retained delimiter token IDs" in error for error in errors)


def _mixed_protocol_censor_payload(
    cell: ExperimentCell,
) -> tuple[list[dict], dict, tuple[Question, ...]]:
    """Return one valid schema-5 protocol censor plus one completed outcome."""

    questions = _truth_questions(cell)
    censored, _ = _protocol_censor(cell, questions[0])
    completed = _record(cell, "q2", current=True)
    # Truth-aware validation requires the complete MCQ option distribution.
    completed["per_agent"][0]["option_logprobs"]["B"] = 0.0
    meta = _current_meta(cell) | {
        "completed_question_count": 1,
        "length_censored_question_count": 0,
        "protocol_censored_question_count": 1,
        "artifact_schema_counts": {str(ARTIFACT_SCHEMA_VERSION): 2},
        "mean_reasoning_tokens": None,
        "mean_reasoning_tokens_exact": 0.0,
        "exact_reasoning_question_count": 1,
        "nonexact_reasoning_question_count": 1,
    }
    return [censored, completed], meta, questions


def test_schema5_accepts_mixed_completed_and_protocol_censored_outcomes(cell):
    records, meta, questions = _mixed_protocol_censor_payload(cell)

    assert validate_completion_payload(
        cell,
        ("q1", "q2"),
        records,
        meta,
        expected_questions=questions,
    ) == ()


def test_schema5_accepts_complete_concurrent_wave_with_multiple_censors():
    cell = ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="independent",
        benchmark="gpqa",
        n_agents=2,
        n_samples=1,
        n_questions=2,
        seed=7,
    )
    questions = _truth_questions(cell)
    length_record, length_error = _full_envelope_censor(cell, questions[0])
    length_coordinate = length_record["observed_topology_coordinates"][0]

    prompt_ids = [11, 12]
    completion_ids = [QWEN_THINK_END_TOKEN_ID, QWEN_IM_END_TOKEN_ID]
    context_limit = get_model(cell.model_size).max_model_len
    requested = context_limit - len(prompt_ids) - 128
    protocol_error = GenerationProtocolCensorError(
        protocol_violation_codes=["unexpected_reasoning_delimiter"],
        finish_reason="stop",
        requested_output_tokens=requested,
        completion_tokens=len(completion_ids),
        output_capacity_floor_tokens=requested_generation_tokens(
            cell.reasoning_level
        ),
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        decoded_completion="</think>",
        server_content="</think>",
        server_reasoning="",
        seed=cell.seed + 1,
        serving_profile=cell.model_size,
        effective_context_limit=context_limit,
        completion_think_end_positions=[0],
        endpoint_generation="test-endpoint-generation",
        created_at=102.0,
    ).enrich(
        qid=questions[0].qid,
        agent_id="agent1",
        round_idx=0,
        generation_role="topology",
    )
    protocol_coordinate = {
        "coordinate_key": "topology:agent1:0",
        "request": {
            "generation_role": "topology",
            "qid": questions[0].qid,
            "agent_id": "agent1",
            "round": 0,
            "seed": cell.seed + 1,
            "sample_index": None,
            "peer_context": {"sha256": completion._EMPTY_SHA256, "utf8_bytes": 0},
            "max_tokens": 4096,
            "elicit_cot": True,
        },
        "outcome": {
            "termination_status": TERMINATION_PROTOCOL_CENSORED,
            "agent_output": None,
            "censored_generation": protocol_error.to_censored_generation(),
        },
        "observed_at": 102.0,
        "producer_wall_ms": 6.0,
    }
    censored = runner._length_censored_record(
        cell=cell,
        question=questions[0],
        error=length_error,
        wall_ms=8.0,
        profile_meta={
            "serving_profile": cell.model_size,
            "effective_context_limit": context_limit,
            "tensor_parallel_size": get_model(cell.model_size).tp_size,
        },
        benchmark_contract_sha256=build_question_contract(
            BenchmarkContractKey.from_cell(cell), questions
        )["question_contract_sha256"],
        observed_topology_coordinates=[length_coordinate, protocol_coordinate],
    )
    completed = _record(cell, "q2", current=True)
    completed["per_agent"][0]["option_logprobs"]["B"] = 0.0
    completed["per_agent"][1]["option_logprobs"]["B"] = 0.0
    meta = _current_meta(cell) | {
        "completed_question_count": 1,
        "length_censored_question_count": 1,
        "protocol_censored_question_count": 0,
        "mean_reasoning_tokens": None,
        "mean_reasoning_tokens_exact": 0.0,
        "exact_reasoning_question_count": 1,
        "nonexact_reasoning_question_count": 1,
    }

    assert validate_completion_payload(
        cell,
        ("q1", "q2"),
        [censored, completed],
        meta,
        expected_questions=questions,
    ) == ()

    nondeterministic_terminal = runner._length_censored_record(
        cell=cell,
        question=questions[0],
        error=protocol_error,
        wall_ms=8.0,
        profile_meta={
            "serving_profile": cell.model_size,
            "effective_context_limit": context_limit,
            "tensor_parallel_size": get_model(cell.model_size).tp_size,
        },
        benchmark_contract_sha256=build_question_contract(
            BenchmarkContractKey.from_cell(cell), questions
        )["question_contract_sha256"],
        observed_topology_coordinates=[length_coordinate, protocol_coordinate],
    )
    nondeterministic_meta = meta | {
        "length_censored_question_count": 0,
        "protocol_censored_question_count": 1,
    }
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        [nondeterministic_terminal, completed],
        nondeterministic_meta,
        expected_questions=questions,
    )
    assert any("deterministic map-order" in error for error in errors), errors

    missing_sibling = json.loads(json.dumps(censored))
    missing_sibling["observed_topology_coordinates"] = [protocol_coordinate]
    missing_sibling["efficiency_raw"]["n_turns"] = 1
    missing_sibling["efficiency_raw"]["total_prompt_tokens"] = 2
    missing_sibling["efficiency_raw"]["total_completion_tokens"] = 2
    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        [missing_sibling, completed],
        meta,
        expected_questions=questions,
    )
    assert any("exact legal terminal wave" in error for error in errors), errors


@pytest.mark.parametrize("mutation", ["extra_field", "false_cot", "swapped_output"])
def test_schema5_snapshot_request_and_outcome_are_exactly_bound(cell, mutation):
    records, meta, questions = _mixed_protocol_censor_payload(cell)
    coordinate = records[0]["observed_topology_coordinates"][0]
    if mutation == "extra_field":
        coordinate["request"]["invented"] = True
    elif mutation == "false_cot":
        coordinate["request"]["elicit_cot"] = False
    else:
        coordinate["outcome"]["censored_generation"]["agent_id"] = "agent9"

    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        records,
        meta,
        expected_questions=questions,
    )

    expected = {
        "extra_field": "request has the wrong fields",
        "false_cot": "elicit_cot must be true",
        "swapped_output": "coordinate does not match request",
    }[mutation]
    assert any(expected in error for error in errors), errors


def test_premature_reasoning_eos_is_not_counted_as_reasoning():
    cell = ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="unlimited",
        topology="single_agent",
        benchmark="gpqa",
        n_agents=1,
        n_samples=1,
        n_questions=1,
        seed=7,
    )
    censor = {
        "finish_reason": "stop",
        "completion_token_ids": [QWEN_THINK_START_TOKEN_ID, QWEN_IM_END_TOKEN_ID],
        "completion_think_start_positions": [0],
        "completion_think_end_positions": [],
    }

    assert runner._censored_reasoning_tokens(censor, thinking_enabled=True) == 0
    assert completion._auxiliary_censored_reasoning_tokens(censor, cell) == 0


@pytest.mark.parametrize(
    ("corruption", "error_fragment"),
    [
        ("violation_codes", "violation codes do not match retained response"),
        ("token_hash", "does not match retained token IDs"),
        ("seed", "seed does not match topology schedule"),
        ("terminal", "actual_terminal_token_id does not match completion"),
        ("snapshot", "coordinate/seed is not on the manifest schedule"),
    ],
)
def test_schema5_protocol_censor_rejects_tampered_exact_provenance(
    cell, corruption, error_fragment
):
    records, meta, questions = _mixed_protocol_censor_payload(cell)
    records = json.loads(json.dumps(records))
    censor = records[0]["censored_generation"]
    if corruption == "violation_codes":
        censor["protocol_violation_codes"] = ["unknown_terminal_under_v4"]
    elif corruption == "token_hash":
        censor["completion_token_id_sha256"] = "f" * 64
    elif corruption == "seed":
        censor["seed"] += 1
    elif corruption == "terminal":
        censor["actual_terminal_token_id"] = QWEN_THINK_END_TOKEN_ID
    else:
        records[0]["observed_topology_coordinates"][0]["request"]["seed"] += 1

    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        records,
        meta,
        expected_questions=questions,
    )

    assert any(error_fragment in error for error in errors), errors


def test_schema4_completed_artifacts_remain_valid_under_schema5_reader(cell):
    questions = _truth_questions(cell)
    records = [
        _record(cell, "q1", current=True),
        _record(cell, "q2", current=True),
    ]
    for record in records:
        record["schema_version"] = 4
        record.pop("observed_topology_coordinates")
        record["per_agent"][0]["option_logprobs"]["B"] = 0.0
    meta = _current_meta(cell)
    meta["schema_version"] = 4
    for field in (
        "generation_censor_protocol_version",
        "generation_censor_protocol_hash",
        "protocol_censored_question_count",
        "artifact_schema_counts",
    ):
        meta.pop(field)

    assert validate_completion_payload(
        cell,
        ("q1", "q2"),
        records,
        meta,
        expected_questions=questions,
    ) == ()


def test_schema4_rejects_protocol_censored_status(cell):
    records, meta, questions = _mixed_protocol_censor_payload(cell)
    for record in records:
        record["schema_version"] = 4
        record.pop("observed_topology_coordinates")
    meta["schema_version"] = 4
    for field in (
        "generation_censor_protocol_version",
        "generation_censor_protocol_hash",
        "protocol_censored_question_count",
        "artifact_schema_counts",
    ):
        meta.pop(field)

    errors = validate_completion_payload(
        cell,
        ("q1", "q2"),
        records,
        meta,
        expected_questions=questions,
    )

    assert any(
        "question result termination_status is not registered" in error
        for error in errors
    )
    assert any(
        "schema-4 meta cannot certify protocol-censored results" in error
        for error in errors
    )


def test_current_self_consistency_requires_all_scheduled_outcomes_and_separate_costs():
    cell = _self_consistency_cell()
    rows = [
        _record(cell, "q1", current=True),
        _record(cell, "q2", current=True),
    ]
    rows[0]["self_consistency"] = _self_consistency_payload_for_test(
        cell, "q1", censored_index=1
    )
    rows[1]["self_consistency"] = _self_consistency_payload_for_test(cell, "q2")
    meta = _current_meta(cell)

    assert validate_completion_payload(cell, ("q1", "q2"), rows, meta) == ()
    assert rows[0]["final_answer"] == "A"
    assert rows[0]["per_agent"]
    assert rows[0]["self_consistency"]["self_consistency_conf"] is None
    assert rows[0]["self_consistency"]["semantic_entropy_conf"] is None

    missing = json.loads(json.dumps(rows))
    missing[0]["self_consistency"]["samples"].pop()
    errors = validate_completion_payload(cell, ("q1", "q2"), missing, meta)
    assert any("exactly manifest n_samples" in error for error in errors)

    wrong_seed = json.loads(json.dumps(rows))
    wrong_seed[0]["self_consistency"]["samples"][0]["seed"] += 1
    errors = validate_completion_payload(cell, ("q1", "q2"), wrong_seed, meta)
    assert any("frozen schedule" in error for error in errors)

    duplicated_nested_coordinate = json.loads(json.dumps(rows))
    samples = duplicated_nested_coordinate[0]["self_consistency"]["samples"]
    samples[2]["censored_generation"] = json.loads(
        json.dumps(samples[1]["censored_generation"])
    )
    samples[2]["agent_output"] = None
    samples[2]["termination_status"] = TERMINATION_LENGTH_CENSORED
    auxiliary = duplicated_nested_coordinate[0]["self_consistency"][
        "auxiliary_efficiency_raw"
    ]
    auxiliary["completed_samples"] -= 1
    auxiliary["length_censored_samples"] += 1
    auxiliary["total_prompt_tokens"] += samples[1]["censored_generation"][
        "prompt_tokens"
    ] - rows[0]["self_consistency"]["samples"][2]["agent_output"]["prompt_tokens"]
    auxiliary["total_completion_tokens"] += samples[1]["censored_generation"][
        "completion_tokens"
    ] - rows[0]["self_consistency"]["samples"][2]["agent_output"][
        "completion_tokens"
    ]
    duplicated_nested_coordinate[0]["self_consistency"][
        "length_censored_sample_count"
    ] += 1
    duplicated_nested_coordinate[0]["self_consistency"]["completed_sample_count"] -= 1
    errors = validate_completion_payload(
        cell, ("q1", "q2"), duplicated_nested_coordinate, meta
    )
    assert any("does not match wrapper" in error for error in errors)

    confidence_on_censor = json.loads(json.dumps(rows))
    confidence_on_censor[0]["self_consistency"]["self_consistency_conf"] = 0.5
    errors = validate_completion_payload(
        cell, ("q1", "q2"), confidence_on_censor, meta
    )
    assert any("undefined when any sample censors" in error for error in errors)

    folded_cost = json.loads(json.dumps(rows))
    folded_cost[0]["efficiency_raw"]["total_prompt_tokens"] += folded_cost[0][
        "self_consistency"
    ]["auxiliary_efficiency_raw"]["total_prompt_tokens"]
    errors = validate_completion_payload(cell, ("q1", "q2"), folded_cost, meta)
    assert any("does not match per_agent sum" in error for error in errors)

    empty = json.loads(json.dumps(rows))
    empty[0]["self_consistency"] = {}
    errors = validate_completion_payload(cell, ("q1", "q2"), empty, meta)
    assert any("every configured auxiliary sample" in error for error in errors)


def test_runner_keeps_primary_when_auxiliary_censors_and_runs_later_samples(
    tmp_path, monkeypatch
):
    cell = ExperimentCell.from_dict(
        _self_consistency_cell().to_dict() | {"n_questions": 1}
    )
    question = Question(
        "q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
    )
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    _mock_verified_questions(monkeypatch, cell, [question])
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: ServerEntry(
            model_size="0.6B",
            hf_id="fake",
            host="node",
            port=8000,
            slurm_job_id="server-1",
            started_at=10.0,
        ),
    )

    primary = _current_agent_output(cell, cell.seed)
    topology_result = TopologyResult(
        final_answer="A",
        per_agent=[primary],
        n_turns=1,
        n_messages=0,
        n_rounds=1,
        n_agents=1,
        system_conf=system_confidences([primary], "A"),
    )
    topology_runs = 0

    def run_primary(_question):
        nonlocal topology_runs
        topology_runs += 1
        return topology_result

    monkeypatch.setattr(
        runner,
        "build_topology",
        lambda *args, **kwargs: SimpleNamespace(run=run_primary),
    )

    censor_row, _ = _full_envelope_censor(cell, question)
    censor = censor_row["censored_generation"]
    censor["generation_role"] = "self_consistency"
    censor["sample_index"] = 1
    censor["seed"] = cell.seed + 1001
    seen_samples = []

    class FakeAgent:
        def prepare_calibration(self, _question):
            return None

        def sample_one(self, _question, *, sample_index, base_seed):
            seen_samples.append(sample_index)
            seed = base_seed + sample_index
            if sample_index == 1:
                return SelfConsistencySample(
                    sample_index=sample_index,
                    seed=seed,
                    termination_status=TERMINATION_LENGTH_CENSORED,
                    censored_generation=censor,
                )
            return SelfConsistencySample(
                sample_index=sample_index,
                seed=seed,
                termination_status=TERMINATION_COMPLETED,
                agent_output=_current_agent_output(cell, seed),
            )

    monkeypatch.setattr(runner, "_build_agents", lambda *args, **kwargs: [FakeAgent()])

    runner.run_cell(cell, "sc-censor")

    assert topology_runs == 1
    assert seen_samples == [0, 1, 2]
    cdir = tmp_path / "sc-censor" / "cells" / cell.cell_id
    row = json.loads((cdir / "results.jsonl").read_text())
    assert row["termination_status"] == TERMINATION_COMPLETED
    assert row["final_answer"] == "A"
    assert len(row["per_agent"]) == 1
    assert row["self_consistency"]["length_censored_sample_count"] == 1
    assert row["self_consistency"]["self_consistency_conf"] is None
    assert row["self_consistency"]["semantic_entropy_conf"] is None
    assert row["efficiency_raw"]["total_prompt_tokens"] == primary.prompt_tokens
    assert row["self_consistency"]["auxiliary_efficiency_raw"][
        "total_prompt_tokens"
    ] > primary.prompt_tokens
    assert get_completion_status(
        cell, cdir, expected_qids=("q1",)
    ).status is CompletionState.COMPLETE


def test_runner_score_probe_failure_cannot_resample_primary_chat(
    tmp_path, monkeypatch, cell
):
    cell = ExperimentCell.from_dict(cell.to_dict() | {"n_questions": 1})
    question = Question(
        "q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
    )
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    _mock_verified_questions(monkeypatch, cell, [question])
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())

    class ProbeConnectionError(Exception):
        pass

    monkeypatch.setattr(runner, "_connection_errors", lambda: (ProbeConnectionError,))
    endpoints = [
        ServerEntry(
            model_size="0.6B",
            hf_id="fake",
            host="node1",
            port=8000,
            slurm_job_id="server-1",
            started_at=10.0,
        ),
        ServerEntry(
            model_size="0.6B",
            hf_id="fake",
            host="node2",
            port=8001,
            slurm_job_id="server-2",
            started_at=11.0,
        ),
    ]
    selections = 0

    def select(*args, **kwargs):
        nonlocal selections
        selected = endpoints[min(selections, 1)]
        selections += 1
        return selected

    monkeypatch.setattr(runner, "_pick_endpoint", select)
    probe_attempts = []
    chat_attempts = []

    class FakeClient:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

        def score_options(self, prompt, option_letters):
            probe_attempts.append(self.endpoint)
            if "node1" in self.endpoint:
                raise ProbeConnectionError("forced probe outage")
            return OptionScores(
                probs={"A": 0.8, "B": 0.2},
                raw_logprobs={"A": -0.1, "B": -1.5},
            )

        def chat(self, **kwargs):
            chat_attempts.append((self.endpoint, kwargs["seed"]))
            return ChatResult(
                text="Answer: A",
                prompt_tokens=2,
                completion_tokens=3,
                finish_reason="stop",
                reasoning_token_source="none",
                generation_phase_finish_reasons=["stop"],
                generation_phase_seeds=[kwargs["seed"]],
                generation_phase_prompt_tokens=[2],
                generation_phase_completion_tokens=[3],
                output_capacity_floor_tokens=requested_generation_tokens(
                    cell.reasoning_level
                ),
                generation_phase_requested_max_tokens=[32768 - 2 - 128],
                generation_phase_prompt_token_id_hashes=["0" * 64],
                generation_phase_completion_token_id_hashes=["1" * 64],
                endpoint_generation=(
                    "server-1@node1:8000#10.000000"
                    if "node1" in self.endpoint
                    else "server-2@node2:8001#11.000000"
                ),
            )

    def build_agents(
        _cell,
        base_url,
        _served_model,
        _profile,
        context_tokenizer=None,
        endpoint_generation=None,
        system_prompt=None,
    ):
        return [
            Agent(
                "agent0",
                FakeClient(base_url),
                "system",
                temperature=cell.temperature,
                reasoning_level=cell.reasoning_level,
            )
        ]

    monkeypatch.setattr(runner, "_build_agents", build_agents)

    runner.run_cell(cell, "probe-before-chat")

    assert probe_attempts == [
        "http://node1:8000/v1",
        "http://node2:8001/v1",
    ]
    assert chat_attempts == [("http://node2:8001/v1", cell.seed)]
    cdir = tmp_path / "probe-before-chat" / "cells" / cell.cell_id
    row = json.loads((cdir / "results.jsonl").read_text())
    assert row["final_answer"] == "A"
    assert row["per_agent"][0]["generation_phase_seeds"] == [cell.seed]


def test_all_multiagent_probes_finish_before_first_topology_chat(
    tmp_path, monkeypatch
):
    cell = ExperimentCell(
        model_size="0.6B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="independent",
        benchmark="gpqa",
        n_agents=2,
        n_samples=1,
        n_questions=1,
        seed=7,
    )
    question = Question(
        "q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
    )
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    _mock_verified_questions(monkeypatch, cell, [question])
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())

    class ProbeConnectionError(Exception):
        pass

    monkeypatch.setattr(runner, "_connection_errors", lambda: (ProbeConnectionError,))
    endpoint = ServerEntry(
        model_size="0.6B",
        hf_id="fake",
        host="node1",
        port=8000,
        slurm_job_id="server-1",
        started_at=10.0,
    )
    selections = 0

    def select(*args, **kwargs):
        nonlocal selections
        selections += 1
        return endpoint if selections == 1 else None

    monkeypatch.setattr(runner, "_pick_endpoint", select)
    events = []

    class ProbeClient:
        def __init__(self, agent_index):
            self.agent_index = agent_index

        def score_options(self, prompt, option_letters):
            events.append(("probe", self.agent_index))
            if self.agent_index == 1:
                raise ProbeConnectionError("second agent probe unavailable")
            return OptionScores(
                probs={"A": 0.5, "B": 0.5},
                raw_logprobs={"A": -0.7, "B": -0.7},
            )

        def chat(self, **kwargs):
            events.append(("chat", self.agent_index))
            raise AssertionError("no chat may precede all option probes")

    def build_agents(*args, **kwargs):
        return [
            Agent(
                f"agent{index}",
                ProbeClient(index),
                "system",
                reasoning_level=cell.reasoning_level,
            )
            for index in range(2)
        ]

    monkeypatch.setattr(runner, "_build_agents", build_agents)
    topology_runs = 0

    def forbidden_run(_question):
        nonlocal topology_runs
        topology_runs += 1
        raise AssertionError("topology ran before calibration barrier")

    monkeypatch.setattr(
        runner,
        "build_topology",
        lambda *args, **kwargs: SimpleNamespace(run=forbidden_run),
    )

    with pytest.raises(ProbeConnectionError, match="second agent probe"):
        runner.run_cell(cell, "multiagent-probe-barrier")

    assert events == [("probe", 0), ("probe", 1)]
    assert topology_runs == 0


@pytest.mark.parametrize(
    ("censor_kind", "expected_status"),
    [
        ("length", TERMINATION_LENGTH_CENSORED),
        ("protocol", TERMINATION_PROTOCOL_CENSORED),
    ],
)
def test_runner_retains_one_censor_without_retry_or_failure(
    tmp_path, monkeypatch, cell, censor_kind, expected_status
):
    cell = ExperimentCell.from_dict(cell.to_dict() | {"n_questions": 1})
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    question = Question(
        "q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]
    )
    if censor_kind == "protocol":
        _, error = _protocol_censor(cell, question)
    else:
        _, error = _full_envelope_censor(cell, question)
    _mock_verified_questions(monkeypatch, cell, [question])
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: ServerEntry(
            model_size="0.6B",
            hf_id="fake",
            host="node",
            port=8000,
            slurm_job_id="server-1",
            started_at=10.0,
        ),
    )
    monkeypatch.setattr(
        runner, "LogprobClient", lambda *args, **kwargs: _NoopScoringClient()
    )
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())

    attempts = 0

    class _CensoringAgent:
        agent_id = "agent0"

        def prepare_calibration(self, _question):
            return None

        def answer(self, _question, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise error

    monkeypatch.setattr(
        runner, "_build_agents", lambda *args, **kwargs: [_CensoringAgent()]
    )

    monkeypatch.setattr(
        runner,
        "build_topology",
        lambda _topology, agents, *args, **kwargs: SimpleNamespace(
            run=lambda observed_question: agents[0].answer(
                observed_question, seed=cell.seed
            )
        ),
    )

    runner.run_cell(cell, "censor-once")

    assert attempts == 1
    cdir = tmp_path / "censor-once" / "cells" / cell.cell_id
    rows = [
        json.loads(line)
        for line in (cdir / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["termination_status"] == expected_status
    assert rows[0]["correct"] is False
    assert rows[0]["final_answer"] is None
    assert not (cdir / "failure.json").exists()
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["completed_question_count"] == 0
    assert meta["length_censored_question_count"] == int(censor_kind == "length")
    assert meta["protocol_censored_question_count"] == int(
        censor_kind == "protocol"
    )
    status = get_completion_status(cell, cdir, expected_qids=("q1",))
    assert status.status is CompletionState.COMPLETE
    assert status.completed_question_count == 0
    assert status.length_censored_question_count == int(censor_kind == "length")
    assert status.protocol_censored_question_count == int(censor_kind == "protocol")


def test_runner_repairs_resume_and_recomputes_full_mean(tmp_path, monkeypatch, cell):
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    questions = [
        Question("q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]),
        Question("q2", "gpqa", "two?", "A", AnswerType.MCQ, ["yes", "no"]),
    ]
    _mock_verified_questions(monkeypatch, cell, questions)
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: ServerEntry(
            model_size="0.6B", hf_id="fake", host="node", port=8000
        ),
    )
    monkeypatch.setattr(
        runner, "LogprobClient", lambda *args, **kwargs: _NoopScoringClient()
    )
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())

    output = AgentOutput(
        agent_id="agent0",
        round=0,
        answer_choice="A",
        raw_text="A",
        cot_text="",
        intermediate_results="A",
        option_logprobs={"A": 1.0, "B": 0.0},
        prompt_tokens=2,
        completion_tokens=3,
        reasoning_tokens=0,
        finish_reason="stop",
        reasoning_token_source="none",
        thinking_budget_consumed_tokens=0,
        generation_phase_finish_reasons=["stop"],
        generation_phase_seeds=[7],
        generation_phase_prompt_tokens=[2],
        generation_phase_completion_tokens=[3],
        output_capacity_floor_tokens=requested_generation_tokens(cell.reasoning_level),
        generation_phase_requested_max_tokens=[32768 - 2 - 128],
        generation_phase_prompt_token_id_hashes=["0" * 64],
        generation_phase_completion_token_id_hashes=["1" * 64],
        endpoint_generation="test-endpoint-generation",
    )
    topology_result = TopologyResult(
        final_answer="A",
        per_agent=[output],
        n_turns=1,
        n_messages=0,
        n_rounds=1,
        n_agents=1,
        system_conf=system_confidences([output], "A"),
    )
    monkeypatch.setattr(
        runner, "build_topology", lambda *args, **kwargs: SimpleNamespace(run=lambda q: topology_result)
    )

    cdir = tmp_path / "resume" / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    first = _record(cell, "q1", reasoning_tokens=10)
    (cdir / "results.jsonl").write_text(
        json.dumps(first) + "\n" + "malformed\n" + json.dumps(first) + "\n"
    )

    runner.run_cell(cell, "resume")

    rows = [json.loads(line) for line in (cdir / "results.jsonl").read_text().splitlines()]
    assert [row["qid"] for row in rows] == ["q1", "q2"]
    assert "serving_profile" not in rows[0]
    assert rows[1]["serving_profile"] == "0.6B"
    assert rows[1]["effective_context_limit"] == 32768
    assert rows[1]["tensor_parallel_size"] == 1
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["mean_reasoning_tokens"] is None
    assert meta["mean_reasoning_tokens_exact"] == 0.0
    assert meta["exact_reasoning_question_count"] == 1
    assert meta["nonexact_reasoning_question_count"] == 1
    assert meta["mean_reasoning_word_count_legacy"] == 5.0
    assert meta["serving_profile"] == "0.6B"
    assert meta["serving_profile_counts"] == {"0.6B": 2}
    assert meta["serving_profile_inferred_counts"] == {"0.6B": 1}
    assert get_completion_status(
        cell, cdir, expected_qids=("q1", "q2")
    ).status is CompletionState.COMPLETE
    assert not (cdir / "failure.json").exists()


def test_runner_quarantines_legacy_bad_meta_without_needing_endpoint(
    tmp_path, monkeypatch, cell
):
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    questions = [
        Question("q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]),
        Question("q2", "gpqa", "two?", "A", AnswerType.MCQ, ["yes", "no"]),
    ]
    _mock_verified_questions(monkeypatch, cell, questions)
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: pytest.fail("full recovered coverage must not select an endpoint"),
    )

    cdir = tmp_path / "repair-meta" / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    io.write_jsonl(
        cdir / "results.jsonl",
        [_record(cell, "q1", 10), _record(cell, "q2", 20)],
    )
    bad_meta = _meta(cell)
    bad_meta["config_hash"] = "legacy-wrong-hash"
    io.write_json(cdir / "meta.json", bad_meta)

    runner.run_cell(cell, "repair-meta")

    repaired_meta = json.loads((cdir / "meta.json").read_text())
    assert repaired_meta["config_hash"] == cell.config_hash()
    assert repaired_meta["mean_reasoning_tokens"] is None
    assert repaired_meta["mean_reasoning_tokens_exact"] is None
    assert repaired_meta["exact_reasoning_question_count"] == 0
    assert repaired_meta["nonexact_reasoning_question_count"] == 2
    assert repaired_meta["mean_reasoning_word_count_legacy"] == 15.0
    assert repaired_meta["serving_profile_counts"] == {"0.6B": 2}
    assert repaired_meta["serving_profile_inferred_counts"] == {"0.6B": 2}
    quarantined = list((cdir / "quarantine").glob("meta.*.json"))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text())["config_hash"] == "legacy-wrong-hash"
    assert get_completion_status(
        cell, cdir, expected_qids=("q1", "q2")
    ).status is CompletionState.COMPLETE


def test_full_legacy_32b_repair_reports_historical_standard_profile(
    tmp_path, monkeypatch
):
    cell = _long_32b_cell()
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    questions = [
        Question("q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]),
        Question("q2", "gpqa", "two?", "A", AnswerType.MCQ, ["yes", "no"]),
    ]
    _mock_verified_questions(monkeypatch, cell, questions)
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: pytest.fail(
            "full recovered coverage must not select the newly routed long endpoint"
        ),
    )

    cdir = tmp_path / "legacy-32b" / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    io.write_jsonl(
        cdir / "results.jsonl",
        [_record(cell, "q1", 10), _record(cell, "q2", 20)],
    )
    bad_meta = _meta(cell)
    bad_meta["config_hash"] = "force-lock-protected-rebuild"
    io.write_json(cdir / "meta.json", bad_meta)

    runner.run_cell(cell, "legacy-32b")

    rows = [json.loads(line) for line in (cdir / "results.jsonl").read_text().splitlines()]
    assert all("serving_profile" not in row for row in rows)
    repaired_meta = json.loads((cdir / "meta.json").read_text())
    assert repaired_meta["serving_profile"] == "32B"
    assert repaired_meta["effective_context_limit"] == 16384
    assert repaired_meta["tensor_parallel_size"] == 1
    assert repaired_meta["serving_profile_counts"] == {"32B": 2}
    assert repaired_meta["serving_profile_inferred_counts"] == {"32B": 2}
    assert get_completion_status(
        cell, cdir, expected_qids=("q1", "q2")
    ).status is CompletionState.COMPLETE


def test_partial_legacy_32b_resume_reports_mixed_profile_counts(tmp_path, monkeypatch):
    cell = _long_32b_cell()
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))
    questions = [
        Question("q1", "gpqa", "one?", "A", AnswerType.MCQ, ["yes", "no"]),
        Question("q2", "gpqa", "two?", "A", AnswerType.MCQ, ["yes", "no"]),
    ]
    _mock_verified_questions(monkeypatch, cell, questions)
    monkeypatch.setattr(runner.io, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(runner, "tokenizer_for_profile", lambda _profile: object())
    monkeypatch.setattr(
        runner,
        "_pick_endpoint",
        lambda *args, **kwargs: ServerEntry(
            model_size="32B",
            hf_id="fake",
            host="node",
            port=8000,
            serving_profile="32B-long",
            served_model_name="32B",
            max_model_len=40960,
            tp_size=2,
        ),
    )
    monkeypatch.setattr(
        runner, "LogprobClient", lambda *args, **kwargs: _NoopScoringClient()
    )
    outputs = [
        AgentOutput(
            agent_id=f"agent{agent_id}",
            round=round_idx,
            answer_choice="A",
            raw_text="A",
            cot_text="",
            intermediate_results="A",
            option_logprobs={"A": 1.0, "B": 0.0},
            prompt_tokens=2,
            completion_tokens=(
                33 if (agent_id, round_idx) == (0, 0) else 3
            ),
            reasoning_tokens=30 if (agent_id, round_idx) == (0, 0) else 0,
            finish_reason="stop",
            reasoning_token_source="vllm_native_token_ids",
            thinking_budget_consumed_tokens=(
                30 if (agent_id, round_idx) == (0, 0) else 0
            ),
            generation_phase_finish_reasons=["stop"],
            generation_phase_seeds=[7 + agent_id + 100 * round_idx],
            generation_phase_prompt_tokens=[2],
            generation_phase_completion_tokens=[
                33 if (agent_id, round_idx) == (0, 0) else 3
            ],
            output_capacity_floor_tokens=requested_generation_tokens(
                cell.reasoning_level
            ),
            generation_phase_requested_max_tokens=[40960 - 2 - 128],
            generation_phase_prompt_token_id_hashes=["0" * 64],
            generation_phase_completion_token_id_hashes=["1" * 64],
            endpoint_generation="test-endpoint-generation",
            reasoning_start_token_index=2,
            reasoning_end_token_index=(
                33 if (agent_id, round_idx) == (0, 0) else 3
            ),
            reasoning_start_token_count=1,
            reasoning_end_token_count=1,
            peer_context_tokens=(20 if round_idx > 0 else 0),
            peer_context_sha256=(
                "2" * 64
                if round_idx > 0
                else (
                    "e3b0c44298fc1c149afbf4c8996fb924"
                    "27ae41e4649b934ca495991b7852b855"
                )
            ),
            peer_context_block_token_counts=([10, 10] if round_idx > 0 else []),
        )
        for round_idx in range(2)
        for agent_id in range(3)
    ]
    topology_result = TopologyResult(
        final_answer="A",
        per_agent=outputs,
        n_turns=6,
        n_messages=6,
        n_rounds=2,
        n_agents=3,
        system_conf=system_confidences(outputs[-3:], "A"),
    )
    monkeypatch.setattr(
        runner,
        "build_topology",
        lambda *args, **kwargs: SimpleNamespace(run=lambda question: topology_result),
    )

    cdir = tmp_path / "mixed-32b" / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    io.write_jsonl(cdir / "results.jsonl", [_record(cell, "q1", 10)])

    runner.run_cell(cell, "mixed-32b")

    rows = [json.loads(line) for line in (cdir / "results.jsonl").read_text().splitlines()]
    assert "serving_profile" not in rows[0]
    assert rows[1]["serving_profile"] == "32B-long"
    assert rows[1]["effective_context_limit"] == 40960
    assert rows[1]["tensor_parallel_size"] == 2
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["serving_profile"] == "mixed"
    assert meta["effective_context_limit"] is None
    assert meta["tensor_parallel_size"] is None
    assert meta["serving_profile_counts"] == {"32B": 1, "32B-long": 1}
    assert meta["serving_profile_inferred_counts"] == {"32B": 1}
    assert get_completion_status(
        cell, cdir, expected_qids=("q1", "q2")
    ).status is CompletionState.COMPLETE
