"""Native vLLM thinking-budget and exact token-provenance tests."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from agents_scaling.agents.base_agent import Agent
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import ReasoningLevel
from agents_scaling.experiment.completion import FailureClass, infer_failure_class
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    GENERATION_CENSOR_PROTOCOL_HASH,
    GENERATION_CENSOR_PROTOCOL_VERSION,
    QWEN_END_OF_TEXT_TOKEN_ID,
    QWEN_IM_END_TOKEN_ID,
    QWEN_THINK_END_TOKEN_ID,
    QWEN_THINK_START_TOKEN_ID,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
    ChatResult,
    GenerationProtocolCensorError,
    GenerationTruncationError,
    LogprobClient,
    OptionScores,
    ServerResponseProtocolError,
    ThinkingBudgetProtocolError,
    thinking_budget_protocol_metadata,
    token_ids_sha256,
)
from agents_scaling.serving.context import (
    CONTEXT_RESERVE_TOKENS,
    ContextCapacityError,
)
from agents_scaling.serving.profiles import ServingProfile


class ExactTokenizer:
    """Tiny chat-template/decode surface with Qwen's off-prompt marker behavior."""

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["tokenize"] is True
        if kwargs["enable_thinking"]:
            return [10, 11]
        # Qwen's thinking-disabled prompt contains a *prompt-side* closed pair.
        return [10, QWEN_THINK_START_TOKEN_ID, 99, QWEN_THINK_END_TOKEN_ID, 11]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        pieces = {20: "alpha", 21: " beta", 30: "Answer: A", 31: "done"}
        return "".join(pieces.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def encode(self, text, **kwargs):
        return list(range(max(1, len(text.split()))))


class FixedPromptTokenizer(ExactTokenizer):
    def __init__(self, prompt_ids):
        self.prompt_ids = list(prompt_ids)

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs["tokenize"] is True
        return list(self.prompt_ids)


def _profile(max_model_len: int = 16384) -> ServingProfile:
    return ServingProfile(
        name="test",
        model_size="0.6B",
        hf_id="unused",
        tp_size=1,
        max_model_len=max_model_len,
        served_model_name="0.6B",
    )


def _response(
    *,
    prompt_ids,
    completion_ids,
    reasoning: str | None,
    content: str | None,
    finish_reason: str = "stop",
    prompt_usage: int | None = None,
    completion_usage: int | None = None,
):
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning,
        model_dump=lambda: {
            "content": content,
            "reasoning": reasoning,
            "reasoning_content": reasoning,
        },
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
        token_ids=list(completion_ids),
        model_dump=lambda: {"token_ids": list(completion_ids)},
    )
    return SimpleNamespace(
        prompt_token_ids=list(prompt_ids),
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=len(prompt_ids) if prompt_usage is None else prompt_usage,
            completion_tokens=(
                len(completion_ids)
                if completion_usage is None
                else completion_usage
            ),
        ),
        model_dump=lambda: {"prompt_token_ids": list(prompt_ids)},
    )


def _client(
    response,
    *,
    max_model_len: int = 16384,
    tokenizer=None,
    endpoint_generation: str | None = None,
):
    client = LogprobClient(
        base_url="http://unused.invalid/v1",
        model="0.6B",
        serving_profile=_profile(max_model_len),
        context_tokenizer=tokenizer or ExactTokenizer(),
        endpoint_generation=endpoint_generation,
    )
    requests = []

    def submit(**kwargs):
        requests.append(kwargs)
        return response

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=submit))
    )
    return client, requests


def _question() -> Question:
    return Question(
        qid="q1",
        benchmark="gpqa",
        prompt_stem="Which option is correct?",
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=["first", "second"],
    )


def _numeric_question() -> Question:
    return Question(
        qid="math-1",
        benchmark="math",
        prompt_stem="What is six times seven?",
        answer_key="42",
        answer_type=AnswerType.NUMERIC,
    )


class StubAgentClient:
    def __init__(self, result: ChatResult | GenerationTruncationError):
        self.result = result
        self.score_called = False
        self.events = []

    def chat(self, **kwargs):
        self.events.append(("chat", kwargs.get("seed")))
        if isinstance(self.result, GenerationTruncationError):
            raise self.result
        return self.result

    def score_options(self, prompt, option_letters):
        self.score_called = True
        self.events.append(("score", None))
        return OptionScores(
            probs={letter: 1.0 / len(option_letters) for letter in option_letters},
            raw_logprobs={letter: -1.0 for letter in option_letters},
            endpoint_generation="calibration-endpoint-generation",
        )


class SequencedAgentClient(StubAgentClient):
    def __init__(self, results):
        self.results = iter(results)
        self.score_called = False
        self.events = []

    def chat(self, **kwargs):
        self.events.append(("chat", kwargs.get("seed")))
        result = next(self.results)
        if isinstance(result, GenerationTruncationError):
            raise result
        return result


class ProbeFailsOnceClient(StubAgentClient):
    def __init__(self, result):
        super().__init__(result)
        self.probe_attempts = 0

    def score_options(self, prompt, option_letters):
        self.probe_attempts += 1
        self.events.append(("score", self.probe_attempts))
        if self.probe_attempts == 1:
            raise ConnectionError("probe unavailable")
        return OptionScores(
            probs={"A": 0.75, "B": 0.25},
            raw_logprobs={"A": -0.1, "B": -1.2},
            endpoint_generation="recovered-calibration-endpoint-generation",
        )


def test_option_probe_precedes_length_censored_chat():
    truncation = GenerationTruncationError(
        finish_reason="length",
        requested_output_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
        completion_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
    )
    client = StubAgentClient(truncation)
    agent = Agent(
        "agent0", client, "system", reasoning_level=ReasoningLevel.OFF
    )

    with pytest.raises(GenerationTruncationError) as caught:
        agent.answer(_question(), max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE)

    assert client.score_called
    assert client.events == [("score", None), ("chat", None)]
    assert caught.value.qid == "q1"
    assert caught.value.agent_id == "agent0"
    assert caught.value.round == 0
    assert caught.value.generation_role == "topology"
    assert caught.value.sample_index is None
    assert caught.value is truncation
    assert infer_failure_class(truncation) is FailureClass.CONTEXT_CAPACITY


def test_failed_option_probe_never_issues_or_reissues_primary_chat():
    client = ProbeFailsOnceClient(
        ChatResult(text="Answer: A", finish_reason="stop")
    )
    agent = Agent("agent0", client, "system", reasoning_level=ReasoningLevel.OFF)

    with pytest.raises(ConnectionError, match="probe unavailable"):
        agent.answer(_question(), seed=7)

    assert client.events == [("score", 1)]
    output = agent.answer(_question(), seed=7)
    assert output.answer_choice == "A"
    assert (
        output.calibration_endpoint_generation
        == "recovered-calibration-endpoint-generation"
    )
    assert client.events == [("score", 1), ("score", 2), ("chat", 7)]

    # Successful question-only probes are cached across auxiliary generations.
    agent.answer(_question(), seed=1007)
    assert client.probe_attempts == 2
    assert client.events[-1] == ("chat", 1007)


def test_self_consistency_retains_censor_and_runs_every_scheduled_seed():
    truncation = GenerationTruncationError(
        finish_reason="length",
        requested_output_tokens=16000,
        output_capacity_floor_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
        completion_tokens=16000,
        seed=1002,
    )
    client = SequencedAgentClient(
        [
            ChatResult(text="42", finish_reason="stop"),
            truncation,
            ChatResult(text="40", finish_reason="stop"),
            ChatResult(text="41", finish_reason="stop"),
            ChatResult(text="43", finish_reason="stop"),
        ]
    )
    agent = Agent("agent0", client, "system", reasoning_level=ReasoningLevel.OFF)

    outcomes = agent.sample(_numeric_question(), n=5, base_seed=1001)

    assert len(outcomes) == 5
    assert [outcome.seed for outcome in outcomes] == [1001, 1002, 1003, 1004, 1005]
    assert [outcome.termination_status for outcome in outcomes] == [
        "completed",
        "length_censored",
        "completed",
        "completed",
        "completed",
    ]
    record = outcomes[1].censored_generation
    assert record is not None
    assert record["generation_role"] == "self_consistency"
    assert record["sample_index"] == 1
    assert record["seed"] == 1002
    assert record["sampling_attempt_count"] == 1
    assert not client.score_called
    assert [event for event in client.events] == [
        ("chat", 1001),
        ("chat", 1002),
        ("chat", 1003),
        ("chat", 1004),
        ("chat", 1005),
    ]


def test_agent_output_records_capacity_floor_and_exact_requested_max():
    client = StubAgentClient(
        ChatResult(
            text="42",
            finish_reason="stop",
            output_capacity_floor_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
            generation_phase_requested_max_tokens=[15872],
            endpoint_generation="server-gen-agent",
        )
    )
    agent = Agent("agent0", client, "system", reasoning_level=ReasoningLevel.OFF)

    output = agent.answer(_numeric_question(), seed=7)

    assert output.output_capacity_floor_tokens == ANSWER_GENERATION_TOKEN_ALLOWANCE
    assert output.generation_phase_requested_max_tokens == [15872]
    assert output.endpoint_generation == "server-gen-agent"
    assert output.to_dict()["output_capacity_floor_tokens"] == (
        ANSWER_GENERATION_TOKEN_ALLOWANCE
    )
    assert output.to_dict()["generation_phase_requested_max_tokens"] == [15872]
    assert output.to_dict()["endpoint_generation"] == "server-gen-agent"


def test_finite_budget_is_one_native_chat_with_exact_token_ids():
    prompt_ids = [10, 11]
    completion_ids = [
        QWEN_THINK_START_TOKEN_ID,
        20,
        21,
        QWEN_THINK_END_TOKEN_ID,
        30,
        QWEN_IM_END_TOKEN_ID,
    ]
    client, requests = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            reasoning="alpha beta",
            content="Answer: A",
        ),
        endpoint_generation="server-gen-stopped",
    )

    result = client.chat(
        system="system",
        user="question",
        max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 2,
        seed=41,
        enable_thinking=True,
        thinking_budget=2,
    )

    assert len(requests) == 1
    request = requests[0]
    assert request["extra_body"]["thinking_token_budget"] == 2
    assert request["extra_body"]["return_token_ids"] is True
    assert request["seed"] == 41
    expected_requested_max = 16384 - len(prompt_ids) - CONTEXT_RESERVE_TOKENS
    assert request["max_tokens"] == expected_requested_max
    assert result.reasoning_tokens == 2
    assert result.reasoning_token_source == "vllm_native_token_ids"
    assert result.thinking_budget_saturated
    assert result.thinking_budget_consumed_tokens == 2
    assert result.generation_phase_count == 1
    assert result.generation_phase_seeds == [41]
    assert result.output_capacity_floor_tokens == (
        ANSWER_GENERATION_TOKEN_ALLOWANCE + 2
    )
    assert result.generation_phase_requested_max_tokens == [expected_requested_max]
    assert result.endpoint_generation == "server-gen-stopped"
    assert result.reasoning_start_token_index == len(prompt_ids)
    assert result.reasoning_end_token_index == len(prompt_ids) + 3
    assert result.generation_phase_prompt_token_id_hashes == [
        token_ids_sha256(prompt_ids)
    ]
    assert result.generation_phase_completion_token_id_hashes == [
        token_ids_sha256(completion_ids)
    ]


def test_finite_budget_natural_close_is_not_saturated():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            21,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha beta",
        content="Answer: A",
    )
    client, _ = _client(response)
    result = client.chat(
        system="system",
        user="question",
        max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
        seed=2,
        enable_thinking=True,
        thinking_budget=5,
    )
    assert result.reasoning_tokens == 2
    assert not result.thinking_budget_saturated


def test_off_accepts_prompt_markers_but_rejects_generated_markers():
    prompt_ids = [10, QWEN_THINK_START_TOKEN_ID, 99, QWEN_THINK_END_TOKEN_ID, 11]
    client, requests = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=[30, QWEN_IM_END_TOKEN_ID],
            reasoning=None,
            content="Answer: A",
        )
    )
    result = client.chat(
        system="system", user="question", seed=4, enable_thinking=False
    )
    assert len(requests) == 1
    assert result.reasoning_tokens == 0
    assert result.reasoning_start_token_count == 0
    assert result.reasoning_end_token_count == 0
    assert result.text == "Answer: A"

    bad, _ = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=[QWEN_THINK_START_TOKEN_ID, 30, QWEN_IM_END_TOKEN_ID],
            reasoning=None,
            content="Answer: A",
        )
    )
    with pytest.raises(GenerationProtocolCensorError) as caught:
        bad.chat(system="system", user="question", seed=4)
    assert caught.value.protocol_violation_codes == (
        "unexpected_reasoning_delimiter",
    )
    assert caught.value.to_censored_generation()["server_reasoning"] is None


def test_later_generated_end_is_literal_parser_visible_answer_content():
    completion_ids = [
        QWEN_THINK_START_TOKEN_ID,
        20,
        QWEN_THINK_END_TOKEN_ID,
        30,
        QWEN_THINK_END_TOKEN_ID,
        31,
        QWEN_IM_END_TOKEN_ID,
    ]
    client, _ = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=completion_ids,
            reasoning="alpha",
            content=f"Answer: A<{QWEN_THINK_END_TOKEN_ID}>done",
        )
    )

    result = client.chat(
        system="system",
        user="question",
        max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
        seed=3,
        enable_thinking=True,
        thinking_budget=5,
    )

    assert result.reasoning_text == "alpha"
    assert result.text == f"Answer: A<{QWEN_THINK_END_TOKEN_ID}>done"
    assert result.reasoning_tokens == 1
    assert result.reasoning_start_token_index == 2
    assert result.reasoning_end_token_index == 4
    assert result.reasoning_start_token_count == 1
    assert result.reasoning_end_token_count == 2


def test_repeated_generated_start_fails_closed_with_diagnostics():
    completion_ids = [
        QWEN_THINK_START_TOKEN_ID,
        QWEN_THINK_START_TOKEN_ID,
        20,
        QWEN_THINK_END_TOKEN_ID,
        30,
        QWEN_IM_END_TOKEN_ID,
    ]
    client, _ = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=completion_ids,
            reasoning=f"<{QWEN_THINK_START_TOKEN_ID}>alpha",
            content="Answer: A",
        )
    )
    with pytest.raises(GenerationProtocolCensorError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=3,
            enable_thinking=True,
            thinking_budget=5,
        )
    assert caught.value.protocol_violation_codes == (
        "reasoning_delimiter_contract",
    )
    record = caught.value.to_censored_generation()
    assert record["completion_token_ids"] == completion_ids
    assert record["sampling_attempt_count"] == 1


def test_premature_reasoning_eos_retains_full_recomputable_protocol_censor():
    completion_ids = [QWEN_THINK_START_TOKEN_ID, 20, 21, QWEN_IM_END_TOKEN_ID]
    client, requests = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=completion_ids,
            reasoning="alpha beta",
            content=None,
        ),
        endpoint_generation="server-generation-premature",
    )

    with pytest.raises(GenerationProtocolCensorError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=17,
            enable_thinking=True,
            thinking_budget=5,
        )

    assert len(requests) == 1
    error = caught.value
    assert error.protocol_violation_codes == ("premature_reasoning_eos",)
    payload = error.to_censored_generation()
    assert payload["reason"] == "protocol"
    assert payload["completion_token_ids"] == completion_ids
    assert payload["completion_token_id_sha256"] == token_ids_sha256(completion_ids)
    assert payload["actual_terminal_token_id"] == QWEN_IM_END_TOKEN_ID
    assert payload["generation_censor_protocol_version"] == (
        GENERATION_CENSOR_PROTOCOL_VERSION
    )
    assert payload["generation_censor_protocol_hash"] == (
        GENERATION_CENSOR_PROTOCOL_HASH
    )
    assert payload["endpoint_generation"] == "server-generation-premature"
    assert payload["server_content"] is None
    assert payload["server_reasoning"] == "alpha beta"


def test_qwen_alternate_eos_is_explicit_protocol_v4_censor_not_silent_acceptance():
    completion_ids = [
        QWEN_THINK_START_TOKEN_ID,
        20,
        QWEN_THINK_END_TOKEN_ID,
        30,
        QWEN_END_OF_TEXT_TOKEN_ID,
    ]
    client, requests = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=completion_ids,
            reasoning="alpha",
            content="Answer: A",
        )
    )

    with pytest.raises(GenerationProtocolCensorError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=23,
            enable_thinking=True,
            thinking_budget=5,
        )

    assert len(requests) == 1
    assert caught.value.protocol_violation_codes == ("alternate_eos_under_v4",)
    assert caught.value.to_censored_generation()["actual_terminal_token_id"] == (
        QWEN_END_OF_TEXT_TOKEN_ID
    )


@pytest.mark.parametrize(
    "prompt_ids",
    [
        [10, QWEN_THINK_START_TOKEN_ID, 99, QWEN_THINK_END_TOKEN_ID, 11],
        [10, QWEN_THINK_END_TOKEN_ID, 11],
    ],
)
def test_thinking_prompt_accepts_closed_pairs_and_stray_end(prompt_ids):
    response = _response(
        prompt_ids=prompt_ids,
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    client, requests = _client(
        response,
        tokenizer=FixedPromptTokenizer(prompt_ids),
    )

    result = client.chat(
        system="system",
        user="question",
        max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
        seed=3,
        enable_thinking=True,
        thinking_budget=1,
    )

    assert len(requests) == 1
    assert result.reasoning_start_token_index == len(prompt_ids)
    assert result.reasoning_end_token_index == len(prompt_ids) + 2


def test_thinking_prompt_unmatched_start_fails_locally_before_http():
    prompt_ids = [10, QWEN_THINK_START_TOKEN_ID, 11]
    client, requests = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=[],
            reasoning="",
            content="",
        ),
        tokenizer=FixedPromptTokenizer(prompt_ids),
    )

    with pytest.raises(ThinkingBudgetProtocolError, match="unmatched") as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert not isinstance(caught.value, ServerResponseProtocolError)
    assert infer_failure_class(caught.value) is FailureClass.CONFIGURATION
    assert requests == []


def test_parser_text_disagreement_fails_closed():
    client, _ = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=[QWEN_THINK_START_TOKEN_ID, 20,
                            QWEN_THINK_END_TOKEN_ID, 30, QWEN_IM_END_TOKEN_ID],
            reasoning="different",
            content="Answer: A",
        )
    )
    with pytest.raises(ServerResponseProtocolError, match="parser"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=3,
            enable_thinking=True,
            thinking_budget=5,
        )


def test_length_finish_is_reported_before_terminal_token_validation():
    prompt_ids = [10, 11]
    output_floor = ANSWER_GENERATION_TOKEN_ALLOWANCE + 5
    max_model_len = len(prompt_ids) + CONTEXT_RESERVE_TOKENS + output_floor
    completion_ids = [QWEN_THINK_START_TOKEN_ID, 20] + [21] * (
        output_floor - 2
    )
    expected_server_reasoning = ExactTokenizer().decode(
        completion_ids[1:],
        skip_special_tokens=False,
    )
    client, requests = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            reasoning=expected_server_reasoning,
            content=None,
            finish_reason="length",
        ),
        max_model_len=max_model_len,
        endpoint_generation="server-gen-7",
    )
    with pytest.raises(GenerationTruncationError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=3,
            enable_thinking=True,
            thinking_budget=5,
        )

    error = caught.value.enrich(
        qid="q-audit", agent_id="agent3", round_idx=2
    )
    expected_requested_max = max_model_len - len(prompt_ids) - CONTEXT_RESERVE_TOKENS
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == expected_requested_max
    assert error.requested_output_tokens == expected_requested_max
    assert error.prompt_token_ids == tuple(prompt_ids)
    assert error.completion_token_ids == tuple(completion_ids)
    record = error.to_censored_generation()
    assert set(record) == {
        "reason",
        "finish_reason",
        "sampling_attempt_count",
        "thinking_budget_protocol_version",
        "thinking_budget_protocol_hash",
        "qid",
        "agent_id",
        "round",
        "generation_role",
        "sample_index",
        "seed",
        "serving_profile",
        "effective_context_limit",
        "context_reserve_tokens",
        "prompt_tokens",
        "requested_output_tokens",
        "output_capacity_floor_tokens",
        "completion_tokens",
        "prompt_token_id_sha256",
        "completion_token_id_sha256",
        "prompt_token_ids",
        "completion_token_ids",
        "decoded_completion",
        "server_content",
        "server_reasoning",
        "prompt_think_start_positions",
        "prompt_think_end_positions",
        "completion_think_start_positions",
        "completion_think_end_positions",
        "endpoint_generation",
        "created_at",
    }
    assert record == {
        **record,
        "reason": "length",
        "finish_reason": "length",
        "sampling_attempt_count": 1,
        "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
        "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
        "qid": "q-audit",
        "agent_id": "agent3",
        "round": 2,
        "generation_role": "topology",
        "sample_index": None,
        "seed": 3,
        "serving_profile": "test",
        "effective_context_limit": max_model_len,
        "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
        "prompt_tokens": len(prompt_ids),
        "requested_output_tokens": expected_requested_max,
        "output_capacity_floor_tokens": output_floor,
        "completion_tokens": len(completion_ids),
        "prompt_token_id_sha256": token_ids_sha256(prompt_ids),
        "completion_token_id_sha256": token_ids_sha256(completion_ids),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "server_content": None,
        "server_reasoning": expected_server_reasoning,
        "prompt_think_start_positions": [],
        "prompt_think_end_positions": [],
        "completion_think_start_positions": [0],
        "completion_think_end_positions": [],
        "endpoint_generation": "server-gen-7",
    }
    assert record["decoded_completion"].startswith(
        f"<{QWEN_THINK_START_TOKEN_ID}>alpha beta"
    )
    assert isinstance(record["created_at"], float) and record["created_at"] > 0


def test_non_envelope_length_response_is_retained_as_protocol_censor():
    client, _ = _client(
        _response(
            prompt_ids=[10, 11],
            completion_ids=[QWEN_THINK_START_TOKEN_ID, 20],
            reasoning="alpha",
            content=None,
            finish_reason="length",
        )
    )

    with pytest.raises(GenerationProtocolCensorError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=3,
            enable_thinking=True,
            thinking_budget=5,
        )
    assert caught.value.protocol_violation_codes == (
        "premature_reasoning_eos",
        "unexpected_finish_reason",
    )


@pytest.mark.parametrize("mismatch", ["prompt", "usage"])
def test_server_token_accounting_mismatch_fails(mismatch):
    prompt_ids = [10, 11]
    response = _response(
        prompt_ids=([10, 12] if mismatch == "prompt" else prompt_ids),
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20,
                        QWEN_THINK_END_TOKEN_ID, 30, QWEN_IM_END_TOKEN_ID],
        reasoning="alpha",
        content="Answer: A",
        completion_usage=(999 if mismatch == "usage" else None),
    )
    client, _ = _client(response)
    with pytest.raises(ServerResponseProtocolError):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 5,
            seed=3,
            enable_thinking=True,
            thinking_budget=5,
        )


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [None],
        "not-a-list",
        [SimpleNamespace(), SimpleNamespace()],
    ],
)
def test_malformed_chat_choices_are_configuration_blocking(choices):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    response.choices = choices
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="choice"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


@pytest.mark.parametrize("missing", ["choices", "usage", "message"])
def test_missing_chat_response_structure_is_normalized(missing):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    target = response.choices[0] if missing == "message" else response
    delattr(target, missing)
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match=missing):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


@pytest.mark.parametrize("finish_reason", [None, 1, False, ""])
def test_non_string_finish_reason_is_not_admitted_as_a_censor(finish_reason):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20, QWEN_IM_END_TOKEN_ID],
        reasoning="alpha",
        content=None,
    )
    response.choices[0].finish_reason = finish_reason
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="finish_reason"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", 7),
        ("reasoning", ["alpha"]),
        ("reasoning_content", {"text": "alpha"}),
    ],
)
def test_non_text_parser_fields_block_protocol_censor(field, value):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20, QWEN_IM_END_TOKEN_ID],
        reasoning="alpha",
        content=None,
    )
    setattr(response.choices[0].message, field, value)
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="parser"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


def test_reasoning_parser_alias_disagreement_blocks_protocol_censor():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20, QWEN_IM_END_TOKEN_ID],
        reasoning="alpha",
        content=None,
    )
    response.choices[0].message.reasoning_content = "different"
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="aliases disagree"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


@pytest.mark.parametrize(
    "message_dump",
    [
        {"reasoning": "alpha"},
        {"content": None},
    ],
)
def test_absent_parser_fields_block_protocol_censor(message_dump):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20, QWEN_IM_END_TOKEN_ID],
        reasoning="alpha",
        content=None,
    )
    response.choices[0].message = SimpleNamespace(
        model_dump=lambda: dict(message_dump)
    )
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="field is absent"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


@pytest.mark.parametrize(
    ("reasoning", "content"),
    [("different", None), ("alpha", "")],
)
def test_parser_disagreement_blocks_instead_of_relabeling_censor(
    reasoning,
    content,
):
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[QWEN_THINK_START_TOKEN_ID, 20, QWEN_IM_END_TOKEN_ID],
        reasoning=reasoning,
        content=content,
    )
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="parser fields"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


def test_trustworthy_unexpected_finish_reason_remains_a_protocol_censor():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
        ],
        reasoning="alpha",
        content="Answer: A",
        finish_reason="content_filter",
    )
    client, requests = _client(response)

    with pytest.raises(GenerationProtocolCensorError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert caught.value.protocol_violation_codes == ("unexpected_finish_reason",)
    assert caught.value.finish_reason == "content_filter"
    assert len(requests) == 1


def test_extension_model_dump_failure_is_normalized():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    del response.prompt_token_ids

    def broken_model_dump():
        raise ValueError("broken extension envelope")

    response.model_dump = broken_model_dump
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="model_dump") as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert len(requests) == 1


@pytest.mark.parametrize("decode_result", [ValueError("decode failed"), ["not text"]])
def test_post_response_tokenizer_decode_failure_is_normalized(decode_result):
    class BrokenDecodeTokenizer(ExactTokenizer):
        def decode(self, token_ids, **kwargs):
            if isinstance(decode_result, Exception):
                raise decode_result
            return decode_result

    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    client, requests = _client(response, tokenizer=BrokenDecodeTokenizer())

    with pytest.raises(ServerResponseProtocolError, match="decode"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


def test_malformed_logprob_shape_is_configuration_blocking():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content=None,
    )
    response.choices[0].logprobs = SimpleNamespace(content={"bad": "shape"})
    client, requests = _client(response)

    with pytest.raises(ServerResponseProtocolError, match="logprobs.content"):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert len(requests) == 1


def test_chat_connection_failure_has_exactly_one_transport_attempt():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    client, _ = _client(response)
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        raise APIConnectionError(
            request=httpx.Request("POST", "http://unused.invalid/v1/chat/completions")
        )

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail_once))
    )

    with pytest.raises(APIConnectionError):
        client.chat(
            system="system",
            user="question",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE + 1,
            seed=3,
            enable_thinking=True,
            thinking_budget=1,
        )

    assert attempts == 1


def test_completion_probe_response_structure_is_configuration_blocking():
    response = _response(
        prompt_ids=[10, 11],
        completion_ids=[
            QWEN_THINK_START_TOKEN_ID,
            20,
            QWEN_THINK_END_TOKEN_ID,
            30,
            QWEN_IM_END_TOKEN_ID,
        ],
        reasoning="alpha",
        content="Answer: A",
    )
    client, _ = _client(response)
    attempts = 0

    def malformed_probe(**kwargs):
        nonlocal attempts
        attempts += 1
        return SimpleNamespace(choices=[])

    client._client = SimpleNamespace(
        completions=SimpleNamespace(create=malformed_probe)
    )

    with pytest.raises(ServerResponseProtocolError, match="completion choice"):
        client.score_options("Question\nAnswer: ", ["A", "B"])

    assert attempts == 1


@pytest.mark.parametrize(
    ("enable_thinking", "thinking_budget", "floor_tokens", "prompt_tokens"),
    [
        (False, None, ANSWER_GENERATION_TOKEN_ALLOWANCE, 5),
        (True, 512, ANSWER_GENERATION_TOKEN_ALLOWANCE + 512, 2),
        (True, None, ANSWER_GENERATION_TOKEN_ALLOWANCE + 8192, 2),
    ],
)
def test_capacity_floor_preflight_fails_before_http(
    enable_thinking, thinking_budget, floor_tokens, prompt_tokens
):
    response = _response(
        prompt_ids=[10, 11], completion_ids=[], reasoning="", content=""
    )
    max_model_len = prompt_tokens + CONTEXT_RESERVE_TOKENS + floor_tokens - 1
    client, requests = _client(response, max_model_len=max_model_len)
    with pytest.raises(ContextCapacityError) as caught:
        client.chat(
            system="system",
            user="question",
            max_tokens=floor_tokens,
            seed=4,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
    assert requests == []
    assert caught.value.preflight.prompt_tokens == prompt_tokens
    assert caught.value.preflight.requested_output_tokens == floor_tokens
    assert caught.value.preflight.required_tokens == max_model_len + 1


def test_capacity_floor_exact_boundary_is_admitted_once():
    prompt_ids = [10, QWEN_THINK_START_TOKEN_ID, 99, QWEN_THINK_END_TOKEN_ID, 11]
    floor = ANSWER_GENERATION_TOKEN_ALLOWANCE
    max_model_len = len(prompt_ids) + CONTEXT_RESERVE_TOKENS + floor
    client, requests = _client(
        _response(
            prompt_ids=prompt_ids,
            completion_ids=[30, QWEN_IM_END_TOKEN_ID],
            reasoning=None,
            content="Answer: A",
        ),
        max_model_len=max_model_len,
    )

    result = client.chat(system="system", user="question", seed=4)

    assert len(requests) == 1
    assert requests[0]["max_tokens"] == floor
    assert result.output_capacity_floor_tokens == floor
    assert result.generation_phase_requested_max_tokens == [floor]
    assert client.last_context_preflight is not None
    assert client.last_context_preflight.required_tokens == max_model_len


def test_policy_requires_seed_and_frozen_output_allowance():
    response = _response(
        prompt_ids=[10, 11], completion_ids=[], reasoning="", content=""
    )
    client, requests = _client(response)
    with pytest.raises(ThinkingBudgetProtocolError, match="seed"):
        client.chat(system="system", user="question")
    with pytest.raises(ThinkingBudgetProtocolError, match="capacity floor"):
        client.chat(system="system", user="question", max_tokens=1024, seed=1)
    assert requests == []


def test_thinking_budget_protocol_metadata_is_frozen_and_hashed():
    assert THINKING_BUDGET_PROTOCOL_VERSION == 4
    assert len(THINKING_BUDGET_PROTOCOL_HASH) == 64
    assert thinking_budget_protocol_metadata() == {
        "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
        "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
    }
