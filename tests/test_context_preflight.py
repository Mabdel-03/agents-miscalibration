"""Rendered-chat token capacity checks, including the seven-agent extreme."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.message_builder import build_peer_context
from agents_scaling.benchmarks.formatting import render_question
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import ContextShareLevel, ExperimentCell
from agents_scaling.prompts.system_prompts import get_prompt
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    UNLIMITED_THINKING_TOKEN_ALLOWANCE,
    LogprobClient,
)
from agents_scaling.serving.context import (
    ContextCapacityError,
    TokenizerInitializationError,
    build_chat_messages,
    preflight_chat_context,
    preflight_profile_chat,
    rendered_chat_token_count,
    tokenizer_for_profile,
)
from agents_scaling.serving.profiles import get_serving_profile, serving_profile_for_cell


class RecordingTokenizer:
    """Deterministic exact tokenizer stand-in: one id per whitespace token + controls."""

    def __init__(
        self,
        fixed_count: int | None = None,
        raw_fixed_count: int | None = None,
    ):
        self.fixed_count = fixed_count
        self.raw_fixed_count = raw_fixed_count
        self.calls = []
        self.encode_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        if self.fixed_count is not None:
            return list(range(self.fixed_count))
        # Count content plus two role/control ids per message and two generation-prompt ids.
        count = sum(len(m["content"].split()) + 2 for m in conversation) + 2
        if kwargs["enable_thinking"]:
            count += 1
        return list(range(count))

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        count = self.raw_fixed_count
        if count is None:
            count = len(text.split())
        return list(range(count))


def test_concurrent_first_tokenizer_use_initializes_exactly_once(monkeypatch):
    """Topology threads cannot race Transformers' lazy AutoTokenizer import/load."""
    sentinel = RecordingTokenizer()
    calls = 0

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            # Make duplicate first-miss initialization deterministic on the old code.
            time.sleep(0.02)
            return sentinel

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    tokenizer_for_profile.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(executor.map(tokenizer_for_profile, ["8B"] * 16))
        assert all(tokenizer is sentinel for tokenizer in loaded)
        assert calls == 1
    finally:
        tokenizer_for_profile.cache_clear()


def test_tokenizer_initialization_error_is_configuration_classified(monkeypatch):
    from agents_scaling.experiment.completion import FailureClass, infer_failure_class

    fake_transformers = ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    tokenizer_for_profile.cache_clear()
    try:
        with pytest.raises(TokenizerInitializationError) as exc_info:
            tokenizer_for_profile("8B")
        assert exc_info.value.profile_name == "8B"
        assert infer_failure_class(exc_info.value) is FailureClass.CONFIGURATION
    finally:
        tokenizer_for_profile.cache_clear()


def test_runner_builds_clients_with_one_eagerly_shared_tokenizer(monkeypatch):
    from agents_scaling.experiment import runner

    sentinel = RecordingTokenizer()
    tokenizer_calls = []
    client_kwargs = []

    def load_tokenizer(profile_name):
        tokenizer_calls.append(profile_name)
        return sentinel

    class FakeClient:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)

    monkeypatch.setattr(runner, "tokenizer_for_profile", load_tokenizer)
    monkeypatch.setattr(runner, "LogprobClient", FakeClient)
    monkeypatch.setattr(
        runner,
        "get_prompt",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit immutable prompt must bypass package-relative resources"
        ),
    )
    cell = ExperimentCell(
        model_size="8B",
        context_share_level="artifact_only",
        prompt_complexity_level=0,
        reasoning_level="off",
        topology="decentralized",
        benchmark="gpqa",
        n_agents=3,
    )
    agents = runner._build_agents(
        cell,
        "http://node:8000/v1",
        "8B",
        get_serving_profile("8B"),
        system_prompt="frozen release prompt",
    )

    assert len(agents) == 3
    assert tokenizer_calls == ["8B"]
    assert len(client_kwargs) == 3
    assert all(kwargs["context_tokenizer"] is sentinel for kwargs in client_kwargs)


def test_count_uses_tokenized_generation_chat_template_and_thinking_flag():
    tokenizer = RecordingTokenizer(fixed_count=37)
    messages = build_chat_messages("system", "user text")

    assert rendered_chat_token_count(tokenizer, messages, enable_thinking=True) == 37
    conversation, kwargs = tokenizer.calls[-1]
    assert conversation == messages
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "return_dict": False,
    }


def test_count_handles_transformers_batch_encoding_shape():
    """transformers 5.x may return a mapping; count input_ids, never mapping keys."""

    class MappingTokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            return {"input_ids": list(range(41)), "attention_mask": [1] * 41}

    count = rendered_chat_token_count(
        MappingTokenizer(), build_chat_messages("system", "user"), enable_thinking=False
    )
    assert count == 41


def test_preflight_allows_equality_and_rejects_one_token_over():
    messages = build_chat_messages("", "question")
    exact = preflight_chat_context(
        RecordingTokenizer(fixed_count=100),
        messages,
        requested_output_tokens=28,
        served_context=256,
        profile_name="test",
        enable_thinking=False,
    )
    assert exact.required_tokens == 256
    assert exact.fits

    with pytest.raises(ContextCapacityError) as exc_info:
        preflight_chat_context(
            RecordingTokenizer(fixed_count=100),
            messages,
            requested_output_tokens=29,
            served_context=256,
            profile_name="test",
            enable_thinking=False,
        )
    assert exc_info.value.preflight.required_tokens == 257
    assert "required=257" in str(exc_info.value)


def _peer(i: int, reasoning_tokens: int) -> AgentOutput:
    # Tail marker proves PLUS_COT does not silently truncate the scientific treatment.
    reasoning = " ".join([f"reason{i}"] * reasoning_tokens + [f"TAIL_{i}"])
    return AgentOutput(
        agent_id=f"agent{i}",
        round=0,
        answer_choice="A",
        raw_text="Answer: A",
        cot_text=reasoning,
        intermediate_results="intermediate conclusion",
    )


def _seven_agent_unlimited_cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="32B",
        context_share_level="plus_cot",
        prompt_complexity_level=3,
        reasoning_level="unlimited",
        topology="decentralized",
        benchmark="gpqa",
        n_agents=7,
    )


def _constructed_long_question() -> Question:
    # Exercise the production Question -> user-message renderer.  Repeating a realistic
    # prose fragment constructs a deterministic upper-bound fixture without loading a
    # benchmark or touching the network during unit tests.
    stem = " ".join(
        ["Given the evidence, determine which conclusion is best supported."] * 650
    )
    return Question(
        qid="constructed-worst",
        benchmark="gpqa",
        prompt_stem=stem,
        answer_key="A",
        answer_type=AnswerType.MCQ,
        options=[
            "The first conclusion follows from all stated premises.",
            "The second conclusion reverses the causal direction.",
            "The third conclusion requires an unstated assumption.",
            "The fourth conclusion contradicts the observations.",
        ],
    )


def test_seven_agent_plus_cot_registered_worst_case_fits_long_profile():
    """The shipped six-peer context + real prompt render fits the routed 40K profile."""
    cell = _seven_agent_unlimited_cell()
    profile = serving_profile_for_cell(cell)
    peer_context = build_peer_context(
        [_peer(i, reasoning_tokens=8192) for i in range(cell.n_agents - 1)],
        cell.context_share_level,
        tokenizer=RecordingTokenizer(),
    )
    # build_peer_context applies the registered, fixed 4,000-character PLUS_COT cap.
    assert "TAIL_0" not in peer_context
    tokenizer = RecordingTokenizer()
    output_budget = (
        ANSWER_GENERATION_TOKEN_ALLOWANCE + UNLIMITED_THINKING_TOKEN_ALLOWANCE
    )
    rendered_question = render_question(_constructed_long_question())

    report = preflight_profile_chat(
        profile,
        system=get_prompt(cell.prompt_complexity_level),
        user=f"{rendered_question}\n\n{peer_context}",
        requested_output_tokens=output_budget,
        enable_thinking=True,
        tokenizer=tokenizer,
    )
    assert report.fits
    assert 16384 < report.required_tokens <= 40960

    with pytest.raises(ContextCapacityError):
        preflight_profile_chat(
            get_serving_profile("32B"),
            system=get_prompt(cell.prompt_complexity_level),
            user=f"{rendered_question}\n\n{peer_context}",
            requested_output_tokens=output_budget,
            enable_thinking=True,
            tokenizer=tokenizer,
        )


def test_unbounded_synthetic_context_is_rejected_instead_of_adaptively_sliced():
    """Preflight never introduces a new runtime truncation policy on oversized input."""
    profile = serving_profile_for_cell(_seven_agent_unlimited_cell())
    unbounded_peer_context = " ".join(["peer_reasoning"] * (6 * 8192))
    with pytest.raises(ContextCapacityError) as exc_info:
        preflight_profile_chat(
            profile,
            system="system",
            user=f"worst benchmark question\n\n{unbounded_peer_context}",
            requested_output_tokens=(
                ANSWER_GENERATION_TOKEN_ALLOWANCE
                + UNLIMITED_THINKING_TOKEN_ALLOWANCE
            ),
            enable_thinking=True,
            tokenizer=RecordingTokenizer(),
        )
    assert exc_info.value.preflight.served_context == 40960
    assert exc_info.value.preflight.required_tokens > 40960


def test_client_raises_capacity_error_before_http_submission():
    tokenizer = RecordingTokenizer(fixed_count=40900)
    client = LogprobClient(
        base_url="http://unused.invalid/v1",
        model="32B",
        serving_profile="32B-long",
        context_tokenizer=tokenizer,
    )

    def should_not_submit(**kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("HTTP submission occurred before context preflight")

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=should_not_submit))
    )
    with pytest.raises(ContextCapacityError):
        client.chat(
            system="system",
            user="user",
            max_tokens=ANSWER_GENERATION_TOKEN_ALLOWANCE,
            seed=0,
        )


def test_option_scoring_raw_prompt_preflight_allows_exact_boundary():
    tokenizer = RecordingTokenizer(raw_fixed_count=16255)
    client = LogprobClient(
        base_url="http://unused.invalid/v1",
        model="32B",
        serving_profile="32B",
        context_tokenizer=tokenizer,
    )
    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(top_logprobs=[{" A": -0.1, " B": -2.0}])
                )
            ]
        )

    client._client = SimpleNamespace(
        completions=SimpleNamespace(create=submit)
    )
    scores = client.score_options("literal completion prompt", ["A", "B"])

    assert scores.argmax == "A"
    assert len(submitted) == 1
    assert tokenizer.encode_calls == [
        ("literal completion prompt", {"add_special_tokens": True})
    ]
    assert client.last_context_preflight is not None
    assert client.last_context_preflight.required_tokens == 16384


def test_option_scoring_rejects_oversize_raw_prompt_before_http():
    tokenizer = RecordingTokenizer(raw_fixed_count=16256)
    client = LogprobClient(
        base_url="http://unused.invalid/v1",
        model="32B",
        serving_profile="32B",
        context_tokenizer=tokenizer,
    )

    def should_not_submit(**kwargs):  # pragma: no cover - regression sentinel
        raise AssertionError("HTTP submission occurred before raw completion preflight")

    client._client = SimpleNamespace(
        completions=SimpleNamespace(create=should_not_submit)
    )
    with pytest.raises(ContextCapacityError) as exc_info:
        client.score_options("literal completion prompt", ["A", "B"])
    assert exc_info.value.preflight.prompt_tokens == 16256
    assert exc_info.value.preflight.requested_output_tokens == 1
