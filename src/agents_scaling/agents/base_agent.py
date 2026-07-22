"""A single agent: a system prompt + a LogprobClient, producing an ``AgentOutput``.

``AgentOutput`` deliberately separates the answer, the chain-of-thought, and intermediate
results so the message builder (Axis 2) can selectively expose each field to peers. This
separation is what makes the context-sharing axis a clean knob rather than ad-hoc string
slicing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from agents_scaling.benchmarks.formatting import render_question, score_prompt
from agents_scaling.benchmarks.grading import extract_answer
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.calibration.signals import verbalized_conf
from agents_scaling.config import ReasoningLevel
from agents_scaling.serving.client import (
    ANSWER_GENERATION_TOKEN_ALLOWANCE,
    THINKING_BUDGET_PROTOCOL_HASH,
    THINKING_BUDGET_PROTOCOL_VERSION,
    UNLIMITED_THINKING_TOKEN_ALLOWANCE,
    GenerationProtocolCensorError,
    GenerationTruncationError,
    LogprobClient,
    OptionScores,
    ServerResponseProtocolError,
)

# A reasoning preamble elicits explicit CoT we can capture and (optionally) share.
_COT_HINT = "Think step by step. Show your reasoning, then give your final answer."

# Split the reasoning (CoT) from the final answer for selective sharing.
_FINAL_MARKERS = re.compile(r"(final answer|answer\s*[:=]|therefore,?\s|so the answer)", re.IGNORECASE)


def requested_generation_tokens(
    reasoning_level: ReasoningLevel,
    base_max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
) -> int:
    """Return the treatment-specific minimum safe output capacity.

    Protocol v4 uses the full remaining context envelope as the HTTP ``max_tokens``;
    this floor preserves the answer allowance and requested finite/unlimited reasoning
    capacity when deciding whether a profile can admit the request at all.
    """
    if base_max_tokens != ANSWER_GENERATION_TOKEN_ALLOWANCE:
        raise ValueError(
            "base_max_tokens must equal the frozen answer allowance "
            f"{ANSWER_GENERATION_TOKEN_ALLOWANCE}"
        )
    if not reasoning_level.enable_thinking:
        return base_max_tokens
    allowance = reasoning_level.thinking_budget
    if allowance is None:
        allowance = UNLIMITED_THINKING_TOKEN_ALLOWANCE
    return base_max_tokens + allowance


def render_agent_user_prompt(
    q: Question,
    reasoning_level: ReasoningLevel,
    *,
    peer_context: str = "",
    elicit_cot: bool = True,
) -> str:
    """Build exactly the user message consumed by :meth:`Agent.answer`."""
    user = render_question(q, ask_verbalized_confidence=True)
    if elicit_cot and not reasoning_level.enable_thinking:
        user = f"{_COT_HINT}\n\n{user}"
    if peer_context:
        user = f"{user}\n\n{peer_context}"
    return user


@dataclass
class AgentOutput:
    agent_id: str
    round: int
    answer_choice: str | None          # canonical answer (letter or normalized value)
    raw_text: str                      # full model output
    cot_text: str                      # extracted chain-of-thought (the reasoning portion)
    intermediate_results: str          # scratchpad / sub-conclusions (artifact-adjacent)
    option_logprobs: dict[str, float] = field(default_factory=dict)  # MCQ option distribution
    verbalized_conf: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_text: str = ""            # Axis 4: thinking trace (empty when reasoning off)
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    reasoning_token_source: str = "none"
    reasoning_word_count_legacy: int = 0
    thinking_budget_protocol_version: int = THINKING_BUDGET_PROTOCOL_VERSION
    thinking_budget_protocol_hash: str = THINKING_BUDGET_PROTOCOL_HASH
    thinking_budget_requested: int | None = None
    thinking_budget_saturated: bool = False
    thinking_budget_consumed_tokens: int = 0
    generation_phase_count: int = 1
    generation_phase_finish_reasons: list[str] = field(default_factory=list)
    generation_phase_seeds: list[int] = field(default_factory=list)
    generation_phase_prompt_tokens: list[int] = field(default_factory=list)
    generation_phase_completion_tokens: list[int] = field(default_factory=list)
    output_capacity_floor_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE
    generation_phase_requested_max_tokens: list[int] = field(default_factory=list)
    generation_phase_prompt_token_id_hashes: list[str] = field(default_factory=list)
    generation_phase_completion_token_id_hashes: list[str] = field(default_factory=list)
    endpoint_generation: str | None = None
    reasoning_start_token_index: int | None = None
    reasoning_end_token_index: int | None = None
    # Completion-only counts. The end count includes later literal think-end IDs in
    # parser-visible answer content; reasoning_end_token_index is always the first end.
    reasoning_start_token_count: int = 0
    reasoning_end_token_count: int = 0
    injected_transition_tokens: int = 0
    # Exact audit of the peer context consumed by this generation.  The topology fills
    # these after building the profile-tokenizer-bounded peer blocks.
    peer_context_tokens: int = 0
    peer_context_sha256: str = field(
        default_factory=lambda: hashlib.sha256(b"").hexdigest()
    )
    peer_context_block_token_counts: list[int] = field(default_factory=list)
    peer_context_truncation_marker_count: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "round": self.round,
            "answer": self.answer_choice,
            # Preserve the complete parser-visible answer text.  ``cot_text`` and
            # ``intermediate_results`` are derived views and cannot reconstruct it.
            "raw_text": self.raw_text,
            "option_logprobs": self.option_logprobs,
            "verbalized_conf": self.verbalized_conf,
            "cot_text": self.cot_text,
            "intermediate_results": self.intermediate_results,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_text": self.reasoning_text,
            "finish_reason": self.finish_reason,
            "reasoning_token_source": self.reasoning_token_source,
            "reasoning_word_count_legacy": self.reasoning_word_count_legacy,
            "thinking_budget_protocol_version": self.thinking_budget_protocol_version,
            "thinking_budget_protocol_hash": self.thinking_budget_protocol_hash,
            "thinking_budget_requested": self.thinking_budget_requested,
            "thinking_budget_saturated": self.thinking_budget_saturated,
            "thinking_budget_consumed_tokens": self.thinking_budget_consumed_tokens,
            "generation_phase_count": self.generation_phase_count,
            "generation_phase_finish_reasons": self.generation_phase_finish_reasons,
            "generation_phase_seeds": self.generation_phase_seeds,
            "generation_phase_prompt_tokens": self.generation_phase_prompt_tokens,
            "generation_phase_completion_tokens": self.generation_phase_completion_tokens,
            "output_capacity_floor_tokens": self.output_capacity_floor_tokens,
            "generation_phase_requested_max_tokens": self.generation_phase_requested_max_tokens,
            "generation_phase_prompt_token_id_hashes": self.generation_phase_prompt_token_id_hashes,
            "generation_phase_completion_token_id_hashes": self.generation_phase_completion_token_id_hashes,
            "endpoint_generation": self.endpoint_generation,
            "reasoning_start_token_index": self.reasoning_start_token_index,
            "reasoning_end_token_index": self.reasoning_end_token_index,
            "reasoning_start_token_count": self.reasoning_start_token_count,
            "reasoning_end_token_count": self.reasoning_end_token_count,
            "injected_transition_tokens": self.injected_transition_tokens,
            "peer_context_tokens": self.peer_context_tokens,
            "peer_context_sha256": self.peer_context_sha256,
            "peer_context_block_token_counts": self.peer_context_block_token_counts,
            "peer_context_truncation_marker_count": self.peer_context_truncation_marker_count,
        }


@dataclass(frozen=True)
class SelfConsistencySample:
    """One scheduled auxiliary draw, retained under every observed terminal status.

    Auxiliary draws are scientific observations rather than a bag of successful
    answers.  The coordinate and seed therefore live next to exactly one full stopped
    ``AgentOutput`` or one full length/protocol censorship record.
    """

    sample_index: int
    seed: int
    termination_status: str
    agent_output: AgentOutput | None = None
    censored_generation: dict | None = None

    def to_dict(self) -> dict:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "termination_status": self.termination_status,
            "agent_output": (
                self.agent_output.to_dict()
                if self.agent_output is not None
                else None
            ),
            "censored_generation": self.censored_generation,
        }


def _split_cot(text: str) -> tuple[str, str]:
    """Heuristically split (reasoning, final-segment). Best-effort; full text always kept."""
    m = _FINAL_MARKERS.search(text)
    if not m:
        return text.strip(), text.strip()
    cot = text[: m.start()].strip()
    final_seg = text[m.start() :].strip()
    return cot, final_seg


class Agent:
    def __init__(
        self,
        agent_id: str,
        client: LogprobClient,
        system_prompt: str,
        temperature: float = 0.7,
        reasoning_level: ReasoningLevel = ReasoningLevel.OFF,
    ):
        self.agent_id = agent_id
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_level = reasoning_level
        # The forced option probe depends only on the frozen question prompt/options,
        # not on a stochastic trajectory or peer context.  Cache a successful probe so
        # every generation for this agent/QID uses the same calibration signal and so a
        # later probe failure can never discard an already observed chat response.
        self._option_score_cache: dict[tuple[str, tuple[str, ...]], OptionScores] = {}

    def prepare_calibration(self, q: Question) -> OptionScores | None:
        """Run/cache the question-only calibration probe *before* stochastic chat.

        The runner calls this for every topology participant before ``Topology.run`` so
        a later agent's probe failure cannot cause an earlier agent's observed chat to
        be reissued.  ``answer`` also calls it defensively for non-runner consumers.
        """

        if q.answer_type != AnswerType.MCQ:
            return None
        prompt = score_prompt(q)
        letters = tuple(q.option_letters)
        key = (prompt, letters)
        scores = self._option_score_cache.get(key)
        if scores is None:
            # Deliberately precedes ``chat``.  A connection failure here is therefore
            # safe for the runner to retry without outcome-conditioned resampling.
            scores = self.client.score_options(prompt, list(letters))
            self._option_score_cache[key] = scores
        return scores

    def answer(
        self,
        q: Question,
        round_idx: int = 0,
        peer_context: str = "",
        max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
        seed: int | None = None,
        elicit_cot: bool = True,
    ) -> AgentOutput:
        """Produce one AgentOutput for question ``q``, optionally seeing ``peer_context``.

        Axis 4 finite reasoning uses vLLM 0.21's native token-level budget on one chat
        generation; unlimited reasoning uses the same exact token-ID path.  MCQ
        calibration then uses an independent forced-answer probe (``score_options``),
        keeping ECE on the validated no-thinking option-logprob path.
        When reasoning is OFF this is exactly the original single-pass behavior.
        """
        thinking_on = self.reasoning_level.enable_thinking
        budget = self.reasoning_level.thinking_budget

        user = render_agent_user_prompt(
            q,
            self.reasoning_level,
            peer_context=peer_context,
            elicit_cot=elicit_cot,
        )

        output_capacity_floor_tokens = requested_generation_tokens(
            self.reasoning_level, max_tokens
        )

        # Complete the independent, question-only forced option probe before drawing a
        # stochastic response.  This ordering is part of protocol v4: no score-probe
        # failure can occur after (and thereby cause reissue of) the primary chat.
        option_scores = self.prepare_calibration(q)

        try:
            res = self.client.chat(
                system=self.system_prompt,
                user=user,
                temperature=self.temperature if not thinking_on else None,
                # In protocol v4 this is the admission floor.  LogprobClient computes
                # the actual one-shot max_tokens from the exact rendered prompt.
                max_tokens=output_capacity_floor_tokens,
                seed=seed,
                enable_thinking=thinking_on,
                thinking_budget=budget,
            )
        except GenerationTruncationError as exc:
            exc.enrich(qid=q.qid, agent_id=self.agent_id, round_idx=round_idx)
            raise
        # LogprobClient raises a richly-provenanced censor before returning any non-stop
        # result.  Treat a custom/mocked client that violates that API invariant as a
        # protocol error rather than manufacturing an unauditable censor record.
        if res.finish_reason != "stop":
            raise ServerResponseProtocolError(
                "chat client returned a non-stop result instead of raising its exact "
                f"censored generation: finish_reason={res.finish_reason!r}"
            )
        cot, final_seg = _split_cot(res.text)
        chosen = extract_answer(q, res.text)

        option_logprobs: dict[str, float] = {}
        if option_scores is not None:
            # The forced-answer probe remains independent of generated reasoning and
            # therefore preserves the existing ECE/calibration interpretation.
            option_logprobs = option_scores.probs
            if chosen is None:
                chosen = option_scores.argmax

        # Prefer the model's native thinking trace as the CoT when present.
        cot_out = res.reasoning_text if res.reasoning_text else cot
        return AgentOutput(
            agent_id=self.agent_id,
            round=round_idx,
            answer_choice=chosen,
            raw_text=res.text,
            cot_text=cot_out,
            intermediate_results=final_seg,
            option_logprobs=option_logprobs,
            verbalized_conf=verbalized_conf(res.text),
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            reasoning_text=res.reasoning_text,
            reasoning_tokens=res.reasoning_tokens,
            finish_reason=res.finish_reason,
            reasoning_token_source=res.reasoning_token_source,
            reasoning_word_count_legacy=res.reasoning_word_count_legacy,
            thinking_budget_protocol_version=res.thinking_budget_protocol_version,
            thinking_budget_protocol_hash=res.thinking_budget_protocol_hash,
            thinking_budget_requested=res.thinking_budget_requested,
            thinking_budget_saturated=res.thinking_budget_saturated,
            thinking_budget_consumed_tokens=res.thinking_budget_consumed_tokens,
            generation_phase_count=res.generation_phase_count,
            generation_phase_finish_reasons=res.generation_phase_finish_reasons,
            generation_phase_seeds=res.generation_phase_seeds,
            generation_phase_prompt_tokens=res.generation_phase_prompt_tokens,
            generation_phase_completion_tokens=res.generation_phase_completion_tokens,
            output_capacity_floor_tokens=res.output_capacity_floor_tokens,
            generation_phase_requested_max_tokens=res.generation_phase_requested_max_tokens,
            generation_phase_prompt_token_id_hashes=res.generation_phase_prompt_token_id_hashes,
            generation_phase_completion_token_id_hashes=res.generation_phase_completion_token_id_hashes,
            endpoint_generation=res.endpoint_generation,
            reasoning_start_token_index=res.reasoning_start_token_index,
            reasoning_end_token_index=res.reasoning_end_token_index,
            reasoning_start_token_count=res.reasoning_start_token_count,
            reasoning_end_token_count=res.reasoning_end_token_count,
            injected_transition_tokens=res.injected_transition_tokens,
        )

    def sample_one(
        self,
        q: Question,
        *,
        sample_index: int,
        base_seed: int = 0,
        **kw,
    ) -> SelfConsistencySample:
        """Execute exactly one scheduled auxiliary draw without replacement sampling."""

        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        seed = base_seed + sample_index
        try:
            output = self.answer(q, seed=seed, **kw)
        except GenerationTruncationError as exc:
            exc.enrich(
                qid=exc.qid or q.qid,
                agent_id=exc.agent_id or self.agent_id,
                round_idx=(
                    exc.round
                    if exc.round is not None
                    else int(kw.get("round_idx", 0))
                ),
                generation_role="self_consistency",
                sample_index=sample_index,
            )
            return SelfConsistencySample(
                sample_index=sample_index,
                seed=seed,
                termination_status=(
                    "protocol_censored"
                    if isinstance(exc, GenerationProtocolCensorError)
                    else "length_censored"
                ),
                censored_generation=exc.to_censored_generation(),
            )
        return SelfConsistencySample(
            sample_index=sample_index,
            seed=seed,
            termination_status="completed",
            agent_output=output,
        )

    def sample(
        self, q: Question, n: int, base_seed: int = 0, **kw
    ) -> list[SelfConsistencySample]:
        """Retain all ``n`` scheduled auxiliary outcomes, including both censor types."""

        if n < 0:
            raise ValueError("n must be non-negative")
        return [
            self.sample_one(
                q,
                sample_index=sample_index,
                base_seed=base_seed,
                **kw,
            )
            for sample_index in range(n)
        ]
