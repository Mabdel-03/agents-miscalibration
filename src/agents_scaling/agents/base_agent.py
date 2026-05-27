"""A single agent: a system prompt + a LogprobClient, producing an ``AgentOutput``.

``AgentOutput`` deliberately separates the answer, the chain-of-thought, and intermediate
results so the message builder (Axis 2) can selectively expose each field to peers. This
separation is what makes the context-sharing axis a clean knob rather than ad-hoc string
slicing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents_scaling.benchmarks.formatting import render_question, score_prompt
from agents_scaling.benchmarks.grading import extract_answer
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.calibration.signals import verbalized_conf
from agents_scaling.config import ReasoningLevel
from agents_scaling.serving.client import LogprobClient

# A reasoning preamble elicits explicit CoT we can capture and (optionally) share.
_COT_HINT = "Think step by step. Show your reasoning, then give your final answer."

# Split the reasoning (CoT) from the final answer for selective sharing.
_FINAL_MARKERS = re.compile(r"(final answer|answer\s*[:=]|therefore,?\s|so the answer)", re.IGNORECASE)


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

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "round": self.round,
            "answer": self.answer_choice,
            "option_logprobs": self.option_logprobs,
            "verbalized_conf": self.verbalized_conf,
            "cot_text": self.cot_text,
            "intermediate_results": self.intermediate_results,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_text": self.reasoning_text,
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

    def answer(
        self,
        q: Question,
        round_idx: int = 0,
        peer_context: str = "",
        max_tokens: int = 1024,
        seed: int | None = None,
        elicit_cot: bool = True,
    ) -> AgentOutput:
        """Produce one AgentOutput for question ``q``, optionally seeing ``peer_context``.

        Axis 4 (reasoning) uses a TWO-CALL protocol when thinking is on, so calibration
        stays on the validated MCQ-logprob path and is not contaminated by thinking-mode
        sampling (which is non-greedy):
          1. thinking call (Qwen3 enable_thinking + budget) -> reasoning trace + answer text.
          2. forced-answer probe (``score_options``, /v1/completions, no thinking) -> the
             clean option-letter distribution used for ECE.
        When reasoning is OFF this is exactly the original single-pass behavior.
        """
        thinking_on = self.reasoning_level.enable_thinking
        budget = self.reasoning_level.thinking_budget

        user = render_question(q, ask_verbalized_confidence=True)
        # When the model thinks natively, don't also nudge for CoT in the prompt.
        if elicit_cot and not thinking_on:
            user = f"{_COT_HINT}\n\n{user}"
        if peer_context:
            user = f"{user}\n\n{peer_context}"

        # Give thinking rungs more room so an answer still emits after the <think> block.
        gen_max_tokens = max_tokens
        if thinking_on:
            gen_max_tokens = max_tokens + (budget if budget is not None else 8192)

        res = self.client.chat(
            system=self.system_prompt,
            user=user,
            temperature=self.temperature if not thinking_on else None,
            max_tokens=gen_max_tokens,
            seed=seed,
            enable_thinking=thinking_on,
            thinking_budget=budget,
        )
        cot, final_seg = _split_cot(res.text)
        chosen = extract_answer(q, res.text)

        option_logprobs: dict[str, float] = {}
        if q.answer_type == AnswerType.MCQ:
            # Forced-answer probe (completions endpoint, thinking implicitly off) gives the
            # clean option distribution for ECE — call #2 of the two-call protocol.
            scores = self.client.score_options(score_prompt(q), q.option_letters)
            option_logprobs = scores.probs
            if chosen is None:
                chosen = scores.argmax

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
        )

    def sample(self, q: Question, n: int, base_seed: int = 0, **kw) -> list[AgentOutput]:
        """n stochastic samples for self-consistency / semantic-entropy signals."""
        return [self.answer(q, seed=base_seed + i, **kw) for i in range(n)]
