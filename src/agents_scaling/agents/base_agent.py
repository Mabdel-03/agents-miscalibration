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
    def __init__(self, agent_id: str, client: LogprobClient, system_prompt: str, temperature: float = 0.7):
        self.agent_id = agent_id
        self.client = client
        self.system_prompt = system_prompt
        self.temperature = temperature

    def answer(
        self,
        q: Question,
        round_idx: int = 0,
        peer_context: str = "",
        max_tokens: int = 1024,
        seed: int | None = None,
        elicit_cot: bool = True,
    ) -> AgentOutput:
        """Produce one AgentOutput for question ``q``, optionally seeing ``peer_context``."""
        user = render_question(q, ask_verbalized_confidence=True)
        if elicit_cot:
            user = f"{_COT_HINT}\n\n{user}"
        if peer_context:
            user = f"{user}\n\n{peer_context}"

        res = self.client.chat(
            system=self.system_prompt,
            user=user,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        cot, final_seg = _split_cot(res.text)
        chosen = extract_answer(q, res.text)

        option_logprobs: dict[str, float] = {}
        if q.answer_type == AnswerType.MCQ:
            # Read the clean option distribution from the scoring endpoint.
            scores = self.client.score_options(score_prompt(q), q.option_letters)
            option_logprobs = scores.probs
            # If the chat answer was unparseable, fall back to the logprob argmax.
            if chosen is None:
                chosen = scores.argmax

        return AgentOutput(
            agent_id=self.agent_id,
            round=round_idx,
            answer_choice=chosen,
            raw_text=res.text,
            cot_text=cot,
            intermediate_results=final_seg,
            option_logprobs=option_logprobs,
            verbalized_conf=verbalized_conf(res.text),
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
        )

    def sample(self, q: Question, n: int, base_seed: int = 0, **kw) -> list[AgentOutput]:
        """n stochastic samples for self-consistency / semantic-entropy signals."""
        return [self.answer(q, seed=base_seed + i, **kw) for i in range(n)]
