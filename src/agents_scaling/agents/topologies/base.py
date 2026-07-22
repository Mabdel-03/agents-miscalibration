"""Topology base class + the shared result container.

A topology takes a question and a pool of agents and returns a ``TopologyResult``: the
final system answer, every per-agent output across rounds (for per-agent calibration),
and raw efficiency counters (turns, messages, tokens) that ``efficiency/`` turns into
Kim's coordination metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agents_scaling.agents.base_agent import Agent, AgentOutput
from agents_scaling.agents.message_builder import PeerContextRender, render_peer_context
from agents_scaling.benchmarks.schema import Question
from agents_scaling.config import ContextShareLevel
from agents_scaling.serving.client import ANSWER_GENERATION_TOKEN_ALLOWANCE


@dataclass
class TopologyResult:
    final_answer: str | None
    per_agent: list[AgentOutput] = field(default_factory=list)  # flat across rounds
    # raw efficiency counters (turn-based, per Kim; tokens logged in parallel)
    n_turns: int = 0          # reasoning-response exchanges
    n_messages: int = 0       # inter-agent messages
    n_rounds: int = 0
    n_agents: int = 0
    # system-level confidence under multiple definitions (filled by aggregate.py)
    system_conf: dict[str, float] = field(default_factory=dict)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(o.prompt_tokens for o in self.per_agent)

    @property
    def total_completion_tokens(self) -> int:
        return sum(o.completion_tokens for o in self.per_agent)


class Topology(ABC):
    """Base for all topologies. ``context_level`` is the Axis-2 knob."""

    def __init__(
        self,
        agents: list[Agent],
        context_level: ContextShareLevel = ContextShareLevel.ARTIFACT_ONLY,
        rounds: int = 2,
        max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
        seed: int = 0,
        context_tokenizer: Any | None = None,
    ):
        self.agents = agents
        self.context_level = context_level
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.seed = seed
        self.context_tokenizer = context_tokenizer

    def render_peer_context(self, peers: list[AgentOutput]) -> PeerContextRender:
        if peers and self.context_tokenizer is None:
            raise RuntimeError(
                "coordinated topology requires the exact serving-profile tokenizer"
            )
        return render_peer_context(
            peers,
            self.context_level,
            tokenizer=self.context_tokenizer,
        )

    @staticmethod
    def attach_peer_context_audit(
        output: AgentOutput, rendered: PeerContextRender
    ) -> AgentOutput:
        output.peer_context_tokens = rendered.token_count
        output.peer_context_sha256 = rendered.sha256
        output.peer_context_block_token_counts = list(rendered.block_token_counts)
        output.peer_context_truncation_marker_count = rendered.truncation_marker_count
        return output

    def answer_with_peer_context(
        self,
        agent: Agent,
        q: Question,
        *,
        round_idx: int,
        rendered: PeerContextRender,
        seed: int,
    ) -> AgentOutput:
        """Answer with render provenance attached before a durable proxy persists it."""

        checkpointed_answer = getattr(
            agent,
            "answer_with_peer_context_audit",
            None,
        )
        if callable(checkpointed_answer):
            return checkpointed_answer(
                q,
                round_idx=round_idx,
                peer_context_render=rendered,
                max_tokens=self.max_tokens,
                seed=seed,
            )
        output = agent.answer(
            q,
            round_idx=round_idx,
            peer_context=rendered.text,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        return self.attach_peer_context_audit(output, rendered)

    @abstractmethod
    def run(self, q: Question) -> TopologyResult:
        ...
