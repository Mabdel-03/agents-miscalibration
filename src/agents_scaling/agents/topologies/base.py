"""Topology base class + the shared result container.

A topology takes a question and a pool of agents and returns a ``TopologyResult``: the
final system answer, every per-agent output across rounds (for per-agent calibration),
and raw efficiency counters (turns, messages, tokens) that ``efficiency/`` turns into
Kim's coordination metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agents_scaling.agents.base_agent import Agent, AgentOutput
from agents_scaling.benchmarks.schema import Question
from agents_scaling.config import ContextShareLevel


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
        max_tokens: int = 1024,
        seed: int = 0,
    ):
        self.agents = agents
        self.context_level = context_level
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.seed = seed

    @abstractmethod
    def run(self, q: Question) -> TopologyResult:
        ...
