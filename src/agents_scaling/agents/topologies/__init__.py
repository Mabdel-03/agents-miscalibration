"""Topology factory: map a ``Topology`` enum value to a constructed topology instance."""

from __future__ import annotations

from typing import Any

from agents_scaling.agents.base_agent import Agent
from agents_scaling.agents.topologies.base import Topology as TopologyBase
from agents_scaling.agents.topologies.base import TopologyResult
from agents_scaling.agents.topologies.centralized import Centralized
from agents_scaling.agents.topologies.decentralized import Decentralized
from agents_scaling.agents.topologies.independent import Independent
from agents_scaling.agents.topologies.single_agent import SingleAgent
from agents_scaling.config import ContextShareLevel
from agents_scaling.config import Topology as TopologyEnum
from agents_scaling.serving.client import ANSWER_GENERATION_TOKEN_ALLOWANCE

_REGISTRY = {
    TopologyEnum.SINGLE_AGENT: SingleAgent,
    TopologyEnum.INDEPENDENT: Independent,
    TopologyEnum.DECENTRALIZED: Decentralized,
    TopologyEnum.CENTRALIZED: Centralized,
}


def build_topology(
    topology: TopologyEnum,
    agents: list[Agent],
    context_level: ContextShareLevel,
    rounds: int,
    max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
    seed: int = 0,
    context_tokenizer: Any | None = None,
) -> TopologyBase:
    cls = _REGISTRY[topology]
    return cls(
        agents=agents,
        context_level=context_level,
        rounds=rounds,
        max_tokens=max_tokens,
        seed=seed,
        context_tokenizer=context_tokenizer,
    )


__all__ = [
    "build_topology",
    "TopologyResult",
    "SingleAgent",
    "Independent",
    "Decentralized",
    "Centralized",
]
