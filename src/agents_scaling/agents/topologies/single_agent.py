"""Single-Agent System (SAS): the baseline. One agent, one answer.

Defines P_SA (baseline accuracy), T_SAS (baseline turns), E_SAS (baseline error rate)
that all of Kim's relative coordination metrics are measured against. Uses only the first
agent in the pool so a sweep can reuse the same agent factory.
"""

from __future__ import annotations

from agents_scaling.agents.aggregate import system_confidences
from agents_scaling.agents.topologies.base import Topology, TopologyResult
from agents_scaling.benchmarks.schema import Question


class SingleAgent(Topology):
    def run(self, q: Question) -> TopologyResult:
        agent = self.agents[0]
        out = agent.answer(q, round_idx=0, peer_context="", max_tokens=self.max_tokens, seed=self.seed)
        res = TopologyResult(
            final_answer=out.answer_choice,
            per_agent=[out],
            n_turns=1,          # one reasoning-response exchange
            n_messages=0,       # no inter-agent communication
            n_rounds=1,
            n_agents=1,
        )
        res.system_conf = system_confidences([out], out.answer_choice)
        return res
