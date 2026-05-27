"""Decentralized MAS: all-to-all peer debate over ``rounds`` rounds.

Round 0: every agent answers independently. Round r>0: every agent re-answers, seeing the
*other* agents' round-(r-1) outputs through ``build_peer_context(..., context_level)`` —
this is where Axis 2 bites. Final answer = majority vote over the last round.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents_scaling.agents.aggregate import majority_vote, system_confidences
from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.message_builder import build_peer_context
from agents_scaling.agents.topologies.base import Topology, TopologyResult
from agents_scaling.benchmarks.schema import Question


class Decentralized(Topology):
    def run(self, q: Question) -> TopologyResult:
        all_outputs: list[AgentOutput] = []
        n_messages = 0
        prev_round: list[AgentOutput] = []

        for r in range(self.rounds):
            def _one(i_agent, _r=r, _prev=prev_round):
                i, agent = i_agent
                # All-to-all: agent i sees every OTHER agent's previous-round output.
                peers = [o for o in _prev if o.agent_id != agent.agent_id]
                ctx = build_peer_context(peers, self.context_level) if _r > 0 else ""
                return agent.answer(
                    q,
                    round_idx=_r,
                    peer_context=ctx,
                    max_tokens=self.max_tokens,
                    seed=self.seed + _r * 100 + i,
                )

            with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
                round_outputs = list(ex.map(_one, enumerate(self.agents)))

            if r > 0:
                # Each agent received a message from every other agent this round.
                n_messages += len(self.agents) * (len(self.agents) - 1)
            all_outputs.extend(round_outputs)
            prev_round = round_outputs

        final = majority_vote(prev_round)  # last round decides
        res = TopologyResult(
            final_answer=final,
            per_agent=all_outputs,
            n_turns=self.rounds * len(self.agents),
            n_messages=n_messages,
            n_rounds=self.rounds,
            n_agents=len(self.agents),
        )
        res.system_conf = system_confidences(prev_round, final)
        return res
