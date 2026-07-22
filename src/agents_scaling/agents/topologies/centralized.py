"""Centralized MAS: orchestrator + sub-agents in a star topology.

Round structure:
  * Sub-agents (agents[1:]) answer independently each round; from round 1 on, they see the
    orchestrator's running synthesis through ``build_peer_context`` (Axis 2).
  * The orchestrator (agents[0]) synthesizes the sub-agents' outputs into the system
    answer, also seeing them through ``build_peer_context`` at ``context_level``.

This maps the context-share axis directly onto 'what the orchestrator/sub-agents see of
each other': final answers only -> + intermediates -> + chain-of-thought.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents_scaling.agents.aggregate import system_confidences
from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.topologies.base import Topology, TopologyResult
from agents_scaling.benchmarks.schema import Question


class Centralized(Topology):
    def run(self, q: Question) -> TopologyResult:
        orchestrator = self.agents[0]
        sub_agents = self.agents[1:] or [self.agents[0]]  # degrade gracefully if n==1

        all_outputs: list[AgentOutput] = []
        n_messages = 0
        last_synthesis: AgentOutput | None = None
        last_sub_outputs: list[AgentOutput] = []

        for r in range(self.rounds):
            # Sub-agents answer; from round 1 they see the orchestrator's prior synthesis.
            def _one(i_agent, _r=r, _syn=last_synthesis):
                i, agent = i_agent
                peers = [_syn] if (_r > 0 and _syn is not None) else []
                rendered = self.render_peer_context(peers)
                return self.answer_with_peer_context(
                    agent,
                    q,
                    round_idx=_r,
                    rendered=rendered,
                    seed=self.seed + _r * 100 + i,
                )

            with ThreadPoolExecutor(max_workers=len(sub_agents)) as ex:
                sub_outputs = list(ex.map(_one, enumerate(sub_agents)))
            all_outputs.extend(sub_outputs)
            if r > 0:
                n_messages += len(sub_agents)  # orchestrator -> each sub-agent

            # Orchestrator synthesizes, seeing sub-agent outputs at the chosen context level.
            rendered = self.render_peer_context(sub_outputs)
            synthesis = self.answer_with_peer_context(
                orchestrator,
                q,
                round_idx=r,
                rendered=rendered,
                seed=self.seed + r * 100 + 999,
            )
            all_outputs.append(synthesis)
            n_messages += len(sub_outputs)  # each sub-agent -> orchestrator
            last_synthesis = synthesis
            last_sub_outputs = sub_outputs

        final = last_synthesis.answer_choice if last_synthesis else None
        res = TopologyResult(
            final_answer=final,
            per_agent=all_outputs,
            n_turns=self.rounds * (len(sub_agents) + 1),  # sub-agents + orchestrator per round
            n_messages=n_messages,
            n_rounds=self.rounds,
            n_agents=len(self.agents),
        )
        # System confidence: vote view from the sub-agents; the FINAL PRODUCER is the
        # orchestrator (the model that actually emits the system answer).
        conf = system_confidences(
            last_sub_outputs, final,
            producers=[last_synthesis] if last_synthesis else None,
        )
        if last_synthesis and final is not None:
            # Keep explicit orchestrator_* keys for continuity; they coincide with
            # final_producer_* here (the orchestrator is the sole producer).
            conf["orchestrator_logprob"] = last_synthesis.option_logprobs.get(final, 0.0)
            if last_synthesis.verbalized_conf is not None:
                conf["orchestrator_verbal"] = last_synthesis.verbalized_conf
        res.system_conf = conf
        return res
