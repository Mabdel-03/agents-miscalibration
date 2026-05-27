"""Independent MAS: n agents answer in parallel with NO peer communication, then vote.

The context-share axis is a no-op here (agents never see each other), so the sweep
collapses this topology to a canonical context level. This is the minimum of the
coordination spectrum and the cleanest test of 'does pooling independent opinions change
system calibration vs a single agent'.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents_scaling.agents.aggregate import majority_vote, system_confidences
from agents_scaling.agents.topologies.base import Topology, TopologyResult
from agents_scaling.benchmarks.schema import Question


class Independent(Topology):
    def run(self, q: Question) -> TopologyResult:
        def _one(i_agent):
            i, agent = i_agent
            return agent.answer(
                q, round_idx=0, peer_context="", max_tokens=self.max_tokens, seed=self.seed + i
            )

        # Agents share one vLLM server; threads exploit its continuous batching.
        with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
            outputs = list(ex.map(_one, enumerate(self.agents)))

        final = majority_vote(outputs)
        res = TopologyResult(
            final_answer=final,
            per_agent=outputs,
            n_turns=len(outputs),   # each agent does one exchange, run in parallel
            n_messages=0,           # no inter-agent messages
            n_rounds=1,
            n_agents=len(self.agents),
        )
        res.system_conf = system_confidences(outputs, final)
        return res
