"""Axis 2 linchpin: build the text one agent sees of its peers' work.

``ContextShareLevel`` is the ONLY thing controlling what fields of a peer's ``AgentOutput``
are exposed. Every topology that passes inter-agent messages routes through
``build_peer_context`` so the axis is applied uniformly (and is unit-testable in
isolation). The three levels are strictly nested:

* ARTIFACT_ONLY     -> peer's final answer only
* PLUS_INTERMEDIATE -> + peer's intermediate results (scratchpad / sub-conclusions)
* PLUS_COT          -> + peer's raw chain-of-thought tokens

A monotonicity property holds by construction: the rendered text at a higher level is a
superset (in shared fields) of a lower level. ``shared_token_estimate`` exposes this so
the runner can assert PLUS_COT >= PLUS_INTERMEDIATE >= ARTIFACT_ONLY token counts.
"""

from __future__ import annotations

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.config import ContextShareLevel

_COT_TRUNCATE_CHARS = 4000  # guard against runaway context at PLUS_COT (plan Risk 4)


def _render_peer(out: AgentOutput, level: ContextShareLevel) -> str:
    parts = [f"Agent {out.agent_id} answered: {out.answer_choice}"]
    if out.verbalized_conf is not None:
        parts.append(f"(stated confidence {int(out.verbalized_conf * 100)}%)")
    block = " ".join(parts)

    if level.rank >= ContextShareLevel.PLUS_INTERMEDIATE.rank and out.intermediate_results:
        block += f"\n  Intermediate result: {out.intermediate_results}"
    if level.rank >= ContextShareLevel.PLUS_COT.rank and out.cot_text:
        cot = out.cot_text[:_COT_TRUNCATE_CHARS]
        block += f"\n  Reasoning: {cot}"
    return block


def build_peer_context(peers: list[AgentOutput], level: ContextShareLevel) -> str:
    """Render the peer-context block shown to an agent. Empty string if no peers."""
    if not peers:
        return ""
    rendered = "\n\n".join(_render_peer(p, level) for p in peers)
    return (
        "Here is what other agents concluded. Consider their views, then give your own "
        "answer.\n\n" + rendered
    )


def shared_token_estimate(peers: list[AgentOutput], level: ContextShareLevel) -> int:
    """Cheap whitespace-token estimate of the shared context (for monotonicity checks)."""
    return len(build_peer_context(peers, level).split())
