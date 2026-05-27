"""Axis-2 message builder: the three context levels are strictly nested + monotonic."""

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.message_builder import build_peer_context, shared_token_estimate
from agents_scaling.config import ContextShareLevel


def _peer():
    return AgentOutput(
        agent_id="agent1",
        round=0,
        answer_choice="B",
        raw_text="full text",
        cot_text="First I considered X, then Y, leading to the conclusion.",
        intermediate_results="Therefore the answer is B because of Y.",
        option_logprobs={"A": 0.1, "B": 0.7, "C": 0.2},
        verbalized_conf=0.8,
    )


def test_artifact_only_excludes_cot_and_intermediate():
    ctx = build_peer_context([_peer()], ContextShareLevel.ARTIFACT_ONLY)
    assert "B" in ctx
    assert "Reasoning:" not in ctx
    assert "Intermediate result:" not in ctx


def test_plus_intermediate_includes_intermediate_not_cot():
    ctx = build_peer_context([_peer()], ContextShareLevel.PLUS_INTERMEDIATE)
    assert "Intermediate result:" in ctx
    assert "Reasoning:" not in ctx


def test_plus_cot_includes_everything():
    ctx = build_peer_context([_peer()], ContextShareLevel.PLUS_COT)
    assert "Intermediate result:" in ctx
    assert "Reasoning:" in ctx


def test_token_counts_are_monotonic():
    peers = [_peer()]
    a = shared_token_estimate(peers, ContextShareLevel.ARTIFACT_ONLY)
    i = shared_token_estimate(peers, ContextShareLevel.PLUS_INTERMEDIATE)
    c = shared_token_estimate(peers, ContextShareLevel.PLUS_COT)
    assert a <= i <= c
    assert a < c  # the extremes must differ given non-empty cot/intermediate


def test_no_peers_is_empty():
    assert build_peer_context([], ContextShareLevel.PLUS_COT) == ""
