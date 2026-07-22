"""Axis-2 message builder: nested eligible fields plus exact bounded rendering."""

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.message_builder import (
    PEER_CONTEXT_PROTOCOL_HASH,
    PEER_CONTEXT_PROTOCOL_VERSION,
    PEER_COT_CHAR_LIMIT,
    PEER_RENDERED_BLOCK_TOKEN_LIMIT,
    build_peer_context,
    peer_context_protocol_metadata,
    render_peer_context,
    shared_token_estimate,
)
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


class CharacterTokenizer:
    """Reversible deterministic tokenizer: one Unicode code point per token."""

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False}
        return "".join(chr(token_id) for token_id in token_ids)


TOKENIZER = CharacterTokenizer()


def test_artifact_only_excludes_cot_and_intermediate():
    ctx = build_peer_context(
        [_peer()], ContextShareLevel.ARTIFACT_ONLY, tokenizer=TOKENIZER
    )
    assert "B" in ctx
    assert "Reasoning:" not in ctx
    assert "Intermediate result:" not in ctx


def test_plus_intermediate_includes_intermediate_not_cot():
    ctx = build_peer_context(
        [_peer()], ContextShareLevel.PLUS_INTERMEDIATE, tokenizer=TOKENIZER
    )
    assert "Intermediate result:" in ctx
    assert "Reasoning:" not in ctx


def test_plus_cot_includes_everything():
    ctx = build_peer_context(
        [_peer()], ContextShareLevel.PLUS_COT, tokenizer=TOKENIZER
    )
    assert "Intermediate result:" in ctx
    assert "Reasoning:" in ctx


def test_token_counts_are_monotonic():
    peers = [_peer()]
    a = shared_token_estimate(
        peers, ContextShareLevel.ARTIFACT_ONLY, tokenizer=TOKENIZER
    )
    i = shared_token_estimate(
        peers, ContextShareLevel.PLUS_INTERMEDIATE, tokenizer=TOKENIZER
    )
    c = shared_token_estimate(peers, ContextShareLevel.PLUS_COT, tokenizer=TOKENIZER)
    assert a <= i <= c
    assert a < c  # the extremes must differ given non-empty cot/intermediate


def test_no_peers_is_empty():
    assert build_peer_context(
        [], ContextShareLevel.PLUS_COT, tokenizer=TOKENIZER
    ) == ""


def test_plus_cot_uses_registered_fixed_reasoning_cap():
    peer = _peer()
    peer.cot_text = "x" * 5000 + "TAIL_MARKER"
    rendered = render_peer_context(
        [peer], ContextShareLevel.PLUS_COT, tokenizer=TOKENIZER
    )
    ctx = rendered.text
    assert "TAIL_MARKER" not in ctx
    assert "PEER_CONTEXT_TRUNCATED" in ctx
    assert rendered.block_token_counts[0] <= PEER_RENDERED_BLOCK_TOKEN_LIMIT
    assert rendered.truncation_marker_count == 1


def test_full_rendered_block_cap_covers_large_intermediate_and_is_hash_marked():
    peer = _peer()
    peer.intermediate_results = "0123456789" * 600
    peer.cot_text = "reasoning remains lower priority at the bounded tail"

    rendered = render_peer_context(
        [peer], ContextShareLevel.PLUS_COT, tokenizer=TOKENIZER
    )

    assert rendered.block_token_counts == (PEER_RENDERED_BLOCK_TOKEN_LIMIT,)
    assert rendered.truncation_marker_count == 1
    assert "PEER_CONTEXT_TRUNCATED" in rendered.text
    assert "sha256=" in rendered.text
    assert len(TOKENIZER.encode(rendered.text.split("\n\n", 1)[1], add_special_tokens=False)) <= 4000


def test_peer_cot_protocol_cap_and_digest_are_public_and_auditable():
    assert PEER_COT_CHAR_LIMIT == 4000
    assert PEER_RENDERED_BLOCK_TOKEN_LIMIT == 4000
    assert len(PEER_CONTEXT_PROTOCOL_HASH) == 64
    assert peer_context_protocol_metadata() == {
        "peer_context_protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
        "peer_context_protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
        "peer_cot_char_limit": 4000,
        "peer_rendered_block_token_limit": 4000,
    }
