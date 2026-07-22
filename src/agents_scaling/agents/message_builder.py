"""Token-bounded, auditable rendering of the context shared between agents.

``ContextShareLevel`` is the only scientific knob controlling which peer-output fields
are eligible for sharing.  Every coordinated topology uses this module.  The CoT field
retains its registered 4,000-character cap, and protocol v2 additionally caps each fully
rendered peer block (answer, confidence, intermediate, CoT, and templates together) at
4,000 exact profile-tokenizer tokens.  Any token truncation is explicit and hash-covered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.config import ContextShareLevel


PEER_CONTEXT_PROTOCOL_VERSION = 2
PEER_COT_CHAR_LIMIT = 4000
PEER_RENDERED_BLOCK_TOKEN_LIMIT = 4000
_PEER_CONTEXT_INTRO = (
    "Here is what other agents concluded. Consider their views, then give your own "
    "answer.\n\n"
)
_PEER_ANSWER_TEMPLATE = "Agent {agent_id} answered: {answer_choice}"
_PEER_CONFIDENCE_TEMPLATE = "(stated confidence {confidence_percent}%)"
_PEER_INTERMEDIATE_TEMPLATE = "\n  Intermediate result: {intermediate_results}"
_PEER_REASONING_TEMPLATE = "\n  Reasoning: {cot_text}"
_PEER_TRUNCATION_TEMPLATE = (
    "\n  [PEER_CONTEXT_TRUNCATED original_tokens={original_tokens} "
    "sha256={original_sha256} limit={token_limit}]"
)
_PEER_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PeerContextRender:
    """Rendered context plus exact provenance attached to the consuming output."""

    text: str
    token_count: int
    sha256: str
    block_token_counts: tuple[int, ...] = ()
    truncation_marker_count: int = 0


def _protocol_payload() -> dict[str, object]:
    """Return the canonical, hash-covered peer-context rendering contract."""

    return {
        "version": PEER_CONTEXT_PROTOCOL_VERSION,
        "peer_cot_char_limit": PEER_COT_CHAR_LIMIT,
        "peer_rendered_block_token_limit": PEER_RENDERED_BLOCK_TOKEN_LIMIT,
        "tokenizer_contract": (
            "the exact serving-profile tokenizer; encode/decode with no special tokens"
        ),
        "truncation_strategy": (
            "decode a retained token-ID prefix that re-encodes with the "
            "explicit marker within the per-block limit"
        ),
        "intro": _PEER_CONTEXT_INTRO,
        "answer_template": _PEER_ANSWER_TEMPLATE,
        "confidence_template": _PEER_CONFIDENCE_TEMPLATE,
        "intermediate_template": _PEER_INTERMEDIATE_TEMPLATE,
        "reasoning_template": _PEER_REASONING_TEMPLATE,
        "truncation_template": _PEER_TRUNCATION_TEMPLATE,
        "peer_separator": _PEER_SEPARATOR,
    }


PEER_CONTEXT_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(_protocol_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()


def peer_context_protocol_metadata() -> dict[str, int | str]:
    """Return immutable protocol identifiers recorded in new cell metadata."""

    return {
        "peer_context_protocol_version": PEER_CONTEXT_PROTOCOL_VERSION,
        "peer_context_protocol_hash": PEER_CONTEXT_PROTOCOL_HASH,
        "peer_cot_char_limit": PEER_COT_CHAR_LIMIT,
        "peer_rendered_block_token_limit": PEER_RENDERED_BLOCK_TOKEN_LIMIT,
    }


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    if not hasattr(tokenizer, "encode"):
        raise TypeError("peer-context tokenizer must expose encode()")
    value = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(value, Mapping):
        try:
            value = value["input_ids"]
        except KeyError as exc:
            raise TypeError("peer-context tokenizer mapping has no input_ids") from exc
    if isinstance(value, str):
        raise TypeError("peer-context tokenizer returned text instead of token IDs")
    try:
        ids = tuple(value)
    except TypeError as exc:
        raise TypeError("peer-context tokenizer returned no token-ID sequence") from exc
    if not all(
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        for token_id in ids
    ):
        raise TypeError("peer-context tokenizer returned invalid token IDs")
    return ids


def _decode_ids(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    if not hasattr(tokenizer, "decode"):
        raise TypeError("peer-context truncation requires tokenizer.decode()")
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    if not isinstance(text, str):
        raise TypeError("peer-context tokenizer.decode() returned non-text")
    return text


def _unbounded_peer_block(out: AgentOutput, level: ContextShareLevel) -> str:
    parts = [
        _PEER_ANSWER_TEMPLATE.format(
            agent_id=out.agent_id, answer_choice=out.answer_choice
        )
    ]
    if out.verbalized_conf is not None:
        parts.append(
            _PEER_CONFIDENCE_TEMPLATE.format(
                confidence_percent=int(out.verbalized_conf * 100)
            )
        )
    block = " ".join(parts)
    if (
        level.rank >= ContextShareLevel.PLUS_INTERMEDIATE.rank
        and out.intermediate_results
    ):
        block += _PEER_INTERMEDIATE_TEMPLATE.format(
            intermediate_results=out.intermediate_results
        )
    if level.rank >= ContextShareLevel.PLUS_COT.rank and out.cot_text:
        block += _PEER_REASONING_TEMPLATE.format(
            cot_text=out.cot_text[:PEER_COT_CHAR_LIMIT]
        )
    return block


def _bounded_peer_block(
    out: AgentOutput,
    level: ContextShareLevel,
    tokenizer: Any,
) -> tuple[str, int, bool]:
    original = _unbounded_peer_block(out, level)
    original_ids = _token_ids(tokenizer, original)
    if len(original_ids) <= PEER_RENDERED_BLOCK_TOKEN_LIMIT:
        return original, len(original_ids), False

    original_sha256 = hashlib.sha256(original.encode("utf-8")).hexdigest()
    marker = _PEER_TRUNCATION_TEMPLATE.format(
        original_tokens=len(original_ids),
        original_sha256=original_sha256,
        token_limit=PEER_RENDERED_BLOCK_TOKEN_LIMIT,
    )
    marker_tokens = len(_token_ids(tokenizer, marker))
    if marker_tokens > PEER_RENDERED_BLOCK_TOKEN_LIMIT:
        raise ValueError("peer-context truncation marker exceeds the block token limit")

    # Decode a token-prefix rather than cutting Unicode/code points. Re-encode the
    # candidate every iteration because boundary merges can change when the marker is
    # appended. Prefix length only decreases, so this terminates even for unusual BPEs.
    prefix_tokens = max(0, PEER_RENDERED_BLOCK_TOKEN_LIMIT - marker_tokens)
    while True:
        prefix = _decode_ids(tokenizer, original_ids[:prefix_tokens])
        candidate = prefix + marker
        candidate_tokens = len(_token_ids(tokenizer, candidate))
        if candidate_tokens <= PEER_RENDERED_BLOCK_TOKEN_LIMIT:
            return candidate, candidate_tokens, True
        overflow = candidate_tokens - PEER_RENDERED_BLOCK_TOKEN_LIMIT
        prefix_tokens = max(0, prefix_tokens - max(1, overflow))


def render_peer_context(
    peers: list[AgentOutput],
    level: ContextShareLevel,
    *,
    tokenizer: Any,
) -> PeerContextRender:
    """Render peer context using the exact serving-profile tokenizer.

    The final per-block counts are re-encoded counts, never estimates.  Whole-context
    count/hash provenance allows each consuming generation to be audited independently.
    """

    if not peers:
        return PeerContextRender(
            text="",
            token_count=0,
            sha256=hashlib.sha256(b"").hexdigest(),
        )
    blocks: list[str] = []
    block_counts: list[int] = []
    truncations = 0
    for peer in peers:
        block, token_count, truncated = _bounded_peer_block(peer, level, tokenizer)
        blocks.append(block)
        block_counts.append(token_count)
        truncations += int(truncated)
    text = _PEER_CONTEXT_INTRO + _PEER_SEPARATOR.join(blocks)
    return PeerContextRender(
        text=text,
        token_count=len(_token_ids(tokenizer, text)),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        block_token_counts=tuple(block_counts),
        truncation_marker_count=truncations,
    )


def build_peer_context(
    peers: list[AgentOutput],
    level: ContextShareLevel,
    *,
    tokenizer: Any,
) -> str:
    """Compatibility text view of :func:`render_peer_context`."""

    return render_peer_context(peers, level, tokenizer=tokenizer).text


def shared_token_estimate(
    peers: list[AgentOutput],
    level: ContextShareLevel,
    *,
    tokenizer: Any,
) -> int:
    """Return the exact shared-context count (historical function name retained)."""

    return render_peer_context(peers, level, tokenizer=tokenizer).token_count
