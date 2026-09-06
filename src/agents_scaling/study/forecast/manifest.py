"""Trusted forecast manifest, the rendered shadow-forecast request and anchor mapping (§8.7, §8.4).

The manifest is supplied by the harness, never by model text (§8.7 "the trusted request
manifest — not model-invented text — supplies ...").  For the FINAL_HANDOFF_REPORT node:

* ``scope``: ``[PERSONAL_FINAL, TEAM_SELECTED]``;
* ``mask``: ``q_personal``/``q_team_now`` required, ``q_child_contract``/``q_recover``/
  ``q_preserve`` null (no child contract, no registered extra-budget operation);
* ``observer_role``: ``COMMON_REPORT_READER``; ``information_set``: ``SELF_ONLY`` (IND_VOTE:
  members never saw each other), ``PEER_EXPOSED`` (DEC), ``HUB_STATE`` (CEN_FLAT);
* ``selected_pool_id``: the sealed pool id; ``checkpoint_id``: ``FINAL_HANDOFF_REPORT``;
  ``operation_id``: ``NONE``; ``remaining_allowance``: the episode's exact ledger slack.

The request is ``prompts/templates/forecast.txt`` with its two placeholders filled by
``prompts.render.render_forecast`` (manifest JSON + schema line; the report between the
REPORT delimiters): one user message, thinking off, temperature 0, 256 output tokens
(``types.FORECAST_DECODING``), ``guided_json=False`` (05_vllm_response_shape.md).

Anchors (§8.4 "resolve spans through the serializer parser and tokenizer offsets, never by
searching for a repeated sentinel string"): byte offsets into the user message content
(``task_only_anchor`` = end of the task span, ``state_anchor`` = end of the complete
compiled report, i.e. the end of ``=== END REPORT ===``) are mapped to prompt-token indices
through the tokenizer's offset mapping over the rendered chat text, verified to reproduce
the exact ``prompt_token_ids``.  The anchor token is the last token that *starts* before
the anchor byte; ``straddles`` records a token that also extends past it (e.g. ``?\\n``).
The state anchor is not the last chat-template token; both are recorded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.config import StudyConfig
from agents_scaling.study.forecast.report import EVIDENCE_TOKENS_CAP, Report
from agents_scaling.study.inference.tokens import chat_template_hash, render_chat_text, render_chat_token_ids
from agents_scaling.study.prompts import render as R
from agents_scaling.study.types import (
    FORECAST_DECODING,
    NS_FORECAST,
    PROMPT_TOKENS_CAP,
    PURPOSE_FORECAST,
    Checkpoint,
    Method,
    ProtocolError,
    RequestSpec,
    SeedKey,
)

RENDER_SCHEMA_VERSION = 1
SCOPES: tuple[str, ...] = ("PERSONAL_FINAL", "TEAM_SELECTED")
REQUIRED = "required"
MASK: Mapping[str, str | None] = {
    "q_personal": REQUIRED,
    "q_team_now": REQUIRED,
    "q_child_contract": None,
    "q_recover": None,
    "q_preserve": None,
}
OBSERVER_ROLE = "COMMON_REPORT_READER"
CHECKPOINT_ID = "FINAL_HANDOFF_REPORT"
OPERATION_ID = "NONE"
ALLOWANCE_UNIT = "logical_flops"
INFORMATION_SETS: Mapping[Method, str] = {
    Method.IND_VOTE: "SELF_ONLY",
    Method.DEC: "PEER_EXPOSED",
    Method.CEN_FLAT: "HUB_STATE",
}
ROLE_FORECAST = "forecast"
ANCHOR_NAMES: tuple[str, ...] = ("task_only_anchor", "state_anchor")

_WS_TOKEN = re.compile(r"\S+")


# --------------------------------------------------------------------------- manifest


def forecast_manifest(report: Report) -> dict[str, Any]:
    """The trusted §8.7 manifest of ``report`` (all eight ``FORECAST_MANIFEST_KEYS`` + the unit)."""
    slack = report.budget_metadata.get("slack_flops")
    if slack is None:
        raise ProtocolError(f"report {report.report_id[:12]}: the episode ledger carries no slack (remaining_allowance)")
    manifest = {
        "scope": list(SCOPES),
        "mask": dict(MASK),
        "observer_role": OBSERVER_ROLE,
        "information_set": INFORMATION_SETS[report.method],
        "selected_pool_id": report.pool_id,
        "checkpoint_id": CHECKPOINT_ID,
        "operation_id": OPERATION_ID,
        "remaining_allowance": int(slack),
        "allowance_unit": ALLOWANCE_UNIT,
    }
    missing = [k for k in R.FORECAST_MANIFEST_KEYS if k not in manifest]
    assert not missing, missing
    return manifest


def required_fields(mask: Mapping[str, Any] = MASK) -> tuple[str, ...]:
    return tuple(k for k, v in mask.items() if v == REQUIRED)


def null_fields(mask: Mapping[str, Any] = MASK) -> tuple[str, ...]:
    return tuple(k for k, v in mask.items() if v is None)


# --------------------------------------------------------------------------- request


def render_forecast_request(report: Report, manifest: Mapping[str, Any] | None = None) -> R.Rendered:
    """The single-user-message forecast prompt with byte anchors into its content.

    ``anchors``: ``task_only_anchor`` (end of the task span inside the report),
    ``state_anchor`` (end of ``=== END REPORT ===``), and ``spans`` (manifest, report, task,
    evidence, packets) — all UTF-8 byte offsets into ``messages[0]["content"]``.
    """
    manifest = dict(manifest) if manifest is not None else forecast_manifest(report)
    rendered = R.render_forecast(report.text, manifest)
    report_start, report_end = rendered.anchors["spans"]["report"]
    content = rendered.content.encode("utf-8")
    if content[report_start:report_end].decode("utf-8") != report.text:
        raise ProtocolError("render_forecast did not insert the report bytes verbatim")
    shift = report_start
    task_span = report.spans["task"]
    evidence = report.spans["evidence"]
    anchors: dict[str, Any] = {
        "task_only_anchor": shift + report.task_only_anchor,
        "state_anchor": int(rendered.anchors["state_anchor"]),
        "spans": {
            "manifest": tuple(rendered.anchors["spans"]["manifest"]),
            "report": (report_start, report_end),
            "task": (shift + task_span[0], shift + task_span[1]),
            "evidence": (shift + evidence[0], shift + evidence[1]),
            "packets": [(shift + s, shift + e) for s, e in report.spans["packets"]],
        },
        "content_bytes": len(content),
    }
    return R.Rendered(rendered.messages, anchors)


def forecast_seed_key(report: Report, checkpoint: Checkpoint) -> SeedKey:
    """``(source_id, split, model_cell, episode_rep, actor_slot=0, "forecast", 0, "forecast")``."""
    return SeedKey(
        source_id=report.source_id,
        split=str(report.item["split"]),
        model_cell=checkpoint.model_cell,
        episode_rep=int(report.item["episode_rep"]),
        actor_slot=0,
        purpose=PURPOSE_FORECAST,
        step_slot=0,
        namespace=NS_FORECAST,
    )


def forecast_spec(report: Report, rendered: R.Rendered, cfg: StudyConfig, checkpoint: Checkpoint) -> RequestSpec:
    """The forecast :class:`RequestSpec` (FORECAST_DECODING; identity per §3.6)."""
    return RequestSpec(
        messages=tuple(dict(m) for m in rendered.messages),
        decoding=FORECAST_DECODING,
        checkpoint=checkpoint,
        seed_key=forecast_seed_key(report, checkpoint),
        role=ROLE_FORECAST,
        study_id=cfg.study_id,
        study_seed_hex=cfg.study_seed_hex,
    )


# --------------------------------------------------------------------------- byte → token anchors


def token_offsets(tokenizer: Any, text: str, ids: Sequence[int]) -> list[tuple[int, int]]:
    """Character spans of every token of ``ids`` in ``text``, verified to reproduce ``ids``.

    Fast HF tokenizers supply ``return_offsets_mapping``; whitespace tokenizers (the test
    stubs) are handled by re-encoding each ``\\S+`` run.  Any disagreement with ``ids`` is a
    ``ProtocolError`` (anchors are never guessed).
    """
    ids = list(ids)
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:  # not callable / slow tokenizer / no offsets
        enc = None
    if enc is not None:
        try:
            got = list(enc["input_ids"])
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        except (KeyError, TypeError, ValueError):
            enc = None
        else:
            if got == ids and len(offsets) == len(ids):
                return offsets
    spans = [(m.start(), m.end()) for m in _WS_TOKEN.finditer(text)]
    got = []
    for start, end in spans:
        piece = tokenizer.encode(text[start:end], add_special_tokens=False)
        if isinstance(piece, Mapping):
            piece = piece["input_ids"]
        got.extend(int(i) for i in piece)
    if got == ids and len(spans) == len(ids):
        return spans
    raise ProtocolError("cannot map byte anchors to prompt tokens: the tokenizer offsets do not reproduce the rendered prompt ids")


def anchor_token(offsets: Sequence[tuple[int, int]], char_pos: int) -> dict[str, Any]:
    """The token holding the last character before ``char_pos`` (the last token that starts
    before the anchor); ``straddles`` when it also extends past the anchor."""
    index = None
    for i, (start, end) in enumerate(offsets):
        if end > start and start < char_pos:
            index = i
    if index is None:
        raise ProtocolError(f"no token starts before character {char_pos}")
    start, end = offsets[index]
    return {"index": index, "char_start": start, "char_end": end, "straddles": end > char_pos, "exact": end == char_pos}


def anchor_tokens(tokenizer: Any, messages: Sequence[Mapping[str, Any]], anchors: Mapping[str, Any], *, enable_thinking: bool = False) -> dict[str, Any]:
    """Map the byte anchors of a single-user-message prompt to prompt-token indices.

    Returns ``{"prompt_token_ids", "prompt_tokens", "rendered_text_sha256", "content_char_start",
    "anchors": {name: {byte, char, index, ...}}, "last_prompt_token": n-1}``.
    """
    if len(messages) != 1:
        raise ProtocolError("forecast prompts are single user messages")
    ids = list(render_chat_token_ids(tokenizer, messages, enable_thinking))
    text = render_chat_text(tokenizer, messages, enable_thinking)
    content = str(messages[0]["content"])
    first = text.find(content)
    if first < 0 or text.find(content, first + 1) >= 0:
        raise ProtocolError("the user message content does not occur exactly once in the rendered chat text")
    offsets = token_offsets(tokenizer, text, ids)
    raw = content.encode("utf-8")
    out: dict[str, Any] = {}
    for name in ANCHOR_NAMES:
        byte = int(anchors[name])
        if not 0 < byte <= len(raw):
            raise ProtocolError(f"{name} byte offset {byte} outside the content")
        char = first + len(raw[:byte].decode("utf-8"))  # anchors sit on ASCII delimiter boundaries
        out[name] = {"byte": byte, "char": char, **anchor_token(offsets, char)}
    return {
        "prompt_token_ids": ids,
        "prompt_tokens": len(ids),
        "rendered_text_sha256": identity.sha256_hex(text),
        "content_char_start": first,
        "anchors": out,
        "last_prompt_token": len(ids) - 1,
    }


# --------------------------------------------------------------------------- the capture-stage render


def report_render(report: Report, tokenizer: Any, cfg: StudyConfig, checkpoint: Checkpoint) -> dict[str, Any]:
    """Everything the neural capture stage (N1) needs for one report, as a JSON-safe dict.

    Carries the exact messages, manifest, ``prompt_token_ids`` (pinned reader tokenizer,
    thinking off), the two anchor token indices, the forecast ``request_id`` (so the capture
    can join the shadow forecast record) and the compiled report itself.  Raises
    ``ProtocolError`` when the rendered prompt exceeds the 32,768-token envelope.
    """
    manifest = forecast_manifest(report)
    rendered = render_forecast_request(report, manifest)
    spec = forecast_spec(report, rendered, cfg, checkpoint)
    tokens = anchor_tokens(tokenizer, rendered.messages, rendered.anchors, enable_thinking=bool(FORECAST_DECODING.enable_thinking))
    if tokens["prompt_tokens"] > PROMPT_TOKENS_CAP:
        raise ProtocolError(f"forecast prompt has {tokens['prompt_tokens']} tokens > {PROMPT_TOKENS_CAP} (§4.3)")
    return {
        "schema_version": RENDER_SCHEMA_VERSION,
        "kind": CHECKPOINT_ID,
        "report_id": report.report_id,
        "source_id": report.source_id,
        "method": report.method.value,
        "cell_id": report.cell_id,
        "seal": report.seal,
        "selection_id": report.selection_id,
        "pool_id": report.pool_id,
        "request_id": spec.request_id,
        "input_hash": spec.input_hash,
        "checkpoint": {
            "size": checkpoint.size,
            "hf_id": checkpoint.hf_id,
            "model_revision": checkpoint.model_revision,
            "tokenizer_revision": checkpoint.tokenizer_revision,
            "model_cell": checkpoint.model_cell,
        },
        "decoding": FORECAST_DECODING.as_strings(),
        "chat_template_kwargs": dict(spec.chat_template_kwargs),
        "chat_template_hash": chat_template_hash(tokenizer) if getattr(tokenizer, "chat_template", None) else None,
        "seed_key": spec.seed_key.as_array(),
        "engine_seed": spec.engine_seed,
        "manifest": manifest,
        "messages": [dict(m) for m in rendered.messages],
        "byte_anchors": {k: rendered.anchors[k] for k in ANCHOR_NAMES},
        "byte_spans": {k: v for k, v in rendered.anchors["spans"].items()},
        "content_bytes": rendered.anchors["content_bytes"],
        "prompt_token_ids": tokens["prompt_token_ids"],
        "prompt_tokens": tokens["prompt_tokens"],
        "rendered_text_sha256": tokens["rendered_text_sha256"],
        "anchor_tokens": tokens["anchors"],
        "last_prompt_token": tokens["last_prompt_token"],
        "evidence_tokens": report.evidence_tokens,
        "evidence_tokens_cap": EVIDENCE_TOKENS_CAP,
        "report": report.to_dict(),
    }


__all__ = [
    "ALLOWANCE_UNIT",
    "ANCHOR_NAMES",
    "CHECKPOINT_ID",
    "INFORMATION_SETS",
    "MASK",
    "OBSERVER_ROLE",
    "OPERATION_ID",
    "RENDER_SCHEMA_VERSION",
    "REQUIRED",
    "ROLE_FORECAST",
    "SCOPES",
    "anchor_token",
    "anchor_tokens",
    "forecast_manifest",
    "forecast_seed_key",
    "forecast_spec",
    "null_fields",
    "render_forecast_request",
    "report_render",
    "required_fields",
    "token_offsets",
]
