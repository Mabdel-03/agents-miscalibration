"""Read-only activation capture: anchors, the panel manifest and the CAPTURE cell worker.

Spec 13.2 ("Capture plan") / 13.3 ("Geometry metrics"); plan §1.2 (``capture.py``), §3.3
(the CAPTURE lane), D11 (read-only replay; no intervention in v1).  WP6.

Why this module exists at all
-----------------------------
Nothing here generates.  A CAPTURE cell has no endpoint, no client and no sampler: the
request store already holds ``prompt_token_ids`` and ``response.token_ids`` for every
decision, so the pass is a *teacher-forced replay of exactly those ids* through the same
pinned model, and "instrumented inference must not alter behavioural generation" (spec 13.2)
is made structural rather than promised -- the import closure of this module never reaches
:mod:`agents_scaling.nextdyn.inference`
(``test_read_only_capture_cannot_change_generation``).

Three anchors, resolved from real byte spans (spec 13.2)
-------------------------------------------------------
======================  ===========================================================
``PROMPT_END``          last prompt token before this call's generation
``PRE_FINAL``           last token before the first token of the final-answer *value*,
                        after the generated public rationale
``ANSWER_END``          last token of the final-answer value, before the public
                        confidence value
======================  ===========================================================

ARD36's three-tag report gave ``<final>`` delimiters to scan for; the spec-6.1 output here
is one JSON object, so the spans come from the **parser that read it**:
:attr:`schema.PublicOutput.final_answer_span` and ``confidence_span`` are byte spans of the
*value* regions inside the final text, recorded once by ``schema.parse_public_output`` and
carried on ``DecisionRow.token_offsets_json``.  This module maps those bytes onto the stored
token ids through ``study.neural.anchors.token_bytes`` / ``byte_spans`` /
``last_token_of_span`` (which assert ``b"".join(pieces) == tokenizer.decode(ids)``), so no
anchor is ever a fixed offset from the end of a sequence.

A byte offset is only meaningful against the text it was measured in, so the public span is
located **structurally** -- the bytes after the ``</think>`` token of the replayed sequence,
never a channel flag copied off the replayed record -- and its origin is then *proved*
against the content hash the parser recorded (:func:`_public_origin`).  A continued draw
(amendment A3) is the case that makes this load-bearing: the record replayed is phase 2,
whose ``chat_template_kwargs`` say ``enable_thinking=False`` (``/v1/completions`` renders no
chat template at all) while its prompt *contains* phase 1's thinking span, so a flag-driven
cut would measure every anchor from the start of the reasoning text.  An unproved origin is
a typed missing anchor (:data:`MISSING_ANCHOR_UNRESOLVED`), never an offset.

Two rules the spec states and this module makes mechanical:

* **A missing anchor is a typed ledger row with no vector, never a zero vector.**  An
  invalid answer, a truncated span or a byte→token mapping that fails produces an
  :class:`ActivationRow` with a ``missing_reason`` from :data:`MISSING_REASONS` and a null
  ``storage_hash``.  ``study.neural.storage.ShardWriter`` refuses the other combinations in
  both directions, and nothing substitutes a different anchor: the substitute would be
  chosen *after* the answer is known, which is exactly the correctness-dependent selection
  spec 13.2 forbids.
* **``ANSWER_END`` is pre-confidence only when confidence was not already emitted.**  The
  spec-6.1 field order puts ``confidence`` last, but a model may emit it earlier; the parser
  records that as ``confidence_emitted_before_answer`` and the eligibility travels to the
  analysis on :attr:`tables.ActivationRow.pre_confidence_eligible` rather than being
  re-derived downstream (R1).

Layers
------
Blocks are ``floor(f*L)-1`` for ``f`` in ``study.instrumentation.relative_layers`` and the
**actual** layer count ``L`` of the pinned model -- 36 for Qwen3-8B (8, 17, 26) and 64 for
Qwen3-32B (15, 31, 47).  ``L`` is read from ``models.lock.json`` (``n_layers``) or, failing
that, from the offline snapshot's ``config.json``; it is never a literal here
(``test_activation_layer_and_anchor_offsets_valid``).

Geometry
--------
Participation ratio, entropy effective rank, distances and displacements are **not**
re-derived: :func:`team_geometry` and :func:`agent_geometry` wrap
``ard.neural.geometry.team_metrics`` / ``agent_metrics`` (both on the ``ARD_IMPORTS``
allowlist), which already carry the ``rank_ceiling(d, s) = min(d, s-1)`` ceiling and return
NaN spectral ratios when the centered covariance has no nonzero eigenvalue.  Spec 13.3:
"when total variance is zero, spectral ratios are undefined, not evidence of a
one-dimensional collapse" -- :func:`spectral_ratios_defined` is the one reader of that fact.

Mechanisms reused unchanged: ``study.neural.engine.CaptureEngine`` (hooks block *outputs*),
``study.neural.replay`` (``teacher_forced_sequence``; ``check_prompt_identity`` against the
decision's **own** -- phase 1 -- record, and ``fidelity`` once per pass for the spec-13.2
tolerance numbers), ``study.neural.storage.ShardWriter`` (atomic chunked fp16 storage, resume by
committed key) and ``ard.neural.capture.consolidate_shards`` (``.npz`` before its ``.jsonl``,
the commit marker).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np

from agents_scaling.ard.neural.capture import consolidate_shards
from agents_scaling.ard.neural.geometry import agent_metrics, team_metrics
from agents_scaling.nextdyn import cli, deps, ids
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn.errors import ContractViolation, DependencyUnmet, ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.study.neural import storage as S
from agents_scaling.study.neural.anchors import (
    AnchorError,
    byte_spans,
    last_token_of_span,
    think_end_id,
    token_at_byte,
    token_bytes,
)
from agents_scaling.study.neural.engine import HOOK_CONVENTION, CaptureRequest
from agents_scaling.study.neural.replay import (
    DEFAULT_FIDELITY_TOKENS,
    ReplayError,
    check_prompt_identity,
    fidelity,
    teacher_forced_sequence,
)

# --------------------------------------------------------------------------- vocabulary

ANCHOR_PROMPT_END: Final[str] = "PROMPT_END"
ANCHOR_PRE_FINAL: Final[str] = "PRE_FINAL"
ANCHOR_ANSWER_END: Final[str] = "ANSWER_END"
#: The spec-13.2 anchors, in spec order.  ``study.instrumentation.anchors`` may narrow this
#: set (the preregistered degradation) but may never name something outside it.
ANCHORS: Final[tuple[str, ...]] = (ANCHOR_PROMPT_END, ANCHOR_PRE_FINAL, ANCHOR_ANSWER_END)

#: The stage ``study.neural.storage`` files vectors under; a single-request replay is the
#: "native" stage, and nextdyn has its own run root so nothing collides.
CAPTURE_STAGE: Final[str] = "native"

# -- typed missingness (spec 13.2 "mark the capture missing with a reason") ---------------
#: The decision never reached vLLM (a build-time context refusal): there are no ids at all.
MISSING_CONTEXT_FAILURE: Final[str] = "CONTEXT_FAILURE"
#: The stored completion never closed its thinking channel, so there is no public span.
MISSING_NO_PUBLIC_SPAN: Final[str] = "NO_PUBLIC_SPAN"
#: The public output carries no ``final_answer`` value span -- an invalid answer (spec 13.2:
#: "if an answer is invalid ... mark the capture missing with a reason").
MISSING_INVALID_ANSWER: Final[str] = "INVALID_ANSWER"
#: The recorded byte span points outside the stored token bytes (a truncated span).
MISSING_SPAN_TRUNCATED: Final[str] = "SPAN_TRUNCATED"
#: The byte→token mapping itself failed (a decode that is not prefix-monotone).
MISSING_ANCHOR_UNRESOLVED: Final[str] = "ANCHOR_UNRESOLVED"
#: The value begins at the first byte of the public span, so no token precedes it.
MISSING_NO_PRECEDING_TOKEN: Final[str] = "NO_PRECEDING_TOKEN"
#: The store holds no record for the decision's physical request.
MISSING_RECORD: Final[str] = "RECORD_ABSENT"
MISSING_REASONS: Final[tuple[str, ...]] = (
    MISSING_CONTEXT_FAILURE, MISSING_NO_PUBLIC_SPAN, MISSING_INVALID_ANSWER,
    MISSING_SPAN_TRUNCATED, MISSING_ANCHOR_UNRESOLVED, MISSING_NO_PRECEDING_TOKEN,
    MISSING_RECORD,
)

# -- ``ActivationRow.replay_parity`` ------------------------------------------------------
#: A vector was written in this pass and the stored prompt reproduced under the tokenizer.
PARITY_OK: Final[str] = "ok"
#: A vector was written but the prompt-identity check was not run (an explicit opt-out).
PARITY_UNCHECKED: Final[str] = "identity_unchecked"
#: No vector: the row is the typed ledger line of a missing anchor.
PARITY_MISSING: Final[str] = "missing"
#: The key was already committed by an earlier pass; this row points at that vector.
PARITY_RESUMED: Final[str] = "resumed"

VECTOR_DTYPE: Final[str] = "float16"

#: The scale-free spectral quantities spec 13.3 declares *undefined* at zero total variance.
SPECTRAL_RATIOS: Final[tuple[str, ...]] = (
    "participation_ratio", "entropy_rank", "leading_eigenvalue_share")


class CaptureError(ManifestError):
    """A CAPTURE cell asked to replay nothing, or that replayed nothing it was asked for.

    A replay cell has no endpoint and no grader watching it: a worker that exits 0 having
    captured nothing writes a ``meta.json``, the lane driver marks the cell complete, and the
    run ends with an empty ``native`` stage nobody notices until the analysis finds no rows.
    """


# --------------------------------------------------------------------------- layers


def capture_blocks(num_layers: int, relative_layers: Sequence[float]) -> tuple[int, ...]:
    """Spec 13.2 zero-based capture blocks ``floor(f*L)-1``, unique and ascending.

    Qwen3-8B (L=36) -> (8, 17, 26); Qwen3-32B (L=64) -> (15, 31, 47).  A fraction that leaves
    ``[0, L)`` for this ``L`` -- which is what a two-layer fixture does -- is a
    :class:`ContractViolation`, never a clamp: a silently clamped block would make two models'
    "quarter depth" the same number.
    """
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
        raise ContractViolation(f"layer count must be a positive int, got {num_layers!r}")
    fracs = tuple(float(f) for f in relative_layers)
    if not fracs:
        raise ContractViolation("study.instrumentation.relative_layers is empty")
    blocks = sorted({int(math.floor(f * num_layers)) - 1 for f in fracs})
    outside = [b for b in blocks if b < 0 or b >= num_layers]
    if outside:
        raise ContractViolation(
            f"relative_layers {fracs} give blocks {blocks} for L={num_layers}; {outside} are "
            "outside [0, L) (spec 13.2 floor(f*L)-1)")
    return tuple(blocks)


def model_layer_count(entry: Any = None, *, snapshot_path: str | os.PathLike | None = None) -> int:
    """``L`` of a pinned model: ``models.lock.json``'s ``n_layers``, else the snapshot config.

    Never a literal.  ``entry`` is a :class:`locks.ModelLockEntry` (or anything carrying
    ``n_layers``); ``snapshot_path`` is the offline HF snapshot directory whose ``config.json``
    declares ``num_hidden_layers``.  Both absent is a refusal, because guessing the depth is
    how a capture ends up describing the wrong quarter of a different model.
    """
    declared = getattr(entry, "n_layers", None) if entry is not None else None
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return int(declared)
    if snapshot_path is not None:
        config = Path(snapshot_path) / "config.json"
        if config.is_file():
            value = json.loads(config.read_text(encoding="utf-8")).get("num_hidden_layers")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return int(value)
            raise ContractViolation(f"{config}: no usable num_hidden_layers")
    raise ContractViolation(
        "the model's layer count is unknown: models.lock.json carries no n_layers and no "
        "offline snapshot was given (spec 13.2: derive every model from its pinned config)")


def blocks_for_model(entry: Any, relative_layers: Sequence[float], *,
                     snapshot_path: str | os.PathLike | None = None) -> tuple[int, ...]:
    """:func:`capture_blocks` at the pinned model's own layer count."""
    return capture_blocks(model_layer_count(entry, snapshot_path=snapshot_path), relative_layers)


# --------------------------------------------------------------------------- spans


@dataclass(frozen=True)
class AnswerSpans:
    """The three span facts an anchor needs, as the WP2 parser recorded them.

    Byte offsets are into the **final text** (the post-``</think>`` public span the parser
    saw).  This is a view, not a second declaration: it is built either from a live
    :class:`schema.PublicOutput` or from ``DecisionRow.token_offsets_json``, which
    ``telemetry.decision_row`` writes from exactly those attributes (R1).
    """

    final_answer_span: tuple[int, int] | None
    confidence_span: tuple[int, int] | None
    confidence_emitted_before_answer: bool
    parse_status: str = ""
    #: ``ids.content_hash`` of the final-text bytes the spans index into -- the parser's own
    #: hash (``PublicOutput.content_hash``, written to ``DecisionRow.content_hash`` by
    #: ``telemetry.decision_row``).  It is what proves the origin of a byte offset inside the
    #: replayed token stream; an empty value is "unstated", never "matches".
    content_hash: str = ""
    #: The parsed final-answer value, the fallback proof when a report's final text was cut by
    #: ``final_cap`` and therefore hashes differently from the stored ids.
    final_answer: str | None = None

    @property
    def answer_end_pre_confidence_eligible(self) -> bool:
        """Spec 13.2/6.1: ``ANSWER_END`` describes a pre-confidence state only when the
        confidence value had not already been emitted at that point."""
        return self.final_answer_span is not None and not self.confidence_emitted_before_answer

    @classmethod
    def of(cls, source: Any) -> "AnswerSpans":
        """From a :class:`schema.PublicOutput`, a :class:`tables.DecisionRow`, or self."""
        if isinstance(source, AnswerSpans):
            return source
        if hasattr(source, "token_offsets_json"):
            return cls.from_decision_row(source)
        return cls.from_output(source)

    @classmethod
    def from_output(cls, out: Any) -> "AnswerSpans":
        return cls(
            final_answer_span=_span(getattr(out, "final_answer_span", None)),
            confidence_span=_span(getattr(out, "confidence_span", None)),
            confidence_emitted_before_answer=bool(getattr(out, "confidence_emitted_before_answer", False)),
            parse_status=str(getattr(getattr(out, "parse_status", ""), "value", getattr(out, "parse_status", ""))),
            content_hash=str(getattr(out, "content_hash", "") or ""),
            final_answer=getattr(out, "final_answer", None),
        )

    @classmethod
    def from_decision_row(cls, row: Any) -> "AnswerSpans":
        raw = row.token_offsets_json
        payload = json.loads(raw) if isinstance(raw, str) and raw else {}
        if not isinstance(payload, Mapping):
            raise ManifestError("token_offsets_json is not a JSON object")
        return cls(
            final_answer_span=_span(payload.get("final_answer_span")),
            confidence_span=_span(payload.get("confidence_span")),
            confidence_emitted_before_answer=bool(payload.get("confidence_emitted_before_answer", False)),
            parse_status=str(row.parse_status),
            content_hash=str(row.content_hash or ""),
            final_answer=row.final_answer_raw,
        )


def _span(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    start, stop = (int(v) for v in value)
    return (start, stop)


# --------------------------------------------------------------------------- anchors


@dataclass(frozen=True)
class Anchor:
    """One capture site of one decision, as an absolute offset into the replayed sequence."""

    name: str
    token_offset: int | None
    missing_reason: str | None
    byte_offset: int | None = None
    pre_confidence_eligible: bool | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ANCHORS:
            raise ContractViolation(f"anchor {self.name!r} is not one of {ANCHORS} (spec 13.2)")
        if (self.token_offset is None) != (self.missing_reason is not None):
            raise ContractViolation(
                f"{self.name}: an anchor has a token offset or a missing reason, never both and "
                "never neither (a missing anchor must not become a zero vector)")
        if self.missing_reason is not None and self.missing_reason not in MISSING_REASONS:
            raise ContractViolation(f"unknown missing reason {self.missing_reason!r}")
        if self.token_offset is not None and self.token_offset < 0:
            raise ContractViolation(f"{self.name}: negative token offset {self.token_offset}")
        object.__setattr__(self, "detail", dict(self.detail))

    @property
    def present(self) -> bool:
        return self.token_offset is not None


@dataclass(frozen=True)
class AnchorSet:
    """The spec-13.2 anchors of one decision plus what the resolution saw."""

    decision_id: str
    prompt_tokens: int
    sequence_tokens: int
    anchors: tuple[Anchor, ...]
    parse_status: str = ""
    public_span: tuple[int, int] | None = None
    #: Byte offset, inside the located public span, where the bytes the parser measured begin
    #: (0 unless the reasoning parser dropped leading bytes); ``None`` when nothing was proved.
    public_text_offset: int | None = None
    #: How that origin was proved: ``content_hash`` | ``answer_bytes`` | ``""`` (not proved).
    origin_proof: str = ""

    def by_name(self, name: str) -> Anchor:
        for anchor in self.anchors:
            if anchor.name == name:
                return anchor
        raise KeyError(name)

    @property
    def present_offsets(self) -> tuple[int, ...]:
        return tuple(sorted({a.token_offset for a in self.anchors if a.present}))

    @property
    def missing(self) -> tuple[Anchor, ...]:
        return tuple(a for a in self.anchors if not a.present)

    @property
    def sequence_tokens_needed(self) -> int:
        """Length of the sequence prefix a replay must prefill to reach every anchor."""
        return 0 if not self.present_offsets else max(self.present_offsets) + 1

    def tokens_needed(self, record_prompt_tokens: int) -> int:
        """Completion tokens of *the replayed record* the prefill must include.

        Not ``max(offset) - self.prompt_tokens``: on a continued decision ``prompt_tokens`` is
        phase 1's prompt length while the record being replayed is phase 2, whose own prompt
        already holds phase 1's thinking span and the frozen prefill.  The offsets index the
        composed sequence, so the count is taken against the record's own prompt.
        """
        return max(0, self.sequence_tokens_needed - int(record_prompt_tokens))


def _record_fields(record: Any) -> tuple[tuple[int, ...], tuple[int, ...], bool | None]:
    """``(prompt ids, completion ids, enable_thinking)`` of a record in any of its shapes.

    ``enable_thinking`` is ``None`` when the record cannot state it: an
    ``inference.DecisionRecord`` carries no ``chat_template_kwargs`` at all, and assuming
    ``True`` there would make the spec-5.6 non-thinking companion (the
    ``total_output_envelope`` adapter, which never emits ``</think>``) lose every content
    anchor silently.  Nothing defaults: the channel is located structurally and the origin is
    proved (:func:`_public_origin`), and an unstated flag only decides which *typed* refusal a
    missing ``</think>`` earns.
    """
    if isinstance(record, Mapping):
        prompt = record.get("prompt_token_ids") or ()
        response = record.get("response") or {}
        completion = response.get("token_ids") or ()
        kwargs = record.get("chat_template_kwargs") or {}
    elif hasattr(record, "completion_token_ids"):            # inference.DecisionRecord
        prompt = record.prompt_token_ids
        completion = record.completion_token_ids
        kwargs = {}
    else:                                                    # study.types.RequestRecord
        prompt = record.prompt_token_ids
        completion = (record.response or {}).get("token_ids") or ()
        kwargs = record.chat_template_kwargs or {}
    thinking = kwargs.get("enable_thinking")
    return (tuple(int(t) for t in prompt), tuple(int(t) for t in completion),
            None if thinking is None else bool(thinking))


def _special_token_ids(tokenizer: Any) -> frozenset[int]:
    """Ids whose bytes are template scaffolding, not content (``<|im_end|>`` and friends)."""
    added = getattr(tokenizer, "added_tokens_decoder", None)
    out = {int(k) for k in added} if isinstance(added, Mapping) else set()
    for tid in (getattr(tokenizer, "all_special_ids", None) or ()):
        try:
            out.add(int(tid))
        except (TypeError, ValueError):                      # pragma: no cover - defensive
            continue
    return frozenset(out)


def _public_origin(decoded: bytes, content: Sequence[int], token_spans: Sequence[tuple[int, int]],
                   tokenizer: Any, spans: AnswerSpans) -> tuple[int, str] | None:
    """Byte offset inside the located public span at which the *parser's* bytes begin.

    ``final_answer_span`` is a byte span of the final text ``schema.parse_public_output``
    read; applying it to the replayed token bytes is only correct once the origin of that text
    inside those bytes is proved.  Two proofs, in order:

    * the content hash the parser recorded (``ids.content_hash`` of the final-text bytes,
      carried on :attr:`AnswerSpans.content_hash`) equals the hash of a candidate byte range --
      the candidates differ only by what a reasoning parser is allowed to drop, trailing
      special tokens and leading/trailing whitespace;
    * failing that, the recorded final-answer *value* really does sit inside ``[start, stop)``
      at that origin.  A report cut by ``final_cap`` hashes differently from the ids the
      server stored yet still carries its answer exactly where the parser said it was.

    ``None`` means nothing was proved, and the caller turns that into
    :data:`MISSING_ANCHOR_UNRESOLVED` -- an unproved origin is exactly the correctness-blind
    anchor spec 13.2 forbids.
    """
    starts = [0]
    lead = len(decoded) - len(decoded.lstrip())
    if lead:
        starts.append(lead)
    if spans.content_hash:
        specials = _special_token_ids(tokenizer)
        ends = [len(decoded)]
        last = len(content) - 1
        while last >= 0 and int(content[last]) in specials:
            ends.append(token_spans[last][0])
            last -= 1
        for start in starts:
            for end in ends:
                if end <= start:
                    continue
                chunk = decoded[start:end]
                for candidate in (chunk, chunk.rstrip()):
                    if candidate and ids.content_hash(candidate) == spans.content_hash:
                        return start, "content_hash"
    answer = (spans.final_answer or "").encode("utf-8")
    if answer and spans.final_answer_span is not None:
        begin, stop = spans.final_answer_span
        for start in starts:
            if stop <= begin or start + stop > len(decoded):
                continue
            if answer in decoded[start + begin: start + stop]:
                return start, "answer_bytes"
    return None


def resolve_anchors(record: Any, out: Any, tokenizer: Any, *, decision_id: str = "",
                    prompt_tokens: int | None = None) -> AnchorSet:
    """The spec-13.2 anchors of one decision, as offsets into ``prompt ids + completion ids``.

    Offsets index the teacher-forced sequence ``study.neural.replay.teacher_forced_sequence``
    builds, which is exactly what ``CaptureEngine`` prefills.  ``out`` is the parsed public
    output (or the decision row that carries its spans); ``prompt_tokens`` overrides the
    prompt length for a decision executed as two physical requests, where the decision's own
    prompt is phase 1's and everything after it -- the thinking span, the frozen prefill and
    phase 2's report -- is generated content.

    The public span is located **structurally** (the bytes after the ``</think>`` token of the
    replayed region) and its origin is then proved by :func:`_public_origin`.  The replayed
    record's own ``enable_thinking`` is never the thing that decides the cut: on a continued
    draw that record is phase 2, which ``decoding.continuation_decoding`` builds with
    ``enable_thinking=False`` while its prompt holds the thinking span, so a flag-driven cut
    would put every anchor inside the reasoning text.
    """
    prompt, completion, enable_thinking = _record_fields(record)
    spans = AnswerSpans.of(out)
    prompt_len = int(prompt_tokens) if prompt_tokens is not None else len(prompt)
    sequence = prompt + completion
    if not prompt or prompt_len < 1 or prompt_len > len(sequence):
        return AnchorSet(decision_id=decision_id, prompt_tokens=len(prompt),
                         sequence_tokens=len(sequence), parse_status=spans.parse_status,
                         anchors=tuple(_missing(name, MISSING_CONTEXT_FAILURE, spans)
                                       for name in ANCHORS))

    prompt_anchor = Anchor(ANCHOR_PROMPT_END, prompt_len - 1, None, byte_offset=None,
                           detail={"prompt_tokens": prompt_len, "sequence_tokens": len(sequence)})

    def refused(reason: str, *, public_span: tuple[int, int] | None = None,
                detail: Mapping[str, Any] | None = None) -> AnchorSet:
        content_anchors = []
        for name in (ANCHOR_PRE_FINAL, ANCHOR_ANSWER_END):
            anchor = _missing(name, reason, spans)
            if detail:
                anchor = replace(anchor, detail={**anchor.detail, **dict(detail)})
            content_anchors.append(anchor)
        return AnchorSet(decision_id=decision_id, prompt_tokens=prompt_len,
                         sequence_tokens=len(sequence), parse_status=spans.parse_status,
                         public_span=public_span,
                         anchors=(prompt_anchor, *content_anchors))

    # A continued draw replays phase 2, whose prompt already holds phase 1's thinking span and
    # the frozen prefill: the close delimiter is inside the region by construction, so it is
    # required here whatever the record's own chat_template_kwargs claim.
    continued = prompt_tokens is not None and prompt_len < len(prompt)
    region = sequence[prompt_len:]
    close = think_end_id(tokenizer)
    if close is None and (continued or enable_thinking):
        raise ContractViolation(
            "the tokenizer does not carry '</think>' as one token; the public span of a "
            "native-thinking decision cannot be located on ids")
    if close is not None and close in region:
        cut = region.index(close) + 1
    elif continued or enable_thinking:
        # thinking was on -- or is inside the phase-2 prompt -- and never closed: no public span
        return refused(MISSING_NO_PUBLIC_SPAN)
    else:
        # the spec-5.6 non-thinking companion, or a record that does not state its channel:
        # the whole region is the candidate content span and the origin proof below settles it
        cut = 0
    content = region[cut:]
    base = prompt_len + cut
    public_span = (base, base + len(content))

    if spans.final_answer_span is None or not content:
        return refused(MISSING_INVALID_ANSWER if content else MISSING_NO_PUBLIC_SPAN,
                       public_span=public_span)

    pieces = token_bytes(tokenizer, content)
    byte_spans_ = byte_spans(pieces)
    decoded = b"".join(pieces)
    total = byte_spans_[-1][1] if byte_spans_ else 0
    start, stop = spans.final_answer_span
    if stop > total:
        return refused(MISSING_SPAN_TRUNCATED, public_span=public_span)

    origin = _public_origin(decoded, content, byte_spans_, tokenizer, spans)
    if origin is None:
        return refused(MISSING_ANCHOR_UNRESOLVED, public_span=public_span,
                       detail={"origin": "unproved",
                               "public_span_bytes": total,
                               "content_hash_stated": bool(spans.content_hash)})
    shift, proof = origin
    if shift + stop > total:
        return refused(MISSING_SPAN_TRUNCATED, public_span=public_span)

    pre_final = _anchor_at(ANCHOR_PRE_FINAL, byte_spans_, shift + start, base, total, spans,
                           empty_reason=MISSING_NO_PRECEDING_TOKEN)
    answer_end = _anchor_at(ANCHOR_ANSWER_END, byte_spans_, shift + stop, base, total, spans,
                            empty_reason=MISSING_INVALID_ANSWER)
    answer_end = replace(answer_end, pre_confidence_eligible=spans.answer_end_pre_confidence_eligible)
    return AnchorSet(decision_id=decision_id, prompt_tokens=prompt_len,
                     sequence_tokens=len(sequence), parse_status=spans.parse_status,
                     public_span=public_span, public_text_offset=shift, origin_proof=proof,
                     anchors=(prompt_anchor, pre_final, answer_end))


def _missing(name: str, reason: str, spans: AnswerSpans) -> Anchor:
    eligible = spans.answer_end_pre_confidence_eligible if name == ANCHOR_ANSWER_END else None
    return Anchor(name, None, reason, pre_confidence_eligible=eligible,
                  detail={"parse_status": spans.parse_status})


def _anchor_at(name: str, spans_: Sequence[tuple[int, int]], byte_end: int, base: int,
               total: int, spans: AnswerSpans, *, empty_reason: str) -> Anchor:
    """The token containing byte ``byte_end - 1`` of the public span, absolute.

    ``byte_end`` is the *exclusive* end of the region the anchor sits at the end of: for
    ``PRE_FINAL`` that is the first byte of the final-answer value (so the anchor is the token
    holding the byte immediately before it), for ``ANSWER_END`` the end of that value.
    """
    if byte_end <= 0:
        return _missing(name, empty_reason, spans)
    if byte_end > total:
        return _missing(name, MISSING_SPAN_TRUNCATED, spans)
    try:
        local = last_token_of_span(spans_, byte_end)
    except AnchorError as exc:                                   # pragma: no cover - defensive
        return replace(_missing(name, MISSING_ANCHOR_UNRESOLVED, spans),
                       detail={"error": str(exc)[:200], "parse_status": spans.parse_status})
    token_start, token_stop = spans_[local]
    return Anchor(
        name, base + local, None, byte_offset=byte_end,
        detail={"public_token_index": local, "token_byte_span": [token_start, token_stop],
                # the delimiter bytes and the value can share one token; recorded rather than
                # silently shifted (spec 13.2 "retain delimiter offsets and span mappings")
                "shared_token": bool(token_stop > byte_end),
                "parse_status": spans.parse_status})


def token_offset_of_byte(tokenizer: Any, token_ids: Sequence[int], byte_offset: int) -> int:
    """Index of the token containing ``byte_offset`` of ``decode(token_ids)`` (audit helper)."""
    return token_at_byte(byte_spans(token_bytes(tokenizer, list(token_ids))), int(byte_offset))


# --------------------------------------------------------------------------- the panel


@dataclass(frozen=True)
class CapturePanel:
    """A sealed ``register:panel/<sha>`` (plan D16): which items and arms are captured.

    Selection happens **before outcomes** (spec 13.2), so the register holds ids only: source
    items, and optionally the condition hashes and rounds the panel covers.  ``None`` means
    "not narrowed", never "none".
    """

    sha: str
    source_item_ids: frozenset[str]
    protocol_hashes: frozenset[str] | None = None
    rounds: frozenset[int] | None = None

    def covers(self, row: Any) -> bool:
        if str(row.source_item_id) not in self.source_item_ids:
            return False
        if self.protocol_hashes is not None and str(row.protocol_hash) not in self.protocol_hashes:
            return False
        if self.rounds is not None and int(row.round) not in self.rounds:
            return False
        return True


def panel_payload(source_item_ids: Sequence[str], *, protocol_hashes: Sequence[str] = (),
                  rounds: Sequence[int] = ()) -> dict[str, Any]:
    """The body ``deps.write_register(run_root, "panel", ...)`` seals for a capture panel."""
    entries = sorted({str(i) for i in source_item_ids})
    if not entries:
        raise CaptureError("refusing to seal an empty capture panel")
    payload: dict[str, Any] = {"kind": deps.REGISTER_PANEL, "entries": entries}
    if protocol_hashes:
        payload["protocol_hashes"] = sorted({str(h) for h in protocol_hashes})
    if rounds:
        payload["rounds"] = sorted({int(r) for r in rounds})
    return payload


def load_panel(run_root: str | os.PathLike, sha: str) -> CapturePanel:
    """Read and verify ``register:panel/<sha>``; a tampered or absent register is unmet."""
    body = deps.read_register(run_root, deps.REGISTER_PANEL, sha)
    entries = body.get(deps.REGISTER_ENTRIES_FIELD) or ()
    items = frozenset(str(e.get("source_item_id") if isinstance(e, Mapping) else e) for e in entries)
    if not items:
        raise DependencyUnmet(f"register:panel/{sha} holds no source items")
    protocols = body.get("protocol_hashes")
    rounds = body.get("rounds")
    return CapturePanel(
        sha=str(sha), source_item_ids=items,
        protocol_hashes=None if protocols is None else frozenset(str(p) for p in protocols),
        rounds=None if rounds is None else frozenset(int(r) for r in rounds))


# --------------------------------------------------------------------------- the work list


@dataclass(frozen=True)
class CaptureWorkItem:
    """One physical sample to replay, and every decision that points at it.

    ``generation_uid`` is the unit of replay: a root reference and its bank draw -- and the
    same root under PRIVATE, PEER and every directed arm -- are *one* sample under several
    ``decision_id``s (plan D14; spec 13.2 "reuse root captures across protocols").  The
    vectors are stored once under the uid and the ledger emits one
    :class:`tables.ActivationRow` per decision, so no denominator is reconstructed outside the
    table.
    """

    generation_uid: str
    decision_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    source_item_id: str
    item_variant_id: str
    episode_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    cell_kind: str
    protocol_hash: str
    model_hash: str
    round_: int
    agent_slot: int
    is_reference: bool
    spans: AnswerSpans

    def __post_init__(self) -> None:
        if not self.generation_uid:
            raise CaptureError("a capture work item needs a generation_uid")
        if not self.decision_ids:
            raise CaptureError(f"{self.generation_uid[:14]}: no decision points at this sample")

    @property
    def order_key(self) -> tuple[str, int, int, str]:
        return (self.source_item_id, int(self.round_), int(self.agent_slot), self.generation_uid)


def _request_ids(row: Any) -> tuple[str, ...]:
    payload = json.loads(row.request_hashes_json or "[]")
    if not isinstance(payload, list):
        raise ManifestError(f"{row.decision_id}: request_hashes_json is not a list")
    return tuple(str(v) for v in payload)


def capture_manifest(decisions: Iterable[Any], panel: CapturePanel | None) -> tuple[CaptureWorkItem, ...]:
    """The frozen work list: the panel's decisions, one entry per distinct physical sample.

    ``decisions`` are :class:`tables.DecisionRow` objects read from the generation cells'
    item files (never a dict-key scan, R1).  Order is deterministic and correctness-blind --
    item, round, slot -- so the same panel produces the same list on every pass.
    """
    merged: dict[str, CaptureWorkItem] = {}
    for row in decisions:
        if panel is not None and not panel.covers(row):
            continue
        uid = str(row.generation_uid)
        if not uid:
            continue                    # a decision with no sample behind it: nothing to replay
        item = merged.get(uid)
        if item is None:
            merged[uid] = CaptureWorkItem(
                generation_uid=uid, decision_ids=(str(row.decision_id),),
                request_ids=_request_ids(row), source_item_id=str(row.source_item_id),
                item_variant_id=str(row.item_variant_id),
                episode_ids=(str(row.episode_id),), cell_ids=(str(row.cell_id),),
                cell_kind=str(row.cell_kind), protocol_hash=str(row.protocol_hash),
                model_hash=str(row.receiver_model_hash), round_=int(row.round),
                agent_slot=int(row.agent_slot), is_reference=bool(row.is_reference),
                spans=AnswerSpans.from_decision_row(row))
            continue
        if str(row.decision_id) in item.decision_ids:
            continue
        if AnswerSpans.from_decision_row(row) != item.spans:
            raise CaptureError(
                f"decisions {item.decision_ids[0]} and {row.decision_id} claim the same "
                f"generation_uid {uid[:14]} but record different answer spans; one physical "
                "sample has one public output (plan D14)")
        merged[uid] = replace(
            item,
            decision_ids=item.decision_ids + (str(row.decision_id),),
            episode_ids=tuple(dict.fromkeys(item.episode_ids + (str(row.episode_id),))),
            cell_ids=tuple(dict.fromkeys(item.cell_ids + (str(row.cell_id),))),
            is_reference=item.is_reference and bool(row.is_reference))
    return tuple(sorted(merged.values(), key=lambda w: w.order_key))


def decisions_of_cell(run_root: str | os.PathLike, cell_id: str, source_item_id: str) -> tuple[Any, ...]:
    """The ``decisions`` rows of one generation cell's item file (empty when it has none)."""
    path = T.item_path(run_root, cell_id, source_item_id)
    if not path.is_file():
        return ()
    return tuple(T.ItemFile.read(path).rows.get("decisions", ()))


def resolve_cell_work(run_root: str | os.PathLike, source_item_id: str,
                      options: Mapping[str, Any]) -> tuple[tuple[CaptureWorkItem, ...], CapturePanel | None]:
    """The work list for one CAPTURE cell item, refusing the "nothing was asked for" cases.

    Two empties look identical at the work-list level and must never be confused:

    * **nothing was asked for** -- a *manifest* defect: the cell options name no generation
      cells, or none of the named cells has an ``items/`` directory under this run root.  No
      amount of GPU time would have produced a vector, so this raises.
    * **this item has none** -- legitimate: the generation cells were read and this particular
      item simply contributes no panel decision.  The worker writes an empty item file.
    """
    cell_ids = tuple(str(c) for c in (options.get("cells") or ()))
    if not cell_ids:
        raise CaptureError(
            "CAPTURE cell options carry no 'cells': there is no decision work list to replay. "
            "The capture cell builder must emit the generation cell ids in options['cells'].")
    present = [c for c in cell_ids if (deps.cell_dir(run_root, c) / "items").is_dir()]
    if not present:
        raise CaptureError(
            f"none of the {len(cell_ids)} generation cells named in options['cells'] has an "
            f"items/ directory under {run_root} (first: {cell_ids[0]!r}); the cell ids or the "
            "run root are wrong, so this cell could never capture anything")
    panel_sha = options.get("panel")
    panel = load_panel(run_root, str(panel_sha)) if panel_sha else None
    rows: list[Any] = []
    for cell_id in present:
        rows.extend(decisions_of_cell(run_root, cell_id, source_item_id))
    return capture_manifest(rows, panel), panel


# --------------------------------------------------------------------------- completeness


def root_agent_coverage(items: Sequence[CaptureWorkItem], rows: Sequence[T.ActivationRow], *,
                        capture_all_root_agents: bool) -> dict[str, Any]:
    """Spec 13.2: every root member of the panel is captured **or explicitly missing**.

    ``study.instrumentation.capture_all_root_agents`` is the knob that makes this binding; the
    census it returns is the same shape either way, so the ledger always says how many root
    agents the panel had and how many of them have a row.
    """
    roots = [w for w in items if int(w.round_) == 0]
    expected = {d for w in roots for d in w.decision_ids}
    covered = {str(r.decision_id) for r in rows if str(r.decision_id) in expected}
    present = {str(r.decision_id) for r in rows
               if str(r.decision_id) in expected and r.missing_reason is None}
    missing_only = sorted(covered - present)
    uncovered = sorted(expected - covered)
    census = {
        "root_samples": len(roots),
        "root_decisions": len(expected),
        "root_decisions_with_rows": len(covered),
        "root_decisions_with_vectors": len(present),
        "root_decisions_missing_reason_only": missing_only,
        "root_decisions_without_any_row": uncovered,
        "capture_all_root_agents": bool(capture_all_root_agents),
    }
    if capture_all_root_agents and uncovered:
        raise CaptureError(
            f"{len(uncovered)} root-agent decision(s) of the panel have no activation row at "
            f"all (first: {uncovered[0]}); spec 13.2 requires every root member to be captured "
            "or explicitly missing with a reason")
    return census


def assert_capture_progress(stats: Mapping[str, Any], *, cell_id: str = "") -> Mapping[str, Any]:
    """Refuse a cell that captured nothing it was asked for.

    Zero captures is legitimate in exactly one case: a resume where every (sample, block,
    anchor) key of a non-empty work list is already committed.  Every other zero -- every
    record missing, every anchor unresolved, an engine that returned nothing -- is a defect
    that would otherwise be written out as a complete cell holding no vectors.
    """
    captured, work_items, pending = (int(stats.get(k, 0)) for k in ("captured", "work_items", "pending"))
    if captured > 0 or (work_items > 0 and pending == 0):
        return stats
    raise CaptureError(
        f"CAPTURE cell {cell_id or '?'} captured 0 of {pending} pending samples ({work_items} "
        f"work items; missing_records={stats.get('missing_records')}, "
        f"anchor_errors={stats.get('anchor_errors')}): a cell that captures nothing must fail, "
        "not look complete")


# --------------------------------------------------------------------------- staging


def staging_root(base: str | os.PathLike | None = None, *, tag: str = "nextdyn") -> Path:
    """A node-local staging directory for shard chunks (spec 13.2 replay engineering).

    ``base`` defaults to ``$NEXTDYN_LOCAL_SCRATCH``, then ``$TMPDIR``: the node-local disk of
    the replay job, never the shared filesystem and never a configured scratch path.
    """
    if base is None:
        base = os.environ.get("NEXTDYN_LOCAL_SCRATCH") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{tag}.capture.", dir=str(base)))


# --------------------------------------------------------------------------- the replay pass


@dataclass(frozen=True)
class CaptureConfig:
    """Everything :func:`run_capture` needs that is not the engine, tokenizer or store."""

    #: The model every sample in this pass was generated by.  ``run_capture`` refuses a work
    #: list that mixes models: one pass loads one set of weights, and a foreign-model draft
    #: replayed through them is not a root measurement of either (spec 13.2).
    model_hash: str
    shard_name: str
    anchors: tuple[str, ...] = ANCHORS
    chunk_rows: int = 128
    group_size: int = 8
    check_identity: bool = True
    #: Stored completion tokens scored by the spec-13.2 tolerance report (0 disables it).  The
    #: default is ``study.neural.replay``'s own, not a number invented here.
    fidelity_tokens: int = DEFAULT_FIDELITY_TOKENS

    def __post_init__(self) -> None:
        unknown = [a for a in self.anchors if a not in ANCHORS]
        if unknown:
            raise ContractViolation(f"study.instrumentation.anchors names {unknown}, not in {ANCHORS}")
        if not self.anchors:
            raise ContractViolation("study.instrumentation.anchors is empty")


def shard_name_for(cell_id: str) -> str:
    """A ``ShardWriter`` shard name for a cell id.

    ``storage.CHUNK_RE`` splits a chunk file on ``.c``, so a shard name may not contain one;
    every nextdyn capture cell id starts with ``capture.``.  Flattening the separators keeps
    the cell id readable and leaves exactly one ``.c`` in the chunk name -- the separator.
    """
    flat = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(cell_id))
    return "nextdyn_" + flat


def read_committed(run_root: str | os.PathLike, *, stage: str = CAPTURE_STAGE,
                   uids: Iterable[str] | None = None,
                   ) -> dict[tuple[str, int, str], tuple[Path, Mapping[str, Any]]]:
    """The committed ledger, keyed ``(generation_uid, block, anchor)``.

    It is both the resume filter and the source of the manifest rows an earlier pass already
    wrote.  Reading it is a whole-stage parse: the ``native`` stage is shared by every capture
    cell of the run, and the worker contract is *per item*, so a cell that replays sixteen
    items would otherwise re-parse the campaign's ledger sixteen times.  A caller that drives
    more than one item reads this **once** and passes it to :func:`run_capture` as
    ``committed``; ``uids`` narrows what is retained to the samples a pass can ask about, so
    the filter never holds the whole campaign in memory.
    """
    wanted = None if uids is None else {str(u) for u in uids}
    out: dict[tuple[str, int, str], tuple[Path, Mapping[str, Any]]] = {}
    for path, row in S.iter_rows(run_root, stage):
        uid = str(row["StateSnapshot_id"])
        if wanted is not None and uid not in wanted:
            continue
        out[(uid, int(row["block"]), str(row["anchor_kind"]))] = (path, row)
    return out


def run_capture(items: Sequence[CaptureWorkItem], *, engine: Any, tokenizer: Any, store: Any,
                run_root: str | os.PathLike, config: CaptureConfig,
                staging: str | os.PathLike | None = None,
                committed: Mapping[tuple[str, int, str], tuple[Path, Mapping[str, Any]]] | None = None,
                on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
                ) -> tuple[tuple[T.ActivationRow, ...], dict[str, Any]]:
    """Replay every pending sample teacher-forced and write its anchors x every block.

    The shape is ``study.neural.capture.run_native``'s: resume filter, one
    :class:`CaptureRequest` per sample over the **exact stored token ids**, forward the group,
    write rows.  Chunks are staged node-locally and consolidated into ``run_root`` with the
    ``.npz`` moved before its ``.jsonl`` (the commit marker), so a crash mid-consolidation
    leaves an orphan ``.npz`` -- which ``storage.next_chunk_index`` skips -- and never a chunk
    without vectors.  ``committed`` is a ledger map read once by the caller
    (:func:`read_committed`); when it is omitted this pass reads its own.

    A sample that cannot be replayed at all -- no physical request behind the decision, or a
    store that holds no record for it -- is a **typed ledger row with no vector**, not a cell
    failure: spec 13.2 says an anchor that does not exist is marked missing with a reason, and
    :func:`assert_capture_progress` is what refuses a cell that captured nothing.

    Returns ``(activations_manifest rows, stats)``.  Nothing in here can issue a generation
    request: there is no client, no pool and no sampler in this module's import closure.
    """
    run_root = Path(run_root)
    foreign = sorted({w.model_hash for w in items} - {config.model_hash})
    if foreign:
        # Spec 13.2: a native root activation and a teacher-forced replay of a *foreign*
        # model's draft are not interchangeable root measurements.  One pass loads one model,
        # so a work list carrying another model's samples is a manifest defect, not something
        # to average over.
        raise CaptureError(
            f"this capture pass replays {config.model_hash} but the work list also carries "
            f"{foreign}; a foreign-model draft is replayed in its own cell (spec 13.2)")
    blocks = [int(b) for b in engine.blocks]
    stored = (dict(committed) if committed is not None
              else read_committed(run_root, uids={w.generation_uid for w in items}))
    already = set(stored)
    pending = [w for w in items
               if not all((w.generation_uid, b, a) in already
                          for b in blocks for a in config.anchors)]
    owned = staging is None
    staging_path = Path(staging) if staging is not None else staging_root()
    stats: dict[str, Any] = {
        "work_items": len(items), "pending": len(pending), "captured": 0, "rows": 0,
        "tokens": 0, "missing_records": 0, "context_failures": 0, "anchor_errors": 0,
        "rows_skipped_committed": 0, "anchors_missing": {}, "blocks": blocks,
        "anchors": list(config.anchors), "shard_name": config.shard_name,
        "staging": str(staging_path),
    }
    rows: list[T.ActivationRow] = []
    try:
        with S.ShardWriter(staging_path, CAPTURE_STAGE, config.shard_name,
                           chunk_rows=config.chunk_rows) as writer:
            for group in _chunks(pending, config.group_size):
                requests: list[CaptureRequest] = []
                meta: list[tuple[CaptureWorkItem, AnchorSet, str, Mapping[str, Any] | None]] = []
                for work in group:
                    source = _records_for(work, store)
                    if source.record is None:
                        reason = source.missing_reason or MISSING_RECORD
                        key = "context_failures" if reason == MISSING_CONTEXT_FAILURE else "missing_records"
                        stats[key] += 1
                        rows.extend(_unresolved_rows(work, reason, blocks, config, stats))
                        continue
                    anchors = resolve_anchors(source.record, work.spans, tokenizer,
                                              decision_id=work.decision_ids[0],
                                              prompt_tokens=source.prompt_tokens)
                    if any(a.missing_reason == MISSING_ANCHOR_UNRESOLVED for a in anchors.anchors):
                        stats["anchor_errors"] += 1
                    if not anchors.present_offsets:
                        rows.extend(_rows_for(work, anchors, None, blocks, config, stats,
                                              parity=PARITY_MISSING, writer=writer,
                                              stored=stored, engine=engine))
                        continue
                    parity = (_check_identity(source, tokenizer, work) if config.check_identity
                              else PARITY_UNCHECKED)
                    record_prompt, record_completion, _ = _record_fields(source.record)
                    needed = anchors.tokens_needed(len(record_prompt))
                    if needed > len(record_completion):        # pragma: no cover - defensive
                        raise CaptureError(
                            f"{work.generation_uid[:14]}: an anchor sits past the stored token "
                            f"ids ({needed} > {len(record_completion)} completion tokens)")
                    report = None
                    if config.fidelity_tokens > 0 and "fidelity" not in stats:
                        report = _fidelity_report(source.record, engine, tokenizer,
                                                  n_tokens=config.fidelity_tokens)
                        stats["fidelity"] = report
                    sequence = teacher_forced_sequence(source.record, needed)
                    requests.append(CaptureRequest(work.generation_uid, tuple(sequence),
                                                   anchors.present_offsets))
                    meta.append((work, anchors, parity, report))
                if not requests:
                    continue
                results = {r.seq_id: r for r in engine.forward(requests)}
                for work, anchors, parity, report in meta:
                    result = results[work.generation_uid]
                    rows.extend(_rows_for(work, anchors, result, blocks, config, stats,
                                          parity=parity, writer=writer, stored=stored,
                                          engine=engine, fidelity_report=report))
                    stats["captured"] += 1
                    stats["tokens"] += int(result.n_tokens)
                writer.flush()
                if on_event is not None:
                    on_event("group_done", {"captured": stats["captured"], "rows": stats["rows"]})
        S.atomic_write_json(S.stage_dir(staging_path, CAPTURE_STAGE) / f"stats.{config.shard_name}.json", stats)
        stats["consolidated"] = [str(p) for p in consolidate_shards(staging_path, run_root, stage=CAPTURE_STAGE)]
    finally:
        if owned:
            shutil.rmtree(staging_path, ignore_errors=True)
    rows.extend(_resumed_rows(items, pending, blocks, config, run_root, stored, stats))
    return tuple(rows), stats


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def _fidelity_report(record: Any, engine: Any, tokenizer: Any, *, n_tokens: int) -> dict[str, Any]:
    """The spec-13.2 numbers: "log tolerances and numerical differences rather than demand
    universally bit-identical GPU kernels".

    ``study.neural.replay.fidelity`` teacher-forces the same prefix twice and reports argmax
    agreement with the stored tokens, their mean log-prob and the two-pass logit discrepancy;
    the plan's S2 pilot gate ("capture parity with the prefill span excluded") reads them off
    ``stats['fidelity']`` and off the ledger row's ``extra``.  A report that cannot be produced
    says so in the same place -- it never fails the capture and never silently disappears.
    """
    try:
        report = fidelity(record, engine, int(n_tokens), tokenizer=tokenizer)
    except Exception as exc:                        # noqa: BLE001 - a tolerance number is not a gate
        return {"error": f"{type(exc).__name__}: {exc}"[:200], "hook_convention": HOOK_CONVENTION}
    return {
        "request_id": report["request_id"],
        "scored_tokens": report["scored_tokens"],
        "argmax_agreement": report["argmax_agreement"],
        "mean_logprob": report["mean_logprob"],
        "min_logprob": report["min_logprob"],
        "max_abs_logit_diff": report["two_pass_max_abs_logit_diff"],
        "mean_abs_logit_diff": report["two_pass_mean_abs_logit_diff"],
        "two_pass_argmax_agreement": report["two_pass_argmax_agreement"],
        "identity": report.get("identity"),
        "hook_convention": HOOK_CONVENTION,
    }


def _check_identity(source: "ReplaySource", tokenizer: Any, work: CaptureWorkItem) -> str:
    """``PARITY_OK`` once the decision's own stored prompt reproduces under the tokenizer.

    The check runs against the decision's **own** record -- phase 1 -- never the replayed one.
    A continued draw's phase-2 record carries no messages at all (``ContinuationSpec`` refuses
    them: its prompt is an exact token prefix, phase 1's prompt + the thinking span + the
    frozen prefill, which no chat render reproduces), so checking that record would fail every
    continued decision -- roughly a third of them at ARD36's measured boundary rate.  The
    phase-2 evidence is the prefix relation :func:`_records_for` proves; both hold before this
    returns ``PARITY_OK``.  ``ReplayError`` is wrapped so a verb never surfaces a foreign
    exception type.
    """
    try:
        check_prompt_identity(source.own_record, tokenizer, strict=True)
    except (ReplayError, AnchorError, TypeError, ValueError, KeyError, IndexError) as exc:
        raise CaptureError(
            f"{work.generation_uid[:14]}: the stored prompt of request {work.request_ids[:1]} "
            f"does not reproduce under the pinned tokenizer ({type(exc).__name__}: {exc})"
        ) from exc
    return PARITY_OK


@dataclass(frozen=True)
class ReplaySource:
    """What one work item replays: the record to forward, and the decision's own record.

    They differ exactly when the decision was executed as two physical requests (amendment
    A3): ``record`` is phase 2 -- whose stored prompt already holds phase 1's prompt, thinking
    span and the frozen prefill, so replaying it reproduces the decision's whole sequence --
    while ``own_record`` is phase 1, the only one whose prompt is a chat render and can
    therefore be identity-checked.  ``missing_reason`` is set instead of a record when there is
    nothing to replay.
    """

    record: Any | None
    own_record: Any | None
    prompt_tokens: int | None
    continued: bool
    missing_reason: str | None = None


def _records_for(work: CaptureWorkItem, store: Any) -> ReplaySource:
    """The :class:`ReplaySource` of one sample.

    A naturally closed draw is one physical request and replays as itself.  A continued draw
    is two: phase 2's stored prompt is phase 1's prompt + the thinking span + the frozen
    prefill, so replaying phase 2's record reproduces the decision's whole sequence, and
    ``PROMPT_END`` is pinned back to phase 1's prompt length -- the last prompt token before
    *this call's* generation, not the last prefill token.  The composition is proved here (the
    phase-2 prompt must start with phase 1's), which is the evidence the identity check on
    phase 1 is paired with.
    """
    if not work.request_ids:
        # a decision with no physical request behind it (a build-time context refusal): typed
        # missingness, never a cell failure
        return ReplaySource(None, None, None, False, MISSING_CONTEXT_FAILURE)
    records = [store.get(rid) for rid in work.request_ids]
    if any(record is None for record in records):
        return ReplaySource(None, None, None, False, MISSING_RECORD)
    first, last = records[0], records[-1]
    if len(records) == 1:
        return ReplaySource(last, first, None, False)
    first_prompt, _, _ = _record_fields(first)
    last_prompt, _, _ = _record_fields(last)
    if last_prompt[: len(first_prompt)] != first_prompt:
        raise CaptureError(
            f"{work.generation_uid[:14]}: the continuation prompt does not start with the "
            "decision's own prompt; the stored ids of the two phases do not compose")
    return ReplaySource(last, first, len(first_prompt), True)


def _rows_for(work: CaptureWorkItem, anchors: AnchorSet, result: Any, blocks: Sequence[int],
              config: CaptureConfig, stats: dict[str, Any], *, parity: str, writer: Any,
              stored: Mapping[tuple[str, int, str], tuple[Path, Mapping[str, Any]]],
              engine: Any, fidelity_report: Mapping[str, Any] | None = None,
              ) -> list[T.ActivationRow]:
    out: list[T.ActivationRow] = []
    for anchor in anchors.anchors:
        if anchor.name not in config.anchors:
            continue
        if not anchor.present:
            stats["anchors_missing"][anchor.missing_reason] = (
                stats["anchors_missing"].get(anchor.missing_reason, 0) + 1)
        for block in blocks:
            key = (work.generation_uid, int(block), anchor.name)
            if key in stored:
                # an earlier pass wrote this vector; the row still has to reach the ledger
                stats["rows_skipped_committed"] += 1
                out.extend(_rows_from_stored(work, anchor.name, int(block), stored[key]))
                continue
            vector = (result.residual_at(int(block), anchor.token_offset)
                      if (anchor.present and result is not None) else None)
            # the digest is computed here, not read back off the writer: the manifest row and
            # the stored vector must carry the *same* hash or the join is unverifiable, and
            # ``ShardWriter.add`` refuses a row whose tensor_hash disagrees with its vector
            digest = None if vector is None else S.tensor_hash(
                np.asarray(vector, dtype=np.float16).reshape(-1))
            storage_row = _storage_row(work, anchors, anchor, int(block), result, engine,
                                       tensor_hash=digest, fidelity_report=fidelity_report)
            writer.add(storage_row, vector)
            stats["rows"] += 1
            out.extend(_manifest_rows(work, anchor, int(block), digest, vector,
                                      parity=parity if vector is not None else PARITY_MISSING))
    return out


def _unresolved_rows(work: CaptureWorkItem, reason: str, blocks: Sequence[int],
                     config: CaptureConfig, stats: dict[str, Any]) -> list[T.ActivationRow]:
    """Ledger rows for a sample that could not be replayed at all (no store record)."""
    out: list[T.ActivationRow] = []
    for name in config.anchors:
        stats["anchors_missing"][reason] = stats["anchors_missing"].get(reason, 0) + 1
        anchor = _missing(name, reason, work.spans)
        for block in blocks:
            out.extend(_manifest_rows(work, anchor, int(block), None, None, parity=PARITY_MISSING))
    return out


def _resumed_rows(items: Sequence[CaptureWorkItem], pending: Sequence[CaptureWorkItem],
                  blocks: Sequence[int], config: CaptureConfig, run_root: Path,
                  stored: Mapping[tuple[str, int, str], tuple[Path, Mapping[str, Any]]],
                  stats: dict[str, Any]) -> list[T.ActivationRow]:
    """Manifest rows for samples every one of whose keys an earlier pass already committed.

    The ledger must account for every work item on a resume too, or a cell that resumes looks
    like a cell that captured nothing.
    """
    done = {w.generation_uid for w in pending}
    out: list[T.ActivationRow] = []
    for work in (w for w in items if w.generation_uid not in done):
        for name in config.anchors:
            for block in blocks:
                found = stored.get((work.generation_uid, int(block), name))
                if found is None:                              # pragma: no cover - defensive
                    continue
                out.extend(_rows_from_stored(work, name, int(block), found))
        stats["rows_skipped_committed"] += len(config.anchors) * len(blocks)
    return out


def _rows_from_stored(work: CaptureWorkItem, name: str, block: int,
                      found: tuple[Path, Mapping[str, Any]]) -> list[T.ActivationRow]:
    """Manifest rows built from a ledger line an earlier pass committed.

    Nothing is recomputed and no vector is re-read: the row points at the same stage
    directory a fresh row does and carries the same ``storage_hash``, which is what the
    analysis joins on (a chunk index is not stable -- consolidation renumbers chunks).
    """
    _path, row = found
    missing = row.get("missingness")
    return [
        T.ActivationRow(
            decision_id=decision_id, layer=int(block), anchor=name, model_hash=work.model_hash,
            token_offset=None if row.get("token_offset") is None else int(row["token_offset"]),
            dtype=None if missing else VECTOR_DTYPE,
            shape_json=None if missing else json.dumps([int(row.get("hidden_size") or 0)]),
            storage_hash=row.get("tensor_hash"),
            storage_path=None if missing else f"{S.NEURAL_DIR}/{CAPTURE_STAGE}",
            replay_parity=PARITY_MISSING if missing else PARITY_RESUMED,
            missing_reason=missing,
            pre_confidence_eligible=(work.spans.answer_end_pre_confidence_eligible
                                     if name == ANCHOR_ANSWER_END else None))
        for decision_id in work.decision_ids
    ]


def _storage_row(work: CaptureWorkItem, anchors: AnchorSet, anchor: Anchor, block: int,
                 result: Any, engine: Any, *, tensor_hash: str | None = None,
                 fidelity_report: Mapping[str, Any] | None = None) -> S.ActivationRow:
    """The ``study.neural.storage`` ledger line that travels beside the vector."""
    return S.ActivationRow(
        StateSnapshot_id=work.generation_uid,
        consumer_revision=work.model_hash,
        condition=f"nd:{work.protocol_hash[:12]}:r{int(work.round_)}",
        child_slot=int(work.agent_slot),
        nonce_hash=ids.content_hash(work.generation_uid.encode("utf-8")),
        block=int(block),
        anchor_kind=anchor.name,
        structural_span="prompt" if anchor.name == ANCHOR_PROMPT_END else "public_json",
        token_offset=anchor.token_offset,
        channel="prompt" if anchor.name == ANCHOR_PROMPT_END else ("none" if not anchor.present else "content"),
        generated_token_count=(0 if not anchor.present
                               else max(0, anchor.token_offset - anchors.prompt_tokens + 1)),
        missingness=anchor.missing_reason,
        tensor_hash=tensor_hash,
        measurement_cost=(result.measurement_cost if result is not None
                          else {"sequence_tokens": 0, "share_seconds": 0.0}),
        operationally_available_at_checkpoint=True,
        stage=CAPTURE_STAGE,
        sequence_id=work.generation_uid,
        source_id=work.source_item_id,
        method=work.protocol_hash,
        role=f"slot{int(work.agent_slot)}",
        phase=f"round{int(work.round_)}",
        cell_id=work.cell_ids[0] if work.cell_ids else None,
        episode_id=work.episode_ids[0] if work.episode_ids else None,
        checkpoint=work.model_hash,
        hidden_size=int(engine.hidden_size),
        sequence_tokens=None if result is None else int(result.n_tokens),
        hook_convention=HOOK_CONVENTION,
        extra={
            "decision_ids": list(work.decision_ids),
            "item_variant_id": work.item_variant_id,
            "cell_kind": work.cell_kind,
            "is_reference": bool(work.is_reference),
            "byte_offset": anchor.byte_offset,
            "anchor_detail": dict(anchor.detail),
            "parse_status": anchors.parse_status,
            "public_span": None if anchors.public_span is None else list(anchors.public_span),
            "public_text_offset": anchors.public_text_offset,
            "origin_proof": anchors.origin_proof,
            "pre_confidence_eligible": anchor.pre_confidence_eligible,
            # spec 13.2's logged tolerances travel with the vectors, so a gate reads them off
            # the ledger rather than off a log line (null on every row but the sampled one)
            "fidelity": None if fidelity_report is None else dict(fidelity_report),
        },
        captured_at=time.time(),
    )


def _manifest_rows(work: CaptureWorkItem, anchor: Anchor, block: int,
                   tensor_hash: str | None, vector: Any, *,
                   parity: str) -> list[T.ActivationRow]:
    """One ``activations_manifest`` row per decision that points at this sample."""
    shape = None
    if vector is not None:
        shape = json.dumps([int(np.asarray(vector).reshape(-1).size)])
    return [
        T.ActivationRow(
            decision_id=decision_id, layer=int(block), anchor=anchor.name,
            model_hash=work.model_hash, token_offset=anchor.token_offset,
            dtype=None if vector is None else VECTOR_DTYPE,
            shape_json=shape,
            storage_hash=tensor_hash,
            storage_path=None if vector is None else f"{S.NEURAL_DIR}/{CAPTURE_STAGE}",
            replay_parity=parity,
            missing_reason=anchor.missing_reason,
            pre_confidence_eligible=anchor.pre_confidence_eligible)
        for decision_id in work.decision_ids
    ]


# --------------------------------------------------------------------------- geometry


def team_geometry(states: Any, *, normalised: bool = False, standardizer: Any = None) -> dict[str, Any]:
    """Spec 13.3 team statistics for one ``s x d`` set of comparable states.

    A thin pass-through to ``ard.neural.geometry.team_metrics`` (allowlisted): participation
    ratio, entropy effective rank, leading eigenvalue share, total variance, raw
    min/mean/max distance and cosine distance, with the ``rank_ceiling = min(d, s-1)``
    ceiling spec 13.3 demands.  Nothing here re-derives an estimator.
    """
    return team_metrics(np.asarray(states, dtype=np.float64), standardizer=standardizer,
                        normalised=normalised)


def agent_geometry(states: Any, *, previous: Any = None, agent_ids: Sequence[int] | None = None
                   ) -> list[dict[str, Any]]:
    """Spec 13.3 per-agent statistics (distances, displacement, displacement alignment).

    ``ard.neural.geometry.agent_metrics`` verbatim: near-zero denominators are *recorded* and
    the alignment is NaN, never a division by a near-zero number.
    """
    return agent_metrics(np.asarray(states, dtype=np.float64),
                         previous=None if previous is None else np.asarray(previous, dtype=np.float64),
                         agent_ids=agent_ids)


def spectral_ratios_defined(metrics: Mapping[str, Any]) -> bool:
    """Spec 13.3: with zero total variance the spectral ratios are **undefined**.

    The one reader of that fact.  ``total_variance`` and every ratio come back as NaN from
    :func:`team_geometry` when the centered covariance has no nonzero eigenvalue (identical
    states at a byte-identical root prompt is the expected case, spec 13.2), and this returns
    ``False`` there instead of letting a downstream zero be read as a one-dimensional
    collapse.
    """
    total = float(metrics.get("total_variance", float("nan")))
    if not np.isfinite(total) or total <= 0.0:
        return False
    return all(np.isfinite(float(metrics.get(name, float("nan")))) for name in SPECTRAL_RATIOS)


# --------------------------------------------------------------------------- engine factory


def default_engine_factory(options: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(engine, tokenizer)`` from a pinned offline HF snapshot (``local_files_only``).

    A bare ``KeyError`` here would read as a code bug; it is a *manifest* bug, and the message
    says which builder owes the option: the snapshot is pinned in the frozen manifest, never
    resolved from the environment at replay time.
    """
    from transformers import AutoTokenizer

    from agents_scaling.study.neural.engine import CaptureEngine

    snapshot = options.get("snapshot_path")
    if not snapshot:
        raise CaptureError(
            "CAPTURE cell options carry no 'snapshot_path': the capture cell builder must pin "
            "the offline HF snapshot directory of the replayed model into the frozen manifest")
    snapshot = Path(snapshot)
    if not (snapshot / "config.json").is_file():
        raise CaptureError(f"snapshot {snapshot} is not an offline HF snapshot directory")
    blocks = options.get("blocks")
    if not blocks:
        raise CaptureError("CAPTURE cell options carry no 'blocks'; the worker derives them "
                           "from the pinned layer count and must pass them here")
    engine = CaptureEngine(snapshot, tuple(int(b) for b in blocks),
                           device_map=options.get("device_map", "auto"),
                           max_batch_tokens=int(options.get("max_batch_tokens", 16384)),
                           mask_policy=str(options.get("mask_policy", "causal")))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return engine, tokenizer


#: Injection point: a test or a degraded path replaces this without touching the worker.
ENGINE_FACTORY: Callable[[Mapping[str, Any]], tuple[Any, Any]] = default_engine_factory


# --------------------------------------------------------------------------- the cell worker


def capture_worker(runner: Any, source_item_id: str) -> T.ItemFile:
    """The CAPTURE cell worker (``kinds.KIND_TABLE``); the runner writes what it returns.

    The runner contract this worker reads (WP4 supplies it, WP6 states it):

    ``runner.cell``              the :class:`cells.CellSpec` (``cell_id``, ``options``)
    ``runner.run_root``          the run root holding the item files and the request store
    ``runner.cfg``               the compiled :class:`config.StudyConfig`
    ``runner.store``             (optional) the request store; else ``<run_root>/requests``
    ``runner.capture_engine()``  (optional) ``(engine, tokenizer)``; else ``ENGINE_FACTORY``
    ``runner.models_lock``       (optional) the loaded ``models.lock.json``

    Cell options: ``cells`` (the generation cell ids whose item files hold the decisions),
    ``panel`` (the ``register:panel/<sha>`` this cell captures), ``model`` (the model cell
    name whose layer count sets the blocks), ``snapshot_path``, and optionally ``blocks``,
    ``anchors``, ``staging``, ``device_map``, ``max_batch_tokens``, ``fidelity_tokens``.

    The blocks are always **derived** from the pinned layer count and
    ``study.instrumentation.relative_layers``; an ``options['blocks']`` literal and the engine
    the runner hands over are both checked against them rather than trusted, because on the
    production path the runner supplies the engine and ``engine.blocks`` would otherwise be
    the only thing that decided where the hooks sit.
    """
    cell = runner.cell
    run_root = Path(runner.run_root)
    cfg = runner.cfg
    options = dict(cell.options or {})
    instrumentation = cfg.instrumentation

    items, _panel = resolve_cell_work(run_root, source_item_id, options)
    if not items:
        return T.ItemFile(schema_version=T.SCHEMA_VERSION, cell_id=cell.cell_id,
                          cell_kind=CellKind.CAPTURE, source_item_id=source_item_id,
                          status=T.ItemStatus.COMPLETE, bundle_export_hash=None,
                          rows={"activations_manifest": ()})

    blocks = _blocks_from_lock(runner, options, instrumentation.relative_layers)
    declared = options.get("blocks")
    if declared is not None and tuple(int(b) for b in declared) != blocks:
        # spec 13.2's rule is floor(f*L)-1 at the *pinned* model's own layer count; an
        # options literal that disagrees would describe a different quarter of the network
        raise ContractViolation(
            f"cell options name blocks {tuple(int(b) for b in declared)} but "
            f"{instrumentation.relative_layers} at this model's pinned layer count give "
            f"{blocks} (spec 13.2 floor(f*L)-1)")
    engine, tokenizer = (runner.capture_engine() if hasattr(runner, "capture_engine")
                         else ENGINE_FACTORY({**options, "blocks": tuple(blocks)}))
    if tuple(int(b) for b in engine.blocks) != blocks:
        # the runner supplies the engine on the production path, so the derived blocks are
        # only binding if the engine is held to them -- otherwise engine.blocks silently wins
        raise ContractViolation(
            f"the capture engine hooks blocks {tuple(int(b) for b in engine.blocks)} but this "
            f"cell's pinned layer count and {instrumentation.relative_layers} give {blocks}")
    store = getattr(runner, "store", None) or _default_store(run_root)
    anchors = tuple(options.get("anchors") or instrumentation.anchors)
    config = CaptureConfig(
        model_hash=str(options.get("model_hash") or items[0].model_hash),
        shard_name=str(options.get("shard_name") or shard_name_for(cell.cell_id)),
        anchors=anchors,
        chunk_rows=int(options.get("chunk_rows", 128)),
        group_size=int(options.get("group_size", 8)),
        check_identity=bool(options.get("check_identity", True)),
        fidelity_tokens=int(options.get("fidelity_tokens", DEFAULT_FIDELITY_TOKENS)))
    rows, stats = run_capture(items, engine=engine, tokenizer=tokenizer, store=store,
                              run_root=run_root, config=config, staging=options.get("staging"))
    assert_capture_progress(stats, cell_id=str(cell.cell_id))
    census = root_agent_coverage(items, rows,
                                 capture_all_root_agents=instrumentation.capture_all_root_agents)
    if hasattr(runner, "event"):
        runner.event("captured", **{k: stats[k] for k in ("captured", "rows", "tokens")},
                     root_decisions=census["root_decisions"])
    return T.ItemFile(schema_version=T.SCHEMA_VERSION, cell_id=cell.cell_id,
                      cell_kind=CellKind.CAPTURE, source_item_id=source_item_id,
                      status=T.ItemStatus.COMPLETE, bundle_export_hash=None,
                      rows={"activations_manifest": tuple(rows)})


def _default_store(run_root: Path) -> Any:
    from agents_scaling.study.inference.store import RequestStore

    return RequestStore(run_root / "requests")


def _blocks_from_lock(runner: Any, options: Mapping[str, Any],
                      relative_layers: Sequence[float]) -> tuple[int, ...]:
    """Blocks of the replayed model: its ``models.lock.json`` depth, else the snapshot's.

    The one derivation: :func:`capture_worker` checks both the cell options and the engine
    against what this returns, so no capture ever hooks a layer nobody derived.
    """
    lock = getattr(runner, "models_lock", None)
    entry = None
    name = options.get("model")
    if lock is not None and name:
        entry = lock.models.get(str(name))
    return blocks_for_model(entry, relative_layers, snapshot_path=options.get("snapshot_path"))


# --------------------------------------------------------------------------- the verb


def main(argv: Sequence[str]) -> cli.Report:
    """``nextdyn capture`` -- report the capture ledger of a run root (spec 17.5).

    The heavy replay is a CAPTURE *cell* on the capture lane, driven by the runner; this verb
    is the close-out reader: it counts the committed rows of the ``native`` stage, separates
    present vectors from typed missing anchors, and never touches a GPU.
    """
    parser = cli.common_parser("capture", description="activation capture ledger")
    parser.add_argument("--stage", default=CAPTURE_STAGE, help="storage stage to report")
    args = parser.parse_args(list(argv))
    try:
        run_root = Path(args.run_root) if args.run_root is not None else cli.resolve_run_root(args)
    except Exception as exc:                                   # noqa: BLE001 - a verb never tracebacks
        return cli.Report(blocked=1, detail=f"cannot resolve the run root: {exc}")
    try:
        stage_dir = S.stage_dir(run_root, str(args.stage))
    except ValueError as exc:
        return cli.Report(blocked=1, detail=str(exc))
    if not stage_dir.is_dir():
        return cli.Report(blocked=1, detail=f"no capture stage at {stage_dir}: run the CAPTURE cells first")
    try:
        rows = [row for _, row in S.iter_rows(run_root, str(args.stage))]
    except Exception as exc:                                   # noqa: BLE001
        return cli.Report(failed=1, detail=f"capture stage unreadable: {exc}")
    present = sum(1 for r in rows if r.get("missingness") is None)
    missing = len(rows) - present
    reasons = sorted({str(r.get("missingness")) for r in rows if r.get("missingness")})
    return cli.Report(completed=present, skipped=missing,
                      detail=f"{len(rows)} activation rows; missing reasons: {reasons or 'none'}")


__all__ = [
    "ANCHORS", "ANCHOR_ANSWER_END", "ANCHOR_PRE_FINAL", "ANCHOR_PROMPT_END", "Anchor",
    "AnchorSet", "AnswerSpans", "CAPTURE_STAGE", "CaptureConfig", "CaptureError",
    "CapturePanel", "CaptureWorkItem", "ENGINE_FACTORY", "MISSING_ANCHOR_UNRESOLVED",
    "MISSING_CONTEXT_FAILURE", "MISSING_INVALID_ANSWER", "MISSING_NO_PRECEDING_TOKEN",
    "MISSING_NO_PUBLIC_SPAN", "MISSING_RECORD", "MISSING_REASONS", "MISSING_SPAN_TRUNCATED",
    "PARITY_MISSING", "PARITY_OK", "PARITY_RESUMED", "PARITY_UNCHECKED", "SPECTRAL_RATIOS",
    "VECTOR_DTYPE", "agent_geometry", "assert_capture_progress", "blocks_for_model",
    "capture_blocks", "capture_manifest", "capture_worker", "decisions_of_cell",
    "default_engine_factory", "load_panel", "main", "model_layer_count", "panel_payload",
    "read_committed", "resolve_anchors", "resolve_cell_work", "root_agent_coverage",
    "run_capture",
    "shard_name_for", "spectral_ratios_defined", "staging_root", "team_geometry",
    "token_offset_of_byte",
]
