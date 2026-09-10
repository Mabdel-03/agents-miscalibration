"""ARD36 neural anchors: the four capture sites of one decision (WP6).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §14.2
("Structural anchors") and plan §5.6/§5.8.  The spec's four sites, in
``contract.CAPTURE_ANCHORS`` order:

1. ``PROMPT_END``  — last prompt token, after the packet but before generated reasoning.
   This is exactly study_v4's ``NATIVE_PREFILL`` anchor at ``prompt_len - 1`` and is
   reused as such (plan §5.6: "anchor 1 = existing ``NATIVE_PREFILL``").
2. ``PRE_FINAL``   — last token immediately before ``<final>`` answer content.
3. ``ANSWER_END``  — last answer-content token, before ``<confidence>`` content.
4. ``REPORT_END``  — end of the final report: the last token of the closing
   ``</confidence>`` tag, or the last token of the ``</final>`` closing tag when the
   confidence tag is missing (plan §5.8 "anchor 4 = last token of the report span").

Two rules from the spec are mechanical here:

* **"Save exact token offsets and marker parse validity. No anchor is guessed from a fixed
  128-token offset."**  Every offset is derived from the tokenizer's own byte spans
  (:func:`~agents_scaling.study.neural.anchors.token_bytes` /
  :func:`~agents_scaling.study.neural.anchors.byte_spans` /
  :func:`~agents_scaling.study.neural.anchors.last_token_of_span`, reused verbatim — they
  assert ``b"".join(pieces) == tokenizer.decode(ids)``).  An anchor that cannot be derived
  is stored as a *typed missing* anchor (:data:`ARD_MISSINGNESS_CODES`); it is never
  approximated, and downstream (``ard.neural.capture``) a missing anchor becomes a ledger
  row with **no vector** — never a zero vector.
* **Validity is decided by the frozen report parser, not here.**  Tag statuses come from
  :func:`agents_scaling.ard.report.parse_report` (WP2, spec §6.3); this module only maps
  the spans that parser accepted onto token offsets, and refuses (``AnchorError``) if its
  own span scan disagrees with the parser's per-tag status.  That keeps one definition of
  "a valid ``<final>`` pair" in the run.

The report span analysed here is the same span the parser saw: the completion ids after
the first ``</think>``, minus a trailing ``<|im_end|>``, capped at
``contract.FINAL_TOKENS`` — identical to ``ard.client.decision_from_record``.

Timing labels (plan §5.8, spec §14.4) also live here because they are anchor semantics:
:func:`feature_time_of` and :func:`last_allowed_round`.  The enforced rule is

* ``pre_answer``  (``PROMPT_END``, ``PRE_FINAL`` = ``contract.PROSPECTIVE_ANCHORS``) at
  round *t*: the state precedes round *t*'s answer, so ``last_allowed_round = t - 1`` and
  every target at round ``>= t`` — including the upcoming answer's identity — is
  prospective for it.
* ``post_answer`` (``ANSWER_END``, ``REPORT_END``) at round *t*: the state already
  contains round *t*'s answer, so ``last_allowed_round = t``; the only permitted target at
  or below that round is **that answer's correctness**, and otherwise the target must be a
  round ``t+1`` (or later) quantity.  ``ard.neural.probes`` enforces this.

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import report as R
from agents_scaling.study.neural.anchors import (
    AnchorError,
    byte_spans,
    last_token_of_span,
    token_bytes,
)

__all__ = [
    "ANCHOR_ANSWER_END",
    "ANCHOR_PRE_FINAL",
    "ANCHOR_PROMPT_END",
    "ANCHOR_REPORT_END",
    "ARD_MISSINGNESS_CODES",
    "AnchorError",
    "AnchorSet",
    "CHANNEL_CONTENT",
    "CHANNEL_NONE",
    "CHANNEL_PROMPT",
    "FEATURE_TIME_POST",
    "FEATURE_TIME_PRE",
    "MISSING_ANCHOR_UNRESOLVED",
    "MISSING_CONTEXT_FAILURE",
    "MISSING_FINAL_EMPTY",
    "MISSING_FINAL_MALFORMED",
    "MISSING_FINAL_TAG",
    "MISSING_NOT_REACHED",
    "MISSING_NO_REPORT_SPAN",
    "MISSING_SPAN_TRUNCATED",
    "REPORT_END_FINAL_CLOSE",
    "REPORT_END_CONFIDENCE_CLOSE",
    "TagAnchor",
    "assert_four_anchors",
    "entropy_anchor_offsets",
    "feature_time_of",
    "is_entropy_anchor",
    "k_max_needed",
    "last_allowed_round",
    "report_span",
    "resolve_anchors",
    "resolve_decision_anchors",
    "tag_span",
]

# --------------------------------------------------------------------------- names

ANCHOR_PROMPT_END, ANCHOR_PRE_FINAL, ANCHOR_ANSWER_END, ANCHOR_REPORT_END = K.CAPTURE_ANCHORS

CHANNEL_PROMPT = "prompt"
CHANNEL_CONTENT = "content"
CHANNEL_NONE = "none"

FEATURE_TIME_PRE = "pre_answer"
FEATURE_TIME_POST = "post_answer"
FEATURE_TIMES: tuple[str, ...] = (FEATURE_TIME_PRE, FEATURE_TIME_POST)

#: Where ``REPORT_END`` was taken from (plan §5.8: the two permitted sources).
REPORT_END_CONFIDENCE_CLOSE = "confidence_close"
REPORT_END_FINAL_CLOSE = "final_close"
REPORT_END_NONE = "none"

# --------------------------------------------------------------------------- missingness

#: The completion never produced the token the anchor names (reuses the study_v4 code).
MISSING_NOT_REACHED = "NOT_REACHED"
#: The byte→token mapping itself failed (non-prefix-monotone decode); reuses study_v4.
MISSING_ANCHOR_UNRESOLVED = "ANCHOR_UNRESOLVED"
#: ``<think>`` never opened or never closed, so there is no report span at all.
MISSING_NO_REPORT_SPAN = "NO_REPORT_SPAN"
#: The report span exists but carries no ``<final>`` pair.
MISSING_FINAL_TAG = "FINAL_TAG_MISSING"
#: Duplicate, nested, or out-of-order ``<final>`` delimiters (spec §6.3 malformed).
MISSING_FINAL_MALFORMED = "FINAL_TAG_MALFORMED"
#: ``<final></final>``: there is no answer-content token to anchor on.
MISSING_FINAL_EMPTY = "FINAL_CONTENT_EMPTY"
#: The report span was cut (``finish_reason == "length"`` or the 256-token overrun), so the
#: closing delimiter is not in the stored ids.  Preferred over the tag codes because it
#: explains *why* the tag is absent.
MISSING_SPAN_TRUNCATED = "REPORT_SPAN_TRUNCATED"
#: The decision was refused at build time (spec §6.3 ``context_failure``); no ids exist.
MISSING_CONTEXT_FAILURE = "CONTEXT_FAILURE"

ARD_MISSINGNESS_CODES: tuple[str, ...] = (
    MISSING_NOT_REACHED,
    MISSING_ANCHOR_UNRESOLVED,
    MISSING_NO_REPORT_SPAN,
    MISSING_FINAL_TAG,
    MISSING_FINAL_MALFORMED,
    MISSING_FINAL_EMPTY,
    MISSING_SPAN_TRUNCATED,
    MISSING_CONTEXT_FAILURE,
)


# --------------------------------------------------------------------------- the anchor


@dataclass(frozen=True)
class TagAnchor:
    """One capture site.  Field names mirror ``study.neural.storage.ActivationRow`` so a
    row is built without a translation layer; ``byte_offset`` is kept for the audit trail
    (spec §14.2 "save exact token offsets and marker parse validity")."""

    kind: str
    token_offset: int | None
    missingness: str | None
    channel: str
    structural_span: str
    generated_token_count: int
    byte_offset: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in K.CAPTURE_ANCHORS:
            raise ValueError(f"anchor kind {self.kind!r} not in contract.CAPTURE_ANCHORS")
        if (self.token_offset is None) != (self.missingness is not None):
            raise ValueError("TagAnchor: token_offset is None iff missingness is set")
        if self.missingness is not None and self.missingness not in ARD_MISSINGNESS_CODES:
            raise ValueError(f"unknown missingness {self.missingness!r}")
        if self.token_offset is not None and self.token_offset < 0:
            raise ValueError("token_offset must be >= 0")

    @property
    def present(self) -> bool:
        return self.token_offset is not None

    @property
    def feature_time(self) -> str:
        return feature_time_of(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "token_offset": self.token_offset,
            "missingness": self.missingness,
            "channel": self.channel,
            "structural_span": self.structural_span,
            "generated_token_count": int(self.generated_token_count),
            "byte_offset": self.byte_offset,
            "feature_time": self.feature_time,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class AnchorSet:
    """The four anchors of one decision plus the marker-parse record (spec §14.2)."""

    decision_id: str
    prompt_tokens: int
    completion_tokens: int
    anchors: tuple[TagAnchor, ...]
    parse_status: str
    markers: Mapping[str, Any]
    report_end_source: str
    truncated: bool
    report_span: tuple[int, int] | None = None   # absolute token offsets [start, stop)

    def __post_init__(self) -> None:
        assert_four_anchors(self.anchors)

    def by_kind(self, kind: str) -> TagAnchor:
        for anchor in self.anchors:
            if anchor.kind == kind:
                return anchor
        raise KeyError(kind)   # pragma: no cover - assert_four_anchors already guarantees it

    @property
    def present_offsets(self) -> tuple[int, ...]:
        return tuple(sorted({a.token_offset for a in self.anchors if a.present}))

    @property
    def missing(self) -> tuple[TagAnchor, ...]:
        return tuple(a for a in self.anchors if not a.present)

    @property
    def k_max(self) -> int:
        """Completion tokens a teacher-forced replay must include (0 = prompt only)."""
        return k_max_needed(self.anchors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "parse_status": self.parse_status,
            "report_end_source": self.report_end_source,
            "truncated": bool(self.truncated),
            "report_span": None if self.report_span is None else list(self.report_span),
            "markers": dict(self.markers),
            "anchors": [a.to_dict() for a in self.anchors],
        }


def assert_four_anchors(anchors: Sequence[TagAnchor]) -> None:
    """Spec §14.2: exactly the four contract anchors, in contract order, once each."""
    kinds = tuple(a.kind for a in anchors)
    if kinds != tuple(K.CAPTURE_ANCHORS):
        raise AnchorError(
            f"expected the four contract anchors {tuple(K.CAPTURE_ANCHORS)} in order, got {kinds}"
        )


def k_max_needed(anchors: Iterable[TagAnchor]) -> int:
    return max((int(a.generated_token_count) for a in anchors if a.present), default=0)


# --------------------------------------------------------------------------- timing labels


def feature_time_of(anchor_kind: str) -> str:
    """``pre_answer`` for ``contract.PROSPECTIVE_ANCHORS``, else ``post_answer`` (spec §14.4)."""
    if anchor_kind not in K.CAPTURE_ANCHORS:
        raise ValueError(f"unknown anchor {anchor_kind!r}")
    return FEATURE_TIME_PRE if anchor_kind in K.PROSPECTIVE_ANCHORS else FEATURE_TIME_POST


def last_allowed_round(anchor_kind: str, round_: int) -> int:
    """Information horizon of a state captured at ``anchor_kind`` in round ``round_``.

    ``pre_answer`` → ``round_ - 1`` (the round-``round_`` answer is still upcoming);
    ``post_answer`` → ``round_`` (the answer is already in the state).  See the module
    docstring for the rule ``ard.neural.probes`` enforces with it.
    """
    return int(round_) - 1 if feature_time_of(anchor_kind) == FEATURE_TIME_PRE else int(round_)


def is_entropy_anchor(anchor_kind: str) -> bool:
    """Spec §14.5: full-vocabulary entropy is computed at anchors 1 and 3 only."""
    return anchor_kind in K.ENTROPY_ANCHORS


def entropy_anchor_offsets(anchors: Sequence[TagAnchor]) -> tuple[int, ...]:
    """Token offsets at which the capture pass must also return next-token logits."""
    return tuple(sorted({a.token_offset for a in anchors if a.present and is_entropy_anchor(a.kind)}))


# --------------------------------------------------------------------------- spans


def report_span(completion_ids: Sequence[int]) -> tuple[int, tuple[int, ...]] | None:
    """``(start_index, kept_ids)`` of the report span inside ``completion_ids``.

    Identical accounting to ``ard.client.decision_from_record``: the ids after the first
    ``</think>``, minus a trailing ``<|im_end|>``, truncated to ``contract.FINAL_TOKENS``.
    ``None`` when the completion did not open and close exactly one thinking channel — the
    two cases in which there is no report span at all.
    """
    ids = tuple(int(i) for i in completion_ids or ())
    if not ids or ids[0] != K.THINK_START_ID:
        return None
    try:
        end = ids.index(K.THINK_END_ID, 1)
    except ValueError:
        return None
    final_ids = ids[end + 1:]
    if final_ids and final_ids[-1] == K.IM_END_ID:
        final_ids = final_ids[:-1]
    return end + 1, final_ids[: K.FINAL_TOKENS]


def tag_span(text: str, open_tag: str, close_tag: str) -> tuple[str, tuple[int, int] | None, tuple[int, int] | None]:
    """``(status, content_span, pair_span)`` for one tag pair, in code-point indices.

    The status vocabulary and the acceptance rule are the frozen ones of
    ``ard.report._extract`` (spec §6.3): exactly one opener and one closer, in order.
    ``content_span`` is the **raw** (unstripped) region between the delimiters — the parser
    strips it for the answer string, but an anchor must point at a real token.
    """
    opens = text.count(open_tag)
    closes = text.count(close_tag)
    if opens == 0 and closes == 0:
        return R.TAG_MISSING, None, None
    if opens != 1 or closes != 1:
        start = text.find(open_tag)
        stop = text.rfind(close_tag)
        pair = (start, stop + len(close_tag)) if 0 <= start < stop else None
        return R.TAG_MALFORMED, None, pair
    start = text.index(open_tag)
    stop = text.index(close_tag)
    if stop < start:
        return R.TAG_MALFORMED, None, (stop, start + len(open_tag))
    pair = (start, stop + len(close_tag))
    content = (start + len(open_tag), stop)
    if not text[content[0]: content[1]].strip():
        return R.TAG_EMPTY, content, pair
    return R.TAG_OK, content, pair


def _mask(text: str, spans: Sequence[tuple[int, int] | None]) -> str:
    """Blank out ``spans`` keeping length — the ``ard.report`` rule that a probability
    written inside ``<final>``/``<evidence>`` is never read as the forecast (spec §6.4)."""
    chars = list(text)
    for span in spans:
        if span is None:
            continue
        for i in range(max(0, span[0]), min(len(chars), span[1])):
            chars[i] = " "
    return "".join(chars)


def _byte_offset(text: str, cp_index: int) -> int:
    """UTF-8 byte offset of code-point index ``cp_index`` in ``text``."""
    return len(text[:cp_index].encode("utf-8"))


# --------------------------------------------------------------------------- resolution


def resolve_anchors(
    *,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    finish_reason: str,
    tokenizer: Any,
    decision_id: str = "",
    report: R.Report | None = None,
) -> AnchorSet:
    """The four spec §14.2 anchors of one decision, as absolute token offsets.

    Offsets index the teacher-forced sequence ``prompt_token_ids + completion_token_ids``
    (``study.neural.replay.teacher_forced_sequence``), which is exactly what
    ``CaptureEngine`` prefills.

    ``report`` may be supplied when the caller already parsed the decision (the capture
    runner does); otherwise it is parsed here.  A per-tag disagreement between this
    module's span scan and the parser's status is an :class:`AnchorError`, never a guess.
    """
    prompt_ids = [int(t) for t in prompt_token_ids]
    completion = tuple(int(t) for t in completion_token_ids or ())
    if not prompt_ids:
        raise AnchorError("decision has no prompt tokens")
    prompt_len = len(prompt_ids)

    span = report_span(completion)
    kept: tuple[int, ...] = () if span is None else span[1]
    text = ""
    if kept:
        pieces = token_bytes(tokenizer, kept)
        spans = byte_spans(pieces)
        text = b"".join(pieces).decode("utf-8")
    else:
        pieces, spans = [], []

    if report is None:
        report = R.parse_report(text, finish_reason, completion)
    truncated = R.span_cut(finish_reason, len(_final_ids(completion)))

    prompt_anchor = TagAnchor(
        kind=ANCHOR_PROMPT_END,
        token_offset=prompt_len - 1,
        missingness=None,
        channel=CHANNEL_PROMPT,
        structural_span=f"prompt[0:{prompt_len})",
        generated_token_count=0,
        byte_offset=None,
        detail={"prompt_len": prompt_len, "completion_len": len(completion),
                "reused_anchor": "NATIVE_PREFILL"},
    )

    markers: dict[str, Any] = {
        "parse_status": report.parse_status.value,
        "final_status": report.final_status,
        "evidence_status": report.evidence_status,
        "confidence_status": report.confidence_status,
        "final_overrun": bool(report.final_overrun),
        "finish_reason": str(finish_reason),
        "report_tokens": len(kept),
        "confidence": report.confidence,
    }

    if span is None:
        code = MISSING_NO_REPORT_SPAN
        missing = tuple(
            TagAnchor(kind, None, code, CHANNEL_NONE, "report_span", 0, None,
                      {"reason": report.parse_status.value, "finish_reason": str(finish_reason)})
            for kind in (ANCHOR_PRE_FINAL, ANCHOR_ANSWER_END, ANCHOR_REPORT_END)
        )
        markers["report_span"] = None
        return AnchorSet(decision_id=decision_id, prompt_tokens=prompt_len,
                         completion_tokens=len(completion), anchors=(prompt_anchor, *missing),
                         parse_status=report.parse_status.value, markers=markers,
                         report_end_source=REPORT_END_NONE, truncated=truncated, report_span=None)

    base = prompt_len + span[0]                 # absolute offset of kept[0]
    abs_span = (base, base + len(kept))
    markers["report_span"] = list(abs_span)

    final_status, final_content, final_pair = tag_span(text, K.FINAL_OPEN, K.FINAL_CLOSE)
    evidence_status, _evidence_content, evidence_pair = tag_span(text, K.EVIDENCE_OPEN, K.EVIDENCE_CLOSE)
    masked = _mask(text, (final_pair, evidence_pair))
    confidence_status, _conf_content, confidence_pair = tag_span(masked, K.CONFIDENCE_OPEN, K.CONFIDENCE_CLOSE)
    _check_status("final", final_status, report.final_status)
    _check_status("evidence", evidence_status, report.evidence_status)
    # ``ard.report`` downgrades an OK confidence pair whose text is not a probability in
    # [0, 1] to malformed; the *span* is still the one we found, so only the missing/present
    # distinction has to agree.
    _check_status("confidence", confidence_status, report.confidence_status,
                  equivalent={(R.TAG_OK, R.TAG_MALFORMED)})

    markers.update({
        "final_byte_span": None if final_pair is None else [_byte_offset(text, final_pair[0]), _byte_offset(text, final_pair[1])],
        "confidence_byte_span": None if confidence_pair is None else [_byte_offset(text, confidence_pair[0]), _byte_offset(text, confidence_pair[1])],
    })

    def _unusable(status: str) -> str:
        """The typed reason a ``<final>``-derived anchor cannot be resolved."""
        if truncated and status in (R.TAG_MISSING, R.TAG_MALFORMED):
            return MISSING_SPAN_TRUNCATED
        if status == R.TAG_MALFORMED:
            return MISSING_FINAL_MALFORMED
        if status == R.TAG_EMPTY:
            return MISSING_FINAL_EMPTY
        return MISSING_FINAL_TAG

    detail_common = {"report_tokens": len(kept), "truncated": truncated}

    # ---- anchor 2: last token before the <final> content -------------------------------
    if final_status in (R.TAG_OK, R.TAG_EMPTY) and final_content is not None:
        open_end_cp = final_content[0]                 # first code point of the content
        byte_end = _byte_offset(text, open_end_cp)     # == end of the "<final>" delimiter
        pre_final = _anchor_at(
            ANCHOR_PRE_FINAL, spans, byte_end, base, prompt_len,
            structural_span=f"report[{K.FINAL_OPEN}]",
            detail={**detail_common, "tag": K.FINAL_OPEN},
        )
    else:
        pre_final = TagAnchor(ANCHOR_PRE_FINAL, None, _unusable(final_status), CHANNEL_NONE,
                              f"report[{K.FINAL_OPEN}]", 0, None,
                              {**detail_common, "final_status": final_status})

    # ---- anchor 3: last answer-content token -------------------------------------------
    if final_status == R.TAG_OK and final_content is not None:
        raw = text[final_content[0]: final_content[1]]
        stripped_end_cp = final_content[0] + len(raw.rstrip())
        byte_end = _byte_offset(text, stripped_end_cp)
        answer_end = _anchor_at(
            ANCHOR_ANSWER_END, spans, byte_end, base, prompt_len,
            structural_span="report[<final> content]",
            detail={**detail_common, "content_chars": len(raw.strip())},
        )
    else:
        answer_end = TagAnchor(ANCHOR_ANSWER_END, None, _unusable(final_status), CHANNEL_NONE,
                               "report[<final> content]", 0, None,
                               {**detail_common, "final_status": final_status})

    # ---- anchor 4: end of the report ----------------------------------------------------
    if confidence_status == R.TAG_OK and confidence_pair is not None:
        source = REPORT_END_CONFIDENCE_CLOSE
        byte_end = _byte_offset(text, confidence_pair[1])
        report_end = _anchor_at(
            ANCHOR_REPORT_END, spans, byte_end, base, prompt_len,
            structural_span=f"report[{K.CONFIDENCE_CLOSE}]",
            detail={**detail_common, "source": source},
        )
    elif final_status in (R.TAG_OK, R.TAG_EMPTY) and final_pair is not None:
        source = REPORT_END_FINAL_CLOSE
        byte_end = _byte_offset(text, final_pair[1])
        report_end = _anchor_at(
            ANCHOR_REPORT_END, spans, byte_end, base, prompt_len,
            structural_span=f"report[{K.FINAL_CLOSE}]",
            detail={**detail_common, "source": source, "confidence_status": confidence_status},
        )
    else:
        source = REPORT_END_NONE
        report_end = TagAnchor(ANCHOR_REPORT_END, None, _unusable(final_status), CHANNEL_NONE,
                               "report[end]", 0, None,
                               {**detail_common, "final_status": final_status,
                                "confidence_status": confidence_status})

    anchors = (prompt_anchor, pre_final, answer_end, report_end)
    return AnchorSet(decision_id=decision_id, prompt_tokens=prompt_len,
                     completion_tokens=len(completion), anchors=anchors,
                     parse_status=report.parse_status.value, markers=markers,
                     report_end_source=source, truncated=truncated, report_span=abs_span)


def _final_ids(completion: Sequence[int]) -> tuple[int, ...]:
    """The whole post-``</think>`` span (uncapped) — ``span_cut`` needs its true length."""
    ids = tuple(int(i) for i in completion or ())
    if not ids or ids[0] != K.THINK_START_ID:
        return ()
    try:
        end = ids.index(K.THINK_END_ID, 1)
    except ValueError:
        return ()
    out = ids[end + 1:]
    if out and out[-1] == K.IM_END_ID:
        out = out[:-1]
    return out


def _check_status(tag: str, mine: str, theirs: str, *, equivalent: set[tuple[str, str]] | None = None) -> None:
    if mine == theirs:
        return
    if equivalent and ((mine, theirs) in equivalent or (theirs, mine) in equivalent):
        return
    raise AnchorError(
        f"{tag} tag status disagrees with ard.report.parse_report ({mine!r} != {theirs!r}); "
        "the anchor resolver and the frozen parser must share one definition"
    )


def _anchor_at(
    kind: str,
    spans: Sequence[tuple[int, int]],
    byte_end: int,
    base: int,
    prompt_len: int,
    *,
    structural_span: str,
    detail: Mapping[str, Any],
) -> TagAnchor:
    """The token containing byte ``byte_end - 1`` of the report span, as an absolute offset."""
    try:
        local = last_token_of_span(spans, byte_end)
    except AnchorError as exc:
        return TagAnchor(kind, None, MISSING_ANCHOR_UNRESOLVED, CHANNEL_NONE, structural_span, 0,
                         None, {**dict(detail), "error": str(exc)[:200]})
    offset = base + local
    start, stop = spans[local]
    return TagAnchor(
        kind=kind,
        token_offset=offset,
        missingness=None,
        channel=CHANNEL_CONTENT,
        structural_span=structural_span,
        generated_token_count=offset - prompt_len + 1,
        byte_offset=byte_end,
        detail={**dict(detail), "report_token_index": local,
                "token_byte_span": [start, stop],
                # the delimiter and the neighbouring text can share one token; recorded
                # rather than silently shifted (spec §14.2 "no anchor is guessed")
                "shared_token": bool(stop > byte_end)},
    )


def resolve_decision_anchors(record: Any, tokenizer: Any, *, decision_id: str | None = None) -> AnchorSet:
    """:func:`resolve_anchors` for a ``DecisionRecord``, a ``RequestRecord`` or a mapping
    with the on-disk field names (``prompt_token_ids``, ``response.token_ids``)."""
    if isinstance(record, Mapping):
        prompt = record["prompt_token_ids"]
        response = record.get("response") or {}
        completion = response.get("token_ids") or ()
        finish = str(response.get("finish_reason") or "")
        rid = str(record.get("request_id") or record.get("decision_id") or "")
    elif hasattr(record, "completion_token_ids"):        # ard.client.DecisionRecord
        prompt = record.prompt_token_ids
        completion = record.completion_token_ids
        finish = str(record.finish_reason)
        rid = str(record.request_id)
    else:                                                # study.types.RequestRecord
        prompt = record.prompt_token_ids
        response = record.response or {}
        completion = response.get("token_ids") or ()
        finish = str(response.get("finish_reason") or "")
        rid = str(record.request_id)
    return resolve_anchors(prompt_token_ids=prompt, completion_token_ids=completion,
                           finish_reason=finish, tokenizer=tokenizer,
                           decision_id=str(decision_id) if decision_id is not None else rid)
