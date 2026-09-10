"""ARD36 report parsing: the post-``</think>`` span, strictly (WP2).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §6.3
("Final report"): the frozen three-tag schema

``<final>…</final><evidence>…≤120 tokens…</evidence><confidence>p in [0,1]</confidence>``

with the rules this module enforces literally:

* **"Parsing must not search the private reasoning span for a favorable answer."**
  :func:`parse_report` receives the post-``</think>`` text only.  It never decodes the
  thinking ids; ``completion_ids`` is used *solely* for delimiter accounting (did the
  completion open and close exactly one thinking channel, and was the report span cut).
* **"Missing/malformed answer is an incorrect delivered answer."**  Duplicate, nested,
  unclosed and empty ``<final>`` each get their own
  :class:`~agents_scaling.ard.contract.ParseStatus`; none of them is ``OK``, so none of
  them casts a vote (``contract.VALID_PARSE_STATUSES == {OK}``).
* **"Missing confidence is a missing feature, not automatically 0.5."**  ``confidence`` is
  ``Optional[float]`` and stays ``None`` when the tag is absent, malformed, duplicated or
  outside ``[0, 1]``.
* Spec §6.4: "Do not erase confidence wording embedded in the evidence field."  The
  evidence text is preserved verbatim, and a confidence-looking string *inside*
  ``<evidence>`` is never read as the forecast: the confidence scan runs over the span
  with the ``<final>`` and ``<evidence>`` regions masked out.

Plan §4.3/§5.2: this is ARD36's own ``Report`` dataclass with a content hash through
``study.parse.candidate.jcs_with_numbers`` — deliberately *not* mapped onto the frozen
``study.types.Candidate``, whose ``confidence`` is a required float and whose six-key JSON
schema is a different contract.

Frozen interface: ``docs/ard36/00_contract.md`` §2 (WP2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.study import identity
from agents_scaling.study.parse.candidate import jcs_with_numbers

__all__ = [
    "REPORT_SCHEMA",
    "Report",
    "TAG_MALFORMED",
    "TAG_MISSING",
    "TAG_EMPTY",
    "TAG_OK",
    "context_failure_report",
    "parse_decision",
    "parse_report",
    "report_content_sha256",
    "span_cut",
]

#: Content-hash schema tag; bump only with a new contract version.
REPORT_SCHEMA: str = "ard_report_v1"

TAG_OK = "ok"
TAG_MISSING = "missing"
TAG_MALFORMED = "malformed"
TAG_EMPTY = "empty"

#: A probability literal and nothing else.  ``"0.9 (high)"`` is malformed, not 0.9.
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


# --------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class Report:
    """One parsed ARD36 report (contract §2 WP2).

    ``final_answer`` / ``evidence`` are the exact source excerpts between the tags with
    surrounding whitespace stripped — never a re-generated summary (spec §4.3 "each field
    is represented as an exact source excerpt").  ``confidence`` is missing-or-a-number.
    ``content_sha256`` hashes the whole delivered report (RFC 8785 + ES6 numbers) and is
    the content-level answer-change definition of plan §5.8, beside the canonical class.

    The five contract fields come first; the trailing fields are diagnostics for
    ``decisions.parquet`` and never change the contract's meaning.
    """

    final_answer: str | None
    evidence: str | None
    confidence: float | None
    parse_status: K.ParseStatus
    content_sha256: str
    final_status: str = TAG_MISSING
    evidence_status: str = TAG_MISSING
    confidence_status: str = TAG_MISSING
    final_overrun: bool = False
    raw_sha256: str = ""

    @property
    def valid(self) -> bool:
        """Does this report deliver a gradeable answer (``contract.VALID_PARSE_STATUSES``)?"""
        return self.parse_status in K.VALID_PARSE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready row fragment for ``decisions.parquet`` (spec §18)."""
        return {
            "answer_raw": self.final_answer,
            "evidence_text": self.evidence,
            "confidence": self.confidence,
            "parse_status": self.parse_status.value,
            "content_sha256": self.content_sha256,
            "final_status": self.final_status,
            "evidence_status": self.evidence_status,
            "confidence_status": self.confidence_status,
            "final_overrun": self.final_overrun,
            "raw_sha256": self.raw_sha256,
        }


def report_content_sha256(
    final_answer: str | None,
    evidence: str | None,
    confidence: float | None,
    parse_status: K.ParseStatus,
) -> str:
    """SHA-256 over the canonical report object (``parse/candidate.jcs_with_numbers``).

    ``identity.jcs`` refuses floats on purpose (§3.6 request identities carry no floats);
    the *content* hash may carry one, which is exactly why the candidate module's ES6
    superset is used here.
    """
    payload = {
        "schema": REPORT_SCHEMA,
        "final_answer": final_answer,
        "evidence": evidence,
        "confidence": None if confidence is None else float(confidence),
        "parse_status": parse_status.value,
    }
    return identity.sha256_hex(jcs_with_numbers(payload))


# --------------------------------------------------------------------------- tag extraction


def _extract(text: str, open_tag: str, close_tag: str) -> tuple[str, str | None, tuple[int, int] | None]:
    """Strict single-pair extraction.

    Returns ``(status, content, span)``.  ``status`` is one of :data:`TAG_OK`,
    :data:`TAG_MISSING`, :data:`TAG_MALFORMED`, :data:`TAG_EMPTY`; ``span`` is the
    ``[start, end)`` slice of ``text`` covered by the pair (open tag through close tag),
    present whenever both delimiters were found in order.
    """
    opens = text.count(open_tag)
    closes = text.count(close_tag)
    if opens == 0 and closes == 0:
        return TAG_MISSING, None, None
    if opens != 1 or closes != 1:
        # duplicate (two reports), nested (``<final><final>``), or a stray closer
        start = text.find(open_tag)
        stop = text.rfind(close_tag)
        span = (start, stop + len(close_tag)) if 0 <= start < stop else None
        return TAG_MALFORMED, None, span
    start = text.index(open_tag)
    stop = text.index(close_tag)
    if stop < start:
        return TAG_MALFORMED, None, (stop, start + len(open_tag))
    span = (start, stop + len(close_tag))
    content = text[start + len(open_tag) : stop]
    if not content.strip():
        return TAG_EMPTY, "", span
    return TAG_OK, content.strip(), span


def _mask(text: str, spans: Sequence[tuple[int, int] | None]) -> str:
    """Blank out ``spans`` (keeping length) so a later scan cannot see inside them."""
    chars = list(text)
    for span in spans:
        if span is None:
            continue
        start, stop = max(0, span[0]), min(len(chars), span[1])
        for i in range(start, stop):
            chars[i] = " "
    return "".join(chars)


def _confidence_value(content: str | None) -> float | None:
    if content is None:
        return None
    token = content.strip()
    if not _NUMBER_RE.match(token):
        return None
    try:
        value = float(token)
    except ValueError:  # pragma: no cover - the regex already rejects these
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


# --------------------------------------------------------------------------- delimiters


def _think_status(completion_ids: Sequence[int]) -> tuple[K.ParseStatus | None, int]:
    """Delimiter accounting over the completion ids (contract §3: the ids are exact).

    Returns ``(status, n_final_ids)``; ``status`` is ``None`` when the completion opened
    and closed exactly one thinking channel.  This mirrors
    ``ard.client.decision_from_record`` so a report parsed from a stored record and one
    parsed from a live record agree.
    """
    ids = tuple(int(i) for i in completion_ids or ())
    if not ids:
        return K.ParseStatus.NO_THINK_OPEN, 0
    if ids[0] != K.THINK_START_ID:
        return K.ParseStatus.NO_THINK_OPEN, 0
    try:
        end = ids.index(K.THINK_END_ID, 1)
    except ValueError:
        return K.ParseStatus.PREMATURE_REASONING_EOS, 0
    final_ids = ids[end + 1 :]
    if final_ids and final_ids[-1] == K.IM_END_ID:
        final_ids = final_ids[:-1]
    return None, len(final_ids)


def span_cut(finish_reason: str, n_final_ids: int) -> bool:
    """Was the text handed to the parser a *cut* report span (spec §6.3 ``truncated``)?

    Two disjoint ways to lose the tail of a report, both of which leave the parser with an
    incomplete span: the whole response hit ``max_tokens`` (``finish_reason == "length"``),
    or the model wrote past the frozen 256-token report allowance and the surplus was
    dropped (``final_overrun``; plan §5.2).
    """
    return str(finish_reason) == "length" or int(n_final_ids) > K.FINAL_TOKENS


# --------------------------------------------------------------------------- entry point


def parse_report(
    final_text: str,
    finish_reason: str,
    completion_ids: Sequence[int],
) -> Report:
    """Parse the post-``</think>`` span into a :class:`Report` (contract §2 WP2, spec §6.3).

    ``final_text`` is the report span *only* — ``ard.client.decision_from_record`` decodes
    the first ``contract.FINAL_TOKENS`` ids after the first ``</think>``.  Nothing in this
    function reaches back into the thinking channel: ``completion_ids`` is read for the
    delimiter statuses and the report length, never decoded.

    Status precedence, most structural first:

    1. ``NO_THINK_OPEN`` — the completion did not start with ``<think>``.
    2. ``PREMATURE_REASONING_EOS`` — the thinking channel never closed, so there is no
       report span at all.
    3. ``TRUNCATED`` — the span exists but was cut (see :func:`span_cut`); the tags that
       *were* recovered are still returned for diagnostics, but the status is not ``OK``
       and the report casts no vote.
    4. ``MALFORMED_FINAL`` / ``EMPTY_FINAL`` / ``MISSING_FINAL`` — the ``<final>`` tag.
    5. ``OK``.
    """
    text = "" if final_text is None else str(final_text)
    think_status, n_final_ids = _think_status(completion_ids)
    overrun = int(n_final_ids) > K.FINAL_TOKENS

    final_status, final_answer, final_span = _extract(text, K.FINAL_OPEN, K.FINAL_CLOSE)
    evidence_status, evidence, evidence_span = _extract(text, K.EVIDENCE_OPEN, K.EVIDENCE_CLOSE)
    # Spec §6.3/§6.4: a probability written *inside* the evidence is evidence prose, never
    # the forecast.  Mask both recovered spans before looking for <confidence>.
    masked = _mask(text, (final_span, evidence_span))
    confidence_status, confidence_text, _ = _extract(masked, K.CONFIDENCE_OPEN, K.CONFIDENCE_CLOSE)
    confidence = _confidence_value(confidence_text) if confidence_status == TAG_OK else None
    if confidence_status == TAG_OK and confidence is None:
        confidence_status = TAG_MALFORMED

    if think_status is not None:
        status = think_status
    elif span_cut(finish_reason, n_final_ids):
        status = K.ParseStatus.TRUNCATED
    elif final_status == TAG_MISSING:
        status = K.ParseStatus.MISSING_FINAL
    elif final_status == TAG_MALFORMED:
        status = K.ParseStatus.MALFORMED_FINAL
    elif final_status == TAG_EMPTY:
        status = K.ParseStatus.EMPTY_FINAL
    else:
        status = K.ParseStatus.OK

    if status is not K.ParseStatus.OK:
        # Only an OK report delivers an answer; the excerpts stay for the audit trail but
        # never reach a vote, a packet or a grader.
        answer: str | None = None
    else:
        answer = final_answer

    evidence_value = evidence if evidence_status == TAG_OK else None
    return Report(
        final_answer=answer,
        evidence=evidence_value,
        confidence=confidence,
        parse_status=status,
        content_sha256=report_content_sha256(answer, evidence_value, confidence, status),
        final_status=final_status,
        evidence_status=evidence_status,
        confidence_status=confidence_status,
        final_overrun=overrun,
        raw_sha256=identity.sha256_hex(text),
    )


def context_failure_report() -> Report:
    """The report of a decision refused at build time (spec §6.3 ``context_failure``).

    ``ard.client.ContextTooLong`` is raised before any request exists, so there is no span
    to parse; the episode records this report so the slot still costs one planned
    opportunity (spec §3: "K counts the planned opportunities, including failed ones").
    """
    status = K.ParseStatus.CONTEXT_FAILURE
    return Report(
        final_answer=None,
        evidence=None,
        confidence=None,
        parse_status=status,
        content_sha256=report_content_sha256(None, None, None, status),
        final_status=TAG_MISSING,
        evidence_status=TAG_MISSING,
        confidence_status=TAG_MISSING,
        final_overrun=False,
        raw_sha256=identity.sha256_hex(""),
    )


def parse_decision(record: Mapping[str, Any] | Any) -> Report:
    """Convenience: parse a :class:`~agents_scaling.ard.client.DecisionRecord`.

    Accepts the dataclass or its ``to_dict()`` row, so compaction can re-derive a report
    from ``decisions.parquet`` without re-running inference.
    """
    if isinstance(record, Mapping):
        return parse_report(
            record.get("final_text", ""),
            record.get("finish_reason", ""),
            record.get("completion_token_ids", ()),
        )
    return parse_report(record.final_text, record.finish_reason, record.completion_token_ids)
