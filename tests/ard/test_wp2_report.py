"""ARD36 WP2: the report tag parser (spec §6.3).

The frozen table of outcomes for the post-``</think>`` span, one assertion per rule of
spec §6.3: missing/malformed answers are incorrect *delivered* answers, missing confidence
is a missing feature and never 0.5, and the parser never searches the reasoning span.
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.report import (
    TAG_MALFORMED,
    Report,
    context_failure_report,
    parse_report,
    report_content_sha256,
)
from tests.ard import wp2_support as W

OK = K.ParseStatus.OK


def _ids(n_final: int = 6, *, open_think: bool = True, close_think: bool = True) -> tuple[int, ...]:
    ids: list[int] = []
    if open_think:
        ids.append(K.THINK_START_ID)
    ids += [11, 12]
    if close_think:
        ids.append(K.THINK_END_ID)
        ids += list(range(20, 20 + n_final))
        ids.append(K.IM_END_ID)
    return tuple(ids)


E = f"{K.EVIDENCE_OPEN}two times three{K.EVIDENCE_CLOSE}"
C = f"{K.CONFIDENCE_OPEN}0.8{K.CONFIDENCE_CLOSE}"


@pytest.mark.parametrize(
    ("text", "status", "answer"),
    [
        (f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}{C}", OK, "42"),
        (f"  {K.FINAL_OPEN} 42 {K.FINAL_CLOSE} {E} {C}", OK, "42"),
        (f"the answer is 42 {E}{C}", K.ParseStatus.MISSING_FINAL, None),
        (f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{K.FINAL_OPEN}7{K.FINAL_CLOSE}", K.ParseStatus.MALFORMED_FINAL, None),
        (f"{K.FINAL_OPEN}{K.FINAL_OPEN}42{K.FINAL_CLOSE}{K.FINAL_CLOSE}", K.ParseStatus.MALFORMED_FINAL, None),
        (f"{K.FINAL_OPEN}42 {E}{C}", K.ParseStatus.MALFORMED_FINAL, None),
        (f"42{K.FINAL_CLOSE}{E}", K.ParseStatus.MALFORMED_FINAL, None),
        (f"{K.FINAL_OPEN}{K.FINAL_CLOSE}{E}{C}", K.ParseStatus.EMPTY_FINAL, None),
        (f"{K.FINAL_OPEN}   {K.FINAL_CLOSE}{E}{C}", K.ParseStatus.EMPTY_FINAL, None),
    ],
)
def test_final_tag_table(text: str, status: K.ParseStatus, answer: str | None) -> None:
    report = parse_report(text, "stop", _ids())
    assert report.parse_status is status
    assert report.final_answer == answer
    assert report.valid == (status is OK)


def test_no_think_open_and_premature_eos() -> None:
    body = f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}{C}"
    assert parse_report(body, "stop", _ids(open_think=False)).parse_status is K.ParseStatus.NO_THINK_OPEN
    assert (
        parse_report("", "length", _ids(close_think=False)).parse_status
        is K.ParseStatus.PREMATURE_REASONING_EOS
    )
    # An empty completion is not a silently valid report either.
    assert parse_report("", "stop", ()).parse_status is K.ParseStatus.NO_THINK_OPEN


def test_truncated_span_is_not_ok() -> None:
    body = f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}{C}"
    cut = parse_report(body, "length", _ids())
    assert cut.parse_status is K.ParseStatus.TRUNCATED
    assert cut.final_answer is None and not cut.valid
    # ``final_overrun``: more than FINAL_TOKENS were generated after </think>, so the text
    # handed to the parser is a prefix of the report (plan §5.2).
    over = parse_report(body, "stop", _ids(K.FINAL_TOKENS + 1))
    assert over.final_overrun is True
    assert over.parse_status is K.ParseStatus.TRUNCATED
    exact = parse_report(body, "stop", _ids(K.FINAL_TOKENS))
    assert exact.final_overrun is False and exact.parse_status is OK


def test_missing_confidence_is_missing_never_half() -> None:
    report = parse_report(f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}", "stop", _ids())
    assert report.parse_status is OK
    assert report.confidence is None  # spec §6.3: a missing feature, not 0.5


@pytest.mark.parametrize(
    "tail",
    [
        f"{K.CONFIDENCE_OPEN}high{K.CONFIDENCE_CLOSE}",
        f"{K.CONFIDENCE_OPEN}1.4{K.CONFIDENCE_CLOSE}",
        f"{K.CONFIDENCE_OPEN}-0.1{K.CONFIDENCE_CLOSE}",
        f"{K.CONFIDENCE_OPEN}0.8 (high){K.CONFIDENCE_CLOSE}",
        f"{K.CONFIDENCE_OPEN}0.8{K.CONFIDENCE_CLOSE}{K.CONFIDENCE_OPEN}0.2{K.CONFIDENCE_CLOSE}",
        f"{K.CONFIDENCE_OPEN}0.8",
    ],
)
def test_malformed_confidence_never_becomes_a_number(tail: str) -> None:
    report = parse_report(f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}{tail}", "stop", _ids())
    assert report.parse_status is OK          # a bad confidence never invalidates the answer
    assert report.confidence is None
    assert report.confidence_status in (TAG_MALFORMED, "missing")


def test_confidence_inside_evidence_is_never_the_forecast() -> None:
    """Spec §6.3/§6.4: evidence prose that *looks* like a forecast is evidence."""
    evidence = (
        f"{K.EVIDENCE_OPEN}I would put {K.CONFIDENCE_OPEN}0.99{K.CONFIDENCE_CLOSE} on this"
        f"{K.EVIDENCE_CLOSE}"
    )
    report = parse_report(f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{evidence}", "stop", _ids())
    assert report.parse_status is OK
    assert report.confidence is None
    assert "0.99" in (report.evidence or "")   # and it is preserved verbatim, never erased

    outside = parse_report(
        f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{evidence}{K.CONFIDENCE_OPEN}0.3{K.CONFIDENCE_CLOSE}",
        "stop",
        _ids(),
    )
    assert outside.confidence == 0.3


def test_a_confidence_looking_answer_is_not_a_confidence() -> None:
    report = parse_report(f"{K.FINAL_OPEN}0.5{K.FINAL_CLOSE}{E}", "stop", _ids())
    assert report.final_answer == "0.5" and report.confidence is None


def test_parser_never_reads_the_thinking_span() -> None:
    """Different reasoning ids, identical report: nothing but the delimiters is read."""
    body = f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}{E}{C}"
    a = parse_report(body, "stop", (K.THINK_START_ID, 101, 102, 103, K.THINK_END_ID, 20, K.IM_END_ID))
    b = parse_report(body, "stop", (K.THINK_START_ID, 900, K.THINK_END_ID, 20, K.IM_END_ID))
    assert a == b


def test_content_hash_separates_content_and_is_stable() -> None:
    a = W.report("42")
    b = W.report("42")
    c = W.report("7")
    d = W.report("42", evidence="a different reason")
    e = W.report("42", confidence=0.1)
    assert a.content_sha256 == b.content_sha256
    assert len({a.content_sha256, c.content_sha256, d.content_sha256, e.content_sha256}) == 4
    assert a.content_sha256 == report_content_sha256("42", a.evidence, a.confidence, a.parse_status)


def test_context_failure_report() -> None:
    report = context_failure_report()
    assert isinstance(report, Report)
    assert report.parse_status is K.ParseStatus.CONTEXT_FAILURE
    assert not report.valid and report.final_answer is None and report.confidence is None
