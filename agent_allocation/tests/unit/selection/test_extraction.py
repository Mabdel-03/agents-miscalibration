"""extraction/final.py: one closed <final> element decides validity; nothing else does."""
from __future__ import annotations

import pytest

from agent_allocation.contracts import CompletionStatus, OutputKind, ParseStatus, TransportStatus
from agent_allocation.extraction.final import extract_candidate
from tests.unit.selection import factories as f


def _extract(content, kind=OutputKind.MATH, **kw):
    t = f.task(kind)
    s = f.spec(t)
    return extract_candidate(f.result(s, content, **kw), s, t), s


def test_single_final_is_valid_and_identity_is_the_node_id():
    cand, s = _extract("<rationale>\nadd\n</rationale>\n<final>\n\\frac{1}{2}\n</final>")
    assert cand.parse_status is ParseStatus.VALID
    assert cand.candidate_id == cand.node_id == s.node_id
    assert cand.final_answer == "\\frac{1}{2}"
    assert cand.rationale == "add"
    assert cand.finish_reason == "stop" and cand.output_kind is OutputKind.MATH


def test_length_finish_with_a_closed_final_is_still_valid():
    cand, _ = _extract("<final>42</final> trailing text cut o", finish_reason="length",
                       completion=CompletionStatus.LENGTH)
    assert cand.parse_status is ParseStatus.VALID and cand.final_answer == "42"
    assert cand.finish_reason == "length"


@pytest.mark.parametrize("content", ["The answer is 42.", "Answer: \\boxed{42}", "<rationale>x</rationale>"])
def test_no_final_is_missing_and_no_heuristic_rescues_it(content):
    cand, _ = _extract(content)
    assert cand.parse_status is ParseStatus.MISSING and cand.final_answer is None


@pytest.mark.parametrize("content", [
    "<final>1</final><final>2</final>",          # two finals
    "<final>1",                                  # unclosed (typical truncation)
    "1</final>",                                 # stray closer
    "</final>1<final>",                          # reversed
    "<final><final>1</final>",                   # nested
    "<final>   </final>",                        # blank element holds no answer
])
def test_malformed_finals(content):
    cand, _ = _extract(content, finish_reason="length")
    assert cand.parse_status is ParseStatus.MALFORMED and cand.final_answer is None


def test_tags_are_case_sensitive():
    cand, _ = _extract("<FINAL>1</FINAL>")
    assert cand.parse_status is ParseStatus.MISSING


@pytest.mark.parametrize("content", [None, "", "  \n "])
def test_empty_visible_content(content):
    cand, _ = _extract(content, finish_reason="length", completion=CompletionStatus.LENGTH)
    assert cand.parse_status is ParseStatus.EMPTY and cand.final_answer is None
    assert cand.visible_content == (content or "")


def test_transport_failure_has_no_content_and_is_empty():
    cand, _ = _extract(None, completion=None, transport=TransportStatus.ERROR_ENVELOPE, finish_reason=None)
    assert cand.parse_status is ParseStatus.EMPTY


@pytest.mark.parametrize("transport", [TransportStatus.CACHE_HIT, TransportStatus.PROVIDER_MISMATCH])
def test_rejected_transport_with_a_body_is_never_a_valid_draw(transport):
    cand, _ = _extract("<final>1</final>", transport=transport)
    assert cand.parse_status is ParseStatus.EMPTY and cand.final_answer is None
    assert cand.visible_content == "<final>1</final>"


def test_blocked_is_refusal_even_with_content():
    cand, _ = _extract("<final>1</final>", completion=CompletionStatus.BLOCKED, finish_reason="content_filter")
    assert cand.parse_status is ParseStatus.REFUSAL and cand.final_answer is None


def test_python_strips_exactly_one_outer_fence():
    code = "def f(x):\n    return x + 1"
    cand, _ = _extract(f"<final>\n```python\n{code}\n```\n</final>", kind=OutputKind.PYTHON)
    assert cand.parse_status is ParseStatus.VALID and cand.final_answer == code
    bare, _ = _extract(f"<final>```\n{code}\n```</final>", kind=OutputKind.PYTHON)
    assert bare.final_answer == code
    double, _ = _extract(f"<final>```python\n```python\n{code}\n```\n```</final>", kind=OutputKind.PYTHON)
    assert double.final_answer == f"```python\n{code}\n```"
    other, _ = _extract(f"<final>```json\n{code}\n```</final>", kind=OutputKind.PYTHON)
    assert other.final_answer.startswith("```json")
    unclosed, _ = _extract(f"<final>```python\n{code}</final>", kind=OutputKind.PYTHON)
    assert unclosed.final_answer == f"```python\n{code}"


def test_fence_only_python_final_is_malformed():
    cand, _ = _extract("<final>```python\n```</final>", kind=OutputKind.PYTHON)
    assert cand.parse_status is ParseStatus.MALFORMED


def test_math_final_keeps_its_raw_text_and_fences_are_not_stripped():
    cand, _ = _extract("<final>```\n\\boxed{3}\n```</final>")
    assert cand.final_answer == "```\n\\boxed{3}\n```"


def test_identity_mismatches_raise():
    t = f.task()
    s = f.spec(t)
    other = f.spec(t, root_slot=1)
    with pytest.raises(ValueError):
        extract_candidate(f.result(other, "<final>1</final>"), s, t)
    with pytest.raises(ValueError):
        extract_candidate(f.result(s, "<final>1</final>"), s, f.task(item_id="other"))
