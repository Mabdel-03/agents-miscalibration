"""WP6 anchors: the four spec §14.2 capture sites of one decision (CPU, real tokenizer).

Nothing here loads a checkpoint: the anchors are pure byte→token arithmetic over the
pinned offline Qwen3 tokenizer, so every offset is checked against ``tokenizer.decode``
itself rather than against a re-implementation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from agents_scaling.ard import contract as K
from agents_scaling.ard import report as R
from agents_scaling.ard.neural import anchors as A
from tests.ard import wp6_support as W

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def tok():
    try:
        return W.support.qwen_tokenizer(K.CHECKPOINT_SMALL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-8B tokenizer unavailable offline: {exc!r}")


def resolve(tok, content, *, finish_reason="stop", think="reasoning 一二三", im_end=True,
            task="What is 6 times 7?", report=None):
    """``(anchor_set, prompt_ids, completion_ids)`` for one synthetic decision."""
    _messages, prompt = W.prompt_ids(tok, task)
    completion = W.completion_ids(tok, content, think=think, im_end=im_end)
    aset = A.resolve_anchors(prompt_token_ids=prompt, completion_token_ids=completion,
                             finish_reason=finish_reason, tokenizer=tok, decision_id="d0",
                             report=report)
    return aset, prompt, completion


def assert_offset_matches_decode(tok, aset, anchor, prompt, completion):
    """The anchor's absolute offset is the token holding the last byte of its span.

    Checked against ``tokenizer.decode`` directly: decoding the report span up to and
    including the anchor token must cover ``byte_offset``, and stopping one token earlier
    must not.  That is the definition of "last token of the span", with no re-derivation
    of the tokenizer's own byte accounting.
    """
    span = A.report_span(completion)
    assert span is not None
    base, kept = len(prompt) + span[0], span[1]
    local = anchor.token_offset - base
    assert 0 <= local < len(kept)
    # the anchor points at a real token of the teacher-forced sequence
    assert (list(prompt) + list(completion))[anchor.token_offset] == kept[local]
    upto = len(tok.decode(kept[: local + 1]).encode("utf-8"))
    before = len(tok.decode(kept[:local]).encode("utf-8")) if local else 0
    assert before < anchor.byte_offset <= upto
    assert anchor.generated_token_count == anchor.token_offset - len(prompt) + 1


# --------------------------------------------------------------------------- the four sites


def test_all_four_anchors_resolve_over_the_tag_contract(tok):
    aset, prompt, completion = resolve(tok, W.report_text("42"))

    assert tuple(a.kind for a in aset.anchors) == tuple(K.CAPTURE_ANCHORS)
    assert all(a.present for a in aset.anchors)
    assert aset.parse_status == K.ParseStatus.OK.value
    assert aset.report_end_source == A.REPORT_END_CONFIDENCE_CLOSE

    # 1. PROMPT_END is the last prompt token — study_v4's NATIVE_PREFILL, reused as such.
    prompt_end = aset.by_kind(A.ANCHOR_PROMPT_END)
    assert prompt_end.token_offset == len(prompt) - 1
    assert prompt_end.channel == A.CHANNEL_PROMPT
    assert prompt_end.generated_token_count == 0
    assert prompt_end.detail["reused_anchor"] == "NATIVE_PREFILL"

    text = tok.decode(A.report_span(completion)[1])
    # 2. PRE_FINAL sits on the last token of the "<final>" opener, before any answer byte.
    pre_final = aset.by_kind(A.ANCHOR_PRE_FINAL)
    assert pre_final.byte_offset == len(text[: text.index(K.FINAL_OPEN) + len(K.FINAL_OPEN)].encode("utf-8"))
    # 3. ANSWER_END sits on the last answer-content token, before <confidence>.
    answer_end = aset.by_kind(A.ANCHOR_ANSWER_END)
    assert answer_end.byte_offset == len(text[: text.index(K.FINAL_CLOSE)].encode("utf-8"))
    # 4. REPORT_END sits on the last token of "</confidence>".
    report_end = aset.by_kind(A.ANCHOR_REPORT_END)
    assert report_end.byte_offset == len(text.encode("utf-8"))

    assert pre_final.token_offset < answer_end.token_offset < report_end.token_offset
    for anchor in (pre_final, answer_end, report_end):
        assert_offset_matches_decode(tok, aset, anchor, prompt, completion)


def test_byte_offsets_survive_multibyte_answers(tok):
    """The mapping is over UTF-8 bytes, so a CJK answer must not shift an anchor."""
    aset, prompt, completion = resolve(tok, W.report_text("四十二 — 一二三"))
    text = tok.decode(A.report_span(completion)[1])
    answer_end = aset.by_kind(A.ANCHOR_ANSWER_END)
    assert answer_end.present
    assert answer_end.byte_offset == len(text[: text.index(K.FINAL_CLOSE)].encode("utf-8"))
    assert answer_end.byte_offset != text.index(K.FINAL_CLOSE)     # bytes, not code points
    assert_offset_matches_decode(tok, aset, answer_end, prompt, completion)


def test_report_end_falls_back_to_the_final_close_when_confidence_is_missing(tok):
    aset, prompt, completion = resolve(tok, W.report_text("42", confidence=None))

    assert aset.report_end_source == A.REPORT_END_FINAL_CLOSE
    report_end = aset.by_kind(A.ANCHOR_REPORT_END)
    assert report_end.present
    text = tok.decode(A.report_span(completion)[1])
    end = text.index(K.FINAL_CLOSE) + len(K.FINAL_CLOSE)
    assert report_end.byte_offset == len(text[:end].encode("utf-8"))
    assert report_end.detail["confidence_status"] == R.TAG_MISSING
    assert_offset_matches_decode(tok, aset, report_end, prompt, completion)
    # a missing confidence is a missing feature, never 0.5 (contract §2 WP2)
    assert aset.markers["confidence"] is None
    # the fallback is the *last final-span token*, strictly after the answer content
    assert report_end.token_offset > aset.by_kind(A.ANCHOR_ANSWER_END).token_offset


def test_report_end_uses_the_final_close_when_the_confidence_value_is_not_a_probability(tok):
    """``ard.report`` downgrades ``<confidence>high</confidence>``; the anchor follows it."""
    aset, prompt, completion = resolve(tok, W.report_text("42", confidence="high"))
    assert aset.markers["confidence"] is None
    assert aset.markers["confidence_status"] == R.TAG_MALFORMED
    # the *span* is still well formed, so anchor 4 is the confidence close, not a guess
    assert aset.report_end_source == A.REPORT_END_CONFIDENCE_CLOSE
    assert_offset_matches_decode(tok, aset, aset.by_kind(A.ANCHOR_REPORT_END), prompt, completion)


# --------------------------------------------------------------------------- typed missingness


def test_malformed_report_is_typed_missingness_not_a_guessed_offset(tok):
    """Two ``<final>`` pairs and no confidence: the frozen parser calls the report
    malformed, and none of the three report anchors is invented."""
    body = "\n\n<final>42</final><final>43</final>"
    aset, _prompt, _completion = resolve(tok, body)

    assert aset.parse_status == K.ParseStatus.MALFORMED_FINAL.value
    for kind in (A.ANCHOR_PRE_FINAL, A.ANCHOR_ANSWER_END, A.ANCHOR_REPORT_END):
        anchor = aset.by_kind(kind)
        assert not anchor.present
        assert anchor.token_offset is None and anchor.byte_offset is None
        assert anchor.missingness == A.MISSING_FINAL_MALFORMED
        assert anchor.missingness in A.ARD_MISSINGNESS_CODES
        assert anchor.channel == A.CHANNEL_NONE
    assert aset.report_end_source == A.REPORT_END_NONE
    assert aset.by_kind(A.ANCHOR_PROMPT_END).present     # anchor 1 never depends on the tags


def test_malformed_final_with_a_well_formed_confidence_still_ends_at_a_real_token(tok):
    """Anchor 4 is the *end of the report*, so a valid ``</confidence>`` pair resolves it
    even when ``<final>`` is malformed — from that tag's own byte span, never a guess.
    The invalidity travels with the row in ``markers`` (spec §14.2 "marker parse validity")."""
    body = "\n\n<final>42</final><final>43</final><confidence>0.9</confidence>"
    aset, prompt, completion = resolve(tok, body)

    assert aset.parse_status == K.ParseStatus.MALFORMED_FINAL.value
    assert aset.markers["final_status"] == R.TAG_MALFORMED
    assert aset.by_kind(A.ANCHOR_PRE_FINAL).missingness == A.MISSING_FINAL_MALFORMED
    assert aset.by_kind(A.ANCHOR_ANSWER_END).missingness == A.MISSING_FINAL_MALFORMED

    report_end = aset.by_kind(A.ANCHOR_REPORT_END)
    assert aset.report_end_source == A.REPORT_END_CONFIDENCE_CLOSE
    text = tok.decode(A.report_span(completion)[1])
    assert report_end.byte_offset == len(text.encode("utf-8"))
    assert_offset_matches_decode(tok, aset, report_end, prompt, completion)


def test_truncated_report_is_typed_as_truncation_rather_than_a_missing_tag(tok):
    """A span cut by ``max_tokens`` explains *why* the closing delimiter is absent."""
    aset, _prompt, _completion = resolve(tok, "\n\n<final>42", finish_reason="length", im_end=False)

    assert aset.truncated is True
    assert aset.parse_status == K.ParseStatus.TRUNCATED.value
    assert aset.by_kind(A.ANCHOR_ANSWER_END).missingness == A.MISSING_SPAN_TRUNCATED
    assert aset.by_kind(A.ANCHOR_REPORT_END).missingness == A.MISSING_SPAN_TRUNCATED
    assert aset.by_kind(A.ANCHOR_PRE_FINAL).missingness == A.MISSING_SPAN_TRUNCATED


def test_empty_final_has_no_answer_content_token(tok):
    aset, prompt, completion = resolve(tok, "\n\n<final></final><confidence>0.5</confidence>")

    assert aset.parse_status == K.ParseStatus.EMPTY_FINAL.value
    assert aset.by_kind(A.ANCHOR_ANSWER_END).missingness == A.MISSING_FINAL_EMPTY
    # the opener and the report end are still real tokens, so they are still resolved
    assert aset.by_kind(A.ANCHOR_PRE_FINAL).present
    assert aset.by_kind(A.ANCHOR_REPORT_END).present
    assert_offset_matches_decode(tok, aset, aset.by_kind(A.ANCHOR_PRE_FINAL), prompt, completion)


def test_missing_final_tag_is_its_own_code(tok):
    aset, _prompt, _completion = resolve(tok, "\n\nI think the answer is 42.")
    assert aset.parse_status == K.ParseStatus.MISSING_FINAL.value
    for kind in (A.ANCHOR_PRE_FINAL, A.ANCHOR_ANSWER_END, A.ANCHOR_REPORT_END):
        assert aset.by_kind(kind).missingness == A.MISSING_FINAL_TAG


def test_no_report_span_when_the_thinking_channel_never_closed(tok):
    """``</think>`` absent → there is no report span at all, and it is typed as such."""
    _messages, prompt = W.prompt_ids(tok, "What is 6 times 7?")
    completion = tok.encode("<think>\nstill reasoning", add_special_tokens=False)
    aset = A.resolve_anchors(prompt_token_ids=prompt, completion_token_ids=completion,
                             finish_reason="length", tokenizer=tok)

    assert aset.report_span is None
    assert aset.parse_status == K.ParseStatus.PREMATURE_REASONING_EOS.value
    assert aset.by_kind(A.ANCHOR_PROMPT_END).present
    for kind in (A.ANCHOR_PRE_FINAL, A.ANCHOR_ANSWER_END, A.ANCHOR_REPORT_END):
        assert aset.by_kind(kind).missingness == A.MISSING_NO_REPORT_SPAN


def test_completion_that_never_opened_think_has_no_report_span(tok):
    _messages, prompt = W.prompt_ids(tok, "What is 6 times 7?")
    completion = tok.encode(W.report_text("42"), add_special_tokens=False)
    aset = A.resolve_anchors(prompt_token_ids=prompt, completion_token_ids=completion,
                             finish_reason="stop", tokenizer=tok)
    assert aset.parse_status == K.ParseStatus.NO_THINK_OPEN.value
    assert aset.by_kind(A.ANCHOR_ANSWER_END).missingness == A.MISSING_NO_REPORT_SPAN


def test_a_missing_anchor_carries_no_vector_and_is_never_a_zero_vector(tok, tmp_path):
    """A missing anchor is a ledger row with ``vector_index = -1`` (spec §14.2/§8.13).

    ``TagAnchor`` makes the invariant structural — ``token_offset is None`` iff a typed
    missingness code is set — and ``capture._row`` turns exactly that into a *None* vector,
    which ``storage.ShardWriter`` stores as a row without a vector.  A zero vector in its
    place is refused by the store, so the degenerate encoding cannot be written by accident.
    """
    import numpy as np

    from agents_scaling.ard.neural import capture as C
    from agents_scaling.study.neural import storage as S

    aset, _prompt, _completion = resolve(tok, "\n\nno tags here at all")
    missing = aset.by_kind(A.ANCHOR_ANSWER_END)
    assert not missing.present

    w = C.CaptureWorkItem.from_row(W.work_row("d0"))
    row, vector = C._row(w=w, anchor=missing, block=8, result=None, nonce="n", revision="rev",
                         checkpoint=K.CHECKPOINT_SMALL, hidden_size=32, extra={})
    assert vector is None
    assert row.missingness == A.MISSING_FINAL_TAG and row.token_offset is None
    assert not row.present

    with S.ShardWriter(tmp_path, "native", "ard36.t000") as writer:
        writer.add(row, vector)
        with pytest.raises(S.StorageError):
            writer.add(row, np.zeros(32, dtype=np.float16))   # never a zero vector

    stored = W.chunk_rows(tmp_path)
    assert [r["anchor_kind"] for r in stored] == [A.ANCHOR_ANSWER_END]
    assert stored[0]["vector_index"] == -1
    assert stored[0]["missingness"] == A.MISSING_FINAL_TAG
    assert stored[0]["tensor_hash"] is None
    matrix, _frame = S.load_stage(tmp_path, "native")
    assert matrix.shape[0] == 0          # no vector at all, not a row of zeros


def test_typed_missingness_codes_are_closed(tok):
    with pytest.raises(ValueError):
        A.TagAnchor(A.ANCHOR_PRE_FINAL, None, "SOMETHING_ELSE", A.CHANNEL_NONE, "", 0)
    with pytest.raises(ValueError):
        A.TagAnchor(A.ANCHOR_PRE_FINAL, 3, "NOT_REACHED", A.CHANNEL_NONE, "", 0)
    with pytest.raises(ValueError):
        A.TagAnchor(A.ANCHOR_PRE_FINAL, None, None, A.CHANNEL_NONE, "", 0)
    with pytest.raises(ValueError):
        A.TagAnchor("SOME_OTHER_ANCHOR", 3, None, A.CHANNEL_NONE, "", 0)


# --------------------------------------------------------------------------- exactly four


def test_a_captured_decision_carries_exactly_four_anchor_rows(tok):
    for content in (W.report_text("42"), W.report_text("42", confidence=None),
                    "\n\n<final>1</final><final>2</final>", "\n\nnothing"):
        aset, _prompt, _completion = resolve(tok, content)
        assert len(aset.anchors) == len(K.CAPTURE_ANCHORS) == 4
        assert tuple(a.kind for a in aset.anchors) == tuple(K.CAPTURE_ANCHORS)
        assert len({a.kind for a in aset.anchors}) == 4
        assert len(aset.missing) + len(aset.present_offsets) >= 1


def test_assert_four_anchors_refuses_a_short_or_reordered_set(tok):
    aset, _prompt, _completion = resolve(tok, W.report_text("42"))
    A.assert_four_anchors(aset.anchors)
    with pytest.raises(A.AnchorError):
        A.assert_four_anchors(aset.anchors[:3])
    with pytest.raises(A.AnchorError):
        A.assert_four_anchors(tuple(reversed(aset.anchors)))
    with pytest.raises(A.AnchorError):
        A.assert_four_anchors(aset.anchors + (aset.anchors[0],))
    with pytest.raises(A.AnchorError):
        A.AnchorSet(decision_id="d", prompt_tokens=3, completion_tokens=3,
                    anchors=aset.anchors[:2], parse_status="ok", markers={},
                    report_end_source=A.REPORT_END_NONE, truncated=False)


# --------------------------------------------------------------------------- parser agreement


def test_the_resolver_refuses_to_disagree_with_the_frozen_parser(tok):
    """One definition of a valid ``<final>`` pair: the parser's.  A mismatch raises."""
    _messages, prompt = W.prompt_ids(tok, "What is 6 times 7?")
    completion = W.completion_ids(tok, W.report_text("42"))
    other = R.parse_report("\n\nno tags at all", "stop", completion)
    with pytest.raises(A.AnchorError) as excinfo:
        A.resolve_anchors(prompt_token_ids=prompt, completion_token_ids=completion,
                          finish_reason="stop", tokenizer=tok, report=other)
    assert "parse_report" in str(excinfo.value)


def test_a_decision_without_prompt_tokens_is_refused(tok):
    with pytest.raises(A.AnchorError):
        A.resolve_anchors(prompt_token_ids=(), completion_token_ids=(1, 2),
                          finish_reason="stop", tokenizer=tok)


def test_tag_span_matches_the_frozen_extractor_statuses():
    for text, expected in [
        ("<final>42</final>", R.TAG_OK),
        ("<final></final>", R.TAG_EMPTY),
        ("", R.TAG_MISSING),
        ("<final>a</final><final>b</final>", R.TAG_MALFORMED),
        ("</final>42<final>", R.TAG_MALFORMED),
        ("<final>42", R.TAG_MALFORMED),
    ]:
        mine = A.tag_span(text, K.FINAL_OPEN, K.FINAL_CLOSE)[0]
        theirs = R._extract(text, K.FINAL_OPEN, K.FINAL_CLOSE)[0]
        assert mine == theirs == expected, text


def test_confidence_inside_evidence_is_never_read_as_the_forecast(tok):
    """Spec §6.4: a probability written in the evidence prose is evidence, not the report."""
    body = ("\n\n<final>42</final>"
            "<evidence>we were <confidence>0.99</confidence> sure</evidence>")
    aset, _prompt, _completion = resolve(tok, body)
    assert aset.markers["confidence"] is None
    assert aset.report_end_source == A.REPORT_END_FINAL_CLOSE


# --------------------------------------------------------------------------- timing labels


def test_feature_time_follows_the_contract_anchor_sets():
    assert set(K.PROSPECTIVE_ANCHORS) == {A.ANCHOR_PROMPT_END, A.ANCHOR_PRE_FINAL}
    for kind in K.CAPTURE_ANCHORS:
        expected = A.FEATURE_TIME_PRE if kind in K.PROSPECTIVE_ANCHORS else A.FEATURE_TIME_POST
        assert A.feature_time_of(kind) == expected
    with pytest.raises(ValueError):
        A.feature_time_of("NOT_AN_ANCHOR")


def test_last_allowed_round_is_the_information_horizon():
    for round_ in K.ROUNDS:
        assert A.last_allowed_round(A.ANCHOR_PROMPT_END, round_) == round_ - 1
        assert A.last_allowed_round(A.ANCHOR_PRE_FINAL, round_) == round_ - 1
        assert A.last_allowed_round(A.ANCHOR_ANSWER_END, round_) == round_
        assert A.last_allowed_round(A.ANCHOR_REPORT_END, round_) == round_


def test_entropy_anchors_are_exactly_the_two_contract_anchors(tok):
    assert set(K.ENTROPY_ANCHORS) == {A.ANCHOR_PROMPT_END, A.ANCHOR_ANSWER_END}
    assert [k for k in K.CAPTURE_ANCHORS if A.is_entropy_anchor(k)] == sorted(
        K.ENTROPY_ANCHORS, key=list(K.CAPTURE_ANCHORS).index)

    aset, _prompt, _completion = resolve(tok, W.report_text("42"))
    offsets = A.entropy_anchor_offsets(aset.anchors)
    assert offsets == tuple(sorted({aset.by_kind(A.ANCHOR_PROMPT_END).token_offset,
                                    aset.by_kind(A.ANCHOR_ANSWER_END).token_offset}))
    assert aset.by_kind(A.ANCHOR_PRE_FINAL).token_offset not in offsets
    assert aset.by_kind(A.ANCHOR_REPORT_END).token_offset not in offsets

    # a missing entropy anchor contributes no offset (no vector, no logit row)
    aset2, _p, _c = resolve(tok, "\n\nnothing")
    assert A.entropy_anchor_offsets(aset2.anchors) == (aset2.by_kind(A.ANCHOR_PROMPT_END).token_offset,)


def test_k_max_is_the_deepest_present_anchor(tok):
    aset, prompt, completion = resolve(tok, W.report_text("42"))
    assert aset.k_max == aset.by_kind(A.ANCHOR_REPORT_END).generated_token_count
    assert aset.k_max <= len(completion)
    prompt_only, _p, _c = resolve(tok, "\n\nnothing")
    assert prompt_only.k_max == 0     # only PROMPT_END survives: a prompt-only replay


def test_report_span_accounting_matches_the_client(tok):
    completion = W.completion_ids(tok, W.report_text("42"))
    start, kept = A.report_span(completion)
    assert completion[start - 1] == K.THINK_END_ID
    assert completion[-1] == K.IM_END_ID and K.IM_END_ID not in kept
    assert tok.decode(kept) == W.report_text("42")
    assert len(kept) <= K.FINAL_TOKENS
    assert A.report_span(tok.encode("no think", add_special_tokens=False)) is None
