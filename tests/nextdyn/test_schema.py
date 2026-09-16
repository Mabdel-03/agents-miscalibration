"""The public output: everything noticed is recorded, nothing is repaired (spec 6.1, D6)."""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.schema import ParseStatus, parse_public_output

LIMITS = K.TokenContract(thinking_cap=2048, final_cap=1024, served_window=32768, reserve=256,
                         packet_cap=512, rationale_cap=256, claims_max=4, claim_cap=64)


def words(text: str) -> int:
    return len(text.split())


def _obj(**fields) -> str:
    return json.dumps(fields)


def _ok(**overrides) -> str:
    fields = {"reasoning": "because", "shareable_claims": ["a claim"], "final_answer": "B",
              "confidence": 0.75}
    fields.update(overrides)
    return _obj(**fields)


def parse(text: str, finish: str = schema.FINISH_STOP, ids=(1, 2, 3)):
    return parse_public_output(text, finish, ids, words, limits=LIMITS)


# ---------------------------------------------------------------------------- happy path


def test_a_well_formed_object_parses_into_the_declared_shape():
    out = parse(_ok())
    assert out.parse_status is ParseStatus.OK
    assert (out.reasoning, out.shareable_claims, out.final_answer, out.confidence) == \
        ("because", ("a claim",), "B", 0.75)
    assert out.field_order == schema.DECLARED_FIELD_ORDER
    assert out.valid and not out.order_violation and not out.any_over_limit


def test_field_order_violation_is_recorded_not_discarded():
    """Spec 6.1: "Record field-order violations without discarding an otherwise gradeable
    answer."."""
    out = parse(_obj(final_answer="B", confidence=0.4, reasoning="because",
                     shareable_claims=["a claim"]))
    assert out.order_violation is True
    assert out.parse_status is ParseStatus.OK_ORDER_VIOLATION
    assert out.parse_status in schema.VALID_STATUSES
    assert out.valid and out.final_answer == "B" and out.confidence == 0.4
    assert out.field_order == ("final_answer", "confidence", "reasoning", "shareable_claims")


def test_missing_confidence_is_null_never_half():
    for text in (_obj(reasoning="r", shareable_claims=[], final_answer="B"),
                 _ok(confidence=None), _ok(confidence="high"), _ok(confidence=1.5),
                 _ok(confidence=-0.1), _ok(confidence=True)):
        out = parse(text)
        assert out.confidence is None, text
        assert out.valid


def test_a_confidence_of_zero_is_a_statement_and_survives():
    assert parse(_ok(confidence=0.0)).confidence == 0.0


def test_answer_end_is_pre_confidence_only_when_confidence_not_yet_emitted():
    """Spec 13.2: an ANSWER_END capture is eligible for pre-confidence analysis only when
    confidence has not already been emitted."""
    after = parse(_ok())
    assert after.confidence_emitted_before_answer is False
    assert after.answer_end_pre_confidence_eligible is True

    before = parse(_obj(reasoning="r", shareable_claims=[], confidence=0.9, final_answer="B"))
    assert before.confidence_emitted_before_answer is True
    assert before.answer_end_pre_confidence_eligible is False


def test_the_answer_and_confidence_byte_spans_point_at_their_own_values():
    text = _ok()
    out = parse(text)
    raw = text.encode("utf-8")
    start, end = out.final_answer_span
    assert raw[start:end].decode() == '"B"'
    cstart, cend = out.confidence_span
    assert raw[cstart:cend].decode() == "0.75"


def test_a_key_name_inside_a_string_value_does_not_move_a_span():
    text = _obj(reasoning='the word final_answer appears here', shareable_claims=[],
                final_answer="B", confidence=0.5)
    out = parse(text)
    start, end = out.final_answer_span
    assert text.encode()[start:end].decode() == '"B"'


def test_over_limit_flags_are_flags_and_never_rewrites():
    """Spec 6.1 limits are recorded, never enforced by truncating the stored output; the
    512-token transmitted cap is the only place anything is clipped."""
    long_reasoning = "word " * (LIMITS.rationale_cap + 10)
    out = parse(_ok(reasoning=long_reasoning))
    assert out.over_limits[schema.OVER_RATIONALE] is True
    assert out.reasoning == long_reasoning
    assert out.parse_status is ParseStatus.OK_TRUNCATED_FIELDS and out.valid

    many = parse(_ok(shareable_claims=["a"] * (LIMITS.claims_max + 1)))
    assert many.over_limits[schema.OVER_CLAIM_COUNT] is True
    assert len(many.shareable_claims) == LIMITS.claims_max + 1

    wide = parse(_ok(shareable_claims=["word " * (LIMITS.claim_cap + 2)]))
    assert wide.over_limits[schema.OVER_CLAIM_TOKENS] is True


def test_the_limits_come_from_the_contract_not_from_the_module():
    tight = K.TokenContract(thinking_cap=2048, final_cap=1024, served_window=32768, reserve=256,
                            packet_cap=512, rationale_cap=1, claims_max=1, claim_cap=1)
    out = parse_public_output(_ok(), schema.FINISH_STOP, (1,), words, limits=tight)
    assert out.over_limits[schema.OVER_RATIONALE] is False   # "because" is one word
    assert parse_public_output(_ok(reasoning="two words"), schema.FINISH_STOP, (1,), words,
                               limits=tight).over_limits[schema.OVER_RATIONALE] is True


# ---------------------------------------------------------------------------- failures


def test_each_failure_shape_gets_its_own_status():
    assert parse("no json at all here").parse_status is ParseStatus.NO_JSON
    assert parse("{not valid json").parse_status is ParseStatus.INVALID_JSON
    assert parse("[1, 2]").parse_status is ParseStatus.NO_JSON
    assert parse(_obj(reasoning="r")).parse_status is ParseStatus.MISSING_FINAL
    assert parse("", schema.FINISH_LENGTH).parse_status is ParseStatus.MISSING_FINAL
    assert parse("", schema.FINISH_STOP, ()).parse_status is ParseStatus.PREMATURE_REASONING_EOS
    assert parse('{"final_answer": "B"', schema.FINISH_LENGTH).parse_status is ParseStatus.MISSING_FINAL


def test_an_empty_final_answer_is_not_an_answer():
    out = parse(_ok(final_answer=""))
    assert out.final_answer is None and not out.valid


def test_a_failed_parse_keeps_the_raw_bytes_and_their_hash():
    out = parse("garbage")
    assert out.raw_text == "garbage"
    assert out.content_hash.startswith("c:")
    assert out.final_answer_span is None and out.confidence_span is None


def test_the_parser_never_assigns_a_producer_status():
    statuses = {parse(t).parse_status for t in ("garbage", _ok(), "", "{}")}
    assert not statuses & schema.PRODUCER_ASSIGNED_STATUSES


def test_the_typed_context_failure_output_is_not_gradeable():
    out = schema.context_failure_output()
    assert out.parse_status is ParseStatus.CONTEXT_FAILURE
    assert not out.valid and out.final_answer is None and out.confidence is None


# ---------------------------------------------------------------------------- salvage rule


def test_parse_salvage_rule_is_source_blind_on_adversarial_fixtures():
    """The one salvage rule looks at the *shape* of the text and never at what it says, so a
    right answer and a wrong answer in the same wrapper are salvaged identically."""
    right, wrong = _ok(final_answer="B"), _ok(final_answer="A")
    for wrapper in ("```json\n{body}\n```", "```\n{body}\n```", "  ```json\n{body}\n```  "):
        got = [parse(wrapper.format(body=body)) for body in (right, wrong)]
        assert [g.parse_status for g in got] == [ParseStatus.OK, ParseStatus.OK]
        assert [g.final_answer for g in got] == ["B", "A"]

    # every non-salvageable shape is refused for both answers alike
    adversarial = (
        "here is my answer: {body}",                 # prose before the object
        "{body}\nI hope that helps.",                # prose after the object
        "```json\n{body}\n```\n```json\n{body}\n```",  # two fences
        "text ```json\n{body}\n``` text",            # a fence that is not the whole text
    )
    for shape in adversarial:
        statuses = {parse(shape.format(body=body)).parse_status for body in (right, wrong)}
        assert len(statuses) == 1, shape
        assert statuses <= {ParseStatus.NO_JSON, ParseStatus.INVALID_JSON}, shape


def test_the_salvaged_spans_are_offsets_into_the_original_text():
    text = "```json\n" + _ok() + "\n```"
    out = parse(text)
    start, end = out.final_answer_span
    assert text.encode()[start:end].decode() == '"B"'


def test_the_parser_never_reads_the_thinking_span():
    """The parser is handed the final text only; a ``</think>`` inside the public rationale
    cannot move anything, because the split already happened on token ids."""
    out = parse(_ok(reasoning="I thought </think> about it"))
    assert out.reasoning == "I thought </think> about it" and out.valid


# ---------------------------------------------------------------------------- native split


class _Rec:
    def __init__(self, ids):
        self.completion_token_ids = tuple(ids)


def test_split_native_reasoning_cuts_on_token_ids_not_on_text(stub_tokenizer):
    native_text, final_text = "hidden chain of thought", _ok()
    ids = [*stub_tokenizer.encode(schema.THINK_OPEN), *stub_tokenizer.encode(native_text),
           *stub_tokenizer.encode(schema.THINK_CLOSE), *stub_tokenizer.encode(final_text)]
    native, final = schema.split_native_reasoning(_Rec(ids), stub_tokenizer)
    assert native == native_text
    assert final == stub_tokenizer.decode(stub_tokenizer.encode(final_text))
    assert schema.THINK_CLOSE not in native


def test_a_completion_without_the_closing_delimiter_is_all_final_text(stub_tokenizer):
    ids = stub_tokenizer.encode("just the answer")
    native, final = schema.split_native_reasoning(_Rec(ids), stub_tokenizer)
    assert native == "" and final == "just the answer"


def test_native_and_public_reasoning_fields_not_confused(stub_tokenizer):
    """Spec 6.1: the public ``reasoning`` field is a rationale; the native thinking text is
    stored separately and never becomes the public field."""
    public = _ok(reasoning="public rationale")
    ids = [*stub_tokenizer.encode(schema.THINK_OPEN), *stub_tokenizer.encode("private chain"),
           *stub_tokenizer.encode(schema.THINK_CLOSE), *stub_tokenizer.encode(public)]
    native, final = schema.split_native_reasoning(_Rec(ids), stub_tokenizer)
    out = parse_public_output(final, schema.FINISH_STOP, ids, words, limits=LIMITS)
    assert out.reasoning == "public rationale"
    assert native == "private chain"
    assert native not in (out.reasoning or "") and native not in out.raw_text


def test_an_unknown_over_limits_key_is_refused():
    with pytest.raises(ContractViolation):
        schema.PublicOutput(reasoning=None, shareable_claims=(), final_answer="B", confidence=None,
                            field_order=(), order_violation=False, parse_status=ParseStatus.OK,
                            raw_text="", content_hash="c:0", over_limits={"invented": True},
                            final_answer_span=None, confidence_span=None,
                            confidence_emitted_before_answer=False)


def test_a_numeric_final_answer_is_an_answer():
    """``{"final_answer": 314}`` is how a model writes a MATH answer, and it was thrown away.

    The field was required to be a ``str``, so a complete, well-formed object whose only
    offence was a bare number scored MISSING_FINAL.  Measured on the smoke's first 144 bank
    draws: 20 of them, 14%.  That loss is not random -- it concentrates on numeric-answer
    tasks -- so it would have inflated ``U`` exactly where MATH lives, widened its Manski
    bounds against the other families, and done it silently.
    """
    def answered(answer_json: str):
        return parse('{"reasoning": "because", "shareable_claims": ["a claim"], '
                     '"final_answer": %s, "confidence": 0.75}' % answer_json)

    assert answered("314").final_answer == "314"
    assert answered("314").parse_status is ParseStatus.OK
    assert answered("3.5").final_answer == "3.5"
    assert answered('"314"').final_answer == "314"
    # zero is an answer, and it is falsey -- the shape of this bug is a bare truthiness test
    assert answered("0").final_answer == "0"
    assert answered("0").parse_status is ParseStatus.OK
    # and the things that are still not answers
    for not_an_answer in ("true", "false", "null", "[1]", '{"a": 1}', '""'):
        assert answered(not_an_answer).final_answer is None, not_an_answer
        assert answered(not_an_answer).parse_status is ParseStatus.MISSING_FINAL
    # the span still points at the value, so the capture anchors keep working
    assert answered("314").final_answer_span is not None


def test_the_escape_salvage_rule_is_source_blind_on_adversarial_fixtures():
    """Salvage rule 2 (amendment A5): unescaped backslashes, repaired blind to content.

    Measured on the smoke's first 159 bank draws: 18% of MATH draws were unparseable and every
    one for this reason -- ordinary mathematical prose (``$ a \\equiv -1 \\mod m $``) inside the
    ``reasoning`` string, where ``\\e`` and ``\\m`` are not JSON escapes.  MMLU-Pro and
    TruthfulQA measured 1.000, so the loss sat in one family and would have made the H2
    cross-family contrast partly an artifact of JSON escaping.
    """
    def latex(answer: str, claim: str = "the LCM of 3, 5, 7, 9 is 315"):
        return ('{"reasoning": "rewrite as $ a \\equiv -1 \\mod m $ for each $ m $", '
                '"shareable_claims": ["%s"], "final_answer": "%s", "confidence": 0.9}'
                % (claim, answer))

    right, wrong = parse(latex("8880")), parse(latex("0"))
    assert right.parse_status is ParseStatus.OK and wrong.parse_status is ParseStatus.OK
    assert (right.final_answer, wrong.final_answer) == ("8880", "0")
    # source-blind: the repair depended on the backslashes, not on which answer was given
    assert right.escape_repaired and wrong.escape_repaired
    # the repair is information-preserving: the backslash the model meant is still there
    assert "\\equiv" in right.reasoning and "\\mod" in right.reasoning

    # already-valid JSON is never touched, because the rule is reached only after a failure
    clean = parse(_ok())
    assert clean.parse_status is ParseStatus.OK and not clean.escape_repaired
    # valid escapes survive a repaired document byte-for-byte
    mixed = parse('{"reasoning": "a \\equiv b, then a line\\nbreak and a \\"quote\\"", '
                  '"shareable_claims": [], "final_answer": "7", "confidence": 0.5}')
    assert mixed.parse_status is ParseStatus.OK and mixed.escape_repaired
    assert "\n" in mixed.reasoning and '"quote"' in mixed.reasoning
    assert "\\equiv" in mixed.reasoning

    # and what is still a failure stays one: the rule repairs escapes, it does not invent JSON
    for hopeless in ('{"reasoning": "x", ', "not json at all", '{"reasoning": }', "{,}"):
        assert parse(hopeless).parse_status in (ParseStatus.INVALID_JSON, ParseStatus.NO_JSON)


def test_a_repaired_draw_still_reports_its_spans_against_the_generated_text():
    """The repair shifts offsets; the activation anchors must not follow it.

    ``final_answer_span`` points into the token stream that was actually generated, so it is
    measured on the original body -- a span computed on the repaired string would drift by one
    character per doubled backslash and the capture would read the wrong tokens.
    """
    out = parse('{"reasoning": "a \\equiv b and \\mod c", "shareable_claims": [], '
                '"final_answer": "314", "confidence": 0.9}')
    assert out.escape_repaired and out.final_answer == "314"
    assert out.final_answer_span is not None
    raw = out.raw_text.encode("utf-8")
    start, end = out.final_answer_span
    assert raw[start:end].decode("utf-8").strip().strip('"') == "314"
