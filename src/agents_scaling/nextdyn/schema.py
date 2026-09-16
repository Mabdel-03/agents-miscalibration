"""The public JSON output: one shape, one parser, one salvage rule (spec 6.1; plan D6, §1.6).

Rule R1 says the output shape lives in exactly one place, and this is it.  ARD36 let three
call sites decide independently what "a valid report" was, which is how a missing confidence
became ``0.5`` in one path and ``None`` in another.  Here every consumer reads
:class:`PublicOutput`, and everything the parser noticed is *recorded* rather than repaired:

* a field-order violation is a boolean on the row, never a reason to discard a gradeable
  answer (spec 6.1, ``test_field_order_violation_is_recorded_not_discarded``);
* the 256-token rationale, four-claim and 64-token-claim limits are **flags**
  (:attr:`PublicOutput.over_limits`), never rewrites -- the transmitted packet is where
  clipping happens, and the original bytes stay in the row (spec 6.1);
* a missing, malformed or out-of-range confidence is ``None``, never ``0.5``
  (``test_missing_confidence_is_null_never_half``);
* the thinking span is never read here.  :func:`split_native_reasoning` separates it from
  the final text once, from token ids, and the parser only ever sees the final text.

The **one** salvage rule (source-blind, documented, tested on adversarial fixtures): if the
final text is not itself JSON, and stripping whitespace leaves exactly one Markdown fenced
block with nothing outside it, the fence body is parsed.  Everything else is a parse failure.
The rule looks only at the *shape* of the text, never at what the fields say, so a fenced
wrong answer and a fenced right answer are salvaged identically
(``test_parse_salvage_rule_is_source_blind_on_adversarial_fixtures``).

Byte spans for ``final_answer`` and ``confidence`` are recorded for the activation anchors
(spec 13.2): ``ANSWER_END`` is eligible for pre-confidence analysis only when confidence had
not yet been emitted, which is exactly
:attr:`PublicOutput.confidence_emitted_before_answer` being false.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import ContractViolation

Counter = Callable[[str], int]

REASONING: Final[str] = "reasoning"
SHAREABLE_CLAIMS: Final[str] = "shareable_claims"
FINAL_ANSWER: Final[str] = "final_answer"
CONFIDENCE: Final[str] = "confidence"
#: The displayed order spec 6.1 requests: rationale, claims, answer, then confidence.
DECLARED_FIELD_ORDER: Final[tuple[str, ...]] = (REASONING, SHAREABLE_CLAIMS, FINAL_ANSWER, CONFIDENCE)

#: ``over_limits`` keys (spec 6.1 limits; flags only, never rewrites).
OVER_RATIONALE: Final[str] = "rationale_tokens"
OVER_CLAIM_COUNT: Final[str] = "claims_count"
OVER_CLAIM_TOKENS: Final[str] = "claim_tokens"
OVER_LIMIT_KEYS: Final[tuple[str, ...]] = (OVER_RATIONALE, OVER_CLAIM_COUNT, OVER_CLAIM_TOKENS)

#: vLLM finish reasons that mean the sampler stopped on its own (a complete response).
FINISH_STOP: Final[str] = "stop"
FINISH_LENGTH: Final[str] = "length"

#: The one salvage shape: a single fenced block and nothing else outside it.
_SINGLE_FENCE = re.compile(r"\A```[^\n`]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)

THINK_OPEN: Final[str] = "<think>"
THINK_CLOSE: Final[str] = "</think>"


class ParseStatus(str, Enum):
    """Why a public output is (or is not) gradeable (spec 6.1, 15.3)."""

    OK = "ok"
    OK_ORDER_VIOLATION = "ok_order_violation"
    OK_TRUNCATED_FIELDS = "ok_truncated_fields"
    NO_JSON = "no_json"
    INVALID_JSON = "invalid_json"
    MISSING_FINAL = "missing_final"
    PREMATURE_REASONING_EOS = "premature_reasoning_eos"
    NO_THINK_OPEN = "no_think_open"
    CONTEXT_FAILURE = "context_failure"


#: Statuses whose row still carries a gradeable answer.
VALID_STATUSES: Final[frozenset[ParseStatus]] = frozenset(
    {ParseStatus.OK, ParseStatus.OK_ORDER_VIOLATION, ParseStatus.OK_TRUNCATED_FIELDS})

#: Statuses this parser never returns: they are assigned by the producer that alone can see
#: the evidence -- the adapter's delimiter split (``NO_THINK_OPEN``) and the build-time
#: context refusal (``CONTEXT_FAILURE``, ``episode.run_episode``).
PRODUCER_ASSIGNED_STATUSES: Final[frozenset[ParseStatus]] = frozenset(
    {ParseStatus.NO_THINK_OPEN, ParseStatus.CONTEXT_FAILURE})


@dataclass(frozen=True)
class PublicOutput:
    """One parsed public JSON object plus everything the parse noticed about it."""

    reasoning: str | None
    shareable_claims: tuple[str, ...]
    final_answer: str | None
    confidence: float | None
    field_order: tuple[str, ...]
    order_violation: bool
    parse_status: ParseStatus
    raw_text: str
    content_hash: str
    over_limits: Mapping[str, bool]
    final_answer_span: tuple[int, int] | None
    confidence_span: tuple[int, int] | None
    confidence_emitted_before_answer: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "shareable_claims", tuple(self.shareable_claims))
        object.__setattr__(self, "field_order", tuple(self.field_order))
        object.__setattr__(self, "parse_status", ParseStatus(self.parse_status))
        flags = dict(self.over_limits)
        unknown = sorted(set(flags) - set(OVER_LIMIT_KEYS))
        if unknown:
            raise ContractViolation(f"unknown over_limits key(s) {unknown}; declared: {OVER_LIMIT_KEYS}")
        object.__setattr__(self, "over_limits", {k: bool(flags.get(k, False)) for k in OVER_LIMIT_KEYS})
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ContractViolation(f"confidence {self.confidence!r} is outside [0, 1]")

    @property
    def valid(self) -> bool:
        """Gradeable: parsed, with a final answer (spec 6.1: validity is not correctness)."""
        return self.parse_status in VALID_STATUSES and self.final_answer is not None

    @property
    def any_over_limit(self) -> bool:
        return any(self.over_limits.values())

    @property
    def answer_end_pre_confidence_eligible(self) -> bool:
        """Spec 13.2: an ``ANSWER_END`` capture is eligible for pre-confidence analysis only
        when confidence had not already been emitted at that point."""
        return self.final_answer_span is not None and not self.confidence_emitted_before_answer


# --------------------------------------------------------------------------- spans


def _byte(text: str, index: int) -> int:
    return len(text[:index].encode("utf-8"))


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _value_end(text: str, start: int) -> int | None:
    """The end index of the JSON value beginning at ``start`` (strings, numbers, literals)."""
    if start >= len(text):
        return None
    if text[start] == '"':
        i = start + 1
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == '"':
                return i + 1
            i += 1
        return None
    i = start
    while i < len(text) and text[i] not in ",}] \t\r\n":
        i += 1
    return i if i > start else None


def _member_span(body: str, key: str) -> tuple[int, int] | None:
    """Character span of the *value* of top-level member ``key``, or ``None``.

    A deliberately small scanner: it walks the object once, tracking string state and depth,
    so a key name that also occurs inside a string value cannot be mistaken for the member.
    """
    depth = 0
    i = 0
    in_string = False
    key_start: int | None = None
    while i < len(body):
        ch = body[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
                if depth == 1 and key_start is not None:
                    name = body[key_start + 1:i]
                    j = _skip_ws(body, i + 1)
                    if j < len(body) and body[j] == ":":
                        j = _skip_ws(body, j + 1)
                        if name == key:
                            end = _value_end(body, j)
                            return None if end is None else (j, end)
                key_start = None
            i += 1
            continue
        if ch == '"':
            in_string = True
            key_start = i
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return None


# --------------------------------------------------------------------------- the parser


def _json_body(final_text: str) -> tuple[str, int] | None:
    """``(JSON source, character offset into final_text)`` under the one salvage rule."""
    stripped = final_text.strip()
    if not stripped:
        return None
    offset = len(final_text) - len(final_text.lstrip())
    if stripped.startswith("{"):
        return stripped, offset
    match = _SINGLE_FENCE.match(stripped)
    if match is None:
        return None
    body = match.group("body")
    return body, offset + match.start("body")


def _claims(value: Any) -> tuple[tuple[str, ...], bool]:
    """``(claims, malformed)``: a non-list or a non-string entry is dropped, never rewritten."""
    if not isinstance(value, list):
        return (), value is not None
    kept = tuple(v for v in value if isinstance(v, str))
    return kept, len(kept) != len(value)


def _confidence(value: Any) -> float | None:
    """Spec 6.1: malformed or absent confidence is null, never 0.5."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def parse_public_output(final_text: str, finish_reason: str, completion_ids: Sequence[int],
                        counter: Counter, *, limits: K.TokenContract) -> PublicOutput:
    """Parse one final-phase completion into the single public output shape.

    ``limits`` carries the spec 6.1 allowances (``rationale_cap``, ``claims_max``,
    ``claim_cap``) because they are *values*, declared once in ``study.yaml`` and compiled
    into :class:`contract.TokenContract`; no cap is a literal in this module.
    ``completion_ids`` and ``finish_reason`` distinguish a model that stopped on its own from
    one that ran out of budget, which is the only thing the token stream is consulted for --
    the thinking span is never read.
    """
    text = final_text if isinstance(final_text, str) else ""
    content_hash = ids.content_hash(text.encode("utf-8"))
    empty = dict(reasoning=None, shareable_claims=(), final_answer=None, confidence=None,
                 field_order=(), order_violation=False, raw_text=text, content_hash=content_hash,
                 over_limits={}, final_answer_span=None, confidence_span=None,
                 confidence_emitted_before_answer=False)

    if not text.strip():
        status = (ParseStatus.MISSING_FINAL if finish_reason == FINISH_LENGTH
                  else ParseStatus.PREMATURE_REASONING_EOS)
        if not completion_ids and finish_reason != FINISH_LENGTH:
            status = ParseStatus.PREMATURE_REASONING_EOS
        return PublicOutput(parse_status=status, **empty)

    found = _json_body(text)
    if found is None:
        return PublicOutput(parse_status=ParseStatus.NO_JSON, **empty)
    body, offset = found
    try:
        parsed = json.loads(body)
    except ValueError:
        status = ParseStatus.MISSING_FINAL if finish_reason == FINISH_LENGTH else ParseStatus.INVALID_JSON
        return PublicOutput(parse_status=status, **empty)
    if not isinstance(parsed, dict):
        return PublicOutput(parse_status=ParseStatus.INVALID_JSON, **empty)

    field_order = tuple(str(k) for k in parsed)
    known = tuple(k for k in field_order if k in DECLARED_FIELD_ORDER)
    expected = tuple(k for k in DECLARED_FIELD_ORDER if k in known)
    order_violation = known != expected

    reasoning = parsed.get(REASONING)
    reasoning = reasoning if isinstance(reasoning, str) else None
    claims, claims_malformed = _claims(parsed.get(SHAREABLE_CLAIMS))
    final_answer = parsed.get(FINAL_ANSWER)
    final_answer = final_answer if isinstance(final_answer, str) and final_answer != "" else None
    confidence = _confidence(parsed.get(CONFIDENCE))

    over_limits = {
        OVER_RATIONALE: reasoning is not None and counter(reasoning) > limits.rationale_cap,
        OVER_CLAIM_COUNT: len(claims) > limits.claims_max or claims_malformed,
        OVER_CLAIM_TOKENS: any(counter(c) > limits.claim_cap for c in claims),
    }

    def span(key: str) -> tuple[int, int] | None:
        found_span = _member_span(body, key)
        if found_span is None:
            return None
        start, end = found_span
        return (_byte(text, offset + start), _byte(text, offset + end))

    final_answer_span = span(FINAL_ANSWER) if final_answer is not None else None
    confidence_span = span(CONFIDENCE) if CONFIDENCE in parsed else None
    emitted_before = (CONFIDENCE in field_order and FINAL_ANSWER in field_order
                      and field_order.index(CONFIDENCE) < field_order.index(FINAL_ANSWER))

    if final_answer is None:
        status = ParseStatus.MISSING_FINAL
    elif any(over_limits.values()):
        status = ParseStatus.OK_TRUNCATED_FIELDS
    elif order_violation:
        status = ParseStatus.OK_ORDER_VIOLATION
    else:
        status = ParseStatus.OK

    return PublicOutput(
        reasoning=reasoning, shareable_claims=claims, final_answer=final_answer,
        confidence=confidence, field_order=field_order, order_violation=order_violation,
        parse_status=status, raw_text=text, content_hash=content_hash, over_limits=over_limits,
        final_answer_span=final_answer_span, confidence_span=confidence_span,
        confidence_emitted_before_answer=emitted_before,
    )


def context_failure_output() -> PublicOutput:
    """The typed output of a slot whose prompt was refused at build time (spec 15.3).

    A context refusal spends the slot's planned opportunity, so the decision exists with no
    request behind it; it is never a retry and never a second sample.
    """
    return PublicOutput(
        reasoning=None, shareable_claims=(), final_answer=None, confidence=None, field_order=(),
        order_violation=False, parse_status=ParseStatus.CONTEXT_FAILURE, raw_text="",
        content_hash=ids.content_hash(b""), over_limits={}, final_answer_span=None,
        confidence_span=None, confidence_emitted_before_answer=False,
    )


# --------------------------------------------------------------------------- native split


def _token_id(tokenizer: Any, token: str) -> int | None:
    getter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if getter is not None:
        try:
            value = getter(token)
        except Exception:  # pragma: no cover - tokenizer-specific
            value = None
        if isinstance(value, int) and value >= 0:
            return value
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    ids_ = encode(token, add_special_tokens=False)
    if isinstance(ids_, Mapping):
        ids_ = ids_["input_ids"]
    return int(ids_[0]) if len(ids_) == 1 else None


def split_native_reasoning(record: Any, tokenizer: Any) -> tuple[str, str]:
    """``(native_reasoning_text, final_text)`` decoded from the completion token ids.

    The delimiter is a *sampled* token, so the split is done on ids and not on text: a model
    that writes the literal ``</think>`` inside its public rationale cannot move the
    boundary.  When the closing delimiter is absent the whole completion is final text and
    the native span is empty -- the ``NO_THINK_OPEN`` case, which the adapter labels.
    """
    completion = tuple(int(t) for t in getattr(record, "completion_token_ids", ()) or ())
    close = _token_id(tokenizer, THINK_CLOSE)
    if close is None:
        raise ContractViolation(
            f"tokenizer does not carry the {THINK_CLOSE!r} delimiter as one token; the native "
            "thinking split cannot be done on ids")
    open_ = _token_id(tokenizer, THINK_OPEN)
    decode = tokenizer.decode
    if close not in completion:
        return "", decode(list(completion))
    cut = completion.index(close)
    native = completion[:cut]
    if open_ is not None and native and native[0] == open_:
        native = native[1:]
    return decode(list(native)), decode(list(completion[cut + 1:]))


__all__ = [
    "CONFIDENCE", "Counter", "DECLARED_FIELD_ORDER", "FINAL_ANSWER", "FINISH_LENGTH",
    "FINISH_STOP", "OVER_CLAIM_COUNT", "OVER_CLAIM_TOKENS", "OVER_LIMIT_KEYS", "OVER_RATIONALE",
    "PRODUCER_ASSIGNED_STATUSES", "ParseStatus", "PublicOutput", "REASONING", "SHAREABLE_CLAIMS",
    "THINK_CLOSE", "THINK_OPEN", "VALID_STATUSES", "context_failure_output",
    "parse_public_output", "split_native_reasoning",
]
