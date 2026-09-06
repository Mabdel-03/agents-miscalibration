"""§3.5 complete-candidate parser: one-fence salvage, strict JSON, six-key schema (WP3).

Spec §3.5: "The only textual salvage is stripping one enclosing Markdown fence and
surrounding whitespace followed by a strict JSON parse.  Reject duplicate keys, trailing
objects/text, nonfinite values, missing fields and ambiguous channels.  No repair model is
called to make an unsuccessful opportunity disappear."  Bounds: approach ≤ 16,384 code
points; ≤ 32 evidence/alternatives/checks; claim/support/list strings ≤ 8,192;
final_answer ≤ 131,072; finite confidence in [0,1].  Canonical bytes are RFC 8785 JSON
(§3.5 "Use RFC 8785 canonical JSON, preserving UTF-8 string content").

Architecture §1.8; audit §2.0 item 1 (parser recipe) and fixture T2; corrections P1-5 /
amendment E3' (``strip_one_fence`` is the *one* frozen code-string rule shared by
``selection.normalize.code_key`` and ``evaluation/bcb.py``); 05_vllm_response_shape.md
(``content`` is the post-reasoning-parser channel and may start with "\\n\\n"; guided JSON is
not enforced, so this parser is the only gate).

Failure codes (``FAILURE_CODES``): ``TRUNCATED`` (``finish_reason == "length"``), ``EMPTY``,
``NOT_JSON``, ``DUPLICATE_KEY``, ``NONFINITE``, ``TRAILING_TEXT``, ``SCHEMA``,
``AMBIGUOUS_CHANNEL`` (a six-key object is present only in the reasoning channel).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.types import (
    CANDIDATE_LIMITS,
    UNCERTAINTY_LEVELS,
    Candidate,
    Evidence,
)

# --------------------------------------------------------------------------- constants

FAILURE_CODES: tuple[str, ...] = (
    "TRUNCATED",
    "EMPTY",
    "NOT_JSON",
    "DUPLICATE_KEY",
    "NONFINITE",
    "TRAILING_TEXT",
    "SCHEMA",
    "AMBIGUOUS_CHANNEL",
)

#: The six §3.5 keys in wire order (``Candidate`` field order).
CANDIDATE_KEYS: tuple[str, ...] = (
    "approach",
    "evidence",
    "alternatives_considered",
    "failure_checks",
    "final_answer",
    "confidence",
)
EVIDENCE_KEYS: tuple[str, ...] = ("claim", "support", "uncertainty")

#: vLLM finish reasons the parser accepts; anything else is an infrastructure fact the
#: client must have classified already (05_vllm_response_shape.md).
FINISH_STOP = "stop"
FINISH_LENGTH = "length"

#: Frozen E3' fence rule: an opening fence is ``\`\`\``, an optional info string, optional
#: blanks and a newline; the closing fence is ``\`\`\`` (optionally preceded by blanks or a
#: newline) at the very end.  Both must be present for a strip to happen (never one side).
_OPEN_FENCE = re.compile(r"^```[ \t]*[A-Za-z0-9_+.#-]*[ \t]*\r?\n")
_CLOSE_FENCE = re.compile(r"[ \t]*```[ \t]*$")

#: Bound on the number of ``{`` positions probed in the reasoning channel (defensive).
_MAX_REASONING_PROBES = 512


# --------------------------------------------------------------------------- errors


class StrictJSONError(ValueError):
    """Strict JSON loading failed; ``code`` is one of the parser failure codes."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code {code!r}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class SchemaError(StrictJSONError):
    """A well-formed JSON value violates a frozen schema (code ``SCHEMA``)."""

    def __init__(self, detail: str) -> None:
        super().__init__("SCHEMA", detail)


class _DuplicateKey(Exception):
    pass


class _NonFinite(Exception):
    pass


# --------------------------------------------------------------------------- strict JSON


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise _NonFinite(name)


_DECODER = json.JSONDecoder(object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)


def strip_one_fence(text: str) -> str:
    """Strip surrounding whitespace and *exactly one* enclosing Markdown fence (§3.5, E3').

    The fence is removed only when both an opening fence line (```` ``` ```` plus an
    optional info string such as ``json`` or ``python``) and a closing ```` ``` ```` at the
    end are present; the inner text is then whitespace-stripped once more.  A second
    fence inside is left in place (and later fails the JSON parse or the AST parse).
    This is the single frozen code-string rule used for BCB grouping and testing (P1-5).
    """
    if not isinstance(text, str):
        raise TypeError("strip_one_fence expects str")
    stripped = text.strip()
    opening = _OPEN_FENCE.match(stripped)
    if opening is None:
        return stripped
    rest = stripped[opening.end():]
    closing = _CLOSE_FENCE.search(rest)
    if closing is None:
        return stripped
    return rest[: closing.start()].strip()


def load_strict_json(text: str) -> Any:
    """Parse ``text`` (already fence-stripped) as one strict JSON value.

    Raises :class:`StrictJSONError` with code ``EMPTY`` (nothing left), ``NOT_JSON``,
    ``DUPLICATE_KEY`` (any object level), ``NONFINITE`` (``NaN``/``Infinity`` literals or a
    number literal overflowing to infinity) or ``TRAILING_TEXT`` (non-whitespace after the
    first complete value).  Never returns a repaired value.  Every ``ValueError`` the decoder
    raises is ``NOT_JSON`` — ``JSONDecodeError`` and CPython's int-digit guard alike (an
    integer literal above 4,300 digits is a completed model outcome, never a harness crash;
    MUST 2.0-1, review P1-A).
    """
    if not isinstance(text, str):
        raise TypeError("load_strict_json expects str")
    if text.strip() == "":
        raise StrictJSONError("EMPTY", "no content after whitespace/fence stripping")
    try:
        value, end = _DECODER.raw_decode(text, 0)
    except _DuplicateKey as exc:
        raise StrictJSONError("DUPLICATE_KEY", f"duplicate key {exc.args[0]!r}") from None
    except _NonFinite as exc:
        raise StrictJSONError("NONFINITE", f"non-finite literal {exc.args[0]}") from None
    except (ValueError, RecursionError) as exc:  # JSONDecodeError ⊂ ValueError; also the 4,300-digit guard
        raise StrictJSONError("NOT_JSON", str(exc)) from None
    if text[end:].strip() != "":
        raise StrictJSONError("TRAILING_TEXT", f"{len(text) - end} trailing characters")
    _check_finite(value)
    return value


def _check_finite(value: Any) -> None:
    """Number literals such as ``1e999`` parse to ``inf`` without a constant hook."""
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJSONError("NONFINITE", "number literal overflowed to a non-finite float")
    if isinstance(value, list):
        for item in value:
            _check_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _check_finite(item)


# --------------------------------------------------------------------------- schema checks


def check_str(value: Any, name: str, max_len: int) -> str:
    """A JSON string of at most ``max_len`` code points that encodes to UTF-8 (no lone surrogates)."""
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string, got {type(value).__name__}")
    if len(value) > max_len:
        raise SchemaError(f"{name} has {len(value)} code points > {max_len}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise SchemaError(f"{name} contains a lone surrogate (not encodable as UTF-8)") from None
    return value


def check_str_list(value: Any, name: str, max_items: int, max_len: int) -> tuple[str, ...]:
    """A JSON array of at most ``max_items`` strings, each ≤ ``max_len`` code points."""
    if not isinstance(value, list):
        raise SchemaError(f"{name} must be an array, got {type(value).__name__}")
    if len(value) > max_items:
        raise SchemaError(f"{name} has {len(value)} items > {max_items}")
    return tuple(check_str(item, f"{name}[{i}]", max_len) for i, item in enumerate(value))


def check_unit_number(value: Any, name: str) -> float:
    """A non-bool finite JSON number in [0, 1].

    The range check runs on the decoded value itself: an integer literal too large for a
    float (``math.isfinite``/``float()`` would raise ``OverflowError``) is a ``SCHEMA``
    failure like any other out-of-range number, never a harness crash (review P1-A).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError(f"{name} is not finite")
    if not 0 <= value <= 1:
        shown = repr(value) if isinstance(value, float) or abs(value) < 10**20 else f"an integer with {len(str(abs(value)))} digits"
        raise SchemaError(f"{name}={shown} outside [0,1]")
    return float(value)


def check_exact_keys(value: Any, name: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    """A JSON object whose key set is exactly ``keys`` (``additionalProperties: false``)."""
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be an object, got {type(value).__name__}")
    missing = [k for k in keys if k not in value]
    extra = [k for k in value if k not in keys]
    if missing:
        raise SchemaError(f"{name} missing keys {missing}")
    if extra:
        raise SchemaError(f"{name} has unexpected keys {extra}")
    return value


def validate_candidate_object(obj: Any) -> Candidate:
    """Validate a decoded JSON value against the §3.5 schema; return the typed ``Candidate``.

    Raises :class:`SchemaError` on any deviation (exact six keys, evidence records with
    exactly ``claim``/``support``/``uncertainty``, the frozen bounds, finite non-bool
    confidence in [0,1]).  ``confidence`` written as an integer ``0``/``1`` is a valid JSON
    number and is stored as ``float``; its canonical bytes are identical either way.
    """
    body = check_exact_keys(obj, "candidate", CANDIDATE_KEYS)
    approach = check_str(body["approach"], "approach", CANDIDATE_LIMITS["approach"])
    raw_evidence = body["evidence"]
    if not isinstance(raw_evidence, list):
        raise SchemaError(f"evidence must be an array, got {type(raw_evidence).__name__}")
    if len(raw_evidence) > CANDIDATE_LIMITS["list_items"]:
        raise SchemaError(f"evidence has {len(raw_evidence)} items > {CANDIDATE_LIMITS['list_items']}")
    evidence: list[Evidence] = []
    for i, item in enumerate(raw_evidence):
        record = check_exact_keys(item, f"evidence[{i}]", EVIDENCE_KEYS)
        claim = check_str(record["claim"], f"evidence[{i}].claim", CANDIDATE_LIMITS["list_string"])
        support = check_str(record["support"], f"evidence[{i}].support", CANDIDATE_LIMITS["list_string"])
        uncertainty = record["uncertainty"]
        if not isinstance(uncertainty, str) or uncertainty not in UNCERTAINTY_LEVELS:
            raise SchemaError(f"evidence[{i}].uncertainty={uncertainty!r} not in {UNCERTAINTY_LEVELS}")
        evidence.append(Evidence(claim=claim, support=support, uncertainty=uncertainty))
    alternatives = check_str_list(
        body["alternatives_considered"],
        "alternatives_considered",
        CANDIDATE_LIMITS["list_items"],
        CANDIDATE_LIMITS["list_string"],
    )
    checks = check_str_list(
        body["failure_checks"], "failure_checks", CANDIDATE_LIMITS["list_items"], CANDIDATE_LIMITS["list_string"]
    )
    final_answer = check_str(body["final_answer"], "final_answer", CANDIDATE_LIMITS["final_answer"])
    confidence = check_unit_number(body["confidence"], "confidence")
    return Candidate(
        approach=approach,
        evidence=tuple(evidence),
        alternatives_considered=alternatives,
        failure_checks=checks,
        final_answer=final_answer,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- RFC 8785 with numbers


def es6_number(value: int | float) -> str:
    """RFC 8785 §3.2.2.3 number serialization (ECMAScript ``Number::toString``) for finite values.

    ``identity.jcs`` refuses floats (nothing in a request identity may carry one); the
    candidate's ``confidence`` is the only float that enters a canonical hash, so this
    module implements the ES6 rule for it: shortest round-trip digits, ``1`` not ``1.0``,
    ``1e-7`` not ``1e-07``, ``-0`` → ``0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("es6_number expects an int or float")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number cannot be canonicalized")
    if number == 0:
        return "0"
    sign, digits_tuple, exponent = Decimal(repr(number)).as_tuple()
    digits = "".join(map(str, digits_tuple)).lstrip("0") or "0"
    # normalize: strip trailing zeros into the exponent
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped
    k = len(digits)
    n = exponent + k  # value = 0.digits × 10^n
    prefix = "-" if sign else ""
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
        body = mantissa + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return prefix + body


def _jcs_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return es6_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_jcs_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JCS object keys must be str")
            pairs.append((key, item))
        pairs.sort(key=lambda kv: kv[0].encode("utf-16-be"))
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _jcs_value(v) for k, v in pairs) + "}"
    raise TypeError(f"type {type(value).__name__} is not JCS-serializable")


def jcs_with_numbers(obj: Any) -> bytes:
    """RFC 8785 canonical JSON including ES6 number formatting (superset of ``identity.jcs``).

    Byte-identical to :func:`identity.jcs` on float-free input; used only for candidate
    and subtask-result content hashes, never for request identities.
    """
    return _jcs_value(obj).encode("utf-8")


def candidate_canonical_json(candidate: Candidate) -> str:
    """RFC 8785 bytes (as ``str``) of a validated candidate (§3.5)."""
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate_canonical_json expects a Candidate")
    return jcs_with_numbers(candidate.to_dict()).decode("utf-8")


def candidate_sha256(candidate: Candidate) -> str:
    """``CandidateRecord.candidate_sha256`` / ``Packet.candidate_sha256`` of a valid candidate."""
    return identity.sha256_hex(candidate_canonical_json(candidate))


# --------------------------------------------------------------------------- the parser


@dataclass(frozen=True)
class ParsedCandidate:
    """Outcome of :func:`parse_candidate` (the parts of a ``CandidateRecord``).

    ``raw_sha256`` hashes the exact ``content`` string (UTF-8); ``canonical_sha256`` and
    ``canonical_json`` are set only for a valid candidate.  ``detail`` is a human-readable
    reason for the failure code (never fed back to a model).
    """

    valid: bool
    failure_code: str | None
    candidate: Candidate | None
    raw_sha256: str
    canonical_sha256: str | None
    canonical_json: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.valid != (self.failure_code is None) or self.valid != (self.candidate is not None):
            raise ValueError("ParsedCandidate: valid iff failure_code is None iff candidate is set")
        if self.failure_code is not None and self.failure_code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code {self.failure_code!r}")


def _six_key_object_in(text: str) -> bool:
    """True when a strict six-key candidate object can be decoded somewhere in ``text``."""
    probes = 0
    position = text.find("{")
    while position != -1 and probes < _MAX_REASONING_PROBES:
        probes += 1
        try:
            value, _ = _DECODER.raw_decode(text, position)
        except (_DuplicateKey, _NonFinite, ValueError, RecursionError):  # ValueError ⊇ JSONDecodeError, int-digit guard
            value = None
        if isinstance(value, dict) and set(value) == set(CANDIDATE_KEYS):
            return True
        position = text.find("{", position + 1)
    return False


def parse_candidate(content: str | None, finish_reason: str, reasoning: str | None = None) -> ParsedCandidate:
    """Parse one solver ``content`` channel into a §3.5 candidate or a typed failure.

    Order of decisions (each is final, no salvage beyond the one-fence rule):

    1. ``finish_reason == "length"`` → ``TRUNCATED`` (the 8,192 cap is a real outcome, L1).
    2. ``strip_one_fence`` then strict JSON (``EMPTY``/``NOT_JSON``/``DUPLICATE_KEY``/
       ``NONFINITE``/``TRAILING_TEXT``).  When the content yields no JSON value at all
       (``EMPTY``/``NOT_JSON``) but ``reasoning`` contains a six-key object, the failure is
       ``AMBIGUOUS_CHANNEL`` instead (§3.5 "ambiguous channels"; the reasoning channel is
       never parsed as the answer).
    3. :func:`validate_candidate_object` → ``SCHEMA``.

    ``content`` may be ``None`` (vLLM emits ``null`` when nothing follows the think block);
    it hashes as the empty string.  ``finish_reason`` must be ``"stop"`` or ``"length"`` —
    any other value is an infrastructure fact the client must have handled (``ValueError``).
    """
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("content must be a str or None")
    if finish_reason not in (FINISH_STOP, FINISH_LENGTH):
        raise ValueError(f"finish_reason {finish_reason!r} is not a parser outcome (stop|length)")
    if reasoning is not None and not isinstance(reasoning, str):
        raise TypeError("reasoning must be a str or None")
    raw_sha256 = identity.sha256_hex(content)

    def failure(code: str, detail: str) -> ParsedCandidate:
        return ParsedCandidate(False, code, None, raw_sha256, None, None, detail)

    if finish_reason == FINISH_LENGTH:
        return failure("TRUNCATED", "finish_reason=length")
    text = strip_one_fence(content)
    try:
        value = load_strict_json(text)
    except StrictJSONError as exc:
        if exc.code in ("EMPTY", "NOT_JSON") and reasoning and _six_key_object_in(reasoning):
            return failure("AMBIGUOUS_CHANNEL", f"candidate object only in the reasoning channel ({exc.code})")
        return failure(exc.code, exc.detail)
    try:
        candidate = validate_candidate_object(value)
    except SchemaError as exc:
        return failure("SCHEMA", exc.detail)
    canonical = candidate_canonical_json(candidate)
    return ParsedCandidate(True, None, candidate, raw_sha256, identity.sha256_hex(canonical), canonical, None)


__all__ = [
    "CANDIDATE_KEYS",
    "EVIDENCE_KEYS",
    "FAILURE_CODES",
    "FINISH_LENGTH",
    "FINISH_STOP",
    "ParsedCandidate",
    "SchemaError",
    "StrictJSONError",
    "candidate_canonical_json",
    "candidate_sha256",
    "check_exact_keys",
    "check_str",
    "check_str_list",
    "check_unit_number",
    "es6_number",
    "jcs_with_numbers",
    "load_strict_json",
    "parse_candidate",
    "strip_one_fence",
    "validate_candidate_object",
]
