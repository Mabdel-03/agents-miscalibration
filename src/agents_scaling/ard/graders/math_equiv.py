"""MATH-500 symbolic answer grader and gold-independent canonicalizer (WP3).

Spec: §5.1 ("MATH grouping: canonicalize predicted answers using a frozen
parse/normalization procedure independent of the reference.  Grade separately against
the reference with the existing validated symbolic evaluator or a pinned equivalent.
Ambiguous parses/timeouts remain explicit; no solver-model judgment silently substitutes
for the evaluator."), §3 (ambiguous/timeout stay in the denominator but are reported
separately), §18 (``grader_hash`` is part of the frozen item identity).
Plan §9 decision 2: ``lark`` is pinned into ``asys_env`` and drives
``sympy.parsing.latex.parse_latex(..., backend='lark')``; the lark backend is newer than
the retired ANTLR one, so its parse-failure set is a *reported gate metric* with a
hand-rolled fallback for the closed set of failing shapes (recorded in the amendment
register).

Two entry points, deliberately separated:

* :func:`canonical_class` — **gold-independent**.  Used at runtime for voting and for
  BALANCED packet construction.  It never sees a reference answer and never raises.
* :func:`compare` — the evaluator-side equivalence decision between a candidate and the
  reference.  Its status is one of :data:`STATUSES` and is *never* silently
  ``incorrect``: an unparseable or timed-out comparison is ``ambiguous``/``timeout``.

Measured gaps in the lark grammar (2026-09-10, lark 1.3.1 / sympy 1.14.0), each handled
by a hand-rolled rewrite in :func:`normalise` or by the shape router in :func:`parse_answer`:

===========================  ==========================================================
shape                        handling
===========================  ==========================================================
``\\pi``                      absent from ``greek_symbols.lark`` -> parsed through the
                             reserved placeholder :data:`PI_PLACEHOLDER` and substituted
                             back to ``sympy.pi``
``\\text{..}`` / ``\\mbox{..}``  not in the grammar -> TEXT node, compared as a string
``(a,b)`` ``[a,b)`` ``\\{a\\}``  not in the grammar -> SEQ node, compared element-wise
``\\begin{pmatrix}..``        not in the grammar -> SEQ node over ``\\\\``-separated rows
``a \\pm b``                  not in the grammar -> unordered two-element SEQ node
``A \\cup B``                 not in the grammar -> unordered SEQ node over the parts
``^\\circ`` ``\\%`` ``\\$``     stripped before parsing
``\\sqrt2`` (braceless)       rewritten to ``\\sqrt{2}``
``52_8`` ``4210_{5}``         base-n numerals are an atomic string class (converting them
                             to base 10 would grade a base-8 question against a base-10
                             answer)
``\\sigma`` ``\\iota`` ``\\Pi``  still unsupported: reported as an unparsed shape
===========================  ==========================================================
"""

from __future__ import annotations

import math
import random
import re
import signal
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import sympy

from agents_scaling.ard.graders._boxed import extract_boxed

#: Pinned parser backend for ``sympy.parsing.latex.parse_latex``.  Recorded in
#: ``grader_hash`` so a version bump invalidates every stored grade (plan §9 item 2).
PINNED_LARK_VERSION: str = "1.3.1"
LATEX_BACKEND: str = "lark"

STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_TIMEOUT = "timeout"
#: The four statuses of the WP0 interface ``GradeResult.status``.
STATUSES: tuple[str, ...] = (STATUS_CORRECT, STATUS_INCORRECT, STATUS_AMBIGUOUS, STATUS_TIMEOUT)

#: Per-comparison wall-clock ceiling (seconds).  Both parses and the simplification run
#: inside it; exceeding it yields ``timeout``, never ``incorrect``.
DEFAULT_TIMEOUT_S: float = 5.0

#: ``\pi`` is missing from sympy's lark grammar.  ``\upsilon`` is present, is never used
#: in a MATH-500 answer, and round-trips to ``Symbol('upsilon')``, so it is the carrier.
PI_PLACEHOLDER = r"\upsilon"
PI_SYMBOL = sympy.Symbol("upsilon")

#: Node kinds produced by :func:`parse_answer`.
KIND_EMPTY = "empty"
KIND_EXPR = "expr"
KIND_TEXT = "text"
KIND_SEQ = "seq"
KIND_BASE = "base"
KIND_RAW = "raw"

#: Random probe points for the numeric fallback; frozen so the decision is deterministic.
NUMERIC_PROBE_SEED = 0xA3D36
NUMERIC_PROBE_POINTS = 8
NUMERIC_MIN_AGREEMENTS = 3
NUMERIC_RTOL = 1e-9


class GraderTimeout(Exception):
    """Raised when a comparison exceeds its wall-clock ceiling."""


# --------------------------------------------------------------------------- timeout


class _TimeLimit:
    """Wall-clock ceiling using ``SIGALRM`` when it is available.

    ``signal.setitimer`` only works on the main thread of the main interpreter.  Off the
    main thread the guard degrades to a cooperative deadline: :func:`_check_deadline` is
    called between pipeline stages, so a pathological ``simplify`` inside one stage can
    still overrun.  That degradation is recorded on the verdict as ``detail`` rather than
    being hidden.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.deadline = 0.0
        self.enforced = False
        self._previous: Any = None

    def __enter__(self) -> "_TimeLimit":
        self.deadline = time.monotonic() + self.seconds
        if self.seconds > 0 and threading.current_thread() is threading.main_thread():
            try:
                self._previous = signal.signal(signal.SIGALRM, self._fire)
                signal.setitimer(signal.ITIMER_REAL, self.seconds)
                self.enforced = True
            except (ValueError, OSError, AttributeError):  # pragma: no cover - platform
                self.enforced = False
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self.enforced:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, self._previous)
            except (ValueError, OSError, AttributeError):  # pragma: no cover - platform
                pass
        return False

    @staticmethod
    def _fire(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
        raise GraderTimeout("math grader wall-clock ceiling exceeded")

    def check(self) -> None:
        if self.deadline and time.monotonic() > self.deadline:
            raise GraderTimeout("math grader deadline exceeded between stages")


# --------------------------------------------------------------------------- normalise

_WS = re.compile(r"\s+")
_SPACER_COMMANDS = re.compile(r"\\(?:!|,|;|:|quad|qquad|thinspace|medspace|space| )")
_LEFT_RIGHT = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg|mathrm|mathbf|displaystyle)\s*")
_DEGREES = re.compile(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)")
_TEXT_WRAP = re.compile(r"\\(?:text|textbf|textit|mbox|hbox|operatorname)\s*\{([^{}]*)\}")
_THOUSANDS_BRACED = re.compile(r"(?<=\d)\{,\}(?=\d{3})")
_THOUSANDS_PLAIN = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_BRACELESS_SQRT = re.compile(r"\\sqrt\s*(\d)")
_LEADING_EQUATION = re.compile(r"^[A-Za-z](?:_\{?\d+\}?)?\s*=\s*(?=[^=]*$)")
_TRAILING_PUNCT = re.compile(r"[.,;]+$")
_BASE_NUMERAL = re.compile(r"^([0-9A-Fa-f]+)_\{?([0-9]{1,2})\}?$")
_WORD = re.compile(r"^[A-Za-z][A-Za-z '\-]*$")
_LEADING_DOT = re.compile(r"(?<![\d.\w])\.(?=\d)")
_SET_MEMBERSHIP = re.compile(r"^[A-Za-z](?:_\{?\d+\}?)?\s*\\in\s*")
ROW_SEPARATOR_SENTINEL = "\x01"
_BARE_PI = re.compile(r"(?<![\\A-Za-z])pi(?![A-Za-z])")
_ENVIRONMENT = re.compile(r"\\begin\{(pmatrix|bmatrix|vmatrix|matrix|array)\}(.*?)\\end\{\1\}", re.DOTALL)


def _strip_text_groups(text: str) -> str:
    """Unwrap a whole-string ``\\text{...}``; replace a partial one with a space.

    A ``\\text{}`` group that spans the entire answer is prose (``\\text{Evelyn}``,
    ``\\text{even}``) and becomes the bare word.  A group that does not span the answer is a
    unit or annotation (``\\frac{270}7\\text{ degrees}``) and is dropped — as a *space*, so
    that a connective such as ``3\\text{ or }5`` collapses to an unparseable ``3 5`` and is
    reported ambiguous rather than silently becoming ``35``.
    """
    previous = None
    while previous != text:
        previous = text
        match = _TEXT_WRAP.search(text)
        if match is None:
            break
        if match.group(0).strip() == text.strip():
            text = match.group(1).strip()
        else:
            text = (text[: match.start()] + " " + text[match.end() :]).strip()
    return text


def strip_answer_wrapper(text: str) -> str:
    """Return the content of the last ``\\boxed{...}`` if present, else ``text``.

    Uses :func:`agents_scaling.ard.graders._boxed.extract_boxed` (the one brace-balanced
    extractor already validated in this repository) as WP3 requires.
    """
    if not isinstance(text, str):
        raise TypeError("answer must be str")
    boxed = extract_boxed(text)
    return boxed if boxed is not None else text


def normalise(text: str) -> str:
    """Frozen, gold-independent pre-normalisation of a LaTeX answer string.

    Strips ``\\left``/``\\right`` and spacing commands, unwraps ``\\text{}``/``\\mbox{}``,
    drops ``^\\circ``, ``\\$``, ``$`` and ``%``, removes thousands separators, maps
    ``\\dfrac``/``\\tfrac`` to ``\\frac``, braces a braceless ``\\sqrt2``, drops a leading
    ``x =`` and a trailing period, and collapses whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("answer must be str")
    out = unicodedata.normalize("NFKC", text).strip()
    out = strip_answer_wrapper(out).strip()
    # ``\\\\`` is a matrix/array row separator; park it so that the ``\\ `` spacing command
    # below cannot eat one of its backslashes.
    out = out.replace("\\\\", ROW_SEPARATOR_SENTINEL)
    out = out.replace("$$", "").replace(r"\$", "").replace("$", "")
    out = out.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    out = out.replace(r"\cfrac", r"\frac")
    out = _DEGREES.sub("", out)
    out = _LEFT_RIGHT.sub("", out)
    out = _SPACER_COMMANDS.sub(" ", out)
    out = _strip_text_groups(out)
    out = out.replace(r"\%", "").replace("%", "")
    out = _THOUSANDS_BRACED.sub("", out)
    out = _THOUSANDS_PLAIN.sub("", out)
    out = _BRACELESS_SQRT.sub(r"\\sqrt{\1}", out)
    out = _LEADING_DOT.sub("0.", out)
    out = _SET_MEMBERSHIP.sub("", out)
    out = out.replace("\u03c0", r"\pi")
    out = _BARE_PI.sub(lambda _m: r"\pi", out)
    out = out.replace(ROW_SEPARATOR_SENTINEL, " \\\\ ")
    out = _WS.sub(" ", out).strip()
    out = _TRAILING_PUNCT.sub("", out).strip()
    out = _LEADING_EQUATION.sub("", out).strip()
    return out


# --------------------------------------------------------------------------- shape router


@dataclass(frozen=True)
class Node:
    """A parsed answer shape.  ``kind`` is one of the ``KIND_*`` constants."""

    kind: str
    source: str
    expr: Any = None
    text: str = ""
    delims: str = ""
    ordered: bool = True
    parts: tuple["Node", ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def parsed(self) -> bool:
        """Whether the whole shape reached a symbolic/textual class (no RAW leaves)."""
        if self.kind == KIND_RAW:
            return False
        if self.kind == KIND_SEQ:
            return all(part.parsed for part in self.parts)
        return True


def _split_top_level(text: str, separators: tuple[str, ...]) -> list[str] | None:
    """Split on ``separators`` outside braces/brackets/parentheses; ``None`` if unbalanced."""
    depth = 0
    parts: list[str] = []
    current: list[str] = []
    index = 0
    found = False
    while index < len(text):
        char = text[index]
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
            if depth < 0:
                return None
        if depth == 0:
            matched = ""
            for separator in separators:
                if text.startswith(separator, index):
                    matched = separator
                    break
            if matched:
                parts.append("".join(current))
                current = []
                index += len(matched)
                found = True
                continue
        current.append(char)
        index += 1
    if depth != 0:
        return None
    parts.append("".join(current))
    if not found:
        return None
    return [part.strip() for part in parts]


_OPEN_TO_CLOSE = {"(": ")", "[": "]", r"\{": r"\}", "{": "}"}


def _container(text: str) -> tuple[str, str] | None:
    """``(delims, inner)`` when ``text`` is one balanced container, else ``None``."""
    for opener, closer in _OPEN_TO_CLOSE.items():
        if not text.startswith(opener):
            continue
        for other_close in (")", "]", r"\}", "}"):
            if not text.endswith(other_close):
                continue
            inner = text[len(opener) : len(text) - len(other_close)]
            if _split_top_level(inner + "\x00", ("\x00",)) is None:
                continue
            if opener in ("(", "[") and other_close in (")", "]"):
                return opener + other_close, inner
            if opener == r"\{" and other_close == r"\}":
                return "{}", inner
            if opener == "{" and other_close == "}" and not inner.startswith("\\"):
                return "{}", inner
    return None


def _parse_latex_expr(text: str, limit: _TimeLimit | None) -> Any:
    """``parse_latex(..., backend='lark')`` with the ``\\pi`` placeholder round-trip."""
    from sympy.parsing.latex import parse_latex  # local: heavy grammar build

    if limit is not None:
        limit.check()
    carrier = text
    substituted = False
    if r"\pi" in carrier:
        if "upsilon" in carrier:
            raise ValueError(r"cannot carry \pi: the answer already uses \upsilon")
        carrier = re.sub(r"\\pi(?![a-zA-Z])", lambda _m: PI_PLACEHOLDER, carrier)
        substituted = True
    expr = parse_latex(carrier, backend=LATEX_BACKEND)
    if substituted:
        expr = expr.subs(PI_SYMBOL, sympy.pi)
    return expr


def parse_answer(answer: str, *, limit: _TimeLimit | None = None) -> Node:
    """Route a *normalised* answer string to a :class:`Node`.  Never raises on content.

    Containers (tuples, intervals, sets, matrices, ``\\pm`` and ``\\cup`` forms and bare
    comma lists) become ``KIND_SEQ`` nodes whose parts are parsed recursively; prose
    becomes ``KIND_TEXT``; base-n numerals become ``KIND_BASE``; anything the lark
    grammar rejects becomes ``KIND_RAW`` with the parser message in ``reason``.
    """
    text = answer.strip()
    if text == "":
        return Node(KIND_EMPTY, source=text)

    environment = _ENVIRONMENT.fullmatch(text)
    if environment:
        rows = [row.strip() for row in re.split(r"\\\\", environment.group(2)) if row.strip()]
        parts: list[Node] = []
        for row in rows:
            columns = [column.strip() for column in row.split("&")]
            if len(columns) > 1:
                parts.append(Node(
                    KIND_SEQ, source=row, delims="row", ordered=True,
                    parts=tuple(parse_answer(column, limit=limit) for column in columns),
                ))
            else:
                parts.append(parse_answer(row, limit=limit))
        return Node(KIND_SEQ, source=text, delims=environment.group(1), ordered=True, parts=tuple(parts))

    for separator, delims, ordered in ((r"\cup", "cup", False), (r"\pm", "pm", False)):
        pieces = _split_top_level(text, (separator,))
        if pieces and len(pieces) > 1:
            if delims == "pm":
                head, tail = pieces[0], pieces[1]
                pieces = [f"{head} - ({tail})", f"{head} + ({tail})"]
            parts = tuple(parse_answer(piece, limit=limit) for piece in pieces)
            return Node(KIND_SEQ, source=text, delims=delims, ordered=ordered, parts=parts)

    container = _container(text)
    if container is not None:
        delims, inner = container
        pieces = _split_top_level(inner, (",",))
        if pieces is None:
            pieces = [inner] if inner.strip() else []
        parts = tuple(parse_answer(piece, limit=limit) for piece in pieces)
        ordered = delims != "{}"
        return Node(KIND_SEQ, source=text, delims=delims, ordered=ordered, parts=parts)

    bare = _split_top_level(text, (",",))
    if bare and len(bare) > 1:
        parts = tuple(parse_answer(piece, limit=limit) for piece in bare)
        return Node(KIND_SEQ, source=text, delims="list", ordered=False, parts=parts)

    base = _BASE_NUMERAL.fullmatch(text)
    if base and int(base.group(2)) >= 2:
        digits = base.group(1).lstrip("0") or "0"
        return Node(KIND_BASE, source=text, text=f"{digits.upper()}_{int(base.group(2))}")

    if _WORD.fullmatch(text) and any(len(token) >= 3 for token in text.split()):
        return Node(KIND_TEXT, source=text, text=_WS.sub(" ", text.strip().casefold()))

    try:
        expr = _parse_latex_expr(text, limit)
    except GraderTimeout:
        raise
    except Exception as exc:  # lark UnexpectedCharacters, sympy errors, recursion, ...
        return Node(KIND_RAW, source=text, text=text, reason=f"{type(exc).__name__}: {exc}".split("\n")[0][:160])
    return Node(KIND_EXPR, source=text, expr=expr)


# --------------------------------------------------------------------------- equivalence


def _numeric_agreement(left: Any, right: Any, limit: _TimeLimit | None) -> str | None:
    """``correct``/``incorrect`` from random probe points, or ``None`` when undecidable."""
    symbols = sorted(set(left.free_symbols) | set(right.free_symbols), key=str)
    rng = random.Random(NUMERIC_PROBE_SEED)
    agreements = 0
    for _ in range(NUMERIC_PROBE_POINTS):
        if limit is not None:
            limit.check()
        point = {symbol: sympy.Rational(rng.randint(2, 97), rng.randint(2, 13)) for symbol in symbols}
        try:
            a = complex(sympy.N(left.subs(point), 30))
            b = complex(sympy.N(right.subs(point), 30))
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            continue
        if not (math.isfinite(a.real) and math.isfinite(a.imag) and math.isfinite(b.real) and math.isfinite(b.imag)):
            continue
        scale = max(1.0, abs(a), abs(b))
        if abs(a - b) > NUMERIC_RTOL * scale:
            return STATUS_INCORRECT
        agreements += 1
    if agreements >= NUMERIC_MIN_AGREEMENTS:
        return STATUS_CORRECT
    return None


_EXTENDED = (sympy.oo, -sympy.oo, sympy.zoo, sympy.nan)


def _compare_expr(left: Any, right: Any, limit: _TimeLimit | None) -> tuple[str, str]:
    if limit is not None:
        limit.check()
    if left == right:
        return STATUS_CORRECT, "structurally identical"
    try:
        left, right = sympy.simplify(left), sympy.simplify(right)
    except GraderTimeout:
        raise
    except Exception as exc:
        return STATUS_AMBIGUOUS, f"simplify failed: {type(exc).__name__}"
    if left == right:
        return STATUS_CORRECT, "identical after simplification"
    if left.has(*_EXTENDED) or right.has(*_EXTENDED):
        # ``oo - oo`` is ``nan``: differencing cannot decide, and two distinct extended
        # values that survived simplification are distinct.
        if not (left.free_symbols or right.free_symbols):
            return STATUS_INCORRECT, "distinct extended-real values"
        return STATUS_AMBIGUOUS, "extended-real value with free symbols"
    try:
        difference = sympy.simplify(left - right)
    except GraderTimeout:
        raise
    except Exception as exc:
        return STATUS_AMBIGUOUS, f"simplify failed: {type(exc).__name__}"
    if difference.has(sympy.nan, sympy.zoo):
        return STATUS_AMBIGUOUS, "difference is undefined (nan/zoo)"
    try:
        zero = difference.is_zero
    except (TypeError, ValueError):
        zero = None
    if zero is True or difference == 0:
        return STATUS_CORRECT, "simplify(a-b)==0"
    if zero is False:
        return STATUS_INCORRECT, "simplify(a-b) is provably non-zero"
    numeric = _numeric_agreement(left, right, limit)
    if numeric == STATUS_CORRECT:
        return STATUS_CORRECT, "numeric agreement at random points"
    if numeric == STATUS_INCORRECT:
        return STATUS_INCORRECT, "numeric disagreement at a random point"
    return STATUS_AMBIGUOUS, "neither simplification nor numeric probing decided"


def _combine(statuses: list[str]) -> str:
    if STATUS_TIMEOUT in statuses:
        return STATUS_TIMEOUT
    if STATUS_INCORRECT in statuses:
        return STATUS_INCORRECT
    if STATUS_AMBIGUOUS in statuses:
        return STATUS_AMBIGUOUS
    return STATUS_CORRECT


def _match_unordered(left: tuple[Node, ...], right: tuple[Node, ...], limit: _TimeLimit | None) -> tuple[str, str]:
    """Greedy multiset matching for unordered sequences (sets, ``\\pm``, ``\\cup``, lists)."""
    remaining = list(right)
    statuses: list[str] = []
    for node in left:
        deferred: list[tuple[int, str, str]] = []
        matched = False
        for index, candidate in enumerate(remaining):
            status, detail = _compare_nodes(node, candidate, limit)
            if status == STATUS_CORRECT:
                remaining.pop(index)
                matched = True
                break
            deferred.append((index, status, detail))
        if matched:
            continue
        soft = [entry for entry in deferred if entry[1] in (STATUS_AMBIGUOUS, STATUS_TIMEOUT)]
        if soft:
            index, status, detail = soft[0]
            remaining.pop(index)
            statuses.append(status)
            continue
        return STATUS_INCORRECT, f"no partner for element {node.source!r}"
    return _combine(statuses), "unordered element-wise match"


def _compare_nodes(left: Node, right: Node, limit: _TimeLimit | None) -> tuple[str, str]:
    if limit is not None:
        limit.check()
    if left.kind == KIND_EMPTY or right.kind == KIND_EMPTY:
        if left.kind == right.kind:
            return STATUS_CORRECT, "both empty"
        return STATUS_INCORRECT, "one side is empty"
    if left.kind == KIND_RAW or right.kind == KIND_RAW:
        if left.source.casefold() == right.source.casefold():
            return STATUS_CORRECT, "identical unparsed strings"
        reason = left.reason or right.reason
        return STATUS_AMBIGUOUS, f"unparsed shape ({reason})"
    if left.kind != right.kind:
        return STATUS_INCORRECT, f"shape mismatch: {left.kind} vs {right.kind}"
    if left.kind == KIND_TEXT:
        return (STATUS_CORRECT, "text equality") if left.text == right.text else (STATUS_INCORRECT, "text mismatch")
    if left.kind == KIND_BASE:
        return (STATUS_CORRECT, "base numeral equality") if left.text == right.text else (STATUS_INCORRECT, "base numeral mismatch")
    if left.kind == KIND_SEQ:
        if len(left.parts) != len(right.parts):
            return STATUS_INCORRECT, f"arity mismatch: {len(left.parts)} vs {len(right.parts)}"
        if left.ordered != right.ordered:
            # ``(a,b)`` versus ``a,b``: the elements may agree while the notation does not
            # commit to the same object.  Never resolved silently in either direction.
            status, _ = _match_unordered(left.parts, right.parts, limit)
            if status == STATUS_CORRECT:
                return STATUS_AMBIGUOUS, "same elements under different container notation"
            return STATUS_INCORRECT, "container notation and elements both differ"
        if left.ordered and left.delims != right.delims:
            return STATUS_INCORRECT, f"container mismatch: {left.delims} vs {right.delims}"
        if not left.ordered:
            return _match_unordered(left.parts, right.parts, limit)
        statuses: list[str] = []
        for a, b in zip(left.parts, right.parts):
            status, detail = _compare_nodes(a, b, limit)
            if status == STATUS_INCORRECT:
                return STATUS_INCORRECT, f"element mismatch: {detail}"
            statuses.append(status)
        return _combine(statuses), "ordered element-wise match"
    return _compare_expr(left.expr, right.expr, limit)


@dataclass(frozen=True)
class MathVerdict:
    """Outcome of one candidate-versus-reference comparison."""

    status: str
    detail: str
    candidate_normalised: str
    reference_normalised: str
    candidate_parsed: bool
    reference_parsed: bool
    duration_s: float


def compare(candidate: str | None, reference: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> MathVerdict:
    """Grade ``candidate`` against the reference answer ``reference`` (evaluator only).

    Returns one of :data:`STATUSES`.  A missing or empty candidate is ``incorrect`` (a
    non-answer is a genuine failure); an unparseable or timed-out *comparison* is
    ``ambiguous``/``timeout`` and is never reported as ``incorrect`` (spec §5.1).
    """
    started = time.monotonic()
    if candidate is None or not str(candidate).strip():
        return MathVerdict(STATUS_INCORRECT, "empty candidate", "", normalise(reference), False, True, 0.0)
    candidate_norm = normalise(str(candidate))
    reference_norm = normalise(reference)
    with _TimeLimit(timeout_s) as limit:
        try:
            if candidate_norm.casefold() == reference_norm.casefold():
                return MathVerdict(
                    STATUS_CORRECT, "normalised string equality", candidate_norm, reference_norm,
                    True, True, time.monotonic() - started,
                )
            left = parse_answer(candidate_norm, limit=limit)
            right = parse_answer(reference_norm, limit=limit)
            status, detail = _compare_nodes(left, right, limit)
            if not limit.enforced and status in (STATUS_AMBIGUOUS, STATUS_INCORRECT):
                detail = f"{detail} [deadline not signal-enforced]"
            return MathVerdict(
                status, detail, candidate_norm, reference_norm, left.parsed, right.parsed,
                time.monotonic() - started,
            )
        except GraderTimeout as exc:
            return MathVerdict(
                STATUS_TIMEOUT, str(exc), candidate_norm, reference_norm, False, False,
                time.monotonic() - started,
            )
        except RecursionError:
            return MathVerdict(
                STATUS_AMBIGUOUS, "recursion limit while parsing", candidate_norm, reference_norm,
                False, False, time.monotonic() - started,
            )


def self_equivalence(answer: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[str, str]:
    """``(status, detail)`` of comparing a parsed answer with an independent parse of itself.

    Bypasses the normalised-string shortcut of :func:`compare`, so it exercises the
    symbolic path.  Used by the pre-freeze validation over the 500 gold answers
    (``items_build.validate_math_grader``): a shape that cannot even match itself would
    make every grade against it ambiguous.
    """
    normalised = normalise(answer)
    with _TimeLimit(timeout_s) as limit:
        try:
            return _compare_nodes(parse_answer(normalised, limit=limit),
                                  parse_answer(normalised, limit=limit), limit)
        except GraderTimeout as exc:
            return STATUS_TIMEOUT, str(exc)


# --------------------------------------------------------------------------- canonical class


def _rationalise(expr: Any) -> Any:
    """Exact decimal -> ``Rational`` so ``0.5`` and ``\\frac12`` share one canonical class."""
    try:
        return expr.replace(lambda node: node.is_Float, lambda node: sympy.Rational(str(node)))
    except Exception:
        return expr


def _class_of(node: Node, limit: _TimeLimit | None) -> str:
    if node.kind == KIND_EMPTY:
        return "empty:"
    if node.kind == KIND_TEXT:
        return f"text:{node.text}"
    if node.kind == KIND_BASE:
        return f"base:{node.text}"
    if node.kind == KIND_RAW:
        return f"str:{node.source}"
    if node.kind == KIND_SEQ:
        children = [_class_of(part, limit) for part in node.parts]
        if not node.ordered:
            children = sorted(children)
            return "seq[unordered]:(" + ",".join(children) + ")"
        return f"seq[{node.delims}]:(" + ",".join(children) + ")"
    if limit is not None:
        limit.check()
    try:
        return "expr:" + sympy.srepr(sympy.simplify(_rationalise(node.expr)))
    except GraderTimeout:
        raise
    except Exception:
        return f"str:{node.source}"


def canonical_class(answer: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> str | None:
    """Gold-independent canonical class of a MATH answer, or ``None`` when empty.

    This is the *voting* key (spec §5.1: canonicalization is independent of the
    reference).  It never raises: a parse failure or a timeout degrades to the
    normalised-string class ``str:<normalised>`` so two byte-identical answers still
    share a class.
    """
    if answer is None:
        return None
    try:
        normalised = normalise(str(answer))
    except TypeError:
        return None
    if normalised == "":
        return None
    with _TimeLimit(timeout_s) as limit:
        try:
            return _class_of(parse_answer(normalised, limit=limit), limit)
        except (GraderTimeout, RecursionError):
            return f"str:{normalised}"
        except Exception:
            return f"str:{normalised}"


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "GraderTimeout",
    "KIND_BASE",
    "KIND_EMPTY",
    "KIND_EXPR",
    "KIND_RAW",
    "KIND_SEQ",
    "KIND_TEXT",
    "LATEX_BACKEND",
    "MathVerdict",
    "Node",
    "PINNED_LARK_VERSION",
    "STATUSES",
    "STATUS_AMBIGUOUS",
    "STATUS_CORRECT",
    "STATUS_INCORRECT",
    "STATUS_TIMEOUT",
    "canonical_class",
    "compare",
    "normalise",
    "parse_answer",
    "self_equivalence",
    "strip_answer_wrapper",
]
