"""Independent evaluator for the E3 straight-line program traces (WP3).

Spec §5.4: "Generate 64 fresh, deterministic small program-trace tasks ... The task is to
evaluate a straight-line integer program modulo 97.  All operations and inputs are
visible.  A CPU interpreter computes the reference privately. ... Example permitted
updates: ``x=(a*x+b)%97``, ``x=(x+y)%97``, ``y=(x+2*y+b)%97``; track sequential assignment
semantics exactly.  Use coefficients 2-11 and seeds 0-96.  Disallow division, imports,
loops, randomness, file access, and undefined behavior.  **A second independent evaluator
validates the gold output.**"  Spec §11 additionally needs the per-step state so that the
corrupted transition of a candidate trace and one matched valid transition can be
verified and reported (``MINORITY_CHECKED``/``MAJORITY_CHECKED``).

This module is that second evaluator.  It is deliberately **independent of the generator's
interpreter (WP5)**: it uses its own hand-written lexer and recursive-descent parser and
never touches :mod:`ast`, ``eval``/``exec``, ``compile`` or ``sympy``, so a defect in the
generator's evaluation path cannot be mirrored here.  The plan's cross-check ("two
evaluators agree on all 64 golds, corrupted trace differs") compares the two.

Constants come from :mod:`agents_scaling.ard.contract` (``RECEIVER_MODULUS``,
``RECEIVER_COEFFICIENTS``); nothing here is re-declared.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agents_scaling.ard.contract import RECEIVER_MODULUS

STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_TIMEOUT = "timeout"
STATUSES: tuple[str, ...] = (STATUS_CORRECT, STATUS_INCORRECT, STATUS_AMBIGUOUS, STATUS_TIMEOUT)

#: Tokens that may never appear in a trace program (spec §5.4 "disallow ...").
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "/", "**", "^", "import", "def", "for", "while", "lambda", "open", "eval", "exec",
    "class", "return", "print", "[", "]", ".", "&", "|", "<", ">", "!", '"', "'", ";",
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_INT = re.compile(r"[0-9]+")


class TraceProgramError(ValueError):
    """The program text violates the frozen E3 program grammar."""


# --------------------------------------------------------------------------- lexer


@dataclass(frozen=True)
class Token:
    kind: str          # "ident" | "int" | "op"
    text: str
    column: int


def _tokenise(line: str, line_no: int) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char.isspace():
            index += 1
            continue
        match = _IDENT.match(line, index)
        if match:
            tokens.append(Token("ident", match.group(0), index))
            index = match.end()
            continue
        match = _INT.match(line, index)
        if match:
            tokens.append(Token("int", match.group(0), index))
            index = match.end()
            continue
        if char in "+-*%()=":
            tokens.append(Token("op", char, index))
            index += 1
            continue
        raise TraceProgramError(f"line {line_no}: illegal character {char!r}")
    return tokens


# --------------------------------------------------------------------------- parser


@dataclass(frozen=True)
class Statement:
    line_no: int
    target: str
    source: str
    modulus: int | None


_FORBIDDEN_WORDS = tuple(token for token in FORBIDDEN_TOKENS if token.isalpha())
_FORBIDDEN_SYMBOLS = tuple(token for token in FORBIDDEN_TOKENS if not token.isalpha())
_FORBIDDEN_WORD_RE = re.compile(r"\b(?:" + "|".join(_FORBIDDEN_WORDS) + r")\b")


def _forbidden_scan(source: str) -> None:
    """Reject the spec §5.4 disallowed constructs.

    Keywords are matched on word boundaries (so a variable named ``form`` is not read as
    ``for``); operator/punctuation tokens are matched as substrings.
    """
    lowered = source.lower()
    for token in _FORBIDDEN_SYMBOLS:
        if token in lowered:
            raise TraceProgramError(f"forbidden construct {token!r} in program text")
    match = _FORBIDDEN_WORD_RE.search(lowered)
    if match:
        raise TraceProgramError(f"forbidden construct {match.group(0)!r} in program text")


def parse_program(source: str) -> tuple[Statement, ...]:
    """Split ``source`` into single-assignment statements, rejecting anything else."""
    if not isinstance(source, str):
        raise TypeError("program source must be str")
    _forbidden_scan(source)
    statements: list[Statement] = []
    for line_no, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        tokens = _tokenise(line, line_no)
        if len(tokens) < 3 or tokens[0].kind != "ident" or tokens[1].text != "=":
            raise TraceProgramError(f"line {line_no}: expected '<name> = <expression>'")
        if any(token.text == "=" for token in tokens[2:]):
            raise TraceProgramError(f"line {line_no}: more than one assignment on a line")
        body = line[line.index("=") + 1 :].strip()
        modulus: int | None = None
        remainder = body
        tail = _tokenise(body, line_no)
        for position in range(len(tail) - 2, -1, -1):
            if tail[position].text == "%":
                if tail[position + 1].kind != "int" or position + 2 != len(tail):
                    raise TraceProgramError(f"line {line_no}: '%' must be followed by a final integer")
                modulus = int(tail[position + 1].text)
                remainder = body[: tail[position].column].strip()
                break
        if "%" in remainder:
            raise TraceProgramError(f"line {line_no}: only one trailing '% m' is permitted")
        statements.append(Statement(line_no, tokens[0].text, remainder, modulus))
    if not statements:
        raise TraceProgramError("program has no statements")
    return tuple(statements)


class _ExpressionParser:
    """Recursive descent over ``expr := term (('+'|'-') term)*`` with integer semantics."""

    def __init__(self, tokens: Sequence[Token], env: Mapping[str, int], line_no: int) -> None:
        self.tokens = list(tokens)
        self.env = env
        self.line_no = line_no
        self.position = 0

    def _peek(self) -> Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self) -> Token:
        token = self._peek()
        if token is None:
            raise TraceProgramError(f"line {self.line_no}: expression ended early")
        self.position += 1
        return token

    def evaluate(self) -> int:
        value = self._expr()
        if self.position != len(self.tokens):
            raise TraceProgramError(f"line {self.line_no}: trailing tokens in expression")
        return value

    def _expr(self) -> int:
        value = self._term()
        while True:
            token = self._peek()
            if token is None or token.text not in "+-":
                return value
            self._take()
            right = self._term()
            value = value + right if token.text == "+" else value - right

    def _term(self) -> int:
        value = self._factor()
        while True:
            token = self._peek()
            if token is None or token.text != "*":
                return value
            self._take()
            value = value * self._factor()

    def _factor(self) -> int:
        token = self._peek()
        if token is not None and token.text in "+-":
            self._take()
            value = self._factor()
            return -value if token.text == "-" else value
        return self._primary()

    def _primary(self) -> int:
        token = self._take()
        if token.kind == "int":
            return int(token.text)
        if token.kind == "ident":
            following = self._peek()
            if following is not None and following.text == "(":
                raise TraceProgramError(f"line {self.line_no}: function calls are not permitted")
            if token.text not in self.env:
                raise TraceProgramError(f"line {self.line_no}: undefined variable {token.text!r}")
            return self.env[token.text]
        if token.text == "(":
            value = self._expr()
            closing = self._take()
            if closing.text != ")":
                raise TraceProgramError(f"line {self.line_no}: unbalanced parentheses")
            return value
        raise TraceProgramError(f"line {self.line_no}: unexpected token {token.text!r}")


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class Step:
    """One executed assignment and the whole variable state after it."""

    index: int
    line_no: int
    target: str
    value: int
    state: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Evaluation:
    final_target: str
    final_value: int
    steps: tuple[Step, ...]
    modulus: int


def evaluate_program(
    source: str,
    *,
    modulus: int = RECEIVER_MODULUS,
    initial: Mapping[str, int] | None = None,
) -> Evaluation:
    """Execute a straight-line program with exact sequential assignment semantics.

    Every assignment is reduced modulo ``modulus`` (Python's ``%`` on ints, which is the
    non-negative residue).  A statement carrying its own ``% m`` must agree with
    ``modulus``.  Any forbidden construct, undefined variable or malformed line raises
    :class:`TraceProgramError` — this evaluator never guesses.
    """
    if not isinstance(modulus, int) or modulus < 2:
        raise ValueError("modulus must be an integer >= 2")
    env: dict[str, int] = {name: int(value) % modulus for name, value in (initial or {}).items()}
    steps: list[Step] = []
    statements = parse_program(source)
    for index, statement in enumerate(statements):
        if statement.modulus is not None and statement.modulus != modulus:
            raise TraceProgramError(
                f"line {statement.line_no}: statement modulus {statement.modulus} != {modulus}"
            )
        tokens = _tokenise(statement.source, statement.line_no)
        if not tokens:
            raise TraceProgramError(f"line {statement.line_no}: empty right-hand side")
        value = _ExpressionParser(tokens, env, statement.line_no).evaluate() % modulus
        env[statement.target] = value
        steps.append(Step(index, statement.line_no, statement.target, value,
                          tuple(sorted(env.items()))))
    last = steps[-1]
    return Evaluation(last.target, last.value, tuple(steps), modulus)


# --------------------------------------------------------------------------- trace checking


@dataclass(frozen=True)
class StepCheck:
    index: int
    line_no: int
    target: str
    claimed: int | None
    reference: int
    agrees: bool


_CLAIM_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=.*?(-?\d+)\s*$")


def parse_claimed_trace(text: str) -> tuple[tuple[str, int], ...]:
    """``((target, value), ...)`` from a rendered candidate trace.

    Accepts ``x = 41`` and ``x = (3*x + 5) % 97 = 41`` — the value is the last integer on
    the line.  Lines that do not assign are ignored, so surrounding prose is tolerated.
    """
    claims: list[tuple[str, int]] = []
    for line in str(text).splitlines():
        match = _CLAIM_LINE.match(line)
        if match:
            claims.append((match.group(1), int(match.group(2))))
    return tuple(claims)


def check_trace(
    source: str,
    claimed: Sequence[tuple[str, int]] | Sequence[int] | str,
    *,
    modulus: int = RECEIVER_MODULUS,
    initial: Mapping[str, int] | None = None,
) -> tuple[StepCheck, ...]:
    """Verify a supplied candidate trace step by step against this evaluator.

    ``claimed`` may be the rendered trace text, a sequence of values, or a sequence of
    ``(target, value)`` pairs.  The returned tuple is aligned with the reference steps;
    :func:`first_divergence` picks out the corrupted transition of spec §5.4.
    """
    reference = evaluate_program(source, modulus=modulus, initial=initial)
    if isinstance(claimed, str):
        pairs: list[tuple[str, int] | None] = list(parse_claimed_trace(claimed))
    else:
        pairs = []
        for entry in claimed:
            if isinstance(entry, tuple):
                pairs.append((str(entry[0]), int(entry[1])))
            else:
                pairs.append((None, int(entry)))  # type: ignore[arg-type]
    checks: list[StepCheck] = []
    for step in reference.steps:
        claim = pairs[step.index] if step.index < len(pairs) else None
        if claim is None:
            checks.append(StepCheck(step.index, step.line_no, step.target, None, step.value, False))
            continue
        name, value = claim
        agrees = value % modulus == step.value and (name is None or name == step.target)
        checks.append(StepCheck(step.index, step.line_no, step.target, value, step.value, agrees))
    return tuple(checks)


def first_divergence(checks: Sequence[StepCheck]) -> StepCheck | None:
    """The first step at which a candidate trace departs from the reference, if any."""
    for check in checks:
        if not check.agrees:
            return check
    return None


# --------------------------------------------------------------------------- grading


@dataclass(frozen=True)
class TraceVerdict:
    status: str
    detail: str
    extracted: int | None
    reference: int


def canonical_class(answer: str | None, *, modulus: int = RECEIVER_MODULUS) -> str | None:
    """Gold-independent voting class: the residue the answer states, or ``None``."""
    value = extract_residue(answer, modulus=modulus)
    return None if value is None else f"mod{modulus}:{value}"


def extract_residue(answer: str | None, *, modulus: int = RECEIVER_MODULUS) -> int | None:
    """The last integer in ``answer`` that is a valid residue, or ``None``."""
    if answer is None:
        return None
    integers = [int(token) for token in re.findall(r"-?\d+", str(answer))]
    for value in reversed(integers):
        if 0 <= value < modulus:
            return value
    return None


def grade(answer: str | None, reference_value: int, *, modulus: int = RECEIVER_MODULUS) -> TraceVerdict:
    """Grade a receiver answer against the privately computed reference (spec §11).

    "The model can also reject both supplied choices; grade its produced answer rather
    than forcing agreement with the packet" — so this grades the stated residue, never a
    choice between the two supplied traces.
    """
    if answer is None or not str(answer).strip():
        return TraceVerdict(STATUS_INCORRECT, "empty answer", None, reference_value)
    value = extract_residue(answer, modulus=modulus)
    if value is None:
        return TraceVerdict(STATUS_AMBIGUOUS, f"no residue in 0..{modulus - 1} found", None, reference_value)
    if value == reference_value % modulus:
        return TraceVerdict(STATUS_CORRECT, "residue matches the reference", value, reference_value)
    return TraceVerdict(STATUS_INCORRECT, "residue differs from the reference", value, reference_value)


__all__ = [
    "Evaluation",
    "FORBIDDEN_TOKENS",
    "STATUSES",
    "STATUS_AMBIGUOUS",
    "STATUS_CORRECT",
    "STATUS_INCORRECT",
    "STATUS_TIMEOUT",
    "Statement",
    "Step",
    "StepCheck",
    "Token",
    "TraceProgramError",
    "TraceVerdict",
    "canonical_class",
    "check_trace",
    "evaluate_program",
    "extract_residue",
    "first_divergence",
    "grade",
    "parse_claimed_trace",
    "parse_program",
]
