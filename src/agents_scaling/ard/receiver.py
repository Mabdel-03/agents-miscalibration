"""ARD36 E3 frozen-content receiver assay: the generator, the packets and the cell (WP5).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §5.4 (the 64
controlled program tasks) and §11 (the assay itself); plan §5.5.

Why this module exists at all
-----------------------------

E3 is the one experiment in ARD36 whose *content is fixed by construction*.  Spec §11:
"All comparisons use the same task and candidate content.  This separates social-count
sensitivity from the independent generation of better roots at larger scale."  Every packet
a model sees here is built by this module, never by a model, so the endorsement-count
contrast is within-program and content-matched and a difference in behaviour cannot be a
difference in what was shown.

The five frozen attributes
--------------------------

Per program, the following are **frozen per-program attributes** and are identical across
all five presentations, both reasoning budgets, both completion seeds and both checkpoints
(plan §5.5):

1. the program text (depth, operator mix, coefficients, seeds);
2. the two candidate traces — one correct, one carrying a single arithmetic error at an
   internal step propagated consistently afterwards;
3. the **trace order** (which candidate is shown as A), from the recorded
   :data:`SCHEDULE`, balanced within depth x operator-mix strata;
4. the **provisional-answer correctness**, from the same schedule and balance;
5. the two preregistered check locations (the corrupted internal transition and one
   matched valid transition of the same update form, never the final transition and never
   the final output).

Only the presentation varies.  That is what makes ``MINORITY_*`` versus ``MAJORITY_*`` a
within-program contrast: :func:`build_packet` renders the two count assignments
byte-identically apart from the two swapped endorsement counts
(:func:`endorsement_line`), and the checked/unchecked pair differs only in the *status*
column of the same two check locations — ``NOT CHECKED`` in the unchecked counterpart, with
no additional explanation (spec §11).

Two independent evaluators
--------------------------

:func:`interpret` is the generator's own CPU interpreter: it walks the *typed*
:class:`Update` records with plain integer arithmetic.  ``ard/graders/trace.py`` (WP3) is
the second, independent evaluator: a hand-written lexer, parser and stepper over the
*rendered text*.  :func:`verify_programs` runs both over all 64 programs and requires
agreement on every gold, which therefore also validates the rendering — a stronger check
than the plan's sketch of "restricted ``exec`` versus a hand-written stepper", and without
putting ``exec`` on a runtime path.  ``ard.graders`` is imported **inside** that function
(and inside nothing else here), because it is the package that can read the protected gold
namespace and the contract §0.2 firewall is about the module-level import closure.  The E3
reference is *computed*, never read from a protected file, so :func:`run_receiver_cell`
grades against :func:`interpret` and needs no grader at all.

Permitted program grammar
-------------------------

Spec §5.4 permits exactly three update forms and disallows division, imports, loops,
randomness, file access and undefined behaviour.  :func:`assert_permitted_program` enforces
this as an **allowlist**: every rendered line must match one of four exact regexes, so a
forbidden construct cannot appear by omission from a denylist.  Every program is also
checked for state dependence — every update reads a prior variable, and every assigned
value is read by a later step or is the final output, so no program computes an independent
unused value.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.study import identity
from agents_scaling.study.types import ProtocolError

__all__ = [
    "AnswerLogitMargin",
    "CANDIDATE_LABELS",
    "CHECK_NOT_CHECKED",
    "CHECK_FAILED",
    "CHECK_VERIFIED",
    "CandidateTrace",
    "FORM_AFFINE_X",
    "FORM_AFFINE_Y",
    "FORM_SUM_X",
    "LocalCheck",
    "MARGIN_NO_DIFFERENCE",
    "MARGIN_NO_LOGPROBS",
    "MARGIN_OK",
    "MARGIN_OUTSIDE_TOP_K",
    "MARGIN_TOKEN_NOT_FOUND",
    "MARGIN_UNAVAILABLE",
    "MIXES",
    "MIX_AFFINE",
    "MIX_COUPLED",
    "ORDER_CORRECT_FIRST",
    "ORDER_WRONG_FIRST",
    "OUTCOMES",
    "OUTCOME_INVALID",
    "OUTCOME_PICKED_CORRECT",
    "OUTCOME_PICKED_WRONG",
    "OUTCOME_PRODUCED_OTHER_CORRECT",
    "OUTCOME_PRODUCED_OTHER_WRONG",
    "PROVISIONAL_CORRECT",
    "PROVISIONAL_WRONG",
    "Program",
    "ProgramSchedule",
    "RECEIVER_FAMILY",
    "RECEIVER_TASK_FAMILY",
    "ReceiverError",
    "ReceiverOutcome",
    "ReceiverPacket",
    "STRATA",
    "TraceStep",
    "Update",
    "answer_logit_margin",
    "assert_permitted_program",
    "build_packet",
    "build_programs",
    "candidate_traces",
    "endorsement_line",
    "endorsements",
    "extract_residue",
    "first_differing_token",
    "grade_receiver",
    "interpret",
    "program_for",
    "programs",
    "receiver_task",
    "receiver_task_loader",
    "receiver_tasks",
    "render_receiver_prompt",
    "run_receiver_cell",
    "schedule",
    "verify_programs",
]

MODULUS: Final[int] = K.RECEIVER_MODULUS

#: Grader tag of the synthetic E3 programs; equals ``ard.graders.TRACE_FAMILY`` (asserted
#: in the WP5 tests, not imported, so the runtime closure stays clear of the grader).  It is
#: *not* the ``ArdTask.family`` of a program row: WP4's ``cells.interleave_families`` admits
#: only ``contract.FAMILIES``, so a program row carries :data:`RECEIVER_TASK_FAMILY` in that
#: column and the real E3 tag in ``stratum`` — the discriminator is the ``receiver`` cohort.
RECEIVER_FAMILY: Final[str] = "receiver_trace"
#: The ``contract.FAMILIES`` member a synthetic program row declares (the tasks are integer
#: arithmetic, so MATH500 is the honest one of the two).
RECEIVER_TASK_FAMILY: Final[str] = K.Family.MATH500.value
PROGRAM_ID_PREFIX: Final[str] = "receiver:p"

# --------------------------------------------------------------------------- the grammar

#: The three permitted update forms (spec §5.4, verbatim).
FORM_AFFINE_X: Final[str] = "x_affine"     # x = (a*x + b) % 97
FORM_SUM_X: Final[str] = "x_sum"           # x = (x + y) % 97
FORM_AFFINE_Y: Final[str] = "y_affine"     # y = (x + 2*y + b) % 97
FORMS: Final[tuple[str, ...]] = (FORM_AFFINE_X, FORM_SUM_X, FORM_AFFINE_Y)

#: Variables each form reads and writes — the state-dependence analysis reads only this.
FORM_READS: Final[Mapping[str, tuple[str, ...]]] = {
    FORM_AFFINE_X: ("x",), FORM_SUM_X: ("x", "y"), FORM_AFFINE_Y: ("x", "y"),
}
FORM_WRITES: Final[Mapping[str, str]] = {
    FORM_AFFINE_X: "x", FORM_SUM_X: "x", FORM_AFFINE_Y: "y",
}

MIX_AFFINE: Final[str] = "affine"
MIX_COUPLED: Final[str] = "coupled"
MIXES: Final[tuple[str, ...]] = (MIX_AFFINE, MIX_COUPLED)

#: The form draw pool of each operator mix (drawn uniformly, so the pool *is* the mix).
MIX_POOLS: Final[Mapping[str, tuple[str, ...]]] = {
    MIX_AFFINE: (FORM_AFFINE_X, FORM_AFFINE_X, FORM_AFFINE_Y, FORM_SUM_X),
    MIX_COUPLED: (FORM_SUM_X, FORM_AFFINE_Y, FORM_SUM_X, FORM_AFFINE_X),
}

#: depth x operator mix, in the frozen stratum order.
STRATA: Final[tuple[tuple[int, str], ...]] = tuple(
    (depth, mix) for depth in K.RECEIVER_DEPTHS for mix in MIXES
)
PROGRAMS_PER_STRATUM: Final[int] = K.RECEIVER_PROGRAMS // len(STRATA)

ORDER_CORRECT_FIRST: Final[str] = "CORRECT_FIRST"
ORDER_WRONG_FIRST: Final[str] = "WRONG_FIRST"
ORDERS: Final[tuple[str, ...]] = (ORDER_CORRECT_FIRST, ORDER_WRONG_FIRST)

PROVISIONAL_CORRECT: Final[str] = "PROVISIONAL_CORRECT"
PROVISIONAL_WRONG: Final[str] = "PROVISIONAL_WRONG"
PROVISIONALS: Final[tuple[str, ...]] = (PROVISIONAL_CORRECT, PROVISIONAL_WRONG)

#: The four schedule cells, balanced within every stratum (plan §5.5: 16 programs per cell).
SCHEDULE_CELLS: Final[tuple[tuple[str, str], ...]] = tuple(
    (order, provisional) for order in ORDERS for provisional in PROVISIONALS
)
PROGRAMS_PER_CELL_PER_STRATUM: Final[int] = PROGRAMS_PER_STRATUM // len(SCHEDULE_CELLS)

CANDIDATE_LABELS: Final[tuple[str, str]] = ("A", "B")

CHECK_VERIFIED: Final[str] = "VERIFIED"
CHECK_FAILED: Final[str] = "FAILED"
CHECK_NOT_CHECKED: Final[str] = "NOT CHECKED"

OUTCOME_PICKED_CORRECT: Final[str] = "picked-correct"
OUTCOME_PICKED_WRONG: Final[str] = "picked-wrong"
OUTCOME_PRODUCED_OTHER_CORRECT: Final[str] = "produced-other-correct"
OUTCOME_PRODUCED_OTHER_WRONG: Final[str] = "produced-other-wrong"
OUTCOME_INVALID: Final[str] = "invalid"
OUTCOMES: Final[tuple[str, ...]] = (
    OUTCOME_PICKED_CORRECT, OUTCOME_PICKED_WRONG, OUTCOME_PRODUCED_OTHER_CORRECT,
    OUTCOME_PRODUCED_OTHER_WRONG, OUTCOME_INVALID,
)

MARGIN_OK: Final[str] = "ok"
MARGIN_NO_LOGPROBS: Final[str] = "no_logprobs"
MARGIN_NO_DIFFERENCE: Final[str] = "candidates_share_an_answer"
MARGIN_TOKEN_NOT_FOUND: Final[str] = "answer_token_not_found"
MARGIN_OUTSIDE_TOP_K: Final[str] = "alternative_outside_top_k"
#: WP1's ``DecisionRecord`` keeps the per-token logprob but not the top-k alternatives, so a
#: caller that only has a record records the margin as *missing* under this status — never 0.
MARGIN_UNAVAILABLE: Final[str] = "top_logprobs_not_retained"
MARGIN_STATUSES: Final[tuple[str, ...]] = (
    MARGIN_OK, MARGIN_NO_LOGPROBS, MARGIN_NO_DIFFERENCE, MARGIN_TOKEN_NOT_FOUND,
    MARGIN_OUTSIDE_TOP_K, MARGIN_UNAVAILABLE,
)


class ReceiverError(ProtocolError):
    """A generated program, trace or packet violates the frozen E3 construction."""


# --------------------------------------------------------------------------- determinism

#: One frozen generator seed; every draw is an HMAC of it, so the 64 programs are a pure
#: function of the contract and never of a wall clock, a PID or a hash randomization.
_GENERATOR_SEED: Final[bytes] = hashlib.sha256(
    f"{K.SEED_NAMESPACE}|{K.NS_RECEIVER}|programs|v{K.CONTRACT_VERSION}".encode("utf-8")
).digest()


def _draw(*parts: Any) -> int:
    """A deterministic non-negative integer from the frozen generator seed."""
    return int.from_bytes(
        identity.blind_order_key(_GENERATOR_SEED, K.NS_RECEIVER, *parts), "big"
    )


def _pick(sequence: Sequence[Any], *parts: Any) -> Any:
    return sequence[_draw(*parts) % len(sequence)]


# --------------------------------------------------------------------------- the program


@dataclass(frozen=True)
class Update:
    """One permitted update (spec §5.4).  ``a``/``b`` are ``contract.RECEIVER_COEFFICIENTS``."""

    form: str
    a: int | None = None
    b: int | None = None

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ReceiverError(f"{self.form!r} is not one of the three permitted update forms")
        for name in ("a", "b"):
            value = getattr(self, name)
            if value is not None and int(value) not in K.RECEIVER_COEFFICIENTS:
                raise ReceiverError(
                    f"coefficient {name}={value} outside contract.RECEIVER_COEFFICIENTS")

    @property
    def target(self) -> str:
        return FORM_WRITES[self.form]

    @property
    def reads(self) -> tuple[str, ...]:
        return FORM_READS[self.form]

    def render(self) -> str:
        if self.form == FORM_AFFINE_X:
            return f"x = ({self.a}*x + {self.b}) % {MODULUS}"
        if self.form == FORM_SUM_X:
            return f"x = (x + y) % {MODULUS}"
        return f"y = (x + 2*y + {self.b}) % {MODULUS}"

    def apply(self, x: int, y: int) -> tuple[int, int]:
        """Exact sequential-assignment semantics, integer arithmetic, one modulus."""
        if self.form == FORM_AFFINE_X:
            return (int(self.a) * x + int(self.b)) % MODULUS, y
        if self.form == FORM_SUM_X:
            return (x + y) % MODULUS, y
        return x, (x + 2 * y + int(self.b)) % MODULUS


@dataclass(frozen=True)
class ProgramSchedule:
    """The frozen per-program presentation attributes (plan §5.5)."""

    order: str
    provisional: str

    @property
    def correct_label(self) -> str:
        """Which displayed candidate carries the correct trace."""
        return CANDIDATE_LABELS[0] if self.order == ORDER_CORRECT_FIRST else CANDIDATE_LABELS[1]

    @property
    def wrong_label(self) -> str:
        return CANDIDATE_LABELS[1] if self.order == ORDER_CORRECT_FIRST else CANDIDATE_LABELS[0]

    @property
    def provisional_is_correct(self) -> bool:
        return self.provisional == PROVISIONAL_CORRECT


@dataclass(frozen=True)
class Program:
    """One of the 64 frozen E3 programs and everything derived from it."""

    program_id: str
    index: int
    depth: int
    mix: str
    seed_x: int
    seed_y: int
    updates: tuple[Update, ...]
    schedule: ProgramSchedule
    error_update: int          # 0-based index into ``updates`` of the corrupted transition
    matched_update: int        # 0-based index of the matched valid transition (same form)
    error_delta: int           # the single arithmetic error, added mod 97 at ``error_update``

    # -- the rendered program ---------------------------------------------------------
    @property
    def statements(self) -> tuple[str, ...]:
        """Two initialisations followed by ``depth`` updates, one statement per line."""
        return (f"x = {self.seed_x}", f"y = {self.seed_y}") + tuple(u.render() for u in self.updates)

    @property
    def source(self) -> str:
        return "\n".join(self.statements)

    @property
    def n_statements(self) -> int:
        return 2 + len(self.updates)

    def statement_index(self, update_index: int) -> int:
        """0-based statement index of ``updates[update_index]`` (the two inits come first)."""
        return 2 + int(update_index)

    @property
    def stratum(self) -> tuple[int, str]:
        return (int(self.depth), str(self.mix))

    @property
    def gold(self) -> int:
        return interpret(self)[-1].value

    def identity_object(self) -> dict[str, Any]:
        return {
            "schema": "ard_receiver_program_v1",
            "program_id": self.program_id,
            "depth": int(self.depth),
            "mix": str(self.mix),
            "seed_x": int(self.seed_x),
            "seed_y": int(self.seed_y),
            "updates": [[u.form, u.a, u.b] for u in self.updates],
            "order": self.schedule.order,
            "provisional": self.schedule.provisional,
            "error_update": int(self.error_update),
            "matched_update": int(self.matched_update),
            "error_delta": int(self.error_delta),
        }

    @property
    def content_sha256(self) -> str:
        return identity.sha256_hex(identity.jcs(self.identity_object()))


@dataclass(frozen=True)
class TraceStep:
    """One executed statement: its text and the value it assigns."""

    index: int
    target: str
    text: str
    value: int


# --------------------------------------------------------------------------- allowlist

_INIT_RE = re.compile(r"^(x|y) = (\d{1,2})$")
_AFFINE_X_RE = re.compile(rf"^x = \((\d{{1,2}})\*x \+ (\d{{1,2}})\) % {MODULUS}$")
_SUM_X_RE = re.compile(rf"^x = \(x \+ y\) % {MODULUS}$")
_AFFINE_Y_RE = re.compile(rf"^y = \(x \+ 2\*y \+ (\d{{1,2}})\) % {MODULUS}$")
_LINE_FORMS: Final[tuple[re.Pattern[str], ...]] = (_INIT_RE, _AFFINE_X_RE, _SUM_X_RE, _AFFINE_Y_RE)
#: The only characters a rendered program may contain.
_ALLOWED_CHARS = re.compile(r"^[xy0-9+*%()= \n]*$")


def assert_permitted_program(source: str) -> None:
    """Allowlist check of a rendered program (spec §5.4 "disallow ...").

    Every line must match one of the four permitted statement forms and the whole text may
    only contain ``x``, ``y``, digits and ``+ * % ( ) =``.  Division, exponentiation,
    imports, loops, randomness, file access, calls, subscripting and attribute access are
    therefore all impossible — not because they are on a denylist, but because nothing
    outside the four forms can appear at all.
    """
    if not isinstance(source, str) or not source.strip():
        raise ReceiverError("a rendered program must be non-empty text")
    if not _ALLOWED_CHARS.match(source):
        illegal = sorted({c for c in source if not _ALLOWED_CHARS.match(c)})
        raise ReceiverError(f"rendered program carries characters outside the allowlist: {illegal}")
    for line_no, line in enumerate(source.splitlines(), 1):
        if not line.strip():
            raise ReceiverError(f"line {line_no}: a rendered program has no blank lines")
        if not any(pattern.match(line) for pattern in _LINE_FORMS):
            raise ReceiverError(
                f"line {line_no}: {line!r} is not one of the three permitted update forms "
                "or an initialisation (spec §5.4)")


def assert_state_dependent(program: Program) -> None:
    """Every update reads prior state and no assigned value is left unused (plan §5.5).

    "Each program must depend on prior state rather than computing independent unused
    values": each update's right-hand side reads at least one live variable, and the value
    written by every statement — the two initialisations included — is read by a later
    statement or is the program's output.
    """
    writes: list[str] = ["x", "y"] + [u.target for u in program.updates]
    reads: list[tuple[str, ...]] = [(), ()] + [u.reads for u in program.updates]
    last = len(writes) - 1
    if writes[last] != "x":
        raise ReceiverError(f"{program.program_id}: the final statement must produce the output x")
    for index, (target, read) in enumerate(zip(writes, reads)):
        if index >= 2 and not read:  # pragma: no cover - FORM_READS is never empty
            raise ReceiverError(f"{program.program_id}: update {index - 2} reads no prior state")
        if index == last:
            continue
        if not any(target in reads[later] for later in range(index + 1, last + 1)):
            raise ReceiverError(
                f"{program.program_id}: statement {index} assigns {target} and nothing later "
                "reads it — that is an independent unused value (plan §5.5)")


# --------------------------------------------------------------------------- interpreter


def interpret(program: Program, *, corrupt_at: int | None = None, delta: int = 0) -> tuple[TraceStep, ...]:
    """The generator's CPU interpreter: gold, or a consistently propagated corruption.

    Straight-line, sequential-assignment, one modulus, plain Python integers and no text
    parsing at all — which is precisely what makes it independent of
    ``ard/graders/trace.py``'s lexer/parser/stepper.  With ``corrupt_at`` set (an index into
    ``program.updates``) the value that update assigns is shifted by ``delta`` mod 97 and
    **every later step is recomputed from the perturbed state**, so the error propagates
    consistently instead of being a single isolated typo (spec §5.4).
    """
    x, y = int(program.seed_x) % MODULUS, int(program.seed_y) % MODULUS
    steps = [TraceStep(0, "x", f"x = {program.seed_x}", x),
             TraceStep(1, "y", f"y = {program.seed_y}", y)]
    for index, update in enumerate(program.updates):
        x, y = update.apply(x, y)
        if corrupt_at is not None and index == int(corrupt_at):
            if update.target == "x":
                x = (x + int(delta)) % MODULUS
            else:
                y = (y + int(delta)) % MODULUS
        value = x if update.target == "x" else y
        steps.append(TraceStep(program.statement_index(index), update.target, update.render(), value))
    return tuple(steps)


# --------------------------------------------------------------------------- traces


@dataclass(frozen=True)
class CandidateTrace:
    """One supplied candidate trace.  Neither is labelled correct to the model (spec §11)."""

    label: str
    kind: str                   # "correct" | "corrupted"
    steps: tuple[TraceStep, ...]
    corrupted_statement: int | None = None

    @property
    def final_answer(self) -> int:
        return self.steps[-1].value

    def render(self) -> str:
        """One line per statement, aligned by index with the program's statements.

        The ``<statement> = <value>`` shape is exactly what
        ``ard.graders.trace.parse_claimed_trace`` accepts, so the second evaluator can check
        a rendered trace step by step without a bespoke reader.
        """
        lines = []
        for step in self.steps:
            lines.append(step.text if step.index < 2 else f"{step.text} = {step.value}")
        return "\n".join(lines)

    def value_at(self, statement_index: int) -> int:
        for step in self.steps:
            if step.index == int(statement_index):
                return step.value
        raise ReceiverError(f"no statement {statement_index} in trace {self.label}")


def candidate_traces(program: Program) -> tuple[CandidateTrace, CandidateTrace]:
    """``(candidate A, candidate B)`` in the program's frozen display order.

    One is the interpreter's correct trace; the other carries the single arithmetic error
    at ``program.error_update`` propagated consistently.  Their final answers differ — the
    delta was chosen at build time to guarantee it (:func:`_choose_delta`).
    """
    correct = CandidateTrace("", "correct", interpret(program))
    corrupted = CandidateTrace(
        "", "corrupted",
        interpret(program, corrupt_at=program.error_update, delta=program.error_delta),
        corrupted_statement=program.statement_index(program.error_update),
    )
    if correct.final_answer == corrupted.final_answer:  # pragma: no cover - build-time guard
        raise ReceiverError(
            f"{program.program_id}: the corrupted trace reproduces the correct final answer")
    first, second = ((correct, corrupted) if program.schedule.order == ORDER_CORRECT_FIRST
                     else (corrupted, correct))
    return (
        CandidateTrace(CANDIDATE_LABELS[0], first.kind, first.steps, first.corrupted_statement),
        CandidateTrace(CANDIDATE_LABELS[1], second.kind, second.steps, second.corrupted_statement),
    )


# --------------------------------------------------------------------------- generation


def _form_sequence(depth: int, mix: str, program_id: str) -> list[str]:
    """The update forms of one program, then the two repairs that make it state-dependent.

    ``updates[-1]`` is forced to ``FORM_AFFINE_X`` (the output is ``x`` and the final
    transition varies by coefficient) and ``updates[-2]`` to ``FORM_SUM_X`` (so ``y`` — the
    initialisation and every later ``y`` write — is always read).  Together these give the
    liveness :func:`assert_state_dependent` demands for every program, whatever was drawn.
    """
    pool = MIX_POOLS[mix]
    forms = [str(_pick(pool, program_id, "form", index)) for index in range(int(depth))]
    forms[-1] = FORM_AFFINE_X
    forms[-2] = FORM_SUM_X
    return forms


def _coefficients(form: str, program_id: str, index: int) -> tuple[int | None, int | None]:
    if form == FORM_SUM_X:
        return None, None
    b = int(_pick(K.RECEIVER_COEFFICIENTS, program_id, "b", index))
    if form == FORM_AFFINE_Y:
        return None, b
    return int(_pick(K.RECEIVER_COEFFICIENTS, program_id, "a", index)), b


def _choose_delta(program_id: str, seed_x: int, seed_y: int, updates: Sequence[Update],
                  error_update: int) -> int:
    """The smallest deterministic non-zero shift whose final answers differ (spec §5.4)."""
    order = sorted(range(1, MODULUS), key=lambda d: _draw(program_id, "delta", d))
    base = _run(seed_x, seed_y, updates, None, 0)
    for delta in order:
        if _run(seed_x, seed_y, updates, error_update, delta) != base:
            return int(delta)
    raise ReceiverError(  # pragma: no cover - the composed maps are injective mod a prime
        f"{program_id}: no corruption of update {error_update} changes the final answer")


def _run(seed_x: int, seed_y: int, updates: Sequence[Update],
         corrupt_at: int | None, delta: int) -> int:
    x, y = int(seed_x) % MODULUS, int(seed_y) % MODULUS
    for index, update in enumerate(updates):
        x, y = update.apply(x, y)
        if corrupt_at is not None and index == int(corrupt_at):
            if update.target == "x":
                x = (x + int(delta)) % MODULUS
            else:
                y = (y + int(delta)) % MODULUS
    return x


def schedule() -> dict[str, ProgramSchedule]:
    """The recorded order x provisional schedule, balanced within depth x mix strata.

    Plan §5.5: "assigned by a recorded schedule balanced within depth x operator-mix strata
    (16 programs per order x provisional cell)".  Each of the four strata holds
    :data:`PROGRAMS_PER_STRATUM` programs and gives exactly
    :data:`PROGRAMS_PER_CELL_PER_STRATUM` of them to each of the four cells, so the
    stratified balance and the 16-per-cell total are both exact by construction; the
    *assignment* of which program lands in which cell is a correctness-blind HMAC
    permutation, so it cannot correlate with anything about the program.
    """
    out: dict[str, ProgramSchedule] = {}
    for stratum_index, (depth, mix) in enumerate(STRATA):
        key = f"{depth}|{mix}"
        positions = sorted(range(PROGRAMS_PER_STRATUM),
                           key=lambda p: identity.blind_order_key(
                               _GENERATOR_SEED, K.NS_RECEIVER, "schedule", key, int(p)))
        for rank, position in enumerate(positions):
            order, provisional = SCHEDULE_CELLS[rank // PROGRAMS_PER_CELL_PER_STRATUM]
            index = stratum_index * PROGRAMS_PER_STRATUM + position
            out[_program_id(index)] = ProgramSchedule(order, provisional)
    return out


def _program_id(index: int) -> str:
    return f"{PROGRAM_ID_PREFIX}{int(index):02d}"


def build_programs() -> tuple[Program, ...]:
    """The 64 frozen E3 programs (spec §5.4), balanced over depth x operator mix.

    Deterministic and side-effect free: every draw is an HMAC of the frozen generator seed
    and the program id, so this returns byte-identical programs in every process, on every
    machine, for the life of the contract.
    """
    plan = schedule()
    built: list[Program] = []
    for stratum_index, (depth, mix) in enumerate(STRATA):
        for position in range(PROGRAMS_PER_STRATUM):
            index = stratum_index * PROGRAMS_PER_STRATUM + position
            program_id = _program_id(index)
            forms = _form_sequence(depth, mix, program_id)
            seed_x = int(_pick(K.RECEIVER_SEEDS, program_id, "seed_x"))
            seed_y = int(_pick(K.RECEIVER_SEEDS, program_id, "seed_y"))
            # The corrupted transition is internal: never the first update (so the error has
            # a prior state to corrupt) and never the last (spec §11 excludes the final
            # transition and the final output from the checks).
            internal = list(range(1, depth - 1))
            error_update = int(_pick(internal, program_id, "error_update"))
            # One matched valid transition of the *same* update form, strictly earlier, so
            # both candidates claim the same value there and the check block is content
            # matched.  If the draw left no earlier twin, one earlier slot is rewritten to
            # the corrupted form — an earlier slot is never one of the two forced ones.
            earlier = [j for j in range(error_update) if forms[j] == forms[error_update]]
            if not earlier:
                target = int(_pick(list(range(error_update)), program_id, "matched_repair"))
                forms[target] = forms[error_update]
                earlier = [target]
            matched_update = int(_pick(earlier, program_id, "matched_update"))
            updates = tuple(
                Update(form, *_coefficients(form, program_id, position_index))
                for position_index, form in enumerate(forms)
            )
            delta = _choose_delta(program_id, seed_x, seed_y, updates, error_update)
            program = Program(
                program_id=program_id, index=index, depth=int(depth), mix=str(mix),
                seed_x=seed_x, seed_y=seed_y, updates=updates, schedule=plan[program_id],
                error_update=error_update, matched_update=matched_update, error_delta=delta,
            )
            assert_permitted_program(program.source)
            assert_state_dependent(program)
            built.append(program)
    if len(built) != K.RECEIVER_PROGRAMS:  # pragma: no cover - arithmetic above is exact
        raise ReceiverError(f"built {len(built)} programs, contract says {K.RECEIVER_PROGRAMS}")
    return tuple(built)


_PROGRAMS: list[Program] = []


def programs() -> tuple[Program, ...]:
    """The memoised frozen program set."""
    if not _PROGRAMS:
        _PROGRAMS.extend(build_programs())
    return tuple(_PROGRAMS)


def program_for(program_id: str) -> Program:
    for program in programs():
        if program.program_id == str(program_id):
            return program
    raise ReceiverError(f"no E3 program {program_id!r}")


# --------------------------------------------------------------------------- verification


def verify_programs(built: Sequence[Program] | None = None) -> dict[str, Any]:
    """Cross-check every gold against the second independent evaluator (spec §5.4).

    ``ard.graders.trace`` is imported **here**, inside the function, for the contract §0.2
    reason ``runner.grade_worker`` imports ``ard.graders`` inside itself: the package is the
    one that can read the protected namespace, and the firewall is a statement about the
    module-level import closure.  Nothing on the :func:`run_receiver_cell` path calls this —
    the E3 reference is computed, not read — so the runtime closure stays clear.

    Checks, for all 64 programs: the two evaluators agree on the gold; the correct trace
    verifies step by step; the corrupted trace first diverges at exactly the corrupted
    statement; and the two final answers differ.
    """
    from agents_scaling.ard.graders import trace as TR  # local: evaluator namespace only

    built = tuple(built if built is not None else programs())
    report: dict[str, Any] = {"n_programs": len(built), "disagreements": [], "golds": {}}
    for program in built:
        mine = interpret(program)[-1].value
        theirs = TR.evaluate_program(program.source, modulus=MODULUS)
        if theirs.final_target != "x" or theirs.final_value != mine:
            report["disagreements"].append({
                "program_id": program.program_id, "interpreter": mine,
                "trace_evaluator": theirs.final_value, "target": theirs.final_target})
        report["golds"][program.program_id] = mine
        a, b = candidate_traces(program)
        for candidate in (a, b):
            checks = TR.check_trace(program.source, candidate.render(), modulus=MODULUS)
            divergence = TR.first_divergence(checks)
            if candidate.kind == "correct":
                if divergence is not None:
                    report["disagreements"].append({
                        "program_id": program.program_id, "correct_trace_diverges_at":
                        divergence.index})
            else:
                expected = program.statement_index(program.error_update)
                if divergence is None or divergence.index != expected:
                    report["disagreements"].append({
                        "program_id": program.program_id,
                        "corrupted_trace_first_divergence":
                            None if divergence is None else divergence.index,
                        "expected": expected})
        if a.final_answer == b.final_answer:
            report["disagreements"].append(
                {"program_id": program.program_id, "identical_final_answers": a.final_answer})
    report["agrees"] = not report["disagreements"]
    return report


# --------------------------------------------------------------------------- the packets


@dataclass(frozen=True)
class LocalCheck:
    """One preregistered check location, with both candidates' claims at that location."""

    statement_index: int
    statement_text: str
    claims: tuple[tuple[str, int], ...]     # (candidate label, claimed value)
    statuses: tuple[tuple[str, str], ...]   # (candidate label, VERIFIED / FAILED / NOT CHECKED)
    is_corrupted_location: bool

    def render(self, ordinal: int) -> str:
        head = f"Check {ordinal} - step {self.statement_index + 1} [{self.statement_text}]"
        lines = [head]
        status = dict(self.statuses)
        for label, value in self.claims:
            lines.append(f"  Candidate {label} claims {value}: {status[label]}")
        return "\n".join(lines)


def check_locations(program: Program, *, checked: bool) -> tuple[LocalCheck, ...]:
    """The two preregistered locations (spec §11), in ascending statement order.

    Location 1 is the matched valid transition — the same update form, strictly before the
    corruption, so both candidates claim the same value and the check is content-matched.
    Location 2 is the corrupted internal transition.  Neither is the final transition and
    neither is the final output.  When ``checked`` is false the *same* two locations and the
    *same* claimed values appear with status ``NOT CHECKED`` and no extra explanation.
    """
    reference = {step.index: step for step in interpret(program)}
    a, b = candidate_traces(program)
    out: list[LocalCheck] = []
    for update_index, corrupted in ((program.matched_update, False), (program.error_update, True)):
        statement = program.statement_index(update_index)
        if statement >= program.n_statements - 1:  # pragma: no cover - build-time guarantee
            raise ReceiverError(
                f"{program.program_id}: a check location may not be the final transition")
        claims = tuple((candidate.label, candidate.value_at(statement)) for candidate in (a, b))
        if checked:
            truth = reference[statement].value
            statuses = tuple(
                (label, CHECK_VERIFIED if value == truth else CHECK_FAILED)
                for label, value in claims
            )
        else:
            statuses = tuple((label, CHECK_NOT_CHECKED) for label, _ in claims)
        out.append(LocalCheck(statement, reference[statement].text, claims, statuses, corrupted))
    if len(out) != K.RECEIVER_CHECKS_PER_PACKET:  # pragma: no cover
        raise ReceiverError(f"a checked packet reveals exactly {K.RECEIVER_CHECKS_PER_PACKET} checks")
    return tuple(out)


def endorsements(presentation: str, order: str) -> tuple[int, int]:
    """``(count for candidate A, count for candidate B)`` out of five (spec §11).

    ``MINORITY_*``: one of five endorsements favours the correct candidate, four favour the
    wrong one.  ``MAJORITY_*``: reversed.  Swapping the presentation therefore swaps the two
    numbers and changes nothing else, which is the byte-identity requirement.
    """
    if presentation == "PRIVATE_TRACE":
        raise ReceiverError("PRIVATE_TRACE carries no endorsement metadata")
    minority = presentation.startswith("MINORITY")
    correct_count, wrong_count = (1, 4) if minority else (4, 1)
    if order == ORDER_CORRECT_FIRST:
        return correct_count, wrong_count
    return wrong_count, correct_count


#: The one line that differs between the two count assignments.  Single digits keep the two
#: renderings the same length as well as the same shape.
ENDORSEMENT_HEAD: Final[str] = (
    "Endorsements (synthetic randomized metadata, not independent generation):"
)


def endorsement_line(count_a: int, count_b: int) -> str:
    return f"Candidate A: {int(count_a)} of 5. Candidate B: {int(count_b)} of 5."


@dataclass(frozen=True)
class ReceiverPacket:
    """One rendered E3 packet: the frozen content one model sees in one presentation."""

    program_id: str
    presentation: str
    text: str
    order: str
    provisional: str
    provisional_answer: int
    counts: tuple[int, int] | None
    checks: tuple[LocalCheck, ...]
    candidate_answers: tuple[int, int]
    correct_label: str
    gold: int

    @property
    def content_sha256(self) -> str:
        return identity.sha256_hex(self.text)

    def to_row(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "presentation": self.presentation,
            "order": self.order,
            "provisional": self.provisional,
            "provisional_answer": int(self.provisional_answer),
            "endorsement_counts": None if self.counts is None else list(self.counts),
            "n_checks": len(self.checks),
            "checked": bool(self.checks) and self.checks[0].statuses[0][1] != CHECK_NOT_CHECKED,
            "candidate_answers": list(self.candidate_answers),
            "correct_label": self.correct_label,
            "packet_sha256": self.content_sha256,
        }


_TASK_INSTRUCTION = (
    "Evaluate the straight-line integer program below modulo 97 and report the final value "
    "of x. All operations and inputs are visible; assignment is sequential."
)
_TRACE_NOTE = "Steps are numbered by line, starting at 1."


def receiver_output_contract() -> str:
    """The frozen three-tag report contract, with the E3 answer type spelled out."""
    return (
        "Output contract: reply with exactly these three tags and nothing else.\n"
        f"{K.FINAL_OPEN}the final value of x, an integer from 0 to {MODULUS - 1}{K.FINAL_CLOSE}\n"
        f"{K.EVIDENCE_OPEN}one or two concise, checkable reasons, at most "
        f"{K.EVIDENCE_MAX_TOKENS} tokens{K.EVIDENCE_CLOSE}\n"
        f"{K.CONFIDENCE_OPEN}probability from 0 to 1{K.CONFIDENCE_CLOSE}\n"
        "The answer precedes the confidence. The whole report is at most "
        f"{K.FINAL_TOKENS} tokens."
    )


def build_packet(program: Program, presentation: str) -> ReceiverPacket:
    """Render one of the five presentations of one program (spec §11).

    ``MINORITY_*`` and ``MAJORITY_*`` of the same checkability differ **only** in
    :func:`endorsement_line`; the checked and unchecked counterparts show the same two check
    locations with the same claimed values and differ only in the status column.
    """
    if presentation not in K.RECEIVER_PRESENTATIONS:
        raise ReceiverError(
            f"unknown presentation {presentation!r}; frozen set is {list(K.RECEIVER_PRESENTATIONS)}")
    a, b = candidate_traces(program)
    correct = a if a.kind == "correct" else b
    wrong = b if a.kind == "correct" else a
    provisional_trace = correct if program.schedule.provisional_is_correct else wrong
    provisional_answer = provisional_trace.final_answer

    blocks = [_TASK_INSTRUCTION, "", "Program:", program.source, "", _TRACE_NOTE, "",
              f"Provisional answer: {provisional_answer}"]
    counts: tuple[int, int] | None = None
    checks: tuple[LocalCheck, ...] = ()
    if presentation == "PRIVATE_TRACE":
        blocks += ["Provisional trace:", provisional_trace.render()]
    else:
        checks = check_locations(program, checked=presentation.endswith("_CHECKED"))
        counts = endorsements(presentation, program.schedule.order)
        blocks += ["",
                   f"Candidate {a.label}:", a.render(),
                   "",
                   f"Candidate {b.label}:", b.render(),
                   "",
                   ENDORSEMENT_HEAD, endorsement_line(*counts),
                   "",
                   "Local checks:"]
        blocks += [check.render(ordinal) for ordinal, check in enumerate(checks, 1)]
    blocks += ["", receiver_output_contract()]
    return ReceiverPacket(
        program_id=program.program_id,
        presentation=str(presentation),
        text="\n".join(blocks),
        order=program.schedule.order,
        provisional=program.schedule.provisional,
        provisional_answer=int(provisional_answer),
        counts=counts,
        checks=checks,
        candidate_answers=(a.final_answer, b.final_answer),
        correct_label=program.schedule.correct_label,
        gold=correct.final_answer,
    )


def render_receiver_prompt(packet: ReceiverPacket) -> tuple[dict[str, str], ...]:
    """The single user turn of one receiver decision."""
    return ({"role": "user", "content": packet.text},)


# --------------------------------------------------------------------------- the margin


@dataclass(frozen=True)
class AnswerLogitMargin:
    """The answer-logit margin between the two supplied candidates (spec §11, plan §5.5).

    ``value`` is ``logprob(chosen token) - logprob(alternative token)`` at the first answer
    token that distinguishes the two supplied candidates.  It is **missing, not zero**, when
    the alternative falls outside the requested top-k, and the ``status`` says which
    missingness it is.
    """

    value: float | None
    status: str
    position: int | None = None
    chosen_token: str | None = None
    alternative_token: str | None = None
    chosen_logprob: float | None = None
    alternative_logprob: float | None = None
    top_k: int = K.RECEIVER_TOP_LOGPROBS
    realized_top_k: int = 0

    @property
    def missing(self) -> bool:
        return self.value is None

    def to_row(self) -> dict[str, Any]:
        return {
            "answer_logit_margin": self.value,
            "answer_logit_margin_status": self.status,
            "answer_logit_margin_missing": self.value is None,
            "answer_logit_margin_position": self.position,
            "chosen_token": self.chosen_token,
            "alternative_token": self.alternative_token,
            "chosen_logprob": self.chosen_logprob,
            "alternative_logprob": self.alternative_logprob,
            "top_logprobs_requested": int(self.top_k),
            "top_logprobs_realized": int(self.realized_top_k),
        }


def first_differing_token(
    chosen: str, alternative: str, *, tokenize: Callable[[str], Sequence[str]] | None = None
) -> tuple[int, str | None, str | None] | None:
    """``(index, chosen token, alternative token)`` of the first difference, or ``None``.

    The default tokenisation is per character, which is the right granularity for the E3
    answers (a residue in 0..96): "41" and "58" first differ at index 0, "41" and "48" at
    index 1.  A caller with the serving tokenizer may inject its own.
    """
    cut = list(tokenize(chosen)) if tokenize else list(str(chosen))
    alt = list(tokenize(alternative)) if tokenize else list(str(alternative))
    for index in range(min(len(cut), len(alt))):
        if cut[index] != alt[index]:
            return index, cut[index], alt[index]
    if len(cut) == len(alt):
        return None
    index = min(len(cut), len(alt))
    return index, (cut[index] if index < len(cut) else None), (alt[index] if index < len(alt) else None)


def _entry_spans(content: Sequence[Mapping[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for entry in content:
        token = str(entry.get("token") or "")
        spans.append((offset, offset + len(token)))
        text_parts.append(token)
        offset += len(token)
    return "".join(text_parts), spans


def answer_logit_margin(
    logprobs_content: Sequence[Mapping[str, Any]] | None,
    *,
    chosen_answer: str,
    alternative_answer: str,
    top_k: int = K.RECEIVER_TOP_LOGPROBS,
    tokenize: Callable[[str], Sequence[str]] | None = None,
) -> AnswerLogitMargin:
    """The answer-logit margin from an OpenAI ``choices[0].logprobs.content`` block.

    ``logprobs_content`` is the per-generated-token list the server returns when
    ``logprobs=true, top_logprobs=k``; WP1 sets ``k = contract.RECEIVER_TOP_LOGPROBS`` for
    :data:`agents_scaling.ard.contract.PURPOSE_RECEIVER` requests
    (``client.top_logprobs_for``), which is what makes this computable at all.

    The token carrying the first difference is located inside the ``<final>`` span of the
    reconstructed completion text; the alternative is the highest-ranked top-k entry whose
    token (whitespace stripped) is a prefix of the alternative answer's tail at that
    position.  When no such entry exists the alternative was outside the top-k and the
    margin is missing — :data:`MARGIN_OUTSIDE_TOP_K` — never 0.
    """
    if not logprobs_content:
        return AnswerLogitMargin(None, MARGIN_NO_LOGPROBS, top_k=int(top_k))
    difference = first_differing_token(str(chosen_answer), str(alternative_answer),
                                       tokenize=tokenize)
    if difference is None:
        return AnswerLogitMargin(None, MARGIN_NO_DIFFERENCE, top_k=int(top_k))
    index, _chosen_char, alternative_char = difference
    if alternative_char is None:
        return AnswerLogitMargin(None, MARGIN_NO_DIFFERENCE, top_k=int(top_k))

    text, spans = _entry_spans(logprobs_content)
    search_from = 0
    marker = text.find(K.FINAL_OPEN)
    if marker >= 0:
        search_from = marker + len(K.FINAL_OPEN)
    answer_start = text.find(str(chosen_answer), search_from)
    if answer_start < 0:
        return AnswerLogitMargin(None, MARGIN_TOKEN_NOT_FOUND, top_k=int(top_k))
    offset = answer_start + index
    position = next((i for i, (start, stop) in enumerate(spans) if start <= offset < stop), None)
    if position is None:
        return AnswerLogitMargin(None, MARGIN_TOKEN_NOT_FOUND, top_k=int(top_k))

    entry = logprobs_content[position]
    alternatives = list(entry.get("top_logprobs") or ())
    chosen_logprob = entry.get("logprob")
    tail = str(alternative_answer)[index:]
    for candidate in alternatives:
        token = str(candidate.get("token") or "")
        stripped = token.strip()
        if stripped and (tail.startswith(stripped) or stripped.startswith(tail)):
            value = float(chosen_logprob) - float(candidate.get("logprob"))
            return AnswerLogitMargin(
                value, MARGIN_OK, position=position, chosen_token=str(entry.get("token") or ""),
                alternative_token=token, chosen_logprob=float(chosen_logprob),
                alternative_logprob=float(candidate.get("logprob")), top_k=int(top_k),
                realized_top_k=len(alternatives),
            )
    return AnswerLogitMargin(
        None, MARGIN_OUTSIDE_TOP_K, position=position,
        chosen_token=str(entry.get("token") or ""),
        chosen_logprob=None if chosen_logprob is None else float(chosen_logprob),
        top_k=int(top_k), realized_top_k=len(alternatives),
    )


def margin_from_record(record: Any, packet: ReceiverPacket, *, produced: str | None) -> AnswerLogitMargin:
    """The margin of one decision, or the honest missing status when it cannot be had.

    ``client.DecisionRecord`` keeps the *mean* final-span logprob and the per-token
    logprobs, but the study client does not retain the per-token top-k alternatives, so a
    record alone cannot supply the alternative's logit.  A caller that has the raw
    ``logprobs.content`` (a live response, or a future WP1 field) passes it on the record as
    ``top_logprobs``/``logprobs_content``; otherwise the margin is recorded as missing under
    :data:`MARGIN_UNAVAILABLE` — never 0 (plan §5.5).
    """
    content = (getattr(record, "logprobs_content", None)
               or getattr(record, "top_logprobs", None))
    if not content:
        return AnswerLogitMargin(None, MARGIN_UNAVAILABLE, top_k=int(K.RECEIVER_TOP_LOGPROBS))
    first, second = packet.candidate_answers
    chosen = str(produced if produced is not None else first)
    alternative = str(second if str(first) == chosen else first)
    return answer_logit_margin(content, chosen_answer=chosen, alternative_answer=alternative)


# --------------------------------------------------------------------------- grading

_INT_RE = re.compile(r"-?\d+")


def extract_residue(answer: str | None, *, modulus: int = MODULUS) -> int | None:
    """The last integer in ``answer`` that is a valid residue, or ``None``.

    Vendored rather than imported from ``ard.graders.trace`` for the contract §0.2 reason
    ``graders/_boxed.py`` is vendored: this function is on the runtime path and the grader
    package is the one that can read the protected namespace.  ``tests/ard/test_wp5_receiver.py``
    asserts parity with ``trace.extract_residue`` so the copy cannot drift.
    """
    if answer is None:
        return None
    for value in reversed([int(token) for token in _INT_RE.findall(str(answer))]):
        if 0 <= value < int(modulus):
            return value
    return None


@dataclass(frozen=True)
class ReceiverOutcome:
    """One graded receiver decision (spec §11 primary assay outcomes)."""

    program_id: str
    presentation: str
    outcome: str
    residue: int | None
    gold: int
    answer_raw: str | None
    confidence: float | None
    picked_label: str | None
    order: str
    provisional: str
    provisional_answer: int
    provisional_correct: bool
    endorsement_counts: tuple[int, int] | None
    checked: bool
    depth: int
    mix: str
    margin: AnswerLogitMargin

    @property
    def correct(self) -> bool:
        return self.residue is not None and int(self.residue) == int(self.gold)

    @property
    def retained(self) -> bool | None:
        """Kept a correct provisional answer (spec §11); ``None`` when it was wrong."""
        if not self.provisional_correct:
            return None
        return self.residue is not None and int(self.residue) == int(self.provisional_answer)

    @property
    def corrected(self) -> bool | None:
        """Corrected a wrong provisional answer; ``None`` when it was already correct."""
        if self.provisional_correct:
            return None
        return self.correct

    @property
    def switched_to_correct(self) -> bool:
        return (not self.provisional_correct) and self.correct

    def to_row(self) -> dict[str, Any]:
        return {
            "event": "receiver_outcome",
            "program_id": self.program_id,
            "presentation": self.presentation,
            "outcome": self.outcome,
            "residue": self.residue,
            "gold": int(self.gold),
            "answer_raw": self.answer_raw,
            "final_confidence": self.confidence,
            "final_confidence_missing": self.confidence is None,
            "picked_label": self.picked_label,
            "order": self.order,
            "provisional": self.provisional,
            "provisional_answer": int(self.provisional_answer),
            "provisional_correct": bool(self.provisional_correct),
            "endorsement_counts": None if self.endorsement_counts is None
            else list(self.endorsement_counts),
            "checked": bool(self.checked),
            "depth": int(self.depth),
            "operator_mix": self.mix,
            "correct": self.correct,
            "retained": self.retained,
            "corrected": self.corrected,
            "switched_to_correct": self.switched_to_correct,
            **self.margin.to_row(),
        }


def grade_receiver(
    program: Program,
    packet: ReceiverPacket,
    report: Any,
    *,
    record: Any | None = None,
) -> ReceiverOutcome:
    """Grade one receiver decision against the interpreter gold (spec §11).

    "The model can also reject both supplied choices; grade its produced answer rather than
    forcing agreement with the packet" — so the class is decided by the *produced residue*
    against :func:`interpret`'s reference, and matching a supplied candidate only decides
    ``picked-*`` versus ``produced-other-*``.  An unparseable report, or one with no residue
    in ``0..96``, is :data:`OUTCOME_INVALID`; its confidence is still recorded, because a
    missing confidence is a missing feature and never 0.5 (spec §6.3).
    """
    answer = getattr(report, "final_answer", None)
    valid = bool(getattr(report, "valid", False))
    residue = extract_residue(answer) if valid else None
    gold = packet.gold
    first, second = packet.candidate_answers
    picked_label: str | None = None
    if residue is not None:
        if residue == first:
            picked_label = CANDIDATE_LABELS[0]
        elif residue == second:
            picked_label = CANDIDATE_LABELS[1]
    if residue is None:
        outcome = OUTCOME_INVALID
    elif picked_label is not None:
        outcome = OUTCOME_PICKED_CORRECT if residue == gold else OUTCOME_PICKED_WRONG
    else:
        outcome = (OUTCOME_PRODUCED_OTHER_CORRECT if residue == gold
                   else OUTCOME_PRODUCED_OTHER_WRONG)
    margin = (margin_from_record(record, packet, produced=None if residue is None else str(residue))
              if record is not None
              else AnswerLogitMargin(None, MARGIN_UNAVAILABLE, top_k=int(K.RECEIVER_TOP_LOGPROBS)))
    return ReceiverOutcome(
        program_id=program.program_id,
        presentation=packet.presentation,
        outcome=outcome,
        residue=residue,
        gold=int(gold),
        answer_raw=answer,
        confidence=getattr(report, "confidence", None),
        picked_label=picked_label,
        order=packet.order,
        provisional=packet.provisional,
        provisional_answer=int(packet.provisional_answer),
        provisional_correct=int(packet.provisional_answer) == int(gold),
        endorsement_counts=packet.counts,
        checked=bool(packet.checks) and packet.checks[0].statuses[0][1] != CHECK_NOT_CHECKED,
        depth=int(program.depth),
        mix=str(program.mix),
        margin=margin,
    )


# --------------------------------------------------------------------------- item rows


def receiver_task(program: Program) -> Any:
    """The synthetic :class:`~agents_scaling.ard.items.ArdTask` row of one program.

    ``cells.build_e3_receiver`` takes the programs as ``ArdTask`` rows; they are synthetic
    (no dataset revision, no gold pointer, no options) and their split is ``test`` so
    ``features.training_eligibility`` marks them ``excluded_test`` and no receiver row can
    ever enter controller training (spec §12.1).  ``family`` is
    :data:`RECEIVER_TASK_FAMILY` because WP4's frozen ``cells.interleave_families`` refuses
    anything outside ``contract.FAMILIES``; the E3 tag travels in ``stratum`` and the row is
    told apart by its ``receiver`` cohort, never by its family column.
    """
    from agents_scaling.ard.items import ArdTask  # local: keeps pyarrow off the import path

    packet = build_packet(program, K.RECEIVER_PRESENTATIONS[0])
    return ArdTask(
        item_id=program.program_id,
        family=RECEIVER_TASK_FAMILY,
        dataset_revision="",
        source_id=program.program_id,
        split="test",
        group_id=program.program_id,
        prompt_hash=packet.content_sha256,
        option_permutation="[]",
        gold_location="",                 # the E3 reference is computed, never stored
        grader_hash=program.content_sha256,
        subject=f"{program.depth}|{program.mix}",
        declared_difficulty_metadata=str(program.depth),
        n_options=0,
        answer_format="exactMatch",
        stratum=f"{RECEIVER_FAMILY}:{program.depth}:{program.mix}",
        rank=int(program.index),
        task_text=packet.text,
        options_json="[]",
    )


def receiver_tasks() -> dict[str, Any]:
    """``{program_id: ArdTask}`` for all 64 programs."""
    return {program.program_id: receiver_task(program) for program in programs()}


def receiver_task_loader(run_root: str | os.PathLike) -> dict[str, Any]:
    """A ``runner.ArdCellRunner(task_loader=...)`` that adds the 64 synthetic programs.

    The E3 programs are generated, not published in ``items.parquet``, so the frozen study
    resume rule (``study/runner._todo``: "names <id>, absent from the public export") would
    refuse a RECEIVER cell built on them.  This loader is the seam: WP4's runner already
    accepts ``task_loader``, and passing this one makes a RECEIVER cell resolvable without
    editing the item table or the dispatcher.
    """
    from agents_scaling.ard.items import load_ard_tasks

    tasks: dict[str, Any] = {}
    try:
        tasks.update(load_ard_tasks(run_root))
    except Exception:  # noqa: BLE001 - a run root without a public table is still valid here
        pass
    tasks.update(receiver_tasks())
    return tasks


# --------------------------------------------------------------------------- the cell


def run_receiver_cell(runner: Any, item_id: str) -> None:
    """``RECEIVER`` worker (``kinds.KIND_TABLE``): one frozen-content decision per program.

    Signature is WP4's item-worker seam (``runner.resolve_worker`` →
    ``worker(runner, item_id)``).  One logical decision per (program, presentation, budget,
    completion seed, checkpoint) — spec §11's 64x5x2x2x2 = 2,560 ceiling — graded against
    the interpreter gold, with the outcome class, the final self-reported confidence and the
    answer-logit margin recorded on the episode.
    """
    from agents_scaling.ard.report import parse_decision
    from agents_scaling.ard.types import DecisionRef, EpisodeRecord, decision_id_for, episode_id_for

    started = runner.clock()
    cell = runner.ard_cell
    program = program_for(item_id)
    task = runner.tasks.get(item_id) if hasattr(runner.tasks, "get") else None
    if task is None:
        task = receiver_task(program)
    presentation = str(cell.options.get("presentation") or K.RECEIVER_PRESENTATIONS[0])
    packet = build_packet(program, presentation)
    step_slot = K.RECEIVER_PRESENTATIONS.index(presentation)

    record = runner.decide(
        task=task,
        messages=render_receiver_prompt(packet),
        purpose=K.PURPOSE_RECEIVER,
        agent_id=0,
        step_slot=step_slot,
        namespace=K.NS_RECEIVER,
        episode_rep=cell.episode_rep,
    )
    report = parse_decision(record)
    outcome = grade_receiver(program, packet, report, record=record)
    episode_id = episode_id_for(cell, item_id, episode_rep=cell.episode_rep)
    decision = DecisionRef(
        decision_id=decision_id_for(episode_id, round_=0, agent_id=0, phase=K.PURPOSE_RECEIVER),
        episode_id=episode_id,
        request_id=record.request_id,
        agent_id=0,
        round=0,
        phase=K.PURPOSE_RECEIVER,
        protocol=cell.protocol.value if cell.protocol else "NONE",
        checkpoint=cell.checkpoint,
        n_packets=0 if presentation == "PRIVATE_TRACE" else 2,
        answer_raw=report.final_answer,
        answer_class=None if outcome.residue is None else f"mod{MODULUS}:{outcome.residue}",
        content_sha256=report.content_sha256,
        evidence_text=report.evidence,
        confidence=report.confidence,
        parse_status=report.parse_status.value,
        thinking_tokens=int(record.thinking_tokens),
        final_tokens=int(record.final_tokens),
        prompt_tokens=int(record.prompt_tokens),
        cached_prefill_tokens=int(record.cached_prefill_tokens),
        budget_tokens=int(cell.reasoning_allowance),
        cap_hit=bool(record.cap_hit),
        boundary_injected=bool(record.boundary_injected),
        final_overrun=bool(record.final_overrun),
        answer_logprob=record.answer_logprob,
        finish_reason=str(record.finish_reason),
        latency_s=float(record.latency_s),
        endpoint_id=str(record.endpoint_id),
        flops=dict(record.flops or {}),
    )
    episode = EpisodeRecord(
        episode_id=episode_id,
        item_id=item_id,
        cohort=cell.cohort.value,
        protocol=cell.protocol.value if cell.protocol else "NONE",
        checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS.get(cell.checkpoint, ""),
        reasoning_allowance=int(cell.reasoning_allowance),
        final_allowance=int(cell.final_allowance),
        packet_policy=f"receiver:{presentation}",
        parent_ref=cell.parent_ref,
        condition_manifest_hash=program.content_sha256,
        decisions=(decision,),
        events=({"event": "receiver_packet", **packet.to_row()}, outcome.to_row()),
    )
    runner.write_item(task, (episode,), started=started)


# Import-time contract: the frozen constants and this module's layout must agree.
if len(STRATA) * PROGRAMS_PER_STRATUM != K.RECEIVER_PROGRAMS:  # pragma: no cover
    raise RuntimeError("depth x mix strata do not partition contract.RECEIVER_PROGRAMS")
if PROGRAMS_PER_CELL_PER_STRATUM * len(SCHEDULE_CELLS) != PROGRAMS_PER_STRATUM:  # pragma: no cover
    raise RuntimeError("the order x provisional schedule is not balanced within a stratum")
