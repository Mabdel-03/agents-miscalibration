"""The E2b synthetic mod-97 generator (spec 9.2; plan §1.8).

E2b exists because natural evidence is not *checkable*: a receiver cannot verify a MATH
rationale line by line, so an endorsement effect there confounds "the cue moved me" with "I
could not tell".  A mod-97 straight-line program fixes that: every displayed row is a local
equation the receiver (or the spec-9.3 tool) can test on its own, and the dependency depth of
the answer is a knob the generator controls at a *fixed* displayed row count.

What this module guarantees, and why each guarantee is structural rather than a convention:

* **24 displayed operations at every depth.**  Depth must change dependency structure and
  error localization without changing how much text the receiver reads, so off-chain nodes are
  not dropped -- they hang off ``x0`` and stay on the page.
* **An answer chain of exact depth D ending at node 24.**  "Exact" is asserted by walking the
  parent map, not assumed from how the chain was built.
* **Exactly one violated local equation in the wrong trace**, at a *nonterminal* node **on**
  the chain, with descendants recomputed consistently.  A violation off the chain would not
  reach the answer; a violation at the terminal node would be the answer itself and could not
  be found by local checking.  The multiplier is a unit modulo the prime 97, so a nonzero
  delta survives the rest of the chain -- and :func:`wrong_trace` asserts both the single
  violation and the differing finals rather than trusting that argument.
* **Generator and evaluator are separate implementations.**  ``ard.graders.trace`` is the
  independent evaluator and this module imports *nothing* from it
  (``test_mod97_evaluator_independent_of_generator`` is the AST guard).  The rendered byte
  format below is chosen so that evaluator parses it unchanged.

Rendering contract (spec 9.2, and the reason it is byte-exact).  ``ard.graders.trace``'s
grammar accepts identifiers, integers, ``+ - * % ( ) =`` and one trailing ``% m``; implicit
multiplication, ``mod 97``, subscripted names and an undefined ``x0`` all fail to parse, and
``check_trace`` aligns claimed rows by statement index.  So :func:`render_problem` emits
``x0 = <int>`` as its first statement and then one line per node::

    x0 = 42
    x7 = (23*x3 + 5) % 97

and :func:`render_trace` emits the same statements with the parent *substituted*, so a reader
sees the arithmetic it is being asked to check::

    x0 = 42
    x7 = (23*40 + 5) % 97 = 78

No experimental number lives here: the operation count, the admitted depths and the modulus
come from ``study.yaml``'s ``experiments.E2b`` block through :func:`spec_from_config`
(``config.CONSUMERS`` names this module as their consumer).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study import identity

#: The relative chain-position strata the injected error is balanced over (plan §1.8).  Three
#: bins is a *design* constant of the balancing scheme, not an experimental quantity: it says
#: "early, middle, late", which is the resolution the spec asks for ("balance error position
#: within relative chain-position bins where possible").
CHAIN_POSITION_BINS: Final[tuple[str, ...]] = ("early", "middle", "late")

#: The variable prefix of the rendered program.  ``x0`` is the input; ``x1..xN`` are the
#: displayed operations.  One prefix, declared once, used by both renderers and the parser
#: of :func:`step_id_of`.
VAR_PREFIX: Final[str] = "x"

#: The input variable ``x0``: every off-chain node and the first chain node read it, and it is
#: the first statement of the rendered program.  It is never a valid ``check_step`` step id
#: (:func:`step_ids` excludes it).  The node the *question* asks for is the answer node --
#: ``Mod97Spec.answer_node`` / ``Problem.answer_node``, "the last displayed operation".
INPUT_NODE: Final[int] = 0

PROBLEM_ID_PREFIX: Final[str] = "m97"
PROBLEM_ID_CHARS: Final[int] = 13


# --------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class Mod97Spec:
    """The generator's shape: every value comes from ``study.yaml`` ``experiments.E2b``."""

    n_operations: int
    chain_depths: tuple[int, ...]
    modulus: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_depths", tuple(int(d) for d in self.chain_depths))
        if int(self.n_operations) < 2:
            raise ContractViolation("a mod-97 problem needs at least two displayed operations")
        if int(self.modulus) < 3:
            raise ContractViolation(f"modulus {self.modulus!r} is not a usable modulus")
        if not self.chain_depths:
            raise ContractViolation("experiments.E2b.chain_depths is empty")
        for depth in self.chain_depths:
            if not 2 <= depth <= int(self.n_operations):
                raise ContractViolation(
                    f"chain depth {depth} does not fit {self.n_operations} displayed operations "
                    "(a depth of 1 has no nonterminal chain node to corrupt)")

    @property
    def answer_node(self) -> int:
        """The node the question asks for: the last displayed operation (spec 9.2)."""
        return int(self.n_operations)


def spec_from_config(e2b_block: Any) -> Mod97Spec:
    """Compile ``study.yaml``'s ``experiments.E2b`` block into the generator's shape.

    Reads exactly the three leaves ``config.CONSUMERS`` routes to this module:
    ``total_displayed_operations``, ``chain_depths`` and ``modulus``.
    """
    return Mod97Spec(
        n_operations=int(e2b_block.total_displayed_operations),
        chain_depths=tuple(int(d) for d in e2b_block.chain_depths),
        modulus=int(e2b_block.modulus),
    )


# --------------------------------------------------------------------------- problems


@dataclass(frozen=True)
class Op:
    """One displayed operation ``x_j = (a * x_parent + b) % m``."""

    j: int
    parent: int
    a: int
    b: int


@dataclass(frozen=True)
class Problem:
    """One source problem: 24 operations, an answer chain of exact depth, one input value."""

    problem_id: str
    depth: int
    x0: int
    ops: tuple[Op, ...]
    modulus: int
    chain: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", tuple(self.ops))
        object.__setattr__(self, "chain", tuple(int(c) for c in self.chain))
        nodes = [op.j for op in self.ops]
        if nodes != list(range(1, len(self.ops) + 1)):
            raise ContractViolation("operations must be displayed in topological order 1..n")
        for op in self.ops:
            if op.parent >= op.j:
                raise ContractViolation(
                    f"x{op.j} depends on x{op.parent}: a displayed operation may only read an "
                    "earlier variable (topological order, spec 9.2)")
            if not 1 <= op.a < self.modulus:
                raise ContractViolation(f"multiplier {op.a} is outside 1..{self.modulus - 1}")
            if not 0 <= op.b < self.modulus:
                raise ContractViolation(f"offset {op.b} is outside 0..{self.modulus - 1}")
        if not 0 <= self.x0 < self.modulus:
            raise ContractViolation(f"x0 {self.x0} is outside 0..{self.modulus - 1}")
        if len(self.chain) != int(self.depth) or self.chain[-1] != len(self.ops):
            raise ContractViolation("the answer chain must have exactly `depth` nodes and end at "
                                    "the answer node")

    @property
    def n_operations(self) -> int:
        return len(self.ops)

    @property
    def answer_node(self) -> int:
        return len(self.ops)

    @property
    def item_variant_id(self) -> str:
        """The single variant of a generated problem (``CompositeKey.item_variant_id``)."""
        return f"{self.problem_id}.v0"

    @property
    def source_item_id(self) -> str:
        """A generated problem *is* its own source item (there is no dataset row)."""
        return self.problem_id

    @property
    def rendered_question(self) -> str:
        """``prompts.TaskItem``: the program plus the question, and nothing about the arm."""
        return (f"{render_problem(self)}\n\n"
                f"What is the value of {VAR_PREFIX}{self.answer_node}?")

    def op_of(self, j: int) -> Op:
        for op in self.ops:
            if op.j == int(j):
                return op
        raise ContractViolation(f"no displayed operation {VAR_PREFIX}{j}")

    def depth_of(self, j: int) -> int:
        """Dependency depth of node ``j``: how many operations separate it from ``x0``."""
        steps = 0
        node = int(j)
        while node != INPUT_NODE:
            node = self.op_of(node).parent
            steps += 1
            if steps > self.n_operations:  # pragma: no cover - refused at construction
                raise ContractViolation("cycle in the displayed operations")
        return steps


# --------------------------------------------------------------------------- traces


@dataclass(frozen=True)
class TraceRow:
    """One displayed evaluation row: what the trace *claims* about one operation."""

    j: int
    parent_value_stated: int
    output_stated: int


@dataclass(frozen=True)
class Trace:
    """One displayed trace of one problem: all n rows, the claimed final, the violation.

    The trace carries its :class:`Problem` so that it can render itself (the displayed row
    states the coefficients *and* the substituted parent value, which is exactly what the
    spec-9.3 tool checks) without the coefficients being copied into a second declaration.
    """

    problem: Problem
    rows: tuple[TraceRow, ...]
    final: int
    violated_step: int | None
    error_position_rel: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        if len(self.rows) != self.problem.n_operations:
            raise ContractViolation("a trace displays every row of its problem")
        if self.violated_step is None and self.error_position_rel is not None:
            raise ContractViolation("a correct trace has no error position")
        if self.violated_step is not None and self.error_position_rel is None:
            raise ContractViolation("a wrong trace records where its error sits")

    @property
    def problem_id(self) -> str:
        return self.problem.problem_id

    @property
    def x0(self) -> int:
        return self.problem.x0

    @property
    def correct(self) -> bool:
        return self.violated_step is None

    def row_of(self, j: int) -> TraceRow | None:
        for row in self.rows:
            if row.j == int(j):
                return row
        return None


# --------------------------------------------------------------------------- seeded stream


class _Stream:
    """A deterministic, unbounded stream of integers derived from one seed.

    Not ``random``: the draw sequence is part of the study's identity and must reproduce
    byte-for-byte on any machine and any Python version, so it is a plain SHA-256 counter.
    """

    def __init__(self, seed: bytes, *label: Any) -> None:
        self._base = hashlib.sha256(bytes(seed) + identity.jcs(list(label))).digest()
        self._counter = 0

    def _next_bytes(self) -> bytes:
        block = hashlib.sha256(self._base + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return block

    def below(self, n: int) -> int:
        """A uniform integer in ``0..n-1`` (rejection-sampled, so no modulo bias)."""
        if n < 1:
            raise ContractViolation("below(n) needs n >= 1")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = int.from_bytes(self._next_bytes()[:8], "big")
            if value < limit:
                return value % n

    def sample(self, population: Sequence[int], k: int) -> list[int]:
        """``k`` distinct members of ``population``, drawn without replacement."""
        pool = list(population)
        if k > len(pool):
            raise ContractViolation(f"cannot draw {k} distinct nodes from {len(pool)}")
        out = []
        for _ in range(k):
            out.append(pool.pop(self.below(len(pool))))
        return out


# --------------------------------------------------------------------------- generation


def generate(seed: bytes, depth: int, index: int, *, spec: Mod97Spec) -> Problem:
    """One source problem at ``depth``, deterministic in ``(seed, depth, index)``.

    Spec 9.2: "choose an ordered chain of D indices ending at 24; sample its D-1 other indices
    without replacement from 1..23 and sort them.  A chain node depends on the preceding chain
    node, with the first depending on x0.  Every off-chain node depends on x0."  Sorting the
    sampled indices is what makes display order topological at no extra cost.
    """
    depth = int(depth)
    if depth not in spec.chain_depths:
        raise ContractViolation(f"depth {depth} is not one of the declared "
                                f"chain depths {spec.chain_depths}")
    n = spec.n_operations
    stream = _Stream(seed, "mod97/problem", depth, int(index), n, spec.modulus)
    chain = sorted(stream.sample(range(1, n), depth - 1)) + [n]
    parents = {node: INPUT_NODE for node in range(1, n + 1)}
    for position, node in enumerate(chain):
        parents[node] = INPUT_NODE if position == 0 else chain[position - 1]
    x0 = stream.below(spec.modulus)
    ops = tuple(
        Op(j=j, parent=parents[j], a=1 + stream.below(spec.modulus - 1), b=stream.below(spec.modulus))
        for j in range(1, n + 1)
    )
    problem_id = _problem_id(depth, index, x0, ops, spec.modulus)
    problem = Problem(problem_id=problem_id, depth=depth, x0=x0, ops=ops, modulus=spec.modulus,
                      chain=tuple(chain))
    realized = problem.depth_of(problem.answer_node)
    if realized != depth:                                   # pragma: no cover - construction
        raise ContractViolation(f"answer depth is {realized}, not the requested {depth}")
    return problem


def _problem_id(depth: int, index: int, x0: int, ops: Sequence[Op], modulus: int) -> str:
    payload = [int(depth), int(index), int(x0), int(modulus),
               [[op.j, op.parent, op.a, op.b] for op in ops]]
    return PROBLEM_ID_PREFIX + identity.sha256_hex(identity.jcs(payload))[:PROBLEM_ID_CHARS]


def generate_panel(seed: bytes, *, spec: Mod97Spec, per_depth: int,
                   start_index: int = 0) -> tuple[Problem, ...]:
    """``per_depth`` problems at each declared depth, in declared depth order.

    ``per_depth`` is a planning quantity (``experiments.E2b.source_problems`` divided by the
    number of depths) and is therefore passed in, never read here.
    """
    out: list[Problem] = []
    for depth in spec.chain_depths:
        for i in range(int(per_depth)):
            out.append(generate(seed, depth, int(start_index) + i, spec=spec))
    if len({p.problem_id for p in out}) != len(out):         # pragma: no cover - 13 hex chars
        raise ContractViolation("two generated problems collide on problem_id")
    return tuple(out)


# --------------------------------------------------------------------------- evaluation


def correct_trace(p: Problem) -> Trace:
    """Evaluate all ``n`` rows.  This is the *generator's* interpreter, not the evaluator."""
    values = {INPUT_NODE: p.x0}
    rows: list[TraceRow] = []
    for op in p.ops:
        parent_value = values[op.parent]
        value = (op.a * parent_value + op.b) % p.modulus
        values[op.j] = value
        rows.append(TraceRow(j=op.j, parent_value_stated=parent_value, output_stated=value))
    return Trace(problem=p, rows=tuple(rows), final=values[p.answer_node],
                 violated_step=None, error_position_rel=None)


def chain_position_bin(position: int, depth: int) -> str:
    """Which of ``CHAIN_POSITION_BINS`` a nonterminal chain position falls in.

    ``position`` is 1-based along the chain and the terminal node is excluded, so the
    nonterminal positions are ``1..depth-1`` and the relative coordinate is
    ``(position - 1) / (depth - 1)`` on ``[0, 1)``.
    """
    rel = relative_position(position, depth)
    index = min(int(rel * len(CHAIN_POSITION_BINS)), len(CHAIN_POSITION_BINS) - 1)
    return CHAIN_POSITION_BINS[index]


def relative_position(position: int, depth: int) -> float:
    """``(position - 1) / (depth - 1)`` -- where a nonterminal chain node sits, in ``[0, 1)``."""
    if not 1 <= int(position) <= int(depth) - 1:
        raise ContractViolation(f"chain position {position} is not nonterminal in a depth-{depth} chain")
    return (int(position) - 1) / float(int(depth) - 1)


def wrong_trace(p: Problem, seed: bytes) -> Trace:
    """The wrong trace: one violated local equation, descendants recomputed consistently.

    The violated node is picked by a **stratified blind key**: the key first chooses one of
    ``CHAIN_POSITION_BINS`` and then a node inside it, so the injected error is balanced over
    relative chain position across a panel of problems instead of clustering wherever the
    hash happens to land.  Bins that no nonterminal node occupies (a short chain) are skipped,
    which is the spec's "where possible".

    ``seed`` carries no condition (R5/D4): the two traces of a problem are held fixed for
    every endorsement / checker / budget cell of that problem (spec 9.2).
    """
    base = correct_trace(p)
    nonterminal = p.chain[:-1]
    if not nonterminal:                                      # pragma: no cover - refused by Mod97Spec
        raise ContractViolation("a depth-1 chain has no nonterminal node to corrupt")
    occupied: dict[str, list[tuple[int, int]]] = {}
    for position, node in enumerate(nonterminal, start=1):
        occupied.setdefault(chain_position_bin(position, p.depth), []).append((position, node))
    bins = [name for name in CHAIN_POSITION_BINS if name in occupied]
    # ORDER_WSTAR is the study's "which wrong thing was chosen" namespace (spec 8.2 picks w*
    # among qualifying wrong classes); for E2b the injected error IS that choice.
    stream = _Stream(ids.blind(seed, K.ORDER_WSTAR, p.problem_id), "mod97/wrong")
    chosen_bin = bins[stream.below(len(bins))]
    members = occupied[chosen_bin]
    position, node = members[stream.below(len(members))]
    delta = 1 + stream.below(p.modulus - 1)                  # nonzero modulo the prime modulus

    values = {INPUT_NODE: p.x0}
    rows: list[TraceRow] = []
    for op in p.ops:
        parent_value = values[op.parent]
        value = (op.a * parent_value + op.b) % p.modulus
        if op.j == node:
            value = (value + delta) % p.modulus              # the single violated local equation
        values[op.j] = value
        rows.append(TraceRow(j=op.j, parent_value_stated=parent_value, output_stated=value))
    trace = Trace(problem=p, rows=tuple(rows), final=values[p.answer_node], violated_step=node,
                  error_position_rel=relative_position(position, p.depth))

    violated = tuple(row.j for row in trace.rows if not local_step_valid(p, row))
    if violated != (node,):
        raise ContractViolation(f"the wrong trace violates local steps {violated}, expected "
                                f"exactly ({node},)")
    if any(not local_step_valid(p, row) for row in base.rows):
        raise ContractViolation("the correct trace violates a local step")
    if trace.final == base.final:
        raise ContractViolation("the wrong trace reaches the same final answer as the correct one")
    return trace


def local_step_valid(p: Problem, row: TraceRow) -> bool:
    """Does one displayed row satisfy its own local equation?

    Uses **only** the row's stated parent value, its stated output and the problem's fixed
    coefficients -- never the evaluated state, never the chain, never the final answer.  That
    restriction is what makes the spec-9.3 tool honest: a locally consistent downstream row of
    the wrong trace is genuinely ``valid``, and the tool cannot be used to binary-search the
    injected error by asking for a recomputation.
    """
    op = p.op_of(row.j)
    return (op.a * int(row.parent_value_stated) + op.b) % p.modulus == int(row.output_stated)


# --------------------------------------------------------------------------- rendering


def render_problem(p: Problem) -> str:
    """The program text, in exactly the grammar ``ard.graders.trace`` parses (spec 9.2).

    First statement ``x0 = <int>`` (otherwise every row references an undefined variable), then
    one ``x_j = (a*x_p + b) % m`` per displayed operation, in topological order.
    """
    lines = [f"{VAR_PREFIX}{INPUT_NODE} = {p.x0}"]
    for op in p.ops:
        lines.append(f"{VAR_PREFIX}{op.j} = ({op.a}*{VAR_PREFIX}{op.parent} + {op.b}) "
                     f"% {p.modulus}")
    return "\n".join(lines)


def render_trace(t: Trace) -> str:
    """The displayed trace body: the same statements with the parent value substituted.

    These bytes are the packet body and are **identical under every support condition** for a
    problem (spec 9.1/9.2); the endorsement cue lives in a separate support block.
    """
    p = t.problem
    lines = [f"{VAR_PREFIX}{INPUT_NODE} = {p.x0}"]
    for row in t.rows:
        op = p.op_of(row.j)
        lines.append(f"{VAR_PREFIX}{row.j} = ({op.a}*{row.parent_value_stated} + {op.b}) "
                     f"% {p.modulus} = {row.output_stated}")
    return "\n".join(lines)


def step_ids(p: Problem) -> tuple[int, ...]:
    """The step ids a check request may name: the displayed operations, never ``x0``."""
    return tuple(op.j for op in p.ops)


__all__ = [
    "CHAIN_POSITION_BINS", "INPUT_NODE", "Mod97Spec", "Op", "PROBLEM_ID_PREFIX", "Problem",
    "Trace", "TraceRow", "VAR_PREFIX", "chain_position_bin", "correct_trace", "generate",
    "generate_panel", "local_step_valid", "relative_position", "render_problem", "render_trace",
    "spec_from_config", "step_ids", "wrong_trace",
]
