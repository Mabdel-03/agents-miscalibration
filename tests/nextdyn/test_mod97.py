"""WP5: the E2b mod-97 generator (spec 9.2; spec 18.4 names).

The generator is the *only* place in this study where the experimenter knows the right answer
by construction, so every property the assay depends on is asserted against a second, unrelated
implementation -- ``ard.graders.trace``, a hand-written lexer and recursive-descent parser that
this module's source is checked never to import.  A shared bug cannot hide in two
implementations that share no code and no fixtures.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from agents_scaling.ard.graders import trace as EVAL
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import mod97 as M
from agents_scaling.nextdyn.errors import ContractViolation

SEED = b"nextdyn-e2b-test-seed"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "nextdyn"


@pytest.fixture(scope="module")
def spec() -> M.Mod97Spec:
    """The committed grid: 24 displayed operations, depths 4/12/24, modulus 97."""
    return M.spec_from_config(C.load_study(CONFIG_DIR / "study.yaml").experiments.E2b)


@pytest.fixture(scope="module")
def panel(spec) -> tuple[M.Problem, ...]:
    return M.generate_panel(SEED, spec=spec, per_depth=6)


# --------------------------------------------------------------------------- spec 18.4


def test_mod97_evaluator_independent_of_generator(spec):
    """Spec 18.4.  ``mod97.py`` imports nothing from the evaluator and the evaluator imports
    nothing from nextdyn, and the two agree on a hand-computed fixture.

    Independence is what makes "the correct trace is correct" a *measurement* rather than a
    restatement of how it was built.
    """
    generator = ast.parse((Path(M.__file__)).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(generator):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    assert not any("graders" in name or "trace" in name for name in imported), sorted(imported)

    evaluator = ast.parse(Path(EVAL.__file__).read_text(encoding="utf-8"))
    back = {n.module for n in ast.walk(evaluator) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(name.startswith("agents_scaling.nextdyn") for name in back)

    # Hand-computed fixture: x0 = 5, x1 = (3*5 + 2) % 97 = 17, x2 = (4*17 + 1) % 97 = 69.
    hand = M.Problem(problem_id="m97hand", depth=2, x0=5, modulus=97, chain=(1, 2),
                     ops=(M.Op(j=1, parent=0, a=3, b=2), M.Op(j=2, parent=1, a=4, b=1)))
    trace = M.correct_trace(hand)
    assert [row.output_stated for row in trace.rows] == [17, 69]
    assert trace.final == 69
    assert EVAL.evaluate_program(M.render_problem(hand)).final_value == 69


def test_all_depths_show_24_operations(spec, panel):
    """Spec 18.4.  Depth changes the dependency structure, never how much the receiver reads:
    every problem at every depth displays exactly the declared number of operations."""
    assert set(spec.chain_depths) == {p.depth for p in panel}
    for problem in panel:
        assert problem.n_operations == spec.n_operations == 24
        assert len(M.render_problem(problem).splitlines()) == spec.n_operations + 1
        assert [op.j for op in problem.ops] == list(range(1, 25))


def test_answer_dependency_depth_exact(spec, panel):
    """Spec 18.4.  The answer's path from ``x0`` is exactly D operations -- walked from the
    parent map, not taken on trust from the chain the generator drew -- and every off-chain
    node hangs directly off ``x0`` (depth 1), so no node is a dangling definition."""
    for problem in panel:
        assert problem.depth_of(problem.answer_node) == problem.depth
        assert problem.chain[-1] == problem.answer_node == 24
        off_chain = [op.j for op in problem.ops if op.j not in problem.chain]
        assert len(off_chain) == spec.n_operations - problem.depth
        for node in off_chain:
            assert problem.op_of(node).parent == M.INPUT_NODE
            assert problem.depth_of(node) == 1


def test_wrong_trace_has_exactly_one_violated_local_step(panel):
    """Spec 18.4.  One violated local equation, at a *nonterminal* node **on** the answer
    chain, with every descendant recomputed consistently from the altered parent.

    A violation off the chain would never reach the answer; one at the terminal node would be
    the answer itself and could not be localized by the spec-9.3 tool.
    """
    for problem in panel:
        wrong = M.wrong_trace(problem, SEED)
        violated = [row.j for row in wrong.rows if not M.local_step_valid(problem, row)]
        assert violated == [wrong.violated_step]
        assert wrong.violated_step in problem.chain[:-1]
        assert wrong.violated_step != problem.answer_node
        correct = M.correct_trace(problem)
        assert all(M.local_step_valid(problem, row) for row in correct.rows)


def test_wrong_and_correct_trace_final_answers_differ(panel):
    """Spec 18.4.  A nonzero delta survives the rest of the chain because every multiplier is
    a unit modulo the prime 97 -- asserted per problem, never argued."""
    for problem in panel:
        assert M.correct_trace(problem).final != M.wrong_trace(problem, SEED).final


# --------------------------------------------------------------------------- the evaluator


def test_evaluator_reproduces_every_generated_correct_trace(panel):
    """The independent evaluator agrees with the generator on every row of every problem."""
    for problem in panel:
        program = M.render_problem(problem)
        evaluation = EVAL.evaluate_program(program)
        generated = M.correct_trace(problem)
        assert evaluation.final_target == f"x{problem.answer_node}"
        assert evaluation.final_value == generated.final
        by_target = {step.target: step.value for step in evaluation.steps}
        assert by_target == {"x0": problem.x0,
                             **{f"x{row.j}": row.output_stated for row in generated.rows}}


def test_rendered_trace_aligns_with_the_evaluators_statement_index(panel):
    """``check_trace`` aligns claimed rows by statement index, so the displayed trace must
    open with its own ``x0`` claim; otherwise every row would be compared against its
    predecessor and a clean trace would read as 25 divergences."""
    for problem in panel:
        program = M.render_problem(problem)
        checks = EVAL.check_trace(program, M.render_trace(M.correct_trace(problem)))
        assert all(check.agrees for check in checks)
        wrong = M.wrong_trace(problem, SEED)
        divergence = EVAL.first_divergence(EVAL.check_trace(program, M.render_trace(wrong)))
        assert divergence is not None
        assert divergence.target == f"x{wrong.violated_step}"


def test_rendered_problem_uses_only_the_evaluators_grammar(panel):
    """Implicit multiplication, ``mod 97``, subscripts and an undefined ``x0`` all fail to
    parse in ``ard.graders.trace``; the frozen byte format avoids each of them."""
    for problem in panel[:3]:
        program = M.render_problem(problem)
        assert program.splitlines()[0] == f"x0 = {problem.x0}"
        assert EVAL.parse_program(program)
        assert " mod " not in program and "_" not in program
        for line in program.splitlines()[1:]:
            assert line.endswith(f"% {problem.modulus}")


# --------------------------------------------------------------------------- determinism


def test_generation_is_deterministic_in_seed_depth_and_index(spec):
    """Two runs of the generator produce byte-identical problems, and two indices do not."""
    first = M.generate(SEED, 12, 3, spec=spec)
    assert first == M.generate(SEED, 12, 3, spec=spec)
    assert first.problem_id != M.generate(SEED, 12, 4, spec=spec).problem_id
    assert first.problem_id != M.generate(b"other", 12, 3, spec=spec).problem_id


def test_the_same_two_traces_are_fixed_for_every_condition_of_a_problem(spec):
    """Spec 9.2: "Hold the same two traces fixed for every endorsement/checker/budget
    condition of a source problem" -- so neither trace takes anything but the problem and the
    study seed, and re-deriving them yields the same bytes."""
    problem = M.generate(SEED, 4, 0, spec=spec)
    assert M.render_trace(M.wrong_trace(problem, SEED)) == M.render_trace(M.wrong_trace(problem, SEED))
    assert M.render_trace(M.correct_trace(problem)) == M.render_trace(M.correct_trace(problem))


def test_error_position_is_balanced_over_relative_chain_position_bins(spec):
    """Spec 9.2: "balance error position within relative chain-position bins where possible".

    Over a panel of depth-24 problems each of the three bins is used, and none dominates -- the
    stratified key picks a bin first and a node inside it second, so a hash that happened to
    favour low indices cannot push every injected error to the front of the chain.
    """
    problems = [M.generate(SEED, 24, i, spec=spec) for i in range(30)]
    bins = Counter()
    for problem in problems:
        wrong = M.wrong_trace(problem, SEED)
        position = problem.chain.index(wrong.violated_step) + 1
        bins[M.chain_position_bin(position, problem.depth)] += 1
        assert 0.0 <= wrong.error_position_rel < 1.0
    assert set(bins) == set(M.CHAIN_POSITION_BINS)
    assert max(bins.values()) <= 2 * min(bins.values())


def test_local_step_valid_reads_only_the_stated_parent_and_output(spec):
    """Spec 9.3.  The checker's whole implementation: a row whose stated parent value is
    consistent with its stated output is ``valid`` even when the parent value is itself wrong,
    which is exactly why a downstream row of the wrong trace answers ``valid``."""
    problem = M.generate(SEED, 4, 1, spec=spec)
    op = problem.op_of(problem.chain[-1])
    fabricated = M.TraceRow(j=op.j, parent_value_stated=7,
                            output_stated=(op.a * 7 + op.b) % problem.modulus)
    assert M.local_step_valid(problem, fabricated) is True
    assert M.local_step_valid(problem, M.TraceRow(j=op.j, parent_value_stated=7,
                                                  output_stated=(op.a * 7 + op.b + 1) % 97)) is False

    wrong = M.wrong_trace(problem, SEED)
    downstream = [row for row in wrong.rows
                  if row.j in problem.chain and row.j > wrong.violated_step]
    assert downstream and all(M.local_step_valid(problem, row) for row in downstream)


def test_an_undeclared_depth_is_refused(spec):
    with pytest.raises(ContractViolation):
        M.generate(SEED, 7, 0, spec=spec)


def test_the_generator_shape_comes_from_study_yaml(spec):
    """No experimental number lives in ``mod97.py``: the committed config supplies all three."""
    assert (spec.n_operations, spec.chain_depths, spec.modulus) == (24, (4, 12, 24), 97)
