"""Deployable, gold-blind checkpoint readouts (MASTER_SPEC §5.8, §11, §12; docs/00_contract.md §2 "06").

Readout by checkpoint:

* A single-candidate pool (any family, any view) -> that candidate, ``SINGLE_CANDIDATE`` (valid or not: an invalid
  last answer is delivered as the invalid output it is, and metrics keep it distinct from an abstention).
* CURRENT view, families ``private_single``/``deep``/``pooled`` -> last answer; a pool of more than one candidate is
  a manifest error and raises. POOLED is synthesis: its delivered answer is its own generated candidate, never a
  pick from the four input roots.
* CURRENT view, families ``sample``/``private_team``/``peer``/``cp`` -> text plurality (``selection.plurality``) for
  MATH/OPTION/BOOLEAN, public-example selection (``selection.code``) for PYTHON.
* ARCHIVE view with more than one candidate -> the same rule over every candidate occurrence in the archive,
  labelled ``archive_*`` (repeated serial answers are not independent evidence; the label says so).

The API admits only public inputs: ``PolicyCheckpoint``, ``PublicTask``, ``Candidate``, ``PublicCheckResult`` and an
optional gold-blind equivalence callable. ``PrivateEvalSpec`` and ``GradeResult`` are refused with ``TypeError``
wherever they appear, including inside a ``functools.partial``, bound method or closure passed as the callable.

``PolicyCheckpoint.selector_name`` is the declared family (``plurality`` | ``public_tests_only``); the concrete selector
run is recorded in ``SelectionResult.selector_name`` and must be legal for the declared family
(``concrete_selector_is_legal``), otherwise the checkpoint is refused.

``pool_hash`` = canonical hash of the ordered ``[candidate_id, final_answer]`` pairs. ``public_evidence_hash`` (code
selections only) = canonical hash of the ordered public-check evidence. ``cpu_seconds`` is process CPU time spent in
this call, including the equivalence callable.

Config keys consumed: none (the caller passes ``seed``, the study's top-level ``seed``).
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent_allocation.contracts import (
    Candidate, CheckpointView, GradeResult, OutputKind, ParseStatus, PolicyCheckpoint, PrivateEvalSpec,
    PublicCheckResult, PublicTask, SelectionResult, SelectionStatus, canonical_hash,
)
from agent_allocation.selection.code import public_example_selection
from agent_allocation.selection.keys import option_letters, vote_key
from agent_allocation.selection.plurality import AnswerClass, plurality

__all__ = [
    "DECLARED_SELECTORS", "DECLARED_SELECTOR_VERSION", "FAMILIES", "LAST_ANSWER_FAMILIES", "SELECTING_FAMILIES",
    "SELECTOR_VERSION", "SelectionDetail", "concrete_selector_is_legal", "pool_hash", "select_checkpoint",
    "select_checkpoint_detail", "selector_for",
]

SELECTOR_VERSION = "06-selection-v3"
FAMILIES: tuple[str, ...] = ("sample", "private_single", "private_team", "peer", "deep", "cp", "pooled")
LAST_ANSWER_FAMILIES = frozenset({"private_single", "deep", "pooled"})
SELECTING_FAMILIES = frozenset({"sample", "private_team", "peer", "cp"})

LAST_ANSWER = "last_answer"
PLURALITY_OPTION = "plurality_option"
PLURALITY_BOOLEAN = "plurality_boolean"
PLURALITY_MATH_EXACT = "plurality_math_exact"
PLURALITY_MATH_MUTUAL = "plurality_math_mutual_complete_link"
PUBLIC_EXAMPLE_SELECTION = "public_example_selection"
ARCHIVE_PREFIX = "archive_"

#: Declared selector families on ``PolicyCheckpoint.selector_name`` (config ``benchmarks[].selector``, contract v2
#: R06-1) and the concrete selectors each may run. ``""`` (undeclared) is accepted for hand-built checkpoints.
DECLARED_SELECTORS: dict[str, frozenset[str]] = {
    "plurality": frozenset({LAST_ANSWER, PLURALITY_OPTION, PLURALITY_BOOLEAN, PLURALITY_MATH_EXACT,
                            PLURALITY_MATH_MUTUAL}),
    "public_tests_only": frozenset({LAST_ANSWER, PUBLIC_EXAMPLE_SELECTION}),
}
DECLARED_SELECTOR_VERSION = "v1"

_FORBIDDEN = (PrivateEvalSpec, GradeResult)


@dataclass(frozen=True)
class SelectionDetail:
    """The sealed ``result`` plus gold-blind diagnostics (valid counts, classes, ties, math sensitivity)."""
    result: SelectionResult
    pool_size: int
    n_votes: int                               # text: candidates with a vote key; code: eligible programs
    classes: tuple[AnswerClass, ...] = ()
    tied: tuple[str, ...] = ()                 # tied class ids (text) or tied candidate ids (code), winner first
    winning_support: int | None = None
    equivalence_calls: int = 0
    equivalence_unknown: int = 0
    exact_only_selected_candidate_id: str | None = None   # MATH with an equivalence callable: sensitivity readout
    best_public_fraction: str | None = None               # code: exact fraction "p/q"


def selector_for(family: str, view: CheckpointView, output_kind: OutputKind, pool_size: int, *,
                 math_equivalence: bool = True) -> str:
    """The selector name ``select_checkpoint`` runs for this checkpoint shape (agent 03 may declare it)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown checkpoint family {family!r}")
    view = CheckpointView(view)
    if pool_size < 1:
        raise ValueError("a checkpoint pool must hold at least one candidate")
    prefix = ARCHIVE_PREFIX if view is CheckpointView.ARCHIVE else ""
    if pool_size == 1:
        return prefix + LAST_ANSWER
    if view is CheckpointView.CURRENT and family in LAST_ANSWER_FAMILIES:
        raise ValueError(f"{family} is a last-answer readout; its current pool must hold exactly one candidate")
    kind = OutputKind(output_kind)
    if kind is OutputKind.PYTHON:
        return prefix + PUBLIC_EXAMPLE_SELECTION
    if kind is OutputKind.MATH:
        return prefix + (PLURALITY_MATH_MUTUAL if math_equivalence else PLURALITY_MATH_EXACT)
    return prefix + (PLURALITY_OPTION if kind is OutputKind.OPTION else PLURALITY_BOOLEAN)


def concrete_selector_is_legal(declared: str, concrete: str) -> bool:
    """Whether ``concrete`` (``SelectionResult.selector_name``) may run for a checkpoint declaring ``declared``."""
    base = concrete[len(ARCHIVE_PREFIX):] if concrete.startswith(ARCHIVE_PREFIX) else concrete
    return base in DECLARED_SELECTORS.get(declared, frozenset())


def pool_hash(candidate_ids: tuple[str, ...], candidates: Mapping[str, Candidate]) -> str:
    return canonical_hash([[cid, candidates[cid].final_answer] for cid in candidate_ids])


def _refuse_private(value: Any, where: str) -> None:
    if isinstance(value, _FORBIDDEN):
        raise TypeError(f"selectors never receive {type(value).__name__} ({where})")


def _guard_callable(fn: Any) -> None:
    if not callable(fn):
        raise TypeError("math_equivalent must be callable or None")
    _refuse_private(fn, "math_equivalent")
    seen: list[Any] = [fn]
    while seen:
        obj = seen.pop()
        if isinstance(obj, functools.partial):
            for inner in (obj.func, *obj.args, *obj.keywords.values()):
                _refuse_private(inner, "math_equivalent partial")
                seen.append(inner)
        bound_self = getattr(obj, "__self__", None)
        if bound_self is not None:
            _refuse_private(bound_self, "math_equivalent bound method")
        for cell in getattr(obj, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            _refuse_private(content, "math_equivalent closure")
        for default in getattr(obj, "__defaults__", None) or ():
            _refuse_private(default, "math_equivalent default argument")


def _guard_inputs(checkpoint: Any, task: Any, candidates: Any, public_checks: Any, seed: Any,
                  math_equivalent: Any) -> None:
    _refuse_private(checkpoint, "checkpoint")
    _refuse_private(task, "task")
    if not isinstance(checkpoint, PolicyCheckpoint):
        raise TypeError(f"checkpoint must be PolicyCheckpoint, got {type(checkpoint).__name__}")
    if not isinstance(task, PublicTask):
        raise TypeError(f"task must be PublicTask, got {type(task).__name__}")
    for name, mapping, expected in (("candidates", candidates, Candidate),
                                    ("public_checks", public_checks, PublicCheckResult)):
        _refuse_private(mapping, name)
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a Mapping[str, {expected.__name__}]")
        for key, value in mapping.items():
            _refuse_private(value, f"{name}[{key!r}]")
            if not isinstance(key, str) or not isinstance(value, expected):
                raise TypeError(f"{name}[{key!r}] must map str -> {expected.__name__}, got {type(value).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if math_equivalent is not None:
        _guard_callable(math_equivalent)


def _check_pool(checkpoint: PolicyCheckpoint, task: PublicTask, candidates: Mapping[str, Candidate]) -> None:
    if checkpoint.family not in FAMILIES:
        raise ValueError(f"{checkpoint.policy_id}: unknown family {checkpoint.family!r}")
    if checkpoint.item_id != task.item_id:
        raise ValueError(f"{checkpoint.policy_id}: checkpoint item {checkpoint.item_id} != task {task.item_id}")
    pool = checkpoint.candidate_ids
    if not pool:
        raise ValueError(f"{checkpoint.policy_id}: empty candidate pool")
    if len(set(pool)) != len(pool):
        raise ValueError(f"{checkpoint.policy_id}: duplicate candidate ids would double-count an opportunity")
    for cid in pool:
        candidate = candidates.get(cid)
        if candidate is None:
            raise ValueError(f"{checkpoint.policy_id}: candidate {cid} is not available; select after it is terminal")
        if candidate.candidate_id != cid or candidate.node_id != cid:
            raise ValueError(f"{checkpoint.policy_id}: candidate keyed {cid} carries id {candidate.candidate_id}")
        if candidate.item_id != task.item_id or candidate.output_kind is not task.output_kind:
            raise ValueError(f"{checkpoint.policy_id}: candidate {cid} belongs to another item or output kind")


def select_checkpoint_detail(checkpoint: PolicyCheckpoint, task: PublicTask, candidates: Mapping[str, Candidate],
                             public_checks: Mapping[str, PublicCheckResult], *, seed: int,
                             math_equivalent: Callable[[str, str], bool | None] | None = None) -> SelectionDetail:
    started = time.process_time()
    _guard_inputs(checkpoint, task, candidates, public_checks, seed, math_equivalent)
    _check_pool(checkpoint, task, candidates)
    pool = tuple(checkpoint.candidate_ids)
    view = CheckpointView(checkpoint.view)
    name = selector_for(checkpoint.family, view, task.output_kind, len(pool),
                        math_equivalence=math_equivalent is not None)
    if checkpoint.selector_name and not concrete_selector_is_legal(checkpoint.selector_name, name):
        raise ValueError(f"{checkpoint.policy_id}: declared selector {checkpoint.selector_name!r} does not admit "
                         f"{name!r} for a {task.output_kind.value} task")
    digest = pool_hash(pool, candidates)

    def finish(status: SelectionStatus, selected: str | None, evidence: str | None, **extra: Any) -> SelectionDetail:
        result = SelectionResult(policy_id=checkpoint.policy_id, selected_candidate_id=selected, selector_name=name,
                                 selector_version=SELECTOR_VERSION, pool_hash=digest, public_evidence_hash=evidence,
                                 cpu_seconds=max(0.0, time.process_time() - started), status=status)
        return SelectionDetail(result=result, pool_size=len(pool), **extra)

    if len(pool) == 1:
        valid = _counts_as_vote(candidates[pool[0]], task)
        return finish(SelectionStatus.SINGLE_CANDIDATE, pool[0], None, n_votes=int(valid))

    if task.output_kind is OutputKind.PYTHON:
        outcome = public_example_selection(pool, task, candidates, public_checks, seed=seed)
        best = None if outcome.best_fraction is None else f"{outcome.best_fraction.numerator}/" \
                                                          f"{outcome.best_fraction.denominator}"
        return finish(outcome.status, outcome.selected_candidate_id, outcome.evidence_hash,
                      n_votes=outcome.n_eligible, tied=outcome.tied_candidate_ids, best_public_fraction=best)

    letters = option_letters(task) if task.output_kind is OutputKind.OPTION else None
    votes = []
    for cid in pool:
        key = vote_key(candidates[cid], task, letters=letters)
        if key is not None:
            votes.append((cid, key))
    outcome = plurality(votes, kind=task.output_kind, item_id=task.item_id, seed=seed,
                        math_equivalent=math_equivalent)
    exact_only = None
    if task.output_kind is OutputKind.MATH and math_equivalent is not None:
        exact_only = plurality(votes, kind=task.output_kind, item_id=task.item_id, seed=seed).selected_candidate_id
    status = SelectionStatus.SELECTED if outcome.selected_candidate_id is not None \
        else SelectionStatus.ABSTAINED_ALL_INVALID
    return finish(status, outcome.selected_candidate_id, None, n_votes=outcome.n_votes, classes=outcome.classes,
                  tied=outcome.tied_class_ids,
                  winning_support=None if outcome.winner is None else outcome.winner.support,
                  equivalence_calls=outcome.equivalence_calls, equivalence_unknown=outcome.equivalence_unknown,
                  exact_only_selected_candidate_id=exact_only)


def _counts_as_vote(candidate: Candidate, task: PublicTask) -> bool:
    """Whether a lone candidate would count as a valid vote/program (diagnostic only; it is delivered regardless)."""
    if task.output_kind is OutputKind.PYTHON:
        return candidate.parse_status is ParseStatus.VALID
    return vote_key(candidate, task) is not None


def select_checkpoint(checkpoint: PolicyCheckpoint, task: PublicTask, candidates: Mapping[str, Candidate],
                      public_checks: Mapping[str, PublicCheckResult], *, seed: int,
                      math_equivalent: Callable[[str, str], bool | None] | None = None) -> SelectionResult:
    return select_checkpoint_detail(checkpoint, task, candidates, public_checks, seed=seed,
                                    math_equivalent=math_equivalent).result
