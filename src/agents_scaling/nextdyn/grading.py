"""The GRADE writer: tri-state at the writer, and the only reader of a label (spec 4.6, 17.3;
plan §2.5, D15).

**Writer side (R7).**

ARD36 wrote a boolean ``is_correct`` column next to an ``evaluator_status`` and then
disagreed with itself about what an unparseable answer meant -- one analysis counted it
wrong, another dropped it, a third counted it unknown.  Two mechanisms make that impossible
here:

* **The tri-state is decided once, at the writer.**  :func:`grade_decision` turns a decision
  plus a grader verdict into exactly one of ``C`` / ``W`` / ``U`` and, for a ``U``, a typed
  ``u_reason``.  A delivery or infrastructure failure, an unparseable output, an ambiguous
  grader, a grader timeout and a grader error are all ``U`` -- never ``W`` (spec 15.3:
  ``unresolved_semantics: unknown``).  No boolean column exists anywhere in the schema
  (``grades_private`` has none), so nothing downstream can invent one.
* **One reader.**  :func:`tri_state` is the sanctioned way to read a label and it *refuses* a
  row that carries any label-ish column (:data:`LABELISH_COLUMNS`), so a re-introduced
  ``is_correct`` fails loudly at the first read instead of silently winning.

``validity``, ``delivery_success`` and ``infrastructure_status`` stay on ``decisions``; a
grade never duplicates them (``test_validity_change_is_not_answer_switch``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import schema, tables
from agents_scaling.nextdyn.errors import LeakageError, MaterializeError
from agents_scaling.nextdyn.graders import (
    STATUS_AMBIGUOUS,
    STATUS_CORRECT,
    STATUS_ERROR,
    STATUS_INCORRECT,
    STATUS_TIMEOUT,
    GradeTarget,
    Verdict,
)
from agents_scaling.nextdyn.graders import families as GF
from agents_scaling.study import identity

# --------------------------------------------------------------------------- vocabulary

STATE_CORRECT: Final[str] = "C"
STATE_WRONG: Final[str] = "W"
STATE_UNKNOWN: Final[str] = "U"
STATES: Final[tuple[str, ...]] = (STATE_CORRECT, STATE_WRONG, STATE_UNKNOWN)

GRADING_RESOLVED: Final[str] = "resolved"
GRADING_AMBIGUOUS: Final[str] = "ambiguous"
GRADING_TIMEOUT: Final[str] = "timeout"
GRADING_ERROR: Final[str] = "error"
GRADING_NOT_GRADED: Final[str] = "not_graded"
GRADING_STATUSES: Final[tuple[str, ...]] = (
    GRADING_RESOLVED, GRADING_AMBIGUOUS, GRADING_TIMEOUT, GRADING_ERROR, GRADING_NOT_GRADED,
)

U_NO_ANSWER: Final[str] = "no_answer"
U_PARSE_FAILED: Final[str] = "parse_failed"
U_GRADER_AMBIGUOUS: Final[str] = "grader_ambiguous"
U_GRADER_TIMEOUT: Final[str] = "grader_timeout"
U_GRADER_ERROR: Final[str] = "grader_error"
U_INFRA_FAILED: Final[str] = "infra_failed"
U_CONTEXT_REFUSED: Final[str] = "context_refused"
U_NOT_DELIVERED: Final[str] = "not_delivered"
U_REASONS: Final[tuple[str, ...]] = (
    U_NO_ANSWER, U_PARSE_FAILED, U_GRADER_AMBIGUOUS, U_GRADER_TIMEOUT, U_GRADER_ERROR,
    U_INFRA_FAILED, U_CONTEXT_REFUSED, U_NOT_DELIVERED,
)

AUDIT_NONE: Final[str] = "none"
AUDIT_BLINDED_PASS: Final[str] = "blinded_pass"
AUDIT_BLINDED_FAIL: Final[str] = "blinded_fail"
AUDIT_DISCORDANT: Final[str] = "discordant_flagged"

INFRASTRUCTURE_OK: Final[str] = "ok"
#: The one admitted value of ``study.failure_policy.unresolved_semantics`` (spec 15.3): an
#: unresolved sample is ``U``, never ``W``.  :func:`assert_failure_policy` holds the running
#: configuration to it before a single grade is written.
UNRESOLVED_SEMANTICS: Final[str] = "unknown"
#: The grading rules' own version, independent of the module bytes (which ride in
#: ``graders.families.grader_hash``).
GRADER_VERSION: Final[str] = "nextdyn_grade_v1"

#: Columns that would re-introduce a boolean grade.  :func:`tri_state` refuses a row carrying
#: one, whatever its value, because the row would then have two answers to one question.
LABELISH_COLUMNS: Final[frozenset[str]] = frozenset({
    "is_correct", "correct", "incorrect", "label", "passed", "pass", "accuracy", "score",
    "gold", "answer_key",
})

_VERDICT_TO_STATE: Final[Mapping[str, tuple[str, str, str | None]]] = {
    STATUS_CORRECT: (STATE_CORRECT, GRADING_RESOLVED, None),
    STATUS_INCORRECT: (STATE_WRONG, GRADING_RESOLVED, None),
    STATUS_AMBIGUOUS: (STATE_UNKNOWN, GRADING_AMBIGUOUS, U_GRADER_AMBIGUOUS),
    STATUS_TIMEOUT: (STATE_UNKNOWN, GRADING_TIMEOUT, U_GRADER_TIMEOUT),
    STATUS_ERROR: (STATE_UNKNOWN, GRADING_ERROR, U_GRADER_ERROR),
}


# --------------------------------------------------------------------------- the writer


def assert_failure_policy(policy: Any) -> None:
    """Refuse to grade under a policy that would let an unresolved sample count as wrong."""
    if policy.unresolved_semantics != UNRESOLVED_SEMANTICS:
        raise MaterializeError(
            f"study.failure_policy.unresolved_semantics is {policy.unresolved_semantics!r}; the "
            f"tri-state writer only admits {UNRESOLVED_SEMANTICS!r} (spec 15.3)")


def ungradeable_reason(decision: tables.DecisionRow, out: schema.PublicOutput) -> str | None:
    """Why this sample cannot be graded at all, or ``None`` when it can (spec 15.3)."""
    if decision.infrastructure_status != INFRASTRUCTURE_OK:
        return U_INFRA_FAILED
    if not decision.delivery_success:
        return U_NOT_DELIVERED
    if out.parse_status is schema.ParseStatus.CONTEXT_FAILURE:
        return U_CONTEXT_REFUSED
    if out.parse_status not in schema.VALID_STATUSES:
        return U_PARSE_FAILED
    if out.final_answer is None or not str(out.final_answer).strip():
        return U_NO_ANSWER
    return None


def grade_decision(family: str, out: schema.PublicOutput, gold: Mapping[str, Any], *,
                   decision: tables.DecisionRow, target: GradeTarget) -> tables.GradePrivateRow:
    """One ``grades_private`` row: the tri-state, decided here and nowhere else.

    ``decision`` supplies the key the row is written under (``generation_uid`` and the
    decision ids that share that sample) and the three non-semantic failure columns;
    ``target`` supplies the variant facts an MC grader needs.  The contract's three-argument
    form cannot address a row whose primary key is ``(generation_uid, grader_hash)``, which is
    why those two are keyword arguments (see the WP3 note appended to contract §2.3).
    """
    grader = GF.grader_hash(family)
    gold_ref = identity.sha256_hex(identity.jcs({k: gold[k] for k in sorted(gold)}))
    blocked = ungradeable_reason(decision, out)
    if blocked is not None:
        return _row(decision, family, grader, gold_ref, state=STATE_UNKNOWN,
                    grading_status=GRADING_NOT_GRADED, u_reason=blocked, extracted=None,
                    canonical_class=None)
    try:
        verdict: Verdict = GF.grade(family, out.final_answer, gold, target)
    except Exception as exc:                      # a grader crash is U, never W (spec 4.6)
        return _row(decision, family, grader, gold_ref, state=STATE_UNKNOWN,
                    grading_status=GRADING_ERROR, u_reason=U_GRADER_ERROR,
                    extracted=None, canonical_class=None, detail=f"{type(exc).__name__}: {exc}")
    state, status, reason = _VERDICT_TO_STATE[verdict.status]
    return _row(decision, family, grader, gold_ref, state=state, grading_status=status,
                u_reason=reason, extracted=verdict.extracted,
                canonical_class=verdict.canonical_class)


def _row(decision: tables.DecisionRow, family: str, grader: str, gold_ref: str, *, state: str,
         grading_status: str, u_reason: str | None, extracted: str | None,
         canonical_class: str | None, detail: str = "") -> tables.GradePrivateRow:
    del detail  # the typed reason is the record; free text never decides a label
    if state not in STATES or grading_status not in GRADING_STATUSES:
        raise MaterializeError(f"illegal grade ({state!r}, {grading_status!r})")
    if (u_reason is None) != (state != STATE_UNKNOWN):
        raise MaterializeError(f"a {state} grade carries u_reason {u_reason!r}: every U has a "
                               "typed reason and no resolved grade has one")
    return tables.GradePrivateRow(
        generation_uid=decision.generation_uid, grader_hash=grader,
        decision_ids_json=json.dumps([decision.decision_id]),
        source_item_id=decision.source_item_id, family=family, semantic_correct=state,
        grading_status=grading_status, u_reason=u_reason, extracted=extracted,
        canonical_class=canonical_class, gold_ref_hash=gold_ref, audit_status=AUDIT_NONE,
        grader_version=GRADER_VERSION)


def merge_decision_ids(rows: Sequence[tables.GradePrivateRow]) -> list[tables.GradePrivateRow]:
    """Collapse rows of one ``(generation_uid, grader_hash)`` into one, listing every decision
    id that reuses that sample (D14: a root reference is not a second sample)."""
    import dataclasses

    merged: dict[tuple[str, str], tables.GradePrivateRow] = {}
    ids: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row.generation_uid, row.grader_hash)
        seen = ids.setdefault(key, [])
        for decision_id in json.loads(row.decision_ids_json):
            if decision_id not in seen:
                seen.append(decision_id)
        if key not in merged:
            merged[key] = row
        elif merged[key].semantic_correct != row.semantic_correct:
            raise MaterializeError(
                f"{row.generation_uid} graded {merged[key].semantic_correct} and "
                f"{row.semantic_correct} by the same grader: one sample, one label")
    return [dataclasses.replace(row, decision_ids_json=json.dumps(sorted(ids[key])))
            for key, row in merged.items()]


# --------------------------------------------------------------------------- the reader


def tri_state(row: tables.GradePrivateRow | Mapping[str, Any]) -> str:
    """``"C"`` / ``"W"`` / ``"U"`` -- the only sanctioned way to read a grade.

    Refuses a row carrying any :data:`LABELISH_COLUMNS` key: a boolean grade column is the
    ARD36 defect this schema removed, and a reader that tolerated one would let it back in.
    """
    if isinstance(row, Mapping):
        leaked = sorted(set(row) & LABELISH_COLUMNS)
        if leaked:
            raise LeakageError(f"grade row carries label-ish column(s) {leaked}; the only label "
                               "is semantic_correct (C/W/U) and grading.tri_state reads it")
        state = row.get("semantic_correct")
    else:
        for name in LABELISH_COLUMNS:
            if hasattr(row, name):
                raise LeakageError(f"{type(row).__name__} carries a {name!r} attribute; "
                                   "grades_private has no boolean column (D15)")
        state = getattr(row, "semantic_correct", None)
    if state not in STATES:
        raise MaterializeError(f"semantic_correct {state!r} is not one of {STATES}")
    return str(state)


def tri_state_index(rows: Sequence[tables.GradePrivateRow | Mapping[str, Any]]) -> dict[str, str]:
    """``{generation_uid: state}`` -- what a module that may not import gold receives.

    ``banks.difficulty`` and every analysis estimator take this mapping rather than the grade
    rows, so the generation-side import closure never needs :mod:`agents_scaling.nextdyn.grading`
    (R7) and a label still passes through exactly one reader.
    """
    out: dict[str, str] = {}
    for row in rows:
        uid = row["generation_uid"] if isinstance(row, Mapping) else row.generation_uid
        state = tri_state(row)
        if out.setdefault(uid, state) != state:
            raise MaterializeError(f"{uid} carries two different labels ({out[uid]} and {state})")
    return out


def canonical_class_index(rows: Sequence[tables.GradePrivateRow]) -> dict[str, str | None]:
    """``{generation_uid: canonical_class}`` (gold-independent, but written by the grader)."""
    return {row.generation_uid: row.canonical_class for row in rows}


def counts(rows: Sequence[tables.GradePrivateRow | Mapping[str, Any]]) -> dict[str, int]:
    index = tri_state_index(rows)
    return {state: sum(1 for value in index.values() if value == state) for state in STATES}


# --------------------------------------------------------------------------- the GRADE worker


def grade_worker(runner: Any, source_item_id: str) -> tables.ItemFile:
    """Grade every decision of the graded cells of one item (cpu lane, resumable).

    The runner supplies the cell, the run root and the decisions to grade; the grades go back
    in ``ItemFile.private_rows`` so the runner writes them under
    ``protected/cells/<cell_id>/items/`` (0600 in 0700) and never under ``cells/`` (R7).
    """
    started = time.monotonic()
    cell = runner.cell
    run_root = runner.run_root
    assert_failure_policy(runner.cfg.failure_policy)
    items = I.load_items(run_root)
    item = items[source_item_id]
    store = GF.GoldStore.load(run_root, [item.family])
    gold = store.reference(item.family, source_item_id)
    rows: list[tables.GradePrivateRow] = []
    for decision, out in runner.decisions_for(source_item_id):
        variant = item.variant(decision.item_variant_id)
        target = GradeTarget(family=item.family, source_item_id=source_item_id,
                             item_variant_id=variant.item_variant_id,
                             answer_format=item.answer_format, permutation=variant.permutation,
                             options=variant.options)
        rows.append(grade_decision(item.family, out, gold, decision=decision, target=target))
    del started
    return tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=cell.cell_id, cell_kind=cell.kind,
        source_item_id=source_item_id, status=tables.ItemStatus.COMPLETE,
        bundle_export_hash=None, rows={},
        private_rows={"grades_private": tuple(merge_decision_ids(rows))})


__all__ = [
    "AUDIT_BLINDED_FAIL", "AUDIT_BLINDED_PASS", "AUDIT_DISCORDANT", "AUDIT_NONE",
    "GRADER_VERSION", "GRADING_AMBIGUOUS", "GRADING_ERROR", "GRADING_NOT_GRADED",
    "GRADING_RESOLVED", "GRADING_STATUSES", "GRADING_TIMEOUT", "LABELISH_COLUMNS", "STATES",
    "STATE_CORRECT", "STATE_UNKNOWN", "STATE_WRONG", "U_CONTEXT_REFUSED", "U_GRADER_AMBIGUOUS",
    "U_GRADER_ERROR", "U_GRADER_TIMEOUT", "U_INFRA_FAILED", "U_NOT_DELIVERED", "U_NO_ANSWER",
    "U_PARSE_FAILED", "U_REASONS", "UNRESOLVED_SEMANTICS", "assert_failure_policy", "canonical_class_index", "counts", "grade_decision",
    "grade_worker", "merge_decision_ids", "tri_state", "tri_state_index", "ungradeable_reason",
]
