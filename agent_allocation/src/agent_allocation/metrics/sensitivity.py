"""Exact-string versus mathematical-class readouts for MATH plurality checkpoints (MASTER_SPEC §11.1).

For every sealed MATH checkpoint whose readout is plurality over more than one candidate, the selection is re-run on
the same public inputs twice -- with the injected mutual complete-link equivalence and with exact keys only -- and
both delivered candidates are joined to grades. The sealed selection remains the primary endpoint; these rows are
the declared sensitivity analysis. ``reproduced`` is ``False`` when the re-run merged readout does not match the
sealed one (for example a timeout that resolved differently); such rows are reported, never silently replaced.

Config keys consumed: none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from agent_allocation.contracts import CheckpointView, GradeResult, OutputKind, SelectionStatus
from agent_allocation.metrics.outcomes import CandidateState, candidate_state, delivered_state
from agent_allocation.metrics.tables import MetricsInvariantError, PartitionRecords
from agent_allocation.selection.keys import vote_key
from agent_allocation.selection.plurality import build_classes, memoized_equivalence
from agent_allocation.selection.readout import select_checkpoint_detail
from agent_allocation.selection.seal import SealedSelections, SealError

__all__ = ["MathSensitivityRow", "math_class_sensitivity"]


@dataclass(frozen=True)
class MathSensitivityRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    policy_id: str
    name: str
    family: str
    view: str
    pool_size: int
    n_votes: int
    n_classes_exact: int
    n_classes_merged: int
    equivalence_calls: int
    equivalence_unknown: int
    sealed_selector_name: str
    sealed_selected_candidate_id: str | None
    reproduced: bool
    merged_selected_candidate_id: str | None
    exact_selected_candidate_id: str | None
    same_selection: bool
    merged_delivered_state: str
    exact_delivered_state: str


def math_class_sensitivity(sealed: SealedSelections, records: PartitionRecords, grades: Mapping[str, GradeResult],
                           *, seed: int, math_equivalent: Callable[[str, str], bool | None] | None
                           ) -> tuple[MathSensitivityRow, ...]:
    if not isinstance(sealed, SealedSelections):
        raise SealError("sensitivity rows join grades only to a verified SealedSelections")
    equivalent = None if math_equivalent is None else memoized_equivalence(math_equivalent)
    rows = []
    for item in records.items:
        task = item.task
        if task.output_kind is not OutputKind.MATH:
            continue
        for cp in item.checkpoints:
            selection = sealed.by_policy.get(cp.policy_id)
            if selection is None or len(cp.candidate_ids) < 2 or selection.status not in (
                    SelectionStatus.SELECTED.value, SelectionStatus.ABSTAINED_ALL_INVALID.value):
                continue
            detail = select_checkpoint_detail(cp, task, item.candidates, {}, seed=seed, math_equivalent=equivalent)
            votes = [(cid, key) for cid in cp.candidate_ids
                     if (key := vote_key(item.candidates[cid], task)) is not None]
            exact_classes, _, _ = build_classes(votes, kind=OutputKind.MATH)
            merged = detail.result.selected_candidate_id
            exact = detail.exact_only_selected_candidate_id if equivalent is not None else merged

            def state(cid: str | None) -> str:
                if cid is None:
                    return delivered_state(detail.result, None).value
                if cid not in cp.candidate_ids:
                    raise MetricsInvariantError(f"{cp.policy_id}: sensitivity readout left its pool")
                chosen: CandidateState = candidate_state(item.candidates.get(cid), grades.get(cid))
                return delivered_state(detail.result, chosen).value

            rows.append(MathSensitivityRow(
                partition=records.partition, benchmark=task.benchmark, subtask=task.subtask, item_id=task.item_id,
                policy_id=cp.policy_id, name=cp.name, family=cp.family, view=CheckpointView(cp.view).value,
                pool_size=len(cp.candidate_ids), n_votes=detail.n_votes, n_classes_exact=len(exact_classes),
                n_classes_merged=len(detail.classes), equivalence_calls=detail.equivalence_calls,
                equivalence_unknown=detail.equivalence_unknown, sealed_selector_name=selection.selector_name,
                sealed_selected_candidate_id=selection.selected_candidate_id,
                reproduced=merged == selection.selected_candidate_id, merged_selected_candidate_id=merged,
                exact_selected_candidate_id=exact, same_selection=merged == exact,
                merged_delivered_state=state(merged), exact_delivered_state=state(exact)))
    return tuple(rows)
