"""Per-candidate and per-(checkpoint, item) outcome tables with tri-state correctness.

Rules (MASTER_SPEC §11-§12, contract R9):

* The population of a checkpoint is every planned (policy manifest) item. Nothing is dropped because it is
  unknown, unrun or failed; each row carries a ``delivered_state`` saying why ``y`` is or is not known.
* ``y`` is 1.0 / 0.0 / NaN. Operational failures (all-invalid abstention, invalid output, no output) are
  known zeros; evaluator errors, paused public checks, pending grades, missing selector rows, unsealed
  partitions and unrun jobs are NaN (unknown), never 0.
* Grades are joined onto a selection only when its batch seal verifies (``RunData.is_sealed``, contract v2 batches).
* Coverage is oracle information over the checkpoint's declared pool; POOLED is synthesis, so its
  delivered accuracy is compared with the coverage of its *input* roots only descriptively.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from agent_allocation.analysis.load import RunData
from agent_allocation.contracts import (
    CheckpointView,
    GradeResult,
    GradeStatus,
    SelectionStatus,
)

NAN = math.nan
SYNTHESIS_FAMILIES = frozenset({"pooled"})
SINGLE_READOUT_FAMILIES = frozenset({"private_single", "deep", "pooled"})

KNOWN_STATES = {
    "correct": 1.0,
    "wrong": 0.0,
    "op_invalid_output": 0.0,
    "op_abstained_all_invalid": 0.0,
    "op_no_output": 0.0,
}
UNKNOWN_STATES = (
    "unknown_evaluator_error",
    "unknown_paused_check_error",
    "unknown_selector_inconsistent",
    "pending_grade",
    "missing_selection",
    "unsealed",
    "in_progress",
    "not_started",
)
ADMINISTRATIVE_STATES = frozenset({"in_progress", "not_started"})


def stratum_of(benchmark: str | None, subtask: str | None) -> str:
    if benchmark == "bbh_logic4":
        return f"bbh_logic4/{subtask}"
    return str(benchmark)


def grade_value(grade: GradeResult | None) -> tuple[float, str]:
    """Tri-state correctness of one candidate: (1/0/NaN, status label)."""
    if grade is None:
        return NAN, "pending_grade"
    if grade.status == GradeStatus.GRADED:
        if grade.semantic_correct is None:
            return NAN, "graded_unknown"
        return (1.0, "correct") if grade.semantic_correct else (0.0, "wrong")
    if grade.status == GradeStatus.INVALID_OUTPUT:
        return 0.0, "op_invalid_output"
    if grade.status == GradeStatus.EVALUATOR_ERROR:
        return NAN, "unknown_evaluator_error"
    return NAN, "pending_grade"


def _tri_any(values: list[float]) -> float:
    """Oracle 'any correct' with unknowns: 1 if any 1; 0 if all known 0; else NaN."""
    if any(v == 1.0 for v in values):
        return 1.0
    if values and all(v == 0.0 for v in values):
        return 0.0
    return NAN


def candidate_table(run: RunData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in run.partitions:
        for node_id, spec in sorted(run.jobs.get(partition, {}).items()):
            parsed = run.parsed.get(node_id)
            result, candidate = parsed if parsed else (None, None)
            grade = run.grades.get(node_id)
            value, gstate = grade_value(grade) if parsed else (NAN, "not_generated")
            usage = result.usage if result else None
            check = run.public_checks.get(node_id)
            rows.append({
                "partition_id": partition,
                "node_id": node_id,
                "item_id": spec.item_id,
                "benchmark": spec.benchmark,
                "subtask": run.subtask_of(spec.item_id),
                "stratum": stratum_of(spec.benchmark, run.subtask_of(spec.item_id)),
                "kind": spec.kind.value,
                "model": spec.model,
                "root_slot": spec.root_slot,
                "team_size": spec.team_size,
                "update_round": spec.update_round,
                "reasoning_budget": spec.reasoning_budget,
                "completion_headroom": spec.completion_headroom,
                "max_completion_tokens": spec.max_completion_tokens,
                "n_parents": len(spec.parent_ids),
                "terminal": parsed is not None,
                "attempt_id": result.attempt_id if result else None,
                "transport_status": result.transport_status.value if result else None,
                "completion_status": (result.completion_status.value
                                      if result and result.completion_status else None),
                "finish_reason": result.finish_reason if result else None,
                "cache_status": result.cache_status if result else None,
                "latency_seconds": result.latency_seconds if result else None,
                "observed_provider": result.observed_provider if result else None,
                "parse_status": candidate.parse_status.value if candidate else None,
                "final_answer_present": bool(candidate and candidate.final_answer is not None),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "reasoning_tokens": usage.reasoning_tokens if usage else None,
                "cached_prompt_tokens": usage.cached_prompt_tokens if usage else None,
                "actual_cost_usd": (str(usage.actual_cost_usd)
                                    if usage and usage.actual_cost_usd is not None else None),
                "grade_status": grade.status.value if grade else None,
                "grade_state": gstate,
                "correct": value,
                "operational_success": (grade.operational_success if grade else None),
                "public_check_status": check.status.value if check else None,
                "public_pass_fraction": check.pass_fraction if check else None,
                "public_check_cpu_seconds": check.cpu_seconds if check else None,
                "grade_cpu_seconds": grade.cpu_seconds if grade else None,
            })
    frame = pd.DataFrame(rows)
    for col in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_prompt_tokens"):
        if col in frame:
            frame[col] = frame[col].astype("Int64")
    return frame


def outcome_table(run: RunData) -> pd.DataFrame:
    """One row per planned (partition, checkpoint policy_id): delivered outcome, coverage, pool counts."""
    panels_of = run.item_panels()
    wave_of = run.wave_of()
    rows: list[dict[str, Any]] = []
    for partition in run.partitions:
        selections = run.selections.get(partition, {})
        for cp in run.policies.get(partition, []):
            sealed = run.is_sealed(partition, cp.policy_id)
            benchmark = run.benchmark_of(cp.item_id)
            subtask = run.subtask_of(cp.item_id)
            required = cp.required_node_ids
            n_terminal = sum(1 for nid in required if nid in run.parsed)
            pool_values = [grade_value(run.grades.get(nid))[0] if nid in run.parsed else NAN
                           for nid in cp.candidate_ids]
            pool_valid = sum(1 for nid in cp.candidate_ids
                             if nid in run.parsed and run.parsed[nid][1].parse_status.value == "valid")
            coverage = _tri_any(pool_values)
            input_coverage = NAN
            if cp.family in SYNTHESIS_FAMILIES:
                inputs = [nid for nid in required if nid not in cp.candidate_ids]
                input_coverage = _tri_any([grade_value(run.grades.get(nid))[0] if nid in run.parsed else NAN
                                           for nid in inputs])
            sel = selections.get(cp.policy_id)
            state, y, selected = _delivered(run, cp, sel, sealed, n_terminal, len(required))
            rows.append({
                "partition_id": partition,
                "policy_id": cp.policy_id,
                "item_id": cp.item_id,
                "benchmark": benchmark,
                "subtask": subtask,
                "stratum": stratum_of(benchmark, subtask),
                "wave": wave_of.get(cp.item_id),
                "panels": ",".join(panels_of.get(cp.item_id, ())),
                "name": cp.name,
                "view": cp.view.value,
                "family": cp.family,
                "team_size": cp.team_size,
                "total_cycles": cp.total_cycles,
                "peer_rounds": cp.peer_rounds,
                "reasoning_budget": cp.reasoning_budget,
                "selector_name": cp.selector_name,
                "selector_version": cp.selector_version,
                "n_pool": len(cp.candidate_ids),
                "n_required": len(required),
                "n_required_terminal": n_terminal,
                "n_pool_valid": pool_valid,
                "n_pool_correct": int(sum(1 for v in pool_values if v == 1.0)),
                "n_pool_unknown": int(sum(1 for v in pool_values if v != v)),
                "coverage": coverage,
                "input_coverage": input_coverage,
                "selection_status": SelectionStatus(sel.status).value if sel else None,
                "selection_batch": run.selection_batch.get(partition, {}).get(cp.policy_id),
                "selector_concrete": sel.selector_name if sel else None,
                "selected_candidate_id": selected,
                "selected_in_pool": (selected in cp.candidate_ids) if selected else None,
                "sealed": sealed,
                "delivered_state": state,
                "y": y,
            })
    return pd.DataFrame(rows)


def _delivered(run: RunData, cp, sel, sealed: bool, n_terminal: int, n_required: int
               ) -> tuple[str, float, str | None]:
    if sel is None:
        if n_terminal == 0:
            return "not_started", NAN, None
        if n_terminal < n_required:
            return "in_progress", NAN, None
        return "missing_selection", NAN, None
    if not sealed:
        return "unsealed", NAN, sel.selected_candidate_id
    status = SelectionStatus(sel.status).value
    selected = sel.selected_candidate_id
    if status == SelectionStatus.ABSTAINED_ALL_INVALID.value:
        return "op_abstained_all_invalid", 0.0, selected
    if status == SelectionStatus.PAUSED_CHECK_ERROR.value:
        return "unknown_paused_check_error", NAN, selected
    if status not in (SelectionStatus.SELECTED.value, SelectionStatus.SINGLE_CANDIDATE.value):
        return "unknown_selector_inconsistent", NAN, selected
    if selected is None:
        if status == SelectionStatus.SINGLE_CANDIDATE.value and n_terminal == n_required:
            return "op_no_output", 0.0, None
        return "unknown_selector_inconsistent", NAN, None
    if selected not in run.parsed:
        return "unknown_selector_inconsistent", NAN, selected
    value, gstate = grade_value(run.grades.get(selected))
    if gstate == "graded_unknown":
        return "unknown_evaluator_error", NAN, selected
    return gstate, value, selected


def current_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["view"] == CheckpointView.CURRENT.value]
