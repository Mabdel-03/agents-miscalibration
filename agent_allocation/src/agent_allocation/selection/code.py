"""Public-example selection for code (MASTER_SPEC §11.2). This is selection by a weak verifier, never voting.

Eligible: parse ``VALID`` and a public check with ``CheckStatus.OK`` (``INVALID_CANDIDATE`` means no parseable
program, so it is never delivered). Any ``CheckStatus.INFRASTRUCTURE_ERROR`` in the pool -- or an ``OK`` check with
zero public cases, whose ``pass_fraction`` is ``None`` (contract v2, D4) -- pauses the selector
(``PAUSED_CHECK_ERROR``): unchecked code is never favoured and nothing is guessed. A ``VALID`` candidate without a
public check is a harness ordering error and raises. Among eligible candidates the maximum exact pass fraction wins;
ties go to the smallest :func:`candidate_priority_key`, a frozen hash of ``(seed, item_id, candidate_id)`` that
reads no correctness and no completion order. No eligible candidate -> ``ABSTAINED_ALL_INVALID``.
Private base/plus outcomes are never an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from agent_allocation.contracts import (
    Candidate, CheckStatus, ParseStatus, PublicCheckResult, PublicTask, SelectionStatus, canonical_hash,
)

__all__ = ["PublicSelectionOutcome", "candidate_priority_key", "public_evidence_hash", "public_example_selection"]


@dataclass(frozen=True)
class PublicSelectionOutcome:
    status: SelectionStatus
    selected_candidate_id: str | None
    best_fraction: Fraction | None
    tied_candidate_ids: tuple[str, ...]      # priority order, winner first
    n_eligible: int
    evidence_hash: str


def candidate_priority_key(seed: int, item_id: str, candidate_id: str) -> str:
    return canonical_hash({"purpose": "agent06/public_selection_priority", "seed": seed, "item_id": item_id,
                           "candidate_id": candidate_id})


def _fraction(check: PublicCheckResult) -> Fraction:
    if not check.cases_total:
        raise ValueError(f"public check {check.candidate_id} has no public case; the selector must pause")
    return Fraction(check.cases_passed, check.cases_total)


def public_evidence_hash(pool_ids: Sequence[str], public_checks: Mapping[str, PublicCheckResult]) -> str:
    rows = []
    for cid in pool_ids:
        check = public_checks.get(cid)
        rows.append({"candidate_id": cid, "check": None if check is None else {
            "status": check.status.value, "cases_total": check.cases_total, "cases_passed": check.cases_passed,
            "case_outcomes": list(check.case_outcomes), "code_sha256": check.code_sha256,
            "checker_version": check.checker_version, "public_inputs_sha256": check.public_inputs_sha256,
            "environment_digest": check.environment_digest}})
    return canonical_hash(rows)


def _validate_check(cid: str, check: PublicCheckResult, task: PublicTask) -> None:
    if check.candidate_id != cid or check.item_id != task.item_id:
        raise ValueError(f"public check keyed {cid!r} belongs to {check.item_id}/{check.candidate_id}")
    if not 0 <= check.cases_passed <= check.cases_total:
        raise ValueError(f"public check {cid}: cases_passed {check.cases_passed} outside 0..{check.cases_total}")
    if check.status is CheckStatus.OK and check.case_outcomes:
        if len(check.case_outcomes) != check.cases_total or check.case_outcomes.count("pass") != check.cases_passed:
            raise ValueError(f"public check {cid}: case_outcomes disagree with the pass counts")


def public_example_selection(pool_ids: Sequence[str], task: PublicTask, candidates: Mapping[str, Candidate],
                             public_checks: Mapping[str, PublicCheckResult], *, seed: int) -> PublicSelectionOutcome:
    evidence = public_evidence_hash(pool_ids, public_checks)
    eligible: list[str] = []
    paused = False
    versions: set[tuple[str, str]] = set()
    for cid in pool_ids:
        check = public_checks.get(cid)
        if check is None:
            if candidates[cid].parse_status is ParseStatus.VALID:
                raise ValueError(f"{task.item_id}: valid candidate {cid} has no public check yet; select after checks")
            continue
        _validate_check(cid, check, task)
        if check.status is CheckStatus.INFRASTRUCTURE_ERROR or (check.status is CheckStatus.OK
                                                                 and check.pass_fraction is None):
            paused = True
        elif check.status is CheckStatus.OK and candidates[cid].parse_status is ParseStatus.VALID:
            eligible.append(cid)
            versions.add((check.checker_version, check.public_inputs_sha256))
    if len(versions) > 1:
        raise ValueError(f"{task.item_id}: pool mixes public checker versions/inputs {sorted(versions)}")
    if paused:
        return PublicSelectionOutcome(SelectionStatus.PAUSED_CHECK_ERROR, None, None, (), len(eligible), evidence)
    if not eligible:
        return PublicSelectionOutcome(SelectionStatus.ABSTAINED_ALL_INVALID, None, None, (), 0, evidence)
    best = max(_fraction(public_checks[cid]) for cid in eligible)
    tied = sorted((cid for cid in eligible if _fraction(public_checks[cid]) == best),
                  key=lambda cid: (candidate_priority_key(seed, task.item_id, cid), cid))
    return PublicSelectionOutcome(SelectionStatus.SELECTED, tied[0], best, tuple(tied), len(eligible), evidence)
