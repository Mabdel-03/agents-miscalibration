"""Per-item metric tables joined to grades only through a verified seal (MASTER_SPEC §5, §12, §15.9/14/17/18).

``build_partition_tables(sealed, records, grades)`` is the only grade join. ``sealed`` must be a ``SealedSelections``
from ``selection.seal.load_sealed_selections`` (it cannot be constructed otherwise), so an unsealed partition never
reaches grades. Every checkpoint of every item yields a row -- a checkpoint without a sealed selection is a
``missing_selection`` row (a blocker), never a dropped row or a zero.

Tables (one row type each; columns are the dataclass fields, in order; ``TABLE_SCHEMAS`` lists them):

* ``checkpoints`` -- one row per (item, checkpoint, view): pool counts by candidate state, current-pool coverage,
  archive coverage over ``required_node_ids`` (only what that policy generated), POOLED input-bank coverage, the
  sealed selection, delivered state and bounds, and the same-pool selection gap for candidate-only readouts
  (``None`` for POOLED synthesis). Exact-key agreement columns are text-only (``None`` for code).
* ``bank`` -- finite-bank oracle pass@k for k = 1..bank from the item's CURRENT ``sample/<bank>`` pool only.
* ``slot_transitions`` / ``team_transitions`` -- consecutive CURRENT steps of ``private_single`` (t),
  ``private_team`` (n, r), ``peer`` (n, r) and ``cp`` (h); fine C/W/I/U kinds, the counting identity (raises on a
  residual), coverage loss/recovery and delivered-state changes. PRIVATE_TEAM trajectories are shared across n by
  design: rows for different n repeat the same slot trajectories and must not be summed across n.
* ``root_strata`` -- pre-communication root correctness among roots[0:n] for each team size n, plus the
  independent reference bank roots[max_n:bank].
* ``failures`` -- per item and generation kind/budget: transport, completion and parse status counts.

Invariants checked while building (each raises ``MetricsInvariantError``): sealed pool hash equals the pool the
metrics see; the delivered candidate belongs to its pool; ``delivered <= coverage`` (lo and hi) for candidate-only
readouts; archives nest and archive coverage never decreases along a trajectory; ``sample/k`` pools are
``bank[0:k]`` for every k = 1..bank; every trajectory step is unique. Current coverage and delivered accuracy are
*not* required to be monotone. ``check_fixed_populations`` verifies that every checkpoint name within
(partition, benchmark, family, view, n) covers the same items.

Config keys consumed: none.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.contracts import (
    Candidate, CheckpointView, CompletionStatus, GenerationResult, GenerationSpec, GradeResult, OutputKind,
    ParseStatus, PolicyCheckpoint, PublicTask, SelectionResult, TransportStatus,
)
from agent_allocation.metrics.outcomes import (
    CandidateState, DeliveredState, candidate_state, coverage_bounds, delivered_bounds, delivered_state,
)
from agent_allocation.metrics.passk import bank_pass_at_k
from agent_allocation.metrics.transitions import TRANSITION_KINDS, check_identity, fold, kind_counts
from agent_allocation.selection.keys import option_letters, vote_key
from agent_allocation.selection.readout import FAMILIES, pool_hash
from agent_allocation.selection.seal import SealedSelections, SealError

__all__ = [
    "BankRow", "CheckpointRow", "FailureRow", "ItemRecords", "MetricsInvariantError", "PartitionRecords",
    "PartitionTables", "PopulationError", "RootStratumRow", "SlotTransitionRow", "TABLE_SCHEMAS", "TeamTransitionRow",
    "build_partition_tables", "check_fixed_populations", "population_gaps", "summarize_checkpoints",
]

METRICS_VERSION = "06-metrics-v1"
TRAJECTORY_FAMILIES = ("private_single", "private_team", "peer", "cp")
_FAMILY_ORDER = {family: i for i, family in enumerate(FAMILIES)}


class MetricsInvariantError(ValueError):
    """An algebraic or structural invariant failed: the inputs come from inconsistent populations or pools."""


class PopulationError(MetricsInvariantError):
    """Checkpoints of one comparison group do not cover the same items."""


# --------------------------------------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class ItemRecords:
    task: PublicTask
    checkpoints: tuple[PolicyCheckpoint, ...]
    candidates: Mapping[str, Candidate]
    results: Mapping[str, GenerationResult] = field(default_factory=dict)   # node_id -> result (failure rates)
    specs: Mapping[str, GenerationSpec] = field(default_factory=dict)       # node_id -> spec (kind/budget)


@dataclass(frozen=True)
class PartitionRecords:
    partition: str
    items: tuple[ItemRecords, ...]


# --------------------------------------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class CheckpointRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    policy_id: str
    name: str
    family: str
    view: str
    n: int | None                           # team size: private_team/peer (from name), cp/pooled (team_size)
    step: int | None                        # k (sample), t (private_single), r (private_team/peer), h (cp), m (deep)
    team_size: int
    total_cycles: int
    peer_rounds: int
    reasoning_budget: int
    nominal_generations: int                # len(required_node_ids): the policy's own generation ancestors
    pool_size: int
    pool_n_valid_parse: int
    pool_n_votes: int | None                # text: candidates with a gold-blind vote key; None for code
    pool_n_distinct_answers: int | None     # text: distinct exact keys among votes
    pool_top_answer_support: int | None     # text: largest exact-key class
    pool_unanimous: bool | None             # text: every pool member votes and all keys agree
    pool_n_correct: int
    pool_n_wrong: int
    pool_n_invalid: int
    pool_n_unknown: int
    pool_n_missing: int
    coverage_lo: int
    coverage_hi: int
    archive_size: int
    archive_coverage_lo: int
    archive_coverage_hi: int
    input_bank_coverage_lo: int | None      # POOLED current view only: required ancestors outside the pool
    input_bank_coverage_hi: int | None
    required_n_missing_candidate: int
    required_n_invalid_parse: int
    required_n_transport_failed: int | None  # None when results were not supplied
    required_n_truncated: int | None
    selection_status: str | None
    selector_name: str | None
    declared_selector_name: str
    selector_version: str | None
    selection_cpu_seconds: float | None
    pool_hash: str | None
    public_evidence_hash: str | None
    selected_candidate_id: str | None
    selected_parse_status: str | None
    selected_state: str | None
    delivered_state: str
    delivered_correct: bool | None
    delivered_lo: int
    delivered_hi: int
    candidate_only: bool
    selection_gap_lo: int | None
    selection_gap_hi: int | None


@dataclass(frozen=True)
class BankRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    bank_size: int
    k: int
    c_lo: int
    c_hi: int
    n_invalid: int
    n_unknown: int
    n_missing: int
    pass_at_k_lo: float
    pass_at_k_hi: float
    pass_at_k_lo_exact: str
    pass_at_k_hi_exact: str
    known: bool


@dataclass(frozen=True)
class SlotTransitionRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    family: str
    n: int | None
    from_step: int
    to_step: int
    slot: int
    from_candidate_id: str
    to_candidate_id: str
    state_from: str
    state_to: str
    kind: str
    validity_change: bool | None            # valid parse <-> invalid parse; None if either candidate is missing
    answer_changed: bool | None             # text only, both valid votes: exact key changed; None otherwise


@dataclass(frozen=True)
class TeamTransitionRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    family: str
    n: int | None
    from_step: int
    to_step: int
    n_slots: int
    c_prev: int
    c_next: int
    u_prev: int
    u_next: int
    fully_known: bool
    delta_c: int
    n_w_to_c: int                           # N_WC + N_IC
    n_c_to_w: int                           # N_CW + N_CI
    n_CC: int
    n_CW: int
    n_CI: int
    n_CU: int
    n_WC: int
    n_WW: int
    n_WI: int
    n_WU: int
    n_IC: int
    n_IW: int
    n_II: int
    n_IU: int
    n_UC: int
    n_UW: int
    n_UI: int
    n_UU: int
    coverage_prev_lo: int
    coverage_prev_hi: int
    coverage_next_lo: int
    coverage_next_hi: int
    coverage_lost: bool | None              # known covered -> known uncovered
    coverage_recovered: bool | None         # known uncovered -> known covered
    delivered_prev_state: str
    delivered_next_state: str
    delivered_transition: str | None        # e.g. "correct->wrong" when both states are known outcomes


@dataclass(frozen=True)
class RootStratumRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    n: int
    c0_lo: int
    c0_hi: int
    stratum: str                            # none_correct | correct_minority | correct_half | correct_majority |
                                            # all_correct | unknown
    r0_policy_id: str
    r0_delivered_state: str
    reference_bank_size: int | None         # roots[max_n:bank]; None without a sample bank
    reference_c_lo: int | None
    reference_c_hi: int | None


@dataclass(frozen=True)
class FailureRow:
    partition: str
    benchmark: str
    subtask: str | None
    item_id: str
    generation_kind: str                    # GenerationKind value, or "unknown" without specs
    reasoning_budget: int | None
    n_nodes: int
    n_results: int
    n_transport_ok: int
    n_transport_failed: int
    n_completion_length: int
    n_completion_blocked: int
    n_completion_empty: int
    n_completion_other: int
    n_candidates: int
    n_parse_valid: int
    n_parse_missing_final: int
    n_parse_malformed_final: int
    n_parse_empty_content: int
    n_parse_refusal: int


@dataclass(frozen=True)
class PartitionTables:
    checkpoints: tuple[CheckpointRow, ...]
    bank: tuple[BankRow, ...]
    slot_transitions: tuple[SlotTransitionRow, ...]
    team_transitions: tuple[TeamTransitionRow, ...]
    root_strata: tuple[RootStratumRow, ...]
    failures: tuple[FailureRow, ...]

    def as_dicts(self) -> dict[str, list[dict[str, Any]]]:
        return {f.name: [dataclasses.asdict(row) for row in getattr(self, f.name)]
                for f in dataclasses.fields(self)}


TABLE_ROW_TYPES: dict[str, type] = {
    "checkpoints": CheckpointRow, "bank": BankRow, "slot_transitions": SlotTransitionRow,
    "team_transitions": TeamTransitionRow, "root_strata": RootStratumRow, "failures": FailureRow,
}
TABLE_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    table: tuple((f.name, str(f.type)) for f in dataclasses.fields(row_type))
    for table, row_type in TABLE_ROW_TYPES.items()
}


# --------------------------------------------------------------------------------------------------------- helpers


def _coord(cp: PolicyCheckpoint) -> tuple[int | None, int | None]:
    parts = cp.name.split("/") if cp.name else []
    if not parts or parts[0] != cp.family:
        raise MetricsInvariantError(f"{cp.policy_id}: checkpoint name {cp.name!r} does not start with {cp.family!r}")
    try:
        numbers = [int(p) for p in parts[1:]]
    except ValueError as exc:
        raise MetricsInvariantError(f"{cp.policy_id}: non-integer coordinate in {cp.name!r}") from exc
    expected = 2 if cp.family in ("private_team", "peer") else 1
    if len(numbers) != expected:
        raise MetricsInvariantError(f"{cp.policy_id}: {cp.family} names carry {expected} coordinate(s)")
    if cp.family in ("private_team", "peer"):
        return numbers[0], numbers[1]
    if cp.family == "pooled":
        return numbers[0], None
    if cp.family == "cp":
        return (cp.team_size or None), numbers[0]
    return None, numbers[0]


def _order(cp: PolicyCheckpoint) -> tuple[Any, ...]:
    n, step = _coord(cp)
    return (_FAMILY_ORDER.get(cp.family, len(_FAMILY_ORDER)), n if n is not None else -1,
            step if step is not None else -1, CheckpointView(cp.view).value, cp.policy_id)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _gap_bounds(dstate: DeliveredState, selected: str | None, pool: Sequence[str],
                states: Mapping[str, CandidateState], cov: tuple[int, int]) -> tuple[int, int]:
    if dstate is DeliveredState.CORRECT:
        return 0, 0
    if dstate is DeliveredState.ABSTAINED:
        return cov
    if dstate in (DeliveredState.PAUSED, DeliveredState.MISSING_SELECTION):
        return 0, cov[1]
    others = coverage_bounds(states[c] for c in pool if c != selected)
    if dstate is DeliveredState.UNKNOWN:
        return 0, others[1]
    return others                                   # wrong / invalid_output: delivered is 0


class _Item:
    """One item's joined view: states per candidate id and helpers shared by all tables."""

    def __init__(self, partition: str, item: ItemRecords, grades: Mapping[str, GradeResult]) -> None:
        self.partition = partition
        self.item = item
        self.task = item.task
        ids: set[str] = set()
        for cp in item.checkpoints:
            ids.update(cp.candidate_ids)
            ids.update(cp.required_node_ids)
        self.states = {cid: candidate_state(item.candidates.get(cid), grades.get(cid)) for cid in sorted(ids)}
        self._letters = option_letters(self.task) if self.task.output_kind is OutputKind.OPTION else None
        self._keys: dict[str, str | None] = {}

    def key(self, cid: str) -> str | None:
        """Exact gold-blind vote key of a text candidate (``None`` for code, invalid or absent candidates)."""
        if cid not in self._keys:
            candidate = self.item.candidates.get(cid)
            found = None if candidate is None or self.task.output_kind is OutputKind.PYTHON \
                else vote_key(candidate, self.task, letters=self._letters)
            self._keys[cid] = None if found is None else found.exact
        return self._keys[cid]

    def base(self) -> dict[str, Any]:
        return {"partition": self.partition, "benchmark": self.task.benchmark, "subtask": self.task.subtask,
                "item_id": self.task.item_id}


def _checkpoint_row(it: _Item, cp: PolicyCheckpoint, selection: SelectionResult | None) -> CheckpointRow:
    task, candidates, states = it.task, it.item.candidates, it.states
    view = CheckpointView(cp.view)
    pool, required = tuple(cp.candidate_ids), tuple(cp.required_node_ids)
    if not pool:
        raise MetricsInvariantError(f"{cp.policy_id}: empty candidate pool")
    if not set(pool) <= set(required):
        raise MetricsInvariantError(f"{cp.policy_id}: pool holds candidates the policy did not generate")
    n, step = _coord(cp)
    pool_states = [states[c] for c in pool]
    cov = coverage_bounds(pool_states)
    arch = coverage_bounds(states[c] for c in required)
    counts = {state: sum(s is state for s in pool_states) for state in CandidateState}

    selected_state = None
    if selection is not None:
        if not all(c in candidates for c in pool):
            raise MetricsInvariantError(f"{cp.policy_id}: a sealed selection's pool candidate is no longer available")
        if selection.pool_hash != pool_hash(pool, candidates):
            raise MetricsInvariantError(f"{cp.policy_id}: sealed pool hash differs from the pool being graded")
        chosen = selection.selected_candidate_id
        if chosen is not None:
            if chosen not in pool:
                raise MetricsInvariantError(f"{cp.policy_id}: delivered candidate {chosen} is outside its pool")
            selected_state = states[chosen]
    dstate = delivered_state(selection, selected_state)
    d_lo, d_hi = delivered_bounds(dstate)
    d_hi = min(d_hi, cov[1])                        # every readout delivers a member of its own pool
    candidate_only = not (cp.family == "pooled" and view is CheckpointView.CURRENT)
    if candidate_only and (d_lo > cov[0] or d_hi > cov[1]):
        raise MetricsInvariantError(f"{cp.policy_id}: delivered accuracy exceeds same-pool coverage")
    gap = _gap_bounds(dstate, None if selection is None else selection.selected_candidate_id, pool, states, cov) \
        if candidate_only else (None, None)
    input_bank = (None, None)
    if cp.family == "pooled" and view is CheckpointView.CURRENT:
        input_bank = coverage_bounds(states[c] for c in required if c not in set(pool))

    text = task.output_kind is not OutputKind.PYTHON
    keys = [it.key(c) for c in pool] if text else []
    votes = [k for k in keys if k is not None]
    support = max((votes.count(k) for k in set(votes)), default=0) if text else None
    results = it.item.results
    chosen_candidate = None if selection is None or selection.selected_candidate_id is None \
        else candidates.get(selection.selected_candidate_id)
    return CheckpointRow(
        **it.base(), policy_id=cp.policy_id, name=cp.name, family=cp.family, view=view.value, n=n, step=step,
        team_size=cp.team_size, total_cycles=cp.total_cycles, peer_rounds=cp.peer_rounds,
        reasoning_budget=cp.reasoning_budget, nominal_generations=len(required), pool_size=len(pool),
        pool_n_valid_parse=sum(1 for c in pool if c in candidates and candidates[c].parse_status is ParseStatus.VALID),
        pool_n_votes=len(votes) if text else None, pool_n_distinct_answers=len(set(votes)) if text else None,
        pool_top_answer_support=support,
        pool_unanimous=(len(votes) == len(pool) and len(set(votes)) == 1) if text else None,
        pool_n_correct=counts[CandidateState.CORRECT], pool_n_wrong=counts[CandidateState.WRONG],
        pool_n_invalid=counts[CandidateState.INVALID], pool_n_unknown=counts[CandidateState.UNKNOWN],
        pool_n_missing=counts[CandidateState.MISSING], coverage_lo=cov[0], coverage_hi=cov[1],
        archive_size=len(required), archive_coverage_lo=arch[0], archive_coverage_hi=arch[1],
        input_bank_coverage_lo=input_bank[0], input_bank_coverage_hi=input_bank[1],
        required_n_missing_candidate=sum(1 for c in required if c not in candidates),
        required_n_invalid_parse=sum(1 for c in required
                                     if c in candidates and candidates[c].parse_status is not ParseStatus.VALID),
        required_n_transport_failed=(sum(1 for c in required if c in results
                                         and results[c].transport_status is not TransportStatus.OK)
                                     if results else None),
        required_n_truncated=(sum(1 for c in required if c in results and _truncated(results[c]))
                              if results else None),
        selection_status=None if selection is None else selection.status,
        selector_name=None if selection is None else selection.selector_name,
        declared_selector_name=cp.selector_name,
        selector_version=None if selection is None else selection.selector_version,
        selection_cpu_seconds=None if selection is None else selection.cpu_seconds,
        pool_hash=None if selection is None else selection.pool_hash,
        public_evidence_hash=None if selection is None else selection.public_evidence_hash,
        selected_candidate_id=None if selection is None else selection.selected_candidate_id,
        selected_parse_status=None if chosen_candidate is None else chosen_candidate.parse_status.value,
        selected_state=None if selected_state is None else selected_state.value,
        delivered_state=dstate.value,
        delivered_correct=None if dstate in (DeliveredState.UNKNOWN, DeliveredState.PAUSED,
                                             DeliveredState.MISSING_SELECTION) or d_lo != d_hi else bool(d_lo),
        delivered_lo=d_lo, delivered_hi=d_hi, candidate_only=candidate_only,
        selection_gap_lo=gap[0], selection_gap_hi=gap[1],
    )


def _truncated(result: GenerationResult) -> bool:
    return result.completion_status is CompletionStatus.LENGTH or result.finish_reason == "length"


def _bank_rows(it: _Item, samples: Sequence[PolicyCheckpoint]) -> tuple[list[BankRow], tuple[str, ...] | None]:
    if not samples:
        return [], None
    by_k: dict[int, PolicyCheckpoint] = {}
    for cp in samples:
        _, k = _coord(cp)
        if k in by_k:
            raise MetricsInvariantError(f"{cp.policy_id}: duplicate sample/{k} checkpoint")
        by_k[k] = cp
    size = max(by_k)
    bank = tuple(by_k[size].candidate_ids)
    if len(bank) != size or len(set(bank)) != size:
        raise MetricsInvariantError(f"{it.task.item_id}: sample/{size} does not hold {size} distinct roots")
    for k in range(1, size + 1):
        if k not in by_k:
            raise PopulationError(f"{it.task.item_id}: sample/{k} is missing; every k=1..{size} must be reported")
        if tuple(by_k[k].candidate_ids) != bank[:k]:
            raise MetricsInvariantError(f"{by_k[k].policy_id}: sample/{k} is not the bank prefix root[0:{k}]")
    states = [it.states[c] for c in bank]
    correct = [True if s is CandidateState.CORRECT else None if s in (CandidateState.UNKNOWN, CandidateState.MISSING)
               else False for s in states]
    rows = []
    for k in range(1, size + 1):
        value = bank_pass_at_k(correct, k)
        rows.append(BankRow(**it.base(), bank_size=size, k=k, c_lo=value.c_lo, c_hi=value.c_hi,
                            n_invalid=sum(s is CandidateState.INVALID for s in states),
                            n_unknown=sum(s is CandidateState.UNKNOWN for s in states),
                            n_missing=sum(s is CandidateState.MISSING for s in states),
                            pass_at_k_lo=float(value.lo), pass_at_k_hi=float(value.hi),
                            pass_at_k_lo_exact=_fraction_text(value.lo), pass_at_k_hi_exact=_fraction_text(value.hi),
                            known=value.lo == value.hi))
    return rows, bank


def _transitions(it: _Item, trajectory: Sequence[PolicyCheckpoint],
                 rows_by_policy: Mapping[str, CheckpointRow]) -> tuple[list[SlotTransitionRow],
                                                                     list[TeamTransitionRow]]:
    slot_rows: list[SlotTransitionRow] = []
    team_rows: list[TeamTransitionRow] = []
    steps = sorted(((_coord(cp)[1], cp) for cp in trajectory), key=lambda pair: pair[0])
    seen = [s for s, _ in steps]
    if len(seen) != len(set(seen)):
        raise MetricsInvariantError(f"{it.task.item_id}: duplicate trajectory steps {seen} for {trajectory[0].family}")
    for (s0, a), (s1, b) in zip(steps, steps[1:]):
        if not set(a.required_node_ids) <= set(b.required_node_ids):
            raise MetricsInvariantError(f"{b.policy_id}: archive does not contain the previous step's archive")
        ra, rb = rows_by_policy[a.policy_id], rows_by_policy[b.policy_id]
        if rb.archive_coverage_lo < ra.archive_coverage_lo or rb.archive_coverage_hi < ra.archive_coverage_hi:
            raise MetricsInvariantError(f"{b.policy_id}: archive coverage decreased")
        if s1 != s0 + 1:
            continue
        if len(a.candidate_ids) != len(b.candidate_ids):
            raise MetricsInvariantError(f"{b.policy_id}: slot count changed along a trajectory")
        before = [it.states[c] for c in a.candidate_ids]
        after = [it.states[c] for c in b.candidate_ids]
        counts = kind_counts(zip(before, after))
        delta = check_identity(before, after, counts)
        n, _ = _coord(b)
        for slot, (ca, cb) in enumerate(zip(a.candidate_ids, b.candidate_ids)):
            cand_a, cand_b = it.item.candidates.get(ca), it.item.candidates.get(cb)
            validity = None if cand_a is None or cand_b is None else \
                (cand_a.parse_status is ParseStatus.VALID) != (cand_b.parse_status is ParseStatus.VALID)
            ka, kb = it.key(ca), it.key(cb)
            slot_rows.append(SlotTransitionRow(
                **it.base(), family=b.family, n=n, from_step=s0, to_step=s1, slot=slot, from_candidate_id=ca,
                to_candidate_id=cb, state_from=fold(it.states[ca]), state_to=fold(it.states[cb]),
                kind=fold(it.states[ca]) + fold(it.states[cb]), validity_change=validity,
                answer_changed=None if ka is None or kb is None else ka != kb))
        unknown = lambda states: sum(fold(s) == "U" for s in states)  # noqa: E731
        lost = recovered = None
        if ra.coverage_lo == ra.coverage_hi and rb.coverage_lo == rb.coverage_hi:
            lost = ra.coverage_lo == 1 and rb.coverage_lo == 0
            recovered = ra.coverage_lo == 0 and rb.coverage_lo == 1
        known_outcomes = {"correct", "wrong", "invalid_output", "abstained"}
        team_rows.append(TeamTransitionRow(
            **it.base(), family=b.family, n=n, from_step=s0, to_step=s1, n_slots=len(before),
            c_prev=sum(s is CandidateState.CORRECT for s in before), c_next=sum(s is CandidateState.CORRECT for s in after),
            u_prev=unknown(before), u_next=unknown(after), fully_known=unknown(before) + unknown(after) == 0,
            delta_c=delta, n_w_to_c=counts["WC"] + counts["IC"], n_c_to_w=counts["CW"] + counts["CI"],
            **{f"n_{kind}": counts[kind] for kind in TRANSITION_KINDS},
            coverage_prev_lo=ra.coverage_lo, coverage_prev_hi=ra.coverage_hi, coverage_next_lo=rb.coverage_lo,
            coverage_next_hi=rb.coverage_hi, coverage_lost=lost, coverage_recovered=recovered,
            delivered_prev_state=ra.delivered_state, delivered_next_state=rb.delivered_state,
            delivered_transition=(f"{ra.delivered_state}->{rb.delivered_state}"
                                  if {ra.delivered_state, rb.delivered_state} <= known_outcomes else None)))
    return slot_rows, team_rows


def _stratum(c0: int, n: int) -> str:
    if c0 == 0:
        return "none_correct"
    if c0 == n:
        return "all_correct"
    if 2 * c0 < n:
        return "correct_minority"
    return "correct_half" if 2 * c0 == n else "correct_majority"


def _strata_rows(it: _Item, current: Sequence[PolicyCheckpoint], rows_by_policy: Mapping[str, CheckpointRow],
                 bank: tuple[str, ...] | None) -> list[RootStratumRow]:
    initial: dict[int, PolicyCheckpoint] = {}
    for family in ("private_team", "peer"):                  # peer/n/0 wins when both exist (same roots)
        for cp in current:
            n, step = _coord(cp) if cp.family == family else (None, None)
            if cp.family == family and step == 0 and n is not None:
                initial[n] = cp
    if not initial:
        return []
    max_n = max(initial)
    reference = bank[max_n:] if bank is not None and len(bank) > max_n else None
    ref_states = [it.states[c] for c in reference] if reference else []
    rows = []
    for n in sorted(initial):
        cp = initial[n]
        states = [it.states[c] for c in cp.candidate_ids]
        lo = sum(s is CandidateState.CORRECT for s in states)
        hi = lo + sum(s in (CandidateState.UNKNOWN, CandidateState.MISSING) for s in states)
        rows.append(RootStratumRow(
            **it.base(), n=n, c0_lo=lo, c0_hi=hi, stratum=_stratum(lo, len(states)) if lo == hi else "unknown",
            r0_policy_id=cp.policy_id, r0_delivered_state=rows_by_policy[cp.policy_id].delivered_state,
            reference_bank_size=len(reference) if reference else None,
            reference_c_lo=sum(s is CandidateState.CORRECT for s in ref_states) if reference else None,
            reference_c_hi=(sum(s in (CandidateState.CORRECT, CandidateState.UNKNOWN, CandidateState.MISSING)
                                for s in ref_states) if reference else None)))
    return rows


def _failure_rows(it: _Item) -> list[FailureRow]:
    nodes: set[str] = set()
    for cp in it.item.checkpoints:
        nodes.update(cp.required_node_ids)
    groups: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    for node in sorted(nodes):
        spec = it.item.specs.get(node)
        groups[("unknown", None) if spec is None else (spec.kind.value, spec.reasoning_budget)].append(node)
    rows = []
    for (kind, budget), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        results = [it.item.results[m] for m in members if m in it.item.results]
        cands = [it.item.candidates[m] for m in members if m in it.item.candidates]
        completion = [r.completion_status for r in results if r.transport_status is TransportStatus.OK]
        parse = [c.parse_status for c in cands]
        rows.append(FailureRow(
            **it.base(), generation_kind=kind, reasoning_budget=budget, n_nodes=len(members), n_results=len(results),
            n_transport_ok=sum(r.transport_status is TransportStatus.OK for r in results),
            n_transport_failed=sum(r.transport_status is not TransportStatus.OK for r in results),
            n_completion_length=sum(s is CompletionStatus.LENGTH for s in completion),
            n_completion_blocked=sum(s is CompletionStatus.BLOCKED for s in completion),
            n_completion_empty=sum(s is CompletionStatus.EMPTY for s in completion),
            n_completion_other=sum(s in (CompletionStatus.OTHER, None) for s in completion),
            n_candidates=len(cands), n_parse_valid=parse.count(ParseStatus.VALID),
            n_parse_missing_final=parse.count(ParseStatus.MISSING),
            n_parse_malformed_final=parse.count(ParseStatus.MALFORMED),
            n_parse_empty_content=parse.count(ParseStatus.EMPTY), n_parse_refusal=parse.count(ParseStatus.REFUSAL)))
    return rows


# --------------------------------------------------------------------------------------------------------- build


def build_partition_tables(sealed: SealedSelections, records: PartitionRecords, grades: Mapping[str, GradeResult],
                           *, strict_populations: bool = True) -> PartitionTables:
    """Join grades to a sealed partition. ``strict_populations=False`` is for nested designs (the pilot's r2 subset);
    ``population_gaps`` then lists the differences instead of raising."""
    if not isinstance(sealed, SealedSelections):
        raise SealError("grades are joined only to a verified SealedSelections (seal the partition first)")
    if sealed.partition != records.partition:
        raise ValueError(f"sealed partition {sealed.partition!r} != records partition {records.partition!r}")
    for cid, grade in grades.items():
        if not isinstance(grade, GradeResult) or grade.candidate_id != cid:
            raise TypeError(f"grades[{cid!r}] must be the GradeResult of that candidate")
    policy_ids: set[str] = set()
    item_ids: set[str] = set()
    tables: dict[str, list[Any]] = {name: [] for name in TABLE_ROW_TYPES}
    for item in records.items:
        if item.task.item_id in item_ids:
            raise PopulationError(f"item {item.task.item_id} appears twice in partition {records.partition}")
        item_ids.add(item.task.item_id)
        it = _Item(records.partition, item, grades)
        rows_by_policy: dict[str, CheckpointRow] = {}
        for cp in sorted(item.checkpoints, key=_order):
            if cp.item_id != item.task.item_id or not cp.policy_id.startswith(f"{records.partition}/"):
                raise MetricsInvariantError(f"{cp.policy_id}: checkpoint outside item/partition")
            if cp.family not in FAMILIES:
                raise MetricsInvariantError(f"{cp.policy_id}: unknown family {cp.family!r}")
            if cp.policy_id in policy_ids:
                raise MetricsInvariantError(f"{cp.policy_id}: duplicate checkpoint")
            policy_ids.add(cp.policy_id)
            row = _checkpoint_row(it, cp, sealed.by_policy.get(cp.policy_id))
            rows_by_policy[cp.policy_id] = row
            tables["checkpoints"].append(row)
        current = [cp for cp in sorted(item.checkpoints, key=_order) if CheckpointView(cp.view) is CheckpointView.CURRENT]
        bank_rows, bank = _bank_rows(it, [cp for cp in current if cp.family == "sample"])
        tables["bank"].extend(bank_rows)
        sample_steps = [cp for cp in current if cp.family == "sample"]
        _check_archive_monotone(sample_steps, rows_by_policy)
        trajectories: dict[tuple[str, int | None], list[PolicyCheckpoint]] = defaultdict(list)
        for cp in current:
            if cp.family in TRAJECTORY_FAMILIES:
                trajectories[(cp.family, _coord(cp)[0])].append(cp)
        for key in sorted(trajectories, key=lambda k: (_FAMILY_ORDER[k[0]], k[1] or 0)):
            slots, teams = _transitions(it, trajectories[key], rows_by_policy)
            tables["slot_transitions"].extend(slots)
            tables["team_transitions"].extend(teams)
        tables["root_strata"].extend(_strata_rows(it, current, rows_by_policy, bank))
        tables["failures"].extend(_failure_rows(it))
    stray = sorted(set(sealed.by_policy) - policy_ids)
    if stray:
        raise MetricsInvariantError(f"sealed selections without a checkpoint in these records: {stray[:5]}")
    result = PartitionTables(**{name: tuple(rows) for name, rows in tables.items()})
    if strict_populations:
        check_fixed_populations(result.checkpoints)
    return result


def _check_archive_monotone(steps: Sequence[PolicyCheckpoint], rows_by_policy: Mapping[str, CheckpointRow]) -> None:
    ordered = sorted(steps, key=lambda cp: _coord(cp)[1])
    for a, b in zip(ordered, ordered[1:]):
        if not set(a.required_node_ids) <= set(b.required_node_ids):
            raise MetricsInvariantError(f"{b.policy_id}: archive does not contain the previous step's archive")
        ra, rb = rows_by_policy[a.policy_id], rows_by_policy[b.policy_id]
        if rb.archive_coverage_lo < ra.archive_coverage_lo or rb.archive_coverage_hi < ra.archive_coverage_hi:
            raise MetricsInvariantError(f"{b.policy_id}: archive coverage decreased")


def population_gaps(rows: Iterable[CheckpointRow]) -> list[dict[str, Any]]:
    """Checkpoint names whose item set differs from their (partition, benchmark, family, view, n) group's union."""
    groups: dict[tuple[Any, ...], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        groups[(row.partition, row.benchmark, row.family, row.view, row.n)][row.name].add(row.item_id)
    gaps = []
    for (partition, benchmark, family, view, n), by_name in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        union = set().union(*by_name.values())
        for name in sorted(by_name):
            missing = sorted(union - by_name[name])
            if missing:
                gaps.append({"partition": partition, "benchmark": benchmark, "family": family, "view": view, "n": n,
                             "name": name, "n_group_items": len(union), "missing_item_ids": missing})
    return gaps


def check_fixed_populations(rows: Iterable[CheckpointRow]) -> None:
    """Every checkpoint name in one (partition, benchmark, family, view, n) group covers the same item set."""
    gaps = population_gaps(rows)
    if gaps:
        first = gaps[0]
        raise PopulationError(f"{len(gaps)} checkpoint(s) miss items of their comparison group, e.g. "
                              f"{first['name']} ({first['benchmark']}, {first['view']}) misses "
                              f"{first['missing_item_ids'][:3]}")


def summarize_checkpoints(rows: Iterable[CheckpointRow]) -> list[dict[str, Any]]:
    """Fixed-denominator means per (partition, benchmark, name, view). Unknowns widen bounds; nothing is dropped."""
    groups: dict[tuple[str, str, str, str], list[CheckpointRow]] = defaultdict(list)
    for row in rows:
        groups[(row.partition, row.benchmark, row.name, row.view)].append(row)
    out = []
    for (partition, benchmark, name, view), members in sorted(groups.items()):
        count = len(members)

        def mean(attr: str, subset: Sequence[CheckpointRow] = members) -> float | None:
            values = [getattr(r, attr) for r in subset]
            return None if not subset or any(v is None for v in values) else float(Fraction(sum(values), len(subset)))

        states = defaultdict(int)
        for r in members:
            states[r.delivered_state] += 1
        lo, hi = mean("delivered_lo"), mean("delivered_hi")
        out.append({
            "partition": partition, "benchmark": benchmark, "name": name, "view": view, "family": members[0].family,
            "n_items": count, "delivered_lo": lo, "delivered_hi": hi, "delivered": lo if lo == hi else None,
            "coverage_lo": mean("coverage_lo"), "coverage_hi": mean("coverage_hi"),
            "archive_coverage_lo": mean("archive_coverage_lo"), "archive_coverage_hi": mean("archive_coverage_hi"),
            "selection_gap_lo": mean("selection_gap_lo") if members[0].candidate_only else None,
            "selection_gap_hi": mean("selection_gap_hi") if members[0].candidate_only else None,
            **{f"n_delivered_{state.value}": states[state.value] for state in DeliveredState},
            "blocked": states[DeliveredState.MISSING_SELECTION.value] > 0,
        })
    return out
