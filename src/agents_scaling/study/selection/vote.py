"""Plurality VOTE with multiplicity and blind ties (WP3).

Spec §5.3: "Preserve duplicate votes: five occurrences of the same answer count as five
votes ... Resolve equal counts using a fixed seed-based order independent of answer
labels and protected correctness."  §4.4: "Select the largest equivalence class by
multiplicity, break class ties using a study-seeded order assigned before correctness,
then choose a representative using a separate blind seed order ... If all valid
candidates are singletons, apply that rule and report the event ... With zero valid
candidates, return NO_VALID_CANDIDATE, and score the selected opportunity incorrect.  K
counts the planned opportunities, including failed ones."  Architecture §1.9; audit
fixture T12.

The vendored ``metrics_reference.plurality_vote`` does the counting and tie logic; this
module only builds its public inputs (``PublicCandidate(candidate_id, vote_key, valid)``),
derives ``tie_seed = HMAC-SHA256(study_seed, b"VOTE_TIE")`` and records the audit fields
of a :class:`~agents_scaling.study.types.SelectionRecord`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
from collections import Counter
from collections.abc import Sequence

from agents_scaling.study import metrics_reference
from agents_scaling.study.selection.normalize import vote_key
from agents_scaling.study.types import (
    GROUPING_MODES,
    SELECTOR_VOTE,
    VOTE_TIE_NAMESPACE,
    CandidateRecord,
    ProtocolError,
    SelectionRecord,
)


def vote_tie_seed(study_seed: bytes) -> bytes:
    """``HMAC-SHA256(study_seed, b"VOTE_TIE")`` — the blind tie seed of every VOTE (§5.3)."""
    if not isinstance(study_seed, (bytes, bytearray)) or len(study_seed) == 0:
        raise ValueError("study_seed must be non-empty bytes")
    return hmac.new(bytes(study_seed), VOTE_TIE_NAMESPACE, hashlib.sha256).digest()


def keyed_record(record: CandidateRecord, answer_format: str) -> CandidateRecord:
    """Return ``record`` with ``vote_key``/``grouping_mode`` filled from its public final answer.

    Invalid records keep ``vote_key=None`` and ``grouping_mode=None``.  A record that already
    carries a key which disagrees with the frozen rule is a harness defect (``ProtocolError``),
    never silently overwritten.
    """
    if not isinstance(record, CandidateRecord):
        raise TypeError("keyed_record expects a CandidateRecord")
    if not record.valid or record.candidate is None:
        if record.vote_key is not None:
            raise ProtocolError(f"invalid candidate {record.candidate_id} carries a vote key")
        return record
    key, mode = vote_key(answer_format, record.candidate.final_answer)
    if record.vote_key is not None and (record.vote_key, record.grouping_mode) != (key, mode):
        raise ProtocolError(
            f"candidate {record.candidate_id} carries vote key {record.vote_key!r}/{record.grouping_mode!r} "
            f"but the frozen rule gives {key!r}/{mode!r}"
        )
    return dataclasses.replace(record, vote_key=key, grouping_mode=mode)


def keyed_pool(pool: Sequence[CandidateRecord], answer_format: str) -> list[CandidateRecord]:
    """Key every record of a pool (see :func:`keyed_record`); order is preserved."""
    return [keyed_record(record, answer_format) for record in pool]


def _public(record: CandidateRecord) -> metrics_reference.PublicCandidate:
    eligible = record.valid and record.candidate is not None and record.vote_key is not None
    return metrics_reference.PublicCandidate(record.candidate_id, record.vote_key if eligible else None, eligible)


def vote(
    pool: Sequence[CandidateRecord],
    source_id: str,
    study_seed: bytes,
    answer_format: str,
    *,
    pool_kind: str = "latest_slots",
    prefix_k: int | None = None,
) -> SelectionRecord:
    """Plurality VOTE over a declared pool of ``CandidateRecord`` (§4.4, §5.3).

    ``pool`` is the sealed, ordered candidate pool (every planned opportunity, invalid ones
    included); ``answer_format`` is the task's public format.  Candidate ids must be unique
    (``ValueError`` from the reference helper otherwise).  Returns a ``SelectionRecord`` with
    ``selector_id="VOTE"``: ``valid_count`` counts *keyed* valid candidates (a valid
    candidate with an empty answer is unkeyed and cannot win; it stays in
    ``planned_count``), ``winning_count``/``tied_classes`` from the reference helper,
    ``all_singleton`` when every keyed class has size 1, ``no_valid_candidate`` when none
    is keyed, ``grouping_mode`` = the most common mode among keyed candidates (first in
    ``GROUPING_MODES`` order on ties) and ``grouping_mode_counts`` for every valid one.
    """
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if len(pool) == 0:
        raise ValueError("a planned opportunity pool must not be empty")
    keyed = keyed_pool(pool, answer_format)
    public = [_public(record) for record in keyed]
    selection = metrics_reference.plurality_vote(
        public, source_id=source_id, tie_seed=vote_tie_seed(study_seed)
    )
    eligible = [p for p in public if p.valid]
    modes = Counter(record.grouping_mode for record in keyed if record.valid and record.grouping_mode)
    dominant = None
    if modes:
        top = max(modes.values())
        dominant = sorted((m for m, n in modes.items() if n == top), key=_mode_rank)[0]
    return SelectionRecord(
        selector_id=SELECTOR_VOTE,
        pool_kind=pool_kind,
        pool_candidate_ids=tuple(record.candidate_id for record in keyed),
        selected_candidate_id=selection.candidate_id,
        planned_count=selection.planned_count,
        valid_count=selection.valid_count,
        no_valid_candidate=selection.candidate_id is None,
        answer_format=answer_format,
        grouping_mode=dominant,
        grouping_mode_counts=dict(sorted(modes.items())),
        winning_count=selection.winning_vote_count,
        tied_classes=selection.tied_winning_classes,
        all_singleton=bool(eligible) and selection.winning_vote_count == 1,
        vote_keys={record.candidate_id: record.vote_key for record in keyed},
        prefix_k=prefix_k,
    )


def _mode_rank(mode: str) -> int:
    return GROUPING_MODES.index(mode) if mode in GROUPING_MODES else len(GROUPING_MODES)


def prefix_vote(
    bank: Sequence[CandidateRecord],
    k: int,
    source_id: str,
    study_seed: bytes,
    answer_format: str,
) -> SelectionRecord:
    """VOTE over the first ``k`` draws of an F bank in draw order (§5.4, §5.5 prefix selections).

    ``k`` must satisfy ``1 <= k <= len(bank)``; the bank order is the frozen draw order,
    never re-sorted.  ``pool_kind="bank_prefix"`` and ``prefix_k=k`` are recorded.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    if k > len(bank):
        raise ValueError(f"prefix k={k} exceeds the bank size {len(bank)}")
    return vote(list(bank[:k]), source_id, study_seed, answer_format, pool_kind="bank_prefix", prefix_k=k)


__all__ = ["keyed_pool", "keyed_record", "prefix_vote", "vote", "vote_tie_seed"]
