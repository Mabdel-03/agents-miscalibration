"""ARD36 selection: plurality over canonical classes, blind ties (WP2).

Spec §7 ("End-of-round selection"): "primary selector is plurality over the five current
canonical answer classes.  Ties use a precommitted uniform class-level tie-break, not
first-in-list.  Also report expected correctness under all tied winning classes
analytically where possible.  Invalid answers cannot form a winning content class; if all
outputs are invalid, selection fails."  Spec §5.3/§3: multiplicity is preserved (five
occurrences of one answer are five votes) and K counts the *planned* opportunities,
including failed ones.  Spec §7 also fixes the mandatory FRESH cumulative baseline over
5/10/15/20 independent draws (``contract.FRESH_PREFIX_K``).

Why ARD36 does not call ``study.selection.vote.vote()`` (contract §3, verified): that
function re-keys every record through the frozen ``selection.normalize.vote_key``
dispatch, which knows only the study's HLE/BCB answer formats, and raises
``ProtocolError`` on a caller-supplied key.  ARD36's classes come from
``ard.graders.canonical_class`` (sympy canonical form for MATH-500, the option letter for
MMLU-Pro), so this module builds ``metrics_reference.PublicCandidate`` rows itself and
calls the vendored ``metrics_reference.plurality_vote`` directly — with the *same*
``VOTE_TIE`` seed derivation, imported verbatim from ``study.selection.vote`` (lines
39-43) rather than re-derived here.

Nothing in this module knows what the correct answer is.  ``canonical_class`` is
gold-independent by construction, and :func:`plurality_expected_correctness` receives a
class→correct mapping from the *analysis* side; it never imports a grader or a gold file.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter as _Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.study import metrics_reference
from agents_scaling.study.metrics_reference import PublicCandidate, plurality_vote
from agents_scaling.study.selection.vote import vote_tie_seed  # the VOTE_TIE derivation
from agents_scaling.study.types import ProtocolError

__all__ = [
    "ArdSelection",
    "PublicCandidate",
    "SELECTOR_CONFIDENCE",
    "SELECTOR_PLURALITY",
    "VoteCandidate",
    "class_counts",
    "confidence_weighted_plurality",
    "fresh_prefix_votes",
    "plurality",
    "plurality_expected_correctness",
    "prefix_vote",
    "public_candidates",
    "vote_tie_seed",
]

SELECTOR_PLURALITY = "plurality"
SELECTOR_CONFIDENCE = "confidence_weighted"

#: Weight resolution of the confidence-weighted selector.  Confidences are model-reported
#: probabilities with three decimals; quantizing before the comparison keeps the tie test
#: exact instead of float-noise dependent.
_WEIGHT_QUANTUM = 1_000_000


# --------------------------------------------------------------------------- pool rows


@dataclass(frozen=True)
class VoteCandidate:
    """One planned opportunity in a declared pool (spec §3: failed ones stay in the pool).

    ``canonical_class`` is the gold-independent class of the delivered answer, ``None``
    when the report was invalid *or* valid-but-unkeyable (an empty or unparseable answer):
    such a candidate cannot form a winning content class but still counts in
    ``planned_count``.  ``confidence`` is optional and is never imputed.
    """

    candidate_id: str
    canonical_class: str | None
    valid: bool = True
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} is outside [0, 1]")

    @property
    def keyed(self) -> bool:
        """Eligible to win: structurally valid *and* carrying a canonical class."""
        return bool(self.valid) and isinstance(self.canonical_class, str) and bool(self.canonical_class)


def public_candidates(pool: Sequence[VoteCandidate]) -> list[PublicCandidate]:
    """The public rows handed to ``metrics_reference.plurality_vote``.

    No correctness, gold answer or hidden field exists on a ``PublicCandidate``; an
    unkeyed candidate is passed through as ``valid=False`` so it occupies its planned
    opportunity without casting a vote.
    """
    return [PublicCandidate(c.candidate_id, c.canonical_class if c.keyed else None, c.keyed) for c in pool]


def class_counts(pool: Sequence[VoteCandidate]) -> dict[str, int]:
    """Canonical-class multiset of a pool (multiplicity preserved, spec §5.3)."""
    counter = _Counter(c.canonical_class for c in pool if c.keyed)
    return {str(k): int(v) for k, v in sorted(counter.items())}


# --------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class ArdSelection:
    """One sealed ARD36 selection over a declared pool (spec §7, §18 ``episodes``)."""

    selector: str
    pool_kind: str
    source_id: str
    pool_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    selected_class: str | None
    planned_count: int
    valid_count: int
    winning_count: int | None
    tied_classes: int | None
    tied_winning_classes: tuple[str, ...]
    all_singleton: bool
    selection_failed: bool
    class_counts: Mapping[str, int] = field(default_factory=dict)
    unkeyed_count: int = 0
    prefix_k: int | None = None
    missing_confidence_count: int = 0
    degenerate_weights: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "pool_kind": self.pool_kind,
            "source_id": self.source_id,
            "pool_candidate_ids": list(self.pool_candidate_ids),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_class": self.selected_class,
            "planned_count": self.planned_count,
            "valid_count": self.valid_count,
            "winning_count": self.winning_count,
            "tied_classes": self.tied_classes,
            "tied_winning_classes": list(self.tied_winning_classes),
            "all_singleton": self.all_singleton,
            "selection_failed": self.selection_failed,
            "class_counts": dict(self.class_counts),
            "unkeyed_count": self.unkeyed_count,
            "prefix_k": self.prefix_k,
            "missing_confidence_count": self.missing_confidence_count,
            "degenerate_weights": self.degenerate_weights,
        }


def _check_pool(pool: Sequence[VoteCandidate], pool_kind: str) -> None:
    if len(pool) == 0:
        raise ValueError(f"a planned {pool_kind!r} opportunity pool must not be empty")
    ids = [c.candidate_id for c in pool]
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate candidate ids would double-count a planned opportunity")


def _tied_classes(pool: Sequence[VoteCandidate], winning_count: int | None) -> tuple[str, ...]:
    if not winning_count:
        return ()
    counts = class_counts(pool)
    return tuple(sorted(name for name, n in counts.items() if n == winning_count))


def _class_of(pool: Sequence[VoteCandidate], candidate_id: str | None) -> str | None:
    if candidate_id is None:
        return None
    for candidate in pool:
        if candidate.candidate_id == candidate_id:
            return candidate.canonical_class
    return None


# --------------------------------------------------------------------------- plurality


def plurality(
    pool: Sequence[VoteCandidate],
    *,
    source_id: str,
    study_seed: bytes,
    pool_kind: str = "current",
    prefix_k: int | None = None,
) -> ArdSelection:
    """Plurality over the canonical classes of a declared pool (spec §7).

    Multiplicity is preserved; equal-size winning classes are resolved by the frozen
    ``VOTE_TIE`` HMAC class order, which reads only ``(source_id, candidate_id)`` and the
    study seed — never the answer label, the input order, the confidence or any later gold
    mapping.  With no keyed candidate the selection *fails* (``selection_failed=True``,
    ``selected_candidate_id=None``) and the opportunity is scored incorrect downstream; it
    is never dropped from the denominator.
    """
    _check_pool(pool, pool_kind)
    if pool_kind not in K.POOL_KINDS and not pool_kind.startswith("bank_prefix"):
        # informative only: WP4/WP7 add derived pool kinds (e.g. suffix, fork branches)
        pass
    selection = plurality_vote(
        public_candidates(pool), source_id=str(source_id), tie_seed=vote_tie_seed(study_seed)
    )
    keyed = [c for c in pool if c.keyed]
    return ArdSelection(
        selector=SELECTOR_PLURALITY,
        pool_kind=str(pool_kind),
        source_id=str(source_id),
        pool_candidate_ids=tuple(c.candidate_id for c in pool),
        selected_candidate_id=selection.candidate_id,
        selected_class=_class_of(pool, selection.candidate_id),
        planned_count=selection.planned_count,
        valid_count=selection.valid_count,
        winning_count=selection.winning_vote_count,
        tied_classes=selection.tied_winning_classes,
        tied_winning_classes=_tied_classes(pool, selection.winning_vote_count),
        all_singleton=bool(keyed) and selection.winning_vote_count == 1,
        selection_failed=selection.candidate_id is None,
        class_counts=class_counts(pool),
        unkeyed_count=len(pool) - len(keyed),
        prefix_k=prefix_k,
        missing_confidence_count=sum(1 for c in keyed if c.confidence is None),
    )


def prefix_vote(
    bank: Sequence[VoteCandidate],
    k: int,
    *,
    source_id: str,
    study_seed: bytes,
    pool_kind: str = "bank_prefix",
) -> ArdSelection:
    """Plurality over the first ``k`` draws of a bank **in draw order** (spec §7 FRESH).

    The bank order is the frozen draw order (``contract.fresh_draw_index``); it is never
    re-sorted, so the prefix at ``k`` is exactly "the independent candidates accumulated so
    far" and the cumulative baseline is decision-matched.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    if k > len(bank):
        raise ValueError(f"prefix k={k} exceeds the bank size {len(bank)}")
    return plurality(
        list(bank[:k]), source_id=source_id, study_seed=study_seed, pool_kind=pool_kind, prefix_k=k
    )


def fresh_prefix_votes(
    bank: Sequence[VoteCandidate],
    *,
    source_id: str,
    study_seed: bytes,
    ks: Sequence[int] = K.FRESH_PREFIX_K,
) -> dict[int, ArdSelection]:
    """The mandatory FRESH cumulative baseline at k = 5, 10, 15, 20 (spec §7).

    Prefixes longer than the realized bank are skipped rather than padded, so a downscoped
    run reports the ks it actually has instead of silently voting over fewer candidates.
    """
    out: dict[int, ArdSelection] = {}
    for k in ks:
        if int(k) <= len(bank):
            out[int(k)] = prefix_vote(bank, int(k), source_id=source_id, study_seed=study_seed)
    return out


# --------------------------------------------------------------------------- weighted


def _class_order_seed(tie_seed: bytes) -> bytes:
    """The class-order sub-seed of ``metrics_reference.plurality_vote`` (same construction)."""
    return hmac.new(tie_seed, b"VOTE_CLASS_ORDER", hashlib.sha256).digest()


def _representative_order_seed(tie_seed: bytes) -> bytes:
    return hmac.new(tie_seed, b"VOTE_REPRESENTATIVE_ORDER", hashlib.sha256).digest()


def _blind_class_key(source_id: str, group: Sequence[VoteCandidate], seed: bytes) -> tuple:
    return min(
        (metrics_reference._tie_key(source_id, c.candidate_id, seed), c.candidate_id) for c in group
    )


def confidence_weighted_plurality(
    pool: Sequence[VoteCandidate],
    *,
    source_id: str,
    study_seed: bytes,
    pool_kind: str = "current",
    prefix_k: int | None = None,
) -> ArdSelection:
    """Confidence-weighted plurality (spec §12.4 static selection competitor).

    The winning class maximizes the **sum of reported confidences** over its members.
    Spec §6.3 forbids imputing a missing confidence, so a keyed candidate with no
    confidence contributes weight 0 and is counted in ``missing_confidence_count``; if no
    keyed candidate reports one at all the weights are degenerate and the selector falls
    back to plain plurality with ``degenerate_weights=True`` recorded, rather than
    inventing 0.5s.

    Class and representative ties use the identical blind construction as
    ``metrics_reference.plurality_vote`` (same ``VOTE_TIE`` sub-seeds, same length-prefixed
    HMAC message), so the two selectors agree exactly whenever the weights reduce to
    counts — asserted in ``tests/ard/test_wp2_vote.py``.
    """
    _check_pool(pool, pool_kind)
    keyed = [c for c in pool if c.keyed]
    missing = sum(1 for c in keyed if c.confidence is None)
    weights: dict[str, int] = {}
    groups: dict[str, list[VoteCandidate]] = {}
    for candidate in keyed:
        name = str(candidate.canonical_class)
        groups.setdefault(name, []).append(candidate)
        weights[name] = weights.get(name, 0) + (
            0 if candidate.confidence is None else int(round(float(candidate.confidence) * _WEIGHT_QUANTUM))
        )
    if not keyed or max(weights.values(), default=0) <= 0:
        base = plurality(
            pool, source_id=source_id, study_seed=study_seed, pool_kind=pool_kind, prefix_k=prefix_k
        )
        return ArdSelection(
            **{
                **base.to_row(),
                "selector": SELECTOR_CONFIDENCE,
                "pool_candidate_ids": tuple(base.pool_candidate_ids),
                "tied_winning_classes": tuple(base.tied_winning_classes),
                "missing_confidence_count": missing,
                "degenerate_weights": bool(keyed),
            }
        )
    tie_seed = vote_tie_seed(study_seed)
    best = max(weights.values())
    tied = sorted(name for name, w in weights.items() if w == best)
    winner = min(
        tied, key=lambda name: _blind_class_key(str(source_id), groups[name], _class_order_seed(tie_seed))
    )
    representative = min(
        groups[winner],
        key=lambda c: (
            metrics_reference._tie_key(str(source_id), c.candidate_id, _representative_order_seed(tie_seed)),
            c.candidate_id,
        ),
    )
    return ArdSelection(
        selector=SELECTOR_CONFIDENCE,
        pool_kind=str(pool_kind),
        source_id=str(source_id),
        pool_candidate_ids=tuple(c.candidate_id for c in pool),
        selected_candidate_id=representative.candidate_id,
        selected_class=winner,
        planned_count=len(pool),
        valid_count=len(keyed),
        winning_count=len(groups[winner]),
        tied_classes=len(tied),
        tied_winning_classes=tuple(tied),
        all_singleton=all(len(g) == 1 for g in groups.values()),
        selection_failed=False,
        class_counts=class_counts(pool),
        unkeyed_count=len(pool) - len(keyed),
        prefix_k=prefix_k,
        missing_confidence_count=missing,
        degenerate_weights=False,
    )


# --------------------------------------------------------------------------- analytic tie


def plurality_expected_correctness(
    selection: ArdSelection, class_is_correct: Mapping[str, bool]
) -> float | None:
    """Expected selected correctness under a *uniform* draw over the tied winning classes.

    Spec §7: "Also report expected correctness under all tied winning classes analytically
    where possible."  The realized selection resolves the tie with one blind HMAC draw;
    this is the expectation of that draw, which is the lower-variance estimand.

    ``class_is_correct`` is supplied by the analysis stage (the class→correct mapping of a
    graded item).  This function never grades anything: it is pure arithmetic over a
    mapping, which is why ``ard/vote.py`` stays outside the gold firewall.  Returns
    ``0.0`` for a failed selection (an all-invalid pool is scored incorrect, spec §7) and
    ``None`` when a tied class is missing from the mapping — the ``class_grade_inconsistent``
    case of plan §10, which is excluded with its count reported rather than guessed.
    """
    if selection.selection_failed or not selection.tied_winning_classes:
        return 0.0
    values: list[float] = []
    for name in selection.tied_winning_classes:
        if name not in class_is_correct:
            return None
        values.append(1.0 if bool(class_is_correct[name]) else 0.0)
    return sum(values) / float(len(values))


def selection_failed(pool: Sequence[VoteCandidate]) -> bool:
    """True when no candidate in the pool can form a winning content class (spec §7)."""
    return not any(c.keyed for c in pool)
