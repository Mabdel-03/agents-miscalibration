"""Small v4 metric/identity reference helpers, not the experiment runtime.

This module does not normalize answers, construct public tests, call a judge,
load gold, implement RFC 8785, schedule work, or claim a protected-data boundary.
Its public candidate/selector interfaces deliberately exclude correctness.
Authoritative labels are joined only by the evaluation helper after selection.
Production implementations must separately enforce provenance and access control.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import math
from typing import Mapping, Sequence


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased finite-bank estimator for IID complete attempts on one task.

    IID/exchangeability is a sampling-design assumption, not verifiable from
    these counts. Communicating workers or recursive subtasks are not attempts.
    Protected correctness is used only in this offline oracle metric.
    """
    _integer("n", n, 1)
    _integer("c", c)
    _integer("k", k, 1)
    if c > n or k > n:
        raise ValueError("require c <= n and k <= n")
    if n - c < k:
        return 1.0
    denominator = math.comb(n, k)
    numerator = denominator - math.comb(n - c, k)
    return float(Fraction(numerator, denominator))


def mean_task_pass_at_k(outcomes: Sequence[Sequence[bool]], k: int) -> float:
    """Equal-source-task mean, retaining all-failed banks in the denominator.

    Each row contains the complete fixed bank for one independent source task.
    Invalid planned model attempts must already be encoded as False. An absent
    infrastructure artifact must be resolved under the study failure contract,
    rather than silently inserted, removed, or replaced here.
    """
    if not outcomes:
        raise ValueError("at least one source-task bank is required")
    estimates = []
    for row in outcomes:
        if any(type(value) is not bool for value in row):
            raise ValueError("each outcome must be an explicit bool")
        estimates.append(pass_at_k(len(row), sum(row), k))
    return math.fsum(estimates) / len(estimates)


@dataclass(frozen=True)
class PublicCandidate:
    """Public selection object; vote_key comes from the frozen public adapter.

    Candidate IDs identify sampling opportunities. Equal vote keys retain their
    multiplicity; repeated references to one candidate ID are a harness error.
    No correctness, gold answer, or hidden-test field exists here.
    """

    candidate_id: str
    vote_key: str | None
    valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a nonempty string")
        if type(self.valid) is not bool:
            raise ValueError("valid must be an explicit bool")
        if self.valid and (not isinstance(self.vote_key, str) or not self.vote_key):
            raise ValueError("valid candidates require a nonempty public vote key")
        if not self.valid and self.vote_key is not None:
            raise ValueError("invalid candidates must not receive a vote key")


@dataclass(frozen=True)
class Selection:
    candidate_id: str | None
    planned_count: int
    valid_count: int
    winning_vote_count: int | None
    tied_winning_classes: int | None


def _validate_pool(candidates: Sequence[PublicCandidate]) -> None:
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("duplicate candidate IDs would double-count an opportunity")


def _tie_key(source_id: str, candidate_id: str, tie_seed: bytes) -> bytes:
    if not isinstance(tie_seed, bytes) or not tie_seed:
        raise ValueError("tie_seed must be nonempty bytes")
    # Length prefixes avoid delimiter collisions; no answer or correctness enters.
    parts = [source_id.encode("utf-8"), candidate_id.encode("utf-8")]
    message = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
    return hmac.new(tie_seed, message, hashlib.sha256).digest()


def plurality_vote(
    candidates: Sequence[PublicCandidate], *, source_id: str, tie_seed: bytes
) -> Selection:
    """Preserve vote multiplicity and resolve class/representative ties blindly.

    Equal-size winning classes compete through a class-order hash. A separately
    derived blind order chooses the representative within the winning class.
    Input order, answer labels and any later gold mapping do not enter tie order.
    All invalid candidates yield no answer, not a removed evaluation item.
    """
    _validate_pool(candidates)
    groups: dict[str, list[PublicCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.valid:
            assert candidate.vote_key is not None
            groups[candidate.vote_key].append(candidate)
    valid_count = sum(map(len, groups.values()))
    if not groups:
        return Selection(None, len(candidates), 0, 0, 0)
    maximum = max(map(len, groups.values()))
    tied = [group for group in groups.values() if len(group) == maximum]
    class_seed = hmac.new(tie_seed, b"VOTE_CLASS_ORDER", hashlib.sha256).digest()
    representative_seed = hmac.new(tie_seed, b"VOTE_REPRESENTATIVE_ORDER", hashlib.sha256).digest()
    winning_group = min(
        tied,
        key=lambda group: min(
            (_tie_key(source_id, item.candidate_id, class_seed), item.candidate_id) for item in group
        ),
    )
    representative = min(
        winning_group,
        key=lambda item: (_tie_key(source_id, item.candidate_id, representative_seed), item.candidate_id),
    )
    return Selection(representative.candidate_id, len(candidates), valid_count, maximum, len(tied))


def judge_best(
    candidates: Sequence[PublicCandidate],
    public_scores: Mapping[str, float],
    *,
    source_id: str,
    tie_seed: bytes,
) -> Selection:
    """Select using already-acquired public scores, never authoritative labels.

    This does not call or validate a judge. The runtime must prove that scores
    were acquired from permitted inputs within budget before calling this helper.
    """
    _validate_pool(candidates)
    valid = [candidate for candidate in candidates if candidate.valid]
    if set(public_scores) != {candidate.candidate_id for candidate in valid}:
        raise ValueError("require exactly one public score per valid candidate")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1
           for value in public_scores.values()):
        raise ValueError("public scores must be finite numeric values in [0,1], not labels")
    if not valid:
        return Selection(None, len(candidates), 0, None, None)
    best_score = max(public_scores.values())
    chosen = min(
        (candidate for candidate in valid if public_scores[candidate.candidate_id] == best_score),
        key=lambda item: (_tie_key(source_id, item.candidate_id, tie_seed), item.candidate_id),
    )
    return Selection(chosen.candidate_id, len(candidates), len(valid), None, None)


@dataclass(frozen=True)
class EvaluatedPool:
    planned_count: int
    valid_count: int
    candidate_mean_correctness: float
    oracle_coverage: bool
    selected_correctness: bool
    selection_gap: int


def evaluate_sealed_selection(
    candidates: Sequence[PublicCandidate],
    selection: Selection,
    protected_correctness: Mapping[str, bool],
) -> EvaluatedPool:
    """Offline join after a candidate-restricted selection is sealed.

    Labels are required for every valid candidate, never supplied to selectors.
    All-invalid pools remain one failed item and all planned attempts remain in
    the candidate-mean denominator. Synthesis outputs need their own endpoint.
    """
    _validate_pool(candidates)
    if not candidates:
        raise ValueError("a planned opportunity pool must not be empty")
    valid_ids = {item.candidate_id for item in candidates if item.valid}
    if set(protected_correctness) != valid_ids:
        raise ValueError("require labels for exactly the valid candidate IDs")
    if any(type(value) is not bool for value in protected_correctness.values()):
        raise ValueError("protected correctness must be explicit bool values")
    if selection.planned_count != len(candidates) or selection.valid_count != len(valid_ids):
        raise ValueError("selection counts do not match the sealed pool")
    if selection.candidate_id is not None and selection.candidate_id not in valid_ids:
        raise ValueError("selected candidate is outside the eligible sealed pool")
    oracle = any(protected_correctness.values())
    selected = False if selection.candidate_id is None else protected_correctness[selection.candidate_id]
    return EvaluatedPool(
        planned_count=len(candidates),
        valid_count=len(valid_ids),
        candidate_mean_correctness=sum(protected_correctness.values()) / len(candidates),
        oracle_coverage=oracle,
        selected_correctness=selected,
        selection_gap=int(oracle) - int(selected),
    )


@dataclass(frozen=True)
class FrozenRequest:
    """Explicit comparison fields for reference alias checks, not a request hash.

    canonical_decoding and canonical_caps are exact bytes from the real frozen
    serializer. This module does not replace the study's RFC8785 request-ID code.
    No outer agent/process/budget label is needed unless it changes these fields.
    """

    source_id: str
    study_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_hash: str
    engine_digest: str
    input_utf8: bytes
    input_token_ids: tuple[int, ...]
    canonical_decoding: bytes
    semantic_seed: int
    canonical_caps: bytes
    precision: str
    hook_hash: str

    def __post_init__(self) -> None:
        for name in ("source_id", "study_id", "model_revision", "tokenizer_revision",
                     "chat_template_hash", "engine_digest", "precision", "hook_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be resolved before an exact alias check")
        for name in ("input_utf8", "canonical_decoding", "canonical_caps"):
            if not isinstance(getattr(self, name), bytes):
                raise ValueError(f"{name} must contain exact serialized bytes")
        if not isinstance(self.input_token_ids, tuple):
            raise ValueError("input_token_ids must be an immutable tuple")
        for token_id in self.input_token_ids:
            _integer("token_id", token_id)
        _integer("semantic_seed", self.semantic_seed)


@dataclass(frozen=True)
class BankEntry:
    logical_id: str
    request: FrozenRequest


def exact_bank_alias(
    canonical: Sequence[BankEntry], requested: Sequence[BankEntry]
) -> dict[str, str]:
    """Map alternate logical names only after exact ordered request equality.

    A neutral N=1 request cannot alias an informed-team prompt. Different seeds,
    caps, token IDs, hooks, precision, runtime or even one prompt byte also reject
    reuse. Sibling order is retained, so an arbitrary permutation is not a prefix.
    """
    if len(canonical) != len(requested):
        raise ValueError("banks must have the same ordered length")
    for bank in (canonical, requested):
        if len({entry.logical_id for entry in bank}) != len(bank):
            raise ValueError("logical bank entry IDs must be unique")
    if any(left.request != right.request for left, right in zip(canonical, requested)):
        raise ValueError("request identity mismatch: reuse is not exact")
    return {right.logical_id: left.logical_id for left, right in zip(canonical, requested)}
