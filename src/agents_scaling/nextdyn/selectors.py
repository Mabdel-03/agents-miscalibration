"""Sealed pools and the five selectors (spec 6.5, 15.1; plan §1.9, D9).

A *selection* is the study's answer to "given the team's five answers, which one ships?".  It
has to be decided from public material only, and it has to be decided **before** any grade is
joined -- ARD36's selector audit was joined afterwards and the gap between "the deployable
selector" and "the oracle" could not be read off the tables at all.  Three structures make
that mechanical here:

1.  :class:`SealedPool` is content-addressed.  Its ``pool_id`` hashes the candidates, so a
    pool cannot be re-opened, re-ordered or extended after a selector has run on it; the
    ``register:pool/<sha>`` dependency (``deps``) is what a SELECTION_AUDIT cell waits for.
2.  :class:`Candidate` carries no correctness field at all.  ``canonical_class`` is the
    gold-independent voting class the *grader-free* canonicalizer produced, ``valid`` is "an
    answer was extractable" (spec 6.1 validity is not correctness), and there is nowhere to
    put a label.  The only selector that sees labels is ``oracle``, and it receives them
    through an explicit :class:`AnalysisOnlyContext` that the runtime never constructs --
    which is what ``test_selector_service_cannot_read_private_grades`` checks as an import
    closure (this module imports neither ``grading`` nor any dataset module, R7).
3.  A selector that could not run is ``unexecuted``, never "0% accurate".  A missing score, an
    unfitted mapping or an absent analysis-only context all produce a :class:`Selection` with
    ``selected_id=None`` and a typed ``fallback_status``; the denominator is untouched.

Tie-breaking (D9).  ARD36 broke ties on the candidate id under a VOTE_TIE HMAC; that key was
correct but the *class* was chosen through a representative, so two arms that presented the
same classes in a different order could disagree.  Here :func:`tie_break` is a pure function of
``(source_item_id, canonical_class, selection_seed)``: no protocol, no agent order, no
confidence, no arrival time (``test_tie_break_is_invariant_to_protocol_and_agent_order``).

Producer of the ``selections`` table (``tables._PRODUCERS``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Protocol, Sequence, runtime_checkable

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.ids import CellKind, CompositeKey
from agents_scaling.nextdyn.tables import FALLBACK_STATUSES, SelectionRow
from agents_scaling.study import identity

# --------------------------------------------------------------------------- vocabulary

#: ``selections.fallback_status`` values, bound to the table's declaration (R1).
FALLBACK_NONE: Final[str] = "none"
FALLBACK_AUDIT: Final[str] = "audit_failed_fallback_plurality"
FALLBACK_UNEXECUTED: Final[str] = "unexecuted"
FALLBACK_NO_CANDIDATE: Final[str] = "no_valid_candidate"
if {FALLBACK_NONE, FALLBACK_AUDIT, FALLBACK_UNEXECUTED, FALLBACK_NO_CANDIDATE} != set(FALLBACK_STATUSES):
    raise RuntimeError("selector fallback statuses must be exactly the selections vocabulary (R1)")

#: Registry names.  ``study.yaml``'s ``selection.primary`` / ``secondary_deployed`` /
#: ``analysis_only`` / ``audit_failure_fallback`` all name entries of :data:`SELECTORS`.
CURRENT_ROUND_PLURALITY: Final[str] = "current_round_plurality"
RANDOM_VALID: Final[str] = "random_valid"
CONFIDENCE_WEIGHTED: Final[str] = "confidence_weighted"
POINTWISE_AUDIT: Final[str] = "pointwise_audit"
ORACLE: Final[str] = "oracle_analysis"

#: The gold-independent label the oracle is allowed to read, through an analysis-only context.
LABEL_CORRECT: Final[str] = "C"

SELECTOR_VERSION: Final[str] = "v1"
POOL_ID_PREFIX: Final[str] = "pool:"
POOL_ID_CHARS: Final[int] = 32

#: ``selection.tie_break`` in ``study.yaml`` names the rule this module implements.
TIE_BREAK_RULE: Final[str] = "item_class_hash"


# --------------------------------------------------------------------------- pools


@dataclass(frozen=True)
class Candidate:
    """One selectable answer.  No correctness field exists on this object, by construction."""

    candidate_id: str                  # the candidate's ``decisions.decision_id``
    generation_uid: str
    content_hash: str | None
    canonical_class: str | None        # gold-independent voting class; None when not gradeable
    valid: bool
    confidence: float | None           # never imputed; None stays None (spec 6.3)
    packet_hash: str
    slot: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ContractViolation("a candidate needs a non-empty candidate_id")
        if type(self.valid) is not bool:
            raise ContractViolation("Candidate.valid must be an explicit bool")
        if self.valid and not self.canonical_class:
            raise ContractViolation(
                f"{self.candidate_id}: a valid candidate has a canonical voting class")
        if not self.valid and self.canonical_class is not None:
            raise ContractViolation(
                f"{self.candidate_id}: an invalid candidate carries no voting class")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ContractViolation(f"confidence {self.confidence!r} is outside [0, 1]")


@dataclass(frozen=True)
class SealedPool:
    """The candidates of one selection event, frozen and content-addressed."""

    source_item_id: str
    episode_id: str
    cell_kind: CellKind
    round_: int
    scope: str
    candidates: tuple[Candidate, ...]
    protocol_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_kind", ids.coerce_kind(self.cell_kind))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.source_item_id or not self.episode_id:
            raise ContractViolation("a sealed pool names its item and its episode")
        if not self.candidates:
            raise ContractViolation("an empty pool is a planning error, never a selection event")
        seen = [c.candidate_id for c in self.candidates]
        if len(set(seen)) != len(seen):
            raise ContractViolation("a candidate id appears twice: one opportunity, one candidate")

    @property
    def pool_id(self) -> str:
        """Content-addressed over everything a selector may read (``register:pool/<sha>``).

        The candidate *order* is inside the hash on purpose: a pool that was re-ordered is a
        different pool, even though no selector's answer may depend on that order.
        """
        payload = [self.source_item_id, self.episode_id, self.cell_kind.value, int(self.round_),
                   self.scope, self.protocol_hash,
                   [[c.candidate_id, c.generation_uid, c.canonical_class, bool(c.valid),
                     c.packet_hash, int(c.slot)] for c in self.candidates]]
        return POOL_ID_PREFIX + identity.sha256_hex(identity.jcs(payload))[:POOL_ID_CHARS]

    @property
    def valid_candidates(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.valid)

    @property
    def unique_packet_hashes(self) -> tuple[str, ...]:
        """One entry per distinct displayed packet: what a generative selector draws on."""
        out: list[str] = []
        for candidate in self.candidates:
            if candidate.packet_hash and candidate.packet_hash not in out:
                out.append(candidate.packet_hash)
        return tuple(out)

    def candidate_of(self, candidate_id: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ContractViolation(f"{candidate_id} is not a member of {self.pool_id}")

    def audit_key(self, index: int, *, study_id: str, item_variant_id: str, audit_model_hash: str,
                  audit_budget_hash: str) -> CompositeKey:
        """The SELECTION_AUDIT composite key of one candidate of this pool (contract §2.4)."""
        return CompositeKey(
            study_id=study_id, cell_kind=CellKind.SELECTION_AUDIT,
            source_item_id=self.source_item_id, item_variant_id=item_variant_id,
            root_team_id=self.pool_id, protocol_hash=self.protocol_hash,
            receiver_model_hash=audit_model_hash, budget_hash=audit_budget_hash,
            replicate_id=0, round=int(self.round_), agent_slot=int(index), phase=K.PHASE_AUDIT)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Score:
    """A generative selector's parsed score for one candidate packet."""

    packet_hash: str
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ContractViolation("a score is a number, never a label")
        if not 0.0 <= float(self.value) <= 1.0:
            raise ContractViolation(f"p_correct {self.value!r} is outside [0, 1]")


@dataclass(frozen=True)
class Failure:
    """A generative selector's typed failure for one candidate packet (never a label)."""

    packet_hash: str
    reason: str


@dataclass(frozen=True)
class Selection:
    """What one selector did to one pool.  ``failed`` never means "chose wrongly"."""

    selector: str
    version: str
    pool_id: str
    selected_id: str | None
    selected_class: str | None
    tie_class_ids: tuple[str, ...]
    fallback_status: str
    cost: Mapping[str, float]
    failed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "tie_class_ids", tuple(self.tie_class_ids))
        object.__setattr__(self, "cost", dict(self.cost))
        if self.fallback_status not in FALLBACK_STATUSES:
            raise ContractViolation(f"unknown fallback_status {self.fallback_status!r}")

    @property
    def selection_id(self) -> str:
        """Carries ``<selector>.<version>`` so a second selector needs no id scheme of its own."""
        return f"{self.selector}.{self.version}|{self.pool_id[len(POOL_ID_PREFIX):][:12]}"

    @property
    def tie_broken(self) -> bool:
        return len(self.tie_class_ids) > 1


@dataclass(frozen=True)
class AnalysisOnlyContext:
    """The explicit envelope a label must travel in to reach a selector.

    Only ``oracle_analysis`` reads it and only the analysis builds it: the runtime selection
    service has no code path that constructs one (import-closure test 18.1).
    """

    labels: Mapping[str, str]          # candidate_id -> tri-state label, "C" when correct

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", dict(self.labels))


@dataclass(frozen=True)
class SelectionContext:
    """Everything a selector may read that is not in the pool itself."""

    study_seed: bytes
    selection_seed: bytes
    scores: Mapping[str, float] = field(default_factory=dict)      # packet_hash -> p_correct
    score_failures: Mapping[str, str] = field(default_factory=dict)
    confidence_weights: "ConfidenceWeights | None" = None
    analysis_only: AnalysisOnlyContext | None = None
    cost: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", dict(self.scores))
        object.__setattr__(self, "score_failures", dict(self.score_failures))
        object.__setattr__(self, "cost", dict(self.cost))
        if not self.selection_seed:
            raise ContractViolation("a selection needs its blind selection seed")


@dataclass(frozen=True)
class ConfidenceWeights:
    """The ``confidence_weighted`` policy, compiled from ``analysis.yaml`` (plan §1.9).

    ``mapping`` names the dev quantity a reported confidence is mapped through, ``fallback``
    the quantity used when the task's dev support is below ``min_task_support``, and
    ``missing_confidence_weight`` what a candidate with *no* reported confidence contributes.
    The last one is the delicate case: the spec forbids imputing a confidence, so the fallback
    weight is used for *ranking only* and the candidate's ``confidence`` stays ``None``
    everywhere it is recorded (``test_no_imputation``).
    """

    mapping: str
    fallback: str
    min_task_support: int
    missing_confidence_weight: str
    complete_confidence_sensitivity: bool
    task_rate: float | None = None         # the task-specific dev root success rate
    task_support: int = 0
    base_rate: float | None = None         # the global dev base rate
    complete_cases_only: bool = False      # the declared sensitivity arm

    @classmethod
    def from_profile(cls, profile: Any, *, task_rate: float | None = None, task_support: int = 0,
                     base_rate: float | None = None,
                     complete_cases_only: bool = False) -> "ConfidenceWeights":
        """Compile ``analysis.yaml``'s ``confidence_weighted`` block plus the fitted dev rates."""
        block = profile.confidence_weighted
        return cls(mapping=block.mapping, fallback=block.fallback,
                   min_task_support=int(block.min_task_support),
                   missing_confidence_weight=block.missing_confidence_weight,
                   complete_confidence_sensitivity=bool(block.complete_confidence_sensitivity),
                   task_rate=task_rate, task_support=int(task_support), base_rate=base_rate,
                   complete_cases_only=bool(complete_cases_only))

    @property
    def fitted(self) -> bool:
        """Is there any dev quantity to map through?  An unfitted selector is *unexecuted*."""
        return self.task_rate is not None or self.base_rate is not None

    def prior(self) -> float | None:
        """The dev rate in force: the task's own, or the global base rate below support."""
        if self.task_rate is not None and int(self.task_support) >= int(self.min_task_support):
            return float(self.task_rate)
        return None if self.base_rate is None else float(self.base_rate)

    def weight_of(self, confidence: float | None) -> float | None:
        """The ranking weight of one candidate.  ``None`` means "drop it from the ranking".

        A reported confidence is *mapped*, not used raw: models are badly calibrated and a
        raw 0.9 from a task the model solves 30% of the time is not worth a raw 0.9 from a
        task it solves 90% of the time.  The mapping is the monotone affine recentring of the
        reported value on the dev rate in force,

            w = prior + (confidence - 1/2) * 2 * min(prior, 1 - prior)   clipped to [0, 1],

        which is order-preserving within a pool (so the selector is still "trust the most
        confident"), anchors the mean on the declared dev quantity, and needs no fitted curve
        the pilot has not produced yet.  A candidate that reported **no** confidence keeps
        ``confidence = None`` everywhere it is recorded; here it contributes the fallback rate
        as a weight (or is dropped entirely under the declared complete-case sensitivity),
        which is a ranking decision and never a model statement.
        """
        prior = self.prior()
        if prior is None:
            return None
        if confidence is None:
            return None if self.complete_cases_only else prior
        spread = 2.0 * min(prior, 1.0 - prior)
        return min(1.0, max(0.0, prior + (float(confidence) - 0.5) * spread))


# --------------------------------------------------------------------------- tie-break


def tie_break(source_item_id: str, classes: Sequence[str], selection_seed: bytes) -> str:
    """The winning class among equal-sized classes: ``min sha256(JCS([item, class, seed]))``.

    Reads the item, the class string and the frozen selection seed -- nothing about the
    protocol, the agent order, the confidence or any later gold mapping (D9, spec 6.5).
    """
    names = sorted({str(c) for c in classes})
    if not names:
        raise ContractViolation("tie_break needs at least one class")
    if not selection_seed:
        raise ContractViolation("tie_break needs the frozen selection seed")
    return min(names, key=lambda cls: (_class_key(source_item_id, cls, selection_seed), cls))


def _class_key(source_item_id: str, canonical_class: str, selection_seed: bytes) -> str:
    return identity.sha256_hex(identity.jcs(
        [str(source_item_id), str(canonical_class), bytes(selection_seed).hex()]))


def _representative(pool: SealedPool, members: Sequence[Candidate], seed: bytes) -> Candidate:
    """Which member of the winning class ships: blind in the candidate id, order-invariant."""
    return min(members, key=lambda c: (_class_key(pool.source_item_id, c.candidate_id, seed),
                                       c.candidate_id))


def _no_valid_candidate(name: str, pool: SealedPool, cost: Mapping[str, float]) -> Selection:
    return Selection(selector=name, version=SELECTOR_VERSION, pool_id=pool.pool_id,
                     selected_id=None, selected_class=None, tie_class_ids=(),
                     fallback_status=FALLBACK_NO_CANDIDATE, cost=cost, failed=True)


def _unexecuted(name: str, pool: SealedPool, cost: Mapping[str, float]) -> Selection:
    return Selection(selector=name, version=SELECTOR_VERSION, pool_id=pool.pool_id,
                     selected_id=None, selected_class=None, tie_class_ids=(),
                     fallback_status=FALLBACK_UNEXECUTED, cost=cost, failed=True)


def _pick(name: str, pool: SealedPool, ctx: SelectionContext, *, ranked: Mapping[str, float],
          fallback_status: str = FALLBACK_NONE) -> Selection:
    """Choose the highest-ranked class, breaking equal ranks with :func:`tie_break`."""
    if not ranked:
        return _no_valid_candidate(name, pool, ctx.cost)
    best = max(ranked.values())
    tied = sorted(cls for cls, value in ranked.items() if value == best)
    winner = tie_break(pool.source_item_id, tied, ctx.selection_seed) if len(tied) > 1 else tied[0]
    members = [c for c in pool.valid_candidates if c.canonical_class == winner]
    chosen = _representative(pool, members, ctx.selection_seed)
    return Selection(selector=name, version=SELECTOR_VERSION, pool_id=pool.pool_id,
                     selected_id=chosen.candidate_id, selected_class=winner,
                     tie_class_ids=tuple(tied), fallback_status=fallback_status, cost=ctx.cost,
                     failed=False)


# --------------------------------------------------------------------------- the protocol


@runtime_checkable
class Selector(Protocol):
    """What every selector must be able to do (plan §1.9).

    ``needs_cells`` is the only thing the planner asks: a selector that needs generation is
    dispatched as a SELECTION_AUDIT cell and is reached through ``audit_requests`` /
    ``parse_score``; one that does not is computed in-process at materialize time.
    """

    name: str
    version: str
    deployable: bool
    needs_cells: bool

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection: ...

    def audit_requests(self, pool: SealedPool, ctx: Any) -> Mapping[str, Any]: ...

    def parse_score(self, record: Any) -> Score | Failure: ...


class _BaseSelector:
    """Shared refusals: a non-generative selector has no request builder and no parser."""

    name: str = ""
    version: str = SELECTOR_VERSION
    deployable: bool = True
    needs_cells: bool = False

    def audit_requests(self, pool: SealedPool, ctx: Any) -> Mapping[str, Any]:
        del pool, ctx
        raise ContractViolation(f"{self.name} is not a generative selector; it needs no cells")

    def parse_score(self, record: Any) -> Score | Failure:
        del record
        raise ContractViolation(f"{self.name} produces no scores to parse")


class CurrentRoundPlurality(_BaseSelector):
    """Spec 6.5 primary: plurality over the current round's canonical classes.

    Multiplicity is preserved (five identical answers are five votes) and an all-invalid pool
    yields ``no_valid_candidate`` rather than a silent drop -- the opportunity stays in the
    denominator and is scored incorrect downstream.
    """

    name = CURRENT_ROUND_PLURALITY

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection:
        counts: dict[str, float] = {}
        for candidate in pool.valid_candidates:
            counts[str(candidate.canonical_class)] = counts.get(str(candidate.canonical_class), 0.0) + 1.0
        return _pick(self.name, pool, ctx, ranked=counts)


class RandomValid(_BaseSelector):
    """A blind draw among the valid candidates: the "is voting doing anything?" baseline."""

    name = RANDOM_VALID

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection:
        valid = pool.valid_candidates
        if not valid:
            return _no_valid_candidate(self.name, pool, ctx.cost)
        chosen = _representative(pool, valid, ctx.selection_seed)
        return Selection(selector=self.name, version=self.version, pool_id=pool.pool_id,
                         selected_id=chosen.candidate_id, selected_class=chosen.canonical_class,
                         tie_class_ids=(), fallback_status=FALLBACK_NONE, cost=ctx.cost,
                         failed=False)


class ConfidenceWeighted(_BaseSelector):
    """Plurality weighted by the *mapped* reported confidence (plan §1.9).

    An unfitted mapping (no dev rate at all) is ``unexecuted``: the selector could not run, and
    reporting 0% accuracy for it would be a measurement of the harness, not of the selector.

    So is an empty ranking under the declared complete-confidence sensitivity.  There
    ``weight_of(None)`` drops every candidate that reported no confidence, and a pool where
    *nobody* reported one leaves nothing to rank -- but the pool is not empty and its
    candidates are valid, so ``no_valid_candidate`` would be a false statement about the pool
    (spec 4.6: validity is answer extractability) and, being scored incorrect downstream,
    would charge this selector with a wrong answer on every such item.  "Could not run here"
    is ``unexecuted``, which keeps the opportunity out of the selector's numerator *and* its
    denominator.  An all-invalid pool is still ``no_valid_candidate``.
    """

    name = CONFIDENCE_WEIGHTED

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection:
        weights = ctx.confidence_weights
        if weights is None or not weights.fitted:
            return _unexecuted(self.name, pool, ctx.cost)
        ranked: dict[str, float] = {}
        for candidate in pool.valid_candidates:
            weight = weights.weight_of(candidate.confidence)
            if weight is None:
                continue
            cls = str(candidate.canonical_class)
            ranked[cls] = ranked.get(cls, 0.0) + float(weight)
        if not ranked:
            return (_no_valid_candidate(self.name, pool, ctx.cost) if not pool.valid_candidates
                    else _unexecuted(self.name, pool, ctx.cost))
        return _pick(self.name, pool, ctx, ranked=ranked)


class PointwiseAudit(_BaseSelector):
    """The generative verifier (spec 6.5): one ``p_correct`` per unique displayed packet.

    Ties in score fall through :func:`tie_break` like any other tie.  A packet whose score
    failed to parse is *not* given a score of zero: the whole selection falls back to the
    preregistered current-round vote and records ``audit_failed_fallback_plurality``, which is
    an execution status and never a semantic error label.
    """

    name = POINTWISE_AUDIT
    needs_cells = True

    def __init__(self, *, request_builder: Callable[[SealedPool, Any], Mapping[str, Any]] | None = None,
                 parser: Callable[[Any], Score | Failure] | None = None) -> None:
        # Every generative selector owns its own request builder and parser (plan §1.9); the
        # concrete ones live in ``audit.py`` and are injected so this module stays free of the
        # prompt and inference imports (and therefore of their closures).
        self._request_builder = request_builder
        self._parser = parser

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection:
        valid = pool.valid_candidates
        if not valid:
            return _no_valid_candidate(self.name, pool, ctx.cost)
        missing = [c for c in valid if c.packet_hash not in ctx.scores]
        if missing:
            if not ctx.scores and not ctx.score_failures:
                return _unexecuted(self.name, pool, ctx.cost)
            fallback = SELECTORS[CURRENT_ROUND_PLURALITY].select(pool, ctx)
            return Selection(selector=self.name, version=self.version, pool_id=pool.pool_id,
                             selected_id=fallback.selected_id,
                             selected_class=fallback.selected_class,
                             tie_class_ids=fallback.tie_class_ids,
                             fallback_status=FALLBACK_AUDIT, cost=ctx.cost,
                             failed=fallback.failed)
        ranked: dict[str, float] = {}
        for candidate in valid:
            cls = str(candidate.canonical_class)
            ranked[cls] = max(ranked.get(cls, 0.0), float(ctx.scores[candidate.packet_hash]))
        return _pick(self.name, pool, ctx, ranked=ranked)

    def audit_requests(self, pool: SealedPool, ctx: Any) -> Mapping[str, Any]:
        if self._request_builder is None:
            raise ContractViolation(
                f"{self.name} has no request builder bound; nextdyn.audit binds one when the "
                "SELECTION_AUDIT cell runs")
        return self._request_builder(pool, ctx)

    def parse_score(self, record: Any) -> Score | Failure:
        if self._parser is None:
            raise ContractViolation(f"{self.name} has no score parser bound")
        return self._parser(record)


class OracleAnalysis(_BaseSelector):
    """The analysis-only ceiling: ships a correct candidate whenever the pool holds one.

    ``deployable=False`` and the labels arrive only through :class:`AnalysisOnlyContext`, so a
    deployed-selector list that named it is a compile error (``contract.NON_DEPLOYABLE_SELECTORS``)
    and a runtime call without the envelope is ``unexecuted``, not a silent gold read.
    """

    name = ORACLE
    deployable = False

    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection:
        envelope = ctx.analysis_only
        if envelope is None:
            return _unexecuted(self.name, pool, ctx.cost)
        valid = pool.valid_candidates
        if not valid:
            return _no_valid_candidate(self.name, pool, ctx.cost)
        correct = [c for c in valid if envelope.labels.get(c.candidate_id) == LABEL_CORRECT]
        members = correct or list(valid)
        chosen = _representative(pool, members, ctx.selection_seed)
        return Selection(selector=self.name, version=self.version, pool_id=pool.pool_id,
                         selected_id=chosen.candidate_id, selected_class=chosen.canonical_class,
                         tie_class_ids=(), fallback_status=FALLBACK_NONE, cost=ctx.cost,
                         failed=False)


#: The registry.  A new selector is one class plus one entry here (plan §1.10); ``audit.py``
#: dispatches every entry whose ``needs_cells`` is true without being edited.
SELECTORS: Final[Mapping[str, Selector]] = {
    CURRENT_ROUND_PLURALITY: CurrentRoundPlurality(),
    RANDOM_VALID: RandomValid(),
    CONFIDENCE_WEIGHTED: ConfidenceWeighted(),
    POINTWISE_AUDIT: PointwiseAudit(),
    ORACLE: OracleAnalysis(),
}


def selector_for(name: str) -> Selector:
    try:
        return SELECTORS[name]
    except KeyError as exc:
        raise ContractViolation(f"unknown selector {name!r}; registered: {sorted(SELECTORS)}") from exc


def deployed_selectors(selection: K.SelectionSpec) -> tuple[str, ...]:
    """The selectors ``study.yaml`` deploys: the primary first, then the secondaries.

    Refuses a non-deployable entry here as well as at compile time, because this is the list
    the runtime actually iterates.
    """
    names = (selection.primary, *selection.secondary_deployed)
    for name in names:
        if not selector_for(name).deployable:
            raise LeakageError(f"{name} is analysis-only and may never be deployed (spec 17.4)")
    if selection.tie_break != TIE_BREAK_RULE:
        raise ContractViolation(
            f"selection.tie_break is {selection.tie_break!r}; this module implements "
            f"{TIE_BREAK_RULE!r} (min sha256(JCS([source_item_id, canonical_class, selection_seed])))")
    if selector_for(selection.audit_failure_fallback).needs_cells:
        raise ContractViolation("the audit failure fallback must be a selector that needs no cells")
    return names


def deployed_for(contract: K.StudyContract) -> tuple[Selector, ...]:
    """The selector objects this study deploys, in declared order (``StudyContract.selection``).

    The runtime iterates *objects*, not names, so a deployed list that named an analysis-only
    selector fails here as well as at compile time -- the two checks are the same function.
    """
    return tuple(selector_for(name) for name in deployed_selectors(contract.selection))


def score_all(pool: SealedPool, ctx: SelectionContext, *,
              selectors: Mapping[str, Selector] | None = None) -> tuple[Selection, ...]:
    """Every registered selector applied to one sealed pool, in registry order."""
    registry = SELECTORS if selectors is None else selectors
    return tuple(registry[name].select(pool, ctx) for name in registry)


# --------------------------------------------------------------------------- rows


def selection_row(pool: SealedPool, selection: Selection, *, audit_cell_id: str | None = None,
                  cost_tokens: int = 0, cost_flops: int | None = None) -> SelectionRow:
    """The one ``selections`` row of one (selector, pool).  Grades are joined afterwards."""
    return SelectionRow(
        selection_id=selection.selection_id,
        episode_id=pool.episode_id,
        cell_kind=pool.cell_kind.value,
        source_item_id=pool.source_item_id,
        round=int(pool.round_),
        selector=selection.selector,
        selector_version=selection.version,
        pool_id=pool.pool_id,
        pool_scope=pool.scope,
        candidate_decision_ids_json=json.dumps([c.candidate_id for c in pool.candidates]),
        valid_count=len(pool.valid_candidates),
        selected_decision_id=selection.selected_id,
        selected_class=selection.selected_class,
        tied_class_ids_json=json.dumps(list(selection.tie_class_ids)),
        tie_broken=selection.tie_broken,
        fallback_status=selection.fallback_status,
        audit_cell_id=audit_cell_id,
        cost_tokens=int(cost_tokens),
        cost_flops=cost_flops,
    )


__all__ = [
    "AnalysisOnlyContext", "CONFIDENCE_WEIGHTED", "CURRENT_ROUND_PLURALITY", "Candidate",
    "ConfidenceWeighted", "ConfidenceWeights", "CurrentRoundPlurality", "FALLBACK_AUDIT",
    "FALLBACK_NONE", "FALLBACK_NO_CANDIDATE", "FALLBACK_UNEXECUTED", "Failure", "ORACLE",
    "OracleAnalysis", "POINTWISE_AUDIT", "PointwiseAudit", "RANDOM_VALID", "RandomValid",
    "SELECTORS", "SELECTOR_VERSION", "Score", "SealedPool", "Selection", "SelectionContext",
    "Selector", "TIE_BREAK_RULE", "deployed_for", "deployed_selectors", "score_all",
    "selection_row", "selector_for", "tie_break",
]
