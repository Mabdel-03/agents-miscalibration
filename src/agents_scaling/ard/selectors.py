"""ARD36 static selection competitors and the set-wise LLM audit (WP5).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §12.4 ("Static
selection competitors"), §7 ("End-of-round selection"); plan §5.8 ("Selected correctness is
long-form over selector x pool").

Spec §12.4, in full: *"Using frozen generated pools, evaluate root/current/final candidate
selection via plurality, confidence-weighted plurality, CURRENT, and a dynamics-only static
selector.  All are trained/calibrated only on training/validation items.  Archived
candidates are available to every archive selector.  Budget a small set-wise LLM selector
audit on held-out root, PEER-final, and GATE-final pools: 240x3 = 720 additional logical 8B
calls, with the same 1,024+256 cap.  Return a candidate ID only; do not allow a newly
synthesized answer.  Candidate order is randomized.  This is a secondary diagnostic, not the
common primary selector."*

One sealed pool, five selectors
-------------------------------

The decisive comparison of §12.4 — *"whether changing trajectories adds value beyond
selecting more intelligently from already generated candidates"* — is only meaningful if
every selector ranks the **same** candidates under the **same** eligibility rule.  So the
unit here is the :class:`SealedPool`: an ordered, content-addressed set of candidates with

* one :data:`POOL_ID_PREFIX`-prefixed :attr:`SealedPool.pool_id`, hashed over the pool's
  content and *nothing selector-specific*, so the ``plurality`` / ``confidence_weighted`` /
  ``CURRENT`` / ``dynamics_only`` / ``llm_audit`` rows of one (item, round, pool) join on
  one key; and
* one validity convention, :data:`VALIDITY_CONVENTION` — a candidate may be selected iff it
  is structurally valid *and* carries a canonical class, which is exactly
  ``vote.VoteCandidate.keyed``.  Spec §7: "Invalid answers cannot form a winning content
  class; if all outputs are invalid, selection fails."  The same rule binds the model
  selector: the audit is only ever shown eligible candidates, so a "clever" audit cannot
  win by resurrecting an unparseable report that plurality was forbidden to count.

``plurality`` and ``confidence_weighted`` delegate to :mod:`agents_scaling.ard.vote`
verbatim (one implementation of the frozen ``VOTE_TIE`` blind tie-break in the study).
``CURRENT`` and ``dynamics_only`` score the frozen v1 feature row of each candidate through
the frozen controller's matching heads — the two heads
:mod:`agents_scaling.ard.controller` fits under ``HEAD_CURRENT`` and ``HEAD_DYNAMICS``.
Features are *attached to the candidates by the caller* rather than re-derived here: the
frozen builder is ``features.features_from_state`` and needs a ``RoundState``, so a caller
holding only ``decisions.parquet`` rows gets the two model rows back with
``status="features_unavailable"`` and a null selection instead of a second, drifting feature
implementation.

The audit returns an ID, never an answer
----------------------------------------

:func:`parse_audit_choice` accepts **only** an exact presented label.  Anything else — a
residue, a restated answer, a candidate that was not presented, prose — is refused with a
status (:data:`AUDIT_SYNTHESIZED`, :data:`AUDIT_UNKNOWN_LABEL`, :data:`AUDIT_REFUSED`) and
the row records ``selection_failed``.  There is no path by which the audit's *text* becomes
an answer, which is what "do not allow a newly synthesized answer" has to mean mechanically.

Candidate order is randomized by the frozen correctness-blind HMAC
``identity.blind_order_key(study_seed, contract.ORDER_PACKET, pool_id, index)`` — the same
namespace WP2 uses for peer-packet delivery order, because this is the same kind of
decision (in what order does a model see a set of candidate reports) and it must be
independent of validity, confidence and any later gold mapping.  The realized order is
recorded on :class:`AuditPresentation`, so the audit's position effects are analysable.

Nothing in this module reads a gold answer, a grade or an item id as a feature; the
module-level import closure never reaches :mod:`agents_scaling.ard.graders` (contract §0.2).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.ard import vote as VOTE
from agents_scaling.study import identity
from agents_scaling.study.types import ProtocolError

__all__ = [
    "AUDIT_LABEL_PREFIX",
    "AUDIT_OK",
    "AUDIT_REFUSED",
    "AUDIT_STATUSES",
    "AUDIT_SYNTHESIZED",
    "AUDIT_UNKNOWN_LABEL",
    "AuditOutcome",
    "AuditPresentation",
    "POOL_ID_PREFIX",
    "SELECTOR_CONFIDENCE",
    "SELECTOR_CURRENT",
    "SELECTOR_DYNAMICS",
    "SELECTOR_LLM_AUDIT",
    "SELECTOR_PLURALITY",
    "STATUS_AUDIT_NOT_RUN",
    "STATUS_EMPTY_POOL",
    "STATUS_FEATURES_UNAVAILABLE",
    "STATUS_NO_CONTROLLER",
    "STATUS_NO_KEYED_CANDIDATE",
    "STATUS_OK",
    "SealedPool",
    "SelectorCandidate",
    "SelectorError",
    "SelectorRow",
    "VALIDITY_CONVENTION",
    "audit_order",
    "audit_presentation",
    "collect_pool",
    "parse_audit_choice",
    "render_audit_prompt",
    "run_selector_audit_cell",
    "score_all",
    "score_selector",
    "sealed_pool",
]

SELECTOR_PLURALITY: Final[str] = VOTE.SELECTOR_PLURALITY          # "plurality"
SELECTOR_CONFIDENCE: Final[str] = VOTE.SELECTOR_CONFIDENCE        # "confidence_weighted"
SELECTOR_CURRENT: Final[str] = "CURRENT"
SELECTOR_DYNAMICS: Final[str] = "dynamics_only"
SELECTOR_LLM_AUDIT: Final[str] = "llm_audit"

#: The five registered selectors, in ``contract.SELECTORS`` order.
SELECTOR_ORDER: Final[tuple[str, ...]] = K.SELECTORS

if set(SELECTOR_ORDER) != {SELECTOR_PLURALITY, SELECTOR_CONFIDENCE, SELECTOR_CURRENT,
                           SELECTOR_DYNAMICS, SELECTOR_LLM_AUDIT}:  # pragma: no cover
    raise RuntimeError("ard/selectors.py and contract.SELECTORS disagree about the registered set")

#: ``pool:<sha256>`` — the join key every selector row of one pool carries.
POOL_ID_PREFIX: Final[str] = "pool:"
POOL_SCHEMA: Final[str] = "ard_selector_pool_v1"

#: Spec §7's eligibility rule, named once so every row can state which rule it used.
VALIDITY_CONVENTION: Final[str] = "keyed"   # structurally valid AND carrying a canonical class

STATUS_OK: Final[str] = "ok"
STATUS_EMPTY_POOL: Final[str] = "empty_pool"
STATUS_NO_KEYED_CANDIDATE: Final[str] = "no_keyed_candidate"
STATUS_NO_CONTROLLER: Final[str] = "no_controller"
STATUS_FEATURES_UNAVAILABLE: Final[str] = "features_unavailable"
STATUS_AUDIT_NOT_RUN: Final[str] = "audit_not_run"

AUDIT_LABEL_PREFIX: Final[str] = "CANDIDATE-"
AUDIT_OK: Final[str] = "ok"
AUDIT_UNKNOWN_LABEL: Final[str] = "unknown_label"
AUDIT_SYNTHESIZED: Final[str] = "synthesized_answer"
AUDIT_REFUSED: Final[str] = "refused"
AUDIT_STATUSES: Final[tuple[str, ...]] = (
    AUDIT_OK, AUDIT_UNKNOWN_LABEL, AUDIT_SYNTHESIZED, AUDIT_REFUSED,
)

#: Pool kinds the audit runs over (spec §12.4: held-out root, PEER-final, GATE-final).
AUDIT_POOLS: Final[tuple[str, ...]] = K.SELECTOR_AUDIT_POOLS


class SelectorError(ProtocolError):
    """A selection was requested over a pool or an artifact that cannot support it."""


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class SelectorCandidate:
    """One already-generated candidate in a sealed pool.

    ``features`` is the frozen v1 row of :data:`agents_scaling.ard.features.FEATURES_V1`
    for this candidate, or ``None`` when the caller holds no ``RoundState`` to build one;
    the model selectors then report ``features_unavailable`` rather than scoring a guess.
    ``valid`` / ``canonical_class`` carry the single :data:`VALIDITY_CONVENTION`.
    """

    candidate_id: str
    answer: str | None
    canonical_class: str | None
    valid: bool = True
    confidence: float | None = None
    content_sha256: str = ""
    evidence: str | None = None
    agent_slot: int = -1
    round_: int = 0
    features: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} is outside [0, 1]")

    @property
    def keyed(self) -> bool:
        """:data:`VALIDITY_CONVENTION` — identical to ``vote.VoteCandidate.keyed``."""
        return bool(self.valid) and isinstance(self.canonical_class, str) and bool(self.canonical_class)

    def as_vote_candidate(self) -> VOTE.VoteCandidate:
        return VOTE.VoteCandidate(
            candidate_id=self.candidate_id,
            canonical_class=self.canonical_class,
            valid=bool(self.valid),
            confidence=self.confidence,
        )

    def identity_object(self) -> dict[str, Any]:
        """The part of a candidate that enters the pool id (content, never a score)."""
        return {
            "candidate_id": str(self.candidate_id),
            "answer": self.answer,
            "canonical_class": self.canonical_class,
            "valid": bool(self.valid),
            "keyed": bool(self.keyed),
            "content_sha256": str(self.content_sha256),
            "agent_slot": int(self.agent_slot),
            "round": int(self.round_),
        }


# --------------------------------------------------------------------------- the pool


@dataclass(frozen=True)
class SealedPool:
    """A frozen candidate pool: one id, one validity convention, five selectors.

    The pool is *sealed* in the sense of spec §12.4 ("using frozen generated pools"): the
    candidate order is the stored order, the id is a content hash, and nothing a selector
    does can change either.  Two selectors that disagree about which candidates were
    available would make the §12.4 comparison meaningless, so the id is the guard.
    """

    item_id: str
    pool_kind: str
    round_: int
    candidates: tuple[SelectorCandidate, ...]
    checkpoint: str = K.CHECKPOINT_SMALL
    protocol: str = "NONE"
    cohort: str = K.Cohort.NATURAL.value
    episode_ids: tuple[str, ...] = ()
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "episode_ids", tuple(self.episode_ids))
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise SelectorError(
                f"pool {self.item_id}|{self.pool_kind} lists a candidate twice; one planned "
                "opportunity would be double-counted (spec §3)")
        if not self.source_id:
            object.__setattr__(self, "source_id", f"{self.item_id}|{self.pool_kind}|r{int(self.round_)}")

    # -- identity -------------------------------------------------------------------
    def identity_object(self) -> dict[str, Any]:
        return {
            "schema": POOL_SCHEMA,
            "study_id": K.STUDY_ID,
            "contract_version": int(K.CONTRACT_VERSION),
            "item_id": str(self.item_id),
            "pool_kind": str(self.pool_kind),
            "round": int(self.round_),
            "checkpoint": str(self.checkpoint),
            "protocol": str(self.protocol),
            "cohort": str(self.cohort),
            "validity_convention": VALIDITY_CONVENTION,
            "candidates": [c.identity_object() for c in self.candidates],
        }

    @property
    def pool_id(self) -> str:
        """``pool:<sha>`` — shared by every selector row scored on this pool."""
        return POOL_ID_PREFIX + identity.sha256_hex(identity.jcs(self.identity_object()))

    # -- views ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def eligible(self) -> tuple[SelectorCandidate, ...]:
        """The candidates :data:`VALIDITY_CONVENTION` admits, in stored order."""
        return tuple(c for c in self.candidates if c.keyed)

    @property
    def selection_failed(self) -> bool:
        return not self.eligible

    def vote_pool(self) -> list[VOTE.VoteCandidate]:
        return [c.as_vote_candidate() for c in self.candidates]

    def by_id(self, candidate_id: str | None) -> SelectorCandidate | None:
        if candidate_id is None:
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None


def sealed_pool(
    item_id: str,
    pool_kind: str,
    candidates: Sequence[SelectorCandidate],
    *,
    round_: int = 0,
    **kwargs: Any,
) -> SealedPool:
    """:class:`SealedPool` with the pool kind checked against the frozen vocabularies."""
    known = set(K.POOL_KINDS) | set(AUDIT_POOLS)
    if str(pool_kind) not in known:
        raise SelectorError(f"unknown pool kind {pool_kind!r}; frozen kinds are {sorted(known)}")
    return SealedPool(item_id=str(item_id), pool_kind=str(pool_kind), round_=int(round_),
                      candidates=tuple(candidates), **kwargs)


# --------------------------------------------------------------------------- the row


@dataclass(frozen=True)
class SelectorRow:
    """One (selector x pool) result — the long-form row of plan §5.8.

    Every row of one pool carries the same ``pool_id``, ``pool_candidate_ids`` and
    ``validity_convention``, which is what makes the five selectors comparable.
    """

    selector: str
    pool_id: str
    pool_kind: str
    item_id: str
    round_: int
    checkpoint: str
    protocol: str
    pool_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    selected_class: str | None
    selected_answer: str | None
    planned_count: int
    valid_count: int
    selection_failed: bool
    status: str = STATUS_OK
    detail: str = ""
    scores: Mapping[str, float] = field(default_factory=dict)
    tied_winning_classes: tuple[str, ...] = ()
    validity_convention: str = VALIDITY_CONVENTION

    def to_row(self) -> dict[str, Any]:
        return {
            "event": "selector_row",
            "selector": self.selector,
            "pool_id": self.pool_id,
            "pool": self.pool_kind,
            "item_id": self.item_id,
            "round": int(self.round_),
            "checkpoint": self.checkpoint,
            "protocol": self.protocol,
            "pool_candidate_ids": list(self.pool_candidate_ids),
            "pool_size": len(self.pool_candidate_ids),
            "selected_decision_id": self.selected_candidate_id,
            "selected_class": self.selected_class,
            "selected_answer": self.selected_answer,
            "planned_count": int(self.planned_count),
            "valid_count": int(self.valid_count),
            "selection_failed": bool(self.selection_failed),
            "status": self.status,
            "detail": self.detail,
            "scores": {str(k): float(v) for k, v in dict(self.scores).items()},
            "tied_winning_classes": list(self.tied_winning_classes),
            "validity_convention": self.validity_convention,
        }


def _row(
    selector: str,
    pool: SealedPool,
    chosen: SelectorCandidate | None,
    *,
    status: str = STATUS_OK,
    detail: str = "",
    scores: Mapping[str, float] | None = None,
    tied: Sequence[str] = (),
) -> SelectorRow:
    return SelectorRow(
        selector=str(selector),
        pool_id=pool.pool_id,
        pool_kind=pool.pool_kind,
        item_id=pool.item_id,
        round_=int(pool.round_),
        checkpoint=pool.checkpoint,
        protocol=pool.protocol,
        pool_candidate_ids=tuple(c.candidate_id for c in pool.candidates),
        selected_candidate_id=None if chosen is None else chosen.candidate_id,
        selected_class=None if chosen is None else chosen.canonical_class,
        selected_answer=None if chosen is None else chosen.answer,
        planned_count=len(pool.candidates),
        valid_count=len(pool.eligible),
        selection_failed=chosen is None,
        status=status,
        detail=detail,
        scores=dict(scores or {}),
        tied_winning_classes=tuple(tied),
    )


# --------------------------------------------------------------------------- the audit


@dataclass(frozen=True)
class AuditPresentation:
    """The randomized, recorded presentation of one pool to the audit model.

    ``order`` maps presented position -> index into ``pool.eligible``; ``labels`` are the
    opaque ids the model may return.  Both are recorded so a position effect is analysable
    and so the model's answer can be mapped back without trusting its text.
    """

    pool_id: str
    item_id: str
    pool_kind: str
    order: tuple[int, ...]
    labels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    body: str

    def candidate_for(self, label: str) -> str | None:
        for name, candidate_id in zip(self.labels, self.candidate_ids):
            if name == label:
                return candidate_id
        return None

    def to_row(self) -> dict[str, Any]:
        return {
            "event": "selector_audit_presentation",
            "pool_id": self.pool_id,
            "item_id": self.item_id,
            "pool": self.pool_kind,
            "order": list(self.order),
            "labels": list(self.labels),
            "candidate_ids": list(self.candidate_ids),
            "n_presented": len(self.labels),
        }


@dataclass(frozen=True)
class AuditOutcome:
    """What the audit model returned, mapped back to a candidate id — or refused."""

    status: str
    label: str | None
    candidate_id: str | None
    raw: str
    pool_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == AUDIT_OK and self.candidate_id is not None

    def to_row(self) -> dict[str, Any]:
        return {
            "event": "selector_audit_choice",
            "pool_id": self.pool_id,
            "status": self.status,
            "label": self.label,
            "candidate_id": self.candidate_id,
            "raw": self.raw,
            "returned_candidate_id_only": self.status in (AUDIT_OK, AUDIT_UNKNOWN_LABEL),
        }


def audit_order(pool: SealedPool, *, study_seed: bytes) -> tuple[int, ...]:
    """Correctness-blind presentation order over ``pool.eligible`` (spec §12.4).

    ``identity.blind_order_key(study_seed, contract.ORDER_PACKET, pool_id, index)`` — the
    frozen namespace WP2 uses for peer-packet delivery order.  The key reads only the pool
    id and the position, never the answer, the confidence or the validity, so the order
    cannot correlate with correctness.
    """
    n = len(pool.eligible)
    pool_id = pool.pool_id
    return tuple(sorted(
        range(n),
        key=lambda index: identity.blind_order_key(
            bytes(study_seed), K.ORDER_PACKET, pool_id, int(index)),
    ))


def audit_presentation(pool: SealedPool, *, study_seed: bytes) -> AuditPresentation:
    """Build the randomized candidate block the audit prompt shows."""
    eligible = pool.eligible
    order = audit_order(pool, study_seed=study_seed)
    labels = tuple(f"{AUDIT_LABEL_PREFIX}{position + 1}" for position in range(len(order)))
    candidate_ids = tuple(eligible[index].candidate_id for index in order)
    blocks: list[str] = []
    for label, index in zip(labels, order):
        candidate = eligible[index]
        blocks.append(f"{label}")
        blocks.append(f"  answer: {candidate.answer or ''}")
        blocks.append(f"  evidence: {candidate.evidence or ''}")
    return AuditPresentation(
        pool_id=pool.pool_id, item_id=pool.item_id, pool_kind=pool.pool_kind,
        order=order, labels=labels, candidate_ids=candidate_ids, body="\n".join(blocks),
    )


_AUDIT_INSTRUCTION = (
    "You are auditing a set of candidate answers that were already produced by other "
    "solvers. Choose the one candidate most likely to be correct."
)


def audit_output_contract(labels: Sequence[str]) -> str:
    """The frozen audit output contract: an id from the presented set, nothing else."""
    shown = ", ".join(labels) if labels else "(none)"
    return (
        "Output contract: reply with exactly these three tags and nothing else.\n"
        f"{K.FINAL_OPEN}one candidate id, copied exactly from this list: {shown}"
        f"{K.FINAL_CLOSE}\n"
        f"{K.EVIDENCE_OPEN}one or two concise, checkable reasons, at most "
        f"{K.EVIDENCE_MAX_TOKENS} tokens{K.EVIDENCE_CLOSE}\n"
        f"{K.CONFIDENCE_OPEN}probability from 0 to 1{K.CONFIDENCE_CLOSE}\n"
        f"{K.FINAL_OPEN} must contain a candidate id. Do not write a new answer, do not "
        "edit a candidate, and do not name a candidate that is not listed. The whole "
        f"report is at most {K.FINAL_TOKENS} tokens."
    )


def render_audit_prompt(task_text: str, presentation: AuditPresentation) -> tuple[dict[str, str], ...]:
    """The one audit prompt (spec §12.4: set-wise, id-only, randomized order)."""
    blocks = [
        _AUDIT_INSTRUCTION,
        "",
        f"Task: {task_text}",
        "",
        "Candidates:",
        presentation.body or "(no eligible candidate)",
        "",
        audit_output_contract(presentation.labels),
    ]
    return ({"role": "user", "content": "\n".join(blocks)},)


_LABEL_RE = re.compile(rf"{re.escape(AUDIT_LABEL_PREFIX)}\d+", re.IGNORECASE)


def parse_audit_choice(report: Any, presentation: AuditPresentation) -> AuditOutcome:
    """Map the audit report onto a presented candidate id, or refuse it.

    Spec §12.4: "Return a candidate ID only; do not allow a newly synthesized answer."
    The only accepted content is one presented label.  A report naming a label that was
    not presented is :data:`AUDIT_UNKNOWN_LABEL`; anything else that parsed — a residue, a
    restated answer, prose — is :data:`AUDIT_SYNTHESIZED` and is *discarded*, never used as
    an answer; an unparseable report is :data:`AUDIT_REFUSED`.  In every non-``ok`` case
    the selection fails, exactly as an all-invalid pool fails under spec §7.
    """
    raw = "" if report is None else str(getattr(report, "final_answer", None) or "")
    if not getattr(report, "valid", False) or not raw.strip():
        return AuditOutcome(AUDIT_REFUSED, None, None, raw, presentation.pool_id)
    text = raw.strip()
    upper = text.upper()
    for label in presentation.labels:
        if upper == label.upper():
            return AuditOutcome(AUDIT_OK, label, presentation.candidate_for(label), raw,
                                presentation.pool_id)
    match = _LABEL_RE.search(upper)
    if match and match.group(0) == upper:
        # Well-formed id, but not one of the presented candidates.
        return AuditOutcome(AUDIT_UNKNOWN_LABEL, match.group(0), None, raw, presentation.pool_id)
    if match:
        # A label embedded in other text: the contract asked for the id alone, and
        # accepting the surrounding text would be accepting a synthesized answer.
        return AuditOutcome(AUDIT_SYNTHESIZED, match.group(0), None, raw, presentation.pool_id)
    return AuditOutcome(AUDIT_SYNTHESIZED, None, None, raw, presentation.pool_id)


# --------------------------------------------------------------------------- scoring


def _model_row(
    selector: str,
    head: str,
    pool: SealedPool,
    controller: Any | None,
) -> SelectorRow:
    """``CURRENT`` / ``dynamics_only``: argmax of the frozen head over eligible candidates."""
    eligible = pool.eligible
    if not eligible:
        return _row(selector, pool, None, status=STATUS_NO_KEYED_CANDIDATE,
                    detail="no candidate satisfies the pool's validity convention")
    if controller is None:
        return _row(selector, pool, None, status=STATUS_NO_CONTROLLER,
                    detail=f"the {selector} selector needs a frozen controller with a {head!r} head")
    missing = [c.candidate_id for c in eligible if c.features is None]
    if missing:
        return _row(selector, pool, None, status=STATUS_FEATURES_UNAVAILABLE,
                    detail=("frozen v1 features are built by features.features_from_state on a "
                            f"RoundState; {len(missing)} candidate(s) carry none"))
    try:
        scores = controller.predict(head, [dict(c.features or {}) for c in eligible])
    except Exception as exc:  # noqa: BLE001 - the artifact's own message is the detail
        return _row(selector, pool, None, status=STATUS_NO_CONTROLLER, detail=str(exc))
    values = [float(v) for v in scores]
    best = max(values)
    # Ties break on the candidate id, which is content-addressed and correctness-blind.
    chosen = min((c for c, v in zip(eligible, values) if v == best),
                 key=lambda c: c.candidate_id)
    return _row(selector, pool, chosen,
                scores={c.candidate_id: v for c, v in zip(eligible, values)})


def score_selector(
    selector: str,
    pool: SealedPool,
    *,
    study_seed: bytes,
    controller: Any | None = None,
    audit: AuditOutcome | None = None,
) -> SelectorRow:
    """One selector's :class:`SelectorRow` over ``pool`` (spec §12.4)."""
    if selector not in SELECTOR_ORDER:
        raise SelectorError(f"unknown selector {selector!r}; frozen set is {list(SELECTOR_ORDER)}")
    if not pool.candidates:
        return _row(selector, pool, None, status=STATUS_EMPTY_POOL, detail="the pool is empty")

    if selector in (SELECTOR_PLURALITY, SELECTOR_CONFIDENCE):
        fn = VOTE.plurality if selector == SELECTOR_PLURALITY else VOTE.confidence_weighted_plurality
        selection = fn(pool.vote_pool(), source_id=pool.source_id, study_seed=bytes(study_seed),
                       pool_kind=pool.pool_kind)
        chosen = pool.by_id(selection.selected_candidate_id)
        return _row(selector, pool, chosen, tied=selection.tied_winning_classes,
                    status=STATUS_OK if chosen is not None else STATUS_NO_KEYED_CANDIDATE,
                    detail="" if chosen is not None else "selection failed: no keyed candidate")
    if selector == SELECTOR_CURRENT:
        from agents_scaling.ard.controller import HEAD_CURRENT  # local: avoids a numpy import cost

        return _model_row(selector, HEAD_CURRENT, pool, controller)
    if selector == SELECTOR_DYNAMICS:
        from agents_scaling.ard.controller import HEAD_DYNAMICS

        return _model_row(selector, HEAD_DYNAMICS, pool, controller)

    # ---- llm_audit -----------------------------------------------------------------
    if audit is None:
        status = STATUS_NO_KEYED_CANDIDATE if pool.selection_failed else STATUS_AUDIT_NOT_RUN
        return _row(selector, pool, None, status=status,
                    detail="the LLM audit has not run on this pool")
    if audit.pool_id and audit.pool_id != pool.pool_id:
        raise SelectorError(
            f"audit outcome was produced on pool {audit.pool_id[:20]} but is being scored on "
            f"{pool.pool_id[:20]}; selector rows of one (item, round, pool) share one pool id")
    chosen = pool.by_id(audit.candidate_id)
    if chosen is not None and not chosen.keyed:  # pragma: no cover - only eligible ids are shown
        chosen = None
    return _row(selector, pool, chosen,
                status=STATUS_OK if chosen is not None else audit.status,
                detail="" if chosen is not None else f"audit returned {audit.raw!r}")


def score_all(
    pool: SealedPool,
    *,
    study_seed: bytes,
    controller: Any | None = None,
    audit: AuditOutcome | None = None,
) -> tuple[SelectorRow, ...]:
    """All five registered selectors on one sealed pool, in ``contract.SELECTORS`` order."""
    rows = tuple(
        score_selector(selector, pool, study_seed=study_seed, controller=controller, audit=audit)
        for selector in SELECTOR_ORDER
    )
    pool_ids = {row.pool_id for row in rows}
    if len(pool_ids) != 1:  # pragma: no cover - the pool id is a pure function of the pool
        raise SelectorError(f"selector rows disagree about the pool id: {sorted(pool_ids)}")
    return rows


# --------------------------------------------------------------------------- pool reader

#: Which already-written decisions form each audit pool (spec §12.4).
POOL_RULES: Final[Mapping[str, dict[str, Any]]] = {
    "root": {"round": 0, "protocol": None},
    "peer_final": {"round": K.UPDATE_ROUNDS, "protocol": K.Protocol.PEER.value},
    "gate_final": {"round": K.UPDATE_ROUNDS, "protocol": K.Protocol.GATE.value},
}


def _candidate_from_decision(row: Mapping[str, Any]) -> SelectorCandidate:
    """One ``decisions``-grain row (``types.DecisionRef.to_dict()``) as a pool candidate."""
    answer_class = row.get("answer_class")
    parse_status = str(row.get("parse_status") or "")
    valid = parse_status in {s.value for s in K.VALID_PARSE_STATUSES} and bool(answer_class)
    confidence = row.get("confidence")
    return SelectorCandidate(
        candidate_id=str(row.get("decision_id") or ""),
        answer=row.get("answer_raw"),
        canonical_class=answer_class if valid else None,
        valid=valid,
        confidence=None if confidence is None else float(confidence),
        content_sha256=str(row.get("content_sha256") or ""),
        evidence=row.get("evidence_text"),
        agent_slot=int(row.get("agent_id") or 0),
        round_=int(row.get("round") or 0),
    )


def collect_pool(
    run_root: str | os.PathLike,
    item_id: str,
    pool_kind: str,
    *,
    cells: Sequence[str] | None = None,
    checkpoint: str = K.CHECKPOINT_SMALL,
) -> SealedPool:
    """Read one item's sealed pool out of the already-written cell item files.

    The generation cells publish ``cells/<cell_id>/items/<item_id>.json``
    (``types.ArdItemResult``); this reads them the way ``runner.grade_worker`` does — by
    decision id, de-duplicated, because a root decision is content-addressed and is shared
    by every protocol that drew it (contract §3).  The order is the sorted decision id, so
    the pool (and therefore its id) does not depend on directory iteration order.

    ``cells`` restricts the scan to a named list (a manifest option); without one every
    cell directory under the run root is scanned.
    """
    rule = POOL_RULES.get(str(pool_kind))
    if rule is None:
        raise SelectorError(
            f"unknown audit pool {pool_kind!r}; spec §12.4 declares {list(AUDIT_POOLS)}")
    base = Path(run_root) / "cells"
    if cells is not None:
        directories = [base / str(cell_id) for cell_id in cells]
    else:
        directories = sorted(p for p in base.glob("*") if p.is_dir()) if base.is_dir() else []
    seen: dict[str, SelectorCandidate] = {}
    episodes: set[str] = set()
    for directory in directories:
        path = directory / "items" / f"{item_id}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - a corrupt item file is a WP4 concern
            continue
        for episode in payload.get("episodes") or []:
            if str(episode.get("checkpoint") or checkpoint) != str(checkpoint):
                continue
            protocol = str(episode.get("protocol") or "")
            if rule["protocol"] is not None and protocol != rule["protocol"]:
                continue
            for decision in episode.get("decisions") or []:
                if int(decision.get("round") or 0) != int(rule["round"]):
                    continue
                if bool(decision.get("fallback")):
                    # A WP4 baseline-executor decision is never protocol evidence.
                    continue
                candidate = _candidate_from_decision(decision)
                if not candidate.candidate_id:
                    continue
                seen.setdefault(candidate.candidate_id, candidate)
                episodes.add(str(episode.get("episode_id") or ""))
    ordered = tuple(seen[key] for key in sorted(seen))
    protocol = str(rule["protocol"] or "NONE")
    return sealed_pool(
        item_id, str(pool_kind), ordered, round_=int(rule["round"]),
        checkpoint=str(checkpoint), protocol=protocol,
        episode_ids=tuple(sorted(e for e in episodes if e)),
    )


# --------------------------------------------------------------------------- the cell


def run_selector_audit_cell(runner: Any, item_id: str) -> None:
    """``SELECTOR_AUDIT`` worker (``kinds.KIND_TABLE``): one set-wise audit call per pool.

    Signature is WP4's item-worker seam (``runner.resolve_worker`` →
    ``worker(runner, item_id)``).  One logical 8B call per (item, pool) at the frozen
    1,024 + 256 cap — spec §12.4's 240x3 = 720 calls — and five
    :class:`SelectorRow` records, all sharing the pool's id, written into the item file's
    episode events so the WP7 compaction can join them to the ``plurality`` /
    ``confidence_weighted`` rows it computes itself.
    """
    from agents_scaling.ard.report import parse_decision
    from agents_scaling.ard.types import DecisionRef, EpisodeRecord, decision_id_for, episode_id_for

    started = runner.clock()
    cell = runner.ard_cell
    task = runner.tasks[item_id]
    pool_kind = str(cell.options.get("pool") or AUDIT_POOLS[0])
    study_seed = bytes(runner.cfg.study_seed)
    pool = collect_pool(runner.run_root, item_id, pool_kind,
                        cells=cell.options.get("cells"), checkpoint=cell.checkpoint)
    presentation = audit_presentation(pool, study_seed=study_seed)
    episode_id = episode_id_for(cell, item_id, episode_rep=cell.episode_rep)

    outcome: AuditOutcome | None = None
    decisions: tuple[Any, ...] = ()
    if presentation.labels:
        record = runner.decide(
            task=task,
            messages=render_audit_prompt(task.task_text, presentation),
            purpose=K.PURPOSE_SELECTOR,
            agent_id=0,
            step_slot=int(pool.round_),
            namespace=K.NS_SELECTOR,
            episode_rep=cell.episode_rep,
        )
        report = parse_decision(record)
        outcome = parse_audit_choice(report, presentation)
        decisions = (DecisionRef(
            decision_id=decision_id_for(episode_id, round_=pool.round_, agent_id=0,
                                        phase=K.PURPOSE_SELECTOR),
            episode_id=episode_id,
            request_id=record.request_id,
            agent_id=0,
            round=int(pool.round_),
            phase=K.PURPOSE_SELECTOR,
            protocol=cell.protocol.value if cell.protocol else "NONE",
            checkpoint=cell.checkpoint,
            answer_raw=report.final_answer,
            answer_class=None,                 # an audit returns an id, never a content class
            content_sha256=report.content_sha256,
            evidence_text=report.evidence,
            confidence=report.confidence,
            parse_status=report.parse_status.value,
            thinking_tokens=int(record.thinking_tokens),
            final_tokens=int(record.final_tokens),
            prompt_tokens=int(record.prompt_tokens),
            cached_prefill_tokens=int(record.cached_prefill_tokens),
            budget_tokens=int(cell.reasoning_allowance),
            cap_hit=bool(record.cap_hit),
            boundary_injected=bool(record.boundary_injected),
            final_overrun=bool(record.final_overrun),
            answer_logprob=record.answer_logprob,
            finish_reason=str(record.finish_reason),
            latency_s=float(record.latency_s),
            endpoint_id=str(record.endpoint_id),
            flops=dict(record.flops or {}),
        ),)
    else:
        runner.event("selector_audit_empty_pool", source_id=item_id, pool=pool_kind)

    rows = score_all(pool, study_seed=study_seed, controller=None, audit=outcome)
    events: list[dict[str, Any]] = [presentation.to_row()]
    if outcome is not None:
        events.append(outcome.to_row())
    events += [row.to_row() for row in rows]
    episode = EpisodeRecord(
        episode_id=episode_id,
        item_id=item_id,
        cohort=cell.cohort.value,
        protocol=cell.protocol.value if cell.protocol else "NONE",
        checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS.get(cell.checkpoint, ""),
        reasoning_allowance=int(cell.reasoning_allowance),
        final_allowance=int(cell.final_allowance),
        packet_policy=f"selector_audit:{pool_kind}",
        parent_ref=cell.parent_ref,
        root_candidate_ids=tuple(c.candidate_id for c in pool.candidates),
        condition_manifest_hash=pool.pool_id,
        decisions=decisions,
        events=tuple(events),
    )
    runner.write_item(task, (episode,), started=started)


# The frozen v1 schema is the only feature vocabulary a model selector may read.
for _name in (SELECTOR_CURRENT, SELECTOR_DYNAMICS):
    assert _name in SELECTOR_ORDER  # noqa: S101 - import-time contract, not a test
del _name
assert set(FE.DYNAMICS_ONLY_FEATURES) <= set(FE.FEATURES_V1)  # noqa: S101
