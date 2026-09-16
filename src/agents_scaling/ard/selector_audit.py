"""Why three of the five registered selectors are null, and the offline repair of two.

``compact.selection_rows`` emits one row per (selector x pool) for all five selectors of
``contract.SELECTORS``, but computes only ``plurality`` and ``confidence_weighted``; the
other three fall into an ``else: chosen = None`` branch.  Sixty per cent of
``round_selection`` is therefore structurally null, and a reader who groups by selector
gets three cells that *look* like a measured zero and are in fact an empty cell.  That is
the generator of the "a lone correct agent reaches the team answer 0 of 186 times"
artifact: the denominator is real, the numerator was never computed.

The three are null for three different reasons, and the distinction is the whole point:

``CURRENT`` and ``dynamics_only``
    Not blocked on generation.  ``selectors.score_selector`` scores them by argmax of a
    frozen controller head (``controller.HEAD_CURRENT`` / ``HEAD_DYNAMICS``) over the
    frozen v1 feature row of each eligible candidate.  Both inputs are missing in this
    run: the single frozen artifact carries only the two fork heads (``p_private``,
    ``p_peer``) because ``fit_controller`` was called with ``current_train=None``
    (``meta.n_current_rows == 0``), so ``predict`` raises ``no head 'CURRENT'``; and no
    caller ever attached features to a candidate.  **Both are recoverable from data that
    already exists**, which is what this module does: rebuild the frozen v1 features from
    the compacted ``decisions`` grain, fit the two heads on train/validation items only,
    and score the same pools ``compact`` builds.  No new generation, no new model family.

``llm_audit``
    Genuinely blocked on generation.  Spec §12.4 budgets 720 logical 8B calls in
    ``SELECTOR_AUDIT`` cells; this run has none, in any manifest or cell directory.  No
    offline computation can substitute — the selector *is* a model call.

Neither cause is the one ``compact.selection_rows`` names.  Its docstring says the three
"appear with a null ``selected_decision_id`` until their cells run", which would be true
only if some cell's output were joined into the table.  ``compact`` never imports
``selectors`` and never reads ``EpisodeRecord.events``, so the ``selector_row`` records a
``SELECTOR_AUDIT`` cell writes would stay invisible even after the cells ran.  Running the
audit is necessary and not sufficient; the join is missing too.

What this module refuses to do
------------------------------

The refitted heads are **not** the preregistered CURRENT.  The preregistered artifact was
frozen without them, and a head fitted after the test labels exist cannot be relabelled as
the frozen one however carefully it is trained.  So every repaired row carries
``selector_source`` naming the refit, the fit itself is refused any test item
(``controller.CurrentFrame.validate`` raises on one), and the output is a *separate* table:
nothing here overwrites ``round_selection``, and no repaired number may be reported as the
sealed §12.4 comparison without an amendment saying when the head was fitted.

Loudness instead of nulls
-------------------------

Every row this module emits carries ``unavailable_reason``, drawn from
``selectors.STATUS_*`` so there is one vocabulary in the study, plus
:data:`REASON_NOT_WIRED` for the compaction gap that no selector-side status can express.
An empty string means the row is a real selection.  ``compact.SCHEMAS`` is a strict
whitelist and ``_project`` raises on an undeclared column, so the same column has to be
added to ``compact`` by its owner; the diff is in this module's report, and until it lands
the join key ``(episode_id, cell_id, round, selector, pool)`` carries the reason here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.ard import graders as G
from agents_scaling.ard import selectors as SEL

__all__ = [
    "REASON_AUDIT_NOT_RUN",
    "REASON_EMPTY_POOL",
    "REASON_FEATURES_UNAVAILABLE",
    "REASON_NONE",
    "REASON_NOT_WIRED",
    "REASON_NO_CONTROLLER_HEAD",
    "REASON_NO_KEYED_CANDIDATE",
    "REASON_ORDER",
    "SOURCE_REFIT",
    "SOURCE_UNAVAILABLE",
    "STATIC_SELECTORS",
    "SelectorAvailability",
    "SelectorRepair",
    "availability_rows",
    "build_repair",
    "features_by_decision",
    "fit_static_heads",
    "main",
    "probe_run",
    "static_current_rows",
    "unavailable_reason",
]

# --------------------------------------------------------------------------- vocabulary

#: A row that carries a real selection.  Never ``None``: a null reason beside a null
#: selection is the ambiguity this column exists to remove.
REASON_NONE: Final[str] = ""

#: Reused verbatim from :mod:`agents_scaling.ard.selectors` so the emitted reason is the
#: status the selector itself would have returned, not a second vocabulary for one fact.
REASON_EMPTY_POOL: Final[str] = SEL.STATUS_EMPTY_POOL
REASON_NO_KEYED_CANDIDATE: Final[str] = SEL.STATUS_NO_KEYED_CANDIDATE
REASON_NO_CONTROLLER_HEAD: Final[str] = SEL.STATUS_NO_CONTROLLER
REASON_FEATURES_UNAVAILABLE: Final[str] = SEL.STATUS_FEATURES_UNAVAILABLE
REASON_AUDIT_NOT_RUN: Final[str] = SEL.STATUS_AUDIT_NOT_RUN

#: The one fact no selector-side status can carry: the selector may have produced rows in
#: its own cell, and ``compact.selection_rows`` has no join that would read them.
REASON_NOT_WIRED: Final[str] = "not_wired_into_compaction"

#: Declaration order, most specific cause first — a reader scanning the column sees the
#: repairable causes before the structural one.
REASON_ORDER: Final[tuple[str, ...]] = (
    REASON_NONE, REASON_EMPTY_POOL, REASON_NO_KEYED_CANDIDATE, REASON_FEATURES_UNAVAILABLE,
    REASON_NO_CONTROLLER_HEAD, REASON_AUDIT_NOT_RUN, REASON_NOT_WIRED,
)

#: The two selectors this module can compute without a model call.
STATIC_SELECTORS: Final[tuple[str, ...]] = (SEL.SELECTOR_CURRENT, SEL.SELECTOR_DYNAMICS)

SOURCE_REFIT: Final[str] = "refit_offline_v1"
SOURCE_UNAVAILABLE: Final[str] = "unavailable"

#: Where the repair lands.  Separate tables, never a rewrite of ``round_selection``.
REPAIR_TABLE: Final[str] = "round_selection_static"
AVAILABILITY_TABLE: Final[str] = "selector_availability"
SUMMARY_FILE: Final[str] = "selector_audit.json"

#: The pools ``compact.selection_rows`` builds, in its order.
POOLS: Final[tuple[str, ...]] = K.POOL_KINDS

_VALID_PARSE: Final[frozenset[str]] = frozenset(s.value for s in K.VALID_PARSE_STATUSES)


# --------------------------------------------------------------------------- the probe


@dataclass(frozen=True)
class SelectorAvailability:
    """One selector's measured availability in one run root — the diagnosis, as a row."""

    selector: str
    available: bool
    reason: str
    detail: str
    #: True when only a model call can supply what is missing (spec §12.4's 720 8B calls).
    needs_generation: bool
    #: True when the missing input can be rebuilt from artifacts the run already wrote.
    repairable_offline: bool

    def to_row(self) -> dict[str, Any]:
        return {
            "selector": self.selector, "available": bool(self.available),
            "unavailable_reason": self.reason, "detail": self.detail,
            "needs_generation": bool(self.needs_generation),
            "repairable_offline": bool(self.repairable_offline),
        }


def _controller_heads(run_root: Path) -> dict[str, list[str]]:
    """``{controller_sha: [head, ...]}`` for every frozen artifact under the run root."""
    from agents_scaling.ard import controller as CT

    base = run_root / CT.CONTROLLER_DIR
    if not base.is_dir():
        return {}
    out: dict[str, list[str]] = {}
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        try:
            out[directory.name] = sorted(CT.load_controller(directory).models)
        except CT.ControllerError as exc:          # a tampered artifact is still a finding
            out[directory.name] = [f"<unloadable: {exc}>"]
    return out


def _audit_cell_ids(run_root: Path) -> list[str]:
    """``SELECTOR_AUDIT`` cell directories, by the kind prefix ``cells.cell_id`` builds."""
    from agents_scaling.ard.types import ArdCellKind

    base = run_root / "cells"
    if not base.is_dir():
        return []
    prefix = f"{ArdCellKind.SELECTOR_AUDIT.value}."
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix))


def probe_run(run_root: str | Path) -> dict[str, Any]:
    """Measure, in one run root, what each null selector is actually waiting for."""
    from agents_scaling.ard import controller as CT

    root = Path(run_root)
    heads = _controller_heads(root)
    audit_cells = _audit_cell_ids(root)
    has_current = any(CT.HEAD_CURRENT in v for v in heads.values())
    has_dynamics = any(CT.HEAD_DYNAMICS in v for v in heads.values())
    return {
        "run_root": str(root),
        "controller_artifacts": heads,
        "has_current_head": has_current,
        "has_dynamics_head": has_dynamics,
        "selector_audit_cells": audit_cells,
        "n_selector_audit_cells": len(audit_cells),
        # The compaction gap is a property of the code, not of the run: assert it here so a
        # future wiring change makes this probe, and the report it feeds, wrong and loud.
        "compaction_joins_wp5_rows": _compaction_joins_wp5_rows(),
    }


def _compaction_joins_wp5_rows() -> bool:
    """Does ``compact.selection_rows`` read WP5 selector output at all?

    A ``SELECTOR_AUDIT`` cell writes its five :class:`selectors.SelectorRow` records into
    its episode's ``events``, so reading that attribute is the only path by which a WP5
    selection could reach ``round_selection``.  Read off the compiled function rather than
    asserted from memory: wire the join and this flips without an edit here.
    """
    from agents_scaling.ard import compact as CP

    return "events" in getattr(CP.selection_rows, "__code__").co_names


def availability_rows(probe: Mapping[str, Any]) -> list[SelectorAvailability]:
    """The five registered selectors with this run's measured cause for each."""
    from agents_scaling.ard import controller as CT

    wired = bool(probe.get("compaction_joins_wp5_rows"))
    unwired = "" if wired else ("; and compact.selection_rows never joins WP5 selector "
                                "output, so running the cell would not populate the table")
    out = [
        SelectorAvailability(SEL.SELECTOR_PLURALITY, True, REASON_NONE,
                             "computed in compact.selection_rows from the round's decisions",
                             False, False),
        SelectorAvailability(SEL.SELECTOR_CONFIDENCE, True, REASON_NONE,
                             "computed in compact.selection_rows from the round's decisions",
                             False, False),
    ]
    for selector, head, present in (
        (SEL.SELECTOR_CURRENT, CT.HEAD_CURRENT, probe.get("has_current_head")),
        (SEL.SELECTOR_DYNAMICS, CT.HEAD_DYNAMICS, probe.get("has_dynamics_head")),
    ):
        if present:
            out.append(SelectorAvailability(
                selector, False, REASON_FEATURES_UNAVAILABLE,
                f"a frozen {head!r} head exists, but no caller attaches frozen v1 features "
                "to the candidates of a pool" + unwired, False, True))
        else:
            out.append(SelectorAvailability(
                selector, False, REASON_NO_CONTROLLER_HEAD,
                f"no frozen controller carries a {head!r} head (fit_controller ran with "
                "current_train=None), and no caller attaches frozen v1 features to a pool's "
                "candidates; both are rebuildable from already-generated data" + unwired,
                False, True))
    out.append(SelectorAvailability(
        SEL.SELECTOR_LLM_AUDIT, False, REASON_AUDIT_NOT_RUN,
        f"{int(probe.get('n_selector_audit_cells') or 0)} SELECTOR_AUDIT cells in the run "
        "root; spec §12.4's 720 logical 8B calls were never spent, and no offline "
        "computation can stand in for a model call" + unwired, True, False))
    return out


# --------------------------------------------------------------------------- state view

# ``features.features_from_state`` is the frozen v1 builder and the module docstring of
# ``selectors`` forbids a second one, so the compacted ``decisions`` grain is adapted to the
# small part of the ``states.RoundState`` surface that builder touches — ``agents[slot].root``,
# ``latest``, ``reports_of_round``, ``canonical_class`` and ``family`` — rather than
# reimplemented.  ``canonical_class`` returns the class the run already recorded: the real
# state classifies report text with the injected gold-independent classifier, and re-running
# it here would risk a different answer for the same report.


class _Report:
    """The report surface ``features_from_state`` reads off a ``ReportRef``."""

    __slots__ = ("final_answer", "confidence", "valid", "content_sha256", "canonical_class")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.final_answer = row.get("answer_raw")
        self.confidence = row.get("confidence")
        self.valid = (str(row.get("parse_status") or "") in _VALID_PARSE
                      and bool(row.get("answer_class")))
        self.content_sha256 = str(row.get("content_sha256") or "")
        self.canonical_class = row.get("answer_class") if self.valid else None


class _Ref:
    """One decision as a ``states.ReportRef``."""

    __slots__ = ("decision_id", "agent_slot", "round_", "report", "row")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.decision_id = str(row.get("decision_id") or "")
        self.agent_slot = int(row.get("agent_id") or 0)
        self.round_ = int(row.get("round") or 0)
        self.report = _Report(row)
        self.row = row


class _Agent:
    __slots__ = ("agent_slot", "root", "latest")

    def __init__(self, slot: int, root: _Ref, latest: _Ref) -> None:
        self.agent_slot, self.root, self.latest = slot, root, latest


class _Runtime:
    """The three runtime quantities that live on a ``DecisionRecord``, not on a report."""

    __slots__ = ("answer_logprob", "thinking_tokens", "cap_hit")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.answer_logprob = row.get("answer_logprob")
        self.thinking_tokens = row.get("thinking_tokens") or 0
        self.cap_hit = bool(row.get("cap_hit"))


class _StateView:
    """The ``RoundState`` surface of one team at the end of one round."""

    __slots__ = ("agents", "archive", "family")

    def __init__(self, agents: Sequence[_Agent], archive: Mapping[int, tuple[_Ref, ...]],
                 family: str) -> None:
        self.agents, self.archive, self.family = tuple(agents), dict(archive), family

    def latest(self, slot: int) -> _Ref:
        return self.agents[int(slot)].latest

    def reports_of_round(self, round_: int) -> tuple[_Ref, ...]:
        return tuple(self.archive.get(int(round_), ()))

    def canonical_class(self, report: _Report) -> str | None:
        return report.canonical_class


#: How a row of ``decisions`` / ``episodes`` names the team it belongs to.  ``episode_id``
#: alone is *not* that name: 4,888 episode rows of this run carry 4,240 distinct ids,
#: because a ROOTS bank aliases the PRIVATE episode of the same item and two FORK cells
#: branch off one parent (defect D-11).  The frozen v1 feature of a decision depends on the
#: team state around it, so an episode-id-only key would hand one decision the other cell's
#: peers.
CellEpisode = tuple[str, str]


def episode_key(row: Mapping[str, Any]) -> CellEpisode:
    """``(cell_id, episode_id)`` — ``""`` for a v1 table that predates the ``cell_id`` column."""
    return str(row.get("cell_id") or ""), str(row.get("episode_id") or "")


def features_by_decision(
    decisions: Sequence[Mapping[str, Any]],
    episodes: Mapping[CellEpisode, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[str, int]]:
    """Frozen v1 features per ``(cell_id, episode_id, decision_id)``, plus coverage counters.

    An episode-round is featurisable only when it is a whole five-agent team over
    ``contract.ROUNDS`` with a round 0 — the shape ``features_from_state`` requires, since
    it reads ``state.latest(j)`` for every slot and each agent's root.  The one-agent
    RECEIVER probes and the sixteen-draw ROOTS banks are neither, and the FORK branches
    carry no round 0 of their own; those decisions get no features, and the selectors that
    need them stay unavailable with ``features_unavailable`` rather than a guessed row.
    """
    by_episode: dict[CellEpisode, dict[int, dict[int, _Ref]]] = defaultdict(
        lambda: defaultdict(dict))
    runtime: dict[str, _Runtime] = {}
    for row in decisions:
        ref = _Ref(row)
        by_episode[episode_key(row)][ref.round_][ref.agent_slot] = ref
        runtime[ref.decision_id] = _Runtime(row)

    features: dict[tuple[str, str, str], dict[str, float]] = {}
    counters: dict[str, int] = defaultdict(int)
    slots = set(range(K.N_AGENTS))
    for key, rounds in by_episode.items():
        meta = episodes.get(key)
        n_rounds = len(rounds)
        if meta is None:
            counters["skipped_no_episode_row"] += n_rounds
            continue
        if 0 not in rounds:
            counters["skipped_no_root_round"] += n_rounds
            continue
        if not all(set(rounds[r]) == slots for r in rounds):
            counters["skipped_not_five_agent_team"] += n_rounds
            continue
        if not all(r in K.ROUNDS for r in rounds):
            counters["skipped_round_outside_contract"] += n_rounds
            continue
        family = str(meta.get("family") or "")
        archive: dict[int, tuple[_Ref, ...]] = {}
        latest: dict[int, _Ref] = {}
        for round_ in sorted(rounds):
            archive[round_] = tuple(rounds[round_][s] for s in range(K.N_AGENTS))
            latest.update(rounds[round_])
            view = _StateView(
                [_Agent(s, rounds[0][s], latest[s]) for s in range(K.N_AGENTS)],
                archive, family)
            for slot in range(K.N_AGENTS):
                ref = rounds[round_][slot]
                features[(*key, ref.decision_id)] = FE.features_from_state(
                    view, slot, round_=round_, records=runtime)
            counters["featurised_rounds"] += 1
    counters["featurised_decisions"] = len(features)
    counters["featurised_distinct_decision_ids"] = len({k[2] for k in features})
    return features, dict(counters)


# --------------------------------------------------------------------------- the refit


def _tri_state(label: Mapping[str, Any] | None) -> bool | None:
    """``compact._is_correct``'s rule, on a labels row: the status, never ``is_correct``."""
    if label is None:
        return None
    status = str(label.get("evaluator_status") or "")
    if status == G.STATUS_CORRECT:
        return True
    if status == G.STATUS_INCORRECT:
        return False
    return None


def static_current_rows(
    features: Mapping[tuple[str, str, str], Mapping[str, float]],
    decisions: Sequence[Mapping[str, Any]],
    episodes: Mapping[CellEpisode, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Any], dict[str, int]]:
    """Every featurised decision as a ``controller.CurrentRow``, once per decision id.

    A root decision is content-addressed and is shared by every protocol that drew it
    (contract §3): 67,104 ``decisions`` rows of this run carry 52,768 distinct ids.  Each
    alias sees a different team, so each carries different features — but they are one
    generated candidate with one grade, and fitting on all of them would weight the aliased
    roots by their alias count.  The frame therefore keeps the first alias in sorted
    ``(cell_id, episode_id)`` order and counts the rest, which is deterministic and
    reported rather than silent.
    """
    from agents_scaling.ard import controller as CT

    rows: list[Any] = []
    counters: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    for row in sorted(decisions, key=lambda r: (*episode_key(r), str(r.get("decision_id")))):
        decision_id = str(row.get("decision_id") or "")
        vector = features.get((*episode_key(row), decision_id))
        if vector is None:
            counters["no_features"] += 1
            continue
        if decision_id in seen:
            counters["aliased_duplicates_dropped"] += 1
            continue
        seen.add(decision_id)
        meta = episodes.get(episode_key(row), {})
        correct = _tri_state(labels.get(decision_id))
        rows.append(CT.CurrentRow(
            row_id=decision_id, item_id=str(meta.get("item_id") or ""),
            family=str(meta.get("family") or ""), split=str(meta.get("split") or ""),
            round_=int(row.get("round") or 0), agent_slot=int(row.get("agent_id") or 0),
            features=dict(vector), episode_id=str(row.get("episode_id") or ""),
            y_correct=None if correct is None else int(correct),
        ))
    counters["rows"] = len(rows)
    return rows, dict(counters)


def fit_static_heads(
    rows: Sequence[Any],
    *,
    study_seed: bytes = b"\x00" * 32,
    penalties: Sequence[float] = K.PENALTY_GRID,
    n_folds: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit ``CURRENT`` and ``dynamics_only`` on training-eligible, graded rows only.

    Spec §12.4: "All are trained/calibrated only on training/validation items."  The filter
    is ``features.training_eligibility``, and ``CurrentFrame.validate`` refuses a test row
    outright, so the firewall is enforced twice — once when the frame is built and once by
    the frame itself.  The heads are the same L2-logistic head ``controller.fit_controller``
    would have fitted, selected by the same one-SE rule over ``contract.PENALTY_GRID``;
    what is *not* the same is when they were fitted, which is why the artifact this returns
    is written beside the frozen controller and never over it.
    """
    from agents_scaling.ard import controller as CT

    eligible = [r for r in rows
                if FE.training_eligibility(r.split) == "eligible" and r.y_correct is not None]
    if not eligible:
        raise CT.ControllerError(
            "no training-eligible graded row: the CURRENT head cannot be fitted from this run")
    frame = CT.CurrentFrame(tuple(eligible)).validate()
    folds = CT.DEFAULT_FOLDS if n_folds is None else int(n_folds)
    y = frame.y()
    models: dict[str, Any] = {}
    selections: dict[str, dict[str, Any]] = {}
    for head, names in ((CT.HEAD_CURRENT, FE.FEATURES_V1),
                        (CT.HEAD_DYNAMICS, FE.DYNAMICS_ONLY_FEATURES)):
        model, selection = CT._fit_head(
            frame.matrix(names), y, frame.clusters(), frame.strata(), names,
            seed=bytes(study_seed), penalties=penalties, n_folds=folds)
        models[head] = model
        selections[head] = {**selection.to_dict(), "n_rows": int(len(frame))}
    meta = {
        "fitted_by": __name__,
        "fit_is_post_hoc": True,
        "n_current_rows": int(len(frame)),
        "n_train_items": len({r.item_id for r in eligible}),
        "splits": sorted({r.split for r in eligible}),
        "base_rate": float(sum(int(r.y_correct) for r in eligible) / len(eligible)),
        "penalty_grid": list(penalties),
        "n_folds": folds,
        "note": ("CURRENT and dynamics_only heads fitted offline after the run; NOT the "
                 "preregistered frozen controller, which carries no such head."),
    }
    controller = CT.FrozenController(
        feature_names=tuple(FE.FEATURES_V1), models=models, selections=selections, meta=meta)
    info = {
        "n_rows": int(len(frame)), "n_items": meta["n_train_items"],
        "base_rate": meta["base_rate"],
        "penalty": {head: float(sel["penalty"]) for head, sel in selections.items()},
        "controller_sha256": controller.sha256,
    }
    return controller, info


# --------------------------------------------------------------------------- scoring


def _candidate(
    row: Mapping[str, Any],
    features: Mapping[tuple[str, str, str], Mapping[str, float]],
    key: CellEpisode,
) -> Any:
    """One decisions row as a pool candidate, carrying the features of *this* team's state."""
    vector = features.get((*key, str(row.get("decision_id") or "")))
    valid = str(row.get("parse_status") or "") in _VALID_PARSE and bool(row.get("answer_class"))
    confidence = row.get("confidence")
    return SEL.SelectorCandidate(
        candidate_id=str(row.get("decision_id") or ""),
        answer=row.get("answer_raw"),
        canonical_class=row.get("answer_class") if valid else None,
        valid=valid,
        confidence=None if confidence is None else float(confidence),
        content_sha256=str(row.get("content_sha256") or ""),
        evidence=row.get("evidence_text"),
        agent_slot=int(row.get("agent_id") or 0),
        round_=int(row.get("round") or 0),
        features=None if vector is None else dict(vector),
    )


def unavailable_reason(status: str) -> str:
    """The ``unavailable_reason`` a ``selectors.SelectorRow`` status implies."""
    return REASON_NONE if status == SEL.STATUS_OK else str(status)


@dataclass(frozen=True)
class SelectorRepair:
    """The repair's two tables, the numbers that justify them, and the refitted heads."""

    availability: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    #: The post-hoc ``FrozenController`` the static rows were scored with, or ``None``.
    controller: Any | None = None


def build_repair(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_root: str | Path,
    study_seed: bytes = b"\x00" * 32,
    refit: bool = True,
    controller: Any | None = None,
) -> SelectorRepair:
    """Diagnose the five selectors and, where possible, compute the two static ones.

    The pool construction mirrors ``compact.selection_rows`` exactly — ``root`` is round 0,
    ``current`` is this round, ``archive`` is every round so far, and ROOTS cells are
    skipped because the bank is selected *from*, never selected *over* — so a repaired row
    joins its ``round_selection`` row on
    ``(episode_id, cell_id, round, selector, pool)``.

    ``controller`` supplies already-fitted heads (and suppresses the refit); ``refit=False``
    with no controller is the diagnosis-only mode, in which every static row is
    ``no_controller`` — which is exactly the state ``round_selection`` is in today.
    """
    probe = probe_run(run_root)
    availability = availability_rows(probe)
    episode_rows = list(tables.get("episodes", ()))
    episodes = {episode_key(e): e for e in episode_rows}
    labels = {str(l["decision_id"]): l for l in tables.get("labels", ())}
    decisions = list(tables.get("decisions", ()))
    keyed_by_cell = any(r.get("cell_id") for r in decisions)

    features, coverage = features_by_decision(decisions, episodes)
    fit_info: dict[str, Any] = {"status": "supplied" if controller is not None else
                                ("not_requested" if not refit else "unavailable")}
    if refit and controller is None:
        rows, frame_counts = static_current_rows(features, decisions, episodes, labels)
        controller, fit_info = fit_static_heads(rows, study_seed=study_seed)
        fit_info["status"] = "fitted"
        fit_info["frame"] = frame_counts

    by_episode: dict[CellEpisode, dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in decisions:
        by_episode[episode_key(row)][int(row.get("round") or 0)].append(row)

    out: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for key, rounds in sorted(by_episode.items()):
        meta = episodes.get(key)
        if meta is None:
            counts["episodes_without_metadata"] += 1
            continue
        cell_id, episode_id = key
        cell_id = cell_id or str(meta.get("cell_id") or "")
        cell_kind = str(meta.get("cell_kind") or cell_id.split(".", 1)[0])
        if cell_kind == "ROOTS":
            # The bank is selected from, never selected over (compact.selection_rows, M2).
            counts["roots_rounds_skipped"] += len(rounds)
            continue
        archive: list[Mapping[str, Any]] = []
        root = sorted(rounds.get(0, ()), key=lambda r: int(r.get("agent_id") or 0))
        for round_ in sorted(rounds):
            current = sorted(rounds[round_], key=lambda r: int(r.get("agent_id") or 0))
            archive = archive + list(current)
            for pool_name, members in (("root", root), ("current", current),
                                       ("archive", archive)):
                candidates = [_candidate(row, features, key) for row in members]
                pool = SEL.sealed_pool(
                    str(meta.get("item_id") or ""), pool_name, candidates, round_=round_,
                    checkpoint=str(meta.get("checkpoint") or K.CHECKPOINT_SMALL),
                    protocol=str(meta.get("protocol") or "NONE"),
                    cohort=str(meta.get("cohort") or K.Cohort.NATURAL.value),
                    episode_ids=(episode_id,))
                for selector in (*STATIC_SELECTORS, SEL.SELECTOR_LLM_AUDIT):
                    scored = SEL.score_selector(selector, pool, study_seed=bytes(study_seed),
                                                controller=controller, audit=None)
                    chosen = pool.by_id(scored.selected_candidate_id)
                    correct = (None if chosen is None
                               else _tri_state(labels.get(chosen.candidate_id)))
                    reason = unavailable_reason(scored.status)
                    counts[f"{selector}:{reason or 'ok'}"] += 1
                    out.append({
                        "episode_id": episode_id, "cell_id": cell_id, "round": int(round_),
                        "item_id": str(meta.get("item_id") or ""),
                        "protocol": str(meta.get("protocol") or ""),
                        "checkpoint": str(meta.get("checkpoint") or ""),
                        "cohort": str(meta.get("cohort") or ""),
                        "split": str(meta.get("split") or ""),
                        "selector": selector, "pool": pool_name,
                        "selected_decision_id": scored.selected_candidate_id,
                        "selected_class": scored.selected_class,
                        "selected_correct": correct,
                        "valid_count": int(scored.valid_count),
                        "planned_count": int(scored.planned_count),
                        "pool_size": len(pool),
                        "status": scored.status,
                        "unavailable_reason": reason,
                        "detail": scored.detail,
                        "selector_source": (SOURCE_UNAVAILABLE if scored.selected_candidate_id
                                            is None else SOURCE_REFIT),
                    })

    summary = {
        "probe": probe,
        "feature_coverage": coverage,
        "fit": fit_info,
        "rows": len(out),
        "by_selector_reason": {k: v for k, v in sorted(counts.items())},
        "reason_vocabulary": list(REASON_ORDER),
        # A v1 table has no cell_id, so (cell_id, episode_id) collapses to the episode id and
        # the 648 aliased episode rows of this run merge.  Say so rather than emit rows whose
        # team membership is a guess.
        "team_key": "cell_id+episode_id" if keyed_by_cell else "episode_id_only",
        "n_episode_rows": len(episode_rows),
        "n_distinct_team_keys": len(episodes),
    }
    return SelectorRepair([a.to_row() for a in availability], out, summary, controller)


# --------------------------------------------------------------------------- CLI


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def load_tables(tables_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """The four compacted tables this module reads, by name."""
    base = Path(tables_dir)
    return {name: _read_parquet(base / f"{name}.parquet")
            for name in ("decisions", "episodes", "labels", "items")}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.selector_audit",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--tables", required=True, help="the compacted tables directory to read")
    p.add_argument("--run-root", required=True, help="the run root holding cells/ and controller/")
    p.add_argument("--out", default=None, help="where to write the repair (defaults to --tables)")
    p.add_argument("--no-refit", action="store_true",
                   help="diagnose only; do not fit the CURRENT / dynamics_only heads")
    p.add_argument("--save-controller", action="store_true",
                   help="also write the refitted heads under <out>/controller_refit/<sha>/")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = load_tables(args.tables)
    out = Path(args.out) if args.out else Path(args.tables)
    repair = build_repair(tables, run_root=args.run_root, refit=not args.no_refit)

    out.mkdir(parents=True, exist_ok=True)
    from agents_scaling.study.aggregate import _write_parquet

    _write_parquet(out / f"{AVAILABILITY_TABLE}.parquet", repair.availability)
    _write_parquet(out / f"{REPAIR_TABLE}.parquet", repair.rows)
    if args.save_controller and repair.controller is not None:
        # Under its own directory name: a post-hoc head must never be mistaken for the
        # artifact a ``controller:<sha>`` dependency gate resolves.
        saved = repair.controller.save(
            out / "controller_refit" / repair.controller.sha256)
        repair.summary["fit"]["artifact"] = str(saved)
    (out / SUMMARY_FILE).write_text(json.dumps(repair.summary, indent=2, sort_keys=True),
                                    encoding="utf-8")
    print(json.dumps({
        "availability": repair.availability,
        "rows": repair.summary["rows"],
        "fit": repair.summary["fit"],
        "feature_coverage": repair.summary["feature_coverage"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
