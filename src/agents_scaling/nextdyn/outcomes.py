"""Per-agent, per-team and per-transition outcomes (plan §4.1; spec 4.2-4.6, 8.5, 10.2).

Why this module exists at all, and why it is three tables rather than one: ARD36 computed
"coverage" once, in a column named ``coverage_current``, and every downstream claim read that
column without ever being able to re-derive it.  The repair found that the same statistic had
two definitions in the codebase and that a boolean ``is_correct`` had been mixed with a
three-state grade, so a team with an ungradeable member was silently counted as a team with a
wrong member.  Here the three states ``C``/``W``/``U`` are carried all the way through:

* every coverage, margin and accuracy number is a **pair** (``*_lo``, ``*_hi``).  ``lo`` counts
  only resolved correct slots, ``hi`` credits every ``U`` as possibly correct, so the pair
  brackets every feasible labelling of the unresolved slots (spec 4.6; spec 18.3
  ``test_lower_upper_coverage_bounds_include_all_feasible_labels``);
* a **switch of answer** and a **change of validity** are different events and live in
  different columns (``switched_class`` vs ``validity_change``), because ARD36 reported a
  C -> unparsed move as a "correct to wrong" transition;
* the three-state bookkeeping identity ``C_{r+1} - C_r = N_WC + N_UC - N_CW - N_CU`` is
  asserted on **every** consecutive pair with a fatal residual
  (:func:`assert_three_state_identity`), which is the one check that would have caught the
  ARD36 transition table drifting away from its own counts;
* hazard and preservation are reported **twice** -- as a baseline-root-cohort trajectory (the
  item set fixed at r=0) and as a state-conditional hazard (the risk set re-evaluated each
  round) -- and never merged, because the two answer different questions and ARD36's single
  number was a mixture of both.

Grades are read only through ``grading.tri_state`` (the sole sanctioned reader, spec 4.6); no
boolean grade column exists anywhere in nextdyn.  Columns are declared once, as the row
dataclasses ``tables.TeamRoundRow`` / ``AgentRoundRow`` / ``TransitionRow`` (R1).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.errors import ContractViolation, MaterializeError
from agents_scaling.nextdyn.ids import BANK_KINDS, CellKind

#: The three semantic states (spec 4.6).  ``config.load_study(...).analysis`` declares the same
#: alphabet as ``semantic_correctness_states``; :func:`assert_state_alphabet` holds the two to
#: each other rather than letting a config edit drift away from the code.
STATE_CORRECT: Final[str] = "C"
STATE_WRONG: Final[str] = "W"
STATE_UNRESOLVED: Final[str] = "U"
STATES: Final[tuple[str, ...]] = (STATE_CORRECT, STATE_WRONG, STATE_UNRESOLVED)

#: The nine transition kinds over three states (plan §4.1).
TRANSITION_KINDS: Final[tuple[str, ...]] = (
    "CC", "CW", "CU", "WC", "WW", "WU", "UC", "UW", "UU",
)

PLURALITY_STRICT_MAJORITY: Final[str] = "strict_majority"
PLURALITY_UNIQUE: Final[str] = "unique_plurality"
PLURALITY_TIED: Final[str] = "tied"
PLURALITY_NONE: Final[str] = "none"
PLURALITY_STATUSES: Final[tuple[str, ...]] = (
    PLURALITY_STRICT_MAJORITY, PLURALITY_UNIQUE, PLURALITY_TIED, PLURALITY_NONE,
)

#: n-gram width of the copying statistic (spec 8.5 "8-gram overlap").
COPY_NGRAM: Final[int] = 8

#: Where ``derive`` writes, relative to the run root, and where it reads the protected grades.
DERIVED_DIR: Final[str] = "derived"
TABLES_DIR: Final[str] = "tables"
PROTECTED_TABLES_DIR: Final[str] = f"{tables.PROTECTED_DIR}/{TABLES_DIR}"

#: The kinds whose decisions carry an answer that can be graded into a state.  SELECTION_AUDIT
#: rows carry a *score*, not an answer, so they never become an agent round.
OUTCOME_KINDS: Final[frozenset[CellKind]] = frozenset(
    BANK_KINDS | {CellKind.EPISODE, CellKind.RECEIVER})

#: A two-phase receiver episode writes two rows at (round 0, slot 0); only the decision phase
#: carries the answer, so the plan phase never becomes an agent round (it would collide on the
#: ``agent_rounds`` primary key, which is exactly the D-11 shape).
NON_ANSWER_PHASES: Final[frozenset[str]] = frozenset({K.PHASE_RECEIVER_PLAN})


# --------------------------------------------------------------------------- agent states


@dataclass(frozen=True)
class AgentState:
    """One slot's observed state in one round (plan §4.1 ``agent_state``)."""

    slot: int
    valid: bool
    cls: str | None
    state: str
    u_reason: str | None
    confidence: float | None
    answer_logprob_mean: float | None = None
    decision_id: str = ""
    generation_uid: str = ""
    source_item_id: str = ""

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ContractViolation(f"agent state {self.state!r} is not one of {STATES}")
        if self.state == STATE_CORRECT and not self.valid:
            raise ContractViolation(
                f"slot {self.slot}: an invalid delivered answer cannot be graded C "
                "(validity and semantic correctness are separate columns, spec 4.6)")


def assert_state_alphabet(analysis_block: Any) -> None:
    """``study.analysis.semantic_correctness_states`` must be exactly the coded alphabet.

    This is the only reader of that leaf: the estimators below are written for three states and
    nothing else, so a config edit that added a fourth would be a silent mis-analysis rather
    than a failure.  ``analysis_block`` is a compiled ``config.AnalysisBlock``.
    """
    declared = analysis_block.semantic_correctness_states
    if tuple(declared) != STATES:
        raise ContractViolation(
            f"study.analysis.semantic_correctness_states {tuple(declared)} != {STATES}: the "
            "estimators in outcomes.py are written for exactly three states")


def _tri_state(grade_row: Mapping[str, Any]) -> str:
    """``grading.tri_state`` when WP3 has landed; the same rule, stated once, until it has.

    The rule is the spec-4.6 one: the grade column is ``semantic_correct`` and it is a
    three-valued string.  A boolean correctness column is refused rather than coerced, which is
    the check that makes "no boolean grade exists anywhere" enforceable.
    """
    try:  # pragma: no cover - exercised once WP3 lands grading.py
        from agents_scaling.nextdyn.grading import tri_state
    except ImportError:
        pass
    else:
        return tri_state(grade_row)
    for forbidden in ("is_correct", "correct", "semantic_correct_bool"):
        if forbidden in grade_row:
            raise ContractViolation(
                f"grade row carries a boolean column {forbidden!r}; grades are the three-state "
                "semantic_correct only (D15)")
    state = grade_row.get("semantic_correct")
    if state not in STATES:
        raise ContractViolation(f"grades_private.semantic_correct {state!r} not in {STATES}")
    return str(state)


def agent_state(decision_row: Mapping[str, Any], grade_row: Mapping[str, Any] | None = None) -> AgentState:
    """The state of one decision: ``U`` unless a grade resolved it, never imputed.

    An ungraded decision is ``U`` with ``u_reason='not_graded'`` -- never wrong.  The decision's
    own ``validity``/``parse_status`` decide *delivery*; the grade decides *semantics*; the two
    never substitute for each other (spec 18.3 ``test_validity_change_is_not_answer_switch``).
    """
    valid = bool(decision_row.get("validity"))
    if grade_row is None:
        state, u_reason = STATE_UNRESOLVED, "not_graded"
    else:
        state = _tri_state(grade_row)
        u_reason = grade_row.get("u_reason") if state == STATE_UNRESOLVED else None
    cls = decision_row.get("canonical_class")
    if cls is None and grade_row is not None:
        cls = grade_row.get("canonical_class")
    if not valid:
        # A delivery failure has no answer to group by; a class here would create a phantom
        # plurality candidate out of an undelivered draft.
        cls = None
        if state == STATE_CORRECT:
            raise ContractViolation(
                f"decision {decision_row.get('decision_id')} is graded C with validity=False")
    confidence = decision_row.get("confidence")
    return AgentState(
        slot=int(decision_row.get("agent_slot") or 0),
        valid=valid,
        cls=None if cls is None else str(cls),
        state=state,
        u_reason=None if u_reason is None else str(u_reason),
        confidence=None if confidence is None else float(confidence),
        answer_logprob_mean=(None if decision_row.get("answer_logprob_mean") is None
                             else float(decision_row["answer_logprob_mean"])),
        decision_id=str(decision_row.get("decision_id") or ""),
        generation_uid=str(decision_row.get("generation_uid") or ""),
        source_item_id=str(decision_row.get("source_item_id") or ""),
    )


# --------------------------------------------------------------------------- team rounds


def _class_counts(states: Sequence[AgentState]) -> Counter:
    return Counter(s.cls for s in states if s.valid and s.cls is not None)


def _entropy_norm(counts: Counter, n_valid: int) -> float | None:
    if n_valid < 2 or not counts:
        return None
    shares = [c / n_valid for c in counts.values()]
    entropy = -math.fsum(p * math.log(p) for p in shares if p > 0)
    return float(entropy / math.log(n_valid))


def team_round(
    states: Sequence[AgentState],
    n_agents: int,
    focal_slot: int | None,
    w_star: str | None,
    root_classes: Mapping[int, str | None],
    *,
    root_states: Mapping[int, str] | None = None,
    prev_h_lo: float = 0.0,
    prev_h_hi: float = 0.0,
    prev_q_lo: float | None = None,
    c0: int | None = None,
) -> dict[str, Any]:
    """One ``team_rounds`` row's measured columns (spec 4.2-4.5).

    ``root_classes``/``root_states`` are the same team's round-0 classes and states, which is
    what makes "a non-focal agent that started wrong now holds a correct class" a *within-team*
    statement rather than a population rate.  ``prev_h_lo``/``prev_h_hi`` carry historical
    coverage forward, so ``h_*`` is nondecreasing by construction (spec 18.3
    ``test_historical_coverage_is_nondecreasing``) and ``q_* <= h_*`` always
    (``test_active_coverage_is_bounded_by_historical_coverage``).
    """
    slots = [s.slot for s in states]
    if len(set(slots)) != len(slots):
        raise ContractViolation(f"team round holds slot(s) twice: {sorted(slots)}")
    if len(states) != int(n_agents):
        raise ContractViolation(
            f"team round holds {len(states)} slots but the team has {n_agents} agents; a missing "
            "slot is an incomplete round, never a smaller team (spec 6.4)")
    counts = Counter(s.state for s in states)
    c, w, u = counts[STATE_CORRECT], counts[STATE_WRONG], counts[STATE_UNRESOLVED]
    if c + w + u != n_agents:
        raise ContractViolation(f"C+W+U = {c + w + u} != N = {n_agents}")
    n_valid = sum(1 for s in states if s.valid)
    fully_graded = u == 0

    q_lo = 1.0 if c > 0 else 0.0
    q_hi = 1.0 if c + u > 0 else 0.0
    h_lo = max(float(prev_h_lo), q_lo)
    h_hi = max(float(prev_h_hi), q_hi)

    class_counts = _class_counts(states)
    correct_classes = {s.cls for s in states if s.state == STATE_CORRECT and s.cls is not None}
    wrong_classes = {s.cls for s in states if s.state == STATE_WRONG and s.cls is not None}
    top = max(class_counts.values()) if class_counts else 0
    winners = sorted(cls for cls, n in class_counts.items() if n == top) if class_counts else []
    if not winners:
        plurality_status, plurality_class = PLURALITY_NONE, None
    elif len(winners) > 1:
        plurality_status, plurality_class = PLURALITY_TIED, None
    elif top * 2 > n_agents:
        plurality_status, plurality_class = PLURALITY_STRICT_MAJORITY, winners[0]
    else:
        plurality_status, plurality_class = PLURALITY_UNIQUE, winners[0]
    if winners and all(cls in correct_classes or cls in wrong_classes for cls in winners):
        plurality_correct = sum(1.0 for cls in winners if cls in correct_classes) / len(winners)
    else:
        plurality_correct = None            # an unresolved winner is unknown, never zero

    strict_majority_correct = bool(
        plurality_status == PLURALITY_STRICT_MAJORITY and plurality_class in correct_classes)
    unanimous_correct = bool(n_valid == n_agents and len(class_counts) == 1
                             and next(iter(class_counts)) in correct_classes)
    # Wrong unanimity needs one *shared wrong* class: five agents that are each wrong but hold
    # five different classes are not a unanimity, and five unresolved agents are not either.
    wrong_unanimity = bool(n_valid == n_agents and len(class_counts) == 1
                           and next(iter(class_counts)) in wrong_classes
                           and next(iter(class_counts)) not in correct_classes)

    wrong_counts = Counter(s.cls for s in states if s.state == STATE_WRONG and s.cls is not None)
    w_total = sum(wrong_counts.values())
    max_wrong = max(wrong_counts.values()) if wrong_counts else 0

    root_states = dict(root_states or {})
    nonfocal = [s for s in states if focal_slot is None or s.slot != focal_slot]
    n_init_wrong = sum(1 for s in nonfocal if root_states.get(s.slot) == STATE_WRONG)
    n_init_correct = sum(1 for s in nonfocal if root_states.get(s.slot) == STATE_CORRECT)
    now_c = sum(1 for s in nonfocal
                if root_states.get(s.slot) == STATE_WRONG and s.state == STATE_CORRECT)
    now_wstar = sum(1 for s in nonfocal
                    if root_states.get(s.slot) == STATE_CORRECT and w_star is not None
                    and s.cls == w_star)
    nonfocal_unknown = sum(1 for s in nonfocal if s.state == STATE_UNRESOLVED)

    focal = next((s for s in states if focal_slot is not None and s.slot == focal_slot), None)
    row: dict[str, Any] = {
        "n_agents": int(n_agents),
        "c_count": int(c), "w_count": int(w), "u_count": int(u),
        "n_valid": int(n_valid), "fully_graded": bool(fully_graded),
        "class_hist_json": json.dumps({str(k): int(v) for k, v in sorted(class_counts.items())},
                                      sort_keys=True),
        "slot_classes_json": json.dumps({str(s.slot): s.cls for s in sorted(states, key=lambda x: x.slot)},
                                        sort_keys=True),
        "m_lo": c / n_agents, "m_hi": (c + u) / n_agents,
        "q_lo": q_lo, "q_hi": q_hi, "h_lo": h_lo, "h_hi": h_hi,
        "plurality_class": plurality_class,
        "plurality_status": plurality_status,
        "plurality_correct": plurality_correct,
        "strict_majority_correct": strict_majority_correct,
        "unanimous_correct": unanimous_correct,
        "wrong_unanimity": wrong_unanimity,
        "max_class_share": (top / n_valid) if n_valid else None,
        "class_entropy_norm": _entropy_norm(class_counts, n_valid),
        "wrong_class_max_share": (max_wrong / w_total) if w_total else None,
        "wrong_collision_prob": (math.fsum((n / w_total) ** 2 for n in wrong_counts.values())
                                 if w_total else None),
        "arithmetic_correct_minority": bool(0 < c < n_agents / 2),
        "outnumbered_correct": bool(c > 0 and c < max_wrong),
        "all_singleton": bool(class_counts and all(n == 1 for n in class_counts.values())),
        "w_star_class": w_star,
        "c0": None if c0 is None else int(c0),
        "focal_slot": focal_slot,
        "focal_class": None if focal is None else focal.cls,
        "focal_retained_root_class": (None if focal is None
                                      else bool(focal.cls is not None
                                                and focal.cls == root_classes.get(focal.slot))),
        "focal_state": None if focal is None else focal.state,
        "nonfocal_init_wrong_now_c": (now_c / n_init_wrong) if n_init_wrong else None,
        "nonfocal_init_correct_now_wstar": (now_wstar / n_init_correct) if n_init_correct else None,
        "n_nonfocal_init_wrong": int(n_init_wrong),
        "n_nonfocal_init_correct": int(n_init_correct),
        "nonfocal_unknown": int(nonfocal_unknown),
    }
    # Recovery (spec 4.5): the team never held a correct answer before this round and holds
    # one now.  ``prev_q_lo is None`` marks the root round, which cannot recover from anything.
    row["recovered_lo"] = bool(prev_q_lo is not None and float(prev_h_lo) == 0.0 and q_lo == 1.0)
    return row


# --------------------------------------------------------------------------- transitions


def _shingles(text: str | None, width: int = COPY_NGRAM) -> frozenset[tuple[str, ...]]:
    tokens = (text or "").split()
    if len(tokens) < width:
        return frozenset()
    return frozenset(tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def transitions(
    prev: Sequence[AgentState],
    cur: Sequence[AgentState],
    visible_classes_by_slot: Mapping[int, Sequence[str | None]],
    prev_rationales: Mapping[int, str | None],
    *,
    w_star: str | None = None,
    cur_rationales: Mapping[int, str | None] | None = None,
    received_slots_by_slot: Mapping[int, Sequence[int]] | None = None,
    placebo_rationales: Sequence[str | None] = (),
) -> list[dict[str, Any]]:
    """One row per slot for the move ``r -> r+1`` (plan §4.1 derived ``transitions``).

    ``copied_exact`` and ``ngram_overlap_max`` are measured against the rationales this agent
    actually received; ``placebo_overlap_max`` is the *same* statistic against same-item
    rationales it did not receive, so a baseline stylistic overlap cannot be read as copying.
    """
    by_slot_prev = {s.slot: s for s in prev}
    by_slot_cur = {s.slot: s for s in cur}
    if set(by_slot_prev) != set(by_slot_cur):
        raise ContractViolation(
            f"a transition must match the same agent slots: {sorted(by_slot_prev)} -> {sorted(by_slot_cur)}")
    cur_rationales = dict(cur_rationales or {})
    received_slots_by_slot = dict(received_slots_by_slot or {})
    placebo = [_shingles(text) for text in placebo_rationales]
    rows: list[dict[str, Any]] = []
    for slot in sorted(by_slot_prev):
        a, b = by_slot_prev[slot], by_slot_cur[slot]
        visible = [c for c in visible_classes_by_slot.get(slot, ()) if c is not None]
        received = list(received_slots_by_slot.get(slot, ()))
        own = _shingles(cur_rationales.get(slot))
        received_texts = [prev_rationales.get(s) for s in received] or \
            [prev_rationales.get(s) for s in by_slot_prev if s != slot and s in prev_rationales]
        own_text = cur_rationales.get(slot)
        copied_exact = bool(own_text) and any(
            text is not None and text == own_text for text in received_texts)
        overlaps = [_jaccard(own, _shingles(text)) for text in received_texts if text]
        rows.append({
            "agent_slot": int(slot),
            "state_from": a.state,
            "state_to": b.state,
            "kind": f"{a.state}{b.state}",
            "cls_from": a.cls,
            "cls_to": b.cls,
            "switched_class": bool(a.valid and b.valid and a.cls is not None
                                   and b.cls is not None and a.cls != b.cls),
            "validity_change": bool(a.valid != b.valid),
            "adopted_wstar": bool(w_star is not None and b.cls == w_star and a.cls != w_star),
            "adopted_visible_class": bool(b.cls is not None and b.cls != a.cls and b.cls in visible),
            "copied_exact": copied_exact,
            "ngram_overlap_max": (max(overlaps) if overlaps else None),
            "placebo_overlap_max": (max((_jaccard(own, p) for p in placebo), default=None)
                                    if placebo else None),
            "confidence_from": a.confidence,
            "confidence_to": b.confidence,
        })
    return rows


def transition_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Counts of the nine kinds, every kind present (a zero is a measurement, not a gap)."""
    counts = Counter(str(r["kind"]) for r in rows)
    unknown = sorted(set(counts) - set(TRANSITION_KINDS))
    if unknown:
        raise ContractViolation(f"unknown transition kind(s) {unknown}")
    return {kind: int(counts.get(kind, 0)) for kind in TRANSITION_KINDS}


def assert_three_state_identity(
    team_prev: Mapping[str, Any],
    team_cur: Mapping[str, Any],
    trans_rows: Sequence[Mapping[str, Any]],
) -> int:
    """``C_{r+1} - C_r == N_WC + N_UC - N_CW - N_CU``; a nonzero residual is FATAL.

    This is the bookkeeping identity of a three-state count.  It cannot fail unless the
    transition rows and the team counts were built from different populations -- which is
    exactly the defect it exists to catch -- so it raises rather than reports (plan D13).
    """
    counts = transition_counts(trans_rows)
    delta = int(team_cur["c_count"]) - int(team_prev["c_count"])
    predicted = counts["WC"] + counts["UC"] - counts["CW"] - counts["CU"]
    residual = delta - predicted
    if residual != 0:
        raise ContractViolation(
            f"three-state identity violated: C_(r+1) - C_r = {delta} but "
            f"N_WC + N_UC - N_CW - N_CU = {predicted} (residual {residual}); counts {counts}")
    return 0


def alpha_beta(trans_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``alpha_r = N_WC / N_W`` and ``beta_r = N_CW / N_C`` on the U-free sub-panel (spec 4.3).

    Pairs touching ``U`` on either side are excluded rather than folded in, and the excluded
    count is returned, so the denominator is never silently the whole team.
    """
    clean = [r for r in trans_rows if "U" not in str(r["kind"])]
    counts = transition_counts(clean) if clean else {k: 0 for k in TRANSITION_KINDS}
    n_w = counts["WC"] + counts["WW"]
    n_c = counts["CC"] + counts["CW"]
    return {
        "alpha": (counts["WC"] / n_w) if n_w else None,
        "beta": (counts["CW"] / n_c) if n_c else None,
        "n_w": n_w, "n_c": n_c,
        "n_excluded_unresolved": len(trans_rows) - len(clean),
    }


def drift_residual(team_prev: Mapping[str, Any], team_cur: Mapping[str, Any],
                   alpha: float | None, beta: float | None) -> float | None:
    """``E[C_{r+1} - C_r | C_r = k] = (N - k)alpha - k*beta``; the observed minus the predicted."""
    if alpha is None or beta is None:
        return None
    n = int(team_prev["n_agents"])
    k = int(team_prev["c_count"])
    return float((int(team_cur["c_count"]) - k) - ((n - k) * alpha - k * beta))


# --------------------------------------------------------------------------- decomposition


def failure_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``1 - A_R = (1 - H_R) + (H_R - Q_R) + (Q_R - A_R)``, exact on fully graded teams.

    ``rows`` are terminal-round ``team_rounds`` rows carrying ``a_lo``.  Teams that are not
    fully graded are excluded and counted: the identity is an identity only where every slot
    has a resolved state (spec 18.3 ``test_failure_decomposition_on_fully_graded_fixture``).
    """
    usable = [r for r in rows if r.get("fully_graded") and r.get("a_lo") is not None]
    if not usable:
        return {"n": 0, "n_excluded": len(rows), "no_history": None, "lost": None,
                "selection": None, "residual": None, "a": None}
    h = sum(float(r["h_lo"]) for r in usable) / len(usable)
    q = sum(float(r["q_lo"]) for r in usable) / len(usable)
    a = sum(float(r["a_lo"]) for r in usable) / len(usable)
    terms = {"no_history": 1.0 - h, "lost": h - q, "selection": q - a}
    residual = (1.0 - a) - math.fsum(terms.values())
    return {"n": len(usable), "n_excluded": len(rows) - len(usable), "a": a,
            "h": h, "q": q, **terms, "residual": float(residual)}


def coverage_bounds(rows: Sequence[Mapping[str, Any]], column: str = "q") -> dict[str, float]:
    """The ``[lo, hi]`` interval a set of team rounds admits under every labelling of ``U``."""
    if not rows:
        return {"lo": 0.0, "hi": 0.0, "n": 0}
    lo = sum(float(r[f"{column}_lo"]) for r in rows) / len(rows)
    hi = sum(float(r[f"{column}_hi"]) for r in rows) / len(rows)
    if lo > hi + 1e-12:
        raise ContractViolation(f"{column}_lo {lo} exceeds {column}_hi {hi}")
    return {"lo": lo, "hi": hi, "n": len(rows)}


# --------------------------------------------------------------------------- derive


def _read(tables_dir: Path, name: str) -> list[dict[str, Any]]:
    try:
        return tables.read_table(tables_dir, name)
    except MaterializeError:
        return []


@dataclass(frozen=True)
class DeriveInputs:
    """Everything :func:`derive` reads, resolved once so the walk order is visible."""

    decisions: tuple[dict[str, Any], ...]
    grades: Mapping[str, Mapping[str, Any]]
    root_teams: Mapping[str, Mapping[str, Any]]
    root_private: Mapping[str, Mapping[str, Any]]
    transmissions: tuple[dict[str, Any], ...]
    selections_graded: tuple[dict[str, Any], ...]


def load_inputs(run_root: str | os.PathLike, *, tables_dir: str | os.PathLike | None = None,
                derived_dir: str | os.PathLike | None = None) -> DeriveInputs:
    root = Path(run_root)
    pub = Path(tables_dir) if tables_dir is not None else root / TABLES_DIR
    prot = root / PROTECTED_TABLES_DIR
    der = Path(derived_dir) if derived_dir is not None else root / DERIVED_DIR
    grades: dict[str, dict[str, Any]] = {}
    for row in _read(prot, "grades_private"):
        grades.setdefault(str(row["generation_uid"]), row)
    return DeriveInputs(
        decisions=tuple(_read(pub, "decisions")),
        grades=grades,
        root_teams={str(r["root_team_id"]): r for r in _read(pub, "root_teams")},
        root_private={str(r["root_team_id"]): r for r in _read(prot, "root_teams_private")},
        transmissions=tuple(_read(pub, "transmissions")),
        selections_graded=tuple(_read(der, "selections_graded")),
    )


def _episode_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["cell_kind"]), str(row["episode_id"])


def _experiment_of(row: Mapping[str, Any]) -> str:
    """The experiment label a decision row belongs to, read off its cell id's first field.

    ``cells.cell_id_for`` puts the kind first and the experiment tag in ``extra``; until WP4
    lands, the cell id's leading kind is the only label that always exists, so it is used and
    the (richer) experiment tag is filled from the cell id when present.
    """
    cell_id = str(row.get("cell_id") or "")
    parts = [p for p in cell_id.split("/") if p]
    return parts[0] if parts else str(row["cell_kind"])


def derive(run_root: str | os.PathLike, *, tables_dir: str | os.PathLike | None = None,
           out_dir: str | os.PathLike | None = None,
           score_rounds: Sequence[int] | None = None) -> dict[str, Any]:
    """Write ``derived/{team_rounds,agent_rounds,transitions}.parquet`` and report the census.

    The three-state identity is asserted on every consecutive pair as the rows are built, so a
    derived directory can never be published with a nonzero residual (plan §4.5 gate 11).

    ``score_rounds`` is ``contract.team.score_rounds``: the rounds at which outcomes are
    scored, which is **not** the same as the rounds an episode generated.  Spec §11.2 runs the
    horizon extension to ``r=8`` once and scores prefixes, so an episode that generated eight
    revisions still contributes rows only at the declared checkpoints, and a transition pairs
    *consecutive scored* rounds.  Passing ``None`` scores every round present, which is what a
    fixture wants and what the census reports as ``unscored_rounds == 0``.
    """
    root = Path(run_root)
    out = Path(out_dir) if out_dir is not None else root / DERIVED_DIR
    data = load_inputs(run_root, tables_dir=tables_dir, derived_dir=out)

    by_round: dict[tuple[str, str], dict[int, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict))
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in data.decisions:
        kind = str(row["cell_kind"])
        try:
            cell_kind = CellKind(kind)
        except ValueError:
            continue
        if cell_kind not in OUTCOME_KINDS or str(row.get("phase")) in NON_ANSWER_PHASES:
            continue
        key = _episode_key(row)
        by_round[key][int(row["round"])][int(row["agent_slot"])] = row
        meta.setdefault(key, row)

    # received packets per receiving decision, for the visible-class and copying statistics
    received: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in data.transmissions:
        received[str(t["receiver_decision_id"])].append(t)
    reasoning_by_decision = {str(r["decision_id"]): r.get("public_reasoning") for r in data.decisions}
    class_by_decision = {str(r["decision_id"]): r for r in data.decisions}
    selected = {(str(r["cell_kind"]), str(r["episode_id"]), int(r["round"])): r
                for r in data.selections_graded}
    reasoning_by_item: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for r in data.decisions:
        reasoning_by_item[str(r["source_item_id"])].append(
            (str(r["episode_id"]), r.get("public_reasoning")))

    scored = None if score_rounds is None else {int(r) for r in score_rounds}
    unscored = 0
    team_rows: list[tables.TeamRoundRow] = []
    agent_rows: list[tables.AgentRoundRow] = []
    trans_rows: list[tables.TransitionRow] = []
    residual_checks = 0

    for key in sorted(by_round):
        cell_kind, episode_id = key
        head = meta[key]
        team_id = str(head.get("root_team_id") or "")
        team = data.root_teams.get(team_id, {})
        private = data.root_private.get(team_id, {})
        focal_slot = team.get("focal_slot")
        focal_slot = None if focal_slot is None else int(focal_slot)
        w_star = private.get("w_star_class")
        c0 = private.get("c0")
        composition = private.get("composition")
        rounds = sorted(by_round[key])
        if scored is not None:
            kept = [r for r in rounds if r in scored]
            unscored += len(rounds) - len(kept)
            rounds = kept
        if not rounds:
            continue
        states_by_round: dict[int, list[AgentState]] = {}
        for r in rounds:
            slots = by_round[key][r]
            states_by_round[r] = [
                agent_state(slots[s], data.grades.get(str(slots[s]["generation_uid"])))
                for s in sorted(slots)
            ]
        root_states = {s.slot: s.state for s in states_by_round.get(rounds[0], ())}
        root_classes = {s.slot: s.cls for s in states_by_round.get(rounds[0], ())}

        h_lo = h_hi = 0.0
        continuous = True
        prev_q_lo: float | None = None
        prev_team: dict[str, Any] | None = None
        prev_states: list[AgentState] | None = None
        prev_round: int | None = None
        for r in rounds:
            states = states_by_round[r]
            n_agents = len(states)
            row = team_round(states, n_agents, focal_slot, w_star, root_classes,
                             root_states=root_states, prev_h_lo=h_lo, prev_h_hi=h_hi,
                             prev_q_lo=prev_q_lo, c0=c0)
            h_lo, h_hi = row["h_lo"], row["h_hi"]
            continuous = bool(continuous and row["q_lo"] > 0)
            sel = selected.get((cell_kind, episode_id, r))
            a_lo = None if sel is None or sel.get("selected_correct") is None else float(sel["selected_correct"])
            baseline = bool(root_states and any(s == STATE_CORRECT for s in root_states.values()))
            hazard_at_risk = prev_team is not None and int(prev_team["c_count"]) > 0
            discovery_at_risk = prev_team is not None and int(prev_team["c_count"]) == 0
            team_rows.append(tables.TeamRoundRow(
                cell_kind=cell_kind, episode_id=episode_id, round=int(r),
                source_item_id=str(head["source_item_id"]),
                item_variant_id=str(head["item_variant_id"]),
                family=str(team.get("family") or head.get("family") or ""),
                experiment=_experiment_of(head),
                cohort=str(team.get("cohort") or ("natural" if not team_id else "")),
                composition=None if composition is None else str(composition),
                protocol_hash=str(head.get("protocol_hash") or ""),
                display_label=head.get("display_label"),
                model_hash=str(head["receiver_model_hash"]),
                budget_hash=str(head["budget_hash"]),
                replicate_id=int(head["replicate_id"]),
                a_lo=a_lo, fail_no_history=None, fail_lost=None, fail_selection=None,
                baseline_c0_positive=baseline,
                continuous_preservation_lo=(bool(continuous) if baseline else None),
                terminal_preservation_lo=(bool(row["q_lo"] > 0) if baseline else None),
                at_risk_hazard=bool(hazard_at_risk),
                hazard_event=(bool(row["c_count"] == 0) if hazard_at_risk else None),
                at_risk_discovery=bool(discovery_at_risk),
                discovery_event=(bool(row["c_count"] > 0) if discovery_at_risk else None),
                **{k: v for k, v in row.items() if k != "recovered_lo"},
                recovered_lo=bool(row["recovered_lo"]),
            ))
            for s in states:
                decision = by_round[key][r][s.slot]
                packets = received.get(str(decision["decision_id"]), ())
                agent_rows.append(tables.AgentRoundRow(
                    cell_kind=cell_kind, episode_id=episode_id, round=int(r), agent_slot=int(s.slot),
                    decision_id=s.decision_id, generation_uid=s.generation_uid,
                    source_item_id=str(head["source_item_id"]),
                    item_variant_id=str(head["item_variant_id"]),
                    family=str(team.get("family") or head.get("family") or ""),
                    experiment=_experiment_of(head),
                    cohort=str(team.get("cohort") or ("natural" if not team_id else "")),
                    composition=None if composition is None else str(composition),
                    protocol_hash=str(head.get("protocol_hash") or ""),
                    display_label=head.get("display_label"),
                    model_hash=str(head["receiver_model_hash"]),
                    budget_hash=str(head["budget_hash"]),
                    replicate_id=int(head["replicate_id"]),
                    state=s.state, cls=s.cls, valid=s.valid, u_reason=s.u_reason,
                    confidence=s.confidence, answer_logprob_mean=s.answer_logprob_mean,
                    retained_root_class=bool(s.cls is not None and s.cls == root_classes.get(s.slot)),
                    is_focal=bool(focal_slot is not None and s.slot == focal_slot),
                    received_slots_json=json.dumps(sorted(int(p["sender_slot"]) for p in packets)),
                ))
            if prev_team is not None and prev_states is not None and prev_round is not None:
                visible: dict[int, list[str | None]] = {}
                received_slots: dict[int, list[int]] = {}
                for s in states:
                    decision = by_round[key][r][s.slot]
                    packets = received.get(str(decision["decision_id"]), ())
                    received_slots[s.slot] = [int(p["sender_slot"]) for p in packets]
                    visible[s.slot] = [
                        (class_by_decision.get(str(p.get("source_decision_id")), {}) or {}).get("canonical_class")
                        for p in packets]
                prev_rationales = {p.slot: reasoning_by_decision.get(p.decision_id) for p in prev_states}
                cur_rationales = {s.slot: reasoning_by_decision.get(s.decision_id) for s in states}
                placebo = [text for eid, text in reasoning_by_item[str(head["source_item_id"])]
                           if eid != episode_id and text]
                rows = transitions(prev_states, states, visible, prev_rationales, w_star=w_star,
                                   cur_rationales=cur_rationales,
                                   received_slots_by_slot=received_slots, placebo_rationales=placebo)
                assert_three_state_identity(prev_team, row, rows)
                residual_checks += 1
                for t in rows:
                    trans_rows.append(tables.TransitionRow(
                        cell_kind=cell_kind, episode_id=episode_id, round=int(prev_round),
                        round_to=int(r), source_item_id=str(head["source_item_id"]), **t))
            prev_team, prev_states, prev_round, prev_q_lo = row, states, r, row["q_lo"]

    out.mkdir(parents=True, exist_ok=True)
    tables.write_derived_table(team_rows, out / "team_rounds.parquet")
    tables.write_derived_table(agent_rows, out / "agent_rounds.parquet")
    tables.write_derived_table(trans_rows, out / "transitions.parquet")
    return {
        "team_rounds": len(team_rows), "agent_rounds": len(agent_rows),
        "transitions": len(trans_rows), "identity_checks": residual_checks,
        "identity_residual": 0, "unscored_rounds": unscored, "out_dir": str(out),
    }


__all__ = [
    "AgentState", "COPY_NGRAM", "DERIVED_DIR", "DeriveInputs", "NON_ANSWER_PHASES",
    "OUTCOME_KINDS", "PLURALITY_NONE", "PLURALITY_STATUSES", "PLURALITY_STRICT_MAJORITY",
    "PLURALITY_TIED", "PLURALITY_UNIQUE", "PROTECTED_TABLES_DIR", "STATES", "STATE_CORRECT",
    "STATE_UNRESOLVED", "STATE_WRONG", "TABLES_DIR", "TRANSITION_KINDS", "agent_state",
    "alpha_beta", "assert_state_alphabet", "assert_three_state_identity", "coverage_bounds",
    "derive", "drift_residual", "failure_decomposition", "load_inputs", "team_round",
    "transition_counts", "transitions",
]
