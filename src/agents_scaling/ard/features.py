"""ARD36 frozen v1 behavioural feature schema (WP5).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §12.2
("Cheap feature set") and plan §5.4.  This module is the *only* place an ARD36 controller
feature is defined, and the schema is frozen at
:data:`agents_scaling.ard.contract.FEATURE_SCHEMA_VERSION`.

What the spec fixes, and what this module therefore does
-------------------------------------------------------

* **"Predefine a maximum of approximately twenty behavioral features."**  The nineteen
  entries of :data:`FEATURE_GROUPS` are those features, one group per spec bullet.  A
  group expands into one or more *design columns* — a value plus its missing flag, or a
  one-hot — and the flattened column list is :data:`FEATURES_V1`.  The group count, not
  the column count, is the quantity the spec caps; ``len(FEATURE_GROUPS) <= 20`` is a test.

* **"No item ID, reference answer, judged correctness, root pass rate, future stability,
  or final consensus feature is allowed."**  :data:`contract.FORBIDDEN_FEATURE_SOURCES` is
  the mechanical form of that sentence and :func:`assert_allowed_source` is the refusal.
  Every builder in this module runs its own output *and* any caller-supplied ``extra``
  payload through it, so a forbidden value cannot enter a row even by accident.  The check
  is on names because the values that carry gold in this study are always named (a graded
  ``is_correct``, a constructed ``C0``, a ``y_private`` label); a numeric column whose
  provenance is a gold join is a code-review question that the WP7 runtime column scan
  (plan §10) answers on the emitted ``features.parquet``, not one a name check can decide.

* **"Own previous confidence and confidence change; unavailable-history flags."**  At
  ``t = 0`` there is no previous round, so the three history columns take the frozen
  imputation :data:`HISTORY_IMPUTATION` (``0.0``) with ``history_unavailable = 1``.  That
  flag is *non-constant* in the fork frame precisely because the free ``t=0`` PRIVATE/PEER
  pairs are training rows beside the paid ``t=1``/``t=2`` forks (plan §5.4).

* **"Mean peer answer logprob, recorded only if runtime access is available to the
  controller."**  Answer logprob, thinking length and cap-hit live on the
  :class:`~agents_scaling.ard.client.DecisionRecord`, not on the report, so they arrive
  through an injected ``records`` mapping (``decision_id -> record``).  Absent access, the
  value is the frozen ``0.0`` imputation and its companion missing flag is set — never a
  silent zero.

* Spec §18 ``features.parquet`` is ``decision_or_state_id, feature_time,
  last_allowed_round, feature_schema_version, behavior_features, neural_feature_file,
  model_revision, training_eligibility``; :meth:`FeatureRow.to_row` emits exactly that,
  with ``behavior_features`` the JSON object of :data:`FEATURES_V1`.

The gold firewall (contract §0.2) is why the canonical classifier is *injected* through
``RoundState.classify`` rather than imported: this module never reaches
``ard.graders`` and therefore never reaches ``benchmarks`` or ``sympy``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from agents_scaling.ard import contract as K

__all__ = [
    "BEHAVIOUR_FEATURE_CAP",
    "DYNAMICS_ONLY_FEATURES",
    "FEATURES_V1",
    "FEATURE_GROUPS",
    "FEATURE_TIME_POST_ANSWER",
    "FEATURE_TIME_PRE_UPDATE",
    "FEATURE_TIMES",
    "FeatureFirewallError",
    "FeatureGroup",
    "FeatureRow",
    "HISTORY_IMPUTATION",
    "NO_FAMILY_FEATURES",
    "TRAINING_ELIGIBLE_SPLITS",
    "assert_allowed_source",
    "assert_no_forbidden_source",
    "build_feature_row",
    "feature_vector",
    "features_from_state",
    "peer_class_entropy",
    "team_features",
    "training_eligibility",
]

#: Spec §12.2 "a maximum of approximately twenty behavioral features".
BEHAVIOUR_FEATURE_CAP: Final[int] = 20

#: The frozen imputation of every history feature at ``t = 0`` (plan §5.4).
HISTORY_IMPUTATION: Final[float] = 0.0

FEATURE_TIME_PRE_UPDATE: Final[str] = "pre_update"
FEATURE_TIME_POST_ANSWER: Final[str] = "post_answer"
#: ``features.parquet.feature_time``.  ``pre_update`` rows are the controller's
#: EXPOSURE_VALUE rows: everything they read exists before the round-(t+1) update, so they
#: may predict what that update produces.  ``post_answer`` rows are CURRENT rows: they read
#: the answer itself and may therefore only predict *that* answer's correctness (spec §14.4
#: states the rule for the neural anchors; it binds the behavioural rows identically).
FEATURE_TIMES: Final[tuple[str, ...]] = (FEATURE_TIME_PRE_UPDATE, FEATURE_TIME_POST_ANSWER)

#: Spec §12.1: "No branch, round, candidate, alternative option permutation, or challenge
#: team from a test item can enter training or calibration."
TRAINING_ELIGIBLE_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})


class FeatureFirewallError(ValueError):
    """A feature name or payload key reaches a forbidden source (spec §12.2 line 420)."""


# --------------------------------------------------------------------------- the schema


@dataclass(frozen=True)
class FeatureGroup:
    """One spec §12.2 bullet: a behavioural feature and the columns it expands into."""

    name: str
    columns: tuple[str, ...]
    doc: str
    #: False for the design indicators (round, family) and for the columns a
    #: dynamics-only selector may not read (confidence, logprob).
    dynamics_only: bool = False


FEATURE_GROUPS: Final[tuple[FeatureGroup, ...]] = (
    FeatureGroup(
        "own_confidence", ("own_confidence", "own_confidence_missing"),
        "Own current confidence and its missing flag; missing is never imputed to 0.5.",
    ),
    FeatureGroup(
        "own_answer_logprob", ("own_answer_logprob", "own_answer_logprob_missing"),
        "Mean answer-token logprob of the own current report (runtime access required).",
    ),
    FeatureGroup(
        "own_answer_length", ("own_answer_length",),
        "Character length of the delivered answer span.", True,
    ),
    FeatureGroup(
        "own_previous_confidence",
        ("prev_confidence", "prev_confidence_delta", "history_unavailable"),
        "Previous-round confidence, its change, and the unavailable-history flag.",
    ),
    FeatureGroup(
        "changed_since_prev", ("changed_since_prev",),
        "Did the own answer change since the previous round (class, else content hash).",
        True,
    ),
    FeatureGroup(
        "changed_since_root", ("changed_since_root",),
        "Did the own answer change since the root report.", True,
    ),
    FeatureGroup(
        "own_class_count", ("own_class_count",),
        "How many of the five current answers share the own canonical class.", True,
    ),
    FeatureGroup(
        "own_class_count_prior", ("own_class_count_prior",),
        "The same count one round earlier (frozen imputation at t=0).", True,
    ),
    FeatureGroup(
        "peer_unique_classes", ("peer_unique_classes",),
        "Number of distinct canonical classes among the four peers.", True,
    ),
    FeatureGroup(
        "peer_class_entropy", ("peer_class_entropy",),
        "Shannon entropy (base 2) of the peer canonical-class distribution.", True,
    ),
    FeatureGroup(
        "largest_competing_count", ("largest_competing_count",),
        "Size of the largest peer class that is not the own class.", True,
    ),
    FeatureGroup(
        "count_margin", ("count_margin",),
        "own_class_count minus largest_competing_count.", True,
    ),
    FeatureGroup(
        "peer_confidence",
        ("peer_confidence_mean", "peer_confidence_max", "own_minus_peer_confidence",
         "peer_confidence_missing"),
        "Mean/max observable peer confidence and own-minus-peer, with a missing flag; the "
        "controller's logged access is part of its frozen interface (spec §12.2).",
    ),
    FeatureGroup(
        "peer_answer_logprob",
        ("peer_answer_logprob_mean", "peer_answer_logprob_missing"),
        "Mean peer answer logprob, recorded only when runtime access is available.",
    ),
    FeatureGroup(
        "thinking_length", ("thinking_length",),
        "Realized reasoning length of the own current decision, in tokens.", True,
    ),
    FeatureGroup(
        "cap_hit", ("cap_hit",),
        "Whole-response cap flag of the own current decision.", True,
    ),
    FeatureGroup(
        "valid", ("valid",),
        "Output-validity flag: the own report delivered a gradeable answer.", True,
    ),
    FeatureGroup(
        "round_index", tuple(f"round_is_{t}" for t in K.ROUNDS),
        "Round index, one-hot over contract.ROUNDS.",
    ),
    FeatureGroup(
        "family", tuple(f"family_is_{f.value}" for f in K.FAMILIES),
        "Task-family indicator; spec §12.2 also requires a no-family model and "
        "leave-one-family-out transfer, which NO_FAMILY_FEATURES supports.",
    ),
)

#: The frozen v1 design columns, in order.  Contract §2 names this ``FEATURES_V1``.
FEATURES_V1: Final[tuple[str, ...]] = tuple(
    column for group in FEATURE_GROUPS for column in group.columns
)

#: The same schema without the family indicators (spec §12.2 "also report a model without
#: them and leave-one-family-out transfer").
NO_FAMILY_FEATURES: Final[tuple[str, ...]] = tuple(
    column for group in FEATURE_GROUPS if group.name != "family" for column in group.columns
)

#: Spec §12.4's "dynamics-only static selector": counts, changes and structure, with no
#: self-reported confidence and no logprob — the Minority-Sentinel-inspired comparator.
DYNAMICS_ONLY_FEATURES: Final[tuple[str, ...]] = tuple(
    column for group in FEATURE_GROUPS if group.dynamics_only for column in group.columns
) + tuple(f"round_is_{t}" for t in K.ROUNDS)


# --------------------------------------------------------------------------- firewall


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(name: str) -> set[str]:
    return set(_WORD.findall(str(name).lower()))


_FORBIDDEN_LOWER: Final[tuple[str, ...]] = tuple(
    sorted({source.lower() for source in K.FORBIDDEN_FEATURE_SOURCES})
)
#: Each forbidden source as its word-token sequence (``"root_pass_rate"`` →
#: ``("root", "pass", "rate")``).
_FORBIDDEN_SEQUENCES: Final[tuple[tuple[str, ...], ...]] = tuple(
    tuple(_WORD.findall(source)) for source in _FORBIDDEN_LOWER
)


def _token_list(name: str) -> list[str]:
    return _WORD.findall(str(name).lower())


def assert_allowed_source(name: str) -> str:
    """Refuse ``name`` when it names or contains a :data:`contract.FORBIDDEN_FEATURE_SOURCES`.

    Two rules, and both are deliberately mechanical (spec §12.2 line 420 is a prohibition
    on *sources*, so the check must not need re-adjudication per feature):

    1. the lowered name may not contain a forbidden source as a substring — this catches
       the compact spellings ``constructedC0``, ``goldanswer``;
    2. the name's word tokens may not contain a forbidden source's token sequence
       *contiguously* — ``is_correct`` → ``("is", "correct")``, ``y_peer`` →
       ``("y", "peer")``.

    Contiguity is what keeps the rule usable: ``changed_since_root`` is a legitimate
    behavioural feature and must not be refused merely because ``root_pass_rate`` starts
    with the word ``root``, while ``root_pass_rate`` itself is refused by both rules.
    """
    text = str(name).lower()
    for source in _FORBIDDEN_LOWER:
        if source in text:
            raise FeatureFirewallError(
                f"feature {name!r} is derived from the forbidden source {source!r} "
                "(spec §12.2: no item id, gold, correctness, pass rate or future state)"
            )
    tokens = _token_list(text)
    for sequence in _FORBIDDEN_SEQUENCES:
        width = len(sequence)
        if not width:
            continue
        for start in range(0, len(tokens) - width + 1):
            if tuple(tokens[start:start + width]) == sequence:
                raise FeatureFirewallError(
                    f"feature {name!r} reaches the forbidden source "
                    f"{'_'.join(sequence)!r} (spec §12.2 line 420)"
                )
    return str(name)


def assert_no_forbidden_source(payload: Mapping[str, Any], *, what: str = "feature row") -> None:
    """Run every key of ``payload`` through :func:`assert_allowed_source`."""
    for key in payload:
        try:
            assert_allowed_source(key)
        except FeatureFirewallError as exc:
            raise FeatureFirewallError(f"{what}: {exc}") from None


# The frozen schema must itself survive the firewall — checked at import, not at test time.
for _column in FEATURES_V1:
    assert_allowed_source(_column)
del _column


# --------------------------------------------------------------------------- helpers


def _record_of(records: Mapping[str, Any] | None, decision_id: str) -> Any | None:
    if not records:
        return None
    return records.get(str(decision_id))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def peer_class_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy (base 2) of a class-count multiset; ``0.0`` for 0 or 1 class."""
    total = float(sum(int(c) for c in counts))
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        p = float(count) / total
        if p > 0.0:
            out -= p * math.log(p, 2)
    return float(out)


def _class_or_none(state: Any, report: Any) -> str | None:
    if report is None:
        return None
    try:
        return state.canonical_class(report)
    except Exception:  # pragma: no cover - a classifier defect must not fake a feature
        return None


def _changed(state: Any, current: Any, other: Any) -> float:
    """Answer-change flag: canonical class when both are keyed, else the content hash.

    Plan §5.8 records the change flag under *both* definitions; the feature uses the class
    where the class exists (that is the dynamics quantity) and falls back to the report
    content hash when a side is unkeyable, so an invalid→invalid pair with different bytes
    is still a change.
    """
    if other is None:
        return 0.0
    a, b = _class_or_none(state, current), _class_or_none(state, other)
    if a is not None and b is not None:
        return 1.0 if a != b else 0.0
    return 1.0 if current.content_sha256 != other.content_sha256 else 0.0


def _previous_ref(state: Any, agent_slot: int, round_: int) -> Any | None:
    """The agent's report at round ``round_ - 1``, from the archive, or ``None`` at t=0."""
    if int(round_) <= 0:
        return None
    for ref in state.reports_of_round(int(round_) - 1):
        if int(ref.agent_slot) == int(agent_slot):
            return ref
    return None


# --------------------------------------------------------------------------- the builder


def features_from_state(
    state: Any,
    agent_slot: int,
    *,
    round_: int,
    records: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """The frozen v1 feature vector of one agent in one round state (contract §2 WP5).

    ``state`` is a WP2 :class:`~agents_scaling.ard.states.RoundState` at the end of round
    ``round_``; the returned row is the controller's view *before* the round-``round_+1``
    update, which is the counterfactual feature boundary of spec §19 ("freeze before the
    intervention").  ``records`` maps ``decision_id -> DecisionRecord`` and supplies the
    three runtime quantities that are not on a report (answer logprob, thinking length,
    cap hit); without it those columns take the frozen imputation and their missing flags
    are set.

    Every value is a float, and every key is in :data:`FEATURES_V1`.
    """
    slot = int(agent_slot)
    round_ = int(round_)
    if round_ not in K.ROUNDS:
        raise ValueError(f"round must be one of {K.ROUNDS}, got {round_}")
    agent = state.agents[slot]
    own_ref = state.latest(slot)
    own = own_ref.report
    own_class = _class_or_none(state, own)
    own_record = _record_of(records, own_ref.decision_id)

    # ---- own report -------------------------------------------------------------
    own_conf = _float_or_none(own.confidence)
    own_logprob = _float_or_none(getattr(own_record, "answer_logprob", None))
    answer = own.final_answer or ""

    # ---- history ----------------------------------------------------------------
    prev_ref = _previous_ref(state, slot, round_)
    prev = None if prev_ref is None else prev_ref.report
    prev_conf = None if prev is None else _float_or_none(prev.confidence)
    history_unavailable = prev is None

    # ---- team counts ------------------------------------------------------------
    current_classes = [_class_or_none(state, state.latest(j).report) for j in range(K.N_AGENTS)]
    own_count = sum(1 for c in current_classes if own_class is not None and c == own_class)
    if history_unavailable:
        prior_count = HISTORY_IMPUTATION
    else:
        prior_refs = state.reports_of_round(round_ - 1)
        prior_classes = [_class_or_none(state, ref.report) for ref in prior_refs]
        prev_class = _class_or_none(state, prev)
        prior_count = float(
            sum(1 for c in prior_classes if prev_class is not None and c == prev_class)
        )

    # ---- peers ------------------------------------------------------------------
    peer_slots = [j for j in range(K.N_AGENTS) if j != slot]
    peer_refs = [state.latest(j) for j in peer_slots]
    peer_classes = [c for j, c in enumerate(current_classes) if j != slot and c is not None]
    peer_counts: dict[str, int] = {}
    for name in peer_classes:
        peer_counts[name] = peer_counts.get(name, 0) + 1
    competing = [n for name, n in peer_counts.items() if name != own_class]
    largest_competing = float(max(competing)) if competing else 0.0

    peer_confidences = [c for c in (_float_or_none(r.report.confidence) for r in peer_refs)
                        if c is not None]
    peer_logprobs = [
        v for v in (_float_or_none(getattr(_record_of(records, r.decision_id), "answer_logprob", None))
                    for r in peer_refs)
        if v is not None
    ]
    peer_conf_mean = sum(peer_confidences) / len(peer_confidences) if peer_confidences else HISTORY_IMPUTATION
    peer_conf_max = max(peer_confidences) if peer_confidences else HISTORY_IMPUTATION
    own_minus_peer = (
        float(own_conf) - float(peer_conf_mean)
        if own_conf is not None and peer_confidences else HISTORY_IMPUTATION
    )

    row: dict[str, float] = {
        "own_confidence": float(own_conf) if own_conf is not None else HISTORY_IMPUTATION,
        "own_confidence_missing": 0.0 if own_conf is not None else 1.0,
        "own_answer_logprob": float(own_logprob) if own_logprob is not None else HISTORY_IMPUTATION,
        "own_answer_logprob_missing": 0.0 if own_logprob is not None else 1.0,
        "own_answer_length": float(len(answer)),
        "prev_confidence": float(prev_conf) if prev_conf is not None else HISTORY_IMPUTATION,
        "prev_confidence_delta": (
            float(own_conf) - float(prev_conf)
            if own_conf is not None and prev_conf is not None else HISTORY_IMPUTATION
        ),
        "history_unavailable": 1.0 if history_unavailable else 0.0,
        "changed_since_prev": _changed(state, own, prev) if prev is not None else HISTORY_IMPUTATION,
        "changed_since_root": _changed(state, own, agent.root.report),
        "own_class_count": float(own_count),
        "own_class_count_prior": float(prior_count),
        "peer_unique_classes": float(len(peer_counts)),
        "peer_class_entropy": peer_class_entropy(list(peer_counts.values())),
        "largest_competing_count": largest_competing,
        "count_margin": float(own_count) - largest_competing,
        "peer_confidence_mean": float(peer_conf_mean),
        "peer_confidence_max": float(peer_conf_max),
        "own_minus_peer_confidence": float(own_minus_peer),
        "peer_confidence_missing": 0.0 if peer_confidences else 1.0,
        "peer_answer_logprob_mean": (
            float(sum(peer_logprobs) / len(peer_logprobs)) if peer_logprobs else HISTORY_IMPUTATION
        ),
        "peer_answer_logprob_missing": 0.0 if peer_logprobs else 1.0,
        "thinking_length": float(getattr(own_record, "thinking_tokens", 0) or 0),
        "cap_hit": 1.0 if bool(getattr(own_record, "cap_hit", False)) else 0.0,
        "valid": 1.0 if bool(own.valid) else 0.0,
    }
    for t in K.ROUNDS:
        row[f"round_is_{t}"] = 1.0 if int(t) == round_ else 0.0
    family = str(getattr(state, "family", "") or "")
    for member in K.FAMILIES:
        row[f"family_is_{member.value}"] = 1.0 if family == member.value else 0.0

    missing = [name for name in FEATURES_V1 if name not in row]
    if missing:  # pragma: no cover - a schema/builder disagreement is a defect
        raise FeatureFirewallError(f"builder did not emit the frozen columns {missing}")
    extra = [name for name in row if name not in FEATURES_V1]
    if extra:  # pragma: no cover
        raise FeatureFirewallError(f"builder emitted columns outside FEATURES_V1: {extra}")
    assert_no_forbidden_source(row)
    return {name: float(row[name]) for name in FEATURES_V1}


def team_features(
    state: Any, *, round_: int, records: Mapping[str, Any] | None = None
) -> tuple[dict[str, float], ...]:
    """:func:`features_from_state` for all five agents, indexed by true slot."""
    return tuple(
        features_from_state(state, slot, round_=round_, records=records)
        for slot in range(K.N_AGENTS)
    )


def feature_vector(row: Mapping[str, float], names: Sequence[str] = FEATURES_V1) -> list[float]:
    """``row`` as a dense vector in ``names`` order; a missing column is an error."""
    out: list[float] = []
    for name in names:
        if name not in row:
            raise FeatureFirewallError(f"feature row has no column {name!r}")
        out.append(float(row[name]))
    return out


def training_eligibility(split: str) -> str:
    """``eligible`` / ``excluded_test`` (spec §12.1: no test item enters training)."""
    return "eligible" if str(split) in TRAINING_ELIGIBLE_SPLITS else "excluded_test"


# --------------------------------------------------------------------------- the row


@dataclass(frozen=True)
class FeatureRow:
    """One ``features.parquet`` row (spec §18)."""

    decision_or_state_id: str
    feature_time: str
    last_allowed_round: int
    behavior_features: Mapping[str, float]
    model_revision: str = ""
    neural_feature_file: str | None = None
    training_eligibility: str = "eligible"
    feature_schema_version: int = K.FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.feature_time not in FEATURE_TIMES:
            raise ValueError(f"feature_time must be one of {FEATURE_TIMES}, got {self.feature_time!r}")
        if int(self.last_allowed_round) not in K.ROUNDS:
            raise ValueError(f"last_allowed_round must be one of {K.ROUNDS}")
        assert_no_forbidden_source(self.behavior_features)

    def to_row(self) -> dict[str, Any]:
        return {
            "decision_or_state_id": str(self.decision_or_state_id),
            "feature_time": str(self.feature_time),
            "last_allowed_round": int(self.last_allowed_round),
            "feature_schema_version": int(self.feature_schema_version),
            "behavior_features": json.dumps(
                {name: float(self.behavior_features[name]) for name in FEATURES_V1},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ),
            "neural_feature_file": self.neural_feature_file,
            "model_revision": str(self.model_revision),
            "training_eligibility": str(self.training_eligibility),
        }


def build_feature_row(
    state: Any,
    agent_slot: int,
    *,
    round_: int,
    split: str,
    records: Mapping[str, Any] | None = None,
    feature_time: str = FEATURE_TIME_PRE_UPDATE,
    row_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    neural_feature_file: str | None = None,
) -> FeatureRow:
    """Build one ``features.parquet`` row, refusing any forbidden source.

    ``extra`` is the caller's escape hatch for a future additive column; it is run through
    the firewall *before* it is merged, so a caller cannot smuggle ``is_correct``,
    ``constructed_C0`` or ``y_private`` into a feature row.  ``row_id`` defaults to the
    agent's ``state:<sha>`` id, which is the identity spec §18 joins on.
    """
    behaviour = features_from_state(state, agent_slot, round_=round_, records=records)
    if extra:
        assert_no_forbidden_source(extra, what="extra feature payload")
        for key, value in extra.items():
            if key in FEATURES_V1:
                raise FeatureFirewallError(
                    f"extra column {key!r} would overwrite the frozen v1 schema"
                )
            behaviour[str(key)] = float(value)
    identifier = row_id or state.agents[int(agent_slot)].state_id
    checkpoint = str(getattr(state, "checkpoint", "") or "")
    return FeatureRow(
        decision_or_state_id=str(identifier),
        feature_time=str(feature_time),
        last_allowed_round=int(round_),
        behavior_features=behaviour,
        model_revision=K.REVISIONS.get(checkpoint, ""),
        neural_feature_file=neural_feature_file,
        training_eligibility=training_eligibility(split),
    )
