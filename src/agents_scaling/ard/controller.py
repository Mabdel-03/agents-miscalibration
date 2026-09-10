"""ARD36 E4 controller: ``CURRENT``, ``EXPOSURE_VALUE`` and the frozen GATE policy (WP5).

Spec ``expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md`` §12.1-§12.3,
plan §5.4.  Three things live here and nothing else:

1. **Two L2-logistic heads on the fork frame.**  Spec §12.1: "Train two regularized
   logistic predictors, ``p_private`` and ``p_peer``, on branch correctness.  Their
   difference estimates ``u``."  The frame is the free ``t=0`` PRIVATE/PEER pairs plus the
   paid PEER forks at ``t=1,2`` (:mod:`agents_scaling.ard.forks`), so every row is a
   *pair* of outcomes for one shared prefix and every round at which GATE acts is covered.
   Fitting reuses the frozen study readout verbatim — ``readout.LogisticModel``,
   ``readout.grouped_stratified_folds`` (item-grouped, family-stratified folds) and
   ``readout.one_se_select`` over ``contract.PENALTY_GRID``.  Spec §12.1: "Maximum
   feature/hyperparameter choices are fixed below; do not search a large controller
   family" — the grid is those four penalties and nothing else.

2. **A validation-only comparator.**  Spec §12.1: "Compare to direct ridge regression on
   the paired {-1,0,1} difference as a validation-only model choice."  :class:`RidgeComparator`
   is fitted on training rows and *scored* on validation; it never enters the policy and
   never enters the frozen artifact's decision path.  The model choice — difference of two
   logistic heads — is preregistered, so the comparator is reported, not obeyed.

3. **``CURRENT`` and its baselines.**  Spec §12.2: "For CURRENT use a matching regularized
   correctness predictor.  Baselines: answer frequency, raw confidence,
   validation-recalibrated confidence, answer logprob, and frequency+confidence."  The
   recalibration is ``study/neural/calibration.fit_scope`` on the *validation* rows, so
   the recalibrated baseline is exactly what its name says.

Freezing and the ``controller:<sha>`` dependency
------------------------------------------------

:meth:`FrozenController.save` writes ``controller/<sha>/MODEL.json`` in the tamper-evident
canonical-JSON pattern of ``study/neural/readout.py`` ``FrozenG.save``/``load``
(lines 1029-1103): the payload is hashed as canonical JSON, the digest is stamped into the
object, and :func:`load_controller` recomputes it and refuses a modified artifact.  The
same rule is WP4's ``runner._content_digest`` rule 2, which is what makes
``depends_on: ["controller:<sha>"]`` a hard gate — a GATE cell whose artifact was edited
exits ``EXIT_DEPS_UNMET`` instead of running with an unknown policy.

The policy itself
-----------------

Spec §12.3: ``u[j] = p_private(x_j) - p_peer(x_j)``; ``protected = argmax_j u`` with ties
broken by a frozen random schedule.  That schedule is
``identity.blind_order_key(study_seed, contract.ORDER_PROTECT, pairing_id, round, slot)``
— the *same* HMAC that drives RANDOM_PROTECT's uniform choice in
``ard.protocols.random_protected_slot``, so the two arms share one tie-break and differ
only in the signal.  Spec §12.3 also fixes the language: this selects the agent *predicted
to benefit most from private continuation*, never "the correct agent", and there is no
test-tuned threshold — if every ``u`` is negative the controller still protects one agent,
which is the declared failure mode.

Nothing here reads a gold answer, a grade or an item id.  Outcomes arrive as columns of an
already-joined analysis frame that the caller builds on the evaluator side; the frame
builder refuses a test item and refuses to let ``y_private``/``y_peer`` reach the design
matrix (:meth:`PairedFrame.validate`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.neural import calibration as CAL
from agents_scaling.study.neural import readout as R
from agents_scaling.study.types import ProtocolError

__all__ = [
    "BASELINE_ANSWER_LOGPROB",
    "BASELINE_CONFIDENCE_RAW",
    "BASELINE_CONFIDENCE_RECALIBRATED",
    "BASELINE_FREQUENCY",
    "BASELINE_FREQUENCY_CONFIDENCE",
    "CONTROLLER_DIR",
    "CONTROLLER_FILE",
    "CONTROLLER_KIND",
    "ControllerError",
    "CurrentFrame",
    "CurrentRow",
    "FrozenController",
    "HEAD_CURRENT",
    "HEAD_DYNAMICS",
    "HEAD_PEER",
    "HEAD_PRIVATE",
    "PairedFrame",
    "PairedRow",
    "ProtectionChoice",
    "RidgeComparator",
    "brier_decomposition",
    "choose_protected",
    "controller_sha256",
    "evaluate_controller",
    "evaluate_current",
    "fit_controller",
    "freeze_controller",
    "load_controller",
    "protect_schedule_key",
    "within_item_auc",
]

#: Mirrors ``ard.runner.CONTROLLER_DIR`` / ``CONTROLLER_FILE`` (WP4 owns that module; the
#: values are asserted equal in ``tests/ard/test_wp5_controller.py`` so the freeze and the
#: dependency gate can never disagree about where the artifact lives).
CONTROLLER_DIR: Final[str] = "controller"
CONTROLLER_FILE: Final[str] = "MODEL.json"
CONTROLLER_KIND: Final[str] = "ARD36_CONTROLLER"
CONTROLLER_SCHEMA_VERSION: Final[int] = 1
SHA_FIELD: Final[str] = "controller_sha256"

HEAD_PRIVATE: Final[str] = "p_private"
HEAD_PEER: Final[str] = "p_peer"
HEAD_CURRENT: Final[str] = "CURRENT"
HEAD_DYNAMICS: Final[str] = "dynamics_only"

BASELINE_FREQUENCY: Final[str] = "baseline_frequency"
BASELINE_CONFIDENCE_RAW: Final[str] = "baseline_confidence_raw"
BASELINE_CONFIDENCE_RECALIBRATED: Final[str] = "baseline_confidence_recalibrated"
BASELINE_ANSWER_LOGPROB: Final[str] = "baseline_answer_logprob"
BASELINE_FREQUENCY_CONFIDENCE: Final[str] = "baseline_frequency_confidence"

#: Feature subsets of the frozen v1 schema used by the spec §12.2 baselines.
BASELINE_FEATURES: Final[Mapping[str, tuple[str, ...]]] = {
    BASELINE_FREQUENCY: ("own_class_count",),
    BASELINE_ANSWER_LOGPROB: ("own_answer_logprob", "own_answer_logprob_missing"),
    BASELINE_FREQUENCY_CONFIDENCE: (
        "own_class_count", "own_confidence", "own_confidence_missing",
    ),
}

DEFAULT_FOLDS: Final[int] = 5
FOLD_NAMESPACE: Final[str] = "ARD36_CONTROLLER_FOLDS"
_EPS: Final[float] = 1e-12


class ControllerError(ProtocolError):
    """The controller frame, fit or frozen artifact is not usable as specified."""


# --------------------------------------------------------------------------- frames


@dataclass(frozen=True)
class PairedRow:
    """One fork-frame row: features of a shared prefix plus both branch outcomes.

    ``y_private`` / ``y_peer`` are the *branch correctness* of spec §12.1 — one stochastic
    outcome per branch, "noisy supervision, not an exact individual causal effect".  They
    are outcomes, never features: :meth:`PairedFrame.validate` refuses a frame whose
    feature dict carries either name.
    """

    row_id: str
    item_id: str                 # the grouping cluster; never a feature
    family: str
    split: str
    round_: int
    agent_slot: int
    features: Mapping[str, float]
    y_private: int | None = None
    y_peer: int | None = None
    team_id: str = ""
    episode_id: str = ""
    parent_state_id: str = ""

    @property
    def team(self) -> str:
        return self.team_id or f"{self.item_id}|r{int(self.round_)}"

    @property
    def paired(self) -> bool:
        return self.y_private is not None and self.y_peer is not None

    @property
    def difference(self) -> int | None:
        """The paired ``{-1, 0, 1}`` difference the ridge comparator regresses on."""
        if not self.paired:
            return None
        return int(self.y_private) - int(self.y_peer)


@dataclass(frozen=True)
class CurrentRow:
    """One CURRENT row: post-answer features of a produced candidate plus its correctness."""

    row_id: str
    item_id: str
    family: str
    split: str
    round_: int
    agent_slot: int
    features: Mapping[str, float]
    y_correct: int | None = None
    episode_id: str = ""


def _validate_rows(rows: Sequence[Any], names: Sequence[str], *, what: str) -> None:
    if not rows:
        raise ControllerError(f"{what} is empty")
    outcomes = {"y_private", "y_peer", "y_correct"}
    for row in rows:
        if FE.training_eligibility(row.split) != "eligible":
            raise ControllerError(
                f"{what}: row {row.row_id!r} is from split {row.split!r}; spec §12.1 "
                "forbids any test item in training or calibration"
            )
        leaked = sorted(outcomes & set(row.features))
        if leaked:
            raise ControllerError(
                f"{what}: row {row.row_id!r} carries the outcome column(s) {leaked} in its "
                "features; the label is not a feature (spec §12.2)"
            )
        FE.assert_no_forbidden_source(row.features, what=f"{what} row {row.row_id!r}")
        missing = [n for n in names if n not in row.features]
        if missing:
            raise ControllerError(f"{what}: row {row.row_id!r} lacks the columns {missing}")


@dataclass(frozen=True)
class PairedFrame:
    """The E4 fork frame (spec §12.1) — one row per shared prefix and focal agent."""

    rows: tuple[PairedRow, ...]
    feature_names: tuple[str, ...] = FE.FEATURES_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    def validate(self) -> "PairedFrame":
        _validate_rows(self.rows, self.feature_names, what="fork frame")
        return self

    def __len__(self) -> int:
        return len(self.rows)

    def matrix(self, names: Sequence[str] | None = None) -> np.ndarray:
        names = tuple(names or self.feature_names)
        return np.asarray(
            [FE.feature_vector(row.features, names) for row in self.rows], dtype=np.float64
        )

    def y(self, which: str) -> np.ndarray:
        values = [getattr(row, which) for row in self.rows]
        return np.asarray([-1 if v is None else int(v) for v in values], dtype=np.int64)

    def observed(self, which: str) -> np.ndarray:
        return np.asarray([getattr(row, which) is not None for row in self.rows], dtype=bool)

    def clusters(self) -> list[str]:
        return [str(row.item_id) for row in self.rows]

    def strata(self) -> list[str]:
        return [str(row.family) for row in self.rows]

    def teams(self) -> list[str]:
        return [row.team for row in self.rows]

    def subset(self, mask: np.ndarray) -> "PairedFrame":
        keep = np.asarray(mask, dtype=bool)
        return PairedFrame(tuple(r for r, m in zip(self.rows, keep) if m), self.feature_names)


@dataclass(frozen=True)
class CurrentFrame:
    """The CURRENT frame — post-answer rows of already generated candidates."""

    rows: tuple[CurrentRow, ...]
    feature_names: tuple[str, ...] = FE.FEATURES_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    def validate(self) -> "CurrentFrame":
        _validate_rows(self.rows, self.feature_names, what="CURRENT frame")
        return self

    def __len__(self) -> int:
        return len(self.rows)

    def matrix(self, names: Sequence[str] | None = None) -> np.ndarray:
        names = tuple(names or self.feature_names)
        return np.asarray(
            [FE.feature_vector(row.features, names) for row in self.rows], dtype=np.float64
        )

    def y(self) -> np.ndarray:
        return np.asarray([-1 if r.y_correct is None else int(r.y_correct) for r in self.rows],
                          dtype=np.int64)

    def observed(self) -> np.ndarray:
        return np.asarray([r.y_correct is not None for r in self.rows], dtype=bool)

    def clusters(self) -> list[str]:
        return [str(row.item_id) for row in self.rows]

    def strata(self) -> list[str]:
        return [str(row.family) for row in self.rows]

    def confidence(self) -> np.ndarray:
        """Raw self-reported confidence, ``nan`` where it is missing (never 0.5)."""
        out = []
        for row in self.rows:
            missing = float(row.features.get("own_confidence_missing", 0.0)) > 0.5
            out.append(float("nan") if missing else float(row.features["own_confidence"]))
        return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- fitting


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("inf"), 0.0
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def _folds(clusters: Sequence[str], strata: Sequence[str], seed: bytes, n_folds: int) -> np.ndarray:
    n = min(int(n_folds), max(1, len(set(clusters))))
    return R.grouped_stratified_folds(clusters, strata, n, seed, namespace=FOLD_NAMESPACE)


def _select_penalty(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    strata: Sequence[str],
    *,
    seed: bytes,
    penalties: Sequence[float],
    n_folds: int,
) -> R.Selection:
    """One-SE penalty choice on item-grouped, family-stratified folds (spec §12.1)."""
    folds = _folds(clusters, strata, seed, n_folds)
    unique = sorted(set(int(f) for f in folds))
    candidates: list[tuple[tuple[int | None, int | None, float], float, float]] = []
    for penalty in penalties:
        losses: list[float] = []
        for fold in unique:
            test = folds == fold
            train = ~test
            if not train.any() or not test.any():
                continue
            model = R.LogisticModel(penalty=float(penalty)).fit(X[train], y[train])
            p = model.predict_proba(X[test])
            losses.append(float(np.mean((p - y[test]) ** 2)))
        mean, se = _mean_se(losses)
        candidates.append(((None, None, float(penalty)), mean, se))
    return R.one_se_select(candidates, block_order=())


def _fit_head(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[str],
    strata: Sequence[str],
    names: Sequence[str],
    *,
    seed: bytes,
    penalties: Sequence[float],
    n_folds: int,
) -> tuple[R.LogisticModel, R.Selection]:
    selection = _select_penalty(X, y, clusters, strata, seed=seed, penalties=penalties,
                               n_folds=n_folds)
    model = R.LogisticModel(penalty=float(selection.penalty), feature_names=list(names))
    model.fit(X, y)
    return model, selection


@dataclass
class RidgeComparator:
    """Validation-only ridge on the paired ``{-1, 0, 1}`` difference (spec §12.1).

    Reported beside ``u = p_private − p_peer``, never substituted for it: the model choice
    is preregistered, so this is a *comparator*, and its numbers are validation numbers.
    """

    penalty: float = 1.0
    feature_names: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, d: np.ndarray) -> "RidgeComparator":
        X = np.asarray(X, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > _EPS, std, 1.0)
        Z = (X - self.mean_) / self.std_
        n_features = Z.shape[1]
        A = Z.T @ Z + float(self.penalty) * np.eye(n_features)
        b = Z.T @ (d - d.mean())
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = float(d.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.std_
        return Z @ self.coef_ + self.intercept_

    def to_dict(self) -> dict[str, Any]:
        return {
            "penalty": float(self.penalty),
            "feature_names": list(self.feature_names),
            "mean": [float(v) for v in (self.mean_ if self.mean_ is not None else [])],
            "std": [float(v) for v in (self.std_ if self.std_ is not None else [])],
            "coef": [float(v) for v in (self.coef_ if self.coef_ is not None else [])],
            "intercept": float(self.intercept_),
        }


# --------------------------------------------------------------------------- metrics


def _auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Rank AUC with tie correction; ``None`` when one class is absent."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(sorted_scores):
        stop = index
        while stop + 1 < len(sorted_scores) and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index:stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def within_item_auc(
    scores: Sequence[float], y: Sequence[int], items: Sequence[str]
) -> dict[str, Any]:
    """Mean within-item AUC over items with mixed outcomes (plan §5.8).

    An all-correct or all-wrong item returns ``None`` for that item and increments
    ``excluded_items``, so the denominator of a panel row can never equal its row count.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    values: list[float] = []
    excluded = 0
    for item in sorted(set(str(i) for i in items)):
        mask = np.asarray([str(i) == item for i in items], dtype=bool)
        auc = _auc(scores[mask], y[mask])
        if auc is None:
            excluded += 1
        else:
            values.append(float(auc))
    return {
        "within_item_auc": float(np.mean(values)) if values else None,
        "eligible_items": len(values),
        "excluded_items": excluded,
    }


def brier_decomposition(p: Sequence[float], y: Sequence[int], *, n_bins: int = 10) -> dict[str, float]:
    """Murphy decomposition ``brier = reliability − resolution + uncertainty``.

    Equal-mass bins (plan §5.8 makes equal-mass the primary binning); the identity is
    exact up to the binning, and ``residual`` reports the gap so a caller can see it.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "residual": float("nan"), "n_bins": 0}
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    uncertainty = base * (1.0 - base)
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, min(int(n_bins), n))
    reliability = resolution = 0.0
    used = 0
    for index in bins:
        if len(index) == 0:
            continue
        used += 1
        weight = len(index) / float(n)
        p_bar = float(p[index].mean())
        y_bar = float(y[index].mean())
        reliability += weight * (p_bar - y_bar) ** 2
        resolution += weight * (y_bar - base) ** 2
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "residual": float(brier - (reliability - resolution + uncertainty)),
        "n_bins": used,
    }


def _cox(p: Sequence[float], y: Sequence[int]) -> dict[str, float]:
    out = CAL.cox_intercept_slope(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.int64))
    return {"cox_intercept": float(out["cox_intercept"]), "cox_slope": float(out["cox_slope"])}


# --------------------------------------------------------------------------- the artifact


@dataclass
class FrozenController:
    """The frozen E4 controller: heads, baselines, penalties and provenance.

    Immutable by hash: :meth:`save` stamps the canonical-JSON digest into the artifact and
    :func:`load_controller` refuses a file whose bytes no longer hash to it.
    """

    feature_names: tuple[str, ...]
    models: dict[str, R.LogisticModel]
    selections: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_features: dict[str, tuple[str, ...]] = field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    comparator: dict[str, Any] | None = None
    prior: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CONTROLLER_SCHEMA_VERSION
    feature_schema_version: int = K.FEATURE_SCHEMA_VERSION

    # -- prediction ----------------------------------------------------------------
    def names_for(self, head: str) -> tuple[str, ...]:
        if head in self.baseline_features:
            return tuple(self.baseline_features[head])
        if head == HEAD_DYNAMICS:
            return tuple(FE.DYNAMICS_ONLY_FEATURES)
        return tuple(self.feature_names)

    def predict(self, head: str, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """``p(head)`` for a sequence of feature dicts."""
        if head == BASELINE_CONFIDENCE_RAW:
            return self._raw_confidence(rows)
        if head == BASELINE_CONFIDENCE_RECALIBRATED:
            return self._recalibrated_confidence(rows)
        try:
            model = self.models[head]
        except KeyError as exc:
            raise ControllerError(f"the frozen controller has no head {head!r}") from exc
        names = self.names_for(head)
        X = np.asarray([FE.feature_vector(row, names) for row in rows], dtype=np.float64)
        return np.asarray(model.predict_proba(X), dtype=np.float64)

    def _raw_confidence(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        out = []
        for row in rows:
            missing = float(row.get("own_confidence_missing", 0.0)) > 0.5
            out.append(float(self.prior) if missing else float(row["own_confidence"]))
        return np.asarray(out, dtype=np.float64)

    def _recalibrated_confidence(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if not self.calibration:
            raise ControllerError(
                "the recalibrated-confidence baseline needs the validation calibration; "
                "this artifact was frozen without one"
            )
        scope = CAL.ScopeCalibration.from_dict(self.calibration)
        q = [None if float(row.get("own_confidence_missing", 0.0)) > 0.5
             else float(row["own_confidence"]) for row in rows]
        methods = [str(K.STUDY_ID)] * len(rows)
        return np.asarray(scope.apply(q, methods), dtype=np.float64)

    def u(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """``u = p_private − p_peer`` (spec §12.3), the exposure value of each agent."""
        return self.predict(HEAD_PRIVATE, rows) - self.predict(HEAD_PEER, rows)

    # -- serialization -------------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "kind": CONTROLLER_KIND,
            "study_id": K.STUDY_ID,
            "contract_version": int(K.CONTRACT_VERSION),
            "feature_schema_version": int(self.feature_schema_version),
            "feature_names": list(self.feature_names),
            "dynamics_features": list(FE.DYNAMICS_ONLY_FEATURES),
            "baseline_features": {k: list(v) for k, v in sorted(self.baseline_features.items())},
            "models": {name: model.to_dict() for name, model in sorted(self.models.items())},
            "selections": {k: dict(v) for k, v in sorted(self.selections.items())},
            "calibration": self.calibration,
            "comparator": self.comparator,
            "prior": float(self.prior),
            "penalty_grid": list(K.PENALTY_GRID),
            "meta": dict(self.meta),
        }

    @property
    def sha256(self) -> str:
        return controller_sha256(self.payload())

    def save(self, directory: str | os.PathLike) -> Path:
        """Write ``<directory>/MODEL.json`` (or the file itself) and return its path."""
        payload = self.payload()
        _refuse_non_finite(payload)
        digest = controller_sha256(payload)
        path = Path(directory)
        if path.suffix != ".json":
            path = path / CONTROLLER_FILE
        io.write_json(path, {**payload, SHA_FIELD: digest})
        return path


def controller_sha256(payload: Mapping[str, Any]) -> str:
    """The canonical-JSON digest of a controller payload (``FrozenG.save`` pattern).

    Byte-for-byte the rule WP4's ``runner._content_digest`` applies to a
    ``controller:<sha>`` register: ``sha256`` of ``json.dumps(body, sort_keys=True,
    separators=(",", ":"), ensure_ascii=False)`` over the payload *without* its
    ``controller_sha256`` field.
    """
    body = {k: v for k, v in dict(payload).items() if k != SHA_FIELD}
    return identity.sha256_hex(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _refuse_non_finite(obj: Any, path: str = "payload") -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            _refuse_non_finite(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _refuse_non_finite(value, f"{path}[{index}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ControllerError(f"{path} is {obj!r}; a frozen controller may not carry NaN/inf")


def load_controller(path: str | os.PathLike) -> FrozenController:
    """Load ``controller/<sha>/MODEL.json``, refusing a modified artifact.

    Accepts the file or its directory.  Any edit to the payload — one coefficient, one
    penalty, one meta string — changes the canonical-JSON digest and raises, which is what
    makes ``depends_on: ["controller:<sha>"]`` a real gate (plan §5.3).
    """
    file = Path(path)
    if file.is_dir():
        file = file / CONTROLLER_FILE
    if not file.is_file():
        raise ControllerError(f"no frozen controller at {file}")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ControllerError(f"{file} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ControllerError(f"{file} is not a controller artifact")
    claimed = payload.get(SHA_FIELD)
    if not isinstance(claimed, str):
        raise ControllerError(f"{file} carries no {SHA_FIELD}")
    computed = controller_sha256(payload)
    if computed != claimed.lower():
        raise ControllerError(
            f"{file}: {SHA_FIELD} mismatch (frozen artifact was modified): "
            f"{claimed[:12]} != {computed[:12]}"
        )
    if payload.get("kind") != CONTROLLER_KIND:
        raise ControllerError(f"{file} is a {payload.get('kind')!r} artifact, not {CONTROLLER_KIND}")
    if int(payload.get("feature_schema_version", -1)) != int(K.FEATURE_SCHEMA_VERSION):
        raise ControllerError(
            f"{file} was frozen against feature schema v{payload.get('feature_schema_version')}, "
            f"this build is v{K.FEATURE_SCHEMA_VERSION}"
        )
    return FrozenController(
        feature_names=tuple(payload["feature_names"]),
        models={name: R.LogisticModel.from_dict(data)
                for name, data in (payload.get("models") or {}).items()},
        selections={k: dict(v) for k, v in (payload.get("selections") or {}).items()},
        baseline_features={k: tuple(v) for k, v in (payload.get("baseline_features") or {}).items()},
        calibration=payload.get("calibration"),
        comparator=payload.get("comparator"),
        prior=float(payload.get("prior", 0.5)),
        meta=dict(payload.get("meta") or {}),
        schema_version=int(payload.get("schema_version", CONTROLLER_SCHEMA_VERSION)),
        feature_schema_version=int(payload["feature_schema_version"]),
    )


def freeze_controller(run_root: str | os.PathLike, controller: FrozenController) -> str:
    """Write ``<run_root>/controller/<sha>/MODEL.json`` and return the ``<sha>``."""
    digest = controller.sha256
    controller.save(Path(run_root) / CONTROLLER_DIR / digest)
    return digest


# --------------------------------------------------------------------------- fit


def fit_controller(
    train_frame: PairedFrame,
    val_frame: PairedFrame,
    *,
    study_seed: bytes = b"\x00" * 32,
    penalties: Sequence[float] = K.PENALTY_GRID,
    n_folds: int = DEFAULT_FOLDS,
    current_train: CurrentFrame | None = None,
    current_val: CurrentFrame | None = None,
    meta: Mapping[str, Any] | None = None,
) -> FrozenController:
    """Fit ``p_private``, ``p_peer``, the comparator, ``CURRENT`` and its baselines.

    Contract §2 signature.  ``train_frame``/``val_frame`` are fork frames (spec §12.1);
    the CURRENT frames are optional so a controller can be frozen for GATE before the
    static-selector work has a labelled candidate frame — the artifact then simply carries
    no ``CURRENT`` head, and :mod:`agents_scaling.ard.selectors` refuses to use one.
    """
    train_frame.validate()
    val_frame.validate()
    names = tuple(train_frame.feature_names)
    if tuple(val_frame.feature_names) != names:
        raise ControllerError("train and validation frames must share one feature schema")
    seed = bytes(study_seed)

    models: dict[str, R.LogisticModel] = {}
    selections: dict[str, dict[str, Any]] = {}
    for head, outcome in ((HEAD_PRIVATE, "y_private"), (HEAD_PEER, "y_peer")):
        observed = train_frame.observed(outcome)
        if not observed.any():
            raise ControllerError(f"the fork frame has no {outcome} outcome for {head}")
        sub = train_frame.subset(observed)
        X = sub.matrix(names)
        y = sub.y(outcome)
        model, selection = _fit_head(X, y, sub.clusters(), sub.strata(), names,
                                     seed=seed, penalties=penalties, n_folds=n_folds)
        models[head] = model
        selections[head] = {**selection.to_dict(), "n_rows": int(len(sub))}

    comparator = _fit_comparator(train_frame, val_frame, names, models, penalties=penalties)

    baseline_features: dict[str, tuple[str, ...]] = {}
    calibration: dict[str, Any] | None = None
    calibration_rows = "none"
    prior = 0.5
    if current_train is not None:
        current_train.validate()
        observed = current_train.observed()
        if not observed.any():
            raise ControllerError("the CURRENT frame carries no correctness outcome")
        rows = [r for r, m in zip(current_train.rows, observed) if m]
        sub = CurrentFrame(tuple(rows), names)
        y = sub.y()
        prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        for head, subset in (
            (HEAD_CURRENT, names),
            (HEAD_DYNAMICS, FE.DYNAMICS_ONLY_FEATURES),
            (BASELINE_FREQUENCY, BASELINE_FEATURES[BASELINE_FREQUENCY]),
            (BASELINE_ANSWER_LOGPROB, BASELINE_FEATURES[BASELINE_ANSWER_LOGPROB]),
            (BASELINE_FREQUENCY_CONFIDENCE, BASELINE_FEATURES[BASELINE_FREQUENCY_CONFIDENCE]),
        ):
            X = sub.matrix(subset)
            model, selection = _fit_head(X, y, sub.clusters(), sub.strata(), subset,
                                         seed=seed, penalties=penalties, n_folds=n_folds)
            models[head] = model
            selections[head] = {**selection.to_dict(), "n_rows": int(len(sub))}
            if head not in (HEAD_CURRENT, HEAD_DYNAMICS):
                baseline_features[head] = tuple(subset)
        # Spec §12.2 says *validation*-recalibrated confidence.  With no validation CURRENT
        # frame the fit falls back to the training rows and says so, rather than silently
        # relabelling a training recalibration as a validation one.
        calibration_rows = "validation" if current_val is not None else "train_fallback"
        calibration = _fit_confidence_recalibration(current_val or sub, seed=seed,
                                                    penalties=penalties)

    info = {
        "n_train_rows": int(len(train_frame)),
        "n_val_rows": int(len(val_frame)),
        "n_train_items": len(set(train_frame.clusters())),
        "n_val_items": len(set(val_frame.clusters())),
        "rounds": sorted({int(r.round_) for r in train_frame.rows}),
        "n_current_rows": 0 if current_train is None else int(len(current_train)),
        "confidence_recalibration_rows": calibration_rows,
        "penalty_grid": list(penalties),
        "n_folds": int(n_folds),
        "model_choice": "difference_of_two_logistic_heads",
        "comparator_is_validation_only": True,
        "runtime_feature_access": "records-mapping when the caller supplies one",
        **dict(meta or {}),
    }
    return FrozenController(
        feature_names=names,
        models=models,
        selections=selections,
        baseline_features=baseline_features,
        calibration=calibration,
        comparator=comparator,
        prior=prior,
        meta=info,
    )


def _fit_comparator(
    train_frame: PairedFrame,
    val_frame: PairedFrame,
    names: Sequence[str],
    models: Mapping[str, R.LogisticModel],
    *,
    penalties: Sequence[float],
) -> dict[str, Any]:
    """Ridge on the paired difference, scored on validation only (spec §12.1)."""
    train_paired = train_frame.subset(np.asarray([r.paired for r in train_frame.rows], dtype=bool))
    val_paired = val_frame.subset(np.asarray([r.paired for r in val_frame.rows], dtype=bool))
    if not len(train_paired) or not len(val_paired):
        return {"status": "unavailable", "reason": "no paired rows on both sides"}
    Xtr = train_paired.matrix(names)
    dtr = np.asarray([r.difference for r in train_paired.rows], dtype=np.float64)
    Xva = val_paired.matrix(names)
    dva = np.asarray([r.difference for r in val_paired.rows], dtype=np.float64)
    ridge = RidgeComparator(penalty=float(max(penalties)), feature_names=list(names)).fit(Xtr, dtr)
    u_val = (models[HEAD_PRIVATE].predict_proba(Xva) - models[HEAD_PEER].predict_proba(Xva))
    d_hat = ridge.predict(Xva)
    return {
        "status": "fitted",
        "ridge": ridge.to_dict(),
        "validation": {
            "n_rows": int(len(val_paired)),
            "ridge_paired_ranking_accuracy": _paired_ranking_accuracy(d_hat, dva),
            "u_paired_ranking_accuracy": _paired_ranking_accuracy(u_val, dva),
            "ridge_mse": float(np.mean((d_hat - dva) ** 2)),
            "u_mse": float(np.mean((u_val - dva) ** 2)),
        },
        "used_by_policy": False,
    }


def _paired_ranking_accuracy(score: Sequence[float], d: Sequence[float]) -> float | None:
    """Fraction of discordant pairs whose sign the score gets right; ties score 0.5."""
    score = np.asarray(score, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    mask = d != 0
    if not mask.any():
        return None
    hits = np.where(score[mask] == 0, 0.5, (np.sign(score[mask]) == np.sign(d[mask])).astype(float))
    return float(np.mean(hits))


def _fit_confidence_recalibration(
    frame: CurrentFrame, *, seed: bytes, penalties: Sequence[float]
) -> dict[str, Any] | None:
    """``study/neural/calibration.fit_scope`` on the validation rows (spec §12.2)."""
    observed = frame.observed()
    rows = [r for r, m in zip(frame.rows, observed) if m]
    if not rows:
        return None
    sub = CurrentFrame(tuple(rows), frame.feature_names)
    q = [None if math.isnan(v) else float(v) for v in sub.confidence()]
    y = [int(v) for v in sub.y()]
    methods = [str(K.STUDY_ID)] * len(rows)
    try:
        scope = CAL.fit_scope(
            q, y, methods, sub.clusters(), sub.strata(),
            scope="ard36_current_confidence", seed=seed, penalties=penalties,
        )
    except CAL.CalibrationError:
        return None
    return scope.to_dict()


# --------------------------------------------------------------------------- the policy


def protect_schedule_key(
    study_seed: bytes, pairing_id: str, round_: int, slot: int
) -> bytes:
    """The frozen ``ORDER_PROTECT`` HMAC (spec §12.3).

    Identical to ``ard.protocols.random_protected_slot``'s key, which is the point: GATE's
    argmax tie-break and RANDOM_PROTECT's uniform draw come from one recorded schedule, so
    the two arms differ only in the signal.
    """
    return identity.blind_order_key(
        bytes(study_seed), K.ORDER_PROTECT, str(pairing_id), int(round_), int(slot)
    )


@dataclass(frozen=True)
class ProtectionChoice:
    """One recorded GATE decision: who was protected, and the ``u`` that chose them."""

    episode_id: str
    pairing_id: str
    round_: int
    protected_slot: int
    u: tuple[float, ...]
    tied_slots: tuple[int, ...]
    controller_sha256: str

    def to_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "pairing_id": self.pairing_id,
            "round": int(self.round_),
            "protected_slot": int(self.protected_slot),
            "u": [float(v) for v in self.u],
            "tied_slots": [int(s) for s in self.tied_slots],
            "controller_sha256": self.controller_sha256,
            "all_negative": all(float(v) < 0.0 for v in self.u),
        }


def protection_choice(
    controller: FrozenController,
    state: Any,
    *,
    round_: int,
    episode_id: str = "",
    records: Mapping[str, Any] | None = None,
) -> ProtectionChoice:
    """The full record behind :func:`choose_protected` (spec §12.3)."""
    rows = FE.team_features(state, round_=int(round_), records=records)
    u = controller.u(rows)
    best = float(np.max(u))
    tied = tuple(slot for slot in range(K.N_AGENTS) if float(u[slot]) == best)
    pairing_id = str(getattr(state, "pairing_id", "") or "")
    protected = min(
        tied, key=lambda slot: protect_schedule_key(state.study_seed, pairing_id, round_, slot)
    )
    return ProtectionChoice(
        episode_id=str(episode_id or getattr(state, "episode_id", "")),
        pairing_id=pairing_id,
        round_=int(round_),
        protected_slot=int(protected),
        u=tuple(float(v) for v in u),
        tied_slots=tied,
        controller_sha256=controller.sha256,
    )


def choose_protected(
    controller: FrozenController,
    state: Any,
    *,
    round_: int,
    episode_id: str = "",
    records: Mapping[str, Any] | None = None,
) -> int:
    """``argmax_j u`` with the frozen HMAC tie-break (contract §2 WP5, spec §12.3).

    Not "the correct agent": the agent *predicted to benefit most from private
    continuation*.  There is no threshold — if every ``u`` is negative one agent is still
    protected, which spec §12.3 names as a meaningful failure mode rather than a bug.
    """
    return protection_choice(controller, state, round_=round_, episode_id=episode_id,
                             records=records).protected_slot


def random_protected_slot(state: Any, *, round_: int) -> int:
    """RANDOM_PROTECT's uniform choice from the same schedule (spec §12.3).

    Delegates to WP2's ``ard.protocols.random_protected_slot`` so there is exactly one
    implementation of the recorded schedule in the study.
    """
    from agents_scaling.ard import protocols as P  # local: avoids a WP2 import cycle

    return P.random_protected_slot(state, round_=int(round_))


# --------------------------------------------------------------------------- evaluation


def evaluate_controller(
    controller: FrozenController,
    frame: PairedFrame,
    *,
    study_seed: bytes = b"\x00" * 32,
) -> dict[str, Any]:
    """Held-out evaluation of the exposure-value heads (plan §5.8).

    Within-item AUC, paired ranking accuracy, Brier with its reliability/resolution
    decomposition, Cox intercept and slope, and policy relevance — the accuracy of the
    argmax-``u`` protection choice against the realized paired outcomes, beside the oracle
    and the uniform-random baseline that RANDOM_PROTECT realizes.
    """
    frame.validate()
    rows = [dict(r.features) for r in frame.rows]
    items = [str(r.item_id) for r in frame.rows]
    out: dict[str, Any] = {"n_rows": int(len(frame)), "n_items": len(set(items))}
    for head, outcome in ((HEAD_PRIVATE, "y_private"), (HEAD_PEER, "y_peer")):
        observed = frame.observed(outcome)
        if not observed.any():
            out[head] = {"status": "no outcome"}
            continue
        p = controller.predict(head, [r for r, m in zip(rows, observed) if m])
        y = frame.y(outcome)[observed]
        sub_items = [i for i, m in zip(items, observed) if m]
        out[head] = {
            "n": int(observed.sum()),
            **within_item_auc(p, y, sub_items),
            **brier_decomposition(p, y),
            **_cox(p, y),
            "pooled_auc": _auc(p, y),
            "mean_p": float(np.mean(p)),
        }
    paired = np.asarray([r.paired for r in frame.rows], dtype=bool)
    if paired.any():
        sub = frame.subset(paired)
        u = controller.u([dict(r.features) for r in sub.rows])
        d = np.asarray([r.difference for r in sub.rows], dtype=np.float64)
        out["paired"] = {
            "n": int(len(sub)),
            "n_discordant": int(np.sum(d != 0)),
            "paired_ranking_accuracy": _paired_ranking_accuracy(u, d),
            "mean_u": float(np.mean(u)),
            "mean_difference": float(np.mean(d)),
        }
        out["policy"] = _policy_relevance(sub, u, d, study_seed=study_seed)
    out["comparator"] = controller.comparator
    return out


def _policy_relevance(
    frame: PairedFrame, u: np.ndarray, d: np.ndarray, *, study_seed: bytes
) -> dict[str, Any]:
    """How often the argmax-``u`` protection choice picks a non-harmed agent (spec §12.5)."""
    teams: dict[str, list[int]] = {}
    for index, row in enumerate(frame.rows):
        teams.setdefault(row.team, []).append(index)
    chosen_gain: list[float] = []
    oracle_gain: list[float] = []
    mean_gain: list[float] = []
    hits: list[float] = []
    for team, index in sorted(teams.items()):
        values = [float(u[i]) for i in index]
        best = max(values)
        tied = [i for i, v in zip(index, values) if v == best]
        pairing = frame.rows[tied[0]].team_id or team
        pick = min(tied, key=lambda i: protect_schedule_key(
            study_seed, pairing, frame.rows[i].round_, frame.rows[i].agent_slot))
        chosen_gain.append(float(d[pick]))
        oracle_gain.append(float(max(d[i] for i in index)))
        mean_gain.append(float(np.mean([d[i] for i in index])))
        hits.append(1.0 if d[pick] >= 0 else 0.0)
    return {
        "n_teams": len(teams),
        "policy_no_harm_rate": float(np.mean(hits)) if hits else None,
        "mean_chosen_gain": float(np.mean(chosen_gain)) if chosen_gain else None,
        "mean_oracle_gain": float(np.mean(oracle_gain)) if oracle_gain else None,
        "mean_random_gain": float(np.mean(mean_gain)) if mean_gain else None,
        "regret_vs_oracle": (
            float(np.mean(oracle_gain) - np.mean(chosen_gain)) if chosen_gain else None
        ),
    }


def evaluate_current(
    controller: FrozenController, frame: CurrentFrame, *, heads: Sequence[str] | None = None
) -> dict[str, Any]:
    """CURRENT and its spec §12.2 baselines on identical rows and identical folds."""
    frame.validate()
    observed = frame.observed()
    rows = [dict(r.features) for r, m in zip(frame.rows, observed) if m]
    items = [str(r.item_id) for r, m in zip(frame.rows, observed) if m]
    y = frame.y()[observed]
    names = heads or (
        HEAD_CURRENT, HEAD_DYNAMICS, BASELINE_FREQUENCY, BASELINE_CONFIDENCE_RAW,
        BASELINE_CONFIDENCE_RECALIBRATED, BASELINE_ANSWER_LOGPROB,
        BASELINE_FREQUENCY_CONFIDENCE,
    )
    out: dict[str, Any] = {"n_rows": len(rows), "n_items": len(set(items))}
    for head in names:
        try:
            p = controller.predict(head, rows)
        except ControllerError as exc:
            out[head] = {"status": "unavailable", "detail": str(exc)}
            continue
        out[head] = {
            **within_item_auc(p, y, items),
            **brier_decomposition(p, y),
            **_cox(p, y),
            "pooled_auc": _auc(p, y),
        }
    return out


# --------------------------------------------------------------------------- frame builder


def paired_rows_from_forks(
    forks: Sequence[Any],
    features_by_state: Mapping[str, Mapping[str, float]],
    outcomes: Mapping[str, int],
    *,
    item_split: Mapping[str, str],
    item_family: Mapping[str, str],
) -> list[PairedRow]:
    """Join ``forks.parquet`` rows to frozen prefix features and branch outcomes.

    ``features_by_state`` maps a ``state:<sha>`` shared prefix to the frozen v1 features of
    the focal agent *at that prefix* (spec §19: frozen before the intervention);
    ``outcomes`` maps a child ``decision_id`` to its 0/1 correctness, which the caller
    reads from ``labels.parquet`` on the evaluator side.  Rows whose prefix has no features
    or whose branches are not both resolved are skipped, and the pair is emitted once per
    (prefix, focal agent).
    """
    from agents_scaling.ard.forks import ACTION_PEER, ACTION_PRIVATE  # local: WP2 vocabulary

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for fork in forks:
        key = (str(fork.shared_prefix_hash), int(fork.focal_agent))
        entry = grouped.setdefault(key, {"forks": {}, "round": int(fork.parent_round),
                                         "item": str(fork.item_id), "episode": str(fork.episode_id)})
        child = tuple(fork.child_decision_ids)
        entry["forks"][str(fork.action)] = child[0] if child else None
    out: list[PairedRow] = []
    for (prefix, slot), entry in sorted(grouped.items()):
        features = features_by_state.get(prefix)
        if features is None:
            continue
        private_id = entry["forks"].get(ACTION_PRIVATE)
        peer_id = entry["forks"].get(ACTION_PEER)
        y_private = outcomes.get(str(private_id)) if private_id else None
        y_peer = outcomes.get(str(peer_id)) if peer_id else None
        if y_private is None or y_peer is None:
            continue
        item = entry["item"]
        out.append(PairedRow(
            row_id=f"{prefix}#a{slot}",
            item_id=item,
            family=str(item_family.get(item, "")),
            split=str(item_split.get(item, "")),
            round_=int(entry["round"]),
            agent_slot=int(slot),
            features=dict(features),
            y_private=int(y_private),
            y_peer=int(y_peer),
            team_id=f"{item}|r{int(entry['round'])}",
            episode_id=str(entry["episode"]),
            parent_state_id=prefix,
        ))
    return out
