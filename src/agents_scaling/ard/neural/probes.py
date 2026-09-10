"""ARD36 neural probes: regularized linear readouts with named targets (WP6).

Spec §14.4 ("Prospective evaluation"); plan §5.6 and §5.8.

The estimator is reused, not rewritten: ``study.neural.readout.LogisticModel`` (standard-
scaled L2 logistic regression), ``readout.grouped_stratified_folds`` (every item and all
its derivatives stay in one fold, strata balanced, deterministic in the study seed) and
``readout.one_se_select`` (the one-SE penalty rule).  What this module adds is the ARD36
evaluation contract:

* **Targets are named explicitly** — ``CURRENT`` (this agent's answer is correct) and
  ``EXPOSURE_VALUE`` (``u = p_private - p_peer``, the value of protecting this agent),
  fitted at each round at which GATE acts.  ``u`` is kept as a continuous quantity for the
  ranking and policy statistics and binarised at 0 for the probe's Brier/AUC, because a
  positive ``u`` is exactly "protection is worth spending on this agent".
* **Every arm runs on IDENTICAL folds and IDENTICAL rows** (spec §14.4 "compare a
  behavior-only baseline"): the neural arm, the shuffled-within-item neural control, the
  option/answer-class baseline, the behaviour-only baseline and the combined arm all see
  one fold vector and one row index built once, in :func:`run_probes`.
* **The projection is train-only** (spec §14.4): a fixed 256-dimensional Gaussian random
  projection whose seed is derived from the study seed and whose centering statistics come
  from the training rows of that fold alone.
* **Timing is enforced in the frame builder** (spec §14.4, plan §5.8).  Each row carries
  ``feature_time`` and ``last_allowed_round`` from
  :mod:`agents_scaling.ard.neural.anchors`; :func:`build_frame` refuses a row whose target
  would make a post-answer state predict the identity of an answer it already contains.
  Pre-answer anchors may predict the upcoming answer; post-answer anchors may predict only
  that answer's correctness or a round ``t+1`` target.
* **No layer sweep on test** (spec §14.4): :func:`select_site` chooses the (layer, anchor)
  on development rows only and returns a :class:`SiteLock`; :func:`evaluate_locked` refuses
  any other site on test rows.

Reported per arm (spec §14.4): within-item AUC, paired ranking accuracy, Brier, and policy
relevance (the accuracy of the argmax-``u`` protection choice).  A high *pooled* AUC is
never called minority recognition; the pooled value is reported beside the within-item one
and labelled.

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as A
from agents_scaling.study import identity
from agents_scaling.study.neural.readout import (
    LogisticModel,
    ReadoutError,
    grouped_stratified_folds,
    one_se_select,
)

__all__ = [
    "ARMS",
    "ARM_ANSWER_CLASS",
    "ARM_BEHAVIOUR",
    "ARM_COMBINED",
    "ARM_NEURAL",
    "ARM_SHUFFLED",
    "PROBE_FOLD_NAMESPACE",
    "PROJECTION_DIM",
    "ProbeFrame",
    "ProbeReport",
    "ProbeRow",
    "RandomProjection",
    "ReadoutError",
    "SiteLock",
    "TARGETS",
    "TARGET_CURRENT",
    "TARGET_EXPOSURE_VALUE",
    "TimingViolation",
    "brier_score",
    "build_frame",
    "evaluate_locked",
    "paired_ranking_accuracy",
    "policy_relevance",
    "run_probes",
    "select_site",
    "shuffle_within_item",
    "within_item_auc",
]

TARGET_CURRENT = "CURRENT"
TARGET_EXPOSURE_VALUE = "EXPOSURE_VALUE"
TARGETS: tuple[str, ...] = (TARGET_CURRENT, TARGET_EXPOSURE_VALUE)

#: ``CURRENT`` is a correctness target of the round the state belongs to; ``EXPOSURE_VALUE``
#: is a value target of the *next* update, i.e. of the round the protection would act on.
TARGET_KIND: Mapping[str, str] = {TARGET_CURRENT: "correctness", TARGET_EXPOSURE_VALUE: "value"}

ARM_NEURAL = "neural"
ARM_SHUFFLED = "neural_shuffled_within_item"
ARM_BEHAVIOUR = "behaviour_only"
ARM_ANSWER_CLASS = "answer_class"
ARM_COMBINED = "neural_plus_behaviour"
ARMS: tuple[str, ...] = (ARM_NEURAL, ARM_SHUFFLED, ARM_BEHAVIOUR, ARM_ANSWER_CLASS, ARM_COMBINED)

#: Spec §14.4: "A train-only fixed random projection to 256 dimensions may reduce cost."
PROJECTION_DIM = 256
PROBE_FOLD_NAMESPACE = "ARD_PROBE_FOLDS"
PROJECTION_NAMESPACE = "ARD_PROBE_PROJECTION"
SHUFFLE_NAMESPACE = "ARD_PROBE_SHUFFLE"
_EPS = 1e-12


class TimingViolation(ReadoutError):
    """A (feature_time, last_allowed_round, target) combination spec §14.4 forbids."""


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class ProbeRow:
    """One captured state with its behavioural context and its targets.

    ``vector_index`` indexes the state matrix handed to :func:`build_frame`.  ``targets``
    maps a target name to its value: ``CURRENT`` is 0/1, ``EXPOSURE_VALUE`` is the
    continuous ``u = p_private - p_peer``.  ``target_round`` says which round each target
    is *about* — that is what the timing rule is checked against.
    """

    row_id: str
    item_id: str
    episode_id: str
    agent_id: int
    protocol: str
    round_: int
    layer: int
    anchor: str
    vector_index: int
    split: str = ""
    family: str = ""
    behaviour: Mapping[str, float] = field(default_factory=dict)
    answer_class: str | None = None
    targets: Mapping[str, float] = field(default_factory=dict)
    target_round: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.anchor not in K.CAPTURE_ANCHORS:
            raise ValueError(f"anchor {self.anchor!r} not in contract.CAPTURE_ANCHORS")

    @property
    def feature_time(self) -> str:
        return A.feature_time_of(self.anchor)

    @property
    def last_allowed_round(self) -> int:
        return A.last_allowed_round(self.anchor, self.round_)

    def round_of(self, target: str) -> int:
        """The round a target is about (defaults: ``CURRENT`` → this round, a value target
        → the next round, because protection acts on the update that follows the state)."""
        if target in self.target_round:
            return int(self.target_round[target])
        return int(self.round_) if TARGET_KIND.get(target) == "correctness" else int(self.round_) + 1


def check_timing(row: ProbeRow, target: str) -> None:
    """Spec §14.4 / plan §5.8, enforced (see the module docstring for the rule).

    A target strictly after ``last_allowed_round`` is always permitted.  A target at or
    before it is permitted only when it is the correctness of the very answer the state
    contains — i.e. a ``post_answer`` state and a correctness target at exactly
    ``last_allowed_round``.
    """
    target_round = row.round_of(target)
    horizon = row.last_allowed_round
    if target_round > horizon:
        return
    if (row.feature_time == A.FEATURE_TIME_POST
            and TARGET_KIND.get(target) == "correctness"
            and target_round == horizon):
        return
    raise TimingViolation(
        f"{row.row_id}: a {row.feature_time} state at round {row.round_} (anchor {row.anchor}, "
        f"last_allowed_round={horizon}) may not predict {target} at round {target_round} "
        "(spec 14.4: post-answer features cannot establish the cause of an answer already emitted)"
    )


# --------------------------------------------------------------------------- frames


@dataclass(frozen=True)
class ProbeFrame:
    """Rows + the state matrix + one fold vector, shared by every arm."""

    target: str
    rows: tuple[ProbeRow, ...]
    H: np.ndarray                     # [n, hidden] states, in row order
    y: np.ndarray                     # [n] binary labels
    value: np.ndarray                 # [n] the continuous target (u); == y for correctness
    folds: np.ndarray                 # [n] fold ids, grouped by item
    behaviour_names: tuple[str, ...]
    B: np.ndarray                     # [n, n_behaviour]
    C: np.ndarray                     # [n, n_classes] answer/option-class one-hot
    class_names: tuple[str, ...]
    items: tuple[str, ...]
    groups: tuple[str, ...]           # (episode, round) protection decision groups
    seed: bytes

    @property
    def n(self) -> int:
        return len(self.rows)

    def sites(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted({(int(r.layer), r.anchor) for r in self.rows}))

    def index_of_site(self, layer: int, anchor: str) -> np.ndarray:
        return np.asarray([i for i, r in enumerate(self.rows) if int(r.layer) == int(layer) and r.anchor == anchor],
                          dtype=np.int64)


def build_frame(
    rows: Sequence[ProbeRow],
    H: np.ndarray,
    *,
    target: str,
    study_seed: bytes,
    n_folds: int = 5,
) -> ProbeFrame:
    """Assemble one frame for ``target``, enforcing the §14.4 timing rule on every row.

    ``H`` is the state matrix; ``row.vector_index`` selects each row's state.  Rows whose
    target is absent are dropped (a missing target is missing, never imputed); rows whose
    target the timing rule forbids raise :class:`TimingViolation` — the builder is the
    enforcement point, so no caller can quietly assemble an anachronistic frame.
    """
    if target not in TARGETS:
        raise ReadoutError(f"unknown probe target {target!r}; expected one of {TARGETS}")
    H = np.asarray(H, dtype=np.float64)
    kept: list[ProbeRow] = []
    for row in rows:
        if target not in row.targets or row.targets[target] is None:
            continue
        check_timing(row, target)
        kept.append(row)
    if not kept:
        raise ReadoutError(f"no rows carry the target {target!r}")
    index = np.asarray([int(r.vector_index) for r in kept], dtype=np.int64)
    if index.min() < 0 or index.max() >= H.shape[0]:
        raise ReadoutError("vector_index outside the state matrix")
    value = np.asarray([float(r.targets[target]) for r in kept], dtype=np.float64)
    y = (value > 0.0).astype(np.int64) if TARGET_KIND[target] == "value" else (value > 0.5).astype(np.int64)

    names = sorted({k for r in kept for k in r.behaviour})
    B = np.asarray([[float(r.behaviour.get(k, 0.0)) for k in names] for r in kept], dtype=np.float64)
    classes = sorted({str(r.answer_class) for r in kept if r.answer_class is not None})
    C = np.zeros((len(kept), max(1, len(classes))), dtype=np.float64)
    for i, row in enumerate(kept):
        if row.answer_class is not None and row.answer_class in classes:
            C[i, classes.index(str(row.answer_class))] = 1.0
    items = tuple(str(r.item_id) for r in kept)
    strata = [str(r.family or "unknown") for r in kept]
    folds = grouped_stratified_folds(list(items), strata, int(n_folds), study_seed,
                                     namespace=PROBE_FOLD_NAMESPACE)
    groups = tuple(f"{r.episode_id}|{r.round_of(target)}" for r in kept)
    return ProbeFrame(target=target, rows=tuple(kept), H=H[index], y=y, value=value, folds=folds,
                      behaviour_names=tuple(names), B=B, C=C, class_names=tuple(classes),
                      items=items, groups=groups, seed=study_seed)


# --------------------------------------------------------------------------- features


@dataclass
class RandomProjection:
    """A fixed Gaussian random projection to ``dim`` dimensions (spec §14.4).

    The matrix is deterministic in ``(study_seed, tag, dim, hidden)`` — it is *fixed*, not
    refit per fold — and the centering vector is estimated on the **training rows only**,
    so no test row influences the representation.
    """

    dim: int = PROJECTION_DIM
    tag: str = "probe"
    matrix_: np.ndarray | None = None
    mean_: np.ndarray | None = None

    def fit(self, H_train: np.ndarray, *, study_seed: bytes) -> "RandomProjection":
        H_train = np.asarray(H_train, dtype=np.float64)
        if H_train.ndim != 2 or H_train.shape[0] == 0:
            raise ReadoutError("the projection needs a non-empty training matrix")
        hidden = int(H_train.shape[1])
        dim = int(min(self.dim, hidden))
        digest = identity.blind_order_key(study_seed, PROJECTION_NAMESPACE, self.tag, dim, hidden)
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        self.matrix_ = rng.standard_normal((hidden, dim)) / math.sqrt(dim)
        self.mean_ = H_train.mean(axis=0)
        return self

    def transform(self, H: np.ndarray) -> np.ndarray:
        if self.matrix_ is None or self.mean_ is None:
            raise ReadoutError("the projection is not fitted")
        return (np.asarray(H, dtype=np.float64) - self.mean_) @ self.matrix_

    @property
    def feature_names(self) -> list[str]:
        return [] if self.matrix_ is None else [f"rp{i}" for i in range(self.matrix_.shape[1])]


def shuffle_within_item(H: np.ndarray, items: Sequence[str], *, study_seed: bytes, tag: str = "shuffle") -> np.ndarray:
    """The §14.4 control: permute the state vectors **within** each item.

    Any signal that survives this is item-level, not state-level: the control keeps every
    item's marginal distribution of states and destroys only the row↔state pairing.
    """
    H = np.asarray(H, dtype=np.float64)
    out = H.copy()
    order = np.arange(H.shape[0])
    for item in sorted(set(items)):
        idx = order[np.asarray([str(i) == str(item) for i in items], dtype=bool)]
        if idx.size < 2:
            continue
        digest = identity.blind_order_key(study_seed, SHUFFLE_NAMESPACE, tag, str(item))
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        out[idx] = H[rng.permutation(idx)]
    return out


# --------------------------------------------------------------------------- metrics


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUC with mid-ranks for ties; ``None`` when one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def within_item_auc(p: np.ndarray, y: np.ndarray, items: Sequence[str]) -> dict[str, Any]:
    """Mean AUC **inside** each item, with its eligible/excluded counts (plan §5.8).

    An item whose rows are all-correct or all-wrong returns no AUC and increments the
    excluded counter, so a panel row's AUC denominator can never equal its row count.
    """
    per_item: list[float] = []
    excluded = 0
    for item in sorted(set(items)):
        mask = np.asarray([str(i) == str(item) for i in items], dtype=bool)
        value = _auc(np.asarray(p)[mask], np.asarray(y)[mask])
        if value is None:
            excluded += 1
        else:
            per_item.append(value)
    return {
        "within_item_auc": float(np.mean(per_item)) if per_item else None,
        "eligible_items": len(per_item),
        "excluded_items": excluded,
        "pooled_auc": _auc(np.asarray(p), np.asarray(y)),
        "pooled_auc_note": "pooled AUC is not minority recognition (spec 14.4)",
    }


def paired_ranking_accuracy(p: np.ndarray, y: np.ndarray, items: Sequence[str]) -> dict[str, Any]:
    """Fraction of within-item (correct, wrong) pairs the probe orders correctly; ties 0.5."""
    wins = 0.0
    pairs = 0
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    for item in sorted(set(items)):
        mask = np.asarray([str(i) == str(item) for i in items], dtype=bool)
        pi, yi = p[mask], y[mask]
        for a in np.flatnonzero(yi == 1):
            for b in np.flatnonzero(yi == 0):
                pairs += 1
                wins += 1.0 if pi[a] > pi[b] else (0.5 if pi[a] == pi[b] else 0.0)
    return {"paired_ranking_accuracy": (wins / pairs) if pairs else None, "pairs": pairs}


def policy_relevance(p: np.ndarray, value: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    """Accuracy of the argmax-``u`` protection choice (spec §14.4 "policy relevance").

    One protection decision per group (an episode-round): the probe picks the agent with
    the highest predicted score; the choice is scored against the agent with the highest
    realized ``u`` and against the realized value it captured.  Groups with a single
    candidate or an all-tied realized ``u`` are excluded and counted.
    """
    p = np.asarray(p, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    hits = 0
    decided = 0
    excluded = 0
    captured: list[float] = []
    oracle: list[float] = []
    for group in sorted(set(map(str, groups))):
        mask = np.asarray([str(g) == group for g in groups], dtype=bool)
        if mask.sum() < 2 or float(np.ptp(value[mask])) <= _EPS:
            excluded += 1
            continue
        decided += 1
        pick = int(np.argmax(p[mask]))
        best = int(np.argmax(value[mask]))
        hits += int(pick == best)
        captured.append(float(value[mask][pick]))
        oracle.append(float(value[mask][best]))
    n_per_group = float(np.mean([sum(1 for g in groups if str(g) == group) for group in set(map(str, groups))]))
    return {
        "policy_accuracy": (hits / decided) if decided else None,
        "decisions": decided,
        "excluded_groups": excluded,
        "chance_accuracy": (1.0 / n_per_group) if n_per_group else None,
        "mean_captured_value": float(np.mean(captured)) if captured else None,
        "mean_oracle_value": float(np.mean(oracle)) if oracle else None,
    }


# --------------------------------------------------------------------------- fitting


def _matrix_for(arm: str, frame: ProbeFrame, H: np.ndarray, projection: RandomProjection | None) -> np.ndarray:
    if arm in (ARM_NEURAL, ARM_SHUFFLED):
        return projection.transform(H) if projection is not None else H
    if arm == ARM_BEHAVIOUR:
        return frame.B
    if arm == ARM_ANSWER_CLASS:
        return frame.C
    if arm == ARM_COMBINED:
        neural = projection.transform(H) if projection is not None else H
        return np.column_stack([neural, frame.B])
    raise ReadoutError(f"unknown arm {arm!r}")   # pragma: no cover


def _fit_arm(
    arm: str, frame: ProbeFrame, index: np.ndarray, *, penalties: Sequence[float], projection_dim: int,
) -> np.ndarray:
    """Out-of-fold probabilities for one arm on ``index``, on the frame's own folds."""
    H_full = (shuffle_within_item(frame.H, frame.items, study_seed=frame.seed, tag=frame.target)
              if arm == ARM_SHUFFLED else frame.H)
    p = np.full(index.size, np.nan, dtype=np.float64)
    folds = frame.folds[index]
    for fold in sorted(set(int(f) for f in folds)):
        test_mask = folds == fold
        train_mask = ~test_mask
        if train_mask.sum() < 2 or test_mask.sum() == 0:
            continue
        train_rows, test_rows = index[train_mask], index[test_mask]
        projection = None
        if arm in (ARM_NEURAL, ARM_SHUFFLED, ARM_COMBINED):
            projection = RandomProjection(dim=projection_dim, tag=f"{frame.target}:{arm}")
            projection.fit(H_full[train_rows], study_seed=frame.seed)   # train-only
        X_train = _matrix_for(arm, _subset(frame, train_rows), H_full[train_rows], projection)
        X_test = _matrix_for(arm, _subset(frame, test_rows), H_full[test_rows], projection)
        penalty = _select_penalty(X_train, frame.y[train_rows], penalties)
        model = LogisticModel(penalty=penalty).fit(X_train, frame.y[train_rows])
        p[test_mask] = model.predict_proba(X_test)
    return p


def _subset(frame: ProbeFrame, index: np.ndarray) -> ProbeFrame:
    idx = np.asarray(index, dtype=np.int64)
    return ProbeFrame(
        target=frame.target,
        rows=tuple(frame.rows[i] for i in idx),
        H=frame.H[idx],
        y=frame.y[idx],
        value=frame.value[idx],
        folds=frame.folds[idx],
        behaviour_names=frame.behaviour_names,
        B=frame.B[idx],
        C=frame.C[idx],
        class_names=frame.class_names,
        items=tuple(frame.items[i] for i in idx),
        groups=tuple(frame.groups[i] for i in idx),
        seed=frame.seed,
    )


def _select_penalty(X: np.ndarray, y: np.ndarray, penalties: Sequence[float]) -> float:
    """Inner one-SE selection over ``penalties`` on the training rows only.

    Falls back to the strongest penalty when the training fold has a single class (the
    model is then a constant and the choice cannot be informative anyway).
    """
    penalties = tuple(float(p) for p in penalties)
    if len(set(np.asarray(y).tolist())) < 2 or X.shape[0] < 4:
        return max(penalties)
    n_inner = min(3, X.shape[0] // 2)
    candidates: list[tuple[tuple[int | None, int | None, float], float, float]] = []
    for penalty in penalties:
        losses: list[float] = []
        for fold in range(n_inner):
            mask = (np.arange(X.shape[0]) % n_inner) == fold
            if mask.all() or (~mask).sum() < 2:
                continue
            model = LogisticModel(penalty=penalty).fit(X[~mask], y[~mask])
            losses.append(brier_score(model.predict_proba(X[mask]), y[mask]))
        if not losses:
            continue
        mean = float(np.mean(losses))
        se = float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0
        candidates.append(((None, None, penalty), mean, se))
    if not candidates:
        return max(penalties)
    return float(one_se_select(candidates, block_order=()).penalty)


@dataclass(frozen=True)
class SiteLock:
    """The (layer, anchor) chosen on development rows — the only site test rows may use."""

    layer: int
    anchor: str
    target: str
    selected_on: str
    within_item_auc: float | None
    candidates: tuple[tuple[int, str], ...]

    def check(self, layer: int, anchor: str) -> None:
        if (int(layer), str(anchor)) != (int(self.layer), str(self.anchor)):
            raise ReadoutError(
                f"no layer sweep on test (spec 14.4): the locked site is "
                f"({self.layer}, {self.anchor}), asked for ({layer}, {anchor})"
            )


@dataclass(frozen=True)
class ProbeReport:
    """One target × one site: every arm on identical folds and rows."""

    target: str
    layer: int
    anchor: str
    feature_time: str
    n_rows: int
    n_items: int
    arms: Mapping[str, Mapping[str, Any]]
    incremental: Mapping[str, Any]
    folds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "layer": int(self.layer), "anchor": self.anchor,
            "feature_time": self.feature_time, "n_rows": int(self.n_rows),
            "n_items": int(self.n_items), "folds": int(self.folds),
            "arms": {k: dict(v) for k, v in self.arms.items()},
            "incremental": dict(self.incremental),
        }


def run_probes(
    frame: ProbeFrame,
    *,
    layer: int,
    anchor: str,
    arms: Sequence[str] = ARMS,
    penalties: Sequence[float] = K.PENALTY_GRID,
    projection_dim: int = PROJECTION_DIM,
    site_lock: SiteLock | None = None,
) -> ProbeReport:
    """Fit every arm at one site on identical folds and rows and report §14.4's four numbers."""
    if site_lock is not None:
        site_lock.check(layer, anchor)
    index = frame.index_of_site(layer, anchor)
    if index.size == 0:
        raise ReadoutError(f"no rows at site (layer={layer}, anchor={anchor})")
    items = [frame.items[i] for i in index]
    groups = [frame.groups[i] for i in index]
    y = frame.y[index]
    value = frame.value[index]
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        p = _fit_arm(arm, frame, index, penalties=penalties, projection_dim=projection_dim)
        scored = np.isfinite(p)
        metrics: dict[str, Any] = {
            "n_scored": int(scored.sum()),
            "brier": brier_score(p[scored], y[scored]) if scored.any() else None,
            **within_item_auc(p[scored], y[scored], [items[i] for i in np.flatnonzero(scored)]),
            **paired_ranking_accuracy(p[scored], y[scored], [items[i] for i in np.flatnonzero(scored)]),
            **policy_relevance(p[scored], value[scored], [groups[i] for i in np.flatnonzero(scored)]),
        }
        metrics["probabilities"] = p.tolist()
        results[arm] = metrics
    incremental = {}
    for baseline in (ARM_BEHAVIOUR, ARM_ANSWER_CLASS, ARM_SHUFFLED):
        if baseline in results and ARM_NEURAL in results:
            incremental[f"{ARM_NEURAL}_minus_{baseline}"] = {
                "brier": _delta(results[ARM_NEURAL]["brier"], results[baseline]["brier"]),
                "within_item_auc": _delta(results[ARM_NEURAL]["within_item_auc"], results[baseline]["within_item_auc"]),
                "policy_accuracy": _delta(results[ARM_NEURAL]["policy_accuracy"], results[baseline]["policy_accuracy"]),
            }
    return ProbeReport(target=frame.target, layer=int(layer), anchor=str(anchor),
                       feature_time=A.feature_time_of(anchor), n_rows=int(index.size),
                       n_items=len(set(items)), arms=results, incremental=incremental,
                       folds=int(len(set(int(f) for f in frame.folds[index]))))


def _delta(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else float(a - b)


def select_site(
    frame: ProbeFrame,
    *,
    sites: Iterable[tuple[int, str]] | None = None,
    arm: str = ARM_NEURAL,
    penalties: Sequence[float] = K.PENALTY_GRID,
    projection_dim: int = PROJECTION_DIM,
    selected_on: str = "train+validation",
) -> SiteLock:
    """Choose (layer, anchor) on development rows only and lock it (spec §14.4).

    Selection maximises within-item AUC; ties break towards the primary relative depth and
    then the contract's anchor order, so the lock is deterministic.
    """
    candidates = tuple(sites) if sites is not None else frame.sites()
    if not candidates:
        raise ReadoutError("no candidate sites")
    scored: list[tuple[float, int, int, tuple[int, str]]] = []
    best_auc: dict[tuple[int, str], float | None] = {}
    for site in candidates:
        report = run_probes(frame, layer=site[0], anchor=site[1], arms=(arm,),
                            penalties=penalties, projection_dim=projection_dim)
        auc = report.arms[arm]["within_item_auc"]
        best_auc[site] = auc
        scored.append((-(auc if auc is not None else -np.inf), site[0],
                       list(K.CAPTURE_ANCHORS).index(site[1]), site))
    scored.sort()
    site = scored[0][3]
    return SiteLock(layer=int(site[0]), anchor=str(site[1]), target=frame.target,
                    selected_on=selected_on, within_item_auc=best_auc[site],
                    candidates=candidates)


def evaluate_locked(
    frame: ProbeFrame,
    lock: SiteLock,
    *,
    arms: Sequence[str] = ARMS,
    penalties: Sequence[float] = K.PENALTY_GRID,
    projection_dim: int = PROJECTION_DIM,
) -> ProbeReport:
    """Evaluate the **locked** site on held-out rows; any other site is refused."""
    if lock.target != frame.target:
        raise ReadoutError(f"lock is for target {lock.target!r}, frame is {frame.target!r}")
    return run_probes(frame, layer=lock.layer, anchor=lock.anchor, arms=arms,
                      penalties=penalties, projection_dim=projection_dim, site_lock=lock)
