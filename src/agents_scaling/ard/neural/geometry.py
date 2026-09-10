"""ARD36 neural geometry: the ``neural_geometry`` and ``neural_agent`` tables (WP6).

Spec §14.3 ("Features") and plan §5.8 ("Neural panel").  The study_v4 estimator
``study.neural.geometry.spectrum_metrics`` is reused verbatim — it already computes, for
one set of ``s`` comparable states, the participation ratio, the eigenvalue-entropy
effective rank with the ``rank_ceiling(d, s) = min(d, s-1)`` ceiling (``N-1 = 4`` at
``N = 5``), raw norms, mean/min/max pairwise distance and the cosine distance in both the
raw and the unit-normalised representation.  **This module adds the five quantities spec
§14.3 asks for beyond it**:

* leading eigenvalue share ``λ1 / Σλ`` and total variance ``Σλ`` (team grain);
* per-agent distance to the peer centroid (the centroid of the *other four* agents) and to
  the nearest peer;
* per-agent displacement from its own ``t-1`` state;
* the alignment (cosine) of that displacement with the **pre-update** peer-versus-self
  direction ``centroid(peers at t-1) - self at t-1``, with near-zero denominators recorded
  rather than divided through (spec §14.3: "recording near-zero denominators").

Two rules from the spec are structural here:

* "Raw and normalized metrics must be shown together."  Every team row is emitted twice,
  once with ``normalised = False`` (raw, centered states) and once with
  ``normalised = True`` (each state divided by its norm first), sharing one key.
* "A team-level rank statistic is permutation invariant and cannot, by itself, tell which
  agent is correct."  That is why ``neural_agent`` exists and carries the per-agent axis;
  the nonclaim travels in :data:`TEAM_STATISTIC_NONCLAIM`.

Grain (plan §5.8): ``neural_geometry`` is keyed by item × protocol × round × layer ×
anchor; ``neural_agent`` adds the agent axis.  Both carry ``feature_time`` and
``last_allowed_round`` from :mod:`agents_scaling.ard.neural.anchors`.  Column names and
order are the frozen ones in ``ard.compact.SCHEMAS`` (WP4); anything this module adds is
appended after them.

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as A
from agents_scaling.study.neural.geometry import (
    GeometryError,
    Standardizer,
    rank_ceiling,
    spectrum_metrics,
    unit_normalize,
)

__all__ = [
    "AGENT_COLUMNS",
    "DEGENERATE_EPS",
    "GEOMETRY_COLUMNS",
    "GeometryError",
    "StateSet",
    "TEAM_STATISTIC_NONCLAIM",
    "agent_metrics",
    "leading_eigenvalue_share",
    "neural_tables",
    "ordered_columns",
    "peer_centroid",
    "team_metrics",
    "rank_ceiling",
    "total_variance",
]

#: Denominators below this (in the raw state units) are reported, never divided by.
DEGENERATE_EPS: float = 1e-8

TEAM_STATISTIC_NONCLAIM = (
    "A team-level rank statistic is permutation-invariant: it may describe a team's "
    "susceptibility to convergence but cannot say which agent is correct. Agent selection "
    "requires the per-agent features in neural_agent (spec 14.3)."
)

#: ``ard.compact.SCHEMAS["neural_geometry"]`` (frozen by WP4); extras are appended after.
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "item_id", "protocol", "round", "layer", "anchor", "participation_ratio", "entropy_rank",
    "leading_eigenvalue_share", "total_variance", "mean_pairwise", "min_pairwise",
    "max_pairwise", "cosine_distance", "normalised", "feature_time", "last_allowed_round",
)
#: ``ard.compact.SCHEMAS["neural_agent"]``.
AGENT_COLUMNS: tuple[str, ...] = (
    "item_id", "protocol", "round", "layer", "anchor", "agent_id", "state_norm",
    "distance_to_peer_centroid", "distance_to_nearest_peer", "own_displacement",
    "displacement_alignment", "near_zero_denominator", "feature_time", "last_allowed_round",
)


# --------------------------------------------------------------------------- additions


def leading_eigenvalue_share(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """``λ1 / Σλ`` over the nonzero eigenvalues; NaN when the spectrum is empty."""
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    if lam.size == 0:
        return float("nan")
    return float(lam.max() / lam.sum())


def total_variance(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """``Σλ`` — the trace of the centered covariance, i.e. the mean squared distance of the
    states from their centroid times ``s/(s-1)``.  The scale-carrying half of spec §14.3."""
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    if lam.size == 0:
        return float("nan")
    return float(lam.sum())


def peer_centroid(X: np.ndarray, agent: int) -> np.ndarray:
    """Centroid of every state except ``agent`` (spec §14.3 "distance to the peer centroid")."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise GeometryError("a peer centroid needs at least two states")
    mask = np.ones(X.shape[0], dtype=bool)
    mask[int(agent)] = False
    return X[mask].mean(axis=0)


# --------------------------------------------------------------------------- team grain


def team_metrics(
    X: np.ndarray,
    *,
    standardizer: Standardizer | None = None,
    normalised: bool = False,
) -> dict[str, Any]:
    """Every §14.3 team statistic for one ``s x d`` set of comparable states.

    ``normalised=False`` runs :func:`spectrum_metrics` on the raw states;
    ``normalised=True`` unit-normalises every state first, which is the scale-free variant
    the spec asks to be shown *beside* the raw one.  The returned mapping carries the
    frozen ``neural_geometry`` metric names plus the study_v4 spectrum block.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise GeometryError(f"state matrix must be 2-d and non-empty, got shape {X.shape}")
    base = unit_normalize(X) if normalised else X
    metrics = spectrum_metrics(base, standardizer=standardizer)
    lam = np.asarray(metrics.eigenvalues, dtype=np.float64)
    return {
        "participation_ratio": float(metrics.participation_ratio),
        "entropy_rank": float(metrics.entropy_effective_rank),
        "leading_eigenvalue_share": leading_eigenvalue_share(lam),
        "total_variance": total_variance(lam),
        "mean_pairwise": float(metrics.raw_pairwise_mean),
        "min_pairwise": float(metrics.raw_pairwise_min),
        "max_pairwise": float(metrics.raw_pairwise_max),
        "cosine_distance": float(metrics.cosine_distance_unit),
        "normalised": bool(normalised),
        # ---- kept beside the frozen columns -----------------------------------------
        "cosine_distance_standardized": float(metrics.cosine_distance_standardized),
        "participation_ratio_unit": float(metrics.participation_ratio_unit),
        "entropy_rank_unit": float(metrics.entropy_effective_rank_unit),
        "state_norm_mean": float(metrics.raw_norm_mean),
        "state_norm_std": float(metrics.raw_norm_std),
        "n_states": int(metrics.s),
        "hidden_size": int(metrics.d),
        "rank_ceiling": int(metrics.rank_ceiling),
        "n_nonzero_eigenvalues": int(metrics.n_nonzero_eigenvalues),
        "eigenvalues": [float(v) for v in metrics.eigenvalues],
        "spectrum_testable": bool(metrics.spectrum_testable),
        "representation": ("unit_normalised+" if normalised else "raw+") + metrics.representation,
    }


# --------------------------------------------------------------------------- agent grain


def agent_metrics(
    X: np.ndarray,
    *,
    previous: np.ndarray | None = None,
    agent_ids: Sequence[int] | None = None,
    eps: float = DEGENERATE_EPS,
) -> list[dict[str, Any]]:
    """The §14.3 per-agent statistics for one round.

    ``X`` is the ``s x d`` matrix of this round's states; ``previous`` (optional) is the
    same agents' states at ``t-1``, in the same row order.  With ``previous`` given, each
    agent also gets

    * ``own_displacement`` = ``||x_t - x_{t-1}||``;
    * ``displacement_alignment`` = ``cos(x_t - x_{t-1}, c_{t-1} - x_{t-1})`` where
      ``c_{t-1}`` is the **pre-update** centroid of that agent's peers — the direction that
      points from the agent to its peers *before* the round it is about to run;
    * ``near_zero_denominator``: True when either norm is ``<= eps``.  The alignment is then
      ``NaN``; it is never computed by dividing by a near-zero number (spec §14.3).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise GeometryError("per-agent geometry needs at least two states")
    s = X.shape[0]
    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != X.shape:
            raise GeometryError(f"previous states {previous.shape} != current {X.shape}")
    ids = list(range(s)) if agent_ids is None else [int(a) for a in agent_ids]
    if len(ids) != s:
        raise GeometryError(f"agent_ids has {len(ids)} entries for {s} states")

    rows: list[dict[str, Any]] = []
    for i in range(s):
        centroid = peer_centroid(X, i)
        deltas = X - X[i]
        peer_d = np.linalg.norm(np.delete(deltas, i, axis=0), axis=1)
        nearest = int(np.argmin(peer_d))
        nearest_id = ids[nearest if nearest < i else nearest + 1]
        row: dict[str, Any] = {
            "agent_id": ids[i],
            "state_norm": float(np.linalg.norm(X[i])),
            "distance_to_peer_centroid": float(np.linalg.norm(X[i] - centroid)),
            "distance_to_nearest_peer": float(peer_d.min()),
            "own_displacement": float("nan"),
            "displacement_alignment": float("nan"),
            "near_zero_denominator": False,
            "nearest_peer_agent": nearest_id,
            "displacement_denominator": float("nan"),
            "direction_denominator": float("nan"),
            "has_previous_state": previous is not None,
        }
        if previous is not None:
            displacement = X[i] - previous[i]
            direction = peer_centroid(previous, i) - previous[i]
            d_norm = float(np.linalg.norm(displacement))
            dir_norm = float(np.linalg.norm(direction))
            degenerate = d_norm <= eps or dir_norm <= eps
            row["own_displacement"] = d_norm
            row["displacement_denominator"] = d_norm
            row["direction_denominator"] = dir_norm
            row["near_zero_denominator"] = bool(degenerate)
            row["displacement_alignment"] = (
                float("nan") if degenerate else float(np.dot(displacement, direction) / (d_norm * dir_norm))
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- tables


@dataclass(frozen=True)
class StateSet:
    """One (item, protocol, round, layer, anchor) group of comparable agent states.

    ``states`` is ``s x d`` in ``agent_ids`` order — the five agents of one team at one
    capture site.  Raw activation coordinates are only comparable within one checkpoint
    (spec §14.1), so ``checkpoint`` is part of the group's identity.
    """

    item_id: str
    protocol: str
    round_: int
    layer: int
    anchor: str
    states: np.ndarray
    agent_ids: tuple[int, ...]
    checkpoint: str = K.CHECKPOINT_SMALL
    episode_id: str = ""
    cohort: str = K.Cohort.CAPTURE.value
    split: str = ""
    family: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.states, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != len(self.agent_ids):
            raise GeometryError(
                f"states {getattr(self.states, 'shape', None)} do not match agent_ids {self.agent_ids}"
            )
        if self.anchor not in K.CAPTURE_ANCHORS:
            raise GeometryError(f"anchor {self.anchor!r} not in contract.CAPTURE_ANCHORS")
        object.__setattr__(self, "states", X)

    @property
    def key(self) -> tuple[str, str, str, int, int, str]:
        return (self.checkpoint, self.item_id, self.protocol, int(self.round_), int(self.layer), self.anchor)

    @property
    def previous_key(self) -> tuple[str, str, str, int, int, str]:
        return (self.checkpoint, self.item_id, self.protocol, int(self.round_) - 1, int(self.layer), self.anchor)

    @property
    def feature_time(self) -> str:
        return A.feature_time_of(self.anchor)

    @property
    def last_allowed_round(self) -> int:
        return A.last_allowed_round(self.anchor, self.round_)


def _identity(state_set: StateSet) -> dict[str, Any]:
    return {
        "item_id": state_set.item_id,
        "protocol": state_set.protocol,
        "round": int(state_set.round_),
        "layer": int(state_set.layer),
        "anchor": state_set.anchor,
    }


def _tail(state_set: StateSet) -> dict[str, Any]:
    return {
        "feature_time": state_set.feature_time,
        "last_allowed_round": state_set.last_allowed_round,
        "checkpoint": state_set.checkpoint,
        "episode_id": state_set.episode_id,
        "cohort": state_set.cohort,
        "split": state_set.split,
        "family": state_set.family,
        **dict(state_set.extra),
    }


def neural_tables(
    state_sets: Iterable[StateSet],
    *,
    standardizer: Standardizer | None = None,
    eps: float = DEGENERATE_EPS,
) -> dict[str, list[dict[str, Any]]]:
    """``{"neural_geometry": [...], "neural_agent": [...]}`` for a collection of groups.

    Every team row is emitted twice (``normalised`` False/True, spec §14.3 "raw and
    normalized metrics must be shown together").  A group whose ``round - 1`` sibling is
    present in the same collection (same checkpoint/item/protocol/layer/anchor, matching
    ``agent_ids``) gets the displacement and alignment columns; otherwise those are ``NaN``
    with ``has_previous_state = False``.  Rows are ordered by the group key so the table is
    reproducible.
    """
    groups = list(state_sets)
    by_key = {g.key: g for g in groups}
    if len(by_key) != len(groups):
        dupes = sorted({g.key for g in groups if sum(1 for h in groups if h.key == g.key) > 1})
        raise GeometryError(f"duplicate state-set keys: {dupes[:4]}")

    geometry_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda g: g.key):
        identity = _identity(group)
        tail = _tail(group)
        for normalised in (False, True):
            metrics = team_metrics(group.states, standardizer=standardizer, normalised=normalised)
            geometry_rows.append({**identity, **metrics, **tail})
        previous = by_key.get(group.previous_key)
        prev_states = None
        if previous is not None and tuple(previous.agent_ids) == tuple(group.agent_ids):
            prev_states = previous.states
        for row in agent_metrics(group.states, previous=prev_states, agent_ids=group.agent_ids, eps=eps):
            agent_rows.append({**identity, **row, **tail})
    return {"neural_geometry": geometry_rows, "neural_agent": agent_rows}


def ordered_columns(table: str, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The frozen ``ard.compact`` columns first, then anything this module added."""
    frozen = GEOMETRY_COLUMNS if table == "neural_geometry" else AGENT_COLUMNS
    seen: list[str] = [c for c in frozen]
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen
