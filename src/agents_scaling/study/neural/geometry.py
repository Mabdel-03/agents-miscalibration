"""Geometry metrics for comparable agent-state vectors (N3; spec §8.5, brief R3).

For ``s`` comparable state vectors of one (item, configuration, checkpoint) this module
reports, exactly as §8.5 asks:

* raw norms and raw pairwise distances (never normalized away);
* mean pairwise cosine distance on (a) the centered, development-standardized
  representation and (b) the unit-normalized representation;
* the covariance eigenvalues of the centered representation, the participation ratio
  ``(sum λ)^2 / sum λ^2`` and the entropy effective rank ``exp(-sum p log p)``;
* the rank ceiling ``min(d_eff, s-1)`` printed beside every estimate;
* fixed-``s`` Monte Carlo averaging over 100 hash-determined subsamples (seeded from the
  study seed + item + configuration) whenever more than ``s`` states exist;
* a 64-d development-fitted PCA per checkpoint, clipped to the training rank, with a
  ``fit``/``fit_transform``/``transform`` API (fit inside training folds for prediction) and
  a separately frozen display transform for plots.

Comparable-state assembly reads the N1 ``ActivationRow`` store through
:func:`load_states`, an adapter over ``storage.load_stage`` that normalizes the metadata
frame to the columns ``id, item, method, role, phase, block, anchor, slot``.  Nothing here
reads labels; the association with outcomes is a join done in ``analysis``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents_scaling.study import identity

#: Spec §8.5: "Freeze 100 hash-determined subsamples per episode".
DEFAULT_SUBSAMPLES = 100
#: Spec §8.5: "Use a 64-dimensional development-fitted PCA by default".
DEFAULT_PCA_RANK = 64
#: Below this eigenvalue mass (relative to the largest) an eigenvalue is numerically zero.
EIGEN_REL_TOL = 1e-10
SUBSAMPLE_NAMESPACE = "GEOMETRY_SUBSAMPLE"
#: Minimum ``s`` for a dimension claim (§8.5 "if s<3, report pairwise-distance summaries").
MIN_S_FOR_SPECTRUM = 3

#: Role contract of the brief (R3): comparable role states per flagship N=5 policy.  The
#: guaranteed count is the minimum the role contract promises; hub and worker states of
#: CEN_FLAT are *different roles* and are never pooled (§8.3).
ROLE_CONTRACTS: Mapping[str, tuple[tuple[str, str, int], ...]] = {
    "IND_VOTE": (("INDEPENDENT_SOLVER", "ROOT", 5),),
    "DEC": (("DECENTRALIZED_MEMBER", "TERMINAL", 5), ("DECENTRALIZED_MEMBER", "ROOT", 5)),
    "CEN_FLAT": (("CENTRAL_HUB", "ROOT", 1), ("CENTRAL_WORKER", "COORDINATION", 1)),
}

META_COLUMNS: tuple[str, ...] = ("id", "item", "method", "role", "phase", "block", "anchor", "slot")


class GeometryError(ValueError):
    """An input that cannot be reduced to a geometry estimate (never silently zeroed)."""


# --------------------------------------------------------------------------- standardization


@dataclass(frozen=True)
class Standardizer:
    """Development-fitted per-dimension centering/scaling (§8.5 "development-standardized")."""

    mean: np.ndarray
    scale: np.ndarray
    n_fit: int

    @classmethod
    def fit(cls, X: np.ndarray, *, eps: float = 1e-6) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise GeometryError("a standardizer needs at least two development states")
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=1)
        scale = np.where(scale > eps, scale, 1.0)
        return cls(mean=mean, scale=scale, n_fit=int(X.shape[0]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Standardizer":
        return cls(np.asarray(data["mean"], dtype=np.float64), np.asarray(data["scale"], dtype=np.float64), int(data["n_fit"]))


# --------------------------------------------------------------------------- spectra


def participation_ratio(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """``(sum λ)^2 / sum λ^2`` over the nonzero eigenvalues; NaN when there are none."""
    lam = _nonzero(eigenvalues)
    if lam.size == 0:
        return float("nan")
    return float(lam.sum() ** 2 / np.sum(lam**2))


def entropy_effective_rank(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """``exp(-sum p log p)`` with ``p = λ / sum λ`` over the nonzero eigenvalues."""
    lam = _nonzero(eigenvalues)
    if lam.size == 0:
        return float("nan")
    p = lam / lam.sum()
    return float(math.exp(-float(np.sum(p * np.log(p)))))


def rank_ceiling(d_eff: int, s: int) -> int:
    """Spec §8.5: the centered covariance rank is at most ``min(d_eff, s-1)``."""
    return int(max(0, min(int(d_eff), int(s) - 1)))


def _nonzero(eigenvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    lam = lam[np.isfinite(lam)]
    if lam.size == 0:
        return lam
    top = float(lam.max())
    if top <= 0:
        return lam[:0]
    return lam[lam > top * EIGEN_REL_TOL]


def covariance_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Nonzero eigenvalues (descending) of the sample covariance ``(1/(s-1)) Xc^T Xc``.

    Computed through the ``s x s`` Gram matrix so ``d = 5,120`` costs nothing; the number of
    returned values never exceeds ``min(d, s-1)``.
    """
    X = np.asarray(X, dtype=np.float64)
    s, d = X.shape
    if s < 2:
        return np.zeros(0, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    gram = Xc @ Xc.T / (s - 1)
    lam = np.linalg.eigvalsh(gram)[::-1]
    lam = np.clip(lam, 0.0, None)
    return _nonzero(lam)[: rank_ceiling(d, s)]


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Condensed Euclidean distances (``s(s-1)/2`` values, ``i<j`` order)."""
    X = np.asarray(X, dtype=np.float64)
    s = X.shape[0]
    if s < 2:
        return np.zeros(0, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    iu = np.triu_indices(s, k=1)
    return np.sqrt(np.clip(d2[iu], 0.0, None))


def mean_pairwise_cosine_distance(X: np.ndarray) -> float:
    """Mean of ``1 - cos(x_i, x_j)`` over ``i<j``; NaN with fewer than two states or a zero vector."""
    X = np.asarray(X, dtype=np.float64)
    s = X.shape[0]
    if s < 2:
        return float("nan")
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms == 0):
        return float("nan")
    U = X / norms[:, None]
    cos = U @ U.T
    iu = np.triu_indices(s, k=1)
    return float(np.mean(1.0 - cos[iu]))


def unit_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise GeometryError("a zero-norm state cannot be unit-normalized (missing anchors must not be zero vectors)")
    return X / norms


@dataclass(frozen=True)
class SpectrumMetrics:
    """Every §8.5 estimate for one set of ``s`` comparable states, with its ceiling."""

    s: int
    d: int
    d_eff: int
    rank_ceiling: int
    raw_norm_mean: float
    raw_norm_std: float
    raw_norms: tuple[float, ...]
    raw_pairwise_mean: float
    raw_pairwise_min: float
    raw_pairwise_max: float
    cosine_distance_standardized: float
    cosine_distance_unit: float
    eigenvalues: tuple[float, ...]
    n_nonzero_eigenvalues: int
    participation_ratio: float
    entropy_effective_rank: float
    participation_ratio_unit: float
    entropy_effective_rank_unit: float
    spectrum_testable: bool
    representation: str

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["raw_norms"] = list(self.raw_norms)
        out["eigenvalues"] = list(self.eigenvalues)
        return out


def spectrum_metrics(X: np.ndarray, *, standardizer: Standardizer | None = None, projector: "DevPCA | None" = None) -> SpectrumMetrics:
    """All §8.5 estimates for the ``s x d`` matrix ``X`` of comparable states.

    ``standardizer`` is the development-fitted per-dimension scaling (centering is always
    applied within the set); ``projector`` optionally maps the *standardized* states to the
    development PCA coordinates before the spectrum (then ``d_eff`` is the PCA rank).  Raw
    norms/distances and the unit-normalized cosine distance are always on the raw vectors.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise GeometryError(f"state matrix must be 2-d and non-empty, got shape {X.shape}")
    s, d = X.shape
    raw_norms = np.linalg.norm(X, axis=1)
    raw_pd = pairwise_distances(X)
    unit = unit_normalize(X) if np.all(raw_norms > 0) else None
    Z = standardizer.transform(X) if standardizer is not None else X.copy()
    Zc = Z - Z.mean(axis=0, keepdims=True)
    if projector is not None:
        Zc = projector.transform(Zc)
        representation = f"centered+standardized→pca{projector.rank}" if standardizer is not None else f"centered→pca{projector.rank}"
        d_eff = projector.rank
    else:
        representation = "centered+standardized" if standardizer is not None else "centered"
        d_eff = d
    ceiling = rank_ceiling(d_eff, s)
    lam = covariance_eigenvalues(Zc) if s >= 2 else np.zeros(0)
    lam_unit = covariance_eigenvalues(unit - unit.mean(axis=0, keepdims=True)) if (unit is not None and s >= 2) else np.zeros(0)
    testable = s >= MIN_S_FOR_SPECTRUM
    return SpectrumMetrics(
        s=int(s), d=int(d), d_eff=int(d_eff), rank_ceiling=ceiling,
        raw_norm_mean=float(raw_norms.mean()), raw_norm_std=float(raw_norms.std(ddof=0)), raw_norms=tuple(float(v) for v in raw_norms),
        raw_pairwise_mean=float(raw_pd.mean()) if raw_pd.size else float("nan"),
        raw_pairwise_min=float(raw_pd.min()) if raw_pd.size else float("nan"),
        raw_pairwise_max=float(raw_pd.max()) if raw_pd.size else float("nan"),
        cosine_distance_standardized=mean_pairwise_cosine_distance(Zc),
        cosine_distance_unit=mean_pairwise_cosine_distance(X),
        eigenvalues=tuple(float(v) for v in lam), n_nonzero_eigenvalues=int(lam.size),
        participation_ratio=participation_ratio(lam) if testable else float("nan"),
        entropy_effective_rank=entropy_effective_rank(lam) if testable else float("nan"),
        participation_ratio_unit=participation_ratio(lam_unit) if testable else float("nan"),
        entropy_effective_rank_unit=entropy_effective_rank(lam_unit) if testable else float("nan"),
        spectrum_testable=testable, representation=representation,
    )


# --------------------------------------------------------------------------- fixed-s subsampling


def subsample_seed(study_seed: bytes, item: str, config: str, index: int) -> int:
    """Hash-determined seed of subsample ``index`` for (item, config): HMAC(study_seed, ...)."""
    digest = identity.blind_order_key(study_seed, SUBSAMPLE_NAMESPACE, str(item), str(config), int(index))
    return int.from_bytes(digest[:8], "big")


def subsample_indices(n: int, s: int, study_seed: bytes, item: str, config: str, *, count: int = DEFAULT_SUBSAMPLES) -> np.ndarray:
    """``count x s`` index matrix: each row is a without-replacement draw of ``s`` of ``n``."""
    if s > n:
        raise GeometryError(f"cannot draw s={s} of n={n} states without replacement")
    out = np.empty((count, s), dtype=np.int64)
    for k in range(count):
        rng = np.random.default_rng(subsample_seed(study_seed, item, config, k))
        out[k] = np.sort(rng.choice(n, size=s, replace=False))
    return out


_AVERAGED_FIELDS: tuple[str, ...] = (
    "raw_norm_mean", "raw_norm_std", "raw_pairwise_mean", "raw_pairwise_min", "raw_pairwise_max",
    "cosine_distance_standardized", "cosine_distance_unit", "participation_ratio", "entropy_effective_rank",
    "participation_ratio_unit", "entropy_effective_rank_unit",
)


@dataclass(frozen=True)
class FixedSMetrics:
    """Fixed-``s`` estimate: either the single full set (``n == s``) or the Monte Carlo mean
    over the hash-determined subsamples (``n > s``); ``n < s`` is reported, not imputed."""

    s: int
    n_available: int
    n_subsamples: int
    status: str  # "exact" | "subsampled" | "insufficient"
    rank_ceiling: int
    d_eff: int
    metrics: dict[str, float]
    metrics_sd: dict[str, float]
    eigenvalues_mean: tuple[float, ...]
    full_set: SpectrumMetrics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "s": self.s, "n_available": self.n_available, "n_subsamples": self.n_subsamples, "status": self.status,
            "rank_ceiling": self.rank_ceiling, "d_eff": self.d_eff, "metrics": dict(self.metrics), "metrics_sd": dict(self.metrics_sd),
            "eigenvalues_mean": list(self.eigenvalues_mean), "full_set": None if self.full_set is None else self.full_set.to_dict(),
        }


def fixed_s_metrics(
    X: np.ndarray,
    s: int,
    *,
    study_seed: bytes,
    item: str,
    config: str,
    standardizer: Standardizer | None = None,
    projector: "DevPCA | None" = None,
    count: int = DEFAULT_SUBSAMPLES,
) -> FixedSMetrics:
    """§8.5 fixed-``s`` estimate with the full-membership spectrum kept as descriptive data."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    full = spectrum_metrics(X, standardizer=standardizer, projector=projector) if n >= 1 else None
    d_eff = full.d_eff if full is not None else (projector.rank if projector is not None else int(X.shape[1]))
    if n < s:
        return FixedSMetrics(s=int(s), n_available=int(n), n_subsamples=0, status="insufficient", rank_ceiling=rank_ceiling(d_eff, n),
                             d_eff=int(d_eff), metrics={k: float("nan") for k in _AVERAGED_FIELDS}, metrics_sd={}, eigenvalues_mean=(), full_set=full)
    if n == s:
        assert full is not None
        return FixedSMetrics(s=int(s), n_available=int(n), n_subsamples=1, status="exact", rank_ceiling=full.rank_ceiling, d_eff=int(d_eff),
                             metrics={k: float(getattr(full, k)) for k in _AVERAGED_FIELDS}, metrics_sd={k: 0.0 for k in _AVERAGED_FIELDS},
                             eigenvalues_mean=full.eigenvalues, full_set=full)
    draws = subsample_indices(n, s, study_seed, item, config, count=count)
    per: dict[str, list[float]] = {k: [] for k in _AVERAGED_FIELDS}
    ceiling = rank_ceiling(d_eff, s)
    eig = np.zeros((count, ceiling), dtype=np.float64)
    for k, idx in enumerate(draws):
        m = spectrum_metrics(X[idx], standardizer=standardizer, projector=projector)
        for name in _AVERAGED_FIELDS:
            per[name].append(float(getattr(m, name)))
        lam = np.asarray(m.eigenvalues)
        eig[k, : min(lam.size, ceiling)] = lam[:ceiling]
    metrics = {k: float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan") for k, v in per.items()}
    sds = {k: float(np.nanstd(v, ddof=0)) if np.any(np.isfinite(v)) else float("nan") for k, v in per.items()}
    return FixedSMetrics(s=int(s), n_available=int(n), n_subsamples=int(count), status="subsampled", rank_ceiling=ceiling, d_eff=int(d_eff),
                         metrics=metrics, metrics_sd=sds, eigenvalues_mean=tuple(float(v) for v in eig.mean(axis=0)), full_set=full)


# --------------------------------------------------------------------------- PCA


@dataclass
class DevPCA:
    """Development-fitted PCA, clipped to the training rank (§8.5, §8.8).

    ``fit`` centers on the training mean and keeps ``min(rank, n_train-1, d)`` components
    (the training rank); ``transform`` never sees test data during fitting, which is what the
    fold-isolation test checks.  ``components_hash`` is the immutable identity of a frozen map.
    """

    rank_requested: int = DEFAULT_PCA_RANK
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None  # [rank, d]
    explained_variance_: np.ndarray | None = None
    n_fit_: int = 0
    d_: int = 0
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.components_ is not None

    @property
    def rank(self) -> int:
        return 0 if self.components_ is None else int(self.components_.shape[0])

    @property
    def clipped(self) -> bool:
        return self.rank < self.rank_requested

    @staticmethod
    def training_rank(rank: int, n: int, d: int) -> int:
        return int(max(0, min(int(rank), int(n) - 1, int(d))))

    def fit(self, X: np.ndarray) -> "DevPCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise GeometryError("PCA needs at least two training states")
        n, d = X.shape
        r = self.training_rank(self.rank_requested, n, d)
        mean = X.mean(axis=0)
        Xc = X - mean
        # thin SVD through the Gram matrix when n << d (the 5,120-d case), else directly
        if n <= d:
            gram = Xc @ Xc.T
            w, V = np.linalg.eigh(gram)
            order = np.argsort(w)[::-1]
            w, V = w[order], V[:, order]
            keep = w > max(float(w[0]), 0.0) * 1e-12 if w.size and w[0] > 0 else np.zeros(w.shape, dtype=bool)
            w, V = w[keep], V[:, keep]
            comps = (Xc.T @ V) / np.sqrt(np.clip(w, 1e-300, None))[None, :]
            comps = comps.T
            var = w / (n - 1)
        else:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps, var = Vt, S**2 / (n - 1)
        r = min(r, comps.shape[0])
        comps = comps[:r]
        # fix signs deterministically (largest-|coefficient| entry positive)
        for i in range(comps.shape[0]):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i, j] < 0:
                comps[i] = -comps[i]
        self.mean_, self.components_, self.explained_variance_ = mean, np.ascontiguousarray(comps), np.asarray(var[:r])
        self.n_fit_, self.d_ = int(n), int(d)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise GeometryError("PCA is not fitted")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def components_hash(self) -> str:
        if self.components_ is None or self.mean_ is None:
            raise GeometryError("PCA is not fitted")
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.mean_, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.components_, dtype="<f8").tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        if self.components_ is None:
            raise GeometryError("PCA is not fitted")
        return {
            "rank_requested": self.rank_requested, "rank": self.rank, "clipped": self.clipped, "n_fit": self.n_fit_, "d": self.d_,
            "label": self.label, "components_hash": self.components_hash, "mean": self.mean_.tolist(),
            "components": self.components_.tolist(), "explained_variance": self.explained_variance_.tolist(), "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DevPCA":
        pca = cls(rank_requested=int(data["rank_requested"]), label=str(data.get("label", "")), extra=dict(data.get("extra") or {}))
        pca.mean_ = np.asarray(data["mean"], dtype=np.float64)
        pca.components_ = np.asarray(data["components"], dtype=np.float64)
        pca.explained_variance_ = np.asarray(data["explained_variance"], dtype=np.float64)
        pca.n_fit_, pca.d_ = int(data["n_fit"]), int(data["d"])
        if pca.components_hash != data.get("components_hash", pca.components_hash):
            raise GeometryError("PCA components hash mismatch (corrupt frozen transform)")
        return pca

    def save(self, path: str | os.PathLike) -> Path:
        """Frozen transform: ``<path>.npz`` (arrays) + ``<path>.json`` (metadata + hash)."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        npz_path, json_path = _pca_paths(base)
        np.savez(npz_path, mean=self.mean_, components=self.components_, explained_variance=self.explained_variance_)
        meta = {k: v for k, v in self.to_dict().items() if k not in ("mean", "components", "explained_variance")}
        json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return json_path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "DevPCA":
        npz_path, json_path = _pca_paths(Path(path))
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as arrays:
            data = {**meta, "mean": arrays["mean"], "components": arrays["components"], "explained_variance": arrays["explained_variance"]}
        return cls.from_dict(data)


def _pca_paths(base: Path) -> tuple[Path, Path]:
    """``<base>.npz`` / ``<base>.json`` by string concatenation (``with_suffix`` would eat a
    dotted stem such as ``display_pca.32B.b15.NATIVE_PREFILL``)."""
    name = base.name[:-5] if base.name.endswith((".json", ".npz")) else base.name
    stem = base.with_name(name)
    return Path(str(stem) + ".npz"), Path(str(stem) + ".json")


def display_transform(X_dev: np.ndarray, *, rank: int = DEFAULT_PCA_RANK, label: str = "display") -> DevPCA:
    """The separately frozen display transform (§8.5): fitted once on development states,
    used only for plots, never for prediction."""
    pca = DevPCA(rank_requested=rank, label=label)
    pca.fit(X_dev)
    pca.extra["purpose"] = "display_only"
    return pca


# --------------------------------------------------------------------------- comparable-state assembly


def _slot_of(row: Mapping[str, Any]) -> int | None:
    extra = row.get("extra")
    if isinstance(extra, Mapping):
        for key in ("actor_slot", "slot"):
            v = extra.get(key)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
    v = row.get("child_slot")
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)


def normalize_meta(frame: Any) -> Any:
    """Adapter: the ``storage.load_stage`` frame → the N3 metadata frame
    (``id, item, method, role, phase, block, anchor, slot, vector_index, present`` + the
    original columns).  Works on a pandas frame or a list of row dicts."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame.from_records(list(frame))
    out = frame.copy()
    out["id"] = out["StateSnapshot_id"].astype(str) if "StateSnapshot_id" in out else out.get("id")
    out["item"] = out["source_id"] if "source_id" in out else out.get("item")
    out["anchor"] = out["anchor_kind"] if "anchor_kind" in out else out.get("anchor")
    for col in ("method", "role", "phase", "block"):
        if col not in out:
            out[col] = None
    out["block"] = out["block"].astype(int)
    out["slot"] = [_slot_of(r) for r in out.to_dict("records")] if len(out) else []
    if "vector_index" not in out:
        out["vector_index"] = -1
    out["present"] = out["vector_index"].astype(int) >= 0
    return out


def load_states(
    run_root: str | os.PathLike,
    stage: str,
    *,
    blocks: Iterable[int] | None = None,
    anchor_kinds: Iterable[str] | None = None,
    include_missing: bool = False,
) -> tuple[np.ndarray, Any]:
    """``(X float32 [n_present, d], meta)`` from the N1 store with the normalized columns."""
    from agents_scaling.study.neural.storage import load_stage

    X, frame = load_stage(run_root, stage, blocks=blocks, anchor_kinds=anchor_kinds, include_missing=include_missing)
    meta = normalize_meta(frame)
    return np.asarray(X, dtype=np.float32), meta


@dataclass(frozen=True)
class StateGroup:
    """The comparable states of one (item, method, role, phase, block, anchor)."""

    item: str
    method: str
    role: str
    phase: str
    block: int
    anchor: str
    ids: tuple[str, ...]
    slots: tuple[int | None, ...]
    rows: tuple[int, ...]  # row indices into X
    n_missing: int

    @property
    def config(self) -> str:
        return f"{self.method}|{self.role}|{self.phase}|b{self.block}|{self.anchor}"

    @property
    def s_available(self) -> int:
        return len(self.rows)


def assemble_comparable_states(meta: Any, *, blocks: Iterable[int] | None = None, anchor_kinds: Iterable[str] | None = None) -> list[StateGroup]:
    """Group present vectors by (item, method, role, phase, block, anchor); missing anchors
    are counted (``n_missing``), never imputed.  Hub and worker rows stay separate roles."""
    want_blocks = None if blocks is None else {int(b) for b in blocks}
    want_anchors = None if anchor_kinds is None else set(anchor_kinds)
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in meta.to_dict("records"):
        if want_blocks is not None and int(row["block"]) not in want_blocks:
            continue
        if want_anchors is not None and row["anchor"] not in want_anchors:
            continue
        key = (str(row["item"]), str(row["method"]), str(row["role"]), str(row["phase"]), int(row["block"]), str(row["anchor"]))
        g = groups.setdefault(key, {"ids": [], "slots": [], "rows": [], "missing": 0})
        if bool(row.get("present", int(row.get("vector_index", -1)) >= 0)):
            g["ids"].append(str(row["id"]))
            g["slots"].append(row.get("slot"))
            g["rows"].append(int(row["vector_index"]))
        else:
            g["missing"] += 1
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        order = np.argsort(np.asarray(g["ids"], dtype=object).astype(str), kind="stable")
        out.append(StateGroup(*key, ids=tuple(g["ids"][i] for i in order), slots=tuple(g["slots"][i] for i in order),
                              rows=tuple(g["rows"][i] for i in order), n_missing=int(g["missing"])))
    return out


def contract_s(method: str, role: str, phase: str) -> int | None:
    """The guaranteed comparable count of the role contract, or ``None`` when unregistered."""
    for r, p, s in ROLE_CONTRACTS.get(str(method), ()):
        if r == role and (p == phase or p == "*"):
            return int(s)
    return None


__all__ = [
    "DEFAULT_PCA_RANK",
    "DEFAULT_SUBSAMPLES",
    "META_COLUMNS",
    "MIN_S_FOR_SPECTRUM",
    "ROLE_CONTRACTS",
    "DevPCA",
    "FixedSMetrics",
    "GeometryError",
    "SpectrumMetrics",
    "Standardizer",
    "StateGroup",
    "assemble_comparable_states",
    "contract_s",
    "covariance_eigenvalues",
    "display_transform",
    "entropy_effective_rank",
    "fixed_s_metrics",
    "load_states",
    "mean_pairwise_cosine_distance",
    "normalize_meta",
    "pairwise_distances",
    "participation_ratio",
    "rank_ceiling",
    "spectrum_metrics",
    "subsample_indices",
    "subsample_seed",
    "unit_normalize",
]
