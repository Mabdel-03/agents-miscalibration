"""N3 geometry tests (CPU): exact spectra, rank ceilings, hash-determined subsampling, PCA
fold isolation and the comparable-state assembly adapter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents_scaling.study.neural import geometry as G

SEED = bytes(range(32))


def states_with_spectrum(eigenvalues: list[float], s: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """``s x d`` centered states whose sample covariance has exactly ``eigenvalues`` (rest 0)."""
    k = len(eigenvalues)
    assert k <= s - 1 <= d
    # orthonormal score directions orthogonal to the all-ones vector (centering-invariant)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(s), rng.normal(size=(s, k))]))
    scores = Q[:, 1 : k + 1] * np.sqrt((s - 1) * np.asarray(eigenvalues))[None, :]
    basis, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return scores @ basis.T


def test_participation_ratio_and_effective_rank_exact():
    lam = [4.0, 2.0, 1.0, 1.0]
    pr = sum(lam) ** 2 / sum(v * v for v in lam)
    p = np.asarray(lam) / sum(lam)
    er = math.exp(-float(np.sum(p * np.log(p))))
    assert G.participation_ratio(lam) == pytest.approx(pr)
    assert G.entropy_effective_rank(lam) == pytest.approx(er)
    # degenerate cases: equal eigenvalues → both equal the count; a single one → 1
    assert G.participation_ratio([3.0, 3.0, 3.0]) == pytest.approx(3.0)
    assert G.entropy_effective_rank([3.0, 3.0, 3.0]) == pytest.approx(3.0)
    assert G.entropy_effective_rank([5.0]) == pytest.approx(1.0)
    assert math.isnan(G.participation_ratio([]))
    # zeros are dropped (nonzero eigenvalues only, spec §8.5)
    assert G.participation_ratio([2.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_spectrum_metrics_recover_planted_eigenvalues_and_ceiling():
    rng = np.random.default_rng(1)
    lam = [5.0, 3.0, 1.0]
    X = states_with_spectrum(lam, s=5, d=40, rng=rng)
    m = G.spectrum_metrics(X)
    assert m.s == 5 and m.d == 40 and m.rank_ceiling == 4 == min(40, 5 - 1)
    assert list(np.round(m.eigenvalues, 8)) == pytest.approx(lam)
    assert m.n_nonzero_eigenvalues == 3 <= m.rank_ceiling
    assert m.participation_ratio == pytest.approx(G.participation_ratio(lam))
    assert m.entropy_effective_rank == pytest.approx(G.entropy_effective_rank(lam))
    assert m.spectrum_testable
    # raw quantities are on the raw vectors
    assert m.raw_norms == pytest.approx(tuple(np.linalg.norm(X, axis=1)))
    assert m.raw_pairwise_mean == pytest.approx(float(G.pairwise_distances(X).mean()))
    assert len(G.pairwise_distances(X)) == 10


def test_rank_ceiling_binds_when_states_exceed_dimension():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))  # s-1 = 11 > d = 3
    m = G.spectrum_metrics(X)
    assert m.rank_ceiling == 3 and m.n_nonzero_eigenvalues <= 3
    # N=1 has no within-team covariance dimension
    one = G.spectrum_metrics(rng.normal(size=(1, 8)))
    assert one.rank_ceiling == 0 and one.n_nonzero_eigenvalues == 0 and not one.spectrum_testable
    assert math.isnan(one.participation_ratio) and math.isnan(one.raw_pairwise_mean)
    two = G.spectrum_metrics(rng.normal(size=(2, 8)))
    assert two.rank_ceiling == 1 and not two.spectrum_testable and math.isfinite(two.raw_pairwise_mean)


def test_standardized_and_unit_cosine_distances():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 20)) * np.linspace(1, 10, 20)[None, :]
    std = G.Standardizer.fit(rng.normal(size=(50, 20)) * np.linspace(1, 10, 20)[None, :])
    m = G.spectrum_metrics(X, standardizer=std)
    assert m.representation == "centered+standardized"
    Z = std.transform(X)
    Zc = Z - Z.mean(axis=0)
    assert m.cosine_distance_standardized == pytest.approx(G.mean_pairwise_cosine_distance(Zc))
    assert m.cosine_distance_unit == pytest.approx(G.mean_pairwise_cosine_distance(G.unit_normalize(X)))
    with pytest.raises(G.GeometryError):
        G.unit_normalize(np.vstack([X, np.zeros(20)]))


def test_fixed_s_subsampling_is_hash_deterministic_and_averages():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(9, 30))
    a = G.fixed_s_metrics(X, 5, study_seed=SEED, item="hle:1", config="IND_VOTE|ROOT", count=20)
    b = G.fixed_s_metrics(X, 5, study_seed=SEED, item="hle:1", config="IND_VOTE|ROOT", count=20)
    assert a.status == "subsampled" and a.n_subsamples == 20 and a.rank_ceiling == 4
    assert a.metrics == b.metrics and a.eigenvalues_mean == b.eigenvalues_mean
    other = G.fixed_s_metrics(X, 5, study_seed=SEED, item="hle:2", config="IND_VOTE|ROOT", count=20)
    assert other.metrics["participation_ratio"] != a.metrics["participation_ratio"]
    draws = G.subsample_indices(9, 5, SEED, "hle:1", "IND_VOTE|ROOT", count=20)
    assert draws.shape == (20, 5) and all(len(set(r)) == 5 for r in draws)
    # the Monte Carlo mean equals the explicit average over the same draws
    expected = np.mean([G.spectrum_metrics(X[idx]).participation_ratio for idx in draws])
    assert a.metrics["participation_ratio"] == pytest.approx(expected)
    # exact when n == s; insufficient when n < s (reported, not imputed)
    exact = G.fixed_s_metrics(X[:5], 5, study_seed=SEED, item="i", config="c")
    assert exact.status == "exact" and exact.n_subsamples == 1 and exact.metrics["participation_ratio"] == pytest.approx(G.spectrum_metrics(X[:5]).participation_ratio)
    short = G.fixed_s_metrics(X[:3], 5, study_seed=SEED, item="i", config="c")
    assert short.status == "insufficient" and short.rank_ceiling == 2 and math.isnan(short.metrics["participation_ratio"])
    assert short.full_set is not None and short.full_set.s == 3
    with pytest.raises(G.GeometryError):
        G.subsample_indices(3, 5, SEED, "i", "c")


def test_pca_fold_isolation_and_clipping(tmp_path):
    rng = np.random.default_rng(5)
    train = rng.normal(size=(20, 50))
    test_a = rng.normal(size=(7, 50))
    test_b = rng.normal(size=(7, 50)) * 100.0
    pca = G.DevPCA(rank_requested=64).fit(train)
    # clipped to the training rank min(64, 20-1, 50) = 19
    assert pca.rank == 19 and pca.clipped
    assert G.DevPCA.training_rank(64, 20, 50) == 19
    hash_before = pca.components_hash
    za = pca.transform(test_a)
    pca.transform(test_b)
    assert pca.components_hash == hash_before  # transform never refits
    assert np.allclose(pca.transform(test_a), za)
    # the map depends only on the training states, not on which held-out set is transformed
    other = G.DevPCA(rank_requested=64).fit(train)
    assert other.components_hash == hash_before
    assert np.allclose(other.transform(test_a), za)
    # a fit on different data is a different map
    assert G.DevPCA(rank_requested=64).fit(np.vstack([train, test_a])).components_hash != hash_before
    # projections reproduce the centered training data when the rank is not clipped
    full = G.DevPCA(rank_requested=8).fit(rng.normal(size=(30, 8)))
    assert full.rank == 8 and not full.clipped
    # fit_transform == fit + transform; components are orthonormal
    ft = G.DevPCA(rank_requested=5).fit_transform(train)
    assert np.allclose(ft, G.DevPCA(rank_requested=5).fit(train).transform(train))
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(19), atol=1e-8)
    # frozen display transform round trip (dotted stem must survive)
    disp = G.display_transform(train, rank=6, label="display")
    path = disp.save(tmp_path / "display_pca.32B.b15.NATIVE_PREFILL")
    assert path.name == "display_pca.32B.b15.NATIVE_PREFILL.json"
    loaded = G.DevPCA.load(tmp_path / "display_pca.32B.b15.NATIVE_PREFILL")
    assert loaded.components_hash == disp.components_hash and loaded.extra["purpose"] == "display_only"
    assert np.allclose(loaded.transform(test_a), disp.transform(test_a))
    with pytest.raises(G.GeometryError):
        G.DevPCA(rank_requested=4).fit(train[:1])


def test_spectrum_with_projector_uses_pca_rank_as_d_eff():
    rng = np.random.default_rng(6)
    dev = rng.normal(size=(40, 30))
    pca = G.DevPCA(rank_requested=8).fit(dev)
    m = G.spectrum_metrics(rng.normal(size=(5, 30)), projector=pca)
    assert m.d_eff == 8 and m.rank_ceiling == 4 and m.representation.endswith("pca8")
    m2 = G.spectrum_metrics(rng.normal(size=(12, 30)), projector=pca)
    assert m2.rank_ceiling == 8 and m2.n_nonzero_eigenvalues <= 8


def test_assemble_comparable_states_from_store_rows():
    import pandas as pd

    rows = []
    for item in ("hle:a", "bcb:b"):
        for slot in range(5):
            rows.append({"StateSnapshot_id": f"{item}-{slot}", "source_id": item, "method": "IND_VOTE", "role": "INDEPENDENT_SOLVER", "phase": "ROOT",
                         "block": 15, "anchor_kind": "NATIVE_PREFILL", "vector_index": len(rows), "extra": {"actor_slot": slot}, "child_slot": None})
        rows.append({"StateSnapshot_id": f"{item}-hub", "source_id": item, "method": "CEN_FLAT", "role": "CENTRAL_HUB", "phase": "ROOT", "block": 15,
                     "anchor_kind": "NATIVE_PREFILL", "vector_index": len(rows), "extra": {}, "child_slot": None})
        rows.append({"StateSnapshot_id": f"{item}-w", "source_id": item, "method": "CEN_FLAT", "role": "CENTRAL_WORKER", "phase": "COORDINATION", "block": 15,
                     "anchor_kind": "GENERATED_512", "vector_index": -1, "missingness": "NOT_REACHED", "extra": {}, "child_slot": None})
    meta = G.normalize_meta(pd.DataFrame.from_records(rows))
    assert set(G.META_COLUMNS) <= set(meta.columns)
    groups = G.assemble_comparable_states(meta, blocks=[15])
    by = {(g.item, g.role, g.anchor): g for g in groups}
    ind = by[("hle:a", "INDEPENDENT_SOLVER", "NATIVE_PREFILL")]
    assert ind.s_available == 5 and ind.slots == (0, 1, 2, 3, 4) and ind.n_missing == 0
    assert G.contract_s("IND_VOTE", "INDEPENDENT_SOLVER", "ROOT") == 5
    hub = by[("hle:a", "CENTRAL_HUB", "NATIVE_PREFILL")]
    assert hub.s_available == 1  # never pooled with workers
    worker = by[("hle:a", "CENTRAL_WORKER", "GENERATED_512")]
    assert worker.s_available == 0 and worker.n_missing == 1  # missing anchors are counted, never zero vectors
    assert G.contract_s("DEC", "DECENTRALIZED_MEMBER", "TERMINAL") == 5 and G.contract_s("DEC", "X", "Y") is None
