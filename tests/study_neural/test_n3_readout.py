"""N3 family-G tests (CPU): fold construction, fold-isolated features, the one-SE rule, the
nested-CV harness on separable vs pure-noise data, the freeze round trip and the bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from agents_scaling.study.neural import readout as R
from tests.study_neural.n3_support import synthetic_frames

SEED = bytes(range(32))
SMALL = dict(ranks=(4, 8), penalties=(1.0, 100.0), text_prefer="tfidf", seed=SEED, text_dims=8)


def test_grouped_stratified_folds_keep_clusters_together_and_balance_strata():
    clusters = [f"hle:{i}" for i in range(20)] * 3 + [f"bcb:{i}" for i in range(15)] * 3
    strata = ["hle"] * 60 + ["bcb"] * 45
    folds = R.grouped_stratified_folds(clusters, strata, 5, SEED)
    assert folds.shape == (105,)
    by_cluster = {}
    for c, f in zip(clusters, folds):
        by_cluster.setdefault(c, set()).add(int(f))
    assert all(len(v) == 1 for v in by_cluster.values())  # every derivative of an item in one fold
    for stratum in ("hle", "bcb"):
        counts = np.bincount([f for c, f in zip(clusters, folds) if c.startswith(stratum)], minlength=5)
        assert counts.max() - counts.min() <= 3  # round-robin within the stratum (3 rows per cluster)
    assert np.array_equal(folds, R.grouped_stratified_folds(clusters, strata, 5, SEED))
    assert not np.array_equal(folds, R.grouped_stratified_folds(clusters, strata, 5, b"\x01" * 32))


def test_one_se_rule_prefers_regularization_then_shallow_block_then_small_rank():
    cands = [((47, 128, 0.1), 0.200, 0.010), ((15, 16, 100.0), 0.208, 0.010), ((31, 32, 1000.0), 0.207, 0.010), ((15, 64, 1000.0), 0.209, 0.010), ((47, 16, 10.0), 0.215, 0.010)]
    sel = R.one_se_select(cands, block_order=(15, 31, 47))
    assert (sel.block, sel.rank, sel.penalty) == (15, 64, 1000.0)  # within 1 SE: strongest penalty, then shallowest
    sel2 = R.one_se_select([c for c in cands if c[0][2] != 1000.0], block_order=(15, 31, 47))
    assert (sel2.block, sel2.rank, sel2.penalty) == (15, 16, 100.0)
    tight = R.one_se_select([((47, 128, 0.1), 0.200, 0.001), ((15, 16, 1000.0), 0.208, 0.001)], block_order=(15, 31, 47))
    assert tight.penalty == 0.1 and tight.block == 47


def test_fold_features_are_fitted_on_the_training_rows_only():
    frames = synthetic_frames(6, seed=1, neural_signal=2.0)
    n = len(frames.rows)
    train, test = np.arange(0, n - 6), np.arange(n - 6, n)
    ff = R.fit_fold_features(frames, train, blocks=frames.blocks, ranks=(4,), text_prefer="tfidf", expansions=(7,))
    assert ff.text.kind == "tfidf_svd"
    pca = ff.neural[(frames.blocks[0], 4)].pca
    same = R.fit_fold_features(frames, train, blocks=frames.blocks, ranks=(4,), text_prefer="tfidf")
    assert same.neural[(frames.blocks[0], 4)].pca.components_hash == pca.components_hash
    # a different training set gives a different map; the test rows never touched the fit
    other = R.fit_fold_features(frames, np.arange(3, n), blocks=frames.blocks, ranks=(4,), text_prefer="tfidf")
    assert other.neural[(frames.blocks[0], 4)].pca.components_hash != pca.components_hash
    Xa, names_a = ff.matrix(frames, test, variant="augmented", block=frames.blocks[0], rank=4)
    Xb, names_b = ff.matrix(frames, test, variant="baseline")
    Xe, _ = ff.matrix(frames, test, variant="text_expanded", extra=7)
    Xo, _ = ff.matrix(frames, test, variant="observables")
    assert Xa.shape[1] == Xb.shape[1] + 4 + len(R.NORM_STAT_NAMES)
    assert Xe.shape[1] == Xb.shape[1] + 7 == Xa.shape[1]  # parameter-count matched
    assert Xo.shape[1] == len(ff.obs.feature_names) < Xb.shape[1]
    assert names_a[-3:] == list(R.NORM_STAT_NAMES) and names_a[: len(names_b)] == names_b
    # the PCA rank is clipped to the training rank
    tiny = R.fit_fold_features(frames, np.arange(0, 3), blocks=frames.blocks, ranks=(8,), text_prefer="tfidf")
    assert tiny.neural[(frames.blocks[0], 8)].effective_rank == 2


def test_nested_cv_recovers_a_positive_contrast_on_separable_data():
    frames = synthetic_frames(15, seed=7, neural_signal=4.0)
    cfg = R.GConfig(blocks=frames.blocks, **SMALL)
    result = R.nested_cv(frames, cfg)
    losses = result.losses
    assert len(losses) == len(frames.rows) and losses["outer_fold"].nunique() == 5
    assert np.isfinite(losses[[f"brier_{v}" for v in R.VARIANTS]].to_numpy()).all()
    wb = result.summary["weighted_brier"]
    assert wb["augmented"] < wb["baseline"] - 0.03
    assert result.summary["contrast_baseline_minus_augmented"] > 0.03
    # the signal block was selected by every outer fold
    assert all(f["augmented_selection"]["block"] == frames.blocks[0] for f in result.selections)
    boot = R.confirmation_contrast(losses, seed=11, n_resamples=500)
    assert boot["bootstrap"]["estimate"] > 0.03 and boot["bootstrap"]["p_one_sided"] < 0.05
    assert boot["bootstrap"]["ci_low"] > 0
    assert set(boot["by_method"]) == {"IND_VOTE", "DEC", "CEN_FLAT"}


def test_nested_cv_is_near_zero_on_pure_noise():
    frames = synthetic_frames(15, seed=8, neural_signal=0.0)
    cfg = R.GConfig(blocks=frames.blocks, **SMALL)
    result = R.nested_cv(frames, cfg)
    contrast = result.summary["contrast_baseline_minus_augmented"]
    assert abs(contrast) < 0.03
    boot = R.confirmation_contrast(result.losses, seed=12, n_resamples=500)
    assert boot["bootstrap"]["p_one_sided"] > 0.05
    assert boot["bootstrap"]["ci_low"] <= 0.0 <= boot["bootstrap"]["ci_high"] + 0.01


def test_freeze_round_trip_and_hash_guard(tmp_path):
    frames = synthetic_frames(8, seed=9, neural_signal=2.5)
    cfg = R.GConfig(blocks=frames.blocks, **SMALL)
    frozen = R.freeze(frames, cfg, meta={"seal": "x"})
    json_path, npz_path = frozen.save(tmp_path)
    loaded = R.FrozenG.load(tmp_path)
    p0 = frozen.predict(frames)
    p1 = loaded.predict(frames)
    for v in R.VARIANTS:
        assert np.allclose(p0[v], p1[v])
    payload = json_path.read_text()
    assert '"sha256"' in payload and '"components_hash"' in payload and '"feature_lists"' in payload
    assert loaded.selections["augmented"].block == frames.blocks[0]
    # a modified artifact is refused
    assert '"kind": "G_FROZEN"' in payload
    json_path.write_text(payload.replace('"kind": "G_FROZEN"', '"kind": "G_FROZEN_TAMPERED"', 1))
    with pytest.raises(R.ReadoutError):
        R.FrozenG.load(tmp_path)


def test_bootstrap_p_values_are_uniform_under_the_null():
    rng = np.random.default_rng(123)
    clusters = [f"hle:{i}" for i in range(20)] + [f"bcb:{i}" for i in range(20)]
    sds = ["hle"] * 20 + ["bcb"] * 20
    ps = []
    for k in range(200):
        values = rng.normal(size=40)
        ps.append(R.paired_cluster_bootstrap(values, clusters, sds, n_resamples=199, seed=k, alternative="greater").p_one_sided)
    ps = np.asarray(ps)
    assert 0.40 < ps.mean() < 0.60
    assert 0.03 <= np.mean(ps < 0.10) <= 0.20
    assert 0.38 <= np.mean(ps < 0.50) <= 0.62
    assert ps.min() >= 1.0 / 200  # +1/+1 correction: never zero


def test_bootstrap_estimate_weights_superdomains_equally_and_flags_small_panels():
    values = np.r_[np.full(30, 0.10), np.full(10, -0.30)]
    clusters = [f"hle:{i}" for i in range(30)] + [f"bcb:{i}" for i in range(10)]
    sds = ["hle"] * 30 + ["bcb"] * 10
    res = R.paired_cluster_bootstrap(values, clusters, sds, n_resamples=100, seed=1)
    assert res.estimate == pytest.approx(0.5 * 0.10 + 0.5 * (-0.30))
    assert res.weights == {"bcb": 0.5, "hle": 0.5} and res.n_clusters == {"hle": 30, "bcb": 10}
    assert res.status == "degenerate"  # zero within-domain variance → no p-value, never p=0
    small = R.paired_cluster_bootstrap(np.random.default_rng(0).normal(size=12), [f"hle:{i}" for i in range(12)], ["hle"] * 12, n_resamples=100, seed=2)
    assert small.estimation_only and small.weights == {"hle": 1.0}
    big = R.paired_cluster_bootstrap(np.random.default_rng(0).normal(size=40), clusters, sds, n_resamples=100, seed=3)
    assert not big.estimation_only and big.status == "ok" and 0 < big.p_one_sided <= 1


def test_item_weights_and_per_item_averaging():
    w = R.item_weights(["hle", "hle", "bcb"])
    assert w.tolist() == pytest.approx([0.25, 0.25, 0.5])
    frames = synthetic_frames(2, seed=3)
    losses = frames.rows[["source_id", "method", "superdomain", "cluster"]].copy()
    losses["brier_baseline"] = np.arange(len(losses), dtype=float)
    losses["brier_augmented"] = 0.0
    items = R.per_item_losses(losses, ["brier_baseline", "brier_augmented"])
    first = losses.loc[losses["source_id"] == "hle:dev0000", "brier_baseline"].to_numpy()
    assert len(items) == 4 and items.set_index("source_id").loc["hle:dev0000", "brier_baseline"] == pytest.approx(first.mean())
    cov = R.method_coverage(losses)
    assert cov["methods"] == ["CEN_FLAT", "DEC", "IND_VOTE"] and cov["n_items_total"] == cov["n_items_complete"] == 4 and cov["n_items_dropped"] == 0


def test_per_item_losses_drop_items_missing_a_method_instead_of_reweighting():
    """P1-6 regression: an item lacking one method must not enter the paired contrast with
    fewer methods than the others (unequal weights); it is dropped and counted."""
    frames = synthetic_frames(3, seed=4)
    losses = frames.rows[["source_id", "method", "superdomain", "cluster"]].copy()
    losses["brier_baseline"] = np.where(losses["method"] == "DEC", 1.0, 0.0)
    losses["brier_augmented"] = 0.0
    partial = losses[~((losses["source_id"] == "hle:dev0000") & (losses["method"] == "DEC"))].reset_index(drop=True)
    cov = R.method_coverage(partial)
    assert cov["n_items_total"] == 6 and cov["n_items_complete"] == 5 and cov["dropped_items"] == ["hle:dev0000"]
    items = R.per_item_losses(partial, ["brier_baseline", "brier_augmented"])
    assert len(items) == 5 and "hle:dev0000" not in set(items["source_id"]) and np.allclose(items["brier_baseline"].to_numpy(), 1 / 3)
    lax = R.per_item_losses(partial, ["brier_baseline", "brier_augmented"], require_complete=False)
    assert len(lax) == 6 and lax.set_index("source_id").loc["hle:dev0000", "brier_baseline"] == 0.0  # the unequal weighting the default forbids
    out = R.confirmation_contrast(partial, seed=1, n_resamples=50)
    assert out["n_items"] == 5 and out["n_items_dropped"] == 1 and out["dropped_items"] == ["hle:dev0000"] and out["n_items_complete"] == 5
    assert out["losses"]["method_coverage"]["n_items_dropped"] == 1 and out["bootstrap"]["estimate"] == pytest.approx(1 / 3)


def test_row_from_report_reads_missing_confidence_and_invalid_forecast():
    render = {"report_id": "r", "source_id": "hle:x", "method": "DEC", "selection_id": "s", "prompt_tokens": 100, "evidence_tokens": 50,
              "report": {"item": {"domain": "hle", "split": "dev", "answer_format": "exactMatch", "N": 5, "B": 4}, "selected_candidate": None, "selected_is_sentinel": True,
                         "selected_personal_confidence": None, "personal_confidence_missing": True, "vote_metadata": {"winning_count": None, "tied_classes": None, "all_singleton": True},
                         "budget_metadata": {"B_flops": 10.0, "spent_flops": 8.0, "slack_flops": 2.0, "calls_by_role": {"root": 5, "revise": 3}, "stop_reason": "BUDGET"}, "text": "t", "task_text": "a b c"}}
    row = R.row_from_report(render, forecast={"parse_status": "NOT_JSON", "parsed": None})
    assert row["personal_missing"] and row["personal_conf"] is None and row["selected_is_sentinel"]
    assert row["calls_total"] == 8.0 and row["spent_frac"] == pytest.approx(0.8) and row["stop_reason"] == "BUDGET"
    assert row["forecast_valid"] is False and row["q_team_now"] is None and row["task_tokens"] == 3.0
    enc = R.ObservableEncoder().fit(__import__("pandas").DataFrame([row]))
    X = enc.transform(__import__("pandas").DataFrame([row]))
    assert X.shape == (1, len(enc.feature_names)) and X[0, enc.feature_names.index("personal_missing")] == 1.0
