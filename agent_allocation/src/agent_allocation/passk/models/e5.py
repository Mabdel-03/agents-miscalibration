"""E5: prospective prediction of allocation outcomes on unseen problems (dev = L96 -> T200).

Entry points: :func:`fit_and_predict` (writes ``sealed/t200/``) and :func:`score` (after the T200 release).
:func:`validate_split` runs the whole pipeline on a synthetic held-out split of development items.

Training pools (PREREG): 1x1; 2x1, 1x2; 4x1, 2x2, 1x4; 16x1, 8x2, 2x8, 1x16; primitive 1x8; C48 only 2x16, 4x8,
8x4 -- all prefixes of A slots 0-15 / deep replicates. The withheld aggregate targets (4,4), (8,1), (4,2), (2,4) are
never built into a training frame (and the guard raises if one appears).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agent_allocation.passk.features import FEATURES, LATENCY, diag_features
from agent_allocation.passk.models import glm as glm_mod
from agent_allocation.passk.models.inputs import (
    BENCHMARKS, CONFIGS_16A, CONFIGS_8A, MIXTURE, PASSK_SEED, PREDICT_CONFIGS, ClassSource, build_prefix_pools,
    fitting_draws, label_value, load_analysis_inputs, load_panels, observed_pool_table, parse_config, sha256_file,
)
from agent_allocation.passk.models.predictors import (
    PREDICTORS, P2Predictor, RidgePredictor, WITHHELD, config_rank, guard_training_frame, ridge_design,
)

VERSION = "passk-e5-v1"
TRAIN_CONFIGS = ("1x1", "2x1", "1x2", "4x1", "2x2", "1x4", "16x1", "8x2", "2x8", "1x16", "1x8")
C48_ONLY_CONFIGS = ("2x16", "4x8", "8x4")
WITHHELD_IDS = tuple(f"{n}x{a}" for n, a in sorted(WITHHELD))
DEV_PARTITIONS = ("legacy_main", "e1_fill", "c48_extra")
assert not set(TRAIN_CONFIGS + C48_ONLY_CONFIGS) & set(WITHHELD_IDS)


# ---------------------------------------------------------------------- development frames
def dev_draw_frame(draws: pd.DataFrame, items, feats: pd.DataFrame) -> pd.DataFrame:
    d = fitting_draws(draws, items, a_slots=20, partitions=DEV_PARTITIONS)
    d = d.assign(y=[label_value(v) for v in d["correct"]], a=d.allowance_a.astype(int))
    d = d[~d.y.isna()]
    d = d[["candidate_id", "item_id", "benchmark", "partition", "a", "y"]].merge(
        feats.drop(columns=["benchmark"]), on="item_id", how="inner")
    return d


def dev_pool_frame(draws: pd.DataFrame, pools: pd.DataFrame | None, cls: ClassSource, items, c48,
                   feats: pd.DataFrame) -> pd.DataFrame:
    items = list(items)
    c48 = set(c48 or [])
    dev = draws[draws.partition.isin(DEV_PARTITIONS)]
    if pools is not None:
        p = pools[pools.item_id.isin(items) & (pools.view == "current") if "view" in pools else pools.item_id.isin(
            items)]
        keep = p.config_id.isin(TRAIN_CONFIGS) | (p.config_id.isin(C48_ONLY_CONFIGS) & p.item_id.isin(c48))
        p = p[keep & ~p.panel.isin(["V48", "T200"])].drop_duplicates(["item_id", "config_id"])
    else:
        sub = dev[(dev.allowance_a > 1) | (dev.replicate < 16)]
        p = build_prefix_pools(sub, items, TRAIN_CONFIGS, "L96")
        if c48:
            p = pd.concat([p, build_prefix_pools(sub, [i for i in items if i in c48], C48_ONLY_CONFIGS, "C48")])
    part = dict(zip(dev.candidate_id, dev.partition))
    obs = observed_pool_table(p, dev, cls)
    obs["member_partitions"] = [[part.get(c, "?") for c in m] for m in p.member_ids]
    obs = obs.rename(columns={"plurality": "delivered"})
    obs["benchmark"] = obs.item_id.map(dict(zip(dev.item_id, dev.benchmark)))
    obs = obs.merge(feats.drop(columns=["benchmark"]), on="item_id", how="inner")
    obs = obs[obs.coverage.notna() | obs.delivered.notna()]
    guard_training_frame(obs)
    return obs.reset_index(drop=True)


def impute_features(feats: pd.DataFrame, items, benchmarks: dict, train_feats: pd.DataFrame,
                    default_provider: dict | None = None) -> pd.DataFrame:
    """Every item gets a feature row; items without diagnostics get development means (flagged)."""
    have = set(feats.item_id)
    means = train_feats[FEATURES].mean()
    rows = []
    for i in items:
        if i not in have:
            rows.append({"item_id": i, "benchmark": benchmarks.get(i), "provider": (default_provider or {}).get(
                i, "mancer"), "collection": "imputed", "n_diag": 0, **means.to_dict()})
    out = pd.concat([feats, pd.DataFrame(rows)], ignore_index=True) if rows else feats.copy()
    out["imputed"] = ~out.item_id.isin(have)
    return out


def predict_frame(feats: pd.DataFrame, configs=PREDICT_CONFIGS) -> pd.DataFrame:
    rows = []
    for r in feats.to_dict("records"):
        for cid in configs:
            n, a = parse_config(cid)
            rows.append({**r, "config_id": cid, "n": n, "a": a})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- fitting
def choose_latency(pool_frame: pd.DataFrame, seed: int) -> dict:
    """Keep latency only if grouped CV of P1 (delivered) improves with it."""
    out = {}
    for use in (False, True):
        X, _ = ridge_design(pool_frame, "P1", use)
        k = pool_frame.delivered.notna().to_numpy()
        best, table = glm_mod.cv_ridge(X[k], pool_frame.delivered.to_numpy(float)[k],
                                       pool_frame.item_id.to_numpy()[k], item_intercept=False,
                                       ridges=(1.0, 10.0, 100.0), seed=seed)
        out[use] = min(r["log_loss"] for r in table)
    return {"use_latency": bool(out[True] < out[False]), "cv_log_loss_without": out[False],
            "cv_log_loss_with": out[True]}


def fit_models(draw_frame: pd.DataFrame, pool_frame: pd.DataFrame, *, seed: int = PASSK_SEED,
               n_sim: int = 2000) -> dict:
    assert config_rank(pool_frame) == 3, "rank([1, log n, log a]) must be 3 on every training frame"
    guard_training_frame(pool_frame)
    lat = choose_latency(pool_frame, seed)
    models: dict = {"latency": lat}
    for name in PREDICTORS:
        if name.startswith("P2"):
            models[name] = P2Predictor(predictor=name, use_latency=lat["use_latency"], n_sim=n_sim).fit(
                draw_frame, pool_frame, seed=seed)
        else:
            models[name] = RidgePredictor(predictor=name, use_latency=lat["use_latency"]).fit(pool_frame, seed=seed)
    return models


def predict_all(models: dict, frame: pd.DataFrame, seed: int = PASSK_SEED) -> pd.DataFrame:
    out = []
    for name in PREDICTORS:
        m = models[name]
        p = m.predict(frame, seed=seed) if isinstance(m, P2Predictor) else m.predict(frame)
        out.append(pd.DataFrame({"item_id": frame.item_id.to_numpy(), "benchmark": frame.benchmark.to_numpy(),
                                 "config_id": frame.config_id.to_numpy(), "predictor": name,
                                 "p_delivered": p.p_delivered.to_numpy(), "p_coverage": p.p_coverage.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def fit_and_predict(run_root: Path | str | None = None, *, draws: pd.DataFrame | None = None,
                    pools: pd.DataFrame | None = None, classes: pd.DataFrame | None = None,
                    panels: dict | None = None, out_dir: Path | str | None = None, seed: int = PASSK_SEED,
                    n_sim: int = 2000, write: bool = True, overwrite: bool = False) -> dict:
    if draws is None:
        draws, pools, classes = load_analysis_inputs(Path(run_root))
    if panels is None:
        panels = load_panels(Path(run_root))
    out_dir = Path(out_dir) if out_dir is not None else (Path(run_root) / "sealed" / "t200" if write else None)
    if write and (out_dir / "predictions.parquet").exists() and not overwrite:
        raise FileExistsError(f"{out_dir}/predictions.parquet exists (sealed artifacts are write-once)")
    for col in ("correct", "grade_state"):
        if col in draws.columns and draws.loc[draws.partition.isin(["t200_diag", "t200_bank", "v48_bank"]),
                                              col].notna().any():
            raise RuntimeError(f"held-out labels present in draws ({col}); E5 fits only before the release")
    cls = ClassSource(draws=draws, table=classes)
    l96, t200, c48 = list(panels["L96"]), list(panels["T200"]), list(panels.get("C48", []))
    feats_all = diag_features(draws, cls, pools=pools)
    dev_feats = feats_all[feats_all.item_id.isin(l96)]
    res = _fit_predict_core(draws, pools, cls, l96, c48, t200, dev_feats, feats_all, seed=seed, n_sim=n_sim)
    if write:
        _write(out_dir, res, seed=seed, n_sim=n_sim, cls=cls)
    return res


def _fit_predict_core(draws, pools, cls, train_items, c48, test_items, dev_feats, feats_all, *, seed, n_sim,
                      bench_map=None) -> dict:
    draw_frame = dev_draw_frame(draws, train_items, dev_feats)
    pool_frame = dev_pool_frame(draws, pools, cls, train_items, c48, dev_feats)
    models = fit_models(draw_frame, pool_frame, seed=seed, n_sim=n_sim)
    bench_map = bench_map or dict(zip(draws.item_id, draws.benchmark))
    test_feats = impute_features(feats_all[feats_all.item_id.isin(test_items)], test_items, bench_map, dev_feats)
    frame = predict_frame(test_feats)
    pred = predict_all(models, frame, seed=seed)
    return {"predictions": pred, "features": test_feats, "models": models, "draw_frame": draw_frame,
            "pool_frame": pool_frame}


def _write(out_dir: Path, res: dict, *, seed: int, n_sim: int, cls: ClassSource) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    res["predictions"].to_parquet(out_dir / "predictions.parquet", index=False)
    res["features"].to_parquet(out_dir / "features.parquet", index=False)
    for name, m in res["models"].items():
        payload = m if isinstance(m, dict) else m.to_dict()
        (out_dir / "models" / f"{name}.json").write_text(json.dumps(payload, indent=1, default=float))
    pf = res["pool_frame"]
    (out_dir / "seeds.json").write_text(json.dumps({
        "passk_seed": seed, "n_sim": n_sim, "cv_folds": 5, "train_configs": list(TRAIN_CONFIGS),
        "c48_only_configs": list(C48_ONLY_CONFIGS), "withheld": list(WITHHELD_IDS)}, indent=1))
    manifest = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "classes_source": cls.source, "n_train_pools": int(len(pf)),
                "train_configs_present": sorted(set(pf.config_id)), "n_train_draws": int(len(res["draw_frame"])),
                "train_partitions": sorted(set(res["draw_frame"].partition)),
                "n_items_predicted": int(res["predictions"].item_id.nunique()),
                "n_items_imputed": int(res["features"].imputed.sum())}
    manifest["sha256"] = {str(p.relative_to(out_dir)): sha256_file(p) for p in sorted(out_dir.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------------- evaluation
def observed_t200(draws: pd.DataFrame, pools: pd.DataFrame | None, cls: ClassSource, t200) -> pd.DataFrame:
    td = draws[draws.partition == "t200_bank"]
    if pools is not None:
        tp = pools[(pools.panel == "T200") & pools.config_id.isin(PREDICT_CONFIGS)]
    else:
        tp = build_prefix_pools(td, t200, PREDICT_CONFIGS, "T200")
    return observed_pool_table(tp, td, cls).rename(columns={"plurality": "delivered"})


def _weighted(values: pd.Series, bench: pd.Series, weights: dict) -> float:
    parts = {b: values[bench == b].mean() for b in BENCHMARKS if (bench == b).any()}
    w = {b: weights[b] for b in parts}
    tot = sum(w.values())
    return float(sum(parts[b] * w[b] for b in parts) / tot) if tot else float("nan")


def _calibration(y, p, bins=10) -> tuple[pd.DataFrame, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "mean_obs": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return pd.DataFrame(rows), float(ece)


def score_frames(pred: pd.DataFrame, obs: pd.DataFrame, *, min_known: int = 3) -> dict:
    m = pred.merge(obs[["item_id", "config_id", "coverage", "delivered"]], on=["item_id", "config_id"], how="inner")
    m["sq_del"] = (m.p_delivered - m.delivered) ** 2
    m["sq_cov"] = (m.p_coverage - m.coverage) ** 2
    # primary: item-level mean Brier over known 16A configurations (>= min_known)
    k16 = m[m.config_id.isin(CONFIGS_16A) & m.delivered.notna()]
    item = k16.groupby(["predictor", "item_id", "benchmark"]).agg(brier=("sq_del", "mean"),
                                                                 n_known=("sq_del", "size")).reset_index()
    dropped = item[item.n_known < min_known]
    item = item[item.n_known >= min_known]
    rows = []
    for name, g in m.groupby("predictor"):
        gi = item[item.predictor == name]
        kd, kc = g[g.delivered.notna()], g[g.coverage.notna()]
        k16g = kd[kd.config_id.isin(CONFIGS_16A)]
        wh = kd[kd.config_id.isin(WITHHELD_IDS)]
        p = np.clip(k16g.p_delivered.to_numpy(float), 1e-6, 1 - 1e-6)
        yv = k16g.delivered.to_numpy(float)
        cal, ece = _calibration(yv, p) if len(yv) else (pd.DataFrame(), float("nan"))
        rows.append({
            "predictor": name,
            "primary_brier_mixture": _weighted(gi.brier, gi.benchmark, MIXTURE),
            "primary_brier_macro": _weighted(gi.brier, gi.benchmark, {b: 1.0 for b in BENCHMARKS}),
            "n_items_primary": int(len(gi)),
            "coverage_brier_16A": float(kc[kc.config_id.isin(CONFIGS_16A)].sq_cov.mean()),
            "log_loss_16A": float(-np.mean(yv * np.log(p) + (1 - yv) * np.log1p(-p))) if len(yv) else np.nan,
            "mae_16A": float(np.mean(np.abs(p - yv))) if len(yv) else np.nan,
            "ece_16A": ece,
            "ranking_accuracy_16A": _ranking_accuracy(k16g),
            "brier_withheld_4x4": float(wh[wh.config_id == "4x4"].sq_del.mean()),
            "brier_withheld_8A": float(wh[wh.config_id.isin(["8x1", "4x2", "2x4"])].sq_del.mean()),
            "brier_all_8A": float(kd[kd.config_id.isin(CONFIGS_8A)].sq_del.mean()),
        })
    summary = pd.DataFrame(rows)
    s = summary.set_index("predictor")
    contrasts = {}
    for a_, b_ in (("P2", "P1"), ("P1-conc", "P1"), ("P2-conc", "P2"), ("P1", "P0"), ("P2", "P0"), ("P3", "P1")):
        if a_ in s.index and b_ in s.index:
            ia = item[item.predictor == a_].set_index("item_id")
            ib = item[item.predictor == b_].set_index("item_id")
            j = ia.join(ib, lsuffix="_a", rsuffix="_b", how="inner")
            contrasts[f"{a_}-{b_}"] = _weighted(j.brier_a - j.brier_b, j.benchmark_a, MIXTURE)
    resid = m[m.delivered.notna()].assign(resid=lambda f: f.delivered - f.p_delivered).groupby(
        ["predictor", "benchmark"]).resid.agg(["mean", "size"]).reset_index()
    policy = policy_readout(m)
    return {"summary": summary, "contrasts": pd.DataFrame([{"contrast": k, "mixture_diff": v}
                                                            for k, v in contrasts.items()]),
            "item_brier": item, "dropped_items": dropped, "residuals": resid, "policy": policy, "merged": m}


def _ranking_accuracy(g: pd.DataFrame) -> float:
    hits, tot = 0.0, 0
    for _, h in g.groupby("item_id"):
        recs = list(zip(h.delivered, h.p_delivered))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                (oi, pi_), (oj, pj) = recs[i], recs[j]
                if oi == oj:
                    continue
                tot += 1
                hits += 0.5 if pi_ == pj else float((oi > oj) == (pi_ > pj))
    return hits / tot if tot else float("nan")


def policy_readout(m: pd.DataFrame) -> pd.DataFrame:
    """21A readout: spend the 5 diagnostic draws (5A), then pick the 16A configuration with the highest predicted
    delivered probability per item. Reported against every fixed 16A configuration (never against the realized
    per-item ceiling)."""
    k = m[m.config_id.isin(CONFIGS_16A) & m.delivered.notna()]
    rows = []
    for name, g in k.groupby("predictor"):
        pick = g.loc[g.groupby("item_id").p_delivered.idxmax()]
        rows.append({"predictor": name, "policy": "select_per_item", "budget_A": 21,
                     "accuracy_mixture": _weighted(pick.delivered, pick.benchmark, MIXTURE),
                     "accuracy_macro": _weighted(pick.delivered, pick.benchmark, {b: 1.0 for b in BENCHMARKS}),
                     "n_items": int(len(pick))})
    one = k[k.predictor == k.predictor.iloc[0]] if len(k) else k
    for cid, g in one.groupby("config_id"):
        rows.append({"predictor": None, "policy": f"fixed_{cid}", "budget_A": 16,
                     "accuracy_mixture": _weighted(g.delivered, g.benchmark, MIXTURE),
                     "accuracy_macro": _weighted(g.delivered, g.benchmark, {b: 1.0 for b in BENCHMARKS}),
                     "n_items": int(len(g))})
    return pd.DataFrame(rows)


def score(run_root: Path | str | None = None, *, draws=None, pools=None, classes=None, t200=None,
          pred_dir: Path | str | None = None, out_dir: Path | str | None = None, observed=None,
          write: bool = True) -> dict:
    pred_dir = Path(pred_dir) if pred_dir is not None else Path(run_root) / "sealed" / "t200"
    pred = pd.read_parquet(pred_dir / "predictions.parquet")
    if observed is None:
        if draws is None:
            draws, pools, classes = load_analysis_inputs(Path(run_root))
        if t200 is None:
            t200 = list(load_panels(Path(run_root))["T200"])
        observed = observed_t200(draws, pools, ClassSource(draws=draws, table=classes), t200)
    res = score_frames(pred, observed)
    if write:
        out = Path(out_dir) if out_dir is not None else Path(run_root) / "scores" / "e5"
        out.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            if isinstance(v, pd.DataFrame) and k != "merged":
                v.to_parquet(out / f"{k}.parquet", index=False)
    return res


# ---------------------------------------------------------------------- development validation
def holdout_split(items_bench: dict[str, str], n_test_per_bench: int = 6, salt: str = "wpm-e5-validate"
                  ) -> tuple[list[str], list[str]]:
    train, test = [], []
    for b in BENCHMARKS:
        its = sorted([i for i, x in items_bench.items() if x == b],
                     key=lambda i: hashlib.sha256(f"{salt}|{i}".encode()).hexdigest())
        test += its[:n_test_per_bench]
        train += its[n_test_per_bench:]
    return train, test


def validate_split(draws: pd.DataFrame, items: list[str], *, classes=None, pools=None, c48=None,
                   n_test_per_bench: int = 6, seed: int = PASSK_SEED, n_sim: int = 1000, min_known: int = 3) -> dict:
    """Fit on the development items minus a hashed held-out subset, predict the subset's diag features (legacy slots
    20-24), and score against the subset's own development pools (prefix pools; whatever configurations exist)."""
    cls = ClassSource(draws=draws, table=classes)
    bench = dict(zip(draws.item_id, draws.benchmark))
    train, test = holdout_split({i: bench[i] for i in items}, n_test_per_bench)
    feats_all = diag_features(draws, cls, pools=pools)
    dev_feats = feats_all[feats_all.item_id.isin(train)]
    res = _fit_predict_core(draws, pools, cls, train, [i for i in (c48 or []) if i in train], test, dev_feats,
                            feats_all, seed=seed, n_sim=n_sim)
    dev = draws[draws.partition.isin(DEV_PARTITIONS)]
    sub = dev[(dev.allowance_a > 1) | (dev.replicate < 16)]
    obs = observed_pool_table(build_prefix_pools(sub, test, PREDICT_CONFIGS, "holdout"), sub, cls).rename(
        columns={"plurality": "delivered"})
    sc = score_frames(res["predictions"], obs, min_known=min_known)
    return {**res, "score": sc, "observed": obs, "train": train, "test": test}
