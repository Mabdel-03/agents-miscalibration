"""Selector comparison under the LEX objective: statistical predictors vs the LLM allocator vs simple rules.

    python -m agent_allocation.passk.selcmp --study-root R [--n-sim 2000] [--overwrite]

Post-hoc and exploratory (every label is released). Owner objective (``docs/passk_cont/PREREG_LEX_front.md``):
(1) accuracy, (2) consistency p^2 + (1 - p)^2, (3) latency = parallel wall-clock of the chosen pool PLUS the
selector's own overhead; accuracy ties within delta = 0.02, then consistency ties within 0.02, then fastest.

Overheads (added to the scored latency of every item):

* P0 and fixed options: none;
* P1, P2, P3, P1-conc, P2-conc and the diagnostic-agreement heuristic: the five diagnostic calls' parallel
  wall-clock (max of their five latencies; L96 = legacy A slots 20-24, T200 = ``t200_diag`` r0-4);
* LLM A0: its decision call latency (``result.latency_seconds`` of every attempt in ``e8a/raw``, summed over retries);
* LLM A1: the diagnostics' wall-clock plus its decision latency.

Menus: M15 = every n x a with n*a <= 16, n, a in {1, 2, 4, 8, 16} (15 options, present on the C48 banks and the T200
pools) and M5 = the LLM allocator's 1x16, 2x8, 4x4, 8x2, 16x1.

Statistical predictions are cross-fitted: the E5 code paths (``models.e5`` frames, ``fit_models`` with its latency
choice and grouped-CV ridge selection, withheld-target guard unchanged) are refit five times on L96 minus one outer
fold and predict that fold's items for every M15 option. The outer folds are ``glm.grouped_folds(L96, 5, 20260925)``,
i.e. exactly the problem-grouped fold assignment the sealed E5 fit used for its own ridge CV on L96. A final fit on
all of L96 predicts T200 for M15 (post-hoc; the sealed 9-option predictions stay the confirmatory ones).

Scoring: on C48 with the per-item bank estimates of ``lex/front_estimates.parquet`` (expected p, C, latency); on T200
with realized delivered correctness (readouts ``S_known``, passk mode) and realized wall-clock (slowest member of the
chosen T200 pool), where consistency is not measurable per item. Outputs go only to ``R/lex/selcmp/``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation.passk.lexfront import DELTA, lex_choose, parse

M15 = ("1x1", "2x1", "4x1", "8x1", "16x1", "1x2", "2x2", "4x2", "8x2", "1x4", "2x4", "4x4", "1x8", "2x8", "1x16")
M5 = ("1x16", "2x8", "4x4", "8x2", "16x1")
M9 = ("16x1", "8x2", "4x4", "2x8", "1x16", "8x1", "4x2", "2x4", "1x8")     # the sealed E5 menu
MENUS = {"M15": M15, "M5": M5}
STAT = ("P0", "P1", "P2", "P3", "P1-conc", "P2-conc")
FOLDS = 5
TAUS = (0.6, 0.8, 1.0)
DEFAULT = "1x16"
REFERENCE = "fixed:1x16"
OUT_PARTS = ("lex", "selcmp")
EPS = 1e-12            # float slack on the delta comparisons (0.80 - 0.02 must tie 0.78)
CI_TOL = 1e-9          # a percentile bound this close to 0 touches zero (0/1 T200 outcomes give bounds like 3e-18)
assert len(set(M15)) == 15 and all(n * a <= 16 for n, a in map(parse, M15)) and set(M5) <= set(M15)


class CrossFitLeakError(RuntimeError):
    """An item's prediction would come from a model that saw that item's draws or pool outcomes."""


# ------------------------------------------------------------------------------------------------------------------
# lexicographic helpers
# ------------------------------------------------------------------------------------------------------------------

def lex_order(table: pd.DataFrame, *, acc: str, lat: str, cons: str | None = None, delta: float = DELTA,
              key: str = "selector") -> list[str]:
    """Population ranking by repeated lexicographic selection.

    Rank 1 is the lexicographic winner of all rows (accuracy within ``delta`` of the best -> consistency within
    ``delta`` of the best among those, when ``cons`` is given -> lowest latency -> ``key``); rank 2 is the winner of
    the remaining rows, and so on. Rows with a missing accuracy or latency (or consistency, when used) are left out.
    """
    cols = [acc, lat] + ([cons] if cons else [])
    remaining = table.dropna(subset=cols)
    order: list[str] = []
    while len(remaining):
        s = remaining[remaining[acc] >= remaining[acc].max() - delta - EPS]
        if cons:
            s = s[s[cons] >= s[cons].max() - delta - EPS]
        winner = s.sort_values([lat, key], kind="mergesort").iloc[0][key]
        order.append(winner)
        remaining = remaining[remaining[key] != winner]
    return order


def ci_side(ci: Sequence[float], tol: float = CI_TOL) -> int:
    """+1 if the 95% interval lies strictly above zero, -1 if strictly below, 0 if it includes or touches zero
    (a bound within ``tol`` of 0 counts as touching: it is a discrete bootstrap landing on no difference)."""
    lo, hi = float(ci[0]), float(ci[1])
    if lo > tol:
        return 1
    if hi < -tol:
        return -1
    return 0


def macro_mean(values: Sequence[float], benchmarks: Sequence[str]) -> float:
    """Equal-weight mean over benchmarks of the per-benchmark means (missing values skipped)."""
    s = pd.Series(np.asarray(values, float)).groupby(np.asarray(benchmarks)).mean().dropna()
    return float(s.mean()) if len(s) else float("nan")


# ------------------------------------------------------------------------------------------------------------------
# cross-fitting (E5 code paths)
# ------------------------------------------------------------------------------------------------------------------

def outer_folds(items: Iterable[str], seed: int | None = None) -> dict[str, int]:
    """item -> outer fold: the E5 problem-grouped fold function over the L96 items (whole problems per fold)."""
    from agent_allocation.passk.models import glm
    from agent_allocation.passk.models.inputs import PASSK_SEED
    items = sorted(set(items))
    fid = glm.grouped_folds(items, FOLDS, PASSK_SEED if seed is None else seed)
    return {i: int(f) for i, f in zip(items, fid)}


def check_training_frames(test_items: Iterable[str], **frames: pd.DataFrame) -> None:
    """Raise if any training frame carries a row of a held-out item."""
    test = set(test_items)
    for name, frame in frames.items():
        leak = test & set(frame["item_id"])
        if leak:
            raise CrossFitLeakError(f"{name} contains {len(leak)} held-out item(s), e.g. {sorted(leak)[:3]}")


def assert_crossfit(pred: pd.DataFrame, train_items_by_fold: Mapping[Any, Iterable[str]]) -> None:
    """Every prediction row names the fold whose model made it; that model's training items exclude the row's item."""
    if "fold" not in pred.columns:
        raise CrossFitLeakError("predictions carry no fold column")
    for fold, g in pred.groupby("fold"):
        if fold not in train_items_by_fold:
            raise CrossFitLeakError(f"fold {fold!r} has no recorded training set")
        leak = set(g["item_id"]) & set(train_items_by_fold[fold])
        if leak:
            raise CrossFitLeakError(f"fold {fold!r} predicts {len(leak)} of its own training items, e.g. "
                                    f"{sorted(leak)[:3]}")


def fit_predict(draws: pd.DataFrame, pools: pd.DataFrame | None, cls, feats_all: pd.DataFrame,
                train_items: Sequence[str], test_items: Sequence[str], c48: Iterable[str], configs: Sequence[str], *,
                seed: int, n_sim: int, default_provider: Mapping[str, str] | None = None
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One E5 fit on ``train_items`` (``_fit_predict_core`` with a configurable menu) -> predictions for
    ``test_items`` x ``configs``. Raises :class:`CrossFitLeakError` when the train and test sets touch."""
    from agent_allocation.passk.models import e5
    train_items, test_items = sorted(set(train_items)), sorted(set(test_items))
    if set(train_items) & set(test_items):
        raise CrossFitLeakError("train and test item sets overlap")
    dev_feats = feats_all[feats_all.item_id.isin(train_items)]
    c48_train = [i for i in sorted(set(c48)) if i in set(train_items)]
    draw_frame = e5.dev_draw_frame(draws, train_items, dev_feats)
    pool_frame = e5.dev_pool_frame(draws, pools, cls, train_items, c48_train, dev_feats)
    check_training_frames(test_items, draw_frame=draw_frame, pool_frame=pool_frame, dev_feats=dev_feats)
    models = e5.fit_models(draw_frame, pool_frame, seed=seed, n_sim=n_sim)
    bench_map = dict(zip(draws.item_id, draws.benchmark))
    test_feats = e5.impute_features(feats_all[feats_all.item_id.isin(test_items)], test_items, bench_map, dev_feats,
                                    default_provider)
    frame = e5.predict_frame(test_feats, configs)
    pred = e5.predict_all(models, frame, seed=seed)
    pred = pred.assign(n=pred.config_id.map(lambda c: parse(c)[0]), a=pred.config_id.map(lambda c: parse(c)[1]),
                       imputed=pred.item_id.map(dict(zip(test_feats.item_id, test_feats.imputed))))
    ridges = {}
    for name in STAT:
        m = models[name]
        ridges[name] = ({"response": m.cv["response"]["best_ridge"], "eta": m.cv["eta"]["best_ridge"]}
                        if name.startswith("P2") else {t: m.cv[t]["best_ridge"] for t in ("delivered", "coverage")})
    info = {"n_train_items": len(train_items), "n_test_items": len(test_items), "n_train_pools": int(len(pool_frame)),
            "n_train_draws": int(len(draw_frame)), "train_configs_present": sorted(set(pool_frame.config_id)),
            "latency_choice": models["latency"], "best_ridges": ridges}
    return pred, info


def crossfit(draws: pd.DataFrame, pools: pd.DataFrame | None, cls, feats_all: pd.DataFrame, items: Sequence[str],
             c48: Iterable[str], configs: Sequence[str], *, seed: int, n_sim: int,
             folds: Mapping[str, int] | None = None, log: Callable[[str], None] = print
             ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Out-of-fold predictions for every item of ``items`` (each from the fold model that never saw it)."""
    folds = dict(folds) if folds is not None else outer_folds(items)
    preds, infos, train_sets = [], {}, {}
    for f in sorted(set(folds.values())):
        test = sorted(i for i in items if folds[i] == f)
        train = sorted(i for i in items if folds[i] != f)
        t0 = time.time()
        pred, info = fit_predict(draws, pools, cls, feats_all, train, test, c48, configs, seed=seed, n_sim=n_sim)
        preds.append(pred.assign(fold=f))
        infos[f] = {**info, "test_items": test, "seconds": round(time.time() - t0, 1)}
        train_sets[f] = train
        log(f"[selcmp] fold {f}: {len(train)} train / {len(test)} test items, {info['n_train_pools']} pools, "
            f"latency={info['latency_choice']['use_latency']} ({time.time() - t0:.0f}s)")
    oof = pd.concat(preds, ignore_index=True)
    assert_crossfit(oof, train_sets)
    per = oof.groupby(["item_id", "config_id", "predictor"]).size()
    if (per != 1).any() or set(oof.item_id) != set(items):
        raise CrossFitLeakError("out-of-fold predictions do not cover every item exactly once per cell")
    return oof, {"folds": infos, "train_sets": train_sets, "assignment": folds}


# ------------------------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------------------------

def diag_wallclock(draws: pd.DataFrame) -> dict[str, float]:
    """item -> parallel wall-clock of its five diagnostic calls (max of their latencies)."""
    from agent_allocation.passk.features import diag_draws
    d = diag_draws(draws)
    sizes = d.groupby("item_id").size()
    if (sizes != 5).any():
        raise RuntimeError(f"items without exactly five diagnostic draws: {sizes[sizes != 5].to_dict()}")
    return d.groupby("item_id").latency_s.max().astype(float).to_dict()


def llm_decision_latency(study_root: Path) -> pd.DataFrame:
    """(arm, panel, item_id) -> decision latency summed over every attempt (parse retries and transport retries)."""
    rows = []
    for line in (study_root / "e8a" / "calls.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = json.loads(gzip.decompress((study_root / row["raw"]).read_bytes()))
        lat = (payload.get("result") or {}).get("latency_seconds")
        rows.append({"arm": row["arm"], "panel": row["panel"], "item_id": row["item_id"],
                     "latency": float(lat) if lat is not None else np.nan, "final": bool(row.get("final"))})
    calls = pd.DataFrame(rows)
    out = calls.groupby(["arm", "panel", "item_id"]).agg(
        decision_latency=("latency", lambda s: float(np.nansum(s))), attempts=("latency", "size"),
        attempts_missing_latency=("latency", lambda s: int(s.isna().sum())),
        decided=("final", "any")).reset_index()
    return out


def realized_tables(study_root: Path, draws: pd.DataFrame, panel: str) -> tuple[dict, dict]:
    """(item, config) -> realized delivered correctness (S_known, passk) and realized wall-clock (slowest member)."""
    ro = pd.read_parquet(study_root / "analysis_inputs" / "readouts.parquet")
    ro = ro[(ro["mode"] == "passk") & (ro.panel == panel)]
    pools = pd.read_parquet(study_root / "analysis_inputs" / "pools.parquet")
    pools = pools[(pools.panel == panel) & (pools.view == "current")]
    lat_of = dict(zip(draws.candidate_id, draws.latency_s.astype(float)))
    acc = {(r.item_id, r.config_id): (float(r.S_known) if pd.notna(r.S_known) else np.nan)
           for r in ro.itertuples(index=False)}
    lat = {}
    for r in pools.itertuples(index=False):
        vals = [lat_of.get(m, np.nan) for m in r.member_ids]
        lat[(r.item_id, r.config_id)] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
    return acc, lat


# ------------------------------------------------------------------------------------------------------------------
# selectors
# ------------------------------------------------------------------------------------------------------------------

def latency_lookup(est: pd.DataFrame, shard_of: Mapping[str, str], *, leave_item_out: bool
                   ) -> Callable[[str, str], float]:
    """Predicted latency of (item, config): the C48 bank-latency mean per (shard, config), leaving the item out."""
    e = est.dropna(subset=["latency"]).assign(shard=est.item_id.map(shard_of))
    agg = e.groupby(["shard", "config"]).latency.agg(["sum", "count"])
    own = dict(zip(zip(e.item_id, e.config), e.latency))
    tot = {k: (float(v["sum"]), int(v["count"])) for k, v in agg.iterrows()}

    def lookup(item: str, config: str) -> float:
        s, c = tot.get((shard_of.get(item), config), (np.nan, 0))
        if leave_item_out and (item, config) in own:
            s, c = s - own[(item, config)], c - 1
        return s / c if c > 0 else np.nan
    return lookup


def stat_choices(pred: pd.DataFrame, menu: Sequence[str], lat_pred: Callable[[str, str], float]) -> pd.Series:
    """The LEX rule on predicted delivered accuracy (C from the prediction, predicted latency) -> item -> config."""
    g = pred[pred.config_id.isin(menu)]
    p = g.p_delivered.to_numpy(float)
    frame = pd.DataFrame({"item_id": g.item_id.to_numpy(), "config": g.config_id.to_numpy(), "p": p,
                          "C": p * p + (1 - p) * (1 - p),
                          "latency": [lat_pred(i, c) for i, c in zip(g.item_id, g.config_id)],
                          "n": g.config_id.map(lambda c: parse(c)[0]).to_numpy(),
                          "a": g.config_id.map(lambda c: parse(c)[1]).to_numpy()})
    if frame.latency.isna().any():
        raise RuntimeError("missing predicted latency for some (item, option)")
    chosen = lex_choose(frame)
    return chosen.set_index("item_id")["config"]


def heuristic_fit(train_items: Sequence[str], f1: Mapping[str, float], bench: Mapping[str, str],
                  outcome: Callable[[str, str], tuple[float, float, float]], overhead: Mapping[str, float],
                  menu: Sequence[str], delta: float = DELTA) -> dict[str, Any]:
    """Choose (tau, X) for "plurality share of the five diagnostic answers >= tau -> X, else 1x16" by the LEX
    objective over the training items: macro accuracy (all items) -> macro consistency (items where it is measured,
    i.e. bank estimates) -> macro latency including the diagnostic overhead -> fewer tokens (n*a of X) -> larger tau."""
    items = list(train_items)
    b = [bench[i] for i in items]
    rows = []
    for tau in TAUS:
        for x in menu:
            vals = [outcome(i, x if f1[i] >= tau - 1e-9 else DEFAULT) for i in items]
            acc = [v[0] for v in vals]
            cons = [v[1] for v in vals]
            lat = [v[2] + overhead[i] for v, i in zip(vals, items)]
            n, a = parse(x)
            rows.append({"rule": f"tau={tau:g}|X={x}", "tau": tau, "X": x, "acc": macro_mean(acc, b),
                         "C": macro_mean(cons, b), "lat": macro_mean(lat, b), "tokens": n * a,
                         "share_X": float(np.mean([f1[i] >= tau - 1e-9 for i in items]))})
    table = pd.DataFrame(rows)
    s = table[table.acc >= table.acc.max() - delta - EPS]
    if s.C.notna().any():
        s = s[s.C >= s.C.max() - delta - EPS]
    s = s.assign(neg_tau=-s.tau).sort_values(["lat", "tokens", "neg_tau"], kind="mergesort")
    best = s.iloc[0]
    return {"tau": float(best.tau), "X": str(best.X), "train_acc": float(best.acc), "train_C": float(best.C),
            "train_lat": float(best.lat), "n_train_items": len(items), "table": table}


def heuristic_apply(rule: Mapping[str, Any], items: Iterable[str], f1: Mapping[str, float]) -> pd.Series:
    return pd.Series({i: rule["X"] if f1[i] >= rule["tau"] - 1e-9 else DEFAULT for i in items})


# ------------------------------------------------------------------------------------------------------------------
# scoring and statistics
# ------------------------------------------------------------------------------------------------------------------

def _fmt_lvl(r: Mapping[str, Any], scale: float = 1.0, digits: int = 3) -> str:
    lo, hi = r["ci95"]
    return f"{r['mean'] * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def _z(x: float) -> float:
    """Float noise at zero (e.g. -3.5e-18 from a discrete bootstrap) prints as +0.0, not -0.0."""
    return 0.0 if abs(x) <= CI_TOL else x


def _fmt_diff(r: Mapping[str, Any] | None, scale: float = 1.0, digits: int = 1) -> str:
    if not r:
        return "--"
    lo, hi = r["ci95"]
    return (f"{_z(r['estimate']) * scale:+.{digits}f} [{_z(lo) * scale:+.{digits}f}, "
            f"{_z(hi) * scale:+.{digits}f}]")


class Stats:
    """Thin wrappers over ``passk.stats`` (benchmark-stratified item bootstrap, 10,000 replicates, seed 20260925)."""

    def __init__(self, n_resamples: int, seed: int):
        self.n, self.seed = n_resamples, seed

    def level(self, frame: pd.DataFrame, col: str, weights) -> dict[str, Any]:
        from agent_allocation.passk.stats import level
        r = level(frame, col, weights=weights, n_resamples=self.n, seed=self.seed)
        return {"mean": float(r["mean"]), "ci95": [float(x) for x in r["ci95_percentile"]],
                "n_items": int(sum(r["n"].values()))}

    def diff(self, frame: pd.DataFrame, a: str, b: str, weights) -> dict[str, Any]:
        from agent_allocation.passk.stats import contrast
        r = contrast(frame, a, b, weights=weights, n_resamples=self.n, seed=self.seed, n_family=1)
        return {"estimate": float(r["estimate"]), "ci95": [float(x) for x in r["ci95_percentile"]],
                "p": float(r["p"]), "n_items": int(r["n_items"])}


def winner_share(wide: Mapping[str, pd.DataFrame], names: Sequence[str], weights: Mapping[str, float], *,
                 cons: bool, n_resamples: int, seed: int, delta: float = DELTA) -> dict[str, float]:
    """Share of bootstrap replicates (same resampler as ``passk.stats``) in which each selector is the population
    LEX winner among ``names``. ``wide``: metric -> frame (item_id, benchmark, <selector columns>)."""
    from agent_allocation.passk.stats import DOMAINS, resampler
    base = wide["acc"].sort_values("item_id")
    rs = resampler(base, n_resamples=n_resamples, seed=seed)
    tot = sum(weights[d] for d in DOMAINS if (base.benchmark == d).any())
    metrics = ["acc", "lat"] + (["C"] if cons else [])
    means = {m: np.zeros((n_resamples, len(names))) for m in metrics}
    for d in DOMAINS:
        if not (base.benchmark == d).any():
            continue
        for m in metrics:
            sub = wide[m][wide[m].benchmark == d].sort_values("item_id")
            vals = sub[list(names)].to_numpy(float)                          # (n_d, S)
            for lo in range(0, n_resamples, 1000):
                idx = rs.idx[d][lo:lo + 1000]
                with np.errstate(invalid="ignore"):
                    means[m][lo:lo + 1000] += weights[d] / tot * np.nanmean(vals[idx], axis=1)
    acc = means["acc"]
    tie = acc >= acc.max(axis=1, keepdims=True) - delta - EPS
    if cons:
        c = np.where(tie, means["C"], -np.inf)
        tie &= means["C"] >= c.max(axis=1, keepdims=True) - delta - EPS
    lat = np.where(tie, means["lat"], np.inf)
    win = lat.argmin(axis=1)
    counts = np.bincount(win, minlength=len(names))
    return {n: float(counts[j] / n_resamples) for j, n in enumerate(names)}


# ------------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------------

def _selector_frame(items: pd.DataFrame, choice: pd.Series, *, value: Callable[[str, str], tuple], overhead=None,
                    name: str, kind: str, menu: str, overhead_kind: str) -> pd.DataFrame:
    f = items[["item_id", "benchmark"]].copy()
    f["choice"] = f.item_id.map(choice)
    if f.choice.isna().any():
        raise RuntimeError(f"{name}: no choice for {int(f.choice.isna().sum())} items")
    vals = [value(i, c) for i, c in zip(f.item_id, f.choice)]
    f["acc"] = [v[0] for v in vals]
    f["C"] = [v[1] for v in vals]
    f["overhead"] = [float(overhead(i)) if overhead is not None else 0.0 for i in f.item_id]
    f["lat"] = np.array([v[2] for v in vals], float) + f.overhead.to_numpy(float)
    f["solver_A"] = [parse(c)[0] * parse(c)[1] for c in f.choice]
    f = f.assign(selector=name, kind=kind, menu=menu, overhead_kind=overhead_kind)
    return f


def _overhead_fn(kind: str, diag: Mapping[str, float], llm: Mapping[tuple, float], arm: str | None = None):
    if kind == "none":
        return None
    if kind == "diag":
        return lambda i: diag[i]
    if kind == "llm":
        return lambda i: llm[(arm, i)]
    if kind == "diag+llm":
        return lambda i: diag[i] + llm[(arm, i)]
    raise ValueError(kind)


def analyze(study_root: Path, *, n_sim: int = 2000, n_resamples: int = 10_000, overwrite: bool = False,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    from agent_allocation.passk.features import diag_features
    from agent_allocation.passk.models import e5
    from agent_allocation.passk.models.inputs import (ClassSource, PASSK_SEED, load_analysis_inputs,
                                                      load_panels)
    from agent_allocation.passk.stats import MACRO, MIXTURE, SEED

    root = Path(study_root).resolve()
    out_dir = root.joinpath(*OUT_PARTS)
    if "sealed" in out_dir.relative_to(root).parts:
        raise RuntimeError("refusing to write under sealed/")
    if (out_dir / "report.json").exists() and not overwrite:
        raise FileExistsError(f"{out_dir}/report.json exists (pass --overwrite to replace the selcmp outputs)")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---------------------------------------------------------------- inputs
    e5._require_complete_classes(root)
    draws, pools, classes = load_analysis_inputs(root)
    panels = load_panels(root)
    l96, c48, t200 = sorted(panels["L96"]), sorted(panels["C48"]), sorted(panels["T200"])
    assert set(c48) <= set(l96)
    shard_of = json.loads((root / "artifacts" / "shards" / "assignment.json").read_text())["assignment"]
    bench = dict(zip(draws.item_id, draws.benchmark))
    cls = ClassSource(draws=draws, table=classes, pools=pools)
    e5._track_fallback(cls)
    feats = diag_features(draws, cls, pools=pools)
    f1 = dict(zip(feats.item_id, feats.f1.astype(float)))
    diag = diag_wallclock(draws)
    llm_tab = llm_decision_latency(root)
    llm = {(r.arm, r.item_id): float(r.decision_latency) for r in llm_tab.itertuples(index=False)}
    est = pd.read_parquet(root / "lex" / "front_estimates.parquet")
    est_idx = est.set_index(["item_id", "config"])
    split = pd.read_parquet(root / "lex" / "front_splithalf.parquet").set_index("item_id")
    winners17 = pd.read_parquet(root / "lex" / "front_winners.parquet").set_index("item_id")["config"]
    e8a_l96 = pd.read_parquet(root / "e8a" / "items_L96.parquet").set_index("item_id")
    e8a_t200 = pd.read_parquet(root / "e8a" / "items_T200.parquet").set_index("item_id")
    sealed = pd.read_parquet(root / "sealed" / "t200" / "predictions.parquet")
    for c in M15 + ("1x25",):
        if est[(est.config == c)].p.notna().sum() != len(c48):
            raise RuntimeError(f"bank estimate missing for {c}")

    # ---------------------------------------------------------------- statistical predictions
    oof, cf = crossfit(draws, pools, cls, feats, l96, c48, M15, seed=PASSK_SEED, n_sim=n_sim, log=log)
    oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    t0 = time.time()
    t200_pred, full_info = fit_predict(draws, pools, cls, feats, l96, t200, c48, M15, seed=PASSK_SEED, n_sim=n_sim)
    t200_pred = t200_pred.assign(fold="all_L96")
    assert_crossfit(t200_pred, {"all_L96": l96})
    t200_pred.to_parquet(out_dir / "t200_m15_predictions.parquet", index=False)
    e5._require_no_fallback(cls)
    log(f"[selcmp] full L96 fit -> T200 ({time.time() - t0:.0f}s)")
    chk = sealed.merge(t200_pred, on=["item_id", "config_id", "predictor"], suffixes=("_sealed", "_refit"))
    reproduction = {"n_cells": int(len(chk)), "max_abs_diff_p_delivered": float(
        (chk.p_delivered_sealed - chk.p_delivered_refit).abs().max())}

    # OOF quality against the C48 bank estimates (context for the policy results)
    q = oof[oof.item_id.isin(c48)].merge(est[["item_id", "config", "p"]], left_on=["item_id", "config_id"],
                                         right_on=["item_id", "config"])
    oof_quality = {}
    for name, g in q.groupby("predictor"):
        rho = [pd.Series(h.p_delivered.to_numpy()).corr(pd.Series(h.p.to_numpy()), method="spearman")
               for _, h in g.groupby("item_id") if h.p.nunique() > 1]     # constant bank p: undefined
        oof_quality[name] = {"mae_vs_bank_p": float((g.p_delivered - g.p).abs().mean()),
                             "brier_vs_bank_p": float(((g.p_delivered - g.p) ** 2).mean()),
                             "mean_within_item_spearman": float(np.nanmean(rho))}

    # ---------------------------------------------------------------- C48
    c48_items = pd.DataFrame({"item_id": c48, "benchmark": [bench[i] for i in c48]})

    def bank_value(i: str, c: str) -> tuple[float, float, float]:
        r = est_idx.loc[(i, c)]
        return float(r.p), float(r.C), float(r.latency)

    lat_loo = latency_lookup(est, shard_of, leave_item_out=True)
    frames: list[pd.DataFrame] = []
    for c in M15 + ("1x25",):
        frames.append(_selector_frame(c48_items, pd.Series({i: c for i in c48}), value=bank_value,
                                      name=f"fixed:{c}", kind="fixed" if c != "1x25" else "reference",
                                      menu=c, overhead_kind="none"))
    for name in STAT:
        pr = oof[(oof.predictor == name) & oof.item_id.isin(c48)]
        ok = "none" if name == "P0" else "diag"
        for mname, menu in MENUS.items():
            frames.append(_selector_frame(c48_items, stat_choices(pr, menu, lat_loo), value=bank_value,
                                          overhead=_overhead_fn(ok, diag, llm), name=f"stat:{name}/{mname}",
                                          kind="stat", menu=mname, overhead_kind=ok))

    # heuristic, cross-fitted on the same outer folds
    ro_l96_acc, ro_l96_lat = realized_tables(root, draws, "L96")

    c48_set = set(c48)

    def train_outcome(i: str, c: str) -> tuple[float, float, float]:
        if i in c48_set:
            return bank_value(i, c)
        return ro_l96_acc.get((i, c), np.nan), np.nan, ro_l96_lat.get((i, c), np.nan)

    heur_rules: dict[str, Any] = {}
    for mname, menu in MENUS.items():
        choice = {}
        heur_rules[mname] = {}
        for f, info in cf["folds"].items():
            rule = heuristic_fit(cf["train_sets"][f], f1, bench, train_outcome, diag, menu)
            heur_rules[mname][f"fold{f}"] = {k: v for k, v in rule.items() if k != "table"}
            test_c48 = [i for i in info["test_items"] if i in c48_set]
            choice.update(heuristic_apply(rule, test_c48, f1).to_dict())
        frames.append(_selector_frame(c48_items, pd.Series(choice), value=bank_value,
                                      overhead=_overhead_fn("diag", diag, llm), name=f"heuristic/{mname}",
                                      kind="heuristic", menu=mname, overhead_kind="diag"))
    for arm, ok in (("A0", "llm"), ("A1", "diag+llm")):
        frames.append(_selector_frame(c48_items, e8a_l96[f"{arm}_choice"], value=bank_value,
                                      overhead=_overhead_fn(ok, diag, llm, arm), name=f"LLM:{arm}", kind="llm",
                                      menu="M5", overhead_kind=ok))
    for mname, menu in (("M15", M15), ("M5", M5)):
        ch = lex_choose(est[est.config.isin(menu)]).set_index("item_id")["config"]
        frames.append(_selector_frame(c48_items, ch, value=bank_value, name=f"oracle/{mname}", kind="ceiling",
                                      menu=mname, overhead_kind="none"))
    frames.append(_selector_frame(c48_items, winners17, value=bank_value, name="oracle/M17", kind="ceiling",
                                  menu="M17", overhead_kind="none"))
    sh = c48_items.assign(choice="split-half", acc=c48_items.item_id.map(split.p), C=c48_items.item_id.map(split.C),
                          overhead=0.0, lat=c48_items.item_id.map(split.latency), solver_A=np.nan,
                          selector="splithalf/M17", kind="reference", menu="M17", overhead_kind="none")
    frames.append(sh)
    c48_long = pd.concat(frames, ignore_index=True)
    c48_long.to_parquet(out_dir / "c48_selector_items.parquet", index=False)

    st = Stats(n_resamples, SEED)
    wide = {m: c48_long.pivot(index=["item_id", "benchmark"], columns="selector", values=m).reset_index()
            for m in ("acc", "C", "lat")}
    selectors = list(dict.fromkeys(c48_long.selector))
    meta = c48_long.drop_duplicates("selector").set_index("selector")
    deployable = [s for s in selectors if meta.loc[s, "kind"] in ("fixed", "stat", "heuristic", "llm")]
    c48_rows = []
    for s in selectors:
        g = c48_long[c48_long.selector == s]
        row = {"selector": s, "kind": meta.loc[s, "kind"], "menu": meta.loc[s, "menu"],
               "overhead_kind": meta.loc[s, "overhead_kind"],
               "acc": st.level(g, "acc", MACRO), "C": st.level(g, "C", MACRO), "lat": st.level(g, "lat", MACRO),
               "mean_overhead_s": macro_mean(g.overhead, g.benchmark),
               "mean_solver_A": macro_mean(g.solver_A, g.benchmark),
               "choices": {str(k): int(v) for k, v in g.choice.value_counts().items()}}
        if s != REFERENCE:
            row["minus_1x16"] = {m: st.diff(wide[m], s, REFERENCE, MACRO) for m in ("acc", "C", "lat")}
        c48_rows.append(row)
    pop = pd.DataFrame([{"selector": r["selector"], "acc": r["acc"]["mean"], "C": r["C"]["mean"],
                         "lat": r["lat"]["mean"]} for r in c48_rows])
    order = lex_order(pop[pop.selector.isin(deployable)], acc="acc", cons="C", lat="lat")
    order_all = lex_order(pop, acc="acc", cons="C", lat="lat")
    share = winner_share(wide, deployable, MACRO, cons=True, n_resamples=n_resamples, seed=SEED)
    for r in c48_rows:
        r["lex_rank_deployable"] = order.index(r["selector"]) + 1 if r["selector"] in order else None
        r["lex_rank_all"] = order_all.index(r["selector"]) + 1
        r["winner_share_deployable"] = share.get(r["selector"])
    llm_contrasts_c48 = {}
    for s in [x for x in deployable if meta.loc[x, "kind"] in ("stat", "heuristic")]:
        llm_contrasts_c48[s] = {arm: {m: st.diff(wide[m], s, f"LLM:{arm}", MACRO) for m in ("acc", "C", "lat")}
                                for arm in ("A0", "A1")}
    log(f"[selcmp] C48 scored ({time.time() - started:.0f}s)")

    # ---------------------------------------------------------------- T200
    t_items = pd.DataFrame({"item_id": t200, "benchmark": [bench[i] for i in t200]})
    t_acc, t_lat = realized_tables(root, draws, "T200")

    def realized_value(i: str, c: str) -> tuple[float, float, float]:
        return t_acc.get((i, c), np.nan), np.nan, t_lat.get((i, c), np.nan)

    lat_c48 = latency_lookup(est, shard_of, leave_item_out=False)
    tframes: list[pd.DataFrame] = []
    for c in M15:
        tframes.append(_selector_frame(t_items, pd.Series({i: c for i in t200}), value=realized_value,
                                       name=f"fixed:{c}", kind="fixed", menu=c, overhead_kind="none"))
    for name in STAT:
        ok = "none" if name == "P0" else "diag"
        for mname, menu in MENUS.items():
            tframes.append(_selector_frame(t_items, stat_choices(t200_pred[t200_pred.predictor == name], menu,
                                                                 lat_c48), value=realized_value,
                                           overhead=_overhead_fn(ok, diag, llm), name=f"stat:{name}/{mname}",
                                           kind="stat", menu=mname, overhead_kind=ok))
        for mname, menu in (("M9sealed", M9), ("M5sealed", M5)):
            tframes.append(_selector_frame(t_items, stat_choices(sealed[sealed.predictor == name], menu, lat_c48),
                                           value=realized_value, overhead=_overhead_fn(ok, diag, llm),
                                           name=f"sealed:{name}/{mname[:2]}", kind="stat-sealed", menu=mname,
                                           overhead_kind=ok))
    heur_t200 = {}
    for mname, menu in MENUS.items():
        rule = heuristic_fit(l96, f1, bench, train_outcome, diag, menu)
        heur_t200[mname] = {k: v for k, v in rule.items() if k != "table"}
        heur_t200[mname]["table"] = rule["table"].round(5).to_dict(orient="records")
        tframes.append(_selector_frame(t_items, heuristic_apply(rule, t200, f1), value=realized_value,
                                       overhead=_overhead_fn("diag", diag, llm), name=f"heuristic/{mname}",
                                       kind="heuristic", menu=mname, overhead_kind="diag"))
    for arm, ok in (("A0", "llm"), ("A1", "diag+llm")):
        tframes.append(_selector_frame(t_items, e8a_t200[f"{arm}_choice"], value=realized_value,
                                       overhead=_overhead_fn(ok, diag, llm, arm), name=f"LLM:{arm}", kind="llm",
                                       menu="M5", overhead_kind=ok))
    for mname, menu in MENUS.items():         # realized in-sample ceiling: best realized outcome, then fastest
        ch = {}
        for i in t200:
            opts = [(t_acc.get((i, c), np.nan), t_lat.get((i, c), np.nan), parse(c)[0] * parse(c)[1], c)
                    for c in menu]
            opts = [o for o in opts if np.isfinite(o[0]) and np.isfinite(o[1])]
            best = max(o[0] for o in opts)
            ch[i] = sorted([o for o in opts if o[0] >= best - DELTA], key=lambda o: (o[1], o[2]))[0][3]
        tframes.append(_selector_frame(t_items, pd.Series(ch), value=realized_value, name=f"oracle/{mname}",
                                       kind="ceiling", menu=mname, overhead_kind="none"))
    t_long = pd.concat(tframes, ignore_index=True)
    t_long.to_parquet(out_dir / "t200_selector_items.parquet", index=False)
    twide = {m: t_long.pivot(index=["item_id", "benchmark"], columns="selector", values=m).reset_index()
             for m in ("acc", "lat")}
    tsel = list(dict.fromkeys(t_long.selector))
    tmeta = t_long.drop_duplicates("selector").set_index("selector")
    tdeploy = [s for s in tsel if tmeta.loc[s, "kind"] in ("fixed", "stat", "stat-sealed", "heuristic", "llm")]
    t_rows = []
    for s in tsel:
        g = t_long[t_long.selector == s]
        row = {"selector": s, "kind": tmeta.loc[s, "kind"], "menu": tmeta.loc[s, "menu"],
               "overhead_kind": tmeta.loc[s, "overhead_kind"],
               "acc_mixture": st.level(g, "acc", MIXTURE), "acc_macro": st.level(g, "acc", MACRO),
               "lat_macro": st.level(g, "lat", MACRO), "lat_mixture": st.level(g, "lat", MIXTURE),
               "mean_overhead_s": macro_mean(g.overhead, g.benchmark),
               "mean_solver_A": macro_mean(g.solver_A, g.benchmark),
               "n_unknown_acc": int(g.acc.isna().sum()),
               "choices": {str(k): int(v) for k, v in g.choice.value_counts().items()}}
        if s != REFERENCE:
            row["minus_1x16"] = {"acc_macro": st.diff(twide["acc"], s, REFERENCE, MACRO),
                                 "acc_mixture": st.diff(twide["acc"], s, REFERENCE, MIXTURE),
                                 "lat_macro": st.diff(twide["lat"], s, REFERENCE, MACRO)}
        t_rows.append(row)
    tpop = pd.DataFrame([{"selector": r["selector"], "acc_macro": r["acc_macro"]["mean"],
                          "acc_mixture": r["acc_mixture"]["mean"], "lat_macro": r["lat_macro"]["mean"],
                          "lat_mixture": r["lat_mixture"]["mean"]} for r in t_rows])
    torder_macro = lex_order(tpop[tpop.selector.isin(tdeploy)], acc="acc_macro", lat="lat_macro")
    torder_mix = lex_order(tpop[tpop.selector.isin(tdeploy)], acc="acc_mixture", lat="lat_mixture")
    tshare = winner_share(twide, tdeploy, MACRO, cons=False, n_resamples=n_resamples, seed=SEED)
    tshare_mix = winner_share(twide, tdeploy, MIXTURE, cons=False, n_resamples=n_resamples, seed=SEED)
    for r in t_rows:
        s = r["selector"]
        r["lex_rank_macro"] = torder_macro.index(s) + 1 if s in torder_macro else None
        r["lex_rank_mixture"] = torder_mix.index(s) + 1 if s in torder_mix else None
        r["winner_share_macro"] = tshare.get(s)
        r["winner_share_mixture"] = tshare_mix.get(s)
    llm_contrasts_t200 = {}
    for s in [x for x in tdeploy if tmeta.loc[x, "kind"] in ("stat", "stat-sealed", "heuristic")]:
        llm_contrasts_t200[s] = {arm: {"acc_macro": st.diff(twide["acc"], s, f"LLM:{arm}", MACRO),
                                       "acc_mixture": st.diff(twide["acc"], s, f"LLM:{arm}", MIXTURE),
                                       "lat_macro": st.diff(twide["lat"], s, f"LLM:{arm}", MACRO)}
                                 for arm in ("A0", "A1")}
    log(f"[selcmp] T200 scored ({time.time() - started:.0f}s)")

    # ---------------------------------------------------------------- report
    folds_info = {str(f): {k: v for k, v in info.items()} for f, info in cf["folds"].items()}
    fold_balance = pd.crosstab(pd.Series(cf["assignment"]).rename("fold"),
                               pd.Series({i: bench[i] for i in l96}).rename("benchmark"))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "post-hoc, exploratory (all labels released; not pre-registered beyond PREREG_LEX_front.md)",
        "delta": DELTA, "menus": {"M15": list(M15), "M5": list(M5), "M9sealed": list(M9)},
        "definitions": {
            "accuracy": "C48: bank-estimated delivered accuracy p (tie-averaged); T200: realized S_known (passk)",
            "consistency": "C48: p^2 + (1-p)^2 from the bank p; T200: not measurable per item (one realization)",
            "latency": "chosen option's parallel wall-clock (C48: bank expected max of n call latencies; T200: slowest "
                       "member of the realized pool) + selector overhead",
            "overhead": {"none": "fixed options, P0, oracles, split-half",
                         "diag": "max of the item's five diagnostic call latencies",
                         "llm": "E8a decision latency summed over every attempt",
                         "diag+llm": "diagnostic wall-clock + decision latency (A1)"},
            "stat_rule": "LEX rule per item on predicted p_delivered, C = p^2+(1-p)^2 of the prediction, latency = "
                         "C48 bank-latency mean per (shard, option) (leave-item-out on C48)",
            "heuristic_rule": "if the plurality share f1 of the five diagnostic answers >= tau then X else 1x16; "
                              "(tau, X) in {0.6, 0.8, 1.0} x menu chosen on the training items by macro accuracy "
                              "(C48 items: bank p; other L96 items: realized S_known) -> macro consistency (bank "
                              "items only) -> macro latency incl. diagnostics -> fewer tokens -> larger tau",
            "population_rank": "repeated LEX selection over deployable selectors (fixed M15, stat, heuristic, LLM): "
                               "macro accuracy within 0.02 of the best -> consistency within 0.02 (C48 only) -> "
                               "lowest macro latency",
            "winner_share": "share of 10,000 benchmark-stratified bootstrap replicates in which the selector is the "
                            "population LEX winner among the deployable selectors (C48: macro; T200: "
                            "winner_share_macro and winner_share_mixture)",
            "headline": "per view (C48 macro, T200 macro, T200 mixture): in_top_band = accuracy within delta of the "
                        "best deployable row; within_delta_of_1x16 = accuracy at most delta below always-1x16; "
                        "ci_side = +1/-1 if the 95% percentile CI lies above/below 0, else 0 (a bound within 1e-9 "
                        "of 0 counts as touching); like-for-like statistical rows = C48 stat M5, T200 sealed M5",
        },
        "crossfit": {"outer_folds": "glm.grouped_folds(L96, 5, seed 20260925) = the E5 ridge-CV fold assignment on "
                                    "L96 (verified identical to the sealed fit's)",
                     "fold_benchmark_counts": {str(k): {str(b): int(n) for b, n in v.items()}
                                               for k, v in fold_balance.to_dict(orient="index").items()},
                     "folds": folds_info, "full_fit": full_info,
                     "sealed_reproduction": reproduction, "n_sim": n_sim,
                     "oof_quality_vs_c48_bank": oof_quality, "class_source": cls.stats()},
        "overheads": {"diag_wallclock_macro_c48": macro_mean([diag[i] for i in c48], [bench[i] for i in c48]),
                      "diag_wallclock_macro_t200": macro_mean([diag[i] for i in t200], [bench[i] for i in t200]),
                      "llm_decision": llm_tab.groupby(["arm", "panel"]).agg(
                          mean_s=("decision_latency", "mean"), max_s=("decision_latency", "max"),
                          attempts=("attempts", "sum"), missing=("attempts_missing_latency", "sum"))
                      .reset_index().to_dict(orient="records")},
        "heuristic_rules": {"c48_crossfit": heur_rules, "t200_fit_on_L96": heur_t200},
        "c48": {"weights": "macro", "selectors": c48_rows, "lex_order_deployable": order,
                "lex_order_all": order_all, "stat_minus_llm": llm_contrasts_c48},
        "t200": {"selectors": t_rows, "lex_order_macro": torder_macro, "lex_order_mixture": torder_mix,
                 "stat_minus_llm": llm_contrasts_t200},
        "runtime_s": round(time.time() - started, 1),
    }
    report["headline"] = headline_facts(report)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


# ------------------------------------------------------------------------------------------------------------------
# markdown
# ------------------------------------------------------------------------------------------------------------------

def _choices_str(ch: Mapping[str, int], k: int = 4) -> str:
    items = sorted(ch.items(), key=lambda kv: -kv[1])
    s = ", ".join(f"{c} {n}" for c, n in items[:k])
    return s + (f", +{len(items) - k} more" if len(items) > k else "")


def c48_table(report: Mapping[str, Any], *, only: Iterable[str] | None = None) -> str:
    rows = report["c48"]["selectors"]
    if only is not None:
        keep = set(only)
        rows = [r for r in rows if r["selector"] in keep]
    rows = sorted(rows, key=lambda r: (r["lex_rank_deployable"] is None, r["lex_rank_deployable"] or 0,
                                       r["lex_rank_all"]))
    out = ["| selector | menu | overhead | accuracy [95% CI] | consistency [95% CI] | latency s [95% CI] | "
           "Δacc vs 1x16 (pts) | ΔC vs 1x16 (pts) | Δlat vs 1x16 (s) | LEX rank | P(win) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r.get("minus_1x16", {})
        rank = r["lex_rank_deployable"] if r["lex_rank_deployable"] is not None else "ref"
        share = f"{r['winner_share_deployable']:.2f}" if r.get("winner_share_deployable") is not None else "--"
        out.append(f"| {r['selector']} | {r['menu']} | {r['overhead_kind']} | {_fmt_lvl(r['acc'])} | "
                   f"{_fmt_lvl(r['C'])} | {_fmt_lvl(r['lat'], digits=0)} | {_fmt_diff(d.get('acc'), 100)} | "
                   f"{_fmt_diff(d.get('C'), 100)} | {_fmt_diff(d.get('lat'), 1, 0)} | {rank} | {share} |")
    return "\n".join(out)


def _share(r: Mapping[str, Any], key: str) -> str:
    return f"{r[key]:.2f}" if r.get(key) is not None else "--"


def t200_table(report: Mapping[str, Any], *, only: Iterable[str] | None = None, order: str = "macro") -> str:
    """T200 rows under both weightings (accuracy level, Δ vs 1x16, LEX rank and bootstrap win share for each),
    sorted by the ``order`` ("macro" or "mixture") LEX rank."""
    if order not in ("macro", "mixture"):
        raise ValueError(order)
    rows = report["t200"]["selectors"]
    if only is not None:
        keep = set(only)
        rows = [r for r in rows if r["selector"] in keep]
    rk_key = f"lex_rank_{order}"
    rows = sorted(rows, key=lambda r: (r[rk_key] is None, r[rk_key] or 0))
    out = ["| selector | menu | overhead | acc mixture [95% CI] | acc macro [95% CI] | latency s (macro) [95% CI] | "
           "Δacc mixture vs 1x16 (pts) | Δacc macro vs 1x16 (pts) | Δlat vs 1x16 (s) | rank mixture | rank macro | "
           "P(win) mixture | P(win) macro |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r.get("minus_1x16", {})
        rk = r["lex_rank_macro"] if r["lex_rank_macro"] is not None else "ref"
        rm = r["lex_rank_mixture"] if r["lex_rank_mixture"] is not None else "ref"
        out.append(f"| {r['selector']} | {r['menu']} | {r['overhead_kind']} | {_fmt_lvl(r['acc_mixture'])} | "
                   f"{_fmt_lvl(r['acc_macro'])} | {_fmt_lvl(r['lat_macro'], digits=0)} | "
                   f"{_fmt_diff(d.get('acc_mixture'), 100)} | {_fmt_diff(d.get('acc_macro'), 100)} | "
                   f"{_fmt_diff(d.get('lat_macro'), 1, 0)} | {rm} | {rk} | {_share(r, 'winner_share_mixture')} | "
                   f"{_share(r, 'winner_share_macro')} |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------
# headline facts (generated from the numbers, per panel and weighting, so the prose cannot outrun them)
# ------------------------------------------------------------------------------------------------------------------

# view -> (panel, weights, accuracy key, rank key, win-share key, minus_1x16 key, stat_minus_llm key)
VIEWS = {
    "C48_macro": ("c48", "macro", "acc", "lex_rank_deployable", "winner_share_deployable", "acc", "acc"),
    "T200_macro": ("t200", "macro", "acc_macro", "lex_rank_macro", "winner_share_macro", "acc_macro", "acc_macro"),
    "T200_mixture": ("t200", "mixture", "acc_mixture", "lex_rank_mixture", "winner_share_mixture", "acc_mixture",
                     "acc_mixture"),
}
# the statistical rows that share the LLM allocator's 5-option menu: cross-fitted M5 on C48, sealed M5 on T200
LIKE_FOR_LIKE = {"c48": ("stat", "M5"), "t200": ("stat-sealed", "M5sealed")}
STAT_KINDS = ("stat", "stat-sealed")


def _predictor(selector: str) -> str:
    return selector.split(":", 1)[1].split("/", 1)[0]


def _contrast(c: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not c:
        return None
    return {"estimate": float(c["estimate"]), "ci95": [float(x) for x in c["ci95"]], "p": float(c["p"]),
            "ci_side": ci_side(c["ci95"])}


def headline_facts(report: Mapping[str, Any], delta: float = DELTA) -> dict[str, Any]:
    """Per view (C48 macro, T200 macro, T200 mixture): where each statistical and LLM row sits against the LEX
    accuracy tie band, its contrast with always-1x16, its rank and win share, and the like-for-like
    statistical-minus-LLM accuracy contrasts. ``ci_side`` is +1 / -1 when the percentile CI is above / below 0,
    else 0; ``in_top_band`` is accuracy within ``delta`` of the best deployable row (the set latency can decide);
    ``within_delta_of_1x16`` is accuracy no more than ``delta`` below always-1x16."""
    out: dict[str, Any] = {}
    for view, (panel, weights, acc_k, rank_k, share_k, d_k, c_k) in VIEWS.items():
        rows = report[panel]["selectors"]
        dep = [r for r in rows if r.get(rank_k) is not None]
        best = max(r[acc_k]["mean"] for r in dep)
        ref_acc = next(r for r in rows if r["selector"] == REFERENCE)[acc_k]["mean"]

        def entry(r: Mapping[str, Any]) -> dict[str, Any]:
            a = float(r[acc_k]["mean"])
            return {"selector": r["selector"], "predictor": _predictor(r["selector"]) if r["kind"] in STAT_KINDS
                    else None, "kind": r["kind"], "menu": r["menu"], "acc": a,
                    "in_top_band": bool(a >= best - delta - EPS),
                    "within_delta_of_1x16": bool(a >= ref_acc - delta - EPS),
                    "minus_1x16": _contrast(r.get("minus_1x16", {}).get(d_k)),
                    "rank": r[rank_k], "winner_share": r[share_k]}

        stat = [entry(r) for r in dep if r["kind"] in STAT_KINDS]
        llm = {r["selector"]: entry(r) for r in dep if r["kind"] == "llm"}
        best_llm_rank = min(e["rank"] for e in llm.values())
        kind, menu = LIKE_FOR_LIKE[panel]
        lfl = [r["selector"] for r in rows if r["kind"] == kind and r["menu"] == menu]
        vs_llm = {}
        for arm in ("A0", "A1"):
            cs = [{"selector": s, **_contrast(report[panel]["stat_minus_llm"][s][arm][c_k])} for s in lfl]
            est = [c["estimate"] for c in cs]
            vs_llm[f"LLM:{arm}"] = {
                "rows": cs, "n": len(cs), "estimate_range": [min(est), max(est)],
                "n_ci_above_zero": sum(c["ci_side"] > 0 for c in cs),
                "n_ci_below_zero": sum(c["ci_side"] < 0 for c in cs),
                "ci_includes_or_touches_zero": [c["selector"] for c in cs if c["ci_side"] == 0],
                "p_range": [min(c["p"] for c in cs), max(c["p"] for c in cs)]}
        stat_share = max(stat, key=lambda e: (e["winner_share"] or 0.0, -e["rank"]))
        others = {r["selector"]: entry(r) for r in dep if r["selector"] == REFERENCE or r["kind"] == "heuristic"}
        out[view] = {
            "panel": panel, "weights": weights, "delta": delta, "best_deployable_acc": float(best),
            "band_floor": float(best - delta), "acc_1x16": float(ref_acc),
            "stat_rows": stat, "llm": llm, "reference_rows": others,
            "stat_like_for_like": lfl,
            "stat_outside_top_band": [e["selector"] for e in stat if not e["in_top_band"]],
            "stat_ranked_below_best_llm": [e["selector"] for e in sorted(stat, key=lambda e: e["rank"])
                                           if e["rank"] > best_llm_rank],
            "best_llm_rank": best_llm_rank,
            "stat_ranks": [min(e["rank"] for e in stat), max(e["rank"] for e in stat)],
            "max_stat_winner_share": {"selector": stat_share["selector"],
                                      "share": float(stat_share["winner_share"] or 0.0)},
            "stat_minus_llm_like_for_like": vs_llm,
        }
    return out


def _side_text(c: Mapping[str, Any]) -> str:
    if c["ci_side"] > 0:
        return "CI above 0"
    if c["ci_side"] < 0:
        return "CI below 0"
    return "CI touches 0" if any(abs(x) <= CI_TOL for x in c["ci95"]) else "CI includes 0"


def _fmt_c(c: Mapping[str, Any] | None) -> str:
    if not c:
        return "--"
    lo, hi = c["ci95"]
    if all(abs(x) <= CI_TOL for x in (c["estimate"], lo, hi)):
        return "no difference on any item"
    return (f"{_z(c['estimate']) * 100:+.1f} pts [{_z(lo) * 100:+.1f}, {_z(hi) * 100:+.1f}], {_side_text(c)}, "
            f"p = {c['p']:.3f}")


def _fmt_entry(s: str, e: Mapping[str, Any]) -> str:
    return (f"{s} {e['acc']:.3f}, vs 1x16 {_fmt_c(e['minus_1x16'])}, "
            f"{'inside' if e['in_top_band'] else 'outside'} the top band, rank {e['rank']}, "
            f"P(win) {e['winner_share']:.3f}")


def render_findings(facts: Mapping[str, Any]) -> list[str]:
    """Markdown bullets for :func:`headline_facts`, one block per view."""
    titles = {"C48_macro": "C48, macro weights (expected per-item values; ranking uses accuracy -> consistency -> "
                           "latency)",
              "T200_macro": "T200, macro weights (one realized outcome per item; accuracy -> latency)",
              "T200_mixture": "T200, mixture weights (aime .10, hmmt .10, omni .50, gpqa .30: the study's primary "
                              "T200 estimand and E8a's pre-registered test; accuracy -> latency)"}
    lines: list[str] = []
    for view, f in facts.items():
        lines += [f"**{titles.get(view, view)}.** Best deployable accuracy {f['best_deployable_acc']:.3f}, so the "
                  f"LEX tie band starts at {f['band_floor']:.3f}; always-1x16 is {f['acc_1x16']:.3f}.", ""]
        by_pred: dict[str, list[dict]] = {}
        for e in f["stat_rows"]:
            by_pred.setdefault(e["predictor"], []).append(e)
        lfl = set(f["stat_like_for_like"])
        lines.append("- Statistical models (accuracy and contrast shown for the row on the LLM's 5-option menu; "
                     "band counts and ranks over all of the predictor's rows):")
        for p, es in by_pred.items():
            rep = next((e for e in es if e["selector"] in lfl), es[0])
            n_in = sum(e["in_top_band"] for e in es)
            n_1x16 = sum(e["within_delta_of_1x16"] for e in es)
            lo_r, hi_r = min(e["rank"] for e in es), max(e["rank"] for e in es)
            lines.append(f"  - {p}: {n_in} of {len(es)} rows inside the top band, {n_1x16} of {len(es)} within 0.02 "
                         f"of 1x16; {rep['selector']} {rep['acc']:.3f}, vs 1x16 {_fmt_c(rep['minus_1x16'])}; "
                         f"rank{'s' if lo_r != hi_r else ''} {lo_r}{f'-{hi_r}' if lo_r != hi_r else ''}.")
        lines.append("- LLM selectors: " + "; ".join(_fmt_entry(s, e) for s, e in f["llm"].items()) + ".")
        lines.append("- Reference rows: " + "; ".join(_fmt_entry(s, e) for s, e in f["reference_rows"].items())
                     + ".")
        vs = []
        for arm, v in f["stat_minus_llm_like_for_like"].items():
            lo, hi = v["estimate_range"]
            inc = v["ci_includes_or_touches_zero"]
            vs.append(f"vs {arm} {lo * 100:+.1f} to {hi * 100:+.1f} pts, CI above 0 for {v['n_ci_above_zero']} of "
                      f"{v['n']}" + (f" (includes or touches 0: {', '.join(inc)})" if inc else "")
                      + f", p {v['p_range'][0]:.3f}-{v['p_range'][1]:.3f}")
        lines.append(f"- Statistical minus LLM accuracy on the LLM's own 5-option menu "
                     f"({', '.join(f['stat_like_for_like'])}): " + "; ".join(vs) + ".")
        below = f["stat_ranked_below_best_llm"]
        lines.append(f"- Ranking: statistical rows rank {f['stat_ranks'][0]}-{f['stat_ranks'][1]}; the better LLM "
                     f"ranks {f['best_llm_rank']}; statistical rows ranked below it: "
                     + (", ".join(below) if below else "none") + ".")
        ms = f["max_stat_winner_share"]
        lines += [f"- Bootstrap win share: " + ", ".join(f"{s} {e['winner_share']:.3f}" for s, e in f["llm"].items())
                  + f"; highest statistical row {ms['selector']} {ms['share']:.3f}.", ""]
    lines.append("CI = 95% percentile interval of the 10,000-replicate benchmark-stratified bootstrap; p = the "
                 "null-centred studentised bootstrap p of the same contrast (unadjusted). The two can disagree at "
                 "these sample sizes. \"CI touches 0\" = a bound exactly at 0 (0/1 outcomes), which is not an "
                 "exclusion of 0.")
    return lines


HEADLINE_C48 = ("fixed:1x16", "fixed:1x8", "fixed:1x25", "stat:P0/M5", "stat:P1/M5", "stat:P1-conc/M5", "stat:P2/M5",
                "stat:P2-conc/M5", "stat:P3/M5", "stat:P2/M15", "stat:P3/M15", "heuristic/M5", "heuristic/M15",
                "LLM:A0", "LLM:A1", "oracle/M5", "oracle/M15")
HEADLINE_T200 = ("fixed:1x16", "fixed:2x8", "sealed:P0/M5", "sealed:P1/M5", "sealed:P1-conc/M5", "sealed:P2/M5",
                 "sealed:P2-conc/M5", "sealed:P3/M5", "sealed:P3/M9", "stat:P3/M15", "heuristic/M5", "heuristic/M15",
                 "LLM:A0", "LLM:A1", "oracle/M5")


def render_markdown(report: Mapping[str, Any]) -> str:
    cf = report["crossfit"]
    lines = [
        "# passk_cont_v1 LEX selector comparison (post-hoc, exploratory)",
        "",
        f"Generated {report['created_at']} by `python -m agent_allocation.passk.selcmp`. Objective: accuracy -> "
        "consistency -> latency (incl. selector overhead), ties within 0.02. C48 is scored with the per-item bank "
        "estimates (expected values, macro over benchmarks); T200 with realized outcomes (one realization per item, "
        "so consistency is not measurable and the ranking is accuracy -> latency).",
        "",
        "Overheads: `diag` = max of the item's five diagnostic call latencies "
        f"(macro mean {report['overheads']['diag_wallclock_macro_c48']:.0f} s on C48, "
        f"{report['overheads']['diag_wallclock_macro_t200']:.0f} s on T200); `llm` = the E8a decision call latency "
        "summed over retries; `diag+llm` = both (A1). LLM mean decision latency: " + ", ".join(
            f"{r['arm']}/{r['panel']} {r['mean_s']:.0f} s" for r in report["overheads"]["llm_decision"]) + ".",
        "",
        "Statistical selectors are cross-fitted (5 problem-grouped folds over L96 = the E5 CV fold assignment; each "
        "C48 item's predictions come from a fit that never saw it). The refit on all of L96 reproduces the sealed "
        f"T200 predictions (max |diff| = {cf['sealed_reproduction']['max_abs_diff_p_delivered']:.2g} over "
        f"{cf['sealed_reproduction']['n_cells']} cells).",
        "",
        "`LEX rank` ranks the deployable selectors (fixed M15 options, statistical, heuristic, LLM); `ref` rows "
        "(1x25, oracles, split-half) are references outside that competition. `P(win)` = share of bootstrap "
        "replicates in which the selector is the population LEX winner; on T200 it is given under both the mixture "
        "and the macro weights, and the two rankings differ.",
        "",
        "## Headline: statistical vs LLM selectors, by panel and weighting",
        "",
        "Generated from the numbers in this report (`report.json` → `headline`). The answer depends on the panel "
        "and, on T200, on the benchmark weighting; read each block on its own.",
        "",
        *render_findings(report["headline"]),
        "",
        "C48 (bank estimates, macro; M5 rows are like-for-like with the LLM allocator's menu):",
        "",
        c48_table(report, only=HEADLINE_C48),
        "",
        "T200 (realized outcomes; `sealed:` = the sealed 9-option predictions, `stat:` = post-hoc M15 refit), "
        "sorted by the mixture rank:",
        "",
        t200_table(report, only=HEADLINE_T200, order="mixture"),
        "",
        "## C48 (bank estimates, macro), all selectors",
        "",
        c48_table(report),
        "",
        "## C48: statistical and heuristic selectors minus the LLM selectors",
        "",
        "| selector | vs | Δacc (pts) | p (acc) | ΔC (pts) | Δlat (s) |",
        "|---|---|---|---|---|---|",
    ]
    for s, d in report["c48"]["stat_minus_llm"].items():
        for arm, m in d.items():
            lines.append(f"| {s} | LLM:{arm} | {_fmt_diff(m['acc'], 100)} | {m['acc']['p']:.3f} | "
                         f"{_fmt_diff(m['C'], 100)} | {_fmt_diff(m['lat'], 1, 0)} |")
    lines += ["", "## C48 option distributions", "", "| selector | choices |", "|---|---|"]
    for r in report["c48"]["selectors"]:
        lines.append(f"| {r['selector']} | {_choices_str(r['choices'], 6)} |")
    lines += ["", "## T200 (realized outcomes), all selectors, sorted by the macro rank", "", t200_table(report), "",
              "## T200: statistical and heuristic selectors minus the LLM selectors", "",
              "| selector | vs | Δacc mixture (pts) | p mixture | Δacc macro (pts) | p macro | Δlat (s) |",
              "|---|---|---|---|---|---|---|"]
    for s, d in report["t200"]["stat_minus_llm"].items():
        for arm, m in d.items():
            lines.append(f"| {s} | LLM:{arm} | {_fmt_diff(m['acc_mixture'], 100)} | {m['acc_mixture']['p']:.3f} | "
                         f"{_fmt_diff(m['acc_macro'], 100)} | {m['acc_macro']['p']:.3f} | "
                         f"{_fmt_diff(m['lat_macro'], 1, 0)} |")
    lines += ["", "## T200 option distributions", "", "| selector | choices |", "|---|---|"]
    for r in report["t200"]["selectors"]:
        lines.append(f"| {r['selector']} | {_choices_str(r['choices'], 6)} |")
    lines += ["", "## Heuristic rules chosen", "", "| fit | menu | tau | X | train acc | train C | train latency |",
              "|---|---|---|---|---|---|---|"]
    for menu, folds in report["heuristic_rules"]["c48_crossfit"].items():
        for f, r in folds.items():
            lines.append(f"| {f} | {menu} | {r['tau']:g} | {r['X']} | {r['train_acc']:.3f} | {r['train_C']:.3f} | "
                         f"{r['train_lat']:.0f} |")
    for menu, r in report["heuristic_rules"]["t200_fit_on_L96"].items():
        lines.append(f"| all L96 -> T200 | {menu} | {r['tau']:g} | {r['X']} | {r['train_acc']:.3f} | "
                     f"{r['train_C']:.3f} | {r['train_lat']:.0f} |")
    lines += ["", "## Cross-fit details", "",
              "| fold | train items | test items | train pools | latency feature | P1 ridge (delivered) | "
              "P2 ridge (response/eta) |", "|---|---|---|---|---|---|---|"]
    for f, info in cf["folds"].items():
        r = info["best_ridges"]
        lines.append(f"| {f} | {info['n_train_items']} | {info['n_test_items']} | {info['n_train_pools']} | "
                     f"{info['latency_choice']['use_latency']} | {r['P1']['delivered']:g} | "
                     f"{r['P2']['response']:g}/{r['P2']['eta']:g} |")
    lines += ["", "Out-of-fold predictions vs the C48 bank p (all 15 M15 cells per item):", "",
              "| predictor | MAE | squared error | mean within-item Spearman |", "|---|---|---|---|"]
    for name, q in cf["oof_quality_vs_c48_bank"].items():
        lines.append(f"| {name} | {q['mae_vs_bank_p']:.3f} | {q['brier_vs_bank_p']:.3f} | "
                     f"{q['mean_within_item_spearman']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.selcmp")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--n-sim", type=int, default=2000)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true", help="replace existing R/lex/selcmp outputs")
    args = parser.parse_args(argv)
    report = analyze(Path(args.study_root), n_sim=args.n_sim, n_resamples=args.n_resamples,
                     overwrite=args.overwrite)
    print(json.dumps({"c48_order": report["c48"]["lex_order_deployable"][:10],
                      "t200_order_macro": report["t200"]["lex_order_macro"][:10],
                      "runtime_s": report["runtime_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
