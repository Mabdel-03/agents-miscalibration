"""E2: C48 depth-response fit -> sealed V48 predictions, and the post-release scorer.

Entry points: :func:`fit_and_predict` (writes ``sealed/v48/``) and :func:`score` (after the V48 release).

Models (``model`` column):

=========================  ==================  ============  ===========  ====================
model                      response            classes       treatment    ablation role
=========================  ==================  ============  ===========  ====================
beta_full_diffuse          Beta(1/2,1/2) cell  all depths    diffuse      full
beta_full_concentrated     Beta cell           all depths    concentrated full
glm_full_diffuse           pooled glm          all depths    diffuse      pooled response
glm_full_concentrated      pooled glm          all depths    concentrated pooled response
shallow_diffuse            A-only + glm shift  A draws only  diffuse      shallow-only
shallow_concentrated       A-only + glm shift  A draws only  concentrated shallow-only
beta_nocomp                Beta cell           none          singletons   depth w/o competition
ridge                      unstructured ridge  (features)    --           unstructured ridge
=========================  ==================  ============  ===========  ====================

Fitting data: C48 items' development draws only (legacy A slots 0-15, legacy deep r0, ``e1_fill``, ``c48_extra``)
via :func:`inputs.fitting_draws`, which never returns a ``v48_bank``/``t200_*`` row; a guard re-checks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agent_allocation.passk.models import glm as glm_mod
from agent_allocation.passk.models.answers import item_classes, modal_margin, simulate_pools
from agent_allocation.passk.models.inputs import (
    BENCHMARKS, CONFIGS_16A, CONFIGS_8A, CROSSOVER_K, DEPTHS, PASSK_SEED, PREDICT_CONFIGS, ClassSource,
    build_prefix_pools, config_id, fitting_draws, label_value, load_analysis_inputs, load_panels,
    observed_pool_table, parse_config, pool_outcome, sha256_file,
)
from agent_allocation.passk.models.response import (
    beta_draws, cell_counts, coverage_from_draws, fit_pooled, shallow_draws,
)

VERSION = "passk-e2-v1"
FENCED = ("v48_bank", "t200_bank", "t200_diag")
STRUCTURED = {
    "beta_full_diffuse": ("beta", "diffuse", True, "all"),
    "beta_full_concentrated": ("beta", "concentrated", True, "all"),
    "glm_full_diffuse": ("glm", "diffuse", True, "all"),
    "glm_full_concentrated": ("glm", "concentrated", True, "all"),
    "shallow_diffuse": ("shallow", "diffuse", True, "shallow"),
    "shallow_concentrated": ("shallow", "concentrated", True, "shallow"),
    "beta_nocomp": ("beta", "diffuse", False, "all"),
}
MODELS = tuple(STRUCTURED) + ("ridge",)
PI = (0.05, 0.95)          # per-item 90% posterior interval
POP_PI = (0.025, 0.975)    # population (macro) 95% predictive interval


def needed_configs() -> list[str]:
    extra = [config_id(k, 1) for k in CROSSOVER_K] + [config_id(1, k) for k in CROSSOVER_K]
    return list(dict.fromkeys(list(PREDICT_CONFIGS) + extra))


def _guard(fit: pd.DataFrame) -> None:
    bad = fit[fit.partition.isin(FENCED)]
    if len(bad):
        raise RuntimeError(f"E2 fit frame contains {len(bad)} fenced rows ({sorted(set(bad.partition))})")


# ---------------------------------------------------------------------- unstructured ridge
def _item_features(fd_item: pd.DataFrame, ic) -> dict:
    labs = np.array([label_value(v) for v in fd_item["correct"]])
    feats = {}
    for a in DEPTHS:
        m = (fd_item.allowance_a.to_numpy() == a) & ~np.isnan(labs)
        feats[f"rate{a}"] = (labs[m].sum() + 0.5) / (m.sum() + 1.0)
    tot = ic.correct_counts.sum() + ic.wrong_counts.sum()
    top = max(ic.correct_counts.max(initial=0), ic.wrong_counts.max(initial=0))
    feats["top_share"] = top / tot if tot else 0.0
    feats["n_classes"] = ic.n_classes / max(len(fd_item), 1)
    feats["novote"] = ic.n_novote / max(len(fd_item), 1)
    return feats


def _ridge_row(bench: str, n: int, a: int, f: dict) -> list[float]:
    ln, la = np.log2(n), np.log2(a)
    ra = f[f"rate{a}"]
    return [float(bench == b) for b in BENCHMARKS] + [
        ln, la, ra, f["rate1"], f["top_share"], f["n_classes"], f["novote"], ln * ra, ln * f["top_share"], la * ra]


def fit_ridge(fit: pd.DataFrame, cls: ClassSource, items_bench: dict[str, str], configs: list[str],
              rng: np.random.Generator, subsets: int = 6) -> dict:
    lab_map = dict(zip(fit.candidate_id, [label_value(v) for v in fit["correct"]]))
    X, yc, yp, grp = [], [], [], []
    feats = {}
    for item, g in fit.groupby("item_id"):
        ic = item_classes(g, cls)
        feats[item] = _item_features(g, ic)
        classes = cls.classes(f"e2fit:{item}", list(g.sort_values(["allowance_a", "replicate"]).candidate_id))
        for cid in configs:
            n, a = parse_config(cid)
            ids = list(g[g.allowance_a == a].sort_values("replicate").candidate_id)
            if len(ids) < n:
                continue
            picks = [ids[:n]] + [list(rng.choice(ids, size=n, replace=False)) for _ in range(subsets - 1)
                                 if len(ids) > n]
            for mem in picks:
                out = pool_outcome(mem, classes, lab_map)
                if np.isnan(out["coverage"]) or np.isnan(out["plurality"]):
                    continue
                X.append(_ridge_row(items_bench[item], n, a, feats[item]))
                yc.append(out["coverage"])
                yp.append(out["plurality"])
                grp.append(item)
    X = np.asarray(X, float)
    models = {}
    for name, y in (("coverage", np.asarray(yc)), ("plurality", np.asarray(yp))):
        best, table = glm_mod.cv_ridge(X, y, grp, item_intercept=False, ridges=(0.3, 1.0, 3.0, 10.0, 30.0))
        models[name] = {"fit": glm_mod.fit_glm(X, y, grp, item_intercept=False, ridge=best, laplace=True),
                        "cv": table, "ridge": best}
    return {"models": models, "features": feats, "n_rows": int(len(X))}


# ---------------------------------------------------------------------- fit + predict
def fit_and_predict(run_root: Path | str | None = None, *, draws: pd.DataFrame | None = None,
                    pools: pd.DataFrame | None = None, classes: pd.DataFrame | None = None,
                    c48: list[str] | None = None, out_dir: Path | str | None = None, S: int = 800, inner: int = 48,
                    seed: int = PASSK_SEED, write: bool = True, overwrite: bool = False) -> dict:
    """Fit every E2 model on C48 development draws and predict all V48 items (= C48 items) x configurations."""
    if draws is None:
        draws, pools, classes = load_analysis_inputs(Path(run_root))
    if c48 is None:
        c48 = list(load_panels(Path(run_root))["C48"])
    out_dir = Path(out_dir) if out_dir is not None else (Path(run_root) / "sealed" / "v48" if write else None)
    if write and (out_dir / "predictions.parquet").exists() and not overwrite:
        raise FileExistsError(f"{out_dir}/predictions.parquet exists (sealed artifacts are write-once)")
    rng = np.random.default_rng(seed)
    fit = fitting_draws(draws, c48)
    _guard(fit)
    cls = ClassSource(draws=draws, table=classes)
    counts = cell_counts(fit)
    pooled = fit_pooled(fit, seed=seed)
    bench = dict(zip(fit.item_id, fit.benchmark))
    items = [i for i in c48 if i in bench]
    configs = needed_configs()

    pred_rows, sims = [], {}
    beta_P = beta_draws(counts, items, S, rng)
    for item in items:
        g = fit[fit.item_id == item]
        ic_all = item_classes(g, cls)
        ic_sh = item_classes(g, cls, depths=(1,), scope_id=f"e2fit:{item}")
        resp = {"beta": beta_P[item], "glm": pooled.draws(item, bench[item], S, rng),
                "shallow": shallow_draws(counts, pooled, item, bench[item], S, rng)}
        for model, (rkind, treat, comp, cscope) in STRUCTURED.items():
            P = resp[rkind]
            ic = ic_all if cscope == "all" else ic_sh
            for cid in configs:
                n, a = parse_config(cid)
                plur, cov = simulate_pools(P[:, DEPTHS.index(a)], ic, n, rng, treatment=treat, competition=comp,
                                           inner=inner)
                sims[(item, cid, model)] = (plur, cov)
                pred_rows.append(_pred_row(item, bench[item], cid, model, plur, cov))
    ridge = fit_ridge(fit, cls, bench, configs, rng)
    for item in items:
        f = ridge["features"].get(item)
        if f is None:
            continue
        for cid in configs:
            n, a = parse_config(cid)
            x = np.asarray([_ridge_row(bench[item], n, a, f)])
            out = {}
            for name in ("coverage", "plurality"):
                m = ridge["models"][name]["fit"]
                b0, coef, _ = m.sample_params(S, rng)
                out[name] = 1.0 / (1.0 + np.exp(-(b0 + coef @ x[0])))
            sims[(item, cid, "ridge")] = (out["plurality"], out["coverage"])
            pred_rows.append(_pred_row(item, bench[item], cid, "ridge", out["plurality"], out["coverage"]))
    pred = pd.DataFrame(pred_rows)
    pred = _add_ranks(pred)
    cross = _crossover(pred, sims)
    pop = _population(pred, sims, rng)
    illus = _illustrative(cross)
    curves = []
    for item in illus.item_id:
        g = fit[fit.item_id == item]
        ic = item_classes(g, cls)
        for rkind, P in (("beta", beta_P[item]), ("glm", pooled.draws(item, bench[item], S, rng))):
            for j, a in enumerate(DEPTHS):
                pa = P[:, j]
                h_over_a = -np.log1p(-np.clip(pa, 0, 1 - 1e-12)) / a
                row = {"item_id": item, "benchmark": bench[item], "response": rkind, "a": a,
                       "p_mean": float(pa.mean()), "p_lo": float(np.quantile(pa, PI[0])),
                       "p_hi": float(np.quantile(pa, PI[1])), "h_over_a_median": float(np.median(h_over_a)),
                       "h_over_a_lo": float(np.quantile(h_over_a, PI[0])),
                       "h_over_a_hi": float(np.quantile(h_over_a, PI[1]))}
                for treat in ("diffuse", "concentrated"):
                    mm = modal_margin(pa, ic, rng, treatment=treat)
                    row[f"margin_{treat}_mean"] = float(mm.mean())
                    row[f"margin_{treat}_lo"] = float(np.quantile(mm, PI[0]))
                    row[f"margin_{treat}_hi"] = float(np.quantile(mm, PI[1]))
                    row[f"p_correct_modal_{treat}"] = float(np.mean(mm > 0))
                curves.append(row)
    result = {"predictions": pred, "crossover": cross, "population": pop, "illustrative": illus,
              "illustrative_curves": pd.DataFrame(curves), "treatment_gap": _treatment_gap(pred),
              "pooled": pooled, "ridge": ridge, "classes_source": cls.source, "classes_stats": cls.stats()}
    if write:
        _write(out_dir, result, seed=seed, S=S, inner=inner, items=items, fit=fit)
    return result


def _pred_row(item, benchmark, cid, model, plur, cov) -> dict:
    n, a = parse_config(cid)
    return {"item_id": item, "benchmark": benchmark, "config_id": cid, "n": n, "a": a, "model": model,
            "pred_coverage": float(np.mean(cov)), "pred_plurality": float(np.mean(plur)),
            "pi_lo": float(np.quantile(plur, PI[0])), "pi_hi": float(np.quantile(plur, PI[1])),
            "cov_pi_lo": float(np.quantile(cov, PI[0])), "cov_pi_hi": float(np.quantile(cov, PI[1])),
            "in_menu": cid in PREDICT_CONFIGS}


def _budget_group(cid: str) -> str | None:
    return "16A" if cid in CONFIGS_16A else ("8A" if cid in CONFIGS_8A else None)


def _add_ranks(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["budget"] = [_budget_group(c) for c in pred.config_id]
    pred["pred_rank"] = np.nan
    m = pred.budget.notna()
    pred.loc[m, "pred_rank"] = pred[m].groupby(["item_id", "model", "budget"]).pred_plurality.rank(
        ascending=False, method="min")
    return pred


def _crossover(pred: pd.DataFrame, sims: dict) -> pd.DataFrame:
    rows = []
    for (item, model), g in pred.groupby(["item_id", "model"]):
        for k in CROSSOVER_K:
            deep = sims[(item, config_id(1, k), model)][1]          # P(1 x kA correct) = coverage of one call
            wide = sims[(item, config_id(k, 1), model)][1]          # P(k x A covered)
            dd = deep - wide
            rows.append({"item_id": item, "benchmark": g.benchmark.iloc[0], "model": model, "k": k,
                         "pred_deep": float(np.mean(deep)), "pred_wide_cov": float(np.mean(wide)),
                         "pred_diff": float(np.mean(dd)), "pred_sign": int(np.sign(np.mean(dd))),
                         "p_sign_pos": float(np.mean(dd > 0)),
                         "diff_lo": float(np.quantile(dd, PI[0])), "diff_hi": float(np.quantile(dd, PI[1]))})
    return pd.DataFrame(rows)


def _macro(values: np.ndarray, benches: list[str]) -> np.ndarray:
    """Equal-benchmark macro over items along axis 0 (values: (items, S))."""
    b = np.asarray(benches)
    parts = [values[b == x].mean(0) for x in BENCHMARKS if (b == x).any()]
    return np.mean(parts, axis=0)


def _population(pred: pd.DataFrame, sims: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (model, cid), g in pred.groupby(["model", "config_id"]):
        items, benches = list(g.item_id), list(g.benchmark)
        for q, idx in (("plurality", 0), ("coverage", 1)):
            probs = np.stack([sims[(i, cid, model)][idx] for i in items])          # (items, S)
            outcomes = (rng.uniform(size=probs.shape) < probs).astype(float)      # predictive incl. Bernoulli
            for scope in ("macro",) + BENCHMARKS:
                if scope == "macro":
                    mean_p, pred_o = _macro(probs, benches), _macro(outcomes, benches)
                else:
                    sel = np.asarray(benches) == scope
                    if not sel.any():
                        continue
                    mean_p, pred_o = probs[sel].mean(0), outcomes[sel].mean(0)
                rows.append({"model": model, "config_id": cid, "quantity": q, "scope": scope,
                             "pred_mean": float(mean_p.mean()),
                             "pi_lo": float(np.quantile(pred_o, POP_PI[0])),
                             "pi_hi": float(np.quantile(pred_o, POP_PI[1]))})
    return pd.DataFrame(rows)


MATERIAL_GAP = 0.10


def _treatment_gap(pred: pd.DataFrame) -> pd.DataFrame:
    """Diffuse vs concentrated residual treatments: |difference| in predicted plurality per item x config; gaps above
    MATERIAL_GAP are flagged and reported as uncertainty (the correct class is not declared modal)."""
    rows = []
    for base in ("beta_full", "glm_full", "shallow"):
        d = pred[pred.model == f"{base}_diffuse"].set_index(["item_id", "config_id"]).pred_plurality
        c = pred[pred.model == f"{base}_concentrated"].set_index(["item_id", "config_id"]).pred_plurality
        j = pd.concat({"diffuse": d, "concentrated": c}, axis=1).dropna().reset_index()
        j["base"] = base
        j["gap"] = (j.diffuse - j.concentrated).abs()
        j["material"] = j.gap > MATERIAL_GAP
        rows.append(j)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _illustrative(cross: pd.DataFrame) -> pd.DataFrame:
    """Frozen rule: per benchmark, the item with the largest predicted |crossover| (beta_full_diffuse, any k)."""
    c = cross[cross.model == "beta_full_diffuse"].copy()
    c["abs"] = c.pred_diff.abs()
    c = c.sort_values(["benchmark", "abs", "item_id"], ascending=[True, False, True])
    return c.groupby("benchmark").head(1)[["benchmark", "item_id", "k", "pred_diff"]].reset_index(drop=True)


def _write(out_dir: Path, res: dict, *, seed: int, S: int, inner: int, items: list[str], fit: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    res["predictions"].to_parquet(out_dir / "predictions.parquet", index=False)
    res["crossover"].to_parquet(out_dir / "crossover.parquet", index=False)
    res["population"].to_parquet(out_dir / "population.parquet", index=False)
    res["illustrative"].to_parquet(out_dir / "illustrative.parquet", index=False)
    res["illustrative_curves"].to_parquet(out_dir / "illustrative_curves.parquet", index=False)
    res["treatment_gap"].to_parquet(out_dir / "treatment_gap.parquet", index=False)
    pooled = res["pooled"]
    (out_dir / "models" / "pooled_glm.json").write_text(json.dumps(
        {"fit": pooled.fit.to_dict(), "monotone": pooled.monotone, "ridge": pooled.ridge, "cv": pooled.cv},
        indent=1, default=float))
    rg = res["ridge"]
    (out_dir / "models" / "ridge.json").write_text(json.dumps(
        {k: {"fit": v["fit"].to_dict(), "ridge": v["ridge"], "cv": v["cv"]} for k, v in rg["models"].items()}
        | {"n_rows": rg["n_rows"]}, indent=1, default=float))
    seeds = {"passk_seed": seed, "S": S, "inner": inner, "beta_prior": [0.5, 0.5], "alpha_residual": 1.0,
             "pi": PI, "pop_pi": POP_PI, "cv_folds": 5}
    (out_dir / "seeds.json").write_text(json.dumps(seeds, indent=1))
    manifest = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "classes_source": res["classes_source"], "classes": res.get("classes_stats"), "n_items": len(items),
                "items": items,
                "fit_partitions": sorted(set(fit.partition)), "n_fit_draws": int(len(fit)),
                "models": list(MODELS), "configs": needed_configs()}
    manifest["sha256"] = {str(p.relative_to(out_dir)): sha256_file(p) for p in sorted(out_dir.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------------- scoring (after the V48 release)
def observed_v48(draws: pd.DataFrame, pools: pd.DataFrame | None, cls: ClassSource,
                 v48: list[str]) -> pd.DataFrame:
    vd = draws[draws.partition == "v48_bank"]
    local = build_prefix_pools(vd, v48, needed_configs(), "V48")
    if pools is not None:
        vp = pools[(pools.panel == "V48") & pools.config_id.isin(needed_configs())]
        have = set(zip(vp.item_id, vp.config_id))
        # crossover arms (2x1, 1x2, ...) may be absent from WP-A pools: fill them by the same prefix rule
        extra = local[[(i, c) not in have for i, c in zip(local.item_id, local.config_id)]]
        vp = pd.concat([vp, extra], ignore_index=True)
    else:
        vp = local
    return observed_pool_table(vp, vd, cls)


def score(run_root: Path | str | None = None, *, draws: pd.DataFrame | None = None, pools=None, classes=None,
          v48: list[str] | None = None, pred_dir: Path | str | None = None, out_dir: Path | str | None = None,
          observed: pd.DataFrame | None = None, write: bool = True) -> dict:
    """Score sealed V48 predictions against released V48 outcomes."""
    pred_dir = Path(pred_dir) if pred_dir is not None else Path(run_root) / "sealed" / "v48"
    pred = pd.read_parquet(pred_dir / "predictions.parquet")
    cross = pd.read_parquet(pred_dir / "crossover.parquet")
    pop = pd.read_parquet(pred_dir / "population.parquet")
    if observed is None:
        if draws is None:
            draws, pools, classes = load_analysis_inputs(Path(run_root))
        if v48 is None:
            v48 = list(load_panels(Path(run_root))["V48"])
        observed = observed_v48(draws, pools, ClassSource(draws=draws, table=classes), v48)
    obs = observed[["item_id", "config_id", "coverage", "plurality"]]
    res = score_frames(pred, cross, pop, obs)
    if write:
        out = Path(out_dir) if out_dir is not None else Path(run_root) / "scores" / "e2"
        out.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            if isinstance(v, pd.DataFrame):
                v.to_parquet(out / f"{k}.parquet", index=False)
    return res


def _brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _logscore(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def score_frames(pred: pd.DataFrame, cross: pd.DataFrame, pop: pd.DataFrame, obs: pd.DataFrame) -> dict:
    m = pred.merge(obs, on=["item_id", "config_id"], how="inner")
    rows = []
    for (model, cid), g in m.groupby(["model", "config_id"]):
        row = {"model": model, "config_id": cid, "n_items": int(len(g))}
        for q, col in (("plurality", "pred_plurality"), ("coverage", "pred_coverage")):
            k = g[~g[q].isna()]
            y, p = k[q].to_numpy(float), k[col].to_numpy(float)
            row[f"{q}_n"] = int(len(k))
            row[f"{q}_brier"] = _brier(y, p) if len(k) else np.nan
            row[f"{q}_logscore"] = _logscore(y, p) if len(k) else np.nan
            obs_macro = np.mean([k[k.benchmark == b][q].mean() for b in BENCHMARKS if (k.benchmark == b).any()])
            pred_macro = np.mean([k[k.benchmark == b][col].mean() for b in BENCHMARKS if (k.benchmark == b).any()])
            row[f"{q}_obs_macro"] = float(obs_macro)
            row[f"{q}_pred_macro"] = float(pred_macro)
            row[f"{q}_pop_error"] = float(pred_macro - obs_macro)
            pp = pop[(pop.model == model) & (pop.config_id == cid) & (pop.quantity == q) & (pop.scope == "macro")]
            if len(pp):
                row[f"{q}_pop_pi_covered"] = bool(pp.pi_lo.iloc[0] <= obs_macro <= pp.pi_hi.iloc[0])
        rows.append(row)
    by_config = pd.DataFrame(rows)
    # overall per model over the 9 menu configurations
    menu = m[m.in_menu & ~m.plurality.isna()]
    overall = []
    for model, g in menu.groupby("model"):
        overall.append({"model": model, "plurality_brier": _brier(g.plurality, g.pred_plurality),
                        "plurality_logscore": _logscore(g.plurality, g.pred_plurality),
                        "coverage_brier": _brier(g.coverage.dropna(), g.loc[g.coverage.notna(), "pred_coverage"]),
                        "ranking_accuracy": _ranking_accuracy(g),
                        "pop_pi_coverage_rate": float(by_config[(by_config.model == model) & by_config.config_id.isin(
                            list(CONFIGS_16A + CONFIGS_8A))].plurality_pop_pi_covered.mean())})
    overall = pd.DataFrame(overall)
    # crossover residuals
    cov = obs.set_index(["item_id", "config_id"]).coverage
    crow = []
    for r in cross.itertuples(index=False):
        d = cov.get((r.item_id, config_id(1, r.k)), np.nan)
        w = cov.get((r.item_id, config_id(r.k, 1)), np.nan)
        od = d - w if not (np.isnan(d) or np.isnan(w)) else np.nan
        crow.append({**r._asdict(), "obs_deep": d, "obs_wide_cov": w, "obs_diff": od,
                     "residual": od - r.pred_diff if not np.isnan(od) else np.nan,
                     "sign_agree": (np.sign(od) == r.pred_sign) if not np.isnan(od) else np.nan})
    crossover = pd.DataFrame(crow)
    return {"by_config": by_config, "overall": overall, "crossover": crossover}


def _ranking_accuracy(g: pd.DataFrame) -> float:
    """Paired ranking accuracy: over item x budget x config pairs whose observed plurality differs, the fraction
    where the predicted order agrees (predicted ties count 1/2)."""
    hits, tot = 0.0, 0
    for (_, budget), h in g.groupby(["item_id", "budget"]):
        recs = list(zip(h.plurality, h.pred_plurality))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                (oi, pi_), (oj, pj) = recs[i], recs[j]
                if oi == oj:
                    continue
                tot += 1
                if pi_ == pj:
                    hits += 0.5
                elif (oi > oj) == (pi_ > pj):
                    hits += 1
    return hits / tot if tot else float("nan")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI for the conductor
    import argparse
    ap = argparse.ArgumentParser(prog="python -m agent_allocation.passk.models.e2")
    ap.add_argument("stage", choices=["fit", "score"])
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args(argv)
    if args.stage == "fit":
        fit_and_predict(args.run_root)
    else:
        score(args.run_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
