"""Model-fitting wrappers for the analysis notebooks.

Everything here is a thin, opinionated wrapper implementing the pre-registered choices
from the analysis plan: item-level FE logit with cluster-robust SEs for accuracy,
fractional logit for ECE levels, Type II/III ANOVA + partial eta^2 and GBM + Friedman H2
for interactions, the Monte-Carlo vote model, and the unified recipe engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[1]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
from fitting import power_law_fit as _power_law_fit  # noqa: E402  (analysis/fitting.py)

from . import boot as _boot  # noqa: E402


# ---------------------------------------------------------------- scaling fits

def power_law(x, y) -> dict:
    """analysis/fitting.py::power_law_fit with an explicit dropped-points report.

    Fit only decaying positive quantities (error rate, ECE) — never delta_* (can be
    negative; silently dropping negatives changes the meaning of the fit).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n_nonpos = int(np.sum((x[ok] <= 0) | (y[ok] <= 0)))
    if n_nonpos > 0.1 * ok.sum():
        raise ValueError(
            f"{n_nonpos}/{ok.sum()} nonpositive points would be silently dropped — "
            "power_law is for strictly positive decaying quantities (error, ECE)")
    res = _power_law_fit(x[ok], y[ok])
    res["n_dropped"] = int(ok.sum() - res["n"])
    return res


# ------------------------------------------------------- accuracy: FE logit

FE_LOGIT_FORMULA = (
    "correct ~ C(model_size, Treatment('0.6B'))"
    " + C(reasoning_level, Treatment('off'))"
    " + C(topology, Treatment('independent'))"
    " + C(prompt_complexity_level, Treatment(0))"
    " + C(qid)"
)


def fe_logit(items: pd.DataFrame, formula: str = FE_LOGIT_FORMULA):
    """Item-level logistic with question fixed effects, cluster-robust by design cell.

    Call per benchmark on the core grid. Degenerate questions (answered all-correct or
    all-wrong across every config) carry no FE-conditional information and cause perfect
    separation — they are dropped first. Returns (result, n_dropped_qids, frame).
    """
    import statsmodels.formula.api as smf

    d = items.copy()
    d["correct"] = d["correct"].astype(float)
    solve = d.groupby("qid")["correct"].mean()
    degenerate = solve[(solve == 0) | (solve == 1)].index
    d = d[~d["qid"].isin(degenerate)].copy()
    for c in ("model_size", "reasoning_level", "topology", "qid"):
        if c in d.columns:
            d[c] = d[c].astype(str)  # plain strings: patsy + Treatment coding
    model = smf.logit(formula, data=d)
    try:
        res = model.fit(disp=0, method="lbfgs", maxiter=500,
                        cov_type="cluster", cov_kwds={"groups": d["design_cell"]})
    except Exception:
        import statsmodels.api as sm
        res = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["design_cell"]})
    return res, len(degenerate), d


def tidy_coefs(res, drop_pattern: str = "C(qid)") -> pd.DataFrame:
    """Coefficient table (log-odds + OR + 95% CI), dropping nuisance FE terms."""
    ci = res.conf_int()
    out = pd.DataFrame({"coef": res.params, "lo": ci[0], "hi": ci[1],
                        "p": res.pvalues})
    out = out[~out.index.str.contains(drop_pattern, regex=False)]
    out["OR"] = np.exp(out["coef"])
    return out


# ------------------------------------------------- calibration: fractional logit

def frac_logit(cells: pd.DataFrame, target: str, rhs: str):
    """Papke-Wooldridge fractional logit for bounded outcomes (ECE, Brier reliability).

    Unweighted QMLE with cluster-robust SEs by design cell (n_questions is nearly
    constant post-dedupe; variance weights would only add a false precision story).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = cells.dropna(subset=[target]).copy()
    d[target] = d[target].clip(1e-6, 1 - 1e-6)
    for c in ("model_size", "reasoning_level", "topology", "context_share_level"):
        if c in d.columns:
            d[c] = d[c].astype(str)
    return smf.glm(f"{target} ~ {rhs}", data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["design_cell"]})


def dl_meta(effects: np.ndarray, variances: np.ndarray) -> dict:
    """DerSimonian-Laird pooling — DESCRIPTIVE heterogeneity summary only.

    Cells within a benchmark share questions, so the independence assumption fails;
    headline CIs come from boot.HierBoot, never from this.
    """
    from statsmodels.stats.meta_analysis import combine_effects

    ok = np.isfinite(effects) & np.isfinite(variances) & (variances > 0)
    if ok.sum() < 3:
        return {"pooled": np.nan, "tau2": np.nan, "n": int(ok.sum())}
    res = combine_effects(effects[ok], variances[ok], method_re="dl")
    # statsmodels does NOT truncate tau^2 at 0; a negative tau^2 yields negative RE
    # weights and garbage pooled estimates. Truncate and pool manually.
    tau2 = max(0.0, float(res.tau2))
    w = 1.0 / (variances[ok] + tau2)
    pooled = float(np.sum(w * effects[ok]) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return {"pooled": pooled, "ci_low": pooled - 1.96 * se,
            "ci_upp": pooled + 1.96 * se, "tau2": tau2, "n": int(ok.sum())}


# ------------------------------------------------------------ interactions

def anova_eta2(cells: pd.DataFrame, target: str, factors: list[str],
               typ: int = 2, logit_transform: bool = False) -> pd.DataFrame:
    """OLS with all two-way interactions -> Type II (default) ANOVA + partial eta^2.

    Coverage is unbalanced: rerun with typ=3 (sum contrasts) and compare; disagreement
    means neither is trustworthy over the GBM check.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = cells.dropna(subset=[target]).copy()
    y = target
    if logit_transform:
        n = d["n_questions"].astype(float)
        p = d[target].clip(1 / (2 * n), 1 - 1 / (2 * n))
        d["_y"] = np.log(p / (1 - p))
        y = "_y"
    contrast = ", Sum" if typ == 3 else ""
    terms = " + ".join(f"C({f}{contrast})" for f in factors)
    ols = smf.ols(f"{y} ~ ({terms})**2", data=d.astype({f: str for f in factors})).fit()
    aov = sm.stats.anova_lm(ols, typ=typ)
    aov["partial_eta_sq"] = aov["sum_sq"] / (aov["sum_sq"] + aov.loc["Residual", "sum_sq"])
    return aov


def gbm_interactions(cells: pd.DataFrame, target: str,
                     features: list[str] | None = None,
                     n_splits: int = 5, seed: int = 0) -> dict:
    """Nonparametric interaction check: HistGB + GroupKFold + permutation importance.

    GroupKFold by design_cell — seeds of one config never span train/test. Returns the
    fitted model (on all data), per-fold held-out R^2, and aggregated importances.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import OrdinalEncoder

    if features is None:
        features = ["model_size", "topology", "context_share_level",
                    "prompt_complexity_level", "reasoning_level", "benchmark",
                    "log2_params", "log_reas_tok"]
    cat = [f for f in features if not pd.api.types.is_numeric_dtype(cells[f])
           or f == "prompt_complexity_level"]
    d = cells.dropna(subset=[target]).copy()
    enc = OrdinalEncoder()
    X = d[features].copy()
    X[cat] = enc.fit_transform(X[cat].astype(str))
    X = X.to_numpy(float)
    y = d[target].to_numpy(float)
    groups = d["design_cell"].to_numpy()
    cat_mask = [f in cat for f in features]

    def make():
        return HistGradientBoostingRegressor(
            categorical_features=cat_mask, random_state=seed, max_iter=300)

    r2, imps = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = make().fit(X[tr], y[tr])
        r2.append(m.score(X[te], y[te]))
        pi = permutation_importance(m, X[te], y[te], n_repeats=30, random_state=seed)
        imps.append(pi.importances_mean)
    model = make().fit(X, y)
    importances = pd.DataFrame(
        {"importance": np.mean(imps, axis=0), "std": np.std(imps, axis=0)},
        index=features).sort_values("importance", ascending=False)
    return {"model": model, "X": X, "features": features, "cat_mask": cat_mask,
            "cv_r2": np.array(r2), "importances": importances}


def friedman_h2(model, X: np.ndarray, features: list[str], j: int, k: int,
                grid_resolution: int = 12) -> float:
    """Friedman H^2 for one feature pair, hand-rolled from partial dependence.

    H^2_jk = sum(PD_jk - PD_j - PD_k)^2 / sum(PD_jk^2) over the joint PD grid (all PDs
    centered). sklearn has no built-in H statistic.
    """
    from sklearn.inspection import partial_dependence

    pd_jk = partial_dependence(model, X, [(j, k)], grid_resolution=grid_resolution,
                               kind="average")
    pd_j = partial_dependence(model, X, [j], grid_resolution=grid_resolution,
                              kind="average")
    pd_k = partial_dependence(model, X, [k], grid_resolution=grid_resolution,
                              kind="average")
    f_jk = pd_jk["average"][0]
    f_j = pd_j["average"][0]
    f_k = pd_k["average"][0]
    # Align 1-D grids with the joint grid (sklearn uses identical per-feature grids).
    f_jk = f_jk - f_jk.mean()
    f_j = (f_j - f_j.mean()).reshape(-1, 1)
    f_k = (f_k - f_k.mean()).reshape(1, -1)
    if f_jk.shape != (f_j.shape[0], f_k.shape[1]):
        return np.nan
    num = float(np.sum((f_jk - f_j - f_k) ** 2))
    den = float(np.sum(f_jk ** 2))
    return num / den if den > 0 else np.nan


# --------------------------------------------------------- vote model (H6)

def vote_mc(item_answers: list[list[str]], item_confs: list[list[float]],
            item_keys: list[str], n_agents: int = 3, B: int = 2000,
            seed: int = 0) -> float:
    """Monte-Carlo majority-vote accuracy under agent INDEPENDENCE.

    Per item: a pool of observed (answer, confidence) single draws (pooled across agents
    and seeds of `independent` cells). Each replicate samples n_agents draws with
    replacement and applies the harness vote rule — plurality, ties broken by summed
    confidence mass. The gap between this prediction and observed MAS accuracy splits
    "statistical ensembling" from coordination + error-correlation effects.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    total = 0
    for answers, confs, key in zip(item_answers, item_confs, item_keys):
        if not answers:
            continue
        answers = np.asarray(answers, dtype=object)
        confs = np.nan_to_num(np.asarray(confs, float), nan=0.0)
        pick = rng.integers(0, len(answers), size=(B, n_agents))
        for b in range(B):
            a = answers[pick[b]]
            c = confs[pick[b]]
            uniq, counts = np.unique(a, return_counts=True)
            top = counts == counts.max()
            if top.sum() == 1:
                winner = uniq[np.argmax(counts)]
            else:  # tie -> summed confidence mass among tied answers
                tied = uniq[top]
                mass = [c[a == t].sum() for t in tied]
                winner = tied[int(np.argmax(mass))]
            hits += winner == key
        total += B
    return hits / total if total else np.nan


def phi_error_correlation(agents: pd.DataFrame) -> float:
    """Mean pairwise correlation of agent error indicators across items.

    Feed one group (e.g. one independent-topology design cell, final round). Positive phi
    = agents share errors (homogenization) -> voting gains shrink below the independence
    prediction.
    """
    piv = agents.pivot_table(index="qid", columns="agent_id", values="correct",
                             aggfunc="first")
    piv = piv.dropna(axis=0)
    if piv.shape[1] < 2 or len(piv) < 10:
        return np.nan
    corr = piv.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    return float(np.nanmean(corr[iu]))


# ------------------------------------------------------------- recipes (T5)

def bh_fdr(pvals, alpha: float = 0.05):
    from statsmodels.stats.multitest import multipletests

    ok = np.isfinite(pvals)
    reject = np.zeros(len(pvals), bool)
    padj = np.full(len(pvals), np.nan)
    if ok.sum():
        reject[ok], padj[ok], _, _ = multipletests(np.asarray(pvals)[ok],
                                                   alpha=alpha, method="fdr_bh")
    return reject, padj


def recipe_leaderboard(hb: "_boot.HierBoot", value_col: str = "correct",
                       config_col: str = "design_cell") -> pd.DataFrame:
    """Bootstrap leaderboard over configs: rank distributions from the joint resample.

    Uses the SAME qid resample for every config (pairing preserved), so rank stability is
    measured against question sampling + seed noise. Returns per config: observed mean,
    median rank, 95% rank interval, P(top-3).
    """
    labels, obs, n, est = hb.mean_matrix(value_col, config_col)
    ranks = (-est).argsort(axis=1).argsort(axis=1) + 1  # 1 = best, per replicate
    return pd.DataFrame({
        config_col: labels, "mean": obs, "n_items": n,
        "median_rank": np.nanmedian(ranks, axis=0),
        "rank_lo": np.nanpercentile(ranks, 2.5, axis=0),
        "rank_hi": np.nanpercentile(ranks, 97.5, axis=0),
        "p_top3": (ranks <= 3).mean(axis=0),
    }).sort_values("p_top3", ascending=False).reset_index(drop=True)


def pareto_front(df: pd.DataFrame, cost_col: str = "cost",
                 value_col: str = "accuracy") -> pd.DataFrame:
    """Rows not dominated by any cheaper-and-better row (minimal cost, maximal value)."""
    d = df.dropna(subset=[cost_col, value_col]).sort_values(cost_col)
    best = -np.inf
    keep = []
    for _, row in d.iterrows():
        if row[value_col] > best:
            keep.append(row.name)
            best = row[value_col]
    return df.loc[keep]


def split_half_shrinkage(items: pd.DataFrame, config_col: str = "design_cell",
                         top_k: int = 5, n_splits: int = 20, seed: int = 0) -> dict:
    """Winner's-curse guard: select top-k configs on half the questions, re-estimate on
    the held-out half; report mean selected-half vs holdout-half accuracy gap."""
    rng = np.random.default_rng(seed)
    qids = items["qid"].unique()
    sel_means, hold_means = [], []
    for _ in range(n_splits):
        half = rng.permutation(qids)[: len(qids) // 2]
        a = items[items["qid"].isin(half)]
        b = items[~items["qid"].isin(half)]
        top = (a.groupby(config_col)["correct"].mean()
               .sort_values(ascending=False).head(top_k).index)
        sel_means.append(a[a[config_col].isin(top)]["correct"].mean())
        hold_means.append(b[b[config_col].isin(top)]["correct"].mean())
    sel, hold = float(np.mean(sel_means)), float(np.mean(hold_means))
    return {"selected_half": sel, "holdout_half": hold, "shrinkage": sel - hold}
