"""Generation versus selection: pass@k at budget A against every allocation of the same compute.

Every configuration is decomposed into
  * generation  -- coverage: does its pool contain a correct answer? Two pools per configuration:
                   ``current`` (the candidates its selector actually chooses among) and ``archive`` (every
                   candidate it ever generated, e.g. all rounds of a team);
  * selection   -- delivered accuracy, and the gap coverage - delivered.
The reference is independent sampling at A = 1024 with the same total compute: pass@k (combinatorial over
the 25-sample bank) and the realized prefix coverage of root[0:k].

Beyond the sealed readouts it evaluates, on candidates the run already paid for (no new generation):
  * a verifier sweep: expected accuracy of choosing from each pool with a scorer of pairwise AUC a;
  * portfolios of deep calls at different depths, read out deepest-first or by vote;
  * accuracy by item difficulty (how many of the 25 bank samples were correct);
  * vote calibration (plurality share) and answer diversity of the bank;
  * round dynamics of revising trajectories; width against depth inside teams; scaling slopes.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

import core as c

RUNGS = [2, 4, 8, 16, 25]
AUCS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
BINS = [(0, 0, "0 of 25"), (1, 5, "1-5"), (6, 15, "6-15"), (16, 25, "16-25")]


# ---------------------------------------------------------------- loading
def policy(run):
    po = c.T(run, "policy_outcomes")
    po = po[po["partition_id"] == "main"].copy()
    po["B"] = po["team_size"].astype(int)
    po["R"] = po["total_cycles"].astype(int)
    po["C"] = po["B"] * po["R"] * po["reasoning_budget"].astype(int)
    po["rung"] = po["C"] // c.A0
    return po


def level_mean(df, col, level):
    """Equal-benchmark macro (or one benchmark) mean of an item-level column."""
    if level != "macro":
        return float(df[df["benchmark"] == level][col].mean())
    return float(np.mean([df[df["benchmark"] == b][col].mean() for b in c.BENCH_ORDER]))


def answers(run_root):
    """Per main-panel candidate: item, kind, slot, round, budget, correct (grade) and an answer class key.
    Every graded-correct answer is one class (they all equal the gold); wrong answers are grouped by the
    harness's exact vote key, which can split mathematically equivalent wrong answers (a small bias toward
    the correct class in plurality)."""
    from agent_allocation.contracts import Candidate, PublicTask, load_record
    from agent_allocation.selection.keys import vote_key
    cand = c.T(os.path.join(run_root, "merged"), "candidates")
    cand = cand[cand["partition_id"] == "main"][["node_id", "item_id", "benchmark", "kind", "root_slot", "team_size",
                                                 "update_round", "reasoning_budget", "correct"]]
    tasks = {}
    for s in ("mancer", "parasail"):
        for f in glob.glob(os.path.join(run_root, "shards", s, "manifests", "public", "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                tasks[r["item_id"]] = load_record(PublicTask, r)
    keys = {}
    for s in ("mancer", "parasail"):
        for f in glob.glob(os.path.join(run_root, "shards", s, "parsed", "*", "*.json")):
            d = json.load(open(f))
            cd = load_record(Candidate, d["candidate"])
            if cd.item_id not in tasks:
                continue
            k = vote_key(cd, tasks[cd.item_id])
            keys[cd.node_id] = None if k is None else k.exact
    cand["key"] = cand["node_id"].map(keys)
    cand["cls"] = np.where(cand["correct"] == True, "__CORRECT__", cand["key"])  # noqa: E712
    return cand


# ---------------------------------------------------------------- verifier model
@lru_cache(maxsize=None)
def p_select_correct(n: int, k: int, auc: float, draws: int = 40000) -> float:
    """P(the top-scored of n candidates is correct) when k are correct and a scorer separates correct from wrong
    with pairwise AUC ``auc`` (equal-variance Gaussian scores, d' = sqrt(2) * Phi^-1(auc))."""
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if auc >= 0.9999:
        return 1.0
    if auc <= 0.5:
        return k / n
    d = np.sqrt(2) * norm.ppf(auc)
    rng = np.random.default_rng(1000 * n + k)
    good = rng.normal(d, 1, size=(draws, k)).max(axis=1)
    bad = rng.normal(0, 1, size=(draws, n - k)).max(axis=1)
    return float((good > bad).mean())


def verifier_curve(po, name, view, auc):
    g = po[(po["name"] == name) & (po["view"] == view)].dropna(subset=["n_pool_correct"])
    vals = [p_select_correct(int(n), int(k), auc) for n, k in zip(g["n_pool"], g["n_pool_correct"])]
    g = g.assign(v=vals)
    return g


# ---------------------------------------------------------------- the whole analysis
def run(run_root, N, out, F):
    merged = os.path.join(run_root, "merged")
    po = policy(merged)
    cur = po[po["view"] == "current"]
    arc = po[po["view"] == "archive"]
    sb = c.T(merged, "sample_bank_items")
    sb = sb[sb["partition_id"] == "main"]
    G: dict = {}

    # ---- 1. per-rung comparison table: pass@k(A) against every configuration of the same compute
    rows = []
    for level in ["macro", *c.BENCH_ORDER]:
        for rung in sorted(cur["rung"].unique()):
            k = int(rung)
            sbk = sb[sb["k"] == k]
            pk = level_mean(sbk, "oracle_pass_at_k", level) if len(sbk) else np.nan
            pc = level_mean(sbk, "prefix_coverage", level) if len(sbk) else np.nan
            for name, g in cur[cur["rung"] == rung].groupby("name"):
                a = arc[arc["name"] == name]
                rows.append(dict(level=level, rung=k, name=name, family=g["family"].iloc[0],
                                 B=int(g["B"].iloc[0]), R=int(g["R"].iloc[0]),
                                 delivered=level_mean(g, "y", level),
                                 cov_current=level_mean(g, "coverage", level),
                                 cov_archive=level_mean(a, "coverage", level),
                                 delivered_archive=level_mean(a, "y", level),
                                 n_pool_current=float(g["n_pool"].mean()), n_pool_archive=float(a["n_pool"].mean()),
                                 pass_at_k=pk, prefix_cov=pc))
    rung_tab = pd.DataFrame(rows)
    rung_tab["gap_current"] = rung_tab["cov_current"] - rung_tab["delivered"]
    rung_tab["gap_archive"] = rung_tab["cov_archive"] - rung_tab["delivered"]
    rung_tab["vs_pass_at_k"] = rung_tab["delivered"] - rung_tab["pass_at_k"]
    rung_tab["cov_vs_pass_at_k"] = rung_tab["cov_archive"] - rung_tab["pass_at_k"]
    rung_tab.to_csv(f"{out}/tables/rung_table.csv", index=False)

    def fam_at(level, rung, fam, col, how="max"):
        s = rung_tab[(rung_tab.level == level) & (rung_tab.rung == rung) & (rung_tab.family == fam)][col]
        return None if s.empty else float(s.max() if how == "max" else s.mean())

    G["rung"] = {}
    for level in ["macro", *c.BENCH_ORDER]:
        d = {}
        for rung in RUNGS:
            sub = rung_tab[(rung_tab.level == level) & (rung_tab.rung == rung)]
            d[rung] = {"pass_at_k": sub["pass_at_k"].iloc[0], "vote": fam_at(level, rung, "sample", "delivered"),
                       "n_configs": int(len(sub)),
                       "n_beat_pass_at_k": int((sub["delivered"] > sub["pass_at_k"]).sum()),
                       "n_cov_above_pass_at_k": int((sub["cov_archive"] > sub["pass_at_k"]).sum()),
                       "best_non_deep": float(sub[sub.family != "deep"]["delivered"].max()),
                       "best_non_deep_name": sub[sub.family != "deep"].sort_values("delivered").iloc[-1]["name"],
                       "best_cov_archive": float(sub[sub.family != "deep"]["cov_archive"].max()),
                       "best_cov_archive_name": sub[sub.family != "deep"].sort_values("cov_archive").iloc[-1]["name"],
                       "deep": fam_at(level, rung, "deep", "delivered"),
                       "private_single": fam_at(level, rung, "private_single", "delivered"),
                       "private_single_cov": fam_at(level, rung, "private_single", "cov_archive"),
                       "teams_best": max([v for v in (fam_at(level, rung, "peer", "delivered"),
                                                      fam_at(level, rung, "private_team", "delivered")) if v is not None],
                                         default=None)}
            d[rung] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d[rung].items()}
        G["rung"][level] = d

    # ---- 2. family-level decomposition across budgets (macro), archive vs current
    fam_rows = []
    for (level, fam, rung), g in rung_tab.groupby(["level", "family", "rung"]):
        fam_rows.append(dict(level=level, family=fam, rung=rung, delivered=g["delivered"].max(),
                             delivered_mean=g["delivered"].mean(), cov_archive=g["cov_archive"].max(),
                             cov_current=g["cov_current"].max(), pass_at_k=g["pass_at_k"].iloc[0]))
    fam_tab = pd.DataFrame(fam_rows)
    fam_tab.to_csv(f"{out}/tables/family_decomposition.csv", index=False)
    # capture of the available ceiling: (delivered - pass@1) / (cov_archive - pass@1), macro, per family mean over rungs>=4
    p1 = N["acc"]["macro"]["sample/1"]["acc"]
    cap = {}
    for fam, g in rung_tab[(rung_tab.level == "macro") & (rung_tab.rung >= 4) & (rung_tab.family != "deep")].groupby("family"):
        den = (g["cov_archive"] - p1)
        ok = den > 0.02
        cap[fam] = {"capture_mean": round(float(((g["delivered"] - p1)[ok] / den[ok]).mean()), 3),
                    "gap_archive_mean_pts": round(100 * float(g["gap_archive"].mean()), 1),
                    "gap_current_mean_pts": round(100 * float(g["gap_current"].mean()), 1),
                    "cov_archive_minus_passk_mean_pts": round(100 * float(g["cov_vs_pass_at_k"].mean()), 1)}
    G["capture"] = cap

    # ---- 3. verifier sweep: what selection alone could recover from each pool
    vs = []
    for rung in RUNGS:
        for name, fam, view in ((f"sample/{rung}", "sample", "current"),
                                (f"private_single/{rung}", "private_single", "archive")):
            for auc in AUCS:
                g = verifier_curve(cur if view == "current" else arc, name, view, auc)
                vs.append(dict(rung=rung, config=name, family=fam, view=view, auc=auc,
                               macro=level_mean(g, "v", "macro"),
                               **{b: level_mean(g, "v", b) for b in c.BENCH_ORDER}))
        teams = rung_tab[(rung_tab.level == "macro") & (rung_tab.rung == rung) &
                         rung_tab.family.isin(["peer", "private_team"])]
        if len(teams):
            best = teams.sort_values("cov_archive").iloc[-1]["name"]
            for auc in AUCS:
                g = verifier_curve(arc, best, "archive", auc)
                vs.append(dict(rung=rung, config=best, family="team", view="archive", auc=auc,
                               macro=level_mean(g, "v", "macro"),
                               **{b: level_mean(g, "v", b) for b in c.BENCH_ORDER}))
    vs = pd.DataFrame(vs)
    vs.to_csv(f"{out}/tables/verifier_sweep.csv", index=False)
    G["verifier"] = {}
    for rung in RUNGS:
        deep = N["acc"]["macro"][f"deep/{rung}"]["acc"]
        e = {"deep": deep}
        for fam in ("sample", "private_single", "team"):
            g = vs[(vs.rung == rung) & (vs.family == fam)].sort_values("auc")
            if g.empty:
                continue
            e[fam] = {str(a): round(v, 3) for a, v in zip(g["auc"], g["macro"])}
            e[fam + "_config"] = g["config"].iloc[0]
            for a_, tag in ((0.5, "random"), (0.7, "auc07"), (0.9, "auc09"), (1.0, "perfect")):
                v = g[g["auc"] == a_]["macro"]
                e[f"{fam}_{tag}"] = round(float(v.iloc[0]), 3) if len(v) else None
            reach = g[g["macro"] >= deep]
            e[fam + "_auc_to_match_deep"] = None if reach.empty else float(reach["auc"].min())
        G["verifier"][rung] = e
    # per benchmark: does a perfect verifier on the sample bank reach deep?
    G["perfect_verifier_vs_deep"] = {b: {r: round(100 * (float(vs[(vs.rung == r) & (vs.family == "sample") & (vs.auc == 1.0)][b].iloc[0])
                                                          - N["acc"][b][f"deep/{r}"]["acc"]), 1) for r in RUNGS}
                                      for b in ["macro", *c.BENCH_ORDER]}

    # ---- 4. difficulty strata by bank success count
    bank25 = sb[sb["k"] == 25][["item_id", "benchmark", "bank_correct", "bank_size"]].copy()
    bank25["stratum_d"] = pd.cut(bank25["bank_correct"], [-1, 0, 5, 15, 25], labels=[b[2] for b in BINS])
    y = cur.pivot_table(index="item_id", columns="name", values="y", aggfunc="first")
    strat = bank25.merge(y, left_on="item_id", right_index=True, how="left")
    configs = ["sample/25", "private_single/25", "peer/5/4", "private_team/5/4", "deep/4", "deep/8", "deep/16", "deep/25"]
    st = []
    for lab, g in strat.groupby("stratum_d", observed=False):
        row = {"stratum": str(lab), "n": int(len(g)),
               "by_bench": {b: int((g["benchmark"] == b).sum()) for b in c.BENCH_ORDER}}
        for cfg in configs:
            row[cfg] = round(float(g[cfg].mean()), 3) if len(g) else None
        st.append(row)
    G["strata"] = st
    # items no sample reached: how many does depth solve?
    zero = strat[strat["bank_correct"] == 0]
    G["bank_zero"] = {"n": int(len(zero)), **{f"deep{m}_solved": int((zero[f"deep/{m}"] == 1).sum()) for m in (4, 8, 16, 25)},
                      "any_deep_solved": int((zero[[f"deep/{m}" for m in (2, 4, 8, 16, 25)]] == 1).any(axis=1).sum()),
                      "private25_solved": int((zero["private_single/25"] == 1).sum()),
                      "by_bench": {b: int((zero["benchmark"] == b).sum()) for b in c.BENCH_ORDER}}
    # 2x2: deep/16 against coverage of the first 16 bank samples
    cov16 = sb[sb["k"] == 16].set_index("item_id")["prefix_coverage"]
    d16 = y["deep/16"]
    j = pd.concat([cov16, d16], axis=1, keys=["cov", "deep"]).dropna()
    cv, dp = j["cov"].astype(float), j["deep"].astype(float)
    G["overlap16"] = {"both": int(((cv == 1) & (dp == 1)).sum()), "deep_only": int(((cv == 0) & (dp == 1)).sum()),
                      "bank_only": int(((cv == 1) & (dp == 0)).sum()), "neither": int(((cv == 0) & (dp == 0)).sum()),
                      "n": int(len(j))}
    strat.to_csv(f"{out}/tables/item_strata.csv", index=False)

    # ---- 5. answer-level analyses (vote calibration, diversity, deep portfolios)
    ans = answers(run_root)
    roots = ans[ans["kind"] == "root"]
    cal = []
    div = []
    for item, g in roots.groupby("item_id"):
        g = g.sort_values("root_slot")
        votes = Counter(x for x in g["cls"] if isinstance(x, str))
        if not votes:
            continue
        top, top_n = votes.most_common(1)[0]
        cal.append(dict(item_id=item, benchmark=g["benchmark"].iloc[0], share=top_n / 25,
                        top_correct=top == "__CORRECT__", n_classes=len(votes),
                        any_correct="__CORRECT__" in votes))
    cal = pd.DataFrame(cal)
    cal["deep16"] = cal["item_id"].map(y["deep/16"])
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    cal["share_bin"] = pd.cut(cal["share"], bins, labels=["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"], right=False)
    G["calibration"] = [{"bin": str(b), "n": int(len(g)), "plurality_correct": round(float(g["top_correct"].mean()), 3) if len(g) else None,
                         "deep16": round(float(g["deep16"].mean()), 3) if len(g) else None}
                        for b, g in cal.groupby("share_bin", observed=False)]
    dv = cal.assign(ncls=pd.cut(cal["n_classes"], [0, 1, 3, 7, 26], labels=["1", "2-3", "4-7", "8+"]))
    G["diversity"] = [{"classes": str(b), "n": int(len(g)), "vote25": round(float(g["top_correct"].mean()), 3) if len(g) else None,
                       "coverage25": round(float(g["any_correct"].mean()), 3) if len(g) else None,
                       "deep16": round(float(g["deep16"].mean()), 3) if len(g) else None}
                      for b, g in dv.groupby("ncls", observed=False)]
    G["classes_median"] = float(cal["n_classes"].median())
    cal.to_csv(f"{out}/tables/vote_calibration.csv", index=False)

    # portfolios of deep calls at different depths (one call per depth per item; root[0] is depth 1)
    deep = ans[ans["kind"] == "deep"].copy()
    deep["m"] = deep["reasoning_budget"] // c.A0
    r0 = roots[roots["root_slot"] == 0].assign(m=1)
    port = pd.concat([deep, r0])[["item_id", "benchmark", "m", "cls", "correct"]]
    by_item = {it: {int(r.m): (r.cls, r.correct) for r in g.itertuples()} for it, g in port.groupby("item_id")}
    bank_first = {it: g.sort_values("root_slot") for it, g in roots.groupby("item_id")}
    portfolios = [(16, 8), (16, 4), (16, 8, 1), (16, 4, 4), (8, 4, 2, 1), (16, 8, 4), (16, 8, 1), (25,), (16,), (8, 8),
                  (25, 16, 8, 4, 2, 1), (16, 4, 2, 1), (8, 4, 2), (4, 2, 1)]
    seen = set()
    prow = []
    for P in portfolios:
        if P in seen or len(set(P)) != len(P):
            continue
        seen.add(P)
        C = sum(P)
        res = {"cov": [], "deepest": [], "vote": [], "bench": []}
        for it, d in by_item.items():
            if not all(m in d for m in P):
                continue
            cls = [d[m][0] for m in P]
            cor = [d[m][1] for m in P]
            if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in cor):
                continue
            res["cov"].append(float(any(cor)))
            res["deepest"].append(float(cor[int(np.argmax(P))]))
            cnt = Counter(x for x in cls if isinstance(x, str))
            if cnt:
                best_n = max(cnt.values())
                tied = [k for k, v in cnt.items() if v == best_n]
                pick = next(cl for m, cl in sorted(zip(P, cls), reverse=True) if cl in tied)
                res["vote"].append(float(pick == "__CORRECT__"))
            else:
                res["vote"].append(0.0)
            res["bench"].append(it.split("/")[0])
        dfp = pd.DataFrame({"cov": res["cov"], "deepest": res["deepest"], "vote": res["vote"], "benchmark": res["bench"]})
        prow.append(dict(portfolio="+".join(map(str, P)), C=C, n=len(dfp),
                         cov=level_mean(dfp, "cov", "macro"), deepest=level_mean(dfp, "deepest", "macro"),
                         vote=level_mean(dfp, "vote", "macro"),
                         pass_at_C=(level_mean(sb[sb["k"] == C], "oracle_pass_at_k", "macro") if C <= 25 else None),
                         vote_at_C=(level_mean(sb[sb["k"] == C], "readout", "macro") if C <= 25 else None)))
    prt = pd.DataFrame(prow).sort_values("C")
    prt.to_csv(f"{out}/tables/deep_portfolios.csv", index=False)
    G["portfolios"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in prow]

    # ---- 6. round dynamics of revising trajectories
    ts = c.T(merged, "transitions_summary")
    ts = ts[(ts["partition_id"] == "main") & (ts["level"] == "macro")]
    G["transitions"] = {}
    for traj, g in ts.groupby("trajectory"):
        g = g.sort_values("from_step")
        G["transitions"][traj] = [{"step": int(r.from_step), "W_to_C": round(float(r.W_to_C), 3),
                                   "C_to_W": round(float(r.C_to_W), 3), "net": round(float(r.W_to_C - r.C_to_W), 3),
                                   "correct_next": round(float(r.correct_next), 3)} for r in g.itertuples()]
    ts.to_csv(f"{out}/tables/transitions_macro.csv", index=False)

    # ---- 7. width against depth inside teams at fixed compute
    wd = []
    for fam in ("peer", "private_team"):
        for n in (2, 3, 4, 5):
            for r in (0, 1, 2, 3, 4):
                name = f"{fam}/{n}/{r}"
                if name in F.col:
                    wd.append(dict(family=fam, n=n, r=r, C=n * (1 + r), acc=N["acc"]["macro"][name]["acc"]))
    wd = pd.DataFrame(wd)
    G["team_shape"] = {}
    for C in sorted(set(wd["C"])):
        g = wd[wd["C"] == C]
        if g["n"].nunique() > 1:
            G["team_shape"][int(C)] = [{"name": f"{r.family}/{r.n}/{r.r}", "acc": r.acc} for r in g.itertuples()]
    wd.to_csv(f"{out}/tables/team_shape.csv", index=False)

    # ---- 8. scaling slopes per family: accuracy points per doubling of C (macro), bootstrap over items
    slopes = {}
    fam_names = {"sample": [f"sample/{k}" for k in (1, 2, 4, 8, 16)],
                 "private_single": [f"private_single/{k}" for k in (1, 2, 4, 8, 16)],
                 "deep": ["sample/1", *[f"deep/{m}" for m in (2, 4, 8, 16)]]}
    x = np.log2([1, 2, 4, 8, 16])
    for level in ["macro", *c.BENCH_ORDER]:
        slopes[level] = {}
        for fam, names in fam_names.items():
            js = [F.col[n] for n in names]
            w = F.levels[level]
            yb = np.stack([c.wboot(F.boot, w, j) for j in js], axis=1)       # (B, 5)
            yp = np.array([c.wpoint(F.mat, w, j) for j in js])
            b = np.polyfit(x, yp, 1)[0]
            bb = np.array([np.polyfit(x, row, 1)[0] for row in yb[::10]])
            slopes[level][fam] = {"pts_per_doubling": round(100 * b, 1),
                                  "lo": round(100 * float(np.quantile(bb, 0.025)), 1),
                                  "hi": round(100 * float(np.quantile(bb, 0.975)), 1)}
        # pass@k slope for reference
        pk = [N["oracle"][level][k] for k in (1, 2, 4, 8, 16)]
        slopes[level]["pass_at_k"] = {"pts_per_doubling": round(100 * np.polyfit(x, pk, 1)[0], 1)}
    G["slopes"] = slopes

    return G, rung_tab, fam_tab, vs, prt, cal, strat
