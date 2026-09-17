"""Selection: does the allocation shape change how well a pool can be read out, and what
would read it out better?

Two identities carry this section. Delivering a correct answer requires the pool to hold one, so
per configuration ``delivered = coverage x S`` exactly, where ``S`` is selection success, the
chance of delivering given the pool held a correct answer. And the vote can only rank distinct
answers, so the structure of the answer space bounds what any adjudication over it can do.
"""
from __future__ import annotations

import ast
import collections
import json
import os
import pickle

import numpy as np
import pandas as pd

from agent_allocation.analysis import stats as astats
import isocore as ic
import isoframe

CONC_BINS = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.999, 1.0]
CONC_LABEL = ["≤0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.00", "1.00"]
TOP_M = [1, 2, 3, 5, 10, 25]
TAUS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.01]


# ---------------------------------------------------------------- allocation and selection

def selection_success(run_dir: str, partition: str = "main"):
    """Coverage, concentration and selection success per configuration, informative benchmarks."""
    po = ic.load(run_dir, "policy_outcomes")
    po = ic.add_budget(po[(po["partition_id"] == partition) & (po["view"] == "current")])
    po = po[po["benchmark"].isin(ic.INFORMATIVE) & (po["n_pool"] >= 2)].copy()
    po["conc"] = po["n_pool_correct"] / po["n_pool"]
    g = po.groupby(["name", "family", "rung", "B", "R"]).agg(
        coverage=("coverage", "mean"), conc=("conc", "mean"), y=("y", "mean"),
        n_pool=("n_pool", "mean"), n=("y", "size")).reset_index()
    cov = po[po["coverage"] == 1].groupby("name")["y"].mean().rename("S")
    return g.merge(cov, on="name", how="left")


def success_by_concentration(run_dir: str, partition: str = "main"):
    """Selection success inside concentration bins, to see whether the family adds anything."""
    po = ic.load(run_dir, "policy_outcomes")
    po = ic.add_budget(po[(po["partition_id"] == partition) & (po["view"] == "current")])
    po = po[po["benchmark"].isin(ic.INFORMATIVE) & (po["n_pool"] >= 2) & (po["coverage"] == 1)].copy()
    po["conc"] = po["n_pool_correct"] / po["n_pool"]
    po["bin"] = pd.cut(po["conc"], CONC_BINS, labels=CONC_LABEL, include_lowest=True)
    return (po.groupby(["bin", "family"], observed=True)
            .agg(S=("y", "mean"), n=("y", "size")).reset_index())


def matched_pool(run_dir: str, partition: str = "main", alpha: float = 0.05):
    """Pools of the same size, read out differently: 25 iid samples against structured pools.

    The archive view of a 5-agent, 5-round team is also 25 candidates, so pool size is held fixed
    and only the way those candidates were produced differs.
    """
    po = ic.load(run_dir, "policy_outcomes")
    po = po[po["partition_id"] == partition]
    arms = [("sample/25", "current"), ("private_team/5/4", "archive"),
            ("private_single/25", "archive"), ("peer/5/4", "archive")]
    rows = []
    for name, view in arms:
        g = po[(po["name"] == name) & (po["view"] == view)
               & po["benchmark"].isin(ic.INFORMATIVE)]
        gc = g[g["coverage"] == 1]
        rows.append(dict(name=name, view=view, n_pool=float(g["n_pool"].mean()),
                         coverage=float(g["coverage"].mean()),
                         conc=float((g["n_pool_correct"] / g["n_pool"]).mean()),
                         y=float(g["y"].mean()), S=float(gc["y"].mean())))
    d = pd.DataFrame(rows)

    # Paired item bootstrap of each structured pool against the iid pool of the same size.
    cur = ic.add_budget(po[po["view"] == "current"])
    pop = astats.build_population(cur, [(partition, "sample/25")], view="current", label="matched")
    rs = pop.resampler(ic.REPLICATES, ic.STUDY_SEED)
    levels = isoframe._levels(pop.strata)
    pc = po[po["view"] == "current"].pivot_table(index="item_id", columns="name",
                                                 values="y", aggfunc="first")
    pa = po[po["view"] == "archive"].pivot_table(index="item_id", columns="name",
                                                values="y", aggfunc="first")
    con = []
    for name, view in arms[1:]:
        A = pa if view == "archive" else pc
        diff = {s: (A.reindex(pop.items[s])[name].to_numpy(float)
                    - pc.reindex(pop.items[s])["sample/25"].to_numpy(float))[:, None]
                for s in pop.strata}
        boot = isoframe._boot_means(diff, rs)
        for lv in ["macro3", "math500", "mmlu_pro", "mbpp_plus"]:
            w = levels[lv]
            est = isoframe._wpoint(diff, w, 0)
            db = isoframe._wboot(boot, w, 0)
            lo, hi = np.nanquantile(db, [alpha / 2, 1 - alpha / 2])
            con.append(dict(name=name, view=view, level=lv, estimate=est,
                            ci_lo=float(lo), ci_hi=float(hi)))
    return d, pd.DataFrame(con)


# ---------------------------------------------------------------- answer-space structure

def load_banks(run_dir: str, partition: str = "main", cache: str | None = None):
    """The 25-candidate bank per item with each candidate's final answer and correctness.

    Pool membership comes from the sealed policy manifest; answers come from the parsed records.
    """
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    banks = {}
    with open(f"{run_dir}/manifests/policies/{partition}.jsonl") as fh:
        for line in fh:
            if '"sample/25"' not in line:
                continue
            d = json.loads(line)
            if d.get("name") != "sample/25" or d.get("view") != "current":
                continue
            ids = d["candidate_ids"]
            banks[d["item_id"]] = ast.literal_eval(ids) if isinstance(ids, str) else ids
    ansmap = {}
    for ids in banks.values():
        for c in ids:
            if c in ansmap:
                continue
            with open(f"{run_dir}/parsed/{c[:2]}/{c}.json") as fh:
                rec = json.load(fh)["candidate"]
            ansmap[c] = (rec.get("final_answer"), rec.get("parse_status"),
                         len(rec.get("rationale") or ""))
    out = {"banks": banks, "ans": ansmap}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(out, open(cache, "wb"))
    return out


def answer_structure(run_dir: str, partition: str = "main", cache: str | None = None):
    """Per item: how many distinct answers, how concentrated, and where the correct one ranks."""
    D = load_banks(run_dir, partition, cache)
    banks, ansmap = D["banks"], D["ans"]
    cand = ic.load(run_dir, "candidates")[["node_id", "correct", "benchmark", "item_id"]]
    corr = dict(zip(cand["node_id"], cand["correct"]))
    bench = dict(zip(cand["item_id"], cand["benchmark"]))
    rows = []
    for item, ids in banks.items():
        v = [(ansmap[c][0], corr.get(c)) for c in ids
             if ansmap[c][1] == "valid" and ansmap[c][0] is not None]
        if not v:
            continue
        cnt = collections.Counter(x for x, _ in v)
        grp = {x: bool(y) for x, y in v}
        rk = sorted(cnt.items(), key=lambda kv: -kv[1])
        n = len(v)
        rank = next((i + 1 for i, (x, _) in enumerate(rk) if grp[x]), None)
        rows.append(dict(item_id=item, benchmark=bench[item], n_valid=n,
                         n_distinct=len(cnt), top_share=rk[0][1] / n,
                         margin=(rk[0][1] - (rk[1][1] if len(rk) > 1 else 0)) / n,
                         correct_rank=rank, has_correct=rank is not None,
                         plurality_ok=grp[rk[0][0]]))
    return pd.DataFrame(rows)


def topm_curve(struct: pd.DataFrame):
    """Accuracy from a perfect adjudication restricted to the m most-voted distinct answers."""
    out = []
    for bench, g in struct.groupby("benchmark"):
        for m in TOP_M:
            out.append(dict(benchmark=bench, m=m,
                            acc=float(((g["correct_rank"].notna())
                                       & (g["correct_rank"] <= m)).mean()),
                            plurality=float(g["plurality_ok"].mean()),
                            oracle=float(g["has_correct"].mean())))
    return pd.DataFrame(out)


def selective_verification(struct: pd.DataFrame, taus=TAUS):
    """Adjudicate the top two only when the plurality margin is below tau."""
    out = []
    for bench, g in struct.groupby("benchmark"):
        plur = float(g["plurality_ok"].mean())
        orc = float(g["has_correct"].mean())
        top2 = (g["correct_rank"].notna()) & (g["correct_rank"] <= 2)
        for tau in taus:
            sel = g["margin"] < tau
            acc = float(np.where(sel, top2, g["plurality_ok"]).mean())
            gap = orc - plur
            out.append(dict(benchmark=bench, tau=tau, verified=float(sel.mean()),
                            acc=acc, plurality=plur, oracle=orc,
                            recovered=float((acc - plur) / gap) if gap > 0 else np.nan))
    return pd.DataFrame(out)


def topm_ci(struct: pd.DataFrame, m: int = 2, reps: int = ic.REPLICATES,
            seed: int = ic.STUDY_SEED, alpha: float = 0.05):
    """Item bootstrap of top-m adjudication minus plurality, per benchmark."""
    rng = np.random.default_rng(seed)
    out = []
    for bench, g in struct.groupby("benchmark"):
        d = (((g["correct_rank"].notna()) & (g["correct_rank"] <= m)).astype(float)
             - g["plurality_ok"].astype(float)).to_numpy()
        bs = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
        out.append(dict(benchmark=bench, m=m, estimate=float(d.mean()),
                        ci_lo=float(np.quantile(bs, alpha / 2)),
                        ci_hi=float(np.quantile(bs, 1 - alpha / 2))))
    return pd.DataFrame(out)
