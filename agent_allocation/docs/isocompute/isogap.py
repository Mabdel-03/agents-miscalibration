"""Where the oracle gap actually is, and what would collapse it.

Every configuration satisfies, per item, ``delivered = coverage - selection loss``, where coverage
is 1 when the configuration's own candidate pool held a correct answer. Comparing that coverage to
independent sampling at the same budget splits the gap into a generation-side part (the pool never
held the answer) and a selection-side part (it did, and the selector missed it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, hypergeom

from agent_allocation.analysis import stats as astats
import isocore as ic
import isoframe

CONC_BINS = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.999, 1.0]
CONC_LABEL = ["≤0.20", "0.20-0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.00", "1.00"]
AUCS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0]
SIM_DRAWS = 6000


# ---------------------------------------------------------------- decomposition

def decomposition(run_dir: str, partition: str = "main", rungs=(2, 4, 6, 8, 10, 12, 16, 20, 25)):
    """Per level, per budget, per configuration: coverage deficit and selection loss."""
    po = ic.load(run_dir, "policy_outcomes")
    po = po[po["partition_id"] == partition]
    cur = ic.add_budget(po[po["view"] == "current"])
    arc = ic.add_budget(po[po["view"] == "archive"])
    sb = ic.load(run_dir, "sample_bank_items")
    sb = sb[sb["partition_id"] == partition]

    strata = sorted(set(cur["stratum"]))
    levels = isoframe._levels(strata)
    arc_by_name = {n: g for n, g in arc.groupby("name")}

    def wm(df, col, w):
        return float(sum(wt * df[df["stratum"] == s][col].mean() for s, wt in w.items()
                         if (df["stratum"] == s).any()))

    rows = []
    for level, w in levels.items():
        for rung in rungs:
            sub = sb[sb["k"] == rung]
            if not len(sub):
                continue
            cov_k = float(sum(wt * sub[sub["stratum"] == s]["prefix_coverage"].mean()
                              for s, wt in w.items()))
            orc_k = float(sum(wt * sub[sub["stratum"] == s]["oracle_pass_at_k"].mean()
                              for s, wt in w.items()))
            for name, g in cur[cur["rung"] == rung].groupby("name"):
                a = arc_by_name.get(name)
                rows.append(dict(
                    level=level, rung=rung, name=name, family=g["family"].iloc[0],
                    B=int(g["B"].iloc[0]), R=int(g["R"].iloc[0]),
                    n_pool=float(g["n_pool"].mean()),
                    cov_sample=cov_k, oracle=orc_k,
                    cov_own=wm(g, "coverage", w), y=wm(g, "y", w),
                    cov_arc=wm(a, "coverage", w) if a is not None else np.nan,
                    y_arc=wm(a, "y", w) if a is not None else np.nan,
                    conc=wm(g[g["n_pool"] >= 2], "conc", w) if "conc" in g else np.nan,
                ))
    d = pd.DataFrame(rows)
    d["cov_deficit"] = d["cov_sample"] - d["cov_own"]
    d["sel_loss"] = d["cov_own"] - d["y"]
    d["total_gap"] = d["cov_sample"] - d["y"]
    d["arc_gain"] = d["y_arc"] - d["y"]
    return d


def concentration(run_dir: str, partition: str = "main"):
    """Mean correct fraction of the pool, by family and budget, over informative benchmarks."""
    po = ic.load(run_dir, "policy_outcomes")
    po = ic.add_budget(po[(po["partition_id"] == partition) & (po["view"] == "current")])
    po = po[po["benchmark"].isin(ic.INFORMATIVE) & (po["n_pool"] >= 2)]
    po = po.assign(conc=po["n_pool_correct"] / po["n_pool"])
    return (po.groupby(["rung", "family"])
            .agg(conc=("conc", "mean"), y=("y", "mean"), n=("y", "size")).reset_index())


def selector_response(run_dir: str, partition: str = "main"):
    """P(delivered correct | the pool held a correct answer), against pool concentration."""
    po = ic.load(run_dir, "policy_outcomes")
    po = po[(po["partition_id"] == partition) & (po["view"] == "current")]
    d = po[(po["n_pool"] >= 2) & (po["n_pool_correct"] >= 1)].copy()
    d["conc"] = d["n_pool_correct"] / d["n_pool"]
    d["bin"] = pd.cut(d["conc"], CONC_BINS, labels=CONC_LABEL, include_lowest=True)
    return (d.groupby(["benchmark", "bin"], observed=True)
            .agg(deliver=("y", "mean"), n=("y", "size")).reset_index())


def buckets(run_dir: str, partition: str = "main"):
    """Item mass split three ways for a k-sample prefix, against k.

    The prefix is a draw without replacement from the item's bank, so the number of correct
    candidates among k is hypergeometric. That makes the three shares exact rather than a
    property of the full bank:

    no_correct  the prefix holds no correct answer at all: generation-limited.
    minority    a correct answer is present but is not the majority: verification-limited.
    majority    the correct answer already carries the vote.
    """
    sb = ic.load(run_dir, "sample_bank_items")
    sb = sb[sb["partition_id"] == partition]
    out = []
    for (k, bench), g in sb.groupby(["k", "benchmark"]):
        n = g["bank_size"].to_numpy()
        c = g["bank_correct"].to_numpy()
        k = int(k)
        none = hypergeom.pmf(0, n, c, k)
        # majority means strictly more than half the prefix is correct
        # An exact half is a tie, which a real vote breaks by coin flip; counting it as half
        # keeps even and odd k comparable instead of producing a parity sawtooth.
        xs = np.arange(0, k + 1)
        maj = np.zeros(len(n))
        for x in xs[xs > k / 2.0]:
            maj += hypergeom.pmf(x, n, c, k)
        if k % 2 == 0:
            maj += 0.5 * hypergeom.pmf(k // 2, n, c, k)
        out.append(dict(partition=partition, k=k, benchmark=bench, n_items=len(g),
                        no_correct=float(np.mean(none)),
                        minority=float(np.mean(np.clip(1.0 - none - maj, 0.0, None))),
                        majority=float(np.mean(maj)),
                        oracle=float(g["oracle_pass_at_k"].mean()),
                        readout=float(g["readout"].mean()),
                        random_draw=float(g["mean_draw_correct"].mean())))
    return pd.DataFrame(out).sort_values(["benchmark", "k"]).reset_index(drop=True)


# ---------------------------------------------------------------- verifier sweep

def _pick_correct(c: int, n: int, mu: float, rng) -> float:
    """A reranker scoring correct candidates N(mu,1) and wrong ones N(0,1), taking the argmax."""
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if not np.isfinite(mu):
        return 1.0
    sc = rng.standard_normal((SIM_DRAWS, c)) + mu
    sw = rng.standard_normal((SIM_DRAWS, n - c))
    return float((sc.max(axis=1) > sw.max(axis=1)).mean())


def verifier_sweep(run_dir: str, partition: str = "main", k: int = 25, seed: int = ic.STUDY_SEED):
    """Delivered accuracy against reranker pairwise AUC, on the real bank compositions.

    This bounds what a verifier of a given discriminative power would buy from candidates the run
    already paid for. It is not evidence that such a verifier exists for these tasks.
    """
    sb = ic.load(run_dir, "sample_bank_items")
    sb = sb[(sb["partition_id"] == partition) & (sb["k"] == k)]
    rng = np.random.default_rng(seed)
    rows = []
    for bench, g in sb.groupby("benchmark"):
        comp = g.groupby(["bank_correct", "bank_size"]).size().reset_index(name="w")
        tot = int(comp["w"].sum())
        cache = {}
        for auc in AUCS:
            mu = np.inf if auc >= 1.0 else norm.ppf(auc) * np.sqrt(2.0)
            acc = 0.0
            for _, r in comp.iterrows():
                key = (int(r["bank_correct"]), int(r["bank_size"]), auc)
                if key not in cache:
                    cache[key] = _pick_correct(int(r["bank_correct"]), int(r["bank_size"]), mu, rng)
                acc += int(r["w"]) * cache[key]
            rows.append(dict(partition=partition, k=k, benchmark=bench, auc=auc,
                             delivered=acc / tot,
                             plurality=float(g["readout"].mean()),
                             oracle=float(g["oracle_pass_at_k"].mean()),
                             random_draw=float(g["mean_draw_correct"].mean())))
    d = pd.DataFrame(rows)
    mac = (d[d["benchmark"].isin(ic.INFORMATIVE)]
           .groupby("auc")[["delivered", "plurality", "oracle", "random_draw"]].mean()
           .reset_index().assign(benchmark="macro3", partition=partition, k=k))
    return pd.concat([d, mac], ignore_index=True)


def equivalent_auc(sweep: pd.DataFrame, bench: str, value: float) -> float:
    """The reranker AUC whose delivered accuracy matches *value*, by linear interpolation."""
    g = sweep[sweep["benchmark"] == bench].sort_values("delivered")
    return float(np.interp(value, g["delivered"], g["auc"]))


# ---------------------------------------------------------------- the A axis

def allowance_axis(run_dir: str, k: int = 25):
    """Main versus high allowance on the shared items: does a bigger A close the gap?"""
    sb = ic.load(run_dir, "sample_bank_items")
    shared = set(sb[sb["partition_id"] == "high"]["item_id"])
    out = []
    for part, a0 in (("main", 1024), ("high", 4096)):
        d = sb[(sb["partition_id"] == part) & (sb["item_id"].isin(shared)) & (sb["k"] == k)]
        for bench, g in d.groupby("benchmark"):
            f = g["bank_correct"] / g["bank_size"]
            out.append(dict(panel=part, a0=a0, benchmark=bench, n_items=len(g),
                            draw_acc=float(f.mean()),
                            no_correct=float((f == 0).mean()),
                            minority=float(((f > 0) & (f <= 0.5)).mean()),
                            oracle=float(g["oracle_pass_at_k"].mean()),
                            readout=float(g["readout"].mean()),
                            gap=float(g["oracle_pass_at_k"].mean() - g["readout"].mean())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- archive readout

def archive_gain(run_dir: str, partition: str = "main", alpha: float = 0.05):
    """Paired item bootstrap of the archive readout minus the current readout, per configuration.

    Same generations, same cost: the archive view votes over every round's output rather than only
    the last one.
    """
    po = ic.load(run_dir, "policy_outcomes")
    po = po[po["partition_id"] == partition]
    cur = ic.add_budget(po[po["view"] == "current"])
    names = sorted(set(cur[cur["family"].isin(ic.LADDER_FAMILIES)]["name"]))
    cols = [(partition, n) for n in names]
    pop = astats.build_population(cur, cols, view="current", label=f"arc:{partition}")
    rs = pop.resampler(ic.REPLICATES, ic.STUDY_SEED)
    levels = isoframe._levels(pop.strata)

    piv_c = po[po["view"] == "current"].pivot_table(index="item_id", columns="name",
                                                    values="y", aggfunc="first")
    piv_a = po[po["view"] == "archive"].pivot_table(index="item_id", columns="name",
                                                    values="y", aggfunc="first")
    diff = {s: (piv_a.reindex(index=pop.items[s], columns=names).to_numpy(float)
                - piv_c.reindex(index=pop.items[s], columns=names).to_numpy(float))
            for s in pop.strata}
    boot = isoframe._boot_means(diff, rs)
    rows = []
    for level, w in levels.items():
        for j, name in enumerate(names):
            est = isoframe._wpoint(diff, w, j)
            db = isoframe._wboot(boot, w, j)
            lo, hi = np.nanquantile(db, [alpha / 2, 1 - alpha / 2])
            rows.append(dict(level=level, name=name, estimate=est,
                             ci_lo=float(lo), ci_hi=float(hi)))
    return pd.DataFrame(rows)
