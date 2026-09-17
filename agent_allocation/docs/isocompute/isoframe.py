"""Build the master iso-compute frame: one row per (level, rung, configuration).

Every ladder checkpoint on a partition shares one fixed item population, so a single
``build_population`` call yields one value matrix and one resampler. Every accuracy, every
difference and every ratio below is therefore computed on the same items with the same resample
indices, which is what makes the differences paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_allocation.analysis import stats as astats
import isocore as ic

CHUNK = 500


def _boot_means(mat: dict, rs) -> dict:
    """{stratum: (n_resamples, n_columns)} of resampled column means, chunked over replicates."""
    out = {}
    for s, Y in mat.items():
        if Y.shape[0] == 0:
            continue
        B, P = rs.n_resamples, Y.shape[1]
        acc = np.empty((B, P), dtype=float)
        for start in range(0, B, CHUNK):
            sl = slice(start, min(B, start + CHUNK))
            acc[sl] = np.nanmean(Y[rs.idx[s][sl]], axis=1)
        out[s] = acc
    return out


def _levels(strata) -> dict:
    """Weight maps for macro4, macro3 and each benchmark. BBH subtasks always weighted equally."""
    by_bench: dict[str, list[str]] = {}
    for s in strata:
        by_bench.setdefault(s.split("/")[0], []).append(s)
    levels = {}
    benches4 = [b for b in ic.BENCH_ORDER if b in by_bench]
    levels["macro4"] = {s: 1.0 / (len(benches4) * len(by_bench[b]))
                        for b in benches4 for s in by_bench[b]}
    benches3 = [b for b in ic.INFORMATIVE if b in by_bench]
    levels["macro3"] = {s: 1.0 / (len(benches3) * len(by_bench[b]))
                        for b in benches3 for s in by_bench[b]}
    for b in benches4:
        levels[b] = {s: 1.0 / len(by_bench[b]) for s in by_bench[b]}
    return levels


def _wpoint(mat: dict, w: dict, j: int) -> float:
    return float(sum(wt * np.nanmean(mat[s][:, j]) for s, wt in w.items()
                     if s in mat and mat[s].shape[0]))


def _wboot(boot: dict, w: dict, j: int) -> np.ndarray:
    acc = None
    for s, wt in w.items():
        if s not in boot:
            continue
        v = wt * boot[s][:, j]
        acc = v if acc is None else acc + v
    return acc


def build(run_dir: str, partition: str = "main", *, alpha: float = 0.05):
    """Returns (frame, context). Frame rows: level x rung x configuration."""
    oc = ic.outcomes(run_dir, partition)
    idx = ic.checkpoint_index(oc)
    names = list(idx["name"])
    cols = [(partition, n) for n in names]

    pop = astats.build_population(oc, cols, view="current", label=f"iso:{partition}")
    mat = astats.value_matrix(oc, pop, cols, view="current", column="y")
    rs = pop.resampler(ic.REPLICATES, ic.STUDY_SEED)
    boot = _boot_means(mat, rs)
    levels = _levels(pop.strata)
    col_of = {n: j for j, n in enumerate(names)}

    # Per-item oracle pass@k, aligned to the population's item order, one column per k.
    sb = ic.load(run_dir, "sample_bank_items")
    sb = sb[sb["partition_id"] == partition]
    ks = sorted(set(int(k) for k in sb["k"].unique()))
    # Two ceilings, deliberately kept distinct:
    #  * oracle_pass_at_k  -- the combinatorial estimator over the whole bank. Smooth and
    #    unbiased, the curve readers know as pass@k, used for the ladder figure.
    #  * prefix_coverage   -- 1 when the realized prefix root[0:k] contained a correct answer.
    #    Binary and per item, so it pairs with the numerator under the same resample indices.
    #    This is the capture denominator; a revising or communicating configuration can exceed
    #    it by producing an answer that independent sampling never generated.
    omat, cmat = {}, {}
    for s in pop.strata:
        items = pop.items[s]
        sub = sb[sb["stratum"] == s]
        for col, target in (("oracle_pass_at_k", omat), ("prefix_coverage", cmat)):
            piv = (sub.pivot_table(index="item_id", columns="k", values=col, aggfunc="first")
                   .reindex(index=items, columns=ks))
            target[s] = piv.to_numpy(dtype=float)
    oboot = _boot_means(omat, rs)
    cboot = _boot_means(cmat, rs)
    k_of = {k: j for j, k in enumerate(ks)}

    rows = []
    for level, w in levels.items():
        base1 = _wpoint(mat, w, col_of["sample/1"])          # pass@1, the eta floor
        for _, r in idx.iterrows():
            name, rung = r["name"], int(r["rung"])
            j = col_of[name]
            acc = _wpoint(mat, w, j)
            ab = _wboot(boot, w, j)
            alo, ahi = np.nanquantile(ab, [alpha / 2, 1 - alpha / 2])

            # Baseline: the achievable readout of independent sampling at the same budget.
            bname = f"sample/{rung}"
            has_base = bname in col_of and name != bname
            if has_base:
                i = col_of[bname]
                d = acc - _wpoint(mat, w, i)
                db = ab - _wboot(boot, w, i)
                dlo, dhi = np.nanquantile(db, [alpha / 2, 1 - alpha / 2])
                centred = db - np.nanmean(db)
                dp = float(np.mean(np.abs(centred) >= abs(d)))
            else:
                d = dlo = dhi = dp = np.nan

            # Ceiling capture against the oracle pass@k that the same budget makes available.
            if rung in k_of:
                oj = k_of[rung]
                ceil = _wpoint(omat, w, oj)          # reported pass@k (combinatorial)
                cov = _wpoint(cmat, w, oj)           # capture denominator (realized prefix)
                denom = cov - base1
                if denom > 0.02:
                    eta = (acc - base1) / denom
                    cb = _wboot(cboot, w, oj)
                    b1 = _wboot(boot, w, col_of["sample/1"])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ebd = (ab - b1) / (cb - b1)
                    ebd = ebd[np.isfinite(ebd)]
                    elo, ehi = (np.nanquantile(ebd, [alpha / 2, 1 - alpha / 2])
                                if ebd.size else (np.nan, np.nan))
                else:
                    eta = elo = ehi = np.nan
            else:
                eta = elo = ehi = ceil = cov = np.nan

            rows.append(dict(
                partition=partition, level=level, benchmark=level, rung=rung, name=name,
                family=r["family"], B=int(r["B"]), R=int(r["R"]), A=int(r["A"]), C=int(r["C"]),
                acc=acc, acc_lo=float(alo), acc_hi=float(ahi),
                delta=d, delta_lo=dlo, delta_hi=dhi, delta_p=dp,
                eta=eta, eta_lo=elo, eta_hi=ehi, oracle=ceil, coverage=cov, pass1=base1,
                n_items=pop.n_items,
            ))

    frame = pd.DataFrame(rows)
    # BH within each (level) panel over the exploratory delta tests.
    frame["delta_sig"] = False
    for level, g in frame.groupby("level"):
        m = g["delta_p"].to_numpy()
        frame.loc[g.index, "delta_sig"] = ic.bh_fdr(m, 0.05)

    ctx = dict(pop=pop, mat=mat, rs=rs, boot=boot, levels=levels, col_of=col_of,
               omat=omat, oboot=oboot, k_of=k_of, idx=idx, oc=oc, names=names)
    return frame, ctx


def readout_curve(run_dir: str, partition: str = "main") -> pd.DataFrame:
    """Oracle pass@k beside the achievable readout, macro4/macro3/per benchmark, from the bank."""
    sb = ic.load(run_dir, "sample_bank_items")
    sb = sb[sb["partition_id"] == partition]
    strata = sorted(set(sb["stratum"]))
    levels = _levels(strata)
    out = []
    for level, w in levels.items():
        for k, g in sb.groupby("k"):
            o = sum(wt * g[g["stratum"] == s]["oracle_pass_at_k"].mean() for s, wt in w.items())
            c = sum(wt * g[g["stratum"] == s]["prefix_coverage"].mean() for s, wt in w.items())
            r = sum(wt * g[g["stratum"] == s]["readout"].mean() for s, wt in w.items())
            out.append(dict(partition=partition, level=level, k=int(k), oracle=float(o),
                            coverage=float(c), readout=float(r), gap=float(o - r),
                            same_pool_gap=float(c - r)))
    return pd.DataFrame(out)


def paired_contrasts(run_dir: str, pairs, partition: str = "main", *, alpha: float = 0.05):
    """Paired item-clustered contrasts nameA minus nameB, at macro4/macro3/per benchmark."""
    oc = ic.outcomes(run_dir, partition)
    names = sorted({n for p in pairs for n in p})
    cols = [(partition, n) for n in names]
    pop = astats.build_population(oc, cols, view="current", label=f"pairs:{partition}")
    mat = astats.value_matrix(oc, pop, cols, view="current", column="y")
    rs = pop.resampler(ic.REPLICATES, ic.STUDY_SEED)
    boot = _boot_means(mat, rs)
    levels = _levels(pop.strata)
    col_of = {n: j for j, n in enumerate(names)}
    rows = []
    for a, b in pairs:
        ja, jb = col_of[a], col_of[b]
        for level, w in levels.items():
            est = _wpoint(mat, w, ja) - _wpoint(mat, w, jb)
            db = _wboot(boot, w, ja) - _wboot(boot, w, jb)
            lo, hi = np.nanquantile(db, [alpha / 2, 1 - alpha / 2])
            centred = db - np.nanmean(db)
            p = float(np.mean(np.abs(centred) >= abs(est)))
            rows.append(dict(partition=partition, a=a, b=b, level=level, estimate=est,
                             ci_lo=float(lo), ci_hi=float(hi), p=p, n_items=pop.n_items))
    return pd.DataFrame(rows)
