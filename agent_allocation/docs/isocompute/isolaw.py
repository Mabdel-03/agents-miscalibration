"""A conditional scaling law: which axis to scale, given the item's state.

Each item has a state $(p, q_1)$, the per-draw probability of a correct answer and the largest
share held by any single wrong answer. Their difference $m = p - q_1$ is the plurality margin, and
it decides everything a vote can do: as $k$ grows the vote converges to the correct answer exactly
when $m > 0$. That splits items into three regimes with different optimal axes.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

import isocore as ic
import isoselect

KS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 25]
REGIMES = ["decided", "outvoted", "dead"]
REGIME_LABEL = {"decided": "Decided (correct is modal)",
                "outvoted": "Outvoted (correct present, loses)",
                "dead": "Dead (no correct answer)"}
REGIME_COLOR = {"decided": "#BBD7E8", "outvoted": "#D55E00", "dead": "#5A5A5A"}


def _bank_rows(run_dir, partition, cache):
    D = isoselect.load_banks(run_dir, partition, cache)
    cand = ic.load(run_dir, "candidates")[["node_id", "correct", "benchmark", "item_id"]]
    corr = dict(zip(cand["node_id"], cand["correct"]))
    bench = dict(zip(cand["item_id"], cand["benchmark"]))
    for item, ids in D["banks"].items():
        seq = [(D["ans"][c][0], bool(corr.get(c)), D["ans"][c][1]) for c in ids]
        yield item, bench[item], seq


def item_state(run_dir: str, partition: str = "main", cache: str | None = None):
    """Per item: p, the largest wrong-answer share q1, the margin, and the regime."""
    rows = []
    for item, bench, seq in _bank_rows(run_dir, partition, cache):
        v = [(a, c) for a, c, stt in seq if stt == "valid" and a is not None]
        if not v:
            continue
        n = len(v)
        cnt = collections.Counter(a for a, _ in v)
        grp = {a: c for a, c in v}
        p = sum(cnt[a] for a in cnt if grp[a]) / n
        wrong = [cnt[a] for a in cnt if not grp[a]]
        q1 = (max(wrong) / n) if wrong else 0.0
        rows.append(dict(item_id=item, benchmark=bench, n=n, p=p, q1=q1, m=p - q1))
    d = pd.DataFrame(rows)
    d["regime"] = np.where(d["p"] == 0, "dead", np.where(d["m"] > 0, "decided", "outvoted"))
    return d


def vote_curve(run_dir: str, partition: str = "main", cache: str | None = None,
               draws: int = 3000, seed: int = ic.STUDY_SEED):
    """P(the plurality of k draws is correct), per item, by resampling the realised bank."""
    rng = np.random.default_rng(seed)
    rows = []
    for item, bench, seq in _bank_rows(run_dir, partition, cache):
        v = [(a, c) for a, c, stt in seq if stt == "valid" and a is not None]
        if not v:
            continue
        n = len(v)
        cnt = collections.Counter(a for a, _ in v)
        grp = {a: c for a, c in v}
        keys = list(cnt)
        counts = np.array([cnt[a] for a in keys])
        isc = np.array([grp[a] for a in keys])
        p = counts[isc].sum() / n
        q1 = (counts[~isc].max() / n) if (~isc).any() else 0.0
        urn = np.repeat(np.arange(len(keys)), counts)
        for k in KS:
            if k > n:
                continue
            idx = np.argsort(rng.random((draws, n)), axis=1)[:, :k]
            oh = (urn[idx][:, :, None] == np.arange(len(keys))[None, None, :]).sum(axis=1)
            rows.append(dict(item_id=item, benchmark=bench, k=k, p=p, q1=q1, m=p - q1,
                             acc=float(isc[oh.argmax(axis=1)].mean())))
    return pd.DataFrame(rows)


def width_law(curve: pd.DataFrame, m_min: float = 0.05):
    """Samples needed to resolve a decided item, against its margin.

    A binomial argument gives k* proportional to m^-2; this fits the exponent rather than
    assuming it.
    """
    c = curve[(curve["m"] > m_min) & curve["benchmark"].isin(ic.INFORMATIVE)]
    rows = []
    for item, g in c.groupby("item_id"):
        g = g.sort_values("k")
        lim = float(g["acc"].iloc[-1])
        if lim < 0.5:
            continue
        hit = g[g["acc"] >= 0.95 * lim]
        if hit.empty:
            continue
        rows.append(dict(item_id=item, m=float(g["m"].iloc[0]), kstar=float(hit["k"].iloc[0])))
    o = pd.DataFrame(rows)
    lm, lk = np.log(o["m"].to_numpy()), np.log(o["kstar"].to_numpy())
    b, a = np.polyfit(lm, lk, 1)
    fit = dict(coef=float(np.exp(a)), exponent=float(b), n=int(len(o)),
               corr=float(np.corrcoef(lm, lk)[0, 1]))
    return o, fit


def axis_by_regime(run_dir: str, state: pd.DataFrame, partition: str = "main",
                   budgets=(4, 8, 16), reps: int = 10000, seed: int = ic.STUDY_SEED):
    """Width against depth-in-one-call at matched compute, split by regime."""
    po = ic.load(run_dir, "policy_outcomes")
    po = po[(po["partition_id"] == partition) & (po["view"] == "current")]
    y = po.pivot_table(index="item_id", columns="name", values="y", aggfunc="first")
    d = state.set_index("item_id").join(y)
    d = d[d["benchmark"].isin(ic.INFORMATIVE)]
    rng = np.random.default_rng(seed)
    rows = []
    for C in budgets:
        w, dp = f"sample/{C}", f"deep/{C}"
        for reg, g in d.groupby("regime"):
            diff = (g[dp] - g[w]).to_numpy(float)
            bs = diff[rng.integers(0, len(diff), (reps, len(diff)))].mean(axis=1)
            rows.append(dict(C=C, regime=reg, n=int(len(g)),
                             width=float(g[w].mean()), depth=float(g[dp].mean()),
                             diff=float(diff.mean()),
                             ci_lo=float(np.quantile(bs, 0.025)),
                             ci_hi=float(np.quantile(bs, 0.975))))
    return pd.DataFrame(rows)


def allowance_transition(state_lo: pd.DataFrame, state_hi: pd.DataFrame):
    """Regime transitions on shared items when the per-draw allowance is raised."""
    j = state_lo.merge(state_hi, on=["item_id", "benchmark"], suffixes=("_lo", "_hi"))
    j = j[j["benchmark"].isin(ic.INFORMATIVE)]
    return j, pd.crosstab(j["regime_lo"], j["regime_hi"])


def _auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def predictability(run_dir: str, state: pd.DataFrame, partition: str = "main",
                   cache: str | None = None, k0: int = 4):
    """Can a cheap pilot tell you the regime before you commit the budget?"""
    cand = ic.load(run_dir, "candidates")
    rt = dict(zip(cand["node_id"], cand["reasoning_tokens"]))
    rc = dict(zip(cand["node_id"], cand["rendered_chars"]))
    D = isoselect.load_banks(run_dir, partition, cache)
    corr = dict(zip(cand["node_id"], cand["correct"]))
    rows = []
    for item, ids in D["banks"].items():
        f0 = ids[:k0]
        v = [(D["ans"][c][0], bool(corr.get(c))) for c in f0 if D["ans"][c][1] == "valid"]
        cnt = collections.Counter(a for a, _ in v)
        n = max(len(v), 1)
        rk = sorted(cnt.items(), key=lambda kv: -kv[1])
        ent = -sum((c / n) * np.log(c / n) for c in cnt.values()) if cnt else 0.0
        rows.append(dict(item_id=item, margin=(rk[0][1] - (rk[1][1] if len(rk) > 1 else 0)) / n
                         if rk else 0.0,
                         n_distinct=len(cnt), entropy=ent,
                         mean_reason=float(np.mean([rt.get(c) or 0 for c in f0])),
                         mean_chars=float(np.mean([rc.get(c) or 0 for c in f0]))))
    f = pd.DataFrame(rows).set_index("item_id").join(state.set_index("item_id"))
    f = f[f["benchmark"].isin(ic.INFORMATIVE)]
    f["not_decided"] = (f["regime"] != "decided").astype(int)
    sig = [("margin", -1), ("n_distinct", 1), ("entropy", 1),
           ("mean_reason", 1), ("mean_chars", 1)]
    out = [dict(scope="pooled", n=int(len(f)), base=float(f["not_decided"].mean()),
                **{c: _auc(f["not_decided"], s * f[c]) for c, s in sig})]
    for b, g in f.groupby("benchmark"):
        out.append(dict(scope=b, n=int(len(g)), base=float(g["not_decided"].mean()),
                        **{c: _auc(g["not_decided"], s * g[c]) for c, s in sig}))
    return f, pd.DataFrame(out)


def prescription(state: pd.DataFrame, axis: pd.DataFrame, C: int = 8):
    """Predicted gain from spending a budget on depth rather than width, per benchmark.

    gain = share(outvoted) x effect(outvoted) + share(dead) x effect(dead), with the effects
    taken from the matched-compute comparison at the same budget.
    """
    eff = axis[axis["C"] == C].set_index("regime")["diff"]
    rows = []
    for b, g in state[state["benchmark"].isin(ic.INFORMATIVE)].groupby("benchmark"):
        sh = g["regime"].value_counts(normalize=True)
        pred = sum(float(sh.get(r, 0.0)) * float(eff.get(r, 0.0)) for r in REGIMES)
        rows.append(dict(benchmark=b, decided=float(sh.get("decided", 0.0)),
                         outvoted=float(sh.get("outvoted", 0.0)),
                         dead=float(sh.get("dead", 0.0)), predicted_gain=pred))
    return pd.DataFrame(rows)
