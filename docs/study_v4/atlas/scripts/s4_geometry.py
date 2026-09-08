"""A12: representational geometry, recomputed from the raw fp16 vectors (not the shipped table).

Reports scale-free (participation ratio, effective rank) AND scale-carrying (pairwise distance,
norm) metrics side by side, because they dissociate: reading only the scale-free family produced
a wrong 'convergence' conclusion in this study.
"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import _common as C

E = C.Emitter("s4_geometry.py")
IDX = pd.read_parquet(f"{C.CACHE}/index.parquet")
IDX["domain"] = np.where(IDX.source_id.astype(str).str.startswith("bcb"), "bcb", "hle")
ARMS = {"IND": ("IND_VOTE", "INDEPENDENT_SOLVER", "ROOT"), "DEC": ("DEC", "DECENTRALIZED_MEMBER", "TERMINAL")}
S_FIXED, N_SUB = 5, 100


def load(subset):
    return np.load(f"{C.CACHE}/{subset}.fp16.npy", mmap_mode="r")


def spectrum(X):
    """All geometry of one group of s states, scale-free and scale-carrying."""
    X = np.asarray(X, dtype=np.float64)
    s = len(X)
    Xc = X - X.mean(0)
    G = Xc @ Xc.T / max(s - 1, 1)                       # s x s Gram; same nonzero spectrum as the covariance
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    ev = ev[ev > ev.max() * 1e-12] if ev.max() > 0 else ev
    tot = ev.sum()
    pr = (tot ** 2 / (ev ** 2).sum()) if len(ev) and (ev ** 2).sum() > 0 else np.nan
    p = ev / tot if tot > 0 else ev
    er = float(np.exp(-(p * np.log(p + 1e-30)).sum())) if len(p) else np.nan
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(s, 1)
    U = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    cos = 1.0 - (U @ U.T)[iu]
    return dict(pr=float(pr), er=er, lam1_share=float(ev.max() / tot) if tot > 0 else np.nan,
                total_var=float(tot), pair_mean=float(D[iu].mean()), pair_min=float(D[iu].min()),
                pair_max=float(D[iu].max()), norm_mean=float(np.linalg.norm(X, axis=1).mean()),
                cos_mean=float(cos.mean()))


def groups(subset, arm, order_by=None, take=None):
    """Per-item geometry for one arm; IND is subsampled to s=5 (hash-seeded) like the study's own rule."""
    meth, role, phase = ARMS[arm]
    d = IDX[(IDX.subset == subset) & (IDX.method == meth) & (IDX.role == role) & (IDX.phase == phase)]
    X = load(subset)
    out = []
    for sid, g in d.groupby("source_id"):
        if order_by is not None:
            g = g.sort_values(order_by)
            g = g.head(take) if take and take > 0 else g.tail(-take)
        rows = g.out_row.values
        if len(rows) < S_FIXED: continue
        V = np.asarray(X[np.sort(rows)], dtype=np.float32)
        if len(V) == S_FIXED:
            rec = spectrum(V); rec["status"] = "exact"; rec["n_avail"] = len(V)
        else:
            seed = int(hashlib.sha256(f"{subset}|{arm}|{sid}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            recs = [spectrum(V[rng.choice(len(V), S_FIXED, replace=False)]) for _ in range(N_SUB)]
            rec = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
            rec["status"] = "subsampled"; rec["n_avail"] = len(V)
        rec.update(item=sid, domain="bcb" if str(sid).startswith("bcb") else "hle")
        out.append(rec)
    return pd.DataFrame(out)


METRICS = ["pr", "er", "lam1_share", "total_var", "pair_mean", "pair_min", "pair_max", "norm_mean", "cos_mean"]

# ---- coverage caveat: the native capture was partial ----
g_tab = C.geometry()
E.emit("geom.table_status", g_tab.status.value_counts().to_dict(), n=len(g_tab), source="neural/tables/geometry.parquet")
E.emit("geom.items_per_method", g_tab.groupby("method").item.nunique().to_dict(),
       note="IND_VOTE geometry exists for a minority of the 300-item panel: the native capture is ~72% complete")
_dr = g_tab[(g_tab.method == "DEC") & (g_tab.phase == "ROOT")]
E.emit("geom.dec_root_unavailable",
       dict(rows=int(len(_dr)), status=_dr.status.value_counts().to_dict(),
            usable=int((_dr.status.isin(["exact", "subsampled"])).sum())),
       note="no DEC-root group has the contracted 5 comparable states: only one root member per episode was "
            "captured, so there is no within-episode before/after baseline anywhere in the data")

# ---- paired arm contrast across the trajectory ----
traj, store = {}, {}
for subset in ("31_NATIVE_PREFILL", "31_GENERATED_32", "31_GENERATED_128", "31_GENERATED_512",
               "31_FINAL_OBJECT_CLOSE", "15_FINAL_OBJECT_CLOSE", "47_FINAL_OBJECT_CLOSE"):
    a, b = groups(subset, "IND"), groups(subset, "DEC")
    store[subset] = (a, b)
    j = a.set_index("item").join(b.set_index("item"), lsuffix="_I", rsuffix="_D", how="inner")
    rec = {"n_items": int(len(j)), "n_avail_IND": float(a.n_avail.mean()), "n_avail_DEC": float(b.n_avail.mean())}
    for m in METRICS:
        x, y = j[f"{m}_I"].values, j[f"{m}_D"].values
        ok = ~(np.isnan(x) | np.isnan(y))
        rec[m] = dict(IND=float(x[ok].mean()), DEC=float(y[ok].mean()),
                      ratio=float(y[ok].mean() / x[ok].mean()) if x[ok].mean() else None,
                      p=float(stats.ttest_rel(y[ok], x[ok]).pvalue), n=int(ok.sum()))
    traj[subset] = rec
E.emit("geom.trajectory", traj, note="paired on items; IND subsampled to s=5 from ~58 available, DEC exact 5")
per_dom = {}
for subset in ("31_FINAL_OBJECT_CLOSE",):
    a, b = store[subset]
    for dom in ("hle", "bcb"):
        j = a[a.domain == dom].set_index("item").join(b[b.domain == dom].set_index("item"),
                                                      lsuffix="_I", rsuffix="_D", how="inner")
        per_dom[dom] = {"n_items": int(len(j)),
                        **{m: dict(IND=float(j[f"{m}_I"].mean()), DEC=float(j[f"{m}_D"].mean()),
                                   p=float(stats.ttest_rel(j[f"{m}_D"], j[f"{m}_I"]).pvalue)) for m in METRICS}}
E.emit("geom.answer_anchor_by_domain", per_dom)

# ---- completion-length confound: does the answer-anchor geometry track how long the answer took? ----
conf = {}
for dom in ("hle", "bcb"):
    short = groups("31_FINAL_OBJECT_CLOSE", "IND", order_by="completion_tokens", take=S_FIXED)
    long_ = groups("31_FINAL_OBJECT_CLOSE", "IND", order_by="completion_tokens", take=-S_FIXED)
    s_, l_ = short[short.domain == dom].set_index("item"), long_[long_.domain == dom].set_index("item")
    j = s_.join(l_, lsuffix="_S", rsuffix="_L", how="inner")
    if len(j) < 15: continue
    conf[dom] = {"n_items": int(len(j)),
                 **{m: dict(short=float(j[f"{m}_S"].mean()), long=float(j[f"{m}_L"].mean()),
                            p=float(stats.ttest_rel(j[f"{m}_L"], j[f"{m}_S"]).pvalue)) for m in ("pr", "er", "pair_mean", "norm_mean")}}
ct = IDX[(IDX.subset == "31_FINAL_OBJECT_CLOSE")].groupby("method").completion_tokens.mean()
conf["mean_completion_tokens_by_method"] = ct.round(1).to_dict()
conf["mean_prompt_tokens_by_method"] = IDX[(IDX.subset == "31_FINAL_OBJECT_CLOSE")].groupby("method").prompt_tokens.mean().round(1).to_dict()
E.emit("geom.length_confound", conf,
       note="within IND only: the 5 shortest vs the 5 longest completions per item, same anchor")

# ---- do the arms differ in context length? (the other confound) ----
E.emit("geom.arm_context", IDX[IDX.subset == "31_FINAL_OBJECT_CLOSE"].groupby("method").agg(
    prompt_tokens=("prompt_tokens", "mean"), completion_tokens=("completion_tokens", "mean"),
    n=("rid", "size")).round(1).to_dict("index"),
    note="peer-exposed members carry the peer packets in context; the arms are not length-matched")

# ---- geometry vs episode behaviour (agreement recomputed from item files, not the defective column) ----
ag = pd.read_parquet(f"{C.DATA}/dec_final_agreement.parquet") if os.path.exists(f"{C.DATA}/dec_final_agreement.parquet") else None
dec = store["31_FINAL_OBJECT_CLOSE"][1]
if ag is not None:
    j = dec.merge(ag.rename(columns={"sid": "item"}), on="item", how="inner")
    ep = C.episodes(); acc = ep[(ep.module == "A") & (ep.method == "DEC") & (ep.checkpoint == "32B")].set_index("source_id").acc
    j["acc"] = j.item.map(acc)
    d0 = C.difficulty().set_index("source_id")
    j["p1"] = j.item.map(d0.pass_at_1)
    out = {}
    for dom in ("hle", "bcb"):
        k = j[j.domain == dom]
        u, s_ = k[k.unanimous == 1], k[k.unanimous < 1]
        blk = {"n": int(len(k)), "n_unanimous": int(len(u)), "n_split": int(len(s_))}
        for m in ("pr", "er", "pair_mean", "pair_max", "cos_mean", "norm_mean"):
            r = stats.spearmanr(k[m], k.agreement)
            blk[m] = dict(unanimous=float(u[m].mean()) if len(u) else None,
                          split=float(s_[m].mean()) if len(s_) else None,
                          welch_p=float(stats.ttest_ind(u[m], s_[m], equal_var=False).pvalue) if len(u) > 3 and len(s_) > 3 else None,
                          rho_agreement=float(r.statistic), p_rho=float(r.pvalue))
            if k.acc.notna().sum() > 20 and k.acc.nunique() > 1:
                cw = k[k.acc > 0.5][m]; wr = k[k.acc <= 0.5][m]
                blk[m]["correct"] = float(cw.mean()) if len(cw) else None
                blk[m]["wrong"] = float(wr.mean()) if len(wr) else None
                blk[m]["p_correct"] = float(stats.ttest_ind(cw, wr, equal_var=False).pvalue) if len(cw) > 3 and len(wr) > 3 else None
        # difficulty control on the headline metric
        kk = k.dropna(subset=["pr", "agreement", "p1"])
        if len(kk) > 25:
            ra = stats.spearmanr(kk.pr, kk.agreement).statistic
            rb = stats.spearmanr(kk.pr, kk.p1).statistic
            rc = stats.spearmanr(kk.agreement, kk.p1).statistic
            blk["pr_partial_given_difficulty"] = float((ra - rb * rc) / np.sqrt((1 - rb ** 2) * (1 - rc ** 2)))
        # leave-3-out fragility on the split arm
        if len(s_) > 6:
            drop = s_.nlargest(3, "pr").index
            blk["pr_welch_p_drop_top3_split"] = float(stats.ttest_ind(u.pr, s_.drop(drop).pr, equal_var=False).pvalue)
        out[dom] = blk
    E.emit("geom.dec_consensus", out,
           note="agreement recomputed from item files; geometry.parquet's own outcome_agreement is 30% null and matches 54% of the time")

# ---- IND full-pool dimensionality (all ~58 draws, not a fixed 5) ----
full = []
d = IDX[(IDX.subset == "31_FINAL_OBJECT_CLOSE") & (IDX.method == "IND_VOTE")]
X = load("31_FINAL_OBJECT_CLOSE")
for sid, g in d.groupby("source_id"):
    if len(g) < 10: continue
    V = np.asarray(X[np.sort(g.out_row.values)], dtype=np.float32)
    r = spectrum(V); r.update(item=sid, n=len(V), domain="bcb" if str(sid).startswith("bcb") else "hle")
    full.append(r)
F = pd.DataFrame(full)
sel = C.selections()
ind = sel[(sel.module == "A") & (sel.method == "IND_VOTE") & (sel.selector_id == "VOTE") &
          (sel.pool_kind == "archive")].set_index("source_id")
F["cand_mean"] = F.item.map(ind.candidate_mean); F["oracle"] = F.item.map(ind.oracle_coverage.astype(float))
F["all_singleton"] = F.item.map(ind.all_singleton.astype(float)); F["gap"] = F.item.map(ind.selection_gap)
fp = {"n_items": int(len(F)), "mean_states": float(F.n.mean()), "mean_pr": float(F.pr.mean())}
for dom in ("hle", "bcb"):
    k = F[F.domain == dom]
    fp[dom] = {"n": int(len(k)),
               **{o: dict(rho=float(stats.spearmanr(k.pr, k[o]).statistic), p=float(stats.spearmanr(k.pr, k[o]).pvalue))
                  for o in ("cand_mean", "oracle", "all_singleton", "gap") if k[o].notna().sum() > 20}}
    kk = k.dropna(subset=["pr", "cand_mean"])
    if len(kk) > 20:
        q = pd.qcut(kk.pr, 3, labels=["low", "mid", "high"])
        fp[dom]["cand_mean_by_pr_tercile"] = kk.groupby(q, observed=True).cand_mean.mean().round(4).to_dict()
E.emit("geom.ind_full_pool", fp, note="dimensionality of an item's whole 64-draw answer-state distribution")

for k, (a, b) in store.items():
    a.assign(arm="IND").to_parquet(f"{C.DATA}/geom_{k}_IND.parquet")
    b.assign(arm="DEC").to_parquet(f"{C.DATA}/geom_{k}_DEC.parquet")
E.save("geometry")
