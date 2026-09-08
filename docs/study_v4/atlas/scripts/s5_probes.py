"""A13/A14: what is linearly decodable from the residual stream, and from the report reader's state."""
import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy import stats
import _common as C

E = C.Emitter("s5_probes.py")
IDX = pd.read_parquet(f"{C.CACHE}/index.parquet")
IDX["domain"] = np.where(IDX.source_id.astype(str).str.startswith("bcb"), "bcb", "hle")
LAB = pd.read_parquet("/tmp/claude-225593/-orcd-data-tpoggio-001-mabdel03-agents-scaling/"
                      "ba72bd31-8ff0-45c2-b3b0-7ae1ad469566/scratchpad/mech/cand_labels.parquet") \
    if os.path.exists("/tmp/claude-225593/-orcd-data-tpoggio-001-mabdel03-agents-scaling/ba72bd31-8ff0-45c2-b3b0-7ae1ad469566/scratchpad/mech/cand_labels.parquet") else None
K, CVAL = 64, 0.05


def Z(subset, k=K):
    z = np.load(f"{C.CACHE}/{subset}.pca256.npz")["Z"][:, :k].astype(np.float32)
    d = IDX[IDX.subset == subset].sort_values("out_row").reset_index(drop=True)
    assert len(d) == len(z), (len(d), len(z))
    if LAB is not None:
        d = d.merge(LAB.drop_duplicates("rid").drop(columns=[c for c in ("cell_id", "source_id") if c in LAB.columns]),
                    on="rid", how="left")
    return z, d


def within_item_auc(items, score, y):
    out = []
    df = pd.DataFrame(dict(i=items, s=score, y=y))
    for _, g in df.groupby("i"):
        if g.y.nunique() == 2 and len(g) >= 4: out.append(roc_auc_score(g.y, g.s))
    return (float(np.mean(out)), len(out)) if out else (None, 0)


def probe(Zi, y, groups, center=False, items=None):
    Zi = Zi.copy()
    if center: Zi = Zi - pd.DataFrame(Zi).groupby(np.asarray(groups)).transform("mean").values
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Zi, y, groups):
        oof[te] = LogisticRegression(C=CVAL, max_iter=3000).fit(Zi[tr], y[tr]).decision_function(Zi[te])
    wi = within_item_auc(items if items is not None else groups, oof, y)
    return dict(pooled_auc=float(roc_auc_score(y, oof)), within_item_auc=wi[0], within_item_n=wi[1], n=int(len(y)))


# ================= what the answer-token state knows about its own candidate =================
res = {}
for blk in (15, 31, 47):
    z, d = Z(f"{blk}_FINAL_OBJECT_CLOSE")
    m = (d.method == "IND_VOTE") & d.valid.fillna(False) & d.correct.notna()
    zi, di = z[m.values], d[m].reset_index(drop=True)
    y = di.correct.astype(int).values
    for dom in ("all", "hle", "bcb"):
        mm = np.ones(len(di), bool) if dom == "all" else (di.domain == dom).values
        if mm.sum() < 200: continue
        r = probe(zi[mm], y[mm], di.source_id[mm].values)
        rc = probe(zi[mm], y[mm], di.source_id[mm].values, center=True)
        cw = within_item_auc(di[mm].source_id.values, di[mm].conf.fillna(0.5).values, y[mm])
        res[f"b{blk}|{dom}"] = dict(n=int(mm.sum()), items=int(di[mm].source_id.nunique()), acc=float(y[mm].mean()),
                                    pooled_auc=r["pooled_auc"], within_item=r["within_item_auc"],
                                    within_item_centred=rc["within_item_auc"], within_item_n=r["within_item_n"],
                                    conf_pooled_auc=float(roc_auc_score(y[mm], di[mm].conf.fillna(0.5))),
                                    conf_within_item=cw[0])
E.emit("probe.candidate_correctness", res,
       note="'within_item' asks whether the state distinguishes right from wrong SIBLINGS of the same question")

# ================= when in the trajectory does the state become informative =================
traj = {}
for a in ("NATIVE_PREFILL", "GENERATED_32", "GENERATED_128", "GENERATED_512", "FINAL_OBJECT_CLOSE"):
    z, d = Z(f"31_{a}")
    m = (d.method == "IND_VOTE") & d.correct.notna()
    zi, di = z[m.values], d[m].reset_index(drop=True)
    y = di.correct.astype(int).values
    r = probe(zi, y, di.source_id.values); rc = probe(zi, y, di.source_id.values, center=True)
    it = di.groupby("source_id").agg(y=("correct", "mean"))
    Zit = pd.DataFrame(zi).groupby(di.source_id.values).mean().loc[it.index].values
    oof = np.zeros(len(it))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Zit):
        oof[te] = Ridge(alpha=10).fit(Zit[tr], it.y.values[tr]).predict(Zit[te])
    traj[a] = dict(n=int(len(di)), pooled_auc=r["pooled_auc"], within_item=r["within_item_auc"],
                   within_item_centred=rc["within_item_auc"],
                   item_difficulty_rho=float(stats.spearmanr(oof, it.y.values).statistic), items=int(len(it)))
E.emit("probe.trajectory", traj, note="block 31, independent solvers; the last column is between-item difficulty")

# ================= is the stated confidence represented before it is written =================
cf = {}
for a in ("GENERATED_128", "GENERATED_512", "FINAL_OBJECT_CLOSE"):
    z, d = Z(f"31_{a}")
    m = (d.method == "IND_VOTE") & d.valid.fillna(False) & d.conf.notna()
    zi, di = z[m.values], d[m].reset_index(drop=True); g = di.source_id.values
    oof = np.zeros(len(di))
    for tr, te in GroupKFold(5).split(zi, di.conf, g):
        oof[te] = Ridge(alpha=100).fit(zi[tr], di.conf.values[tr]).predict(zi[te])
    r2 = 1 - ((oof - di.conf) ** 2).sum() / ((di.conf - di.conf.mean()) ** 2).sum()
    wi = [stats.spearmanr(x.a, x.b).statistic for _, x in
          pd.DataFrame(dict(a=oof, b=di.conf.values, i=g)).groupby("i") if x.b.nunique() > 1 and len(x) >= 5]
    cf[a] = dict(n=int(len(di)), r2=float(r2), rho=float(stats.spearmanr(oof, di.conf).statistic),
                 within_item_rho=float(np.nanmean(wi)) if wi else None, items_with_variation=len(wi))
E.emit("probe.confidence_representation", cf,
       note="the value is fully decodable only at the answer token, i.e. after it has been emitted")

# ================= majority membership vs correctness =================
mj = {}
z, d = Z("31_FINAL_OBJECT_CLOSE")
m = (d.method == "IND_VOTE") & d.valid.fillna(False) & (d.n_valid >= 4)
zi, di = z[m.values], d[m].reset_index(drop=True)
for dom in ("hle", "bcb"):
    mm = (di.domain == dom).values
    y = di.is_plurality.fillna(False).astype(int).values[mm]
    if y.min() == y.max(): continue
    r = probe(zi[mm], y, di.source_id[mm].values, center=True)
    cw = within_item_auc(di[mm].source_id.values, di[mm].conf.fillna(0.5).values, y)
    mj[dom] = dict(base_rate=float(y.mean()), within_item_auc=r["within_item_auc"], items=r["within_item_n"],
                   conf_within_item=cw[0], n=int(mm.sum()))
E.emit("probe.majority_membership", mj,
       note="compare with probe.candidate_correctness: mainstream-ness is more legible than correctness")

# ================= hub: is delegation predictable, and is it just difficulty? =================
hub = {}
z, d = Z("31_NATIVE_PREFILL", k=16)
h = (d.method == "CEN_FLAT") & (d.role == "CENTRAL_HUB") & (d.phase == "ROOT")
zh, dh = z[h.values], d[h].reset_index(drop=True)
ep = C.episodes(); cen = ep[(ep.module == "A") & (ep.method == "CEN_FLAT") & (ep.checkpoint == "32B")]
calls = cen.set_index("source_id").calls_admitted; acc = cen.set_index("source_id").acc
agg = pd.DataFrame(dict(item=dh.source_id.unique()))
Zh = pd.DataFrame(zh).groupby(dh.source_id.values).first()
agg = agg.set_index("item").join(Zh)
agg["calls"] = agg.index.map(calls); agg["acc"] = agg.index.map(acc)
d0 = C.difficulty().set_index("source_id"); agg["p1"] = agg.index.map(d0.pass_at_1)
agg["domain"] = np.where(agg.index.astype(str).str.startswith("bcb"), "bcb", "hle")
agg = agg.dropna(subset=["calls"])
feat = [c for c in agg.columns if isinstance(c, int)]
def auc_probe(frame, target):
    y = frame[target].values.astype(int)
    if y.min() == y.max() or len(y) < 40: return None
    X = frame[feat].values; oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, frame.index.values):
        oof[te] = LogisticRegression(C=CVAL, max_iter=3000).fit(X[tr], y[tr]).decision_function(X[te])
    return dict(auc=float(roc_auc_score(y, oof)), n=int(len(y)), base=float(y.mean()))
agg["multi"] = (agg.calls >= 2).astype(int)
hub["pooled_multi_call"] = auc_probe(agg, "multi")
for dom in ("hle", "bcb"):
    hub[f"{dom}_multi_call"] = auc_probe(agg[agg.domain == dom], "multi")
zb = agg[(agg.domain == "hle") & (agg.p1 == 0)]
hub["hle_never_band_multi_call"] = auc_probe(zb, "multi")
hub["episode_correct"] = auc_probe(agg.assign(y=(agg.acc > 0.5).astype(int)), "y")
hub["multi_call_rate_by_difficulty"] = dict(never=float(agg[agg.p1 == 0].multi.mean()),
                                            solvable=float(agg[agg.p1 > 0].multi.mean()))
E.emit("probe.hub_delegation", hub,
       note="the pooled figure is largely a domain classifier; inside the never-solved HLE band the signal is gone")

# ================= the report reader's state =================
sys.path.insert(0, "/orcd/data/tpoggio/001/mabdel03/agents_scaling/src")
from agents_scaling.study.neural.storage import load_stage
X, fr = load_stage(C.RUN_ROOT, "report", include_missing=False)
fr = fr.reset_index(drop=True); X = np.asarray(X, dtype=np.float32)
fr["k"] = fr.source_id.astype(str) + "|" + fr.method.astype(str)
sel = C.selections()
sel = sel[(sel.module == "A") & (sel.checkpoint == "32B") & (sel.split == "main") &
          (sel.selector_id == "VOTE")][["source_id", "method", "selected_correct", "oracle_coverage"]].drop_duplicates(["source_id", "method"])
def vec(anchor, blk):
    d = fr[(fr.anchor_kind == anchor) & (fr.block == blk)]
    return d.set_index("k"), X[d.vector_index.values]
rep, arch = {}, {}
for blk in (15, 31, 47):
    mt, Vt = vec("TASK_ONLY_ANCHOR", blk); ms, Vs = vec("STATE_ANCHOR", blk)
    common = sorted(set(mt.index) & set(ms.index))
    Vt = Vt[[mt.index.get_loc(k) for k in common]]; Vs = Vs[[ms.index.get_loc(k) for k in common]]
    info = mt.loc[common][["method", "source_id"]].reset_index(drop=True)
    lab = info.merge(sel, on=["source_id", "method"], how="left")
    keep = lab.selected_correct.notna().values
    y = lab.selected_correct.fillna(False).astype(int).values
    meth = pd.Categorical(info.method).codes
    for name, V in (("task_only", Vt), ("report_state", Vs), ("evidence_delta", Vs - Vt)):
        pred = np.zeros(len(meth), int)
        for tr, te in GroupKFold(5).split(V, meth, info.source_id):
            p = PCA(n_components=32, random_state=0).fit(V[tr])
            pred[te] = LogisticRegression(C=CVAL, max_iter=3000).fit(p.transform(V[tr]), meth[tr]).predict(p.transform(V[te]))
        arch[f"b{blk}|{name}"] = dict(accuracy=float((pred == meth).mean()), chance=1 / 3, n=int(len(meth)))
        oof = np.zeros(keep.sum())
        Vk, yk, gk = V[keep], y[keep], info.source_id.values[keep]
        for tr, te in GroupKFold(5).split(Vk, yk, gk):
            p = PCA(n_components=16, random_state=0).fit(Vk[tr])
            oof[te] = LogisticRegression(C=CVAL, max_iter=3000).fit(p.transform(Vk[tr]), yk[tr]).decision_function(p.transform(Vk[te]))
        rep[f"b{blk}|{name}"] = dict(auc=float(roc_auc_score(yk, oof)), n=int(len(yk)), base=float(yk.mean()))
    if blk == 31:
        disp = np.linalg.norm(Vs - Vt, axis=1) / np.linalg.norm(Vt, axis=1)
        lab2 = lab.assign(disp=disp, domain=np.where(lab.source_id.astype(str).str.startswith("bcb"), "bcb", "hle"))
        dd = {}
        for dom, g in lab2.dropna(subset=["selected_correct"]).groupby("domain"):
            dd[dom] = dict(n=int(len(g)), mean_rel_disp=float(g.disp.mean()),
                           rho_selected_correct=float(stats.spearmanr(g.disp, g.selected_correct.astype(float)).statistic),
                           p_selected=float(stats.spearmanr(g.disp, g.selected_correct.astype(float)).pvalue),
                           rho_oracle=float(stats.spearmanr(g.disp, g.oracle_coverage.astype(float)).statistic),
                           p_oracle=float(stats.spearmanr(g.disp, g.oracle_coverage.astype(float)).pvalue))
        dd["by_method"] = lab2.groupby("method").disp.mean().round(4).to_dict()
        E.emit("probe.report_displacement", dd,
               note="how far the compiled evidence moves the reader's state, relative to the task-only state")
E.emit("probe.report_architecture", arch, note="which orchestration produced this report, 3-way, chance 1/3")
E.emit("probe.report_correctness", rep, note="whether the delivered answer is right; compare task_only against report_state")

E.save("probes")
