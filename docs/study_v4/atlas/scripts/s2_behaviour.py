"""A2-A7, A10, A11: configuration -> behaviour. Every number the atlas cites about outcomes."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import _common as C

E = C.Emitter("s2_behaviour.py")
sel, ep = C.selections(), C.episodes()
main_sel = sel[sel.split == "main"]
fs = C.flagship(ep)                       # module A, 32B, N5, B4, main
diff = C.difficulty()

# ================= B1/B6/B7: the five policies at matched allowance =================
g = fs.groupby("method")
pol = {}
for m, d in g:
    b = C.boot_mean(d.acc)
    pol[m] = dict(acc=b["mean"], lo=b["lo"], hi=b["hi"], n=b["n"],
                  conf=float(d.native_confidence.mean(skipna=True)),
                  valid=float(d.native_final_valid.mean(skipna=True)),
                  calls=float(d.calls_admitted.mean()),
                  spent_frac=float((d.spent_flops / d.B_flops).mean()),
                  stop=d.stop_reason.value_counts(normalize=True).round(3).to_dict())
E.emit("policy.flagship", pol, n=len(fs), source="module A, 32B, N=5, B=4, main split")
E.emit("policy.by_domain", {m: {dom: round(float(x.acc.mean()), 4) for dom, x in d.groupby("domain")}
                            for m, d in g})
# paired contrasts between every policy pair, pooled and per domain
pairs = {}
for i, a in enumerate(C.POLICIES):
    for b in C.POLICIES[i + 1:]:
        va, vb, items = C.pair_on(fs, "method", "acc", a, b)
        r = C.boot_paired(va, vb); r["items"] = len(items)
        per = {}
        for dom in ("hle", "bcb"):
            fd = fs[fs.domain == dom]
            xa, xb, it = C.pair_on(fd, "method", "acc", a, b)
            per[dom] = C.boot_paired(xa, xb)
        r["by_domain"] = per
        pairs[f"{a}-{b}"] = r
E.emit("policy.paired_contrasts", pairs, note="paired on items; percentile bootstrap, 10k")

# ================= C10: difficulty as master moderator =================
fd = fs.merge(diff, on="source_id", how="inner")
E.emit("difficulty.policy_by_band",
       fd.pivot_table(index="method", columns="band", values="acc", aggfunc="mean", observed=True).round(4).to_dict())
E.emit("difficulty.items_per_band", fd.drop_duplicates("source_id").band.value_counts().to_dict())
E.emit("difficulty.single_draw_by_band",
       fd.drop_duplicates("source_id").groupby("band", observed=True).pass_at_1.mean().round(4).to_dict())
# spread between best and worst policy inside each band
sp = fd.pivot_table(index="band", columns="method", values="acc", aggfunc="mean", observed=True)
E.emit("difficulty.policy_spread_by_band", (sp.max(axis=1) - sp.min(axis=1)).round(4).to_dict(),
       note="best minus worst policy accuracy within each difficulty band")
E.emit("difficulty.policy_spread_pooled",
       float(fs.groupby("method").acc.mean().max() - fs.groupby("method").acc.mean().min()))
# non-tautological never-band census: were they ever solved in any flagship episode?
nb = set(diff[diff.band == "never"].source_id)
allep = ep[(ep.split == "main") & (ep.checkpoint == "32B")]
solved = allep[allep.source_id.isin(nb) & (allep.acc > 0)].source_id.nunique()
E.emit("difficulty.never_band_census",
       dict(items=len(nb), episodes=int(len(allep[allep.source_id.isin(nb)])),
            items_ever_solved=int(solved), solve_events=int((allep[allep.source_id.isin(nb)].acc > 0).sum())),
       note="items with 0/10 neutral draws, checked against every 32B main episode")
# cross-policy agreement pooled vs inside the contested band
def phi_matrix(frame):
    piv = frame.pivot_table(index="source_id", columns="method", values="acc", aggfunc="max")
    piv = piv.dropna()
    if len(piv) < 10: return None, 0
    cols = [c for c in C.POLICIES if c in piv.columns]
    vals = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = piv[a].values, piv[b].values
            if x.std() == 0 or y.std() == 0: continue
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return (float(np.mean(vals)) if vals else None), len(piv)
p_all, n_all = phi_matrix(fd)
p_con, n_con = phi_matrix(fd[fd.band == "contested"])
E.emit("difficulty.policy_agreement_phi", dict(pooled=p_all, pooled_items=n_all,
                                               contested=p_con, contested_items=n_con),
       note="mean pairwise phi of per-item correctness across the five policies")

# ================= B2/B3: pool quality and the selection gap =================
arc = main_sel[(main_sel.module == "A") & (main_sel.checkpoint == "32B") & (main_sel.pool_kind == "archive")]
selq = {}
for (m, s), d in arc.groupby(["method", "selector_id"]):
    selq[f"{m}|{s}"] = dict(selected=float(d.selected_correct.mean()),
                            oracle=float(d.oracle_coverage.mean()),
                            cand_mean=float(d.candidate_mean.mean()),
                            gap=float(d.selection_gap.mean()),
                            valid=float(d.valid_count.mean()),
                            winning=float(d.winning_count.mean(skipna=True)),
                            all_singleton=float(d.all_singleton.mean()),
                            n=int(len(d)))
E.emit("selection.archive_quality", selq, source="module A archive pools, main split")
E.emit("selection.by_domain",
       {f"{m}|{s}|{dom}": dict(selected=float(x.selected_correct.mean()), oracle=float(x.oracle_coverage.mean()),
                               cand_mean=float(x.candidate_mean.mean()), all_singleton=float(x.all_singleton.mean()),
                               n=int(len(x)))
        for (m, s, dom), x in arc.groupby(["method", "selector_id", "domain"])})

# ================= C4/B6: test-time compute curves (F banks) =================
bk = C.banks()
f00 = bk[(bk.framing.astype(str) == "00") & (bk.checkpoint.astype(str) == "32B")].drop_duplicates("source_id")
f00 = f00[f00.source_id.isin(set(main_sel.source_id))]
passk = {}
for dom, d in f00.groupby("domain"):
    passk[dom] = {k: float(d[f"pass_at_{k}"].mean()) for k in (1, 2, 3, 5, 10)}
passk["pooled"] = {k: float(np.mean([passk[d][k] for d in passk])) for k in (1, 2, 3, 5, 10)}
E.emit("ttc.pass_at_k", passk, n=int(len(f00)), source="F bank, framing 00, 32B, main items")
fb = main_sel[main_sel.pool_kind == "bank_prefix"]
sk = {}
for (s, dom), d in fb.groupby(["selector_id", "domain"]):
    sk[f"{s}|{dom}"] = d.groupby("prefix_k").selected_correct.mean().round(4).to_dict()
for s in ("VOTE", "JUDGE_BEST"):
    d = fb[fb.selector_id == s]
    sk[f"{s}|pooled"] = {k: float(np.mean([v for dom in ("hle", "bcb")
                                           for v in [d[d.domain == dom].groupby("prefix_k").selected_correct.mean().get(k, np.nan)]]))
                         for k in sorted(d.prefix_k.dropna().unique())}
E.emit("ttc.selector_at_k", sk, note="VOTE@k and JUDGE_BEST@k over the same bank prefixes")
E.emit("ttc.gap_at_k", {str(int(k)): round(float(passk["pooled"][int(k)] - sk["VOTE|pooled"][k]), 4)
                        for k in (1, 2, 3, 5, 10) if k in sk["VOTE|pooled"]},
       note="pass@k minus VOTE@k, pooled across domains")

# ================= C4: budget and the caps that bind =================
bud = ep[(ep.split == "main") & (ep.module.isin(["A", "B"])) & (ep.checkpoint == "32B")]
E.emit("budget.by_level", {f"{m}|B{int(b)}": dict(acc=float(d.acc.mean()),
                                                  spent_frac=float((d.spent_flops / d.B_flops).mean()),
                                                  calls=float(d.calls_admitted.mean()),
                                                  stop=d.stop_reason.value_counts(normalize=True).round(3).to_dict(),
                                                  n=int(len(d)))
                           for (m, b), d in bud.groupby(["method", "B"])})
bp = {}
for m in C.POLICIES:
    for a, b in ((1, 2), (2, 4), (1, 4)):
        d = bud[bud.method == m]
        va, vb, items = C.pair_on(d, "B", "acc", b, a)
        if len(items) >= 20: bp[f"{m}|B{b}-B{a}"] = C.boot_paired(va, vb)
E.emit("budget.paired_contrasts", bp)
E.emit("budget.stop_reason_overall",
       ep[ep.split == "main"].groupby(["checkpoint", "B"]).stop_reason.value_counts(normalize=True).round(3)
       .unstack(fill_value=0).to_dict("index"))

# ================= C5: model scale =================
DENSE = ["IND_VOTE", "DEC", "CEN_FLAT"]
m_items = set(ep[(ep.split == "main") & (ep.module == "M")].source_id)
ms = pd.concat([ep[(ep.split == "main") & (ep.module == "M")],
                ep[(ep.split == "main") & (ep.module == "A") & (ep.method.isin(DENSE)) &
                   (ep.source_id.isin(m_items))]], ignore_index=True)
E.emit("scale.panel_note", dict(items=len(m_items), methods=DENSE),
       note="the 32B row of the dense panel is module A restricted to the M-panel items and the three "
            "dense-panel methods (amendment M1: the 32B row aliases main A)")
E.emit("scale.dense_panel", {f"{m}|{c}": dict(acc=float(d.acc.mean()), conf=float(d.native_confidence.mean(skipna=True)),
                                              valid=float(d.native_final_valid.mean(skipna=True)),
                                              calls=float(d.calls_admitted.mean()),
                                              spent_frac=float((d.spent_flops / d.B_flops).mean()), n=int(len(d)))
                             for (m, c), d in ms.groupby(["method", "checkpoint"])})
E.emit("scale.by_domain", {f"{m}|{c}|{dom}": round(float(x.acc.mean()), 4)
                           for (m, c, dom), x in ms.groupby(["method", "checkpoint", "domain"])})
scp = {}
for m in ("IND_VOTE", "DEC", "CEN_FLAT"):
    d = ms[ms.method == m]
    for a, b in (("8B", "32B"), ("8B", "14B"), ("4B", "32B"), ("14B", "32B")):
        va, vb, items = C.pair_on(d, "checkpoint", "acc", a, b)
        if len(items) >= 20:
            r = C.boot_paired(va, vb)
            r["by_domain"] = {dom: C.boot_paired(*C.pair_on(d[d.domain == dom], "checkpoint", "acc", a, b)[:2])
                              for dom in ("hle", "bcb")}
            scp[f"{m}|{a}-{b}"] = r
E.emit("scale.paired_contrasts", scp)

# ================= B5: calibration =================
cal = ep[(ep.split == "main") & ep.native_confidence.notna()].copy()
cal["conf"] = cal.native_confidence.astype(float)
def cal_block(d):
    b = ((d.conf - d.acc) ** 2).mean()
    return dict(n=int(len(d)), acc=float(d.acc.mean()), conf=float(d.conf.mean()),
                brier=float(b), bias=float((d.conf - d.acc).mean()),
                share_ge95=float((d.conf >= 0.95).mean()), share_eq50=float((d.conf == 0.5).mean()),
                resolution=float(d[d.acc > 0].conf.mean() - d[d.acc == 0].conf.mean()) if d.acc.nunique() > 1 else None)
E.emit("calib.by_policy_flagship", {m: cal_block(d) for m, d in cal[(cal.module == "A") & (cal.checkpoint == "32B")].groupby("method")})
_calm = cal[cal.source_id.isin(m_items) & (((cal.module == "M")) |
            ((cal.module == "A") & cal.method.isin(DENSE)))]
E.emit("calib.by_checkpoint", {f"{c}": cal_block(d) for c, d in _calm.groupby("checkpoint")})
E.emit("calib.by_checkpoint_method", {f"{m}|{c}": cal_block(d) for (m, c), d in _calm.groupby(["method", "checkpoint"])})
E.emit("calib.by_domain_flagship", {f"{m}|{dom}": cal_block(d)
                                    for (m, dom), d in cal[(cal.module == "A") & (cal.checkpoint == "32B")].groupby(["method", "domain"])})
# reliability curve on the flagship panel
cc = cal[(cal.module == "A") & (cal.checkpoint == "32B")].copy()
cc["bin"] = pd.cut(cc.conf, [-.01, .5, .7, .8, .9, .95, 1.01],
                   labels=["<=0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"])
E.emit("calib.reliability", cc.groupby("bin", observed=True).agg(n=("acc", "size"), acc=("acc", "mean"),
                                                                conf=("conf", "mean")).round(4).to_dict("index"))
# excluding the 0.50 placeholder, does the size trend survive?
nz = _calm[_calm.conf != 0.5]
E.emit("calib.by_checkpoint_excl_placeholder", {c: cal_block(d) for c, d in nz.groupby("checkpoint")})

# ================= B8: run-to-run stability (module E) =================
Emod = ep[(ep.split == "main") & (ep.module == "E")]
if len(Emod):
    st = {}
    for m, d in Emod.groupby("method"):
        per_rep = d.groupby("episode_rep").acc.mean()
        per_item = d.groupby(["source_id"]).acc.agg(["mean", "std", "size"])
        st[m] = dict(rep_min=float(per_rep.min()), rep_max=float(per_rep.max()),
                     rep_range=float(per_rep.max() - per_rep.min()), n_reps=int(per_rep.size),
                     within_item_sd=float(per_item["std"].mean()), items=int(per_item.shape[0]))
    E.emit("stability.by_policy", st, source="module E, 6 repeated episodes")

# ================= structural axes: N, degree, framing =================
Nmod = ep[(ep.split == "main") & (ep.module == "N")]
E.emit("membership.by_N", {f"{m}|N{int(n)}": dict(acc=float(d.acc.mean()), calls=float(d.calls_admitted.mean()),
                                                  conf=float(d.native_confidence.mean(skipna=True)), n=int(len(d)))
                           for (m, n), d in Nmod.groupby(["method", "N"])})
narc = main_sel[(main_sel.module == "N") & (main_sel.selector_id == "VOTE") & (main_sel.pool_kind == "archive")]
E.emit("membership.pool_by_N", {f"{m}|N{int(n)}": dict(selected=float(d.selected_correct.mean()),
                                                       oracle=float(d.oracle_coverage.mean()), n=int(len(d)))
                                for (m, n), d in narc.groupby(["method", "N"])})
Dmod = ep[(ep.split == "main") & (ep.module == "D")]
if len(Dmod):
    E.emit("degree.by_level", {int(k): dict(acc=float(d.acc.mean()), n=int(len(d)))
                               for k, d in Dmod.groupby("degree")})
fr = bk[(bk.checkpoint.astype(str) == "32B")].drop_duplicates(["source_id", "framing"])
fr = fr[fr.source_id.isin(set(main_sel.source_id))]
E.emit("framing.bank_pass", {str(f): dict(p1=float(d.pass_at_1.mean()), p10=float(d.pass_at_10.mean()), n=int(len(d)))
                             for f, d in fr.groupby("framing")})
frs = {}
for f in ("01", "10", "11"):
    va, vb, items = C.pair_on(fr, "framing", "pass_at_1", f, "00")
    if len(items) >= 20: frs[f"{f}-00.pass1"] = C.boot_paired(va, vb)
    va, vb, items = C.pair_on(fr, "framing", "pass_at_10", f, "00")
    if len(items) >= 20: frs[f"{f}-00.pass10"] = C.boot_paired(va, vb)
E.emit("framing.paired_contrasts", frs)
# framing x confidence needs the module-N (framing 00) arm against the module-A (framing 11) arm on shared items
_fa = cal[(cal.checkpoint == "32B") & (cal.method == "IND_VOTE") & (cal.split == "main") &
          (cal.module.isin(["A", "N"])) & (cal.N == 5) & (cal.B == 4)]
E.emit("framing.confidence", {str(f): dict(conf=float(d.conf.mean()), acc=float(d.acc.mean()), n=int(len(d)))
                              for f, d in _fa.groupby("framing")})
_c0, _c1, _it = C.pair_on(_fa, "framing", "conf", "11", "00")
_a0, _a1, _ = C.pair_on(_fa, "framing", "acc", "11", "00")
E.emit("framing.confidence_contrast", dict(conf=C.boot_paired(_c0, _c1), acc=C.boot_paired(_a0, _a1)),
       note="vote-aware framing 11 minus neutral 00, independent sampling, paired on the shared 100-item panel")

# ================= B7: validity =================
E.emit("validity.by_checkpoint", {c: dict(valid_final=float(d.native_final_valid.mean(skipna=True)),
                                          complete=float(d.complete_candidates.mean(skipna=True)), n=int(len(d)))
                                  for c, d in ms.groupby("checkpoint")})
E.emit("validity.by_policy", {m: dict(valid_final=float(d.native_final_valid.mean(skipna=True)),
                                      complete=float(d.complete_candidates.mean(skipna=True)))
                              for m, d in fs.groupby("method")})

E.save("behaviour")
